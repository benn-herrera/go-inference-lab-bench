package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// FullAttentionGatedBuilder implements gated multi-head attention with QK-norm and MRoPE.
// Direct translation of buildFullAttnLayer / buildFullAttnLayerCached in qwen35.go.
type FullAttentionGatedBuilder struct{}

func (b *FullAttentionGatedBuilder) Contract() BuilderContract {
	return BuilderContract{
		Kind:            KindSWAAttention,
		RequiredWeights: []string{WeightAttnQ, WeightAttnK, WeightAttnV, WeightAttnOutput},
		OptionalWeights: []string{WeightAttnQNorm, WeightAttnKNorm},
		RequiredParams:  []string{ParamHeadDim, ParamNHeads, ParamNKVHeads, ParamRoPENRot, ParamRoPEFreqBase},
		ConfigSchema: map[string][]string{
			ConfigQHasGate:  nil, // bool, any value
			ConfigQKNorm:    {NormRMS, NormL2, ""},
			ConfigRope:      {RopeMulti, RopeImrope, RopeStandard, ""},
			ConfigOutputGate: {GateSigmoid, ""},
		},
	}
}

// splitGatedProjection splits a joint Q+gate projection into separate Q and gate tensors.
func splitGatedProjection(ctx *ggml.GraphContext, qg ggml.Tensor, headDim, nHeads, nTokens int64) (ggml.Tensor, ggml.Tensor) {
	elemSz := qg.ElementSize()
	q := ggml.View3D(ctx, qg, headDim, nHeads, nTokens,
		elemSz*int(headDim)*2, elemSz*int(headDim)*2*int(nHeads), 0)
	gate := ggml.View3D(ctx, qg, headDim, nHeads, nTokens,
		elemSz*int(headDim)*2, elemSz*int(headDim)*2*int(nHeads), elemSz*int(headDim))
	gate = ggml.Cont2D(ctx, gate, headDim*nHeads, nTokens)
	return q, gate
}

func (b *FullAttentionGatedBuilder) BuildStateless(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	zeroFill *[]ggml.Tensor) ggml.Tensor {

	headDim := int64(params.Ints[ParamHeadDim])
	nHeads := int64(params.Ints[ParamNHeads])
	nKVHeads := int64(params.Ints[ParamNKVHeads])
	rmsEps := params.Floats[ParamRMSEps]
	nTokens := inputs.NTokens

	// Q+gate joint projection
	qg := ggml.MulMat(ctx, weights[WeightAttnQ], cur)
	q, gate := splitGatedProjection(ctx, qg, headDim, nHeads, nTokens)

	// QK norm
	q = rmsNormApply(ctx, q, weights[WeightAttnQNorm], rmsEps)

	k := projectReshape3D(ctx, weights[WeightAttnK], cur, headDim, nKVHeads, nTokens)
	k = rmsNormApply(ctx, k, weights[WeightAttnKNorm], rmsEps)

	v := projectReshape3D(ctx, weights[WeightAttnV], cur, headDim, nKVHeads, nTokens)

	// MRoPE
	nRot := params.Ints[ParamRoPENRot]
	sections := ropeSections(params)
	freqBase := params.Floats[ParamRoPEFreqBase]
	q, k = applyRoPEMultiPair(ctx, q, k, inputs.InpPos, nRot, sections, freqBase, ropeMultiMode(config))

	// Permute for attention
	q = ggml.Permute(ctx, q, 0, 2, 1, 3)
	k = ggml.Permute(ctx, k, 0, 2, 1, 3)
	v = ggml.Permute(ctx, v, 0, 2, 1, 3)

	// Scaled dot-product attention with GQA. This builder is decoder-only and
	// does not expose ConfigKQPrec; pass true to force F32 (today's behavior).
	cur = scaledDotProductAttention(ctx, q, k, v, inputs.InpMask, attentionScale(headDim), nHeads, nTokens, inputs.Captures, inputs.FlashAttn, true)

	// Gate + output projection
	cur = ggml.Mul(ctx, cur, ggml.Sigmoid(ctx, gate))
	cur = ggml.MulMat(ctx, weights[WeightAttnOutput], cur)
	return cur
}

func (b *FullAttentionGatedBuilder) BuildCached(
	ctx *ggml.GraphContext, gf *ggml.Graph, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	cache *LayerCache) ggml.Tensor {

	headDim := int64(params.Ints[ParamHeadDim])
	nHeads := int64(params.Ints[ParamNHeads])
	nKVHeads := int64(params.Ints[ParamNKVHeads])
	rmsEps := params.Floats[ParamRMSEps]
	nNew := inputs.NTokens
	nKV := inputs.NKV
	seqPos := inputs.SeqPos

	// Q+gate joint projection
	qg := ggml.MulMat(ctx, weights[WeightAttnQ], cur)
	q, gate := splitGatedProjection(ctx, qg, headDim, nHeads, nNew)

	q = rmsNormApply(ctx, q, weights[WeightAttnQNorm], rmsEps)

	kNew := projectReshape3D(ctx, weights[WeightAttnK], cur, headDim, nKVHeads, nNew)
	kNew = rmsNormApply(ctx, kNew, weights[WeightAttnKNorm], rmsEps)
	vNew := projectReshape3D(ctx, weights[WeightAttnV], cur, headDim, nKVHeads, nNew)

	// RoPE
	nRot := params.Ints[ParamRoPENRot]
	sections := ropeSections(params)
	freqBase := params.Floats[ParamRoPEFreqBase]
	q, kNew = applyRoPEMultiPair(ctx, q, kNew, inputs.InpPos, nRot, sections, freqBase, ropeMultiMode(config))

	// KV cache writeback (in-graph GPU copy). Gated on WriteKV: PMM shadow
	// (Stream B) passes set this false to read mainline K/V without overwriting it.
	if inputs.WriteKV {
		writeCacheKV(ctx, gf, kNew, vNew, cache, seqPos, nKVHeads)
	}

	// For attention: prefill uses inline K/V, decode reads from cache
	kAttn, vAttn := selectCachedKV(ctx, cache, seqPos, kNew, vNew, headDim, nKV, nKVHeads)

	q = ggml.Permute(ctx, q, 0, 2, 1, 3)

	// Decoder-only builder; ConfigKQPrec not exposed. Force F32 (today's behavior).
	cur = scaledDotProductAttention(ctx, q, kAttn, vAttn, inputs.InpMask, attentionScale(headDim), nHeads, nNew, inputs.Captures, inputs.FlashAttn, true)
	cur = ggml.Mul(ctx, cur, ggml.Sigmoid(ctx, gate))
	cur = ggml.MulMat(ctx, weights[WeightAttnOutput], cur)
	return cur
}
