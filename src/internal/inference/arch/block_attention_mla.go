package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// MLAAttentionBuilder implements Multi-head Latent Attention (DeepSeek V2 / GLM-4).
// Low-rank Q and KV compression with Q-nope absorption and post-attention V decompression.
type MLAAttentionBuilder struct{}

// buildMLAQueryPath computes the MLA Q path: low-rank compress → norm → expand → split →
// RoPE on positional component → Q-nope absorption → concat absorbed + positional.
// Returns qFinal: [kvLoraRank+ropeDim, nHeads, seqLen].
func buildMLAQueryPath(ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	pos ggml.Tensor, nHeads, headKDim, ropeDim, seqLen int64,
	rmsEps, freqBase float32) ggml.Tensor {
	nopeDim := headKDim - ropeDim

	qCompressed := ggml.MulMat(ctx, weights[WeightAttnQA], cur)
	qCompressed = rmsNormApply(ctx, qCompressed, weights[WeightAttnQANorm], rmsEps)
	qExpanded := ggml.MulMat(ctx, weights[WeightAttnQB], qCompressed)
	q := ggml.Reshape3D(ctx, qExpanded, headKDim, nHeads, seqLen)

	qNopeNb1 := q.Nb(1)
	qNope := ggml.View3D(ctx, q, nopeDim, nHeads, seqLen,
		qNopeNb1, q.Nb(2), 0)
	qPe := ggml.View3D(ctx, q, ropeDim, nHeads, seqLen,
		qNopeNb1, q.Nb(2), int(nopeDim)*q.ElementSize())

	qPe = defaultRopeExt(ctx, qPe, pos, int(ropeDim), freqBase)

	qNopePerm := ggml.Permute(ctx, qNope, 0, 2, 1, 3)
	qAbsorbed := ggml.MulMat(ctx, weights[WeightAttnKB], qNopePerm)
	qAbsorbed = ggml.Permute(ctx, qAbsorbed, 0, 2, 1, 3)
	return ggml.Concat(ctx, qAbsorbed, qPe, 0)
}

// buildMLAOutputTail decompresses V (kqv x attn_v_b), merges heads, and applies
// the output projection. Shared tail for MLA's stateless and cached paths.
func buildMLAOutputTail(ctx *ggml.GraphContext, kqv ggml.Tensor, weights map[string]ggml.Tensor,
	nHeads, seqLen int64) ggml.Tensor {
	// V decompression: attn_v_b [kvLoraRank, headVDim, nHeads] @ kqv [kvLoraRank, seqLen, nHeads]
	// Batched matmul on dim 2 (nHeads), contracts on dim 0 (kvLoraRank) -> [headVDim, seqLen, nHeads]
	decompressed := ggml.MulMat(ctx, weights[WeightAttnVB], kqv)

	// Merge heads: permute to [headVDim, nHeads, seqLen] then flatten
	headVDim := decompressed.Ne(0)
	merged := ggml.Permute(ctx, decompressed, 0, 2, 1, 3)
	cur := ggml.Cont2D(ctx, merged, headVDim*nHeads, seqLen)
	cur = ggml.MulMat(ctx, weights[WeightAttnOutput], cur)
	return cur
}

func (b *MLAAttentionBuilder) Contract() BuilderContract {
	return BuilderContract{
		Kind: KindAttention,
		RequiredWeights: []string{
			WeightAttnQA, WeightAttnQANorm, WeightAttnQB,
			WeightAttnKVAMQA, WeightAttnKVANorm,
			WeightAttnKB, WeightAttnVB,
			WeightAttnOutput,
		},
		RequiredParams: []string{
			ParamNHeads, ParamRMSEps, ParamRoPENRot, ParamRoPEFreqBase,
			ParamKVLoraRank, ParamHeadKDimMLA,
		},
	}
}

func (b *MLAAttentionBuilder) BuildStateless(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	zeroFill *[]ggml.Tensor) ggml.Tensor {

	nHeads := int64(params.Ints[ParamNHeads])
	rmsEps := params.Floats[ParamRMSEps]
	nTokens := inputs.NTokens
	kvLoraRank := int64(params.Ints[ParamKVLoraRank])
	headKDim := int64(params.Ints[ParamHeadKDimMLA])
	ropeDim := int64(params.Ints[ParamRoPENRot])
	freqBase := params.Floats[ParamRoPEFreqBase]

	qFinal := buildMLAQueryPath(ctx, cur, weights, inputs.InpPos,
		nHeads, headKDim, ropeDim, nTokens, rmsEps, freqBase)

	// KV path: compress → split → norm + RoPE
	kvFull := ggml.MulMat(ctx, weights[WeightAttnKVAMQA], cur) // [kvLoraRank+ropeDim, nTokens]

	// Split into compressed KV and positional K
	kvCompressed := ggml.View2D(ctx, kvFull, kvLoraRank, nTokens,
		kvFull.Nb(1), 0)
	kPe := ggml.View3D(ctx, kvFull, ropeDim, int64(1), nTokens,
		kvFull.Nb(1), kvFull.Nb(1),
		int(kvLoraRank)*kvFull.ElementSize())

	// Norm the compressed KV
	kvCompressed = rmsNormApply(ctx, kvCompressed, weights[WeightAttnKVANorm], rmsEps)

	// RoPE on positional K only
	kPe = defaultRopeExt(ctx, kPe, inputs.InpPos, int(ropeDim), freqBase)

	// K: concat compressed + positional (MQA: 1 KV head)
	kvCompressed3d := ggml.Reshape3D(ctx, kvCompressed, kvLoraRank, int64(1), nTokens)
	kFinal := ggml.Concat(ctx, kvCompressed3d, kPe, 0) // [kvLoraRank+ropeDim, 1, nTokens]

	// V: just the compressed representation
	vFinal := ggml.Reshape3D(ctx, kvCompressed, kvLoraRank, int64(1), nTokens)

	// Attention (MQA: 1 KV head, n_heads Q heads — GQA broadcasting)
	qPerm := ggml.Permute(ctx, qFinal, 0, 2, 1, 3) // [kvLoraRank+ropeDim, nTokens, nHeads]
	kPerm := ggml.Permute(ctx, kFinal, 0, 2, 1, 3) // [kvLoraRank+ropeDim, nTokens, 1]
	vPerm := ggml.Permute(ctx, vFinal, 0, 2, 1, 3) // [kvLoraRank, nTokens, 1]

	kqScale := attentionScale(kvLoraRank + ropeDim)
	kq := ggml.MulMat(ctx, kPerm, qPerm) // [nTokens, nTokens, nHeads] with GQA broadcast
	kq = ggml.SoftMaxExt(ctx, kq, inputs.InpMask, kqScale, 0.0)

	vT := ggml.Cont(ctx, ggml.Transpose(ctx, vPerm))
	kqv := ggml.MulMat(ctx, vT, kq) // [kvLoraRank, nTokens, nHeads]

	return buildMLAOutputTail(ctx, kqv, weights, nHeads, nTokens)
}

func (b *MLAAttentionBuilder) BuildCached(
	ctx *ggml.GraphContext, gf *ggml.Graph, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	cache *LayerCache) ggml.Tensor {

	nHeads := int64(params.Ints[ParamNHeads])
	rmsEps := params.Floats[ParamRMSEps]
	nNew := inputs.NTokens
	nKV := inputs.NKV
	seqPos := inputs.SeqPos
	kvLoraRank := int64(params.Ints[ParamKVLoraRank])
	headKDim := int64(params.Ints[ParamHeadKDimMLA])
	ropeDim := int64(params.Ints[ParamRoPENRot])
	freqBase := params.Floats[ParamRoPEFreqBase]
	kDim := kvLoraRank + ropeDim // total K cache dim per entry

	qFinal := buildMLAQueryPath(ctx, cur, weights, inputs.InpPos,
		nHeads, headKDim, ropeDim, nNew, rmsEps, freqBase)

	// KV path for new tokens
	kvFull := ggml.MulMat(ctx, weights[WeightAttnKVAMQA], cur)
	kvCompressed := ggml.View2D(ctx, kvFull, kvLoraRank, nNew, kvFull.Nb(1), 0)
	kPeNew := ggml.View3D(ctx, kvFull, ropeDim, int64(1), nNew,
		kvFull.Nb(1), kvFull.Nb(1),
		int(kvLoraRank)*kvFull.ElementSize())

	kvCompressed = rmsNormApply(ctx, kvCompressed, weights[WeightAttnKVANorm], rmsEps)
	kPeNew = defaultRopeExt(ctx, kPeNew, inputs.InpPos, int(ropeDim), freqBase)

	kvCompressed3d := ggml.Reshape3D(ctx, kvCompressed, kvLoraRank, int64(1), nNew)
	kNew := ggml.Concat(ctx, kvCompressed3d, kPeNew, 0) // [kDim, 1, nNew]

	// Cache writeback: K only (MLA: V derived from K's compressed portion).
	// Emit in-graph cpy into the cache buffer at seqPos. Gated on WriteKV:
	// PMM shadow (Stream B) passes set this false to read mainline K without overwriting.
	kc := cache.Tensors[CacheK]
	if inputs.WriteKV {
		kForCache := ggml.Cont(ctx, ggml.Permute(ctx, kNew, 0, 2, 1, 3)) // [kDim, nNew, 1]
		const float32Size = 4
		kView := ggml.View3D(ctx, kc, kDim, nNew, int64(1), kc.Nb(1), kc.Nb(2), seqPos*int(kDim)*float32Size)
		gf.BuildForwardExpand(ggml.Cpy(ctx, kForCache, kView))
	}

	// For attention: build K and V from cache or inline
	var kAttn, vAttn ggml.Tensor
	if seqPos == 0 {
		// Prefill: use inline
		kAttn = ggml.Cont(ctx, ggml.Permute(ctx, kNew, 0, 2, 1, 3))
		// V = compressed portion of K (first kvLoraRank elements)
		vNew := ggml.Reshape3D(ctx, kvCompressed, kvLoraRank, int64(1), nNew)
		vAttn = ggml.Cont(ctx, ggml.Permute(ctx, vNew, 0, 2, 1, 3))
	} else {
		// Decode: read K from cache, derive V from K's compressed portion
		kAttn = ggml.View3D(ctx, kc, kDim, nKV, int64(1), kc.Nb(1), kc.Nb(2), 0)
		// V = first kvLoraRank elements of each K entry
		vAttn = ggml.View3D(ctx, kc, kvLoraRank, nKV, int64(1), kc.Nb(1), kc.Nb(2), 0)
	}

	// Attention
	qPerm := ggml.Permute(ctx, qFinal, 0, 2, 1, 3)
	kqScale := attentionScale(kDim)
	kq := ggml.MulMat(ctx, kAttn, qPerm)
	kq = ggml.SoftMaxExt(ctx, kq, inputs.InpMask, kqScale, 0.0)

	vT := ggml.Cont(ctx, ggml.Transpose(ctx, vAttn))
	kqv := ggml.MulMat(ctx, vT, kq)

	return buildMLAOutputTail(ctx, kqv, weights, nHeads, nNew)
}
