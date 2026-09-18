package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// AttentionBuilder implements standard multi-head attention with GQA and RoPE.
// Supports config-based param overrides for per-block head dims/counts,
// optional K/V (shared KV for non-KV layers), V-norm, sliding window mask,
// and RoPE frequency factors.
type AttentionBuilder struct{}

func (b *AttentionBuilder) Contract() BuilderContract {
	return BuilderContract{
		Kind:            KindAttention,
		RequiredWeights: []string{WeightAttnQ, WeightAttnOutput},
		OptionalWeights: []string{WeightAttnK, WeightAttnV, WeightAttnQNorm, WeightAttnKNorm, WeightRoPEFreqs,
			WeightAttnQBias, WeightAttnKBias, WeightAttnVBias, WeightAttnOutputBias},
		RequiredParams: []string{ParamHeadDim, ParamNHeads, ParamNKVHeads, ParamRoPENRot, ParamRoPEFreqBase},
		ConfigSchema: map[string][]string{
			ConfigRope:          {RopeStandard, RopeNeox, RopeAxial2D, RopeMRopeVision, RopeNone, ""},
			ConfigKQPrec:        {KQPrecNative, ""},
			ConfigVNorm:         {NormRMS, ""},
			ParamSlidingWindow:  nil,
			ConfigSharedKVGroup: nil,
			ParamHeadDim:        nil,
			ParamNHeads:         nil,
			ParamNKVHeads:       nil,
			ParamRoPENRot:       nil,
			ParamRoPEFreqBase:   nil,
			ConfigKQScale:       nil,
			ConfigNonCausal:     nil,
			ConfigQKVFused:      nil,
		},
	}
}

// attnParams reads attention dimensions from config overrides or global params.
func attnParams(params *ResolvedParams, config map[string]any) (headDim, nHeads, nKVHeads int64, nRot, ropeMode int, freqBase, kqScale float32) {
	headDim = int64(configIntOr(config, ParamHeadDim, params))
	nHeads = int64(configIntOr(config, ParamNHeads, params))
	nKVHeads = int64(configIntOr(config, ParamNKVHeads, params))
	nRot = configIntOr(config, ParamRoPENRot, params)
	freqBase = configFloatOr(config, ParamRoPEFreqBase, params)
	kqScale = configFloatOr(config, ConfigKQScale, params)
	if kqScale == 0 {
		kqScale = attentionScale(headDim)
	}
	if configStrOr(config, ConfigRope, "") == RopeNeox {
		ropeMode = ggml.RopeTypeNeoX
	}
	return
}

// kqForceF32 resolves the ConfigKQPrec knob to the bool consumed by
// scaledDotProductAttention. Default (unset / "") preserves the decoder
// behavior of forcing F32 accumulation on the K·Q matmul; KQPrecNative opts
// out so the matmul uses native precision (clip-encoder parity).
func kqForceF32(config map[string]any) bool {
	return configStrOr(config, ConfigKQPrec, "") != KQPrecNative
}

// qkvResult holds the prepared Q, K, V tensors from the KV pipeline.
// HasKV indicates whether this layer projected its own K/V (vs reading from SharedKV).
type qkvResult struct {
	Q, K, V  ggml.Tensor
	HasKV    bool
	NKVHeads int64 // actual kv_heads (may be inferred from K weight shape)
}

// prepareQKV runs the full KV preparation pipeline: Q/K/V projection, optional QK-norm,
// optional V-norm, RoPE (with optional frequency factors), and SharedKV read/write.
func (b *AttentionBuilder) prepareQKV(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	headDim, nHeads, nKVHeads int64, nRot, ropeMode int, freqBase float32,
) qkvResult {
	nTokens := inputs.NTokens
	hasKV := !weights[WeightAttnK].IsNil()

	q := projectReshape3DClamped(ctx, weights[WeightAttnQ], cur, headDim, nHeads, nTokens,
		clampFor(inputs.LinearClamps, WeightAttnQ))
	q = addProjBias(ctx, q, weights[WeightAttnQBias], headDim, nHeads)

	var k, v ggml.Tensor
	if hasKV {
		// Infer actual n_kv_heads from K weight shape when it differs from the param.
		// Handles models where SWA and full attention have different kv_head counts
		// but only one n_kv_heads param exists in the GGUF.
		if actual := weights[WeightAttnK].Ne(1) / headDim; actual > 0 {
			nKVHeads = actual
		}
		k = projectReshape3DClamped(ctx, weights[WeightAttnK], cur, headDim, nKVHeads, nTokens,
			clampFor(inputs.LinearClamps, WeightAttnK))
		k = addProjBias(ctx, k, weights[WeightAttnKBias], headDim, nKVHeads)
		if !weights[WeightAttnV].IsNil() {
			v = projectReshape3DClamped(ctx, weights[WeightAttnV], cur, headDim, nKVHeads, nTokens,
				clampFor(inputs.LinearClamps, WeightAttnV))
			v = addProjBias(ctx, v, weights[WeightAttnVBias], headDim, nKVHeads)
		} else {
			v = k
		}
	}

	// Optional QK-norm
	if !weights[WeightAttnQNorm].IsNil() {
		rmsEps := params.Floats[ParamRMSEps]
		q = rmsNormApply(ctx, q, weights[WeightAttnQNorm], rmsEps)
		if hasKV {
			k = rmsNormApply(ctx, k, weights[WeightAttnKNorm], rmsEps)
		}
	}

	// Optional V-norm (raw RMS norm, no learned weights)
	if hasKV && configStrOr(config, ConfigVNorm, "") == NormRMS {
		v = ggml.RmsNorm(ctx, v, params.Floats[ParamRMSEps])
	}

	// RoPE dispatch on the configured mode.
	switch configStrOr(config, ConfigRope, "") {
	case RopeNone:
		// No rotation. ViT-style encoders set `rope = "none"` and supply
		// position information via a separate position-embedding tensor
		// added before the layer loop.
	case RopeAxial2D:
		// Generic axial (2D) RoPE: rotate the first half of head_dim by the
		// per-token X position and the second half by the Y position. q/k are
		// [head_dim, n_heads, n_tokens] here, which is exactly applyRope2D's
		// expected [head_dim, n_heads, n_patches] layout (n_tokens==n_patches
		// for an encoder). Reuses the vision-tower 2D-RoPE math.
		if inputs.PosX.IsNil() || inputs.PosY.IsNil() {
			// axial2d is a tower-only mode; a block selecting it MUST run in a
			// graph that supplies PosX/PosY. Reaching here is a wiring bug, not
			// user input, and prepareQKV has no error channel — fail loud.
			panic("AttentionBuilder: rope=\"axial2d\" requires GraphInputs.PosX and PosY, " +
				"but one or both are nil (graph-wiring bug — this mode is tower-only)")
		}
		q = applyRope2D(ctx, q, inputs.PosX, inputs.PosY, headDim, freqBase)
		if hasKV {
			k = applyRope2D(ctx, k, inputs.PosX, inputs.PosY, headDim, freqBase)
		}
	case RopeMRopeVision:
		// Multi-section vision M-RoPE (Qwen3-VL tower). Mirrors qwen3vl.cpp
		// op-for-op:
		//   ggml_rope_multi(ctx, Q, positions, nullptr,
		//       d_head/2, {d_head/4 ×4}, GGML_ROPE_TYPE_VISION,
		//       32768, 10000, 1, 0, 1, 32, 1)
		// n_dims = head_dim/2, sections = {head_dim/4 ×4}, mode = VISION,
		// n_ctx_orig = 32768, freq_base = θ (10000, from ParamRoPEFreqBase),
		// freq_scale 1, ext_factor 0, attn_factor 1, beta_fast 32, beta_slow 1.
		if inputs.InpPosVision.IsNil() {
			// mrope_vision is a tower-only mode; a block selecting it MUST run in
			// a graph that supplies InpPosVision. Reaching here is a wiring bug,
			// not user input, and prepareQKV has no error channel — fail loud.
			panic("AttentionBuilder: rope=\"mrope_vision\" requires GraphInputs.InpPosVision, " +
				"but it is nil (graph-wiring bug — this mode is tower-only)")
		}
		sec := [4]int{int(headDim / 4), int(headDim / 4), int(headDim / 4), int(headDim / 4)}
		nDimsV := int(headDim / 2)
		q = ggml.RopeMulti(ctx, q, inputs.InpPosVision, ggml.NilTensor(),
			nDimsV, sec, ggml.RopeTypeVision, 32768, freqBase, 1.0, 0.0, 1.0, 32.0, 1.0)
		if hasKV {
			k = ggml.RopeMulti(ctx, k, inputs.InpPosVision, ggml.NilTensor(),
				nDimsV, sec, ggml.RopeTypeVision, 32768, freqBase, 1.0, 0.0, 1.0, 32.0, 1.0)
		}
	default:
		// RopeStandard / RopeNeox / "" → 1D RopeExt (unchanged equiv-tested path).
		freqFactors := weights[WeightRoPEFreqs]
		if freqFactors.IsNil() {
			freqFactors = ggml.NilTensor()
		}
		q = ggml.RopeExt(ctx, q, inputs.InpPos, freqFactors, nRot, ropeMode, 0,
			freqBase, 1.0, 0.0, 1.0, 32.0, 1.0)
		if hasKV {
			k = ggml.RopeExt(ctx, k, inputs.InpPos, freqFactors, nRot, ropeMode, 0,
				freqBase, 1.0, 0.0, 1.0, 32.0, 1.0)
		}
	}

	// Update SharedKV if this is a KV layer
	if hasKV && inputs.SharedKV != nil {
		if group := configStrOr(config, ConfigSharedKVGroup, ""); group != "" {
			inputs.SharedKV.K[group] = k
			inputs.SharedKV.V[group] = v
		}
	}

	// Non-KV layers get K/V from SharedKV
	if !hasKV && inputs.SharedKV != nil {
		group := configStrOr(config, ConfigSharedKVGroup, "")
		k = inputs.SharedKV.K[group]
		v = inputs.SharedKV.V[group]
	}

	return qkvResult{Q: q, K: k, V: v, HasKV: hasKV, NKVHeads: nKVHeads}
}

// selectMask returns the mask tensor to feed into softmax for this block.
// Precedence:
//  1. `non_causal = true` on the block → NilTensor (bidirectional attention,
//     no causal triangle). Used by ViT-style encoders. SoftMaxExt and
//     FlashAttnExt both tolerate a nil mask per the opt-prefix convention.
//  2. `sliding_window = true` on the block AND the SWA mask is allocated
//     (which only happens when the global ParamSlidingWindow window-size
//     param is > 0) → InpMaskSWA.
//  3. Otherwise → InpMask (the standard causal mask, optionally relaxed by
//     the arch-level NonCausal flag which already affects InpMask's content).
//
// The non_causal block-config short-circuits SWA: a block declaring both
// is a config error caught at TOML load (validate_lines flags conflicting
// flags), but at runtime non_causal wins to keep behavior well-defined.
func selectMask(config map[string]any, inputs *GraphInputs) ggml.Tensor {
	if configBoolOr(config, ConfigNonCausal, false) {
		return ggml.NilTensor()
	}
	if configBoolOr(config, ParamSlidingWindow, false) && !inputs.InpMaskSWA.IsNil() {
		return inputs.InpMaskSWA
	}
	return inputs.InpMask
}

func (b *AttentionBuilder) BuildStateless(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	zeroFill *[]ggml.Tensor) ggml.Tensor {

	headDim, nHeads, nKVHeads, nRot, ropeMode, freqBase, kqScale := attnParams(params, config)
	kv := b.prepareQKV(ctx, cur, weights, params, config, inputs,
		headDim, nHeads, nKVHeads, nRot, ropeMode, freqBase)
	mask := selectMask(config, inputs)

	q := ggml.Permute(ctx, kv.Q, 0, 2, 1, 3)
	k := ggml.Permute(ctx, kv.K, 0, 2, 1, 3)
	v := ggml.Permute(ctx, kv.V, 0, 2, 1, 3)

	cur = scaledDotProductAttention(ctx, q, k, v, mask, kqScale, nHeads, inputs.NTokens, inputs.Captures, inputs.FlashAttn, kqForceF32(config))
	out := mulMatClamped(ctx, weights[WeightAttnOutput], cur, clampFor(inputs.LinearClamps, WeightAttnOutput))
	return addOutputBias(ctx, out, weights[WeightAttnOutputBias])
}

func (b *AttentionBuilder) BuildCached(
	ctx *ggml.GraphContext, gf *ggml.Graph, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	cache *LayerCache) ggml.Tensor {

	headDim, nHeads, nKVHeads, nRot, ropeMode, freqBase, kqScale := attnParams(params, config)
	kv := b.prepareQKV(ctx, cur, weights, params, config, inputs,
		headDim, nHeads, nKVHeads, nRot, ropeMode, freqBase)
	mask := selectMask(config, inputs)

	q := ggml.Permute(ctx, kv.Q, 0, 2, 1, 3)
	nKVHeadsActual := kv.NKVHeads
	if kv.HasKV {
		if inputs.WriteKV {
			writeCacheKV(ctx, gf, kv.K, kv.V, cache, inputs.SeqPos, nKVHeadsActual)
		}
		kAttn, vAttn := selectCachedKV(ctx, cache, inputs.SeqPos, kv.K, kv.V, headDim, inputs.NKV, nKVHeadsActual)
		cur = scaledDotProductAttention(ctx, q, kAttn, vAttn, mask, kqScale, nHeads, inputs.NTokens, inputs.Captures, inputs.FlashAttn, kqForceF32(config))
	} else {
		group := configStrOr(config, ConfigSharedKVGroup, "")
		kAttn, vAttn := selectSharedKV(ctx, cache, inputs.SeqPos, inputs.SharedKV, group, headDim, nKVHeadsActual)
		cur = scaledDotProductAttention(ctx, q, kAttn, vAttn, mask, kqScale, nHeads, inputs.NTokens, inputs.Captures, inputs.FlashAttn, kqForceF32(config))
	}

	out := mulMatClamped(ctx, weights[WeightAttnOutput], cur, clampFor(inputs.LinearClamps, WeightAttnOutput))
	return addOutputBias(ctx, out, weights[WeightAttnOutputBias])
}

// addProjBias adds an optional per-projection bias to a 3D Q/K/V tensor shaped
// [head_dim, n_heads, n_tokens]. bias is the stored 1D weight of length
// head_dim*n_heads (the projection out-features); it is reshaped to
// [head_dim, n_heads] so ggml broadcasts it over the token dimension — exactly
// the per-out-feature bias the reference adds to the fused projection before
// the head split. Nil bias (every decoder/Gemma path) is a no-op, leaving the
// graph byte-identical.
func addProjBias(ctx *ggml.GraphContext, t, bias ggml.Tensor, headDim, nHeads int64) ggml.Tensor {
	if bias.IsNil() {
		return t
	}
	return ggml.Add(ctx, t, ggml.Reshape2D(ctx, bias, headDim, nHeads))
}

// addOutputBias adds an optional output-projection bias to the 2D attention
// output [n_embd, n_tokens]. bias is a 1D [n_embd] tensor, broadcast over
// tokens. Nil bias (every decoder/Gemma path) is a no-op.
func addOutputBias(ctx *ggml.GraphContext, t, bias ggml.Tensor) ggml.Tensor {
	if bias.IsNil() {
		return t
	}
	return ggml.Add(ctx, t, bias)
}
