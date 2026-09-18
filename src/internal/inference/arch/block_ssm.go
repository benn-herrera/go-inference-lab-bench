package arch

import (
	ggml "inference-lab-bench/internal/ggml"
)

// newZeroFilledInput creates a tensor, marks it as input, and appends it to the zero-fill list.
func newZeroFilledInput(ctx *ggml.GraphContext, typ ggml.GGMLType, zeroFill *[]ggml.Tensor, dims ...int64) ggml.Tensor {
	var t ggml.Tensor
	switch len(dims) {
	case 3:
		t = ggml.NewTensor3D(ctx, typ, dims[0], dims[1], dims[2])
	case 4:
		t = ggml.NewTensor4D(ctx, typ, dims[0], dims[1], dims[2], dims[3])
	}
	ggml.SetInput(t)
	*zeroFill = append(*zeroFill, t)
	return t
}

// GatedDeltaNetBuilder implements the gated delta-net SSM block.
// Direct translation of buildSSMLayer / buildSSMLayerCached in qwen35.go.
type GatedDeltaNetBuilder struct{}

func (b *GatedDeltaNetBuilder) Contract() BuilderContract {
	return BuilderContract{
		Kind:            KindRecurrent,
		RequiredWeights: []string{WeightAttnQKV, WeightAttnGate, WeightSSMA, WeightSSMAlpha, WeightSSMBeta,
			WeightSSMConv1D, WeightSSMDTBias, WeightSSMOut},
		OptionalWeights: []string{WeightSSMNorm},
		RequiredParams: []string{ParamConvChannels, ParamSSMDTRank, ParamSSMDState, ParamSSMNGroup,
			ParamHeadVDim, ParamSSMDConv, ParamSSMDInner, ParamNEmbd},
		ConfigSchema: map[string][]string{
			ConfigConvActivation: {ActivationSiLU, ""},
			ConfigQKNorm:         {NormL2, NormRMS, ""},
			ConfigGateNorm:       {NormRMS, ""},
			ConfigGateActivation: {ActivationSiLU, ""},
		},
	}
}

func (b *GatedDeltaNetBuilder) BuildStateless(
	ctx *ggml.GraphContext, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	zeroFill *[]ggml.Tensor) ggml.Tensor {

	nTokens := inputs.NTokens
	rmsEps := params.Floats[ParamRMSEps]
	nSeqs := int64(1)
	cc := int64(params.Ints[ParamConvChannels])
	dtRank := int64(params.Ints[ParamSSMDTRank])
	dState := int64(params.Ints[ParamSSMDState])
	nGroup := int64(params.Ints[ParamSSMNGroup])
	hvDim := int64(params.Ints[ParamHeadVDim])
	dConv := int64(params.Ints[ParamSSMDConv])
	dInner := int64(params.Ints[ParamSSMDInner])
	nEmbd := int64(params.Ints[ParamNEmbd])

	qkvMixed := ggml.Reshape3D(ctx, ggml.MulMat(ctx, weights[WeightAttnQKV], cur), cc, nTokens, nSeqs)
	z := ggml.MulMat(ctx, weights[WeightAttnGate], cur)

	beta := ggml.Sigmoid(ctx, ggml.Reshape4D(ctx, ggml.MulMat(ctx, weights[WeightSSMBeta], cur), 1, dtRank, nTokens, nSeqs))

	alpha := ggml.Reshape3D(ctx, ggml.MulMat(ctx, weights[WeightSSMAlpha], cur), dtRank, nTokens, nSeqs)
	alpha = ggml.Softplus(ctx, ggml.Add(ctx, alpha, weights[WeightSSMDTBias]))
	g := ggml.Reshape4D(ctx, ggml.Mul(ctx, alpha, weights[WeightSSMA]), 1, dtRank, nTokens, nSeqs)

	// Conv1d with zero initial state
	convStates := newZeroFilledInput(ctx, ggml.TypeF32, zeroFill, dConv-1, cc, nSeqs)

	qkvT := ggml.Transpose(ctx, qkvMixed)
	convInput := ggml.Concat(ctx, convStates, qkvT, 0)
	convOut := ggml.Silu(ctx, ggml.SSMConv(ctx, convInput, weights[WeightSSMConv1D]))

	// Split Q, K, V
	qkvDim := dState*nGroup*2 + hvDim*dtRank
	nb1QKV := ggml.RowSize(ggml.TypeF32, qkvDim)

	qSSM := ggml.View4D(ctx, convOut, dState, nGroup, nTokens, nSeqs,
		ggml.RowSize(ggml.TypeF32, dState), nb1QKV, nb1QKV*int(nTokens), 0)
	kSSM := ggml.View4D(ctx, convOut, dState, nGroup, nTokens, nSeqs,
		ggml.RowSize(ggml.TypeF32, dState), nb1QKV, nb1QKV*int(nTokens),
		int(dState*nGroup)*convOut.ElementSize())
	vSSM := ggml.View4D(ctx, convOut, hvDim, dtRank, nTokens, nSeqs,
		ggml.RowSize(ggml.TypeF32, hvDim), nb1QKV, nb1QKV*int(nTokens),
		ggml.RowSize(ggml.TypeF32, 2*dState*nGroup))

	qSSM = ggml.L2Norm(ctx, qSSM, rmsEps)
	kSSM = ggml.L2Norm(ctx, kSSM, rmsEps)

	if nGroup != dtRank {
		qSSM = ggml.Repeat4D(ctx, qSSM, dState, dtRank, nTokens, nSeqs)
		kSSM = ggml.Repeat4D(ctx, kSSM, dState, dtRank, nTokens, nSeqs)
	}

	// Delta-net fused op with zero initial state.
	// ggml gated_delta_net contract (v0.12.0+): state is [D, K, n_seqs] with
	// D = S_v*S_v*H (S_v=hvDim value dim, H=dtRank heads) and K = snapshot-slot
	// count (1 here — bench does not use spec-decode state snapshots). The
	// output layout (per-token block then state tail) is unchanged for K=1, so
	// the views below are unaffected. (Earlier ggml took state as [S_v,S_v,H,seqs].)
	ssmState := newZeroFilledInput(ctx, ggml.TypeF32, zeroFill, hvDim*hvDim*dtRank, 1, nSeqs)

	deltaOut := ggml.GatedDeltaNet(ctx, qSSM, kSSM, vSSM, g, beta, ssmState)

	ssmOutput := ggml.View4D(ctx, deltaOut, hvDim, dtRank, nTokens, nSeqs,
		ggml.RowSize(ggml.TypeF32, hvDim),
		ggml.RowSize(ggml.TypeF32, hvDim*dtRank),
		ggml.RowSize(ggml.TypeF32, hvDim*dtRank*nTokens), 0)

	// Gated normalization
	z4d := ggml.Reshape4D(ctx, z, hvDim, dtRank, nTokens, nSeqs)
	normOut := ssmOutput
	if !weights[WeightSSMNorm].IsNil() {
		normOut = rmsNormApply(ctx, ssmOutput, weights[WeightSSMNorm], rmsEps)
	}
	cur = ggml.Mul(ctx, normOut, ggml.Silu(ctx, z4d))

	cur = ggml.Reshape2D(ctx, cur, dInner, nTokens*nSeqs)

	cur = ggml.MulMat(ctx, weights[WeightSSMOut], cur)
	cur = ggml.Reshape2D(ctx, cur, nEmbd, nTokens*nSeqs)
	return cur
}

func (b *GatedDeltaNetBuilder) BuildCached(
	ctx *ggml.GraphContext, gf *ggml.Graph, cur ggml.Tensor, weights map[string]ggml.Tensor,
	params *ResolvedParams, config map[string]any, inputs *GraphInputs,
	cache *LayerCache) ggml.Tensor {

	nNew := inputs.NTokens
	rmsEps := params.Floats[ParamRMSEps]
	nSeqs := int64(1)
	cc := int64(params.Ints[ParamConvChannels])
	dtRank := int64(params.Ints[ParamSSMDTRank])
	dState := int64(params.Ints[ParamSSMDState])
	nGroup := int64(params.Ints[ParamSSMNGroup])
	hvDim := int64(params.Ints[ParamHeadVDim])
	dConv := int64(params.Ints[ParamSSMDConv])
	dInner := int64(params.Ints[ParamSSMDInner])
	nEmbd := int64(params.Ints[ParamNEmbd])

	qkvMixed := ggml.Reshape3D(ctx, ggml.MulMat(ctx, weights[WeightAttnQKV], cur), cc, nNew, nSeqs)
	z := ggml.MulMat(ctx, weights[WeightAttnGate], cur)

	beta := ggml.Sigmoid(ctx, ggml.Reshape4D(ctx, ggml.MulMat(ctx, weights[WeightSSMBeta], cur), 1, dtRank, nNew, nSeqs))

	alpha := ggml.Reshape3D(ctx, ggml.MulMat(ctx, weights[WeightSSMAlpha], cur), dtRank, nNew, nSeqs)
	alpha = ggml.Softplus(ctx, ggml.Add(ctx, alpha, weights[WeightSSMDTBias]))
	g := ggml.Reshape4D(ctx, ggml.Mul(ctx, alpha, weights[WeightSSMA]), 1, dtRank, nNew, nSeqs)

	// Conv1d with cached conv state
	convSt3d := ggml.Reshape3D(ctx, cache.Tensors[CacheConvState], dConv-1, cc, nSeqs)
	qkvT := ggml.Transpose(ctx, qkvMixed)
	convInput := ggml.Concat(ctx, convSt3d, qkvT, 0)

	// Conv state writeback: copy new tail of convInput into cache (in-graph GPU copy).
	// Gated on WriteSSM: PMM shadow fork-variant passes set this false to read
	// mainline conv_state without advancing it; persistent-variant sets true to
	// capture the post-shadow state for the next decode token.
	if inputs.WriteSSM {
		newConvSt := ggml.View3D(ctx, convInput, dConv-1, cc, nSeqs,
			convInput.Nb(1), convInput.Nb(2),
			int(convInput.Ne(0)-(dConv-1))*convInput.ElementSize())
		newConvStCont := ggml.Cont(ctx, newConvSt)
		gf.BuildForwardExpand(ggml.Cpy(ctx, newConvStCont, cache.Tensors[CacheConvState]))
	}

	convOut := ggml.Silu(ctx, ggml.SSMConv(ctx, convInput, weights[WeightSSMConv1D]))

	// Split Q, K, V
	qkvDim := dState*nGroup*2 + hvDim*dtRank
	nb1QKV := ggml.RowSize(ggml.TypeF32, qkvDim)

	qSSM := ggml.View4D(ctx, convOut, dState, nGroup, nNew, nSeqs,
		ggml.RowSize(ggml.TypeF32, dState), nb1QKV, nb1QKV*int(nNew), 0)
	kSSM := ggml.View4D(ctx, convOut, dState, nGroup, nNew, nSeqs,
		ggml.RowSize(ggml.TypeF32, dState), nb1QKV, nb1QKV*int(nNew),
		int(dState*nGroup)*convOut.ElementSize())
	vSSM := ggml.View4D(ctx, convOut, hvDim, dtRank, nNew, nSeqs,
		ggml.RowSize(ggml.TypeF32, hvDim), nb1QKV, nb1QKV*int(nNew),
		ggml.RowSize(ggml.TypeF32, 2*dState*nGroup))

	qSSM = ggml.L2Norm(ctx, qSSM, rmsEps)
	kSSM = ggml.L2Norm(ctx, kSSM, rmsEps)

	if nGroup != dtRank {
		qSSM = ggml.Repeat4D(ctx, qSSM, dState, dtRank, nNew, nSeqs)
		kSSM = ggml.Repeat4D(ctx, kSSM, dState, dtRank, nNew, nSeqs)
	}

	// Delta-net with cached SSM state. ggml gated_delta_net contract (v0.12.0+)
	// requires state shape [D, K, n_seqs] with D = S_v*S_v*H and K = snapshot
	// slots (1 — no spec-decode snapshots). Same cache buffer (D*nSeqs elements),
	// reshaped from the old [S_v, S_v, H, nSeqs] view; the output parsing and
	// state writeback below are unchanged (the K=1 output layout matches).
	ssmStateIn := ggml.Reshape3D(ctx, cache.Tensors[CacheSSMState], hvDim*hvDim*dtRank, 1, nSeqs)
	deltaOut := ggml.GatedDeltaNet(ctx, qSSM, kSSM, vSSM, g, beta, ssmStateIn)

	ssmOutput := ggml.View4D(ctx, deltaOut, hvDim, dtRank, nNew, nSeqs,
		ggml.RowSize(ggml.TypeF32, hvDim), ggml.RowSize(ggml.TypeF32, hvDim*dtRank),
		ggml.RowSize(ggml.TypeF32, hvDim*dtRank*nNew), 0)

	// SSM state writeback: copy updated state back to cache (in-graph GPU copy).
	// Gated on WriteSSM (see conv state writeback above for rationale).
	if inputs.WriteSSM {
		newSSMSt := ggml.View4D(ctx, deltaOut, hvDim, hvDim, dtRank, nSeqs,
			ggml.RowSize(ggml.TypeF32, hvDim), ggml.RowSize(ggml.TypeF32, hvDim*hvDim),
			ggml.RowSize(ggml.TypeF32, hvDim*hvDim*dtRank),
			ggml.RowSize(ggml.TypeF32, hvDim*dtRank*nNew*nSeqs))
		newSSMCont := ggml.Cont(ctx, newSSMSt)
		gf.BuildForwardExpand(ggml.Cpy(ctx, newSSMCont, cache.Tensors[CacheSSMState]))
	}

	// Gated normalization
	z4d := ggml.Reshape4D(ctx, z, hvDim, dtRank, nNew, nSeqs)
	normOut := ssmOutput
	if !weights[WeightSSMNorm].IsNil() {
		normOut = rmsNormApply(ctx, ssmOutput, weights[WeightSSMNorm], rmsEps)
	}
	cur = ggml.Mul(ctx, normOut, ggml.Silu(ctx, z4d))

	cur = ggml.Reshape2D(ctx, cur, dInner, nNew*nSeqs)
	cur = ggml.MulMat(ctx, weights[WeightSSMOut], cur)
	cur = ggml.Reshape2D(ctx, cur, nEmbd, nNew*nSeqs)
	return cur
}
