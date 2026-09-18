package arch

// graph_rlb.go — RLB-specific graph helpers.
//
// Three exported methods on *GenericModel for per-block RLB prefill:
//
//   RLBForwardEmbed   — token embedding lookup (+ optional embed scale)
//   RLBForwardLayer   — single-layer forward pass on a host-side hidden state
//   RLBProjectLogits  — output_norm + output matmul on a host-side hidden state
//
// All three use a lazily-allocated rlbScratch context/scheduler that is
// intentionally separate from the main-path cachedCtx/cachedSched. This
// guarantees that RLB graph builds never corrupt the main-path scheduler
// state, at the cost of ~100-140 lines of duplicated graph-building glue.
// The duplication is explicit and accepted (see Decision A in
// memory/rlb_block_recurrence_rewrite.md).
//
// Thread safety: rlbScratch is initialized once via sync.Once. The methods
// themselves are not safe for concurrent use — the caller (generateRLB) is
// single-threaded.

import (
	"fmt"
	"math"
	"sync"
	"unsafe"

	ggml "inference-lab-bench/internal/ggml"
)

// rlbScratch holds the lazily-allocated ggml context and scheduler for RLB
// graph helpers. Never shared with the main-path cachedCtx/cachedSched.
type rlbScratch struct {
	ctx   *ggml.GraphContext
	sched *ggml.Sched
}

// rlbState owns all RLB-specific state on a GenericModel, isolated from the
// main inference path. Keeping it in a sub-struct means model.go never needs
// to know about RLB internals — the only RLB concern leaking into model.go is
// the single rlb field and the Free() call in cleanup paths.
type rlbState struct {
	// BlockRanges holds precomputed per-block layer ranges for per-block RLB
	// prefill. [blockIdx] = {firstLayer, lastLayerInclusive}. Populated at
	// model init time from the arch's full_attn_interval parameter; nil for
	// arches without full-attn cadence (the RLB driver falls back to a single
	// block covering all layers).
	BlockRanges [][2]int

	// scratch and once manage the lazily-allocated ggml context and scheduler
	// for RLB graph helpers. Isolated from main-path cachedCtx/cachedSched;
	// never touched by ForwardCached.
	scratch *rlbScratch
	once    sync.Once
}

// InitBlockRanges populates BlockRanges from the arch's full_attn_interval
// cadence. Each block spans fullAttnInterval layers ending on a full-attn
// layer. If fullAttnInterval is zero or negative, BlockRanges is left nil and
// the RLB driver falls back to a single block covering all layers.
func (s *rlbState) InitBlockRanges(nLayers, fullAttnInterval int) {
	if fullAttnInterval <= 0 {
		return
	}
	for start := 0; start < nLayers; start += fullAttnInterval {
		end := start + fullAttnInterval - 1
		if end >= nLayers {
			end = nLayers - 1
		}
		s.BlockRanges = append(s.BlockRanges, [2]int{start, end})
	}
}

// Free releases the lazily-allocated scratch context/scheduler, if any.
// Idempotent and safe to call from both the error-cleanup path and Close().
func (s *rlbState) Free() {
	if s.scratch != nil {
		s.scratch.sched.Free()
		s.scratch.ctx.Free()
		s.scratch = nil
	}
}

// RLBBlockRanges returns the precomputed per-block layer ranges used by the
// per-block RLB prefill driver. Empty if the arch has no full_attn_interval
// cadence, in which case the driver falls back to a single block covering all
// layers.
func (m *GenericModel) RLBBlockRanges() [][2]int {
	return m.rlb.BlockRanges
}

// ensureRLBScratch initializes m.rlb.scratch exactly once on the first RLB
// call. Sized identically to the main-path resources in createComputeResources.
func (m *GenericModel) ensureRLBScratch() error {
	var initErr error
	m.rlb.once.Do(func() {
		ctx := ggml.NewGraphContext(graphCtxSize(), ggml.AllocPermDisallow)
		if ctx == nil {
			initErr = fmt.Errorf("rlb: failed to create graph context")
			return
		}
		sched := ggml.NewSched(m.Store.GPU, m.Store.CPU, maxGraphNodes)
		if sched == nil {
			ctx.Free()
			initErr = fmt.Errorf("rlb: failed to create scheduler")
			return
		}
		m.rlb.scratch = &rlbScratch{ctx: ctx, sched: sched}
	})
	return initErr
}

// RLBForwardEmbed builds a 1-op graph doing token embedding lookup
// (+ optional embed scale per arch params). Returns [n_embd * nTokens] float32,
// row-major: token i occupies result[i*n_embd : (i+1)*n_embd].
func (m *GenericModel) RLBForwardEmbed(tokens []int32) ([]float32, error) {
	if err := m.ensureRLBScratch(); err != nil {
		return nil, err
	}

	nEmbd := m.Params.Ints[ParamNEmbd]
	nTokens := int64(len(tokens))

	sc := m.rlb.scratch
	sc.ctx.Reset()
	sc.sched.Reset()
	gctx := sc.ctx

	inpTokens := ggml.NewTensor1D(gctx, ggml.TypeI32, nTokens)
	ggml.SetInput(inpTokens)

	x := ggml.GetRows(gctx, m.Store.Global(WeightTokenEmbd), inpTokens)
	if m.Def.Architecture.EmbedScale {
		x = ggml.Scale(gctx, x, float32(math.Sqrt(float64(nEmbd))))
	}
	ggml.SetOutput(x)

	gf := ggml.NewGraph(gctx, maxGraphNodes)
	gf.BuildForwardExpand(x)

	if !sc.sched.AllocGraph(gf) {
		return nil, fmt.Errorf("rlb embed: failed to allocate graph")
	}

	ggml.TensorSet(inpTokens, unsafe.Pointer(&tokens[0]), 0, inpTokens.Nbytes())

	if status := sc.sched.Compute(gf); status != ggml.StatusSuccess {
		return nil, fmt.Errorf("rlb embed: %w (status=%d)", ErrComputeFailed, status)
	}

	result := make([]float32, int(nTokens)*nEmbd)
	ggml.TensorGet(x, unsafe.Pointer(&result[0]), 0, x.Nbytes())
	return result, nil
}

// RLBForwardLayer runs a single layer as its own graph. Reads hiddenIn
// [n_embd * len(tokenPositions)] float32, runs exactly one block builder
// and its FFN, and writes the post-layer hidden state to hiddenOut.
//
// writeKV / writeSSM control whether attention K/V and SSM state, respectively,
// are written back to the cache during this layer's forward pass. Existing
// callers (RLB prefill) pass true for both — the cache is being filled. PMM
// (Poor Man's Mythos) shadow passes pass false (fork variant) or selectively
// (persistent variant captures SSM via writeSSM=true). See GraphInputs doc.
//
// If logitsOut != nil (length >= n_vocab), also projects through
// output_norm + output matmul and writes logits for the last token
// position into logitsOut.
//
// hiddenIn and hiddenOut may alias — the implementation copies hiddenIn
// before writing hiddenOut when they alias.
//
// Validates:
//   - il in [0, n_layers)
//   - len(hiddenIn) == n_embd * len(tokenPositions)
//   - block does not use shared_kv_group (not supported in RLB v1)
//   - arch does not use per-layer embedding injection (Gemma 4 — unsupported)
func (m *GenericModel) RLBForwardLayer(
	cache *GenericCache, il int,
	hiddenIn []float32, tokenPositions []int32,
	flashAttn bool,
	writeKV, writeSSM bool,
	hiddenOut []float32,
	logitsOut []float32,
) error {
	nLayers := m.Params.Ints[ParamNLayers]
	nEmbd := m.Params.Ints[ParamNEmbd]
	nVocab := m.Params.Ints[ParamNVocab]
	rmsEps := m.Params.Floats[ParamRMSEps]
	nTokens := int64(len(tokenPositions))

	// --- Validation ---
	if il < 0 || il >= nLayers {
		return fmt.Errorf("rlb layer %d: out of range [0, %d)", il, nLayers)
	}
	if len(hiddenIn) != nEmbd*int(nTokens) {
		return fmt.Errorf("rlb layer %d: hiddenIn len %d != n_embd(%d)*nTokens(%d)",
			il, len(hiddenIn), nEmbd, nTokens)
	}
	// Per-layer embedding injection (Gemma 4) is not supported in RLB v1.
	if !m.Store.Global(WeightTokEmbdPerLayer).IsNil() {
		return fmt.Errorf("rlb layer %d: arch uses per-layer embedding injection (Gemma 4); not supported in RLB v1", il)
	}
	// shared_kv_group guard: RLB does not support cross-layer KV sharing.
	blockName := m.LayerBlockNames[il]
	blockDef := m.Def.Blocks[blockName]
	if group := configStrOr(blockDef.Config, ConfigSharedKVGroup, ""); group != "" {
		return fmt.Errorf("rlb layer %d: block %q uses shared_kv_group=%q; not supported in RLB v1",
			il, blockName, group)
	}

	if err := m.ensureRLBScratch(); err != nil {
		return err
	}

	// If hiddenIn and hiddenOut alias, copy hiddenIn before we write to hiddenOut.
	workIn := hiddenIn
	if len(hiddenOut) > 0 && len(hiddenIn) > 0 && &hiddenIn[0] == &hiddenOut[0] {
		workIn = make([]float32, len(hiddenIn))
		copy(workIn, hiddenIn)
	}

	// seqPos for this prefill pass: the first tokenPosition value.
	// For per-block RLB prefill, all passes use seqPos=0 (full inline K/V path).
	seqPos := int(tokenPositions[0])
	nKV := int64(seqPos) + nTokens

	sc := m.rlb.scratch
	sc.ctx.Reset()
	sc.sched.Reset()
	gctx := sc.ctx

	// --- Build input tensors ---
	// x: [n_embd, nTokens] F32 — the hidden state coming in.
	inpHidden := ggml.NewTensor2D(gctx, ggml.TypeF32, int64(nEmbd), nTokens)
	ggml.SetInput(inpHidden)

	// IMROPE decoders (Qwen3-VL) need a [4·nTokens] position buffer. RLB is a
	// text-only research path, so the 4 blocks are all the same sequential
	// positions (the IMROPE→NEOX text reduction). Other archs: 1-D [nTokens].
	inpPos := ggml.NewTensor1D(gctx, ggml.TypeI32, m.decoderPositionCount(nTokens))
	ggml.SetInput(inpPos)

	inpMask := ggml.NewTensor2D(gctx, ggml.TypeF32, nKV, nTokens)
	ggml.SetInput(inpMask)

	swaWindow, hasSWA := m.Params.Ints[ParamSlidingWindow]
	var inpMaskSWA ggml.Tensor
	if hasSWA && swaWindow > 0 {
		inpMaskSWA = ggml.NewTensor2D(gctx, ggml.TypeF32, nKV, nTokens)
		ggml.SetInput(inpMaskSWA)
	}

	effectiveFlashAttn := flashAttn && !m.Def.Architecture.NoFlashAttn
	inputs := &GraphInputs{
		InpPos:       inpPos,
		InpMask:      inpMask,
		InpMaskSWA:   inpMaskSWA,
		NTokens:      nTokens,
		NKV:          nKV,
		SeqPos:       seqPos,
		CurrentLayer: il,
		SharedKV:     &SharedKVState{K: make(map[string]ggml.Tensor), V: make(map[string]ggml.Tensor)},
		FlashAttn:    effectiveFlashAttn,
		WriteKV:      writeKV,
		WriteSSM:     writeSSM,
	}

	// Build graph before layer to allow BuildCached to emit in-graph KV/SSM writes.
	gf := ggml.NewGraph(gctx, maxGraphNodes)

	// --- Attention/SSM block ---
	// Bounds check above guarantees il is in range, so Store.Layer cannot return nil here.
	lt := m.Store.Layer(il)

	x := inpHidden
	xPreBlock := x
	if !lt.Common[WeightAttnNorm].IsNil() {
		cur := rmsNormApply(gctx, x, lt.Common[WeightAttnNorm], rmsEps)
		cur = m.BlockBuilders[il].BuildCached(gctx, gf, cur, lt.Block, m.Params, blockDef.Config, inputs, &cache.Layers[il])
		if !lt.Common[WeightAttnPostNorm].IsNil() {
			cur = rmsNormApply(gctx, cur, lt.Common[WeightAttnPostNorm], rmsEps)
		}
		x = ggml.Add(gctx, xPreBlock, cur)
	}

	// --- FFN block ---
	x = m.buildFFNBlock(gctx, x, lt, il, rmsEps, inputs)

	// layer_output_scale (Gemma 4 dense/MoE)
	if !lt.Common[WeightLayerOutputScale].IsNil() {
		x = ggml.Mul(gctx, x, lt.Common[WeightLayerOutputScale])
	}

	// x is the post-layer hidden state: [n_embd, nTokens]
	ggml.SetOutput(x)

	// --- Optional logits projection (last layer of block only) ---
	var logitsTensor ggml.Tensor
	if logitsOut != nil {
		// Project the last token position through output_norm + output matmul.
		last := ggml.View2D(gctx, x, int64(nEmbd), 1, x.Nb(1), int(nTokens-1)*x.Nb(1))
		xn := rmsNormApply(gctx, last, m.Store.Global(WeightOutputNorm), rmsEps)
		logitsTensor = ggml.MulMat(gctx, m.Store.Global(WeightOutput), xn)
		if cap, ok := m.Params.Floats[ParamLogitSoftcap]; ok && cap > 0 {
			logitsTensor = ggml.Scale(gctx, logitsTensor, 1.0/cap)
			logitsTensor = ggml.Tanh(gctx, logitsTensor)
			logitsTensor = ggml.Scale(gctx, logitsTensor, cap)
		}
		ggml.SetOutput(logitsTensor)
		gf.BuildForwardExpand(logitsTensor)
	}

	gf.BuildForwardExpand(x)

	if !sc.sched.AllocGraph(gf) {
		return fmt.Errorf("rlb layer %d: failed to allocate graph", il)
	}

	// --- Set inputs ---
	// inpHidden is always consumed by the block/FFN graph, so it must be
	// allocated — no guard. The positional/mask inputs are conditional:
	// a single-layer graph for a recurrent_ssm block references neither
	// inpPos nor inpMask, so the scheduler leaves them unallocated and
	// TensorSet would crash in ggml_backend_tensor_set's buf-null assert.
	// Guard each conditional input on TensorHasBuffer.
	ggml.TensorSet(inpHidden, unsafe.Pointer(&workIn[0]), 0, inpHidden.Nbytes())

	if ggml.TensorHasBuffer(inpPos) {
		posData := tokenPositions
		if m.usesImropeDecoder() {
			// Replicate the supplied sequential positions across all 4 IMROPE
			// blocks [t|h|w|e] (text reduction; RLB carries no image spans).
			posData = make([]int32, 4*len(tokenPositions))
			for b := 0; b < 4; b++ {
				copy(posData[b*len(tokenPositions):], tokenPositions)
			}
		}
		ggml.TensorSet(inpPos, unsafe.Pointer(&posData[0]), 0, inpPos.Nbytes())
	}

	if ggml.TensorHasBuffer(inpMask) {
		// RLB + vision is unsupported in v1 (Engine.Generate rejects it
		// upstream); no bidirectional spans here.
		maskData := buildCausalMaskData(cache.maskBuf, nTokens, nKV, seqPos, m.Def.Architecture.NonCausal, nil)
		ggml.TensorSet(inpMask, unsafe.Pointer(&maskData[0]), 0, inpMask.Nbytes())
	}

	if hasSWA && swaWindow > 0 && ggml.TensorHasBuffer(inpMaskSWA) {
		swaMaskData := buildSWAMaskData(cache.swaMaskBuf, nTokens, nKV, seqPos, swaWindow, nil)
		ggml.TensorSet(inpMaskSWA, unsafe.Pointer(&swaMaskData[0]), 0, inpMaskSWA.Nbytes())
	}

	if status := sc.sched.Compute(gf); status != ggml.StatusSuccess {
		return fmt.Errorf("rlb layer %d: %w (status=%d)", il, ErrComputeFailed, status)
	}

	// --- Read back hidden state ---
	if len(hiddenOut) > 0 {
		ggml.TensorGet(x, unsafe.Pointer(&hiddenOut[0]), 0, x.Nbytes())
	}

	// --- Read back logits if requested ---
	if logitsOut != nil && !logitsTensor.IsNil() {
		need := nVocab
		if len(logitsOut) < need {
			need = len(logitsOut)
		}
		buf := make([]float32, nVocab)
		ggml.TensorGet(logitsTensor, unsafe.Pointer(&buf[0]), 0, logitsTensor.Nbytes())
		copy(logitsOut, buf[:need])
	}

	return nil
}

// RLBProjectLogits runs output_norm + output matmul (+ optional softcap)
// on host-side hidden input, producing logits for the final sequence position.
// hiddenIn is [n_embd * nTokens] float32 (row-major, same layout as RLBForwardLayer output).
// Returns a slice of length n_vocab.
func (m *GenericModel) RLBProjectLogits(hiddenIn []float32, nTokens int) ([]float32, error) {
	nEmbd := m.Params.Ints[ParamNEmbd]
	nVocab := m.Params.Ints[ParamNVocab]
	rmsEps := m.Params.Floats[ParamRMSEps]

	if len(hiddenIn) != nEmbd*nTokens {
		return nil, fmt.Errorf("rlb project logits: hiddenIn len %d != n_embd(%d)*nTokens(%d)",
			len(hiddenIn), nEmbd, nTokens)
	}

	if err := m.ensureRLBScratch(); err != nil {
		return nil, err
	}

	sc := m.rlb.scratch
	sc.ctx.Reset()
	sc.sched.Reset()
	gctx := sc.ctx

	inpHidden := ggml.NewTensor2D(gctx, ggml.TypeF32, int64(nEmbd), int64(nTokens))
	ggml.SetInput(inpHidden)

	// Slice last token: [n_embd, 1]
	last := ggml.View2D(gctx, inpHidden, int64(nEmbd), 1, inpHidden.Nb(1), (nTokens-1)*inpHidden.Nb(1))
	xn := rmsNormApply(gctx, last, m.Store.Global(WeightOutputNorm), rmsEps)
	logits := ggml.MulMat(gctx, m.Store.Global(WeightOutput), xn)

	if cap, ok := m.Params.Floats[ParamLogitSoftcap]; ok && cap > 0 {
		logits = ggml.Scale(gctx, logits, 1.0/cap)
		logits = ggml.Tanh(gctx, logits)
		logits = ggml.Scale(gctx, logits, cap)
	}

	ggml.SetOutput(logits)

	gf := ggml.NewGraph(gctx, maxGraphNodes)
	gf.BuildForwardExpand(logits)

	if !sc.sched.AllocGraph(gf) {
		return nil, fmt.Errorf("rlb project logits: failed to allocate graph")
	}

	ggml.TensorSet(inpHidden, unsafe.Pointer(&hiddenIn[0]), 0, inpHidden.Nbytes())

	if status := sc.sched.Compute(gf); status != ggml.StatusSuccess {
		return nil, fmt.Errorf("rlb project logits: %w (status=%d)", ErrComputeFailed, status)
	}

	result := make([]float32, nVocab)
	ggml.TensorGet(logits, unsafe.Pointer(&result[0]), 0, logits.Nbytes())
	return result, nil
}
