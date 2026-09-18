package arch

import (
	"maps"

	ggml "inference-lab-bench/internal/ggml"
)

// CaptureFlags is a bitfield controlling what intermediate values ForwardStateless collects.
type CaptureFlags uint32

const (
	// CaptureAttnWeights captures the post-softmax attention weight matrix per layer.
	CaptureAttnWeights CaptureFlags = 1 << iota
	// CaptureNamed enables NamedTensor(name, t) recording. Used for the
	// Phase 8 numerical-equivalence workflow: dump named intermediate
	// tensors from the forward pass for side-by-side diff against
	// llama-mtmd's matching cb() callbacks.
	CaptureNamed
)

// ForwardCaptures is passed via GraphInputs.Captures to collect inference-time tensors.
// Nil ForwardCaptures (the default) adds zero overhead to the forward pass.
// Results are populated before ForwardStateless returns, after compute is complete.
type ForwardCaptures struct {
	Flags CaptureFlags
	// AttnWeights[il] is the flat post-softmax attention matrix for layer il.
	// Shape: [nHeads * nKV * nTokens]float32 (ggml layout: nKV fastest).
	// Nil for layers that do not use scaledDotProductAttention (SSM, MLA).
	AttnWeights [][]float32
	// NHeads and NTokens are set from the forward pass, needed to interpret AttnWeights.
	NHeads  int64
	NTokens int64

	// NamedTensors holds the flat float32 contents of each captured tensor,
	// keyed by the name passed to NamedTensor(). Names are caller-defined
	// (e.g. "vision.patch_embd", "vision.layer_0.attn_out"); values are
	// row-major flat slices following ggml ne-order conventions. Populated
	// after compute when CaptureNamed is set.
	NamedTensors map[string][]float32

	// currentLayer is set by the blkFn closure before each block builder call
	// so scaledDotProductAttention can write into the right attnTensors slot.
	currentLayer int
	// attnTensors[il] holds the captured kq tensor for layer il; nil if layer has no SDPA.
	// Pre-allocated to nLayers; filled during graph construction; read after compute.
	attnTensors []ggml.Tensor

	// namedTensorRefs holds the in-graph tensor handles captured by
	// NamedTensor() during graph construction. Read back after sched.Compute
	// and stored into NamedTensors.
	namedTensorRefs map[string]ggml.Tensor
}

// NamedTensor marks `t` as a graph output and records it under `name`
// for readback after compute. No-op when caps is nil or CaptureNamed is
// off — zero cost on the production path. Caller is responsible for
// keeping names unique within a forward pass (later calls with the same
// name overwrite earlier ones).
//
// Each capture is funneled through `ggml.Cont` to give it an
// independent contiguous buffer. Without that step, the scheduler can
// reuse the source tensor's buffer for downstream computation,
// overwriting earlier captures with later layers' data. Matches the
// pattern in scaledDotProductAttention's `kqCap := ggml.Cont(...)`
// path that powers CaptureAttnWeights.
func (caps *ForwardCaptures) NamedTensor(gctx *ggml.GraphContext, gf *ggml.Graph, name string, t ggml.Tensor) {
	if caps == nil || caps.Flags&CaptureNamed == 0 {
		return
	}
	if t.IsNil() {
		return
	}
	if caps.namedTensorRefs == nil {
		caps.namedTensorRefs = make(map[string]ggml.Tensor)
	}
	cont := ggml.Cont(gctx, t)
	ggml.SetOutput(cont)
	gf.BuildForwardExpand(cont)
	caps.namedTensorRefs[name] = cont
}

// SharedKVState passes in-graph K/V tensors from KV layers to non-KV layers.
// KV layers update these after computing K/V; non-KV layers read them for attention.
type SharedKVState struct {
	K map[string]ggml.Tensor // group → latest K
	V map[string]ggml.Tensor // group → latest V
}

// GraphInputs holds shared input tensors for the forward pass.
//
// WriteKV / WriteSSM contract: every BlockBuilder.BuildCached implementation that
// emits a cache writeback (K/V tensor cpy or SSM state cpy) MUST gate the
// writeback on the corresponding flag. Vanilla / mainline passes set both to
// true; PMM (Poor Man's Mythos) shadow passes set them to false (fork variant)
// or selectively (persistent variant captures SSM via WriteSSM=true). Cache
// READS are not gated — only writes — because shadow passes read the mainline
// state and must be free to do so without snapshot/restore plumbing in the
// graph builders. Defaults are false; ForwardCached and the RLB layer dispatch
// must set both explicitly. Missing a write-gate breaks PMM silently;
// TestKVCacheWritePassMatchesVanillaForward is the canary.
type GraphInputs struct {
	InpPos       ggml.Tensor
	InpMask      ggml.Tensor
	InpMaskSWA   ggml.Tensor      // sliding-window attention mask (nil if unused)
	NTokens      int64            // number of new tokens being processed
	NKV          int64            // total KV length (seqPos + nNew for cached, nTokens for stateless)
	SeqPos       int              // cache position (0 for stateless)
	Captures     *ForwardCaptures // nil = no capture
	SharedKV     *SharedKVState
	CurrentLayer int  // set by runLayers before calling the block builder
	FlashAttn    bool // use ggml_flash_attn_ext instead of explicit KQ+softmax+V matmul
	// WriteKV controls whether attention layers write K/V to the persistent cache.
	// Must be true for all vanilla/mainline passes; false for shadow (Stream B) passes.
	// Every BlockBuilder.BuildCached that emits KV writes MUST gate the write on this flag.
	WriteKV bool
	// WriteSSM controls whether SSM layers write conv_state/ssm_state back to cache.
	// False for fork-variant shadow passes; true for persistent-variant (capture needed).
	WriteSSM bool
	// PosX / PosY are per-token X/Y grid positions for axial (2D) RoPE
	// (rope="axial2d"). Nil for every decoder path — only a 2D-positional
	// tower (e.g. a ViT encoder) supplies them.
	PosX ggml.Tensor
	PosY ggml.Tensor
	// InpPosVision is the 4-channel vision M-RoPE position buffer for
	// rope="mrope_vision" (Qwen3-VL tower). A single I32 tensor of length
	// 4*n_patches laid out channel-major [y,x,y,x] (see VisionMRopePositions),
	// fed directly to ggml_rope_multi with GGML_ROPE_TYPE_VISION. Nil for every
	// non-vision-mrope path (decoder, axial2d, none) — only the Qwen3-VL vision
	// graph supplies it (wired in P5).
	InpPosVision ggml.Tensor
	// LinearClamps optionally maps a builder weight key (e.g. WeightAttnQ,
	// WeightFFNDown) to the input/output clamp applied around that weight's
	// matmul. Nil for every decoder path (Go zero value), so every lookup
	// yields an inactive LinearClamp and mulMatClamped degrades to a bare
	// ggml.MulMat — byte-identical to the un-clamped decoder graph. Only the
	// Gemma-4 vision tower, whose activations grow unboundedly without the
	// clamps, populates this map.
	LinearClamps map[string]LinearClamp
}

// LayerCache holds per-layer cache tensors.
type LayerCache struct {
	Tensors     map[string]ggml.Tensor // cache name → tensor (e.g. "k", "v", "conv_state", "ssm_state")
	MaxSeqLen   int
	SharedGroup string // shared group name (empty = not shared)
}

// BuilderKind categorizes a builder for palette routing and diagram rendering.
// The zero value is intentionally invalid — an unset Kind is a programming error.
type BuilderKind int

const (
	KindAttention    BuilderKind = iota + 1
	KindSWAAttention             // sliding-window / gated attention (palette: swa green)
	KindRecurrent
	KindFFN
)

// BuilderContract declares the expected weights, params, and config for a builder.
// Used by Validate() to catch mismatches at TOML load time.
type BuilderContract struct {
	Kind            BuilderKind         // block category — must be set by every builder
	RequiredWeights []string            // weight keys that must be present
	OptionalWeights []string            // weight keys that may be present
	RequiredParams  []string            // param names read from ResolvedParams
	ConfigSchema    map[string][]string // config key → valid values (nil slice = any value)
	ExpertRouted    bool                // true for MoE builders that route to expert weight banks
}

// BlockBuilder constructs ggml graph subgraphs for attention/SSM blocks.
type BlockBuilder interface {
	Contract() BuilderContract

	BuildStateless(ctx *ggml.GraphContext, input ggml.Tensor, weights map[string]ggml.Tensor,
		params *ResolvedParams, config map[string]any, inputs *GraphInputs,
		zeroFill *[]ggml.Tensor) ggml.Tensor

	BuildCached(ctx *ggml.GraphContext, gf *ggml.Graph, input ggml.Tensor, weights map[string]ggml.Tensor,
		params *ResolvedParams, config map[string]any, inputs *GraphInputs,
		cache *LayerCache) ggml.Tensor
}

// FFNBuilder constructs ggml graph subgraphs for feed-forward networks.
//
// inputs carries the shared per-pass graph inputs; FFN builders use it only
// for GraphInputs.LinearClamps (the clamp-around-matmul map). It is nil-safe
// via clampFor — every decoder path passes a GraphInputs whose LinearClamps
// is nil, so the clamps degrade to bare matmuls (byte-identical graph).
type FFNBuilder interface {
	Contract() BuilderContract

	BuildFFN(ctx *ggml.GraphContext, input ggml.Tensor, weights map[string]ggml.Tensor,
		params *ResolvedParams, config map[string]any, inputs *GraphInputs) ggml.Tensor
}

// Builder registries
var blockBuilders = map[string]BlockBuilder{}
var ffnBuilders = map[string]FFNBuilder{}

func init() {
	blockBuilders["attention"] = &AttentionBuilder{}
	blockBuilders["mla_attention"] = &MLAAttentionBuilder{}
	blockBuilders["full_attention_gated"] = &FullAttentionGatedBuilder{}
	blockBuilders["gated_delta_net"] = &GatedDeltaNetBuilder{}
	ffnBuilders["swiglu"] = &gluBuilder{activation: ActivationSiLU}
	ffnBuilders["geglu"] = &gluBuilder{activation: ActivationGELU}
	// geglu_quick: gated FFN with quick-GELU (x*sigmoid(1.702x)) on the gate,
	// matching CLIP-style vision encoders (llama.cpp FFN_GELU_QUICK). Gemma 4's
	// ViT FFN uses this; the text decoder stays on tanh-GELU "geglu".
	ffnBuilders["geglu_quick"] = &gluBuilder{activation: ActivationGELUQuick}
	ffnBuilders["mlp"] = &mlpBuilder{}
	ffnBuilders[FFNSymMoE] = &MoEBuilder{}
}

// GetBlockBuilder returns a registered block builder by name.
func GetBlockBuilder(name string) (BlockBuilder, bool) {
	b, ok := blockBuilders[name]
	return b, ok
}

// GetFFNBuilder returns a registered FFN builder by name.
func GetFFNBuilder(name string) (FFNBuilder, bool) {
	b, ok := ffnBuilders[name]
	return b, ok
}

// FFNBuilderIsExpertRouted returns true if the named FFN builder declares ExpertRouted
// in its contract. Returns false for unknown builder names.
func FFNBuilderIsExpertRouted(name string) bool {
	b, ok := ffnBuilders[name]
	if !ok {
		return false
	}
	return b.Contract().ExpertRouted
}

// GetBlockBuilders returns a copy of the block builder registry.
func GetBlockBuilders() map[string]BlockBuilder {
	m := make(map[string]BlockBuilder, len(blockBuilders))
	maps.Copy(m, blockBuilders)
	return m
}

// GetFFNBuilders returns a copy of the FFN builder registry.
func GetFFNBuilders() map[string]FFNBuilder {
	m := make(map[string]FFNBuilder, len(ffnBuilders))
	maps.Copy(m, ffnBuilders)
	return m
}

// ropeSections converts the IntArr rope_sections param to [4]int for ggml.RopeMulti.
func ropeSections(params *ResolvedParams) [4]int {
	arr := params.IntArr[ParamRoPESections]
	var s [4]int
	for i := 0; i < 4 && i < len(arr); i++ {
		s[i] = arr[i]
	}
	return s
}
