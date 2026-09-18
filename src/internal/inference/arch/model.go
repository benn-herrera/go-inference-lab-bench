package arch

import (
	"fmt"

	ggufparser "github.com/gpustack/gguf-parser-go"

	"inference-lab-bench/internal/ggml"
	"inference-lab-bench/internal/log"
)

// GenericModel is a model loaded from a GGUF file using an architecture definition.
type GenericModel struct {
	Def             *ArchDef
	Params          *ResolvedParams
	Weights         *ResolvedWeights
	Store           *WeightStore     // immutable weight storage (read-only after load)
	LayerBlockNames []string         // which block each layer uses
	BlockBuilders   []BlockBuilder   // per-layer block builder
	FFNBuilders     []FFNBuilder     // per-layer FFN builder
	FFNConfigs      []map[string]any // per-layer FFN config (from [ffn.config] / [ffn_alt.config])

	// Vision tower (multimodal models only — nil for unimodal arch). All
	// three fields are co-populated by the loader when ArchDef.Vision is
	// non-nil. Tensors share the WeightStore with the decoder; the
	// per-tower split is logical-name routing only.
	VisionResolved *VisionResolved
	VisionTensors  *VisionTensors
	VisionParams   *VisionParams
	VisionBuilders *VisionBuilders

	// Canonical module map and supporting data for diagram rendering.
	CanonicalModuleMap *ModuleMap
	HeadDim            int // elements per attention head (for flash attention geometry check)
	TensorDims         TensorDimsMap
	ModelPath          string

	// Persistent compute resources for ForwardCached. Created at model load, reused across tokens.
	cachedCtx   *ggml.GraphContext
	cachedSched *ggml.Sched

	// Pre-allocated scratch buffer for the hot decode path (ForwardCached).
	logitBuf []float32 // reused by readLogitsInto each token

	// rlb bundles all RLB-specific state (block ranges, lazy scratch
	// context/scheduler). Defined in graph_rlb.go so RLB internals stay out
	// of this file.
	rlb rlbState
}

// NewGenericModelFromGGUF loads a GGUF model using the named architecture definition.
// archDir is the directory containing architecture definition TOML files.
// If gf is non-nil it is used directly; otherwise the GGUF is parsed from modelPath.
// mmprojPath, when non-empty, attaches a paired vision/audio sidecar GGUF so
// the model loads with multimodal support.
func NewGenericModelFromGGUF(memStats ggml.MemoryStats, maxSeqLen int, archDef *ArchDef, modelPath, archDir string, gf *ggufparser.GGUFFile, mmprojPath string) (*GenericModel, error) {
	if archDef == nil {
		return nil, fmt.Errorf("valid ArchDef required")
	}
	reader, err := NewModelReaderGGUF(archDef, modelPath, gf, mmprojPath)
	if err != nil {
		return nil, err
	}

	return newGenericModelFromReader(memStats, maxSeqLen, reader, archDef, modelPath)
}

// NewGenericModelFromSafetensors loads a safetensors model using the named architecture definition.
// archDir is the directory containing architecture definition TOML and stmap files.
// stDir is the directory containing the safetensors shards and config.json.
// enableMmproj is the --auto-mmproj gate, forwarded to the reader to
// suppress vision-tower setup when false (see NewModelReaderSafetensors).
func NewGenericModelFromSafetensors(memStats ggml.MemoryStats, maxSeqLen int, archDef *ArchDef, stDir, archDir string, enableMmproj bool) (*GenericModel, error) {
	if archDef == nil {
		return nil, fmt.Errorf("valid ArchDef required")
	}
	reader, err := NewModelReaderSafetensors(archDef, stDir, archDir, enableMmproj)
	if err != nil {
		return nil, err
	}

	return newGenericModelFromReader(memStats, maxSeqLen, reader, archDef, stDir)
}

// newGenericModelFromReader is the shared loading path used by both GGUF and safetensors.
// Takes an already-constructed ModelReader and loaded ArchDef. Always closes the
// reader before returning. On error, releases any resources allocated so far.
func newGenericModelFromReader(memStats ggml.MemoryStats, maxSeqLen int, reader ModelReader, def *ArchDef, modelPath string) (*GenericModel, error) {
	b := &genericModelBuilder{
		memStats:  memStats,
		maxSeqLen: maxSeqLen,
		reader:    reader,
		def:       def,
		modelPath: modelPath,
	}
	return b.build()
}

// genericModelBuilder owns the partial state accumulated while constructing a
// GenericModel. Splitting the loader into phase methods on this struct keeps
// the top-level control flow readable without passing a dozen locals around.
//
// Ownership model for cleanup:
//   - Before buildWeightStore(): gpu, cpu, weightCtx, weightBuf (if allocated)
//     are owned by the builder and must be freed individually on error.
//   - After buildWeightStore(): store owns all of the above; store.Close()
//     frees everything in one call.
//   - After createComputeResources(): m.cachedCtx / m.cachedSched are owned by
//     the model; cleanupOnError frees them before closing store.
type genericModelBuilder struct {
	// Inputs
	memStats  ggml.MemoryStats
	maxSeqLen int
	reader    ModelReader
	def       *ArchDef
	modelPath string

	// Resolved architecture
	params      *ResolvedParams
	weights     *ResolvedWeights
	headDim     int
	canonicalMM *ModuleMap
	tensorDims  TensorDimsMap

	// Backends + weight arena — ownership transfers to store once built.
	gpu               *ggml.Backend
	cpu               *ggml.Backend
	weightCtx         *ggml.GraphContext
	weightBuf         *ggml.Buffer
	weightTensorIndex map[string]ggml.Tensor

	// Vision tower state (multimodal only; nil for unimodal arch).
	// Resolved in resolveArch alongside the decoder params/weights; bound
	// to actual tensor handles in buildWeightStore from the same
	// weightTensorIndex the decoder uses.
	visionResolved *VisionResolved
	visionParams   *VisionParams
	visionTensors  *VisionTensors

	// Built model — owns the store and compute resources from phase H onward.
	store *WeightStore
	m     *GenericModel
}

func (b *genericModelBuilder) build() (m *GenericModel, err error) {
	defer b.reader.Close()
	defer func() {
		if err != nil {
			b.cleanupOnError()
		}
	}()

	if err = b.checkMemory(); err != nil {
		return nil, err
	}
	if err = b.resolveArch(); err != nil {
		return nil, err
	}
	if err = b.initBackendsAndArena(); err != nil {
		return nil, err
	}
	if err = b.uploadWeights(); err != nil {
		return nil, err
	}
	if err = b.buildWeightStore(); err != nil {
		return nil, err
	}
	if err = b.assignBuilders(); err != nil {
		return nil, err
	}
	if err = b.createComputeResources(); err != nil {
		return nil, err
	}
	return b.m, nil
}

// cleanupOnError releases any resources the builder has allocated so far.
// Safe to call at any phase — uses store ownership as the cut line.
func (b *genericModelBuilder) cleanupOnError() {
	// Model-scoped compute resources (created after store).
	if b.m != nil {
		if b.m.cachedSched != nil {
			b.m.cachedSched.Free()
			b.m.cachedSched = nil
		}
		if b.m.cachedCtx != nil {
			b.m.cachedCtx.Free()
			b.m.cachedCtx = nil
		}
		b.m.rlb.Free()
	}
	// Store owns gpu/cpu/weightCtx/weightBuf — its Close frees everything.
	if b.store != nil {
		b.store.Close()
		return
	}
	// Store not yet built — free raw resources individually.
	if b.weightBuf != nil {
		b.weightBuf.Free()
	}
	if b.weightCtx != nil {
		b.weightCtx.Free()
	}
	if b.gpu != nil {
		b.gpu.Free()
	}
	if b.cpu != nil {
		b.cpu.Free()
	}
}

// checkMemory verifies the model fits in available VRAM/RAM.
func (b *genericModelBuilder) checkMemory() error {
	memReq := b.reader.MinMemoryRequired(b.maxSeqLen)
	if b.memStats.IsUMA {
		if memReq.UnifiedRAM() > uint64(b.memStats.VRAM.Available()) {
			return fmt.Errorf("insufficient unified RAM min required: %d  available: %d",
				memReq.UnifiedRAM(),
				b.memStats.VRAM.Available(),
			)
		}
		return nil
	}
	if memReq.TotalVRAM() >= uint64(b.memStats.VRAM.Available()) || memReq.OverheadRAM >= uint64(b.memStats.RAM.Available()) {
		return fmt.Errorf("insufficient VRAM/RAM min required: %d/%d  available: %d/%d",
			memReq.TotalVRAM(), memReq.OverheadRAM,
			b.memStats.VRAM.Available(), b.memStats.RAM.Available(),
		)
	}
	return nil
}

// resolveArch resolves params, weights, and builds the canonical module map
// and tensor dims (consumed by gen-arch-diagram).
func (b *genericModelBuilder) resolveArch() error {
	params, err := ResolveParams(b.def, b.reader)
	if err != nil {
		return fmt.Errorf("resolving params: %w", err)
	}
	weights, err := ResolveWeights(b.def, params)
	if err != nil {
		return fmt.Errorf("resolving weights: %w", err)
	}
	b.params = params
	b.weights = weights
	b.headDim = params.Ints[ParamHeadDim]
	b.canonicalMM = BuildModuleMap(b.def, weights)
	b.tensorDims = BuildTensorDimsMap(weights, func(name string) (int64, int64, int64, bool) {
		if s, ok := b.reader.TensorSpec(name); ok {
			return s.Ne[0], s.Ne[1], int64(s.Size), true
		}
		return 0, 0, 0, false
	}, b.headDim)

	// Vision tower (multimodal models only). The arch.toml's [vision]
	// block declares the *possibility* of a vision tower; the model
	// file's own metadata says whether this specific load is multimodal.
	// Drive setup from that metadata flag (`vision.has_encoder` — set
	// by the stmap's [[derived_metadata]] from config.json's
	// `vision_config` block on the safetensors path, or by the mmproj
	// GGUF's `clip.has_vision_encoder` once mmproj loading lands).
	//
	// When the flag is set, *every* declared vision weight must actually
	// be present in the model file — partial coverage is a corrupt
	// upload or a stmap bug, and silently ignoring it would surface
	// later as confusing inference errors. Loud load-time failure with
	// a named missing tensor is the right pattern.
	if b.def.Vision != nil && visionMetadataDeclared(b.reader) {
		if err := validateVisionWeightsPresent(b.def, b.reader); err != nil {
			return fmt.Errorf("vision metadata declares encoder but: %w", err)
		}
		vr, err := ResolveVisionWeights(b.def, params)
		if err != nil {
			return fmt.Errorf("resolving vision weights: %w", err)
		}
		vp, err := ResolveVisionParams(b.def, b.reader)
		if err != nil {
			return fmt.Errorf("resolving vision params: %w", err)
		}
		b.visionResolved = vr
		b.visionParams = vp
	}
	return nil
}

// visionMetadataDeclared returns true when the loaded model file
// declares a vision tower via its own metadata. The canonical signal
// is the `vision.has_encoder` capability flag (populated for
// safetensors via the stmap's `config_key_present` derived op on
// `config.json.vision_config`; populated for mmproj GGUF via the
// upstream `clip.has_vision_encoder` convention once mmproj loading
// lands). Returns false when the flag isn't set or resolves to zero —
// the model is unimodal regardless of what the arch.toml allows.
func visionMetadataDeclared(reader ModelReader) bool {
	v, ok := reader.GetU32(KeyVisionHasEncoder)
	return ok && v != 0
}

// validateVisionWeightsPresent confirms every declared vision-tower
// weight (per-layer common weights + globals + projector) exists in
// the model reader's tensor index. Called only when
// visionMetadataDeclared returns true — at that point a missing weight
// is a real defect, not an "unimodal model" condition. Returns the
// first missing tensor's name so the caller can fail loudly.
func validateVisionWeightsPresent(def *ArchDef, reader ModelReader) error {
	missing := func(name string) error {
		return fmt.Errorf("required vision tensor %q not in model file", name)
	}
	// Globals.
	for _, name := range def.Vision.Weights.Global {
		if _, ok := reader.TensorSpec(name); !ok {
			return missing(name)
		}
	}
	// Projector globals (when declared).
	if def.Projector != nil {
		for _, name := range def.Projector.Weights {
			if _, ok := reader.TensorSpec(name); !ok {
				return missing(name)
			}
		}
	}
	// Per-layer common weights, checked at layer 0 only. (Per-layer
	// uniformity is a structural assumption the encoder forward relies
	// on; verifying layer 0 catches stmap and converter bugs without
	// O(n_layers × n_weights) scanning.)
	prefix := ExpandPrefix(def.Vision.Layers.Prefix, 0)
	for _, suffix := range def.Vision.Layers.CommonWeights {
		full := prefix + suffix
		if _, ok := reader.TensorSpec(full); !ok {
			return missing(full)
		}
	}
	return nil
}

// initBackendsAndArena brings up the GPU/CPU backends, creates the weight
// context, builds the tensor-name → tensor index from reader specs, and
// allocates all weight storage on the GPU.
func (b *genericModelBuilder) initBackendsAndArena() error {
	b.gpu = ggml.GPUInit()
	if b.gpu == nil {
		return fmt.Errorf("failed to init GPU backend")
	}
	b.cpu = ggml.CPUInit()

	nTensors := int64(b.reader.TensorCount())
	ctxSize := int64(ggml.TensorOverhead())*nTensors + 1*1024*1024
	b.weightCtx = ggml.NewGraphContext(int(ctxSize), ggml.AllocPermDisallow)
	if b.weightCtx == nil {
		return fmt.Errorf("failed to create weight context")
	}

	b.weightTensorIndex = make(map[string]ggml.Tensor, nTensors)
	for _, name := range b.reader.TensorNames() {
		spec, ok := b.reader.TensorSpec(name)
		if !ok {
			continue
		}
		t := makeTensorFromSpec(b.weightCtx, spec.Type, spec.Ne[0], spec.Ne[1], spec.Ne[2], spec.Ne[3])
		ggml.SetName(t, name)
		b.weightTensorIndex[name] = t
	}

	b.weightBuf = ggml.AllocCtxTensors(b.weightCtx, b.gpu)
	if b.weightBuf == nil {
		return fmt.Errorf("failed to alloc GPU VRAM")
	}
	return nil
}

// uploadWeights reads each tensor's raw bytes from the reader, runs
// ggml_validate_row_data on them, and uploads them to GPU VRAM.
//
// ggml_validate_row_data inspects each block's scale and delta fields for
// NaN/Inf at near-zero cost on quantized types; for float types it scans every
// element. A single bad block scale in a Q4_K weight propagates silently to
// every output row that block touches during inference, producing the "first
// token EOS / garbage text" symptom that took hours to diagnose in the
// safetensors load path. Catching it here turns that into a loud one-line
// load-time error naming the offending tensor.
func (b *genericModelBuilder) uploadWeights() error {
	for _, name := range b.reader.TensorNames() {
		t, ok := b.weightTensorIndex[name]
		if !ok {
			continue
		}
		spec, _ := b.reader.TensorSpec(name)
		buf := make([]byte, spec.Size)
		if err := b.reader.ReadTensor(name, buf); err != nil {
			return fmt.Errorf("read tensor %s: %w", name, err)
		}
		if !ggml.ValidateRowData(spec.Type, buf) {
			return fmt.Errorf("tensor %s failed ggml_validate_row_data "+
				"(NaN/Inf in raw bytes — see stderr for offending block index; "+
				"likely cause: corrupted weight file or reader type/shape mismatch)", name)
		}
		ggml.TensorSetBytes(t, buf, 0)
	}
	return nil
}

// buildWeightStore wraps the backends + weight arena in a WeightStore,
// resolves global and per-layer tensor maps, handles tied embeddings, and
// validates that each block builder's required weights are present.
//
// On success, the store takes ownership of gpu, cpu, weightCtx, weightBuf —
// subsequent cleanup runs through store.Close() only.
func (b *genericModelBuilder) buildWeightStore() error {
	store := &WeightStore{
		global: make(map[string]ggml.Tensor),
		ctx:    b.weightCtx,
		Buffer: b.weightBuf,
		GPU:    b.gpu,
		CPU:    b.cpu,
	}

	for logicalName, tensorName := range b.weights.Global {
		t, ok := b.weightTensorIndex[tensorName]
		if !ok {
			t = ggml.NilTensor()
		}
		store.global[logicalName] = t
	}

	if b.def.Architecture.TiedEmbeddings && store.Global(WeightOutput).IsNil() {
		store.global[WeightOutput] = store.Global(WeightTokenEmbd)
	}

	// Transfer ownership to store before any error return so cleanupOnError
	// uses store.Close() instead of double-freeing via individual fields.
	b.store = store

	if store.Global(WeightTokenEmbd).IsNil() || store.Global(WeightOutputNorm).IsNil() {
		return fmt.Errorf("missing required global weight tensor: %s or %s", WeightTokenEmbd, WeightOutputNorm)
	}
	if store.Global(WeightOutput).IsNil() {
		return fmt.Errorf("missing required global weight tensor: %s", WeightOutput)
	}

	nLayers := b.params.Ints[ParamNLayers]
	store.layers = make([]LayerTensors, nLayers)

	for i, lw := range b.weights.Layers {
		lt := LayerTensors{
			Common: make(map[string]ggml.Tensor),
			Block:  make(map[string]ggml.Tensor),
			FFN:    make(map[string]ggml.Tensor),
		}

		for logicalName, tensorName := range lw.Common {
			t, ok := b.weightTensorIndex[tensorName]
			if !ok {
				t = ggml.NilTensor()
			}
			lt.Common[logicalName] = t
		}

		blockDef := b.def.Blocks[lw.BlockName]
		for logicalName, tensorName := range lw.Block {
			t, ok := b.weightTensorIndex[tensorName]
			if !ok {
				// Fallback: try the raw suffix as a global tensor name (e.g. rope_freqs.weight)
				if suffix, has := blockDef.Weights[logicalName]; has {
					t, ok = b.weightTensorIndex[suffix]
				}
			}
			if !ok {
				t = ggml.NilTensor()
			}
			lt.Block[logicalName] = t
		}

		// n_kv_shared_layers: non-KV layers still have weights in the model but
		// must NOT compute their own K/V or write to cache. Null out K/V weights
		// so the builder's hasKV check correctly identifies them as non-KV.
		if nKVShared, ok := b.params.Ints[ParamNKVSharedLayers]; ok && nKVShared > 0 {
			nKVFromStart := nLayers - nKVShared
			if i >= nKVFromStart {
				lt.Block[WeightAttnK] = ggml.NilTensor()
				lt.Block[WeightAttnV] = ggml.NilTensor()
			}
		}

		for logicalName, tensorName := range lw.FFN {
			t, ok := b.weightTensorIndex[tensorName]
			if !ok {
				t = ggml.NilTensor()
			}
			lt.FFN[logicalName] = t
		}

		// FFN alt weights
		for logicalName, tensorName := range lw.FFNAlt {
			t, ok := b.weightTensorIndex[tensorName]
			if !ok {
				t = ggml.NilTensor()
			}
			lt.FFN[logicalName] = t
		}

		store.layers[i] = lt

		// Validate that all required weights for this layer's block builder are present.
		if bb, ok := GetBlockBuilder(blockDef.Builder); ok {
			contract := bb.Contract()
			for _, req := range contract.RequiredWeights {
				if lt.Block[req].IsNil() {
					return fmt.Errorf("layer %d (block %q): required weight %q is missing from model", i, lw.BlockName, req)
				}
			}
		}
	}

	// Vision tower: bind resolved vision weight names to the loaded
	// tensor handles from the same weightTensorIndex the decoder uses.
	// All vision tensors share the WeightStore's GPU buffer; the
	// per-tower split is logical-name routing only. Any missing
	// resolved name becomes NilTensor and is the caller's problem
	// (BuildVisionGraph validates the must-haves up front).
	if b.visionResolved != nil {
		b.visionTensors = BuildVisionTensors(b.visionResolved, b.weightTensorIndex)

		// Attach Gemma4ClippableLinear clamp scalars. Read through the
		// ModelReader so a single loader serves both GGUF and safetensors
		// (clamp scalars are exposed under canonical names by both readers).
		clamps, err := LoadVisionClampsFromReader(b.reader, b.def, b.visionResolved)
		if err != nil {
			log.Warn("vision clamp load: %v", err)
		} else if clamps != nil {
			b.visionTensors.Clamps = clamps
		}
	}

	return nil
}

// assignBuilders instantiates the model struct, assigns per-layer block and
// FFN builders (including per-layer alt-FFN routing via layerHasAltFFN), and
// emits the load summary log.
func (b *genericModelBuilder) assignBuilders() error {
	nLayers := b.params.Ints[ParamNLayers]

	m := &GenericModel{
		Def:                b.def,
		Params:             b.params,
		Weights:            b.weights,
		Store:              b.store,
		CanonicalModuleMap: b.canonicalMM,
		HeadDim:            b.headDim,
		TensorDims:         b.tensorDims,
		ModelPath:          b.modelPath,
		VisionResolved:     b.visionResolved,
		VisionTensors:      b.visionTensors,
		VisionParams:       b.visionParams,
	}
	b.m = m

	// Vision tower builders (multimodal only). Mirrors the decoder block/FFN
	// resolution above: pick the uniform vision block + FFN builders from
	// [vision.blocks]/[vision.ffn] and capture their weight remaps. nil for
	// unimodal archs (visionParams nil).
	if b.visionParams != nil {
		vb, err := ResolveVisionBuilders(b.def, b.visionParams)
		if err != nil {
			return fmt.Errorf("resolving vision builders: %w", err)
		}
		m.VisionBuilders = vb
	}

	m.LayerBlockNames = make([]string, nLayers)
	m.BlockBuilders = make([]BlockBuilder, nLayers)
	for i, lw := range b.weights.Layers {
		m.LayerBlockNames[i] = lw.BlockName
		bb, ok := GetBlockBuilder(b.def.Blocks[lw.BlockName].Builder)
		if !ok {
			return fmt.Errorf("layer %d: unknown block builder %q", i, b.def.Blocks[lw.BlockName].Builder)
		}
		m.BlockBuilders[i] = bb
	}

	fb, ok := GetFFNBuilder(b.def.FFN.Builder)
	if !ok {
		return fmt.Errorf("unknown FFN builder %q", b.def.FFN.Builder)
	}
	var fbAlt FFNBuilder
	if b.def.FFNAlt != nil {
		fbAlt, ok = GetFFNBuilder(b.def.FFNAlt.Builder)
		if !ok {
			return fmt.Errorf("unknown FFN alt builder %q", b.def.FFNAlt.Builder)
		}
	}
	m.FFNBuilders = make([]FFNBuilder, nLayers)
	m.FFNConfigs = make([]map[string]any, nLayers)
	for i, lw := range b.weights.Layers {
		// Use alt FFN if this layer has alt weights in the model
		if fbAlt != nil && len(lw.FFNAlt) > 0 && layerHasAltFFN(b.store, lw) {
			m.FFNBuilders[i] = fbAlt
			m.FFNConfigs[i] = b.def.FFNAlt.Config
		} else {
			m.FFNBuilders[i] = fb
			m.FFNConfigs[i] = b.def.FFN.Config
		}
	}

	// Compute block boundaries for per-block RLB. InitBlockRanges is a no-op
	// when full_attn_interval is absent or zero; the per-block driver then
	// falls back to a single block covering all layers.
	m.rlb.InitBlockRanges(nLayers, b.params.Ints[ParamFullAttnInterval])

	blockCounts := make(map[string]int)
	for _, name := range m.LayerBlockNames {
		blockCounts[name]++
	}
	var blockSummary string
	for name, count := range blockCounts {
		if blockSummary != "" {
			blockSummary += " + "
		}
		blockSummary += fmt.Sprintf("%d %s", count, name)
	}
	log.Info("%s: %d layers (%s), %s=%d, %s=%d/%d, %s=%d",
		b.def.Architecture.Name, nLayers, blockSummary,
		ParamNEmbd, b.params.Ints[ParamNEmbd], ParamNHeads, b.params.Ints[ParamNHeads], b.params.Ints[ParamNKVHeads], ParamHeadDim, b.params.Ints[ParamHeadDim])
	return nil
}

// createComputeResources allocates the cached forward-pass graph context,
// scheduler, and scratch buffers reused across decode tokens.
func (b *genericModelBuilder) createComputeResources() error {
	b.m.cachedCtx = ggml.NewGraphContext(graphCtxSize(), ggml.AllocPermDisallow)
	if b.m.cachedCtx == nil {
		return fmt.Errorf("failed to create cached graph context")
	}
	b.m.cachedSched = ggml.NewSched(b.store.GPU, b.store.CPU, maxGraphNodes)
	if b.m.cachedSched == nil {
		return fmt.Errorf("failed to create cached scheduler")
	}
	b.m.logitBuf = make([]float32, b.params.Ints[ParamNVocab])
	return nil
}

// MemoryStats returns the model's current memory allocation.
func (m *GenericModel) MemoryStats() (weightBytes uint64, cacheBytes uint64) {
	if m.Store == nil || m.Store.Buffer == nil {
		return 0, 0
	}
	// WeightStore.Buffer.Size() returns the total allocated GPU buffer.
	bufSize := m.Store.Buffer.Size()

	// We don't have a precise split between weights and cache in the buffer,
	// but we can estimate: weight size is the sum of all tensor specs.
	// For now, the whole buffer is reported as allocated.
	weightBytes = uint64(bufSize)
	cacheBytes = 0 // Cache is included in the same buffer; no separate accounting.
	return weightBytes, cacheBytes
}

// layerHasAltFFN checks if the alt FFN weights actually exist in the GGUF for this layer.
// Skips alt weights whose GGUF tensor name also appears in standard FFN, block, or common
// weights — those exist in both dense and MoE models and can't distinguish them.
func layerHasAltFFN(store *WeightStore, lw ResolvedLayerWeights) bool {
	// Collect GGUF tensor names used by non-alt weight sources.
	shared := make(map[string]struct{})
	for _, tn := range lw.Common {
		shared[tn] = struct{}{}
	}
	for _, tn := range lw.Block {
		shared[tn] = struct{}{}
	}
	for _, tn := range lw.FFN {
		shared[tn] = struct{}{}
	}

	lt := store.layers[lw.Index]
	for logicalName, tensorName := range lw.FFNAlt {
		if _, ok := shared[tensorName]; ok {
			continue
		}
		if t := lt.FFN[logicalName]; !t.IsNil() {
			return true
		}
	}
	return false
}

// Close releases all GPU resources.
func (m *GenericModel) Close() {
	if m.cachedSched != nil {
		m.cachedSched.Free()
		m.cachedSched = nil
	}
	if m.cachedCtx != nil {
		m.cachedCtx.Free()
		m.cachedCtx = nil
	}
	m.rlb.Free()
	if m.Store != nil {
		m.Store.Close()
		m.Store = nil
	}
}

// --- Pure-Go GGUF reader types ---

// tensorSpec holds tensor metadata extracted from gguf-parser-go,
// replacing the C shape context. Dimensions are padded to 4 elements (unused = 1).
type tensorSpec struct {
	Type ggml.GGMLType // ggml type (matches ggml.NewTensor*D)
	Ne   [4]int64      // dims; unused = 1
	Size int           // total bytes
}

// goGGUFReader provides GGUF metadata access using gguf-parser-go.
// Embedded in ggufModelReader to supply the metadata-reading methods of
// ModelReader (GetU32, GetF32, GetArrInts, GetArrBools, GetTensorDim).
type goGGUFReader struct {
	kvs         ggufparser.GGUFMetadataKVs
	tensorSpecs map[string]tensorSpec
}

// getBool reads a GGUF boolean scalar. Lowercase because it's reader-
// internal — callers reach it via the public GetU32 BOOL→U32 promotion
// path (mmproj's `clip.has_vision_encoder` is BOOL but we expose it as
// the U32 flag `vision.has_encoder`).
func (r *goGGUFReader) getBool(key string) (bool, bool) {
	kv, ok := r.kvs.Get(key)
	if !ok {
		return false, false
	}
	if kv.ValueType != ggufparser.GGUFMetadataValueTypeBool {
		return false, false
	}
	if b, ok := kv.Value.(bool); ok {
		return b, true
	}
	return false, false
}

func (r *goGGUFReader) GetU32(key string) (uint32, bool) {
	kv, ok := r.kvs.Get(key)
	if !ok {
		return 0, false
	}
	if kv.ValueType != ggufparser.GGUFMetadataValueTypeUint32 {
		return 0, false
	}
	return kv.ValueUint32(), true
}

func (r *goGGUFReader) GetF32(key string) (float32, bool) {
	kv, ok := r.kvs.Get(key)
	if !ok {
		return 0, false
	}
	if kv.ValueType != ggufparser.GGUFMetadataValueTypeFloat32 {
		return 0, false
	}
	return kv.ValueFloat32(), true
}

func (r *goGGUFReader) GetArrInts(key string) ([]int, bool) {
	kv, ok := r.kvs.Get(key)
	if !ok {
		return nil, false
	}
	if kv.ValueType != ggufparser.GGUFMetadataValueTypeArray {
		return nil, false
	}
	av := kv.ValueArray()
	switch av.Type {
	case ggufparser.GGUFMetadataValueTypeInt32:
		raw := av.ValuesInt32()
		arr := make([]int, len(raw))
		for i, v := range raw {
			arr[i] = int(v)
		}
		return arr, len(arr) > 0
	case ggufparser.GGUFMetadataValueTypeUint32:
		raw := av.ValuesUint32()
		arr := make([]int, len(raw))
		for i, v := range raw {
			arr[i] = int(v)
		}
		return arr, len(arr) > 0
	default:
		return nil, false
	}
}

func (r *goGGUFReader) GetArrBools(key string) ([]bool, bool) {
	kv, ok := r.kvs.Get(key)
	if !ok {
		return nil, false
	}
	if kv.ValueType != ggufparser.GGUFMetadataValueTypeArray {
		return nil, false
	}
	av := kv.ValueArray()
	if av.Type != ggufparser.GGUFMetadataValueTypeBool {
		return nil, false
	}
	arr := av.ValuesBool()
	return arr, len(arr) > 0
}

func (r *goGGUFReader) GetTensorDim(name string, dim int) (int64, bool) {
	spec, ok := r.tensorSpecs[name+".weight"]
	if !ok {
		spec, ok = r.tensorSpecs[name]
	}
	if !ok {
		return 0, false
	}
	if dim < 0 || dim > 3 {
		return 0, false
	}
	return spec.Ne[dim], true
}

// buildTensorSpecs constructs a map of tensor name → tensorSpec from a parsed GGUF file.
// Dimensions are padded to 4 elements; missing trailing dimensions default to 1.
func buildTensorSpecs(f *ggufparser.GGUFFile) map[string]tensorSpec {
	specs := make(map[string]tensorSpec, len(f.TensorInfos))
	for _, ti := range f.TensorInfos {
		var ne [4]int64
		ne[0], ne[1], ne[2], ne[3] = 1, 1, 1, 1
		for d := 0; d < len(ti.Dimensions) && d < 4; d++ {
			ne[d] = int64(ti.Dimensions[d])
		}
		specs[ti.Name] = tensorSpec{
			Type: ggml.GGMLType(ti.Type),
			Ne:   ne,
			Size: int(ti.Bytes()),
		}
	}
	return specs
}
