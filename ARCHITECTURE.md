# ARCHITECTURE — Inference Lab Bench

Companion to `CONVENTIONS.md`: codebase structure, data flows, and invariants.

---

## Central Invariants

**Zero model-specific Go code.** See SPEC.md §"Architecture Definition Contract" — the requirement and its rationale live there.

**Zero magic hard-coded values** - any string or number that must agree in multiple places must be defined as a constant and referenced only via the constant name. Renaming a map key or DSL file format symbol must be doable via a single edit of a value rather than a global search and replace.

---

## System Diagram

```
HTTP request
    │
    ▼
apiserver/completions.go          parse JSON, extract GenerateParams
    │
    ▼
apiserver/server.go (Engine())    look up loaded model → *inference.Engine
    │
    ▼
inference/engine.go (Generate())
    ├─ tokenizer.go               EncodeChat (Jinja2 chat template from GGUF, via gonja)
    ├─ generateCached / generateStateless / generateDiffusion
    │       │
    │       ▼
    │   arch/graph.go             ForwardCached / ForwardStateless / ForwardStatelessAllLogits → build ggml graph per-layer
    │       ├─ Store.Layer()      resolve per-layer tensors from WeightStore
    │       ├─ BlockBuilders[il]  BuildStateless / BuildCached per layer
    │       └─ FFNBuilders[il]    BuildFFN per layer
    │           │
    │           ▼
    │       ggml/ops_graph.go     Go wrappers → C ggml ops (GPU)
    │
    ├─ sample()                   Greedy or Top-P
    └─ onToken callback           SSE streaming or buffered
```

---

## Package Map

### `bench/` — CLI entry point
Thin cobra command wrappers delegating to `internal/`. `serve-api` exposes `--log`/`--log-level` flags; batch commands (`gen-arch-diagram`) use default INFO-to-stderr logger.

### `internal/log/` — structured logging (leaf package)
Zero-external-dependency. Imports stdlib only — no project package may be imported by it. See **Logging** section.

### `internal/model/` — model discovery
Two-pass directory scan: `*.gguf` glob first, then `*.st/` enumeration via `os.ReadDir`. Architectures with no matching `.arch.toml` are filtered out. `ModelFormat` (`FormatGGUF` / `FormatSafetensors`) is carried on `ModelInfo.Format`; `List()` rescans for new files and `.st/` dirs on each call; `Get()` / `tryLoadOne()` checks `.gguf` before falling back to `.st/`.

Safetensors discovery reads the index for tensor inventory and `config.json` for architecture class and numeric params, resolving the architecture via `arch.FindSTMapByHFClass()`. A missing `config.json` yields "unknown" architecture and zeroed numeric fields; a corrupt one logs a debug warning and continues with partial data.

### `internal/apiserver/` — HTTP surface
Handlers for the API, control, and diagnostic endpoints, plus config loading — there is no separate `internal/config/` package.

`Server` holds an `engines` map (`modelName → *inference.Engine`) guarded by `enginesMu sync.Mutex`, alongside `httpServer`, a `pending` WaitGroup, and an injected `util.BenchPaths`. The mutex is held while looking up, evicting, or creating an engine — single-client R&D, but the lock keeps the eviction/creation race honest. `NewServer(paths, cfg, manager)` takes `BenchPaths` from the cobra entry point — `apiserver` never calls `util.ResolvePaths()` itself.

Endpoint and payload contracts: SPEC.md §"HTTP API". Config keys, defaults, and per-request override precedence: SPEC.md §"Configuration Contract".

### `internal/inference/engine.go`

```go
type DiffusionParams struct {
    Steps       int    // 0 = use default (64)
    BlockLength int    // 0 = single block (global); >0 = block-based left-to-right
    Algorithm   string // "" or "confidence" = max-softmax; stub for future variants
}

type GenerateParams struct {
    MaxTokens       int
    Temperature     float32
    TopP            float32
    Stateless       bool             // bypass KV cache (for testing/comparison)
    ThinkingEnabled bool             // passed to template as enable_thinking variable
    LogProbs        bool             // include log-probabilities in response
    TopLogProbs     int              // number of top log-probabilities per token
    FlashAttention  *bool            // nil = use server default; true/false = per-request override
    Diffusion       *DiffusionParams // nil = not diffusion (ignored for autoregressive models)
}
```

`Generate()` flow:
1. `EncodeChat` → token IDs (template handles thinking mode natively via `enable_thinking`)
2. Dispatch on generation strategy:
   - `e.IsDiffusion()` → `generateDiffusion` (block-based iterative masked denoising)
   - `params.Stateless` → `generateStateless` (full-sequence recompute per token)
   - default → `generateCached` (KV/SSM-cached autoregressive decode)

### Generation strategies

Three generation paths live in separate files under `internal/inference/`:

| File | Strategy | Forward call | Notes |
|---|---|---|---|
| `generate_cached.go` | Autoregressive + KV cache | `ForwardCached` | Prefill in one pass; decode one token at a time |
| `generate_stateless.go` | Autoregressive, no cache | `ForwardStateless` | Full recompute each step; supports `ForwardCaptures` |
| `generate_diffusion.go` | Diffusion (iterative denoising) | `ForwardStatelessAllLogits` | Block-based masked denoising; no prefill/decode split |

`Engine.Generate()` dispatches via `IsDiffusion()` first; stateless flag is consulted only on the autoregressive path.

**Diffusion generation** (`generateDiffusion`):
1. Initialize output positions with `mask_token_id` (read from GGUF via `MaskTokenID()`).
2. Divide output into left-to-right blocks of size `DiffusionParams.BlockLength` (0 = single global block).
3. For each block: run `DiffusionParams.Steps` denoising steps. Each step calls `ForwardStatelessAllLogits` on the full sequence (prompt + all output positions), extracts logits for masked positions in the current block, scores each position by max-softmax confidence, and unmasks the highest-confidence positions according to a linear schedule.
4. Emit output tokens in sequence order after all blocks are resolved, stopping at the first stop token.

Streaming requests on diffusion models produce a single burst response after all denoising steps complete (not token-by-token); a warning is logged.

### `internal/inference/arch/`
Structural types (`ModuleMap`), TOML DSL parsing, graph construction, forward pass.
See subsections below.

### `internal/ggml/`
Go wrappers for ggml C ops (~90 functions). All CGo for graph ops is isolated here.
See "CGo Layer" section. All CGo is confined to this package.

---

## TOML DSL (`ArchDef`)

An `*.arch.toml` file maps directly onto `arch.ArchDef`:

| Section | Go field | Purpose |
|---|---|---|
| `[architecture]` | `ArchMeta` | Name, `tied_embeddings`, `embed_scale`, `non_causal`, `generation`, `shift_logits` — see **`ArchMeta` flags** below |
| `[params]` | `ParamsDef` | GGUF metadata key mappings; `[params.derived]` = arithmetic expressions; `[params.defaults]` = fallback values |
| `[weights.global]` | `WeightsDef` | Global tensor names: `token_embd`, `output_norm`, `output` |
| `[layers]` | `LayersDef` | `count` (param ref), `prefix` (must contain `@{layer_idx}`), `[layers.routing]`, `[layers.common_weights]` |
| `[layers.routing]` | `RoutingDef` | Binary routing: `rule` expression or `pattern` array param, `if_true`/`if_false` block names |
| `[layers.common_weights]` | `map[string]string` | Per-layer tensors shared by all block types (e.g. `attn_norm`) |
| `[blocks.<name>]` | `BlockDef` | Block-type weights, config, cache specs |
| `[ffn]` / `[ffn_alt]` | `FFNDef` | FFN builder + weights; `ffn_alt` for per-layer routing (dense vs MoE) |
| `[tokens]` | `TokensDef` | `think_open`, `think_close` token strings; `stop_tokens` string array — each entry added to the generation stop set alongside EOS |

**Routing:** `RoutingDef` supports two mutually exclusive modes:
- `rule` — expression evaluated per-layer (`@{layer_idx}`, `${param}` refs); nonzero → `if_true` block
- `pattern` — GGUF array param name; `pattern[layer_idx]` nonzero → `if_true` block

**`CacheDef.Shared`:** a string group name. All layers with the same `shared` value in a
given cache entry share one physical cache tensor (allocated on the first layer in the
group). Used for TOML-declared sharing; see also param-driven sharing via `n_kv_shared_layers`.

**`ArchMeta` flags:** `EmbedScale` (`embed_scale`), `NonCausal` (`non_causal`),
`Generation` (`generation`), `ShiftLogits` (`shift_logits`) — effects and
parse-time validation rules: SPEC.md §"Architecture Definition Contract".
`Generation` selects the generation path `Engine.Generate()` dispatches to
via `IsDiffusion()`.

**Expressions:** `@{layer_idx}` = engine builtin; `${param}` = resolved GGUF param;
bare names = derived param refs; `tensor.ne[dim]` = tensor shape. Full spec including
operators, `?` optional suffix, and `[params.defaults]` semantics:
`models/arch/MODEL_ARCH_TOML_DSL_SPEC.md`. **Read it before writing or modifying any `.arch.toml`.**

**Validation** runs at parse time (`Validate()` in `arch.go`): unknown keys, missing
weights, bad param refs, and contract violations all produce errors with source line
numbers. A successful `arch.Load()` means the definition is structurally sound.

**Adding a new architecture:** Copy the closest existing `.arch.toml`. The validator
and builder contracts will identify exactly what is wrong.

**For safetensors models:** also create a matching `.arch.stmap.toml` file that maps
HF config.json keys → GGUF metadata keys and HF tensor names → our short names. The
stmap's `architecture.hf_class` must match `config.json["architectures"][0]` for the
target architecture. Without it, `NewModelReaderSafetensors` will fail to locate a stmap.

---

## Block Builder System

Defined in `arch/blocks.go`:

```go
type BlockBuilder interface {
    Contract() BuilderContract           // declares expected weights, params, config
    BuildStateless(...) ggml.Tensor      // full-sequence forward pass
    BuildCached(ctx, gf, input, weights, params, config, inputs, cache) ggml.Tensor
}

type FFNBuilder interface {
    Contract() BuilderContract
    BuildFFN(...) ggml.Tensor
}
```

`BuildCached` receives a `*ggml.Graph` (`gf`) in addition to the graph context. Builders call `gf.BuildForwardExpand(ggml.Cpy(...))` to emit in-graph copy ops directly into KV cache tensors on the GPU — no CPU writeback is involved.

Registered in `init()`:
```
block builders: attention, full_attention_gated, mla_attention, gated_delta_net
FFN builders:   swiglu, geglu, moe
```

`attention` is the standard multi-head attention builder. Sliding-window attention is not
a separate builder — it is `attention` with `sliding_window = "true"` in block config,
which causes `selectMask` to route to `inputs.InpMaskSWA` instead of `inputs.InpMask`.

**`AttentionBuilder` contract:**
- Required weights: `attn_q`, `attn_output`
- Optional weights: `attn_k`, `attn_v`, `attn_q_norm`, `attn_k_norm`, `rope_freqs`
- Config schema (all optional): `head_dim`, `n_heads`, `n_kv_heads`, `rope_n_rot`,
  `rope_freq_base`, `kq_scale` — per-block overrides for attention dims; `sliding_window`
  — set to `"true"` to use SWA mask; `shared_kv_group` — group name for SharedKV;
  `v_norm` — `"rms"` applies RMS norm to V without learned weights; `rope` — `"standard"`
  or `"neox"` rope mode

`BuilderContract` declares `RequiredWeights`, `OptionalWeights`, `RequiredParams`, and
`ConfigSchema`. `Validate()` calls each builder's `Contract()` at TOML load time — the
primary protection against stale definitions.

**Adding a new block builder:**
1. Implement `BlockBuilder` (or `FFNBuilder`) with an explicit `Contract()`
2. Register in `init()` in `blocks.go`
3. Reference by name in TOML: `[blocks.<name>] builder = "my_builder"`

**NilTensor sentinel:** Optional weights absent from the model yield the zero-value
`ggml.Tensor`, which satisfies `IsNil()`. Every builder checks `IsNil()` and skips the op.

---

## Model Loading (`arch/model.go`)

**`ModelReader` interface** (`arch/model_reader.go`): abstracts metadata + tensor loading behind a single contract. Both GGUF and safetensors implement `ModelReader` — `newGenericModelFromReader()` is format-agnostic:

```go
type ModelReader interface {
    GetU32, GetF32, GetArrInts, GetArrBools  // metadata access (GGUF KV or safetensors config.json)
    GetTensorDim(tensorName, dim)            // shape query
    TensorCount() int                         // total tensor count
    TensorNames() []string                    // all tensor names
    TensorSpec(name string) (TensorSpec, bool) // shape, dtype, byte size
    ReadTensor(name string, buf []byte) error // raw bytes into caller buffer
    Close() error                            // release file handles
}

type TensorSpec struct {
    Type ggml.GGMLType     // ggml type constant (e.g., ggml.TypeF16)
    Ne   [4]int64          // dimensions, padded to 4 (unused = 1)
    Size int               // total bytes in ggml format
}
```

**Construction**: `NewModelReader(path, *GGUFFile)` for GGUF; `NewModelReaderSafetensors(archDef, stDir, archDir)` for safetensors. Both return `ModelReader`.

**`NewGenericModel(archName, modelPath, archDir, gf)`** → `*GenericModel` (GGUF path):
1. Parse `.arch.toml` → `ArchDef`
2. Create `ModelReader` from GGUF file (reuses pre-parsed `gf` if provided)
3. Delegates to `newGenericModelFromReader()` (steps 3–9 below)

**`NewSafetensorsModel(archName, stDir, archDir)`** → `*GenericModel` (safetensors path):
1. Parse `.arch.toml` → `ArchDef`
2. Create `ModelReader` via `NewModelReaderSafetensors(def, stDir, archDir)`:
   a. Load `model.safetensors.index.json` (500KB guard) or parse single `model.safetensors`
   b. Read `config.json`, extract `architectures[0]` as HF class
   c. Find matching `.arch.stmap.toml` via `FindSTMapByHFClass()` — scans all stmap files, returns the one with `architecture.hf_class` matching
   d. Build param value map: stmap `[params]` (1:1 HF→GGUF) + `[gguf_metadata]` (literals) + `[[derived_metadata]]` (computed from config.json keys)
   e. Precompute tensor specs (dtype → ggml type mapping; BF16 mapped to F16)
   f. Synthesize derived tensors per `[[derived_tensors]]` (no safetensors source — generated F32 bytes stored in `derivedTensors` map)
   g. Open all shard files for `ReadAt`
   h. Return `stModelReader` implementing `ModelReader`
3. Delegates to `newGenericModelFromReader()` (steps 3–9 below)

**`newGenericModelFromReader(reader, def, modelPath)`** → `*GenericModel` (shared path):

The loader is structured as a `genericModelBuilder` struct with phase methods, called sequentially from `build()`. The builder owns partial state until each phase succeeds; on failure, `cleanupOnError` releases everything allocated so far. The `WeightStore` is the ownership cut line: before `buildWeightStore` runs the builder owns `gpu`/`cpu`/`weightCtx`/`weightBuf` individually; after, `store.Close()` frees all four in one call. Compute resources (`cachedCtx`, `cachedSched`) created in the final phase are owned by the model and freed before `store.Close()`.

Phases (each a method on `genericModelBuilder`):
1. `checkMemory` — verify the model fits available VRAM/RAM via `reader.MinMemoryRequired(maxSeqLen)`
2. `resolveArch` — resolve params → `ResolvedParams`, then **validate**: every required param is present, typed, and nonzero where zero would cause silent downstream divergence (e.g. `rms_eps`, `rope_freq_base`). Bails out with a named error before any allocation. Then resolve weights → `ResolvedLayerWeights` per layer, build `CanonicalModuleMap` and `TensorDimsMap`.
3. `initBackendsAndArena` — bring up GPU/CPU backends, create the weight context (`AllocPermDisallow`), build the tensor-name → tensor index from reader specs, allocate weight VRAM via `AllocCtxTensors`.
4. `uploadWeights` — for each weight tensor: `reader.ReadTensor()` into a byte buffer → `ggml.ValidateRowData(spec.Type, buf)` → `ggml.TensorSetBytes(t, buf, 0)`. Validation inspects block scale/delta fields for quantized types (near-zero cost) and every element for float types; a NaN/Inf byte is a hard load error with the offending tensor name. This catches corrupt weight files and reader type/shape mismatches at load time, not as NaN logits mid-generation.
5. `buildWeightStore` — wrap the backends + arena in a `WeightStore`, transferring ownership; resolve global and per-layer tensor maps; handle tied embeddings; validate required globals are present.
6. `assignBuilders` — determine per-layer FFN routing (`ffn` vs `ffn_alt`), assign `BlockBuilders[]` and `FFNBuilders[]` per layer.
7. `createComputeResources` — create persistent `cachedCtx` (`AllocPermDisallow`, sized via `graphCtxSize()`) and `cachedSched` (sized to `maxGraphNodes`) for reuse across decode tokens; allocate `ffnScratch` and `logitBuf` scratch buffers for the hot decode path.

```go
type GenericModel struct {
    Def               *ArchDef
    Params            *ResolvedParams
    Weights           *ResolvedWeights       // tensor name maps per layer (format-agnostic above ModelReader);
                                             // ResolvedLayerWeights.Prefix is the canonical per-layer prefix source
    Store             *WeightStore           // GPU-resident tensors
    LayerBlockNames   []string               // which block builder each layer uses
    BlockBuilders     []BlockBuilder
    FFNBuilders       []FFNBuilder
    FFNConfigs        []map[string]any       // per-layer FFN config ([ffn.config] / [ffn_alt.config])
    CanonicalModuleMap *ModuleMap            // immutable; always Clone() before modifying
    HeadDim           int                    // attention head dimension
    TensorDims        TensorDimsMap          // tensor shape info for diagnostics
    ModelPath         string                 // path to GGUF or .st/ directory

    // Persistent compute resources for ForwardCached. Created at load, reused across tokens.
    cachedCtx   *ggml.GraphContext  // sized via arch.graphCtxSize(); AllocPermDisallow
    cachedSched *ggml.Sched         // sized to arch.maxGraphNodes

    // Pre-allocated scratch buffers for the hot decode path.
    ffnScratch map[string]ggml.Tensor // reused by buildFFNBlock each layer
    logitBuf   []float32              // reused by readLogits each token
}
```

---

## Cache (`arch/cache.go`)

`GenericModel.NewCache(maxSeqLen)` allocates per-layer cache tensors on GPU VRAM and pre-allocates `maskBuf` and `swaMaskBuf` (each sized to `maxSeqLen`) for reuse in `buildCausalMaskData` / `buildSWAMaskData` during cached decode. `ForwardStateless` allocates its own mask buffers and does not use these fields.

`GenericCache.Clear()` zeroes the entire cache backing buffer in a single `cacheBuf.Clear(0)` backend call rather than iterating per-tensor.

Two sharing mechanisms keep non-KV layers from allocating redundant buffers:

**TOML-declared sharing (`CacheDef.Shared`)**  
Set `shared = "groupName"` on a cache entry in a block definition. All layers whose block
type carries that entry with the same `shared` value reuse a single tensor, allocated on
the first layer in the group. `LayerCache.SharedGroup` is set to the group name for
diagnostic purposes.

**Param-driven sharing (`n_kv_shared_layers`)**  
When the GGUF param `n_kv_shared_layers` is present and nonzero, the last `N` layers
(indices `nLayers − N` through `nLayers − 1`) are treated as non-KV layers and reuse the
most recently allocated KV cache tensor for their block type. The last KV layer's
`LayerCache` is found by tracking `lastKVByBlock[blockName]` as allocation proceeds. The
non-KV layer's `LayerCache.SharedGroup` is set to the block name.

In both cases, non-KV layers hold a reference to the same `ggml.Tensor` as the KV layer.
During the forward pass, `selectSharedKV` reads from this shared cache tensor rather than
a per-layer buffer.

---

## Forward Pass (`arch/graph.go`)

Entry points:
- `ForwardStateless(tokenIDs, caps, flashAttn) ([]float32, error)` — full recompute; returns logits for the last token position
- `ForwardStatelessAllLogits(tokenIDs, flashAttn) ([]float32, error)` — full recompute; returns logits for **all** token positions, row-major by position: position `p` occupies `allLogits[p*nVocab:(p+1)*nVocab]`. Used exclusively by `generateDiffusion`.
- `ForwardCached(gc, tokenIDs, flashAttn) ([]float32, error)` — KV-cached; returns logits for the last token

Both stateless entry points share a private `forwardStatelessCore` helper that builds the graph context, scheduler, inputs, and runs all layers. The two stateless entry points differ only in how they extract logits after `sched.Compute`.

All three entry points follow the same layer-execution frame:
1. Extract params; allocate graph context + GPU scheduler. Stateless paths build a fresh `GraphContext` per call sized via `graphCtxSize()` (= `ggml.GraphContextSize(maxGraphNodes)` — derived from ggml's own cgraph + tensor-overhead accounting). `ForwardCached` reuses the persistent `cachedCtx` / `cachedSched` allocated at model load. All call sites pass `maxGraphNodes` (= 16384) consistently to the context, `NewGraph`, and `NewSched` — these three must agree.
2. Build input tensors (`inpTokens`, `inpPos`, `inpMask`)
3. Embedding scaling: if `ArchMeta.EmbedScale`, multiply token embeddings by `sqrt(n_embd)`
4. SWA mask: if `sliding_window` param present and nonzero, allocate `inpMaskSWA` and set it on `GraphInputs`
5. Per-layer embedding setup: `buildPerLayerEmbedSetup` prepares the combined per-layer embedding tensor (returns NilTensor if unused)
6. `runLayers`: for each layer, `Store.Layer(il)` → `BlockBuilders[il].Build*()` → optional `attn_post_norm` → residual add → `buildFFNBlock()` (includes optional `ffn_post_norm`) → optional per-layer embedding injection → optional `layer_output_scale` multiply
7. Final norm + LM head (`buildLogits`); execute (`sched.Compute`); read back logits + any captures

Stateless vs cached diverge in input setup, KV cache writeback strategy, and position tracking.

**`ForwardCached` reuse model:** `ForwardCached` calls `cachedCtx.Reset()` and `cachedSched.Reset()` at the top of each call and reuses the persistent `cachedCtx`/`cachedSched` instances created at model load — no scheduler or graph context is allocated per token. KV writes (K, V, conv_state, ssm_state, MLA K) are emitted as `ggml_cpy` ops directly into the persistent cache buffer via `writeCacheKV` / `gf.BuildForwardExpand(ggml.Cpy(...))` and execute on the GPU during `sched.Compute()`. There is no post-compute CPU writeback loop.

### `runLayers` — per-layer features

```
for each layer il:
    lt = Store.Layer(il)
    if !attn_norm.IsNil():
        cur = rmsNormApply(attn_norm) → BlockBuilder.Build*()
        if !attn_post_norm.IsNil(): cur = rmsNormApply(attn_post_norm)
        x += cur
    x = buildFFNBlock(x)     // includes ffn_post_norm if present
    if perLayerEmbd present and !pe_inp_gate.IsNil():
        x = perLayerEmbedInject(x)
    if !layer_output_scale.IsNil():
        x *= layer_output_scale
```

`perLayerEmbedInject` slices this layer's embedding from the combined tensor prepared by
`buildPerLayerEmbedSetup`, gates it through `pe_inp_gate` (GELU), optionally applies
`pe_post_norm`, and adds the result to the residual stream.

### `buildLogits` — logit softcapping

After the final norm and LM head projection, if the `logit_softcapping` param is present
and nonzero, logits are capped:

```
logits = logits * (1/cap) → tanh → * cap
```

This is `cap * tanh(logits / cap)`, implemented as three sequential `ggml.Scale` /
`ggml.Tanh` ops.

### `SharedKVState` — in-graph K/V sharing

`SharedKVState` is allocated fresh on each forward pass and passed through `GraphInputs`.
It carries in-graph tensors (not weights) from KV layers to non-KV layers within the same
`runLayers` call:

1. KV layers (`attn_k` present): after projecting K and V, write both into
   `SharedKV.K[group]` and `SharedKV.V[group]` keyed by the block's `shared_kv_group`
   config value.
2. Non-KV layers (`attn_k` absent): read `SharedKV.K[group]` and `SharedKV.V[group]`
   for attention. In cached mode, `selectSharedKV` additionally reads from the shared
   `LayerCache` tensor rather than a per-layer KV buffer.

`SharedKVState` tensors are live only for the duration of the graph build. The cache
backing for non-KV layers is handled separately — see "Cache" section below.

### SWA mask

`buildSWAMaskData(nQuery, nKV, startPos, window)` produces a float32 mask where positions
outside the sliding window (key > query or key < query − window) are set to −Inf. The
mask is allocated as `inpMaskSWA` in `GraphInputs` only when the `sliding_window` GGUF
param is present and nonzero; otherwise `InpMaskSWA` is a NilTensor. `selectMask` in
`AttentionBuilder` chooses between `InpMask` and `InpMaskSWA` based on the block's
`sliding_window` config key.

### `ForwardCaptures` — optional tensor capture

`ForwardStateless` accepts `*ForwardCaptures` (nil = no capture):

```go
type CaptureFlags uint32
const CaptureAttnWeights CaptureFlags = 1 << iota

type ForwardCaptures struct {
    Flags       CaptureFlags
    AttnWeights [][]float32  // [n_layers][nHeads*nTokens*nKV], nKV fastest
    NHeads      int64
    NTokens     int64
}
```

When `CaptureAttnWeights` is set, `scaledDotProductAttention` marks the post-softmax
tensor as a graph output; data is read back after `sched.Compute`. Layers without
`scaledDotProductAttention` (MLA, delta-net) leave their slot nil.

**Capture is stateless-only by design.** `ForwardCached` has no capture parameter —
cached mode has no single clean matrix to extract across prefill + per-token decode calls.
Data collection and KV-cache optimization must not be entangled. Any research question
answerable via a stateless pass should use that path. Do not add capture to
`ForwardCached` without explicit research into cached-mode mechanics as a dedicated goal.

---

## CGo Layer (`internal/ggml/`)

All CGo is confined here. `ggml_lib/` is a thin C wrapper over ggml ops — no model-specific
logic, no C++.

**Type and signature conventions** (enforced as the API for every Go caller):
- `ggml.GGMLType` — named integer type for tensor element types (e.g. `TypeF32`, `TypeQ4_K`). Distinct from `int` so that tensor type arguments cannot be silently swapped with unrelated ints. The neighbouring types it must not be confused with: ne dimensions are `int64`, mode flags are `int`.
- `ggml.AllocPerm` — named type controlling whether a `GraphContext` arena holds tensor *data* (`AllocPermAllow`) or only tensor *descriptors* (`AllocPermDisallow`, the normal graph-build case). `NewGraphContext(memSize int, allocPerm AllocPerm)` is non-variadic — every caller must declare its intent explicitly. There is no `AllocPermDefault`.
- `opt*` parameter prefix — nullable tensor parameters in op wrappers use the `opt` prefix to flag that callers may pass `NilTensor()` (e.g. `SoftMaxExt(ctx, a, optMask, ...)`, `RopeExt(ctx, a, pos, optFreqFactors, ...)`, `FlashAttnExt(ctx, q, k, v, optMask, ...)`). Required tensor parameters have no prefix.

**Principled context sizing** (`backend.go`):
- `GraphOverheadCustom(size int, grads bool) int` — exact bytes ggml requires for a cgraph structure (nodes array, leafs array, hash tables) of the given size. Thin wrapper over `ggml_graph_overhead_custom`.
- `GraphContextSize(maxNodes int) int` — minimum context arena bytes needed to build a graph of up to `maxNodes` nodes: `GraphOverheadCustom + TensorOverhead*maxNodes + 64` alignment slop. Use this with `NewGraphContext` to size graph contexts precisely from the declared maxNodes budget — never ad-hoc multipliers.
- `Buffer.Clear(byte)` — single backend call that writes `value` to every byte in the buffer (including alignment padding). Used by `GenericCache.Clear` instead of per-tensor zero loops.

**ggml log routing (`ggml/logging.go`):**  
`ggml_log_set` registers a CGo callback in the C layer (`ggml_ops.c`). All ggml diagnostic
output — previously printed directly to stderr — is intercepted and forwarded to the Go
`internal/log` logger at the appropriate level. This eliminates uncontrolled stderr writes
from the C layer; all log output passes through the single Go logger and is subject to
the same level gate and file sink. Registration is not automatic on import — each CLI
entry point that loads models must call `ggmlmod.InitLogging()` explicitly after
`log.InitLogger`. `logging.go` is the only file in `ggml/` that imports `internal/log`;
the rest of the package has no dependency on it.

**Go wrappers added for Gemma4 support (in `ggml/`):**
- `Gelu(ctx, a)` — element-wise GELU activation (used by GeGLU and per-layer embed injection)
- `Tanh(ctx, a)` — element-wise tanh (used by logit softcapping)
- `MulMatSetPrecF32(t)` — sets F32 accumulation precision on a `mul_mat` tensor (no return value; mutates in place)

**Go wrappers for load-time validation (in `ggml/`):**
- `ValidateRowData(typ GGMLType, data []byte) bool` — thunks to upstream `ggml_validate_row_data`. Full element scan for float types; per-block scale/delta scan for quantized types. Returns true on empty input. Called by `newGenericModelFromReader` before every `TensorSetBytes`. Unit tested in `ggml/validate_test.go` (F32 NaN/Inf, I32 pass-through, Q4_K block-scale NaN).

**GGUF metadata reading**: `arch/model.go` uses pure-Go `gguf-parser-go` — no CGo. The `goGGUFReader` adapter provides the metadata-access portion of `ModelReader` (GetU32/F32/ArrInts/ArrBools/TensorDim) with type-checked KV access (guards against gguf-parser-go's panic-on-wrong-type behavior). Split GGUF files are rejected at parse time.

**Safetensors parsing**: `arch/safetensors_index.go` and `arch/model_reader_safetensors.go` are pure Go — JSON index parsing, shard header reading, and BF16→F16 conversion are all done without CGo. The `stModelReader` implements `ModelReader` using `os.ReadAt` on shard files. Tensor transforms (norm-shift, V-head reorder, etc.) live in `arch/model_reader_safetensors_xform.go`; the derived-metadata and derived-tensor handler registries live in `arch/model_reader_safetensors_derived.go`.

**Verified ggml semantics** (do not change without re-verifying):
- `ggml_mul_mat(A, B)`: contracts over `A→ne[0] == B→ne[0]`; result `[A→ne[1], B→ne[1], ...]`; GQA broadcasting when `B→ne[2] % A→ne[2] == 0`
- `ggml_permute(ctx, a, ax0..ax3)`: input dim i → output position ax_i
- `ggml_rms_norm`: normalizes over `ne[0]` per slice
- `ggml_soft_max_ext(a, mask, scale, max_bias)`: `softmax(scale * a + mask)` over `ne[0]`

---

## Defensive Checks

Silent numerical failures are the highest-cost class of bug in this codebase — a
single NaN in a Q4_K block scale propagates through every row the block touches
and surfaces as "the model produces gibberish," forcing hours of bisection
through the load and forward paths. The checks below convert that class of
failure into loud load-time or sampler-time errors with the offending
tensor/position named, plus two debugging aids (param dumps and the
GGUF↔safetensors equivalence test).

**Param validation (load time, `arch/params.go`)**  
After `ResolveParams`, the loader asserts that every required param is present,
typed correctly, and nonzero where zero would cause silent downstream
divergence. Concrete failures this catches:
- `rms_eps = 0` (from untyped JSON `1e-6` silently truncated to `uint32 0`) →
  `1/sqrt(0) = Inf` in every RMS norm
- `rope_freq_base` present in `Ints` but not `Floats` → positional encoding
  computed against zero base → NaN positions → first-token-EOS output
- Missing `n_heads` / `n_kv_heads` / `head_dim` → attention scale computation
  fails later rather than at load

Fails before any VRAM allocation; the error names the param and the reader
that produced it.

**Weight validation (load time, `arch/model.go`)**  
Every tensor's raw bytes are validated via `ggml.ValidateRowData(spec.Type,
buf)` before `TensorSetBytes` uploads them to VRAM. The underlying
`ggml_validate_row_data`:
- Float types (F32, F16, BF16, F64) — scans every element for NaN/Inf
- Quantized types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
  Q8_K) — scans each block's scale/delta fields for NaN/Inf (near-zero cost;
  quants themselves are integers so they cannot be NaN)
- MXFP4 — E8M0 scale validation
- Integer types (I8, I16, I32, I64) — pass-through (no representation for NaN)

On failure the C layer prints `ggml_validate_row_data: found nan value at
block N` to stderr (routed through the Go logger) and the Go wrapper returns
false. `newGenericModelFromReader` converts that into a named load error that
identifies the tensor and the likely cause (corrupt file or reader type/shape
mismatch).

Unit tested in `internal/ggml/validate_test.go`: F32 finite/empty/NaN/Inf,
I32 pass-through, Q4_K well-formed block passes, Q4_K F16 NaN d-scale fails.

**Logit validation (sample time, `inference/sampler.go`)**  
`ValidateLogits` runs at every sampler chokepoint. Autoregressive paths
(`generate_cached`, `generate_stateless`, `generate_rlb`) hit it transitively
via `Engine.sample()` in `engine.go`, which validates before any sampling
math. The diffusion path (`generate_diffusion`) calls it explicitly after
each `ForwardStatelessAllLogits`, wrapped with `blockNum` / `step` context.
If any logit is NaN/Inf the request fails with a named error rather than
emitting garbage tokens. Cost is one linear scan of `n_vocab` per call —
memory-bandwidth bound, negligible vs. a full forward pass.

**Param dumps (debugging aid, `arch/model_reader_safetensors.go`,
`arch/model_reader_gguf.go`)**  
Both readers emit a sorted `[param] key = value` dump at DEBUG level listing
every metadata KV seen from the source file. When porting a new architecture
to safetensors, visual diff of the two dumps is the fastest way to find a
stmap error: any param present in one but absent/wrong in the other is the
bug. Keep these dumps symmetric — a difference in dump format between loaders
hides exactly the class of bug they are there to surface.

**Equivalence testing (acceptance gate, `make equiv-test`)**  
`test_chat_equiv.sh gguf-st` compares logprobs between the GGUF and safetensors
loaders for the same model, at the same sampler step, on the same prompt.
Non-identical above GPU FP variance is a hard failure. This test is on the
default Makefile testing path and is the authoritative gate for any change
touching the load or inference path — it must be green before a change is
considered complete.

---

## Logging

See **Internal Logging** section in `CONVENTIONS.md` — authoritative for format, initialization, CLI flags, ggml routing, and constraints. Summary: `internal/log` package, `<HH:MM:SS>[LEVEL]` format, dual stderr/file output, `ggml.InitLogging()` routes C library output. Dependency leaf — imports stdlib only.

---

## Visualization System

Four SVG renderers share a unified palette (`Pal()` in `archdiagram/palette.go`).
To change any color, update the palette map — all renderers reflect it. No inline hex strings.

- `RenderArchDiagram(def, outPath)` — `*.arch.svg` from `ArchDef`; invoked by `bench gen-arch-diagram`
- `RenderLayersDiagram(def, mm, svgPath, subtitle, dims)` — per-layer tensor detail; invoked by `bench gen-arch-diagram`
- `RenderVisionDiagram(def, w)` — `*.vision.svg` overview when `def.Vision != nil`; see "Vision / Multimodal Subsystem" section
- `RenderVisionLayersDiagram(def, nLayers, w)` — `*.vision.layers.svg` exploded per-layer when `def.Example.VisionNLayers > 0`

---

## Weight Store (`arch/weight_store.go`)

`WeightStore` holds GPU-resident tensors, immutable after construction. Logical names
(e.g. `attn_q`, `ffn_gate`) are short names from TOML, not GGUF tensor names; translation
is via `ResolvedLayerWeights` in `arch/weights.go`.
- `Global(name) ggml.Tensor`
- `Layer(idx) map[string]ggml.Tensor`

---

## Safetensors Loading Pipeline

Safetensors models from HuggingFace use a different directory layout and naming convention
than GGUF. The loading pipeline translates both into the format-agnostic `ModelReader`
interface so that inference code is identical above the abstraction boundary.

### Directory Convention

Required `.st/` directory layout and the tokenizer sidecar requirement:
SPEC.md §"Model Format Contract". Implementation wiring: format detection
happens at discovery time in `internal/model`; `ModelInfo.Format`
(`FormatGGUF` / `FormatSafetensors`) carries the result into `NewEngine()`.

### Index Parsing (`safetensors_index.go`)

`LoadSafetensorsIndex(stDir)` handles both sharded and single-file models:
1. **Sharded**: reads `model.safetensors.index.json` (500KB guard), parses `weight_map`
   to build ordered shard list, then reads each shard's 8-byte header length + JSON header
   to populate `STTensorEntry` per tensor (dtype, shape, absolute data offset, shard assignment).
2. **Single-file**: reads `model.safetensors` directly, parses its JSON header.

Both paths produce a `SafetensorsIndex` with `Tensors` map and `Shards` list.

### STMap Resolution (`stmap.go`, `model_reader_safetensors.go`)

Safetensors tensor/param names differ from GGUF conventions. A per-architecture
`.arch.stmap.toml` file provides the translation (see
`models/arch/MODEL_ARCH_STMAP_TOML_DSL_SPEC.md` for the user-facing contract).
The lookup chain:

1. `config.json["architectures"][0]` → HF class (e.g. `"LlamaForCausalLM"`)
2. `FindSTMapByHFClass(archDir, hfClass)` scans `*.arch.stmap.toml` files, returns the
   one with `architecture.hf_class` matching
3. STMap provides:
   - `[params]`: HF config.json key → GGUF metadata key mapping (1:1, scalar)
   - `[layer_prefix].hf`: per-layer prefix with `{N}` substitution
   - `[tensors]`: our short tensor name → HF short tensor name
   - `[tensors.global]`: our short global name → HF full tensor name
   - `[gguf_metadata]`: literal GGUF metadata values for arch-level constants with no `config.json` source
   - `[[derived_metadata]]`: GGUF metadata values computed at load time from one or more `config.json` keys (via the `derivedMetadataOps` registry in `model_reader_safetensors_derived.go`)
   - `[[derived_tensors]]`: GGUF tensors synthesized at load time, no safetensors source (via the `derivedTensorOps` registry)
   - `[[transforms]]`: per-tensor F32 transforms (scalar add, V-head reorder, etc.) applied after read, before final dtype cast

Param resolution: HF config.json value is stored under the GGUF key name, so the
existing `ResolveParams()` machinery works without changes. Scalar params come
through `[params]` directly; arrays and computed values come through
`[[derived_metadata]]` (e.g. `string_array_eq` for Gemma 4's
`layer_types` → `sliding_window_pattern` translation). `GetArrInts` /
`GetArrBools` work for any derived metadata that produces an array.

Tensor name resolution at load time:
- Per-layer: `layer_prefix.hf` (with `{N}` resolved) + `tensors[our_short_name]` = full HF tensor key
- Global: `tensors.global[our_short_name]` used directly
- Derived: synthesized tensors live in a parallel `derivedTensors` map (no HF source), consulted before the HF-name lookup path in every reader method

### Reader Construction (`model_reader_safetensors.go`)

`NewModelReaderSafetensors(archDef, stDir, archDir)` builds an `stModelReader`
implementing `ModelReader`:

1. Parse safetensors index → `SafetensorsIndex`
2. Read `config.json` (with `json.Number` preserved — see "Param coercion" below)
3. Extract HF class → find matching stmap
4. Build param value map from stmap `[params]` + `config.json`
5. Inject literal values from stmap `[gguf_metadata]`
6. Apply `[[derived_metadata]]` handlers — runs each registered op against
   `config.json` and writes the result into the param map under `spec.Target`.
   Same param namespace as 1:1 and literal entries, so the `[param]` debug
   dump shows everything in one sorted list (visual GGUF↔safetensors diffing
   stays a single-flat-list operation)
7. Build `tensorSpecs` from the safetensors index (dtype → ggml type mapping
   via `stDtypeToGGML`)
8. Apply `[[derived_tensors]]` handlers — runs each registered op against
   `config.json` and produces F32 little-endian bytes stored in a
   `derivedTensors` map keyed by GGUF tensor name. These appear in
   `TensorCount` / `TensorNames` / `TensorSpec` / `ReadTensor` alongside
   safetensors-sourced tensors (consulted first in each method); downstream
   code is format-agnostic and cannot tell them apart
9. Open all shard files for `ReadAt`-based random access
10. Allocate the slow-path scratch ggml context (`AllocPermAllow`, chunked
    working arena for any tensor that needs F32 conversion or transforms)

**Derived ops registry**: handlers are registered by name in
`model_reader_safetensors_derived.go` — `derivedMetadataOps` and
`derivedTensorOps`. Adding a new op = adding an entry to the appropriate
registry, mirroring the block builder pattern. The TOML side stays
declarative; the actual computation is Go-side. Currently registered:
- `derivedMetadataOps`: `string_array_eq`, `copy_param`
- `derivedTensorOps`: `rope_freqs_proportional`

**BF16 conversion**: bench currently stores BF16 source tensors as **F16** at load (the
target type in `stDtypeToGGML`; the load fast-path is explicitly disabled for BF16,
forcing the convert). The conversion goes BF16 → F32 (exact bit-expansion: the 16 BF16
bits become the upper half of an F32, lower 16 zero) → F16 (round-to-nearest). F16
preserves BF16's mantissa exactly (7 bits ⊂ F16's 10); the only loss is exponent range,
and any BF16 magnitude outside F16 range becomes `inf`, caught loudly by
`ggml.ValidateRowData` at load. Exceptions stored as F32 (not F16): rank-1 tensors (norms)
and conv1d weights — see the type-promotion in `model_reader_safetensors.go`.

ggml *does* have a native BF16 type (`GGML_TYPE_BF16 = 30`) and its Metal backend supports
BF16 ops on Metal 3.1+ / `has_bfloat` GPUs (M3+ — confirmed `kernel_mul_mv_bf16_*` kernels
and the `supports_op` BF16 gate). Storing BF16 natively (no conversion, exact, same VRAM as
F16) is therefore viable; the F16 conversion is a current choice whose one observable
property is that `.st` weights stay bit-identical to F16 GGUFs under `gguf-st` equiv (→ 0.0).

### Engine Integration (`engine.go`)

`NewEngine()` dispatches on `info.Format`:
- `FormatSafetensors`: loads tokenizer from `tokenizer.gguf` sidecar in the `.st/`
  directory (a minimal GGUF with only tokenizer metadata), then calls
  `arch.NewSafetensorsModel(archName, stDir, archDir)`
- `FormatGGUF`: existing GGUF path; reuses pre-parsed GGUF to avoid double-parsing

For safetensors models, `NLayers` may be 0 if `config.json` lacked the field — falls
back to the resolved model param (`m.Params.GetInt("n_layers")`).

### Model Discovery (`model/safetensors.go`)

`ParseSafetensorsDir(stDir, archDir)` produces `GGUFMetadata` for the model manager:
1. Load safetensors index for tensor inventory
2. Convert `STTensorEntry` → `TensorInfo` (offset=0, no unified safetensors offset concept)
3. Read `config.json` for architecture class and numeric params (`num_hidden_layers`,
   `hidden_size`, etc.)
4. Resolve HF class → architecture name via `FindSTMapByHFClass()`
5. Return `GGUFMetadata` with resolved architecture name

The manager (`manager.go`) scans both `*.gguf` files and `*.st/` directories during
`scan()` and `List()`, filtering by whether the resolved architecture has a matching
`.arch.toml` file. `tryLoadOne()` checks GGUF first, then falls back to `.st/`.

---

## Vision / Multimodal Subsystem

Multimodal models (Gemma 4, Qwen3.5-VL) carry a second encoder tower alongside
the text decoder. The encoder runs as a separate forward graph per image, and its
projected output embeddings are spliced into the decoder's input stream before the
layer loop.

### Where the code lives

Vision code follows the `vision_*.go` naming convention within its package.
`internal/inference/arch/` holds everything model-side — weight/param resolution
and the encoder graph entry point `BuildVisionGraph` in `vision.go`, plus M-RoPE
position buffers, the splice seam, image preprocessing, and the Gemma 4
clamp-scalar loader in their respective `vision_*.go` files. `internal/inference/`
holds the request-side half: `<|image|>` placeholder expansion, producing the
`VisionPrefill` the forward paths consume. `archdiagram/` holds the two SVG
renderers, wired for emission from the `gen-arch-diagram` command.

One placement is not guessable from naming: `handleConv3DTemporalSplit` lives in
`model_reader_safetensors_derived.go` but is **not** in the `derivedTensorOps`
registry — `buildDerivedTensors` dispatches it separately because it reads
source shard data rather than just `config.json`, which the registry's handler
signature cannot express. See "Construction Across Two Formats" below.

**TOML examples:** `models/arch/gemma4.arch.toml` and `qwen35.arch.toml`, `[vision]`
/ `[projector]` sections. `models/arch/qwen35.arch.stmap.toml` carries the
safetensors-side `[vision]` tensor mappings.

### Data-Driven Two-Tower Purity (invariant #1)

`BuildVisionGraph` serves two architecturally distinct vision towers from one
procedural graph. Every fork is keyed on a data property — never on an
architecture name. Key dispatch points:

| Property | Gemma 4 | Qwen3.5-VL |
|---|---|---|
| `params.NormType` | `VisionNormRMS` (default) → weight-only `rmsNormApply` | `VisionNormLayerNorm` → `ggml_norm` + weight + bias (`layerNormApply`) |
| `params.ProjectorType` | `ProjectorLinearPostNorm` — avg-pool + scale + unit-RMSNorm + single linear | `ProjectorMLP` — reshape-only merger + `w0` → GELU → `w1` with biases |
| `ConfigRope` in block config | `axial2d` — NeoX RoPE on the two halves of each head's dim using `PosX`/`PosY` (`applyRope2D`) | `mrope_vision` — 4-channel M-RoPE via `GGML_ROPE_TYPE_VISION` using `InpPosVision` |
| Second patch-embed kernel present | No — single Conv2D; reshape + transpose | Yes — two Conv2D summed + bias; merge-grouped permute/reshape |
| `posEmbd.Ne(2) > 1` | Yes — axial table `[n_embd, max_per_axis, 2]`; independent X/Y GetRows, summed | No — learned grid `[n_embd, n_patches²]`; bilinearly interpolated to the patch grid via `ggml_interpolate` |
| `ConfigQKVFused` | false | true — `splitFusedQKV` slices the fused `[n_embd, 3*n_embd]` weight + bias into contiguous View2D thirds before dispatch |

The encoder layer loop in `BuildVisionGraph` dispatches into the **same** generic
`BlockBuilder` / `FFNBuilder` implementations the decoder uses (`attention` with
`rope=axial2d` or `rope=mrope_vision`; `geglu_quick` (quick-GELU) for Gemma; `mlp` for Qwen). No
vision-specific builder exists. Adding a new ViT family is primarily a TOML +
stmap operation: add `[vision]` / `[projector]` blocks to the arch and stmap
files; update the dispatch forks only if the new tower introduces a genuinely new
data-driven property.

### Construction Across Two Formats (invariant #16)

Vision weights arrive via two formats, both presenting the same canonical tensor
names downstream:

**GGUF mmproj sidecar**: `mmproj-<name>.gguf` contains the encoder + projector
weights (GGUF namespace `v.*` / `mm.*`). The GGUF `ModelReader` merges the mmproj
sidecar into the main model's tensor namespace at load time (`model_reader_gguf.go`).

**Safetensors `.st/` path**: vision weights live in the same `.st/` directory as
the decoder weights. The stmap's `[vision]` tensor-mapping section translates HF
tensor names to the canonical short names that `ResolveVisionWeights` expects.
`conv3d_temporal_split` is a special case: Qwen3-VL's HF repo stores a single
rank-5 Conv3D patch-embed kernel `[oc, ic, kt, kh, kw]`; the mmproj GGUF splits
it at conversion time into two rank-4 Conv2D kernels (`v.patch_embd.weight` /
`v.patch_embd.weight.1`). For the safetensors path, `buildDerivedTensors` in
`model_reader_safetensors.go` dispatches `handleConv3DTemporalSplit` (defined in
`model_reader_safetensors_derived.go`) to perform that split at load time. This op
reads directly from the shard file rather than from `config.json`, which is why it
is dispatched separately from the config-only `derivedTensorOps` registry — the
registry's handler signature only receives `config.json`, making it structurally
incapable of reaching shard data.

**Clamp scalars** (`vision_clamp.go`): Gemma 4's `Gemma4ClippableLinear` pairs
every linear weight with four `<base>.input_min` / `input_max` / `output_min` /
`output_max` scalar tensors. `LoadVisionClampsFromReader` discovers and reads them
through the `ModelReader` boundary — the same canonical names exist in both the
GGUF mmproj and the safetensors directory, so one function serves both formats.
Clamps are consumed by `mulMatClamped` in the encoder layer loop and projector.

### Preprocessing Pipeline (`arch/vision_preproc.go`)

`PreprocessImage` follows the llama.cpp `mtmd_image_preprocessor_dyn_size`
pipeline:

1. **smartResize** — aspect-preserving scale to a target whose pixel count falls
   in `[MinPixels, MaxPixels]` and whose dimensions are multiples of
   `AlignSize = patch_size × n_merge`. Matches `calc_size_preserved_ratio` in
   `tools/mtmd/mtmd-image.cpp`.
2. **PAD_CEIL fit** — scale the source by `min(tgtW/srcW, tgtH/srcH)` (aspect-fit
   into the target), center the result, and pad the border with black. This is
   llama.cpp's default `img_tool::resize` behavior; using a stretch-to-fill resize
   instead causes every patch to shift for any image whose aspect ratio doesn't
   already match the aligned target (confirmed to produce 0.185 logprob
   divergence on Qwen before the fix).
3. **resizeBilinearLlama** — align-corners, 2-tap bilinear, no antialias (a literal
   port of llama.cpp's `img_tool::resize_bilinear`). `golang.org/x/image/draw`'s
   `BiLinear` is NOT used: it is half-pixel and antialiases on downscale, shifting
   every resized pixel relative to the reference.
4. **packPixelsF32** — channel-major F32 `[W, H, 3, 1]` in `[0, 1]`; the encoder
   graph rescales to `[-1, 1]` via `ggml_scale_bias(inp_raw, 2.0, -1.0)`.

Preprocessing parameters (`PatchSize`, `NMerge`, `ImageMinTokens`,
`ImageMaxTokens`) come from `arch.toml`'s `[vision]` section via
`PreprocConfigFromArchDef`. No per-architecture hard-coded values exist in Go.

### Splice and Decoder Integration (`arch/vision_splice.go`, `inference/vision_prefill.go`)

`prepareVisionPrefill` preprocesses each attached image and expands the tokenized
prompt so each `<|image|>` placeholder becomes N copies, where N is
`PreprocessedImage.NTokens(nMerge)`. This expansion is computed from actual
preprocessed dimensions, not from the `n_image_tokens` field in the arch.toml
(which is diagram-only).

`buildVisionSplice` is called once per request during graph construction — before
`runLayers`, not inside the decode loop. For each image it:

1. Creates graph input tensors (`inpRaw`, `posX`/`posY` for axial towers,
   `posMRope` for M-RoPE towers — only the pair the active tower's graph consumes).
2. Calls `BuildVisionGraph` to build the encoder + projector sub-graph; checks
   that the projector output token count matches the declared splice length (a
   preprocess↔TOML drift guard).
3. Chains a `ggml.SetRows` op to overwrite the image's placeholder rows in
   `inpEmbd` with the projected embeddings.

`fillVisionInputs` writes the actual pixel + position data after `sched.AllocGraph`
backs the input tensors with memory.

**Decoder image-token attention**: by default, image tokens in the decoder are
causal (the surrounding text). `[vision].decoder_non_causal = true` (set by Gemma
4, not Qwen3.5-VL) causes `visionDecoderSpans` to return `MaskSpan` entries for
each image's position run, which `buildCausalMaskData` opens bidirectionally. This
mirrors llama-mtmd's `mtmd_decode_use_non_causal`, which is true only for
GEMMA3/GEMMA4V. The flag is data-driven; no arch-name branch.

`visionImages` (the `[]VisionSpliceInput` slice) threads through the forward
functions as nil for non-vision callers: `buildVisionSplice` short-circuits on
`len(images) == 0` and returns the input embedding tensor unchanged.

**Token ID rewrite**: `RewriteTokenIDsForVision` replaces all image placeholder
positions with token ID 0 (the padding token) before the decoder embedding lookup
so the `GetRows` and `perLayerEmbedInject` ops see a valid vocabulary index. Those
rows are immediately overwritten by the `SetRows` splice.

### Diagramming

`RenderVisionDiagram` and `RenderVisionLayersDiagram` in `archdiagram/` emit an
overview SVG and a fully-exploded per-layer SVG respectively. Both renderers read
entirely from `def.Vision` and `def.Projector` — no architecture-name branches.
Differences such as norm type, fused vs. separate QKV, dual-conv patch embed, and
projector layer count are all data-driven from the TOML. They share the unified
palette (`archdiagram/palette.go`) — invariant #7.

`gen_arch_diagram.go` emits vision SVGs automatically when the parsed `ArchDef`
has `def.Vision != nil`; the exploded per-layer diagram is emitted when
`def.Example.VisionNLayers > 0`.

### Equivalence Testing and Fidelity

Vision correctness is gated by `test_vision_equiv.sh` (two modes: `llama` —
bench GGUF vs. llama-server; `gguf-st` — bench safetensors vs. bench GGUF).
Threshold, full protocol, and the reference-build sensitivity behind that
threshold: **`test_vision_equiv.sh`** in `CONVENTIONS.md`.

For diagnosing suspected encoder or preprocessing divergence, the logprob gate
is insufficient — an 18% projected-embedding error was shown to move logprobs
less than 0.001. The correct tool is a **per-element checkpoint-diff**: run
`bin/vision_capture` (bench) against `MTMD_DEBUG_GRAPH=1 llama-mtmd-cli`
(brew llama.cpp) and compare per-stage tensor sums and edge values. The logprob
gate catches gross regressions (wrong answer, preproc-stretch class of bug);
checkpoint-diff catches algorithmic or norm-order errors.

---

## Key Invariants

1. **No model-specific Go code.** SPEC.md §"Architecture Definition Contract".
2. **`ModuleMap` is structural-only.** Built once from resolved weights in `BuildModuleMap`; consumed by `gen-arch-diagram`. Never mutated post-construction.
3. **NilTensor means absent.** Optional weights not present in the model return zero-value `ggml.Tensor`. Every builder checks `IsNil()`.
4. **`os.Exit` / `log.Fatal` placement.** SPEC.md §"System Invariants". Wiring fact: `util.ResolvePaths()` returns `(BenchPaths, error)`; `BenchPaths` is injected from the cobra entry into `Server` and `Engine` — no library code calls `ResolvePaths` itself.
5. **CGo confinement.** SPEC.md §"System Invariants". GGUF metadata is read via pure-Go `gguf-parser-go`; safetensors index and tensor data are parsed in pure Go.
6. **Builder contracts enforced at parse time.** Adding a required weight to `Contract()` requires updating all `.arch.toml` files using that builder.
7. **Palette is single-source.** All diagram colors from `diagramPalette()`.
8. **TOML is the data language.** SPEC.md §"Architecture Definition Contract".
9. **`internal/log` is a dependency leaf.** Imports stdlib only.
10. **Utility placement:** project-wide → `src/internal/util/project_util.go`; package-internal, multi-file → `src/internal/<pkg>/<pkg>_util.go`; single-file → define in that file.
11. **Test with all models.** Llama is easy; Qwen3.5, Qwen3.5-MoE, DeepSeek2, Gemma4 edge-case.
12. **Arch editor: TOML is canonical, palette/builders dynamic.** No hardcoded colors/names in JS.
13. **Data capture is stateless-only.** SPEC.md §"System Invariants".
14. **Template owns thinking mode.** `enable_thinking` passed to gonja template. No post-render manipulation.
15. **SharedKV ordering validated at parse time.** `shared_kv_group` with no `attn_k` producer is a validation error.
16. **`ModelReader` is the format boundary.** SPEC.md §"System Invariants". Everything above `newGenericModelFromReader()` is format-agnostic — this is why Model Loading below is described once, not per-format.
17. **Tokenizer is GGUF-only.** SPEC.md §"Model Format Contract".
18. **`NewGraphContext` requires explicit `AllocPerm`.** No default. Graph-build contexts pass `AllocPermDisallow`; data-arena scratch contexts (load-time type conversion) pass `AllocPermAllow`. Caller intent must be visible at the call site.
19. **Single source of truth for the cgraph node budget.** `arch.maxGraphNodes = 16384` drives both the context arena (via `arch.graphCtxSize()` → `ggml.GraphContextSize`) and every `NewGraph` / `NewSched` call in the arch package. Never use a literal node count.
20. **Canonical logical weight names live in `arch_util.go`.** `WeightAttnNorm`, `WeightFFNNorm` (and the `Cache*` keys) are constants; never inline these literal strings in graph or module code.
21. **`ResolvedLayerWeights.Prefix` is the canonical per-layer prefix.** All consumers (module map, tensor dims, diagram code) read `lw.Prefix` (already includes the trailing dot). Never reconstruct `blk.<N>.` from `layer_idx` ad hoc.

## Metrics and Timing

`InferenceMetrics` captures timing/throughput for each generation, returned by `engine.Generate()` and serialized in the API `usage` field:

```go
type InferenceMetrics struct {
    PromptTokens     int                  // input tokens
    CompletionTokens int                  // output tokens
    FinishReason     string               // "stop" or "length"
    PrefillDuration  time.Duration        // prefill wall-clock
    DecodeDuration   time.Duration        // all decode steps wall-clock
    TotalDuration    time.Duration        // total generation wall-clock
    TokenLogProbs    []TokenLogProb       // nil if not requested
}
```

Throughput methods: `TokensPerSec()` (decode), `PrefillTokensPerSec()`, `TotalTokensPerSec()`. All return 0 when denominator ≤ 0.

Wire-level `usage` object shape and streaming inclusion rules: SPEC.md §"HTTP API".

---

## Logprobs

`ComputeTopLogProbs` in `sampler.go` computes log-probabilities via stable log-softmax
on raw logits after each `sample()` call. Works in both cached and stateless modes.

```go
type TokenLogProb struct {
    ID       int32
    Token    string
    LogProb  float64
    Bytes    ByteArray  // JSON-marshals as integer array, not base64
    TopProbs []TopLogProb
}
```

`ByteArray` is a custom `[]byte` type that marshals as `[51, 52]` instead of Go's
default base64 encoding — matches the OpenAI/llama.cpp response format.

### Equivalence Testing

Gate modes and the pass threshold: CONVENTIONS.md §"Validating Changes".
Validates: tokenization, chat template rendering, forward pass correctness,
and sampling.

`test_chat_equiv.sh` and `test_rlb.sh` invoke the harness via `bash test_inference.sh`,
which is a thin launcher over `tools/test_inference.py` — a stdlib-only Python
program that owns SSE streaming, payload construction, server lifecycle,
loop-mode REPL, and the logprob fingerprint line that `test_chat_equiv.sh` greps
for. Streaming is always on; bench's `stream_options.include_usage` final chunk
carries the `usage` block that drives `parse_response`'s timing/throughput
display.
