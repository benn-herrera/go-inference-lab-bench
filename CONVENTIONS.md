# CONVENTIONS – Inference Lab Bench

From-scratch Go LLM inference engine for R&D into inference mechanics. Multi-model API server, data-driven architecture definition via TOML DSL, KV-cached and stateless inference, and visualization tooling.

## Project Priorities

- **R&D focus** — clarity and tinkerability over production concerns; don't make choices that close doors
- **Consistency** — naming, data patterns, conventions. Surprises are expensive in R&D. TOML is the static data language except where constrained (e.g. JSON for OpenAPI payloads). Use discipline-specific terminology where high-consensus terms exist.
- **Data-driven architecture purity** — model-specific Go code subverts the design. The TOML DSL + block builder system is the whole point; adding architectures should be a data-writing operation, not a coding one.
- **Portability** — macOS only now; Linux/Windows ports coming. Don't accumulate portability debt. Platform-specific surface area is limited to: ggml cmake build options, `ggml_lib` compiler flags, and Metal-specific binding wiring.

## Current Status

Known working models (links in README.md):
- Llama-3.2-3B-Instruct-f16.gguf, llama-3.2-3b-instruct-q4_k_m.gguf (`llama` arch)
- qwen35-9b-opus46-mix-i1-Q4_K_M.gguf (`qwen35` arch)
- DeepSeek-V2-Lite-Chat.Q4_K_M.gguf (`deepseek2` arch)
- gemma-4-E4B-it.gguf (F16, dense, `gemma4` arch — from canonical Google safetensors via `tools/hf_to_gguf.sh --bench-convert`)
- Gemma 4 MoE (`gemma4` arch, auto-detected via `[ffn_alt]` GGUF weights)
- LLaDA-MoE-7B-A1B-Instruct gguf (`llada-moe` arch) — MoE diffusion model
- LLaDA-8B-Instruct gguf (`llada` arch) — dense diffusion model
- Qwen3.5-9B.st/ (`qwen35` arch)
- LLaDA-8B-Instruct.st/ (`llada` arch)
- gemma-4-E4B-it.st/ (`gemma4` arch, dense) — verified bit-identical to the F16 GGUF path via `make equiv-test` (`gguf-st` mode)

**Multimodal models**: Gemma 4 dense variants are released as text+image+audio multimodal models. `tools/hf_to_gguf.sh --bench-convert` automatically produces a paired `mmproj-<name>.gguf` sidecar alongside the decoder GGUF when `config.json` declares a vision/audio/video config. **Image input is implemented and passing equivalence** (`test_vision_equiv.sh`): the ViT encoder + projector run from the mmproj sidecar and splice into the decoder token stream (see `BuildVisionGraph` in `arch/vision.go`, `vision_splice.go`, and ARCHITECTURE.md "Vision / Multimodal Subsystem"). Audio input is not yet wired (the audio tower ships in the mmproj but has no input path). Sidecar pairing and filtering rules: SPEC.md §"Model Format Contract".

- **Two model formats**: GGUF (`*.gguf`) and safetensors (`*.st/` directories) — layout, tokenizer sidecar, and auto-detection: SPEC.md §"Model Format Contract".
- KV-cached and stateless inference; TOML DSL drives loading, graph construction, and cache allocation regardless of source format. Only C code is a thin model-agnostic ggml op wrapper. Zero C++.
- HTTP server: Bearer auth, model listing, streaming SSE + non-streaming completions, logprobs, timing/throughput in `usage` response — full contract: SPEC.md §"HTTP API".
- **Zero model-specific code**: chat templates executed from GGUF `tokenizer.chat_template` via gonja; BOS/EOS from GGUF metadata. Thinking mode controlled entirely by template `enable_thinking` variable — no post-render prompt manipulation.
- BPE tokenizer: dual mode — GPT-2 byte-level (Llama, Qwen) + SentencePiece (Gemma); greedy + top-p sampling; logprob computation via stable log-softmax
- KV cache + SSM state: prefill in one pass, decode one token at a time
- Stateless mode (`bench_custom.stateless`) — only mode supporting `ForwardCaptures`. Data collection and KV-cache optimization are intentionally separated; do not add capture to `ForwardCached`.
- SVG architecture visualizer: `bench gen-arch-diagram` generates `*.arch.svg` and `*.layers.svg` from TOML

## Architecture

- **Language**: Go + pure C via CGo (ggml op wrappers only)
- **Model definition**: TOML DSL (`models/arch/*.arch.toml`)
- **Inference backend**: ggml (git submodule), gpu-accelerated
- **API**: OpenAI-compatible chat completions plus bench-specific extensions. Endpoints, request/response shapes, and extension fields: SPEC.md §"HTTP API". Debugging aid: request-level overrides of server config defaults are logged as `[req] param overrides: ...`.
- **Diagnostics**: `/diag/` serves files from `bin/diag/` — a useful location for dumping R&D diagnostic output (SVGs, etc.) that can be viewed in-browser while the server is running.
- **Config**: `config/api_config.toml`. Section/key list, defaults, and per-request override precedence: SPEC.md §"Configuration Contract".

### ggml wrappers

Type and signature conventions, context sizing, and the `NilTensor` sentinel: ARCHITECTURE.md §"CGo Layer" and Key Invariants #3, #18–21. The `//export` collision hazard on `ggmlGoLogCallback` is under §"Internal Logging" below.

## Source Layout

```
Makefile                        Top-level: test, integration-test, equiv-test, symlinks bin/ delegates the rest to src/
config/
  api_config.toml               API server config (models.default, auth, listen, max_seq_len)
models/
  *.gguf                        Model files (not committed)
  *.st/                         Safetensors directories (not committed)
  arch/                         Architecture TOML definitions + generated SVGs
    MODEL_ARCH_TOML_DSL_SPEC.md DSL specification
    MODEL_ARCH_STMAP_TOML_DSL_SPEC.md STMap file DSL for safetensors name mapping
    block_svg/                  Hand-crafted SVG fragments per block builder
src/
  Makefile                      Go build, ggml build, test
  go.mod / go.sum               Module: inference-lab-bench
  cmd/                          CLI entry point — cobra subcommands, thin wrappers only
  internal/
    log/                        Structured leveled logger; see ### Internal Logging
    util/                       Project-wide utilities (LoadTOML, WriteJSON, BenchPaths, ResolvePaths, extension constants)
    apiserver/                  HTTP handlers, config loading, /ctl and /diag endpoints
    model/                      Model discovery: GGUF scanning + safetensors directory enumeration
    inference/                  Engine, generation strategies (one generate_*.go per strategy), tokenizer, sampler, metrics
      arch/                     The data-driven core: TOML DSL parsing, param/weight resolution, block builders
                                (one block_*.go per builder), model readers per format, cache, forward pass, vision
      archdiagram/              SVG diagram renderers (separate package)
    ggml/                       Go wrappers for ggml ops (~90 functions); all CGo confined here
  ggml_lib/                     C op wrappers + ggml build
  third_party/ggml/             ggml git submodule
tools/
  test_inference.py             Test harness implementation (stdlib-only python; streams SSE; loop mode; all-models mode; invoked via test_inference.sh) - stateless prompt submission to allow for perfectly repeatable testing
test_inference.sh               Thin launcher → tools/test_inference.py (preserves env-var/CLI contract; works with bench or llama-server via USE_LLAMA)
test_chat_equiv.sh              Logprob equivalence test: bench vs llama-server, stateless vs non-stateless, flash vs standard
test_vision_equiv.sh            Vision equivalence gate: bench vs llama-server (llama mode) or bench GGUF vs .st (gguf-st mode); semantic + logprob comparison
```

All SVG renderers share a single palette from `archdiagram/palette.go:Pal()`. To change any color, update that map only.

## Build & Run

The `make` targets are not thin aliases: they handle ggml C compilation, cgo flags, symlinks, and output placement. A direct `go build` produces a binary that is missing all of it.

```bash
make                    # build bench binary, symlink config+models into bin/
make serve              # build + start API server (--log bin/bench.log --log-level INFO)
make arch-diagrams      # rebuild SVG diagrams from models/arch/*.arch.toml
make st-tok-ggufs       # (re)generate tokenizer.gguf sidecar in every models/*.st/ dir
make test               # run go unit tests
make integration-test   # test inference end to end for all models
make equiv-test         # logprob equivalence vs llama-server + GGUF↔safetensors cross-check

# CLI subcommands
./bin/bench serve-api
./bin/bench gen-arch-diagram [--layers] [--blocks] <input.toml> [output.svg]

# Safetensors conversion tools
tools/setup_venv.sh       One-time setup: Python venv + llama.cpp convert script
tools/hf_to_gguf.sh             Tokenizer sidecar generation + full convert passthrough
tools/hf_to_gguf.sh --bench-tokenizer <model-name>  # generates tokenizer.gguf in .st/ dir

# After adding a new models/<name>.st/ directory, build its tokenizer.gguf sidecar:
make st-tok-ggufs               # builds the sidecar in every .st/ dir missing one
#   — the sidecar is required to load a safetensors model (bench's tokenizer
#     path is GGUF-only). `make serve` invokes this automatically; running
#     `./bin/bench serve-api` directly does not, so run it manually after
#     dropping a new .st/ directory into models/.

# Test harness (assumes server running; FORCE_NEW_SERVER=true to kill+restart)
./test_inference.sh "What is 2+2?"
./test_inference.sh --loop
FORCE_NEW_SERVER=true bash test_inference.sh "Hi"
MODEL=Qwen3.5-4B_abliterated.Q4_K_M MAX_TOKENS=100 bash test_inference.sh "Hi"
THINK=true ELIDE_THINK=false bash test_inference.sh "Hi"
ALL_MODELS=true bash test_inference.sh "Hi"   # test every loaded model in sequence
DIFFUSION_STEPS=32 DIFFUSION_BLOCK_LENGTH=64 bash test_inference.sh "Hi"   # diffusion params (ignored on autoregressive models)

# API
curl localhost:11116/api/v1/models
curl -X POST localhost:11116/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Hi"}]}'
```

**Validating Changes**

The acceptance gate for every change, without exception — mechanical fixes and refactors included:

```bash
make test && make integration-test && make equiv-test
```

No partial gates. `make integration-test` and `make equiv-test` cannot run concurrently with each other or with themselves (port and GPU resource contention) — all validation is strictly sequential.

* `make test` — Go unit tests.
* `make integration-test` — inference end-to-end for all discovered models via `test_inference.sh` (`ALL_MODELS=true` by default); fails on any `<NO-MODEL-RESPONSE>` or `[ERROR]` log line.
* `make equiv-test` — the authoritative correctness gate: four `test_chat_equiv.sh` modes plus `test_vision_equiv.sh`.

  | Mode | CHECK | REF | Proves |
  |---|---|---|---|
  | `llama` | bench (GGUF) | `llama-server` (Homebrew) | Bench's forward pass matches an independent reference implementation |
  | `stateless` | bench, cached | bench, stateless | The stateless (no-KV-cache) path matches the cached path |
  | `standard-attention` | bench, Flash Attention | bench, standard attention | FA2 and non-FA2 code paths agree |
  | `gguf-st` | bench, safetensors | bench, GGUF | Safetensors and GGUF loaders produce equivalent inference for the same model |

  Each mode compares the top-1 token's logprob on a fixed prompt for every applicable model; `|check_lp - ref_lp| > PASS_THRESH` is a hard failure. `PASS_THRESH` default: `0.001`.

* When adding or changing an architecture, also run `make arch-diagrams` and confirm the new arch renders correctly in the generated SVGs.
* Run `ALL_MODELS=true bash test_inference.sh "..."` before declaring any inference change complete, even though `make integration-test` already covers it by default — useful standalone during iterative debugging. Llama is easy mode — edge cases surface on Qwen3.5, Qwen3.5-MoE, DeepSeek2, and Gemma4.

**test_vision_equiv.sh** — vision equivalence gate. Two modes:

- `llama` (default) — cross-backend: bench (GGUF) vs llama-server (GGUF) on the same weights. `GGUF_ONLY=true` is set automatically because llama.cpp cannot load safetensors. CHECK = bench (system under test); REF = llama-server (authoritative reference).
- `gguf-st` — cross-format, bench only: bench's safetensors vision path (CHECK) vs bench's GGUF vision path (REF) for the same logical model; llama is not involved. Enumerates only logical models with both a GGUF decoder and a complete `.st/` directory.

Structure mirrors `test_chat_equiv.sh`: a per-mode trio of `setup_equiv_<name>` / `gather_check_<name>` / `gather_ref_<name>` functions plus shared `interstitial` and `finish`. CHECK is gathered first so a hard bench failure exits before the (slower) reference collection. Cross-function state is global by convention; per-function scratch is `local`.

**Prompt requirements** — both requirements stem from the same root cause: the two backends must see identical token layouts for logprob comparison to be meaningful.

1. **Image-first ordering**: the image marker must lead every prompt so the image is processed through the KV cache before any question text on BOTH engines. llama-mtmd always decodes the image ubatch first; an image spliced mid-sentence (e.g. `"Is @<img> a color…"`) gives bench a different image↔text causal layout than llama — that is a real, precision-invariant layout difference, not FP noise.
2. **Format-unambiguous wording**: avoid phrasings that induce a near-tie between token formattings. E.g. `"Answer in one word: left or right"` pulls ~8% mass onto a capitalized `Right` competitor, amplifying the FP floor into a visible logprob delta (blue-tie was 0.0186 with the suffix, 0.0019 without). When the choice is inherent and can't be dropped (color/grayscale), pin the exact tokens: `"Describe the image colorspace. Answer in one word with exactly one of 'color' or 'grayscale'."` — this collapsed gemma-31B color from 0.20 → 0.00000. The effect scales with model depth: small models tolerate the ambiguity, deeper models amplify it hard. See `memory/gemma-vision-logprob-deltas-are-prompt-artifacts.md` for the full three-instance record and the "bench faithful throughout" conclusion.

**`VISION_PASS_THRESH`** (default `0.0075`) — max `|check_lp - ref_lp|` answer-token logprob delta before a row's logprob column is FAIL. The residual after image-first + format-unambiguous prompting is the cross-backend F16 floor: the vision path is long and the mmproj is F16 on both sides, so identical ops accumulate sub-LSB differently — visible only on contested tail tokens. gemma-4 sits well under the floor (≤~0.003); **Qwen3.5 rides it at ~0.005–0.006** (subject/color), because its 27-layer encoder + decoder accumulates more F16 residual than gemma's shallower tower.

Crucially, that floor is **sensitive to the reference binary's build**. The reference is whatever `llama-server` the user has installed (Homebrew here), which is **intentionally not pinned** — anyone running the gate uses their own llama.cpp, and forcing a pinned system package (or building llama.cpp C++ under `tools/`, which is reference-source-only and not part of the build) is not reasonable. A llama.cpp reference *rebuild* shifts tail-token logprobs ~0.001 even when no vision or Metal op changed. Measured concretely: across the brew `9410 → 9430` bump the Qwen `subject`/`color` rows moved from `<0.004` to `~0.0057 / ~0.0043` while **bench output was byte-identical** and the reference's Qwen vision-graph + Metal-kernel code was **unchanged** (the only vision commit in `b9410..b9430` was additive DeepSeekOCR2 support). So the shift is pure cross-build FP variance at the floor, not an algorithm change on either side.

Because the reference is deliberately unpinned, the threshold **must absorb cross-build (and cross-machine) variance on the deepest model**: `0.004` sat *below* the Qwen floor and was not robustly reproducible across reference rebuilds; `0.0075` clears it with margin while still catching gross regressions (the 0.185 preproc-stretch class of bug). The vision logprob gate is consequently — and by design — **looser than text equiv** (which holds at 0.001): a deep F16 image path validated against an environment-variable reference cannot be held to the same tolerance, and that is not a bench defect. For diagnosing a *suspected* fidelity regression (vs. floor drift), use the per-element checkpoint-diff below, which is reference-build-robust. History: the gate was 0.02 → 0.0075 → 0.004 (prompt-framing artifacts removed in that progression) → 0.0075 (re-widened to the real reference-build-sensitive Qwen floor).

**Per-element checkpoint-diff is the real encoder-fidelity tool** — `bin/vision_capture` vs `MTMD_DEBUG_GRAPH=1 llama-mtmd-cli` (brew llama.cpp). The logprob gate is robust to encoder error (an 18% embedding error moved logprobs <0.001) and mainly catches gross regressions and wrong answers. Use checkpoint-diff when diagnosing suspected encoder or preprocessing divergence.

**Q4 note**: quantized rows legitimately exceed `VISION_PASS_THRESH` — quant activation noise amplifies the mmproj FP diff (vision-only; text Q4 is byte-identical). A Q4 failure is a quant-path concern, not an F16 fidelity regression.

**`.st` and llama mode**: llama-server cannot load safetensors, so every `llama`-mode comparison (both `test_chat_equiv.sh` and `test_vision_equiv.sh`) sets `GGUF_ONLY=true` to drop `.st` models from enumeration — the cross-backend comparison only runs GGUF models both backends can load. `.st` models are exercised instead by the `gguf-st` mode (bench `.st` vs bench GGUF), which doesn't involve llama. (`GGUF_ONLY` is a `test_inference.py` knob: keep only model ids backed by a local `<id>.gguf`.)

**Env knobs** (vision-equiv specific):
- `MODEL` — restrict to one model (default: all mmproj-capable in llama mode; all GGUF+`.st` pairs in gguf-st mode)
- `IMAGES` — comma/whitespace-separated image paths relative to project root (default: `test_data/vision_test.png`, which exercises the bilinear-resize path; `test_data/vision_test_960x624.png` is the no-resize variant for isolating resize bugs)
- `SKIP_BENCH` / `SKIP_LLAMA` — skip the respective side (llama mode only)
- `LOG_DIR` — log directory (default: `bin/test_vision_equiv_logs`)
- `VISION_PASS_THRESH` — logprob threshold (default: `0.0075`; reference-build-sensitive Qwen F16 floor — see threshold rationale above)

**Load-time defensive checks** — the loader runs three cheap sanity checks that turn silent numerical failures into loud load-time errors. Do not disable them without cause:
* `arch.ResolveParams` validates every required param is present and typed correctly; the loader aborts before VRAM allocation if any required key is missing or has a zero/garbage value that would cause silent divergence downstream (e.g. `rms_eps=0`).
* `ggml.ValidateRowData` inspects every tensor's raw bytes before upload — full element scan for float types, block scale/delta scan for quantized types (near-zero cost). Catches corrupt weight files and reader type/shape mismatches at load time rather than as NaN logits mid-generation.
* `ValidateLogits` runs at every sampler chokepoint (cached, stateless, diffusion) and fails the request if any logit is NaN/Inf. Diffusion wraps the error with `blockNum` / `step`; autoregressive paths wrap with `sample:`.

**Param dumps** — both GGUF and safetensors readers emit a sorted `[param] key = value` DEBUG dump of all metadata at load. Visual diff between the two is the fastest way to catch stmap errors when porting a new architecture to safetensors.

## Key Technical Details

### TOML Model DSL

`models/arch/*.arch.toml` declares:
- **Params**: GGUF metadata key mappings + derived arithmetic expressions + `[params.defaults]` fallbacks
- **Layer routing**: expression rules → block type per layer. `@{layer_idx}` = builtin; `${name}` = resolved GGUF param. Evaluated at load time.
- **Weight bindings**: GGUF tensor name templates with `blk.@{layer_idx}.` prefix expansion
- **Cache specs**: per-block-type tensor dimensions and dtypes; `shared` group name for layers that reuse a single cache tensor (e.g. non-KV layers in Gemma4)
- **FFN type**: `[ffn]` / `[ffn_alt]` for per-layer routing (e.g. dense for first N layers, MoE for rest — auto-detected from weights)
- **MoE FFN config** (`[ffn.config]` / `[ffn_alt.config]` for the `moe` builder):
  - `norm_w = "true"` — normalize router weights (static; always on)
  - `norm_w_param = "<param_name>"` — normalize router weights only when the named GGUF integer param is nonzero (dynamic; reads at load time). Mutually exclusive with `norm_w` — using both in the same `[ffn.config]` block is a validation error.
- **Layer routing**: `layers.routing.rule` (expression) OR `layers.routing.pattern` (array param name — nonzero → if_true)
- **Architecture flags** (`embed_scale`, `non_causal`, `generation`, `shift_logits`): effects and parse-time validation rules are SPEC.md §"Architecture Definition Contract".
- **Tokens**: `[tokens]` — `think_open`, `think_close`, `stop_tokens` (string array; each entry added to the generation stop set alongside EOS)

Optional GGUF params use `?` suffix (silently skipped if missing). Full spec: `models/arch/MODEL_ARCH_TOML_DSL_SPEC.md` — **read before writing or modifying any `.arch.toml`**.

### ggml Semantics (verified)

See ARCHITECTURE.md §"Verified ggml semantics".

### Qwen3.5 Architecture
- **Hybrid**: 32 layers — every 4th (3,7,11,...,31) is full softmax attention; rest are delta-net SSM
- **Full attention**: joint Q+gate projection, separate K/V, QK-norm, MRoPE ([11,11,10,0]), GQA (16Q/4KV heads, head_dim=256), sigmoid-gated output
- **Delta-net**: combined QKV projection, conv1d, L2-normalized Q/K, fused gated delta net op, rms_norm × silu(z) output gate
- **Common**: RMSNorm (eps=1e-6), SwiGLU FFN, post-attention norm; tied embeddings
- Exact params: `models/arch/qwen35.arch.toml`

### Gemma4 Architecture
- **ISWA**: layers alternate between sliding-window attention (SWA) and global attention; pattern-based routing via `layers.routing.pattern` (array param `swa_pattern`)
- **SWA**: `attention` builder with `sliding_window = "true"` config; SWA mask built from `sliding_window` GGUF param
- **Shared KV cache**: non-KV layers (SWA layers) reuse the K/V cache of the nearest global-attention layer via `n_kv_shared_layers` param; `CacheDef.Shared` group name for TOML-declared sharing
- **`SharedKVState`**: `{K, V map[string]Tensor}` — in-graph K/V propagated from KV layers to non-KV layers within the same forward pass; keyed by `shared_kv_group` config string
- **GeGLU FFN**: `geglu` builder — `gelu(gate * x) * (up * x) → down`; uses `ggml.Gelu`
- **Embed scaling**: `architecture.embed_scale = true` → multiply token embeddings by `sqrt(n_embd)` before first layer
- **Logit softcapping**: `logit_softcapping` param → `cap * tanh(logits / cap)` applied after final norm; uses `ggml.Tanh`
- **Per-layer embeddings**: `pe_inp_gate` weight per layer (injected into residual stream after FFN); handled by `perLayerEmbedInject` in `graph.go`
- **Post-attention/FFN norms**: `attn_post_norm`, `ffn_post_norm` common weights; applied after block output before residual add
- **Layer output scaling**: `layer_output_scale` weight; multiplied into residual stream after FFN + post-norm
- **`MulMatSetPrecF32`**: forces F32 accumulation on a matmul op (required for correct Gemma4 attention scores)
- Exact params: `models/arch/gemma4.arch.toml`

### Internal Logging

**Package**: `inference-lab-bench/internal/log`. Call-site API, output format, `Level` type, and initialization signatures are derivable from source. The non-obvious items are below.

**CLI flags** (on `serve-api`):
- `--log <path>` — log file path
- `--log-level <level>` — stderr level (`DEBUG|INFO|WARN|ERROR|NONE`), default `INFO`
- --log-file-line — enables logging of source file names and line numbers 
- `make serve` passes `--log bin/bench.log --log-level INFO`
- Batch commands (`gen-arch-diagram`) do not expose these flags

**ggml C library log routing** (`src/internal/ggml/logging.go`):

`ggml.InitLogging()` registers a CGo callback via `ggml_log_set` that routes all ggml C library diagnostic output through the Go logger (DEBUG for most levels; WARN/ERROR for ggml levels 3/4). Call it once in `serve_api.go` immediately after `log.InitLogger`. This eliminates all uncontrolled stderr output — `--log-level NONE` fully suppresses everything including ggml noise.

**CGo export constraint** — do not violate:

`ggmlGoLogCallback` in `logging.go` carries the `//export` directive. CGo auto-generates its C declaration in `_cgo_export.h`. A forward declaration for this symbol must NOT appear in `ggml_ops.h`. If it does, the CGo auto-generated declaration and the hand-written one will conflict at compile time. The symbol's C declaration is local to `ggml_ops.c` only.

**Constraints** (enforce these):
- `log.Fatal` / `os.Exit` placement: SPEC.md §"System Invariants".
- Never pass user-controlled values in the format string position — always use `%s`/`%v` args.
- Think content must be capped at 500 chars before logging; request string fields at 64 chars.
- `LOG_LEVEL` env var does not exist — level comes only from `--log-level`.
- No slog dependency; no third-party logging dependency.

### Path Resolution and Injection

The structural rule — `ResolvePaths` called once in `bench/`; `BenchPaths` injected downstream; library code never calls `ResolvePaths` — is covered by SPEC.md §"System Invariants".

Unique fact: `BENCH_EXE_DIR` env var is honored by `ResolvePaths` for debug/IDE configurations where source CWD and runtime CWD diverge.

### Model Loader Phase Structure (`arch/model.go`)

See ARCHITECTURE.md §"Model Loading" for the full phase sequence and `WeightStore` ownership cut line. Invariant: when adding new phases, place them in the sequence and update `cleanupOnError`; do not pass loose state by parameter.

## Adding a Model Architecture

**Critical invariant**: zero model-specific Go code. Never write `if arch == "foo"` anywhere in Go. All architecture differences must be expressed in the TOML DSL and generic block builders.

### Phase 0: Research — Understand the Architecture

Before writing any code, build a complete picture of the model.

1. **Read the llama.cpp reference implementation** — the authoritative source for how the architecture actually works at inference time:
   - ```make llama-cpp``` will clone our reference version 
   - `tools/llama.cpp/src/models/<arch>.cpp` — full forward pass (layer loop, attention, FFN, any novel ops)
   - `tools/llama.cpp/src/llama-arch.cpp` — GGUF tensor name mappings
   - `tools/llama.cpp/src/llama-hparams.h` — parameter names, defaults, helper functions (e.g., `is_swa()`, `has_kv()`)
   - `tools/llama.cpp/src/llama-model.cpp` — model init (look for arch-specific overrides like `f_attention_scale`, `rope_type`, `layer_reuse_cb`)
   - `tools/llama.cpp/src/llama-graph.cpp` — shared graph construction (attention, KV cache writes, mask construction)
   - `tools/llama.cpp/conversion/<arch>.py` — the authoritative **HF→GGUF tensor/param mapping** for this architecture (its `modify_tensors` / `set_gguf_parameters`). Requires `make llama-cpp` (lives inside the gitignored `tools/llama.cpp/` clone). Best reference when writing the `.arch.stmap.toml` for a safetensors port — one file for "how weights map," paired with `src/models/<arch>.cpp` for "how they're used."

2. **Scan the model file or directory** — verify actual metadata keys and tensor names:

   **GGUF**:
   ```python
   # Use the python gguf library to inspect metadata and tensor inventory
   import gguf
   reader = gguf.GGUFReader("models/<model>.gguf")
   for field in reader.fields.values(): print(field.name, ...)
   for t in reader.tensors: print(t.name, t.shape)
   ```

   **Safetensors** — inspect `config.json` and `model.safetensors.index.json` (or shard headers):
   ```python
   import json
   with open("models/<model>.st/config.json") as f: cfg = json.load(f)
   for t_name, t_info in index["weight_map"].items(): print(t_name, t_info["dtype"], t_info["shape"])
   ```

   Common surprises: tensors named without `.weight` suffix, global vs per-layer tensors, bool arrays stored as GGUF bool type, scalar params stored as arrays.

   **If using safetensors**, you will also need to create a `.arch.stmap.toml` mapping HF names → GGUF-equivalent names (see `models/arch/MODEL_ARCH_STMAP_TOML_DSL_SPEC.md`). The stmap is per-architecture, not per-model — all variants of the same architecture share one stmap file.

3. **Read existing arch TOMLs** — find the closest existing architecture and understand the DSL patterns:
   - `models/arch/MODEL_ARCH_TOML_DSL_SPEC.md` — authoritative DSL spec
   - `models/arch/*.arch.toml` — existing architectures as templates

4. **Identify novel features** — catalog what this architecture does that no existing builder supports. Common categories:
   - New activation functions (need ggml op bindings)
   - New attention variants (config overrides on existing `attention` builder vs new builder)
   - New layer routing patterns (expression-based vs array-based)
   - New cache sharing patterns (param-driven shared KV, sliding window)
   - New pre/post-processing (embedding scaling, logit capping, per-layer embeddings)

### Phase 1: Add Missing ggml Ops (if needed)

If the architecture needs ggml ops not yet exposed:
1. `src/ggml_lib/src/ggml_ops.h` — add function declaration
2. `src/ggml_lib/src/ggml_ops.c` — add implementation (typically a one-line cast+forward to the underlying ggml function)
3. `src/internal/ggml/ops.go` — add Go wrapper function

Pattern: each op is 1 line .h + 1 line .c + 1 function in Go. Keep it mechanical.

### Phase 2: Add New Block Builders (if needed)

Only needed if the architecture uses a computation pattern not covered by existing builders. Check the builder registry in `blocks.go` first.

**Available builders** (as of this writing):
- `attention` — standard multi-head attention with GQA, RoPE. Highly configurable via block config: per-block head dim/count overrides, optional K/V (shared KV for non-KV layers), V-norm, sliding window mask, RoPE frequency factors, NeoX rope mode, custom kq_scale. This is the most flexible builder — prefer extending it via config over writing a new one.
- `full_attention_gated` — gated attention with joint Q+gate projection, MRoPE (Qwen3.5)
- `mla_attention` — multi-latent attention with low-rank Q/KV (DeepSeek2/GLM-4)
- `gated_delta_net` — delta net SSM (Qwen3.5 hybrid)
- `swiglu` — SiLU-gated FFN
- `geglu` — GELU-gated FFN (Gemma 4)
- `moe` — mixture of experts with optional shared expert and expert bias

To add a new builder:
1. `src/internal/inference/arch/block_<type>.go` — implement `BlockBuilder` or `FFNBuilder` (see interfaces in `blocks.go`). Must implement both `BuildStateless` and `BuildCached`.
2. Register in `init()` in `blocks.go`
3. Define `Contract()` — required/optional weights, required params, config schema with allowed values
4. `models/arch/block_svg/<name>.svg` — SVG snippet for diagram rendering

### Phase 3: Extend the TOML DSL (if needed)

If the architecture needs TOML/param features not yet supported:
- **New param types**: `params.go` handles int, float, string, int arrays, bool arrays. Add new `Get*` methods to the `ModelReader` interface if needed (and to both GGUF and safetensors adapters). Note: safetensors models only provide scalar params from `config.json` — array types (`GetArrInts`, `GetArrBools`) are not available from that format.
- **New routing patterns**: `weights.go:resolveBlockName()` supports expression-based (`rule`) and array-based (`pattern`) routing. Array routing uses `IntArr[pattern][layer_idx]` — nonzero → `if_true`, zero → `if_false`.
- **New cache sharing**: `cache.go:NewCache()` supports param-driven sharing via `n_kv_shared_layers` (layers past `nLayers - nKVShared` reuse the last KV layer's cache per block type).
- **Architecture-level flags**: `arch.go:ArchMeta` — add fields like `EmbedScale bool`, `TiedEmbeddings bool`, `NonCausal bool`.

### Phase 4: Extend graph.go (if needed)

`graph.go` owns the layer loop and pre/post-processing. If the architecture has novel features at this level:
- **Embedding scaling** — `m.Def.Architecture.EmbedScale` flag, applied after `GetRows`
- **Logit softcapping** — checked via `m.Params.Floats["logit_softcapping"]`, applied after LM head
- **Post-attention/post-FFN norms** — checked via `lt["attn_post_norm"]` / `lt["ffn_post_norm"]`
- **Per-layer embeddings** — `buildPerLayerEmbedSetup` + `perLayerEmbedInject` (Gemma 4 pattern)
- **Layer output scaling** — `lt["layer_output_scale"]` per-layer scalar
- **SWA mask** — second mask tensor built when `sliding_window` param exists

The layer loop ordering in `runLayers` is: `attn_norm → attention block → post_attn_norm → residual → ffn_norm → FFN → post_ffn_norm → residual → per-layer embed → layer_output_scale`. Verify this matches the reference implementation — ordering differences cause silent numerical divergence.

**Per-layer hook threshold**: `runLayers` implements optional per-layer behaviors as weight- or param-presence nil-checks (see its doc comment for the current list). Adding the (N+1)th hook at N ≥ 5 should trigger a design review: is this the right layer for the feature, or does it warrant an explicit per-layer hook registry? At that scale the readability cost of another nil-check outweighs the benefit of inlining another feature.

### Phase 5: Write the `.arch.toml` File

1. Copy the closest existing `.arch.toml` to `models/arch/<arch-name>.arch.toml`
2. Update `[architecture]` section (name, flags)
3. Update `[params]` — map param names to GGUF metadata keys. Use `?` suffix for optional params.
4. Update `[layers]` — count, prefix, routing
5. Update `[layers.common_weights]` — per-layer tensors shared across block types
6. Update `[blocks.*]` — per-block-type weights, config overrides, cache specs
7. Update `[ffn]` — FFN builder and weights

**If using safetensors**, also write `models/arch/<arch-name>.arch.stmap.toml` mapping HF names → GGUF-equivalent names (see `models/arch/MODEL_ARCH_STMAP_TOML_DSL_SPEC.md` for the full contract). The stmap covers:
- `[params]` — HF `config.json` keys → GGUF metadata keys (1:1 scalar mapping)
- `[layer_prefix].hf` — HF per-layer prefix with `{N}` substitution
- `[tensors]` — our short tensor names → HF short tensor names
- `[tensors.global]` — our short global names → HF full tensor names
- `[gguf_metadata]` — literal GGUF metadata values for arch-level constants the converter writes as fixed values (e.g. `qwen35.ssm.inner_size = 4096`)
- `[[transforms]]` — per-tensor F32 transforms (norm-shift, V-head reorder, etc.) when the converter applies non-trivial tensor data manipulation
- `[[derived_metadata]]` — GGUF metadata computed at load time from `config.json` keys (e.g. Gemma 4's `string_array_eq` for `layer_types` → `sliding_window_pattern`, `copy_param` when one source value populates multiple GGUF keys)
- `[[derived_tensors]]` — GGUF tensors synthesized at load time with no safetensors source (e.g. Gemma 4's `rope_freqs_proportional` for partial-rotary RoPE on top of full-rotary primitive)

**When to use which:**
- 1:1 scalar mapping → `[params]`
- Fixed value (same across all instances of this arch) → `[gguf_metadata]`
- Computed from `config.json` → `[[derived_metadata]]` (use existing op or add a new one to `derivedMetadataOps` in `model_reader_safetensors_derived.go`)
- Synthesized tensor (converter's `generate_extra_tensors` analogue) → `[[derived_tensors]]` (same pattern, `derivedTensorOps`)
- Per-tensor numeric transform (read source, compute, store) → `[[transforms]]`

If the converter has procedural metadata/tensor computation, declare it in the stmap rather than fork the safetensors reader. New op handlers register in Go-side maps the same way block builders do; the TOML side stays declarative.

**Common pitfalls:**
- GGUF tensor names: check for `.weight` suffix presence/absence. Our loader tries both.
- Global vs per-layer tensors: per-layer tensors use `blk.@{layer_idx}.` prefix. Global tensors need `[weights.global]` entries. A per-layer weight entry that references a global-only tensor will use the fallback lookup (raw suffix as global name).
- Config overrides: block config values that are strings reference param names (`head_dim = "head_dim_swa"` → look up `params.Ints["head_dim_swa"]`). Literal numeric values are also supported (`kq_scale = 1.0`).
- Bool arrays from GGUF: stored as `IntArr` (0/1 values) after conversion in `resolveParam`.
- Shared KV groups: blocks that share cache must declare matching `shared_kv_group` config values.

### Phase 6: Debug Numerical Equivalence

This is typically the hardest phase. The model will load and run before the output is correct.

**Debugging strategy** — isolate the first layer/component where values diverge from llama.cpp:

1. Add temporary debug prints in `graph.go` and `block_attention.go` to dump first few float values of intermediate tensors at key checkpoints (after norm, after attention, after FFN, etc.)
2. Add matching debug prints to the llama.cpp reference model (at `cb()` callsites)
3. Run both on the same prompt, compare values checkpoint by checkpoint
4. The first checkpoint where values diverge identifies the buggy component

**Common sources of numerical divergence:**
- Wrong RoPE mode (standard vs NeoX — check `rope_type` in llama.cpp model init)
- Wrong attention scale (`f_attention_scale` — some models use 1.0 instead of 1/sqrt(headDim))
- Param resolution errors — wrong GGUF key, config override pointing to wrong param name
- Weight not loading — global tensor expected at per-layer path, fallback silently returns NilTensor
- Layer loop ordering mismatch — post-norm before vs after residual add
- Wrong mask — SWA layers getting full mask or vice versa
- Shared KV not wired — non-KV layers getting nil K/V from SharedKV because group names don't match

### Verification

Run the acceptance gate — see §"Validating Changes", including the `make arch-diagrams` step that applies when an architecture is added or changed.

**Ensure Docs Stay Up To Date** - CONVENTIONS.md, README.md, ARCHITECTURE.md, SPEC.md must be brought up to date when committing checkpoints.
