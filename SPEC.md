# SPEC – Inference Lab Bench

Normative contract for Inference Lab Bench: what the system must do. Package
map, data flow, and forward-pass mechanics live in `ARCHITECTURE.md`; working
practice and how-to procedures live in `CONVENTIONS.md`. This document does
not restate either.

---

## HTTP API

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/models` | List discovered models |
| POST | `/api/v1/chat/completions` | Chat completion (streaming or non-streaming) |
| GET | `/ctl/` | Control endpoint — `?memstats`, `?quit`, `?quit&now`, `?help` |
| GET | `/diag/*` | Static file server over `bin/diag/` (R&D diagnostic output) |

No other routes exist. There is no `/v1/*` legacy alias.

### Authentication

If `[server].auth_token` is set (non-empty) in `config/api_config.toml`, every
request must carry `Authorization: Bearer <auth_token>` or the server returns
`401`. If `auth_token` is empty (the default), no authentication is enforced.

### Chat Completions Request

`POST /api/v1/chat/completions` body:

| Field | Type | Behavior |
|---|---|---|
| `model` | string | Model ID from `/api/v1/models`, or `"default"` / `""` — resolves via `[models].default` (see Configuration Contract) |
| `messages` | array of `{role, content}` | `content` is either a plain string, or an array of `{type:"text", text}` / `{type:"image_url", image_url:{url}}` parts (OpenAI multimodal shape). `image_url.url` must be a `data:` URI. Up to 16 images per request, 32 MiB per decoded data URI, 8192×8192 px decoded-pixel cap. |
| `stream` | bool | Enables SSE streaming (see Streaming) |
| `stream_options.include_usage` | bool | When `stream:true`, emit a final usage-only chunk before `[DONE]` |
| `max_tokens` | int | Generation cap; `0` = engine default |
| `temperature` | float | `>= 0` overrides server default; negative leaves the default |
| `top_p` | float | `> 0` overrides server default |
| `logprobs` | bool | Include per-token log-probabilities in the response |
| `top_logprobs` | int | Alternatives per token; when `logprobs:true` and `top_logprobs < 1`, defaults to `1` |
| `chat_template_kwargs` | object | Passed to the Jinja2 chat template. Reserved key: `enable_thinking` (bool) — see Thinking Mode below |
| `bench_custom` | object | Bench-specific extensions, all optional — see below |

**`bench_custom` fields:**

| Field | Type | Behavior |
|---|---|---|
| `stateless` | bool | `true` bypasses the KV cache — full-sequence recompute per token. Default `false` (cached). Only mode that supports `ForwardCaptures`; see ARCHITECTURE.md §"`ForwardCaptures`". |
| `flash_attention` | bool | Per-request override of `[inference].flash_attention`. `null`/absent = use server default. |
| `elide_thinking` | bool | Per-request override of `[inference].elide_thinking_default` — strip `<think>...</think>` from output. |
| `diffusion` | object `{steps, block_length, algorithm}` | Diffusion generation parameters. Ignored with a logged warning if the target model is not a diffusion architecture (`generation = "diffusion"` not set — see Architecture Definition Contract). `steps`/`block_length` of `0` use the model's built-in defaults. `algorithm`: `""` or `"confidence"` selects max-softmax unmasking. |

Fields not listed above (e.g. `rlb_*`, `pmm_*`) are unstable research
surface, not covered by this contract.

**Thinking mode**: controlled exclusively by `chat_template_kwargs.enable_thinking`
(bool). If absent, the server injects `[inference].enable_thinking_default`.
There is no other mechanism to enable or disable thinking — no prompt
injection, no post-render manipulation. `bench_custom.elide_thinking`
controls whether `<think>...</think>` content is stripped from the response;
it is independent of whether thinking is enabled.

### Response Schema — Non-Streaming

```json
{
  "id": "chatcmpl-...", "object": "chat.completion", "created": 0, "model": "...",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop",
    "logprobs": {"content": [{"top_logprobs": [{"id":0,"token":"...","logprob":0.0,"bytes":[...]}]}]}
  }],
  "usage": {
    "prompt_tokens": 12, "completion_tokens": 48, "thinking_tokens": 0, "total_tokens": 60,
    "prompt_tokens_per_sec": 240.5, "completion_tokens_per_sec": 35.2, "total_tokens_per_sec": 42.1,
    "prefill_seconds": 0.049, "decode_seconds": 1.374, "total_seconds": 1.423
  }
}
```

`finish_reason` is one of `"stop"`, `"length"` (token budget exhausted), or
`"error"`. `logprobs` is present only when the request set `logprobs:true`.
All `usage` fields except the token counts are `omitempty` — a metrics
collection failure yields an `usage` object with only the token counts.
`thinking_tokens` counts tokens produced while inside a `<think>` block,
independent of whether they were elided from `content`.

### Streaming (SSE)

`stream:true` returns `Content-Type: text/event-stream`, one `data: <chunk>`
line per event, terminated by `data: [DONE]`. Each content chunk has OpenAI's
`chat.completion.chunk` shape with a `delta.content` fragment. The first
chunk carries `delta.role:"assistant"` with no content. The final content
chunk carries `finish_reason` and, if `logprobs:true` was set, the
aggregated `logprobs` for the whole response (bench does not emit per-token
logprobs incrementally). A non-streaming `usage` object is **not** included
by default; it is included, as one extra chunk with `choices:[]` before
`[DONE]`, only when the request set `stream_options.include_usage:true`.

### Models Endpoint

`GET /api/v1/models` returns `{"object":"list","data":[{"id","object":"model","created","owned_by":"local"}, ...]}`
for every model discovered under `[models].directory` (see Model Format
Contract for discovery rules).

### Control Endpoint

`GET /ctl/?memstats` returns current VRAM/RAM allocation stats as JSON.
`GET /ctl/?quit` waits for in-flight requests to complete, then shuts down.
`GET /ctl/?quit&now` shuts down immediately (100ms timeout). `GET /ctl/?help`
(or no recognized query param) lists available commands.

---

## Configuration Contract (`config/api_config.toml`)

| Section | Key | Type | Default if key absent | Meaning |
|---|---|---|---|---|
| `[server]` | `host` | string | `"0.0.0.0"` | Listen address |
| | `port` | int | `11116` | Listen port; must be 1–65535 |
| | `auth_token` | string | `""` (no auth) | Bearer token required on every request when non-empty |
| `[models]` | `default` | string | `"first"` | `"first"`/`"last"` = first/last model in discovery order; any other value is treated as an explicit model ID, falling back to `"first"` with a warning if that ID is not found |
| `[inference]` | `max_seq_len` | int | `8192` | KV cache size in tokens; must be `>= 1` |
| | `max_request_seq_len` | int | `16384` | Hard cap on estimated-prompt-tokens + `max_tokens`; `0` disables the guardrail |
| | `strict_mode` | bool | `true` | `true` = reject over-limit requests (`400`); `false` = log a warning and allow |
| | `log_thinking` | bool | `false` | Log `<think>` content to stderr |
| | `enable_thinking_default` | bool | `false` | **No built-in default** — if this key is omitted from the TOML file, thinking is OFF, regardless of any comment or shipped-config value. Per-request override: `chat_template_kwargs.enable_thinking` |
| | `elide_thinking_default` | bool | `true` | Nil-if-absent, resolved to `true`. Per-request override: `bench_custom.elide_thinking` |
| | `single_resident_model` | bool | `true` | Nil-if-absent, resolved to `true`: loading a new model evicts the previous one. `false` keeps all loaded models resident. No per-request override. |
| | `flash_attention` | bool | `true` | Nil-if-absent, resolved to `true`: use Flash Attention 2 when head geometry allows (Metal FA2). Per-request override: `bench_custom.flash_attention` |

Unknown keys in the TOML file are logged as a warning, not rejected. Any
config key with a documented per-request override follows this precedence:
explicit non-null value in the request &gt; the resolved config value above.

---

## Model Format Contract

Two model formats are supported: GGUF (`*.gguf`) and safetensors
(`*.st/` directories). Format is auto-detected per model at discovery time;
inference code above the `ModelReader` interface (see ARCHITECTURE.md
§"Model Loading") is identical for both.

### GGUF

A standalone model is any `*.gguf` file under `[models].directory` whose
GGUF-declared architecture matches a `models/arch/<name>.arch.toml` file.
Files whose basename starts with `mmproj-` are never listed as standalone
models — they are vision/audio sidecars, consumed only when paired to a
decoder GGUF (below).

### Safetensors (`.st/`)

A safetensors model is a directory named `<id>.st/` under
`[models].directory`, containing:

```
<id>.st/
  model-NNNNN-of-MMMMM.safetensors   (or a single model.safetensors)
  model.safetensors.index.json        (sharded only)
  config.json                         HF model config
  tokenizer.gguf                      tokenizer sidecar (required, vocab only)
```

`config.json["architectures"][0]` must match the `hf_class` declared by
some `models/arch/<name>.arch.stmap.toml` file (see
`models/arch/MODEL_ARCH_STMAP_TOML_DSL_SPEC.md`), and that stmap's base
name must match an existing `<name>.arch.toml`. A `.st/` directory missing
`config.json`, `tokenizer.gguf`, or a resolvable stmap is not a loadable
model.

### Tokenizer Sidecar Requirement

Bench's tokenizer path is GGUF-only. Safetensors models supply no tokenizer
of their own; `tokenizer.gguf` — a minimal GGUF carrying only tokenizer
metadata, generated via `make st-tok-ggufs` (or automatically by `make
serve`) — is required for a `.st/` directory to load. There is no
HuggingFace `tokenizer.json` parsing path.

### Vision (`mmproj`) Sidecar Pairing and Filtering

For GGUF decoders, a paired `mmproj-<name>.gguf` sidecar (containing the
vision/audio encoder + projector) is bound automatically by filename-prefix
matching when the server is started with `--auto-mmproj` (default: off).
`mmproj-*.gguf` files are always excluded from the standalone model list,
regardless of `--auto-mmproj`. For safetensors models, vision weights live
inline in the `.st/` directory and are gated by the same `--auto-mmproj`
flag via `ModelInfo.MmprojEnabled`.

### Format Auto-Detection and Precedence

If both `<id>.gguf` and `<id>.st/` exist for the same `id`, GGUF is
preferred by default; pass `--prefer-st` to the server to prefer `.st/`
instead. Model discovery is a two-pass scan
(`*.gguf` glob, then `*.st/` directory enumeration) filtered to
architectures with a matching `.arch.toml`.

---

## Architecture Definition Contract (TOML DSL)

Full DSL syntax: `models/arch/MODEL_ARCH_TOML_DSL_SPEC.md`. Safetensors
name-mapping syntax: `models/arch/MODEL_ARCH_STMAP_TOML_DSL_SPEC.md`. This
section states only the requirements that make an `.arch.toml` valid or
invalid.

**Zero model-specific Go code.** Every supported architecture is fully
described by `models/arch/<name>.arch.toml`. No Go code may branch on an
architecture name (e.g. `if arch == "qwen35"`); all architecture-specific
behavior is expressed in the TOML DSL and the generic block-builder system.
Adding a new architecture that reuses existing block builders is a
data-writing operation — a new `.arch.toml` (and, for safetensors, a
matching `.arch.stmap.toml`) — never a Go code change.

**TOML is the data language.** Architecture definitions, safetensors name
maps, and the server config are all TOML. JSON is used only for the
OpenAI-compatible HTTP payloads.

**Architecture-level flags** (`[architecture]` in `.arch.toml`), validated
at parse time (`arch.Load()`):

| Flag | Effect | Validation |
|---|---|---|
| `embed_scale` | Multiply input token embeddings by `sqrt(n_embd)` before the layer loop | — |
| `non_causal` | Attention uses a zero mask (bidirectional) instead of a causal mask | Required when `generation = "diffusion"` |
| `generation` | `""` (default, autoregressive) or `"diffusion"` — selects the generation strategy `Engine.Generate()` dispatches to | `generation = "diffusion"` without `non_causal = true` is a validation error |
| `shift_logits` | Diffusion only: output position `p` reads logits at index `p-1` | Meaningful only under `generation = "diffusion"` |

A successful `arch.Load()` / `Validate()` means the definition is
structurally sound: required weights present per block's `Contract()`,
param references resolvable, and the flag constraints above satisfied.

---

## System Invariants

These hold across the whole codebase, not just the TOML DSL:

- **`os.Exit` / `log.Fatal` only at CLI entry.** `log.Fatal` is permitted
  only inside `bench/` cobra command handlers. Library and utility code
  (including `internal/util/paths.go`) returns errors; `util.ResolvePaths()`
  returns `(BenchPaths, error)`.
- **CGo confined to `internal/ggml/`.** No other package imports `"C"`.
  GGUF metadata is read via pure-Go `gguf-parser-go`; safetensors index and
  tensor data are parsed in pure Go.
- **Data capture is stateless-only.** `ForwardCaptures` is accepted only by
  `ForwardStateless`. `ForwardCached` has no capture parameter — cached mode
  has no single clean matrix to extract across prefill + per-token decode
  calls. Data collection and KV-cache optimization must not be entangled.
- **`ModelReader` is the format boundary.** Everything above
  `newGenericModelFromReader()` is format-agnostic. Adding a new model
  format requires only a new `ModelReader` implementation.
- **Tokenizer is GGUF-only.** See Model Format Contract above.

---

## Supported Architectures

An architecture is supported if `models/arch/<name>.arch.toml` exists; the
model manager filters out any model whose declared architecture has no
matching file. As of this writing:

| Architecture | `.arch.toml` | GGUF | Safetensors | Vision | Diffusion |
|---|---|---|---|---|---|
| Llama | `llama.arch.toml` | yes | — | — | — |
| Qwen3.5 (dense + MoE) | `qwen35.arch.toml`, `qwen35moe.arch.toml` | yes | yes | yes | — |
| DeepSeek-V2 | `deepseek2.arch.toml` | yes | — | — | — |
| Gemma 4 (dense + MoE) | `gemma4.arch.toml` | yes | yes | yes | — |
| LLaDA | `llada.arch.toml` | yes | yes | — | yes |
| LLaDA-MoE | `llada-moe.arch.toml` | yes | — | — | yes |

Vision (`yes`) means the arch TOML declares a `[vision]` section and the
model loads a paired `mmproj-*.gguf` (GGUF) or inline vision tensors
(`.st/`) per the Model Format Contract above. Diffusion (`yes`) means
`generation = "diffusion"` is set (see Architecture Definition Contract).
Known-working model files for each architecture are tracked in
`CONVENTIONS.md` §"Current Status" (not part of this contract — that list
changes as new checkpoints are validated).

---

## CLI Surface

### `bench serve-api`

Starts the HTTP API server.

| Flag | Default | Effect |
|---|---|---|
| `--config <path>` | `<exe-dir>/config/api_config.toml` | Config file path |
| `--host <host>` | — | Overrides `[server].host` |
| `--port <port>` | — | Overrides `[server].port` |
| `--log <path>` | — | Log file path (tee with stderr) |
| `--log-level <level>` | `INFO` | Stderr level: `DEBUG\|INFO\|WARN\|ERROR\|NONE` |
| `--log-file-line` | `false` | Include source file:line in log messages |
| `--prefer-st` | `false` | Prefer `.st/` over `.gguf` when both exist for the same model ID |
| `--auto-mmproj` | `false` | Auto-discover and bind `*mmproj*.gguf` sidecars for multimodal support |

### `bench gen-arch-diagram [flags] <input.toml> [output.svg]`

Renders SVG diagrams from an `.arch.toml` file.

| Flag | Default | Effect |
|---|---|---|
| `--layers <n>` | `0` (omit layer-pattern strip) | Layer count for the pattern strip; falls back to `[example].n_layers` from the TOML if `0` |
| `--blocks <dir>` | `<input-dir>/block_svg/` | Directory of per-block SVG fragments |

`output.svg` defaults to `<input-basename>.svg`. When the arch declares
`[vision]`, `<name>.vision.svg` is also emitted (and
`<name>.vision.layers.svg` if `[example].vision_n_layers > 0`). When the
arch declares `[ffn_alt]`, an additional `<name>-<alt-builder>.arch.svg` (+
layers variant) is emitted.

There is no `bench chat` subcommand — `serve-api` and `gen-arch-diagram`
are the complete CLI surface.
