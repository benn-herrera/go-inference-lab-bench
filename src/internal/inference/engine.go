package inference

import (
	"errors"
	"fmt"
	"image"
	"path/filepath"
	"time"

	ggufparser "github.com/gpustack/gguf-parser-go"

	"inference-lab-bench/internal/inference/arch"
	"inference-lab-bench/internal/log"
	"inference-lab-bench/internal/model"
	"inference-lab-bench/internal/ggml"
)

// ChatImage is a single decoded image attached to a chat request.
// Images travel as a flat ordered list on GenerateParams.Images rather than
// nested under ChatMessage — the splice code in arch/graph.go consumes them
// in template-render order (first `<|image|>` placeholder → Images[0], etc.),
// independent of which message they were attached to.
type ChatImage struct {
	Image image.Image // decoded pixels (preprocessing happens inside the engine)
}

// ErrComputeFailed is returned by Generate when the ggml Metal backend fails to
// execute a graph. The underlying Metal context is permanently poisoned after
// this error — the server must evict and re-create the engine to recover.
var ErrComputeFailed = arch.ErrComputeFailed

// IsComputeFailure reports whether err is (or wraps) a Metal compute failure.
func IsComputeFailure(err error) bool {
	return errors.Is(err, ErrComputeFailed)
}

// Engine runs inference for a single loaded model.
type Engine struct {
	model       *arch.GenericModel
	tokenizer   *Tokenizer
	nLayers     int
	maxSeqLen   int
	flashAttn   bool   // server default: use FA2 when head geometry allows
	diagDir     string // directory for diagnostic output (resolved by caller)
	weOpenThink bool   // true if the chat template injects think_open into the prompt under enable_thinking (computed once at load)
}

// NewEngine creates an inference engine for the given model.
// archDir is the directory containing architecture definition TOML files.
// diagDir is the directory where diagnostic files are written.
// maxSeqLen is the KV cache size in tokens (0 = default 8192).
// flashAttention is the server default for FA2 (true = enable when head geometry allows).
func NewEngine(memStats ggml.MemoryStats, info *model.ModelInfo, archDir, diagDir string, maxSeqLen int, flashAttention bool) (*Engine, error) {
	var (
		m   *arch.GenericModel
		tok *Tokenizer
		err error
	)

	archName := info.Metadata.Architecture
	archDef, err := arch.Load(archDir, archName)
	if err != nil {
		return nil, fmt.Errorf("loading arch def %q: %w", archName, err)
	}

	switch info.Format {
	case model.FormatSafetensors:
		// Load tokenizer from the tokenizer.gguf sidecar in the safetensors directory.
		// This is a minimal GGUF containing only tokenizer metadata (no weights).
		tokPath := filepath.Join(info.Path, "tokenizer.gguf")
		tokF, e := ggufparser.ParseGGUFFile(tokPath)
		if e != nil {
			return nil, fmt.Errorf("parsing tokenizer sidecar: %w", e)
		}

		tok, err = NewTokenizerFromGGUF(tokF, tokPath)
		if err != nil {
			return nil, fmt.Errorf("building tokenizer: %w", err)
		}

		m, err = arch.NewGenericModelFromSafetensors(memStats, maxSeqLen, archDef, info.Path, archDir, info.MmprojEnabled)
		if err != nil {
			return nil, fmt.Errorf("creating model: %w", err)
		}

	case model.FormatGGUF, "":
		// Existing GGUF path — functionally identical to prior code.
		f, err := ggufparser.ParseGGUFFile(info.Path)
		if err != nil {
			return nil, fmt.Errorf("parsing GGUF for tokenizer: %w", err)
		}

		tok, err = NewTokenizerFromGGUF(f, info.Path)
		if err != nil {
			return nil, fmt.Errorf("building tokenizer: %w", err)
		}

		// DSL-driven model loading — reuse the already-parsed GGUF to avoid double-parsing.
		m, err = arch.NewGenericModelFromGGUF(memStats, maxSeqLen, archDef, info.Path, archDir, f, info.MmprojPath)
		if err != nil {
			return nil, fmt.Errorf("creating model: %w", err)
		}

	default:
		return nil, fmt.Errorf("unsupported model format: %s", info.Format)
	}

	if maxSeqLen <= 0 {
		maxSeqLen = defaultMaxSeqLen
	}

	validateArchTokens(m.Def.Tokens, tok)

	// For safetensors, NLayers may be 0 if config.json lacked it. Fall back to
	// the resolved model param (config.json → params stmap).
	nLayers := info.Metadata.NLayers
	if nLayers == 0 {
		if n, err := m.Params.GetInt(arch.ParamNLayers); err == nil {
			nLayers = n
		}
	}

	// Resolve flash attention: check head geometry against Metal FA2 supported dims.
	headDim := m.HeadDim
	flashAttn := flashAttention && arch.FlashAttnSupported(headDim)
	switch {
	case !flashAttention:
		log.Info("flash attention: disabled by config")
	case !arch.FlashAttnSupported(headDim):
		log.Warn("flash attention: disabled (fallback) - head_dim=%d not in FA2 supported set", headDim)
	default:
		log.Info("flash attention: enabled (head_dim=%d)", headDim)
	}

	weOpenThink := computeWeOpenThink(tok, m.Def.Tokens.ThinkOpen, m.Def.Tokens.ThinkClose, info.ID)

	return &Engine{
		model:       m,
		tokenizer:   tok,
		nLayers:     nLayers,
		maxSeqLen:   maxSeqLen,
		flashAttn:   flashAttn,
		diagDir:     diagDir,
		weOpenThink: weOpenThink,
	}, nil
}

// computeWeOpenThink determines, once at load, whether the chat template — when
// thinking is enabled — primes an OPEN think block in the prompt, so the
// assistant response begins already inside it (WE open) rather than the model
// emitting the open tag itself.
//
// It renders a fixed trivial probe prompt with enable_thinking=true and tests
// (in token space, for robustness against exotic template construction) whether
// that prompt ENDS INSIDE AN UNCLOSED think_open — i.e. a think_open token
// appears with no think_close after it. Presence alone is insufficient: the
// Qwen3 family is "always-thinking" — it injects a *closed* `<think></think>`
// when thinking is disabled and an *open* `<think>` when enabled, so think_open
// appears in both renders. Only the open-vs-closed distinction discriminates.
//
// Returns false when think_open is unconfigured, not a single vocab entry, or
// the render fails — the safe default is "model opens" (content-default), which
// never destroys an answer.
func computeWeOpenThink(tok *Tokenizer, thinkOpen, thinkClose, name string) bool {
	flag := false
	if thinkOpen != "" {
		if openID, ok := tok.TokenID(thinkOpen); ok {
			closeID := int32(-1) // sentinel: think_close not a vocab token → never closes
			if cid, ok := tok.TokenID(thinkClose); ok {
				closeID = cid
			}
			// Fixed probe: a trivial user message that must NOT itself contain
			// think_open, so user content can't pollute the result.
			probe := []ChatMessage{{Role: "user", Content: "hi"}}
			onIDs, err := tok.EncodeChat(probe, map[string]any{KwEnableThinking: true})
			if err == nil {
				flag = promptOpensThink(onIDs, openID, closeID)
			} else {
				log.Warn("[think] probe render failed for model=%s — defaulting who_opens=model", name)
			}
		}
	}
	log.Info("[think] model=%s think_open=%q who_opens=%s", name, thinkOpen,
		map[bool]string{true: "engine", false: "model"}[flag])
	return flag
}

// promptOpensThink reports whether a token sequence ends inside an unclosed
// think_open block: a think_open token appears with no think_close after it.
// closeID < 0 means think_close is not a vocab token (treated as "never closes").
func promptOpensThink(ids []int32, openID, closeID int32) bool {
	lastOpen, lastClose := -1, -1
	for i, id := range ids {
		switch id {
		case openID:
			lastOpen = i
		case closeID:
			lastClose = i
		}
	}
	return lastOpen >= 0 && lastOpen > lastClose
}

// validateArchTokens checks each non-empty token string from the arch TOML
// against the loaded vocabulary. Warns if a string is not a direct vocab entry
// or encodes to zero tokens.
func validateArchTokens(tokens arch.TokensDef, tok *Tokenizer) {
	check := func(name, value string) {
		if value == "" {
			return
		}
		ids := tok.Encode(value, false)
		if len(ids) == 0 {
			log.Warn("arch token %s=%q encodes to zero tokens — not in vocabulary", name, value)
		} else if !tok.VocabContains(value) {
			log.Warn("arch token %s=%q is not a single vocabulary entry (encodes to %d BPE sub-tokens)", name, value, len(ids))
		}
	}
	check("think_open", tokens.ThinkOpen)
	check("think_close", tokens.ThinkClose)
	for i, s := range tokens.ExtraEOS {
		check(fmt.Sprintf("extra_eos[%d]", i), s)
	}
}

// IsDiffusion reports whether the loaded model uses diffusion generation.
func (e *Engine) IsDiffusion() bool {
	return e.model != nil && e.model.Def.Architecture.Generation == arch.GenerationDiffusion
}

// ThinkOpen returns the TOML-defined think opening tag, or "".
func (e *Engine) ThinkOpen() string { return e.model.Def.Tokens.ThinkOpen }

// ThinkClose returns the TOML-defined think closing tag, or "".
func (e *Engine) ThinkClose() string { return e.model.Def.Tokens.ThinkClose }

// WeOpenThinkBlock reports whether the chat template injects think_open into
// the prompt under the enable_thinking conditional (i.e. WE open the think
// block, so the assistant response begins already inside it). Computed once at
// load. When false, the model emits the open tag itself (content-default) — see
// computeWeOpenThink. Used to seed the apiserver thinkFilter.
func (e *Engine) WeOpenThinkBlock() bool { return e.weOpenThink }

// Close frees all GPU resources.
func (e *Engine) Close() {
	if e.model != nil {
		e.model.Close()
	}
}

// WeightStore returns the model's weight store (for memory stats).
// Returns nil if the engine is not loaded.
func (e *Engine) WeightStore() *arch.WeightStore {
	if e.model == nil {
		return nil
	}
	return e.model.Store
}

func (e *Engine) MemoryStats() ggml.MemoryStats {
	if ws := e.WeightStore(); ws != nil {
		return ggml.DevMemory(ws.GPU, ws.CPU)
	}
	return ggml.MemoryStats{}
}

// DiffusionParams groups diffusion-model-specific generation parameters.
// Ignored for autoregressive models.
type DiffusionParams struct {
	Steps       int    // 0 = use default (64)
	BlockLength int    // 0 = single block (global); >0 = block-based left-to-right
	Algorithm   string // "" or "confidence" = max-softmax; stub for future variants
}

// GenerateParams controls the generation loop.
type GenerateParams struct {
	MaxTokens          int
	Temperature        float32
	TopP               float32
	Stateless          bool             // bypass KV cache (for testing/comparison)
	Streaming          bool             // true if the caller is using SSE streaming
	ThinkingEnabled    bool             // mirrors ChatTemplateKwargs[KwEnableThinking]; used for thinkFilter init + internal logic
	ChatTemplateKwargs map[string]any   // passed to the chat template renderer; llama.cpp-compatible shape
	LogProbs           bool             // include log-probabilities in response
	TopLogProbs        int              // number of top log-probabilities per token (0 = just the chosen token)
	FlashAttention     *bool            // nil = use server default; true/false = per-request override
	Diffusion          *DiffusionParams // nil = not diffusion (ignored for autoregressive models)
	RLB                *RLBParams       // nil = RLB disabled; non-nil = run recurrent logic block generation with these params
	PMM                *PMMParams       // nil = PMM disabled; non-nil = run two-stream decode with these params

	// Images is the flat ordered list of decoded images attached to this
	// request, in template-render order. The Phase 7 splice in
	// arch/graph.go consumes them 1:1 with the `<|image|>` placeholder
	// token positions in the tokenized prompt. nil/empty for text-only
	// requests; the encoder + projector + splice are bypassed in that case.
	Images []ChatImage
}

// DefaultParams returns sensible defaults.
func DefaultParams() GenerateParams {
	return GenerateParams{
		MaxTokens:       512,
		Temperature:     0,
		TopP:            0.9,
		ThinkingEnabled: true,
	}
}

const defaultMaxSeqLen = 8192

// KwEnableThinking is the chat_template_kwargs key that controls
// whether the template emits a think block. Used across inference
// and apiserver.
const KwEnableThinking = "enable_thinking"

// Generate runs the full chat generation loop.
// Uses KV/SSM cache by default; set params.Stateless for full-sequence recomputation.
// Returns metrics for the generation request.
func (e *Engine) Generate(
	messages []ChatMessage,
	params GenerateParams,
	onToken func(token string) bool,
) (*InferenceMetrics, error) {
	// Ensure chat_template_kwargs carries enable_thinking for callers (e.g.
	// CLI paths) that only set ThinkingEnabled. The HTTP handler populates
	// both already; this is the backstop for direct Engine users.
	kwargs := params.ChatTemplateKwargs
	if kwargs == nil {
		kwargs = map[string]any{}
	}
	if _, ok := kwargs[KwEnableThinking]; !ok {
		kwargs[KwEnableThinking] = params.ThinkingEnabled
	}
	promptIDs, err := e.tokenizer.EncodeChat(messages, kwargs)
	if err != nil {
		return nil, fmt.Errorf("encoding chat: %w", err)
	}
	if len(promptIDs) == 0 {
		return nil, fmt.Errorf("empty prompt after encoding")
	}

	// Vision prefill: when images are attached, preprocess each, expand
	// the `<|image|>` placeholders in the tokenized prompt into N copies
	// (one per soft token after pooling), and capture the splice runs
	// the forward path uses to overwrite those positions with projected
	// encoder embeddings.
	var visionSpliceInputs []arch.VisionSpliceInput
	if len(params.Images) > 0 {
		// v1 limitation: vision is only wired into the vanilla cached /
		// stateless paths. Mixing with diffusion / RLB / PMM is rejected
		// with a clean error rather than silently producing gibberish.
		if e.IsDiffusion() {
			return nil, fmt.Errorf("vision input + diffusion generation is not supported")
		}
		if params.RLB != nil {
			return nil, fmt.Errorf("vision input + RLB generation is not supported")
		}
		if params.PMM != nil {
			return nil, fmt.Errorf("vision input + PMM generation is not supported")
		}
		if e.model.Def.Vision == nil {
			return nil, fmt.Errorf("request attached %d image(s) but model %q has no vision tower",
				len(params.Images), e.model.Def.Architecture.Name)
		}
		placeholderID, ok := e.tokenizer.TokenID(e.model.Def.Vision.ImageToken)
		if !ok {
			return nil, fmt.Errorf("vision placeholder token %q not in vocabulary", e.model.Def.Vision.ImageToken)
		}
		prefill, err := prepareVisionPrefill(e.model.Def, promptIDs, placeholderID, params.Images)
		if err != nil {
			return nil, err
		}
		if prefill != nil {
			promptIDs = prefill.ExpandedTokens
			visionSpliceInputs = prefill.toArchSpliceInputs()
			log.Info("vision prefill: %d image(s), expanded prompt %d → %d tokens",
				len(params.Images), len(prefill.Runs), len(promptIDs))
			for i, run := range prefill.Runs {
				log.Info("  image[%d] span: [%d, %d) (%d tokens, bidirectional mask)",
					i, run.Start, run.Start+run.Length, run.Length)
			}
		}
	}
	// The template handles enable_thinking natively — when disabled, it emits an
	// empty think block (e.g. "<think>\n\n</think>\n\n") which is the correct
	// prompt. No post-render stripping needed; this matches llama.cpp behavior.

	// Resolve effective flash attention: per-request override applies config flag;
	// geometry check is enforced at load time (e.flashAttn already incorporates it).
	flashAttn := e.flashAttn
	if params.FlashAttention != nil {
		flashAttn = *params.FlashAttention && arch.FlashAttnSupported(e.model.HeadDim)
	}
	params.FlashAttention = &flashAttn
	log.Info("flash_attention_used=%v", *params.FlashAttention)

	maxTokens := params.MaxTokens

	stopSet := e.tokenizer.BuildStopSet(e.model.Def.Tokens.ExtraEOS)

	metrics := &InferenceMetrics{
		PromptTokens: len(promptIDs),
	}

	start := time.Now()

	// RLB doesn't apply to diffusion models (no SSM state to blend, no
	// per-block recurrence concept). If both are requested, drop RLB and
	// fall through to the diffusion path.
	if params.RLB != nil && e.IsDiffusion() {
		log.Warn("rlb_gen requested on diffusion model; ignoring RLB, performing diffusion generation")
		params.RLB = nil
	}

	// PMM is decode-only autoregressive. It is incompatible with diffusion
	// (no autoregressive decode loop), with stateless (PMM relies on the
	// persistent K/V cache), and takes precedence over RLB if both are set.
	if params.PMM != nil {
		if e.IsDiffusion() {
			log.Warn("pmm requested on diffusion model; ignoring PMM, performing diffusion generation")
			params.PMM = nil
		} else if params.Stateless {
			log.Warn("pmm requires cached mode; ignoring stateless=true")
			params.Stateless = false
		}
	}
	if params.PMM != nil && params.RLB != nil {
		log.Warn("pmm: RLB also set; PMM takes precedence, RLB ignored")
		params.RLB = nil
	}

	var genErr error
	if params.PMM != nil {
		log.Info("pmm generation: prompt=%d tokens", len(promptIDs))
		genErr = e.generatePMM(promptIDs, maxTokens, stopSet, params, onToken, metrics)
	} else if params.RLB != nil {
		if params.Stateless {
			log.Warn("rlb_gen requires cached mode; ignoring stateless=true")
		}
		log.Info("rlb generation: prompt=%d tokens", len(promptIDs))
		genErr = e.generateRLB(promptIDs, maxTokens, stopSet, params, onToken, metrics)
	} else if e.IsDiffusion() {
		if params.Streaming {
			T := 64
			if params.Diffusion != nil && params.Diffusion.Steps > 0 {
				T = params.Diffusion.Steps
			}
			log.Warn("streaming request on diffusion model; response will be returned as a burst after %d denoising steps", T)
		}
		genErr = e.generateDiffusion(promptIDs, maxTokens, stopSet, params, onToken, metrics)
	} else if params.Stateless {
		log.Info("stateless inference (no KV cache), prompt=%d tokens", len(promptIDs))
		genErr = e.generateStateless(promptIDs, maxTokens, stopSet, params, visionSpliceInputs, onToken, metrics)
	} else {
		genErr = e.generateCached(promptIDs, maxTokens, stopSet, params, visionSpliceInputs, onToken, metrics)
	}

	metrics.TotalDuration = time.Since(start)

	return metrics, genErr
}

// enablePerTokenLogging gates verbose per-token diagnostic logging on the
// hot sampling path. Off by default — flipping it to true produces one log
// line per generated token with the sampled ID and top logit, useful when
// debugging flat-distribution or off-by-one-token bugs without rebuilding
// the engine harness. Promote to a config/runtime setting if it proves
// useful often enough to warrant live toggling.
const enablePerTokenLogging = false

func (e *Engine) sample(logits []float32, params GenerateParams) (int32, error) {
	if err := ValidateLogits(logits); err != nil {
		return 0, fmt.Errorf("sample: %w", err)
	}
	var next int32
	if params.Temperature <= 0 {
		next = Greedy(logits)
	} else {
		next = TopP(logits, params.Temperature, params.TopP)
	}
	if enablePerTokenLogging {
		log.Debug("sample: next=%d top_logit=%.4f n_vocab=%d temp=%.2f top_p=%.2f",
			next, logits[next], len(logits), params.Temperature, params.TopP)
	}
	return next, nil
}

