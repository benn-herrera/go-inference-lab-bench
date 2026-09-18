package apiserver

import (
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"strings"
	"time"

	"inference-lab-bench/internal/inference"
	log "inference-lab-bench/internal/log"
	"inference-lab-bench/internal/util"
)

const (
	// charsPerTokenEstimate is a rough BPE heuristic for request-size guardrails.
	charsPerTokenEstimate = 4
	// minPromptTokenEstimate is a floor so very-short prompts don't escape the guardrail check.
	minPromptTokenEstimate = 10
)

// thinkFilter strips <think>...</think> sections from a token stream.
// Handles tags split across token boundaries. Accumulates think content for logging.
type thinkFilter struct {
	open        string // e.g. "<think>" — from arch TOML tokens.think_open
	close       string // e.g. "</think>" — from arch TOML tokens.think_close
	buf         string
	inThink     bool // true while inside think block; set initially if model starts in think mode
	think       strings.Builder
	thinkTokens int // tokens received while inThink was true at entry
	totalTokens int // total tokens received via feed()
}

func (f *thinkFilter) feed(token string) string {
	f.totalTokens++
	if f.inThink {
		f.thinkTokens++
	}
	f.buf += token
	var out strings.Builder
	holdClose := len(f.close) + 1
	holdOpen := len(f.open) + 1
	for {
		if f.inThink {
			if i := strings.Index(f.buf, f.close); i >= 0 {
				f.think.WriteString(f.buf[:i])
				f.buf = f.buf[i+len(f.close):]
				f.inThink = false
			} else {
				// hold tail in case close tag is split across tokens
				if len(f.buf) > holdClose {
					f.think.WriteString(f.buf[:len(f.buf)-holdClose])
					f.buf = f.buf[len(f.buf)-holdClose:]
				}
				break
			}
		} else {
			if f.open != "" {
				if i := strings.Index(f.buf, f.open); i >= 0 {
					out.WriteString(f.buf[:i])
					f.buf = f.buf[i+len(f.open):]
					f.inThink = true
					continue
				}
			}
			// hold tail in case open tag is split across tokens
			safe := len(f.buf) - holdOpen
			if safe <= 0 {
				break
			}
			out.WriteString(f.buf[:safe])
			f.buf = f.buf[safe:]
			break
		}
	}
	return out.String()
}

// flush returns any buffered content at end of generation.
//
// unclosedIsContent controls the fail-safe for a think block that was opened
// (or primed) but never closed before generation stopped:
//   - true:  surface the buffered remainder as CONTENT — the model ended its
//     turn without closing (out-of-norm), so emit the text rather than destroy
//     it. This is the fix for models that answer with thinking enabled but emit
//     no close tag.
//   - false: treat the remainder as incomplete think content and discard it
//     from output — e.g. the token budget was exhausted mid-think (length).
func (f *thinkFilter) flush(unclosedIsContent bool) string {
	if f.inThink {
		if unclosedIsContent {
			// Fail-safe: the block was opened/primed but never closed and
			// generation stopped normally. While inThink, feed() routed the
			// running text into f.think and held only the close-tag tail in
			// f.buf — so the full unclosed remainder is f.think + f.buf.
			// Surface all of it as content rather than destroy it, and reset
			// f.think so it isn't double-counted as elided reasoning.
			out := f.think.String() + f.buf
			f.think.Reset()
			f.buf = ""
			f.inThink = false
			return out
		}
		f.think.WriteString(f.buf) // incomplete think block — treat remainder as think content
		f.buf = ""
	}
	out := f.buf
	f.buf = ""
	f.inThink = false
	return out
}

// matches logging policy in CONVENTIONS.md
const maxThinkLogLen = 500 // cap think content in logs

const maxLogValLen = 64

func truncLogVal(s string) string {
	if len(s) > maxLogValLen {
		return s[:maxLogValLen] + "…"
	}
	return s
}

func (f *thinkFilter) logThinking(model string) {
	s := f.think.String()
	if s == "" {
		return
	}
	if len(s) > maxThinkLogLen {
		s = s[:maxThinkLogLen] + "...[truncated]"
	}
	log.Info("[think] model=%s %s", truncLogVal(model), s)
}

type chatMessage struct {
	Role    string         `json:"role"`
	Content messageContent `json:"content"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type chatCompletionRequest struct {
	Model              string             `json:"model"`
	Messages           []chatMessage      `json:"messages"`
	Stream             bool               `json:"stream"`
	StreamOptions      *streamOptions     `json:"stream_options,omitempty"`       // OpenAI: stream_options.include_usage emits a final usage chunk before [DONE]
	MaxTokens          int                `json:"max_tokens"`
	Temperature        float64            `json:"temperature"`
	TopP               float64            `json:"top_p"`
	LogProbs           bool               `json:"logprobs,omitempty"`             // include log-probabilities
	TopLogProbs        int                `json:"top_logprobs,omitempty"`         // number of top alternatives (default 1)
	ChatTemplateKwargs map[string]any     `json:"chat_template_kwargs,omitempty"` // passed to the chat template (llama.cpp-compatible); reserved key: "enable_thinking" (bool)
	BenchCustom        *benchCustomParams `json:"bench_custom,omitempty"`
}

type diffusionRequestParams struct {
	Steps       int    `json:"steps,omitempty"`
	BlockLength int    `json:"block_length,omitempty"`
	Algorithm   string `json:"algorithm,omitempty"`
}

type benchCustomParams struct {
	Stateless           *bool                   `json:"stateless,omitempty"`
	FlashAttention      *bool                   `json:"flash_attention,omitempty"`
	ElideThinking       *bool                   `json:"elide_thinking,omitempty"`
	EnableRLBOnPrefill  bool                    `json:"enable_rlb_on_prefill,omitempty"`
	UseRLBGen           bool                    `json:"use_rlb_gen,omitempty"`
	RLBAlpha            *float64                `json:"rlb_alpha,omitempty"`              // 0.0-1.0; nil = default (1.0, no blending)
	RLBHaltRule         string                  `json:"rlb_halt_rule,omitempty"`          // halt rule name ("" = default fixed ceiling); see parseHaltRule in generate_rlb.go for menu
	RLBTerminalHaltRule string                  `json:"rlb_terminal_halt_rule,omitempty"` // halt rule for the terminal block only ("" = reuse rlb_halt_rule); Tier 1 rules only have meaningful signal at the terminal block since upstream tops aren't yet projected through lm_head
	RLBMagnitudeNorm    bool                    `json:"rlb_magnitude_norm,omitempty"`     // output-recycling with residual magnitude normalization (decode only)
	UsePMM              bool                    `json:"use_pmm,omitempty"`                // enable Poor Man's Mythos two-stream decode
	PMMSlabFirst        *int                    `json:"pmm_slab_first,omitempty"`         // first slab block index (default 2); nil = default
	PMMSlabLast         *int                    `json:"pmm_slab_last,omitempty"`          // last slab block index (default 5); nil = default
	PMMCap              *int                    `json:"pmm_cap,omitempty"`                // max iterations per slab block during answer phase (default 2); nil = default
	PMMThinkCap         *int                    `json:"pmm_think_cap,omitempty"`          // max iterations per slab block during think phase (default 1); nil = default
	PMMThinkDisabled    *bool                   `json:"pmm_think_disabled,omitempty"`     // skip Stream B entirely during think phase
	PMMBlendAlpha       *float64                `json:"pmm_blend_alpha,omitempty"`        // shadow weight in logit blend: 1.0=pure fork, <1.0 lerps toward mainline (default 1.0)
	Diffusion           *diffusionRequestParams `json:"diffusion,omitempty"`
}

// --- Non-streaming response types ---

type choiceMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type choiceLogProbs struct {
	Content []choiceLogProbEntry `json:"content"`
}

type choiceLogProbEntry struct {
	TopLogProbs []inference.TopLogProb `json:"top_logprobs"`
}

type choice struct {
	Index        int             `json:"index"`
	Message      choiceMessage   `json:"message"`
	FinishReason string          `json:"finish_reason"`
	LogProbs     *choiceLogProbs `json:"logprobs,omitempty"`
}

type usage struct {
	PromptTokens           int     `json:"prompt_tokens"`
	CompletionTokens       int     `json:"completion_tokens"`
	ThinkingTokens         int     `json:"thinking_tokens,omitempty"`
	TotalTokens            int     `json:"total_tokens"`
	PromptTokensPerSec     float64 `json:"prompt_tokens_per_sec,omitempty"`
	CompletionTokensPerSec float64 `json:"completion_tokens_per_sec,omitempty"`
	TotalTokensPerSec      float64 `json:"total_tokens_per_sec,omitempty"`
	PrefillSeconds         float64 `json:"prefill_seconds,omitempty"`
	DecodeSeconds          float64 `json:"decode_seconds,omitempty"`
	TotalSeconds           float64 `json:"total_seconds,omitempty"`
}

type chatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []choice `json:"choices"`
	Usage   usage    `json:"usage"`
}

// --- Streaming delta types ---

type deltaContent struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type streamChoice struct {
	Index        int             `json:"index"`
	Delta        deltaContent    `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
	LogProbs     *choiceLogProbs `json:"logprobs,omitempty"` // populated only on the final chunk when logprobs was requested
}

type streamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
	Usage   *usage         `json:"usage,omitempty"` // populated only on the final chunk when stream_options.include_usage = true
}

// flattenImages collapses per-message decoded-image lists into a single
// ordered slice matching the order the chat template emits image
// placeholders: across messages by index (i ascending), and within each
// message by the order extractImages produced (ascending PartIndex). That
// ordering is the contract the vision splice in arch/graph.go relies on — the
// Nth flattened image must line up with the Nth `<|image|>` placeholder the
// template renders. total is the precomputed sum of inner lengths, used only
// to size the result.
func flattenImages(perMessage [][]decodedImage, total int) []inference.ChatImage {
	flat := make([]inference.ChatImage, 0, total)
	for _, imgs := range perMessage {
		for _, di := range imgs {
			flat = append(flat, inference.ChatImage{Image: di.Image})
		}
	}
	return flat
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.pending.Add(1)
	defer s.pending.Done()
	var req chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	// Resolve default model
	if req.Model == "" || req.Model == "default" {
		req.Model = s.resolveDefaultModel()
	}
	if req.Model == "" {
		log.Error("no models available")
		writeError(w, http.StatusNotFound, "no models available")
		return
	}
	if s.manager.Get(req.Model) == nil {
		log.Error("model not found: %s", req.Model)
		writeError(w, http.StatusNotFound, "model not found")
		return
	}

	eng, err := s.Engine(req.Model)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	params := inference.DefaultParams()

	// Resolve chat_template_kwargs: request overrides server default.
	// enable_thinking is a reserved key — we type-check it here (must be bool)
	// and mirror the resolved value into params.ThinkingEnabled for the
	// thinkFilter initial state and downstream generation logic.
	kwargs := make(map[string]any, len(req.ChatTemplateKwargs)+1)
	maps.Copy(kwargs, req.ChatTemplateKwargs)
	if raw, present := kwargs[inference.KwEnableThinking]; present {
		b, ok := raw.(bool)
		if !ok {
			writeError(w, http.StatusBadRequest, "invalid type for chat_template_kwargs.enable_thinking (expected bool)")
			return
		}
		params.ThinkingEnabled = b
	} else {
		params.ThinkingEnabled = s.cfg.Inference.EnableThinkingDefault
		kwargs[inference.KwEnableThinking] = params.ThinkingEnabled
	}
	params.ChatTemplateKwargs = kwargs

	if req.MaxTokens > 0 {
		params.MaxTokens = req.MaxTokens
	}
	if req.Temperature >= 0 {
		params.Temperature = float32(req.Temperature)
	}
	if req.TopP > 0 {
		params.TopP = float32(req.TopP)
	}
	if req.BenchCustom != nil && req.BenchCustom.Stateless != nil {
		params.Stateless = *req.BenchCustom.Stateless
	}
	params.LogProbs = req.LogProbs
	params.TopLogProbs = req.TopLogProbs
	if params.TopLogProbs < 1 && params.LogProbs {
		params.TopLogProbs = 1
	}

	// Thinking mode is controlled via the enable_thinking template variable
	// passed to gonja in EncodeChat — no prompt injection needed.

	// Phase 7 vision-plan: a request may carry typed multi-part content
	// (text + image_url). Decode any attached images up-front so a bad
	// payload returns 400 before any inference engine work begins, and
	// collect them as a flat ordered list to thread through to the splice
	// code. Per-message image lists below; the flat aggregate becomes
	// params.Images in render order (apiserver order == template render
	// order, since each multi-part message renders its parts in sequence).
	// Per-request image cap. ImageCount is a cheap part-type tally (no decode),
	// so summing it across all messages lets us reject an over-budget request
	// before allocating a single pixel buffer. extractImages also bounds a
	// single message, but the request-level total is the real contract — the
	// per-message bound alone would let an N-message request smuggle in
	// maxImagesPerRequest×N images.
	var totalImageParts int
	for _, m := range req.Messages {
		if m.Content.IsMultiPart() {
			totalImageParts += m.Content.ImageCount()
		}
	}
	if totalImageParts > maxImagesPerRequest {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("too many images: %d (limit %d per request)", totalImageParts, maxImagesPerRequest))
		return
	}

	perMessageImages := make([][]decodedImage, len(req.Messages))
	var totalImages int
	for i, m := range req.Messages {
		if !m.Content.IsMultiPart() {
			continue
		}
		imgs, err := m.Content.extractImages()
		if err != nil {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("messages[%d]: %s", i, err.Error()))
			return
		}
		perMessageImages[i] = imgs
		totalImages += len(imgs)
	}

	// Guardrails: enforce context length limits to prevent system overload
	if s.cfg.Inference.MaxRequestSeqLen > 0 && req.MaxTokens > 0 {
		// Estimate prompt token count via tokenizer.
		// This will be re-encoded inside Generate(), but we need the estimate for guardrails.
		// Sum character counts across all message roles — user, system, and assistant all
		// contribute to the prompt that the model must process.
		var promptChars int
		for _, m := range req.Messages {
			// Only text contributes to the char-based estimate. Image
			// tokens (~260 each for Gemma 4) aren't counted — keeping the
			// guardrail arch-agnostic. The engine fails loudly if total
			// prefill exceeds max_seq_len, so this is a coarse pre-filter
			// not a hard contract.
			promptChars += len(m.Content.TextOnly) + 1 // +1 for newline separator
		}

		estimatedPromptTokens := max(promptChars/charsPerTokenEstimate, minPromptTokenEstimate)
		totalEstimate := estimatedPromptTokens + req.MaxTokens
		if totalEstimate > s.cfg.Inference.MaxRequestSeqLen {
			if s.cfg.Inference.StrictMode {
				log.Error("request exceeds guardrail limit: estimated_prompt=%d + max_tokens=%d = %d > max_request_seq_len=%d",
					estimatedPromptTokens, req.MaxTokens, totalEstimate, s.cfg.Inference.MaxRequestSeqLen)
				writeError(w, http.StatusBadRequest, fmt.Sprintf("request exceeds context limit: %d tokens (prompt=%d + max_tokens=%d); max_request_seq_len=%d",
					totalEstimate, estimatedPromptTokens, req.MaxTokens, s.cfg.Inference.MaxRequestSeqLen))
				return
			} else {
				log.Warn("request exceeds guardrail limit: estimated_prompt=%d + max_tokens=%d = %d > max_request_seq_len=%d (strict_mode=false, allowing)",
					estimatedPromptTokens, req.MaxTokens, totalEstimate, s.cfg.Inference.MaxRequestSeqLen)
			}
		}
	}

	msgs := make([]inference.ChatMessage, len(req.Messages))
	for i, m := range req.Messages {
		if !m.Content.IsMultiPart() {
			msgs[i] = inference.ChatMessage{Role: m.Role, Content: m.Content.TextOnly}
			continue
		}
		// Multi-part: hand the chat template the structured parts list so
		// it emits per-type output (text inline, model's image_token
		// literal at image positions). image_url parts collapse to
		// `{type: "image"}` — pixel data flows separately via
		// params.Images.
		parts := make([]inference.ChatContentPart, len(m.Content.Parts))
		for j, p := range m.Content.Parts {
			switch p.Type {
			case "text":
				parts[j] = inference.ChatContentPart{Type: "text", Text: p.Text}
			case "image_url":
				parts[j] = inference.ChatContentPart{Type: "image"}
			default:
				writeError(w, http.StatusBadRequest, fmt.Sprintf("messages[%d].content[%d]: unsupported part type %q", i, j, p.Type))
				return
			}
		}
		msgs[i] = inference.ChatMessage{Role: m.Role, Parts: parts}
	}

	// Flatten the per-message image lists into a single ordered slice in
	// template-render order (see flattenImages). This matches the order the
	// chat template emits `<|image|>` placeholders, which is what the splice
	// code in arch/graph.go consumes.
	if totalImages > 0 {
		params.Images = flattenImages(perMessageImages, totalImages)
	}
	// Thinking mode is controlled via the enable_thinking template variable
	// passed to gonja in EncodeChat — no prompt injection needed.

	// Resolve flash attention: request overrides server config; null = use config.
	if req.BenchCustom != nil && req.BenchCustom.FlashAttention != nil {
		params.FlashAttention = req.BenchCustom.FlashAttention
	}
	if req.BenchCustom != nil && req.BenchCustom.Diffusion != nil {
		if !eng.IsDiffusion() {
			log.Warn("diffusion params in request ignored: model is not a diffusion model")
		} else {
			params.Diffusion = &inference.DiffusionParams{
				Steps:       req.BenchCustom.Diffusion.Steps,
				BlockLength: req.BenchCustom.Diffusion.BlockLength,
				Algorithm:   req.BenchCustom.Diffusion.Algorithm,
			}
		}
	}
	if req.BenchCustom != nil && (req.BenchCustom.UseRLBGen || req.BenchCustom.EnableRLBOnPrefill) {
		rlb := &inference.RLBParams{
			Prefill: req.BenchCustom.EnableRLBOnPrefill,
		}
		if req.BenchCustom.UseRLBGen {
			rlb.Decode = true
			rlb.HaltRule = req.BenchCustom.RLBHaltRule
			rlb.TerminalHaltRule = req.BenchCustom.RLBTerminalHaltRule
			rlb.MagnitudeNorm = req.BenchCustom.RLBMagnitudeNorm
			if req.BenchCustom.RLBAlpha != nil {
				rlb.Alpha = *req.BenchCustom.RLBAlpha
			}
		}
		params.RLB = rlb
	}
	if req.BenchCustom != nil && req.BenchCustom.UsePMM {
		pmm := &inference.PMMParams{}
		if req.BenchCustom.PMMSlabFirst != nil {
			pmm.SlabFirst = *req.BenchCustom.PMMSlabFirst
		}
		if req.BenchCustom.PMMSlabLast != nil {
			pmm.SlabLast = *req.BenchCustom.PMMSlabLast
		}
		if req.BenchCustom.PMMCap != nil {
			pmm.Cap = *req.BenchCustom.PMMCap
		}
		if req.BenchCustom.PMMThinkCap != nil {
			pmm.ThinkCap = *req.BenchCustom.PMMThinkCap
		}
		if req.BenchCustom.PMMThinkDisabled != nil {
			pmm.ThinkDisabled = *req.BenchCustom.PMMThinkDisabled
		}
		if req.BenchCustom.PMMBlendAlpha != nil {
			pmm.BlendAlpha = *req.BenchCustom.PMMBlendAlpha
		}
		params.PMM = pmm
	}
	params.Streaming = req.Stream

	// Log any request-level parameter overrides of server config defaults.
	{
		var overrides []string
		if req.BenchCustom != nil && req.BenchCustom.Stateless != nil && *req.BenchCustom.Stateless {
			overrides = append(overrides, "stateless=true")
		}
		if raw, present := req.ChatTemplateKwargs[inference.KwEnableThinking]; present {
			if b, ok := raw.(bool); ok && b != s.cfg.Inference.EnableThinkingDefault {
				overrides = append(overrides, fmt.Sprintf("enable_thinking=%v (server: %v)", b, s.cfg.Inference.EnableThinkingDefault))
			}
		}
		if req.BenchCustom != nil && req.BenchCustom.ElideThinking != nil && *req.BenchCustom.ElideThinking != s.cfg.Inference.ShouldElideThink() {
			overrides = append(overrides, fmt.Sprintf("elide_thinking=%v (server: %v)", *req.BenchCustom.ElideThinking, s.cfg.Inference.ShouldElideThink()))
		}
		if req.BenchCustom != nil && req.BenchCustom.FlashAttention != nil && *req.BenchCustom.FlashAttention != s.cfg.Inference.UseFlashAttention() {
			overrides = append(overrides, fmt.Sprintf("flash_attention=%v (server: %v)", *req.BenchCustom.FlashAttention, s.cfg.Inference.UseFlashAttention()))
		}
		if req.BenchCustom != nil && req.BenchCustom.EnableRLBOnPrefill {
			overrides = append(overrides, "rlb.prefill=true")
		}
		if req.BenchCustom != nil && req.BenchCustom.UseRLBGen {
			overrides = append(overrides, "rlb.decode=true")
		}
		if req.BenchCustom != nil && req.BenchCustom.UseRLBGen && req.BenchCustom.RLBAlpha != nil {
			overrides = append(overrides, fmt.Sprintf("rlb.alpha=%.2f", *req.BenchCustom.RLBAlpha))
		}
		if req.BenchCustom != nil && req.BenchCustom.UseRLBGen && req.BenchCustom.RLBHaltRule != "" {
			overrides = append(overrides, fmt.Sprintf("rlb.halt_rule=%q", req.BenchCustom.RLBHaltRule))
		}
		if req.BenchCustom != nil && req.BenchCustom.UseRLBGen && req.BenchCustom.RLBTerminalHaltRule != "" {
			overrides = append(overrides, fmt.Sprintf("rlb.terminal_halt_rule=%q", req.BenchCustom.RLBTerminalHaltRule))
		}
		if req.BenchCustom != nil && req.BenchCustom.UseRLBGen && req.BenchCustom.RLBMagnitudeNorm {
			overrides = append(overrides, "rlb.magnitude_norm=true")
		}
		if req.BenchCustom != nil && req.BenchCustom.UsePMM {
			overrides = append(overrides, "pmm=true")
			if req.BenchCustom.PMMSlabFirst != nil || req.BenchCustom.PMMSlabLast != nil {
				sf := -1
				sl := -1
				if req.BenchCustom.PMMSlabFirst != nil {
					sf = *req.BenchCustom.PMMSlabFirst
				}
				if req.BenchCustom.PMMSlabLast != nil {
					sl = *req.BenchCustom.PMMSlabLast
				}
				overrides = append(overrides, fmt.Sprintf("pmm.slab=[%d,%d]", sf, sl))
			}
			if req.BenchCustom.PMMCap != nil {
				overrides = append(overrides, fmt.Sprintf("pmm.cap=%d", *req.BenchCustom.PMMCap))
			}
			if req.BenchCustom.PMMThinkCap != nil {
				overrides = append(overrides, fmt.Sprintf("pmm.think_cap=%d", *req.BenchCustom.PMMThinkCap))
			}
			if req.BenchCustom.PMMThinkDisabled != nil && *req.BenchCustom.PMMThinkDisabled {
				overrides = append(overrides, "pmm.think_disabled=true")
			}
			if req.BenchCustom.PMMBlendAlpha != nil {
				overrides = append(overrides, fmt.Sprintf("pmm.blend_alpha=%.2f", *req.BenchCustom.PMMBlendAlpha))
			}
		}
		if len(overrides) > 0 {
			log.Info("[req] param overrides: %s", strings.Join(overrides, ", "))
		}
	}

	chunkID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	elide := s.cfg.Inference.ShouldElideThink()
	if req.BenchCustom != nil && req.BenchCustom.ElideThinking != nil {
		elide = *req.BenchCustom.ElideThinking
	}
	// Seed inThink only when WE prime an open think block in THIS request's
	// prompt — i.e. the template opens under thinking (WeOpenThinkBlock, computed
	// at load) AND this request enabled thinking. Both are required: the Qwen3
	// family is "always-thinking" and primes a *closed* <think></think> when
	// thinking is disabled, so an enable_thinking=false request answers directly
	// and must NOT be seeded. When we don't seed, the filter starts content-
	// default and feed()'s open-tag detection enters think-mode where the model
	// actually emits the open tag. Seeding from ThinkingEnabled alone destroyed
	// answers from models that answer without emitting the close tag.
	filter := &thinkFilter{open: eng.ThinkOpen(), close: eng.ThinkClose(), inThink: eng.WeOpenThinkBlock() && params.ThinkingEnabled}

	if req.Stream {
		// --- Streaming (SSE) ---
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		flusher, canFlush := w.(http.Flusher)

		sendChunk := func(delta deltaContent, finishReason *string) {
			chunk := streamChunk{
				ID:      chunkID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model,
				Choices: []streamChoice{{Index: 0, Delta: delta, FinishReason: finishReason}},
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			if canFlush {
				flusher.Flush()
			}
		}

		// Send role delta first
		sendChunk(deltaContent{Role: "assistant"}, nil)

		metrics, genErr := eng.Generate(msgs, params, func(token string) bool {
			if elide {
				if out := filter.feed(token); out != "" {
					sendChunk(deltaContent{Content: out}, nil)
				}
			} else {
				sendChunk(deltaContent{Content: token}, nil)
			}
			return true
		})

		result := s.finalizeGeneration(finalizeArgs{
			model:     req.Model,
			metrics:   metrics,
			genErr:    genErr,
			filter:    filter,
			elide:     elide,
			streaming: true,
			logThink:  s.cfg.Inference.LogThinking,
			onFlush: func(out string) {
				if out != "" {
					sendChunk(deltaContent{Content: out}, nil)
				}
			},
		})

		if genErr != nil && inference.IsComputeFailure(genErr) {
			// GPU backend is permanently poisoned after a compute failure.
			// Evict the engine now so the next request loads a fresh one.
			// We cannot change the HTTP status (headers already sent), but
			// finish_reason:"error" signals the failure to the client.
			log.Error("GPU compute failure for %s — evicting engine for recovery: %v", req.Model, genErr)
			s.evictEngine(req.Model)
		}

		stop := result.finishReason

		// Final finish-reason chunk. When logprobs was requested, attach the
		// aggregated per-token logprobs at choices[0].logprobs — OpenAI places
		// them at the choice level alongside delta. Bench delivers all of them
		// on this terminal chunk rather than per-content-chunk; the engine
		// callback signature does not currently expose per-token logprobs and
		// the aggregated form is sufficient for downstream consumers.
		var finalLogProbs *choiceLogProbs
		if req.LogProbs && metrics != nil && len(metrics.TokenLogProbs) > 0 {
			entries := make([]choiceLogProbEntry, len(metrics.TokenLogProbs))
			for i, tlp := range metrics.TokenLogProbs {
				entries[i] = choiceLogProbEntry{TopLogProbs: tlp.TopProbs}
			}
			finalLogProbs = &choiceLogProbs{Content: entries}
		}
		finishChunk := streamChunk{
			ID:      chunkID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: []streamChoice{{Index: 0, Delta: deltaContent{}, FinishReason: &stop, LogProbs: finalLogProbs}},
		}
		if data, err := json.Marshal(finishChunk); err == nil {
			fmt.Fprintf(w, "data: %s\n\n", data)
			if canFlush {
				flusher.Flush()
			}
		}

		// OpenAI stream_options.include_usage: emit a final chunk with choices=[]
		// and a populated usage object before [DONE]. Only sent when the client
		// opts in, matching OpenAI's semantics.
		if req.StreamOptions != nil && req.StreamOptions.IncludeUsage && metrics != nil {
			u := usage{
				PromptTokens:           metrics.PromptTokens,
				CompletionTokens:       metrics.CompletionTokens,
				ThinkingTokens:         filter.thinkTokens,
				TotalTokens:            metrics.PromptTokens + metrics.CompletionTokens,
				PromptTokensPerSec:     metrics.PrefillTokensPerSec(),
				CompletionTokensPerSec: metrics.TokensPerSec(),
				TotalTokensPerSec:      metrics.TotalTokensPerSec(),
				PrefillSeconds:         metrics.PrefillDuration.Seconds(),
				DecodeSeconds:          metrics.DecodeDuration.Seconds(),
				TotalSeconds:           metrics.TotalDuration.Seconds(),
			}
			usageChunk := streamChunk{
				ID:      chunkID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model,
				Choices: []streamChoice{},
				Usage:   &u,
			}
			data, _ := json.Marshal(usageChunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			if canFlush {
				flusher.Flush()
			}
		}

		fmt.Fprintf(w, "data: [DONE]\n\n")
		if canFlush {
			flusher.Flush()
		}

	} else {
		// --- Non-streaming ---
		// Always feed tokens through the filter for think-token counting;
		// also collect visible output when eliding.
		var visible strings.Builder
		var raw []byte

		metrics, genErr := eng.Generate(msgs, params, func(token string) bool {
			visible.WriteString(filter.feed(token))
			raw = append(raw, token...)
			return true
		})
		if genErr != nil {
			if inference.IsComputeFailure(genErr) {
				// GPU backend is permanently poisoned — evict now so the next
				// request loads a fresh engine.
				log.Error("GPU compute failure for %s — evicting engine for recovery: %v", req.Model, genErr)
				s.evictEngine(req.Model)
				writeError(w, http.StatusServiceUnavailable, "GPU compute failure; please retry")
			} else {
				writeError(w, http.StatusInternalServerError, genErr.Error())
			}
			return
		}

		var flushedTail string
		result := s.finalizeGeneration(finalizeArgs{
			model:     req.Model,
			metrics:   metrics,
			genErr:    genErr,
			filter:    filter,
			elide:     elide,
			streaming: false,
			logThink:  s.cfg.Inference.LogThinking,
			onFlush: func(out string) {
				flushedTail = out
			},
		})

		var content string
		if elide {
			content = strings.TrimSpace(visible.String() + flushedTail)
		} else {
			content = strings.TrimSpace(string(raw))
		}

		var logProbs *choiceLogProbs
		if req.LogProbs && metrics != nil && len(metrics.TokenLogProbs) > 0 {
			entries := make([]choiceLogProbEntry, len(metrics.TokenLogProbs))
			for i, tlp := range metrics.TokenLogProbs {
				entries[i] = choiceLogProbEntry{TopLogProbs: tlp.TopProbs}
			}
			logProbs = &choiceLogProbs{Content: entries}
		}

		resp := chatCompletionResponse{
			ID:      chunkID,
			Object:  "chat.completion",
			Created: created,
			Model:   req.Model,
			Choices: []choice{{
				Index:        0,
				Message:      choiceMessage{Role: "assistant", Content: content},
				FinishReason: result.finishReason,
				LogProbs:     logProbs,
			}},
			Usage: usage{
				PromptTokens:           metrics.PromptTokens,
				CompletionTokens:       metrics.CompletionTokens,
				ThinkingTokens:         filter.thinkTokens,
				TotalTokens:            metrics.PromptTokens + metrics.CompletionTokens,
				PromptTokensPerSec:     metrics.PrefillTokensPerSec(),
				CompletionTokensPerSec: metrics.TokensPerSec(),
				TotalTokensPerSec:      metrics.TotalTokensPerSec(),
				PrefillSeconds:         metrics.PrefillDuration.Seconds(),
				DecodeSeconds:          metrics.DecodeDuration.Seconds(),
				TotalSeconds:           metrics.TotalDuration.Seconds(),
			},
		}
		util.WriteJSON(w, resp)
	}
}

// finalizeArgs carries the inputs to finalizeGeneration. A struct keeps
// the parameter list readable as it exceeds what a positional signature
// would reasonably hold.
type finalizeArgs struct {
	model     string
	metrics   *inference.InferenceMetrics
	genErr    error
	filter    *thinkFilter
	elide     bool
	streaming bool // selects per-branch log ordering (metrics-first for SSE, metrics-last for JSON)
	logThink  bool // s.cfg.Inference.LogThinking
	// onFlush receives the thinkFilter tail content when elide is true.
	// For streaming, the callback writes an SSE chunk (preserving the
	// pre-refactor interleaving between [metrics] and [think] log lines).
	// For non-streaming, the callback stashes the tail for appending to
	// the response body.
	onFlush func(string)
}

// finalizeResult carries outputs needed by each response-shape branch.
type finalizeResult struct {
	finishReason string // OpenAI-compatible finish_reason string
}

// finalizeGeneration runs the end-of-generation work shared between the
// streaming and non-streaming branches: flushing the thinkFilter tail,
// logging think content, emitting the [metrics] log line, and computing
// the OpenAI-compatible finish_reason.
//
// Log ordering matches the pre-refactor sources:
//   - streaming: [metrics] first, then flush (onFlush sendChunks the tail),
//     then [think]
//   - non-streaming: flush first (onFlush stashes the tail for the body),
//     then [think] (gated on elide), then [metrics]
//
// Compute-failure eviction and the HTTP response shape stay at the call
// site — the two branches diverge too much on errors for a single helper
// to cover cleanly. feed()/flush() ordering is preserved: feed() is only
// called from inside eng.Generate's callback, and flush() runs here before
// any content is written out.
func (s *Server) finalizeGeneration(a finalizeArgs) finalizeResult {
	logMetrics := func() {
		if a.metrics == nil {
			return
		}
		log.Info("[metrics] prompt=%d completion=%d think=%d prefill=%.1fms decode=%.1fms total=%.1fms tok/s=%.1f",
			a.metrics.PromptTokens, a.metrics.CompletionTokens, a.filter.thinkTokens,
			float64(a.metrics.PrefillDuration.Microseconds())/1000,
			float64(a.metrics.DecodeDuration.Microseconds())/1000,
			float64(a.metrics.TotalDuration.Microseconds())/1000,
			a.metrics.TokensPerSec())
	}

	// Compute finish_reason BEFORE flushing so the flush fail-safe can branch on
	// it. `length` (max_tokens hit) = truncated mid-think → incomplete reasoning,
	// elide. Otherwise (stop / error) = model ended its turn without closing →
	// surface the unclosed remainder as content rather than destroy it.
	reason := finishReason(a.metrics, a.genErr)
	unclosedIsContent := reason != inference.FinishReasonLength

	if a.streaming {
		logMetrics()
		if a.elide {
			a.onFlush(a.filter.flush(unclosedIsContent))
		}
		if a.logThink {
			a.filter.logThinking(a.model)
		}
	} else {
		if a.elide {
			a.onFlush(a.filter.flush(unclosedIsContent))
			if a.logThink {
				a.filter.logThinking(a.model)
			}
		}
		logMetrics()
	}

	return finalizeResult{finishReason: reason}
}

// finishReason maps generation outcome to an OpenAI-compatible finish_reason string.
// Returns FinishReasonError on any error, FinishReasonLength if the token budget was
// exhausted, and FinishReasonStop otherwise (stop token hit or metrics unavailable).
func finishReason(m *inference.InferenceMetrics, err error) string {
	if err != nil {
		return inference.FinishReasonError
	}
	if m != nil && m.FinishReason == inference.FinishReasonLength {
		return inference.FinishReasonLength
	}
	return inference.FinishReasonStop
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	fmt.Fprintf(w, `{"error":{"message":%q,"type":"api_error"}}`, msg)
}
