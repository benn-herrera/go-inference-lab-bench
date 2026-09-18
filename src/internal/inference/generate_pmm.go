package inference

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"inference-lab-bench/internal/inference/arch"
	log "inference-lab-bench/internal/log"
)

// generate_pmm.go — Poor Man's Mythos (PMM) two-stream decode.
//
// At each decode token:
//   Stream A (mainline): vanilla single-pass ForwardCached. Writes K/V to the
//     persistent cache and updates mainline SSM state. Provides the K/V history
//     that Stream B reads.
//   Stream B (shadow):   per-block iteration on blocks [SlabFirst, SlabLast]
//     using the dH_threshold halt rule with iteration cap. Reads K/V from the
//     mainline cache (never writes). Produces the logits used for sampling.
//     SSM state is forked fresh from mainline each token and discarded.
//
// PMM does not modify perBlockForwardCore. It supplies a unified halt closure
// that single-passes non-slab blocks and applies dH_threshold (capped) on slab
// blocks, plus no-op saveState/blendState.

// PMM defaults. Slab covers blocks 2..5 with cap=2 — small enough to be cheap
// per token, large enough to give the dH_threshold rule room to halt early on
// converged blocks. ThinkCap defaults to 1 so the think phase reduces to a
// single shadow pass (effectively pass-through), which prior testing showed
// is the only way to keep thinking models from collapsing to ~19 think tokens.
const (
	pmmDefaultSlabFirst  = 2
	pmmDefaultSlabLast   = 5
	pmmDefaultCap        = 2
	pmmDefaultThinkCap   = 1
	pmmDefaultBlendAlpha = 1.0 // pure fork; 0 < α < 1 blends toward mainline
	// pmmAgreeLogEvery controls how often agreement-rate is summarized to stderr.
	pmmAgreeLogEvery = 16
)

// PMMParams configures the PMM two-stream decode mechanism.
// Mirrors RLBParams / DiffusionParams: nil = PMM off; non-nil = run.
type PMMParams struct {
	SlabFirst     int     // first block index to iterate (default 2)
	SlabLast      int     // last block index to iterate (default 5)
	Cap           int     // max iterations per slab block during the answer phase (default 2)
	ThinkCap      int     // max iterations per slab block during the think phase (default 1)
	ThinkDisabled bool    // if true, skip Stream B entirely during think phase (use mainline logits)
	BlendAlpha    float64 // shadow weight in logit blend: 1.0=pure fork, 0<α<1 blends toward mainline (default 1.0; 0=use default)
}

// applyDefaults fills any zero-value field with its package default.
// Returns a copy so callers can keep the original around for debugging.
func (p PMMParams) applyDefaults() PMMParams {
	if p.SlabFirst == 0 {
		p.SlabFirst = pmmDefaultSlabFirst
	}
	if p.SlabLast == 0 {
		p.SlabLast = pmmDefaultSlabLast
	}
	if p.Cap == 0 {
		p.Cap = pmmDefaultCap
	}
	if p.ThinkCap == 0 {
		p.ThinkCap = pmmDefaultThinkCap
	}
	if p.BlendAlpha == 0 {
		p.BlendAlpha = pmmDefaultBlendAlpha
	}
	return p
}

// pmmPhase tags each decode token as belonging to the think or answer phase.
// Used to gate cap selection (and optionally Stream B) on the current phase.
type pmmPhase int

const (
	pmmPhaseAnswer pmmPhase = iota
	pmmPhaseThink
)

// pmmPhaseTracker classifies decode tokens as think- or answer-phase by
// scanning a rolling buffer for the model's open/close tags. Sibling
// implementation: thinkFilter in src/internal/apiserver/completions.go —
// same split-tag handling pattern, same transition-timing rules.
//
// Transition timing matches thinkFilter: the token CONTAINING </close> is
// classified as the LAST think token (inThink read after observe but the
// token that completed the close tag is reported as think). Tokens after
// the close-completing one are answer-phase.
type pmmPhaseTracker struct {
	open, close string
	buf         string
	inThink     bool
}

// newPMMPhaseTracker constructs a tracker. initialInThink mirrors
// thinkFilter's inThink: true if the model starts mid-think (typical for
// thinking models with enable_thinking=true).
func newPMMPhaseTracker(open, close string, initialInThink bool) pmmPhaseTracker {
	return pmmPhaseTracker{
		open:    open,
		close:   close,
		inThink: initialInThink,
	}
}

// observe appends tok to the rolling buffer, scans for open/close tags, and
// returns the phase the just-observed token belongs to. The token containing
// </close> is classified as think (last think token); subsequent tokens are
// answer. If open/close are empty, always returns pmmPhaseAnswer.
func (p *pmmPhaseTracker) observe(tok string) pmmPhase {
	if p.open == "" && p.close == "" {
		return pmmPhaseAnswer
	}
	enteredThink := p.inThink
	p.buf += tok
	holdClose := len(p.close) + 1
	holdOpen := len(p.open) + 1
	for {
		if p.inThink {
			if i := strings.Index(p.buf, p.close); i >= 0 {
				p.buf = p.buf[i+len(p.close):]
				p.inThink = false
				continue
			}
			if len(p.buf) > holdClose {
				p.buf = p.buf[len(p.buf)-holdClose:]
			}
			break
		}
		if p.open != "" {
			if i := strings.Index(p.buf, p.open); i >= 0 {
				p.buf = p.buf[i+len(p.open):]
				p.inThink = true
				continue
			}
		}
		safe := len(p.buf) - holdOpen
		if safe <= 0 {
			break
		}
		p.buf = p.buf[safe:]
		break
	}
	// A token that completed </close> entered with inThink=true and exited
	// with inThink=false. It is still classified as think (last think token);
	// answer-phase begins on the NEXT token. A token that opened <think>
	// (entered false, exited true) is classified as think — the open tag
	// itself sits in the think phase boundary.
	if enteredThink || p.inThink {
		return pmmPhaseThink
	}
	return pmmPhaseAnswer
}

// pmmStepRec is the per-token JSONL record.
type pmmStepRec struct {
	Type           string `json:"type"`
	TokenPos       int    `json:"token_pos"`
	Phase          string `json:"phase"` // "think" or "answer"
	MainlineTop1   int32  `json:"mainline_top1"`
	ShadowForkTop1 int32  `json:"shadow_fork_top1"`
	AgreeFork      bool   `json:"agree_fork"`
	Skipped        bool   `json:"skipped,omitempty"` // true when Stream B was skipped (think_disabled)
}

// pmmSessionRec is the JSONL header record emitted at the start of a PMM
// generation request, mirroring rlbSessionRec.
type pmmSessionRec struct {
	Type          string `json:"type"`
	TS            string `json:"ts"`
	Model         string `json:"model"`
	PromptTokens  int    `json:"prompt_tokens"`
	SlabFirst     int    `json:"slab_first"`
	SlabLast      int    `json:"slab_last"`
	Cap           int    `json:"cap"`
	ThinkCap      int     `json:"think_cap"`
	ThinkDisabled bool    `json:"think_disabled"`
	BlendAlpha    float64 `json:"blend_alpha"`
	FlashAttn     bool    `json:"flash_attn"`
}

// pmmDecodeState holds per-request PMM state. Constructed once after prefill;
// step() called per decode token.
type pmmDecodeState struct {
	model     *arch.GenericModel
	cache     *arch.GenericCache
	tokenizer *Tokenizer
	flashAttn bool
	params    PMMParams
	blocks    [][2]int

	dumpFile *os.File

	// phase tracking — drives think-vs-answer cap selection per token
	phase        pmmPhaseTracker
	currentPhase pmmPhase

	// running counters for INFO summary
	agreeForkCount     int
	totalTokens        int
	thinkTokens        int
	thinkShadowSkipped int
}

// newPMMDecodeState constructs the per-request shadow infrastructure. cache is
// expected to have been populated by a vanilla prefill (cache.SeqPos = nPrompt).
// thinkOpen/thinkClose come from the engine's arch tokens; initialInThink
// matches params.ThinkingEnabled (true means the prompt seeds the model
// inside an open think block, identical to thinkFilter's init contract).
func (e *Engine) newPMMDecodeState(
	cache *arch.GenericCache,
	flashAttn bool,
	params PMMParams,
	dumpFile *os.File,
	thinkOpen, thinkClose string,
	initialInThink bool,
) *pmmDecodeState {
	nLayers := e.model.Params.Ints[arch.ParamNLayers]
	blocks := e.model.RLBBlockRanges()
	if len(blocks) == 0 {
		blocks = [][2]int{{0, nLayers - 1}}
	}
	return &pmmDecodeState{
		model:     e.model,
		cache:     cache,
		tokenizer: e.tokenizer,
		flashAttn: flashAttn,
		params:    params,
		blocks:    blocks,
		dumpFile:  dumpFile,
		phase:     newPMMPhaseTracker(thinkOpen, thinkClose, initialInThink),
	}
}

// runShadowPass executes Stream B for the current decode token. Cache SSM state
// on entry is mainline's (left by Stream A); writeSSM=false means the shadow
// run leaves it untouched. writeKV is always false — Stream B never writes K/V.
// cache.SeqPos must be the post-Stream-A position so attention covers [0, t].
func (s *pmmDecodeState) runShadowPass(tokenID int32) ([]float32, error) {
	// SeqPos is post-Stream-A (= t+1). Shadow attends over [0, t]; the layer
	// graph derives nKV from cache.SeqPos and tokenPositions tells RoPE which
	// position the new token sits at. Pass tokenPositions = [t] (= SeqPos-1).
	t := s.cache.SeqPos - 1
	if t < 0 {
		return nil, fmt.Errorf("pmm shadow: cache.SeqPos=%d, expected >0 after Stream A", s.cache.SeqPos)
	}
	positions := []int32{int32(t)}

	// Token embedding via RLB helper — reuses the lazily-allocated scratch ctx.
	hidA, err := s.model.RLBForwardEmbed([]int32{tokenID})
	if err != nil {
		return nil, fmt.Errorf("pmm shadow embed: %w", err)
	}
	logitsBuf := make([]float32, s.model.Params.Ints[arch.ParamNVocab])

	// WriteKV and WriteSSM both false — Stream B never writes to cache.
	forwardLayer := func(il int, hidIn, hidOut, logitsOut []float32) error {
		return s.model.RLBForwardLayer(s.cache, il, hidIn, positions, s.flashAttn,
			false, false, hidOut, logitsOut)
	}

	// Set up the unified halt rule: single-pass non-slab blocks; dH_threshold
	// (capped at s.params.Cap) on slab blocks. Closure over the dH rule from
	// parseHaltRule keeps the alpha computation consistent.
	dHRule, _, _ := parseHaltRule(HaltRuleDHThreshold)
	slabFirst := s.params.SlabFirst
	slabLast := s.params.SlabLast
	cap := s.params.Cap
	if s.currentPhase == pmmPhaseThink {
		cap = s.params.ThinkCap
	}

	unifiedRule := func(history []rlbIterRec) (bool, float64) {
		if len(history) == 0 {
			return true, 1.0
		}
		bi := history[len(history)-1].BlockIdx
		inSlab := bi >= slabFirst && bi <= slabLast
		if !inSlab {
			return true, 1.0 // single pass for non-slab blocks
		}
		if len(history) >= cap {
			_, alpha := dHRule(history)
			return true, alpha
		}
		return dHRule(history)
	}

	// PMM does not blend SSM state per block — it's managed at the variant
	// level via host-side snapshot/restore. Pass no-op closures.
	saveState := func(int) {}
	blendState := func(int, float64) (float64, error) { return 0, nil }

	// Embed closure must return a fresh hidden state — reuse the pre-allocated hidA.
	forwardEmbed := func(_ []int32) ([]float32, error) { return hidA, nil }

	// projectLogits writes into the local logitsBuf and returns it; the caller
	// copies before returning to the engine.
	projectLogits := func(hid []float32, n int) ([]float32, error) {
		out, err := s.model.RLBProjectLogits(hid, n)
		if err != nil {
			return nil, err
		}
		copy(logitsBuf, out)
		return logitsBuf, nil
	}

	// dump — passed through to the session JSONL but with token_pos injected
	// at the shadow caller for clarity. Keep as no-op here since the
	// per-token pmmStepRec is the canonical record; the rlbIterRec stream is
	// noisy for PMM and not the primary diagnostic.
	dumpWrite := func(any) {}

	logits, _, err := perBlockForwardCore(
		[]int32{tokenID},
		s.blocks,
		s.model.Params.Ints[arch.ParamNVocab],
		s.tokenizer.TokenString,
		autoAlpha, // shadow uses the rule's auto-alpha (only consumed by no-op blendState)
		false,     // magnitudeNorm off — PMM is Mythos-shape on slab blocks only
		unifiedRule, cap,
		unifiedRule, cap,
		forwardEmbed,
		forwardLayer,
		projectLogits,
		saveState,
		blendState,
		dumpWrite,
	)
	if err != nil {
		return nil, err
	}
	// Copy out — perBlockForwardCore's logits alias logitsBuf which we own
	// here, but keep the result independent for cleanliness.
	out := make([]float32, len(logits))
	copy(out, logits)
	return out, nil
}

// step runs a single decode token through the PMM two-stream pipeline.
// Returns the logits used for sampling (Stream B output, or mainline when
// the token is in the think phase and ThinkDisabled is set). The caller is
// responsible for incrementing nothing — Stream A's ForwardCached already
// advances cache.SeqPos.
func (s *pmmDecodeState) step(tokenID int32) ([]float32, error) {
	s.totalTokens++

	// Classify the just-arrived token. The tokenizer string is used for the
	// rolling buffer scan — same input the streaming response uses.
	tokStr := s.tokenizer.TokenString(tokenID)
	s.currentPhase = s.phase.observe(tokStr)
	if s.currentPhase == pmmPhaseThink {
		s.thinkTokens++
	}

	// === Stream A: vanilla single-pass forward. Writes K/V + SSM. ===
	mainlineLogits, err := s.model.ForwardCached(s.cache, []int32{tokenID}, s.flashAttn, nil)
	if err != nil {
		return nil, fmt.Errorf("pmm stream A: %w", err)
	}
	// Stream A's logits buffer aliases an internal model buffer that the next
	// ForwardCached call (next token's Stream A) overwrites. Copy out for the
	// agreement comparison below.
	mainlineLogitsCopy := append([]float32(nil), mainlineLogits...)
	mainlineTop1 := Greedy(mainlineLogitsCopy)

	phaseStr := "answer"
	if s.currentPhase == pmmPhaseThink {
		phaseStr = "think"
	}

	// Think-phase short-circuit: skip Stream B entirely and sample from
	// mainline. Logged with Skipped=true; agreement is trivially true.
	if s.currentPhase == pmmPhaseThink && s.params.ThinkDisabled {
		s.thinkShadowSkipped++
		s.agreeForkCount++
		rec := pmmStepRec{
			Type:         "pmm_step",
			TokenPos:     s.totalTokens,
			Phase:        phaseStr,
			MainlineTop1: mainlineTop1,
			AgreeFork:    true,
			Skipped:      true,
		}
		pmmDumpWrite(s.dumpFile, rec)
		if s.totalTokens%pmmAgreeLogEvery == 0 {
			s.logAgreement()
		}
		return mainlineLogitsCopy, nil
	}

	// === Stream B: fork shadow. SSM state on entry is mainline's (Stream A
	// left it); writeSSM=false means it stays untouched after the pass.
	forkLogits, err := s.runShadowPass(tokenID)
	if err != nil {
		return nil, fmt.Errorf("pmm shadow: %w", err)
	}
	// Blend fork and mainline logits before sampling. α=1.0 keeps pure fork
	// (default). α<1.0 lerps toward mainline, dampening shadow overrides.
	α := float32(s.params.BlendAlpha)
	if α < 1.0 {
		for i := range forkLogits {
			forkLogits[i] = (1-α)*mainlineLogitsCopy[i] + α*forkLogits[i]
		}
	}

	forkTop1 := Greedy(forkLogits)
	rec := pmmStepRec{
		Type:           "pmm_step",
		TokenPos:       s.totalTokens,
		Phase:          phaseStr,
		MainlineTop1:   mainlineTop1,
		ShadowForkTop1: forkTop1,
		AgreeFork:      forkTop1 == mainlineTop1,
	}
	if rec.AgreeFork {
		s.agreeForkCount++
	}

	pmmDumpWrite(s.dumpFile, rec)

	if s.totalTokens%pmmAgreeLogEvery == 0 {
		s.logAgreement()
	}

	return forkLogits, nil
}

// logAgreement emits a one-line INFO summary of running shadow-vs-mainline agreement.
func (s *pmmDecodeState) logAgreement() {
	if s.totalTokens == 0 {
		return
	}
	log.Info("pmm: tokens=%d fork_agree=%d/%d (%.1f%%) think_tokens=%d think_skipped=%d",
		s.totalTokens, s.agreeForkCount, s.totalTokens,
		100*float64(s.agreeForkCount)/float64(s.totalTokens),
		s.thinkTokens, s.thinkShadowSkipped)
}

// pmmDumpOpen / pmmDumpWrite mirror their RLB counterparts but write under
// <diagDir>/pmm/. Using a dedicated subdir keeps the two streams' JSONL files
// from interleaving on disk.
func pmmDumpOpen(diagDir, modelPath string) *os.File {
	dir := filepath.Join(diagDir, "pmm")
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Warn("pmm dump: mkdir %s: %v", dir, err)
		return nil
	}
	stem := strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))
	ts := time.Now().Format("20060102-150405")
	path := filepath.Join(dir, fmt.Sprintf("%s.%s.jsonl", stem, ts))
	f, err := os.Create(path)
	if err != nil {
		log.Warn("pmm dump: create %s: %v", path, err)
		return nil
	}
	log.Info("pmm dump: %s", path)
	return f
}

func pmmDumpWrite(f *os.File, rec any) {
	if f == nil {
		return
	}
	b, err := json.Marshal(rec)
	if err != nil {
		log.Warn("pmm dump: marshal: %v", err)
		return
	}
	b = append(b, '\n')
	if _, err := f.Write(b); err != nil {
		log.Warn("pmm dump: write: %v", err)
	}
}

// generatePMM is the engine entry point for PMM two-stream decode. Prefill is
// vanilla ForwardCached (no prefill RLB / PMM in v1); decode runs the
// per-token two-stream loop.
func (e *Engine) generatePMM(
	promptIDs []int32, maxTokens int, stopSet map[int32]bool,
	params GenerateParams,
	onToken func(string) bool, metrics *InferenceMetrics,
) error {
	cache, err := e.model.NewCache(e.maxSeqLen)
	if err != nil {
		return fmt.Errorf("creating cache: %w", err)
	}
	defer cache.Free()

	flashAttn := *params.FlashAttention
	pmmParams := params.PMM.applyDefaults()

	// Validate slab range against block count.
	nLayers := e.model.Params.Ints[arch.ParamNLayers]
	blocks := e.model.RLBBlockRanges()
	if len(blocks) == 0 {
		blocks = [][2]int{{0, nLayers - 1}}
	}
	if pmmParams.SlabFirst < 0 || pmmParams.SlabLast >= len(blocks) || pmmParams.SlabFirst > pmmParams.SlabLast {
		log.Warn("pmm: slab [%d,%d] out of range for %d blocks; clamping",
			pmmParams.SlabFirst, pmmParams.SlabLast, len(blocks))
		if pmmParams.SlabFirst < 0 {
			pmmParams.SlabFirst = 0
		}
		if pmmParams.SlabLast >= len(blocks) {
			pmmParams.SlabLast = len(blocks) - 1
		}
		if pmmParams.SlabFirst > pmmParams.SlabLast {
			pmmParams.SlabFirst = pmmParams.SlabLast
		}
	}

	dumpFile := pmmDumpOpen(e.diagDir, e.model.ModelPath)
	defer func() {
		if dumpFile != nil {
			dumpFile.Close()
		}
	}()

	pmmDumpWrite(dumpFile, pmmSessionRec{
		Type:          "session",
		TS:            time.Now().UTC().Format(time.RFC3339),
		Model:         strings.TrimSuffix(filepath.Base(e.model.ModelPath), filepath.Ext(e.model.ModelPath)),
		PromptTokens:  len(promptIDs),
		SlabFirst:     pmmParams.SlabFirst,
		SlabLast:      pmmParams.SlabLast,
		Cap:           pmmParams.Cap,
		ThinkCap:      pmmParams.ThinkCap,
		ThinkDisabled: pmmParams.ThinkDisabled,
		BlendAlpha:    pmmParams.BlendAlpha,
		FlashAttn:     flashAttn,
	})

	log.Info("pmm decode: slab=[%d,%d] cap=%d think_cap=%d think_disabled=%v blend_alpha=%.2f (blocks=%d)",
		pmmParams.SlabFirst, pmmParams.SlabLast, pmmParams.Cap,
		pmmParams.ThinkCap, pmmParams.ThinkDisabled, pmmParams.BlendAlpha, len(blocks))

	// === PREFILL — plain ForwardCached. ===
	// PMM + vision is unsupported in v1 (Engine.Generate rejects the
	// combination upstream); pass nil for splice inputs unconditionally.
	prefillStart := time.Now()
	logits, err := e.model.ForwardCached(cache, promptIDs, flashAttn, nil)
	if err != nil {
		return fmt.Errorf("pmm prefill: %w", err)
	}
	metrics.PrefillDuration = time.Since(prefillStart)

	dec := e.newPMMDecodeState(cache, flashAttn, pmmParams, dumpFile,
		e.ThinkOpen(), e.ThinkClose(), params.ThinkingEnabled)

	// === DECODE — two-stream per token. ===
	decodeStart := time.Now()
	hitStop := false
	for range maxTokens {
		nextID, err := e.sample(logits, params)
		if err != nil {
			return err
		}
		if stopSet[nextID] {
			hitStop = true
			break
		}
		if params.LogProbs {
			metrics.TokenLogProbs = append(metrics.TokenLogProbs,
				ComputeTopLogProbs(logits, nextID, params.TopLogProbs, e.tokenizer.TokenString))
		}
		if !onToken(e.tokenizer.TokenString(nextID)) {
			hitStop = true
			break
		}
		metrics.CompletionTokens++

		logits, err = dec.step(nextID)
		if err != nil {
			return fmt.Errorf("pmm decode: %w", err)
		}
	}
	metrics.DecodeDuration = time.Since(decodeStart)

	// Final agreement summary on completion.
	dec.logAgreement()

	if hitStop {
		metrics.FinishReason = FinishReasonStop
	} else {
		metrics.FinishReason = FinishReasonLength
	}
	return nil
}
