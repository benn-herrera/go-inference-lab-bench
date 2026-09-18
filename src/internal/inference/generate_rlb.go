package inference

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unsafe"

	ggml "inference-lab-bench/internal/ggml"
	"inference-lab-bench/internal/inference/arch"
	log "inference-lab-bench/internal/log"
)

// RLB experiment diagnostic dump — one JSONL file per RLB generation, written to
// <diagDir>/rlb/<model_stem>.<timestamp>.jsonl. Three record types in first-field
// order: "session" (header), "iter" (per-iteration state snapshot), "summary"
// (totals). The top_k list captures the top-K logprobs each iteration so the
// distribution shape (not just argmax) can be analyzed post-hoc. The iter
// record also includes Tier-1 logit shape stats and Tier-2 hidden/SSM
// convergence norms to give a downstream halt head more signal than raw
// top-1 logprob. Intentionally kept off the metrics contract — this is
// experiment scaffolding.
const rlbDumpTopK = 50

// defaultRLBCeiling is the max iteration count used by adaptive halt rules —
// the upper bound if the predicate never triggers. Single source of truth for
// the RLB loop bound: fixed_N rules encode N in their name; adaptive rules
// fall back here.
const defaultRLBCeiling = 7

// autoAlpha is the user-facing sentinel for "use the halt rule's auto-computed
// alpha per iter". When RLBParams.Alpha equals this value, the effective alpha
// for each blend is taken from the haltRule's second return value (a rule-
// specific confidence-derived blend factor). Otherwise the user-supplied alpha
// is used verbatim. 0.0 is a safe sentinel because a genuine user-supplied
// alpha of 0 would mean "never advance SSM state" — a degenerate no-op — so
// overloading it for "auto" does not lose useful semantics.
const autoAlpha = -1.0

// rlbPrefillMaxIters is the hard ceiling for parked prefill RLB.
// Conservative: 3 iters was the sweet spot in the comparative sweep.
const rlbPrefillMaxIters = 3

// Halt rule names — single source of truth for switch dispatch in parseHaltRule,
// for test cases, and for any in-Go reference to a rule by name. Bash sweep
// harnesses (test_rlb.sh, test_inference.sh) use string literals because they
// can't import Go consts; keeping the shell names in sync with these constants
// is a manual discipline. Defensive coding: a typo'd literal becomes a compile
// error inside Go, instead of silently routing through the unknown-rule
// fallback at runtime.
const (
	HaltRuleFixed1        = "fixed_1"
	HaltRuleLogprobPeak   = "logprob_peak"
	HaltRuleEntropyMin    = "entropy_min"
	HaltRuleGapPeak       = "gap_peak"
	HaltRuleArgmaxStable2 = "argmax_stable_2"
	HaltRuleArgmaxStable3 = "argmax_stable_3"
	HaltRuleDHThreshold   = "dH_threshold"
	HaltRuleDefault       = HaltRuleFixed1
)

// dH_threshold rule constants. Kept at package scope so tests can reference
// them without duplicating magic numbers. threshold is the normalized ΔH
// ceiling below which the block is considered converged (1% of the block
// output norm, calibrated from the auto-alpha sanity sweep where per-block
// dH/outNorm ratios dropped an order of magnitude between iters 1 and 3).
// maxIters is the hard upper bound if the predicate never triggers — a
// safety net, not an operating point. 10 is a headroom choice (binary-
// search doubling from the initial 5, where the sanity sweep showed block 0
// on the sine-degrees prompt would have converged around iter 7-8 on its
// observed decay slope); cheap enough to keep, loose enough that "the cap
// fired" signals a genuinely stuck block rather than a tight budget.
const (
	dHThresholdRatio    = 0.01
	dHThresholdMaxIters = 10
)

// RLBParams groups RLB-specific generation parameters. Mirrors DiffusionParams
// — ignored for non-RLB generation; only non-nil when RLB is enabled.
type RLBParams struct {
	// Prefill: when true, run per-block iteration during prefill with
	// hardcoded defaults (auto-alpha, dH_threshold, max rlbPrefillMaxIters iters).
	Prefill bool
	// Decode: when true, run per-block iteration during decode with
	// the Alpha/HaltRule/TerminalHaltRule params below.
	Decode bool

	// MagnitudeNorm enables residual stream magnitude normalization at block
	// boundaries. When true, after the final iteration of each block, the
	// output is renormalized to the L2 magnitude observed on the first pass.
	// Iteration itself uses Mythos-shape (restore original block input, SSM
	// state evolves). This prevents magnitude compound across decode tokens
	// that causes K/V distribution shift. SSM state save/blend still applies.
	MagnitudeNorm bool

	// Alpha is the SSM-state blend factor applied between iterations within a
	// block. 0.0 == autoAlpha sentinel — the halt rule supplies a confidence-
	// derived alpha per iter. Any other value is used verbatim as a fixed blend
	// factor in (0, 1]. A fixed alpha of 1.0 means "no blend" (full state
	// replacement); values below 1 preserve saved state in proportion. See
	// autoAlpha / parseHaltRule above. Controls decode RLB.
	Alpha float64
	// HaltRule is the halt rule name ("" = default fixed_3). See parseHaltRule
	// for the full menu — iter count is encoded in the rule name. Controls decode RLB.
	HaltRule string
	// TerminalHaltRule is the halt rule applied to the terminal block (the
	// block whose output feeds lm_head). Empty = reuse HaltRule for all
	// blocks. Motivation: Tier 1 logit-shape rules (logprob_peak,
	// entropy_min, gap_peak, argmax_stable_*) only have meaningful signal
	// at the projection point, since upstream blocks' top-1 logprobs are
	// "what lm_head would say given a partial state" — not a convergence
	// signal. This field lets a run use a Tier 2 rule (e.g. dH_threshold)
	// on upstream blocks and a Tier 1 rule on the terminal block, or
	// vice versa. The terminal rule's own maxIters (from parseHaltRule)
	// applies only to the terminal block; upstream blocks use the main
	// rule's maxIters. Controls decode RLB.
	TerminalHaltRule string
}

// ssmBuf holds a single SSM state tensor save buffer for one layer+name pair.
// Promoted to file scope so both perBlockRLBPrefill and newDecodeRLBState can use it.
type ssmBuf struct {
	layer int
	name  string
	data  []byte
}

type rlbSessionRec struct {
	Type             string  `json:"type"`
	TS               string  `json:"ts"`
	Model            string  `json:"model"`
	PromptTokens     int     `json:"prompt_tokens"`
	Iterations       int     `json:"iterations"`
	Alpha            float64 `json:"alpha"`
	FlashAttn        bool    `json:"flash_attn"`
	MagnitudeNorm    bool    `json:"magnitude_norm,omitempty"` // true = output-recycling, magnitude norm, residual blend (no SSM save/blend)
	HaltRule         string  `json:"halt_rule,omitempty"`
	TerminalHaltRule string  `json:"terminal_halt_rule,omitempty"` // set only when user-specified (non-empty RLBParams.TerminalHaltRule)
	TerminalIters    int     `json:"terminal_iters,omitempty"`     // maxIters cap of terminal rule when different from main
	Granularity      string  `json:"granularity,omitempty"`        // "per_block" for new runs
	Phase            string  `json:"phase,omitempty"`              // "prefill" or "decode"
}

type rlbTopKEntry struct {
	ID      int32   `json:"id"`
	Token   string  `json:"token"`
	LogProb float64 `json:"logprob"`
}

type rlbIterRec struct {
	Type        string         `json:"type"`
	Iter        int            `json:"iter"`         // block-local iter index (0 = first pass through the block)
	BlockIdx    int            `json:"block_idx"`    // which block (0..nBlocks-1)
	BlockLayers [2]int         `json:"block_layers"` // [firstLayer, lastLayer] inclusive
	GlobalStep  int            `json:"global_step"`  // monotonic counter across all block-iters in this prefill
	TokenPos    int            `json:"token_pos,omitempty"` // decode token index (0 for prefill records)
	Top1ID      int32          `json:"top1_id"`
	Top1Token   string         `json:"top1_token"`
	Top1LogProb float64        `json:"top1_logprob"`
	ElapsedMs   float64        `json:"elapsed_ms"`
	TopK        []rlbTopKEntry `json:"top_k"`

	// Tier 1 — logit distribution shape. All computed from the same last-layer
	// logits that feed top-1 / top-K. Cheap to add (one pass over nVocab).
	TopKEntropy float64 `json:"top_k_entropy"` // nats over softmax-renormalized top-K
	Rank12Gap   float64 `json:"rank12_gap"`    // top_k[0].logprob − top_k[1].logprob
	Rank1KGap   float64 `json:"rank1k_gap"`    // top_k[0].logprob − top_k[K-1].logprob
	LogitMax    float64 `json:"logit_max"`     // raw max over full vocab logits (not logprob)
	LogitMin    float64 `json:"logit_min"`     // raw min over full vocab logits
	LogitMean   float64 `json:"logit_mean"`    // raw mean over full vocab
	LogitVar    float64 `json:"logit_var"`     // raw variance over full vocab

	// Tier 2 — hidden-state and SSM convergence norms. These are the richer
	// "is recurrence actually converging" signals. BlockInNorm and BlockOutNorm
	// are per-iter snapshots of residual stream magnitude; DeltaHiddenNorm is
	// the per-iter change in block output (0 on iter 0 of each block);
	// SSMStateDeltaPre is the L2 of the SSM state change from the save point
	// to the post-forward state for the blend that set up *this* iter's
	// forward pass (0 on iter 0 of each block, since there's no prior blend).
	BlockInNorm      float64 `json:"block_in_norm"`
	BlockOutNorm     float64 `json:"block_out_norm"`
	DeltaHiddenNorm  float64 `json:"delta_hidden_norm"`
	SSMStateDeltaPre float64 `json:"ssm_state_delta_pre"`
}

type rlbSummaryRec struct {
	Type                  string  `json:"type"`
	Iterations            int     `json:"iterations"`
	CompletedIterations   int     `json:"completed_iterations"`
	ThinkingMsTotal       float64 `json:"thinking_ms_total"`
	CompletedItersByBlock []int   `json:"completed_iters_by_block,omitempty"` // len == nBlocks
}

// rlbDumpOpen creates <diagDir>/rlb/<model_stem>.<ts>.jsonl. Returns nil if the
// directory cannot be created or the file cannot be opened — dump failure never
// fails the request; all write sites tolerate a nil file.
func rlbDumpOpen(diagDir, modelPath string) *os.File {
	dir := filepath.Join(diagDir, "rlb")
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Warn("rlb dump: mkdir %s: %v", dir, err)
		return nil
	}
	stem := strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))
	ts := time.Now().Format("20060102-150405")
	path := filepath.Join(dir, fmt.Sprintf("%s.%s.jsonl", stem, ts))
	f, err := os.Create(path)
	if err != nil {
		log.Warn("rlb dump: create %s: %v", path, err)
		return nil
	}
	log.Info("rlb dump: %s", path)
	return f
}

// rlbDumpWrite marshals rec as JSON and appends it as a single newline-terminated
// line. Silently skips when f is nil.
func rlbDumpWrite(f *os.File, rec any) {
	if f == nil {
		return
	}
	b, err := json.Marshal(rec)
	if err != nil {
		log.Warn("rlb dump: marshal: %v", err)
		return
	}
	b = append(b, '\n')
	if _, err := f.Write(b); err != nil {
		log.Warn("rlb dump: write: %v", err)
	}
}

// logitStats computes full-vocab max/min/mean/var over a raw logits slice.
// One pass; variance uses the two-pass Welford-equivalent on already-computed
// mean for numerical simplicity at this scale.
func logitStats(logits []float32) (lmax, lmin, mean, variance float64) {
	if len(logits) == 0 {
		return 0, 0, 0, 0
	}
	lmax = float64(logits[0])
	lmin = float64(logits[0])
	sum := 0.0
	for _, x := range logits {
		xf := float64(x)
		if xf > lmax {
			lmax = xf
		}
		if xf < lmin {
			lmin = xf
		}
		sum += xf
	}
	mean = sum / float64(len(logits))
	sq := 0.0
	for _, x := range logits {
		d := float64(x) - mean
		sq += d * d
	}
	variance = sq / float64(len(logits))
	return
}

// l2Norm returns sqrt(sum(x[i]^2)) over a float32 slice, computed in float64.
func l2Norm(v []float32) float64 {
	s := 0.0
	for _, x := range v {
		xf := float64(x)
		s += xf * xf
	}
	return math.Sqrt(s)
}

// scaleToMag rescales v in-place so that its L2 norm equals targetMag.
// If the current norm is zero (degenerate), v is left unchanged.
func scaleToMag(v []float32, targetMag float64) {
	cur := l2Norm(v)
	if cur == 0 {
		return
	}
	ratio := float32(targetMag / cur)
	for i := range v {
		v[i] *= ratio
	}
}

// l2Diff returns sqrt(sum((a[i]-b[i])^2)). Caller must ensure equal length.
func l2Diff(a, b []float32) float64 {
	s := 0.0
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		d := float64(a[i]) - float64(b[i])
		s += d * d
	}
	return math.Sqrt(s)
}

// buildRLBIterRec computes the top-k logprob record plus Tier-1 logit shape
// stats and Tier-2 hidden-state norms for one iteration. Shared between the
// human-readable log line and the JSONL dump so the underlying
// ComputeTopLogProbs call happens exactly once per iteration.
//
// blockIn, blockOut: residual-stream snapshots at this iter's block
// input/output. Used for BlockInNorm, BlockOutNorm.
// prevBlockOut: previous iter's block output (within this same block), or nil
// on iter 0. Used for DeltaHiddenNorm.
// ssmStateDeltaPre: L2 norm of the SSM state change from pre-save to post-
// forward captured during the blend that set up this iter's forward (0 on
// iter 0).
func buildRLBIterRec(
	iter int,
	logits []float32,
	elapsed time.Duration,
	tokenString func(int32) string,
	blockIn, blockOut, prevBlockOut []float32,
	ssmStateDeltaPre float64,
) rlbIterRec {
	topID := Greedy(logits)
	tlp := ComputeTopLogProbs(logits, topID, rlbDumpTopK, tokenString)
	entries := make([]rlbTopKEntry, len(tlp.TopProbs))
	for i, p := range tlp.TopProbs {
		entries[i] = rlbTopKEntry{ID: p.ID, Token: p.Token, LogProb: p.LogProb}
	}
	lmax, lmin, lmean, lvar := logitStats(logits)
	entropy := entropyOfTopK(entries)
	r12 := rank2GapTopK(entries)
	r1k := 0.0
	if len(entries) >= 2 {
		r1k = entries[0].LogProb - entries[len(entries)-1].LogProb
	}
	inNorm := l2Norm(blockIn)
	outNorm := l2Norm(blockOut)
	delta := 0.0
	if prevBlockOut != nil {
		delta = l2Diff(blockOut, prevBlockOut)
	}
	return rlbIterRec{
		Type:             "iter",
		Iter:             iter,
		Top1ID:           topID,
		Top1Token:        tokenString(topID),
		Top1LogProb:      tlp.LogProb,
		ElapsedMs:        float64(elapsed.Microseconds()) / 1000,
		TopK:             entries,
		TopKEntropy:      entropy,
		Rank12Gap:        r12,
		Rank1KGap:        r1k,
		LogitMax:         lmax,
		LogitMin:         lmin,
		LogitMean:        lmean,
		LogitVar:         lvar,
		BlockInNorm:      inNorm,
		BlockOutNorm:     outNorm,
		DeltaHiddenNorm:  delta,
		SSMStateDeltaPre: ssmStateDeltaPre,
	}
}

// haltRule is a predicate over RLB iteration history. It returns (halt,
// nextAlpha): halt says whether the thinking loop should stop before running
// the next iteration; nextAlpha is the rule's auto-computed alpha for the
// blend that sets up the next iter's forward. history always contains at
// least iter 0 (prefill) when the rule is evaluated, and nextAlpha is only
// consulted when the caller is in auto-alpha mode (params.RLB.Alpha ==
// autoAlpha). When halt is true, nextAlpha is ignored.
//
// Direction of nextAlpha: HIGH confidence → LOW alpha (preserve state, allow
// honing); LOW confidence → HIGH alpha (allow larger state displacement to
// disambiguate). This matches the empirical finding from the first per-block
// sweep that aggressive state replacement hurts stability.
type haltRule func(history []rlbIterRec) (halt bool, nextAlpha float64)

// entropyOfTopK computes entropy in nats over the softmax-renormalized top-k.
// Matches halting_audit.py topk_entropy(): subtract max logprob, exp, renormalize
// within top-k, compute -sum(p*log(p)). Returns 0 for empty or one-entry slices.
func entropyOfTopK(topk []rlbTopKEntry) float64 {
	if len(topk) == 0 {
		return 0
	}
	maxLP := topk[0].LogProb
	probs := make([]float64, len(topk))
	sum := 0.0
	for i, e := range topk {
		p := math.Exp(e.LogProb - maxLP)
		probs[i] = p
		sum += p
	}
	h := 0.0
	for _, p := range probs {
		p /= sum
		if p > 0 {
			h -= p * math.Log(p)
		}
	}
	return h
}

// rank2GapTopK returns TopK[0].LogProb - TopK[1].LogProb, or 0 if fewer than 2
// entries are present.
func rank2GapTopK(topk []rlbTopKEntry) float64 {
	if len(topk) < 2 {
		return 0
	}
	return topk[0].LogProb - topk[1].LogProb
}

// tailMatches returns the number of consecutive trailing iters in history
// that share history[len-1].Top1ID, capped at maxN. Returns 0 for empty
// history, ≥ 1 otherwise (the current iter always agrees with itself).
func tailMatches(history []rlbIterRec, maxN int) int {
	if len(history) == 0 {
		return 0
	}
	last := history[len(history)-1].Top1ID
	n := 0
	for i := len(history) - 1; i >= 0 && n < maxN; i-- {
		if history[i].Top1ID != last {
			break
		}
		n++
	}
	return n
}

// clamp01 constrains x to [0, 1]. Used to sanitize rule-computed alphas that
// may drift slightly outside the interval due to floating-point arithmetic on
// top-K logprob approximations.
func clamp01(x float64) float64 {
	if math.IsNaN(x) {
		return 0
	}
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// parseHaltRule resolves a requested halt rule name to a rule predicate, a
// human-readable label, and the max iteration count for the RLB thinking loop.
// Single source of truth for both the halt decision and the loop bound.
//
// Fixed rules (fixed_N) encode N in their name and return N as maxIters.
// Adaptive rules (logprob_peak, entropy_min, gap_peak, argmax_stable_*) break
// out early via the predicate and return defaultRLBCeiling as the upper bound
// if they never trigger. Empty or unknown ruleName falls back to fixed_3.
//
// Each rule's closure also produces an auto-alpha ∈ [0, 1]. The auto-alpha is
// consumed when params.RLB.Alpha == autoAlpha; otherwise it is ignored. The
// convention is "HIGH confidence → LOW alpha" (preserve state, allow honing)
// and "LOW confidence → HIGH alpha" (allow larger state displacement).
//
// fixed_N auto-alpha schedule: α = 1/len(history). At rule-call time
// len(history) == iter+1, so the sequence for fixed_3 is {1.0, 0.5, 1/3} for
// the blends that set up iters 1, 2, 3. This is the classical running-mean
// recurrence: after k iters the blended SSM state is the arithmetic mean of
// every post-forward state seen so far. The initial (zero / prior-block
// handoff) state is discarded by α=1.0 on the first blend, and each
// subsequent forward contributes an equal share to the evolving state.
// Constant α=1/N gives geometrically-decaying backward weights that
// overweight the initial state — strictly worse — and is reachable with a
// fixed RLB_ALPHA anyway, so that mode was dropped.
//
// Caveat: fixed_N substitutes iter count as a confidence proxy since the
// rule has no other signal to draw on. This is not a good auto-alpha
// (empirical confidence does not always grow with iter count), merely a
// meaningful one; adaptive rules use measured confidence instead.
//
// dH_threshold rule: first rule to consume a Tier 2 signal directly.
// Halts when the normalized hidden-state delta (||block_out_k −
// block_out_{k-1}|| / ||block_out_k||) falls below dHThresholdRatio —
// i.e. the block's residual-stream contribution is no longer changing
// meaningfully between iters. Uses BlockOutNorm for the denominator
// because pre-norm transformers accumulate residuals additively without
// renormalizing until the global output head, so outNorm grows monotonically
// through the stack and dH/outNorm is the scale-invariant convergence signal.
// Iter 0 is a forced fallthrough (dH is 0 by definition when there is no
// prior block output), so the rule can never halt on the prefill iter.
// Auto-alpha = clamp01(normDH / (2*threshold)): at the halt boundary the
// alpha is 0.5 (half blend), with linear scaling above and clamping below;
// that gives the blend enough authority to keep displacing state when dH is
// still large, and winds it down as the block approaches convergence.
func parseHaltRule(ruleName string) (haltRule, string, int) {
	switch ruleName {
	case "":
		// No rule specified — use default. Label suffix makes it obvious
		// in logs that the caller didn't pick explicitly.
		rule, _, iters := parseHaltRule(HaltRuleDefault)
		return rule, HaltRuleDefault + " (default)", iters

	case HaltRuleLogprobPeak:
		// Halt when the last iter's top-1 logprob is strictly less than the
		// previous iter's — i.e. the peak was at the prior iter.
		// Auto-alpha = 1 − exp(top1_logprob). At certainty (logprob ≈ 0),
		// alpha ≈ 0; at high uncertainty, alpha → 1.
		return func(history []rlbIterRec) (bool, float64) {
			cur := history[len(history)-1]
			nextA := clamp01(1.0 - math.Exp(cur.Top1LogProb))
			if len(history) < 2 {
				return false, nextA
			}
			prev := history[len(history)-2]
			return cur.Top1LogProb < prev.Top1LogProb, nextA
		}, ruleName, defaultRLBCeiling

	case HaltRuleEntropyMin:
		// Halt when the last iter's top-k entropy is strictly greater than the
		// previous iter's — confidence peaked at the prior iter.
		// Auto-alpha = entropy / log(K). At zero entropy (full confidence on
		// top-1), alpha = 0; at uniform entropy, alpha = 1.
		return func(history []rlbIterRec) (bool, float64) {
			cur := history[len(history)-1]
			curE := entropyOfTopK(cur.TopK)
			nextA := 0.0
			if k := len(cur.TopK); k > 1 {
				nextA = clamp01(curE / math.Log(float64(k)))
			}
			if len(history) < 2 {
				return false, nextA
			}
			prevE := entropyOfTopK(history[len(history)-2].TopK)
			return curE > prevE, nextA
		}, ruleName, defaultRLBCeiling

	case HaltRuleGapPeak:
		// Halt when the rank1-rank2 logprob gap is strictly less than the
		// previous iter's gap — the decisiveness peak was at the prior iter.
		// Auto-alpha = exp(−gap). At gap = 0 (tie), alpha = 1; at large gaps,
		// alpha → 0.
		return func(history []rlbIterRec) (bool, float64) {
			cur := history[len(history)-1]
			curG := rank2GapTopK(cur.TopK)
			nextA := clamp01(math.Exp(-curG))
			if len(history) < 2 {
				return false, nextA
			}
			prev := rank2GapTopK(history[len(history)-2].TopK)
			return curG < prev, nextA
		}, ruleName, defaultRLBCeiling

	case HaltRuleArgmaxStable2:
		// Halt when the last 2 iters share top-1.
		// Auto-alpha = (2 − tailMatches) / 2. 1 match → 0.5; 2 matches → 0
		// (and we halt anyway).
		return func(history []rlbIterRec) (bool, float64) {
			m := tailMatches(history, 2)
			nextA := clamp01(float64(2-m) / 2.0)
			if len(history) < 2 {
				return false, nextA
			}
			return history[len(history)-1].Top1ID == history[len(history)-2].Top1ID, nextA
		}, ruleName, defaultRLBCeiling

	case HaltRuleArgmaxStable3:
		// Halt when the last 3 iters share top-1.
		// Auto-alpha = (3 − tailMatches) / 3. 1 match → 0.667; 2 → 0.333;
		// 3 → 0 (and we halt anyway).
		return func(history []rlbIterRec) (bool, float64) {
			m := tailMatches(history, 3)
			nextA := clamp01(float64(3-m) / 3.0)
			if len(history) < 3 {
				return false, nextA
			}
			last := history[len(history)-1].Top1ID
			return last == history[len(history)-2].Top1ID && last == history[len(history)-3].Top1ID, nextA
		}, ruleName, defaultRLBCeiling

	case HaltRuleDHThreshold:
		// Tier 2 convergence rule — halts when normalized block output delta
		// falls below dHThresholdRatio. See parseHaltRule doc comment above
		// for the full rationale.
		return func(history []rlbIterRec) (bool, float64) {
			last := history[len(history)-1]
			// Iter 0 has no prior block output, so DeltaHiddenNorm is 0 by
			// definition. Fall through with a full-replacement alpha so the
			// blend that sets up iter 1 doesn't collapse to zero on an
			// unmeasurable signal.
			if last.Iter == 0 {
				return false, 1.0
			}
			// BlockOutNorm is L2 of the residual stream at block output and
			// is strictly positive in any non-degenerate run; guard the
			// division defensively to avoid NaN propagation if it ever isn't.
			if last.BlockOutNorm == 0 {
				return false, 1.0
			}
			normDH := last.DeltaHiddenNorm / last.BlockOutNorm
			nextA := clamp01(normDH / (2 * dHThresholdRatio))
			if normDH < dHThresholdRatio {
				return true, nextA
			}
			return false, nextA
		}, ruleName, dHThresholdMaxIters

	default:
		// Generic fixed_N: parse "fixed_<int>" for any non-negative N.
		if strings.HasPrefix(ruleName, "fixed_") {
			n, err := strconv.Atoi(ruleName[len("fixed_"):])
			if err == nil && n >= 0 {
				return func(history []rlbIterRec) (bool, float64) {
					return len(history) > n, 1.0 / float64(len(history))
				}, ruleName, n
			}
		}
		log.Warn("rlb: unknown halt rule %q — falling back to %s", ruleName, HaltRuleDefault)
		rule, _, iters := parseHaltRule(HaltRuleDefault)
		return rule, HaltRuleDefault + " (default)", iters
	}
}

// perBlockStats holds aggregate metrics from a perBlockForwardCore run.
type perBlockStats struct {
	thinkTotalMs          float64
	completedItersByBlock []int // len == len(blocks); entry i is how many iters block i ran
	totalItersRun         int   // sum(completedItersByBlock)
}

// buildSSMBufsByBlock allocates per-block SSM state save buffers for all
// (conv_state, ssm_state) tensors in the given block ranges. Used by both
// perBlockRLBPrefill and newDecodeRLBState.
func buildSSMBufsByBlock(cache *arch.GenericCache, blocks [][2]int) [][]ssmBuf {
	result := make([][]ssmBuf, len(blocks))
	for bi, blockRange := range blocks {
		firstL, lastL := blockRange[0], blockRange[1]
		for li := firstL; li <= lastL; li++ {
			for _, name := range []string{arch.CacheConvState, arch.CacheSSMState} {
				t := cache.Layers[li].Tensors[name]
				if t.IsNil() {
					continue
				}
				result[bi] = append(result[bi], ssmBuf{
					layer: li, name: name, data: make([]byte, t.Nbytes()),
				})
			}
		}
	}
	return result
}

// perBlockForwardCore is the pure loop body, factored out for unit testing.
// Used for both prefill and decode — callers supply forward/embed/project
// closures so tests can substitute fakes without loading a model.
//
// forwardEmbed(tokens) returns the initial hidden state.
// forwardLayer(il, hidIn, hidOut, logitsOut) runs one layer; logitsOut is nil
// for non-last layers in a block.
// projectLogits(hid, nTokens) produces final logits after all blocks.
// saveState(blockIdx) saves SSM state for the block before a forward pass.
// blendState(blockIdx, alpha) blends post-forward SSM state with the saved
// pre-forward state. It returns the L2 norm of the pre-blend state delta
// (||post_forward - pre_save||), which is always computed regardless of alpha.
// The returned delta is reported as SSMStateDeltaPre on the next iter's
// record, giving the halt head visibility into how much the prior forward
// pass moved the SSM state.
// dumpWrite(rec) emits one record to the diagnostic dump.
//
// Two iteration modes:
//
// Mythos-shape (magnitudeNorm=false): each iter restores the original block
// input; only SSM state evolves across iters via saveState/blendState.
//
// Output-recycling (magnitudeNorm=true): layers are sealed black boxes — no
// SSM save/blend. Each iter feeds the previous iter's block output (magnitude-
// normalized to magOut) back as input. Block outputs are alpha-blended into an
// accumulator across iterations so the residual handed to the next block
// carries accumulated thinking, not just the last iter's raw output.
// saveState/blendState are NOT called in this mode.
//
// alpha semantics: if alpha == autoAlpha, the effective blend/accumulation
// factor per iter is taken from the halt rule's second return value
// (confidence-derived). Any other value is used verbatim.
//
// Halt rule routing: rule/maxIters applies to every block except the terminal
// block (the one whose output feeds lm_head); terminalRule/terminalMaxIters
// applies to the terminal block. The caller is responsible for defaulting
// terminalRule to rule and terminalMaxIters to maxIters when the user has
// not specified a separate terminal rule; the core makes no assumption and
// always consults whichever pair the caller passed in.
func perBlockForwardCore(
	promptIDs []int32,
	blocks [][2]int,
	nVocab int,
	tokenString func(int32) string,
	alpha float64,
	magnitudeNorm bool,
	rule haltRule, maxIters int,
	terminalRule haltRule, terminalMaxIters int,
	forwardEmbed func([]int32) ([]float32, error),
	forwardLayer func(il int, hidIn, hidOut, logitsOut []float32) error,
	projectLogits func([]float32, int) ([]float32, error),
	saveState func(blockIdx int),
	blendState func(blockIdx int, alpha float64) (float64, error),
	dumpWrite func(rec any),
) (finalLogits []float32, stats perBlockStats, err error) {
	nTokens := len(promptIDs)

	hidA, err := forwardEmbed(promptIDs)
	if err != nil {
		return nil, stats, fmt.Errorf("rlb embed: %w", err)
	}
	hidB := make([]float32, len(hidA))

	// Stable copy of the current block's input, snapshotted before the inner
	// iter loop. In Mythos-shape mode, restored at iter > 0 so each block-iter
	// re-reads the same input while SSM state evolves. In output-recycling mode,
	// not restored (the previous iter's output is recycled as input), but still
	// used as the reference for buildRLBIterRec's BlockInNorm.
	blockInput := make([]float32, len(hidA))

	// Accumulator for alpha-blended block output across iterations. Only used
	// in output-recycling mode (magnitudeNorm=true). Initialized to iter 0's
	// block output; subsequent iters blend into it. This is the value that
	// gets handed to the next block.
	accumulated := make([]float32, len(hidA))

	// Stable copy of the prior iter's block output, reused across iters within
	// a block for DeltaHiddenNorm computation. Reset (via prevBlockOutPtr=nil)
	// at the start of each block. Allocated once outside the block loop.
	prevBlockOutBuf := make([]float32, len(hidA))

	logitsBuf := make([]float32, nVocab)

	stats.completedItersByBlock = make([]int, len(blocks))

	globalStep := 0

	for bi, blockRange := range blocks {
		firstL, lastL := blockRange[0], blockRange[1]
		history := []rlbIterRec{}

		// Per-block halt rule selection. Every block except the last uses
		// the main rule/maxIters; the terminal block uses the terminal
		// pair. Callers are responsible for having already resolved the
		// "no separate terminal rule was specified" case by passing
		// terminalRule == rule and terminalMaxIters == maxIters — the core
		// does not inspect user intent, it just consults whichever pair
		// corresponds to this block's position.
		blockRule := rule
		blockMaxIters := maxIters
		if bi == len(blocks)-1 {
			blockRule = terminalRule
			blockMaxIters = terminalMaxIters
		}

		// Snapshot the block's input. hidA currently holds the input to this
		// block (embedding for block 0, previous block's output for block > 0).
		copy(blockInput, hidA)

		// blockOut tracks which buffer holds the block's output after the final iter.
		// Declared at block scope so the buffer-copy logic after the loop can use it.
		var blockOut []float32

		// Per-block per-iter state: previous iter's block output (for
		// DeltaHiddenNorm) and previous iter's SSM state delta (for
		// SSMStateDeltaPre). Both reset per block.
		var prevBlockOutPtr []float32
		prevSSMDelta := 0.0

		// Magnitude-norm mode captures magIn (block entry magnitude) and
		// magOut (first-pass output magnitude). On iter > 0, the block
		// output is normalized to magIn and blended with the previous
		// iter's input at ratio alpha. The final output is normalized to
		// magOut at handoff. SSM state is saved/restored (not blended)
		// to prevent double-advancement of mutable layer state.
		var magIn, magOut float64

		for iter := 0; ; iter++ {
			if magnitudeNorm {
				// Magnitude-norm treats layers as sealed black boxes for
				// hidden-state blending, but SSM state (conv_state, ssm_state)
				// is mutable and advances during each forward pass. We must
				// save before iter 0 and restore before each subsequent iter
				// to prevent double-advancement.
				if iter > 0 {
					// Restore SSM state to pre-iteration save point.
					delta, err := blendState(bi, 0) // alpha=0 → full restore
					if err != nil {
						return nil, stats, fmt.Errorf("rlb block=%d iter=%d ssm restore: %w", bi, iter, err)
					}
					prevSSMDelta = delta

					// Blend: nextInput = (1−α)·prevIterInput + α·norm(blockOut, magIn)
					// prevIterInput is in blockInput; blockOut may alias hidA or hidB.
					// Use accumulated[] as scratch for the normalized output.
					copy(accumulated, blockOut)
					scaleToMag(accumulated, magIn)
					a := float32(alpha)
					for i := range hidA {
						hidA[i] = (1-a)*blockInput[i] + a*accumulated[i]
					}
					// Save this iter's input as prevIterInput for the next.
					copy(blockInput, hidA)
				} else {
					magIn = l2Norm(hidA)
					saveState(bi)
				}
			} else {
				// Mythos-shape: restore original block input each iter.
				// SSM state evolves via save/blend below.
				if iter > 0 {
					copy(hidA, blockInput)
				}
				saveState(bi)
			}

			// Forward all layers in the block, threading hidden state via ping-pong.
			forwardStart := time.Now()
			in, out := hidA, hidB
			for il := firstL; il <= lastL; il++ {
				var logitsDest []float32
				if il == lastL {
					logitsDest = logitsBuf
				}
				if err := forwardLayer(il, in, out, logitsDest); err != nil {
					return nil, stats, fmt.Errorf("rlb block=%d iter=%d layer=%d: %w", bi, iter, il, err)
				}
				in, out = out, in
			}
			// After the layer loop, `in` holds the block's output (the last
			// RLBForwardLayer wrote into `out`, then the swap made it `in`).
			blockOut = in
			elapsed := time.Since(forwardStart)

			if magnitudeNorm && iter == 0 {
				magOut = l2Norm(blockOut)
			}

			// Build iter record from last-layer projected logits, with Tier 1/2
			// hidden/SSM convergence metrics.
			rec := buildRLBIterRec(iter, logitsBuf, elapsed, tokenString,
				blockInput, blockOut, prevBlockOutPtr, prevSSMDelta)
			rec.BlockIdx = bi
			rec.BlockLayers = [2]int{firstL, lastL}
			rec.GlobalStep = globalStep
			globalStep++
			log.Info("rlb: block=%d iter=%d top1=%q logprob=%.4f time=%.1fms",
				bi, iter, rec.Top1Token, rec.Top1LogProb, rec.ElapsedMs)
			history = append(history, rec)
			dumpWrite(rec)
			stats.thinkTotalMs += rec.ElapsedMs

			// Halt check after appending current iter. The rule also produces an
			// auto-alpha used when the caller is in auto-alpha mode. blockRule
			// and blockMaxIters are selected per block above (terminal vs main).
			halt, ruleAlpha := blockRule(history)
			if halt {
				break
			}
			if iter >= blockMaxIters {
				break
			}

			// Snapshot this iter's blockOut for the next iter's prevBlockOut
			// reference (before the hidA restore destroys it).
			copy(prevBlockOutBuf, blockOut)
			prevBlockOutPtr = prevBlockOutBuf

			if !magnitudeNorm {
				// Mythos-shape: blend SSM state with pre-forward save.
				effectiveAlpha := alpha
				if alpha == autoAlpha {
					effectiveAlpha = ruleAlpha
				}
				delta, err := blendState(bi, effectiveAlpha)
				if err != nil {
					return nil, stats, fmt.Errorf("rlb block=%d iter=%d blend: %w", bi, iter, err)
				}
				prevSSMDelta = delta
			}
		}

		stats.completedItersByBlock[bi] = len(history)
		stats.totalItersRun += len(history)

		if magnitudeNorm && len(history) > 1 {
			// Handoff: normalize final block output to first-pass output
			// magnitude so next block sees the expected range.
			if &blockOut[0] != &hidA[0] {
				copy(hidA, blockOut)
			}
			scaleToMag(hidA, magOut)
		} else {
			// Mythos-shape or single-iter: copy raw block output to hidA.
			if len(blockOut) != len(hidA) {
				return nil, stats, fmt.Errorf("rlb block=%d: hidden buffer length mismatch %d vs %d", bi, len(blockOut), len(hidA))
			}
			if &blockOut[0] != &hidA[0] {
				copy(hidA, blockOut)
			}
		}
	}

	// Advance cache position and project final logits.
	finalLogits, err = projectLogits(hidA, nTokens)
	if err != nil {
		return nil, stats, fmt.Errorf("rlb final project: %w", err)
	}
	return finalLogits, stats, nil
}

// perBlockRLBPrefill runs the per-block recurrent prefill. It is a thin wrapper
// around perBlockForwardCore that supplies model-backed closures and manages the
// SSM state save buffers. rule/maxIters apply to every block except the terminal
// block; terminalRule/terminalMaxIters apply to the terminal block. Callers must
// pre-resolve the "no separate terminal rule" case by passing the main pair for
// both — see generateRLB for the resolution policy.
func (e *Engine) perBlockRLBPrefill(
	cache *arch.GenericCache,
	promptIDs []int32,
	flashAttn bool,
	alpha float64,
	magnitudeNorm bool,
	rule haltRule, maxIters int,
	terminalRule haltRule, terminalMaxIters int,
	dumpFile *os.File,
) (finalLogits []float32, stats perBlockStats, err error) {
	nLayers := e.model.Params.Ints[arch.ParamNLayers]

	// Block ranges: fall back to single block if full_attn_interval is absent.
	blocks := e.model.RLBBlockRanges()
	if len(blocks) == 0 {
		blocks = [][2]int{{0, nLayers - 1}}
	}

	nVocab := e.model.Params.Ints[arch.ParamNVocab]

	ssmBufsByBlock := buildSSMBufsByBlock(cache, blocks)

	// Token positions for seqPos=0 prefill — built once, reused across all block-iters.
	nTokens := len(promptIDs)
	positions := make([]int32, nTokens)
	for i := range positions {
		positions[i] = int32(i)
	}

	saveState := func(bi int) {
		for _, b := range ssmBufsByBlock[bi] {
			t := cache.Layers[b.layer].Tensors[b.name]
			copy(b.data, ggml.TensorGetBytes(t, 0, len(b.data)))
		}
	}

	blendState := func(bi int, a float64) (float64, error) {
		// totalDeltaSq accumulates ||post_forward - pre_save||^2 across all
		// (conv_state, ssm_state) tensors belonging to layers in this block.
		// The returned delta is always measured regardless of a — at a=1.0
		// (no blend), delta still reports how far the forward pass moved the
		// state from the save point.
		var totalDeltaSq float64
		for _, b := range ssmBufsByBlock[bi] {
			t := cache.Layers[b.layer].Tensors[b.name]
			updated := make([]byte, t.Nbytes())
			copy(updated, ggml.TensorGetBytes(t, 0, len(updated)))

			// Reinterpret the fresh copies as float32 views and accumulate
			// the per-tensor squared delta before any blending mutates them.
			n := len(updated) / 4
			curF := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(updated))), n)
			savF := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(b.data))), n)
			for i := 0; i < n; i++ {
				d := float64(curF[i]) - float64(savF[i])
				totalDeltaSq += d * d
			}

			// At a == 1.0 the blend is a no-op; skip the mutation and the
			// writeback to save bandwidth, but keep the delta above.
			if a < 1.0 {
				if err := blendFloat32(updated, b.data, a); err != nil {
					return 0, err
				}
				ggml.TensorSetBytes(t, updated, 0)
			}
		}
		return math.Sqrt(totalDeltaSq), nil
	}

	forwardLayer := func(il int, hidIn, hidOut, logitsOut []float32) error {
		// Prefill RLB writes both K/V and SSM state — it is filling the cache
		// the decode loop will read from.
		return e.model.RLBForwardLayer(cache, il, hidIn, positions, flashAttn, true, true, hidOut, logitsOut)
	}

	projectLogits := func(hid []float32, n int) ([]float32, error) {
		// Set cache.SeqPos so the downstream decode loop sees the right position.
		cache.SeqPos = n
		return e.model.RLBProjectLogits(hid, n)
	}

	dumpWrite := func(rec any) {
		rlbDumpWrite(dumpFile, rec)
	}

	return perBlockForwardCore(
		promptIDs, blocks, nVocab,
		e.tokenizer.TokenString,
		alpha,
		magnitudeNorm,
		rule, maxIters,
		terminalRule, terminalMaxIters,
		e.model.RLBForwardEmbed,
		forwardLayer,
		projectLogits,
		saveState,
		blendState,
		dumpWrite,
	)
}

// decodeRLBState holds pre-allocated resources for per-token block iteration
// during decode. Created once per generation request by newDecodeRLBState;
// step() is called once per decode token.
type decodeRLBState struct {
	blocks         [][2]int
	nVocab         int
	ssmBufsByBlock [][]ssmBuf
	cache          *arch.GenericCache

	tokenString   func(int32) string
	forwardEmbed  func([]int32) ([]float32, error)
	forwardLayer  func(il int, hidIn, hidOut, logitsOut []float32) error
	projectLogits func([]float32, int) ([]float32, error)
	saveState     func(blockIdx int)
	blendState    func(blockIdx int, alpha float64) (float64, error)
	dumpWrite     func(rec any)

	rule             haltRule
	ruleLabel        string
	maxIters         int
	terminalRule     haltRule
	terminalMaxIters int
	alpha            float64
	magnitudeNorm    bool

	tokenIdx int // monotonic decode token counter for JSONL token_pos
}

// newDecodeRLBState constructs a decodeRLBState for one generation request.
// Resolves halt rules, pre-allocates SSM save buffers, and builds closures.
// The forwardLayer closure reads cache.SeqPos at call time so each token's
// iteration set uses the correct decode position.
func (e *Engine) newDecodeRLBState(
	cache *arch.GenericCache,
	blocks [][2]int,
	flashAttn bool,
	params *RLBParams,
	dumpFile *os.File,
) *decodeRLBState {
	nVocab := e.model.Params.Ints[arch.ParamNVocab]

	ssmBufsByBlock := buildSSMBufsByBlock(cache, blocks)

	saveState := func(bi int) {
		for _, b := range ssmBufsByBlock[bi] {
			t := cache.Layers[b.layer].Tensors[b.name]
			copy(b.data, ggml.TensorGetBytes(t, 0, len(b.data)))
		}
	}

	blendState := func(bi int, a float64) (float64, error) {
		var totalDeltaSq float64
		for _, b := range ssmBufsByBlock[bi] {
			t := cache.Layers[b.layer].Tensors[b.name]
			updated := make([]byte, t.Nbytes())
			copy(updated, ggml.TensorGetBytes(t, 0, len(updated)))

			n := len(updated) / 4
			curF := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(updated))), n)
			savF := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(b.data))), n)
			for i := 0; i < n; i++ {
				d := float64(curF[i]) - float64(savF[i])
				totalDeltaSq += d * d
			}

			if a < 1.0 {
				if err := blendFloat32(updated, b.data, a); err != nil {
					return 0, err
				}
				ggml.TensorSetBytes(t, updated, 0)
			}
		}
		return math.Sqrt(totalDeltaSq), nil
	}

	// forwardLayer reads cache.SeqPos at call time — each token's block
	// iterations all run at the same position (the current decode position).
	// Decode RLB writes K/V and SSM state — it is the inference path; PMM is
	// the only consumer that disables writes.
	forwardLayer := func(il int, hidIn, hidOut, logitsOut []float32) error {
		positions := []int32{int32(cache.SeqPos)}
		return e.model.RLBForwardLayer(cache, il, hidIn, positions, flashAttn, true, true, hidOut, logitsOut)
	}

	// projectLogits does NOT set cache.SeqPos — the caller increments it
	// after step() returns.
	projectLogits := func(hid []float32, n int) ([]float32, error) {
		return e.model.RLBProjectLogits(hid, n)
	}

	dumpWriteFn := func(rec any) {
		rlbDumpWrite(dumpFile, rec)
	}

	rule, ruleLabel, maxIters := parseHaltRule(params.HaltRule)
	// Terminal rule defaults to the main rule when unset.
	terminalRule, _, terminalMaxIters := rule, ruleLabel, maxIters
	if params.TerminalHaltRule != "" {
		terminalRule, _, terminalMaxIters = parseHaltRule(params.TerminalHaltRule)
	}

	return &decodeRLBState{
		blocks:           blocks,
		nVocab:           nVocab,
		ssmBufsByBlock:   ssmBufsByBlock,
		cache:            cache,
		tokenString:      e.tokenizer.TokenString,
		forwardEmbed:     e.model.RLBForwardEmbed,
		forwardLayer:     forwardLayer,
		projectLogits:    projectLogits,
		saveState:        saveState,
		blendState:       blendState,
		dumpWrite:        dumpWriteFn,
		rule:             rule,
		ruleLabel:        ruleLabel,
		maxIters:         maxIters,
		terminalRule:     terminalRule,
		terminalMaxIters: terminalMaxIters,
		alpha:            params.Alpha,
		magnitudeNorm:    params.MagnitudeNorm,
	}
}

// step runs one decode token through per-block iteration. Returns logits for
// sampling. Does NOT increment cache.SeqPos — the caller is responsible for
// that after step returns.
func (s *decodeRLBState) step(tokenID int32) ([]float32, perBlockStats, error) {
	s.tokenIdx++
	tokenPos := s.tokenIdx

	// Wrap dumpWrite to inject token_pos into iter records. rlbIterRec is
	// passed by value from perBlockForwardCore, so type-asserting to the value
	// type gives us a copy we can modify before forwarding to the real writer.
	dumpWrite := func(rec any) {
		switch r := rec.(type) {
		case rlbIterRec:
			r.TokenPos = tokenPos
			s.dumpWrite(r)
		default:
			s.dumpWrite(rec)
		}
	}

	return perBlockForwardCore(
		[]int32{tokenID},
		s.blocks,
		s.nVocab,
		s.tokenString,
		s.alpha,
		s.magnitudeNorm,
		s.rule, s.maxIters,
		s.terminalRule, s.terminalMaxIters,
		s.forwardEmbed,
		s.forwardLayer,
		s.projectLogits,
		s.saveState,
		s.blendState,
		dumpWrite,
	)
}

func (e *Engine) generateRLB(
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

	// Block ranges (shared by prefill and decode RLB paths).
	nLayers := e.model.Params.Ints[arch.ParamNLayers]
	blocks := e.model.RLBBlockRanges()
	if len(blocks) == 0 {
		blocks = [][2]int{{0, nLayers - 1}}
	}

	// Open diagnostic dump. Failure is non-fatal.
	dumpFile := rlbDumpOpen(e.diagDir, e.model.ModelPath)
	defer func() {
		if dumpFile != nil {
			dumpFile.Close()
		}
	}()

	var logits []float32

	// === PREFILL ===
	if params.RLB.Prefill {
		prefillMagNorm := params.RLB.Decode && params.RLB.MagnitudeNorm

		// When magnitude-norm is active, prefill uses the same rule/alpha
		// as decode so the K/V cache is congruent with decode iteration.
		// Otherwise, parked defaults (dH_threshold, autoAlpha, capped iters).
		var prefillRule haltRule
		var prefillRuleLabel string
		var prefillMaxIters int
		var prefillAlpha float64

		if prefillMagNorm {
			prefillRule, prefillRuleLabel, prefillMaxIters = parseHaltRule(params.RLB.HaltRule)
			prefillAlpha = params.RLB.Alpha
			log.Info("rlb prefill: rule=%s max_iters=%d alpha=%.2f (mirroring decode)", prefillRuleLabel, prefillMaxIters, prefillAlpha)
		} else {
			prefillRule, prefillRuleLabel, prefillMaxIters = parseHaltRule(HaltRuleDHThreshold)
			if prefillMaxIters > rlbPrefillMaxIters {
				prefillMaxIters = rlbPrefillMaxIters
			}
			prefillAlpha = autoAlpha
			log.Info("rlb prefill: rule=%s max_iters=%d (parked defaults)", prefillRuleLabel, prefillMaxIters)
		}

		rlbDumpWrite(dumpFile, rlbSessionRec{
			Type:          "session",
			TS:            time.Now().UTC().Format(time.RFC3339),
			Model:         strings.TrimSuffix(filepath.Base(e.model.ModelPath), filepath.Ext(e.model.ModelPath)),
			PromptTokens:  len(promptIDs),
			Iterations:    prefillMaxIters,
			Alpha:         prefillAlpha,
			FlashAttn:     flashAttn,
			MagnitudeNorm: prefillMagNorm,
			HaltRule:      prefillRuleLabel,
			Granularity:   "per_block",
			Phase:         "prefill",
		})

		var pbs perBlockStats
		logits, pbs, err = e.perBlockRLBPrefill(
			cache, promptIDs, flashAttn,
			prefillAlpha,
			prefillMagNorm,
			prefillRule, prefillMaxIters,
			prefillRule, prefillMaxIters,
			dumpFile,
		)
		if err != nil {
			return fmt.Errorf("rlb prefill: %w", err)
		}
		metrics.PrefillDuration = time.Duration(pbs.thinkTotalMs * float64(time.Millisecond))
		log.Info("rlb prefill: %d total iters across %d blocks, total=%.1fms",
			pbs.totalItersRun, len(pbs.completedItersByBlock), pbs.thinkTotalMs)
		rlbDumpWrite(dumpFile, rlbSummaryRec{
			Type:                  "summary",
			Iterations:            prefillMaxIters,
			CompletedIterations:   pbs.totalItersRun,
			ThinkingMsTotal:       pbs.thinkTotalMs,
			CompletedItersByBlock: pbs.completedItersByBlock,
		})
	} else {
		// Standard single-pass prefill.
		// RLB + vision is unsupported in v1 (Engine.Generate rejects the
		// combination upstream); pass nil for splice inputs.
		prefillStart := time.Now()
		logits, err = e.model.ForwardCached(cache, promptIDs, flashAttn, nil)
		if err != nil {
			return fmt.Errorf("prefill forward: %w", err)
		}
		metrics.PrefillDuration = time.Since(prefillStart)
	}

	// === DECODE ===
	var decState *decodeRLBState
	if params.RLB.Decode {
		decState = e.newDecodeRLBState(cache, blocks, flashAttn, params.RLB, dumpFile)

		// Emit decode session record.
		alpha := params.RLB.Alpha
		terminalSpecified := params.RLB.TerminalHaltRule != ""
		_, ruleLabel, ruleIters := parseHaltRule(params.RLB.HaltRule)
		sessionRec := rlbSessionRec{
			Type:          "session",
			TS:            time.Now().UTC().Format(time.RFC3339),
			Model:         strings.TrimSuffix(filepath.Base(e.model.ModelPath), filepath.Ext(e.model.ModelPath)),
			PromptTokens:  len(promptIDs),
			Iterations:    ruleIters,
			Alpha:         alpha,
			FlashAttn:     flashAttn,
			MagnitudeNorm: params.RLB.MagnitudeNorm,
			HaltRule:      ruleLabel,
			Granularity:   "per_block",
			Phase:         "decode",
		}
		if terminalSpecified {
			_, termLabel, termIters := parseHaltRule(params.RLB.TerminalHaltRule)
			sessionRec.TerminalHaltRule = termLabel
			sessionRec.TerminalIters = termIters
		}
		rlbDumpWrite(dumpFile, sessionRec)

		log.Info("rlb decode: rule=%s max_iters=%d alpha=%.2f mag_norm=%v",
			decState.ruleLabel, decState.maxIters, decState.alpha, decState.magnitudeNorm)
	}

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

		if decState != nil {
			logits, _, err = decState.step(nextID)
			if err != nil {
				return fmt.Errorf("decode rlb forward: %w", err)
			}
			cache.SeqPos++
		} else {
			logits, err = e.model.ForwardCached(cache, []int32{nextID}, flashAttn, nil)
			if err != nil {
				return fmt.Errorf("decode forward: %w", err)
			}
		}
	}
	metrics.DecodeDuration = time.Since(decodeStart)
	if hitStop {
		metrics.FinishReason = FinishReasonStop
	} else {
		metrics.FinishReason = FinishReasonLength
	}
	return nil
}

// blendFloat32 blends dst toward old: dst = alpha*dst + (1-alpha)*old
// Both slices must be the same size and multiple of 4 bytes.
func blendFloat32(dst, old []byte, alpha float64) error {
	n := len(dst) / 4
	if len(old)/4 != n {
		return fmt.Errorf("blendFloat32: dst/old size mismatch: %d vs %d", len(dst), len(old))
	}
	newF := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(dst))), n)
	oldF := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(old))), n)
	for i := range newF {
		newF[i] = float32(alpha*float64(newF[i]) + (1-alpha)*float64(oldF[i]))
	}
	return nil
}
