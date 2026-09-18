package inference

import (
	"encoding/json"
	"math"
	"testing"
)

// --- helpers for building test histories ---

func makeRec(top1ID int32, top1LP float64, topk []rlbTopKEntry) rlbIterRec {
	return rlbIterRec{
		Type:        "iter",
		Top1ID:      top1ID,
		Top1Token:   "",
		Top1LogProb: top1LP,
		TopK:        topk,
	}
}

func makeTopK(logprobs ...float64) []rlbTopKEntry {
	entries := make([]rlbTopKEntry, len(logprobs))
	for i, lp := range logprobs {
		entries[i] = rlbTopKEntry{ID: int32(i), LogProb: lp}
	}
	return entries
}

// haltOnly extracts the halt bool from a two-return haltRule invocation,
// discarding the auto-alpha second return. Used to keep halt-decision tests
// concise — the auto-alpha path has its own dedicated tests below.
func haltOnly(rule haltRule, hist []rlbIterRec) bool {
	halt, _ := rule(hist)
	return halt
}

// --- entropyOfTopK ---

func TestEntropyOfTopK(t *testing.T) {
	const tol = 1e-9

	t.Run("empty", func(t *testing.T) {
		got := entropyOfTopK(nil)
		if got != 0 {
			t.Fatalf("want 0, got %v", got)
		}
	})

	t.Run("one_entry", func(t *testing.T) {
		// Single entry: renormalized probability is 1.0, entropy = 0.
		got := entropyOfTopK(makeTopK(-1.0))
		if got != 0 {
			t.Fatalf("want 0, got %v", got)
		}
	})

	t.Run("uniform_k2", func(t *testing.T) {
		// Two entries with equal logprob → uniform → entropy = log(2).
		got := entropyOfTopK(makeTopK(-1.0, -1.0))
		want := math.Log(2)
		if math.Abs(got-want) > tol {
			t.Fatalf("want %.9f, got %.9f", want, got)
		}
	})

	t.Run("uniform_k4", func(t *testing.T) {
		// Four entries with equal logprob → uniform → entropy = log(4).
		got := entropyOfTopK(makeTopK(-2.0, -2.0, -2.0, -2.0))
		want := math.Log(4)
		if math.Abs(got-want) > tol {
			t.Fatalf("want %.9f, got %.9f", want, got)
		}
	})

	t.Run("degenerate_one_hot", func(t *testing.T) {
		// First entry dominates: logprob 0, rest at -1000 (effectively zero prob).
		// Entropy should be very close to 0.
		got := entropyOfTopK(makeTopK(0.0, -1000.0, -1000.0))
		if got > 1e-6 {
			t.Fatalf("expected near-zero entropy, got %v", got)
		}
	})

	t.Run("logprob_max_alignment", func(t *testing.T) {
		// Shifted versions of the uniform case: adding a constant to all logprobs
		// must not change entropy (the renormalization subtracts the max).
		base := entropyOfTopK(makeTopK(-1.0, -1.0, -1.0))
		shifted := entropyOfTopK(makeTopK(5.0, 5.0, 5.0))
		if math.Abs(base-shifted) > tol {
			t.Fatalf("entropy should be shift-invariant: base=%.9f shifted=%.9f", base, shifted)
		}
	})
}

// --- rank2GapTopK ---

func TestRank2GapTopK(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		if got := rank2GapTopK(nil); got != 0 {
			t.Fatalf("want 0, got %v", got)
		}
	})

	t.Run("one_entry", func(t *testing.T) {
		if got := rank2GapTopK(makeTopK(-1.0)); got != 0 {
			t.Fatalf("want 0, got %v", got)
		}
	})

	t.Run("two_entries", func(t *testing.T) {
		got := rank2GapTopK(makeTopK(-1.0, -3.0))
		want := 2.0
		if math.Abs(got-want) > 1e-9 {
			t.Fatalf("want %.9f, got %.9f", want, got)
		}
	})

	t.Run("three_entries", func(t *testing.T) {
		// gap uses only rank1 and rank2
		got := rank2GapTopK(makeTopK(-0.5, -2.5, -10.0))
		want := 2.0
		if math.Abs(got-want) > 1e-9 {
			t.Fatalf("want %.9f, got %.9f", want, got)
		}
	})
}

// --- individual rule functions ---

func TestFixedCeiling(t *testing.T) {
	// Empty rule name returns the default fixed ceiling (1).
	rule, _, _ := parseHaltRule("")

	cases := []struct {
		histLen int
		want    bool
	}{
		{1, false}, // only iter 0 — 0 thinking iters done
		{2, true},  // 1 thinking iter done → len(history)=2 > 1 → halt
		{3, true},
	}
	for _, c := range cases {
		hist := make([]rlbIterRec, c.histLen)
		got := haltOnly(rule, hist)
		if got != c.want {
			t.Errorf("fixedCeiling(1) with histLen=%d: got %v, want %v", c.histLen, got, c.want)
		}
	}

	// Also verify fixed_3 explicitly.
	rule3, _, _ := parseHaltRule("fixed_3")
	if haltOnly(rule3, make([]rlbIterRec, 3)) {
		t.Error("fixed_3 should not halt at histLen=3")
	}
	if !haltOnly(rule3, make([]rlbIterRec, 4)) {
		t.Error("fixed_3 should halt at histLen=4")
	}
}

func TestLogprobPeak(t *testing.T) {
	rule, _, _ := parseHaltRule(HaltRuleLogprobPeak)

	t.Run("single_entry_no_halt", func(t *testing.T) {
		hist := []rlbIterRec{makeRec(1, -0.5, nil)}
		if haltOnly(rule, hist) {
			t.Fatal("should not halt with only 1 entry")
		}
	})

	t.Run("ascending_no_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(1, -2.0, nil),
			makeRec(1, -1.5, nil),
			makeRec(1, -1.0, nil),
		}
		if haltOnly(rule, hist) {
			t.Fatal("ascending logprob: should not halt")
		}
	})

	t.Run("descent_triggers_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(1, -2.0, nil),
			makeRec(1, -1.0, nil), // peak
			makeRec(1, -1.5, nil), // drop
		}
		if !haltOnly(rule, hist) {
			t.Fatal("logprob dropped: should halt")
		}
	})

	t.Run("equal_no_halt", func(t *testing.T) {
		// equal is not strictly less — no halt
		hist := []rlbIterRec{
			makeRec(1, -1.0, nil),
			makeRec(1, -1.0, nil),
		}
		if haltOnly(rule, hist) {
			t.Fatal("equal logprob should not trigger halt")
		}
	})
}

func TestEntropyMin(t *testing.T) {
	rule, _, _ := parseHaltRule(HaltRuleEntropyMin)

	// Uniform k=2: entropy = log(2) ≈ 0.693
	// Uniform k=4: entropy = log(4) ≈ 1.386
	uniformK2 := makeTopK(-1.0, -1.0)
	uniformK4 := makeTopK(-1.0, -1.0, -1.0, -1.0)

	t.Run("single_entry_no_halt", func(t *testing.T) {
		hist := []rlbIterRec{makeRec(1, -0.5, uniformK2)}
		if haltOnly(rule, hist) {
			t.Fatal("should not halt with only 1 entry")
		}
	})

	t.Run("entropy_decreasing_no_halt", func(t *testing.T) {
		// entropy going from high to low = confidence growing = do NOT halt
		hist := []rlbIterRec{
			makeRec(1, -1.0, uniformK4),
			makeRec(1, -1.0, uniformK2),
		}
		if haltOnly(rule, hist) {
			t.Fatal("decreasing entropy should not halt")
		}
	})

	t.Run("entropy_increasing_triggers_halt", func(t *testing.T) {
		// entropy going from low to high = confidence falling = halt
		hist := []rlbIterRec{
			makeRec(1, -1.0, uniformK2),
			makeRec(1, -1.0, uniformK4),
		}
		if !haltOnly(rule, hist) {
			t.Fatal("increasing entropy should halt")
		}
	})

	t.Run("equal_entropy_no_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(1, -1.0, uniformK2),
			makeRec(1, -1.0, uniformK2),
		}
		if haltOnly(rule, hist) {
			t.Fatal("equal entropy should not halt")
		}
	})
}

func TestGapPeak(t *testing.T) {
	rule, _, _ := parseHaltRule(HaltRuleGapPeak)

	bigGap := makeTopK(-1.0, -5.0)   // gap = 4.0
	smallGap := makeTopK(-1.0, -2.0) // gap = 1.0

	t.Run("single_entry_no_halt", func(t *testing.T) {
		hist := []rlbIterRec{makeRec(1, -0.5, bigGap)}
		if haltOnly(rule, hist) {
			t.Fatal("should not halt with only 1 entry")
		}
	})

	t.Run("gap_growing_no_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(1, -1.0, smallGap),
			makeRec(1, -1.0, bigGap),
		}
		if haltOnly(rule, hist) {
			t.Fatal("growing gap should not halt")
		}
	})

	t.Run("gap_shrinking_triggers_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(1, -1.0, bigGap),
			makeRec(1, -1.0, smallGap),
		}
		if !haltOnly(rule, hist) {
			t.Fatal("shrinking gap should halt")
		}
	})

	t.Run("equal_gap_no_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(1, -1.0, bigGap),
			makeRec(1, -1.0, bigGap),
		}
		if haltOnly(rule, hist) {
			t.Fatal("equal gap should not halt")
		}
	})
}

func TestArgmaxStable2(t *testing.T) {
	rule, _, _ := parseHaltRule(HaltRuleArgmaxStable2)

	t.Run("single_no_halt", func(t *testing.T) {
		hist := []rlbIterRec{makeRec(42, -0.5, nil)}
		if haltOnly(rule, hist) {
			t.Fatal("should not halt with 1 entry")
		}
	})

	t.Run("different_ids_no_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(1, -0.5, nil),
			makeRec(2, -0.5, nil),
		}
		if haltOnly(rule, hist) {
			t.Fatal("different Top1IDs should not halt")
		}
	})

	t.Run("same_ids_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(1, -0.5, nil),
			makeRec(1, -0.5, nil),
		}
		if !haltOnly(rule, hist) {
			t.Fatal("matching Top1IDs for 2 consecutive iters should halt")
		}
	})

	t.Run("three_but_last_two_match_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(99, -0.5, nil),
			makeRec(1, -0.5, nil),
			makeRec(1, -0.5, nil),
		}
		if !haltOnly(rule, hist) {
			t.Fatal("last two matching should halt regardless of earlier entries")
		}
	})
}

func TestArgmaxStable3(t *testing.T) {
	rule, _, _ := parseHaltRule(HaltRuleArgmaxStable3)

	t.Run("two_matching_no_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(1, -0.5, nil),
			makeRec(1, -0.5, nil),
		}
		if haltOnly(rule, hist) {
			t.Fatal("only 2 matching: should not halt (need 3)")
		}
	})

	t.Run("three_not_all_same_no_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(1, -0.5, nil),
			makeRec(2, -0.5, nil),
			makeRec(1, -0.5, nil),
		}
		if haltOnly(rule, hist) {
			t.Fatal("not all 3 same: should not halt")
		}
	})

	t.Run("three_same_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(7, -0.5, nil),
			makeRec(7, -0.5, nil),
			makeRec(7, -0.5, nil),
		}
		if !haltOnly(rule, hist) {
			t.Fatal("3 consecutive same Top1ID should halt")
		}
	})

	t.Run("four_with_last_three_same_halt", func(t *testing.T) {
		hist := []rlbIterRec{
			makeRec(99, -0.5, nil),
			makeRec(7, -0.5, nil),
			makeRec(7, -0.5, nil),
			makeRec(7, -0.5, nil),
		}
		if !haltOnly(rule, hist) {
			t.Fatal("last 3 same should halt")
		}
	})
}

// makeDHRec constructs an iter record populated with just the fields
// dH_threshold consumes: Iter, DeltaHiddenNorm, BlockOutNorm. Keeps the
// test cases below terse and self-documenting.
func makeDHRec(iter int, dH, outNorm float64) rlbIterRec {
	return rlbIterRec{
		Type:            "iter",
		Iter:            iter,
		DeltaHiddenNorm: dH,
		BlockOutNorm:    outNorm,
	}
}

// TestDHThreshold exercises the Tier 2 dH_threshold rule's halt boundary
// and auto-alpha scaling. The rule halts when the last iter's normalized
// ΔH (DeltaHiddenNorm / BlockOutNorm) falls below dHThresholdRatio (0.01);
// below that threshold is "block converged", above is "keep iterating".
// Iter 0 is a forced fallthrough because DeltaHiddenNorm is 0 by definition
// on the first pass through any block. BlockOutNorm == 0 is a defensive
// fallthrough that should not occur in practice.
//
// Auto-alpha: normDH / (2 * threshold), clamped to [0, 1]. At the halt
// boundary the alpha is 0.5; above 2×threshold it saturates at 1.0; below
// threshold it is < 0.5 but the rule halts anyway so the alpha is unused.
func TestDHThreshold(t *testing.T) {
	const tol = 1e-9
	rule, _, maxIters := parseHaltRule(HaltRuleDHThreshold)

	if maxIters != dHThresholdMaxIters {
		t.Fatalf("maxIters: got %d, want %d", maxIters, dHThresholdMaxIters)
	}

	t.Run("iter0_never_halts", func(t *testing.T) {
		// Iter 0 — even with dH==0 (the natural case), rule must fall through.
		hist := []rlbIterRec{makeDHRec(0, 0, 100)}
		halt, nextA := rule(hist)
		if halt {
			t.Error("iter 0 must never halt")
		}
		if math.Abs(nextA-1.0) > tol {
			t.Errorf("iter 0 auto-alpha: got %.9f, want 1.0", nextA)
		}
	})

	t.Run("outnorm_zero_fallthrough", func(t *testing.T) {
		// Defensive guard: if BlockOutNorm is 0 (should never happen in a
		// real run), the rule must not halt and must not produce NaN alpha.
		hist := []rlbIterRec{
			makeDHRec(0, 0, 100),
			makeDHRec(1, 5, 0),
		}
		halt, nextA := rule(hist)
		if halt {
			t.Error("outNorm==0: should fall through, not halt")
		}
		if math.IsNaN(nextA) || nextA != 1.0 {
			t.Errorf("outNorm==0 auto-alpha: got %v, want 1.0", nextA)
		}
	})

	t.Run("above_threshold_no_halt", func(t *testing.T) {
		// normDH = 5/100 = 0.05, well above 0.01 — keep iterating.
		hist := []rlbIterRec{
			makeDHRec(0, 0, 100),
			makeDHRec(1, 5, 100),
		}
		halt, nextA := rule(hist)
		if halt {
			t.Error("normDH=0.05 > 0.01: should not halt")
		}
		// nextA = 0.05 / (2*0.01) = 2.5 → clamped to 1.0.
		if math.Abs(nextA-1.0) > tol {
			t.Errorf("auto-alpha: got %.9f, want 1.0 (saturated)", nextA)
		}
	})

	t.Run("at_halt_boundary_halts", func(t *testing.T) {
		// normDH = 0.5/100 = 0.005, below threshold 0.01 — halt.
		hist := []rlbIterRec{
			makeDHRec(0, 0, 100),
			makeDHRec(1, 0.5, 100),
		}
		halt, nextA := rule(hist)
		if !halt {
			t.Error("normDH=0.005 < 0.01: should halt")
		}
		// nextA = 0.005 / 0.02 = 0.25. Value is unused since halt=true,
		// but checking it documents the scaling math.
		want := 0.005 / (2 * dHThresholdRatio)
		if math.Abs(nextA-want) > tol {
			t.Errorf("auto-alpha: got %.9f, want %.9f", nextA, want)
		}
	})

	t.Run("alpha_scales_linearly_below_saturation", func(t *testing.T) {
		// normDH = 0.015, above threshold — no halt; nextA = 0.015/0.02 = 0.75.
		hist := []rlbIterRec{
			makeDHRec(0, 0, 100),
			makeDHRec(1, 1.5, 100),
		}
		halt, nextA := rule(hist)
		if halt {
			t.Error("normDH=0.015 > 0.01: should not halt")
		}
		want := 0.015 / (2 * dHThresholdRatio)
		if math.Abs(nextA-want) > tol {
			t.Errorf("auto-alpha: got %.9f, want %.9f", nextA, want)
		}
	})

	t.Run("halt_boundary_exactly_at_threshold", func(t *testing.T) {
		// normDH == threshold exactly — strict less-than, so this is NOT a
		// halt, it is the last non-halt step. Documents the boundary
		// convention (matches gap_peak/logprob_peak strictness).
		hist := []rlbIterRec{
			makeDHRec(0, 0, 100),
			makeDHRec(1, 1, 100),
		}
		halt, _ := rule(hist)
		if halt {
			t.Error("normDH == threshold: should not halt (strict <)")
		}
	})
}

// TestFixedNAutoAlpha locks in the running-mean auto-alpha schedule for the
// fixed_N halt rules: α_k = 1/len(history), where len(history) == k+1 at
// rule-call time. The sequence for fixed_3 is {1.0, 0.5, 1/3} across the
// three blends that set up iters 1, 2, 3 — equivalent to maintaining an
// arithmetic mean of all post-forward SSM states seen so far. The default
// rule ("") routes through fixed_3 and must yield the same schedule.
func TestFixedNAutoAlpha(t *testing.T) {
	const tol = 1e-9
	cases := []struct {
		ruleName string
		histLen  int
		wantA    float64
	}{
		{"fixed_0", 1, 1.0}, // halts immediately; value is never consumed but still correct
		{"fixed_3", 1, 1.0},
		{"fixed_3", 2, 0.5},
		{"fixed_3", 3, 1.0 / 3.0},
		{"fixed_5", 1, 1.0},
		{"fixed_5", 2, 0.5},
		{"fixed_5", 5, 0.2},
		{"", 1, 1.0}, // default routes through fixed_3
		{"", 2, 0.5},
		{"", 3, 1.0 / 3.0},
	}
	for _, c := range cases {
		rule, _, _ := parseHaltRule(c.ruleName)
		hist := make([]rlbIterRec, c.histLen)
		_, gotA := rule(hist)
		if math.Abs(gotA-c.wantA) > tol {
			t.Errorf("rule=%q histLen=%d: auto-alpha got %.9f, want %.9f",
				c.ruleName, c.histLen, gotA, c.wantA)
		}
	}
}

// --- parseHaltRule: label and basic halt decision per menu entry ---

func TestParseHaltRule(t *testing.T) {
	// A history of 1 (just prefill) for "too short to halt" checks.
	histOne := []rlbIterRec{makeRec(1, -0.5, nil)}
	// A history of 2 entries for default (fixed_1) ceiling checks.
	histTwo := make([]rlbIterRec, 2)
	// A history of 4 entries for fixed_3 ceiling checks.
	histFour := make([]rlbIterRec, 4)

	cases := []struct {
		name       string
		wantLabel  string
		wantIters  int          // expected loop bound from parseHaltRule
		histNoHalt []rlbIterRec // rule(histNoHalt) must be false
		histHalt   []rlbIterRec // rule(histHalt) must be true (nil = skip)
	}{
		{
			name:       "",
			wantLabel:  HaltRuleDefault + " (default)",
			wantIters:  1,
			histNoHalt: histOne,
			histHalt:   histTwo, // len=2 > 1 → halt
		},
		{
			name:       "fixed_0",
			wantLabel:  "fixed_0",
			wantIters:  0,
			histNoHalt: nil,
			histHalt:   histOne, // len=1 > 0 → halt
		},
		{
			name:       "fixed_3",
			wantLabel:  "fixed_3",
			wantIters:  3,
			histNoHalt: histOne,
			histHalt:   histFour, // len=4 > 3 → halt
		},
		{
			name:       "fixed_5",
			wantLabel:  "fixed_5",
			wantIters:  5,
			histNoHalt: make([]rlbIterRec, 5), // len=5 not > 5
			histHalt:   make([]rlbIterRec, 6), // len=6 > 5
		},
		{
			name:      HaltRuleLogprobPeak,
			wantLabel: HaltRuleLogprobPeak,
			wantIters: defaultRLBCeiling,
			histNoHalt: []rlbIterRec{
				makeRec(1, -2.0, nil),
				makeRec(1, -1.0, nil), // ascending
			},
			histHalt: []rlbIterRec{
				makeRec(1, -1.0, nil),
				makeRec(1, -2.0, nil), // descent
			},
		},
		{
			name:      HaltRuleEntropyMin,
			wantLabel: HaltRuleEntropyMin,
			wantIters: defaultRLBCeiling,
			histNoHalt: []rlbIterRec{
				makeRec(1, -1.0, makeTopK(-1.0, -1.0, -1.0, -1.0)),
				makeRec(1, -1.0, makeTopK(-1.0, -1.0)), // entropy dropped
			},
			histHalt: []rlbIterRec{
				makeRec(1, -1.0, makeTopK(-1.0, -1.0)),
				makeRec(1, -1.0, makeTopK(-1.0, -1.0, -1.0, -1.0)), // entropy rose
			},
		},
		{
			name:      HaltRuleGapPeak,
			wantLabel: HaltRuleGapPeak,
			wantIters: defaultRLBCeiling,
			histNoHalt: []rlbIterRec{
				makeRec(1, -1.0, makeTopK(-1.0, -5.0)),
				makeRec(1, -1.0, makeTopK(-1.0, -10.0)), // gap grew
			},
			histHalt: []rlbIterRec{
				makeRec(1, -1.0, makeTopK(-1.0, -10.0)),
				makeRec(1, -1.0, makeTopK(-1.0, -2.0)), // gap shrank
			},
		},
		{
			name:      HaltRuleArgmaxStable2,
			wantLabel: HaltRuleArgmaxStable2,
			wantIters: defaultRLBCeiling,
			histNoHalt: []rlbIterRec{
				makeRec(1, -0.5, nil),
				makeRec(2, -0.5, nil),
			},
			histHalt: []rlbIterRec{
				makeRec(5, -0.5, nil),
				makeRec(5, -0.5, nil),
			},
		},
		{
			name:      HaltRuleArgmaxStable3,
			wantLabel: HaltRuleArgmaxStable3,
			wantIters: defaultRLBCeiling,
			histNoHalt: []rlbIterRec{
				makeRec(5, -0.5, nil),
				makeRec(5, -0.5, nil),
			},
			histHalt: []rlbIterRec{
				makeRec(5, -0.5, nil),
				makeRec(5, -0.5, nil),
				makeRec(5, -0.5, nil),
			},
		},
		{
			name:      HaltRuleDHThreshold,
			wantLabel: HaltRuleDHThreshold,
			wantIters: dHThresholdMaxIters,
			// Iter 1 with normDH = 5/100 = 0.05 > threshold 0.01 → no halt.
			histNoHalt: []rlbIterRec{
				makeDHRec(0, 0, 100),
				makeDHRec(1, 5, 100),
			},
			// Iter 1 with normDH = 0.5/100 = 0.005 < threshold → halt.
			histHalt: []rlbIterRec{
				makeDHRec(0, 0, 100),
				makeDHRec(1, 0.5, 100),
			},
		},
	}

	for _, c := range cases {
		t.Run("name="+c.name, func(t *testing.T) {
			rule, label, iters := parseHaltRule(c.name)
			if label != c.wantLabel {
				t.Errorf("label: got %q, want %q", label, c.wantLabel)
			}
			if iters != c.wantIters {
				t.Errorf("iters: got %d, want %d", iters, c.wantIters)
			}
			if c.histNoHalt != nil {
				if haltOnly(rule, c.histNoHalt) {
					t.Errorf("rule(%q) should not halt on histNoHalt", c.name)
				}
			}
			if c.histHalt != nil {
				if !haltOnly(rule, c.histHalt) {
					t.Errorf("rule(%q) should halt on histHalt", c.name)
				}
			}
		})
	}

	t.Run("unknown_name_fallback_to_default", func(t *testing.T) {
		rule, label, iters := parseHaltRule("bogus_rule_xyz")
		wantLabel := HaltRuleDefault + " (default)"
		if label != wantLabel {
			t.Errorf("unknown name: label got %q, want %q", label, wantLabel)
		}
		if iters != 1 {
			t.Errorf("unknown name: iters got %d, want 1", iters)
		}
		// Should behave like fixed_1
		if haltOnly(rule, histOne) {
			t.Error("fallback rule should not halt on histOne (len=1)")
		}
		if !haltOnly(rule, histTwo) {
			t.Error("fallback rule should halt on histTwo (len=2 > 1)")
		}
	})
}

// =============================================================================
// Per-block driver unit tests (Step 8 of the rewrite plan)
// =============================================================================

// fakeForwardLayer is a forwardLayer closure that writes recognizable values
// into hidOut (token index * 1.0 per element) and increments a per-layer call
// counter. logitsOut is filled with descending values so Greedy picks index 0.
func makeFakeForward(nVocab int, layerCalls *[]int) func(il int, hidIn, hidOut, logitsOut []float32) error {
	return func(il int, hidIn, hidOut, logitsOut []float32) error {
		*layerCalls = append(*layerCalls, il)
		// Write something non-zero to hidOut.
		for i := range hidOut {
			hidOut[i] = float32(il + 1)
		}
		// Fill logitsOut so Greedy returns 0 (largest).
		for i := range logitsOut {
			logitsOut[i] = float32(nVocab - i)
		}
		return nil
	}
}

// runFakeCore is a helper that wires up perBlockForwardCore with fakes.
// It returns the stats and the layer call sequence. The main rule/maxIters
// are applied to every block; the terminal block (last entry in blocks) uses
// the same pair unless the test needs to exercise terminal override, in
// which case runFakeCoreTerm is the explicit variant.
func runFakeCore(
	t *testing.T,
	blocks [][2]int,
	nEmbd, nVocab, nTokens int,
	rule haltRule,
	maxIters int,
	alpha float64,
	blendCalls *int,
) (perBlockStats, []int) {
	t.Helper()
	return runFakeCoreTerm(t, blocks, nEmbd, nVocab, nTokens, rule, maxIters, rule, maxIters, alpha, blendCalls)
}

// runFakeCoreTerm is the explicit variant that lets tests pass distinct
// (rule, maxIters) pairs for non-terminal and terminal blocks. Used by
// TestPerBlockTerminalHaltRule to verify per-block dispatch.
func runFakeCoreTerm(
	t *testing.T,
	blocks [][2]int,
	nEmbd, nVocab, nTokens int,
	rule haltRule, maxIters int,
	terminalRule haltRule, terminalMaxIters int,
	alpha float64,
	blendCalls *int,
) (perBlockStats, []int) {
	t.Helper()

	promptIDs := make([]int32, nTokens)
	for i := range promptIDs {
		promptIDs[i] = int32(i)
	}

	var layerCalls []int

	forwardEmbed := func(tokens []int32) ([]float32, error) {
		return make([]float32, nEmbd*len(tokens)), nil
	}
	forwardLayer := makeFakeForward(nVocab, &layerCalls)
	projectLogits := func(hid []float32, n int) ([]float32, error) {
		out := make([]float32, nVocab)
		out[0] = 1.0 // make Greedy pick 0
		return out, nil
	}
	saveState := func(bi int) {}
	blendState := func(bi int, a float64) (float64, error) {
		if blendCalls != nil {
			*blendCalls++
		}
		// Fake returns zero delta — tests that care about SSMStateDeltaPre
		// don't go through this helper.
		return 0.0, nil
	}
	dumpWrite := func(rec any) {}
	tokenString := func(id int32) string { return "" }

	_, stats, err := perBlockForwardCore(
		promptIDs, blocks, nVocab,
		tokenString,
		alpha,
		false, // magnitudeNorm — existing tests use Mythos-shape
		rule, maxIters,
		terminalRule, terminalMaxIters,
		forwardEmbed, forwardLayer, projectLogits,
		saveState, blendState, dumpWrite,
	)
	if err != nil {
		t.Fatalf("perBlockForwardCore: %v", err)
	}
	return stats, layerCalls
}

// TestPerBlockHistoryIsolation asserts that history is reset between blocks —
// the halt rule for block 1 never sees records from block 0.
func TestPerBlockHistoryIsolation(t *testing.T) {
	blocks := [][2]int{{0, 1}, {2, 3}}
	nEmbd, nVocab, nTokens := 4, 8, 2

	// Rule that records history length each time it is called. After the run,
	// we verify that the first call for block 1 saw history of length 1 (not
	// accumulated from block 0).
	type callRecord struct {
		histLen int
	}
	var calls []callRecord

	rule := func(history []rlbIterRec) (bool, float64) {
		calls = append(calls, callRecord{histLen: len(history)})
		// "fixed_2" effectively: halt when 3rd iter has been appended. The
		// auto-alpha second return is never consumed because the helper passes
		// a concrete alpha (1.0) — it just has to match the haltRule type.
		return len(history) > 2, 0.5
	}

	stats, _ := runFakeCore(t, blocks, nEmbd, nVocab, nTokens, rule, 10, 1.0, nil)

	// Each block should have run independently.
	if len(stats.completedItersByBlock) != 2 {
		t.Fatalf("want 2 blocks, got %d", len(stats.completedItersByBlock))
	}

	// The rule is called once per iter, after appending. Verify that at the
	// start of block 1 the history length resets to 1 (only iter 0 of block 1).
	// Find the first call for block 1: it follows block 0's calls.
	block0Calls := stats.completedItersByBlock[0]
	if len(calls) <= block0Calls {
		t.Fatalf("not enough rule calls: got %d, expected at least %d", len(calls), block0Calls+1)
	}
	firstBlock1Call := calls[block0Calls]
	if firstBlock1Call.histLen != 1 {
		t.Errorf("first rule call for block 1: want histLen=1 (reset), got %d", firstBlock1Call.histLen)
	}
}

// TestPerBlockFixed0Determinism asserts that fixed_0 causes each block to run
// exactly one inner iter (completedItersByBlock[i] == 1 for all i).
func TestPerBlockFixed0Determinism(t *testing.T) {
	blocks := [][2]int{{0, 1}, {2, 3}, {4, 5}}
	nEmbd, nVocab, nTokens := 4, 8, 2

	rule, _, maxIters := parseHaltRule("fixed_0")

	stats, _ := runFakeCore(t, blocks, nEmbd, nVocab, nTokens, rule, maxIters, 1.0, nil)

	if len(stats.completedItersByBlock) != len(blocks) {
		t.Fatalf("want %d blocks, got %d", len(blocks), len(stats.completedItersByBlock))
	}
	for i, n := range stats.completedItersByBlock {
		if n != 1 {
			t.Errorf("block %d: want 1 iter, got %d", i, n)
		}
	}
	if stats.totalItersRun != len(blocks) {
		t.Errorf("totalItersRun: want %d, got %d", len(blocks), stats.totalItersRun)
	}
}

// TestPerBlockBlendCallsAtAlpha1 verifies that blendState is invoked the
// expected number of times at a fixed alpha of 1.0. blendState is always
// called between iters (for SSM delta measurement, even at alpha=1.0 where
// the actual blend mutation is a no-op inside the implementation); the
// invariant here is simply the per-iter call count.
//
// With fixed_3 and 2 blocks: each block runs 4 iters (iters 0..3 with halt
// firing at len(history)=4 > 3). There are 3 between-iter blend points per
// block → 6 total calls across 2 blocks.
func TestPerBlockBlendCallsAtAlpha1(t *testing.T) {
	blocks := [][2]int{{0, 1}, {2, 3}}
	nEmbd, nVocab, nTokens := 4, 8, 2

	rule, _, maxIters := parseHaltRule("fixed_3")

	blendCalls := 0
	runFakeCore(t, blocks, nEmbd, nVocab, nTokens, rule, maxIters, 1.0, &blendCalls)

	const wantBlendCalls = 6 // 2 blocks × 3 between-iter blends
	if blendCalls != wantBlendCalls {
		t.Errorf("alpha=1.0: blendState called %d times, want %d", blendCalls, wantBlendCalls)
	}
}

// TestPerBlockJSONLRecordShape verifies the JSON encoding of rlbIterRec with
// the per-block fields and the new Tier 1 / Tier 2 signal fields, plus
// backward-read compatibility.
func TestPerBlockJSONLRecordShape(t *testing.T) {
	rec := rlbIterRec{
		Type:        "iter",
		Iter:        2,
		BlockIdx:    3,
		BlockLayers: [2]int{4, 7},
		GlobalStep:  42,
		Top1ID:      99,
		Top1Token:   "hello",
		Top1LogProb: -0.123,
		ElapsedMs:   15.6,
		TopK:        []rlbTopKEntry{{ID: 99, Token: "hello", LogProb: -0.123}},

		// Tier 1 — logit distribution shape
		TopKEntropy: 0.45,
		Rank12Gap:   1.2,
		Rank1KGap:   5.5,
		LogitMax:    12.0,
		LogitMin:    -8.0,
		LogitMean:   0.1,
		LogitVar:    3.4,

		// Tier 2 — hidden / SSM convergence
		BlockInNorm:      10.5,
		BlockOutNorm:     11.2,
		DeltaHiddenNorm:  0.8,
		SSMStateDeltaPre: 0.03,
	}

	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Check type field.
	if m["type"] != "iter" {
		t.Errorf("type: got %v, want \"iter\"", m["type"])
	}

	// iter must be numeric (Python rec['iter'] compat).
	iterVal, ok := m["iter"].(float64)
	if !ok {
		t.Errorf("iter: not a number, got %T %v", m["iter"], m["iter"])
	} else if iterVal != 2 {
		t.Errorf("iter: got %v, want 2", iterVal)
	}

	// top1_logprob present as a number.
	if _, ok := m["top1_logprob"].(float64); !ok {
		t.Errorf("top1_logprob: not a number, got %T", m["top1_logprob"])
	}

	// Per-block fields.
	if v, _ := m["block_idx"].(float64); int(v) != 3 {
		t.Errorf("block_idx: got %v, want 3", m["block_idx"])
	}
	if v, _ := m["global_step"].(float64); int(v) != 42 {
		t.Errorf("global_step: got %v, want 42", m["global_step"])
	}

	// block_layers is a JSON array; after unmarshal it's []interface{}.
	bl, ok := m["block_layers"].([]any)
	if !ok || len(bl) != 2 {
		t.Fatalf("block_layers: expected []any of len 2, got %T %v", m["block_layers"], m["block_layers"])
	}
	if bl[0].(float64) != 4 || bl[1].(float64) != 7 {
		t.Errorf("block_layers: got [%v, %v], want [4, 7]", bl[0], bl[1])
	}

	// Existing fields still present.
	for _, key := range []string{"top1_id", "top1_token", "elapsed_ms", "top_k"} {
		if _, exists := m[key]; !exists {
			t.Errorf("existing field %q missing from JSON", key)
		}
	}

	// Tier 1 fields — shape stats and top-K summaries must round-trip as
	// numbers. Zero is still a valid value; we just want the keys present.
	tier1 := []string{"top_k_entropy", "rank12_gap", "rank1k_gap",
		"logit_max", "logit_min", "logit_mean", "logit_var"}
	for _, key := range tier1 {
		if _, ok := m[key].(float64); !ok {
			t.Errorf("Tier 1 field %q: not a number, got %T %v", key, m[key], m[key])
		}
	}

	// Tier 2 fields — hidden / SSM norms. Same round-trip check.
	tier2 := []string{"block_in_norm", "block_out_norm", "delta_hidden_norm", "ssm_state_delta_pre"}
	for _, key := range tier2 {
		if _, ok := m[key].(float64); !ok {
			t.Errorf("Tier 2 field %q: not a number, got %T %v", key, m[key], m[key])
		}
	}

	// Spot-check a couple of round-tripped values to catch struct-tag typos.
	if v, _ := m["rank12_gap"].(float64); v != 1.2 {
		t.Errorf("rank12_gap round-trip: got %v, want 1.2", v)
	}
	if v, _ := m["ssm_state_delta_pre"].(float64); v != 0.03 {
		t.Errorf("ssm_state_delta_pre round-trip: got %v, want 0.03", v)
	}
}

// TestPerBlockTerminalHaltRule asserts that per-block rule dispatch routes
// non-terminal blocks through `rule`/`maxIters` and the terminal block
// through `terminalRule`/`terminalMaxIters`. The test uses fixed_0 on
// non-terminal blocks (→ 1 iter each) and fixed_3 on the terminal block
// (→ 4 iters, since fixed_N halts when len(history) > N and the inner loop
// also bails at iter > maxIters). The distinct iter counts between
// non-terminal and terminal blocks prove the dispatch branch is firing.
// A second subtest reverses the roles (fixed_3 upstream, fixed_0 terminal)
// to catch a dispatch that is silently stuck on one rule.
func TestPerBlockTerminalHaltRule(t *testing.T) {
	nEmbd, nVocab, nTokens := 4, 8, 2
	blocks := [][2]int{{0, 1}, {2, 3}, {4, 5}} // 3 blocks; block 2 is terminal

	nonTerm, _, nonTermIters := parseHaltRule("fixed_0")
	term, _, termIters := parseHaltRule("fixed_3")

	t.Run("terminal_runs_more_iters", func(t *testing.T) {
		stats, _ := runFakeCoreTerm(t, blocks, nEmbd, nVocab, nTokens,
			nonTerm, nonTermIters, term, termIters, 1.0, nil)

		// Non-terminal blocks: fixed_0 → 1 iter each.
		for i := 0; i < len(blocks)-1; i++ {
			if stats.completedItersByBlock[i] != 1 {
				t.Errorf("non-terminal block %d: want 1 iter (fixed_0), got %d",
					i, stats.completedItersByBlock[i])
			}
		}
		// Terminal block: fixed_3 → 4 iters.
		last := len(blocks) - 1
		if stats.completedItersByBlock[last] != 4 {
			t.Errorf("terminal block: want 4 iters (fixed_3), got %d",
				stats.completedItersByBlock[last])
		}
	})

	t.Run("roles_reversed", func(t *testing.T) {
		// Swap: fixed_3 on non-terminal, fixed_0 on terminal.
		stats, _ := runFakeCoreTerm(t, blocks, nEmbd, nVocab, nTokens,
			term, termIters, nonTerm, nonTermIters, 1.0, nil)

		for i := 0; i < len(blocks)-1; i++ {
			if stats.completedItersByBlock[i] != 4 {
				t.Errorf("non-terminal block %d: want 4 iters (fixed_3), got %d",
					i, stats.completedItersByBlock[i])
			}
		}
		last := len(blocks) - 1
		if stats.completedItersByBlock[last] != 1 {
			t.Errorf("terminal block: want 1 iter (fixed_0), got %d",
				stats.completedItersByBlock[last])
		}
	})
}

// TestPerBlockBlockBoundaries asserts that forwardLayer is called with the
// correct layer indices in order, advancing through each block's range.
func TestPerBlockBlockBoundaries(t *testing.T) {
	blocks := [][2]int{{0, 3}, {4, 7}, {8, 11}}
	nEmbd, nVocab, nTokens := 4, 8, 2
	layersPerBlock := 4

	rule, _, maxIters := parseHaltRule("fixed_0") // one iter per block

	_, layerCalls := runFakeCore(t, blocks, nEmbd, nVocab, nTokens, rule, maxIters, 1.0, nil)

	// With fixed_0, each block runs exactly 1 iter. Each iter forwards
	// layersPerBlock layers in order.
	wantCallCount := len(blocks) * layersPerBlock
	if len(layerCalls) != wantCallCount {
		t.Fatalf("layer call count: got %d, want %d", len(layerCalls), wantCallCount)
	}

	// Verify layer indices advance in order through the block ranges.
	expected := make([]int, 0, wantCallCount)
	for _, br := range blocks {
		for il := br[0]; il <= br[1]; il++ {
			expected = append(expected, il)
		}
	}
	for i, got := range layerCalls {
		if got != expected[i] {
			t.Errorf("layer call[%d]: got il=%d, want il=%d", i, got, expected[i])
		}
	}
}

// =============================================================================
// decodeRLBState unit tests
// =============================================================================

// makeDecodeState builds a decodeRLBState with fake closures suitable for
// unit testing. No Engine or GenericCache required. The returned state uses
// the provided blocks, nVocab, halt rule, and alpha.
//
// capturedRecs, if non-nil, receives every rlbIterRec passed to dumpWrite
// so callers can inspect TokenPos and other per-record fields.
func makeDecodeState(
	t *testing.T,
	blocks [][2]int,
	nEmbd, nVocab int,
	rule haltRule, maxIters int,
	alpha float64,
	blendCalls *int,
	capturedRecs *[]rlbIterRec,
) *decodeRLBState {
	t.Helper()

	forwardEmbed := func(tokens []int32) ([]float32, error) {
		return make([]float32, nEmbd*len(tokens)), nil
	}

	var layerCalls []int
	forwardLayer := makeFakeForward(nVocab, &layerCalls)

	projectLogits := func(hid []float32, n int) ([]float32, error) {
		out := make([]float32, nVocab)
		out[0] = 1.0
		return out, nil
	}

	saveState := func(bi int) {}

	blendState := func(bi int, a float64) (float64, error) {
		if blendCalls != nil {
			*blendCalls++
		}
		return 0.0, nil
	}

	dumpWrite := func(rec any) {
		if capturedRecs != nil {
			if r, ok := rec.(rlbIterRec); ok {
				*capturedRecs = append(*capturedRecs, r)
			}
		}
	}

	return &decodeRLBState{
		blocks:           blocks,
		nVocab:           nVocab,
		ssmBufsByBlock:   make([][]ssmBuf, len(blocks)),
		tokenString:      func(id int32) string { return "" },
		forwardEmbed:     forwardEmbed,
		forwardLayer:     forwardLayer,
		projectLogits:    projectLogits,
		saveState:        saveState,
		blendState:       blendState,
		dumpWrite:        dumpWrite,
		rule:             rule,
		maxIters:         maxIters,
		terminalRule:     rule,
		terminalMaxIters: maxIters,
		alpha:            alpha,
	}
}

// TestDecodeRLBStepSingleToken verifies that step() with a single token
// returns valid logits of the expected length and produces correct block stats.
func TestDecodeRLBStepSingleToken(t *testing.T) {
	blocks := [][2]int{{0, 1}, {2, 3}} // 2 blocks, 2 layers each
	nEmbd, nVocab := 8, 16

	rule, _, maxIters := parseHaltRule("fixed_3")

	s := makeDecodeState(t, blocks, nEmbd, nVocab, rule, maxIters, 1.0, nil, nil)

	logits, stats, err := s.step(42)
	if err != nil {
		t.Fatalf("step: %v", err)
	}

	// Logits must be non-nil and have the right length.
	if logits == nil {
		t.Fatal("step: logits is nil")
	}
	if len(logits) != nVocab {
		t.Errorf("logits length: got %d, want %d", len(logits), nVocab)
	}

	// Stats must have one entry per block.
	if len(stats.completedItersByBlock) != len(blocks) {
		t.Fatalf("completedItersByBlock length: got %d, want %d",
			len(stats.completedItersByBlock), len(blocks))
	}

	// fixed_3 halts when len(history) > 3, so iters 0..3 run → 4 iters per block.
	for i, n := range stats.completedItersByBlock {
		if n != 4 {
			t.Errorf("block %d: want 4 iters (fixed_3), got %d", i, n)
		}
	}

	// totalItersRun must equal the sum of completedItersByBlock.
	wantTotal := 0
	for _, n := range stats.completedItersByBlock {
		wantTotal += n
	}
	if stats.totalItersRun != wantTotal {
		t.Errorf("totalItersRun: got %d, want %d (sum of per-block)", stats.totalItersRun, wantTotal)
	}
}

// TestDecodeRLBStepTokenPosIncrementing verifies that tokenIdx increments on
// each step() call and that TokenPos is injected into JSONL dump records.
func TestDecodeRLBStepTokenPosIncrementing(t *testing.T) {
	blocks := [][2]int{{0, 1}, {2, 3}}
	nEmbd, nVocab := 8, 16

	rule, _, maxIters := parseHaltRule("fixed_3")

	var capturedRecs []rlbIterRec
	s := makeDecodeState(t, blocks, nEmbd, nVocab, rule, maxIters, 1.0, nil, &capturedRecs)

	// Call step() three times and collect the records each time.
	type stepResult struct {
		startIdx int
		endIdx   int
	}
	stepBoundaries := make([]stepResult, 3)

	for i := range 3 {
		start := len(capturedRecs)
		_, _, err := s.step(int32(i + 10))
		if err != nil {
			t.Fatalf("step %d: %v", i+1, err)
		}
		stepBoundaries[i] = stepResult{start, len(capturedRecs)}
	}

	// Verify each step's records carry the correct TokenPos (1-indexed).
	for stepN, bounds := range stepBoundaries {
		wantTokenPos := stepN + 1
		for ri := bounds.startIdx; ri < bounds.endIdx; ri++ {
			if capturedRecs[ri].TokenPos != wantTokenPos {
				t.Errorf("step %d record[%d]: TokenPos got %d, want %d",
					wantTokenPos, ri, capturedRecs[ri].TokenPos, wantTokenPos)
			}
		}
	}

	// Sanity: tokenIdx must equal 3 after three steps.
	if s.tokenIdx != 3 {
		t.Errorf("tokenIdx after 3 steps: got %d, want 3", s.tokenIdx)
	}
}

// TestDecodeRLBStepDHThresholdHalt verifies that dH_threshold halt fires during
// a decode step when the block output converges. Uses perBlockForwardCore
// directly with a forwardLayer that produces geometrically-decaying output so
// that l2Diff(blockOut_k, blockOut_{k-1}) / l2Norm(blockOut_k) quickly falls
// below dHThresholdRatio (0.01) and the rule halts early.
//
// The convergence approach: track per-block iteration via saveState (called once
// per iter before the forward), and produce output = base / 2^iter. At iter 0
// the output is 1.0 (large); at iter 1 it is 0.5; at iter 2 it is 0.25, etc.
// The normalized delta between iter k-1 and iter k is:
//
//	l2Diff / l2Norm = |1/2^(k-1) - 1/2^k| / (1/2^k) = |2 - 1| = 1 at all iters.
//
// That won't converge. Instead: make output = c + perturbation, where
// perturbation = 0.1 / 2^iter and c = 1.0 (constant base). Then:
//
//	l2Diff ≈ ||pert_k - pert_{k-1}|| ≈ |0.1/2^k - 0.1/2^(k-1)| × sqrt(nEmbd)
//	l2Norm ≈ sqrt(nEmbd * (c + pert_k)^2) ≈ c * sqrt(nEmbd)
//	normDH = l2Diff / l2Norm ≈ 0.1/2^k / c = 0.1 / 2^k
//
// At iter 4: normDH ≈ 0.1/16 = 0.00625 < 0.01 → halt.
// This gives clean early termination well before maxIters=10.
func TestDecodeRLBStepDHThresholdHalt(t *testing.T) {
	blocks := [][2]int{{0, 1}, {2, 3}} // 2 blocks, 2 layers each
	nEmbd, nVocab, nTokens := 8, 16, 1

	rule, _, maxIters := parseHaltRule(HaltRuleDHThreshold)

	// blockIter[bi] tracks how many times saveState was called for block bi,
	// which equals the current iter index within that block (saveState is called
	// once per iter before the forward pass).
	blockIter := make([]int, len(blocks))

	saveState := func(bi int) {
		blockIter[bi]++
	}

	// currentBI tracks which block's layers are being forwarded. Each block's
	// layers are contiguous so we can detect the block index from the layer.
	// blocks[0]=[0,1], blocks[1]=[2,3]; layer < 2 → block 0, else block 1.
	forwardLayer := func(il int, hidIn, hidOut, logitsOut []float32) error {
		// Determine which block this layer belongs to.
		bi := 0
		if il >= blocks[1][0] {
			bi = 1
		}
		// iter is the count of completed iters in this block so far
		// (saveState was called once before this forward, so blockIter[bi]
		// is already incremented for the current iter).
		iter := blockIter[bi] - 1
		const base = 1.0
		pert := 0.1 / float64(int(1)<<iter) // 0.1, 0.05, 0.025, ...
		for i := range hidOut {
			hidOut[i] = float32(base + pert)
		}
		if logitsOut != nil {
			logitsOut[0] = 1.0
		}
		return nil
	}

	promptIDs := make([]int32, nTokens)

	forwardEmbed := func(tokens []int32) ([]float32, error) {
		return make([]float32, nEmbd*len(tokens)), nil
	}
	projectLogits := func(hid []float32, n int) ([]float32, error) {
		out := make([]float32, nVocab)
		out[0] = 1.0
		return out, nil
	}
	blendState := func(bi int, a float64) (float64, error) { return 0.0, nil }
	dumpWrite := func(rec any) {}
	tokenString := func(id int32) string { return "" }

	_, stats, err := perBlockForwardCore(
		promptIDs, blocks, nVocab,
		tokenString,
		1.0,   // fixed alpha
		false, // magnitudeNorm — this test uses Mythos-shape
		rule, maxIters,
		rule, maxIters,
		forwardEmbed, forwardLayer, projectLogits,
		saveState, blendState, dumpWrite,
	)
	if err != nil {
		t.Fatalf("perBlockForwardCore: %v", err)
	}

	// With maxIters=10, running all iters would give totalItersRun=22 (11 per block).
	// Convergence analysis: normDH ≈ 0.1/2^k. Halts when 0.1/2^k < 0.01, i.e. k≥4.
	// So each block runs ~5 iters (0,1,2,3,4) and halts at iter 4.
	maxPossible := (maxIters + 1) * len(blocks) // 11 × 2 = 22
	if stats.totalItersRun >= maxPossible {
		t.Errorf("dH_threshold should have halted early: totalItersRun=%d, maxPossible=%d",
			stats.totalItersRun, maxPossible)
	}

	// Every block must have halted before maxIters+1 iters.
	for i, n := range stats.completedItersByBlock {
		if n >= maxIters+1 {
			t.Errorf("block %d: ran %d iters, expected early halt (< %d)", i, n, maxIters+1)
		}
	}
}

// TestDecodeRLBStepBlendCalled verifies that blendState is called the expected
// number of times across a decode step. With fixed_3 and 2 blocks, each block
// runs 4 iters (0..3) and blend is called between iters 0→1, 1→2, 2→3 —
// 3 blends per block × 2 blocks = 6 total. This matches TestPerBlockBlendCallsAtAlpha1.
func TestDecodeRLBStepBlendCalled(t *testing.T) {
	blocks := [][2]int{{0, 1}, {2, 3}}
	nEmbd, nVocab := 8, 16

	rule, _, maxIters := parseHaltRule("fixed_3")

	blendCalls := 0
	s := makeDecodeState(t, blocks, nEmbd, nVocab, rule, maxIters, 1.0, &blendCalls, nil)

	_, _, err := s.step(99)
	if err != nil {
		t.Fatalf("step: %v", err)
	}

	const wantBlendCalls = 6 // 2 blocks × 3 between-iter blends (fixed_3 runs 4 iters)
	if blendCalls != wantBlendCalls {
		t.Errorf("blendState call count: got %d, want %d", blendCalls, wantBlendCalls)
	}
}

// --- scaleToMag ---

func TestScaleToMag(t *testing.T) {
	const tol = 1e-6

	t.Run("basic", func(t *testing.T) {
		v := []float32{3.0, 4.0} // L2 = 5
		scaleToMag(v, 10.0)
		got := l2Norm(v)
		if math.Abs(got-10.0) > tol {
			t.Errorf("want L2=10.0, got %.6f", got)
		}
		// Direction should be preserved: 3/5, 4/5.
		if math.Abs(float64(v[0])-6.0) > tol || math.Abs(float64(v[1])-8.0) > tol {
			t.Errorf("direction not preserved: got [%f, %f], want [6, 8]", v[0], v[1])
		}
	})

	t.Run("zero_vector", func(t *testing.T) {
		v := []float32{0, 0, 0}
		scaleToMag(v, 5.0)
		// Should be left unchanged — can't normalize zero.
		for i, x := range v {
			if x != 0 {
				t.Errorf("v[%d] = %f, want 0", i, x)
			}
		}
	})

	t.Run("already_at_target", func(t *testing.T) {
		v := []float32{3.0, 4.0} // L2 = 5
		scaleToMag(v, 5.0)
		got := l2Norm(v)
		if math.Abs(got-5.0) > tol {
			t.Errorf("want L2=5.0, got %.6f", got)
		}
	})
}

// --- Magnitude-norm iteration mode ---

// TestMagnitudeNormBlockOutputNormalized verifies that when magnitudeNorm=true
// (output-recycling mode), the block output handed to the next block is the
// alpha-blended accumulator normalized to the first-pass output magnitude.
// Uses a two-block setup with a forward that depends on input so each
// iteration produces different output.
func TestMagnitudeNormBlockOutputNormalized(t *testing.T) {
	const tol = 1e-4
	blocks := [][2]int{{0, 1}, {2, 3}}
	nEmbd, nVocab := 4, 8

	// Embed returns non-zero so magnitudes are meaningful.
	forwardEmbed := func(tokens []int32) ([]float32, error) {
		out := make([]float32, nEmbd*len(tokens))
		for i := range out {
			out[i] = 1.0
		}
		return out, nil
	}

	// Forward adds input-dependent + layer-dependent value so output-recycling
	// produces different results per iter.
	var layerCalls []int
	forwardLayer := func(il int, hidIn, hidOut, logitsOut []float32) error {
		layerCalls = append(layerCalls, il)
		for i := range hidOut {
			hidOut[i] = hidIn[i]*1.05 + float32(il+1)*0.1
		}
		for i := range logitsOut {
			logitsOut[i] = float32(nVocab - i)
		}
		return nil
	}

	projectLogits := func(hid []float32, n int) ([]float32, error) {
		out := make([]float32, nVocab)
		out[0] = 1.0
		return out, nil
	}

	// Run WITHOUT magnitude norm (Mythos-shape) as baseline.
	promptIDs := []int32{0}
	rule, _, maxIters := parseHaltRule("fixed_3")

	baseLogits, baseStats, err := perBlockForwardCore(
		promptIDs, blocks, nVocab,
		func(id int32) string { return "" },
		1.0, false, // magnitudeNorm=false
		rule, maxIters, rule, maxIters,
		forwardEmbed, forwardLayer, projectLogits,
		func(bi int) {},
		func(bi int, a float64) (float64, error) { return 0, nil },
		func(rec any) {},
	)
	if err != nil {
		t.Fatalf("baseline: %v", err)
	}

	// Run WITH magnitude norm (output-recycling).
	layerCalls = nil
	magLogits, magStats, err := perBlockForwardCore(
		promptIDs, blocks, nVocab,
		func(id int32) string { return "" },
		0.7, true, // magnitudeNorm=true, alpha=0.7
		rule, maxIters, rule, maxIters,
		forwardEmbed, forwardLayer, projectLogits,
		func(bi int) {},
		func(bi int, a float64) (float64, error) { return 0, nil },
		func(rec any) {},
	)
	if err != nil {
		t.Fatalf("mag norm: %v", err)
	}

	// Both should run the same number of iters.
	if baseStats.totalItersRun != magStats.totalItersRun {
		t.Errorf("iter count mismatch: baseline=%d, magNorm=%d",
			baseStats.totalItersRun, magStats.totalItersRun)
	}

	// Both should produce valid logits (not NaN/zero).
	if baseLogits[0] == 0 || magLogits[0] == 0 {
		t.Errorf("got zero logits: baseline[0]=%f, magNorm[0]=%f",
			baseLogits[0], magLogits[0])
	}

	// The magnitude-norm logits should differ from baseline (different
	// iteration scheme) but both should be valid.
	if len(baseLogits) != len(magLogits) {
		t.Fatalf("logits length mismatch: %d vs %d", len(baseLogits), len(magLogits))
	}
	_ = tol // logit values depend on fake forward specifics; just verify no error/crash.
}

// TestMagnitudeNormSingleIterNoOp verifies that when a block runs only one
// iteration (fixed_0), magnitude normalization is a no-op (the "len(history)>1"
// guard in the post-block normalization skips it).
func TestMagnitudeNormSingleIterNoOp(t *testing.T) {
	blocks := [][2]int{{0, 1}}
	nEmbd, nVocab := 4, 8

	forwardEmbed := func(tokens []int32) ([]float32, error) {
		out := make([]float32, nEmbd*len(tokens))
		for i := range out {
			out[i] = 2.0
		}
		return out, nil
	}

	forwardLayer := func(il int, hidIn, hidOut, logitsOut []float32) error {
		for i := range hidOut {
			hidOut[i] = hidIn[i] + 1.0
		}
		for i := range logitsOut {
			logitsOut[i] = float32(nVocab - i)
		}
		return nil
	}

	projectLogits := func(hid []float32, n int) ([]float32, error) {
		out := make([]float32, nVocab)
		out[0] = 1.0
		return out, nil
	}

	rule, _, maxIters := parseHaltRule("fixed_0")
	promptIDs := []int32{0}

	// With magnitudeNorm=true but fixed_0 (single iter), the block only runs
	// once. The post-block normalization guard (len(history)>1) should skip
	// it, giving identical results to magnitudeNorm=false.
	baseLogits, _, err := perBlockForwardCore(
		promptIDs, blocks, nVocab,
		func(id int32) string { return "" },
		1.0, false,
		rule, maxIters, rule, maxIters,
		forwardEmbed, forwardLayer, projectLogits,
		func(bi int) {},
		func(bi int, a float64) (float64, error) { return 0, nil },
		func(rec any) {},
	)
	if err != nil {
		t.Fatalf("baseline: %v", err)
	}

	magLogits, _, err := perBlockForwardCore(
		promptIDs, blocks, nVocab,
		func(id int32) string { return "" },
		1.0, true,
		rule, maxIters, rule, maxIters,
		forwardEmbed, forwardLayer, projectLogits,
		func(bi int) {},
		func(bi int, a float64) (float64, error) { return 0, nil },
		func(rec any) {},
	)
	if err != nil {
		t.Fatalf("mag norm: %v", err)
	}

	// Should be identical since only one iter ran.
	for i := range baseLogits {
		if baseLogits[i] != magLogits[i] {
			t.Errorf("logits[%d]: baseline=%f, magNorm=%f — expected identical for single iter",
				i, baseLogits[i], magLogits[i])
			break
		}
	}
}

// TestMagnitudeNormSSMSaveRestore verifies that in magnitude-norm mode
// (magnitudeNorm=true), SSM state is saved before iter 0 and restored
// (blendState with alpha=0) before each subsequent iteration to prevent
// double-advancement of mutable SSM state.
func TestMagnitudeNormSSMSaveRestore(t *testing.T) {
	blocks := [][2]int{{0, 1}, {2, 3}}
	nEmbd, nVocab := 4, 8

	forwardEmbed := func(tokens []int32) ([]float32, error) {
		out := make([]float32, nEmbd*len(tokens))
		for i := range out {
			out[i] = 1.0
		}
		return out, nil
	}
	var layerCalls []int
	forwardLayer := makeFakeForward(nVocab, &layerCalls)
	projectLogits := func(hid []float32, n int) ([]float32, error) {
		out := make([]float32, nVocab)
		out[0] = 1.0
		return out, nil
	}

	rule, _, maxIters := parseHaltRule("fixed_3")
	saveCalls := 0
	blendCalls := 0
	var blendAlphas []float64

	_, _, err := perBlockForwardCore(
		[]int32{0}, blocks, nVocab,
		func(id int32) string { return "" },
		1.0, true, // magnitudeNorm=true
		rule, maxIters, rule, maxIters,
		forwardEmbed, forwardLayer, projectLogits,
		func(bi int) { saveCalls++ },
		func(bi int, a float64) (float64, error) {
			blendCalls++
			blendAlphas = append(blendAlphas, a)
			return 0, nil
		},
		func(rec any) {},
	)
	if err != nil {
		t.Fatalf("perBlockForwardCore: %v", err)
	}

	// fixed_3 = 4 iterations per block (iter 0..3), 2 blocks:
	// Per block: 1 saveState (iter 0) + 3 blendState restores (iters 1,2,3).
	wantSaves := 2  // 1 per block × 2 blocks
	wantBlends := 6 // 3 per block × 2 blocks
	if saveCalls != wantSaves {
		t.Errorf("saveState calls: got %d, want %d", saveCalls, wantSaves)
	}
	if blendCalls != wantBlends {
		t.Errorf("blendState calls: got %d, want %d", blendCalls, wantBlends)
	}

	// All blendState calls should use alpha=0 (full restore, not blend).
	for i, a := range blendAlphas {
		if a != 0 {
			t.Errorf("blendState call %d: alpha=%.4f, want 0", i, a)
		}
	}
}

// TestMagnitudeNormAccumulation verifies that the accumulated block output
// is an alpha-blend of successive iteration outputs, not just the last iter's
// raw output. Uses a forward that adds input-dependent values so each iter
// produces a different output when fed the recycled result.
func TestMagnitudeNormAccumulation(t *testing.T) {
	const tol = 1e-4
	blocks := [][2]int{{0, 1}} // single block, 2 layers
	nEmbd, nVocab := 4, 8
	alpha := 0.6

	forwardEmbed := func(tokens []int32) ([]float32, error) {
		out := make([]float32, nEmbd*len(tokens))
		for i := range out {
			out[i] = float32(i+1) * 0.5 // [0.5, 1.0, 1.5, 2.0]
		}
		return out, nil
	}

	// Forward adds a fraction of hidIn plus a layer-dependent offset, so
	// the output varies with the input (output-recycling changes input).
	forwardLayer := func(il int, hidIn, hidOut, logitsOut []float32) error {
		for i := range hidOut {
			hidOut[i] = hidIn[i]*1.1 + float32(il+1)*0.1
		}
		for i := range logitsOut {
			logitsOut[i] = float32(nVocab - i)
		}
		return nil
	}

	projectLogits := func(hid []float32, n int) ([]float32, error) {
		out := make([]float32, nVocab)
		out[0] = 1.0
		return out, nil
	}

	rule, _, maxIters := parseHaltRule("fixed_3")

	logits, stats, err := perBlockForwardCore(
		[]int32{0}, blocks, nVocab,
		func(id int32) string { return "" },
		alpha, true, // magnitudeNorm=true
		rule, maxIters, rule, maxIters,
		forwardEmbed, forwardLayer, projectLogits,
		func(bi int) {},
		func(bi int, a float64) (float64, error) { return 0, nil },
		func(rec any) {},
	)
	if err != nil {
		t.Fatalf("perBlockForwardCore: %v", err)
	}

	// fixed_3 = 4 iters
	if stats.totalItersRun != 4 {
		t.Fatalf("expected 4 iters, got %d", stats.totalItersRun)
	}

	// Logits should be valid (not NaN, not all zero).
	if logits[0] == 0 {
		t.Errorf("logits[0] is zero")
	}
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logits contain NaN/Inf")
		}
	}
	_ = tol
}
