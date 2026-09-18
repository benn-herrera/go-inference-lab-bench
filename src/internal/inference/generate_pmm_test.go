package inference

import (
	"testing"
)

// TestPMMParamsApplyDefaults verifies that zero-value fields are populated
// with the documented defaults and that explicitly set non-zero values pass
// through unchanged.
func TestPMMParamsApplyDefaults(t *testing.T) {
	tests := []struct {
		name string
		in   PMMParams
		want PMMParams
	}{
		{
			name: "all zero -> defaults",
			in:   PMMParams{},
			want: PMMParams{
				SlabFirst:  pmmDefaultSlabFirst,
				SlabLast:   pmmDefaultSlabLast,
				Cap:        pmmDefaultCap,
				ThinkCap:   pmmDefaultThinkCap,
				BlendAlpha: pmmDefaultBlendAlpha,
			},
		},
		{
			name: "fully specified passes through",
			in:   PMMParams{SlabFirst: 1, SlabLast: 7, Cap: 5, ThinkCap: 3, ThinkDisabled: true, BlendAlpha: 0.2},
			want: PMMParams{SlabFirst: 1, SlabLast: 7, Cap: 5, ThinkCap: 3, ThinkDisabled: true, BlendAlpha: 0.2},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.in.applyDefaults()
			if got != tt.want {
				t.Errorf("applyDefaults() = %+v; want %+v", got, tt.want)
			}
		})
	}
}

// TestPMMPhaseTracker exercises think→answer phase classification across the
// boundary cases that matter at decode time: contiguous-tag input, tags
// split across token boundaries, initialInThink seeding, and empty tags.
func TestPMMPhaseTracker(t *testing.T) {
	const open = "<think>"
	const close = "</think>"

	// Helper: feed a token sequence and collect the phase reported per token.
	run := func(p *pmmPhaseTracker, toks []string) []pmmPhase {
		out := make([]pmmPhase, len(toks))
		for i, tok := range toks {
			out[i] = p.observe(tok)
		}
		return out
	}

	t.Run("basic think->answer transitions", func(t *testing.T) {
		p := newPMMPhaseTracker(open, close, false)
		toks := []string{"hi ", "<think>", "reason", "</think>", "answer"}
		got := run(&p, toks)
		// "hi " is answer; "<think>" enters think; "reason" is think; "</think>"
		// is the LAST think token (last-think rule); "answer" is answer.
		want := []pmmPhase{pmmPhaseAnswer, pmmPhaseThink, pmmPhaseThink, pmmPhaseThink, pmmPhaseAnswer}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("token[%d]=%q: got %v, want %v", i, toks[i], got[i], want[i])
			}
		}
	})

	t.Run("open tag split across two tokens", func(t *testing.T) {
		p := newPMMPhaseTracker(open, close, false)
		// "<thi" alone is a held-tail prefix — still answer until completion.
		// On ">", the open tag completes and the token transitions us into think.
		toks := []string{"<thi", "nk>", "thought", "</think>", "ans"}
		got := run(&p, toks)
		// "<thi": held tail, no transition yet, pre-token state was answer → answer.
		// "nk>": completes <think>, transitions to think → think.
		// "thought": think.
		// "</think>": last think token.
		// "ans": answer.
		want := []pmmPhase{pmmPhaseAnswer, pmmPhaseThink, pmmPhaseThink, pmmPhaseThink, pmmPhaseAnswer}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("token[%d]=%q: got %v, want %v", i, toks[i], got[i], want[i])
			}
		}
	})

	t.Run("close tag split </thi+nk>", func(t *testing.T) {
		p := newPMMPhaseTracker(open, close, true)
		toks := []string{"thinking", "</thi", "nk>", "the answer"}
		got := run(&p, toks)
		// initialInThink=true → "thinking" think; "</thi" held tail still inThink → think;
		// "nk>" completes close → last think token → think; "the answer" → answer.
		want := []pmmPhase{pmmPhaseThink, pmmPhaseThink, pmmPhaseThink, pmmPhaseAnswer}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("token[%d]=%q: got %v, want %v", i, toks[i], got[i], want[i])
			}
		}
	})

	t.Run("close tag split </th+ink>", func(t *testing.T) {
		p := newPMMPhaseTracker(open, close, true)
		toks := []string{"x", "</th", "ink>", "y"}
		got := run(&p, toks)
		want := []pmmPhase{pmmPhaseThink, pmmPhaseThink, pmmPhaseThink, pmmPhaseAnswer}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("token[%d]=%q: got %v, want %v", i, toks[i], got[i], want[i])
			}
		}
	})

	t.Run("initialInThink classifies first token as think without open tag", func(t *testing.T) {
		p := newPMMPhaseTracker(open, close, true)
		got := p.observe("first")
		if got != pmmPhaseThink {
			t.Errorf("got %v, want pmmPhaseThink", got)
		}
	})

	t.Run("empty open/close: always answer", func(t *testing.T) {
		p := newPMMPhaseTracker("", "", false)
		toks := []string{"<think>", "reason", "</think>", "answer"}
		got := run(&p, toks)
		for i, ph := range got {
			if ph != pmmPhaseAnswer {
				t.Errorf("token[%d]=%q: got %v, want pmmPhaseAnswer", i, toks[i], ph)
			}
		}
	})

	t.Run("empty open/close with initialInThink=true: still answer", func(t *testing.T) {
		// Empty tags mean "no think phase ever" — that overrides the seed.
		p := newPMMPhaseTracker("", "", true)
		got := p.observe("anything")
		if got != pmmPhaseAnswer {
			t.Errorf("got %v, want pmmPhaseAnswer", got)
		}
	})
}

// TestPMMUnifiedHaltRule exercises the slab+cap+dH halt-rule logic in
// isolation. Non-slab blocks must always return halt=true on the first
// invocation (single pass); slab blocks must respect cap; below-cap slab
// invocations delegate to dH_threshold.
func TestPMMUnifiedHaltRule(t *testing.T) {
	dHRule, _, _ := parseHaltRule(HaltRuleDHThreshold)

	// Closure construction mirrors runShadowPass.
	makeRule := func(slabFirst, slabLast, cap int) func(history []rlbIterRec) (bool, float64) {
		return func(history []rlbIterRec) (bool, float64) {
			if len(history) == 0 {
				return true, 1.0
			}
			bi := history[len(history)-1].BlockIdx
			inSlab := bi >= slabFirst && bi <= slabLast
			if !inSlab {
				return true, 1.0
			}
			if len(history) >= cap {
				_, alpha := dHRule(history)
				return true, alpha
			}
			return dHRule(history)
		}
	}

	rule := makeRule(2, 5, 2)

	// Non-slab block: any history length, must halt immediately.
	for _, bi := range []int{0, 1, 6, 7, 8} {
		halt, _ := rule([]rlbIterRec{{BlockIdx: bi, Iter: 0}})
		if !halt {
			t.Errorf("non-slab block bi=%d should halt on first iter", bi)
		}
	}

	// Slab block, iter 0 (history len 1): dH rule fall-through (Iter 0 always
	// returns false from dHRule). Must NOT halt yet.
	halt, _ := rule([]rlbIterRec{{BlockIdx: 3, Iter: 0, BlockOutNorm: 1.0}})
	if halt {
		t.Errorf("slab block at iter 0 should not halt (dH rule fall-through)")
	}

	// Slab block, history reaches cap (len == 2): force halt regardless of dH.
	history := []rlbIterRec{
		{BlockIdx: 3, Iter: 0, BlockOutNorm: 1.0, DeltaHiddenNorm: 0.5},
		{BlockIdx: 3, Iter: 1, BlockOutNorm: 1.0, DeltaHiddenNorm: 0.4}, // not converged by dH
	}
	halt, _ = rule(history)
	if !halt {
		t.Errorf("slab block at cap should halt regardless of dH")
	}

	// Slab block, history len < cap, dH below threshold: dH rule says halt.
	history = []rlbIterRec{
		{BlockIdx: 3, Iter: 0, BlockOutNorm: 100.0, DeltaHiddenNorm: 0.0},
		{BlockIdx: 3, Iter: 1, BlockOutNorm: 100.0, DeltaHiddenNorm: 0.001}, // ratio = 1e-5 < threshold
	}
	halt, _ = rule(history)
	if !halt {
		t.Errorf("slab block with dH below threshold should halt")
	}
}
