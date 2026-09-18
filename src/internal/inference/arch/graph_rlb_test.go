package arch

import (
	"os"
	"testing"

	ggufparser "github.com/gpustack/gguf-parser-go"

	"inference-lab-bench/internal/ggml"
)

// TestRLBForwardLayerStackEquivalence verifies that the per-layer RLB forward
// helpers (RLBForwardEmbed → RLBForwardLayer × nLayers → RLBProjectLogits)
// produce logits that are byte-identical (or within FP rounding tolerance) to
// a single monolithic ForwardCached call over the same prompt.
//
// This is the critical correctness gate for the per-block RLB rewrite: if these
// two paths diverge, the per-block driver built on top of the helpers is
// structurally unsound.
//
// Set BENCH_TEST_MODEL_PATH to a GGUF file path to run. The model's arch name
// is read from GGUF metadata — any supported arch works; Qwen 3.5 is the
// primary target.
//
// Optionally set BENCH_ARCH_DIR to override the arch definition directory.
// Default: walks up from the test file location to find models/arch/.
func TestRLBForwardLayerStackEquivalence(t *testing.T) {
	modelPath := os.Getenv("BENCH_TEST_MODEL_PATH")
	if modelPath == "" {
		t.Skip("BENCH_TEST_MODEL_PATH not set; skipping RLB forward-path equivalence test " +
			"(set to a supported GGUF file path to run)")
	}

	archDir := os.Getenv("BENCH_ARCH_DIR")
	if archDir == "" {
		archDir = findArchDir(t)
	}

	// Parse GGUF to extract arch name and reuse the parsed file for model load.
	gf, err := ggufparser.ParseGGUFFile(modelPath)
	if err != nil {
		t.Fatalf("parsing GGUF: %v", err)
	}
	archName := gf.Architecture().Architecture
	if archName == "" {
		t.Fatalf("GGUF metadata has no general.architecture field")
	}
	t.Logf("arch=%s path=%s", archName, modelPath)

	archDef, err := Load(archDir, archName)
	if err != nil {
		t.Fatalf("loading arch def %q: %v", archName, err)
	}

	// Obtain real memory stats so checkMemory passes.
	gpu := ggml.GPUInit()
	if gpu == nil {
		t.Fatal("ggml.GPUInit() returned nil")
	}
	defer gpu.Free()
	cpu := ggml.CPUInit()
	defer cpu.Free()
	memStats := ggml.DevMemory(gpu, cpu)

	const maxSeqLen = 256
	model, err := NewGenericModelFromGGUF(memStats, maxSeqLen, archDef, modelPath, archDir, gf, "")
	if err != nil {
		t.Fatalf("loading model: %v", err)
	}
	defer model.Close()

	// Five token IDs small enough to be valid in any vocabulary.
	promptIDs := []int32{0, 1, 2, 3, 4}
	nTokens := len(promptIDs)
	nEmbd := model.Params.Ints[ParamNEmbd]
	nLayers := model.Params.Ints[ParamNLayers]
	t.Logf("n_layers=%d n_embd=%d n_vocab=%d", nLayers, nEmbd, model.Params.Ints[ParamNVocab])

	// --- Main-path run ---
	cache1, err := model.NewCache(maxSeqLen)
	if err != nil {
		t.Fatalf("NewCache (main): %v", err)
	}
	defer cache1.Free()

	// flashAttn=false for determinism; both paths must match this setting.
	stackLogitsRaw, err := model.ForwardCached(cache1, promptIDs, false, nil)
	if err != nil {
		t.Fatalf("ForwardCached: %v", err)
	}
	// MUST copy: readLogitsInto aliases an internal buffer reused on the next call.
	stackLogits := append([]float32(nil), stackLogitsRaw...)

	// --- RLB-path run ---
	cache2, err := model.NewCache(maxSeqLen)
	if err != nil {
		t.Fatalf("NewCache (rlb): %v", err)
	}
	defer cache2.Free()

	hidA, err := model.RLBForwardEmbed(promptIDs)
	if err != nil {
		t.Fatalf("RLBForwardEmbed: %v", err)
	}
	if len(hidA) != nEmbd*nTokens {
		t.Fatalf("RLBForwardEmbed returned %d floats, want %d (n_embd=%d * nTokens=%d)",
			len(hidA), nEmbd*nTokens, nEmbd, nTokens)
	}

	// Ping-pong hidden state through all layers.
	hidB := make([]float32, len(hidA))

	// Token positions for a prefill starting at seqPos=0.
	positions := make([]int32, nTokens)
	for i := range positions {
		positions[i] = int32(i)
	}

	in, out := hidA, hidB
	for il := 0; il < nLayers; il++ {
		if err := model.RLBForwardLayer(cache2, il, in, positions, false, true, true, out, nil); err != nil {
			t.Fatalf("RLBForwardLayer il=%d: %v", il, err)
		}
		in, out = out, in
	}
	// `in` now holds the final post-stack hidden state.

	rlbLogits, err := model.RLBProjectLogits(in, nTokens)
	if err != nil {
		t.Fatalf("RLBProjectLogits: %v", err)
	}

	// --- Compare ---
	if len(rlbLogits) != len(stackLogits) {
		t.Fatalf("logits length mismatch: stack=%d rlb=%d", len(stackLogits), len(rlbLogits))
	}

	diffCount := 0
	maxAbsDiff := float32(0)
	firstDiffIdx := -1
	for i := range stackLogits {
		d := stackLogits[i] - rlbLogits[i]
		if d != 0 {
			if firstDiffIdx < 0 {
				firstDiffIdx = i
			}
			diffCount++
			if d < 0 {
				d = -d
			}
			if d > maxAbsDiff {
				maxAbsDiff = d
			}
		}
	}

	if diffCount == 0 {
		t.Logf("logits are byte-identical (%d elements)", len(stackLogits))
		return
	}

	// Non-zero diff: tolerate up to 1e-5 max abs diff (FP reordering on GPU
	// is a known source of non-associativity). Report but do not fail if within
	// tolerance; fail loudly if exceeded so the delta is surfaced.
	const tol = float32(1e-5)
	if maxAbsDiff > tol {
		t.Errorf("logits diverge beyond tolerance: %d/%d positions differ, "+
			"max abs diff=%g (tol=%g), first diverging idx=%d (stack=%g rlb=%g)",
			diffCount, len(stackLogits), maxAbsDiff, tol,
			firstDiffIdx, stackLogits[firstDiffIdx], rlbLogits[firstDiffIdx])
	} else {
		t.Logf("logits differ but within tolerance: %d/%d positions, max abs diff=%g (tol=%g)",
			diffCount, len(stackLogits), maxAbsDiff, tol)
	}
}
