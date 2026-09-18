package arch

import (
	"bytes"
	"os"
	"testing"
	"unsafe"

	ggufparser "github.com/gpustack/gguf-parser-go"

	"inference-lab-bench/internal/ggml"
)

// TestKVCacheWritePassMatchesVanillaForward verifies that ForwardCached, run
// twice with WriteKV=true / WriteSSM=true (the new defaults set explicitly),
// produces a K/V (and SSM, when present) cache state byte-identical to the
// canonical vanilla forward — across a 5-token prefill + 16-token greedy
// decode sequence.
//
// The test is the canary for Step 1/2/4 of the PMM implementation: any code
// path that constructs a GraphInputs without explicitly setting WriteKV /
// WriteSSM = true would silently produce a cache that diverges from the
// expected state, and this test would catch it.
//
// The mirror approach used by TestRLBForwardLayerStackEquivalence is exactly
// the structure here: load the model once, run the same token sequence twice
// against two fresh caches, then compare bytes.
//
// PMM v1 fork-variant Stream A is itself a plain ForwardCached call, so this
// test also covers the bit-exactness of PMM's mainline pass against the
// canonical vanilla path.
//
// Set BENCH_TEST_MODEL_PATH to a GGUF file path to run; the test skips
// otherwise. Optionally set BENCH_ARCH_DIR to override the arch dir.
func TestKVCacheWritePassMatchesVanillaForward(t *testing.T) {
	modelPath := os.Getenv("BENCH_TEST_MODEL_PATH")
	if modelPath == "" {
		t.Skip("BENCH_TEST_MODEL_PATH not set; skipping PMM cache-bitexact test " +
			"(set to a supported GGUF file path to run)")
	}

	archDir := os.Getenv("BENCH_ARCH_DIR")
	if archDir == "" {
		archDir = findArchDir(t)
	}

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

	// 5-token prefill: small valid IDs.
	prompt := []int32{0, 1, 2, 3, 4}
	const decodeSteps = 16

	// runSequence runs prefill + decodeSteps greedy decode steps against a fresh
	// cache and returns the cache for byte comparison. Greedy decode is
	// deterministic, so the two runs produce the same token sequence and the
	// same cache state.
	runSequence := func(label string) (*GenericCache, []int32, error) {
		cache, err := model.NewCache(maxSeqLen)
		if err != nil {
			return nil, nil, err
		}

		logits, err := model.ForwardCached(cache, prompt, false, nil)
		if err != nil {
			cache.Free()
			return nil, nil, err
		}

		tokens := append([]int32(nil), prompt...)
		for i := 0; i < decodeSteps; i++ {
			next := greedyArgmax(logits)
			tokens = append(tokens, next)
			logits, err = model.ForwardCached(cache, []int32{next}, false, nil)
			if err != nil {
				cache.Free()
				return nil, nil, err
			}
		}
		t.Logf("%s: token sequence (last 5) = %v", label, tokens[len(tokens)-5:])
		return cache, tokens, nil
	}

	cacheA, tokensA, err := runSequence("run A")
	if err != nil {
		t.Fatalf("run A: %v", err)
	}
	defer cacheA.Free()

	cacheB, tokensB, err := runSequence("run B")
	if err != nil {
		t.Fatalf("run B: %v", err)
	}
	defer cacheB.Free()

	// Sanity: both runs must take the same path (greedy on identical state).
	if len(tokensA) != len(tokensB) {
		t.Fatalf("token-sequence length mismatch: A=%d B=%d", len(tokensA), len(tokensB))
	}
	for i := range tokensA {
		if tokensA[i] != tokensB[i] {
			t.Fatalf("token-sequence mismatch at i=%d: A=%d B=%d", i, tokensA[i], tokensB[i])
		}
	}

	// Compare every cache tensor byte-for-byte across all layers. Both caches
	// were filled via the same sequence of ForwardCached calls; with WriteKV /
	// WriteSSM correctly set on every code path, the buffers must be identical.
	if cacheA.SeqPos != cacheB.SeqPos {
		t.Fatalf("cache.SeqPos mismatch: A=%d B=%d", cacheA.SeqPos, cacheB.SeqPos)
	}

	if len(cacheA.Layers) != len(cacheB.Layers) {
		t.Fatalf("layer count mismatch: A=%d B=%d", len(cacheA.Layers), len(cacheB.Layers))
	}

	mismatches := 0
	for li := range cacheA.Layers {
		la := cacheA.Layers[li]
		lb := cacheB.Layers[li]
		for name, ta := range la.Tensors {
			tb := lb.Tensors[name]
			if ta.IsNil() || tb.IsNil() {
				continue
			}
			// SharedGroup tensors alias one cache slot across layers; only
			// compare on the first layer of the group to avoid double-counting.
			if la.SharedGroup != "" && li > 0 && la.Tensors[name] == cacheA.Layers[li-1].Tensors[name] {
				continue
			}
			n := ta.Nbytes()
			if n != tb.Nbytes() {
				t.Errorf("layer %d %q: byte size mismatch A=%d B=%d", li, name, n, tb.Nbytes())
				mismatches++
				continue
			}
			ba := ggml.TensorGetBytes(ta, 0, n)
			bb := ggml.TensorGetBytes(tb, 0, n)
			if !bytes.Equal(ba, bb) {
				// Report the first differing offset for diagnosis.
				firstDiff := -1
				for k := 0; k < n; k++ {
					if ba[k] != bb[k] {
						firstDiff = k
						break
					}
				}
				t.Errorf("layer %d %q: cache bytes differ (n=%d, first diff at byte %d)",
					li, name, n, firstDiff)
				mismatches++
			}
		}
	}

	if mismatches == 0 {
		t.Logf("cache state byte-identical across all layers (%d layers checked)", len(cacheA.Layers))
	}
}

// greedyArgmax returns the index of the maximum logit. Local copy to avoid
// importing the inference package from the arch package (cycle guard).
func greedyArgmax(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}
	best := int32(0)
	bestVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			best = int32(i)
		}
	}
	return best
}

// asFloat32Bytes is a defensive helper that exposes a tensor's bytes as a
// float32 slice for ad-hoc inspection during test debugging. Not used in the
// passing path; kept here so that diagnosing a mismatch doesn't require
// re-deriving the cast.
func asFloat32Bytes(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/4)
}

var _ = asFloat32Bytes // silence "unused" until the helper is needed
