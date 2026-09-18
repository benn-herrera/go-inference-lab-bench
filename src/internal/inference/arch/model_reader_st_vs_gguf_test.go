package arch

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	ggufparser "github.com/gpustack/gguf-parser-go"
)

// TestSafetensorsMatchesGGUF compares every tensor in the safetensors build
// against the reference GGUF file for the same model. Runs only when the env
// vars are set — otherwise skipped (this is not a self-contained unit test).
//
// Usage:
//
//	ST_VS_GGUF_MODEL=qwen35 \
//	ST_VS_GGUF_ST_DIR=/Users/benn/projects/inference-lab-bench/models/Qwen3.5-9B.st \
//	ST_VS_GGUF_GGUF=/Users/benn/projects/inference-lab-bench/models/Qwen3.5-9B.gguf.bin \
//	ST_VS_GGUF_ARCH_DIR=/Users/benn/projects/inference-lab-bench/models/arch \
//	go test ./src/internal/inference/arch -run TestSafetensorsMatchesGGUF -v
func TestSafetensorsMatchesGGUF(t *testing.T) {
	archName := os.Getenv("ST_VS_GGUF_MODEL")
	stDir := os.Getenv("ST_VS_GGUF_ST_DIR")
	ggufPath := os.Getenv("ST_VS_GGUF_GGUF")
	archDir := os.Getenv("ST_VS_GGUF_ARCH_DIR")

	if archName == "" || stDir == "" || ggufPath == "" || archDir == "" {
		t.Skip("ST_VS_GGUF_* env vars not set")
	}

	archDef, err := Load(archDir, archName)
	if err != nil {
		t.Fatalf("loading arch def: %v", err)
	}

	// ----- safetensors reader -----
	stReader, err := NewModelReaderSafetensors(archDef, stDir, archDir, MmprojEnabled)
	if err != nil {
		t.Fatalf("opening safetensors reader: %v", err)
	}
	defer stReader.Close()

	// ----- GGUF reader -----
	gf, err := ggufparser.ParseGGUFFile(ggufPath)
	if err != nil {
		t.Fatalf("parsing GGUF: %v", err)
	}
	ggufReader, err := NewModelReaderGGUF(archDef, ggufPath, gf, "")
	if err != nil {
		t.Fatalf("opening GGUF reader: %v", err)
	}
	defer ggufReader.Close()

	// ----- Collect tensor names from both -----
	stNames := stReader.TensorNames()
	ggufNamesAll := ggufReader.TensorNames()
	sort.Strings(ggufNamesAll)

	stSet := make(map[string]struct{}, len(stNames))
	for _, n := range stNames {
		stSet[n] = struct{}{}
	}
	ggufSet := make(map[string]struct{}, len(ggufNamesAll))
	for _, n := range ggufNamesAll {
		ggufSet[n] = struct{}{}
	}

	// Report set differences
	var onlyInST []string
	for _, n := range stNames {
		if _, ok := ggufSet[n]; !ok {
			onlyInST = append(onlyInST, n)
		}
	}
	var onlyInGGUF []string
	for _, n := range ggufNamesAll {
		if _, ok := stSet[n]; !ok {
			onlyInGGUF = append(onlyInGGUF, n)
		}
	}
	if len(onlyInST) > 0 {
		t.Errorf("%d tensors only in safetensors: %v", len(onlyInST), onlyInST[:min(10, len(onlyInST))])
	}
	if len(onlyInGGUF) > 0 {
		t.Errorf("%d tensors only in GGUF: %v", len(onlyInGGUF), onlyInGGUF[:min(10, len(onlyInGGUF))])
	}

	// ----- Compare all common tensors -----
	mismatchCount := 0
	shapeMismatch := 0
	sizeMismatch := 0
	typeMismatch := 0
	byteMismatch := 0
	var firstMismatchName string
	var firstMismatchDiff string

	for _, name := range stNames {
		if _, ok := ggufSet[name]; !ok {
			continue
		}
		stSpec, _ := stReader.TensorSpec(name)
		ggufSpec, _ := ggufReader.TensorSpec(name)

		mismatch := false

		if stSpec.Type != ggufSpec.Type {
			mismatch = true
			typeMismatch++
			if firstMismatchName == "" {
				firstMismatchName = name
				firstMismatchDiff = "type"
			}
		}
		if stSpec.Ne != ggufSpec.Ne {
			mismatch = true
			shapeMismatch++
			if firstMismatchName == "" {
				firstMismatchName = name
				firstMismatchDiff = "shape"
			}
		}
		if stSpec.Size != ggufSpec.Size {
			mismatch = true
			sizeMismatch++
			if firstMismatchName == "" {
				firstMismatchName = name
				firstMismatchDiff = "size"
			}
		}

		if !mismatch {
			// Compare raw bytes
			stBuf := make([]byte, stSpec.Size)
			if err := stReader.ReadTensor(name, stBuf); err != nil {
				t.Errorf("%s: st read err: %v", name, err)
				continue
			}
			ggBuf := make([]byte, ggufSpec.Size)
			if err := ggufReader.ReadTensor(name, ggBuf); err != nil {
				t.Errorf("%s: gguf read err: %v", name, err)
				continue
			}
			if !bytes.Equal(stBuf, ggBuf) {
				mismatch = true
				byteMismatch++
				if firstMismatchName == "" {
					firstMismatchName = name
					firstMismatchDiff = "bytes"
				}
				// Diagnostic: for first few mismatches, show value-level diff
				if mismatchCount < 3 && stSpec.Type == 0 && stSpec.Size <= 256 {
					st32 := decodeF32(stBuf)
					gg32 := decodeF32(ggBuf)
					t.Logf("  ST[%s]   = %v", name, st32)
					t.Logf("  GGUF[%s] = %v", name, gg32)
					// Diff positions
					var diffs []int
					for i := range st32 {
						if st32[i] != gg32[i] {
							diffs = append(diffs, i)
						}
					}
					t.Logf("  diff indices: %v (count=%d)", diffs, len(diffs))
				}
			}
		}

		if mismatch {
			mismatchCount++
			t.Logf("MISMATCH %s: st{type=%d ne=%v size=%d} gguf{type=%d ne=%v size=%d}",
				name,
				stSpec.Type, stSpec.Ne, stSpec.Size,
				ggufSpec.Type, ggufSpec.Ne, ggufSpec.Size)
		}
	}

	t.Logf("compared %d tensors, %d mismatches (%d type, %d shape, %d size, %d bytes)",
		len(stNames), mismatchCount, typeMismatch, shapeMismatch, sizeMismatch, byteMismatch)
	if firstMismatchName != "" {
		t.Logf("first mismatch: %s (%s)", firstMismatchName, firstMismatchDiff)
	}

	if mismatchCount > 0 {
		t.Errorf("found %d mismatches", mismatchCount)
	}
}

// decodeF32 decodes a little-endian F32 byte slice into a slice of float32.
func decodeF32(buf []byte) []float32 {
	n := len(buf) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(buf[i*4]) | uint32(buf[i*4+1])<<8 |
			uint32(buf[i*4+2])<<16 | uint32(buf[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

// quickCheck: called with both files open, print dimensions of first N tensors
// to help debug.
func listTensorDims(t *testing.T, r ModelReader, label string, limit int) {
	names := r.TensorNames()
	sort.Strings(names)
	if limit > len(names) {
		limit = len(names)
	}
	for _, n := range names[:limit] {
		spec, _ := r.TensorSpec(n)
		t.Logf("[%s] %s type=%d ne=%v size=%d", label, n, spec.Type, spec.Ne, spec.Size)
	}
}

// min is a helper since Go's builtin min requires 1.21+.
// (It's available on most modern Go versions; this file defines a local
// version just in case.)
func minIntDuplicate(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ensure filepath stays imported
var _ = filepath.Join
