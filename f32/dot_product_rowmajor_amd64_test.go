//go:build amd64

package f32

import "testing"

func TestDotProductRowMajorAMD64ThresholdPredicates(t *testing.T) {
	indexedEnabled := []struct {
		rows int
		dims int
	}{
		{4, 64}, {256, 64}, {4, 128}, {256, 128}, {64, 768}, {32, 2048},
	}
	for _, tc := range indexedEnabled {
		if !batchDotIndexedSIMDEligible(tc.rows, tc.dims, tc.dims) {
			t.Fatalf("indexed rows=%d dims=%d unexpectedly gated", tc.rows, tc.dims)
		}
	}

	indexedGated := []struct {
		rows int
		dims int
	}{
		{1, 64}, {4, 63}, {256, 768}, {64, 2048}, {256, 2048},
	}
	for _, tc := range indexedGated {
		if batchDotIndexedSIMDEligible(tc.rows, tc.dims, tc.dims) {
			t.Fatalf("indexed rows=%d dims=%d unexpectedly enabled", tc.rows, tc.dims)
		}
	}
	if batchDotIndexedSIMDEligible(4, 64, 63) {
		t.Fatalf("indexed queryLen<dims unexpectedly enabled")
	}

	stridedEnabled := []struct {
		rows   int
		dims   int
		stride int
	}{
		{4, 64, 64}, {256, 64, 80}, {13, 128, 128}, {13, 128, 144},
		{256, 128, 144}, {64, 768, 768}, {13, 768, 784},
		{8, 2048, 2048}, {16, 2048, 2064}, {32, 2048, 2048},
	}
	for _, tc := range stridedEnabled {
		if !batchDotStridedSIMDEligible(tc.rows, tc.dims, tc.stride, tc.dims) {
			t.Fatalf("strided rows=%d dims=%d stride=%d unexpectedly gated", tc.rows, tc.dims, tc.stride)
		}
	}

	stridedGated := []struct {
		rows   int
		dims   int
		stride int
	}{
		{1, 64, 64}, {4, 63, 63}, {4, 128, 128}, {8, 128, 144}, {32, 128, 144},
		{13, 768, 768}, {64, 768, 784}, {256, 768, 768},
		{4, 2048, 2048}, {13, 2048, 2064}, {16, 2048, 2048},
		{64, 2048, 2048}, {64, 2048, 2064}, {256, 2048, 2064},
	}
	for _, tc := range stridedGated {
		if batchDotStridedSIMDEligible(tc.rows, tc.dims, tc.stride, tc.dims) {
			t.Fatalf("strided rows=%d dims=%d stride=%d unexpectedly enabled", tc.rows, tc.dims, tc.stride)
		}
	}
	if batchDotStridedSIMDEligible(4, 64, 0, 64) {
		t.Fatalf("strided stride=0 unexpectedly enabled")
	}
	if batchDotStridedSIMDEligible(4, 64, -1, 64) {
		t.Fatalf("strided stride=-1 unexpectedly enabled")
	}
}

func TestDotProductRowMajorAMD64FalseReturnUsesGoFallbackForTails(t *testing.T) {
	const dims = 64
	base := deterministicF32Vector(701, 3*dims)
	query := deterministicF32Vector(702, dims)

	savedDotProductImpl := dotProductImpl
	dotProductImpl = func(a, b []float32) float32 {
		t.Fatalf("false-return fallback tail used dotProduct dispatcher")
		return 0
	}
	defer func() { dotProductImpl = savedDotProductImpl }()

	indexedRowIDs := []uint32{0, 1, 99, 2}
	indexedGot := make([]float32, len(indexedRowIDs))
	indexedWant := make([]float32, len(indexedRowIDs))
	dotProductIndexedGo(indexedWant, base, query, indexedRowIDs, dims)
	if DotProductIndexed(indexedGot, base, query, indexedRowIDs, dims) {
		t.Fatalf("indexed ragged batch unexpectedly reported optimized")
	}
	assertCloseSlice(t, indexedGot, indexedWant)

	stridedGot := make([]float32, 4)
	stridedWant := make([]float32, 4)
	dotProductStridedGo(stridedWant, base, query, 4, dims, dims)
	if DotProductStrided(stridedGot, base, query, 4, dims, dims) {
		t.Fatalf("strided ragged batch unexpectedly reported optimized")
	}
	assertCloseSlice(t, stridedGot, stridedWant)
}
