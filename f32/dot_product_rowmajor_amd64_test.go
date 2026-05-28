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

func TestDotProductRowMajorAMD64FallbackUsesDotProductForValidRows(t *testing.T) {
	const dims = 64
	base := deterministicF32Vector(701, 3*dims)
	query := deterministicF32Vector(702, dims)
	savedDotProductImpl := dotProductImpl
	defer func() { dotProductImpl = savedDotProductImpl }()

	t.Run("indexed", func(t *testing.T) {
		calls := 0
		dotProductImpl = func(a, b []float32) float32 {
			calls++
			if len(a) != dims || len(b) != dims {
				t.Fatalf("dotProduct len(a)=%d len(b)=%d, want %d", len(a), len(b), dims)
			}
			return float32(100 + calls)
		}

		rowIDs := []uint32{0, 99, 2}
		got := make([]float32, len(rowIDs))
		if DotProductIndexed(got, base, query, rowIDs, dims) {
			t.Fatalf("indexed rows<4 unexpectedly reported optimized")
		}
		if calls != 2 {
			t.Fatalf("indexed fallback dotProduct calls=%d, want 2", calls)
		}
		assertCloseSlice(t, got, []float32{101, 0, 102})
	})

	t.Run("strided", func(t *testing.T) {
		calls := 0
		dotProductImpl = func(a, b []float32) float32 {
			calls++
			if len(a) != dims || len(b) != dims {
				t.Fatalf("dotProduct len(a)=%d len(b)=%d, want %d", len(a), len(b), dims)
			}
			return float32(200 + calls)
		}

		got := make([]float32, 3)
		if DotProductStrided(got, base[:2*dims], query, 3, dims, dims) {
			t.Fatalf("strided rows<4 unexpectedly reported optimized")
		}
		if calls != 2 {
			t.Fatalf("strided fallback dotProduct calls=%d, want 2", calls)
		}
		assertCloseSlice(t, got, []float32{201, 202, 0})
	})
}
