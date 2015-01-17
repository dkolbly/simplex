package simplex

import (
	"math"
	"math/rand"
	"testing"
)

func TestSimplex2(t *testing.T) {
	r := rand.New(rand.NewSource(101))
	n := New(r)
	a := n.Noise2(0, 1.25)
	a0 := -0.2248
	if math.Abs(a-a0) > 0.00001 {
		t.Errorf("Got %.4f, expected %.4f", a, a0)
	}

	// make sure we get a different answer with a different seed
	n = New(rand.New(rand.NewSource(102)))
	a2 := n.Noise2(0, 1.25)
	if math.Abs(a-a2) < 0.0001 {
		t.Errorf("got %.4f and %.4f with different seeds", a, a2)
	}

	var minValue, maxValue float64

	for i := 0; i < 1000000; i++ {
		x := r.Float64()
		y := r.Float64()

		a := n.Noise2(x, y)
		if i == 0 {
			minValue = a
			maxValue = a
		} else {
			if a < minValue {
				minValue = a
			}
			if a > maxValue {
				maxValue = a
			}
		}
	}
	if minValue < -1 {
		t.Errorf("got min value %.4f, expected no less than -1", minValue)
	}
	if maxValue > 1 {
		t.Errorf("got max value %.4f, expected no more than 1", maxValue)
	}
}

func TestSimplex3(t *testing.T) {
	r := rand.New(rand.NewSource(101))
	n := New(r)
	var minValue, maxValue float64

	for i := 0; i < 1000000; i++ {
		x := r.Float64()
		y := r.Float64()
		z := r.Float64()

		a := n.Noise3(x, y, z)
		if i == 0 {
			minValue = a
			maxValue = a
		} else {
			if a < minValue {
				minValue = a
			}
			if a > maxValue {
				maxValue = a
			}
		}
	}
	if minValue < -1 {
		t.Errorf("got min value %.4f, expected no less than -1", minValue)
	}
	if maxValue > 1 {
		t.Errorf("got max value %.4f, expected no more than 1", maxValue)
	}
}

func TestSimplex4(t *testing.T) {
	r := rand.New(rand.NewSource(101))
	n := New(r)
	var minValue, maxValue float64

	for i := 0; i < 1000000; i++ {
		x := r.Float64()
		y := r.Float64()
		z := r.Float64()
		w := r.Float64()

		a := n.Noise4(x, y, z, w)
		if i == 0 {
			minValue = a
			maxValue = a
		} else {
			if a < minValue {
				minValue = a
			}
			if a > maxValue {
				maxValue = a
			}
		}
	}
	if minValue < -1 {
		t.Errorf("got min value %.4f, expected no less than -1", minValue)
	}
	if maxValue > 1 {
		t.Errorf("got max value %.4f, expected no more than 1", maxValue)
	}
}

// on my machine (charon) we get about 145 ns/op
func BenchmarkSimplex(b *testing.B) {
	r := rand.New(rand.NewSource(101))
	n := New(r)

	x := 0.001
	y := 0.0001
	for i := 0; i < b.N; i++ {
		n.Noise2(x, y)
		x += 0.00000011
		y += 0.00000012
	}
}
