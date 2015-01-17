package simplex

import (
	"testing"
	"math"
	"math/rand"
)

func TestSimplex2(t *testing.T) {
	r := rand.New(rand.NewSource(101))
	n := New(r)
	a := n.Noise2(0, 1.25)
	a0 := -0.2248
	if math.Abs(a - a0) > 0.00001 {
		t.Errorf("Got %.4f, expected %.4f", a, a0)
	}

	// make sure we get a different answer with a different seed
	n = New(rand.New(rand.NewSource(102)))
	a2 := n.Noise2(0, 1.25)
	if math.Abs(a - a2) < 0.0001 {
		t.Errorf("got %.4f and %.4f with different seeds", a, a2)
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
