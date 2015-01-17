// Package simplex implements simplex noise
package simplex

/*
 *  Based on the Java implementation of SimplexNoise
 *  by Stefan Gustavson
 *
 *  obtained from
 *  http://webstaff.itn.liu.se/~stegu/simplexnoise/SimplexNoise.java
 *  on 2014-02-16
 *
 *  and described in
 *  http://staffwww.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
 */

// this is the original comment from the Java code:

/*
 * A speed-improved simplex noise algorithm for 2D, 3D and 4D in Java.
 *
 * Based on example code by Stefan Gustavson (stegu@itn.liu.se).
 * Optimisations by Peter Eastman (peastman@drizzle.stanford.edu).
 * Better rank ordering method by Stefan Gustavson in 2012.
 *
 * This could be speeded up even further, but it's useful as it is.
 *
 * Version 2012-03-09
 *
 * This code was placed in the public domain by its original author,
 * Stefan Gustavson. You may use it as you see fit, but
 * attribution is appreciated.
 *
 */

import (
	"math"
	"math/rand"
)

type Simplex struct {
	// this is a permutation of the numbers 0-255
	mix [256]uint8
}

func New(r *rand.Rand) *Simplex {
	s := &Simplex{}
	// initialize it
	for i := 0; i < 256; i++ {
		s.mix[i] = uint8(i)
	}
	// now randomize the permutation
	for i := 0; i < 255; i++ {
		j := r.Int31() & 0xFF
		if int(j) > i {
			s.mix[i], s.mix[j] = s.mix[j], s.mix[i]
		}
	}
	return s
}

type grad2 struct {
	dx, dy float64
}

type grad3 struct {
	dx, dy, dz float64
}

type grad4 struct {
	dx, dy, dz, dw float64
}

func (g grad3) dot(x, y float64) float64 {
	return g.dx*x + g.dy*y
}

func (g grad3) dot3(x, y, z float64) float64 {
	return g.dx*x + g.dy*y + g.dz*z
}

func (g grad4) dot(x, y, z, w float64) float64 {
	return g.dx*x + g.dy*y + g.dz*z + g.dw*w
}

var g3 = [...]grad3{
	grad3{1, 1, 0},
	grad3{-1, 1, 0},
	grad3{1, -1, 0},
	grad3{-1, -1, 0},

	grad3{1, 0, 1},
	grad3{-1, 0, 1},
	grad3{1, 0, -1},
	grad3{-1, 0, -1},

	grad3{0, 1, 1},
	grad3{0, -1, 1},
	grad3{0, 1, -1},
	grad3{0, -1, -1},
}

var g4 = [...]grad4{
	grad4{0, 1, 1, 1}, grad4{0, 1, 1, -1}, grad4{0, 1, -1, 1}, grad4{0, 1, -1, -1},
	grad4{0, -1, 1, 1}, grad4{0, -1, 1, -1}, grad4{0, -1, -1, 1}, grad4{0, -1, -1, -1},
	grad4{1, 0, 1, 1}, grad4{1, 0, 1, -1}, grad4{1, 0, -1, 1}, grad4{1, 0, -1, -1},
	grad4{-1, 0, 1, 1}, grad4{-1, 0, 1, -1}, grad4{-1, 0, -1, 1}, grad4{-1, 0, -1, -1},
	grad4{1, 1, 0, 1}, grad4{1, 1, 0, -1}, grad4{1, -1, 0, 1}, grad4{1, -1, 0, -1},
	grad4{-1, 1, 0, 1}, grad4{-1, 1, 0, -1}, grad4{-1, -1, 0, 1}, grad4{-1, -1, 0, -1},
	grad4{1, 1, 1, 0}, grad4{1, 1, -1, 0}, grad4{1, -1, 1, 0}, grad4{1, -1, -1, 0},
	grad4{-1, 1, 1, 0}, grad4{-1, 1, -1, 0}, grad4{-1, -1, 1, 0}, grad4{-1, -1, -1, 0},
}

func (s *Simplex) getPerm(k int) int {
	return int(s.mix[k & 0xff])
}

func (s *Simplex) getPermMod12(k int) int {
	return s.getPerm(k) % 12
}

func fastfloor(x float64) int {
	return int(math.Floor(x))
}

var F2 = 0.5 * (math.Sqrt(3.0) - 1.0)
var G2 = (3.0 - math.Sqrt(3.0)) / 6.0

const F3 = 1.0 / 3.0
const G3 = 1.0 / 6.0

var F4 = (math.Sqrt(5.0) - 1.0) / 4.0
var G4 = (5.0 - math.Sqrt(5.0)) / 20.0

func (s *Simplex) Noise2(x, y float64) float64 {
	//double n0, n1, n2; // Noise contributions from the three corners
	// Skew the input space to determine which simplex cell we're in
	h := (x + y) * F2 // Hairy factor for 2D
	i := fastfloor(x + h)
	j := fastfloor(y + h)
	t := float64(i+j) * G2

	X0 := float64(i) - t // Unskew the cell origin back to (x,y) space
	Y0 := float64(j) - t
	x0 := x - float64(X0) // The x,y distances from the cell origin
	y0 := y - float64(Y0)

	//log.Printf("X (%d,%d) x (%g,%g)", X0, Y0, x0, y0)

	// For the 2D case, the simplex shape is an equilateral triangle.
	// Determine which simplex we are in.
	var i1, j1 int // Offsets for second (middle) corner of simplex in (i,j) coords
	if x0 > y0 {
		// lower triangle, XY order: (0,0)->(1,0)->(1,1)
		i1 = 1
		j1 = 0
	} else {
		i1 = 0
		j1 = 1
	} // upper triangle, YX order: (0,0)->(0,1)->(1,1)

	// A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
	// a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
	// c = (3-sqrt(3))/6
	x1 := x0 - float64(i1) + G2 // Offsets for middle corner in (x,y) unskewed coords
	y1 := y0 - float64(j1) + G2
	x2 := x0 - 1.0 + 2.0*G2 // Offsets for last corner in (x,y) unskewed coords
	y2 := y0 - 1.0 + 2.0*G2
	// Work out the hashed gradient indices of the three simplex corners
	ii := i & 255;
	jj := j & 255;
	gi0 := s.getPermMod12(ii + s.getPerm(jj))
	gi1 := s.getPermMod12(ii + i1 + s.getPerm(jj+j1))
	gi2 := s.getPermMod12(ii + 1 + s.getPerm(jj+1))
	// Calculate the contribution from the three corners
	t0 := 0.5 - x0*x0 - y0*y0
	var n0 float64
	if t0 < 0 {
		n0 = 0.0
	} else {
		t0 *= t0
		n0 = t0 * t0 * g3[gi0].dot(x0, y0) // (x,y) of grad3 used for 2D gradient
	}
	t1 := 0.5 - x1*x1 - y1*y1
	var n1 float64
	if t1 < 0 {
		n1 = 0.0
	} else {
		t1 *= t1
		n1 = t1 * t1 * g3[gi1].dot(x1, y1)
	}
	t2 := 0.5 - x2*x2 - y2*y2
	var n2 float64
	if t2 < 0 {
		n2 = 0.0
	} else {
		t2 *= t2
		n2 = t2 * t2 * g3[gi2].dot(x2, y2)
	}
	// Add contributions from each corner to get the final noise value.
	// The result is scaled to return values in the interval [-1,1].
	return 70.0 * (n0 + n1 + n2)
}

func (s *Simplex) Noise3(x, y, z float64) float64 {
	//double n0, n1, n2, n3; // Noise contributions from the four corners
	// Skew the input space to determine which simplex cell we're in
	h := (x + y + z) * F3 // Very nice and simple skew factor for 3D

	i := fastfloor(x + h)
	j := fastfloor(y + h)
	k := fastfloor(z + h)

	t := float64(i+j+k) * G3
	X0 := float64(i) - t // Unskew the cell origin back to (x,y,z) space
	Y0 := float64(j) - t
	Z0 := float64(k) - t

	x0 := x - float64(X0) // The x,y,z distances from the cell origin
	y0 := y - float64(Y0)
	z0 := z - float64(Z0)

	// For the 3D case, the simplex shape is a slightly irregular tetrahedron.
	// Determine which simplex we are in.
	var i1, j1, k1 int // Offsets for second corner of simplex in (i,j,k) coords
	var i2, j2, k2 int // Offsets for third corner of simplex in (i,j,k) coords
	if x0 >= y0 {
		if y0 >= z0 {
			i1 = 1
			j1 = 0
			k1 = 0
			i2 = 1
			j2 = 1
			k2 = 0 // X Y Z order
		} else if x0 >= z0 {
			i1 = 1
			j1 = 0
			k1 = 0
			i2 = 1
			j2 = 0
			k2 = 1 // X Z Y order
		} else {
			i1 = 0
			j1 = 0
			k1 = 1
			i2 = 1
			j2 = 0
			k2 = 1
		} // Z X Y order
	} else { // x0<y0
		if y0 < z0 {
			i1 = 0
			j1 = 0
			k1 = 1
			i2 = 0
			j2 = 1
			k2 = 1 // Z Y X order
		} else if x0 < z0 {
			i1 = 0
			j1 = 1
			k1 = 0
			i2 = 0
			j2 = 1
			k2 = 1 // Y Z X order
		} else {
			i1 = 0
			j1 = 1
			k1 = 0
			i2 = 1
			j2 = 1
			k2 = 0
		} // Y X Z order
	}
	// A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
	// a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
	// a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
	// c = 1/6.
	x1 := x0 - float64(i1) + G3 // Offsets for second corner in (x,y,z) coords
	y1 := y0 - float64(j1) + G3
	z1 := z0 - float64(k1) + G3
	x2 := x0 - float64(i2) + 2.0*G3 // Offsets for third corner in (x,y,z) coords
	y2 := y0 - float64(j2) + 2.0*G3
	z2 := z0 - float64(k2) + 2.0*G3
	x3 := x0 - 1.0 + 3.0*G3 // Offsets for last corner in (x,y,z) coords
	y3 := y0 - 1.0 + 3.0*G3
	z3 := z0 - 1.0 + 3.0*G3
	// Work out the hashed gradient indices of the four simplex corners
	//int ii = i & 255;
	//int jj = j & 255;
	//int kk = k & 255;
	gi0 := s.getPermMod12(i + s.getPerm(j+s.getPerm(k)))
	gi1 := s.getPermMod12(i + i1 + s.getPerm(j+j1+s.getPerm(k+k1)))
	gi2 := s.getPermMod12(i + i2 + s.getPerm(j+j2+s.getPerm(k+k2)))
	gi3 := s.getPermMod12(i + 1 + s.getPerm(j+1+s.getPerm(k+1)))
	// Calculate the contribution from the four corners
	t0 := 0.6 - x0*x0 - y0*y0 - z0*z0
	var n0, n1, n2, n3 float64
	if t0 < 0 {
		n0 = 0.0
	} else {
		t0 *= t0
		n0 = t0 * t0 * g3[gi0].dot3(x0, y0, z0)
	}
	t1 := 0.6 - x1*x1 - y1*y1 - z1*z1
	if t1 < 0 {
		n1 = 0.0
	} else {
		t1 *= t1
		n1 = t1 * t1 * g3[gi1].dot3(x1, y1, z1)
	}
	t2 := 0.6 - x2*x2 - y2*y2 - z2*z2
	if t2 < 0 {
		n2 = 0.0
	} else {
		t2 *= t2
		n2 = t2 * t2 * g3[gi2].dot3(x2, y2, z2)
	}
	t3 := 0.6 - x3*x3 - y3*y3 - z3*z3
	if t3 < 0 {
		n3 = 0.0
	} else {
		t3 *= t3
		n3 = t3 * t3 * g3[gi3].dot3(x3, y3, z3)
	}
	// Add contributions from each corner to get the final noise value.
	// The result is scaled to stay just inside [-1,1]
	return 32.0 * (n0 + n1 + n2 + n3)
}

func ifexpr(cond bool, t, f int) int {
	if cond {
		return t
	} else {
		return f
	}
}

func (s *Simplex) Noise4(x, y, z, w float64) float64 {
	// Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
	h := (x + y + z + w) * F4 // Factor for 4D skewing
	i := fastfloor(x + h)
	j := fastfloor(y + h)
	k := fastfloor(z + h)
	l := fastfloor(w + h)
	t := float64(i+j+k+l) * G4 // Factor for 4D unskewing
	X0 := float64(i) - t                           // Unskew the cell origin back to (x,y,z,w) space
	Y0 := float64(j) - t
	Z0 := float64(k) - t
	W0 := float64(l) - t
	x0 := x - float64(X0) // The x,y,z,w distances from the cell origin
	y0 := y - float64(Y0)
	z0 := z - float64(Z0)
	w0 := w - float64(W0)
	// For the 4D case, the simplex is a 4D shape I won't even try to describe.
	// To find out which of the 24 possible simplices we're in, we need to
	// determine the magnitude ordering of x0, y0, z0 and w0.
	// Six pair-wise comparisons are performed between each possible pair
	// of the four coordinates, and the results are used to rank the numbers.
	rankx := 0
	ranky := 0
	rankz := 0
	rankw := 0

	if x0 > y0 {
		rankx++
	} else {
		ranky++
	}
	if x0 > z0 {
		rankx++
	} else {
		rankz++
	}
	if x0 > w0 {
		rankx++
	} else {
		rankw++
	}
	if y0 > z0 {
		ranky++
	} else {
		rankz++
	}
	if y0 > w0 {
		ranky++
	} else {
		rankw++
	}
	if z0 > w0 {
		rankz++
	} else {
		rankw++
	}
	var i1, j1, k1, l1 int // The integer offsets for the second simplex corner
	var i2, j2, k2, l2 int // The integer offsets for the third simplex corner
	var i3, j3, k3, l3 int // The integer offsets for the fourth simplex corner
	// simplex[c] is a 4-vector with the numbers 0, 1, 2 and 3 in some order.
	// Many values of c will never occur, since e.g. x>y>z>w makes x<z, y<w and x<w
	// impossible. Only the 24 indices which have non-zero entries make any sense.
	// We use a thresholding to set the coordinates in turn from the largest magnitude.
	// Rank 3 denotes the largest coordinate.
	i1 = ifexpr(rankx >= 3, 1, 0)
	j1 = ifexpr(ranky >= 3, 1, 0)
	k1 = ifexpr(rankz >= 3, 1, 0)
	l1 = ifexpr(rankw >= 3, 1, 0)
	// Rank 2 denotes the second largest coordinate.
	i2 = ifexpr(rankx >= 2, 1, 0)
	j2 = ifexpr(ranky >= 2, 1, 0)
	k2 = ifexpr(rankz >= 2, 1, 0)
	l2 = ifexpr(rankw >= 2, 1, 0)
	// Rank 1 denotes the second smallest coordinate.
	i3 = ifexpr(rankx >= 1, 1, 0)
	j3 = ifexpr(ranky >= 1, 1, 0)
	k3 = ifexpr(rankz >= 1, 1, 0)
	l3 = ifexpr(rankw >= 1, 1, 0)
	// The fifth corner has all coordinate offsets = 1, so no need to compute that.
	x1 := x0 - float64(i1) + G4 // Offsets for second corner in (x,y,z,w) coords
	y1 := y0 - float64(j1) + G4
	z1 := z0 - float64(k1) + G4
	w1 := w0 - float64(l1) + G4
	x2 := x0 - float64(i2) + 2.0*G4 // Offsets for third corner in (x,y,z,w) coords
	y2 := y0 - float64(j2) + 2.0*G4
	z2 := z0 - float64(k2) + 2.0*G4
	w2 := w0 - float64(l2) + 2.0*G4
	x3 := x0 - float64(i3) + 3.0*G4 // Offsets for fourth corner in (x,y,z,float64(w)) coords
	y3 := y0 - float64(j3) + 3.0*G4
	z3 := z0 - float64(k3) + 3.0*G4
	w3 := w0 - float64(l3) + 3.0*G4
	x4 := x0 - 1.0 + 4.0*G4 // Offsets for last corner in (x,y,z,w) coords
	y4 := y0 - 1.0 + 4.0*G4
	z4 := z0 - 1.0 + 4.0*G4
	w4 := w0 - 1.0 + 4.0*G4
	// Work out the hashed gradient indices of the five simplex corners
	ii := i // & 255;
	jj := j // & 255;
	kk := k // & 255;
	ll := l // & 255;

	p := func(n int) int { return s.getPerm(n) }
	//#define p(n)  get_perm(n)

	gi0 := p(ii+p(jj+p(kk+p(ll)))) % 32
	gi1 := p(ii+i1+p(jj+j1+p(kk+k1+p(ll+l1)))) % 32
	gi2 := p(ii+i2+p(jj+j2+p(kk+k2+p(ll+l2)))) % 32
	gi3 := p(ii+i3+p(jj+j3+p(kk+k3+p(ll+l3)))) % 32
	gi4 := p(ii+1+p(jj+1+p(kk+1+p(ll+1)))) % 32

	// Calculate the contribution from the five corners
	var n0, n1, n2, n3, n4 float64 // Noise contributions from the five corners
	t0 := 0.6 - x0*x0 - y0*y0 - z0*z0 - w0*w0
	if t0 < 0 {
		n0 = 0.0
	} else {
		t0 *= t0
		n0 = t0 * t0 * g4[gi0].dot(x0, y0, z0, w0)
	}

	t1 := 0.6 - x1*x1 - y1*y1 - z1*z1 - w1*w1
	if t1 < 0 {
		n1 = 0.0
	} else {
		t1 *= t1
		n1 = t1 * t1 * g4[gi1].dot(x1, y1, z1, w1)
	}

	t2 := 0.6 - x2*x2 - y2*y2 - z2*z2 - w2*w2
	if t2 < 0 {
		n2 = 0.0
	} else {
		t2 *= t2
		n2 = t2 * t2 * g4[gi2].dot(x2, y2, z2, w2)
	}

	t3 := 0.6 - x3*x3 - y3*y3 - z3*z3 - w3*w3
	if t3 < 0 {
		n3 = 0.0
	} else {
		t3 *= t3
		n3 = t3 * t3 * g4[gi3].dot(x3, y3, z3, w3)
	}

	t4 := 0.6 - x4*x4 - y4*y4 - z4*z4 - w4*w4
	if t4 < 0 {
		n4 = 0.0
	} else {
		t4 *= t4
		n4 = t4 * t4 * g4[gi4].dot(x4, y4, z4, w4)
	}

	// Sum up and scale the result to cover the range [-1,1]
	return 27.0 * (n0 + n1 + n2 + n3 + n4)
}
