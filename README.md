Simplex Noise in Go
===================

This package implements simplex noise in Go, based fairly directly on
the implementation in Java by Stefan Gustavson as described in
http://staffwww.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf

``` Go
    r := rand.New(rand.NewSource(101))
    s := NewSimplex(r)

    x := 1.234
    y := 2.345
    a := s.Noise2(x, y)
    fmt.Printf("2D noise value at (%g, %g) is %g\n", x, y, a)
```


