!---------------------------------------------------------------------
!               DIAGONALIZATION SUBROUTINES
!---------------------------------------------------------------------

subroutine rs(nm, n, a, w, z, fv1, ierr)
  implicit none

  integer, intent(in) :: n, nm
  integer, intent(out) :: ierr
  double precision, intent(in) :: a(nm, n)
  double precision, intent(out) :: w(n), z(nm, n), fv1(n)

  if (n .le. nm) goto 10
  ierr = 10 * n
  goto 50

10 call tred2(nm, n, a, w, fv1, z)
  call tql2(nm, n, w, fv1, z, ierr)
50 return
end subroutine rs

subroutine tred2(nm, n, a, d, e, z)
  implicit none

  integer, intent(in) :: nm, n
  integer :: i, j, k, ii, l, jp1
  double precision, intent(in) :: a(nm, n)
  double precision, intent(out) :: d(n), e(n), z(nm, n)
  double precision :: f, g, h, hh, scale

  do 100 i = 1, n
     do 80 j = i, n
80      z(j, i) = a(j, i)

     d(i) = a(n, i)
100 continue

  if (n .eq. 1) goto 510

  do 300 ii = 2, n
     i = n + 2 - ii
     l = i - 1
     h = 0.0d0
     scale = 0.0d0
     if (l .lt. 2) goto 130

     do 120 k = 1, l
120   scale = scale + dabs(d(k))

     if (scale .ne. 0.0d0) goto 140
130  e(i) = d(l)

     do 135 j = 1, l
        d(j) = z(l, j)
        z(i, j) = 0.0d0
        z(j, i) = 0.0d0
135  continue

     goto 290

140  do 150 k = 1, l
        d(k) = d(k) / scale
        h = h + d(k) * d(k)
150  continue

     f = d(l)
     g = -dsign(dsqrt(h), f)
     e(i) = scale * g
     h = h - f * g
     d(l) = f - g

     do 170 j = 1, l
170     e(j) = 0.0d0

     do 240 j = 1, l
        f = d(j)
        z(j, i) = f
        g = e(j) + z(j, j) * f
        jp1 = j + 1
        if (l .lt. jp1) goto 220

        do 200 k = jp1, l
           g = g + z(k, j) * d(k)
           e(k) = e(k) + z(k, j) * f
200     continue

220     e(j) = g
240  continue

     f = 0.0d0

     do 245 j = 1, l
        e(j) = e(j) / h
        f = f + e(j) * d(j)
245  continue

     hh = f / (h + h)

     do 250 j = 1, l
250     e(j) = e(j) - hh * d(j)

     do 280 j = 1, l
        f = d(j)
        g = e(j)

        do 260 k = j, l
260        z(k, j) = z(k, j) - f * e(k) - g * d(k)

        d(j) = z(l, j)
        z(i, j) = 0.0d0
280  continue

290  d(i) = h
300 continue

  do 500 i = 2, n
     l = i - 1
     z(n, l) = z(l, l)
     z(l, l) = 1.0d0
     h = d(i)
     if (h .eq. 0.0d0) goto 380

     do 330 k = 1, l
330     d(k) = z(k, i) / h

     do 360 j = 1, l
        g = 0.0d0

        do 340 k = 1, l
340        g = g + z(k, i) * z(k, j)

        do 350 k = 1, l
350        z(k, j) = z(k, j) - g * d(k)
360  continue

380  do 400 k = 1, l
400     z(k, i) = 0.0d0
500 continue

510 do 520 i = 1, n
     d(i) = z(n, i)
     z(n, i) = 0.0d0
520 continue

  z(n, n) = 1.0d0
  e(1) = 0.0d0
  return
end subroutine tred2

subroutine tql2(nm, n, d, e, z, ierr)
  implicit none

  integer, intent(in) :: nm, n
  integer, intent(out) :: ierr
  integer :: i, j, k, l, m, ii, l1, l2, mml
  double precision, intent(inout) :: d(n), e(n), z(nm, n)
  double precision :: c, c2, c3, dl1, el1, f, g, h, p, r, s, s2, tst1, tst2
  double precision :: pythag

  ierr = 0
  if (n .eq. 1) goto 1001

  do 100 i = 2, n
100  e(i - 1) = e(i)

  f = 0.0d0
  tst1 = 0.0d0
  e(n) = 0.0d0

  do 240 l = 1, n
     j = 0
     h = dabs(d(l)) + dabs(e(l))
     if (tst1 .lt. h) tst1 = h

     do 110 m = l, n
        tst2 = tst1 + dabs(e(m))
        if (tst2 .eq. tst1) goto 120
110  continue

120  if (m .eq. l) goto 220
130  if (j .eq. 30) goto 1000
     j = j + 1

     l1 = l + 1
     l2 = l1 + 1
     g = d(l)
     p = (d(l1) - g) / (2.0d0 * e(l))
     r = pythag(p, 1.0d0)
     d(l) = e(l) / (p + dsign(r, p))
     d(l1) = e(l) * (p + dsign(r, p))
     dl1 = d(l1)
     h = g - d(l)
     if (l2 .gt. n) goto 145

     do 140 i = l2, n
140     d(i) = d(i) - h

145  f = f + h

     p = d(m)
     c = 1.0d0
     c2 = c
     el1 = e(l1)
     s = 0.0d0
     mml = m - l

     do 200 ii = 1, mml
        c3 = c2
        c2 = c
        s2 = s
        i = m - ii
        g = c * e(i)
        h = c * p
        r = pythag(p, e(i))
        e(i + 1) = s * r
        s = e(i) / r
        c = p / r
        p = c * d(i) - s * g
        d(i + 1) = h + s * (c * g + s * d(i))

        do 180 k = 1, n
           h = z(k, i + 1)
           z(k, i + 1) = s * z(k, i) + c * h
           z(k, i) = c * z(k, i) - s * h
180     continue
200  continue

     p = -s * s2 * c3 * el1 * e(l) / dl1
     e(l) = s * p
     d(l) = c * p
     tst2 = tst1 + dabs(e(l))
     if (tst2 .gt. tst1) goto 130
220  d(l) = d(l) + f
240 continue

  do 300 ii = 2, n
     i = ii - 1
     k = i
     p = d(i)

     do 260 j = ii, n
        if (d(j) .ge. p) goto 260
        k = j
        p = d(j)
260  continue

     if (k .eq. i) goto 300
     d(k) = d(i)
     d(i) = p

     do 280 j = 1, n
        p = z(j, i)
        z(j, i) = z(j, k)
        z(j, k) = p
280  continue
300 continue

  goto 1001

1000 ierr = l
1001 return
end subroutine tql2

double precision function pythag(a, b)
  implicit none

  double precision, intent(in) :: a, b
  double precision :: p, r, s, t, u

  p = dmax1(dabs(a), dabs(b))
  if (p .eq. 0.0d0) goto 20
  r = (dmin1(dabs(a), dabs(b)) / p)**2
10 continue
  t = 4.0d0 + r
  if (t .eq. 4.0d0) goto 20
  s = r / t
  u = 1.0d0 + 2.0d0 * s
  p = u * p
  r = (s / u)**2 * r
  goto 10

20 pythag = p
  return
end function pythag
