extern crate lapack;
use lapack::fortran as lapack;
use num_complex::Complex;

pub fn rs(nm: usize, n: usize, a: &mut [f64], w: &mut [f64], z: &mut [f64], fv1: &mut [f64]) -> i32 {
    if n > nm {
        return (10 * n) as i32;
    }
    tred2(nm, n, a, w, fv1, z);
    let mut ierr = 0;
    tql2(nm, n, w, fv1, z, &mut ierr);
    ierr
}

pub fn tred2(nm: usize, n: usize, a: &mut [f64], d: &mut [f64], e: &mut [f64], z: &mut [f64]) {
    for i in 0..n {
        for j in i..n {
            z[j + i * nm] = a[j + i * nm];
        }
        d[i] = a[(n - 1) + i * nm];
    }

    if n == 1 {
        return;
    }

    for ii in 1..n {
        let i = n - ii;
        let l = i - 1;
        let mut h = 0.0;
        let mut scale = 0.0;

        if l >= 1 {
            for k in 0..=l {
                scale += d[k].abs();
            }

            if scale != 0.0 {
                for k in 0..=l {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }

                let f = d[l];
                let g = -((f.signum()) * h.sqrt());
                e[i] = scale * g;
                h -= f * g;
                d[l] = f - g;

                for j in 0..=l {
                    e[j] = 0.0;
                }

                for j in 0..=l {
                    let f = d[j];
                    z[j + i * nm] = f;
                    let mut g = e[j] + z[j + j * nm] * f;
                    for k in (j + 1)..=l {
                        g += z[k + j * nm] * d[k];
                        e[k] += z[k + j * nm] * f;
                    }
                    e[j] = g;
                }

                let mut f = 0.0;
                for j in 0..=l {
                    e[j] /= h;
                    f += e[j] * d[j];
                }

                let hh = f / (h + h);
                for j in 0..=l {
                    e[j] -= hh * d[j];
                }

                for j in 0..=l {
                    let f = d[j];
                    let g = e[j];
                    for k in j..=l {
                        z[k + j * nm] -= f * e[k] + g * d[k];
                    }
                    d[j] = z[l + j * nm];
                    z[i + j * nm] = 0.0;
                }
            } else {
                e[i] = d[l];
                for j in 0..=l {
                    d[j] = z[l + j * nm];
                    z[i + j * nm] = 0.0;
                    z[j + i * nm] = 0.0;
                }
            }
        } else {
            e[i] = d[l];
        }

        d[i] = h;
    }

    for i in 1..n {
        let l = i - 1;
        z[n - 1 + l * nm] = z[l + l * nm];
        z[l + l * nm] = 1.0;
        let h = d[i];
        if h != 0.0 {
            for k in 0..l {
                d[k] = z[k + i * nm] / h;
            }

            for j in 0..l {
                let mut g = 0.0;
                for k in 0..l {
                    g += z[k + i * nm] * z[k + j * nm];
                }

                for k in 0..l {
                    z[k + j * nm] -= g * d[k];
                }
            }
        }

        for k in 0..l {
            z[k + i * nm] = 0.0;
        }
    }

    for i in 0..n {
        d[i] = z[(n - 1) + i * nm];
        z[(n - 1) + i * nm] = 0.0;
    }

    z[(n - 1) + (n - 1) * nm] = 1.0;
    e[0] = 0.0;
}

pub fn tql2(nm: usize, n: usize, d: &mut [f64], e: &mut [f64], z: &mut [f64], ierr: &mut i32) {
    *ierr = 0;
    if n == 1 {
        return;
    }

    for i in 1..n {
        e[i - 1] = e[i];
    }

    let mut f = 0.0;
    let mut tst1 = 0.0;
    e[n - 1] = 0.0;

    for l in 0..n {
        let mut j = 0;
        let mut h;
        let mut tst2;
        let l1 = l + 1;
        let l2 = l1 + 1;

        loop {
            h = d[l].abs() + e[l].abs();
            if tst1 < h {
                tst1 = h;
            }

            let mut m = l;
            while m < n {
                tst2 = tst1 + e[m].abs();
                if tst2 == tst1 {
                    break;
                }
                m += 1;
            }

            if m == l {
                break;
            }

            if j == 30 {
                *ierr = l as i32;
                return;
            }
            j += 1;

            let g = d[l];
            let p = (d[l1] - g) / (2.0 * e[l]);
            let r = pythag(p, 1.0);
            d[l] = e[l] / (p + p.signum() * r);
            d[l1] = e[l] * (p + p.signum() * r);
            let dl1 = d[l1];
            let h = g - d[l];

            if l2 < n {
                for i in l2..n {
                    d[i] -= h;
                }
            }

            f += h;
            let mut p = d[m];
            let mut c = 1.0;
            let mut c2 = c;
            let el1 = e[l1];
            let mut s = 0.0;

            for ii in (1..=(m - l)).rev() {
                let mut c3 = c2;
                c2 = c;
                let s2 = s;
                let i = m - ii;
                let g = c * e[i];
                let h = c * p;
                let r = pythag(p, e[i]);
                e[i + 1] = s * r;
                s = e[i] / r;
                c = p / r;
                p = c * d[i] - s * g;
                d[i + 1] = h + s * (c * g + s * d[i]);

                for k in 0..n {
                    h = z[k + (i + 1) * nm];
                    z[k + (i + 1) * nm] = s * z[k + i * nm] + c * h;
                    z[k + i * nm] = c * z[k + i * nm] - s * h;
                }
            }

            p = -s * s2 * c3 * el1 * e[l] / dl1;
            e[l] = s * p;
            d[l] = c * p;
        }

        d[l] += f;
    }

    for ii in 1..n {
        let i = ii - 1;
        let mut k = i;
        let mut p = d[i];

        for j in ii..n {
            if d[j] < p {
                k = j;
                p = d[j];
            }
        }

        if k != i {
            d[k] = d[i];
            d[i] = p;

            for j in 0..n {
                p = z[j + i * nm];
                z[j + i * nm] = z[j + k * nm];
                z[j + k * nm] = p;
            }
        }
    }
}

fn pythag(a: f64, b: f64) -> f64 {
    let p = a.abs().max(b.abs());
    if p == 0.0 {
        return 0.0;
    }
    let r = (a.abs().min(b.abs()) / p).powi(2);
    p * (1.0 + r).sqrt()
}

fn main() {
    let nm = 100;
    let n = 5;
    let mut a = vec![0.0; nm * n];
    let mut w = vec![0.0; n];
    let mut z = vec![0.0; nm * n];
    let mut fv1 = vec![0.0; n];

    // Fill a with your data
    for i in 0..n {
        for j in 0..n {
            a[j + i * nm] = (i * n + j) as f64;
        }
    }

    let ierr = rs(nm, n, &mut a, &mut w, &mut z, &mut fv1);

    if ierr != 0 {
        println!("Error: {:?}", ierr);
    } else {
        println!("Eigenvalues: {:?}", w);
        println!("Eigenvectors: {:?}", z);
    }
}
