extern crate lapack;
extern crate num_complex;
extern crate num_traits;

use lapack::*;
use num_complex::Complex64;
use std::f64::consts::PI;

// findmz equivalent function
fn findmz(emax: f64, vmin: f64, nsf: f64, zmin: f64, zmax: f64) -> i32 {
    let em = emax + vmin.abs();
    let pz = (1.0 / (2.0 * PI)).sqrt() * em.sqrt();
    let wz = 2.0 * PI / pz;
    let mz = ((2.75 + 0.125 * nsf) * (zmax - zmin) / wz + 1.0) as i32;
    println!("mz = {}, setting mz = 550", mz);
    550
}

// basis equivalent function
fn basis(d: &mut Vec<f64>, ix: &mut Vec<i32>, iy: &mut Vec<i32>, imax: i32, dmax: f64, ei: f64, theta: f64, phi: f64, a1: f64, a2: f64, b2: f64) -> (i32, i32) {
    let mut n = 0;
    let mut n00 = 0;

    let ax = a1;
    let ay = 0.0;
    let bx = a2;
    let by = b2;

    let auc = (ax * by).abs();
    let rec_unit = 2.0 * PI / auc;
    let gax = by * rec_unit;
    let gay = -bx * rec_unit;
    let gbx = -ay * rec_unit;
    let gby = ax * rec_unit;

    let ered = ei;
    let thetad = theta * PI / 180.0;
    let phid = phi * PI / 180.0;

    let pkx = ered.sqrt() * thetad.sin() * phid.cos();
    let pky = ered.sqrt() * thetad.sin() * phid.sin();

    for i1 in -imax..=imax {
        for i2 in -imax..=imax {
            let gx = gax * (i1 as f64) + gbx * (i2 as f64);
            let gy = gay * (i1 as f64) + gby * (i2 as f64);
            let eint = (pkx + gx).powi(2) + (pky + gy).powi(2);
            let di = eint - ered;
            if di < dmax {
                n += 1;
                ix.push(i1);
                iy.push(i2);
                d.push(di);
                if i1 == 0 && i2 == 0 {
                    n00 = n;
                }
            }
        }
    }

    (n, n00)
}

// tshape equivalent function
fn tshape(a: f64, b: f64, m: usize, w: &mut Vec<f64>, x: &mut Vec<f64>, t: &mut Vec<Vec<f64>>) {
    let mut ww = vec![0.0; m + 1];
    let mut xx = vec![0.0; m + 1];
    let mut tt = vec![vec![0.0; m + 1]; m + 1];

    let n = m + 1;
    lobatto(a, b, n, &mut ww, &mut xx);

    for i in 0..n {
        ww[i] = ww[i].sqrt();
    }

    for i in 0..n {
        let mut ff = 0.0;
        for j in 0..n {
            if j == i {
                continue;
            }
            let mut gg = 1.0 / (xx[i] - xx[j]);
            ff += gg;
            for k in 0..n {
                if k == j || k == i {
                    continue;
                }
                gg *= (xx[j] - xx[k]) / (xx[i] - xx[k]);
            }
            tt[j][i] = ww[j] * gg / ww[i];
        }
        tt[i][i] = ff;
    }

    for i in 0..m {
        w.push(ww[i + 1]);
        x.push(xx[i + 1]);
        for j in 0..=i {
            let mut hh = 0.0;
            for k in 0..n {
                hh += tt[k][i + 1] * tt[k][j + 1];
            }
            t[i][j] = hh;
            t[j][i] = hh;
        }
    }
}

// lobatto equivalent function
fn lobatto(a: f64, b: f64, n: usize, w: &mut Vec<f64>, x: &mut Vec<f64>) {
    let l = (n + 1) / 2;
    let pi = PI;
    let shift = 0.5 * (b + a);
    let scale = 0.5 * (b - a);
    let weight = (b - a) / (n * (n - 1)) as f64;

    x[0] = a;
    w[0] = weight;

    for k in 1..l {
        let mut p1 = 1.0;
        let mut z = (pi * (4.0 * k as f64 - 3.0) / (4.0 * n as f64 - 2.0)).cos();
        for _ in 0..7 {
            let mut p2 = 0.0;
            
            for j in 1..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 - 1.0) * z * p2 - (j as f64 - 1.0) * p3) / j as f64;
            }
            p2 = (n as f64 - 1.0) * (p2 - z * p1) / (1.0 - z * z);
            let p3 = (2.0 * z * p2 - n as f64 * (n as f64 - 1.0) * p1) / (1.0 - z * z);
            z -= p2 / p3;
        }
        x[k] = shift - scale * z;
        x[n - k] = shift + scale * z;
        w[k] = weight / (p1 * p1);
        w[n - k] = w[k];
    }

    x[n - 1] = b;
    w[n - 1] = weight;
}

// waves equivalent function
fn waves(a: Complex64, b: Complex64, c: Complex64, zmax: f64) -> Complex64 {
    let q = (a - b * c) * Complex64::new((PI / 3.0).cos(), (PI / 3.0).sin());
    let mut w0 = vec![Complex64::new(0.0, 0.0); 1];
    let mut wr = vec![0.0; 1];
    let mut wi = vec![0.0; 1];
    let mut vl = vec![0.0; 1];
    let mut vr = vec![0.0; 1];
    let mut work = vec![0.0; 4];
    let mut info = 0;

    unsafe {
        dgeev(b'N', b'N', 1, &mut [a.re, a.im, b.re, b.im, c.re, c.im, zmax, q.re, q.im], 1, &mut wr , &mut wi, &mut vl, 1, &mut vr, 1, &mut work, 4, &mut info);
    }

    w0[0] = Complex64::new(wr[0], wi[0]);
    w0[0]
}

#[cfg(test)]
mod tests {
    use std::f64::EPSILON;

    use super::*;

    #[test]
    fn test_findmz() {
        let result = findmz(1.0, -1.0, 2.0, 0.0, 1.0);
        assert_eq!(result, 550);
    }

    #[test]
    fn test_basis() {
        let mut d = Vec::new();
        let mut ix = Vec::new();
        let mut iy = Vec::new();
        let (n, n00) = basis(&mut d, &mut ix, &mut iy, 5, 10.0, 1.0, 45.0, 45.0, 1.0, 2.0, 3.0);
        assert!(n > 0);
        assert!(n00 >= 0);
    }

    #[test]
    fn test_tshape() {
        let mut w = Vec::new();
        let mut x = Vec::new();
        let mut t = vec![vec![0.0; 5]; 5];
        tshape(0.0, 1.0, 4, &mut w, &mut x, &mut t);
        assert_eq!(w.len(), 4);
        assert_eq!(x.len(), 4);
        assert_eq!(t.len(), 4);
        assert_eq!(t[0].len(), 4);
    }

    #[test]
    fn test_lobatto() {
        let mut w = vec![0.0; 5];
        let mut x = vec![0.0; 5];
        lobatto(0.0, 1.0, 5, &mut w, &mut x);
        assert_eq!(w.len(), 5);
        assert_eq!(x.len(), 5);
    }

    #[test]
    fn test_waves() {
        let a = Complex64::new(1.0, 0.0);
        let b = Complex64::new(0.0, 1.0);
        let c = Complex64::new(1.0, 1.0);
        let result = waves(a, b, c, 1.0);
        assert!((result.re - 1.0).abs() < EPSILON);
        assert!((result.im).abs() < EPSILON);
    }
}


