use num_complex::Complex64;
use std::fs::File;
use std::io::{BufRead, BufReader};

// Constants
const NZFIXED_MAX: usize = 1000; // Adjust as needed
const NVFCFIXED_MAX: usize = 100; // Adjust as needed

pub struct Constants {
    hemass: f64,
    rmlmda: f64,
}

impl Constants {
    pub fn new(hemass: f64, rmlmda: f64) -> Self {
        Constants { hemass, rmlmda }
    }
}

pub fn load_fixed_pot(
    nzfixed: usize,
    nfc: usize,
    fourierfile: &str,
    rmlmda: f64,
) -> Vec<Vec<Complex64>> {
    let mut vfcfixed = vec![vec![Complex64::new(0.0, 0.0); nfc]; nzfixed];
    let file = File::open(fourierfile).expect("Failed to open the file");
    let reader = BufReader::new(file);

    // Discard the first 5 lines
    for _ in 0..5 {
        reader.lines().next();
    }

    // Read in the Fourier components
    for i in 0..nfc {
        for j in 0..nzfixed {
            if let Some(Ok(line)) = reader.lines().next() {
                let parts: Vec<&str> = line.trim().split_whitespace().collect();
                if parts.len() == 2 {
                    let real: f64 = parts[0].parse().expect("Failed to parse real part");
                    let imag: f64 = parts[1].parse().expect("Failed to parse imaginary part");
                    vfcfixed[j][i] = Complex64::new(real, imag);
                }
            }
        }
    }

    // Scale to the program units
    for row in &mut vfcfixed {
        for elem in row {
            *elem *= rmlmda;
        }
    }

    vfcfixed
}

pub fn potent(
    stepzmin: f64,
    stepzmax: f64,
    nzfixed: usize,
    vfcfixed: &Vec<Vec<Complex64>>,
    nfc: usize,
    m: usize,
    z: Vec<f64>,
) -> Vec<Vec<Complex64>> {
    let mut vfc = vec![vec![Complex64::new(0.0, 0.0); nfc]; m];

    for i in 0..nfc {
        for j in 0..m {
            let zindex = (z[j] - stepzmin) / (stepzmax - stepzmin) * (nzfixed - 1) as f64 + 1.0;
            let indexlow = zindex.floor() as usize;

            if zindex == indexlow as f64 {
                vfc[j][i] = vfcfixed[indexlow][i];
            } else {
                let atmp = vfcfixed[indexlow][i];
                let btmp = vfcfixed[indexlow + 1][i];
                let vrealtmp = atmp + (btmp - atmp) * (zindex - indexlow as f64);
                vfc[j][i] = vrealtmp;
            }
        }
    }

    vfc
}

fn main() {
    let constants = Constants::new(1.0, 2.0);
    let fourierfile = "path_to_fourier_file.txt";
    let nzfixed = 10;
    let nfc = 5;

    let vfcfixed = load_fixed_pot(nzfixed, nfc, fourierfile, constants.rmlmda);
    let stepzmin = 0.0;
    let stepzmax = 1.0;
    let m = 10;
    let z = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    let vfc = potent(stepzmin, stepzmax, nzfixed, &vfcfixed, nfc, m, z);
    
    // Output the results for verification
    for row in vfc {
        for val in row {
            print!("{:?} ", val);
        }
        println!();
    }
}
