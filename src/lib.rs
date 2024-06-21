mod scatsub;
mod diagsub;
mod potsub;

use std::fs::File;
use std::io::{BufRead, BufReader};

extern crate num_complex;
use num_complex::Complex;

// Constants
const NZFIXED_MAX: usize = 1000; // Adjust as needed
const NVFCFIXED_MAX: usize = 100; // Adjust as needed
const NMAX: usize = 1000; // Adjust as needed
const NFCX: usize = 100; // Adjust as needed
const LMAX: usize = 901; // Adjust as needed
const MMAX: usize = 100; // Adjust as needed

struct Constants {
    hemass: f64,
    rmlmda: f64,
}

impl Constants {
    fn new(hemass: f64) -> Self {
        Constants {
            hemass,
            rmlmda: 2.0 * hemass / 4.18020,
        }
    }
}

fn load_fixed_pot(nzfixed: usize, nfc: usize, fourierfile: &str, rmlmda: f64) -> Vec<Vec<Complex<f64>>> {
    let mut vfcfixed = vec![vec![Complex::new(0.0, 0.0); nfc]; nzfixed];
    let file = File::open(fourierfile).expect("Failed to open the file");
    let reader = BufReader::new(file);

    // Skip the first 5 lines
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
                    vfcfixed[j][i] = Complex::new(real, imag) * rmlmda;
                }
            }
        }
    }

    vfcfixed
}

fn find_mz(emax: f64, vmin: f64, nsf: i32, zmin: f64, zmax: f64) -> (usize, Vec<f64>) {
    let m = MMAX; // Placeholder for calculation
    let w = vec![0.0; m];
    let z = vec![0.0; m];
    (m, z)
}

fn potent(stepzmin: f64, stepzmax: f64, nzfixed: usize, vfcfixed: &Vec<Vec<Complex<f64>>>, nfc: usize, m: usize, z: Vec<f64>) -> Vec<Vec<Complex<f64>>> {
    let mut vfc = vec![vec![Complex::new(0.0, 0.0); nfc]; m];

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
    let constants = Constants::new(1.0); // Replace with actual value of hemass
    let inputfile = "input.conf"; // Replace with actual input file name

    println!("Multiscat: Close Coupled Scattering Program");
    println!("=============================================");
    println!("");

    // Read input parameters from config file
    let mut config_lines = BufReader::new(File::open(inputfile).expect("Failed to open input file"));
    let mut line = String::new();

    // Read Fourier labels file
    config_lines.read_line(&mut line).expect("Error reading Fourier labels file");
    let fourier_labels_file = line.trim();

    // Read scattering conditions file
    line.clear();
    config_lines.read_line(&mut line).expect("Error reading scattering conditions file");
    let scatt_cond_file = line.trim();

    println!("Fourier labels file = {}", fourier_labels_file);
    println!("Loading scattering conditions from {}", scatt_cond_file);

    // Open scattering conditions file
    let mut scatt_cond_lines = BufReader::new(File::open(scatt_cond_file).expect("Failed to open scattering conditions file"));

    // Skip the first line of conditions file
    line.clear();
    scatt_cond_lines.read_line(&mut line).expect("Error reading scattering conditions file");

    // Read other parameters
    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let itest: i32 = line.trim().parse().expect("Failed to parse itest");

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let ipc: i32 = line.trim().parse().expect("Failed to parse ipc");
    let ipc = if ipc < 0 { 0 } else if ipc > 1 { 1 } else { ipc };

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let nsf: i32 = line.trim().parse().expect("Failed to parse nsf");
    let nsf = if nsf < 2 { 2 } else if nsf > 5 { 10 } else { nsf };
    let eps = 0.5 * 10.0.powf(-nsf as f64);

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let nfc: usize = line.trim().parse().expect("Failed to parse nfc");

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let (zmin, zmax): (f64, f64) = {
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        (parts[0].parse().expect("Failed to parse zmin"), parts[1].parse().expect("Failed to parse zmax"))
    };

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let vmin: f64 = line.trim().parse().expect("Failed to parse vmin");

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let dmax: f64 = line.trim().parse().expect("Failed to parse dmax");

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let imax: usize = line.trim().parse().expect("Failed to parse imax");

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let a1: f64 = line.trim().parse().expect("Failed to parse a1");

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let a2: f64 = line.trim().parse().expect("Failed to parse a2");

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let b2: f64 = line.trim().parse().expect("Failed to parse b2");

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let nzfixed: usize = line.trim().parse().expect("Failed to parse nzfixed");

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let (stepzmin, stepzmax): (f64, f64) = {
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        (parts[0].parse().expect("Failed to parse stepzmin"), parts[1].parse().expect("Failed to parse stepzmax"))
    };

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let startindex: i32 = line.trim().parse().expect("Failed to parse startindex");

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let endindex: i32 = line.trim().parse().expect("Failed to parse endindex");

    line.clear();
    config_lines.read_line(&mut line).expect("Error reading input file");
    let hemass: f64 = line.trim().parse().expect("Failed to parse hemass");

    // Preliminary calculations
    let rmlmda = 2.0 * hemass / 4.180;
    // Further setup based on constants
    let constants = Constants::new(hemass);
    let mut vfcfixed = vec![vec![Complex::new(0.0, 0.0); NVFCFIXED_MAX]; NZFIXED_MAX];

    // Label the Fourier components
    let mut fourier_labels_reader = BufReader::new(File::open(fourier_labels_file).expect("Failed to open Fourier labels file"));
    let mut ivx = vec![0; NFCX];
    let mut ivy = vec![0; NFCX];
    let mut nfc00 = 0;

    for i in 0..nfc {
        let mut line = String::new();
        fourier_labels_reader.read_line(&mut line).expect("Error reading Fourier labels file");
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.len() == 2 {
            ivx[i] = parts[0].parse().expect("Failed to parse ivx");
            ivy[i] = parts[1].parse().expect("Failed to parse ivy");
        }
        if ivx[i] == 0 && ivy[i] == 0 {
            nfc00 = i + 1; // Store 1-based index
        }
    }

    // Main loop for potential files
    for in_idx in startindex..=endindex {
        let fourierfile = format!("pot{:05}.in", in_idx);
        let outfile = format!("diffrac{:05}.out", in_idx);

        // Initialize potential
        vfcfixed = load_fixed_pot(nzfixed, nfc, &fourierfile, constants.rmlmda);

        // Do scattering calculations
        println!("\nCalculating scattering for potential: {}", fourierfile);
        println!("Energy / meV    Theta / deg    Phi / deg        I00         Sum");

        let mut scatt_cond_line = String::new();
        while scatt_cond_lines.read_line(&mut scatt_cond_line).unwrap() > 0 {
            let parts: Vec<&str> = scatt_cond_line.trim().split_whitespace().collect();
            let ei: f64 = parts[0].parse().expect("Failed to parse ei");
            let theta: f64 = parts[1].parse().expect("Failed to parse theta");
            let phi: f64 = parts[2].parse().expect("Failed to parse phi");

            let (m, z) = find_mz(0.0, 0.0, nsf, zmin, zmax); // Placeholder values for emax and vmin
            let mut vfc = vec![vec![Complex::new(0.0, 0.0); nfc]; m];
            vfc = potent(stepzmin, stepzmax, nzfixed, &vfcfixed, nfc, m, z);

            // Placeholder functions
            let d = vec![0.0; NMAX];
            let ix = vec![0; NMAX];
            let iy = vec![0; NMAX];
            let n = 0;
            let n00 = 0;
            let e = vec![vec![0.0; NMAX]; MMAX];
            let f = vec![vec![Complex::new(0.0, 0.0); NMAX]; MMAX];
            let t = vec![vec![0.0; MMAX]; MMAX];
            let a = vec![Complex::new(0.0, 0.0); NMAX];
            let b = vec![Complex::new(0.0, 0.0); NMAX];
            let c = vec![Complex::new(0.0, 0.0); NMAX];
            let p = vec![0.0; NMAX];
            let s = vec![0.0; NMAX];

            // Placeholder calls to functions not provided
            waves(0.0, 0.0, 0.0, 0.0, 0.0); // Example call
            precon(m, n, &mut vfc, nfc, nfc00, &d, &e, &f, &t);
            gmres(&mut x, &mut xx, &mut y, m, &ix, &iy, n, n00, &vfc, &ivx, &ivy, nfc, &a, &b, &c, &d, &e, &f, &p, &s, &t, eps, ipc, 0);

            // Placeholder output call
            output(ei, theta, phi, &ix, &iy, n, n00, &d, &p, itest);

            scatt_cond_line.clear();
        }
    }
}
