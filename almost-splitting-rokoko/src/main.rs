mod sampler;

use incomplete_rexl::{add_mod, multiply_mod, power_mod, sub_mod};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use sampler::{sample_challenge, Challenge, DEGREE, HALF_DEGREE};
use std::time::Instant;

const DEFAULT_SAMPLES: u64 = 1 << 22;
const CHUNK_SIZE: u64 = 1 << 14;
const BASE_SEED: u64 = 0x5049_4b4b_552d_4550;

#[derive(Clone, Copy)]
struct Row {
    q: u64,
    weight: usize,
    bound: f64,
    heuristic_exponent: f64,
}

const ROWS: [Row; 8] = [
    Row {
        q: 127,
        weight: 3,
        bound: 2.9,
        heuristic_exponent: -7.977,
    },
    Row {
        q: 383,
        weight: 4,
        bound: 3.4,
        heuristic_exponent: -11.162,
    },
    Row {
        q: 1151,
        weight: 4,
        bound: 3.7,
        heuristic_exponent: -14.337,
    },
    Row {
        q: 1279,
        weight: 4,
        bound: 3.7,
        heuristic_exponent: -14.642,
    },
    Row {
        q: 1663,
        weight: 4,
        bound: 3.8,
        heuristic_exponent: -15.399,
    },
    Row {
        q: 2687,
        weight: 5,
        bound: 4.0,
        heuristic_exponent: -16.784,
    },
    Row {
        q: 5119,
        weight: 5,
        bound: 4.0,
        heuristic_exponent: -18.643,
    },
    Row {
        q: 6143,
        weight: 5,
        bound: 4.1,
        heuristic_exponent: -19.169,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Fq2 {
    a: u64,
    b: u64,
}

impl Fq2 {
    const ZERO: Self = Self { a: 0, b: 0 };
    const ONE: Self = Self { a: 1, b: 0 };

    fn is_zero(self) -> bool {
        self.a == 0 && self.b == 0
    }
}

#[derive(Clone)]
struct Field {
    q: u64,
    nonsquare: u64,
}

impl Field {
    fn new(q: u64) -> Self {
        let nonsquare = (2..q)
            .find(|&x| power_mod(x, (q - 1) / 2, q) == q - 1)
            .expect("odd prime must have a quadratic non-residue");
        Self { q, nonsquare }
    }

    fn add(&self, x: Fq2, y: Fq2) -> Fq2 {
        Fq2 {
            a: add_mod(x.a, y.a, self.q),
            b: add_mod(x.b, y.b, self.q),
        }
    }

    fn sub(&self, x: Fq2, y: Fq2) -> Fq2 {
        Fq2 {
            a: sub_mod(x.a, y.a, self.q),
            b: sub_mod(x.b, y.b, self.q),
        }
    }

    fn neg(&self, x: Fq2) -> Fq2 {
        Fq2 {
            a: if x.a == 0 { 0 } else { self.q - x.a },
            b: if x.b == 0 { 0 } else { self.q - x.b },
        }
    }

    fn mul(&self, x: Fq2, y: Fq2) -> Fq2 {
        let aa = multiply_mod(x.a, y.a, self.q);
        let bb = multiply_mod(x.b, y.b, self.q);
        let dbb = multiply_mod(self.nonsquare, bb, self.q);
        let ab = multiply_mod(x.a, y.b, self.q);
        let ba = multiply_mod(x.b, y.a, self.q);
        Fq2 {
            a: add_mod(aa, dbb, self.q),
            b: add_mod(ab, ba, self.q),
        }
    }

    fn pow(&self, mut x: Fq2, mut exp: u64) -> Fq2 {
        let mut acc = Fq2::ONE;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = self.mul(acc, x);
            }
            x = self.mul(x, x);
            exp >>= 1;
        }
        acc
    }

    fn add_signed(&self, acc: Fq2, coeff: i8, value: Fq2) -> Fq2 {
        match coeff {
            -2 => self.sub(self.sub(acc, value), value),
            -1 => self.sub(acc, value),
            0 => acc,
            1 => self.add(acc, value),
            2 => self.add(self.add(acc, value), value),
            _ => unreachable!("challenge coefficients are in [-2,2]"),
        }
    }

    fn frobenius(&self, x: Fq2) -> Fq2 {
        Fq2 {
            a: x.a,
            b: if x.b == 0 { 0 } else { self.q - x.b },
        }
    }
}

#[derive(Clone)]
struct SlotSystem {
    field: Field,
    powers: Vec<[Fq2; DEGREE]>,
}

impl SlotSystem {
    fn new(q: u64) -> Self {
        assert!(is_prime(q), "q={q} is not prime");
        assert_eq!(multiplicative_order_mod(q % 256, 256), 2);

        let field = Field::new(q);
        let alpha = primitive_256th_root(&field);
        assert_eq!(field.pow(alpha, 128), field.neg(Fq2::ONE));

        let mut roots = Vec::with_capacity(DEGREE);
        for k in 0..DEGREE {
            roots.push(field.pow(alpha, (2 * k + 1) as u64));
        }

        let mut used = vec![false; DEGREE];
        let mut representatives = Vec::with_capacity(HALF_DEGREE);
        for i in 0..DEGREE {
            if used[i] {
                continue;
            }
            let root = roots[i];
            let conjugate = field.frobenius(root);
            representatives.push(root);
            for j in 0..DEGREE {
                if roots[j] == root || roots[j] == conjugate {
                    used[j] = true;
                }
            }
        }
        assert_eq!(representatives.len(), HALF_DEGREE);

        let powers = representatives
            .iter()
            .map(|&root| {
                let mut row = [Fq2::ZERO; DEGREE];
                row[0] = Fq2::ONE;
                for j in 1..DEGREE {
                    row[j] = field.mul(row[j - 1], root);
                }
                row
            })
            .collect();

        Self { field, powers }
    }

    fn non_unit_difference(&self, left: &Challenge, right: &Challenge) -> bool {
        for slot_powers in &self.powers {
            let mut acc = Fq2::ZERO;
            for i in 0..left.weight {
                let p = left.positions[i] as usize;
                acc = self.field.add_signed(acc, left.signs[i], slot_powers[p]);
            }
            for i in 0..right.weight {
                let p = right.positions[i] as usize;
                acc = self.field.add_signed(acc, -right.signs[i], slot_powers[p]);
            }
            if acc.is_zero() {
                return true;
            }
        }
        false
    }
}

#[derive(Default, Clone, Copy)]
struct Counts {
    tests: u64,
    non_units: u64,
    equal_resamples: u64,
    sampler_attempts: u64,
}

impl Counts {
    fn add(self, other: Counts) -> Counts {
        Counts {
            tests: self.tests + other.tests,
            non_units: self.non_units + other.non_units,
            equal_resamples: self.equal_resamples + other.equal_resamples,
            sampler_attempts: self.sampler_attempts + other.sampler_attempts,
        }
    }
}

struct ResultRow {
    row: Row,
    counts: Counts,
    seconds: f64,
}

fn run_row(row: Row, samples: u64) -> ResultRow {
    let slots = SlotSystem::new(row.q);
    let started = Instant::now();
    let chunks = samples.div_ceil(CHUNK_SIZE);
    let counts = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let start = chunk * CHUNK_SIZE;
            let len = CHUNK_SIZE.min(samples - start);
            let mut rng = SmallRng::seed_from_u64(BASE_SEED ^ row.q.rotate_left(17) ^ chunk);
            let mut counts = Counts::default();

            while counts.tests < len {
                let (left, left_attempts) = sample_challenge(&mut rng, row.weight, row.bound);
                let (right, right_attempts) = sample_challenge(&mut rng, row.weight, row.bound);
                counts.sampler_attempts += left_attempts + right_attempts;
                if left == right {
                    counts.equal_resamples += 1;
                    continue;
                }
                counts.tests += 1;
                if slots.non_unit_difference(&left, &right) {
                    counts.non_units += 1;
                }
            }
            counts
        })
        .reduce(Counts::default, Counts::add);

    ResultRow {
        row,
        counts,
        seconds: started.elapsed().as_secs_f64(),
    }
}

fn primitive_256th_root(field: &Field) -> Fq2 {
    let exp = (field.q * field.q - 1) / 256;
    for b in 1..field.q {
        for a in 0..field.q {
            let candidate = Fq2 { a, b };
            let root = field.pow(candidate, exp);
            if field.pow(root, 128) == field.neg(Fq2::ONE) {
                return root;
            }
        }
    }
    panic!("failed to find a primitive 256th root for q={}", field.q);
}

fn multiplicative_order_mod(x: u64, modulus: u64) -> u64 {
    let mut acc = 1 % modulus;
    for order in 1..=modulus {
        acc = (acc * x) % modulus;
        if acc == 1 {
            return order;
        }
    }
    panic!("{x} has no multiplicative order modulo {modulus}");
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

fn parse_samples() -> u64 {
    let mut args = std::env::args().skip(1);
    let mut samples = DEFAULT_SAMPLES;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--samples" => {
                let value = args.next().expect("--samples requires a value");
                samples = value.parse().expect("invalid --samples value");
            }
            "--help" | "-h" => {
                println!("usage: cargo run --release -- [--samples N]");
                std::process::exit(0);
            }
            _ => panic!("unknown argument: {arg}"),
        }
    }
    samples
}

fn format_epsilon(non_units: u64, tests: u64) -> String {
    if non_units == 0 {
        format!("<2^{{-{:.3}}}", (tests as f64).log2())
    } else {
        let epsilon = non_units as f64 / tests as f64;
        format!("\\approx2^{{{:.3}}}", epsilon.log2())
    }
}

fn main() {
    sampler::require_avx512();
    let samples = parse_samples();
    println!("samples_per_row={samples}");
    println!(
        "q,s,B,tests,non_units,epsilon,minus_log2_epsilon,heuristic_log2,avg_sampler_attempts,equal_resamples,seconds"
    );

    let results: Vec<_> = ROWS
        .iter()
        .copied()
        .map(|row| run_row(row, samples))
        .collect();

    for result in &results {
        let epsilon = result.counts.non_units as f64 / result.counts.tests as f64;
        let minus_log2 = if result.counts.non_units == 0 {
            f64::INFINITY
        } else {
            -epsilon.log2()
        };
        let avg_attempts =
            result.counts.sampler_attempts as f64 / (2.0 * result.counts.tests as f64);
        println!(
            "{},{},{:.1},{},{},{:.12e},{:.6},{:.3},{:.6},{},{:.3}",
            result.row.q,
            result.row.weight,
            result.row.bound,
            result.counts.tests,
            result.counts.non_units,
            epsilon,
            minus_log2,
            result.row.heuristic_exponent,
            avg_attempts,
            result.counts.equal_resamples,
            result.seconds
        );
    }

    println!();
    println!("latex_epsilon_cells");
    for result in &results {
        println!(
            "q={} & {} \\\\",
            result.row.q,
            format_epsilon(result.counts.non_units, result.counts.tests)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_system_has_expected_roots() {
        for row in ROWS {
            let slots = SlotSystem::new(row.q);
            assert_eq!(slots.powers.len(), HALF_DEGREE);
            for powers in &slots.powers {
                let root = powers[1];
                assert_eq!(slots.field.pow(root, 128), slots.field.neg(Fq2::ONE));
                assert_eq!(slots.field.pow(root, 256), Fq2::ONE);
            }
        }
    }

    #[test]
    fn copied_sampler_respects_weight_and_bound() {
        let mut rng = SmallRng::seed_from_u64(7);
        for row in ROWS {
            for _ in 0..256 {
                let (challenge, _) = sample_challenge(&mut rng, row.weight, row.bound);
                assert_eq!(challenge.weight, row.weight);
                assert!(
                    sampler::op_norm_sq_sparse(&challenge.positions, &challenge.signs, row.weight)
                        .sqrt()
                        <= row.bound + 1e-12
                );
            }
        }
    }

    #[test]
    fn inverse_mod_is_available_from_incomplete_rexl() {
        assert_eq!(multiply_mod(3, incomplete_rexl::inv_mod(3, 127), 127), 1);
    }

    #[test]
    fn slot_nonunit_matches_polynomial_gcd() {
        for row in ROWS {
            let slots = SlotSystem::new(row.q);
            let mut rng = SmallRng::seed_from_u64(0xdecafbad ^ row.q);
            for _ in 0..128 {
                let left = sampler::sample_attempt(&mut rng, row.weight);
                let right = sampler::sample_attempt(&mut rng, row.weight);
                let by_slots = slots.non_unit_difference(&left, &right);
                let by_gcd = nonunit_by_polynomial_gcd(row.q, &left, &right);
                assert_eq!(by_slots, by_gcd, "q={}", row.q);
            }
        }
    }

    fn nonunit_by_polynomial_gcd(q: u64, left: &Challenge, right: &Challenge) -> bool {
        let mut diff = vec![0u64; DEGREE];
        for i in 0..left.weight {
            add_coeff(q, &mut diff, left.positions[i] as usize, left.signs[i]);
        }
        for i in 0..right.weight {
            add_coeff(q, &mut diff, right.positions[i] as usize, -right.signs[i]);
        }
        trim(&mut diff);
        if diff.is_empty() {
            return true;
        }

        let mut cyclo = vec![0u64; DEGREE + 1];
        cyclo[0] = 1;
        cyclo[DEGREE] = 1;
        let gcd = poly_gcd(q, cyclo, diff);
        gcd.len() > 1
    }

    fn add_coeff(q: u64, coeffs: &mut [u64], index: usize, value: i8) {
        match value {
            -2 => coeffs[index] = sub_mod(sub_mod(coeffs[index], 1, q), 1, q),
            -1 => coeffs[index] = sub_mod(coeffs[index], 1, q),
            0 => {}
            1 => coeffs[index] = add_mod(coeffs[index], 1, q),
            2 => coeffs[index] = add_mod(add_mod(coeffs[index], 1, q), 1, q),
            _ => unreachable!(),
        }
    }

    fn poly_gcd(q: u64, mut a: Vec<u64>, mut b: Vec<u64>) -> Vec<u64> {
        trim(&mut a);
        trim(&mut b);
        while !b.is_empty() {
            let r = poly_rem(q, &a, &b);
            a = b;
            b = r;
        }
        if let Some(&lead) = a.last() {
            let inv = incomplete_rexl::inv_mod(lead, q);
            for coeff in &mut a {
                *coeff = multiply_mod(*coeff, inv, q);
            }
        }
        a
    }

    fn poly_rem(q: u64, a: &[u64], b: &[u64]) -> Vec<u64> {
        assert!(!b.is_empty());
        let mut r = a.to_vec();
        let inv_lead = incomplete_rexl::inv_mod(*b.last().unwrap(), q);
        while r.len() >= b.len() && !r.is_empty() {
            let shift = r.len() - b.len();
            let factor = multiply_mod(*r.last().unwrap(), inv_lead, q);
            if factor != 0 {
                for (i, &b_coeff) in b.iter().enumerate() {
                    let product = multiply_mod(factor, b_coeff, q);
                    r[shift + i] = sub_mod(r[shift + i], product, q);
                }
            }
            trim(&mut r);
        }
        r
    }

    fn trim(poly: &mut Vec<u64>) {
        while poly.last() == Some(&0) {
            poly.pop();
        }
    }
}
