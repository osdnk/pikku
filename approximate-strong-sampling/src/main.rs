mod sampler;

use incomplete_rexl::{add_mod, multiply_mod, power_mod, sub_mod};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use sampler::{sample_challenge, Challenge, DEGREE, HALF_DEGREE};
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::time::Instant;

const DEFAULT_SAMPLES: u64 = 1 << 22;
const CHUNK_SIZE: u64 = 1 << 14;
const BASE_SEED: u64 = 0x5049_4b4b_552d_4550;
const TOP_ANCHORS: usize = 16;

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

    #[cfg(test)]
    fn is_zero(self) -> bool {
        self.a == 0 && self.b == 0
    }

    fn dense_index(self, q: u64) -> usize {
        (self.a * q + self.b) as usize
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

    fn evaluate_slots_into(&self, challenge: &Challenge, out: &mut [Fq2; HALF_DEGREE]) {
        for (value, slot_powers) in out.iter_mut().zip(&self.powers) {
            let mut acc = Fq2::ZERO;
            for i in 0..challenge.weight {
                let p = challenge.positions[i] as usize;
                acc = self
                    .field
                    .add_signed(acc, challenge.signs[i], slot_powers[p]);
            }
            *value = acc;
        }
    }

    #[cfg(test)]
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

    fn has_matching_slot(&self, left: &Challenge, right_slots: &[Fq2; HALF_DEGREE]) -> bool {
        for (slot_powers, &right_slot) in self.powers.iter().zip(right_slots) {
            let mut acc = Fq2::ZERO;
            for i in 0..left.weight {
                let p = left.positions[i] as usize;
                acc = self.field.add_signed(acc, left.signs[i], slot_powers[p]);
            }
            if acc == right_slot {
                return true;
            }
        }
        false
    }
}

#[derive(Default, Clone, Copy)]
struct Counts {
    samples: u64,
    sampler_attempts: u64,
}

impl Counts {
    fn add(self, other: Counts) -> Counts {
        Counts {
            samples: self.samples + other.samples,
            sampler_attempts: self.sampler_attempts + other.sampler_attempts,
        }
    }
}

struct ResultRow {
    row: Row,
    counts: Counts,
    anchor_samples: u64,
    direct_samples: u64,
    top_anchors: Vec<AnchorCandidate>,
    direct_counts: Vec<DirectCounts>,
    max_slot_count: u16,
    max_slot: usize,
    seconds: f64,
}

#[derive(Clone)]
struct AnchorCandidate {
    challenge: Challenge,
    score_count: u64,
}

#[derive(Default, Clone, Copy)]
struct DirectCounts {
    samples: u64,
    non_units: u64,
    equal: u64,
    sampler_attempts: u64,
}

impl DirectCounts {
    fn add(self, other: DirectCounts) -> DirectCounts {
        DirectCounts {
            samples: self.samples + other.samples,
            non_units: self.non_units + other.non_units,
            equal: self.equal + other.equal,
            sampler_attempts: self.sampler_attempts + other.sampler_attempts,
        }
    }
}

fn run_row(row: Row, samples: u64, anchor_samples: u64, direct_samples: u64) -> ResultRow {
    let slots = SlotSystem::new(row.q);
    let slot_bins = (row.q as usize) * (row.q as usize);
    let bins = HALF_DEGREE
        .checked_mul(slot_bins)
        .expect("slot histogram is too large");
    let histograms: Vec<_> = (0..bins).map(|_| AtomicU16::new(0)).collect();
    let max_observed = AtomicU64::new(0);
    let started = Instant::now();
    let chunks = samples.div_ceil(CHUNK_SIZE);

    let counts = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let start = chunk * CHUNK_SIZE;
            let len = CHUNK_SIZE.min(samples - start);
            let mut rng = SmallRng::seed_from_u64(BASE_SEED ^ row.q.rotate_left(17) ^ chunk);
            let mut counts = Counts::default();
            let mut slot_values = [Fq2::ZERO; HALF_DEGREE];

            while counts.samples < len {
                let (challenge, attempts) = sample_challenge(&mut rng, row.weight, row.bound);
                counts.samples += 1;
                counts.sampler_attempts += attempts;
                slots.evaluate_slots_into(&challenge, &mut slot_values);
                for (slot, &value) in slot_values.iter().enumerate() {
                    let histogram_index = slot * slot_bins + value.dense_index(row.q);
                    let previous = histograms[histogram_index].fetch_add(1, Ordering::Relaxed);
                    assert!(previous < u16::MAX, "histogram counter overflow");
                    update_max(&max_observed, previous + 1, slot);
                }
            }
            counts
        })
        .reduce(Counts::default, Counts::add);

    let encoded_max = max_observed.load(Ordering::Relaxed);
    let top_anchors = top_anchors(row, &slots, &histograms, slot_bins, anchor_samples);
    let direct_counts = direct_against_anchors(row, &slots, &top_anchors, direct_samples);

    ResultRow {
        row,
        counts,
        anchor_samples,
        direct_samples,
        top_anchors,
        direct_counts,
        max_slot_count: (encoded_max >> 32) as u16,
        max_slot: (encoded_max & 0xffff_ffff) as usize,
        seconds: started.elapsed().as_secs_f64(),
    }
}

fn top_anchors(
    row: Row,
    slots: &SlotSystem,
    histograms: &[AtomicU16],
    slot_bins: usize,
    anchor_samples: u64,
) -> Vec<AnchorCandidate> {
    assert!(anchor_samples > 0);
    let chunks = anchor_samples.div_ceil(CHUNK_SIZE);
    (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let start = chunk * CHUNK_SIZE;
            let len = CHUNK_SIZE.min(anchor_samples - start);
            let mut rng = SmallRng::seed_from_u64(
                BASE_SEED ^ row.q.rotate_left(17) ^ 0xa4c3_686f_7253_4545 ^ chunk,
            );
            let mut slot_values = [Fq2::ZERO; HALF_DEGREE];
            let mut best = Vec::with_capacity(TOP_ANCHORS);

            for _ in 0..len {
                let (anchor, _) = sample_challenge(&mut rng, row.weight, row.bound);
                slots.evaluate_slots_into(&anchor, &mut slot_values);
                let score_count: u64 = slot_values
                    .iter()
                    .enumerate()
                    .map(|(slot, &value)| {
                        let histogram_index = slot * slot_bins + value.dense_index(row.q);
                        histograms[histogram_index].load(Ordering::Relaxed) as u64
                    })
                    .sum();
                insert_anchor_candidate(
                    &mut best,
                    AnchorCandidate {
                        challenge: anchor,
                        score_count,
                    },
                );
            }
            best
        })
        .reduce_with(|mut left, right| {
            for candidate in right {
                insert_anchor_candidate(&mut left, candidate);
            }
            left
        })
        .expect("at least one anchor chunk is present")
}

fn insert_anchor_candidate(best: &mut Vec<AnchorCandidate>, candidate: AnchorCandidate) {
    if best
        .iter()
        .any(|other| other.challenge == candidate.challenge)
    {
        return;
    }
    let position = best
        .iter()
        .position(|other| candidate.score_count > other.score_count)
        .unwrap_or(best.len());
    if position < TOP_ANCHORS {
        best.insert(position, candidate);
        best.truncate(TOP_ANCHORS);
    } else if best.len() < TOP_ANCHORS {
        best.push(candidate);
    }
}

fn direct_against_anchors(
    row: Row,
    slots: &SlotSystem,
    anchors: &[AnchorCandidate],
    direct_samples: u64,
) -> Vec<DirectCounts> {
    assert!(direct_samples > 0);
    assert!(!anchors.is_empty());
    let mut anchor_slots = vec![[Fq2::ZERO; HALF_DEGREE]; anchors.len()];
    for (out, anchor) in anchor_slots.iter_mut().zip(anchors) {
        slots.evaluate_slots_into(&anchor.challenge, out);
    }
    let chunks = direct_samples.div_ceil(CHUNK_SIZE);
    (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let start = chunk * CHUNK_SIZE;
            let len = CHUNK_SIZE.min(direct_samples - start);
            let mut rng = SmallRng::seed_from_u64(
                BASE_SEED ^ row.q.rotate_left(17) ^ 0xd15e_c7c0_ffee_0000 ^ chunk,
            );
            let mut counts = vec![DirectCounts::default(); anchors.len()];
            for _ in 0..len {
                let (challenge, attempts) = sample_challenge(&mut rng, row.weight, row.bound);
                for ((counts, anchor), anchor_slots) in
                    counts.iter_mut().zip(anchors).zip(&anchor_slots)
                {
                    counts.samples += 1;
                    counts.sampler_attempts += attempts;
                    if challenge == anchor.challenge {
                        counts.equal += 1;
                    } else if slots.has_matching_slot(&challenge, anchor_slots) {
                        counts.non_units += 1;
                    }
                }
            }
            counts
        })
        .reduce(
            || vec![DirectCounts::default(); anchors.len()],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left = left.add(right);
                }
                left
            },
        )
}

fn update_max(max_observed: &AtomicU64, count: u16, slot: usize) {
    let encoded = ((count as u64) << 32) | slot as u64;
    let mut current = max_observed.load(Ordering::Relaxed);
    while encoded > current {
        match max_observed.compare_exchange_weak(
            current,
            encoded,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(next) => current = next,
        }
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

fn parse_args() -> (u64, Option<u64>, Option<u64>) {
    let mut args = std::env::args().skip(1);
    let mut samples = DEFAULT_SAMPLES;
    let mut anchor_samples = None;
    let mut direct_samples = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--samples" => {
                let value = args.next().expect("--samples requires a value");
                samples = value.parse().expect("invalid --samples value");
            }
            "--anchor-samples" => {
                let value = args.next().expect("--anchor-samples requires a value");
                anchor_samples = Some(value.parse().expect("invalid --anchor-samples value"));
            }
            "--direct-samples" => {
                let value = args.next().expect("--direct-samples requires a value");
                direct_samples = Some(value.parse().expect("invalid --direct-samples value"));
            }
            "--help" | "-h" => {
                println!(
                    "usage: cargo run --release -- [--samples N] [--anchor-samples N] [--direct-samples N]"
                );
                std::process::exit(0);
            }
            _ => panic!("unknown argument: {arg}"),
        }
    }
    (samples, anchor_samples, direct_samples)
}

fn format_power_of_two(value: f64) -> String {
    format!("\\approx2^{{{:.3}}}", value.log2())
}

fn main() {
    sampler::require_avx512();
    let (samples, anchor_samples, direct_samples) = parse_args();
    let anchor_samples = anchor_samples.unwrap_or(samples);
    let direct_samples = direct_samples.unwrap_or(samples);
    println!("samples_per_row={samples}");
    println!("anchor_samples_per_row={anchor_samples}");
    println!("direct_samples_per_row={direct_samples}");
    println!(
        "q,s,B,samples,anchor_samples,direct_samples,max_slot,max_slot_count,max_slot_probability,q2_max_slot_probability,union_epsilon_bound,minus_log2_union_bound,best_anchor_score_bound,minus_log2_best_anchor_score,best_direct_non_units,best_direct_epsilon,minus_log2_best_direct_epsilon,best_direct_equal,worst_direct_rank,worst_direct_anchor_score,worst_direct_epsilon,minus_log2_worst_direct_epsilon,heuristic_log2,avg_sampler_attempts,avg_direct_sampler_attempts,worst_direct_anchor,seconds"
    );

    let results: Vec<_> = ROWS
        .iter()
        .copied()
        .map(|row| run_row(row, samples, anchor_samples, direct_samples))
        .collect();

    for result in &results {
        let max_probability = result.max_slot_count as f64 / result.counts.samples as f64;
        let q2_max_probability = (result.row.q * result.row.q) as f64 * max_probability;
        let epsilon_bound = HALF_DEGREE as f64 * max_probability;
        let best_anchor_score =
            result.top_anchors[0].score_count as f64 / result.counts.samples as f64;
        let best_direct_counts = result.direct_counts[0];
        let best_direct_epsilon =
            best_direct_counts.non_units as f64 / best_direct_counts.samples as f64;
        let (worst_direct_rank, worst_direct_counts) = result
            .direct_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, counts)| counts.non_units)
            .expect("at least one direct count");
        let worst_direct_anchor_score =
            result.top_anchors[worst_direct_rank].score_count as f64 / result.counts.samples as f64;
        let worst_direct_epsilon =
            worst_direct_counts.non_units as f64 / worst_direct_counts.samples as f64;
        let avg_attempts = result.counts.sampler_attempts as f64 / result.counts.samples as f64;
        let avg_direct_attempts =
            best_direct_counts.sampler_attempts as f64 / best_direct_counts.samples as f64;
        println!(
            "{},{},{:.1},{},{},{},{},{},{:.12e},{:.6},{:.12e},{:.6},{:.12e},{:.6},{},{:.12e},{:.6},{},{},{:.12e},{:.12e},{:.6},{:.3},{:.6},{:.6},{},{:.3}",
            result.row.q,
            result.row.weight,
            result.row.bound,
            result.counts.samples,
            result.anchor_samples,
            result.direct_samples,
            result.max_slot,
            result.max_slot_count,
            max_probability,
            q2_max_probability,
            epsilon_bound,
            -epsilon_bound.log2(),
            best_anchor_score,
            -best_anchor_score.log2(),
            best_direct_counts.non_units,
            best_direct_epsilon,
            -best_direct_epsilon.log2(),
            best_direct_counts.equal,
            worst_direct_rank,
            worst_direct_anchor_score,
            worst_direct_epsilon,
            -worst_direct_epsilon.log2(),
            result.row.heuristic_exponent,
            avg_attempts,
            avg_direct_attempts,
            format_challenge(&result.top_anchors[worst_direct_rank].challenge),
            result.seconds
        );
    }

    println!();
    println!("latex_cells");
    for result in &results {
        let max_probability = result.max_slot_count as f64 / result.counts.samples as f64;
        let q2_max_probability = (result.row.q * result.row.q) as f64 * max_probability;
        let epsilon_bound = HALF_DEGREE as f64 * max_probability;
        let anchor_score = result.top_anchors[0].score_count as f64 / result.counts.samples as f64;
        let direct_epsilon = result
            .direct_counts
            .iter()
            .map(|counts| counts.non_units as f64 / counts.samples as f64)
            .fold(0.0f64, f64::max);
        println!(
            "q={} & ${:.3}$ & ${}$ & ${}$ & ${}$ \\\\",
            result.row.q,
            q2_max_probability,
            format_power_of_two(epsilon_bound),
            format_power_of_two(anchor_score),
            format_power_of_two(direct_epsilon)
        );
    }
}

fn format_challenge(challenge: &Challenge) -> String {
    (0..challenge.weight)
        .map(|i| format!("{}:{:+}", challenge.positions[i], challenge.signs[i]))
        .collect::<Vec<_>>()
        .join("|")
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
