use crate::config::{
    commitment_rank, folded_norm_bound, projection_norm_bound, FOLD_CHALLENGE_OP_NORM_BOUND,
    FOLD_CHALLENGE_WEIGHT, FOLD_INPUTS, PROJECTION_BATCH_POINTS, PROJECTION_LAYERS,
    PROJECTION_ROWS, WITNESS_COEFF_BOUND,
};
use crate::prover::{FoldProof, ProverTimings};
use crate::statement::{Instance, InstanceTimings};
use rokoko::common::config::{DEGREE, MOD_Q};
use rokoko::protocol::config::{to_kb, SizeableProof};
use std::time::Duration;

pub(crate) struct Timings {
    pub(crate) sample: Duration,
    pub(crate) setup: Duration,
    pub(crate) instance: Duration,
    pub(crate) instance_parts: InstanceTimings,
    pub(crate) key_derivation: Duration,
    pub(crate) prover_total: Duration,
    pub(crate) prover: ProverTimings,
    pub(crate) verify: Duration,
    pub(crate) output: Duration,
}

pub(crate) fn commitment_bits(instance: &Instance) -> usize {
    instance
        .commitment
        .data
        .iter()
        .map(|element| element.compact_size_in_bits())
        .sum()
}

pub(crate) fn sumcheck_round_bits(proof: &FoldProof) -> usize {
    proof
        .round_polynomials
        .iter()
        .map(|pair| pair.iter().map(|c| c.size_in_bits()).sum::<usize>())
        .sum()
}

pub(crate) fn debatching_bits(proof: &FoldProof) -> usize {
    proof
        .terminal_values
        .iter()
        .map(|element| element.compact_size_in_bits())
        .sum()
}

pub(crate) fn projection_bits(proof: &FoldProof) -> usize {
    proof
        .projection_trace
        .iter()
        .chain(&proof.batched_projection)
        .map(|element| element.compact_size_in_bits())
        .sum()
}

pub(crate) fn print_report(
    m: usize,
    timings: &Timings,
    skip_output_check: bool,
    rounds: usize,
    instance: &Instance,
    proof: &FoldProof,
) {
    println!("pikku-fold schemelet");
    println!("m = 2^{} ({})", m.ilog2(), m);
    println!("rank = {}", commitment_rank(m));
    println!("q = {}", MOD_Q);
    println!("degree = {}", DEGREE);
    println!("fold_inputs = {}", FOLD_INPUTS);
    println!("witness_coeff_bound = {}", WITNESS_COEFF_BOUND);
    println!("fold_challenge_weight = {}", FOLD_CHALLENGE_WEIGHT);
    println!(
        "fold_challenge_op_norm_bound = {:.3}",
        FOLD_CHALLENGE_OP_NORM_BOUND
    );
    println!("folded_norm_bound = {:.3}", folded_norm_bound(m));
    println!("projection_layers = {}", PROJECTION_LAYERS);
    println!("projection_rows = {}", PROJECTION_ROWS);
    println!("projection_batch_points = {}", PROJECTION_BATCH_POINTS);
    println!("projection_norm_bound = {:.3}", projection_norm_bound(m));
    println!("sumcheck_rounds = {rounds}");
    println!("commitment_kb = {:.3}", to_kb(commitment_bits(instance)));
    println!("projection_kb = {:.3}", to_kb(projection_bits(proof)));
    println!(
        "sumcheck_rounds_kb = {:.3}",
        to_kb(sumcheck_round_bits(proof))
    );
    println!("debatching_kb = {:.3}", to_kb(debatching_bits(proof)));
    println!(
        "total_communication_kb = {:.3}",
        to_kb(projection_bits(proof) + sumcheck_round_bits(proof) + debatching_bits(proof))
    );
    println!("sample_ms = {:.3}", ms(timings.sample));
    println!("setup_ms = {:.3}", ms(timings.setup));
    println!("instance_ms = {:.3}", ms(timings.instance));
    for (col, time) in timings.instance_parts.commit_columns.iter().enumerate() {
        if col + 1 == timings.instance_parts.commit_columns.len() {
            println!("  commit_acc_ms = {:.3}", ms(*time));
        } else {
            println!("  commit_input{col}_ms = {:.3}", ms(*time));
        }
    }
    println!("  claims_ms = {:.3}", ms(timings.instance_parts.claims));
    println!(
        "key_derivation_ms = {:.3}",
        ms(timings.instance_parts.key_derivation + timings.key_derivation)
    );
    println!("prover_total_ms = {:.3}", ms(timings.prover_total));
    println!(
        "  prover_projection_ms = {:.3}",
        ms(timings.prover.projection)
    );
    println!("  prover_sumcheck_ms = {:.3}", ms(timings.prover.sumcheck));
    println!("  prover_fold_ms = {:.3}", ms(timings.prover.fold));
    println!("output_check_ms = {:.3}", ms(timings.output));
    println!("verify_total_ms = {:.3}", ms(timings.verify));
    println!(
        "output_check = {}",
        if skip_output_check {
            "skipped"
        } else {
            "verified"
        }
    );
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
