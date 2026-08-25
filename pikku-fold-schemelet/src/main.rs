mod args;
mod coarse_projection;
mod commitment;
mod config;
mod eval;
mod eval_claims;
mod fold;
mod output;
mod proj_sumcheck;
mod projection;
mod prover;
mod qe_vec;
mod report;
mod statement;
mod sumcheck;
#[cfg(test)]
mod tests;
mod transcript;
mod verifier;
mod witness;

use crate::args::parse_args;
use crate::commitment::CommitmentKey;
use crate::config::{commitment_rank, folded_norm_bound};
use crate::output::verify_output;
use crate::prover::prove_fold;
use crate::report::{print_report, Timings};
use crate::statement::build_instance;
use crate::verifier::verify_fold;
use crate::witness::sample_witness;
use rokoko::common::init_common;
use std::process;
use std::time::Instant;

fn main() {
    let args = parse_args();
    if let Err(err) = run(args) {
        eprintln!("{err}");
        process::exit(1);
    }
}

fn run(args: args::Args) -> Result<(), String> {
    init_common();
    if args.log_m == 0 {
        return Err("log_m must be positive".to_string());
    }
    let m = 1usize
        .checked_shl(args.log_m as u32)
        .ok_or_else(|| "log_m is too large for this target".to_string())?;

    let sample_start = Instant::now();
    let witness = sample_witness(m);
    let sample = sample_start.elapsed();

    let setup_start = Instant::now();
    let commitment_key = CommitmentKey::sample(m, commitment_rank(m));
    let setup_time = setup_start.elapsed();

    let commit_start = Instant::now();
    let (instance, instance_timings) = build_instance(&commitment_key, &witness);
    let commit_time = commit_start.elapsed();

    let fold_start = Instant::now();
    let prover_message = prove_fold(m, &instance, &witness)?;
    let fold_time = fold_start.elapsed();

    let verify_start = Instant::now();
    let verifier_message = verify_fold(m, &instance, &prover_message.proof)?;
    let verify_core = verify_start.elapsed();

    let output_start = Instant::now();
    let mut output_derivation = std::time::Duration::ZERO;
    if !args.skip_output_check {
        output_derivation = verify_output(
            &commitment_key,
            &prover_message.folded_witness,
            &verifier_message.folded_commitment,
            &verifier_message.folded_claim,
            folded_norm_bound(m),
        )?;
    }
    let output_time = output_start.elapsed() - output_derivation;

    let timings = Timings {
        sample,
        setup: setup_time,
        instance: commit_time - instance_timings.key_derivation,
        instance_parts: instance_timings,
        key_derivation: output_derivation,
        prover_total: fold_time,
        prover: prover_message.timings,
        verify: verify_core,
        output: output_time,
    };
    print_report(
        m,
        &timings,
        args.skip_output_check,
        prover_message.proof.round_polynomials.len(),
        &instance,
        &prover_message.proof,
    );
    Ok(())
}
