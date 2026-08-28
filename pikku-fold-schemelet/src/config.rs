use rokoko::common::config::DEGREE;

pub(crate) const DEFAULT_LOG_M: usize = 20;
// Ranks from the parameter estimates, set for 32 sequential folding rounds
// at INPUT_COEFF_INFINITY_BOUND = 2^5.
pub(crate) fn commitment_rank(m: usize) -> usize {
    match m.ilog2() {
        ..=18 => 14,
        19..=20 => 15,
        _ => 16,
    }
}
pub(crate) const FRESH_INPUTS: usize = 2;
pub(crate) const ACCUMULATORS: usize = 1;
pub(crate) const FOLD_INPUTS: usize = FRESH_INPUTS + ACCUMULATORS;
pub(crate) const ACCUMULATOR_COL: usize = FRESH_INPUTS;
pub(crate) const FRESH_SELECTOR_VARS: usize = FRESH_INPUTS.ilog2() as usize;
const _: () = assert!(FRESH_INPUTS.is_power_of_two());
// INPUT_COEFF_INFINITY_BOUND = 2^5 in the parameter estimates.
pub(crate) const WITNESS_COEFF_BOUND: u64 = 32;
pub(crate) const FOLD_CHALLENGE_WEIGHT: usize = 23;
pub(crate) const FOLD_CHALLENGE_OP_NORM_BOUND: f64 = 8.357;
pub(crate) const FOLD_CHALLENGE_LABEL: &[u8] = b"pikku-fold-fixed-weight-challenge";

pub(crate) const PROJECTION_LAYERS: usize = 3;
pub(crate) const PROJECTION_ROWS: usize = 2 * DEGREE;
// The number of trace-batching points, mu in the paper.
pub(crate) const PROJECTION_BATCH_POINTS: usize = 2;
pub(crate) const JL_UPPER_TAIL_BETA: f64 = 343.2;

pub(crate) fn witness_norm_bound(m: usize) -> f64 {
    ((m * DEGREE) as f64).sqrt() * (WITNESS_COEFF_BOUND - 1) as f64
}

pub(crate) fn folded_norm_bound(m: usize) -> f64 {
    witness_norm_bound(m) * (1.0 + FRESH_INPUTS as f64 * FOLD_CHALLENGE_OP_NORM_BOUND)
}

pub(crate) fn projection_norm_bound(m: usize) -> f64 {
    (FRESH_INPUTS as f64).sqrt()
        * witness_norm_bound(m)
        * JL_UPPER_TAIL_BETA.powf(PROJECTION_LAYERS as f64 / 2.0)
}
