use crate::config::{
    ACCUMULATOR_COL, FOLD_CHALLENGE_LABEL, FOLD_CHALLENGE_OP_NORM_BOUND, FOLD_CHALLENGE_WEIGHT,
    FOLD_INPUTS, FRESH_INPUTS,
};
use crate::ifma::{fold_pass, FixedMultiplier};
use rokoko::common::hash::HashWrapper;
use rokoko::common::matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix};
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::common::short_challenge::sample_fixed_weight_challenge_into;

pub(crate) fn fold_challenges(transcript: &mut HashWrapper) -> Vec<RingElement> {
    let mut out = vec![RingElement::zero(Representation::IncompleteNTT); FOLD_INPUTS];
    for element in &mut out[..FRESH_INPUTS] {
        sample_fixed_weight_challenge_into::<FOLD_CHALLENGE_WEIGHT>(
            transcript,
            FOLD_CHALLENGE_OP_NORM_BOUND,
            FOLD_CHALLENGE_LABEL,
            element,
        );
    }
    out[ACCUMULATOR_COL] = RingElement::constant(1, Representation::IncompleteNTT);
    out
}

pub(crate) fn fold_witness(
    witness: &VerticallyAlignedMatrix<RingElement>,
    challenges: &[RingElement],
) -> VerticallyAlignedMatrix<RingElement> {
    let m = witness.height;
    debug_assert!(
        challenges[ACCUMULATOR_COL] == RingElement::constant(1, Representation::IncompleteNTT)
    );
    let multipliers: [FixedMultiplier; FRESH_INPUTS] =
        std::array::from_fn(|col| FixedMultiplier::new(&challenges[col]));
    let columns: [&[RingElement]; FRESH_INPUTS] = std::array::from_fn(|col| witness.col(col));
    let data = unsafe { fold_pass(witness.col(ACCUMULATOR_COL), columns, &multipliers) };
    VerticallyAlignedMatrix {
        data,
        width: 1,
        height: m,
        used_cols: 1,
    }
}

pub(crate) fn fold_commitment(
    commitment: &HorizontallyAlignedMatrix<RingElement>,
    challenges: &[RingElement],
) -> Vec<RingElement> {
    let mut out = vec![RingElement::zero(Representation::IncompleteNTT); commitment.height];
    let mut tmp = RingElement::zero(Representation::IncompleteNTT);
    for row in 0..commitment.height {
        for col in 0..FOLD_INPUTS {
            tmp *= (&commitment[(row, col)], &challenges[col]);
            out[row] += &tmp;
        }
    }
    out
}
