use crate::config::{FOLD_INPUTS, WITNESS_COEFF_BOUND};
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::common::sampling::sample_random_short_vector;

pub(crate) fn sample_witness(m: usize) -> VerticallyAlignedMatrix<RingElement> {
    VerticallyAlignedMatrix {
        height: m,
        width: FOLD_INPUTS,
        used_cols: FOLD_INPUTS,
        data: sample_random_short_vector(
            m * FOLD_INPUTS,
            WITNESS_COEFF_BOUND,
            Representation::IncompleteNTT,
        ),
    }
}
