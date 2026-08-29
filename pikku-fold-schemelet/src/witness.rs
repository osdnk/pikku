use crate::config::{FOLD_INPUTS, WITNESS_COEFF_BOUND};
use rand::Rng;
use rokoko::common::config::MOD_Q;
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::ring_arithmetic::{Representation, RingElement};

fn sample_uniform_short_vector(
    size: usize,
    bound: u64,
    representation: Representation,
) -> Vec<RingElement> {
    assert!(bound <= i64::MAX as u64);
    assert!(bound < MOD_Q / 2);

    let signed_bound = bound as i64;
    let mut rng = rand::rng();
    let mut elements = Vec::with_capacity(size);

    for _ in 0..size {
        let mut element = RingElement::new(Representation::Coefficients);
        for coefficient in &mut element.v {
            let sampled = rng.random_range(-signed_bound..=signed_bound);
            *coefficient = if sampled < 0 {
                MOD_Q - sampled.unsigned_abs()
            } else {
                sampled as u64
            };
        }
        element.to_representation(representation);
        elements.push(element);
    }

    elements
}

pub(crate) fn sample_witness(m: usize) -> VerticallyAlignedMatrix<RingElement> {
    VerticallyAlignedMatrix {
        height: m,
        width: FOLD_INPUTS,
        used_cols: FOLD_INPUTS,
        data: sample_uniform_short_vector(
            m * FOLD_INPUTS,
            WITNESS_COEFF_BOUND,
            Representation::IncompleteNTT,
        ),
    }
}
