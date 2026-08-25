use crate::commitment::CommitmentKey;
use crate::config::FOLD_INPUTS;
use crate::eval::mle_evaluate;
use rokoko::common::hash::HashWrapper;
use rokoko::common::matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix};
use rokoko::common::ring_arithmetic::{Representation, RingElement};

pub(crate) struct EvalClaim {
    pub(crate) point: Vec<RingElement>,
    pub(crate) value: RingElement,
}

pub(crate) struct Instance {
    pub(crate) commitment: HorizontallyAlignedMatrix<RingElement>,
    pub(crate) claims: Vec<EvalClaim>,
}

pub(crate) struct InstanceTimings {
    pub(crate) commit_columns: Vec<std::time::Duration>,
    pub(crate) claims: std::time::Duration,
    pub(crate) key_derivation: std::time::Duration,
}

pub(crate) fn build_instance(
    commitment_key: &CommitmentKey,
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> (Instance, InstanceTimings) {
    let rank = crate::config::commitment_rank(witness.height);
    let mut commitment = HorizontallyAlignedMatrix {
        data: vec![
            rokoko::common::ring_arithmetic::RingElement::zero(Representation::IncompleteNTT);
            rank * witness.used_cols
        ],
        width: witness.used_cols,
        height: rank,
    };
    let mut commit_columns = Vec::with_capacity(witness.used_cols);
    let mut key_derivation = std::time::Duration::ZERO;
    for col in 0..witness.used_cols {
        let start = std::time::Instant::now();
        let (column, derivation) = commitment_key.commit_column(witness.col(col));
        commit_columns.push(start.elapsed() - derivation);
        key_derivation += derivation;
        for (row, value) in column.into_iter().enumerate() {
            commitment[(row, col)] = value;
        }
    }
    let claims_start = std::time::Instant::now();
    let point_vars = witness.height.ilog2() as usize;
    let mut sampler = HashWrapper::new();
    sampler.update_with_ring_element(&RingElement::random(Representation::IncompleteNTT));
    let claims = (0..FOLD_INPUTS)
        .map(|col| {
            let mut point = vec![RingElement::zero(Representation::IncompleteNTT); point_vars];
            sampler.sample_ring_element_ntt_slots_same_vec_into(&mut point);
            let value = mle_evaluate(witness.col(col), &point);
            EvalClaim { point, value }
        })
        .collect();
    let timings = InstanceTimings {
        commit_columns,
        claims: claims_start.elapsed(),
        key_derivation,
    };
    (Instance { commitment, claims }, timings)
}
