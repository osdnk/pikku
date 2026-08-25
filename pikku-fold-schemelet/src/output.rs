use crate::commitment::CommitmentKey;
use crate::eval::mle_evaluate;
use crate::statement::EvalClaim;
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::norms::l2_norm;
use rokoko::common::ring_arithmetic::RingElement;

pub(crate) fn verify_output(
    commitment_key: &CommitmentKey,
    witness: &VerticallyAlignedMatrix<RingElement>,
    commitment: &[RingElement],
    claim: &EvalClaim,
    norm_bound: f64,
) -> Result<std::time::Duration, String> {
    let (recomputed, derivation) = commitment_key.commit(witness);
    for (row, expected) in commitment.iter().enumerate() {
        if recomputed[(row, 0)] != *expected {
            return Err("folded commitment mismatch".to_string());
        }
    }
    let norm = l2_norm(&witness.data);
    if norm > norm_bound {
        return Err(format!(
            "folded witness norm {norm} exceeds bound {norm_bound}"
        ));
    }
    if mle_evaluate(witness.col(0), &claim.point) != claim.value {
        return Err("folded evaluation claim mismatch".to_string());
    }
    Ok(derivation)
}
