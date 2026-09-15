use crate::field_sumcheck::diagonal_value;
use crate::ifma::contract_pass;
use crate::proj_sumcheck::embed_qe;
use crate::qe_vec::expand_eq_soa;
use rokoko::common::ring_arithmetic::{QuadraticExtension, RingElement};

// The point is diagonal, so the eq table is F_{q^2}-valued.
pub(crate) fn mle_evaluate(values: &[RingElement], point: &[RingElement]) -> RingElement {
    assert_eq!(values.len(), 1 << point.len());
    let layers: Vec<QuadraticExtension> = point.iter().map(diagonal_value).collect();
    let alpha = embed_qe(&QuadraticExtension { coeffs: [0, 1] });
    unsafe { contract_pass(values, &expand_eq_soa(&layers), &alpha) }.remove(0)
}
