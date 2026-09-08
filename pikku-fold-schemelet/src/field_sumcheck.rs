// Slot batching is F_{q^2}-linear and commutes with every diagonal factor of
// the witness rounds (eval points, batching and round challenges, layer eq
// tables), so those rounds run on the slot-batched witness over F_{q^2}. The
// ring factor t_1(r_2) of the projection term is absorbed into delta.
use crate::proj_sumcheck::embed_qe;
use crate::qe_vec::QeVec;
use crate::sumcheck::slot_batch;
use incomplete_rexl::{eltwise_fma_mod, eltwise_mult_mod};
use rokoko::common::config::{DEGREE, HALF_DEGREE, MOD_Q};
use rokoko::common::ring_arithmetic::{QuadraticExtension, Representation, RingElement};
use rokoko::common::sumcheck_element::SumcheckElement;

pub(crate) fn diagonal_value(element: &RingElement) -> QuadraticExtension {
    let mut homogenized = element.clone();
    homogenized.from_incomplete_ntt_to_homogenized_field_extensions();
    let slots = homogenized.split_into_quadratic_extensions();
    debug_assert!(slots.iter().all(|slot| *slot == slots[0]));
    slots[0]
}

pub(crate) fn delta_times(
    delta: &[QuadraticExtension; HALF_DEGREE],
    t: &RingElement,
) -> [QuadraticExtension; HALF_DEGREE] {
    let mut homogenized = t.clone();
    homogenized.from_incomplete_ntt_to_homogenized_field_extensions();
    let slots = homogenized.split_into_quadratic_extensions();
    let mut out = [QuadraticExtension::zero(); HALF_DEGREE];
    for i in 0..HALF_DEGREE {
        out[i] *= (&delta[i], &slots[i]);
    }
    out
}

// Phi_delta as a 2 x DEGREE matrix over F_q, read off from the unit vectors.
pub(crate) struct SlotBatcher {
    rows: [Vec<u64>; 2],
}

impl SlotBatcher {
    pub(crate) fn new(delta: &[QuadraticExtension; HALF_DEGREE]) -> Self {
        let mut rows = [vec![0u64; DEGREE], vec![0u64; DEGREE]];
        let mut unit = RingElement::zero(Representation::IncompleteNTT);
        for k in 0..DEGREE {
            unit.v = [0; DEGREE];
            unit.v[k] = 1;
            let image = slot_batch(&unit, delta);
            rows[0][k] = image.coeffs[0];
            rows[1][k] = image.coeffs[1];
        }
        SlotBatcher { rows }
    }

    pub(crate) fn apply(&self, element: &RingElement) -> QuadraticExtension {
        debug_assert!(element.representation == Representation::IncompleteNTT);
        let mut products = [0u64; DEGREE];
        let mut coeffs = [0u64; 2];
        for (limb, row) in coeffs.iter_mut().zip(&self.rows) {
            eltwise_mult_mod(&mut products, row, &element.v, MOD_Q);
            *limb = products.iter().sum::<u64>() % MOD_Q;
        }
        QuadraticExtension { coeffs }
    }

    pub(crate) fn apply_all(&self, elements: &[RingElement]) -> Vec<QuadraticExtension> {
        elements.iter().map(|element| self.apply(element)).collect()
    }
}

// sum_b eq_b w_b for eq_b = e0_b + e1_b alpha, as two F_q-weighted sums.
pub(crate) fn evaluate_column(values: &[RingElement], eq: &QeVec) -> RingElement {
    assert_eq!(values.len(), eq.len());
    let mut acc = [vec![0u64; DEGREE], vec![0u64; DEGREE]];
    let mut next = vec![0u64; DEGREE];
    for (index, value) in values.iter().enumerate() {
        debug_assert!(value.representation == Representation::IncompleteNTT);
        for (limb, weights) in acc.iter_mut().zip([&eq.limb0, &eq.limb1]) {
            eltwise_fma_mod(&mut next, &value.v, weights[index], limb, MOD_Q);
            std::mem::swap(limb, &mut next);
        }
    }
    let mut out = RingElement::zero(Representation::IncompleteNTT);
    let mut alpha_part = RingElement::zero(Representation::IncompleteNTT);
    alpha_part.v.copy_from_slice(&acc[1]);
    out *= (&alpha_part, &embed_qe(&QuadraticExtension { coeffs: [0, 1] }));
    let mut plain = RingElement::zero(Representation::IncompleteNTT);
    plain.v.copy_from_slice(&acc[0]);
    out += &plain;
    out
}
