use crate::config::{ACCUMULATORS, FRESH_INPUTS, PROJECTION_BATCH_POINTS};
use rokoko::common::arithmetic::field_to_ring_element_into;
use rokoko::common::config::HALF_DEGREE;
use rokoko::common::hash::HashWrapper;
use rokoko::common::ring_arithmetic::{QuadraticExtension, Representation, RingElement};
use rokoko::common::sumcheck_element::SumcheckElement;
use rokoko::protocol::sumcheck_utils::common::{HighOrderSumcheckData, SumcheckBaseData};
use rokoko::protocol::sumcheck_utils::elephant_cell::ElephantCell;
use rokoko::protocol::sumcheck_utils::linear::LinearSumcheck;
use rokoko::protocol::sumcheck_utils::polynomial::Polynomial;
use rokoko::protocol::sumcheck_utils::ring_to_field_combiner::RingToFieldCombiner;

pub(crate) fn claim_batching_challenges(transcript: &mut HashWrapper) -> Vec<RingElement> {
    let sampled = PROJECTION_BATCH_POINTS + FRESH_INPUTS;
    let mut out = vec![RingElement::zero(Representation::IncompleteNTT); sampled + ACCUMULATORS];
    transcript.sample_ring_element_ntt_slots_same_vec_into(&mut out[..sampled]);
    out[sampled] = RingElement::constant(1, Representation::IncompleteNTT);
    out
}

pub(crate) fn slot_batching_challenges(
    transcript: &mut HashWrapper,
) -> [QuadraticExtension; HALF_DEGREE] {
    let mut element = RingElement::zero(Representation::IncompleteNTT);
    transcript.sample_ring_element_into(&mut element);
    element.from_incomplete_ntt_to_homogenized_field_extensions();
    element.split_into_quadratic_extensions()
}

pub(crate) fn slot_batch(
    element: &RingElement,
    delta: &[QuadraticExtension; HALF_DEGREE],
) -> QuadraticExtension {
    let mut homogenized = element.clone();
    homogenized.from_incomplete_ntt_to_homogenized_field_extensions();
    let slots = homogenized.split_into_quadratic_extensions();
    let mut acc = QuadraticExtension::zero();
    let mut tmp = QuadraticExtension::zero();
    for (slot, challenge) in slots.iter().zip(delta.iter()) {
        tmp *= (slot, challenge);
        acc += &tmp;
    }
    acc
}

pub(crate) fn slot_batch_poly(
    poly: &Polynomial<RingElement>,
    delta: &[QuadraticExtension; HALF_DEGREE],
) -> Polynomial<QuadraticExtension> {
    let mut out = Polynomial::<QuadraticExtension>::new(0);
    for i in 0..poly.num_coefficients {
        out.coefficients[i] = slot_batch(&poly.coefficients[i], delta);
    }
    out.num_coefficients = poly.num_coefficients;
    out
}

pub(crate) fn round_challenge(transcript: &mut HashWrapper) -> (QuadraticExtension, RingElement) {
    let mut field_value = QuadraticExtension::zero();
    transcript.sample_field_element_into(&mut field_value);
    let mut ring_value = RingElement::zero(Representation::IncompleteNTT);
    field_to_ring_element_into(&mut ring_value, &field_value);
    ring_value.from_homogenized_field_extensions_to_incomplete_ntt();
    (field_value, ring_value)
}

pub(crate) struct SumcheckExecution {
    pub(crate) round_polynomials: Vec<[QuadraticExtension; 2]>,
}

pub(crate) fn execute_sumcheck_prover(
    field_combiner: &RingToFieldCombiner,
    leaves: &[ElephantCell<LinearSumcheck<RingElement>>],
    rounds: usize,
    transcript: &mut HashWrapper,
) -> SumcheckExecution {
    let mut round_polynomials = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let mut poly = Polynomial::<QuadraticExtension>::new(0);
        field_combiner.univariate_polynomial_into(&mut poly);
        transcript.update_with_quadratic_extension_slice(&poly.coefficients);
        let (_, ring_value) = round_challenge(transcript);
        for leaf in leaves {
            leaf.borrow_mut().partial_evaluate(&ring_value);
        }
        round_polynomials.push([poly.coefficients[0], poly.coefficients[2]]);
    }
    SumcheckExecution { round_polynomials }
}

pub(crate) struct VerifiedSumcheck {
    pub(crate) field_points: Vec<QuadraticExtension>,
    pub(crate) ring_points: Vec<RingElement>,
    pub(crate) final_claim: QuadraticExtension,
}

pub(crate) fn verify_sumcheck_rounds(
    round_polynomials: &[[QuadraticExtension; 2]],
    rounds: usize,
    mut running_claim: QuadraticExtension,
    transcript: &mut HashWrapper,
) -> Result<VerifiedSumcheck, String> {
    if round_polynomials.len() != rounds {
        return Err("wrong number of sumcheck rounds".to_string());
    }
    let mut field_points = Vec::with_capacity(rounds);
    let mut ring_points = Vec::with_capacity(rounds);
    for pair in round_polynomials {
        let mut linear = running_claim;
        linear -= &pair[0];
        linear -= &pair[0];
        linear -= &pair[1];
        let poly = Polynomial::<QuadraticExtension> {
            coefficients: [pair[0], linear, pair[1], QuadraticExtension::zero()],
            num_coefficients: 3,
        };
        transcript.update_with_quadratic_extension_slice(&poly.coefficients);
        let (field_value, ring_value) = round_challenge(transcript);
        running_claim = poly.at(&field_value);
        field_points.push(field_value);
        ring_points.push(ring_value);
    }
    Ok(VerifiedSumcheck {
        field_points,
        ring_points,
        final_claim: running_claim,
    })
}
