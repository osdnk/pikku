use crate::commitment::CommitmentKey;
use crate::config::{commitment_rank, folded_norm_bound, witness_norm_bound};
use crate::eval::mle_evaluate;
use crate::output::verify_output;
use crate::prover::{prove_fold, ProverMessage};
use crate::statement::{build_instance, EvalClaim, Instance};
use crate::verifier::verify_fold;
use crate::witness::sample_witness;
use rokoko::common::init_common;
use rokoko::common::matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix};
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::protocol::sumcheck_utils::common::EvaluationSumcheckData;
use rokoko::protocol::sumcheck_utils::linear::BasicEvaluationLinearSumcheck;
use std::sync::LazyLock;

const TEST_LOG_M: usize = 18;

struct Fixture {
    m: usize,
    witness: VerticallyAlignedMatrix<RingElement>,
    key: CommitmentKey,
    instance: Instance,
    prover_message: ProverMessage,
}

static FIXTURE: LazyLock<Fixture> = LazyLock::new(|| {
    init_common();
    let m = 1 << TEST_LOG_M;
    let witness = sample_witness(m);
    let key = CommitmentKey::sample(m, commitment_rank(m));
    let (instance, _) = build_instance(&key, &witness);
    let prover_message = prove_fold(m, &instance, &witness).unwrap();
    Fixture {
        m,
        witness,
        key,
        instance,
        prover_message,
    }
});

fn tampered_instance(f: &Fixture, tamper: impl FnOnce(&mut Instance)) -> Instance {
    let mut instance = Instance {
        commitment: HorizontallyAlignedMatrix {
            data: f.instance.commitment.data.clone(),
            width: f.instance.commitment.width,
            height: f.instance.commitment.height,
        },
        claims: f
            .instance
            .claims
            .iter()
            .map(|claim| EvalClaim {
                point: claim.point.clone(),
                value: claim.value.clone(),
            })
            .collect(),
    };
    tamper(&mut instance);
    instance
}

#[test]
fn end_to_end_fold_with_sumcheck_passes() {
    let f = &*FIXTURE;
    let verifier_message = verify_fold(f.m, &f.instance, &f.prover_message.proof).unwrap();
    verify_output(
        &f.key,
        &f.prover_message.folded_witness,
        &verifier_message.folded_commitment,
        &verifier_message.folded_claim,
        folded_norm_bound(f.m),
    )
    .unwrap();
}

#[test]
fn mle_conventions_are_consistent() {
    let f = &*FIXTURE;
    let point = &f.instance.claims[0].point;
    let folded: Vec<RingElement> = point.iter().rev().cloned().collect();
    let mut evaluator = BasicEvaluationLinearSumcheck::<RingElement>::new(f.m);
    evaluator.load_from(f.witness.col(0));
    let by_folding = evaluator.evaluate(&folded).clone();
    assert_eq!(by_folding, mle_evaluate(f.witness.col(0), point));
    assert_eq!(by_folding, f.instance.claims[0].value);
}

#[test]
fn tampered_claim_value_fails_sumcheck() {
    let f = &*FIXTURE;
    let instance = tampered_instance(f, |instance| {
        instance.claims[0].value += &RingElement::constant(1, Representation::IncompleteNTT);
    });
    let result = verify_fold(f.m, &instance, &f.prover_message.proof);
    assert!(result.is_err());
}

#[test]
fn tampered_commitment_fails_sumcheck() {
    let f = &*FIXTURE;
    let instance = tampered_instance(f, |instance| {
        instance.commitment.data[0] += &RingElement::constant(1, Representation::IncompleteNTT);
    });
    let result = verify_fold(f.m, &instance, &f.prover_message.proof);
    assert!(result.is_err());
}

#[test]
fn tampered_terminal_value_fails_terminal_check() {
    let f = &*FIXTURE;
    let mut proof = f.prover_message.proof.clone();
    proof.terminal_values[0] += &RingElement::constant(1, Representation::IncompleteNTT);
    let result = verify_fold(f.m, &f.instance, &proof);
    assert!(result.err().unwrap().contains("terminal"));
}

#[test]
fn truncated_proof_fails() {
    let f = &*FIXTURE;
    let mut proof = f.prover_message.proof.clone();
    proof.round_polynomials.pop();
    let result = verify_fold(f.m, &f.instance, &proof);
    assert!(result.err().unwrap().contains("rounds"));
}

#[test]
fn oversized_folded_witness_fails_norm_check() {
    let f = &*FIXTURE;
    let verifier_message = verify_fold(f.m, &f.instance, &f.prover_message.proof).unwrap();
    let result = verify_output(
        &f.key,
        &f.prover_message.folded_witness,
        &verifier_message.folded_commitment,
        &verifier_message.folded_claim,
        witness_norm_bound(f.m),
    );
    assert!(result.unwrap_err().contains("norm"));
}

#[test]
fn tampered_projection_trace_fails_trace_check() {
    let f = &*FIXTURE;
    let mut proof = f.prover_message.proof.clone();
    proof.projection_trace[0].v[0] ^= 1;
    let result = verify_fold(f.m, &f.instance, &proof);
    assert!(result.err().unwrap().contains("projection"));
}

#[test]
fn oversized_projection_trace_fails_norm_check() {
    let f = &*FIXTURE;
    let mut proof = f.prover_message.proof.clone();
    proof.projection_trace[0].v[0] = rokoko::common::config::MOD_Q / 2;
    let result = verify_fold(f.m, &f.instance, &proof);
    assert!(result.err().unwrap().contains("norm"));
}

#[test]
fn tampered_batched_projection_fails_trace_check() {
    let f = &*FIXTURE;
    let mut proof = f.prover_message.proof.clone();
    proof.batched_projection[0] += &RingElement::constant(1, Representation::IncompleteNTT);
    let result = verify_fold(f.m, &f.instance, &proof);
    assert!(result.err().unwrap().contains("projection"));
}

#[test]
fn first_coarse_projection_matches_reference() {
    use crate::coarse_projection::project_first_coarse;
    use crate::projection::projection_shape;
    use rokoko::common::matrix::VerticallyAlignedMatrix;
    use rokoko::common::projection_matrix::ProjectionMatrix;
    use rokoko::protocol::project_coarse::project_ring;
    let f = &*FIXTURE;
    let ratios = projection_shape(f.m).unwrap();
    let mut matrix = ProjectionMatrix::new(ratios[0], crate::config::PROJECTION_ROWS);
    let mut sampler = rokoko::common::hash::HashWrapper::new();
    matrix.sample(&mut sampler);
    let input = VerticallyAlignedMatrix {
        data: f.witness.data[..crate::config::FRESH_INPUTS * f.m].to_vec(),
        width: 1,
        height: crate::config::FRESH_INPUTS * f.m,
        used_cols: 1,
    };
    let fast = project_first_coarse(&input, &matrix);
    let reference = project_ring(&input, &matrix);
    assert_eq!(fast.data, reference.data);
}

#[test]
fn eq_expansion_matches_u64_tensor() {
    use crate::proj_sumcheck::expand_eq_qe;
    use rokoko::common::arithmetic::precompute_structured_values_fast;
    use rokoko::common::ring_arithmetic::QuadraticExtension;
    init_common();
    let layers_u64: Vec<u64> = vec![3, 17, 4242, 999_999_999, 5];
    let layers_qe: Vec<QuadraticExtension> = layers_u64
        .iter()
        .map(|&v| QuadraticExtension { coeffs: [v, 0] })
        .collect();
    let expected = precompute_structured_values_fast(&layers_u64);
    let actual = expand_eq_qe(&layers_qe);
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(&expected) {
        assert_eq!(a.coeffs, [*e, 0]);
    }
}

#[test]
fn folded_one_hot_table_matches_eq_expansion() {
    use crate::proj_sumcheck::{embed_qe, expand_eq_qe};
    use crate::sumcheck::round_challenge;
    use rokoko::common::hash::HashWrapper;
    use rokoko::common::ring_arithmetic::QuadraticExtension;
    use rokoko::protocol::sumcheck_utils::common::SumcheckBaseData;
    use rokoko::protocol::sumcheck_utils::linear::LinearSumcheck;
    init_common();
    let index = 0b10110101usize;
    let mut data = vec![RingElement::zero(Representation::IncompleteNTT); 256];
    data[index] = RingElement::constant(1, Representation::IncompleteNTT);
    let mut table = LinearSumcheck::from_data(data);
    let mut sampler = HashWrapper::new();
    let mut field_points: Vec<QuadraticExtension> = vec![];
    for _ in 0..8 {
        let (field_value, ring_value) = round_challenge(&mut sampler);
        table.partial_evaluate(&ring_value);
        field_points.push(field_value);
    }
    let msb: Vec<QuadraticExtension> = field_points.iter().rev().cloned().collect();
    assert_eq!(
        *table.final_evaluations(),
        embed_qe(&expand_eq_qe(&msb)[index])
    );
}

#[test]
fn tampered_round_polynomial_fails() {
    let f = &*FIXTURE;
    let mut proof = f.prover_message.proof.clone();
    proof.round_polynomials[0][0] +=
        &rokoko::common::ring_arithmetic::QuadraticExtension { coeffs: [1, 0] };
    let result = verify_fold(f.m, &f.instance, &proof);
    assert!(result.err().unwrap().contains("mismatch"));
}

#[test]
fn accumulate_kernel_matches_scalar() {
    use crate::proj_sumcheck::{accumulate_j_columns, accumulate_j_columns_scalar};
    use rokoko::common::hash::HashWrapper;
    use rokoko::common::projection_matrix::ProjectionMatrix;
    use rokoko::common::ring_arithmetic::QuadraticExtension;
    use rokoko::common::sumcheck_element::SumcheckElement;
    init_common();
    let mut sampler = HashWrapper::new();
    let mut matrix = ProjectionMatrix::new(64, 256);
    matrix.sample(&mut sampler);
    let mut weights = vec![QuadraticExtension::zero(); 256];
    for weight in &mut weights {
        sampler.sample_field_element_into(weight);
    }
    let soa = crate::qe_vec::QeVec {
        limb0: weights.iter().map(|w| w.coeffs[0]).collect(),
        limb1: weights.iter().map(|w| w.coeffs[1]).collect(),
    };
    let fast = accumulate_j_columns(&matrix, &soa);
    let reference = accumulate_j_columns_scalar(&matrix, &weights);
    for (index, expected) in reference.iter().enumerate() {
        assert_eq!(fast.get(index), *expected);
    }
}

#[test]
fn soa_eq_expansion_and_dot_match_reference() {
    use crate::proj_sumcheck::{dot_qe, expand_eq_qe};
    use crate::qe_vec::expand_eq_soa;
    use rokoko::common::hash::HashWrapper;
    use rokoko::common::ring_arithmetic::QuadraticExtension;
    use rokoko::common::sumcheck_element::SumcheckElement;
    init_common();
    let mut sampler = HashWrapper::new();
    let mut layers = vec![QuadraticExtension::zero(); 9];
    let mut other_layers = vec![QuadraticExtension::zero(); 9];
    for layer in layers.iter_mut().chain(other_layers.iter_mut()) {
        sampler.sample_field_element_into(layer);
    }
    let reference = expand_eq_qe(&layers);
    let soa = expand_eq_soa(&layers);
    for (index, expected) in reference.iter().enumerate() {
        assert_eq!(soa.get(index), *expected);
    }
    let other = expand_eq_soa(&other_layers);
    let other_reference = expand_eq_qe(&other_layers);
    assert_eq!(soa.dot(&other), dot_qe(&reference, &other_reference));
}
