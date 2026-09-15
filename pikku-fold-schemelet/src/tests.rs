use crate::commitment::CommitmentKey;
use crate::config::{commitment_rank, folded_norm_bound, witness_norm_bound, WITNESS_COEFF_BOUND};
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

#[test]
fn configured_bounds_and_ranks_match_estimates() {
    assert_eq!(WITNESS_COEFF_BOUND, 1 << 10);
    assert_eq!(commitment_rank(1 << 18), 12);
    assert_eq!(commitment_rank(1 << 20), 12);
    assert_eq!(commitment_rank(1 << 22), 13);
}

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
fn coarse_layers_match_ring_reference() {
    use crate::projection::{project_witness, projection_shape, sample_projection_matrices};
    use rokoko::protocol::project_coarse::project_ring;
    let f = &*FIXTURE;
    let ratios = projection_shape(f.m).unwrap();
    let mut sampler = rokoko::common::hash::HashWrapper::new();
    let matrices = sample_projection_matrices(&ratios, &mut sampler);
    let (levels, _, _) = project_witness(&f.witness, &matrices);
    let input = VerticallyAlignedMatrix {
        data: f.witness.data[..crate::config::FRESH_INPUTS * f.m].to_vec(),
        width: 1,
        height: crate::config::FRESH_INPUTS * f.m,
        used_cols: 1,
    };
    let level0 = project_ring(&input, &matrices[0]);
    assert_eq!(levels[0].data, level0.data);
    let level1 = project_ring(&levels[0], &matrices[1]);
    assert_eq!(levels[1].data, level1.data);
}

#[test]
fn witness_passes_match_reference() {
    use crate::coarse_layers::prepare_i16;
    use crate::field_sumcheck::{evaluate_column, SlotBatcher};
    use crate::ifma::{contract_pass, dot_rows_pass};
    use crate::proj_sumcheck::embed_qe;
    use crate::qe_vec::expand_eq_soa;
    use crate::sumcheck::slot_batching_challenges;
    use crate::vnni::{contract_i16, dot_rows_i16};
    use rokoko::common::hash::HashWrapper;
    use rokoko::common::ring_arithmetic::QuadraticExtension;
    use rokoko::common::sumcheck_element::SumcheckElement;
    init_common();
    let mut sampler = HashWrapper::new();
    let delta = slot_batching_challenges(&mut sampler);
    let batcher = SlotBatcher::new(&delta);
    let elements: Vec<RingElement> = (0..1 << 10)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, WITNESS_COEFF_BOUND))
        .collect();
    let elements_16 = prepare_i16(&elements);
    let [r0, r1] = unsafe { dot_rows_pass(&elements, &batcher.rows()) };
    let [c0, c1] = unsafe { dot_rows_i16(&elements_16, &SlotBatcher::coefficient_rows(&delta)) };
    for (x, element) in elements.iter().enumerate() {
        let expected = batcher.apply(element);
        assert_eq!([r0[x], r1[x]], expected.coeffs);
        assert_eq!([c0[x], c1[x]], expected.coeffs);
    }
    let mut layers = vec![QuadraticExtension::zero(); 8];
    for layer in layers.iter_mut() {
        sampler.sample_field_element_into(layer);
    }
    let weights = expand_eq_soa(&layers);
    let alpha = embed_qe(&QuadraticExtension { coeffs: [0, 1] });
    let contracted = unsafe { contract_pass(&elements, &weights, &alpha) };
    let contracted_16 = unsafe { contract_i16(&elements_16, &weights, &alpha) };
    for (t, block) in elements.chunks_exact(weights.len()).enumerate() {
        let expected = evaluate_column(block, &weights);
        assert_eq!(contracted[t], expected);
        assert_eq!(contracted_16[t], expected);
    }
}

#[test]
#[cfg(not(feature = "derived-key"))]
fn commitment_matches_ring_reference() {
    init_common();
    let (height, rank) = (1 << 10, 3);
    let key = CommitmentKey::sample(height, rank);
    let witness = sample_witness(height);
    let (commitment, _) = key.commit(&witness);
    let mut tmp = RingElement::zero(Representation::IncompleteNTT);
    for row in 0..rank {
        for col in 0..witness.used_cols {
            let mut expected = RingElement::zero(Representation::IncompleteNTT);
            for (a, w) in key.rows[row * height..][..height]
                .iter()
                .zip(witness.col(col))
            {
                tmp *= (a, w);
                expected += &tmp;
            }
            assert_eq!(commitment[(row, col)], expected);
        }
    }
}

#[test]
fn fold_pass_matches_ring_fold() {
    use crate::fold::{fold_challenges, fold_witness};
    let f = &*FIXTURE;
    let mut sampler = rokoko::common::hash::HashWrapper::new();
    let challenges = fold_challenges(&mut sampler);
    let folded = fold_witness(&f.witness, &challenges);
    let reference = rokoko::protocol::fold::fold(&f.witness, &challenges);
    assert_eq!(folded.data, reference.data);
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

#[test]
fn slot_batcher_matches_slot_batch() {
    use crate::field_sumcheck::{delta_times, SlotBatcher};
    use crate::sumcheck::{slot_batch, slot_batching_challenges};
    use rokoko::common::hash::HashWrapper;
    init_common();
    let mut sampler = HashWrapper::new();
    let delta = slot_batching_challenges(&mut sampler);
    let batcher = SlotBatcher::new(&delta);
    let t = RingElement::random(Representation::IncompleteNTT);
    let scaled = SlotBatcher::new(&delta_times(&delta, &t));
    for _ in 0..8 {
        let w = RingElement::random(Representation::IncompleteNTT);
        assert_eq!(batcher.apply(&w), slot_batch(&w, &delta));
        let mut tw = RingElement::zero(Representation::IncompleteNTT);
        tw *= (&t, &w);
        assert_eq!(scaled.apply(&w), slot_batch(&tw, &delta));
    }
}

#[test]
fn column_evaluation_matches_claim() {
    use crate::field_sumcheck::{diagonal_value, evaluate_column};
    use crate::qe_vec::expand_eq_soa;
    let f = &*FIXTURE;
    for (col, claim) in f.instance.claims.iter().enumerate() {
        let point: Vec<_> = claim.point.iter().map(diagonal_value).collect();
        assert_eq!(
            evaluate_column(f.witness.col(col), &expand_eq_soa(&point)),
            claim.value
        );
    }
}
