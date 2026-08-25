use crate::config::{
    FRESH_INPUTS, FRESH_SELECTOR_VARS, PROJECTION_BATCH_POINTS, PROJECTION_LAYERS,
};
use crate::eval_claims::{batched_claim, form_eval_claims};
use crate::fold::fold_challenges;
use crate::proj_sumcheck::{
    accumulate_j_columns, build_l0, embed_qe, one_minus, scaled_embedded_table,
};
use crate::projection::{
    batched_projections, j_batched_vectors, project_witness, projection_shape,
    sample_batching_tensors, sample_projection_matrices,
};
use crate::qe_vec::expand_eq_soa;
use crate::statement::Instance;
use crate::sumcheck::{
    claim_batching_challenges, execute_sumcheck_prover, round_challenge, slot_batch_poly,
    slot_batching_challenges,
};
use crate::transcript::statement_transcript;
use rokoko::common::config::HALF_DEGREE;
use rokoko::common::hash::HashWrapper;
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::ring_arithmetic::{QuadraticExtension, Representation, RingElement};
use rokoko::protocol::fold::fold;
use rokoko::protocol::sumcheck_utils::combiner::Combiner;
use rokoko::protocol::sumcheck_utils::common::{HighOrderSumcheckData, SumcheckBaseData};
use rokoko::protocol::sumcheck_utils::elephant_cell::ElephantCell;
use rokoko::protocol::sumcheck_utils::linear::LinearSumcheck;
use rokoko::protocol::sumcheck_utils::polynomial::Polynomial;
use rokoko::protocol::sumcheck_utils::product::ProductSumcheck;
use rokoko::protocol::sumcheck_utils::ring_to_field_combiner::RingToFieldCombiner;

#[derive(Clone)]
pub(crate) struct FoldProof {
    pub(crate) projection_trace: Vec<RingElement>,
    pub(crate) batched_projection: Vec<RingElement>,
    pub(crate) round_polynomials: Vec<[QuadraticExtension; 2]>,
    pub(crate) terminal_values: Vec<RingElement>,
}

pub(crate) struct ProverTimings {
    pub(crate) projection: std::time::Duration,
    pub(crate) sumcheck: std::time::Duration,
    pub(crate) fold: std::time::Duration,
}

pub(crate) struct ProverMessage {
    pub(crate) proof: FoldProof,
    pub(crate) folded_witness: VerticallyAlignedMatrix<RingElement>,
    pub(crate) timings: ProverTimings,
}

// Rounds over the internal boundary variables. The eval claims are padded by
// prefix (1 - X_j) factors, and every unbound padding factor sums to 1 over
// {0, 1}, so their whole contribution to a prefix round is the analytic
// gamma * E * (1 - X) with gamma = prod (1 - r_j) over the bound rounds;
// no padded tables exist. The projection claim contributes the two-table
// product, whose suffix boundaries have collapsed into the precomputed
// intermediate projection (sum over the inner hypercube of a product of
// matrix MLEs is the MLE of the matrix product).
#[allow(clippy::too_many_arguments)]
fn run_prefix_stage(
    product: &ProductSumcheck<RingElement>,
    leaves: [&ElephantCell<LinearSumcheck<RingElement>>; 2],
    rounds: usize,
    padded_eval_sum: &RingElement,
    delta: &[QuadraticExtension; HALF_DEGREE],
    transcript: &mut HashWrapper,
    gamma: &mut RingElement,
    round_polynomials: &mut Vec<[QuadraticExtension; 2]>,
    field_points: &mut Vec<QuadraticExtension>,
) {
    for _ in 0..rounds {
        let mut ring_poly = Polynomial::<RingElement>::new(0);
        product.univariate_polynomial_into(&mut ring_poly);
        let mut padding = RingElement::zero(Representation::IncompleteNTT);
        padding *= (&*gamma, padded_eval_sum);
        if ring_poly.num_coefficients < 2 {
            ring_poly.num_coefficients = 2;
        }
        ring_poly.coefficients[0] += &padding;
        ring_poly.coefficients[1] -= &padding;
        let field_poly = slot_batch_poly(&ring_poly, delta);
        transcript.update_with_quadratic_extension_slice(&field_poly.coefficients);
        let (field_value, ring_value) = round_challenge(transcript);
        for leaf in leaves {
            leaf.borrow_mut().partial_evaluate(&ring_value);
        }
        let mut next_gamma = RingElement::zero(Representation::IncompleteNTT);
        next_gamma *= (&*gamma, &one_minus(&ring_value));
        *gamma = next_gamma;
        round_polynomials.push([field_poly.coefficients[0], field_poly.coefficients[2]]);
        field_points.push(field_value);
    }
}

pub(crate) fn prove_fold(
    m: usize,
    instance: &Instance,
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> Result<ProverMessage, String> {
    let mut transcript = statement_transcript(m, instance);

    let projection_start = std::time::Instant::now();
    let coarse_ratios = projection_shape(m)?;
    let matrices = sample_projection_matrices(&coarse_ratios, &mut transcript);
    let (mut levels, projection_trace) = project_witness(witness, &matrices);
    transcript.update_with_ring_element_slice(&projection_trace);
    let tensors = sample_batching_tensors(&mut transcript);
    let j_batched = j_batched_vectors(&matrices[PROJECTION_LAYERS - 1], &tensors);
    let fine_input = levels.pop().ok_or("missing projection levels")?;
    let middle_input = levels.pop().ok_or("missing projection levels")?;
    let batched_projection = batched_projections(&fine_input, &j_batched);
    let projection_time = projection_start.elapsed();
    transcript.update_with_ring_element_slice(&batched_projection);

    let sumcheck_start = std::time::Instant::now();
    let batching = claim_batching_challenges(&mut transcript);
    let delta = slot_batching_challenges(&mut transcript);
    let (proj_batching, eval_batching) = batching.split_at(PROJECTION_BATCH_POINTS);

    let padded_eval_sum = batched_claim(instance, eval_batching);
    let gadgets = form_eval_claims(m, instance, witness, eval_batching);

    let mut gamma = RingElement::constant(1, Representation::IncompleteNTT);
    let output_vars = matrices[PROJECTION_LAYERS - 1].projection_height.ilog2() as usize;
    let middle_vars = middle_input.height.ilog2() as usize;
    let witness_vars = FRESH_SELECTOR_VARS + m.ilog2() as usize;
    let total_rounds = output_vars + middle_vars + witness_vars;
    let mut round_polynomials = Vec::with_capacity(total_rounds);
    let mut field_points = Vec::with_capacity(output_vars + middle_vars);

    // L_0 = sum_i d_i * j_batched_i, so the stage-A claim sum_c L_0[c] v_2[c]
    // equals sum_i d_i * vbtd_i, matching the verifier's initial claim.
    let l0_leaf = ElephantCell::new(LinearSumcheck::from_data(build_l0(
        &j_batched,
        proj_batching,
    )));
    let fine_leaf = ElephantCell::new(LinearSumcheck::from_data(fine_input.data));
    let product_a = ProductSumcheck::new(l0_leaf.clone(), fine_leaf.clone());
    run_prefix_stage(
        &product_a,
        [&l0_leaf, &fine_leaf],
        output_vars,
        &padded_eval_sum,
        &delta,
        &mut transcript,
        &mut gamma,
        &mut round_polynomials,
        &mut field_points,
    );

    // Boundary contraction: T_1[c] = L_0(r_1) * sum_r J_1[r, c] * eq(r_1)[r].
    // Claim continuity across the boundary is v_2 = A_1 v_1. Rounds bind the
    // least-significant variable first, so the reversed challenges are the
    // MS-first layers the eq expansion expects.
    let l0_terminal = l0_leaf.borrow().final_evaluations().clone();
    let r1_msb: Vec<QuadraticExtension> = field_points.iter().rev().cloned().collect();
    let eq_r1 = expand_eq_soa(&r1_msb);
    let s1 = accumulate_j_columns(&matrices[PROJECTION_LAYERS - 2], &eq_r1);
    let t1_leaf = ElephantCell::new(LinearSumcheck::from_data(scaled_embedded_table(
        &s1,
        &l0_terminal,
    )));
    let middle_leaf = ElephantCell::new(LinearSumcheck::from_data(middle_input.data));
    let product_b = ProductSumcheck::new(t1_leaf.clone(), middle_leaf.clone());
    run_prefix_stage(
        &product_b,
        [&t1_leaf, &middle_leaf],
        middle_vars,
        &padded_eval_sum,
        &delta,
        &mut transcript,
        &mut gamma,
        &mut round_polynomials,
        &mut field_points,
    );

    // A_0(r_2, X_3) factors through the block structure as
    // eq(r_2-block, X_3-block) * mle[M_0](r_2-row, X_3-col); the block index is
    // the top log2(r_0) bits on both boundaries. eqB and S stay separate
    // product factors: collapsing either into the witness table would change
    // the multilinear extension off the cube and the verifier could no longer
    // evaluate the terminal from the matrix description. One of the two is
    // always in a dummy round, so the true round degree stays 2.
    let t1_terminal = t1_leaf.borrow().final_evaluations().clone();
    let block_vars = middle_vars - output_vars;
    let column_vars = witness_vars - block_vars;
    let r2_msb: Vec<QuadraticExtension> =
        field_points[output_vars..].iter().rev().cloned().collect();
    let s = accumulate_j_columns(&matrices[0], &expand_eq_soa(&r2_msb[block_vars..]));
    let s_leaf = ElephantCell::new(LinearSumcheck::from_data_with_prefixed_sufixed_data(
        (0..s.len()).map(|index| embed_qe(&s.get(index))).collect(),
        block_vars,
        0,
    ));
    let block_leaf = ElephantCell::new(LinearSumcheck::from_data_with_prefixed_sufixed_data(
        scaled_embedded_table(&expand_eq_soa(&r2_msb[..block_vars]), &t1_terminal),
        0,
        column_vars,
    ));
    let witness_leaf = ElephantCell::new(LinearSumcheck::from_data(
        witness.data[..FRESH_INPUTS * m].to_vec(),
    ));
    let product_c = ElephantCell::new(ProductSumcheck::new(
        ElephantCell::new(ProductSumcheck::new(s_leaf.clone(), block_leaf.clone())),
        witness_leaf.clone(),
    ));

    let one = RingElement::constant(1, Representation::IncompleteNTT);
    let chain_children: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> =
        vec![product_c, gadgets.combiner.clone()];
    let mut final_combiner = Combiner::new(chain_children);
    final_combiner.load_challenges_from(&[one, gamma.clone()]);
    let mut field_combiner = RingToFieldCombiner::new(ElephantCell::new(final_combiner));
    field_combiner.load_challenges_from(delta);

    let mut chain_leaves = vec![s_leaf, block_leaf, witness_leaf];
    chain_leaves.extend(gadgets.leaves());
    let execution = execute_sumcheck_prover(
        &field_combiner,
        &chain_leaves,
        witness_vars,
        &mut transcript,
    );
    round_polynomials.extend(execution.round_polynomials);

    let terminal_values = gadgets.terminal_values();
    transcript.update_with_ring_element_slice(&terminal_values);
    drop(field_combiner);
    drop(chain_leaves);
    drop(gadgets);
    let sumcheck_time = sumcheck_start.elapsed();

    let fold_start = std::time::Instant::now();
    let challenges = fold_challenges(&mut transcript);
    let folded_witness = fold(witness, &challenges);
    let fold_time = fold_start.elapsed();
    Ok(ProverMessage {
        proof: FoldProof {
            projection_trace,
            batched_projection,
            round_polynomials,
            terminal_values,
        },
        folded_witness,
        timings: ProverTimings {
            projection: projection_time,
            sumcheck: sumcheck_time,
            fold: fold_time,
        },
    })
}
