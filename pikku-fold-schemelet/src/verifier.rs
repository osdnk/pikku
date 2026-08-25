use crate::config::{
    projection_norm_bound, FOLD_INPUTS, FRESH_INPUTS, FRESH_SELECTOR_VARS, PROJECTION_BATCH_POINTS,
    PROJECTION_LAYERS,
};
use crate::eval_claims::{batched_claim, eval_terminal};
use crate::fold::{fold_challenges, fold_commitment};
use crate::proj_sumcheck::{accumulate_j_columns, build_l0, embed_qe, eq_layers_qe, one_minus};
use crate::projection::{
    check_batched_projection, coeff_l2_norm, j_batched_vectors, projection_shape,
    sample_batching_tensors, sample_projection_matrices, trace_values, TRACE_RING_LEN,
};
use crate::prover::FoldProof;
use crate::qe_vec::expand_eq_soa;
use crate::statement::{EvalClaim, Instance};
use crate::sumcheck::{
    claim_batching_challenges, slot_batch, slot_batching_challenges, verify_sumcheck_rounds,
};
use crate::transcript::statement_transcript;
use rokoko::common::ring_arithmetic::{Representation, RingElement};

pub(crate) struct VerifierMessage {
    pub(crate) folded_commitment: Vec<RingElement>,
    pub(crate) folded_claim: EvalClaim,
}

pub(crate) fn verify_fold(
    m: usize,
    instance: &Instance,
    proof: &FoldProof,
) -> Result<VerifierMessage, String> {
    let mut transcript = statement_transcript(m, instance);

    let coarse_ratios = projection_shape(m)?;
    let matrices = sample_projection_matrices(&coarse_ratios, &mut transcript);
    if proof.projection_trace.len() != TRACE_RING_LEN {
        return Err("wrong projection trace length".to_string());
    }
    transcript.update_with_ring_element_slice(&proof.projection_trace);
    if coeff_l2_norm(&proof.projection_trace) > projection_norm_bound(m) {
        return Err("projection trace norm exceeds bound".to_string());
    }
    let tensors = sample_batching_tensors(&mut transcript);
    if proof.batched_projection.len() != PROJECTION_BATCH_POINTS {
        return Err("wrong number of batched projections".to_string());
    }
    transcript.update_with_ring_element_slice(&proof.batched_projection);
    let trace = trace_values(&proof.projection_trace);
    for (tensor, batched) in tensors.iter().zip(&proof.batched_projection) {
        if !check_batched_projection(tensor, &trace, batched) {
            return Err("batched projection trace check failed".to_string());
        }
    }

    let batching = claim_batching_challenges(&mut transcript);
    let delta = slot_batching_challenges(&mut transcript);
    let (proj_batching, eval_batching) = batching.split_at(PROJECTION_BATCH_POINTS);

    let mut ring_claim = RingElement::zero(Representation::IncompleteNTT);
    let mut tmp = RingElement::zero(Representation::IncompleteNTT);
    for (value, challenge) in proof.batched_projection.iter().zip(proj_batching) {
        tmp *= (value, challenge);
        ring_claim += &tmp;
    }
    ring_claim += &batched_claim(instance, eval_batching);
    let running_claim = slot_batch(&ring_claim, &delta);

    let output_vars = matrices[PROJECTION_LAYERS - 1].projection_height.ilog2() as usize;
    let input_vars = (FRESH_INPUTS * m).ilog2() as usize;
    let middle_vars = input_vars - coarse_ratios[0].ilog2() as usize;
    let witness_vars = FRESH_SELECTOR_VARS + m.ilog2() as usize;
    let prefix_rounds = output_vars + middle_vars;
    let rounds = prefix_rounds + witness_vars;
    let verified = verify_sumcheck_rounds(
        &proof.round_polynomials,
        rounds,
        running_claim,
        &mut transcript,
    )?;
    if proof.terminal_values.len() != FOLD_INPUTS {
        return Err("wrong number of terminal values".to_string());
    }

    // Challenges arrive LS-first; reversed they are the MS-first global point
    // [X_3 witness | X_2 middle | X_1 output].
    let msb_field: Vec<_> = verified.field_points.iter().rev().cloned().collect();
    let witness_point = &msb_field[..witness_vars];
    let middle_point = &msb_field[witness_vars..witness_vars + middle_vars];
    let output_point = &msb_field[witness_vars + middle_vars..];
    let block_vars = middle_vars - output_vars;

    let j_batched = j_batched_vectors(&matrices[PROJECTION_LAYERS - 1], &tensors);
    let l0 = build_l0(&j_batched, proj_batching);
    let eq_output = expand_eq_soa(output_point);
    let mut l0_terminal = RingElement::zero(Representation::IncompleteNTT);
    for (index, value) in l0.iter().enumerate() {
        tmp *= (value, &embed_qe(&eq_output.get(index)));
        l0_terminal += &tmp;
    }

    let middle_matrix_eval = expand_eq_soa(middle_point).dot(&accumulate_j_columns(
        &matrices[PROJECTION_LAYERS - 2],
        &eq_output,
    ));
    let block_eval = eq_layers_qe(&middle_point[..block_vars], &witness_point[..block_vars]);
    let coarse_matrix_eval = expand_eq_soa(&witness_point[block_vars..]).dot(
        &accumulate_j_columns(&matrices[0], &expand_eq_soa(&middle_point[block_vars..])),
    );

    let selector_weights = expand_eq_soa(&witness_point[..FRESH_SELECTOR_VARS]);
    let mut witness_eval = RingElement::zero(Representation::IncompleteNTT);
    for (index, value) in proof.terminal_values[..FRESH_INPUTS].iter().enumerate() {
        tmp *= (value, &embed_qe(&selector_weights.get(index)));
        witness_eval += &tmp;
    }

    // The slot batching is applied only to the completed ring product below:
    // the diagonal (subfield) factors commute with it, the two general ring
    // factors L_0(r_1) and mle[w_all](r_3) do not.
    let mut proj_terminal = RingElement::zero(Representation::IncompleteNTT);
    proj_terminal *= (&l0_terminal, &embed_qe(&middle_matrix_eval));
    let mut scaled = RingElement::zero(Representation::IncompleteNTT);
    let mut block_times_coarse = block_eval;
    block_times_coarse *= &coarse_matrix_eval;
    scaled *= (&proj_terminal, &embed_qe(&block_times_coarse));
    proj_terminal *= (&scaled, &witness_eval);

    let witness_rounds = verified.ring_points[prefix_rounds..].to_vec();
    let eval_part = eval_terminal(
        m,
        instance,
        eval_batching,
        &proof.terminal_values,
        &witness_rounds,
    );

    let mut gamma = RingElement::constant(1, Representation::IncompleteNTT);
    for point in &verified.ring_points[..prefix_rounds] {
        let mut next = RingElement::zero(Representation::IncompleteNTT);
        next *= (&gamma, &one_minus(point));
        gamma = next;
    }
    let mut terminal = RingElement::zero(Representation::IncompleteNTT);
    terminal *= (&gamma, &eval_part);
    terminal += &proj_terminal;
    if slot_batch(&terminal, &delta) != verified.final_claim {
        return Err("sumcheck terminal claim mismatch".to_string());
    }

    transcript.update_with_ring_element_slice(&proof.terminal_values);
    let challenges = fold_challenges(&mut transcript);
    let folded_commitment = fold_commitment(&instance.commitment, &challenges);

    let mut folded_value = RingElement::zero(Representation::IncompleteNTT);
    for (value, challenge) in proof.terminal_values.iter().zip(&challenges) {
        tmp *= (value, challenge);
        folded_value += &tmp;
    }
    let folded_point: Vec<RingElement> = witness_rounds
        .iter()
        .rev()
        .skip(FRESH_SELECTOR_VARS)
        .cloned()
        .collect();

    Ok(VerifierMessage {
        folded_commitment,
        folded_claim: EvalClaim {
            point: folded_point,
            value: folded_value,
        },
    })
}
