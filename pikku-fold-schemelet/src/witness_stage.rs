// Witness rounds. Every unbound variable is summed over the cube, so the low
// rounds run on tables contracted over the top (block) variables; one ring
// pass at the low point gives the top-round tables and the terminal values
// (slot batching is linear: Phi(sum eq w) = sum eq Phi(w)).
use crate::config::{ACCUMULATOR_COL, FOLD_INPUTS, FRESH_INPUTS, FRESH_SELECTOR_VARS};
use crate::eval_claims::weight_layers;
use crate::field_sumcheck::{delta_times, diagonal_value, evaluate_column, SlotBatcher};
use crate::ifma::{contract_pass, dot_rows_pass, DEGREE};
use crate::proj_sumcheck::{embed_qe, expand_eq_qe};
use crate::qe_vec::{expand_eq_soa, QeVec};
use crate::statement::Instance;
use crate::sumcheck::execute_sumcheck_prover;
use crate::vnni::{contract_i16, dot_rows_i16};
use rokoko::common::config::HALF_DEGREE;
use rokoko::common::hash::HashWrapper;
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::ring_arithmetic::{QuadraticExtension, RingElement};
use rokoko::common::sumcheck_element::SumcheckElement;
use rokoko::protocol::project_coarse::Signed16RingElement;
use rokoko::protocol::sumcheck_utils::combiner::Combiner;
use rokoko::protocol::sumcheck_utils::common::{HighOrderSumcheckData, SumcheckBaseData};
use rokoko::protocol::sumcheck_utils::elephant_cell::ElephantCell;
use rokoko::protocol::sumcheck_utils::linear::LinearSumcheck;
use rokoko::protocol::sumcheck_utils::product::ProductSumcheck;

pub(crate) struct WitnessStageOutput {
    pub(crate) round_polynomials: Vec<[QuadraticExtension; 2]>,
    pub(crate) terminal_values: Vec<RingElement>,
}

type Leaf = ElephantCell<LinearSumcheck<QuadraticExtension>>;
type Node = ElephantCell<dyn HighOrderSumcheckData<Element = QuadraticExtension>>;

fn leaf(data: Vec<QuadraticExtension>, prefix: usize) -> Leaf {
    ElephantCell::new(LinearSumcheck::from_data_with_prefixed_sufixed_data(
        data, prefix, 0,
    ))
}

fn product(a: &Leaf, b: &Leaf) -> Node {
    ElephantCell::new(ProductSumcheck::new(a.clone(), b.clone()))
}

struct Stage {
    combiner: Combiner<QuadraticExtension>,
    leaves: Vec<Leaf>,
}

// projection + gamma * sum_c beta_c * weight_c * witness_c.
fn stage(
    projection: Node,
    projection_leaves: Vec<Leaf>,
    evals: Vec<[Leaf; 2]>,
    eval_batching: &[QuadraticExtension],
    gamma: &QuadraticExtension,
) -> Stage {
    let mut leaves = projection_leaves;
    let mut products: Vec<Node> = Vec::with_capacity(evals.len());
    for pair in evals {
        products.push(product(&pair[0], &pair[1]));
        leaves.extend(pair);
    }
    let mut gadgets = Combiner::new(products);
    gadgets.load_challenges_from(eval_batching);
    let mut combiner = Combiner::new(vec![projection, ElephantCell::new(gadgets)]);
    combiner.load_challenges_from(&[QuadraticExtension::one(), *gamma]);
    Stage { combiner, leaves }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_witness_stage(
    m: usize,
    instance: &Instance,
    witness: &VerticallyAlignedMatrix<RingElement>,
    witness_16: &[Signed16RingElement],
    s: &QeVec,
    block_eq: &[QuadraticExtension],
    delta: &[QuadraticExtension; HALF_DEGREE],
    t1_terminal: &RingElement,
    eval_batching: &[RingElement],
    gamma: &RingElement,
    transcript: &mut HashWrapper,
) -> WitnessStageOutput {
    let block_vars = block_eq.len().ilog2() as usize;
    let top_vars = block_vars - FRESH_SELECTOR_VARS;
    let column_vars = m.ilog2() as usize - top_vars;
    assert_eq!(s.len(), 1usize << column_vars);
    let eval_batching: Vec<QuadraticExtension> = eval_batching.iter().map(diagonal_value).collect();
    let gamma = diagonal_value(gamma);
    let points: Vec<Vec<QuadraticExtension>> = (0..FOLD_INPUTS)
        .map(|col| {
            instance.claims[col]
                .point
                .iter()
                .map(diagonal_value)
                .collect()
        })
        .collect();

    let delta_proj = delta_times(delta, t1_terminal);
    let delta_batcher = SlotBatcher::new(delta);
    let proj_batcher = SlotBatcher::new(&delta_proj);
    let [d0, d1] = SlotBatcher::coefficient_rows(delta);
    let [p0, p1] = SlotBatcher::coefficient_rows(&delta_proj);
    let rows: [[u64; DEGREE]; 4] = [d0, d1, p0, p1];
    let [f0, f1, q0, q1] = unsafe { dot_rows_i16(witness_16, &rows) };
    let [a0, a1] = unsafe { dot_rows_pass(witness.col(ACCUMULATOR_COL), &delta_batcher.rows()) };
    let projected = QeVec::from_limbs(q0, q1);
    let column_table = |col: usize| -> QeVec {
        if col == ACCUMULATOR_COL {
            QeVec::from_limbs(a0.clone(), a1.clone())
        } else {
            QeVec::from_limbs(
                f0[col * m..(col + 1) * m].to_vec(),
                f1[col * m..(col + 1) * m].to_vec(),
            )
        }
    };

    let low_s = leaf(s.to_vec(), 0);
    let low_projected = leaf(projected.contract_top(block_eq).to_vec(), 0);
    let low_evals: Vec<[Leaf; 2]> = (0..FOLD_INPUTS)
        .map(|col| {
            let top_eq = expand_eq_qe(&points[col][..top_vars]);
            [
                leaf(expand_eq_qe(&points[col][top_vars..]), 0),
                leaf(column_table(col).contract_top(&top_eq).to_vec(), 0),
            ]
        })
        .collect();
    let low = stage(
        product(&low_s, &low_projected),
        vec![low_s.clone(), low_projected],
        low_evals,
        &eval_batching,
        &gamma,
    );
    let execution = execute_sumcheck_prover(&low.combiner, &low.leaves, column_vars, transcript);
    let mut round_polynomials = execution.round_polynomials;
    let low_points = execution.field_points;
    let s_final = *low_s.borrow().final_evaluations();
    let low_scales: Vec<QuadraticExtension> = (0..FOLD_INPUTS)
        .map(|col| *low.leaves[2 + 2 * col].borrow().final_evaluations())
        .collect();
    drop(low);

    let low_msb: Vec<QuadraticExtension> = low_points.iter().rev().cloned().collect();
    let low_eq = expand_eq_soa(&low_msb);
    let alpha = embed_qe(&QuadraticExtension { coeffs: [0, 1] });
    let mut partial: Vec<Vec<RingElement>> = unsafe { contract_i16(witness_16, &low_eq, &alpha) }
        .chunks_exact(block_eq.len() / FRESH_INPUTS)
        .map(|column| column.to_vec())
        .collect();
    partial.push(unsafe { contract_pass(witness.col(ACCUMULATOR_COL), &low_eq, &alpha) });

    let top_s = leaf(vec![s_final], block_vars);
    let top_block = leaf(block_eq.to_vec(), 0);
    let top_projected = leaf(
        partial[..FRESH_INPUTS]
            .iter()
            .flat_map(|column| column.iter().map(|u| proj_batcher.apply(u)))
            .collect(),
        0,
    );
    let top_evals: Vec<[Leaf; 2]> = (0..FOLD_INPUTS)
        .map(|col| {
            let layers = weight_layers(col, &points[col][..top_vars]);
            let mut weight = expand_eq_qe(&layers);
            for entry in &mut weight {
                *entry *= &low_scales[col];
            }
            [
                leaf(weight, 0),
                leaf(
                    partial[col]
                        .iter()
                        .map(|u| delta_batcher.apply(u))
                        .collect(),
                    FRESH_SELECTOR_VARS,
                ),
            ]
        })
        .collect();
    let inner = product(&top_s, &top_block);
    let outer: Node = ElephantCell::new(ProductSumcheck::new(inner, top_projected.clone()));
    let top = stage(
        outer,
        vec![top_s, top_block, top_projected],
        top_evals,
        &eval_batching,
        &gamma,
    );
    let execution = execute_sumcheck_prover(&top.combiner, &top.leaves, block_vars, transcript);
    round_polynomials.extend(execution.round_polynomials);
    drop(top);

    let top_msb: Vec<QuadraticExtension> = execution
        .field_points
        .iter()
        .rev()
        .skip(FRESH_SELECTOR_VARS)
        .cloned()
        .collect();
    let top_eq = expand_eq_soa(&top_msb);
    let terminal_values: Vec<RingElement> = partial
        .iter()
        .map(|column| evaluate_column(column, &top_eq))
        .collect();
    transcript.update_with_ring_element_slice(&terminal_values);
    WitnessStageOutput {
        round_polynomials,
        terminal_values,
    }
}
