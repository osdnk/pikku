use crate::config::{ACCUMULATOR_COL, FOLD_INPUTS, FRESH_SELECTOR_VARS};
use crate::field_sumcheck::{diagonal_value, SlotBatcher};
use crate::proj_sumcheck::expand_eq_qe;
use crate::statement::Instance;
use rokoko::common::config::HALF_DEGREE;
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::ring_arithmetic::{QuadraticExtension, Representation, RingElement};
use rokoko::common::structured_row::StructuredRow;
use rokoko::common::sumcheck_element::SumcheckElement;
use rokoko::protocol::sumcheck_utils::combiner::Combiner;
use rokoko::protocol::sumcheck_utils::common::{
    EvaluationSumcheckData, HighOrderSumcheckData,
};
use rokoko::protocol::sumcheck_utils::elephant_cell::ElephantCell;
use rokoko::protocol::sumcheck_utils::linear::{
    LinearSumcheck, StructuredRowEvaluationLinearSumcheck,
};
use rokoko::protocol::sumcheck_utils::product::ProductSumcheck;

pub(crate) struct EvalClaimGadgets {
    weight_leaves: Vec<ElephantCell<LinearSumcheck<QuadraticExtension>>>,
    witness_leaves: Vec<ElephantCell<LinearSumcheck<QuadraticExtension>>>,
    pub(crate) combiner: ElephantCell<Combiner<QuadraticExtension>>,
}

impl EvalClaimGadgets {
    pub(crate) fn leaves(&self) -> Vec<ElephantCell<LinearSumcheck<QuadraticExtension>>> {
        self.weight_leaves
            .iter()
            .chain(&self.witness_leaves)
            .cloned()
            .collect()
    }
}

pub(crate) fn weight_layers<E: SumcheckElement>(col: usize, point: &[E]) -> Vec<E> {
    let selector = if col == ACCUMULATOR_COL { 0 } else { col };
    let mut layers = Vec::with_capacity(FRESH_SELECTOR_VARS + point.len());
    for bit in (0..FRESH_SELECTOR_VARS).rev() {
        layers.push(if (selector >> bit) & 1 == 1 {
            E::one()
        } else {
            E::zero()
        });
    }
    layers.extend_from_slice(point);
    layers
}

pub(crate) fn form_eval_claims(
    m: usize,
    instance: &Instance,
    witness: &VerticallyAlignedMatrix<RingElement>,
    batching: &[RingElement],
    delta: &[QuadraticExtension; HALF_DEGREE],
) -> EvalClaimGadgets {
    let batcher = SlotBatcher::new(delta);
    let mut weight_leaves = Vec::with_capacity(FOLD_INPUTS);
    let mut witness_leaves = Vec::with_capacity(FOLD_INPUTS);
    let mut products: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = QuadraticExtension>>> =
        Vec::with_capacity(FOLD_INPUTS);
    for col in 0..FOLD_INPUTS {
        let point: Vec<QuadraticExtension> =
            instance.claims[col].point.iter().map(diagonal_value).collect();
        let weight = ElephantCell::new(LinearSumcheck::from_data(expand_eq_qe(
            &weight_layers(col, &point),
        )));
        let mut witness_leaf =
            LinearSumcheck::new_with_prefixed_sufixed_data(m, FRESH_SELECTOR_VARS, 0);
        witness_leaf.load_from(&batcher.apply_all(witness.col(col)));
        let witness_leaf = ElephantCell::new(witness_leaf);
        products.push(ElephantCell::new(ProductSumcheck::new(
            weight.clone(),
            witness_leaf.clone(),
        )));
        weight_leaves.push(weight);
        witness_leaves.push(witness_leaf);
    }
    let mut combiner = Combiner::new(products);
    let batching: Vec<QuadraticExtension> = batching.iter().map(diagonal_value).collect();
    combiner.load_challenges_from(&batching);
    EvalClaimGadgets {
        weight_leaves,
        witness_leaves,
        combiner: ElephantCell::new(combiner),
    }
}

pub(crate) fn batched_claim(instance: &Instance, batching: &[RingElement]) -> RingElement {
    let mut claim = RingElement::zero(Representation::IncompleteNTT);
    let mut tmp = RingElement::zero(Representation::IncompleteNTT);
    for (eval_claim, challenge) in instance.claims.iter().zip(batching) {
        tmp *= (&eval_claim.value, challenge);
        claim += &tmp;
    }
    claim
}

pub(crate) fn eval_terminal(
    m: usize,
    instance: &Instance,
    batching: &[RingElement],
    terminal_values: &[RingElement],
    round_points: &Vec<RingElement>,
) -> RingElement {
    let mut terminal = RingElement::zero(Representation::IncompleteNTT);
    let mut tmp = RingElement::zero(Representation::IncompleteNTT);
    let mut weighted = RingElement::zero(Representation::IncompleteNTT);
    for (col, ((claim, challenge), terminal_value)) in instance
        .claims
        .iter()
        .zip(batching)
        .zip(terminal_values)
        .enumerate()
    {
        let mut weight =
            StructuredRowEvaluationLinearSumcheck::<RingElement>::new(m << FRESH_SELECTOR_VARS);
        weight.load_from(StructuredRow {
            tensor_layers: weight_layers(col, &claim.point),
        });
        tmp *= (weight.evaluate(round_points), terminal_value);
        weighted *= (&tmp, challenge);
        terminal += &weighted;
    }
    terminal
}
