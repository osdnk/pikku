use crate::config::{ACCUMULATOR_COL, FOLD_INPUTS, FRESH_SELECTOR_VARS};
use crate::eval::eq_table;
use crate::statement::Instance;
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::common::structured_row::StructuredRow;
use rokoko::protocol::sumcheck_utils::combiner::Combiner;
use rokoko::protocol::sumcheck_utils::common::{
    EvaluationSumcheckData, HighOrderSumcheckData, SumcheckBaseData,
};
use rokoko::protocol::sumcheck_utils::elephant_cell::ElephantCell;
use rokoko::protocol::sumcheck_utils::linear::{
    LinearSumcheck, StructuredRowEvaluationLinearSumcheck,
};
use rokoko::protocol::sumcheck_utils::product::ProductSumcheck;

pub(crate) struct EvalClaimGadgets {
    weight_leaves: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    witness_leaves: Vec<ElephantCell<LinearSumcheck<RingElement>>>,
    pub(crate) combiner: ElephantCell<Combiner<RingElement>>,
}

impl EvalClaimGadgets {
    pub(crate) fn leaves(&self) -> Vec<ElephantCell<LinearSumcheck<RingElement>>> {
        self.weight_leaves
            .iter()
            .chain(&self.witness_leaves)
            .cloned()
            .collect()
    }

    pub(crate) fn terminal_values(&self) -> Vec<RingElement> {
        self.witness_leaves
            .iter()
            .map(|leaf| leaf.borrow().final_evaluations().clone())
            .collect()
    }
}

pub(crate) fn weight_layers(col: usize, point: &[RingElement]) -> Vec<RingElement> {
    let selector = if col == ACCUMULATOR_COL { 0 } else { col };
    let mut layers = Vec::with_capacity(FRESH_SELECTOR_VARS + point.len());
    for bit in (0..FRESH_SELECTOR_VARS).rev() {
        layers.push(RingElement::constant(
            ((selector >> bit) & 1) as u64,
            Representation::IncompleteNTT,
        ));
    }
    layers.extend_from_slice(point);
    layers
}

pub(crate) fn form_eval_claims(
    m: usize,
    instance: &Instance,
    witness: &VerticallyAlignedMatrix<RingElement>,
    batching: &[RingElement],
) -> EvalClaimGadgets {
    let mut weight_leaves = Vec::with_capacity(FOLD_INPUTS);
    let mut witness_leaves = Vec::with_capacity(FOLD_INPUTS);
    let mut products: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> =
        Vec::with_capacity(FOLD_INPUTS);
    for col in 0..FOLD_INPUTS {
        let weight = ElephantCell::new(LinearSumcheck::from_data(eq_table(&weight_layers(
            col,
            &instance.claims[col].point,
        ))));
        let mut witness_leaf =
            LinearSumcheck::new_with_prefixed_sufixed_data(m, FRESH_SELECTOR_VARS, 0);
        witness_leaf.load_from(witness.col(col));
        let witness_leaf = ElephantCell::new(witness_leaf);
        products.push(ElephantCell::new(ProductSumcheck::new(
            weight.clone(),
            witness_leaf.clone(),
        )));
        weight_leaves.push(weight);
        witness_leaves.push(witness_leaf);
    }
    let mut combiner = Combiner::new(products);
    combiner.load_challenges_from(batching);
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
