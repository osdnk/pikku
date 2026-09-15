use crate::config::{ACCUMULATOR_COL, FRESH_SELECTOR_VARS};
use crate::statement::Instance;
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::common::structured_row::StructuredRow;
use rokoko::common::sumcheck_element::SumcheckElement;
use rokoko::protocol::sumcheck_utils::common::EvaluationSumcheckData;
use rokoko::protocol::sumcheck_utils::linear::StructuredRowEvaluationLinearSumcheck;

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
