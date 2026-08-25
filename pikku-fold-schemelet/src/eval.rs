use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::common::structured_row::PreprocessedRow;
use rokoko::protocol::open::evaluation_point_to_structured_row;

pub(crate) fn eq_table(point: &[RingElement]) -> Vec<RingElement> {
    PreprocessedRow::from_structured_row(&evaluation_point_to_structured_row(point))
        .preprocessed_row
}

pub(crate) fn mle_evaluate(values: &[RingElement], point: &[RingElement]) -> RingElement {
    assert_eq!(values.len(), 1 << point.len());
    let table = eq_table(point);
    let mut acc = RingElement::zero(Representation::IncompleteNTT);
    let mut tmp = RingElement::zero(Representation::IncompleteNTT);
    for (value, weight) in values.iter().zip(&table) {
        tmp *= (value, weight);
        acc += &tmp;
    }
    acc
}
