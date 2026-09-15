use crate::coarse_layers::{
    lift_i32, lift_i64, prepare_i16, project_layer0, project_layer1, Layer0Plan,
};
use crate::config::{FRESH_INPUTS, PROJECTION_BATCH_POINTS, PROJECTION_LAYERS, PROJECTION_ROWS};
use rokoko::common::arithmetic::{
    centered_i64_from_u64_mod_q_scalar, inner_product, precompute_structured_values_fast,
};
use rokoko::common::config::{DEGREE, MOD_Q, NOF_BATCHES};
use rokoko::common::hash::HashWrapper;
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::projection_matrix::ProjectionMatrix;
use rokoko::common::ring_arithmetic::RingElement;
use rokoko::hexl::bindings::{eltwise_mult_mod, eltwise_reduce_mod};
use rokoko::protocol::project_coarse::Signed16RingElement;
use rokoko::protocol::project_fine::{compute_j_batched_collectively, project_coefficients};

pub(crate) const TRACE_RING_LEN: usize = PROJECTION_ROWS / DEGREE;
const BATCH_TENSOR_VARS: usize = PROJECTION_ROWS.ilog2() as usize;

// Balances the coarse shrink logs: the chain must take k*m ring elements down
// to PROJECTION_ROWS before the fine layer, so the total shrink exponent is
// split as evenly as possible across the coarse layers, earlier layers taking
// the remainder. The returned ratios are each layer's width/height fan-in.
pub(crate) fn projection_shape(m: usize) -> Result<[usize; PROJECTION_LAYERS - 1], String> {
    let input_log = (FRESH_INPUTS * m).ilog2() as usize;
    let rows_log = PROJECTION_ROWS.ilog2() as usize;
    if input_log < rows_log + PROJECTION_LAYERS - 1 {
        return Err("witness too small for the projection chain".to_string());
    }
    let shrink_log = input_log - rows_log;
    let coarse_layers = PROJECTION_LAYERS - 1;
    let quotient = shrink_log / coarse_layers;
    let remainder = shrink_log % coarse_layers;
    let mut ratios = [0usize; PROJECTION_LAYERS - 1];
    for (layer, ratio) in ratios.iter_mut().enumerate() {
        *ratio = 1usize << (quotient + usize::from(layer < remainder));
    }
    Ok(ratios)
}

pub(crate) fn sample_projection_matrices(
    coarse_ratios: &[usize; PROJECTION_LAYERS - 1],
    transcript: &mut HashWrapper,
) -> Vec<ProjectionMatrix> {
    let mut matrices = Vec::with_capacity(PROJECTION_LAYERS);
    for ratio in coarse_ratios {
        matrices.push(ProjectionMatrix::new(*ratio, PROJECTION_ROWS));
    }
    matrices.push(ProjectionMatrix::new(DEGREE, PROJECTION_ROWS));
    for matrix in &mut matrices {
        matrix.sample(transcript);
    }
    matrices
}

pub(crate) fn project_witness(
    witness: &VerticallyAlignedMatrix<RingElement>,
    matrices: &[ProjectionMatrix],
) -> (
    Vec<VerticallyAlignedMatrix<RingElement>>,
    Vec<RingElement>,
    Vec<Signed16RingElement>,
) {
    let m = witness.height;
    let witness_16 = prepare_i16(&witness.data[..FRESH_INPUTS * m]);
    let plan = Layer0Plan::new(&matrices[0]);
    let image0 = project_layer0(&witness_16, &plan);
    let image1 = project_layer1(&image0, &matrices[1]);
    let levels = vec![lift_i32(&image0), lift_i64(&image1)];
    let trace = project_coefficients(&levels[1], &matrices[PROJECTION_LAYERS - 1]);
    (levels, trace.data, witness_16)
}

pub(crate) fn sample_batching_tensors(transcript: &mut HashWrapper) -> Vec<Vec<u64>> {
    (0..PROJECTION_BATCH_POINTS)
        .map(|_| {
            let layers: Vec<u64> = (0..BATCH_TENSOR_VARS)
                .map(|_| transcript.sample_u64_mod_q())
                .collect();
            precompute_structured_values_fast(&layers)
        })
        .collect()
}

pub(crate) fn j_batched_vectors(
    matrix: &ProjectionMatrix,
    tensors: &[Vec<u64>],
) -> Vec<Vec<RingElement>> {
    assert_eq!(tensors.len(), NOF_BATCHES);
    let paired: [Vec<u64>; NOF_BATCHES] = [tensors[0].clone(), tensors[1].clone()];
    compute_j_batched_collectively(matrix, &paired).into()
}

pub(crate) fn batched_projections(
    fine_input: &VerticallyAlignedMatrix<RingElement>,
    j_batched: &[Vec<RingElement>],
) -> Vec<RingElement> {
    j_batched
        .iter()
        .map(|vector| inner_product(vector, &fine_input.data))
        .collect()
}

pub(crate) fn trace_values(trace: &[RingElement]) -> Vec<u64> {
    trace.iter().flat_map(|element| element.v).collect()
}

pub(crate) fn coeff_l2_norm(trace: &[RingElement]) -> f64 {
    trace_values(trace)
        .iter()
        .map(|&value| {
            let centered = centered_i64_from_u64_mod_q_scalar(value) as f64;
            centered * centered
        })
        .sum::<f64>()
        .sqrt()
}

pub(crate) fn check_batched_projection(
    tensor: &[u64],
    trace: &[u64],
    batched: &RingElement,
) -> bool {
    let mut products = vec![0u64; trace.len()];
    unsafe {
        eltwise_mult_mod(
            products.as_mut_ptr(),
            tensor.as_ptr(),
            trace.as_ptr(),
            trace.len() as u64,
            MOD_Q,
        );
    }
    let mut expected = products.iter().sum::<u64>();
    unsafe {
        eltwise_reduce_mod(&mut expected, &expected, 1, MOD_Q);
    }
    batched.constant_term_from_incomplete_ntt() == expected
}
