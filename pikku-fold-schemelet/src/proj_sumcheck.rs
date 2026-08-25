use crate::qe_vec::QeVec;
use rokoko::common::config::MOD_Q;
use rokoko::common::projection_matrix::ProjectionMatrix;
use rokoko::common::ring_arithmetic::{QuadraticExtension, Representation, RingElement};
use rokoko::common::structured_row::PreprocessedRow;
use rokoko::common::sumcheck_element::SumcheckElement;

pub(crate) use rokoko::protocol::snark::{embed_qe, eq_layers_qe};

const _: () = assert!(
    crate::config::PROJECTION_LAYERS == 3,
    "the chained sumcheck stages assume two coarse layers and one fine layer"
);

pub(crate) fn one_minus(value: &RingElement) -> RingElement {
    let mut out = RingElement::constant(1, Representation::IncompleteNTT);
    out -= value;
    out
}

#[allow(dead_code)]
pub(crate) fn expand_eq_qe(layers_msb: &[QuadraticExtension]) -> Vec<QuadraticExtension> {
    PreprocessedRow::from_layers(layers_msb).preprocessed_row
}

// Lazy accumulation: height * (q - 1) fits in a u64, so the eq-weighted
// column sums run as raw adds off the bit planes and reduce once at the end.
pub(crate) fn accumulate_j_columns(matrix: &ProjectionMatrix, row_weights: &QeVec) -> QeVec {
    assert_eq!(row_weights.len(), matrix.projection_height);
    assert!(matrix.projection_height as u128 * (MOD_Q as u128 - 1) < u64::MAX as u128);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        accumulate_j_columns_avx512(matrix, row_weights)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        let weights: Vec<QuadraticExtension> =
            (0..row_weights.len()).map(|i| row_weights.get(i)).collect();
        let acc = accumulate_j_columns_scalar(matrix, &weights);
        QeVec {
            limb0: acc.iter().map(|v| v.coeffs[0]).collect(),
            limb1: acc.iter().map(|v| v.coeffs[1]).collect(),
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
unsafe fn accumulate_j_columns_avx512(matrix: &ProjectionMatrix, row_weights: &QeVec) -> QeVec {
    use std::arch::x86_64::*;

    let height = matrix.projection_height;
    let width = height * matrix.projection_ratio;
    let bytes = matrix.width;
    assert_eq!(bytes % 8, 0);
    assert!(height as u64 * (MOD_Q / 2 + 1) < 1u64 << 62);
    let pos_ptr = matrix.pos_masks.data.as_ptr();
    let nz_ptr = matrix.non_zero_masks.data.as_ptr();

    let centered = |v: u64| -> i64 {
        if v > MOD_Q / 2 {
            -((MOD_Q - v) as i64)
        } else {
            v as i64
        }
    };
    let w0: Vec<i64> = row_weights.limb0.iter().map(|&v| centered(v)).collect();
    let w1: Vec<i64> = row_weights.limb1.iter().map(|&v| centered(v)).collect();

    let bias = (MOD_Q << 8) as i64;
    let mut limb0 = vec![0u64; width];
    let mut limb1 = vec![0u64; width];

    for tile in 0..bytes / 8 {
        let mut acc = [_mm512_setzero_si512(); 16];
        for row in 0..height {
            let byte_index = row * bytes + tile * 8;
            let pos8 = (pos_ptr.add(byte_index) as *const u64).read_unaligned();
            let nz8 = (nz_ptr.add(byte_index) as *const u64).read_unaligned();
            let add8 = nz8 & pos8;
            let sub8 = nz8 & !pos8;
            let v0 = _mm512_set1_epi64(w0[row]);
            let v1 = _mm512_set1_epi64(w1[row]);
            for byte in 0..8 {
                let add: __mmask8 = (add8 >> (8 * byte)) as u8;
                let sub: __mmask8 = (sub8 >> (8 * byte)) as u8;
                acc[2 * byte] = _mm512_mask_add_epi64(acc[2 * byte], add, acc[2 * byte], v0);
                acc[2 * byte] = _mm512_mask_sub_epi64(acc[2 * byte], sub, acc[2 * byte], v0);
                acc[2 * byte + 1] =
                    _mm512_mask_add_epi64(acc[2 * byte + 1], add, acc[2 * byte + 1], v1);
                acc[2 * byte + 1] =
                    _mm512_mask_sub_epi64(acc[2 * byte + 1], sub, acc[2 * byte + 1], v1);
            }
        }
        let bias_v = _mm512_set1_epi64(bias);
        for byte in 0..8 {
            let col = tile * 64 + byte * 8;
            _mm512_storeu_si512(
                limb0.as_mut_ptr().add(col) as *mut __m512i,
                _mm512_add_epi64(acc[2 * byte], bias_v),
            );
            _mm512_storeu_si512(
                limb1.as_mut_ptr().add(col) as *mut __m512i,
                _mm512_add_epi64(acc[2 * byte + 1], bias_v),
            );
        }
    }

    rexl_reduce(&mut limb0);
    rexl_reduce(&mut limb1);
    QeVec { limb0, limb1 }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn rexl_reduce(values: &mut [u64]) {
    let operand = values.to_vec();
    incomplete_rexl::eltwise_reduce_mod(values, &operand, MOD_Q);
}

#[allow(dead_code)]
pub(crate) fn accumulate_j_columns_scalar(
    matrix: &ProjectionMatrix,
    row_weights: &[QuadraticExtension],
) -> Vec<QuadraticExtension> {
    let width = matrix.projection_height * matrix.projection_ratio;
    let mut acc = vec![QuadraticExtension::zero(); width];
    for (row, weight) in row_weights.iter().enumerate() {
        let (pos, non_zero) = matrix.row_chunks(row);
        for (byte, (&pos_byte, &nz_byte)) in pos.iter().zip(non_zero).enumerate() {
            if nz_byte == 0 {
                continue;
            }
            for bit in 0..8 {
                if (nz_byte >> bit) & 1 == 1 {
                    let col = byte * 8 + bit;
                    if (pos_byte >> bit) & 1 == 1 {
                        acc[col] += weight;
                    } else {
                        acc[col] -= weight;
                    }
                }
            }
        }
    }
    acc
}

#[allow(dead_code)]
pub(crate) fn dot_qe(a: &[QuadraticExtension], b: &[QuadraticExtension]) -> QuadraticExtension {
    let mut acc = QuadraticExtension::zero();
    let mut tmp = QuadraticExtension::zero();
    for (x, y) in a.iter().zip(b) {
        tmp *= (x, y);
        acc += &tmp;
    }
    acc
}

pub(crate) fn build_l0(
    j_batched: &[Vec<RingElement>],
    proj_batching: &[RingElement],
) -> Vec<RingElement> {
    let width = j_batched[0].len();
    let mut out = vec![RingElement::zero(Representation::IncompleteNTT); width];
    let mut tmp = RingElement::zero(Representation::IncompleteNTT);
    for (vector, challenge) in j_batched.iter().zip(proj_batching) {
        for (acc, value) in out.iter_mut().zip(vector) {
            tmp *= (value, challenge);
            *acc += &tmp;
        }
    }
    out
}

pub(crate) fn scaled_embedded_table(values: &QeVec, scale: &RingElement) -> Vec<RingElement> {
    let mut out = Vec::with_capacity(values.len());
    let mut tmp = RingElement::zero(Representation::IncompleteNTT);
    for index in 0..values.len() {
        let embedded = embed_qe(&values.get(index));
        tmp *= (&embedded, scale);
        out.push(tmp.clone());
    }
    out
}
