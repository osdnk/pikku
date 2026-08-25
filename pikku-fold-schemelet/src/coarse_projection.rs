use crate::config::WITNESS_COEFF_BOUND;
use rokoko::common::config::{DEGREE, MOD_Q};
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::projection_matrix::ProjectionMatrix;
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::protocol::project_coarse::prepare_i16_witness;

pub(crate) fn project_first_coarse(
    witness: &VerticallyAlignedMatrix<RingElement>,
    matrix: &ProjectionMatrix,
) -> VerticallyAlignedMatrix<RingElement> {
    let height = matrix.projection_height;
    let row_len = matrix.projection_ratio * height;
    assert!(row_len as u64 * WITNESS_COEFF_BOUND < i32::MAX as u64);

    let witness_16 = prepare_i16_witness(witness);
    let out_height = witness.height / matrix.projection_ratio;
    let mut image = VerticallyAlignedMatrix {
        data: vec![RingElement::zero(Representation::IncompleteNTT); out_height],
        width: 1,
        height: out_height,
        used_cols: 1,
    };
    for element in image.data.iter_mut() {
        element.from_incomplete_ntt_to_even_odd_coefficients();
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        use rokoko::protocol::project_coarse::signed_offset_lists;
        let (pos, pos_bounds, neg, neg_bounds) = signed_offset_lists(matrix);
        for chunk in 0..out_height / height {
            let base = witness_16
                .col_slice(0, chunk * row_len, (chunk + 1) * row_len)
                .as_ptr() as *const u8;
            for inner_row in 0..height {
                unsafe {
                    project_row_i32_avx512(
                        base,
                        &pos[pos_bounds[inner_row]..pos_bounds[inner_row + 1]],
                        &neg[neg_bounds[inner_row]..neg_bounds[inner_row + 1]],
                        &mut image.data[chunk * height + inner_row].v,
                    );
                }
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        assert!(row_len <= u16::MAX as usize + 1);
        let mut pos_by_row = vec![Vec::new(); height];
        let mut neg_by_row = vec![Vec::new(); height];
        for row in 0..height {
            let (pos, non_zero) = matrix.row_chunks(row);
            for (byte, (&pos_byte, &nz_byte)) in pos.iter().zip(non_zero).enumerate() {
                if nz_byte == 0 {
                    continue;
                }
                for bit in 0..8 {
                    if (nz_byte >> bit) & 1 == 1 {
                        let col = (byte * 8 + bit) as u16;
                        if (pos_byte >> bit) & 1 == 1 {
                            pos_by_row[row].push(col);
                        } else {
                            neg_by_row[row].push(col);
                        }
                    }
                }
            }
        }
        for chunk in 0..out_height / height {
            let subwitness = witness_16.col_slice(0, chunk * row_len, (chunk + 1) * row_len);
            for inner_row in 0..height {
                project_row_i32_scalar(
                    subwitness,
                    &pos_by_row[inner_row],
                    &neg_by_row[inner_row],
                    &mut image.data[chunk * height + inner_row].v,
                );
            }
        }
    }

    for element in image.data.iter_mut() {
        element.from_even_odd_coefficients_to_incomplete_ntt_representation();
    }
    image
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
unsafe fn project_row_i32_avx512(
    base: *const u8,
    pos: &[u32],
    neg: &[u32],
    out: &mut [u64; DEGREE],
) {
    use std::arch::x86_64::*;

    const BATCHES: usize = DEGREE / 16;
    let mut acc = [_mm512_setzero_si512(); BATCHES];
    for &offset in pos {
        let ptr = base.add(offset as usize) as *const __m256i;
        for (batch, slot) in acc.iter_mut().enumerate() {
            let values = _mm512_cvtepi16_epi32(_mm256_loadu_si256(ptr.add(batch)));
            *slot = _mm512_add_epi32(*slot, values);
        }
    }
    for &offset in neg {
        let ptr = base.add(offset as usize) as *const __m256i;
        for (batch, slot) in acc.iter_mut().enumerate() {
            let values = _mm512_cvtepi16_epi32(_mm256_loadu_si256(ptr.add(batch)));
            *slot = _mm512_sub_epi32(*slot, values);
        }
    }

    let mut lanes = [0i32; DEGREE];
    for (batch, slot) in acc.iter().enumerate() {
        _mm512_storeu_si512(lanes.as_mut_ptr().add(batch * 16) as *mut __m512i, *slot);
    }
    for (slot, &value) in out.iter_mut().zip(&lanes) {
        *slot = if value >= 0 {
            value as u64
        } else {
            MOD_Q - value.unsigned_abs() as u64
        };
    }
}

#[allow(dead_code)]
fn project_row_i32_scalar(
    subwitness: &[rokoko::protocol::project_coarse::Signed16RingElement],
    pos: &[u16],
    neg: &[u16],
    out: &mut [u64; DEGREE],
) {
    let mut acc = [0i32; DEGREE];
    for &index in pos {
        for (slot, &value) in acc.iter_mut().zip(&subwitness[index as usize].0) {
            *slot += value as i32;
        }
    }
    for &index in neg {
        for (slot, &value) in acc.iter_mut().zip(&subwitness[index as usize].0) {
            *slot -= value as i32;
        }
    }
    for (slot, &value) in out.iter_mut().zip(&acc) {
        *slot = if value >= 0 {
            value as u64
        } else {
            MOD_Q - value.unsigned_abs() as u64
        };
    }
}
