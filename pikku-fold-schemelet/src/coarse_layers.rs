// Coarse layers in the coefficient domain. Layer 0 precomputes, per L2 tile,
// the 40 signed combinations of each group of four witness elements, so a row
// costs one add per group instead of one per nonzero.
#![allow(clippy::needless_range_loop)]
use crate::config::WITNESS_COEFF_BOUND;
use rokoko::common::arithmetic::centered_i16_from_u64_mod_q;
use rokoko::common::config::HALF_DEGREE;
use rokoko::common::config::{DEGREE, MOD_Q};
use rokoko::common::matrix::VerticallyAlignedMatrix;
use rokoko::common::projection_matrix::ProjectionMatrix;
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::hexl::bindings::ntt_inverse;
use rokoko::protocol::project_coarse::Signed16RingElement;
use std::arch::x86_64::*;

pub(crate) fn prepare_i16(elements: &[RingElement]) -> Vec<Signed16RingElement> {
    let mut out = vec![Signed16RingElement([0i16; DEGREE]); elements.len()];
    let mut buffer = RingElement::zero(Representation::IncompleteNTT);
    for (dst, src) in out.iter_mut().zip(elements) {
        unsafe {
            ntt_inverse(buffer.v.as_mut_ptr(), src.v.as_ptr(), HALF_DEGREE, MOD_Q);
            ntt_inverse(
                buffer.v.as_mut_ptr().add(HALF_DEGREE),
                src.v.as_ptr().add(HALF_DEGREE),
                HALF_DEGREE,
                MOD_Q,
            );
        }
        centered_i16_from_u64_mod_q(&mut dst.0, &buffer.v);
    }
    out
}

const GROUP: usize = 4;
const COMBOS: usize = 40;
// Seven groups of four coefficients bounded by 2^10 stay inside i16.
const SUB_GROUPS: usize = 7;
const SUB_TILES: usize = 4;
const TILE_GROUPS: usize = SUB_GROUPS * SUB_TILES;
const LANES: usize = DEGREE;
const ENTRY_BYTES: usize = LANES * 2;
const ZMM_PER_ENTRY: usize = LANES / 32;
const ZMM_PER_ACC: usize = LANES / 16;
const _: () = assert!(SUB_GROUPS as u64 * GROUP as u64 * WITNESS_COEFF_BOUND <= i16::MAX as u64);

#[derive(Clone, Copy)]
#[repr(align(64))]
pub(crate) struct I32Element(pub [i32; LANES]);

// Generation order of build_tile_table, keyed by the base-3 sign code.
fn combo_index_table() -> [u8; 81] {
    let mut table = [u8::MAX; 81];
    let mut next = 0u8;
    let mut visit = |signs: [i8; GROUP]| {
        let code = signs
            .iter()
            .enumerate()
            .map(|(i, &s)| (s + 1) as usize * 3usize.pow(i as u32))
            .sum::<usize>();
        table[code] = next;
        next += 1;
    };
    for i in 0..GROUP {
        let mut s1 = [0i8; GROUP];
        s1[i] = 1;
        visit(s1);
        for j in i + 1..GROUP {
            for sj in [1i8, -1] {
                let mut s2 = s1;
                s2[j] = sj;
                visit(s2);
                for k in j + 1..GROUP {
                    for sk in [1i8, -1] {
                        let mut s3 = s2;
                        s3[k] = sk;
                        visit(s3);
                        for l in k + 1..GROUP {
                            for sl in [1i8, -1] {
                                let mut s4 = s3;
                                s4[l] = sl;
                                visit(s4);
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(next as usize, COMBOS);
    table
}

const SIGN_BIT: u32 = 1 << 31;

// One slot per (tile, row, group): the byte offset of the group's combination
// in the tile table, sign in the top bit; all-zero groups point at the zero
// entry past the table, so every loop has a fixed trip count.
pub(crate) struct Layer0Plan {
    height: usize,
    row_len: usize,
    tile_groups: Vec<usize>,
    slots: Vec<u32>,
}

impl Layer0Plan {
    pub(crate) fn new(matrix: &ProjectionMatrix) -> Self {
        let height = matrix.projection_height;
        let row_len = matrix.projection_ratio * height;
        assert_eq!(row_len % GROUP, 0);
        assert!((row_len as u64) * WITNESS_COEFF_BOUND < i32::MAX as u64);
        let groups = row_len / GROUP;
        let tiles = groups.div_ceil(TILE_GROUPS);
        let tile_groups: Vec<usize> = (0..tiles)
            .map(|tile| (groups - tile * TILE_GROUPS).min(TILE_GROUPS))
            .collect();
        let index_of = combo_index_table();
        let zero_entry = (TILE_GROUPS * COMBOS * ENTRY_BYTES) as u32;
        let mut signs = vec![0i8; row_len];
        let mut rows: Vec<Vec<i8>> = Vec::with_capacity(height);
        for row in 0..height {
            let (pos_bits, nz_bits) = matrix.row_chunks(row);
            for (byte, (&p, &n)) in pos_bits.iter().zip(nz_bits).enumerate() {
                for bit in 0..8 {
                    let col = byte * 8 + bit;
                    if col >= row_len {
                        break;
                    }
                    signs[col] = if (n >> bit) & 1 == 0 {
                        0
                    } else if (p >> bit) & 1 == 1 {
                        1
                    } else {
                        -1
                    };
                }
            }
            rows.push(signs.clone());
        }
        let mut slots = vec![zero_entry; tiles * height * TILE_GROUPS];
        for tile in 0..tiles {
            for (row, row_signs) in rows.iter().enumerate() {
                let base_slot = (tile * height + row) * TILE_GROUPS;
                for g in 0..tile_groups[tile] {
                    let base = (tile * TILE_GROUPS + g) * GROUP;
                    let s = &row_signs[base..base + GROUP];
                    let Some(sign) = s.iter().copied().find(|&v| v != 0) else {
                        continue;
                    };
                    let code = s
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (v * sign + 1) as usize * 3usize.pow(i as u32))
                        .sum::<usize>();
                    let offset = ((g * COMBOS + index_of[code] as usize) * ENTRY_BYTES) as u32;
                    slots[base_slot + g] = if sign > 0 { offset } else { offset | SIGN_BIT };
                }
            }
        }
        Layer0Plan {
            height,
            row_len,
            tile_groups,
            slots,
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn build_tile_table(inputs: *const Signed16RingElement, groups: usize, table: *mut u8) {
    for g in 0..groups {
        let w = inputs.add(g * GROUP) as *const i16;
        let out = table.add(g * COMBOS * ENTRY_BYTES) as *mut i16;
        for b in 0..ZMM_PER_ENTRY {
            let load = |i: usize| _mm512_loadu_si512(w.add(i * LANES + b * 32) as *const __m512i);
            let v = [load(0), load(1), load(2), load(3)];
            let mut next = 0usize;
            let mut store = |value: __m512i| {
                _mm512_storeu_si512(out.add(next * LANES + b * 32) as *mut __m512i, value);
                next += 1;
            };
            for i in 0..GROUP {
                let s1 = v[i];
                store(s1);
                for j in i + 1..GROUP {
                    for sj in 0..2 {
                        let s2 = if sj == 0 {
                            _mm512_add_epi16(s1, v[j])
                        } else {
                            _mm512_sub_epi16(s1, v[j])
                        };
                        store(s2);
                        for k in j + 1..GROUP {
                            for sk in 0..2 {
                                let s3 = if sk == 0 {
                                    _mm512_add_epi16(s2, v[k])
                                } else {
                                    _mm512_sub_epi16(s2, v[k])
                                };
                                store(s3);
                                for l in k + 1..GROUP {
                                    store(_mm512_add_epi16(s3, v[l]));
                                    store(_mm512_sub_epi16(s3, v[l]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn accumulate_row(table: *const u8, slots: &[u32; TILE_GROUPS], acc: *mut i32) {
    let zero = _mm512_setzero_si512();
    let mut a = [_mm512_setzero_si512(); ZMM_PER_ACC];
    for b in 0..ZMM_PER_ACC {
        a[b] = _mm512_load_si512(acc.add(b * 16) as *const __m512i);
    }
    for sub in slots.chunks_exact(SUB_GROUPS) {
        let mut narrow = [_mm512_setzero_si512(); ZMM_PER_ENTRY];
        for &slot in sub {
            let entry = table.add((slot & !SIGN_BIT) as usize) as *const __m512i;
            let negate = ((slot as i32) >> 31) as u32;
            for b in 0..ZMM_PER_ENTRY {
                let e = _mm512_loadu_si512(entry.add(b));
                let e = _mm512_mask_sub_epi16(e, negate, zero, e);
                narrow[b] = _mm512_add_epi16(narrow[b], e);
            }
        }
        for b in 0..ZMM_PER_ENTRY {
            let low = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(narrow[b]));
            let high = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(narrow[b], 1));
            a[2 * b] = _mm512_add_epi32(a[2 * b], low);
            a[2 * b + 1] = _mm512_add_epi32(a[2 * b + 1], high);
        }
    }
    for b in 0..ZMM_PER_ACC {
        _mm512_store_si512(acc.add(b * 16) as *mut __m512i, a[b]);
    }
}

pub(crate) fn project_layer0(
    witness_16: &[Signed16RingElement],
    plan: &Layer0Plan,
) -> Vec<I32Element> {
    let chunks = witness_16.len() / plan.row_len;
    assert_eq!(chunks * plan.row_len, witness_16.len());
    let mut image = vec![I32Element([0; LANES]); chunks * plan.height];
    let mut table = vec![Signed16RingElement([0; LANES]); TILE_GROUPS * COMBOS + 1];
    let table_ptr = table.as_mut_ptr() as *mut u8;
    let tile_len = GROUP * TILE_GROUPS;
    for chunk in 0..chunks {
        let acc = &mut image[chunk * plan.height..(chunk + 1) * plan.height];
        for (tile, &groups) in plan.tile_groups.iter().enumerate() {
            unsafe {
                build_tile_table(
                    witness_16
                        .as_ptr()
                        .add(chunk * plan.row_len + tile * tile_len),
                    groups,
                    table_ptr,
                );
            }
            for (row, acc_row) in acc.iter_mut().enumerate() {
                let slot = (tile * plan.height + row) * TILE_GROUPS;
                let slots = plan.slots[slot..slot + TILE_GROUPS].try_into().unwrap();
                unsafe { accumulate_row(table_ptr, slots, acc_row.0.as_mut_ptr()) };
            }
        }
    }
    image
}

pub(crate) fn project_layer1(input: &[I32Element], matrix: &ProjectionMatrix) -> Vec<[i64; LANES]> {
    let height = matrix.projection_height;
    let row_len = matrix.projection_ratio * height;
    assert_eq!(input.len(), row_len);
    let mut out = vec![[0i64; LANES]; height];
    for (row, acc) in out.iter_mut().enumerate() {
        let (pos_bits, nz_bits) = matrix.row_chunks(row);
        let mut pos = Vec::with_capacity(row_len / 4);
        let mut neg = Vec::with_capacity(row_len / 4);
        for (byte, (&p, &n)) in pos_bits.iter().zip(nz_bits).enumerate() {
            for bit in 0..8 {
                let col = byte * 8 + bit;
                if col < row_len && (n >> bit) & 1 == 1 {
                    if (p >> bit) & 1 == 1 {
                        pos.push(col as u32);
                    } else {
                        neg.push(col as u32);
                    }
                }
            }
        }
        unsafe { accumulate_row_i64(input, &pos, &neg, acc) };
    }
    out
}

#[target_feature(enable = "avx512f")]
unsafe fn accumulate_row_i64(
    input: &[I32Element],
    pos: &[u32],
    neg: &[u32],
    acc: &mut [i64; LANES],
) {
    const ZMM: usize = LANES / 8;
    let mut a = [_mm512_setzero_si512(); ZMM];
    for &index in pos {
        let src = input.as_ptr().add(index as usize) as *const __m256i;
        for b in 0..ZMM {
            a[b] = _mm512_add_epi64(a[b], _mm512_cvtepi32_epi64(_mm256_load_si256(src.add(b))));
        }
    }
    for &index in neg {
        let src = input.as_ptr().add(index as usize) as *const __m256i;
        for b in 0..ZMM {
            a[b] = _mm512_sub_epi64(a[b], _mm512_cvtepi32_epi64(_mm256_load_si256(src.add(b))));
        }
    }
    for b in 0..ZMM {
        _mm512_storeu_si512(acc.as_mut_ptr().add(b * 8) as *mut __m512i, a[b]);
    }
}

fn lift(values: impl Iterator<Item = i64>) -> RingElement {
    let mut element = RingElement::zero(Representation::IncompleteNTT);
    element.from_incomplete_ntt_to_even_odd_coefficients();
    for (slot, value) in element.v.iter_mut().zip(values) {
        *slot = if value >= 0 {
            value as u64
        } else {
            MOD_Q - value.unsigned_abs()
        };
    }
    element.from_even_odd_coefficients_to_incomplete_ntt_representation();
    element
}

pub(crate) fn lift_i32(image: &[I32Element]) -> VerticallyAlignedMatrix<RingElement> {
    let data: Vec<RingElement> = image
        .iter()
        .map(|e| lift(e.0.iter().map(|&v| v as i64)))
        .collect();
    VerticallyAlignedMatrix {
        height: data.len(),
        width: 1,
        used_cols: 1,
        data,
    }
}

pub(crate) fn lift_i64(image: &[[i64; LANES]]) -> VerticallyAlignedMatrix<RingElement> {
    let data: Vec<RingElement> = image.iter().map(|e| lift(e.iter().copied())).collect();
    VerticallyAlignedMatrix {
        height: data.len(),
        width: 1,
        used_cols: 1,
        data,
    }
}
