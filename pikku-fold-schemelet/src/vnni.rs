// Passes over the i16 coefficient copy of the fresh columns: residues split
// into four 13-bit chunks so vpdpwssd multiplies i16 coefficients by chunks
// into i32 pair sums, recombined mod q at the end.
#![allow(clippy::needless_range_loop)]
use crate::ifma::{reduce_u128, Q};
use crate::qe_vec::QeVec;
use rokoko::common::config::DEGREE;
use rokoko::common::ring_arithmetic::{Representation, RingElement};
use rokoko::protocol::project_coarse::Signed16RingElement;
use std::arch::x86_64::*;

const CHUNKS: usize = 4;
const CHUNK_BITS: u32 = 13;
const CHUNK_MASK: u64 = (1 << CHUNK_BITS) - 1;
const ZMM: usize = DEGREE / 32;

fn chunks_of(value: u64) -> [i16; CHUNKS] {
    std::array::from_fn(|j| ((value >> (CHUNK_BITS * j as u32)) & CHUNK_MASK) as i16)
}

fn combine_chunks(s: [i64; CHUNKS], bias: u32) -> u64 {
    let mut n = 0u128;
    let mut b = 0u128;
    for j in 0..CHUNKS {
        n += ((s[j] + (1i64 << bias)) as u128) << (CHUNK_BITS * j as u32);
        b += (1u128 << bias) << (CHUNK_BITS * j as u32);
    }
    let n = reduce_u128(n);
    let b = reduce_u128(b);
    if n >= b {
        n - b
    } else {
        n + Q - b
    }
}

#[repr(align(64))]
struct ChunkRows<const K: usize>([[[i16; DEGREE]; CHUNKS]; K]);

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(crate) unsafe fn dot_rows_i16<const K: usize>(
    elements: &[Signed16RingElement],
    rows: &[[u64; DEGREE]; K],
) -> [Vec<u64>; K] {
    let mut chunk_rows = ChunkRows::<K>([[[0i16; DEGREE]; CHUNKS]; K]);
    for k in 0..K {
        for lane in 0..DEGREE {
            let chunks = chunks_of(rows[k][lane]);
            for j in 0..CHUNKS {
                chunk_rows.0[k][j][lane] = chunks[j];
            }
        }
    }
    let mut out: [Vec<u64>; K] = std::array::from_fn(|_| Vec::with_capacity(elements.len()));
    for element in elements {
        let w: [__m512i; ZMM] = std::array::from_fn(|b| {
            _mm512_load_si512(element.0.as_ptr().add(32 * b) as *const __m512i)
        });
        for k in 0..K {
            let mut sums = [0i64; CHUNKS];
            for j in 0..CHUNKS {
                let mut acc = _mm512_setzero_si512();
                for b in 0..ZMM {
                    let r = _mm512_load_si512(
                        chunk_rows.0[k][j].as_ptr().add(32 * b) as *const __m512i
                    );
                    acc = _mm512_dpwssd_epi32(acc, w[b], r);
                }
                sums[j] = _mm512_reduce_add_epi32(acc) as i64;
            }
            out[k].push(combine_chunks(sums, 30));
        }
    }
    out
}

// out[t] = sum_x weights[x] * elements[t * len + x] in incomplete NTT form.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(crate) unsafe fn contract_i16(
    elements: &[Signed16RingElement],
    weights: &QeVec,
    alpha: &RingElement,
) -> Vec<RingElement> {
    const FLUSH: usize = 256;
    let len = weights.len();
    assert_eq!(elements.len() % len, 0);
    assert_eq!(len % FLUSH, 0);
    assert!(len <= 1 << 36);
    let chunk_weights: [[Vec<u32>; CHUNKS]; 2] = std::array::from_fn(|limb| {
        let limb_values = if limb == 0 {
            &weights.limb0
        } else {
            &weights.limb1
        };
        std::array::from_fn(|j| {
            (0..len / 2)
                .map(|pair| {
                    let lo = chunks_of(limb_values[2 * pair])[j] as u16 as u32;
                    let hi = chunks_of(limb_values[2 * pair + 1])[j] as u16 as u32;
                    lo | (hi << 16)
                })
                .collect()
        })
    });
    let mut out = Vec::with_capacity(elements.len() / len);
    for block in elements.chunks_exact(len) {
        let mut wide = [[[0i64; DEGREE]; CHUNKS]; 2];
        for (flush_index, flush) in block.chunks_exact(FLUSH).enumerate() {
            let mut acc = [[[_mm512_setzero_si512(); 2 * ZMM]; CHUNKS]; 2];
            for (p, pair) in flush.chunks_exact(2).enumerate() {
                let pair_index = flush_index * FLUSH / 2 + p;
                let mut a = [_mm512_setzero_si512(); 2 * ZMM];
                for b in 0..ZMM {
                    let x = _mm512_load_si512(pair[0].0.as_ptr().add(32 * b) as *const __m512i);
                    let y = _mm512_load_si512(pair[1].0.as_ptr().add(32 * b) as *const __m512i);
                    a[2 * b] = _mm512_unpacklo_epi16(x, y);
                    a[2 * b + 1] = _mm512_unpackhi_epi16(x, y);
                }
                for limb in 0..2 {
                    for j in 0..CHUNKS {
                        let e = _mm512_set1_epi32(chunk_weights[limb][j][pair_index] as i32);
                        for b in 0..2 * ZMM {
                            acc[limb][j][b] = _mm512_dpwssd_epi32(acc[limb][j][b], a[b], e);
                        }
                    }
                }
            }
            for limb in 0..2 {
                for j in 0..CHUNKS {
                    for b in 0..2 * ZMM {
                        let dst = wide[limb][j].as_mut_ptr().add(16 * b);
                        let low = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(acc[limb][j][b]));
                        let high =
                            _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(acc[limb][j][b], 1));
                        let sum_low =
                            _mm512_add_epi64(_mm512_loadu_si512(dst as *const __m512i), low);
                        let sum_high = _mm512_add_epi64(
                            _mm512_loadu_si512(dst.add(8) as *const __m512i),
                            high,
                        );
                        _mm512_storeu_si512(dst as *mut __m512i, sum_low);
                        _mm512_storeu_si512(dst.add(8) as *mut __m512i, sum_high);
                    }
                }
            }
        }
        // acc index b = 2k + h, lane p: unpack(lo|hi) of zmm k puts element
        // lane 32k + 8(p / 4) + 4h + p % 4 there.
        let mut parts = [
            RingElement::zero(Representation::IncompleteNTT),
            RingElement::zero(Representation::IncompleteNTT),
        ];
        for limb in 0..2 {
            parts[limb].from_incomplete_ntt_to_even_odd_coefficients();
            for b in 0..2 * ZMM {
                for p in 0..16 {
                    let lane = 32 * (b / 2) + 8 * (p / 4) + 4 * (b % 2) + p % 4;
                    let sums: [i64; CHUNKS] = std::array::from_fn(|j| wide[limb][j][16 * b + p]);
                    parts[limb].v[lane] = combine_chunks(sums, 60);
                }
            }
            parts[limb].from_even_odd_coefficients_to_incomplete_ntt_representation();
        }
        let [plain, alpha_part] = parts;
        let mut value = RingElement::zero(Representation::IncompleteNTT);
        value *= (&alpha_part, alpha);
        value += &plain;
        out.push(value);
    }
    out
}
