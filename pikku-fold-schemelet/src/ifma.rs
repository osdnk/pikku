// IFMA passes with lazy reduction: low and high 52-bit product halves are
// accumulated unreduced (lo < 2^63, hi < 2^59 at every reduction point) and
// reduced once through q = 2^50 - C50.
#![allow(clippy::needless_range_loop)]
use crate::qe_vec::QeVec;
use rokoko::common::config::MOD_Q;
use rokoko::common::ring_arithmetic::{Representation, RingElement, SHIFT_FACTORS};
use std::arch::x86_64::*;

pub(crate) const Q: u64 = 1125899906839937;
const C50: u64 = (1 << 50) - Q;
const C52: u64 = (C50 << 2) % Q;
const C102: u64 = ((C52 as u128 * C50 as u128) % Q as u128) as u64;
const M50: u64 = (1 << 50) - 1;
const _: () = assert!(Q > 1 << 49 && C50 < 1 << 13);
pub(crate) const DEGREE: usize = 128;
const GROUPS: usize = DEGREE / 8;

pub(crate) fn check_modulus() {
    assert_eq!(MOD_Q, Q);
}

#[inline]
pub(crate) fn reduce_lo_hi(lo: u64, hi: u64) -> u64 {
    reduce_u128(((hi as u128) << 52) + lo as u128)
}

#[inline]
pub(crate) fn reduce_u128(n: u128) -> u64 {
    debug_assert!(n < 1 << 112);
    let n = (n >> 50) * C50 as u128 + (n & M50 as u128);
    let n = ((n >> 50) as u64) * C50 + (n as u64 & M50);
    let n = (n >> 50) * C50 + (n & M50);
    if n >= Q {
        n - Q
    } else {
        n
    }
}

#[inline]
unsafe fn fold50(v: __m512i) -> __m512i {
    _mm512_add_epi64(
        _mm512_mullo_epi64(_mm512_srli_epi64(v, 50), _mm512_set1_epi64(C50 as i64)),
        _mm512_and_si512(v, _mm512_set1_epi64(M50 as i64)),
    )
}

#[inline]
unsafe fn reduce_lo_hi_v(lo: __m512i, hi: __m512i) -> __m512i {
    let lo = fold50(lo);
    let hi_t = _mm512_srli_epi64(hi, 50);
    let hi_b = _mm512_and_si512(hi, _mm512_set1_epi64(M50 as i64));
    let h = _mm512_add_epi64(
        _mm512_mullo_epi64(hi_t, _mm512_set1_epi64(C102 as i64)),
        _mm512_mullo_epi64(hi_b, _mm512_set1_epi64(C52 as i64)),
    );
    let t = fold50(_mm512_add_epi64(lo, fold50(h)));
    let q = _mm512_set1_epi64(Q as i64);
    _mm512_min_epu64(t, _mm512_sub_epi64(t, q))
}

#[inline]
unsafe fn hsum(v: __m512i) -> u64 {
    _mm512_reduce_add_epi64(v) as u64
}

#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(crate) unsafe fn dot_rows_pass<const K: usize>(
    elements: &[RingElement],
    rows: &[[u64; DEGREE]; K],
) -> [Vec<u64>; K] {
    let mut out: [Vec<u64>; K] = std::array::from_fn(|_| Vec::with_capacity(elements.len()));
    let mut r = [[_mm512_setzero_si512(); GROUPS]; K];
    for k in 0..K {
        for g in 0..GROUPS {
            r[k][g] = _mm512_loadu_si512(rows[k].as_ptr().add(8 * g) as *const __m512i);
        }
    }
    for element in elements {
        let mut lo = [[_mm512_setzero_si512(); 2]; K];
        let mut hi = [[_mm512_setzero_si512(); 2]; K];
        for g in 0..GROUPS {
            let v = _mm512_load_si512(element.v.as_ptr().add(8 * g) as *const __m512i);
            for k in 0..K {
                lo[k][g & 1] = _mm512_madd52lo_epu64(lo[k][g & 1], v, r[k][g]);
                hi[k][g & 1] = _mm512_madd52hi_epu64(hi[k][g & 1], v, r[k][g]);
            }
        }
        for k in 0..K {
            out[k].push(reduce_lo_hi(
                hsum(_mm512_add_epi64(lo[k][0], lo[k][1])),
                hsum(_mm512_add_epi64(hi[k][0], hi[k][1])),
            ));
        }
    }
    out
}

// out[t] = sum_x weights[x] * elements[t * len + x], weights in F_{q^2}.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(crate) unsafe fn contract_pass(
    elements: &[RingElement],
    weights: &QeVec,
    alpha: &RingElement,
) -> Vec<RingElement> {
    const FLUSH: usize = 1 << 11;
    let len = weights.len();
    assert_eq!(elements.len() % len, 0);
    let mut out = Vec::with_capacity(elements.len() / len);
    for block in elements.chunks_exact(len) {
        let mut acc = [[_mm512_setzero_si512(); GROUPS]; 2];
        let mut lo = [[_mm512_setzero_si512(); GROUPS]; 2];
        let mut hi = [[_mm512_setzero_si512(); GROUPS]; 2];
        for (chunk_index, chunk) in block.chunks(FLUSH).enumerate() {
            for (x, element) in chunk.iter().enumerate() {
                let index = chunk_index * FLUSH + x;
                let w = [
                    _mm512_set1_epi64(weights.limb0[index] as i64),
                    _mm512_set1_epi64(weights.limb1[index] as i64),
                ];
                for g in 0..GROUPS {
                    let v = _mm512_load_si512(element.v.as_ptr().add(8 * g) as *const __m512i);
                    for limb in 0..2 {
                        lo[limb][g] = _mm512_madd52lo_epu64(lo[limb][g], v, w[limb]);
                        hi[limb][g] = _mm512_madd52hi_epu64(hi[limb][g], v, w[limb]);
                    }
                }
            }
            for limb in 0..2 {
                for g in 0..GROUPS {
                    let reduced = reduce_lo_hi_v(lo[limb][g], hi[limb][g]);
                    let sum = _mm512_add_epi64(acc[limb][g], reduced);
                    acc[limb][g] =
                        _mm512_min_epu64(sum, _mm512_sub_epi64(sum, _mm512_set1_epi64(Q as i64)));
                    lo[limb][g] = _mm512_setzero_si512();
                    hi[limb][g] = _mm512_setzero_si512();
                }
            }
        }
        let mut plain = RingElement::zero(Representation::IncompleteNTT);
        let mut alpha_part = RingElement::zero(Representation::IncompleteNTT);
        for g in 0..GROUPS {
            _mm512_store_si512(plain.v.as_mut_ptr().add(8 * g) as *mut __m512i, acc[0][g]);
            _mm512_store_si512(
                alpha_part.v.as_mut_ptr().add(8 * g) as *mut __m512i,
                acc[1][g],
            );
        }
        let mut value = RingElement::zero(Representation::IncompleteNTT);
        value *= (&alpha_part, alpha);
        value += &plain;
        out.push(value);
    }
    out
}

// x * y mod q in [0, 2q) with y fixed, y_shoup = floor(y * 2^52 / q).
#[inline]
unsafe fn mul_shoup(x: __m512i, y: __m512i, y_shoup: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512();
    let t = _mm512_madd52hi_epu64(zero, x, y_shoup);
    let lo = _mm512_madd52lo_epu64(zero, x, y);
    let tq = _mm512_madd52lo_epu64(zero, t, _mm512_set1_epi64(Q as i64));
    _mm512_and_si512(
        _mm512_sub_epi64(lo, tq),
        _mm512_set1_epi64((1i64 << 52) - 1),
    )
}

pub(crate) struct FixedMultiplier {
    even: [u64; HALF],
    odd: [u64; HALF],
    shifted_odd: [u64; HALF],
    even_shoup: [u64; HALF],
    odd_shoup: [u64; HALF],
    shifted_odd_shoup: [u64; HALF],
}
const HALF: usize = DEGREE / 2;

fn shoup(y: u64) -> u64 {
    (((y as u128) << 52) / Q as u128) as u64
}

impl FixedMultiplier {
    pub(crate) fn new(y: &RingElement) -> Self {
        assert!(y.representation == Representation::IncompleteNTT);
        let mut out = FixedMultiplier {
            even: [0; HALF],
            odd: [0; HALF],
            shifted_odd: [0; HALF],
            even_shoup: [0; HALF],
            odd_shoup: [0; HALF],
            shifted_odd_shoup: [0; HALF],
        };
        for i in 0..HALF {
            out.even[i] = y.v[i];
            out.odd[i] = y.v[HALF + i];
            out.shifted_odd[i] =
                ((SHIFT_FACTORS[i] as u128 * y.v[HALF + i] as u128) % Q as u128) as u64;
            out.even_shoup[i] = shoup(out.even[i]);
            out.odd_shoup[i] = shoup(out.odd[i]);
            out.shifted_odd_shoup[i] = shoup(out.shifted_odd[i]);
        }
        out
    }
}

// out[row] = base[row] + sum_c multipliers[c] * columns[c][row]; the lazy
// sum stays below (1 + 4K) q.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(crate) unsafe fn fold_pass<const K: usize>(
    base: &[RingElement],
    columns: [&[RingElement]; K],
    multipliers: &[FixedMultiplier; K],
) -> Vec<RingElement> {
    let q = _mm512_set1_epi64(Q as i64);
    let two_q = _mm512_set1_epi64(2 * Q as i64);
    let four_q = _mm512_set1_epi64(4 * Q as i64);
    let eight_q = _mm512_set1_epi64(8 * Q as i64);
    assert!(K <= 4);
    let reduce = |v: __m512i| {
        let v = _mm512_min_epu64(v, _mm512_sub_epi64(v, eight_q));
        let v = _mm512_min_epu64(v, _mm512_sub_epi64(v, four_q));
        let v = _mm512_min_epu64(v, _mm512_sub_epi64(v, two_q));
        _mm512_min_epu64(v, _mm512_sub_epi64(v, q))
    };
    let mut out: Vec<RingElement> = Vec::with_capacity(base.len());
    let slots = out.spare_capacity_mut();
    for row in 0..base.len() {
        let dst = slots[row].as_mut_ptr();
        std::ptr::addr_of_mut!((*dst).representation).write(Representation::IncompleteNTT);
        let dst = std::ptr::addr_of_mut!((*dst).v) as *mut u64;
        for g in 0..HALF / 8 {
            let mut even = _mm512_load_si512(base[row].v.as_ptr().add(8 * g) as *const __m512i);
            let mut odd =
                _mm512_load_si512(base[row].v.as_ptr().add(HALF + 8 * g) as *const __m512i);
            for c in 0..K {
                let m = &multipliers[c];
                let load = |arr: &[u64; HALF]| {
                    _mm512_loadu_si512(arr.as_ptr().add(8 * g) as *const __m512i)
                };
                let a = _mm512_load_si512(columns[c][row].v.as_ptr().add(8 * g) as *const __m512i);
                let b = _mm512_load_si512(
                    columns[c][row].v.as_ptr().add(HALF + 8 * g) as *const __m512i
                );
                even = _mm512_add_epi64(even, mul_shoup(a, load(&m.even), load(&m.even_shoup)));
                even = _mm512_add_epi64(
                    even,
                    mul_shoup(b, load(&m.shifted_odd), load(&m.shifted_odd_shoup)),
                );
                odd = _mm512_add_epi64(odd, mul_shoup(a, load(&m.odd), load(&m.odd_shoup)));
                odd = _mm512_add_epi64(odd, mul_shoup(b, load(&m.even), load(&m.even_shoup)));
            }
            _mm512_stream_si512(dst.add(8 * g) as *mut __m512i, reduce(even));
            _mm512_stream_si512(dst.add(HALF + 8 * g) as *mut __m512i, reduce(odd));
        }
    }
    _mm_sfence();
    out.set_len(base.len());
    out
}

// commitment[row * width + col] = sum_i key[row * height + i] * columns[col][i],
// every key element read once; the (a + bX)(c + dX) slot products accumulate
// lazily, so the flush interval keeps 2 * FLUSH products under 2^63.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub(crate) unsafe fn commit_pass(
    key: &[RingElement],
    height: usize,
    rank: usize,
    columns: &[&[RingElement]],
) -> Vec<RingElement> {
    const FLUSH: usize = 1 << 10;
    const KINDS: usize = 4;
    let width = columns.len();
    assert_eq!(key.len(), rank * height);
    let q = _mm512_set1_epi64(Q as i64);
    let zeta: [__m512i; HALF / 8] = std::array::from_fn(|g| {
        _mm512_loadu_si512(SHIFT_FACTORS.as_ptr().add(8 * g) as *const __m512i)
    });
    let zeta_shoup: [__m512i; HALF / 8] = std::array::from_fn(|g| {
        let values: [u64; 8] = std::array::from_fn(|l| shoup(SHIFT_FACTORS[8 * g + l]));
        _mm512_loadu_si512(values.as_ptr() as *const __m512i)
    });
    let mut sums = vec![RingElement::zero(Representation::IncompleteNTT); rank * width];
    let mut acc = vec![_mm512_setzero_si512(); rank * width * (HALF / 8) * KINDS];
    let mut wit = vec![_mm512_setzero_si512(); width * 3 * (HALF / 8)];
    for i in 0..height {
        for col in 0..width {
            let element = &columns[col][i];
            for g in 0..HALF / 8 {
                let c = _mm512_load_si512(element.v.as_ptr().add(8 * g) as *const __m512i);
                let d = _mm512_load_si512(element.v.as_ptr().add(HALF + 8 * g) as *const __m512i);
                let zd = mul_shoup(d, zeta[g], zeta_shoup[g]);
                let zd = _mm512_min_epu64(zd, _mm512_sub_epi64(zd, q));
                let base = (col * 3) * (HALF / 8) + g;
                wit[base] = c;
                wit[base + HALF / 8] = d;
                wit[base + 2 * (HALF / 8)] = zd;
            }
        }
        for row in 0..rank {
            let element = &key[row * height + i];
            for g in 0..HALF / 8 {
                let a = _mm512_load_si512(element.v.as_ptr().add(8 * g) as *const __m512i);
                let b = _mm512_load_si512(element.v.as_ptr().add(HALF + 8 * g) as *const __m512i);
                for col in 0..width {
                    let base = (col * 3) * (HALF / 8) + g;
                    let c = wit[base];
                    let d = wit[base + HALF / 8];
                    let zd = wit[base + 2 * (HALF / 8)];
                    let slot = ((row * width + col) * (HALF / 8) + g) * KINDS;
                    let mut even_lo = acc[slot];
                    let mut even_hi = acc[slot + 1];
                    let mut odd_lo = acc[slot + 2];
                    let mut odd_hi = acc[slot + 3];
                    even_lo = _mm512_madd52lo_epu64(even_lo, a, c);
                    even_hi = _mm512_madd52hi_epu64(even_hi, a, c);
                    even_lo = _mm512_madd52lo_epu64(even_lo, b, zd);
                    even_hi = _mm512_madd52hi_epu64(even_hi, b, zd);
                    odd_lo = _mm512_madd52lo_epu64(odd_lo, a, d);
                    odd_hi = _mm512_madd52hi_epu64(odd_hi, a, d);
                    odd_lo = _mm512_madd52lo_epu64(odd_lo, b, c);
                    odd_hi = _mm512_madd52hi_epu64(odd_hi, b, c);
                    acc[slot] = even_lo;
                    acc[slot + 1] = even_hi;
                    acc[slot + 2] = odd_lo;
                    acc[slot + 3] = odd_hi;
                }
            }
        }
        if (i + 1) % FLUSH == 0 || i + 1 == height {
            for row in 0..rank {
                for col in 0..width {
                    let sum = &mut sums[row * width + col];
                    for g in 0..HALF / 8 {
                        let slot = ((row * width + col) * (HALF / 8) + g) * KINDS;
                        for (half, kind) in [(0usize, 0usize), (HALF, 2)] {
                            let reduced = reduce_lo_hi_v(acc[slot + kind], acc[slot + kind + 1]);
                            let ptr = sum.v.as_mut_ptr().add(half + 8 * g) as *mut __m512i;
                            let total = _mm512_add_epi64(_mm512_load_si512(ptr), reduced);
                            _mm512_store_si512(
                                ptr,
                                _mm512_min_epu64(total, _mm512_sub_epi64(total, q)),
                            );
                            acc[slot + kind] = _mm512_setzero_si512();
                            acc[slot + kind + 1] = _mm512_setzero_si512();
                        }
                    }
                }
            }
        }
    }
    sums
}
