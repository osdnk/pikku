//! AVX-512 fixed-weight ternary challenge sampler for `Z[X]/(X^128 + 1)`.
//!
//! This is copied from Rokoko's `common/short_challenge.rs` structure and
//! adjusted so the fixed weight and operator-norm bound are row parameters.

use rand::rngs::SmallRng;
use rand::Rng;
use std::f64::consts::PI;
use std::sync::LazyLock;

pub const DEGREE: usize = 128;
pub const HALF_DEGREE: usize = DEGREE / 2;
pub const MAX_WEIGHT: usize = 5;

const PHASE_LEN: usize = 2 * DEGREE;
const PHASE_MASK: usize = PHASE_LEN - 1;

static PHASE_RE: LazyLock<[f64; PHASE_LEN]> = LazyLock::new(|| {
    let mut arr = [0.0f64; PHASE_LEN];
    for (m, value) in arr.iter_mut().enumerate() {
        let angle = PI * (m as f64) / (DEGREE as f64);
        *value = angle.cos();
    }
    arr
});

static PHASE_IM: LazyLock<[f64; PHASE_LEN]> = LazyLock::new(|| {
    let mut arr = [0.0f64; PHASE_LEN];
    for (m, value) in arr.iter_mut().enumerate() {
        let angle = PI * (m as f64) / (DEGREE as f64);
        *value = angle.sin();
    }
    arr
});

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Challenge {
    pub positions: [u8; MAX_WEIGHT],
    pub signs: [i8; MAX_WEIGHT],
    pub weight: usize,
}

pub fn require_avx512() {
    assert!(
        std::is_x86_feature_detected!("avx512f"),
        "this experiment intentionally requires AVX-512F"
    );
    assert!(
        std::is_x86_feature_detected!("avx2"),
        "this experiment intentionally requires AVX2"
    );
}

#[inline(always)]
pub fn op_norm_sq_sparse(
    positions: &[u8; MAX_WEIGHT],
    signs: &[i8; MAX_WEIGHT],
    weight: usize,
) -> f64 {
    require_avx512();
    unsafe { op_norm_sq_sparse_avx512(positions, signs, weight) }
}

#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn op_norm_sq_sparse_avx512(
    positions: &[u8; MAX_WEIGHT],
    signs: &[i8; MAX_WEIGHT],
    weight: usize,
) -> f64 {
    use std::arch::x86_64::*;

    const NUM_BATCHES: usize = DEGREE / 16;
    const _: () = assert!(HALF_DEGREE == NUM_BATCHES * 8);

    let phase_re_ptr = PHASE_RE.as_ptr();
    let phase_im_ptr = PHASE_IM.as_ptr();

    let mut vr = [_mm512_setzero_pd(); NUM_BATCHES];
    let mut vi = [_mm512_setzero_pd(); NUM_BATCHES];

    let mask_v = _mm256_set1_epi32(PHASE_MASK as i32);
    let lane_index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    for k in 0..weight {
        let p = positions[k] as i32;
        let s = signs[k] as f64;
        let s_v = _mm512_set1_pd(s);
        let step = (2 * p) & (PHASE_MASK as i32);

        let step_v = _mm256_set1_epi32(step);
        let lane_offsets = _mm256_mullo_epi32(lane_index, step_v);
        let big_step = _mm256_set1_epi32(8 * step);

        let mut base = _mm256_add_epi32(_mm256_set1_epi32(p), lane_offsets);
        for b in 0..NUM_BATCHES {
            let idx_v = _mm256_and_si256(base, mask_v);
            let pre = _mm512_i32gather_pd::<8>(idx_v, phase_re_ptr);
            let pim = _mm512_i32gather_pd::<8>(idx_v, phase_im_ptr);
            vr[b] = _mm512_fmadd_pd(s_v, pre, vr[b]);
            vi[b] = _mm512_fmadd_pd(s_v, pim, vi[b]);
            base = _mm256_add_epi32(base, big_step);
        }
    }

    let mut mm = _mm512_fmadd_pd(vr[0], vr[0], _mm512_mul_pd(vi[0], vi[0]));
    for b in 1..NUM_BATCHES {
        let m = _mm512_fmadd_pd(vr[b], vr[b], _mm512_mul_pd(vi[b], vi[b]));
        mm = _mm512_max_pd(mm, m);
    }
    _mm512_reduce_max_pd(mm)
}

pub fn sample_attempt(rng: &mut SmallRng, weight: usize) -> Challenge {
    assert!(weight <= MAX_WEIGHT);

    let mut perm: [u8; DEGREE] = std::array::from_fn(|i| i as u8);
    for i in 0..weight {
        let j = rng.random_range(i..DEGREE);
        perm.swap(i, j);
    }

    let mut positions = [0u8; MAX_WEIGHT];
    let mut signs = [0i8; MAX_WEIGHT];
    for i in 0..weight {
        positions[i] = perm[i];
        signs[i] = if rng.random::<bool>() { 1 } else { -1 };
    }
    for i in 1..weight {
        let mut j = i;
        while j > 0 && positions[j - 1] > positions[j] {
            positions.swap(j - 1, j);
            signs.swap(j - 1, j);
            j -= 1;
        }
    }

    Challenge {
        positions,
        signs,
        weight,
    }
}

pub fn sample_challenge(rng: &mut SmallRng, weight: usize, bound: f64) -> (Challenge, u64) {
    let bound_sq = bound * bound + 1e-12;
    let mut attempts = 0;
    loop {
        attempts += 1;
        let challenge = sample_attempt(rng, weight);
        if op_norm_sq_sparse(&challenge.positions, &challenge.signs, weight) <= bound_sq {
            return (challenge, attempts);
        }
    }
}
