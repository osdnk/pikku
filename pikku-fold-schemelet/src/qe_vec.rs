use incomplete_rexl::{eltwise_fma_mod, eltwise_mult_mod, eltwise_sub_mod};
use rokoko::common::config::MOD_Q;
use rokoko::common::ring_arithmetic::{QuadraticExtension, FIELD_SHIFT_FACTOR};
use rokoko::hexl::bindings::{add_mod, multiply_mod};

pub(crate) struct QeVec {
    pub(crate) limb0: Vec<u64>,
    pub(crate) limb1: Vec<u64>,
}

impl QeVec {
    pub(crate) fn len(&self) -> usize {
        self.limb0.len()
    }

    pub(crate) fn from_limbs(limb0: Vec<u64>, limb1: Vec<u64>) -> Self {
        assert_eq!(limb0.len(), limb1.len());
        QeVec { limb0, limb1 }
    }

    pub(crate) fn to_vec(&self) -> Vec<QuadraticExtension> {
        (0..self.len()).map(|index| self.get(index)).collect()
    }

    // out[x] = sum_t weights[t] * self[t * len + x] with len = self.len() / weights.len().
    pub(crate) fn contract_top(&self, weights: &[QuadraticExtension]) -> QeVec {
        let len = self.len() / weights.len();
        assert_eq!(len * weights.len(), self.len());
        let mut limb0 = vec![0u64; len];
        let mut limb1 = vec![0u64; len];
        let mut scratch = vec![0u64; len];
        for (t, weight) in weights.iter().enumerate() {
            let w0 = weight.coeffs[0];
            let w1 = weight.coeffs[1];
            let alpha_w1 = unsafe { multiply_mod(*FIELD_SHIFT_FACTOR, w1, MOD_Q) };
            let t0 = &self.limb0[t * len..(t + 1) * len];
            let t1 = &self.limb1[t * len..(t + 1) * len];
            eltwise_fma_mod(&mut scratch, t0, w0, &limb0, MOD_Q);
            eltwise_fma_mod(&mut limb0, t1, alpha_w1, &scratch, MOD_Q);
            eltwise_fma_mod(&mut scratch, t1, w0, &limb1, MOD_Q);
            eltwise_fma_mod(&mut limb1, t0, w1, &scratch, MOD_Q);
        }
        QeVec { limb0, limb1 }
    }

    pub(crate) fn get(&self, index: usize) -> QuadraticExtension {
        QuadraticExtension {
            coeffs: [self.limb0[index], self.limb1[index]],
        }
    }

    pub(crate) fn dot(&self, other: &QeVec) -> QuadraticExtension {
        let n = self.len();
        assert_eq!(n, other.len());
        let mut products = vec![0u64; n];
        let mut sums = [0u64; 4];
        for (slot, (a, b)) in [
            (&self.limb0, &other.limb0),
            (&self.limb1, &other.limb1),
            (&self.limb0, &other.limb1),
            (&self.limb1, &other.limb0),
        ]
        .into_iter()
        .enumerate()
        {
            eltwise_mult_mod(&mut products, a, b, MOD_Q);
            sums[slot] = lazy_sum_mod(&products);
        }
        let c0 = unsafe {
            add_mod(
                sums[0],
                multiply_mod(*FIELD_SHIFT_FACTOR, sums[1], MOD_Q),
                MOD_Q,
            )
        };
        let c1 = unsafe { add_mod(sums[2], sums[3], MOD_Q) };
        QuadraticExtension { coeffs: [c0, c1] }
    }
}

pub(crate) fn lazy_sum_mod(values: &[u64]) -> u64 {
    let partials: Vec<u64> = values
        .chunks(4096)
        .map(|chunk| chunk.iter().sum::<u64>())
        .collect();
    let mut reduced = vec![0u64; partials.len()];
    incomplete_rexl::eltwise_reduce_mod(&mut reduced, &partials, MOD_Q);
    let mut acc = 0u64;
    for value in reduced {
        acc = unsafe { add_mod(acc, value, MOD_Q) };
    }
    acc
}

pub(crate) fn expand_eq_soa(layers_msb: &[QuadraticExtension]) -> QeVec {
    let full = 1usize << layers_msb.len();
    let mut limb0 = vec![0u64; full];
    let mut limb1 = vec![0u64; full];
    limb0[0] = 1;
    let mut scratch = vec![0u64; full / 2];
    let zeros = vec![0u64; full / 2];
    let mut n = 1usize;
    for layer in layers_msb.iter().rev() {
        let l0 = layer.coeffs[0];
        let l1 = layer.coeffs[1];
        let alpha_l1 = unsafe { multiply_mod(*FIELD_SHIFT_FACTOR, l1, MOD_Q) };
        {
            let (low, high) = limb0.split_at_mut(n);
            eltwise_fma_mod(&mut scratch[..n], &limb1[..n], alpha_l1, &zeros[..n], MOD_Q);
            eltwise_fma_mod(&mut high[..n], &low[..n], l0, &scratch[..n], MOD_Q);
        }
        {
            let (low, high) = limb1.split_at_mut(n);
            eltwise_fma_mod(&mut scratch[..n], &low[..n], l0, &zeros[..n], MOD_Q);
            let low0 = &limb0[..n];
            eltwise_fma_mod(&mut high[..n], low0, l1, &scratch[..n], MOD_Q);
        }
        {
            let (low, high) = limb0.split_at_mut(n);
            eltwise_sub_mod(&mut scratch[..n], &low[..n], &high[..n], MOD_Q);
            low[..n].copy_from_slice(&scratch[..n]);
        }
        {
            let (low, high) = limb1.split_at_mut(n);
            eltwise_sub_mod(&mut scratch[..n], &low[..n], &high[..n], MOD_Q);
            low[..n].copy_from_slice(&scratch[..n]);
        }
        n *= 2;
    }
    QeVec { limb0, limb1 }
}
