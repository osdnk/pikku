use crate::ifma::commit_pass;
use rokoko::common::matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix};
use rokoko::common::ring_arithmetic::{Representation, RingElement};
#[cfg(not(feature = "derived-key"))]
use rokoko::common::sampling::sample_random_vector;
use std::time::Duration;

#[cfg(feature = "derived-key")]
const DERIVE_CHUNK: usize = 1024;

pub(crate) struct CommitmentKey {
    #[cfg(not(feature = "derived-key"))]
    pub(crate) rows: Vec<RingElement>,
    height: usize,
    rank: usize,
}

impl CommitmentKey {
    #[cfg(not(feature = "derived-key"))]
    pub(crate) fn sample(height: usize, rank: usize) -> Self {
        CommitmentKey {
            rows: sample_random_vector(height * rank, Representation::IncompleteNTT),
            height,
            rank,
        }
    }

    #[cfg(feature = "derived-key")]
    pub(crate) fn sample(height: usize, rank: usize) -> Self {
        CommitmentKey { height, rank }
    }

    #[cfg(not(feature = "derived-key"))]
    pub(crate) fn commit_column(&self, column: &[RingElement]) -> (Vec<RingElement>, Duration) {
        assert_eq!(column.len(), self.height);
        let out = unsafe { commit_pass(&self.rows, self.height, self.rank, &[column]) };
        (out, Duration::ZERO)
    }

    // Every row's chunk is derived from its own AES-CTR stream, so the row
    // keys of one height range sit together and the kernel sees all of them.
    #[cfg(feature = "derived-key")]
    pub(crate) fn commit_column(&self, column: &[RingElement]) -> (Vec<RingElement>, Duration) {
        use rokoko::common::sampling::{AesCtrPublicSampler, PUBLIC_CRS_SEED};
        assert_eq!(column.len(), self.height);
        let mut out = vec![RingElement::zero(Representation::IncompleteNTT); self.rank];
        let mut chunk =
            vec![RingElement::zero(Representation::IncompleteNTT); self.rank * DERIVE_CHUNK];
        let mut samplers: Vec<AesCtrPublicSampler> = (0..self.rank)
            .map(|row| {
                let mut seed = PUBLIC_CRS_SEED.to_vec();
                seed.extend_from_slice(b"row");
                seed.extend_from_slice(&(row as u64).to_le_bytes());
                AesCtrPublicSampler::from_seed(&seed)
            })
            .collect();
        let mut derivation = Duration::ZERO;
        for start in (0..self.height).step_by(DERIVE_CHUNK) {
            let len = DERIVE_CHUNK.min(self.height - start);
            let derive_start = std::time::Instant::now();
            for (row, sampler) in samplers.iter_mut().enumerate() {
                for element in chunk[row * len..(row + 1) * len].iter_mut() {
                    sampler.fill_ring_element(element, Representation::IncompleteNTT);
                }
            }
            derivation += derive_start.elapsed();
            let partial = unsafe {
                commit_pass(
                    &chunk[..self.rank * len],
                    len,
                    self.rank,
                    &[&column[start..start + len]],
                )
            };
            for (acc, value) in out.iter_mut().zip(&partial) {
                *acc += value;
            }
        }
        (out, derivation)
    }

    pub(crate) fn commit(
        &self,
        witness: &VerticallyAlignedMatrix<RingElement>,
    ) -> (HorizontallyAlignedMatrix<RingElement>, Duration) {
        assert_eq!(witness.height, self.height);
        let mut commitment = HorizontallyAlignedMatrix {
            data: vec![
                RingElement::zero(Representation::IncompleteNTT);
                self.rank * witness.used_cols
            ],
            width: witness.used_cols,
            height: self.rank,
        };
        let mut derivation = Duration::ZERO;
        for col in 0..witness.used_cols {
            let (column, column_derivation) = self.commit_column(witness.col(col));
            derivation += column_derivation;
            for (row, value) in column.into_iter().enumerate() {
                commitment[(row, col)] = value;
            }
        }
        (commitment, derivation)
    }
}
