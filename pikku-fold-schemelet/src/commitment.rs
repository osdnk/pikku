use rokoko::common::matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix};
use rokoko::common::ring_arithmetic::{Representation, RingElement};
#[cfg(not(feature = "derived-key"))]
use rokoko::common::sampling::sample_random_vector;
use std::time::Duration;

#[cfg(feature = "derived-key")]
const DERIVE_CHUNK: usize = 4096;

pub(crate) struct CommitmentKey {
    #[cfg(not(feature = "derived-key"))]
    rows: Vec<RingElement>,
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
        let mut out = vec![RingElement::zero(Representation::IncompleteNTT); self.rank];
        let mut tmp = RingElement::zero(Representation::IncompleteNTT);
        for (row, acc) in out.iter_mut().enumerate() {
            let key_row = &self.rows[row * self.height..(row + 1) * self.height];
            for (key, value) in key_row.iter().zip(column) {
                tmp *= (key, value);
                *acc += &tmp;
            }
        }
        (out, Duration::ZERO)
    }

    #[cfg(feature = "derived-key")]
    pub(crate) fn commit_column(&self, column: &[RingElement]) -> (Vec<RingElement>, Duration) {
        use rokoko::common::sampling::{AesCtrPublicSampler, PUBLIC_CRS_SEED};
        assert_eq!(column.len(), self.height);
        let mut out = vec![RingElement::zero(Representation::IncompleteNTT); self.rank];
        let mut tmp = RingElement::zero(Representation::IncompleteNTT);
        let mut chunk = vec![RingElement::zero(Representation::IncompleteNTT); DERIVE_CHUNK];
        let mut derivation = Duration::ZERO;
        for (row, acc) in out.iter_mut().enumerate() {
            let mut seed = PUBLIC_CRS_SEED.to_vec();
            seed.extend_from_slice(b"row");
            seed.extend_from_slice(&(row as u64).to_le_bytes());
            let mut sampler = AesCtrPublicSampler::from_seed(&seed);
            for start in (0..self.height).step_by(DERIVE_CHUNK) {
                let len = DERIVE_CHUNK.min(self.height - start);
                let derive_start = std::time::Instant::now();
                for element in chunk[..len].iter_mut() {
                    sampler.fill_ring_element(element, Representation::IncompleteNTT);
                }
                derivation += derive_start.elapsed();
                for (key, value) in chunk[..len].iter().zip(&column[start..start + len]) {
                    tmp *= (key, value);
                    *acc += &tmp;
                }
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
