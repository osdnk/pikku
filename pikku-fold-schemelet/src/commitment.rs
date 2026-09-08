use rokoko::common::matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix};
use rokoko::common::ring_arithmetic::{Representation, RingElement};
#[cfg(not(feature = "derived-key"))]
use rokoko::common::sampling::sample_random_vector;
use rokoko::common::structured_row::PreprocessedRow;
use rokoko::protocol::commitment_crt::{commit_basic_crt, CrtKey, Plan};
use rokoko::protocol::project_coarse::{prepare_i16_witness, Signed16RingElement};
use std::time::{Duration, Instant};

// A full preprocessed CrtKey is limbs * rank * height * 2 * DEGREE i16 (~45 GB
// at the default size), so the key is preprocessed one height chunk at a time.
const KEY_CHUNK: usize = 1 << 16;

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
    fn chunk_rows(&self, start: usize, len: usize) -> Vec<PreprocessedRow> {
        (0..self.rank)
            .map(|row| PreprocessedRow {
                preprocessed_row: self.rows[row * self.height + start..][..len].to_vec(),
            })
            .collect()
    }

    #[cfg(feature = "derived-key")]
    fn chunk_rows(&self, start: usize, len: usize) -> Vec<PreprocessedRow> {
        use rokoko::common::sampling::{AesCtrPublicSampler, PUBLIC_CRS_SEED};
        (0..self.rank)
            .map(|row| {
                let mut seed = PUBLIC_CRS_SEED.to_vec();
                seed.extend_from_slice(b"row");
                seed.extend_from_slice(&(row as u64).to_le_bytes());
                seed.extend_from_slice(b"chunk");
                seed.extend_from_slice(&(start as u64).to_le_bytes());
                let mut sampler = AesCtrPublicSampler::from_seed(&seed);
                let mut preprocessed_row =
                    vec![RingElement::zero(Representation::IncompleteNTT); len];
                for element in preprocessed_row.iter_mut() {
                    sampler.fill_ring_element(element, Representation::IncompleteNTT);
                }
                PreprocessedRow { preprocessed_row }
            })
            .collect()
    }

    pub(crate) fn commit(
        &self,
        witness: &VerticallyAlignedMatrix<RingElement>,
        coeff_bound: u64,
    ) -> (HorizontallyAlignedMatrix<RingElement>, Duration) {
        assert_eq!(witness.height, self.height);
        let width = witness.used_cols;
        let mut commitment = HorizontallyAlignedMatrix {
            data: vec![RingElement::zero(Representation::IncompleteNTT); self.rank * width],
            width,
            height: self.rank,
        };
        let mut derivation = Duration::ZERO;
        let digits = prepare_i16_witness(witness);
        for start in (0..self.height).step_by(KEY_CHUNK) {
            let len = KEY_CHUNK.min(self.height - start);
            let derive_start = Instant::now();
            let ck = self.chunk_rows(start, len);
            let plan = Plan::for_shape(len, coeff_bound, self.rank);
            let key = CrtKey::preprocess(&ck, self.rank, &plan);
            drop(ck);
            derivation += derive_start.elapsed();
            let chunk = VerticallyAlignedMatrix {
                data: (0..width)
                    .flat_map(|col| {
                        digits.data[col * digits.height + start..][..len].iter().cloned()
                    })
                    .collect::<Vec<Signed16RingElement>>(),
                width,
                height: len,
                used_cols: width,
            };
            let part = commit_basic_crt(&key, &chunk, &plan, self.rank);
            for row in 0..self.rank {
                for col in 0..width {
                    commitment[(row, col)] += &part[(row, col)];
                }
            }
        }
        (commitment, derivation)
    }
}
