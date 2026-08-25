use crate::config::{commitment_rank, FOLD_INPUTS};
use crate::statement::Instance;
use rokoko::common::config::{DEGREE, MOD_Q};
use rokoko::common::hash::HashWrapper;

pub(crate) fn statement_transcript(m: usize, instance: &Instance) -> HashWrapper {
    let mut transcript = HashWrapper::new();
    transcript.update_with_u64(MOD_Q);
    transcript.update_with_u64(DEGREE as u64);
    transcript.update_with_u64(m as u64);
    transcript.update_with_u64(FOLD_INPUTS as u64);
    transcript.update_with_u64(commitment_rank(m) as u64);
    transcript.update_with_ring_element_slice(&instance.commitment.data);
    for claim in &instance.claims {
        transcript.update_with_ring_element_slice(&claim.point);
        transcript.update_with_ring_element(&claim.value);
    }
    transcript
}
