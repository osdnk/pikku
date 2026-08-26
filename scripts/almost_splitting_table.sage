"""Reproduce the degree-128 almost-splitting sampler table.

Run with

    sage scripts/almost_splitting_table.sage

For each selected quadratic-slot modulus q, the script performs the following
calculation.

1. Verify that q is prime and has order two modulo 256.
2. Choose the smallest fixed Hamming weight s for which

       |C_fw(256, s)| >= 2^5 q^2.

3. Compute the operator-norm distribution.  Weights three and four are
   enumerated exactly after quotienting by the norm-preserving rotation and
   global-sign symmetries.  Weight five uses a deterministic sample of 2^22
   challenges because exhaustive enumeration is substantially larger.
4. Scan B in increments of 0.1 and choose the smallest value for which both

       |C_fw(256, s, B)| >= 2^5 q^2

   and the expected number of rejection-sampling trials is at most 10.
5. Print the values used by the LaTeX table and assert the primality, splitting,
   five-bit margin, and expected-trial conditions.

The final error-exponent column is log_2(64/q^2), since e = 2 and the number of
NTT slots is phi/e = 64.  This is the pointwise scale derived from one
quadratic-slot value having probability about q^-2.
"""

from itertools import combinations, islice

import numpy as np
from sage.all import Mod, ZZ, binomial


DEGREE = 128
CONDUCTOR = 256
SLOT_DEGREE = 2
SLOT_COUNT = DEGREE // SLOT_DEGREE
MARGIN_BITS = 5
MAX_EXPECTED_TRIALS = 10
SELECTED_PRIMES = (
    127,
    383,
    641,
    1151,
    1153,
    1279,
    1409,
    1663,
    2687,
    2689,
    3457,
    3583,
    3967,
    4481,
    4993,
    5119,
    5503,
    6143,
    6271,
    6529,
)

EXACT_BATCH_SIZE = 2**12
MONTE_CARLO_SAMPLES = 2**22
MONTE_CARLO_BATCH_SIZE = 2**13
MONTE_CARLO_SEED = 0x50494B4B55

TWIST = np.exp(1j * np.pi * np.arange(DEGREE) / DEGREE)


def challenge_cardinality(weight):
    return ZZ(2) ** weight * binomial(DEGREE, weight)


def challenge_entropy(weight):
    return float(challenge_cardinality(weight).log(2))


def minimum_weight(q):
    target = ZZ(2) ** MARGIN_BITS * q**SLOT_DEGREE
    return next(
        weight
        for weight in range(1, DEGREE + 1)
        if challenge_cardinality(weight) >= target
    )


def operator_norms(coefficients):
    evaluations = np.fft.ifft(coefficients * TWIST, axis=1) * DEGREE
    return np.max(np.abs(evaluations), axis=1)


def exact_operator_norms(weight):
    """Enumerate the operator-norm distribution for a fixed weight.

    Rotation and global negation preserve the operator norm.  Fixing one
    selected coefficient at index zero with sign +1 therefore preserves the
    distribution while reducing its size by a factor 2 * DEGREE / weight.
    """

    sign_count = 2 ** (weight - 1)
    signs = np.array(
        [
            [
                1.0 if (mask >> index) & 1 else -1.0
                for index in range(weight - 1)
            ]
            for mask in range(sign_count)
        ],
        dtype=np.float64,
    )
    count = int(binomial(DEGREE - 1, weight - 1)) * sign_count
    norms = np.empty(count, dtype=np.float64)
    supports = combinations(range(1, DEGREE), weight - 1)
    offset = 0

    while True:
        block = list(islice(supports, EXACT_BATCH_SIZE))
        if not block:
            break
        block = np.asarray(block, dtype=np.int16)
        rows = len(block) * sign_count
        coefficients = np.zeros((rows, DEGREE), dtype=np.float64)
        coefficients[:, 0] = 1.0
        support_rows = np.repeat(block, sign_count, axis=0)
        sign_rows = np.tile(signs, (len(block), 1))
        coefficients[np.arange(rows)[:, None], support_rows] = sign_rows
        norms[offset : offset + rows] = operator_norms(coefficients)
        offset += rows

    assert offset == count
    return norms


def sampled_operator_norms(weight):
    rng = np.random.default_rng(int(MONTE_CARLO_SEED + weight))
    norms = np.empty(MONTE_CARLO_SAMPLES, dtype=np.float64)

    for start in range(0, MONTE_CARLO_SAMPLES, MONTE_CARLO_BATCH_SIZE):
        count = min(MONTE_CARLO_BATCH_SIZE, MONTE_CARLO_SAMPLES - start)
        keys = rng.random((count, DEGREE), dtype=np.float32)
        support = np.argpartition(keys, weight - 1, axis=1)[:, :weight]
        signs = 2 * rng.integers(
            0, 2, size=(count, weight), dtype=np.int8
        ) - 1
        coefficients = np.zeros((count, DEGREE), dtype=np.float64)
        coefficients[np.arange(count)[:, None], support] = signs
        norms[start : start + count] = operator_norms(coefficients)

    return norms


def acceptance_probability(norms, bound):
    return float(np.count_nonzero(norms <= bound + 1e-12)) / len(norms)


def choose_bound(q, weight, norms):
    cardinality = int(challenge_cardinality(weight))
    minimum_acceptance = max(
        1.0 / MAX_EXPECTED_TRIALS,
        float((ZZ(2) ** MARGIN_BITS * q**SLOT_DEGREE) / cardinality),
    )
    for tenth in range(0, 10 * weight + 1):
        bound = tenth / 10.0
        probability = acceptance_probability(norms, bound)
        if probability >= minimum_acceptance:
            return bound, probability
    raise AssertionError("no admissible operator-norm threshold")


weights = {q: minimum_weight(q) for q in SELECTED_PRIMES}
norms_by_weight = {}
for weight in sorted(set(weights.values())):
    if weight <= 4:
        norms_by_weight[weight] = exact_operator_norms(weight)
    else:
        norms_by_weight[weight] = sampled_operator_norms(weight)

print(
    "q,s,base_entropy,B,expected_trials,accepted_entropy,"
    "q^e_entropy,error_exponent,margin"
)
for q in SELECTED_PRIMES:
    assert ZZ(q).is_prime()
    assert Mod(q, CONDUCTOR).multiplicative_order() == SLOT_DEGREE

    weight = weights[q]
    bound, acceptance = choose_bound(q, weight, norms_by_weight[weight])
    base_entropy = challenge_entropy(weight)
    accepted_entropy = base_entropy + np.log2(acceptance)
    slot_entropy = SLOT_DEGREE * np.log2(q)
    error_exponent = np.log2(SLOT_COUNT) - slot_entropy
    margin = accepted_entropy - slot_entropy

    assert 1.0 / acceptance <= MAX_EXPECTED_TRIALS
    assert margin >= MARGIN_BITS
    print(
        f"{q},{weight},{base_entropy:.3f},{bound:.1f},"
        f"{1.0 / acceptance:.2f},{accepted_entropy:.3f},"
        f"{slot_entropy:.3f},{error_exponent:.3f},{margin:.3f}"
    )
