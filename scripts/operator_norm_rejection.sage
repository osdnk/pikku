"""Reproduce the operator-norm rejection table.

The first part estimates the fixed-weight ternary cutoffs in degree 128.
The second part estimates bounded fixed-weight cutoffs for B=2 in degree
64, and the final part estimates ternary cutoffs in degree 256.

Run with

    sage scripts/operator_norm_rejection.sage
"""

from sage.all import RealField, binomial
import numpy as np


RR = RealField(128)
DEGREE = 128
TARGET_BITS = (100, 128)
LOG_TRIALS = (0, 2, 4, 6, 8)
SAMPLES = 2**18
BATCH_SIZE = 2**13
CALIBRATION_SEED = 0x50494B4B55
VALIDATION_SEED = 0x53414D504C45


def fixed_weight_entropy(degree, weight, coefficient_bound):
    return float(
        RR(binomial(degree, weight)).log(2)
        + weight * RR(2 * coefficient_bound).log(2)
    )


def minimum_coefficient_bound(degree):
    required_entropy = max(TARGET_BITS) + max(LOG_TRIALS)
    return next(
        coefficient_bound
        for coefficient_bound in range(1, degree + 1)
        if max(
            fixed_weight_entropy(degree, weight, coefficient_bound)
            for weight in range(1, degree + 1)
        )
        >= required_entropy
    )


COEFFICIENT_BOUNDS = {
    degree: minimum_coefficient_bound(degree)
    for degree in (64, 128, 256)
}
assert COEFFICIENT_BOUNDS == {64: 2, 128: 1, 256: 1}


def challenge_entropy(weight):
    return fixed_weight_entropy(
        DEGREE, weight, COEFFICIENT_BOUNDS[DEGREE]
    )


def minimum_weight(target_bits, log_trials):
    return next(
        weight
        for weight in range(1, DEGREE + 1)
        if challenge_entropy(weight) - log_trials >= target_bits
    )


def sample_operator_norms(seed, weight):
    rng = np.random.default_rng(int(seed))
    twist = np.exp(1j * np.pi * np.arange(DEGREE) / DEGREE)
    norms = np.empty(SAMPLES, dtype=np.float64)

    for start in range(0, SAMPLES, BATCH_SIZE):
        count = min(BATCH_SIZE, SAMPLES - start)
        keys = rng.random((count, DEGREE), dtype=np.float32)
        support = np.argpartition(keys, weight - 1, axis=1)[:, :weight]
        signs = 2 * rng.integers(
            0, 2, size=(count, weight), dtype=np.int8
        ) - 1
        coefficients = np.zeros((count, DEGREE), dtype=np.float64)
        coefficients[np.arange(count)[:, None], support] = signs

        # For zeta = exp(pi*i/DEGREE), the odd canonical evaluations are
        # DEGREE * IFFT(coefficients * (zeta^i)_i).
        evaluations = np.fft.ifft(coefficients * twist, axis=1) * DEGREE
        norms[start : start + count] = np.max(
            np.abs(evaluations), axis=1
        )

    assert np.max(norms) <= weight + 1e-10
    return norms


weights = {
    (target_bits, r): minimum_weight(target_bits, r)
    for target_bits in TARGET_BITS
    for r in LOG_TRIALS
}
calibration = {}
validation = {}
sampled_weights = {
    weights[target_bits, r]
    for target_bits in TARGET_BITS
    for r in LOG_TRIALS
    if r > 0
}
for weight in sorted(sampled_weights):
    calibration[weight] = sample_operator_norms(
        CALIBRATION_SEED + weight, weight
    )
    validation[weight] = sample_operator_norms(
        VALIDATION_SEED + weight, weight
    )

print(f"samples per phase = {SAMPLES}")
print("target_bits,r,weight,base_entropy,target_trials,threshold,validation_trials,entropy")

for target_bits in TARGET_BITS:
    for r in LOG_TRIALS:
        weight = weights[target_bits, r]
        base_entropy = challenge_entropy(weight)
        if r == 0:
            threshold = float(weight)
            acceptance = 1.0
        else:
            target_acceptance = 2.0 ** (-r)
            threshold = np.quantile(
                calibration[weight], target_acceptance, method="higher"
            )
            acceptance = (
                float(np.count_nonzero(validation[weight] <= threshold))
                / int(SAMPLES)
            )
        trials = 1.0 / acceptance
        entropy = base_entropy + np.log2(acceptance)
        print(
            f"{target_bits},{r},{weight},{base_entropy:.3f},{2**r},"
            f"{threshold:.6f},{trials:.4f},{entropy:.3f}"
        )


# The bounded fixed-weight set has (2B)^s * binomial(d, s) elements.
# Degree 64 requires B=2: with B=1, no fixed-weight slice contains even
# 2^100 elements.  For each target and rejection budget, use the smallest
# weight whose unrejected cardinality is at least the target retained
# cardinality times the target number of trials.  As above, calibrate on
# one sample and independently validate on another.
BOUNDED_DEGREE = 64
COEFFICIENT_BOUND = COEFFICIENT_BOUNDS[BOUNDED_DEGREE]
BOUNDED_CALIBRATION_SEED = 0x424F554E444544
BOUNDED_VALIDATION_SEED = 0x56414C4944415445


def bounded_challenge_entropy(weight, coefficient_bound=COEFFICIENT_BOUND):
    return fixed_weight_entropy(
        BOUNDED_DEGREE, weight, coefficient_bound
    )


assert max(
    bounded_challenge_entropy(weight, 1)
    for weight in range(1, BOUNDED_DEGREE + 1)
) < min(TARGET_BITS)
assert max(
    bounded_challenge_entropy(weight)
    for weight in range(1, BOUNDED_DEGREE + 1)
) >= max(TARGET_BITS) + max(LOG_TRIALS)


def bounded_weight(target_bits, log_trials):
    required_entropy = target_bits + log_trials
    return next(
        weight
        for weight in range(1, BOUNDED_DEGREE + 1)
        if bounded_challenge_entropy(weight) >= required_entropy
    )


def sample_bounded_operator_norms(seed, weight):
    rng = np.random.default_rng(int(seed))
    twist = np.exp(1j * np.pi * np.arange(BOUNDED_DEGREE) / BOUNDED_DEGREE)
    norms = np.empty(SAMPLES, dtype=np.float64)

    for start in range(0, SAMPLES, BATCH_SIZE):
        count = min(BATCH_SIZE, SAMPLES - start)
        keys = rng.random((count, BOUNDED_DEGREE), dtype=np.float32)
        support = np.argpartition(keys, weight - 1, axis=1)[:, :weight]
        magnitudes = rng.integers(
            1,
            COEFFICIENT_BOUND + 1,
            size=(count, weight),
            dtype=np.int8,
        )
        signs = 2 * rng.integers(
            0, 2, size=(count, weight), dtype=np.int8
        ) - 1
        coefficients = np.zeros(
            (count, BOUNDED_DEGREE), dtype=np.float64
        )
        coefficients[np.arange(count)[:, None], support] = magnitudes * signs
        evaluations = (
            np.fft.ifft(coefficients * twist, axis=1) * BOUNDED_DEGREE
        )
        norms[start : start + count] = np.max(
            np.abs(evaluations), axis=1
        )

    assert np.max(norms) <= weight * COEFFICIENT_BOUND + 1e-10
    return norms


bounded_weights = {
    (target_bits, r): bounded_weight(target_bits, r)
    for target_bits in TARGET_BITS
    for r in LOG_TRIALS
}
bounded_calibration = {}
bounded_validation = {}
bounded_sampled_weights = {
    bounded_weights[target_bits, r]
    for target_bits in TARGET_BITS
    for r in LOG_TRIALS
    if r > 0
}
for weight in sorted(bounded_sampled_weights):
    bounded_calibration[weight] = sample_bounded_operator_norms(
        BOUNDED_CALIBRATION_SEED + weight, weight
    )
    bounded_validation[weight] = sample_bounded_operator_norms(
        BOUNDED_VALIDATION_SEED + weight, weight
    )

print()
print("target_bits,r,weight,bound,base_entropy,target_trials,threshold,validation_trials,entropy")
for target_bits in TARGET_BITS:
    for r in LOG_TRIALS:
        weight = bounded_weights[target_bits, r]
        base_entropy = bounded_challenge_entropy(weight)
        if r == 0:
            threshold = float(weight * COEFFICIENT_BOUND)
            acceptance = 1.0
        else:
            target_acceptance = 2.0 ** (-r)
            threshold = np.quantile(
                bounded_calibration[weight],
                target_acceptance,
                method="higher",
            )
            acceptance = (
                float(
                    np.count_nonzero(
                        bounded_validation[weight] <= threshold
                    )
                )
                / int(SAMPLES)
            )
        trials = 1.0 / acceptance
        entropy = base_entropy + np.log2(acceptance)
        print(
            f"{target_bits},{r},{weight},{COEFFICIENT_BOUND},"
            f"{base_entropy:.3f},{2**r},{threshold:.6f},"
            f"{trials:.4f},{entropy:.3f}"
        )


# Repeat the sparse fixed-weight ternary staircase in degree 256.
LARGE_DEGREE = 256
LARGE_CALIBRATION_SEED = 0x4C4152474543414C
LARGE_VALIDATION_SEED = 0x4C4152474556414C


def large_challenge_entropy(weight):
    return fixed_weight_entropy(
        LARGE_DEGREE, weight, COEFFICIENT_BOUNDS[LARGE_DEGREE]
    )


def large_weight(target_bits, log_trials):
    required_entropy = target_bits + log_trials
    return next(
        weight
        for weight in range(1, LARGE_DEGREE + 1)
        if large_challenge_entropy(weight) >= required_entropy
    )


def sample_large_operator_norms(seed, weight):
    rng = np.random.default_rng(int(seed))
    twist = np.exp(1j * np.pi * np.arange(LARGE_DEGREE) / LARGE_DEGREE)
    norms = np.empty(SAMPLES, dtype=np.float64)

    for start in range(0, SAMPLES, BATCH_SIZE):
        count = min(BATCH_SIZE, SAMPLES - start)
        keys = rng.random((count, LARGE_DEGREE), dtype=np.float32)
        support = np.argpartition(keys, weight - 1, axis=1)[:, :weight]
        signs = 2 * rng.integers(
            0, 2, size=(count, weight), dtype=np.int8
        ) - 1
        coefficients = np.zeros((count, LARGE_DEGREE), dtype=np.float64)
        coefficients[np.arange(count)[:, None], support] = signs
        evaluations = (
            np.fft.ifft(coefficients * twist, axis=1) * LARGE_DEGREE
        )
        norms[start : start + count] = np.max(
            np.abs(evaluations), axis=1
        )

    assert np.max(norms) <= weight + 1e-10
    return norms


large_weights = {
    (target_bits, r): large_weight(target_bits, r)
    for target_bits in TARGET_BITS
    for r in LOG_TRIALS
}
large_calibration = {}
large_validation = {}
large_sampled_weights = {
    large_weights[target_bits, r]
    for target_bits in TARGET_BITS
    for r in LOG_TRIALS
    if r > 0
}
for weight in sorted(large_sampled_weights):
    large_calibration[weight] = sample_large_operator_norms(
        LARGE_CALIBRATION_SEED + weight, weight
    )
    large_validation[weight] = sample_large_operator_norms(
        LARGE_VALIDATION_SEED + weight, weight
    )

print()
print("target_bits,r,weight,bound,base_entropy,target_trials,threshold,validation_trials,entropy")
for target_bits in TARGET_BITS:
    for r in LOG_TRIALS:
        weight = large_weights[target_bits, r]
        base_entropy = large_challenge_entropy(weight)
        if r == 0:
            threshold = float(weight)
            acceptance = 1.0
        else:
            target_acceptance = 2.0 ** (-r)
            threshold = np.quantile(
                large_calibration[weight],
                target_acceptance,
                method="higher",
            )
            acceptance = (
                float(
                    np.count_nonzero(
                        large_validation[weight] <= threshold
                    )
                )
                / int(SAMPLES)
            )
        trials = 1.0 / acceptance
        entropy = base_entropy + np.log2(acceptance)
        print(
            f"{target_bits},{r},{weight},1,{base_entropy:.3f},"
            f"{2**r},{threshold:.6f},{trials:.4f},{entropy:.3f}"
        )
