"""Verify the rounded ternary-JL parameters printed in the paper.

Run with

    sage scripts/parameters.sage

The global 200-bit certificate and the generator for arbitrary row counts
live in ``ternary_jl_parameters.ipynb``.  This lightweight script only
rechecks the five conservatively rounded, 2*lambda-row parameter sets used
in the paper.
"""

RR = RealField(200)
SECURITY_PARAMETERS = (64, 96, 128, 192, 256)

S_CROSS = RR("2.31057")
PSI = RR("0.5508")
KAPPA_BD = 1 / (2 * erfc(RR(1)))
KAPPA_BE = RR("0.52")
P_ROW = 1 / sqrt(RR(2))

# Rounded constants printed in the paper.  gamma and beta are rounded up,
# whereas alpha is rounded down.  The auxiliary D and r values need only
# satisfy the three modular inequalities checked below.
TABLE = {
    64:  (RR("6.97"), RR("13.53"), RR("171.8"), 16, RR("7.34"), RR("5.37"), 73),
    96:  (RR("8.43"), RR("20.45"), RR("257.5"), 23, RR("8.83"), RR("5.99"), 100),
    128: (RR("9.66"), RR("27.37"), RR("343.2"), 30, RR("10.08"), RR("6.53"), 126),
    192: (RR("11.75"), RR("41.21"), RR("514.6"), 44, RR("12.20"), RR("7.44"), 176),
    256: (RR("13.51"), RR("55.05"), RR("686.0"), 58, RR("13.98"), RR("8.22"), 224),
}


def inverse_erfc(y):
    return find_root(lambda x: erfc(x) - y, RR(0), RR(50))


def moment_threshold(lam, failure):
    candidates = []
    for moment in range(1, 4 * lam + 1):
        log_beta = (
            log_gamma(RR(lam + moment))
            - log_gamma(RR(lam))
            - log(failure)
        ) / moment
        candidates.append((exp(log_beta), moment))
    return min(candidates)


for lam in SECURITY_PARAMETERS:
    rows = 2 * lam
    half_error = RR(2) ** (-(lam + 1))
    row_error = half_error / rows

    gamma_exact = inverse_erfc(row_error / KAPPA_BD)
    alpha_exact = (log(half_error) - rows * log(PSI)) / S_CROSS
    beta_exact, moment = moment_threshold(lam, half_error)

    gamma, alpha, beta, threshold_rows, D, ratio, modulus_loss = TABLE[lam]
    assert gamma >= gamma_exact
    assert alpha <= alpha_exact
    assert beta >= beta_exact

    binomial_tail = sum(
        binomial(rows, i) for i in range(threshold_rows)
    ) / RR(2) ** rows
    assert binomial_tail <= RR(2) ** (-lam)

    ell = sqrt(alpha)
    berry_esseen = KAPPA_BE * sqrt(RR(2)) / sqrt(ratio**2 - 1)
    escape = KAPPA_BD * erfc(D / 2)
    C = D / sqrt(1 - ratio**(-2))

    case_1 = ell / (1 - gamma / D)
    case_2 = 2 * ratio * D * ell / sqrt(RR(threshold_rows))
    case_3 = 2 * C * ell / (
        sqrt(pi) * (P_ROW - 2 * berry_esseen - escape)
    )
    assert modulus_loss > max(case_1, case_2, case_3)

    modular_row_probability = (
        2 * C * ell / (modulus_loss * sqrt(pi))
        + 2 * berry_esseen
        + escape
    )
    assert modular_row_probability < P_ROW

    print(
        f"{lam:3d}: exact=(gamma={float(gamma_exact):.8f}, "
        f"alpha={float(alpha_exact):.8f}, beta={float(beta_exact):.8f}, "
        f"moment={moment}); modular=(T={threshold_rows}, "
        f"D={float(D):.2f}, r={float(ratio):.2f}, b={modulus_loss})"
    )
