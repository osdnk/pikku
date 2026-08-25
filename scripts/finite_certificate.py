import numpy as np
import itertools, sys

def envf(s):
    return max((1 + np.exp(-s)) / 2, (1 + s) ** -0.5)

def signs(k):
    S = np.array(list(itertools.product([-1, 0, 1], repeat=k)), dtype=float)
    P = np.prod(np.where(S == 0, 0.5, 0.25), axis=1)
    return S, P

def certify(k, mu0, s, safety=1e-9, cap=6e7):
    """Certify Phi_k(q; rho2=1-sum q; s) <= env(s)-safety over q_i in [mu0^2, 1], sum q<=1.
       Phi_k = (1/sqrt A) sum_pat P exp(-(s/A) x^2), A=1+rho2*s, x=sum(+-sqrt q_i).
       First-order interval bound with rigorous Lipschitz L per coord."""
    S, P = signs(k)
    q0 = mu0 * mu0
    e = envf(s)
    # rigorous global Lipschitz bound on |dPhi/dq_i| for q_i>=q0, s fixed (see note):
    #   |dPhi/dq_i| <= E_{N(0,2s)}[ |xi|/(4 mu0) + xi^2/4 ] = sqrt(s/pi)/(2 mu0) + s/2
    L = np.sqrt(s / np.pi) / (2 * mu0) + s / 2.0
    # rigorous per-row Hessian bound Lam (see report): moments of N(0,2s)
    m1 = 2 * np.sqrt(s / np.pi); m2 = 2 * s
    m3 = 2 * np.sqrt(2 / np.pi) * (2 * s) ** 1.5; m4 = 12 * s * s
    Eoff = m4 / 16 + m3 / (8 * mu0) + m2 / (16 * mu0**2)
    Edia = m4 / 16 + m3 / (8 * mu0) + m2 / (8 * mu0**2) + m1 / (8 * mu0**3)
    Lam = (k - 1) * Eoff + Edia
    box = np.array([[q0] * k + [1.0] * k], dtype=float)  # [lo(k), hi(k)]
    stack = [box]
    ncert = npro = 0
    worst = np.inf
    while stack:
        B = stack.pop()
        if len(B) > 200000:
            stack.append(B[200000:]); B = B[:200000]
        npro += len(B)
        if npro > cap:
            return False, ncert, npro, worst
        lo, hi = B[:, :k], B[:, k:]
        feas = (lo.sum(axis=1) <= 1.0 + 1e-15)
        B, lo, hi = B[feas], lo[feas], hi[feas]
        if len(B) == 0:
            continue
        hi = np.minimum(hi, np.maximum(q0, 1.0 - (lo.sum(axis=1, keepdims=True) - lo)))
        c = (lo + hi) / 2
        r = (hi - lo) / 2
        rho2 = np.maximum(1.0 - c.sum(axis=1), 0.0)
        A = 1 + rho2 * s
        a = np.sqrt(c)
        x = a @ S.T
        E = np.exp(-(s / A)[:, None] * x * x)
        phi_c = (E @ P) / np.sqrt(A)
        PE = E * P
        s0 = PE.sum(axis=1)
        s2 = (PE * x * x).sum(axis=1)
        s1 = (PE * x) @ S
        common = (s / 2) * A**-1.5 * s0 - A**-0.5 * (s * s / A**2) * s2
        grad = A[:, None]**-0.5 * (-s / (a * A[:, None])) * s1 + common[:, None]
        rs = r.sum(axis=1)
        ub2 = phi_c + (np.abs(grad) * r).sum(axis=1) + 0.5 * Lam * rs * rs
        ub = np.minimum(phi_c + L * rs, ub2)
        ok = ub <= e - safety
        ncert += ok.sum()
        margin = (e - safety) - ub
        if (~ok).any():
            worst = min(worst, margin[~ok].min())
        bad = B[~ok]
        if len(bad) == 0:
            continue
        w = bad[:, k:] - bad[:, :k]
        d = w.argmax(axis=1)
        new = []
        for j in range(k):
            sel = bad[d == j]
            if len(sel) == 0:
                continue
            b1, b2 = sel.copy(), sel.copy()
            mid = (sel[:, j] + sel[:, k + j]) / 2
            b1[:, k + j] = mid
            b2[:, j] = mid
            new += [b1, b2]
        stack.append(np.concatenate(new, axis=0))
    return True, ncert, npro, worst

if __name__ == "__main__":
    mu0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.45
    s = float(sys.argv[2]) if len(sys.argv) > 2 else 2.31
    K = int(np.floor(1 / mu0**2))
    from scipy.special import erfc
    eps = erfc(np.pi / (2 * mu0 * np.sqrt(s)))
    print(f"mu0={mu0}  K={K}  s={s}  env(s)={envf(s):.10f}  erfc slack eps={eps:.3e}")
    print(f"L (Lipschitz per coord) = {np.sqrt(s/np.pi)/(2*mu0)+s/2:.4f}")
    allok = True
    for k in range(1, K + 1):
        ok, n, npro, w = certify(k, mu0, s)
        allok &= ok
        print(f"  k={k}: certified={ok}  cert_boxes={n}  processed={npro}  worst_margin={w:.2e}")
    Psi = envf(s) + eps
    print(f"{'ALL CERTIFIED' if allok else 'INCOMPLETE'}   Psi(s)=env+eps={Psi:.10f}   "
          f"log2 tail bound at this s = {(30*s + 256*np.log(Psi))/np.log(2):.6f}")
