import numpy as np
from mpmath import mp, mpf, log, exp, sqrt, erfc, pi, cos, quad, inf, binomial, gammainc
mp.dps = 40
def bits(x): return float(log(x,2))

# ---- 1. Representation identity: phi_w(s) = E_{xi~N(0,2s)} prod cos^2(w_j xi/2)
#         where phi_w(s) = E_u exp(-s <w,u>^2), u iid D=(1/4,1/2,1/4).
def phi_direct(w, s):
    # exact expectation over u in {-1,0,1}^len(w)
    import itertools
    w = [mpf(x) for x in w]; s=mpf(s); tot=mpf(0)
    for u in itertools.product((-1,0,1), repeat=len(w)):
        p = mpf(1)
        for ui in u: p *= (mpf(1)/2 if ui==0 else mpf(1)/4)
        ip = sum(wi*ui for wi,ui in zip(w,u))
        tot += p*exp(-s*ip*ip)
    return tot
def phi_fourier(w, s):
    w=[mpf(x) for x in w]; s=mpf(s)
    dens = lambda xi: exp(-xi*xi/(4*s))/sqrt(4*pi*s)
    integ = lambda xi: dens(xi)*np.prod([float(cos(wi*xi/2)**2) for wi in w])
    # do it in mpmath
    def f(xi):
        p=mpf(1)
        for wi in w: p*=cos(wi*xi/2)**2
        return dens(xi)*p
    return 2*quad(f, [0, 20*sqrt(s)])  # even integrand
for w,s in ([(1,),2.0], [(0.6,0.8),1.5], [(0.5,0.5,0.5,0.5),2.31], [(0.9,0.3,0.31),3.0]):
    a,b=phi_direct(w,s),phi_fourier(w,s)
    print('phi identity w=%-22s s=%.2f: direct=%.10f fourier=%.10f  match=%s'%(str(w),s,float(a),float(b),abs(a-b)<1e-8))

# ---- 2. Reduction inequality: phi_w(s) <= Phi_K(large,rho2;s) + erfc(pi/(2 mu0 sqrt s))
def env(s): s=mpf(s); return max((1+exp(-s))/2, 1/sqrt(1+s))
def Phi_closure(atoms, rho2, s):
    # E_{xi~N(0,2s)} [ prod cos^2(a_j xi/2) * exp(-rho2 xi^2/4) ]
    s=mpf(s); atoms=[mpf(a) for a in atoms]; rho2=mpf(rho2)
    def f(xi):
        p=exp(-rho2*xi*xi/4)
        for a in atoms: p*=cos(a*xi/2)**2
        return exp(-xi*xi/(4*s))/sqrt(4*pi*s)*p
    return 2*quad(f,[0,20*sqrt(s)])
mu0=mpf('0.38'); s=mpf('2.31')
import itertools
def check_reduction(w):
    w=[mpf(x) for x in w]; nrm=sqrt(sum(x*x for x in w)); w=[x/nrm for x in w]
    large=[x for x in w if abs(x)>mu0]; rho2=sum(x*x for x in w if abs(x)<=mu0)
    lhs=phi_direct(w,s); rhs=Phi_closure(large,rho2,s)+erfc(pi/(2*mu0*sqrt(s)))
    return float(lhs),float(rhs),lhs<=rhs
for w in [(1,),(0.9,0.4),(0.6,0.6,0.5),(0.5,0.5,0.5,0.5),(0.7,0.5,0.4,0.3,0.2)]:
    l,r,ok=check_reduction(w); print('reduction w=%-24s: phi=%.8f  bound=%.8f  holds=%s'%(str(w),l,r,ok))

# ---- 3. Headline numbers
print()
print('crossover s_x root of (1+s)(1+e^-s)^2=4:')
from mpmath import findroot
sx=findroot(lambda s:(1+s)*(1+exp(-s))**2-4, 2.3); print('  s_x =',sx, ' env(s_x)=',env(sx))
# left tail proven bound at s=2.31, mu0=0.38: e^{30 s} (env + erfc)^256
Psi=env(mpf('2.31'))+erfc(pi/(2*mpf('0.38')*sqrt(mpf('2.31'))))
left=exp(30*mpf('2.31'))*Psi**256
print('  LEFT proven: Psi=%.9f  bound=2^%.3f'%(float(Psi),bits(left)))
# conjectured envelope at s_x
leftc=exp(30*sx)*env(sx)**256; print('  LEFT conj env: 2^%.3f'%bits(leftc))
# spike exact
sp=sum(binomial(256,i) for i in range(31))/mpf(2)**256; print('  spike exact: 2^%.3f'%bits(sp))
# right tail: moment-Markov inf_m Gamma(128+m)/Gamma(128)/337^m
from mpmath import gamma, loggamma
best=min((float((loggamma(128+m)-loggamma(128)-m*log(337))/log(2)), m) for m in range(1,400))
print('  RIGHT proven (moment-Markov): 2^%.3f at m=%d'%(best[0],best[1]))
# right tail exp-moment version inf_{0<t<1} e^{-337 t}(1-t)^{-128}
bestc=min((float((-337*mpf(t)/1000 - 128*log(1-mpf(t)/1000))/log(2)), t/1000) for t in range(1,999))
print('  RIGHT proven (MGF chi2, all w): 2^%.3f at t=%.3f'%(bestc[0],bestc[1]))
# single column: two-sided Rademacher-Gaussian. Gaussian value erfc(9.75). Pinelis factor ~1.98 one-sided
gv=erfc(mpf('9.75')); print('  SINGLE Gaussian value erfc(9.75)=2^%.3f'%bits(gv))
print('  SINGLE proven (BD 3.178x one-sided, x2 two-sided): 2^%.3f'%bits(2*mpf('3.178')*gv/2))
