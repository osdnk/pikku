"""Verify the degree-128, weight-32 sampler parameters."""

RBF = RealBallField(256)

phi = 128
s = 32
h = phi // 2
w = s // 2
q = ZZ(18446744073709549441)
K = binomial(h, w) * 2^w


def invertibility_error(rho):
    rho = RBF(rho)
    return h * (2 * rho / (q * K) + rho^2 / q^2) / (1 - RBF(K)^(-2))


epsilon_120 = invertibility_error(RBF(3) / 2)
epsilon_110 = invertibility_error(63)

assert q.nbits() == 64
assert q.is_prime()
assert q % 256 == 129
assert K == 32016101348447354880
assert RBF(K)^2 > RBF(2)^RBF("129.59")
assert epsilon_120 < RBF(2)^(-120)
assert epsilon_110 < RBF(2)^(-110)

print(f"q = {q}")
print(f"K = {K}")
print(f"log2(K^2) = {(RBF(K)^2).log(2)}")
print(f"-log2 epsilon for rho = 3/2: {-epsilon_120.log(2)}")
print(f"-log2 epsilon for rho = 63: {-epsilon_110.log(2)}")
