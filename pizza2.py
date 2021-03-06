import numpy as np
import mpmath as mp
r = 1e-4
n = 4
"""
Creates an n fock state which is used by measure fock state
"""
def mp_comb(n: int, k: int) -> int:
    """
    Uses the mp math package to calculate the binomial coefficient of a given
    number 'num'
    """
    import mpmath as mp
    mp.dps = 160
    n_f = mp.fac(n) / (mp.fac(k) * mp.fac(n - k))

    return n_f
from scipy.special import comb
import math

sigma_m = {}
µ = {}
weight_a = {}#np.zeros(n + 1)

r_neg_2 = np.power(r,-2,dtype=np.longdouble)
r_2 = mp.power(r, 2)#, dtype=np.longdouble)

for j in range(n + 1):
    sigma_m[j] = 0.5 * (1 + (n - j) *r_2 ) / (1 - (n - j) * r_2) * np.identity(2)
    print(sigma_m[j])
    µ[j] = np.zeros(2) # np.zeros([j,0])

for j in range(n + 1):
    weight = (1 - n * (r_2)) / (1 - (n - j) * (r_2))
    weight *= (-1) ** (n - j) * mp_comb(n, j)
    weight_a[j] = weight
    print(weight)

weight_sum = mp.mpf('0')
for i in range(len(weight_a)):
    weight_sum += weight_a[i]
for j in range(n + 1):
    weight_a[j] *= 1 / weight_sum

# return weight_a, µ, sigma_m
