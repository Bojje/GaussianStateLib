import numpy as np
from GaussianStateClass import *

g = GaussianState(modes=2, alpha=1)

g.single_mode_squeeze(1, 0, modes=[1])
g.single_mode_squeeze(1, 0, modes=[2])
g.phase_shift(np.pi / 2, modes=[2])

g.beam_split(0.5, modes=[1, 2])

weight, µ, sigma_j = g.get_zero_fock_state()

g.measurement(1, sigma_j, µ)


sigma_A = g.sigma[:-2, :-2]
sigma_AB = g.sigma[:-2,-2:]
sigma_BA = g.sigma[-2:,:-2]
sigma_B = g.sigma[-2:, -2:]
prod_sigma = sigma_BA.T@np.linalg.inv(sigma_B + sigma_j)@sigma_BA

sigma_A -prod_sigma
