from GaussianStateClass import *
import numpy as np

N = 100

mode_A_xi=[n*2 for n in range(N)][::2]
kA_x = {j+1:n for j,n in enumerate(mode_A_xi)}
mode_B_xi=[n*2 for n in range(N)][1::2]
kB_x = {j+1:n for j,n in enumerate(mode_B_xi)}

mode_A_pi=[n*2 - 1 for n in range(1, N)][::2]
kA_p = {j+1:n for j,n in enumerate(mode_A_pi)}
mode_B_pi=[n*2 - 1 for n in range(1, N)][1::2]
kB_p = {j+1:n for j, n in enumerate(mode_B_pi)}

modes = N
runs = 40
squeezing = np.linspace(0, 1, runs)
A_modes = [n * 2 - 1 for n in range(1,int(modes/2) + 1)]
B_modes = [n * 2 for n in range(1,int(modes/2) + 1)]

# g = GaussianState(modes=N)
# g.single_mode_squeeze(1, 0, modes=[1])
# g.single_mode_squeeze(1, np.pi, modes=[2])

# g.delay(1)

# creating vectors to store the variances for the EPR state
var_q = np.zeros(runs)
var_p = np.zeros(runs)
m_A = 1
m_B = 1
for ii, sq in enumerate(squeezing):
    g = GaussianState(modes=modes)
    modes = g.modes
    A_modes = [n * 2 - 1 for n in range(1,int(modes/2) + 1)]
    B_modes = [n * 2 for n in range(1,int(modes/2) + 1)]
    # EPR state tests
    for n in A_modes:
        # Squeezing in the position quadrature
        # for modes in channel A
        g.single_mode_squeeze(sq, 0, modes=[n])
    for n in B_modes:
        # Squeezing modes in channel B
        g.single_mode_squeeze(sq, 0, modes=[n])
        # Phase rotation in channel B
        g.phase_shift(np.pi / 2, modes=[n])

    # BS1
    for n_a, n_b in zip(A_modes, B_modes):
        g.beam_split(0.5, modes=[n_a, n_b])
    x = np.zeros(2*g.modes)
    x[kA_x[1]] = 1
    x[kB_x[1]] = 1
    print(g.nullifier(x))
    x = np.zeros(2*g.modes)
    x[kA_p[1]] = 1
    x[kB_p[1]] = -1

    s_2 = np.copy(g.sigma)
    g.delay(1)
    modes_new = g.modes
    A_modes = [n * 2 - 1 for n in range(1,int(modes_new/2) + 1)]
    B_modes = [n * 2 for n in range(1,int(modes_new/2) + 1)]

    # BS2 with delay
    for n_a, n_b in zip(A_modes, B_modes):
        g.beam_split(0.5, modes=[n_a, n_b])

    delay = 12
    g.delay(delay)
    modes_new = g.modes
    A_modes = [n * 2 - 1 for n in range(1,int(modes_new/2) + 1)]
    B_modes = [n * 2 for n in range(1,int(modes_new/2) + 1)]

    # BS3
    for n_a, n_b in zip(A_modes, B_modes):
        g.beam_split(0.5, modes=[n_a, n_b])

    k = 30
    x = np.zeros(2*g.modes)
    x[kA_x[k]] = 1
    x[kB_x[k]] = 1
    x[kA_x[k + 1]] = -1
    x[kB_x[k + 1]] = -1
    x[kA_x[k + delay]] = -1
    x[kB_x[k + delay]] = 1
    x[kA_x[k + delay + 1]] = -1
    x[kB_x[k + delay + 1]] = 1
    var_q[ii] = g.nullifier(x)

    x = np.zeros(2*g.modes)
    x[kA_p[k]] = 1
    x[kB_p[k]] = 1
    x[kA_p[k + 1]] = 1
    x[kB_p[k + 1]] = 1
    x[kA_p[k + delay]] = -1
    x[kB_p[k + delay]] = 1
    x[kA_p[k + delay + 1]] = 1
    x[kB_p[k + delay + 1]] = -1
    var_p[ii] = g.nullifier(x)

import matplotlib.pyplot as plt
plt.plot(squeezing, var_p, 'x')
plt.plot(squeezing, 4 * np.exp(-2 * squeezing), label='Expected')
plt.title('var($\hat{n}^p$) for a 2D cluster state')
plt.xlabel('Squeezing (r)')
plt.ylabel('var($\hat{n}^p$)')
plt.show()


plt.plot(squeezing, var_q, 'x', label='Generated by Gaussian class')
plt.plot(squeezing, 4 * np.exp(-2 * squeezing), label='Expected')
#modes_first = [[i * 2- 2, i * 2 - 1] for i in modes]]
plt.legend()
plt.title('var($\hat{n}^q$) for a 2D cluster state')
plt.xlabel('Squeezing (r)')
plt.ylabel('var($\hat{n}^q$)')
plt.show()