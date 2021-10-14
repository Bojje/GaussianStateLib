import numpy as np

class NonGaussianState:
    """
    This class is used to act as a wrapper for the existing class
    'GaussianState'.
    It implements Non-Gaussian operations by creating a summation over different
    Gaussian States, the primary task of this class is to keep track of the
    different weights of the gaussians and update them
    """
    def __init__(self, modes=2, alpha=0):
        from GaussianStateClass import GaussianState
        """
        initializes the first gaussian state, by default a 2 mode vacuum
        state.
        """
        g = GaussianState(modes=modes, alpha=alpha)

        # Save a vector containing the weights of the gaussian class
        self.weight = np.ones(1)
        # Save the first GaussianState object in a dictionary which will contain
        # all of the gaussian states
        self.g_dic = {0: g}

    def measure_fock_state(self, mode=1, n=0):
        """
        This function measures the mode given by mode. Number of photons
        measured is given by 'n' as an int, signifying the number
        of photons measured and it updates the weights object
        """

        # First generate the state which was measured
        weight, µ, sigma_j = self.gen_fock_state(n)
        new_g = {}
        new_weight = np.zeros(1)
        nn = 0
        for j in range(n + 1):
            for i in self.g_dic:
                g = self.g_dic[i]
                # need to generate the weights!
                new_g[nn]= g.measurement(mode, sigma_j)

                nn += 1

            nn += 1

    def gen_fock_state(self, n, r=1e-1):
        """
        Creates an n fock state which is used by measure fock state
        """
        import math
        import mpmath as mp

        def mp_comb(n: int, k: int) -> int:
            """
            Uses the mp math package to calculate the binomial coefficient of a given
            number 'num'
            """
            import mpmath as mp
            mp.dps = 160
            n_f = mp.fac(n) / (mp.fac(k) * mp.fac(n - k))

            return n_f


        sigma_m = {}
        µ = {}
        weight_a = {}#np.zeros(n + 1)

        for j in range(n + 1):
            sigma_m[j] = 0.5 * (1 + (n - j) * r ** 2) / (1 - (n - j) * r ** 2) * np.identity(2)
            µ[j] = np.zeros(2) # np.zeros([j,0])

        r_2 = mp.power(r, 2)
        for j in range(n + 1):
            weight = (1 - n * (r_2)) / (1 - (n - j) * (r_2))
            print(weight)
            weight *= (-1) ** (n - j) * mp_comb(n, j)
            weight_a[j] = weight

        weight_sum = mp.mpf('0')
        for i in range(len(weight_a)):
            weight_sum += weight_a[i]

        for j in range(n + 1):
            weight_a[j] *= 1 / weight_sum

        return weight_a, µ, sigma_m


    def init_fock_state(self,n, r=1e-1, modes=1):
        """
        Creates an n fock state and saves it to the NonGauss object
        """
        from GaussianStateClass import GaussianState
        if modes != 1:
            print('Multiple modes have not been implemented yet!')
        else:
            self.g_dic ={}
            weight, µ, sigma_m = self.gen_fock_state(n, r=r)
            for i in range(len(µ)):
                g = GaussianState(modes=1)
                g.sigma = sigma_m[i]
                g.mu = µ[i]
                self.g_dic[i] = g
            self.weight = weight

    def calc_wigner_func(self, start= -4, stop=4, N=100, mode=1):
        """
        Calculates the Wigner function for a single mode
        """
        NN = len(self.g_dic)
        wigner = np.zeros((N, N))
        modes = [mode]
        for i in range(NN):
            print(i)
            g = self.g_dic[i]
            w = self.weight[i] * g.calc_wigner_func(g.sigma,\
                                                    g.mu, start = start,\
                                                    stop=stop, N=N, mode=modes)
            wigner += w

        return wigner


# Example that plots the wigner plot of the fock state |1>
ng = NonGaussianState()
# weight,µ,sigma_m=ng.gen_fock_state(19)
ng.init_fock_state(4, r=1e-3)
wigner = ng.calc_wigner_func(N=101)

import matplotlib as mpl
mpl.rc('image', cmap='bwr')
# center around 0, so white corresponds to 0
vmin = np.min(wigner)
vmax = np.max(wigner)
norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

import matplotlib.pyplot as plt

q = np.linspace(-4, 4, 101)
qq, pp = np.meshgrid(q,q)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(qq, pp, wigner,vmin=vmin, vmax=vmax, norm=norm)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Wigner Function')# Burde fixe farver
ax.set_xlabel('q')
ax.set_ylabel('p')
plt.show()
# 1/np.pi



#quad data type
