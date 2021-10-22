import numpy as np
from GaussianStateClass import GaussianState
class NonGaussianState:
    """
    This class is used to act as a wrapper for the existing class
    'GaussianState'.
    It implements Non-Gaussian operations by creating a summation over different
    Gaussian States, the primary task of this class is to keep track of the
    different weights of the gaussians and update them
    """
    def __init__(self, modes=2, alpha=0):
        """
        initializes the first gaussian state, by default a 2 mode vacuum
        state.
        The different gaussians are 0-indexed. The weights are also, by virtue
        of being in an array 0-indexed.
        Modes are 1-indexed
        """
        g = GaussianState(modes=modes, alpha=alpha)

        # Save a vector containing the weights of the gaussian class
        # These are set to 1 to begin with
        self.weight_m = np.ones(1)
        # Save the first GaussianState object in a dictionary which will contain
        # all of the gaussian states
        self.g_dic = {0:g}


    def single_mode_squeeze(self, r, theta, modes=1):
        """
        Single mode squeezing operation which is performed on each of the
        different Gaussians saved in the NonGauss class
        """
        for i in self.g_dic:
            g = self.g_dic[i]
            g.single_mode_squeeze(r, theta, modes=[modes])
            # save the squeezed gaussian
            self.g_dic[i] = g

    def phase_shift(self, phi, modes=1):
        """
        Phase shift operation which is performed on each of the
        different Gaussians saved in the NonGauss class
        """
        for i in self.g_dic:
            g = self.g_dic[i]
            g.phase_shift(phi, modes=[modes])
            # save the phase shifted gaussian
            self.g_dic[i] = g

    def beam_split(self, eta, modes=[1, 2], dag=False):
        """
        Beam split operation which is performed on each of the
        different Gaussians saved in the NonGauss class
        """
        for i in self.g_dic:
            g = self.g_dic[i]
            g.beam_split(eta, modes=modes)
            # save the phase shifted gaussian
            self.g_dic[i] = g

    def measure_fock_state(self, mode=1, n=0, r=1e-2):
        """
        This function measures the mode given by mode. Number of photons
        measured is given by 'n' as an int, signifying the number
        of photons measured and it updates the weights object
        """
        import copy

        # First generate the state which was measured
        weight_j, µ_j, sigma_j = self.gen_fock_state(n, r=r)
        print(len(self.g_dic))
        new_weight = np.zeros((n + 1) * len(self.g_dic))
        new_gs = {}
        nn = 0
        if mode < 1:
            num_modes = self.g_dic[0].modes
            print("The measured mode must be between 1 and {}".format(num_modes))
            return None
        else:
            for j in range(n + 1):
                # This loops over all of the different gaussians generated by the
                # gen_fock_state class. The weight corresponding to each is weight_j
                for i in self.g_dic:
                    # This loops over all over the gaussians which are already
                    # contained in the NG class and performs a measurement on each
                    # of them
                    g = copy.deepcopy(self.g_dic[i])
                    # need to generate the weights!
                    weight_G= g.measurement(mode, sigma_j[j], µ_j[j])
                    new_weight[nn] = weight_j[j] * weight_G * self.weight_m[i - 1]
                    nn += 1
                    new_gs[nn] = g

            # Normalize the weights
            new_weight /= np.sum(new_weight)
            # Save the results into self
            self.g_dic = new_gs
            self.weight_m = new_weight
    def homodyne_measurement(self, mode, theta=0, u=np.zeros(2)):
        """
        The homodyne measurement performs the same homodyne measurement across
        all of the different Gaussian distributions which are contained in
        the Non-Gauss class
        """



    def gen_fock_state(self, n, r=1e-1):
        """
        Creates an n fock state which is used by measure fock state
        """
        from scipy.special import comb
        import math
        import mpmath as mp
        import sympy

        sigma_m = {}
        µ = {}
        weight_a = np.zeros(n + 1)

        r_2 = np.power(r, 2, dtype=np.longdouble)

        for j in range(n + 1):
            sigma_m[j] = 0.5 * (1 + (n - j) * r_2) / (1 - (n - j) * r_2) * np.identity(2)
            µ[j] = np.zeros(2) # np.zeros([j,0])

        r_2 = mp.power(r, 2)

        for j in range(n + 1):
            weight = (1 - n * (r_2)) / (1 - (n - j) * (r_2))
            print(weight)
            weight *= (-1) ** (n - j) * comb(n, j)
            weight_a[j] = weight
            print(weight)

        weight_sum = mp.mpf('0')
        # weight_sum = np.sum(weight_a)
        for i in range(len(weight_a)):
            weight_sum += weight_a[i]
        weight_sum = sympy.Float(str(weight_sum),70)#Float(str(weight_sum),50)
        for j in range(n + 1):
            weight_a[j] *= 1 / weight_sum

        return weight_a, µ, sigma_m


    def init_fock_state(self,n, r=1e-1):
        """
        Creates an n fock state and saves it to the NonGauss object.
        if n is an int, the function will create a single mode
        n-foton fock state. If n is given as either a list or an array, it will
        create a fock state for each entry in n
        """
        from GaussianStateClass import GaussianState
        import itertools
        if type(n) == int:
            self.g_dic ={}
            weight, µ, sigma_m = self.gen_fock_state(n, r=r)
            for i in range(len(µ)):
                g = GaussianState(modes=1)
                g.sigma = sigma_m[i]
                g.mu = µ[i]
                self.g_dic[i] =g
            self.weight = weight

        else:
            # First all of the different combinations of gausians are calculated
            set_n = np.arange(n + 1)
            set_n = [np.arange(nn + 1) for nn in n]
            # make iterable cartesian product
            cart_prod = itertools.product(*set_n)

            # the different gaussians are created and stored in a library
            weight, µ, sigma_m = self.gen_fock_state(np.max(n), r=r)



    def calc_wigner_func(self, start= -4, stop=4, N=100, mode=1):
        """
        Calculates the Wigner function for a single mode. It is meant to be
        called by the plot wigner function
        """
        NN = len(self.g_dic)
        wigner = np.zeros((N, N))
        modes = [mode]
        for i in self.g_dic:
            print(i)
            g = self.g_dic[i]
            w = self.weight[i] * g.calc_wigner_func(g.sigma,\
                                                    g.mu, start = start,\
                                                    stop=stop, N=N, mode=modes)
            wigner += w

        return wigner

    def plot_wigner(self, start= -4, stop=4, N=101, mode=1,
                    title='Wigner Function'):
        """
        kek
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        wigner = self.calc_wigner_func(N=N)
        mpl.rc('image', cmap='bwr')
        # center around 0, so white corresponds to 0
        vmin = np.min(wigner)
        vmax = np.max(wigner)
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        q = np.linspace(-4, 4, 101)
        qq, pp = np.meshgrid(q,q)
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(qq, pp, wigner,vmin=vmin, vmax=vmax, norm=norm)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title(title)# Burde fixe farver
        ax.set_xlabel('q')
        ax.set_ylabel('p')
        plt.show()

# Example that plots the wigner plot of the fock state |1>
ng = NonGaussianState()
# weight,µ,sigma_m=ng.gen_fock_state(19)
sq = 0.1
ng.single_mode_squeeze(sq, 0, modes=2)
ng.single_mode_squeeze(sq,0, modes=1)
ng.phase_shift(np.pi / 2, modes=2)
ng.beam_split(0.5, modes=[1, 2])

ng.measure_fock_state(mode=1, n=0, r=1e-2)
# ng.measure_fock_state(mode=2, n=0, r=1e-2)


# g = GaussianState(modes=4)
# g.single_mode_squeeze(1, 0, modes=[4])
# g.plot_wigner_func(modes=[4])
# u = np.linspace(-6, 6, 1000)
# sq = 1
# save_u = np.zeros(np.shape(u))
# for i, uu in enumerate(u):
#     g = GaussianState(modes=10)
#     g.single_mode_squeeze(sq, 0, modes=[4])
#     save_u[i] = g.homodyne_measurement(4,theta=np.pi / 2, u=np.array([uu, 0]))
# import matplotlib.pyplot as plt
# plt.plot(u, save_u)
# plt.rc('font', size=10) #controls default text size

# plt.rc('axes', titlesize=12) #fontsize of the title

# plt.rc('axes', labelsize=12) #fontsize of the x and y labels
# plt.title('Homodyne Measurement Distribution for $\hat{p}$')
# plt.xlabel('u')
# plt.ylabel('$P_{\hat{p}}(u)$')
# # ax.set_ylabel('p')
# plt.show()
# # ng.init_fock_state(0)

sigma_j = ng.g_dic[0].sigma
µ = np.zeros(2)
weight = 1
g.measurement(2, sigma_j, µ)
ng.init_fock_state(4, r=1e-1)
ng.plot_wigner()
