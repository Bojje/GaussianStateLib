import numpy as np
class GaussianState:
    """ Class that defines an N mode Gaussian state based on a vectors of
    means and a covariance matrix. It will initialize a vacuum state in the
    first version of the class and different functions will implement the main
    operations (displacement, squeezing and rotation)
    """
    def __init__(self, modes=2, alpha=0):
        """ Initializes a vacuum state, specifying the number of modes. If a
        value for alpha is given, all modes will be set to a coherent state"""
        # mu is a vectors containing the expectations values of the momemntum
        # and position operators
        self.mu = np.zeros(2 * modes)
        # self.mu.shape = (len(mu), 1)
        # Sigma is the covariance matrix
        self.sigma = np.identity(2 * modes) * 0.5
        self.modes = modes
        if alpha != 0:
            mu_coherent = np.sqrt(2)*np.array([alpha.real,alpha.imag])
            self.mu = np.tile(mu_coherent,modes)

    def displacement(self, alpha, modes= -1):
        """ The displacement operator displaces the state. We have that
        D(alpha)Sigma D_dag(sigma)=Sigma and q -> q+sqrt(2)Re(alpha)
        p -> p+sqrt(2)Im(alpha).
        alpha indicates the displacement, and modes indicates the mode(s) which should
        be displaced. Mode is by default set to -1, which is interpreted as a
        displacement of alpha for all modes. If modes is given as a list, alpha
        must be either given as a list of complex numbers of exactly equal
        length or as a single complex number which will be applied to all the
        modes in the list
        """
        if modes == -1:
            # Handles the case where every mode is displaced
            if type(alpha) is list:
                if len(alpha) == 1:
                    alpha = alpha[0]
            assert type(alpha) is complex, "If all modes are chosen, alpha must be given\
        as a single complex number or a list of length 1"
            modes = np.arange(self.modes) + 1
            for m in modes:
                self.displacement(alpha, m)

        elif type(modes) is list:
            # A subset of modes are displaced, either with the same displacement
            # or with a unique displacement if alpha is given as a list
            if type(alpha) is not list:
                alpha = [alpha for jj in range(len(modes))]
            elif len(alpha) != len(modes):
                print("alpha must either be single complex\
    number or be a list of equal length to length of the list of modes")
            for ii, m in enumerate(modes):
                self.displacement(mu, alpha[ii], m)

        else:
            sq2_real_alpha = 1 / np.sqrt(2) * (alpha + np.conj(alpha))
            sq2_im_alpha = 1 / (np.sqrt(2) * 1j) * (alpha - np.conj(alpha))
            i = (2 * modes) % (2 * self.modes)
            if i == 0:
                # Normal indexing breaks down at the end of the list
                # (frustratingly) This is kind of strange and could/should
                # probably be fixed in some way
                self.mu[-2:] += np.abs(np.array([sq2_real_alpha, sq2_im_alpha]))
            else:
                self.mu[i - 2:i] += np.abs(np.array([sq2_real_alpha, sq2_im_alpha]))
            #self.mu += #np.array([y for x in mu for y in (x,)*2])
            print('displaced mode', modes)

    def calc_wigner_func(self, sigma, mu, start= -4, stop=4, N=100, mode=[1]):
        """
        This function is currently only called by the NonGauss class. It simply
        calculates the wigner function in a p-space of size NxN, in the range
        [-4,4] (for both x and p). It uses a given sigma and mu and traces out
        all other modes. It should be ultimately used to simplify the
        plot_wigner_func function since it performs the same calculations
        """

        for m in mode:
            mi = m - 1
            sigma_temp = self.sigma[mi * 2:m * 2, mi * 2:m * 2]
            if m == self.modes:
                mu_temp = self.mu[2 * m - 1:]
            else:
                mu_temp = self.mu[2 * m - 1:2 * m]
        # Create a vector representing q
        q = np.linspace(start, stop, N)
        norm = 1 / ((2 * np.pi) * np.sqrt(np.linalg.det(sigma_temp)))
        inv_sig = np.linalg.inv(sigma_temp)
        # r_mu = r - self.mu
        # r_mu_T = r_mu.transpose()
        wigner = np.zeros((N, N))
        for ri, r in enumerate(q):
            for rri, rr in enumerate(q):
                r_mu = np.array([r, rr]) - mu_temp
                r_mu.shape = (2, 1)
                wigner[ri, rri] = norm * np.exp(-0.5 *r_mu.T @ inv_sig @ r_mu)
        return wigner

    def plot_wigner_func(self, start= -4, stop=4, N=100, modes=[1]):
        """ the plot wigner func can only plot a single mode at a time
         Choose the modes we want to plot:
        """
        for m in modes:
            mi = m - 1
            sigma_temp = self.sigma[mi * 2:m * 2, mi * 2:m * 2]
            if m == self.modes:
                mu_temp = self.mu[2 * m - 1:]
            else:
                mu_temp = self.mu[2 * m - 1:2 * m]
        # Create a vector representing r
        q = np.linspace(start, stop, N)
        # r = np.array([y for x in q for y in (x,)*2])
        # r = np.reshape(r,(N, 2)).T
        norm = 1 / ((2 * np.pi) * np.sqrt(np.linalg.det(sigma_temp)))
        inv_sig = np.linalg.inv(sigma_temp)
        # r_mu = r - self.mu
        # r_mu_T = r_mu.transpose()
        wigner = np.zeros((N, N))
        for ri, r in enumerate(q):
            for rri, rr in enumerate(q):
                r_mu = np.array([r, rr]) - mu_temp
                r_mu.shape = (2, 1)
                wigner[rri, ri] = norm * np.exp(-0.5 *r_mu.T @ inv_sig @ r_mu)

        # wigner = norm * np.exp(-0.5 *r_mu.T @ inv_sig @ r_mu)
        print(np.sum(wigner * (q[1]-q[0])**2))
        #print(np.sum(q * wigner)* (q[1]-q[0])**2)
        import matplotlib.pyplot as plt
        qq, pp = np.meshgrid(q,q)
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(qq, pp, wigner)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Wigner Function')
        ax.set_xlabel('q')
        ax.set_ylabel('p')
        plt.show()
        # return q, wigner

    def single_mode_squeeze(self, r, theta, modes=[1]):
        """ Single mode squeezing, implemented using eq 50-52 in 2102.05748"""
        # Generate the indicies used to update the matrix
        #import scipy
        S_theta = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]])
        F = np.cosh(r) * np.identity(2) - np.sinh(r) * S_theta
        # The full matrix is first created as the identity matrix
        F_full = np.identity(self.modes * 2)
        for m in modes:
            F_full[m * 2 - 2:m * 2, m * 2 - 2:m * 2] = F
        F_dag = F_full.T
        self.sigma = F_full@self.sigma@F_dag

        for m in modes:
            self.mu[m * 2 - 2:m * 2] = self.mu[m * 2 - 2:m * 2] @ F


    def phase_shift(self, phi, modes=[1]):
        """ Implements a phase shift of phi radians. Squeezing should be applied
        to generate a state which is not invariant to rotation"""
        R_phi = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        R_full = np.identity(self.modes * 2)
        for m in modes:
            mi = m - 1
            R_full[mi * 2:m * 2, mi * 2:m * 2] = R_phi
        R_phi_dag = R_full.T
        self.sigma = R_full@self.sigma@R_phi_dag
        self.mu = R_full@self.mu


    def two_mode_squeeze(self, r, theta, modes=[1, 2]):
        """ Implements the two mode squeeze operation. Squeezing should be applied
        to generate a state which is not invariant to rotation"""
        print("Several problems with this operation, fixes made in BS but not implemented for two_mode_squeeze. Achtung")
        S_theta = -np.sinh(r) * np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]])
        F_r = np.cosh(r) * np.identity(2)
        F = np.block([[F_r, S_theta], [S_theta, F_r]])

        # Create the large F
        Z_up_r =np.zeros((4, self.modes * 2-4))
        Z_d_l = np.zeros((self.modes * 2- 4, 4))
        Z_d_r = np.zeros((self.modes * 2- 4, self.modes * 2- 4))
        F_full = np.block([[F, Z_up_r],[Z_d_l, Z_d_r]])

        # Creating the permuation matrix
        P_mat = np.identity(2*self.modes)
        start_vec = np.arange(2 * self.modes)
        # Numbers which need to be first
        # Numbers which need to be first
        modes_first = [[i * 2- 2, i * 2 - 1] for i in modes]

        for num in modes_first[::-1]:
            for i in num[::-1]:
                start_vec = np.delete(start_vec,np.where(start_vec==i))# = np.delete(start_vec, i + 4)
        for num in modes_first[::-1]:
            for i in num[::-1]:
                start_vec = np.insert(start_vec, 0, i)
        P_mat = np.take(P_mat, start_vec, axis=0)
        P_inv = np.linalg.inv(P_mat)
        F_full = P_inv@F_full@P_mat
        F_full_dag = F_full.conj().T
        self.sigma = F_full@self.sigma@F_full_dag
        self.mu = self.mu@F_full

    def beam_split(self, eta, modes=[1, 2], dag=False):
        """ Interferes two chosen modes on a beam splitter given a ratio of
        eta
        """
        rt_eta = np.sqrt(eta) * np.identity(2)
        rt_m_eta = np.sqrt(1 - eta) * np.identity(2)
        # if convention == 'upper':
        if dag == True:
            F = np.block([[rt_eta, rt_m_eta], [-rt_m_eta, rt_eta]])
        else:
            F = np.block([[rt_eta, -rt_m_eta], [rt_m_eta, rt_eta]])
        # else:
        #     F = np.block([[rt_eta, rt_m_eta], [-rt_m_eta, rt_eta]])
        # Create the large F
        Z_up_r =np.zeros((4, self.modes * 2-4))
        Z_d_l = np.zeros((self.modes * 2- 4, 4))
        Z_d_r = np.identity(self.modes * 2- 4)
        F_full = np.block([[F, Z_up_r],[Z_d_l, Z_d_r]])
        # Creating the permuation matrix
        P_mat = np.identity(2*self.modes)
        start_vec = np.arange(2 * self.modes)
        # Numbers which need to be first
        modes_first = [[i * 2- 2, i * 2 - 1] for i in modes]

        for num in modes_first[::-1]:
            for i in num[::-1]:
                start_vec = np.delete(start_vec,np.where(start_vec==i))# = np.delete(start_vec, i + 4)
        for num in modes_first[::-1]:
            for i in num[::-1]:
                start_vec = np.insert(start_vec, 0, i)
        P_mat = np.take(P_mat, start_vec, axis=0)
        P_inv = np.linalg.inv(P_mat)
        F_full = P_inv@F_full@P_mat
        self.sigma = F_full@self.sigma@F_full.T
        self.mu = F_full@self.mu

    def delay(self, delay: int):
        from itertools import cycle
        modes = self.modes
        A_modes = [n * 2 - 1 for n in range(1,int(modes/2) + 1)]
        B_modes = [n * 2 for n in range(1,int(modes/2) + 1)]

        N = modes + 4 * delay
        mode_A_xi=[n*2 for n in range(N)][::2]
        mode_B_xi=[n*2 for n in range(N)][1::2]

        mode_A_pi=[n*2 - 1 for n in range(1, N)][::2]
        mode_B_pi=[n*2 + 1 for n in range(N)][1::2]

        comb_a = [ele for comb in zip(cycle(mode_A_xi[:int(modes / 2)]), mode_A_pi[:int(modes / 2)]) for ele in comb]
        start_vec = np.zeros(modes * 2 + 4 * delay)
        for ele in comb_a:
            start_vec[ele] = ele

        comb_b = [ele for comb in zip(cycle(mode_B_xi[:int(modes / 2)]), mode_B_pi[:int(modes / 2)]) for ele in comb]
        for ele in comb_b:
            start_vec[ele + 4 * (delay)] = ele
        # Add vacuum modes to permutation matrix
        vacuum_modes = [n * 2 for n in range(1,delay + 1)]
        vacuum_modes += [A_modes[-1] + n * 2 for n in range(1,delay + 1)]
        #Fill top rows
        start_max = max(start_vec)
        for n in vacuum_modes:
            # print(n)
            ii = n*2+np.array([-2, -1])
            start_vec[ii] = start_max + 1, start_max + 2
            start_max = max(start_vec)
        start_vec = np.array([int(n) for n in start_vec])
        # Finally expand sigma
        sigma_new = np.identity(modes * 2 + 4 * delay) * 0.5
        sigma_new[:modes * 2, :modes * 2] = self.sigma
        # Use the permuation vector to delay modes
        sigma_new[:, :] = sigma_new[start_vec,:]
        sigma_new[:,:] = sigma_new[:,start_vec]
        self.sigma = sigma_new
        self.modes = modes + 2 * delay
        # N^2 memory overhead
        #print('Delayed the modes in channel b with {}'.format(delay))
        mu = np.zeros(modes * 2 + 4 * delay)
        mu[:modes * 2] = self.mu
        self.mu = mu[start_vec]

    def nullifier(self, x):
        """
        Function that takes a binary vector of length 2x modes, in the form of
        (x_1,p_2,x_2,p_2,...,x_n,p_n) and calculates the corresponding nullifier
        in the form of eg. (x_1 - x_2)**2
        """
        return np.sum(self.sigma * (x[np.newaxis,].T@x[np.newaxis,]))


    def add_vacuum(self):
        """
        Function that adds a vacuum mode to the last place in the covariance
        matrix
        """
        self.modes += 1
        new_sigma = np.zeros((2 * self.modes, 2 * self.modes))
        new_sigma[:-2, :-2] = self.sigma
        new_sigma[-2:, -2:] = 1 / 2 * np.identity(2)
        self.sigma = new_sigma
        self.mu = np.append(self.mu, np.zeros(2))

    def permute_mode(self, mode, target_mode):
        """
        A general function to permute a single mode to either the last mode or
        to place the last mode
        it is primarily meant to be called from other functions in the class and
        should be used with care, since it does not respect the distinction
        between an A and a B channel which are used extensively while creating
        cluster states
        """
        print("Implemented, needs testing")
        start_vec = np.arange(2 * self.modes)
        if target_mode == -1:
            start_vec = np.delete(start_vec, 2 * mode - 1)
            start_vec = np.delete(start_vec, 2 * mode - 2)
            start_vec = np.append(start_vec,2 * mode - 2)
            start_vec = np.append(start_vec,2 * mode - 1)
        if mode == -1:
            end_modes = start_vec[-2:][::-1]
            start_vec = np.delete(start_vec, -1)
            start_vec = np.delete(start_vec, -1)
            for j in end_modes:
                start_vec = np.insert(start_vec,2*target_mode-2,j)
        self.sigma[:, :] = self.sigma[start_vec,:]
        self.sigma[:,:] = self.sigma[:,start_vec]

        self.mu = self.mu[start_vec]


    def multi_gauss(self, xi, mu, sigma):
        """
        Function that calculates the normalized multivariate gaussian
        distribution for a given set of means, variances and variables (xi)
        """
        xi_mu = xi - mu
        xi_mu_t =  (xi_mu)[np.newaxis, ].T
        s_inv = np.linalg.inv(sigma)
        s_det_sqrt = np.sqrt(np.linalg.det(2 * np.pi * sigma))

        return np.exp(-0.5 * xi_mu@s_inv@xi_mu_t) / s_det_sqrt

    def measurement(self, mode, sigma_j, mu_j):
        """
        Given a mode measured as being in a given (almost) arbitrary state,
        this function updates the covariance matri(x/cies), means and
        coefficients.
        Homodyne mesurements are handled by a seperate function. Threshold
        detection will also be implemented at some time
        """

        # First we permute the covariance matrix so the chosen mode is made into
        # the last mode
        self.permute_mode(mode, -1)
        sigma_A = self.sigma[:-2, :-2]
        sigma_AB = self.sigma[:-2,-2:] # Upper right
        sigma_BA = self.sigma[-2:,:-2] # lower left
        sigma_B = self.sigma[-2:, -2:]
        inv_sigma = np.linalg.inv(sigma_B + sigma_j)
        # The covariance matrix and mean are updated
        self.sigma = sigma_A - sigma_AB@inv_sigma@sigma_BA
        self.mu = self.mu[:-2] + sigma_AB@inv_sigma@(self.mu[-2:] - mu_j)
        # Number of modes are updated
        self.modes -= 1
        # A vacuum mode is added to replace the measured state
        self.add_vacuum()
        mu_B = self.mu[-2:]

        # And the vacuum mode is returned back to the original place of the
        # measured mode

        weight_update = self.multi_gauss(mu_j, mu_B, sigma_B)
        self.permute_mode(-1, mode)


        return weight_update
        # return None#gauss_gauss

    def homodyne_measurement(self, mode, theta=0, u=np.zeros(2)):
        """
        Performs a homodyne measurement with some given value theta.
        For theta=0 the x-quadrature is measured and for theta=pi/2 the
        p-quadrature is measured
        """
        # A phase shift of theta is applied so we make sure that we measure
        # the chosen quadrature
        self.phase_shift(theta, [mode])
        self.permute_mode(mode, -1)
        sigma_A = self.sigma[:-2, :-2]
        sigma_AB = self.sigma[:-2,-2:]
        sigma_BA = self.sigma[-2:,:-2]
        sigma_B = self.sigma[-2:, -2:]
        pi_mat = np.array([[1, 0], [0, 0]])
        B11 = sigma_B[0, 0]

        self.sigma = sigma_A - 1 / B11 * sigma_AB@pi_mat@sigma_BA
        self.mu = self.mu[:-2] - 1 / B11 * sigma_AB@pi_mat@(u - self.mu[-2:])
        self.modes -= 1
        # A vacuum mode is added to replace the measured state
        self.add_vacuum()

        norm = 1 / np.sqrt(2 * np.pi * B11)
        weight_update = norm * np.exp(-1 / (2 * B11) * (u[0] - self.mu[-2]) ** 2)
        # Move back the measured mode
        self.permute_mode(-1, mode)

        return weight_update

    def get_zero_fock_state(self, r=1e-1):
        """
        The get_zero_fock_state generates the |0><0| state and is meant to be
        used for testing and will be supplanted by a more general function in
        the wrapper class NonGauss
        """
        import scipy.special
        import math
        n = 0


        # Might as well be written as 1/2*identity(2)
        for j in range(n + 1):
            # This loop is simply so that the function can be generalized, it
            # serves no actual function
            sigma_m = 1 / 2 * (1 + (n - j) * r ** 2) / (1 - (n - j) * r ** 2) * np.identity(2)
            µ = np.zeros(2) # np.zeros([j,0])
        for j in range(n + 1):
            weight = (1 - n * r ** 2) / (1 - (n - j) * r ** 2)
            weight *= scipy.special.binom(n, j)
            weight *= (-1) ** (n - j)

        # Needs a normalization to become general

        return weight, µ, sigma_m
