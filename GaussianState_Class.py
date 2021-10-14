class GaussianState:
    import numpy as np
    """ Class that defines an N mode Gaussian state based on a vectors of
    means and a covariance matrix. It will initialize a vacuum state in the
    first version of the class and different functions will implement the main
    operations (displacement, squeezing and rotation)
    """
    def __init__(self, modes=2):
        """ Initializes a vacuum state, specifying the number of modes. It will
        be expanded later to include more exotic states"""
        # mu is a vectors containing the expectations values of the momemntum
        # and position operators
        self.mu = np.zeros(2 * modes)
        # self.mu.shape = (len(mu), 1)
        # Sigma is the covariance matrix
        self.sigma = np.identity(2 * modes) * 0.5
        self.modes = modes

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
                wigner[ri, rri] = norm * np.exp(-0.5 *r_mu.T @ inv_sig @ r_mu)

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
            mi = m - 1
            F_full[mi * 2:m * 2, mi * 2:m * 2] = F
        F_dag = F_full.T
        self.sigma = F_full@self.sigma@F_dag
        for m in modes:
            mi = m - 1
            self.mu[mi * 2:m * 2] = self.mu[mi * 2:m * 2] @ F


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
        # print(start_vec)
        P_mat = np.take(P_mat, start_vec, axis=0)
        P_inv = np.linalg.inv(P_mat)
        F_full = P_inv@F_full@P_mat
        self.sigma = F_full@self.sigma@F_full.T
        self.mu = F_full@self.mu

    def s(self, ABxp1: str,i1: int, ABxp2: str, i2: int) -> float:
        """
        A function which was created to make it much simpler to access and
        work with the covariance matrix only using references to mode A or
        mode B. Primarly relevant when two challenges are used with
        interactions between the two channels
        """

        # Generate dict to look-up indicies for modes
        # To reduce computation time, this should only be done once, but to keep
        # the code clear and readable, it has been moved here

        N = self.modes
        mode_A_xi=[n*2 for n in range(N)][::2]
        kA_x = {j+1:n for j,n in enumerate(mode_A_xi)}
        mode_B_xi=[n*2 for n in range(N)][1::2]
        kB_x = {j+1:n for j,n in enumerate(mode_B_xi)}

        mode_A_xi=[n*2 - 1 for n in range(1, N)][::2]
        kA_p = {j+1:n for j,n in enumerate(mode_A_xi)}
        mode_B_xi=[n*2 - 1 for n in range(1, N)][1::2]
        kB_p = {j+1:n for j, n in enumerate(mode_B_xi)}

        indicies = []

        if ABxp2 == 'Ax':
            indicies += [kA_x[i1]]
        elif ABxp2 == 'Ap':
            indicies += [kA_p[i1]]

        if ABxp2 == 'Bx':
            indicies += [kB_x[i1]]
        elif ABxp2 == 'Bp':
            indicies += [kB_p[i1]]

        if ABxp1 == 'Ax':
            indicies += [kA_x[i1]]
        elif ABxp1 == 'Ap':
            indicies += [kA_p[i1]]

        if ABxp1 == 'Bx':
            indicies += [kB_x[i1]]
        elif ABxp1 == 'Bp':
            indicies += [kB_p[i1]]
        return self.sigma[indicies[0], indicies[1]]
