import numpy as np
# def add_mode(self, n=1):
class GaussianState:
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


#""" Maybe the following should be converted into a function, which is used to
# access sigma """
N = 500
mode_A_xi=[n*2 for n in range(N)][::2]
kA_x = {j+1:n for j,n in enumerate(mode_A_xi)}
mode_B_xi=[n*2 for n in range(N)][1::2]
kB_x = {j+1:n for j,n in enumerate(mode_B_xi)}

mode_A_xi=[n*2 - 1 for n in range(1, N)][::2]
kA_p = {j+1:n for j,n in enumerate(mode_A_xi)}
mode_B_xi=[n*2 - 1 for n in range(1, N)][1::2]
kB_p = {j+1:n for j, n in enumerate(mode_B_xi)}

N = 100
var_q = np.zeros(N)
var_p = np.zeros(N)
m1 = 51 #non-even mode A
m2 = 2 #even mode B
m_A = int(np.ceil(m1 / 2))
m_B = int(m2 / 2)
squeezing = np.linspace(0, 3, N)
for i, sq in enumerate(squeezing):
    gauss = GaussianState(modes=100)
    #gauss.displacement(1 + 0.1j, modes=1)
    # gauss.two_mode_squeeze(sq, np.pi / 2, modes=[m1, m2])
    gauss.single_mode_squeeze(sq, 0, modes=[m1])
    gauss.single_mode_squeeze(sq, 0, modes=[m2])
    gauss.phase_shift(np.pi / 2, modes=[m2])
    gauss.beam_split(0.5, modes=[m1, m2])
    # sigma_old = gauss.sigma
    # gauss.plot_wigner_func(start= -4, stop=4, N=100, modes=[2])
    var_q[i] = gauss.sigma[kA_x[m_A],kA_x[m_A]]+gauss.sigma[kB_x[m_B],kB_x[m_B]] - 2*gauss.sigma[kB_x[m_B],kA_x[m_A]]
    var_p[i] = gauss.sigma[kA_p[m_A],kA_p[m_A]]+gauss.sigma[kB_p[m_B],kB_p[m_B]] + 2*gauss.sigma[kB_p[m_B],kA_p[m_A]]
    # var_p[i] = gauss.sigma[2,2]+gauss.sigma[3,3]-2*gauss.sigma[3,1]
import matplotlib.pyplot as plt
plt.plot(squeezing, var_p, 'x')
plt.plot(squeezing, np.exp(-2 * squeezing), label='Expected')
plt.title('var($\hat{n}^p$) for an EPR state')
plt.xlabel('Squeezing (r)')
plt.ylabel('var($\hat{n}^p$)')
plt.show()


import matplotlib.pyplot as plt
plt.plot(squeezing, var_q, 'x', label='Generated by Gaussian class')
plt.plot(squeezing, np.exp(-2 * squeezing), label='Expected')
#modes_first = [[i * 2- 2, i * 2 - 1] for i in modes]]
plt.legend()
plt.title('var($\hat{n}^q$) for an EPR state')
plt.xlabel('Squeezing (r)')
plt.ylabel('var($\hat{n}^q$)')
plt.show()

# Generating a 1D cluster state:
N = 80#must always be even
NN = 10
squeezing = np.linspace(0, 1, NN)
var_q = np.zeros(NN)
var_q0 = np.zeros(NN)
var_q1 = np.zeros(NN)
var_p = np.zeros(NN)

var_a = np.zeros(NN)
var_b = np.zeros(NN)
var_c = np.zeros(NN)
var_d = np.zeros(NN)
var_e = np.zeros(NN)
var_f = np.zeros(NN)
var_g = np.zeros(NN)
var_h = np.zeros(NN)

m1 = 5 #non-even mode A
m2 = 6 #even mode B
m_A = int(np.ceil(m1 / 2))
m_B = int(m2 / 2)
for ii, sq in enumerate(squeezing):
    gauss = GaussianState(modes=N)
    A_modes = [n * 2 - 1 for n in range(1,int(N/2) + 1)]
    B_modes = [n * 2 for n in range(1,int(N/2) + 1)]
    for i, n in enumerate(A_modes):
        gauss.single_mode_squeeze(sq, 0, modes=[n])
    for i, n in enumerate(B_modes):
        # pass
        gauss.single_mode_squeeze(sq, 0, modes=[n])
        gauss.phase_shift(np.pi / 2, modes=[n])

    # The following checks that the 4 requirements for stage 2 have been passed:
    test_1 = []
    test_1 += [np.sum([round(gauss.sigma[kA_x[i],kA_x[i]], 6)==round(np.exp(-sq)*np.exp(-sq)*0.5, 6) for i in range(1,int(np.ceil(N/2))+1)])]
    test_1 += [np.sum([round(gauss.sigma[kA_p[i],kA_p[i]], 6)==round(np.exp(1 * sq)*np.exp(1 * sq)*1/2, 6)for i in range(1,int(np.ceil(N/2))+1)])]
    test_1 += [np.sum([round(gauss.sigma[kB_x[i],kB_x[i]], 6)==round(np.exp(1 *sq)*np.exp(1 * sq)*1/2, 6) for i in range(1,int(np.ceil(N/2))+1)])]
    test_1 += [np.sum([round(gauss.sigma[kB_p[i],kB_p[i]], 6)==round(np.exp(-1 *sq)*np.exp(-1 * sq)*1/2, 6) for i in range(1,int(np.ceil(N/2))+1)])]
    # print(sum(test_1) == N * 2)
    # Save sigma from stage 2
    s_2 = np.copy(gauss.sigma)
    # print(gauss.sigma[kA_x[1],kA_x[1]])
    for i in np.arange(len(A_modes)):
        A = A_modes[i]
        B = B_modes[i]
        gauss.beam_split(0.5, modes=[A_modes[i], B_modes[i]]) # Right  fukawa for 1D
        # gauss.beam_split(0.5, modes=[ n * 2, n * 2 + 1]) # Right for Mikkel
        # print('1d pair', n * 2 + 1, n * 2)
    # After the BS is added, the following checks must be true:

    s_3 = np.copy(gauss.sigma)

    Ax_3 = [s_3[kA_x[i], kA_x[i]] for i in range(1, int(np.ceil(N / 2) + 1))]
    Ax_3_theory = [0.5* (s_2[kA_x[i], kA_x[i]] + s_2[kB_x[i], kB_x[i]] - s_2[kA_x[i], kB_x[i]] - s_2[kB_x[i], kA_x[i]]) for i in range(1,int(np.ceil(N/2))+1)]

    Ap_3 = [s_3[kA_p[i], kA_p[i]] for i in range(1, int(np.ceil(N / 2) + 1))]
    Ap_3_theory = [1 / 2 * (s_2[kA_p[i], kA_p[i]] + s_2[kB_p[i], kB_p[i]] - s_2[kA_p[i], kB_p[i]] - s_2[kB_p[i], kA_p[i]]) for i in range(1,int(np.ceil(N/2))+1)]

    Bx3_theory = [1 / 2 * (s_2[kB_x[i], kB_x[i]] + s_2[kA_x[i], kA_x[i]] + s_2[kB_x[i], kA_x[i]] + s_2[kA_x[i], kB_x[i]]) for i in range(1,int(np.ceil(N/2))+1)]
    Bx_3 = [s_3[kB_x[i], kB_x[i]] for i in range(2, int(np.ceil(N / 2) + 1))]

    Bp_3_theory = [1 / 2 * (s_2[kB_p[i], kB_p[i]] + s_2[kA_p[i], kA_p[i]] + s_2[kB_p[i], kA_p[i]] + s_2[kA_p[i], kB_p[i]]) for i in range(1,int(np.ceil(N/2))+1)]
    Bp_3 = [s_3[kB_p[i], kB_p[i]] for i in range(1, int(np.ceil(N / 2) + 1))]
    # print(Bp_3)

    ABx_3_theory = [1 / 2 * (s_2[kA_x[i], kA_x[i]] + s_2[kA_x[i], kB_x[i]] - s_2[kB_x[i], kB_x[i]] - s_2[kB_x[i], kA_x[i]]) for i in range(2,int(np.ceil(N/2))+1)]
    ABx_3 = [s_3[kA_x[i], kB_x[i]] for i in range(2, int(np.ceil(N / 2) + 1))]

    ABp_3_theory = [1 / 2 * (s_2[kA_p[i], kA_p[i]] + s_2[kA_p[i], kB_p[i]] - s_2[kB_p[i], kA_p[i]] - s_2[kB_p[i], kB_p[i]]) for i in range(1,int(np.ceil(N/2))+1)]
    ABp_3 = [s_3[kA_p[i], kB_p[i]] for i in range(1, int(np.ceil(N / 2) + 1))]


    BAx_3_theory = [1 / 2 * (s_2[kA_x[i], kA_x[i]] - s_2[kA_x[i], kB_x[i]] + s_2[kB_x[i], kA_x[i]] - s_2[kB_x[i], kB_x[i]]) for i in range(2,int(np.ceil(N/2))+1)]
    BAx_3 = [s_3[kB_x[i], kA_x[i]] for i in range(1, int(np.ceil(N / 2) + 1))]

    BAp_3_theory = [1 / 2 * (s_2[kA_p[i], kA_p[i]] - s_2[kA_p[i], kB_p[i]] + s_2[kB_p[i], kA_p[i]] - s_2[kB_p[i], kB_p[i]]) for i in range(2,int(np.ceil(N/2))+1)]
    BAp_3 = [s_3[kB_p[i], kA_p[i]] for i in range(1, int(np.ceil(N / 2) + 1))]


    AB1x_3_theory = [1 / 2 * (s_2[kA_x[i], kA_x[i - 1]] + s_2[kA_x[i], kB_x[i - 1]] - s_2[kB_x[i], kA_x[i - 1]] - s_2[kB_x[i], kB_x[i - 1]]) for i in range(2,int(np.ceil(N/2))+1)]
    AB1x_3 = [s_3[kA_x[i], kB_x[i - 1]] for i in range(2, int(np.ceil(N / 2) + 1))]

    B1Ax_3_theory = [1 / 2 * (s_2[kA_x[i - 1], kB_x[i]] - s_2[kA_x[i - 1], kB_x[i]] + s_2[kB_x[i - 1], kA_x[i]] - s_2[kB_x[i], kB_x[i - 1]]) for i in range(2,int(np.ceil(N/2))+1)]
    B1Ax_3 = [s_3[kB_x[i - 1], kA_x[i]] for i in range(2, int(np.ceil(N / 2) + 1))]


    AB1p_3_theory = [1 / 2 * (s_2[kA_p[i], kA_p[i - 1]] + s_2[kA_p[i], kB_p[i - 1]] - s_2[kB_p[i], kA_p[i - 1]] - s_2[kB_p[i], kB_p[i - 1]]) for i in range(2,int(np.ceil(N/2))+1)]
    AB1p_3 = [s_3[kA_p[i], kB_p[i - 1]] for i in range(2, int(np.ceil(N / 2) + 1))]

    B1Ap_3_theory = [1 / 2 * (s_2[kA_p[i - 1], kB_p[i]] - s_2[kA_p[i - 1], kB_p[i]] + s_2[kB_p[i - 1], kA_p[i]] - s_2[kB_p[i], kB_p[i - 1]]) for i in range(2,int(np.ceil(N/2))+1)]
    B1Ap_3 = [s_3[kB_p[i - 1], kA_p[i]] for i in range(2, int(np.ceil(N / 2) + 1))]


    # print("Bp_3_theory", sum([x - y for x, y in zip(Bp_3_theory, Bp_3)][1:]))
    # print("Ap_3_theory", sum([x - y for x, y in zip(Ap_3_theory, Ap_3)][1:]))
    # print("Ax_3_theory", sum([x - y for x, y in zip(Ax_3_theory, Ax_3)][1:]))
    # print("ABx_3_theory", sum([x - y for x, y in zip(ABx_3_theory, ABx_3)][1:]))
    # print("BAx_3_theory", sum([x - y for x, y in zip(BAx_3_theory, BAx_3)][1:]))
    # print("ABp_3_theory", sum([x - y for x, y in zip(ABp_3_theory, ABp_3)][1:]))
    # print("BAp_3_theory", sum([x - y for x, y in zip(BAp_3_theory, BAp_3)][1:]))
    # print("AB1x_3_theory", sum([x - y for x, y in zip(AB1x_3_theory, AB1x_3)][1:]))
    # print("B1Ax_3_theory", sum([x - y for x, y in zip(B1Ax_3_theory, B1Ax_3)][1:]))
    # print("AB1x_3_theory", sum([x - y for x, y in zip(AB1p_3_theory, AB1p_3)][1:]))
    # print("B1Ax_3_theory", sum([x - y for x, y in zip(B1Ap_3_theory, B1Ap_3)][1:]))

    var_q[ii] = gauss.sigma[kA_x[m_A],kA_x[m_A]]+gauss.sigma[kB_x[m_B],kB_x[m_B]] + 2*gauss.sigma[kB_x[m_B],kA_x[m_A]]
    var_p[ii] = gauss.sigma[kA_p[m_A],kA_p[m_A]]+gauss.sigma[kB_p[m_B],kB_p[m_B]] - 2*gauss.sigma[kB_p[m_B],kA_p[m_A]]

    for i in range(1, len(A_modes)):
        # gauss.beam_split(0.5, modes=[n * 2, n * 2 - 1])
        gauss.beam_split(0.5, modes=[A_modes[i], B_modes[i]]) # Right fukawa  for 1D
        # print(A_modes[i], B_modes[i - 1])


    # Stage 5 tests (stage 4 is the delay of modes B)
    Ax_5 = 1/2*(np.array(Ax_3)[4]-np.array(AB1x_3)[4]+np.array(Bx_3)[3]-np.array(B1Ax_3)[4])
    print(Ax_5 - gauss.sigma[kA_x[2],kA_x[2]])
    Ap_5 = 1/2*(np.array(Ap_3)[4]-np.array(AB1p_3)[4]+np.array(Bx_3)[4]-np.array(B1Ap_3)[4])
    print(Ap_5 - gauss.sigma[kA_p[2],kA_p[2]])
    Bx_5 = 1/2*(np.array(Bx_3)[4]-np.array(AB1x_3)[4]+np.array(Bx_3)[4]-np.array(B1Ax_3)[4])
    print(Bx_5 - gauss.sigma[kB_x[4],kB_x[4]])
    Ap_5 = 1/2*(np.array(Ap_3)[4]-np.array(AB1p_3)[4]+np.array(Bx_3)[3]-np.array(B1Ap_3)[4])
    print(Ap_5 - gauss.sigma[kB_p[4],kB_p[4]])
    # Bp_5 = 1/2*(np.array(Ax_3)[4]-np.array(ABx_3)[4]+np.array(Bx_3)[4]-np.array(BAx_3)[4])

    k = 20
    s = gauss.sigma
    # The following nullifier works for Mikkels paper
    # nullifier for q
    a = s[kA_x[k], kA_x[k]] + s[kA_x[k], kB_x[k]] - s[kA_x[k], kA_x[k + 1]] + s[kA_x[k], kB_x[k + 1]]
    b = s[kB_x[k], kB_x[k]] + s[kB_x[k], kA_x[k]] - s[kB_x[k], kA_x[k + 1]] + s[kB_x[k], kB_x[k + 1]]
    c = -s[kA_x[k + 1], kA_x[k]] - s[kA_x[k + 1], kB_x[k]] + s[kA_x[k + 1], kA_x[k + 1]] - s[kA_x[k + 1], kB_x[k + 1]]
    d = s[kB_x[k + 1], kA_x[k]] + s[kB_x[k + 1], kB_x[k]] - s[kB_x[k + 1], kA_x[k + 1]] + s[kB_x[k + 1], kB_x[k + 1]]
    var_q[ii] = a + b + c + d
    # nulifier for p
    a = s[kA_p[k], kA_p[k]] + s[kA_p[k], kB_p[k]] + s[kA_p[k], kA_p[k + 1]] - s[kA_p[k], kB_p[k + 1]]
    b = s[kB_p[k], kB_p[k]] + s[kB_p[k], kA_p[k]] + s[kB_p[k], kA_p[k + 1]] - s[kB_p[k], kB_p[k + 1]]
    c = s[kA_p[k + 1], kA_p[k]] + s[kA_p[k + 1], kB_p[k]] + s[kA_p[k + 1], kA_p[k + 1]] - s[kA_p[k + 1], kB_p[k + 1]]
    d = -s[kB_p[k + 1], kA_p[k]] - s[kB_p[k + 1], kB_p[k]] - s[kB_p[k + 1], kA_p[k + 1]] + s[kB_p[k + 1], kB_p[k + 1]]
    var_p[ii] = a + b + c + d
    # TO do check that var_p works at this stage, figure out why the order seems
    # to be switched compared to Mikkel's paper


    # The following nulifier is based on the nulifiers in Mikkels phd

    # # For the 2D cluster state we set an additional delay of 12
    delay = 12
    n = 0
    while n * 2 + 1 + 2 * (delay + 1) < N:
        # gauss.beam_split(0.5, modes=[n * 2 +2 * (delay + 1), n * 2 + 1])
        gauss.beam_split(0.5, modes=[n * 2 +1 + 2 * (delay + 1), n * 2 + 1], dag=True)
        print(n * 2 + 2 * (delay+ 1), n * 2 + 1)
        # print(int(np.ceil((n * 2 + 1) / 2)) - (n * 2 +2 + 2 * (delay + 1)) / 2)
        n += 1
    s_fin = np.copy(gauss.sigma)

    # # Generating the 2D cluster state:
    # # Delay equals 12
    # for n in np.arange(1, int(N / 2) + 1):
    #     gauss.beam_split(0.5, modes=[n * 2, n * 2 + 1])




    # Checking the nulifiers
    # Mode indicies are made into tables for easy look-up
    # Here indicies for the spatial mode A will match the corresponding index for
    # the spatial mode B when there is no

    # First all of the squared contributions are added
    k = 10
    NN = delay
    # copy over sigma to lighten notation
    sig = gauss.sigma
    i = kA_x[k]
    null_A_k = sig[i, kA_x[k]] + sig[i, kB_x[k]] - sig[i, kA_x[k + 1]] - sig[i, kB_x[k + 1]] - sig[i, kA_x[k + NN]] + sig[i, kB_x[k + NN]] - sig[i, kA_x[k + NN + 1]] + sig[i, kB_x[k + NN + 1]]
    var_a[ii] = null_A_k

    i = kB_x[k]
    null_B_k = sig[i, kA_x[k]] + sig[i, kB_x[k]] - sig[i, kA_x[k + 1]] - sig[i, kB_x[k + 1]] - sig[i, kA_x[k + NN]] + sig[i, kB_x[k + NN]] - sig[i, kA_x[k + NN + 1]] + sig[i, kB_x[k + NN + 1]]
    var_b[ii] = null_B_k

    i = kA_x[k + 1]
    null_A_k1 = -sig[i, kA_x[k]] - sig[i, kB_x[k]] + sig[i, kA_x[k + 1]] + sig[i, kB_x[k + 1]] + sig[i, kA_x[k + NN]] - sig[i, kB_x[k + NN]] + sig[i, kA_x[k + NN + 1]] - sig[i, kB_x[k + NN + 1]]
    # print(abs(null_A_k) + abs(null_B_k) + abs(null_A1_k) + abs(null_AN_k) +
    # abs(null_BN1_k) + abs(null_B1_k) + abs(null_BN_k) + abs(null_AN1_k))
    var_c[ii] = null_A_k1

    i = kB_x[k + 1]
    null_B_k1 = -sig[i, kA_x[k]] - sig[i, kB_x[k]] + sig[i, kA_x[k + 1]] + sig[i, kB_x[k + 1]] + sig[i, kA_x[k + NN]] - sig[i, kB_x[k + NN]] + sig[i, kA_x[k + NN + 1]] - sig[i, kB_x[k + NN + 1]]
    var_d[ii] = null_B_k1

    i = kA_x[k + NN]
    null_A_kN = -sig[i, kA_x[k]] - sig[i, kB_x[k]] + sig[i, kA_x[k + 1]] + sig[i, kB_x[k + 1]] + sig[i, kA_x[k + NN]] - sig[i, kB_x[k + NN]] + sig[i, kA_x[k + NN + 1]] - sig[i, kB_x[k + NN + 1]]
    var_e[ii] = null_A_kN

    i = kB_x[k + NN]
    null_B_kN = sig[i, kA_x[k]] + sig[i, kB_x[k]] - sig[i, kA_x[k + 1]] - sig[i, kB_x[k + 1]] - sig[i, kA_x[k + NN]] + sig[i, kB_x[k + NN]] - sig[i, kA_x[k + NN + 1]] + sig[i, kB_x[k + NN + 1]]
    var_f[ii] = null_B_kN

    i = kA_x[k + NN + 1]
    null_A_kN1 = -sig[i, kA_x[k]] - sig[i, kB_x[k]] + sig[i, kA_x[k + 1]] + sig[i, kB_x[k + 1]] + sig[i, kA_x[k + NN]] - sig[i, kB_x[k + NN]] + sig[i, kA_x[k + NN + 1]] - sig[i, kB_x[k + NN + 1]]
    var_g[ii] = null_A_kN1

    i = kB_x[k + NN + 1]
    null_B_kN1 = sig[i, kA_x[k]] + sig[i, kB_x[k]] - sig[i, kA_x[k + 1]] - sig[i, kB_x[k + 1]] - sig[i, kA_x[k + NN]] + sig[i, kB_x[k + NN]] - sig[i, kA_x[k + NN + 1]] + sig[i, kB_x[k + NN + 1]]
    var_h[ii] = null_B_kN1

    var_q1[ii] = null_A_k + null_B_k + null_A_k1 + null_B_k1 + null_A_kN + null_B_kN + null_A_kN1 + null_B_kN1



    i = kA_p[k]
    null_A_k = sig[i, kA_p[k]] + sig[i, kB_p[k]] - sig[i, kA_p[k + 1]] - sig[i, kB_p[k + 1]] - sig[i, kA_p[k + NN]] + sig[i, kB_p[k + NN]] - sig[i, kA_p[k + NN + 1]] + sig[i, kB_p[k + NN + 1]]

    i = kB_p[k]
    null_B_k = sig[i, kA_p[k]] + sig[i, kB_p[k]] - sig[i, kA_p[k + 1]] - sig[i, kB_p[k + 1]] - sig[i, kA_p[k + NN]] + sig[i, kB_p[k + NN]] - sig[i, kA_p[k + NN + 1]] + sig[i, kB_p[k + NN + 1]]

    i = kA_p[k + 1]
    null_A_k1 = -sig[i, kA_p[k]] - sig[i, kB_p[k]] + sig[i, kA_p[k + 1]] + sig[i, kB_p[k + 1]] + sig[i, kA_p[k + NN]] - sig[i, kB_p[k + NN]] + sig[i, kA_p[k + NN + 1]] - sig[i, kB_p[k + NN + 1]]
    # print(abs(null_A_k) + abs(null_B_k) + abs(null_A1_k) + abs(null_AN_k) + abs(null_BN1_k) + abs(null_B1_k) + abs(null_BN_k) + abs(null_AN1_k))
    i = kB_p[k + 1]
    null_B_k1 = -sig[i, kA_p[k]] - sig[i, kB_p[k]] + sig[i, kA_p[k + 1]] + sig[i, kB_p[k + 1]] + sig[i, kA_p[k + NN]] - sig[i, kB_p[k + NN]] + sig[i, kA_p[k + NN + 1]] - sig[i, kB_p[k + NN + 1]]

    i = kA_p[k + NN]
    null_A_kN = -sig[i, kA_p[k]] - sig[i, kB_p[k]] + sig[i, kA_p[k + 1]] + sig[i, kB_p[k + 1]] + sig[i, kA_p[k + NN]] - sig[i, kB_p[k + NN]] + sig[i, kA_p[k + NN + 1]] - sig[i, kB_p[k + NN + 1]]

    i = kB_p[k + NN]
    null_B_kN = sig[i, kA_p[k]] + sig[i, kB_p[k]] - sig[i, kA_p[k + 1]] - sig[i, kB_p[k + 1]] - sig[i, kA_p[k + NN]] + sig[i, kB_p[k + NN]] - sig[i, kA_p[k + NN + 1]] + sig[i, kB_p[k + NN + 1]]

    i = kA_p[k + NN + 1]
    null_A_kN1 = -sig[i, kA_p[k]] - sig[i, kB_p[k]] + sig[i, kA_p[k + 1]] + sig[i, kB_p[k + 1]] + sig[i, kA_p[k + NN]] - sig[i, kB_p[k + NN]] + sig[i, kA_p[k + NN + 1]] - sig[i, kB_p[k + NN + 1]]

    i = kB_p[k + NN + 1]
    null_B_kN1 = sig[i, kA_p[k]] + sig[i, kB_p[k]] - sig[i, kA_p[k + 1]] - sig[i, kB_p[k + 1]] - sig[i, kA_p[k + NN]] + sig[i, kB_p[k + NN]] - sig[i, kA_p[k + NN + 1]] + sig[i, kB_p[k + NN + 1]]

    # var_q1[ii] = null_A_k + null_B_k + null_A_k1 + null_B_k1 + null_A_kN +
    # null_B_kN + null_A_kN1 + null_B_kN1
    print(ii)




import matplotlib.pyplot as plt
plt.plot(squeezing, var_q1, 'x', label='Generated by Gaussian class')
plt.plot(squeezing, 4 * np.exp(-2* squeezing), label='Expected')
#modes_first = [[i * 2- 2, i * 2 - 1] for i in modes]]
plt.legend()
plt.title('var($\hat{n}^q$) for an EPR state')
plt.xlabel('Squeezing (r)')
plt.ylabel('var($\hat{n}^q$)')
plt.show()

import matplotlib.pyplot as plt
plt.plot(var_a, label='null_A_k')
plt.plot(var_b, label='null_B_k')
plt.plot(var_c, label='null_A_k1')
plt.plot(var_d, label='null_B_k1')
plt.plot(var_e, label='null_A_kN')
plt.plot(var_f, label='null_B_kN')
plt.plot(var_g, label='null_A_kN1')
plt.plot(var_h, label='null_B_kN1')
plt.legend()
plt.show()
