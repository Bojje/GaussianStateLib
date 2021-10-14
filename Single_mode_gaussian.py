import numpy as np
# Number of modes are set:

# def add_mode(self, n=1):
class GaussianState:
    """ Class that defines a single mode Gaussian state based on a vectors of
    means and a covariance matrix. It will initialize a vacuum state in the
    first version of the class and different functions will implement the main
    operations (displacement, squeezing and rotation)
    """
    def __init__(self, modes=1):
        """ Initialize a vacuum state. The specification of modes has not been
        implemented to support modes > 1"""
        # mu is a vectors containing the expectations values of the momemntum
        # and position operators
        self.mu = np.zeros(2 * modes)
        # self.mu.shape = (len(mu), 1)
        # Sigma is the covariance matrix
        self.sigma = np.identity(2 * modes) * 0.5
        self.modes = modes
    def displacement(self, alpha):
        """ The displacement operator displaces the state. We have that
        D(alpha)Sigma D_dag(sigma)=Sigma and q -> q+sqrt(2)Re(alpha)
        p -> p+sqrt(2)Im(alpha)
        """
        import numpy as np
        # The displacement 'matrix is simply the identity and will not be
        # included
        # The vector to update the means follows:
        sq2_real_alpha = 1 / np.sqrt(2) * (alpha + np.conj(alpha))
        sq2_im_alpha = 1 / (np.sqrt(2) * 1j) * (alpha - np.conj(alpha))
        self.mu += np.abs(np.array([sq2_real_alpha, sq2_im_alpha]))
    def plot_wigner_func(self, start= -4, stop=4, N=100):
        # Create a vector representing r
        q = np.linspace(start, stop, N)
        # r = np.array([y for x in q for y in (x,)*2])
        # r = np.reshape(r,(N, 2)).T
        norm = 1 / ((2 * np.pi) ** 1 * np.sqrt(np.linalg.det(self.sigma)))
        inv_sig = np.linalg.inv(self.sigma)
        # r_mu = r - self.mu
        # r_mu_T = r_mu.transpose()
        wigner = np.zeros((N, N))
        for ri, r in enumerate(q):
            for rri, rr in enumerate(q):
                r_mu = np.array([r, rr]) - self.mu
                r_mu.shape = (2, 1)
                wigner[ri, rri] = norm * np.exp(-0.5 *r_mu.T @ inv_sig @ r_mu)

        # wigner = norm * np.exp(-0.5 *r_mu.T @ inv_sig @ r_mu)
        # The code below can be uncommented to get the numerical integral of the
        # wigner function for the interval q,p \in \[start,stop]
        np.sum(wigner * (q[1]-q[0])**2)
        import matplotlib.pyplot as plt
        qq, pp = np.meshgrid(q,q)
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(qq, pp, wigner)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Wigner Function')
        ax.set_xlabel('q')
        ax.set_xlabel('p')
        plt.show()
        return wigner

    def single_mode_squeeze(self, r, theta):
        """ Single mode squeezing, implemented using eq 50-52 in 2102.05748"""
        S_theta = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]])
        F = np.cosh(r) * np.identity(2) - np.sinh(r) * S_theta
        F_dag = F.conj().T
        self.sigma = F_dag@self.sigma@F


    def phase_shift(self, phi=np.pi, mode=1):
        """ Implements a phase shift of phi radians. Squeezing should be applied
        to generate a state which is not invariant to rotation"""
        R_phi = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        R_phi_dag = R_phi.conj().T
        self.sigma = R_phi_dag@self.sigma@R_phi
gauss = GaussianState()
#gauss.displacement(0.2 + 0.1j)
#gauss.phase_shift(np.pi)
gauss.single_mode_squeeze(1, 0)
wigner = gauss.plot_wigner_func(start= -5, stop=5, N=100)
