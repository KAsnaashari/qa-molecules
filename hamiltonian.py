class Hamiltonian:
    import numpy as np

    def __init__(self, mol_params, index=(1, 2), tol=0) -> None:
        self.b_e = mol_params['b_e']
        self.d = mol_params['d']
        self.gamma = mol_params['gamma']
        # self.delta_alpha = mol_params['delta_alpha']
        self.alpha_0 = 1 / 3 * (mol_params['alpha_parallel'] + 2 * mol_params['alpha_perp'])
        self.alpha_2 = 2 / 3 * (mol_params['alpha_parallel'] - mol_params['alpha_perp'])
        self.index = index
        self.tol = tol

    def calc_isolated_ham(self, E, B, xi, n):
        import numpy as np
        from utils import mu_b, g_s, pauli_Z, zero_tol, calc_3j
        from utils import epsilon0, c0

        num_states = 2 * n**2
        beta = mu_b * g_s * B
        if np.all(np.equal(np.array(xi), np.zeros(3))):
            xi = None
        if xi is not None:
            xi = np.array(xi)
            I = np.dot(xi, xi) * c0 * epsilon0 / 2
            theta_I = np.arccos(xi[2] / np.linalg.norm(xi))
            d2s = np.zeros(5)
            d2s[0] = d2s[4] = 1 / 4 * np.sqrt(6) * np.square(np.sin(theta_I))
            d2s[1] = 1 / 4 * np.sqrt(6) * np.sin(2 * theta_I)
            d2s[3] = -d2s[1]
            d2s[2] = 1 / 2 * (3 * np.square(np.cos(theta_I)) - 1)

        ham = np.zeros((num_states, num_states))
        for i in range(num_states):
            for j in range(i, num_states):
                n1 = np.floor(np.sqrt(i / 2))
                m1 = i // 2 - (n1 + 1) * n1
                s1 = i % 2 - 1/2
            
                n2 = np.floor(np.sqrt(j / 2))
                m2 = j // 2 - (n2 + 1) * n2
                s2 = j % 2 - 1/2

                if (n1 == n2) and (m1 == m2) and (s1 == s2):
                    ham[i, j] += self.b_e * (n1 + 1) * n1 + beta * s1 + self.gamma * m1 * s1
                if (n1 == n2) and (((m1 == m2 + 1) and (s1 == s2 - 1)) or ((m1 == m2 - 1) and (s1 == s2 + 1))):
                    ham[i, j] += 1/2 * self.gamma * np.sqrt(n1 * (n1 + 1) - m1 * m2)
                if (s1 == s2) and (m1 == m2):
                    if ((n1 == n2 + 1) or (n1 == n2 - 1)):
                        j1 = calc_3j(np.array([[n1, 1, n2], [-m1, 0, m2]]))
                        j2 = calc_3j(np.array([[n1, 1, n2], [0, 0, 0]]))
                        ham[i, j] += -E * self.d * (-1)**m1 * np.sqrt((2 * n1 + 1) * (2 * n2 + 1)) * j1 * j2
                if (xi is not None) and (s1 == s2):
                    if ((n1 == n2 + 2) or (n1 == n2 - 2)) and (np.abs(m1 - m2) <= 2):
                        k = int(m1 - m2)
                        j1 = calc_3j(np.array([[n1, 2, n2], [-m1, k, m2]]))
                        j2 = calc_3j(np.array([[n1, 2, n2], [0, 0, 0]]))
                        ham[i, j] += (-1)**m2 * I * self.alpha_2 / (2 * epsilon0 * c0) \
                                    * np.sqrt((2 * n1 + 1) * (2 * n2 + 1)) * j2 * d2s[k + 2] * j1
                    elif (n1 > 0) and (n1 == n2):
                        j1 = calc_3j(np.array([[n1, 2, n2], [-m1, 0, m2]]))
                        j2 = calc_3j(np.array([[n1, 2, n2], [0, 0, 0]]))
                        ham[i, j] += (-1)**m2 * I * self.alpha_2 / (2 * epsilon0 * c0) \
                                    * np.sqrt((2 * n1 + 1) * (2 * n2 + 1)) * j2 * d2s[2] * j1
        ham += ham.T - np.diag(np.diag(ham))
        return ham

    def calc_ab(self, E, B, xi=None, n=3, all=False):
        import numpy as np
        from utils import mu_b, g_s, pauli_Z, zero_tol

        # hamiltonian = rot + fs + elec + mag
        hamiltonian = self.calc_isolated_ham(E, B, xi, n)
        # print(hamiltonian)

        eig_vals, eig_vecs = np.linalg.eigh(hamiltonian)
        alpha = eig_vecs[:, self.index[0]]
        beta = eig_vecs[:, self.index[1]]

        # if abs(alpha).max() == -alpha.min():
        #     alpha = -alpha
        # if abs(beta).max() != beta.max():
        #     beta = -beta

        if alpha[1] < 0:
            alpha = -alpha
        if beta[1] < 0:
            beta = -beta

        alpha = zero_tol(alpha, self.tol)
        # alpha.real[abs(alpha.real) < tol] = 0.0
        # alpha.imag[abs(alpha.imag) < tol] = 0.0
        
        beta = zero_tol(beta, self.tol)
        # beta.real[abs(beta.real) < tol] = 0.0
        # beta.imag[abs(beta.imag) < tol] = 0.0
        if all:
            return eig_vals, eig_vecs

        return np.array((eig_vals[self.index[0]], eig_vals[self.index[1]])), np.array((alpha, beta))

    def calc_dipoles(self, ab):
        import numpy as np
        from utils import calc_3j, zero_tol

        alpha, beta = ab
        num_states = alpha.shape[0]

        d_0 = np.zeros((2, 2), dtype=complex)
        d_1 = np.zeros((2, 2), dtype=complex)
        d__1 = np.zeros((2, 2), dtype=complex)
        for i in range(num_states):
            for j in range(num_states):
                n1 = np.floor(np.sqrt(i / 2))
                m1 = i // 2 - (n1 + 1) * n1
                s1 = i % 2 - 1/2
            
                n2 = np.floor(np.sqrt(j / 2))
                m2 = j // 2 - (n2 + 1) * n2
                s2 = j % 2 - 1/2

                if (s1 == s2) and (np.abs(n1 - n2) == 1) and (np.abs(m1 - m2) <= 1):
                    j1 = calc_3j(np.array([[n1, 1, n2], [-m1, m1 - m2, m2]]))
                    j2 = calc_3j(np.array([[n1, 1, n2], [0, 0, 0]]))
                    d = self.d * (-1)**m1 * np.sqrt((2 * n1 + 1) * (2 * n2 + 1)) * j1 * j2
                    
                    if (np.abs(m1 - m2) == 0):
                        d_0[0, 0] += d * alpha[i] * alpha[j]
                        d_0[0, 1] += d * alpha[i] * beta[j]
                        d_0[1, 0] += d * beta[i] * alpha[j]
                        d_0[1, 1] += d * beta[i] * beta[j]
                    elif (m1 - m2 == 1):
                        d_1[0, 0] += d * alpha[i] * alpha[j]
                        d_1[0, 1] += d * alpha[i] * beta[j]
                        d_1[1, 0] += d * beta[i] * alpha[j]
                        d_1[1, 1] += d * beta[i] * beta[j]
                    elif (m1 - m2 == -1):
                        d__1[0, 0] += d * alpha[i] * alpha[j]
                        d__1[0, 1] += d * alpha[i] * beta[j]
                        d__1[1, 0] += d * beta[i] * alpha[j]
                        d__1[1, 1] += d * beta[i] * beta[j]

        d_0 = zero_tol(d_0, self.tol)
        d_1 = zero_tol(d_1, self.tol)
        d__1 = zero_tol(d__1, self.tol)

        return d_0, d_1, d__1

    def calc_int_hamiltonian(self, E, B, r, Xi=None, theta=np.pi/2, n=3):
        import numpy as np
        from utils import K

        E1, E2 = E
        B1, B2 = B
        xi1, xi2 = Xi

        (E_a1, E_b1), (alpha1, beta1) = self.calc_ab(E1, B1, xi1, n=n)
        (E_a2, E_b2), (alpha2, beta2) = self.calc_ab(E2, B2, xi2, n=n)
        ab1 = (alpha1, beta1)
        ab2 = (alpha2, beta2)
        
        d_01, d_11, d__11 = self.calc_dipoles(ab1)
        d_02, d_12, d__12 = self.calc_dipoles(ab2)

        ham = np.zeros((4, 4), dtype=complex)

        ham[0, 0] = -3/2 * np.sin(theta)**2 * (d__11[0, 0] * d__12[0, 0] + d_11[0, 0] * d_12[0, 0]) + \
                    3 / np.sqrt(2) * np.sin(theta) * np.cos(theta) * (d_11[0, 0] * d_02[0, 0] + d_01[0, 0] * d_12[0, 0] - d__11[0, 0] * d_02[0, 0] - d_01[0, 0] * d__12[0, 0]) + \
                    1/2 * (1 - 3 * np.cos(theta)**2) * (d_11[0, 0] * d__12[0, 0] + 2 * d_01[0, 0] * d_02[0, 0] + d__11[0, 0] * d_12[0, 0])
        # ham[0, 0] = d_01[0, 0] * d_02[0, 0] - 3/2 * (d__11[0, 0] * d__12[0, 0] + d_11[0, 0] * d_12[0, 0]) + 1/2 * (d__11[0, 0] * d_12[0, 0] + d_11[0, 0] * d__12[0, 0])
        # ham[0, 1] = d_01[0, 0] * d_02[0, 1] - 3/2 * (d__11[0, 0] * d__12[0, 1] + d_11[0, 0] * d_12[0, 1]) + 1/2 * (d__11[0, 0] * d_12[0, 1] + d_11[0, 0] * d__12[0, 1])
        # ham[0, 2] = d_01[0, 1] * d_02[0, 0] - 3/2 * (d__11[0, 1] * d__12[0, 0] + d_11[0, 1] * d_12[0, 0]) + 1/2 * (d__11[0, 1] * d_12[0, 0] + d_11[0, 1] * d__12[0, 0])
        # ham[0, 3] = d_01[0, 1] * d_02[0, 1] - 3/2 * (d__11[0, 1] * d__12[0, 1] + d_11[0, 1] * d_12[0, 1]) + 1/2 * (d__11[0, 1] * d_12[0, 1] + d_11[0, 1] * d__12[0, 1])
        
        
        ham[1, 1] = -3/2 * np.sin(theta)**2 * (d__11[0, 0] * d__12[1, 1] + d_11[0, 0] * d_12[1, 1]) + \
                    3 / np.sqrt(2) * np.sin(theta) * np.cos(theta) * (d_11[0, 0] * d_02[1, 1] + d_01[0, 0] * d_12[1, 1] - d__11[0, 0] * d_02[1, 1] - d_01[0, 0] * d__12[1, 1]) + \
                    1/2 * (1 - 3 * np.cos(theta)**2) * (d_11[0, 0] * d__12[1, 1] + 2 * d_01[0, 0] * d_02[1, 1] + d__11[0, 0] * d_12[1, 1])
        # ham[1, 1] = d_01[0, 0] * d_02[1, 1] - 3/2 * (d__11[0, 0] * d__12[1, 1] + d_11[0, 0] * d_12[1, 1]) + 1/2 * (d__11[0, 0] * d_12[1, 1] + d_11[0, 0] * d__12[1, 1])
        ham[1, 2] = -3/2 * np.sin(theta)**2 * (d__11[0, 1] * d__12[1, 0] + d_11[0, 1] * d_12[1, 0]) + \
                    3 / np.sqrt(2) * np.sin(theta) * np.cos(theta) * (d_11[0, 1] * d_02[1, 0] + d_01[0, 1] * d_12[1, 0] - d__11[0, 1] * d_02[1, 0] - d_01[0, 1] * d__12[1, 0]) + \
                    1/2 * (1 - 3 * np.cos(theta)**2) * (d_11[0, 1] * d__12[1, 0] + 2 * d_01[0, 1] * d_02[1, 0] + d__11[0, 1] * d_12[1, 0])
        # ham[1, 2] = d_01[0, 1] * d_02[1, 0] - 3/2 * (d__11[0, 1] * d__12[1, 0] + d_11[0, 1] * d_12[1, 0]) + 1/2 * (d__11[0, 1] * d_12[1, 0] + d_11[0, 1] * d__12[1, 0])
        # ham[1, 3] = d_01[0, 1] * d_02[1, 1] - 3/2 * (d__11[0, 1] * d__12[1, 1] + d_11[0, 1] * d_12[1, 1]) + 1/2 * (d__11[0, 1] * d_12[1, 1] + d_11[0, 1] * d__12[1, 1])

        ham[2, 2] = -3/2 * np.sin(theta)**2 * (d__11[1, 1] * d__12[0, 0] + d_11[1, 1] * d_12[0, 0]) + \
                    3 / np.sqrt(2) * np.sin(theta) * np.cos(theta) * (d_11[1, 1] * d_02[0, 0] + d_01[1, 1] * d_12[0, 0] - d__11[1, 1] * d_02[0, 0] - d_01[1, 1] * d__12[0, 0]) + \
                    1/2 * (1 - 3 * np.cos(theta)**2) * (d_11[1, 1] * d__12[0, 0] + 2 * d_01[1, 1] * d_02[0, 0] + d__11[1, 1] * d_12[0, 0])
        # ham[2, 2] = d_01[1, 1] * d_02[0, 0] - 3/2 * (d__11[1, 1] * d__12[0, 0] + d_11[1, 1] * d_12[0, 0]) + 1/2 * (d__11[1, 1] * d_12[0, 0] + d_11[1, 1] * d__12[0, 0])
        # ham[2, 3] = d_01[1, 1] * d_02[0, 1] - 3/2 * (d__11[1, 1] * d__12[0, 1] + d_11[1, 1] * d_12[0, 1]) + 1/2 * (d__11[1, 1] * d_12[0, 1] + d_11[1, 1] * d__12[0, 1])
        
        ham[3, 3] = -3/2 * np.sin(theta)**2 * (d__11[1, 1] * d__12[1, 1] + d_11[1, 1] * d_12[1, 1]) + \
                    3 / np.sqrt(2) * np.sin(theta) * np.cos(theta) * (d_11[1, 1] * d_02[1, 1] + d_01[1, 1] * d_12[1, 1] - d__11[1, 1] * d_02[1, 1] - d_01[1, 1] * d__12[1, 1]) + \
                    1/2 * (1 - 3 * np.cos(theta)**2) * (d_11[1, 1] * d__12[1, 1] + 2 * d_01[1, 1] * d_02[1, 1] + d__11[1, 1] * d_12[1, 1])
        # ham[3, 3] = d_01[1, 1] * d_02[1, 1] - 3/2 * (d__11[1, 1] * d__12[1, 1] + d_11[1, 1] * d_12[1, 1]) + 1/2 * (d__11[1, 1] * d_12[1, 1] + d_11[1, 1] * d__12[1, 1])

        ham[1, 0] = ham[0, 1].conjugate()
        ham[2, 0] = ham[0, 2].conjugate()
        ham[2, 1] = ham[1, 2].conjugate()
        ham[3, 0] = ham[0, 3].conjugate()
        ham[3, 1] = ham[1, 3].conjugate()
        ham[3, 2] = ham[2, 3].conjugate()

        ham *= K / r**3

        return ham

    def calc_couplings_biases(self, E, B, r, Xi=(None, None), theta=np.pi/2, n=3):
        import numpy as np

        ham = self.calc_int_hamiltonian(E, B, r, Xi=Xi, theta=theta, n=n)
        # print(ham)
        j_perp = 2 * ham[1, 2]
        j_z = ham[0, 0] + ham[3, 3] - (ham[2, 2] + ham[1, 1])
        w = 0.5 * (ham[0, 0] + ham[2, 2] - (ham[3, 3] + ham[1, 1]))
        k = 0.5 * (ham[0, 0] + ham[1, 1] - (ham[3, 3] + ham[2, 2]))
        v = 0.25 * (ham[0, 0] + ham[1, 1] + ham[2, 2] + ham[3, 3])

        # (E_a1, E_b1), _ = calc_ab(E[0], B[0], b_e, d, gamma, index=(1, 2))
        # (E_a2, E_b2), _ = calc_ab(E[1], B[1], b_e, d, gamma, index=(1, 2))

        h1 = w
        h2 = k

        return np.array([j_perp, j_z, h1, h2]).real

    def calc_E_perp(self, b, xi=None, perp_range=(0.5e5, 10e5)):
        from scipy.optimize import brute, fmin
        from utils import j_to_mhz

        def energy_diff(e, b):
            (e_a, e_b), (a, b) = self.calc_ab(e, b, xi=xi)
            return e_b - e_a

        opt = brute(lambda e: energy_diff(e, b) * j_to_mhz, ranges=(perp_range,), Ns=1000, full_output=True, finish=fmin)
        # print(opt[3])
        return opt[0][0]

    def calc_B_perp(self, e, xi=None):
        from scipy.optimize import brute, fmin
        from utils import j_to_mhz

        def energy_diff(e, b):
            (e_a, e_b), (a, b) = self.calc_ab(e, b, xi=xi)
            return e_b - e_a

        opt = brute(lambda b: energy_diff(e, b) * j_to_mhz, ranges=((0.53, 0.7),), Ns=1000, full_output=True, finish=fmin)
        # print(opt[3])
        return opt[0][0]

    def calc_E_z(self, b, xi=None, theta=np.pi/2, threshold=100, r=1e-9, e_perp=None, perp_range=((0.5e5, 10e5))):
        from scipy.optimize import brute, fmin
        import numpy as np
        from utils import j_to_hz

        if e_perp is None:
            e_perp = self.calc_E_perp(b, xi, perp_range=perp_range)

        def cost_fn(e):
            j_perp, j_z, _, _ = self.calc_couplings_biases((e, e), (b, b), r, Xi=(xi, xi), theta=theta) * j_to_hz
            j_perp, j_z = np.abs((j_perp, j_z))

            return (j_z - j_perp * threshold)**2
            
        opt = brute(cost_fn, ranges=((e_perp, 1.5 * e_perp),), Ns=1000, full_output=True, finish=fmin)
        return opt[0][0]

    def calc_E_perp_z(self, b, xi=None, theta=np.pi/2, threshold=100, perp_range=((0.5e5, 10e5))):
        import numpy as np
        e_perp = self.calc_E_perp(b, xi=xi, perp_range=perp_range)
        e_z = self.calc_E_z(b, xi=xi, theta=theta, threshold=threshold, e_perp=e_perp)
        return np.array([e_perp, e_z])

# from utils import srf_params
# h = Hamiltonian(srf_params)
# print(h.calc_isolated_ham(1e5, 0.54, [0, 0, 1e8], 4))