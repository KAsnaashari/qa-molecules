class Lattice:
    def __init__(self, coords, qubits, hamiltonian, biasers=None) -> None:
        '''
            Prepares the lattice given the 3d coordinates of each molecule, pairs of molecules considered as qubits 
            and the magnetic field as a function of space.

            coords: N*3 matrix of 3D coordinates for N molecules
            qubits: M*2 matrix indicating the molecules grouped together as qubits
                    Each row represents a qubit and contains the indices of the two molecules in the coords matrix.
                    (Order of the indices determines the |0> state of the qubit. |0> = first index up and second index down)
            hamiltonian: a Hamiltonian object initialized with the relevant molecule parameters
        '''
        import numpy as np
        self.coords = coords
        self.num_mols = coords.shape[0]
        self.mol_tril_indices = np.triu_indices(self.num_mols, 1)
        self.qubits = qubits
        if qubits is None:
            self.num_qubits = self.num_mols
            self.qubit_tril_indices = self.mol_tril_indices
        else:
            self.num_qubits = qubits.shape[0]
            self.qubit_tril_indices = np.triu_indices(self.num_qubits, 1)
        self.hamiltonian = hamiltonian
        self.biasers = biasers
        self.edges = self.build_edges()
        self.num_edges = (self.num_mols**2 - self.num_mols) // 2

    def build_edges(self):
        '''
            Builds two N(N-1)/2 arrays of intermolecular distances and angles between the intermolecular vectors and 
            the magnetic field

            returns:
                edges: a tuple of distances and thetas (two N(N-1)/2 arrays)
        '''
        import numpy as np

        diffs = self.coords[:, None, :] - self.coords[None, :, :]
        diffs = diffs[self.mol_tril_indices]
        dists = np.linalg.norm(diffs, axis=1)

        unit_diffs = diffs / np.linalg.norm(diffs, axis=1)[:, None]
        thetas = np.arccos(np.einsum('ik,k->i', unit_diffs, np.array([0, 0, 1])))
        # print(dists, thetas)
        return (dists, thetas)

    def mol_edge_indices(self):
        import numpy as np
        return np.dstack(self.mol_tril_indices)

    def qubit_edge_indices(self):
        import numpy as np
        return np.dstack(self.qubit_tril_indices)

    def calc_couplings_biases_lattice(self, E, B, Xi):
        import numpy as np
        from utils import j_to_hz

        biases = np.zeros(self.num_mols)
        couplings = np.zeros((self.num_edges, 2))
        for i in range(self.num_edges):
            r = self.edges[0][i]
            theta = self.edges[1][i]
            ind_a = self.mol_tril_indices[0][i]
            ind_b = self.mol_tril_indices[1][i]
            r_a = self.coords[ind_a]
            r_b = self.coords[ind_b]

            c = self.hamiltonian.calc_couplings_biases((E(r_a), E(r_b)), (B(r_a), B(r_b)), r, 
                                                        Xi=(Xi(r_a), Xi(r_b)), theta=theta) * j_to_hz
            couplings[i] = c[:2]
            biases[ind_a] += c[2]
            biases[ind_b] += c[3]

        for i in range(self.num_mols):
            (E_a1, E_b1), _ = self.hamiltonian.calc_ab(E(self.coords[i]), B(self.coords[i]))
            biases[i] += (E_b1 - E_a1) * j_to_hz

        return couplings, biases

    def calc_couplings_qubits_2q(self, E, B, Xi, normalize=False):
        import numpy as np
        couplings, biases = self.calc_couplings_biases_lattice(E, B, Xi)
        hs = np.zeros(self.num_qubits)
        deltas = np.zeros(self.num_qubits)
        
        for i in range(self.num_qubits):
            q1, q2 = self.qubits[i]
            hs[i] = biases[q1] - biases[q2]

        qubit_couplings = np.zeros((self.num_qubits, self.num_qubits, 2))

        for k in range(self.num_edges):
            i, j = self.mol_tril_indices[0][k], self.mol_tril_indices[1][k]
            # print(i, j)
            # print(np.where(self.qubits == i)[0])
            if np.where(self.qubits == i)[0] == np.where(self.qubits == j)[0]:
                q = np.where(self.qubits == i)[0]
                deltas[q] = couplings[k, 0]
                continue

            if self.biasers is not None:
                if i in self.biasers[:, 0]:
                    b = np.where(self.biasers[:, 0] == i)[0]
                    q = np.where(self.qubits == j)[0]
                    sign = self.biasers[b, 1]
                    hs[q] += sign * np.abs(couplings[k, 1])
                    continue
                elif j in self.biasers[:, 0]:
                    b = np.where(self.biasers[:, 0] == j)[0]
                    q = np.where(self.qubits == i)[0]
                    sign = self.biasers[b, 1]
                    hs[q] += sign * np.abs(couplings[k, 1])
                    continue
            
            q1, i1 = np.where(self.qubits == i)
            q2, i2 = np.where(self.qubits == j)
            # print(q1, q2)
            if i1 == i2:
                sign = 1
            else:
                sign = -1

            qubit_couplings[q1, q2, 0] += sign * couplings[k, 0] / 2
            qubit_couplings[q1, q2, 1] += sign * couplings[k, 1]
            qubit_couplings[q2, q1, 0] += sign * couplings[k, 0] / 2
            qubit_couplings[q2, q1, 1] += sign * couplings[k, 1]
            # print(qubit_couplings)

        j_perp = qubit_couplings[self.qubit_tril_indices][:, 0]
        j_z = qubit_couplings[self.qubit_tril_indices][:, 1]

        if normalize:
            j = np.max(np.abs(np.concatenate([hs, deltas, j_z, j_perp])))
            hs, deltas, j_z, j_perp = hs / j, deltas / j, j_z / j, j_perp / j

        return hs, deltas, j_z, j_perp

    def calc_couplings_qubits_1q(self, E, B, Xi, MW):
        import numpy as np
        from utils import j_to_hz
        couplings, biases = self.calc_couplings_biases_lattice(E, B, Xi)
        qubit_couplings = np.zeros((self.num_qubits, self.num_qubits, 2))
        hs = np.zeros(self.num_qubits)
        deltas = np.zeros(self.num_qubits)

        for i in range(self.num_qubits):
            w_mw, e_mw = MW(self.coords[i])
            w0 = biases[i]
            # print(w0)
            hs[i] = - (w_mw - w0)
            
            _, ab = self.hamiltonian.calc_ab(E(self.coords[i]), B(self.coords[i]), Xi(self.coords[i]), n=3)
            d_0, d_1, d__1 = self.hamiltonian.calc_dipoles(ab)
            # print(d_0)
            deltas[i] = - d_0[0, 1] * e_mw * j_to_hz

        for k in range(self.num_edges):
            i, j = self.mol_tril_indices[0][k], self.mol_tril_indices[1][k]

            qubit_couplings[i, j, 0] += couplings[k, 0] / 2
            qubit_couplings[i, j, 1] += couplings[k, 1]

        j_perp = qubit_couplings[self.qubit_tril_indices][:, 0]
        j_z = qubit_couplings[self.qubit_tril_indices][:, 1]

        return hs, deltas, j_z, j_perp

# from hamiltonian import Hamiltonian
# import numpy as np
# from utils import srf_params, check_stoq_2q

# h = Hamiltonian(srf_params)
# coords = np.array([[0, 0, 0], [0, 400e-9, 0]])
# l = Lattice(coords, None, h)

# def E(coords):
#     return 1.18e5
# def B(coords):
#     return 0.538
# def Xi(coords):
#     return None
# def MW(coords):
#     return 8336400.374724318, 1

# hs, deltas, j_z, j_perp = l.calc_couplings_qubits_1q(E, B, Xi, MW)
# print(hs, deltas, j_z, j_perp)
# print(check_stoq_2q(hs, deltas, j_z, j_perp))