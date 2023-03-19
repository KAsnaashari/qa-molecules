import numpy as np
import qutip as qt
from scipy.sparse import coo
from utils import srf_params, mols_to_qubits, qubits_to_mols
from hamiltonian import Hamiltonian
from lattice import Lattice

coords = np.array([[0, 0, 0], [0, 0, 500e-9], [500e-9, 0, 0], [500e-9, 0, 500e-9], [0, 500e-9, 0], [500e-9, 500e-9, 0], 
                    [0, 500e-9, 500e-9], [500e-9, 500e-9, 500e-9]])
# coords = np.array([[0, 0, 0], [500e-9, 0, 0], [0, 0, 1000e-9], [500e-9, 0, 1000e-9], [0, 1000e-9, 0], [500e-9, 1000e-9, 0], ])
qubits = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])

valid_states_q = np.array([np.binary_repr(i, width=qubits.shape[0]) for i in range(2**qubits.shape[0])])
valid_states = np.array(list(map(lambda s: qubits_to_mols(s, qubits), valid_states_q)))
# solutions_q = np.array(['010', '101'])
# solutions = np.array(list(map(lambda s: qubits_to_mols(s, qubits), solutions_q)))
print(coords)
print(qubits)
print(valid_states)
# print(solutions)

h = Hamiltonian(srf_params)
l = Lattice(coords, qubits, h, biasers=None)
num_mols = l.num_mols
num_edges = l.num_edges

# print(l.qubit_edge_indices())
def Xi(coords, s):
    if (coords[0] == 0) and (coords[1] == 0):
        return np.array([0, 0, 1e6]) * (1 - s)
    if (coords[0] == 500e-9) and (coords[1] == 0):
        return np.array([0, 0, 3e6]) * (1 - s)
    if (coords[1] == 500e-9) and (coords[2] == 0):
        return np.array([0, 0, 6e6]) * (1 - s)
    if (coords[1] == 500e-9) and (coords[2] == 500e-9):
        return np.array([0, 0, 9e6]) * (1 - s)

def B(coords):
    return 0.6
    # if (coords[0] == 0) and (coords[1] == 0):
    #     return 0.597
    # if (coords[0] == 500e-9) and (coords[1] == 0):
    #     return 0.598
    # if (coords[1] == 500e-9) and (coords[2] == 0):
    #     return 0.599
    # if (coords[1] == 500e-9) and (coords[2] == 500e-9):
    #     return 0.6

e_perp1, e_z1 = e_perp2, e_z2 = e_perp3, e_z3 = e_perp4, e_z4 = h.calc_E_perp_z(B(coords[0]))
# e_perp1, e_z1 = h.calc_E_perp_z(B(coords[0]))
# e_perp2, e_z2 = h.calc_E_perp_z(B(coords[2]))
# e_perp3, e_z3 = h.calc_E_perp_z(B(coords[4]))
# e_perp4, e_z4 = h.calc_E_perp_z(B(coords[6]))
def E(coords, s, h: Hamiltonian):
    return e_perp1 + (e_z1 - e_perp1) * s
    # if (coords[0] == 0) and (coords[1] == 0):
    #     return e_perp1 + (e_z1 - e_perp1) * s
    # if (coords[0] == 500e-9) and (coords[1] == 0):
    #     return e_perp2 + (e_z2 - e_perp2) * s
    # if (coords[1] == 500e-9) and (coords[2] == 0):
    #     return e_perp3 + (e_z3 - e_perp3) * s
    # if (coords[1] == 500e-9) and (coords[2] == 500e-9):
    #     return e_perp4 + (e_z4 - e_perp4) * s

from functools import partial
E1 = partial(E, s=0, h=h)
E2 = partial(E, s=1, h=h)
Xi1 = partial(Xi, s=0)
Xi2 = partial(Xi, s=1)
couplings, biases = l.calc_couplings_biases_lattice(E1, B, Xi1)
print(biases - biases[0])
hs, deltas, j_z, j_perp = l.calc_couplings_qubits_2q(E1, B, Xi1)
print(hs, deltas, j_z, j_perp)
hs, deltas, j_z, j_perp = l.calc_couplings_qubits_2q(E2, B, Xi2)
print(hs, deltas, j_z, j_perp)


# isings = []
# exchanges = []
# z_terms = []
# x_terms = []
# for i in range(num_mols):
#     c1 = None
#     c2 = None
#     for j in range(num_mols):
#         if i == j:
#             o1 = 1 / 2 * qt.sigmaz()
#             o2 = qt.sigmax()
#         else:
#             o1 = qt.qeye(2)
#             o2 = qt.qeye(2)
#         if c1 is None:
#             c1 = o1
#             c2 = o2
#         else:
#             c1 = qt.tensor(c1, o1)
#             c2 = qt.tensor(c2, o2)
#     z_terms.append(c1)
#     x_terms.append(c2)
    
# for i in range(num_edges):
#     c1 = None
#     c2 = None
#     for j in range(num_mols):
#         mols = l.mol_edge_indices()[0][i]
#         if j in mols:
#             o1 = 1 / 2 * qt.sigmaz()
#             if mols[0] == j:
#                 o2 = qt.sigmap()
#             else:
#                 o2 = qt.sigmam()
#         else:
#             o1 = o2 = qt.qeye(2)
#         if c1 is None:
#             c1 = o1
#             c2 = o2
#         else:
#             c1 = qt.tensor(c1, o1)
#             c2 = qt.tensor(c2, o2)
#     isings.append(c1)
#     c2 += c2.dag()
#     c2 /= 2
#     exchanges.append(c2)
    
# def get_ham(t, args) -> qt.Qobj:
#     l = args['lattice']
#     tmax = args['tmax']
#     s = t / tmax
#     Es = partial(E, s=s, h=l.hamiltonian)
#     couplings, biases = l.calc_couplings_biases_lattice(Es, B)
#     # print(couplings)
#     j_perp = couplings[:, 0]
#     j_z = couplings[:, 1]
    
#     ham = None
#     for i in range(num_mols):
#         term = hs[i] * z_terms[i]
#         if ham is None:
#             ham = term
#         else:
#             ham += term
#     for i in range(num_edges):
#         term = j_z[i] * isings[i] + j_perp[i] * exchanges[i]
#         ham += term
    

#     ham *= 2 * np.pi
#     ham.dims = [[2**num_mols], [2**num_mols]]
#     return ham

# args = {
#     'lattice': l,
#     'tmax': 1.5e-3
# }
# tlist = np.linspace(0.0 * args['tmax'], 1.0 * args['tmax'], 5000)
# h0 = get_ham(tlist[0], args)
# psi0 = 1 / np.sqrt(2**len(valid_states)) * sum([qt.basis(2**num_mols, i) for i in valid_states])
# # _, psi0 = h0.groundstate()
# print(psi0)

# evals_mat = np.zeros((len(tlist), 2**num_mols))
# P_mat = np.zeros((len(tlist), 2**num_mols))
# sol_over = np.zeros((len(tlist)))
# val_over = np.zeros(len(tlist))

# idx = [0]
# def process_rho(t, psi):
  
#     # evaluate the Hamiltonian with gradually switched on interaction 
#     H = get_ham(t, args)

#     # find the M lowest eigenvalues of the system
#     evals, ekets = H.eigenstates()

#     evals_mat[idx[0],:] = np.real(evals)
    
#     # find the overlap between the eigenstates and psi 
#     for n, eket in enumerate(ekets):
#         P_mat[idx[0],n] = abs((eket.dag().data * psi.data)[0,0])**2    
    
#     psi2 = psi.full() * psi.conj().full()
#     sol_over[idx[0]] = sum([psi2[i, 0] for i in solutions])
#     val_over[idx[0]] = sum([psi2[i, 0] for i in valid_states])
        
#     idx[0] += 1

# result = qt.mesolve(get_ham, psi0, tlist, [], process_rho, args=args, options=qt.Options(store_states=True))
# np.savez('result_3q_afm.npz', evals_mat=evals_mat, P_mat=P_mat, sol_over=sol_over, val_over=val_over)
# qt.fileio.qsave(result, 'result_3q_afm')