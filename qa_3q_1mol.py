import numpy as np
import qutip as qt
from utils import srf_params, mols_to_qubits, qubits_to_mols, scratch_dir
from hamiltonian import Hamiltonian
from lattice import Lattice

coords = np.array([[0, 0, 0], [0, 500e-9, 0], [0, 1000e-9, 0]])

# valid_states_q = np.array(['00', '01', '10', '11'])
# valid_states = np.array(list(map(lambda s: qubits_to_mols(s, qubits), valid_states_q)))
solutions_q = np.array(['010'])
solutions = np.array([int(b[::-1], 2) for b in solutions_q])
print(f'Coordinates: {coords}')
# print(f'Qubits: {qubits}')
# print(f'Valid States: {valid_states}')
print(f'Solutions: {solutions}')

h = Hamiltonian(srf_params)
l = Lattice(coords, None, h, biasers=None)
num_mols = l.num_mols
num_edges = l.num_edges

# print(l.qubit_edge_indices())
# def B(coords, s):
#     return 0.600
# def Xi(coords, s):
#     return None

# e_perp, e_z = h.calc_E_perp_z(B(None, 0), xi=Xi(None, 0), threshold=20)
# print(f'E_perp = {(e_perp / 1e5):.3f}, E_z = {e_z / 1e5}')
# def E(coords, s):
#     return e_perp + (e_z - e_perp) * s

# def MW(coords, s, l: Lattice):
#     Xis = partial(Xi, s=s)
#     Es = partial(E, s=s)
#     Bs = partial(B, s=s)
#     _, biases = l.calc_couplings_biases_lattice(Es, Bs, Xis)
#     # (e_a, e_b), _ = h.calc_ab(E(coords, s), B(coords, s), xi=Xi(coords, s))
#     w = biases[1] + s * 0
#     e = 0.5 * 1 / (19 * s + 1)
#     return w, e

# from functools import partial
# E1 = partial(E, s=0)
# B1 = partial(B, s=0)
# Xi1 = partial(Xi, s=0)
# MW1 = partial(MW, s=0, l=l)
# E2 = partial(E, s=1)
# B2 = partial(B, s=1)
# Xi2 = partial(Xi, s=1)
# MW2 = partial(MW, s=1, l=l)
# hs, deltas, j_z, j_perp = l.calc_couplings_qubits_1q(E1, B1, Xi1, MW1)
# print(hs, deltas, j_z, j_perp)
# hs, deltas, j_z, j_perp = l.calc_couplings_qubits_1q(E2, B2, Xi2, MW2)
# print(hs, deltas, j_z, j_perp)

isings = []
exchanges = []
z_terms = []
x_terms = []
for i in range(num_mols):
    c1 = None
    c2 = None
    for j in range(num_mols):
        if i == j:
            o1 = 1 / 2 * qt.sigmaz()
            o2 = 1 / 2 * qt.sigmax()
        else:
            o1 = qt.qeye(2)
            o2 = qt.qeye(2)
        if c1 is None:
            c1 = o1
            c2 = o2
        else:
            c1 = qt.tensor(c1, o1)
            c2 = qt.tensor(c2, o2)
    z_terms.append(c1)
    x_terms.append(c2)
    
for i in range(num_edges):
    c1 = None
    c2 = None
    for j in range(num_mols):
        mols = l.mol_edge_indices()[0][i]
        if j in mols:
            o1 = 1 / 2 * qt.sigmaz()
            if mols[0] == j:
                o2 = qt.sigmap()
            else:
                o2 = qt.sigmam()
        else:
            o1 = o2 = qt.qeye(2)
        if c1 is None:
            c1 = o1
            c2 = o2
        else:
            c1 = qt.tensor(c1, o1)
            c2 = qt.tensor(c2, o2)
    isings.append(c1)
    c2 += c2.dag()
    print(c2)
    c2 /= 2
    exchanges.append(c2)
    
# def get_ham(t, args) -> qt.Qobj:
#     from functools import partial
#     l = args['lattice']
#     tmax = args['tmax']
#     stoq = args['stoq']
#     s = t / tmax
#     Es = partial(E, s=s)
#     Bs = partial(B, s=s)
#     Xis = partial(Xi, s=s)
#     MWs = partial(MW, s=s, l=l)
#     hs, deltas, j_z, j_perp = l.calc_couplings_qubits_1q(Es, Bs, Xis, MWs)
#     if stoq:
#         j_perp = np.zeros(j_perp.shape)

#     ham = None
#     for i in range(num_mols):
#         term = hs[i] * z_terms[i] + deltas[i] * x_terms[i]
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

# tmax = 3e-3
# for stoq in [False, True]:
#     args = {
#         'lattice': l,
#         'tmax': tmax,
#         'stoq': stoq
#     }
#     tlist = np.linspace(0.0 * args['tmax'], 1.0 * args['tmax'], 500)

#     evals_mat = np.zeros((len(tlist), 2**num_mols))
#     P_mat = np.zeros((len(tlist), 2**num_mols))
#     sol_over = np.zeros((len(tlist)))
#     val_over = np.zeros(len(tlist))

#     idx = [0]
#     def process_rho(t, psi):
    
#         # evaluate the Hamiltonian with gradually switched on interaction 
#         H = get_ham(t, args)

#         # find the M lowest eigenvalues of the system
#         evals, ekets = H.eigenstates()

#         evals_mat[idx[0],:] = np.real(evals)
        
#         # find the overlap between the eigenstates and psi 
#         for n, eket in enumerate(ekets):
#             P_mat[idx[0],n] = abs((eket.dag().data * psi.data)[0,0])**2    
        
#         psi2 = psi.full() * psi.conj().full()
#         sol_over[idx[0]] = sum([psi2[i, 0] for i in solutions])
#         # val_over[idx[0]] = sum([psi2[i, 0] for i in valid_states])
            
#         idx[0] += 1

#     h0 = get_ham(tlist[0], args)
#     # psi0 = 1 / np.sqrt(2**num_mols) * sum([qt.basis(2**num_mols, i) for i in range(2**num_mols)])
#     _, psi0 = h0.groundstate()
#     print(psi0)

#     result = qt.mesolve(get_ham, psi0, tlist, [], process_rho, args=args, options=qt.Options(store_states=True))
#     np.savez(f'{scratch_dir}/result_3q_1mol{"_stoq" if stoq else ""}_afm_{int(tmax * 1e4)}.npz', evals_mat=evals_mat, P_mat=P_mat, sol_over=sol_over, val_over=val_over)
#     qt.fileio.qsave(result, f'{scratch_dir}/result_3q_1mol{"_stoq" if stoq else ""}_afm_{int(tmax * 1e4)}')