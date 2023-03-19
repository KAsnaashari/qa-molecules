import numpy as np
import qutip as qt
from os import mkdir
from utils import conserved_binary, print_log, srf_params, mols_to_qubits, qubits_to_mols, scratch_dir
from hamiltonian import Hamiltonian
from lattice import Lattice
from scipy.special import comb
from functools import partial
scratch_dir = '../../scratch/qa'

i1 = 3
i2 = 4
r1 = 500e-9
r2 = 1000e-9

coords = []
for i in range(i1):
    for j in range(i2):
        coords.append([i * r2, j * r2, 0])
        coords.append([i * r2, j * r2, r1])
coords = np.array(coords)

qubits = np.array([[2 * i, 2 * i + 1] for i in range(0, i1 * i2)])
# print(qubits)

h = Hamiltonian(srf_params)
l = Lattice(coords, qubits, h, biasers=None)
num_mols = l.num_mols
num_edges = l.num_edges

tmax_list = [5e-3, 10e-3, 15e-3, 20e-3]

from itertools import permutations
conserved_states_q = np.sort(conserved_binary(num_mols))
conserved_states = np.array(list(map(lambda b: int(b, 2), conserved_states_q)))
valid_states_q = []
valid_states = []
for i, s in enumerate(conserved_states):
    try:
        valid_states_q.append(mols_to_qubits(s, qubits))
        valid_states.append(i)
    except ValueError:
        continue
valid_states_q = np.array(valid_states_q)
valid_states = np.array(valid_states)

solutions_q = np.array(['010110100101', '101001011010'])
solutions = np.array(list(map(lambda s: np.where(conserved_states == qubits_to_mols(s, qubits)), solutions_q))).reshape(-1)

# print(l.qubit_edge_indices())
def Xi(coords, s):
    return 0

def B(coords):
    return 0.6

e_perp, e_z = h.calc_E_perp_z(B(coords[0]), threshold=100)
def E(coords, s, h: Hamiltonian):
    return e_perp + (e_z - e_perp) * np.power(s, 1) if s >=0 else e_perp

isings = []
exchanges = []
z_terms = []
x_terms = []
y_terms = []

for i in range(num_mols):
    op_list = []
    for j in range(num_mols):
        op_list.append(qt.qeye(2))
    
    op_list[i] = 1/2 * qt.sigmaz()
    z_terms.append(qt.Qobj(qt.tensor(op_list)))
    op_list[i] = 1/2 * qt.sigmax()
    x_terms.append(qt.Qobj(qt.tensor(op_list)))
    op_list[i] = 1/2 * qt.sigmay()
    y_terms.append(qt.Qobj(qt.tensor(op_list)))
  
def get_ham(t, args) -> qt.Qobj:
    l = args['lattice']
    tmax = args['tmax']
    atol = args['atol']
    s = t / tmax
    Es = partial(E, s=s, h=l.hamiltonian)
    Xis = partial(Xi, s=s)
    couplings, biases = l.calc_couplings_biases_lattice(Es, B, Xis)
    # print(couplings)
    j_perp = couplings[:, 0]
    j_z = couplings[:, 1]
    
    ham = None
    for i in range(num_mols):
        term = biases[i] * z_terms[i]
        if ham is None:
            ham = term.extract_states(conserved_states)
        else:
            ham += term.extract_states(conserved_states)
    for i in range(num_edges):
        (i1, i2) = l.mol_edge_indices()[0][i]
        ising = z_terms[i1] * z_terms[i2]
        ising = ising.extract_states(conserved_states)

        exchange = x_terms[i1] * x_terms[i2] + y_terms[i1] * y_terms[i2]
        exchange = exchange.extract_states(conserved_states)

        term = j_z[i] * ising + j_perp[i] * exchange
        ham += term

    ham *= 2 * np.pi
    return ham.tidyup(atol=atol)

for tmax in tmax_list:
    dt = 101
    args = {
        'lattice': l,
        'tmax': tmax,
        'atol': 0
    }
    tlist = np.linspace(0.0 * args['tmax'], 1.0 * args['tmax'], dt)

    # scratch_dir = '../../scratch/qa/test'
    save_dir = f'{scratch_dir}/result_2d{i1}x{i2}_afm_{int(tmax * 1e4)}_{dt}'
    mkdir(save_dir)
    save_file = f'{save_dir}/result_2d{i1}x{i2}_afm_{int(tmax * 1e4)}_{dt}'
    # save_file = f'results/result_2d{i1}x{i2}_afm_{int(tmax * 1e4)}_test'
    log_file = f'{save_file}.txt'

    print_log(coords, log_file)
    print_log(qubits, log_file)

    psi0 = 1 / np.sqrt(len(valid_states)) * sum([qt.basis(conserved_states.shape[0], i) for i in valid_states])

    evals_mat = np.zeros((len(tlist), conserved_states.shape[0]))
    P_mat = np.zeros((len(tlist), conserved_states.shape[0]))
    sol_over = np.zeros((len(tlist)))
    val_over = np.zeros(len(tlist))

    idx = [0]
    def process_rho(t, psi):
    
        # # evaluate the Hamiltonian with gradually switched on interaction 
        # H = get_ham(t, args)

        # # find the M lowest eigenvalues of the system
        # evals, ekets = H.eigenstates()

        # evals_mat[idx[0],:] = np.real(evals)
        
        # # find the overlap between the eigenstates and psi 
        # for n, eket in enumerate(ekets):
        #     P_mat[idx[0], n] = abs((eket.dag().data * psi.data)[0,0])**2    
        
        psi2 = psi.full() * psi.conj().full()
        sol_over[idx[0]] = sum([psi2[i, 0] for i in solutions])
        val_over[idx[0]] = sum([psi2[i, 0] for i in valid_states])
        
        s = t / tmax
        step = int(s * dt)
        qt.fileio.qsave(psi, f'{save_file}_state{step}')

        print_log(f's = {s:.2f} (step = {step}) done!', log_file)
            
        idx[0] += 1

    result = qt.mesolve(get_ham, psi0, tlist, [], process_rho, args=args, options=qt.Options(store_final_state=True))

    np.savez(f'{save_file}.npz', evals_mat=evals_mat, P_mat=P_mat, sol_over=sol_over, val_over=val_over)
    qt.fileio.qsave(result, save_file)  