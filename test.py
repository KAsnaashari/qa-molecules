
import numpy as np
from utils import srf_params, expspace
from hamiltonian import Hamiltonian
from lattice import Lattice

coords = np.array([[0, 0, 0], [0, 0, 500e-9], [1000e-9, 0, 0], [1000e-9, 0, 500e-9], ])
qubits = np.array([[0, 1], [2, 3]])

# x = 2
# y = 2
# coords = np.empty((x * y * 2, 3))
# qubits = np.empty((x * y, 2), dtype=int)
# r1 = 500e-9
# r2 = 1000e-9
# r3 = 1000e-9

# for i in range(x):
#     for j in range(y):
#         q1 = 2 * (i * x + j)
#         q2 = 2 * (i * x + j) + 1
#         qubits[i * x + j, :] = [q1, q2]

#         c1 = [i * r2, j * r3, 0]
#         c2 = [i * r2, j * r3, r1]
#         coords[q1, :] = c1
#         coords[q2, :] = c2

# print(coords)
# print(qubits)

h = Hamiltonian(srf_params)
l = Lattice(coords, qubits, h, biasers=None)

# print(l.qubit_edge_indices())
def B(coords):
    return 0.600

e_perp, e_z = h.calc_E_perp_z(B(None))
steps = 50
tmax = 1e-3

def E(coords, t, h: Hamiltonian):
    s = t / tmax
    return e_perp + (e_z - e_perp) * s**3

from functools import partial

tlist = np.linspace(0, 1e-3, 100)
couplings_list = []
biases_list = []

for t in tlist:
    Et = partial(E, t=t, h=h)
    couplings, biases = l.calc_couplings_biases_lattice(Et, B)
    couplings_list.append(couplings)
    biases_list.append(biases)

np.savez('couplings.npz', couplings=np.array(couplings_list), biases=np.array(biases_list))