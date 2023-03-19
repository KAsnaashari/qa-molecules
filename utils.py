import numpy as np

base_dir = '/arc/project/st-rkrems-1/kasra/qa/'
scratch_dir = '/scratch/st-rkrems-1/kasra/qa/'

mu_b = 9.27400968e-24
g_s = 2.00231930436256
epsilon0 = 8.8541878128e-12
c0 = 299792458
K = 1 / (4 * np.pi * epsilon0)
j_to_cm = 5.03445e22
cm_to_j = 1/j_to_cm
j_to_hz = 1.50930e33
j_to_mhz = j_to_hz / 1e6
j_to_ghz = j_to_hz / 1e9
hz_to_j = 1/j_to_hz
debye_to_Cm = 3.335640952e-30
angst3_to_Fm2 = 1/8.988e15

srf_params = {
    'b_e': 0.251 * cm_to_j,
    'd': 3.47 * debye_to_Cm,
    'gamma': 2.49e-3 * cm_to_j,
    'alpha_parallel': 123.29e-24 * angst3_to_Fm2,
    'alpha_perp': 192.87e-24 * angst3_to_Fm2
}

sri_params = {
    'b_e': 0.0367 * cm_to_j,
    'd': 6 * debye_to_Cm,
    'gamma': 3.29e-3 * cm_to_j,
    'alpha_parallel': 0 * angst3_to_Fm2,
    'alpha_perp': 0 * angst3_to_Fm2
}

pauli_Z = np.array([[1, 0], [0, -1]])
pauli_X = np.array([[0, 1], [1, 0]])
pauli_Y = np.array([[0, -1j], [1j, 0]])

jjj_coeffs = {}
clebsch_coeffs = {}

def print_log(s, file):
    s = str(s)
    print(s)
    f = open(file, 'a')
    f.write(s)
    f.write('\n')
    f.close()

def zero_tol(a, tol=1e-7):
    import numpy as np
    
    b = np.array(a)
    b[abs(b) < tol] = 0
    return b

def binary_even(n):
    if n == 0:
        return True
    if n % 2 == 0:
        return binary_even(n // 2)
    else:
        return not binary_even((n - 1) // 2)

def xi_to_I(xi):
    xi = np.array(xi)
    I = np.dot(xi, xi) * c0 * epsilon0 / 2
    return I

def expspace(start, stop, steps, p=3):
    delta = stop - start
    print(delta)
    x = np.linspace(0, 1, steps)
    return start + delta * x**p

def mols_to_qubits(n, qubits):
    import numpy as np

    out = []
    n_bin = np.binary_repr(n, width=qubits.shape[0] * qubits.shape[1])
    # print(n_bin)

    for i in range(qubits.shape[0]):
        q1, q2 = qubits[i]
        if (n_bin[q1] == '0') and (n_bin[q2] == '1'):
            out.append('0')
        elif (n_bin[q1] == '1') and (n_bin[q2] == '0'):
            out.append('1')
        else:
            raise ValueError('Invalid molecule state.')
    
    out = ''.join(out)
    return out

def qubits_to_mols(n, qubits):
    import numpy as np

    out_bin = ['' for i in range(qubits.shape[0] * 2)]
    for i in range(qubits.shape[0]):
        q1, q2 = qubits[i]
        if n[i] == '0':
            out_bin[q1] = '0'
            out_bin[q2] = '1'
        elif n[i] == '1':
            out_bin[q1] = '1'
            out_bin[q2] = '0'
        else:
            raise ValueError('Invalid qubit state.')
    
    out_bin = ''.join(out_bin)
    out = int(out_bin, 2)
    return out

def calc_clebsch(j1, j2, j3, m1, m2, m3):
    import qutip as qt
    global clebsch_coeffs

    j = np.array([[j1, j2, j3], [m1, m2, m3]])
    key = ','.join(map(str, j.astype(int).flatten()))
    j = j.astype(float)

    if key in clebsch_coeffs.keys():
        return clebsch_coeffs[key]

    out = qt.clebsch(j1, j2, j3, m1, m2, m3)
    # print('clebsch')

    keys = []
    keys.append(key)
    keys.append(','.join(map(str, (j[:, [1, 0, 2]] * np.array([[1, 1, 1], [-1, -1, -1]])).astype(int).flatten())))
    for k in keys:
        clebsch_coeffs[k] = out

    keys = []
    keys.append(','.join(map(str, j[:, [1, 0, 2]].astype(int).flatten())))
    keys.append(','.join(map(str, (j * np.array([[1, 1, 1], [-1, -1, -1]])).astype(int).flatten())))
    for k in keys:
        clebsch_coeffs[k] = np.power(-1, j1 + j2 - j3) * out

    return out

def calc_3j(j):
    key = ','.join(map(str, j.astype(int).flatten()))
    j = j.astype(float)

    if key in jjj_coeffs.keys():
        return jjj_coeffs[key]

    j1, j2, j3 = j[0]
    m1, m2, m3 = j[1]

    c = calc_clebsch(j1, j2, j3, m1, m2, -m3)
    out = np.power(-1, j1 - j2 - m3) / np.sqrt(2 * j3 + 1) * c
    # print('3j')

    keys = []
    keys.append(key)
    keys.append(','.join(map(str, j[:, [1, 2, 0]].astype(int).flatten())))
    keys.append(','.join(map(str, j[:, [2, 0, 1]].astype(int).flatten())))
    keys.append(','.join(map(str, j[:, [2, 0, 1]].astype(int).flatten())))
    for k in keys:
        jjj_coeffs[k] = out
    
    keys = []
    keys.append(','.join(map(str, j[:, [0, 2, 1]].astype(int).flatten())))
    keys.append(','.join(map(str, j[:, [1, 0, 2]].astype(int).flatten())))
    keys.append(','.join(map(str, j[:, [2, 1, 0]].astype(int).flatten())))
    keys.append(','.join(map(str, (j * np.array([[1, 1, 1], [-1, -1, -1]])).astype(int).flatten())))
    for k in keys:
        jjj_coeffs[k] = np.power(-1, j1 + j2 + j3) * out

    return out

def check_stoq_2q(hs, deltas, jz, j_perp, zero=1e-4, grid=1001):
    import numpy as np

    if (jz == 0) and (j_perp == 0):
        return True
    if jz == j_perp:
        return True

    if (jz < 0) and (j_perp < 0):
        if jz <= j_perp:
            a_xx = j_perp / jz
            a_yy = - j_perp / jz
            d1 = -2 * deltas[0] / np.abs(jz)
            d2 = 2 * deltas[1] / np.abs(jz)
            h1 = -2 * hs[0] / np.abs(jz)
            h2 = 2 * hs[1] / np.abs(jz)
        else:
            a_xx = jz / j_perp
            a_yy = -1
            d1 = -2 * hs[0] / np.abs(jz)
            d2 = 2 * hs[1] / np.abs(jz)
            h1 = -2 * deltas[0] / np.abs(jz)
            h2 = 2 * deltas[1] / np.abs(jz)
    elif (jz >= 0) and (j_perp >= 0):
        if jz <= j_perp:
            a_xx = a_yy = j_perp / jz
            d1 = 2 * deltas[0] / np.abs(jz)
            d2 = 2 * deltas[1] / np.abs(jz)
            h1 = 2 * hs[0] / np.abs(jz)
            h2 = 2 * hs[1] / np.abs(jz)
        else:
            a_xx = jz / j_perp
            a_yy = 1
            d1 = 2 * hs[0] / np.abs(jz)
            d2 = 2 * hs[1] / np.abs(jz)
            h1 = 2 * deltas[0] / np.abs(jz)
            h2 = 2 * deltas[1] / np.abs(jz)
    
    if (h1 == 0) and (h2 == 0) and (d1 == 0) and (d2 == 0):
        return True
    

    tl = np.linspace(0, 2 * np.pi, grid)
    tl_grid, tr_grid = np.meshgrid(tl, tl)

    # h1, h2 = np.array(hs) / jz * 2
    # delta1, delta2 = np.array(deltas) / jz * 2
    # j_xx = j_perp / jz

    ineq1 = np.sin(tl_grid) * np.sin(tr_grid) + a_xx * np.cos(tl_grid) * np.cos(tr_grid) + np.abs(a_yy)
    ineq2 = d2 * np.cos(tr_grid) + h2 * np.sin(tr_grid) + np.abs(np.cos(tl_grid) * np.sin(tr_grid) - a_xx * np.cos(tr_grid) * np.sin(tl_grid))
    ineq3 = d1 * np.cos(tl_grid) + h1 * np.sin(tl_grid) + np.abs(np.cos(tr_grid) * np.sin(tl_grid) - a_xx * np.cos(tl_grid) * np.sin(tr_grid))
    
    a = ineq1 <= zero
    b = ineq2 <= zero
    c = ineq3 <= zero

    ab = np.logical_and(a, b)
    abc = np.logical_and(ab, c)

    return np.any(abc)

def conserved_binary(N):
    s_list = []
    conserved_binary_rec(N, '', 0, 0, s_list)
    return s_list

def conserved_binary_rec(N, s, ones, zeros, s_list):
    if len(s) == N:
        s_list.append(s)

    if ones < N / 2:
        conserved_binary_rec(N, s + '1', ones + 1, zeros, s_list)
    
    if zeros < N / 2:
        conserved_binary_rec(N, s + '0', ones, zeros + 1, s_list)
    