import numpy as np
from utils import check_stoq_2q, scratch_dir

zero = 1e-4
j_z = 1
j_perps = [0.99, 1.01]
hs = np.linspace(0, 2, 201)
ds = np.linspace(0, 2, 201)

stoq = []
for j_perp in j_perps:
    stoq_j = []
    for h in zip(hs, hs):
        stoq_j.append(list(map(lambda d: check_stoq_2q(h, (d, d), j_z, j_perp, zero=zero), ds)))
    stoq.append(stoq_j)
    np.savez(scratch_dir + 'stoq_jhd.npz', stoq=np.array(stoq), j_perps=j_perps, hs=hs, ds=ds, xyz='jhd')
