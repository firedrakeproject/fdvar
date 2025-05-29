import numpy as np
from petsc4py import PETSc

Print = PETSc.Sys.Print
Vec = PETSc.Vec
IS = PETSc.IS

comm = PETSc.COMM_WORLD

nranks = comm.size
rank = comm.rank

# SubVecs: [a], [b]

# local size of vecs
na = 4
nb = 2
n = na + nb

va = Vec().createMPI((na, nranks*na))
vb = Vec().createMPI((nb, nranks*nb))

offa = rank*na
offb = 100 + rank*nb
va.setArray(offa + np.arange(na))
vb.setArray(offb + np.arange(nb))

comm.Barrier()
print(f"subVecs: {rank=} | {va.array = } | {vb.array = }")
comm.Barrier()

# VecNest: [a, b]
offset = rank*n
ia = IS().createGeneral(offset + np.array([0, 1, 2, 3], dtype=np.int32))
ib = IS().createGeneral(offset + np.array([4, 5], dtype=np.int32))

vn = Vec().createNest([va, vb], [ia, ib])
comm.Barrier()
print(f"Continguous: {rank=} | {vn.array = }")
comm.Barrier()

# VecNest: [a0, b0, a1, b1]
ia = IS().createGeneral(offset + np.array([0, 1, 3, 4], dtype=np.int32))
ib = IS().createGeneral(offset + np.array([2, 5], dtype=np.int32))

vn = Vec().createNest([va, vb], [ia, ib])

comm.Barrier()
print(f"Interleaved: {rank=} | {vn.array = }")
comm.Barrier()
