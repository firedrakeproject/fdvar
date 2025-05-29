import numpy as np
from petsc4py import PETSc

Print = PETSc.Sys.Print
Mat = PETSc.Mat
Vec = PETSc.Vec
IS = PETSc.IS

comm = PETSc.COMM_WORLD

nranks = comm.size
rank = comm.rank

# size of blocks
n0l = 4
n1l = 2
nl = n0l + n1l

n0g = n0l*nranks
n1g = n1l*nranks
ng = n0g + n1g

# SubMats:
# [a] [b]
# [c] [d]
ma = Mat().createDense(((n0l, n0g), (n0l, n0g)), comm=comm)
mb = Mat().createDense(((n0l, n0g), (n1l, n1g)), comm=comm)
mc = Mat().createDense(((n1l, n1g), (n0l, n0g)), comm=comm)
md = Mat().createDense(((n1l, n1g), (n1l, n1g)), comm=comm)

offa = 100 + rank
offb = 200 + rank
offc = 300 + rank
offd = 400 + rank
ma.getDenseArray()[:] = offa
mb.getDenseArray()[:] = offb
mc.getDenseArray()[:] = offc
md.getDenseArray()[:] = offd
# 
print(f"{rank=} | ma =\n{ma.getDenseArray()}")
print(f"{rank=} | mb =\n{mb.getDenseArray()}")
print(f"{rank=} | mc =\n{mc.getDenseArray()}")
print(f"{rank=} | md =\n{md.getDenseArray()}")

# MatNest:
# [[a, b]
#  [c, d]]
offset = rank*nl
ia = IS().createGeneral(offset + np.array([0, 1, 3, 4], dtype=np.int32))
ib = IS().createGeneral(offset + np.array([2, 5], dtype=np.int32))

mat = Mat().createNest(
    mats=[[ma, mb], [mc, md]],
    isrows=[ia, ib],
    iscols=[ia, ib])

# print(f"mat =\n{mat.getDenseLocalMatrix()}")
# 
# # vn = Vec().createNest([va, vb], [ia, ib])
# # Print(f"{vn.array = }")
# 
