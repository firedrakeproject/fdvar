import numpy as np
from petsc4py import PETSc

Print = PETSc.Sys.Print
Mat = PETSc.Mat
Vec = PETSc.Vec
IS = PETSc.IS

# size of blocks
n0 = 4
n1 = 2

# SubMats:
# [a] [b]
# [c] [d]
ma = Mat().createDense((n0, n0))
mb = Mat().createDense((n0, n1))
mc = Mat().createDense((n1, n0))
md = Mat().createDense((n1, n1))

ma.getDenseArray()[:] = 1
mb.getDenseArray()[:] = 2
mc.getDenseArray()[:] = 3
md.getDenseArray()[:] = 4

print(f"ma =\n{ma.getDenseArray()}")
print(f"mb =\n{mb.getDenseArray()}")
print(f"mc =\n{mc.getDenseArray()}")
print(f"md =\n{md.getDenseArray()}")

# MatNest:
# [[a, b]
#  [c, d]]
ia = IS().createGeneral([0, 1, 2, 3])
ib = IS().createGeneral([4, 5])

mat = Mat().createNest(
    mats=[[ma, mb], [mc, md]],
    isrows=[ia, ib],
    iscols=[ia, ib])

print(f"mat =\n{mat.getDenseLocalMatrix()}")

# vn = Vec().createNest([va, vb], [ia, ib])
# Print(f"{vn.array = }")
