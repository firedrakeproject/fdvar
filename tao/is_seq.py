import numpy as np
from petsc4py import PETSc

Print = PETSc.Sys.Print
Vec = PETSc.Vec
IS = PETSc.IS

# SubVecs: [a], [b]

# size of vecs
na = 4
nb = 2

va = Vec().createSeq(na)
vb = Vec().createSeq(nb)

va.setArray([1, 2, 3, 4])
vb.setArray([51, 42])

Print(f"{va.array = }")
Print(f"{vb.array = }")

# VecNest: [a, b]
ia = IS().createGeneral([0, 1, 2, 3])
ib = IS().createGeneral([4, 5])

vn = Vec().createNest([va, vb], [ia, ib])
Print(f"{vn.array = }")

# VecNest: [a0, b0, a1, b1]
ia = IS().createGeneral([0, 1, 3, 4])
ib = IS().createGeneral([2, 5])

vn = Vec().createNest([va, vb], [ia, ib])
Print(f"{vn.array = }")

