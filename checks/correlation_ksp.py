import numpy as np
from firedrake import *
from fdvar.correlations import *
from petsctools import set_from_options, inserted_options

nx = 32
L = 0.2
m = 2
sigma2 = 1e-3

mesh = PeriodicUnitIntervalMesh(nx)
V = FunctionSpace(mesh, "CG", 1)

B = ImplicitDiffusionCorrelation(V, sqrt(sigma2), L, m=m, seed=2)

Amat = CorrelationOperatorMat(B, action="apply")

parameters = {
    "ksp_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "fdvar.CorrelationOperatorPC"
}

ksp = PETSc.KSP().create()
ksp.setOperators(Amat)
set_from_options(ksp, parameters, options_prefix="")

rhs, sol = Function(V), Function(V)
with rhs.dat.vec_wo as v:
    v.array[:] = np.random.random_sample(v.array.shape)

with inserted_options(ksp):
    with rhs.dat.vec as vb, sol.dat.vec as vx:
        ksp.solve(vb, vx)
