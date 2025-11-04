import numpy as np
from firedrake import *
from firedrake.adjoint import *
from petsctools import set_from_options, inserted_options
from pyadjoint.optimization.tao_solver import (
    ReducedFunctionalMat, TLMAction, AdjointAction, HessianAction)

def aaorf(W, dt, n):
    Nlocal = W.nlocal_spaces
    if ensemble.ensemble_rank == 0:
        Nlocal -= 1

    V = W.local_spaces[0]
    u = TrialFunction(V)
    v = TestFunction(V)
    
    a = inner(u, v)*dx + Constant(dt)*inner(grad(u), grad(v))*dx
    
    ic = Function(V).interpolate(sin(2*pi*x))
    
    propagators = []
    continue_annotation()
    for _ in range(Nlocal):
        control = Function(V).assign(ic)
        u0, u1 = Function(V), Function(V)
        L = inner(u0, v)*dx
    
        with set_working_tape() as tape:
            u1.assign(control)
            for _ in range(n):
                u0.assign(u1)
                solve(a==L, u1)
            Jm = ReducedFunctional(u1, Control(control), tape=tape)
        propagators.append(Jm)
    pause_annotation()
    
    control = EnsembleFunction(W)
    functional = EnsembleFunction(W)
    
    return AllAtOnceReducedFunctional(functional, Control(control), propagators)

N = 15
n = 10

nx = 16
dt = 0.02

ensemble = Ensemble(COMM_WORLD, 1)

mesh = PeriodicUnitIntervalMesh(nx, comm=ensemble.comm)
x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)

Nlocal = (N+1)//ensemble.ensemble_size
if ensemble.ensemble_rank == 0:
    Nlocal += 1
W = EnsembleFunctionSpace([V for _ in range(Nlocal)], ensemble)

Jhat = aaorf(W, dt, n)
JPhat = aaorf(W, dt*n, 1)

sol = EnsembleFunction(W)
rhs = EnsembleCofunction(W.dual())

parameters = {
    'ksp_view': ':ksp_view.log',
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    'ksp_rtol': 1e-10,
    'ksp_type': 'richardson',
    'pc_type': 'ksp',
    'ksp': {
        'ksp_converged_maxits': None,
        'ksp_convergence_test': 'skip',
        'ksp_max_it': N+1,
        'ksp_type': 'richardson',
        'pc_type': 'none',
    }
}

action = AdjointAction
Amat = ReducedFunctionalMat(Jhat, action=action, comm=COMM_WORLD)
Pmat = ReducedFunctionalMat(JPhat, action=action, comm=COMM_WORLD)

ksp = PETSc.KSP().create(comm=COMM_WORLD)
ksp.setOperators(Amat, Pmat)
set_from_options(ksp, parameters, options_prefix="")

with rhs.vec() as v:
    v.array[:] = np.random.random_sample(v.array.shape)

with inserted_options(ksp):
    with rhs.vec() as vb, sol.vec() as xb:
        ksp.solve(vb, xb)
