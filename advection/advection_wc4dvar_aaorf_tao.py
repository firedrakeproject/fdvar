import firedrake as fd
from firedrake.adjoint import (
    continue_annotation, pause_annotation, minimize,
    stop_annotating, Control, taylor_test)
from firedrake.adjoint import FourDVarReducedFunctional
from firedrake.petsc import PETSc
from advection_utils import *
from fdvar import TAOSolver
from sys import exit

from advection_wc4dvar_aaorf import make_fdvrf
Jhat, control = make_fdvrf()

Print = PETSc.Sys.Print

# the perturbation values need to be held in the
# same type as the control i.e. and EnsembleFunction
x0 = control.copy()

ksp_params = {
    'monitor_short': None,
    # 'converged_rate': None,
    # 'converged_reason': None,
}

tao_params = {
    'tao_view': ':tao_view.log',
    'tao': {
        'monitor': None,
        'converged_reason': None,
        'gatol': 1e-1,
        'grtol': 1e-1,
        'gttol': 1e-1,
    },
    'tao_type': 'nls',
    'tao_nls': {
        'ksp': ksp_params,
        'ksp_type': 'gmres',
        'pc_type': 'lmvm',
        'ksp_rtol': 1e-1,
        # 'ksp_type': 'gmres',
        # 'pc_type': 'python',
        # 'pc_python_type': 'fdvar.AllAtOnceJacobiPC',
        
    },
    # 'tao_cg': {
    #     'ksp': ksp_params,
    #     'ksp_rtol': 1e-1,
    #     'type': 'pr',  # fr-pr-prp-hs-dy
    # },
}
tao = TAOSolver(Jhat, options_prefix="",
                solver_parameters=tao_params)
tao.solve()

xopt = Jhat.control.copy()
J0 = Jhat(x0)
Jopt = Jhat(xopt)

Print(f"{J0 = :.3e} | {Jopt = :.3e} | {Jopt/J0 = :.3e}")


