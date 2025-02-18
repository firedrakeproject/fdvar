import firedrake as fd
from advection_wc4dvar_aaorf import make_fdvrf
from fdvar.mat import *

Jhat, control = make_fdvrf()
ensemble = Jhat.ensemble

saddlepoint_params = {
    'ksp_monitor': None,
    'ksp_converged_rate': None,
    'ksp_rtol': 1e-5,
    'ksp_type': 'fgmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_0_fields': '0,1',  # schur complement on dx
    'pc_fieldsplit_1_fields': '2',

    # diagonal pc
    'pc_fieldsplit_schur_fact_type': 'diag',

    # triangular pc
    'pc_fieldsplit_schur_fact_type': 'upper',

    # inexact constraint pc
    'pc_fieldsplit_0_fields': '2',
    'pc_fieldsplit_1_fields': '1',
    'pc_fieldsplit_2_fields': '0',
    'pc_fieldsplit_type': 'multiplicative',

    'fieldsplit_0': {  # D
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'EnsembleBJacobiPC',
        'sub': {
            'ksp_rtol': 1e-5,
            'ksp_type': 'cg',
            'pc_type': 'gamg',
            'mg_levels': {
                'ksp_type': 'chebyshev',
                'ksp_max_it': 2,
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu',
            }
        }
    }
    'fieldsplit_1': {  # R
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'EnsembleBJacobiPC',
        'sub': {
            'ksp_rtol': 1e-5,
            'ksp_type': 'cg',
            'pc_type': 'icc',
        }
    }
    'fieldsplit_2': {  # S
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'SaddleSchurPC',
        'pc_saddle_schur_observation_type': 'none',  # or low-rank?
        'model_sub': {  # L = I
            'ksp_type': 'preonly',
            'pc_type': 'none',
        }
        'model_sub': {  # Low bandwidth approx - its=N is direct solve
            'ksp_max_it': 1,
            'ksp_type': 'richardson',
            'pc_type': 'none',
        }
        'model_sub': {  # L_M pc - bjacobi with bsize = k
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'AllAtOnceBJacobiPC',
            'pc_aaobjacobi_bsize': 4,
            'sub': {
                'ksp_max_it': 4,
                'ksp_type': 'richardson',
                'pc_type': 'none',
            }
        }
    }
}

ksp, options = FDVarSaddlePointKSP(fdvrf, saddlepoint_params)
b = FDVarSaddlePointRHS(fdvrf)  # TODO
x = b.duplicate()

with options.inserted_options():
    ksp.solve(b, x)
