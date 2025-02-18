import firedrake as fd
from firedrake.petsc import PETSc, OptionsManager
from firedrake.adjoint import pyadjoint  # noqa: F401
from firedrake.matrix import ImplicitMatrix
from pyadjoint.optimization.tao_solver import PETScVecInterface
from advection_wc4dvar_aaorf import make_fdvrf
from mpi4py import MPI
import numpy as np
from typing import Optional, Collection
from fdvar.mat import *


def CovarianceMat(covariancerf):
    space = covariancerf.controls[0].control.function_space()
    comm = space.mesh().comm
    covariance = covariancerf.covariance
    if isinstance(covariance, Collection):
        covariance, power = covariance

    sizes = space.dof_dset.layout_vec.sizes
    shape = (sizes, sizes)
    covmat = PETSc.Mat().createConstantDiagonal(
        shape, covariance, comm=comm)

    covmat.setUp()
    covmat.assemble()
    return covmat


Jhat, control = make_fdvrf()
ensemble = Jhat.ensemble

# >>>>> Covariance

# Covariance Mat
# covrf = Jhat.background_norm
covrf = Jhat.stages[0].model_norm
covmat = CovarianceMat(covrf)

# Covariance KSP
covksp = PETSc.KSP().create(comm=ensemble.comm)
covksp.setOptionsPrefix('cov_')
covksp.setOperators(covmat)

covksp.pc.setType(PETSc.PC.Type.JACOBI)
covksp.setType(PETSc.KSP.Type.PREONLY)
covksp.setFromOptions()
covksp.setUp()
print(PETSc.Options().getAll())

x = covmat.createVecRight()
b = covmat.createVecLeft()

b.array_w[:] = np.random.random_sample(b.array_w.shape)
print(f'{b.norm() = }')
covksp.solve(b, x)
print(f'{x.norm() = }')
