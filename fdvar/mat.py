import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.adjoint import ReducedFunctional
from firedrake.adjoint.fourdvar_reduced_functional import CovarianceNormReducedFunctional
from firedrake.ensemble.ensemble_functionspace import EnsembleFunctionSpaceBase
from pyop2.mpi import MPI
from pyadjoint.optimization.tao_solver import PETScVecInterface
from pyadjoint.enlisting import Enlist
from typing import Optional
from enum import Enum
from functools import partial, cached_property, wraps


def EnsembleMat(ctx):
    # number of columns is row length, and vice-versa
    row_sizes = ctx.row_space.layout_vec.getSizes()
    col_sizes = ctx.col_space.layout_vec.getSizes()

    mat = PETSc.Mat().createPython(
        (row_sizes, col_sizes), ctx,
        comm=ctx.ensemble.global_comm)
    mat.setUp()
    mat.assemble()
    return mat


class EnsembleMatCtxBase:
    def __init__(self, row_space, col_space):
        if not isinstance(row_space, EnsembleFunctionSpaceBase):
            raise ValueError(
                f"EnsembleMat row_space must be EnsembleFunctionSpace not {type(row_space).__name__}")
        if not isinstance(col_space, EnsembleFunctionSpaceBase):
            raise ValueError(
                f"EnsembleMat col_space must be EnsembleFunctionSpace not {type(col_space).__name__}")

        if row_space.ensemble != col_space.ensemble:
            raise ValueError(
                "Ensembles of row and column spaces of EnsembleMat must be the same")

        self.ensemble = row_space.ensemble
        self.row_space = row_space
        self.col_space = col_space

        # input/output Vecs will be copied in/out of these
        # so that base classes can implement mult only in
        # terms of Ensemble objects not Vecs.
        self.x = fd.EnsembleFunction(self.row_space)
        self.y = fd.EnsembleFunction(self.col_space)

    def mult(self, A, x, y):
        with self.x.vec_wo() as xvec:
            x.copy(result=xvec)

        self.mult_impl(A, self.x, self.y)

        with self.y.vec_ro() as yvec:
            yvec.copy(result=y)


class EnsembleBlockDiagonalMatCtx(EnsembleMatCtxBase):
    def __init__(self, blocks, row_space, col_space):
        super().__init__(row_space, col_space)
        self.blocks = blocks

        if self.row_space.nlocal_spaces != self.col_space.nlocal_spaces:
            raise ValueError(
                "EnsembleBlockDiagonalMat row and col spaces must be the same length,"
                f" not {row_space.nlocal_spaces = } and {col_space.nlocal_spaces = }")

        if len(self.blocks) != self.row_space.nlocal_spaces:
            raise ValueError(
                f"EnsembleBlockDiagonalMat requires one submatrix for each of the"
                f" {self.row_space.nlocal_spaces} local subfunctions of the EnsembleFunctionSpace,"
                f" but only {len(self.blocks)} provided.")

        for i, (Vrow, Vcol, block) in enumerate(zip(self.row_space.local_spaces,
                                                    self.col_space.local_spaces,
                                                    self.blocks)):
            # number of columns is row length, and vice-versa
            vc_sizes = Vrow.dof_dset.layout_vec.sizes
            vr_sizes = Vcol.dof_dset.layout_vec.sizes
            mr_sizes, mc_sizes = block.sizes
            if (vr_sizes[0] != mr_sizes[0]) or (vr_sizes[1] != mr_sizes[1]):
                raise ValueError(
                    f"Row sizes {mr_sizes} of block {i} and {vr_sizes} of row_space {i} of EnsembleBlockDiagonalMat must match.")
            if (vc_sizes[0] != mc_sizes[0]) or (vc_sizes[1] != mc_sizes[1]):
                raise ValueError(
                    f"Col sizes of block {i} and col_space {i} of EnsembleBlockDiagonalMat must match.")

    def mult_impl(self, A, x, y):
        for block, xsub, ysub in zip(self.blocks,
                                     x.subfunctions,
                                     y.subfunctions):
            with xsub.dat.vec_ro as xvec, ysub.dat.vec_wo as yvec:
                block.mult(xvec, yvec)


def EnsembleBlockDiagonalMat(blocks, row_space, col_space):
    return EnsembleMat(
        EnsembleBlockDiagonalMatCtx(blocks, row_space, col_space))
