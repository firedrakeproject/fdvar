import firedrake as fd
from firedrake.petsc import PETSc
from firedrake.petsc import OptionsManager
from firedrake.adjoint import ReducedFunctional
from firedrake.adjoint.fourdvar_reduced_functional import CovarianceNormReducedFunctional
from pyop2.mpi import MPI
from pyadjoint.optimization.tao_solver import PETScVecInterface
from pyadjoint.enlisting import Enlist
from typing import Optional
from enum import Enum
from functools import partial, cached_property, wraps


class RFAction(Enum):
    TLM = 'tlm'
    Adjoint = 'adjoint'
    Hessian = 'hessian'
TLM = RFAction.TLM
Adjoint = RFAction.Adjoint
Hessian = RFAction.Hessian


def copy_controls(controls):
    return controls.delist([c.control._ad_init_zero() for c in controls])


def convert_types(overloaded, options):
    overloaded = Enlist(overloaded)
    return overloaded.delist([o._ad_convert_type(o, options=options)
                              for o in overloaded])


def check_rf_action(action):
    def check_rf_action_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.action != action:
                raise NotImplementedError(
                    f'Cannot apply {str(action)} action if {self.action = }')
            return func(*args, **kwargs)
        return wrapper
    return check_rf_action_decorator


class ReducedFunctionalMatCtx:
    """
    PythonMat context to apply action of a pyadjoint.ReducedFunctional.

    Jhat : V -> U
    TLM : V -> U
    Adjoint : U* -> V*
    Hessian : V x U* -> V* | V -> V*

    Parameters
    ----------

        action : RFAction
    """

    def __init__(self, Jhat: ReducedFunctional,
                 action: str = Hessian, *,
                 apply_riesz: bool = False,
                 comm: MPI.Comm = PETSc.COMM_WORLD):
        self.Jhat = Jhat
        self.control_interface = PETScVecInterface(
            Jhat.controls, comm=comm)
        self.apply_riesz = apply_riesz
        if action in (Adjoint, TLM):
            self.functional_interface = PETScVecInterface(
                Jhat.functional, comm=comm)

        if action == Hessian:  # control -> control
            self.xinterface = self.control_interface
            self.yinterface = self.control_interface

            self.x = copy_controls(Jhat.controls)
            self.mult_impl = self._mult_hessian

        elif action == Adjoint:  # functional -> control
            self.xinterface = self.functional_interface
            self.yinterface = self.control_interface

            self.x = Jhat.functional._ad_copy()
            self.mult_impl = self._mult_adjoint

        elif action == TLM:  # control -> functional
            self.xinterface = self.control_interface
            self.yinterface = self.functional_interface

            self.x = copy_controls(Jhat.controls)
            self.mult_impl = self._mult_tlm
        else:
            raise ValueError(
                'Unrecognised {action = }.')

        self.action = action
        self._m = copy_controls(Jhat.controls)
        self._shift = 0

    @classmethod
    def update(cls, obj, x, A, P):
        ctx = A.getPythonContext()
        ctx.control_interface.from_petsc(x, ctx._m)
        ctx.update_tape_values(update_adjoint=True)
        ctx._shift = 0

    def shift(self, A, alpha):
        self._shift += alpha

    def update_tape_values(self, update_adjoint=True):
        _ = self.Jhat(self._m)
        if update_adjoint:
            _ = self.Jhat.derivative(apply_riesz=False)

    def mult(self, A, x, y):
        self.xinterface.from_petsc(x, self.x)
        out = self.mult_impl(A, self.x)
        self.yinterface.to_petsc(y, out)

        if self._shift != 0:
            y.axpy(self._shift, x)

    # @check_rf_action(action=Hessian)
    def _mult_hessian(self, A, x):
        # self.update_tape_values(update_adjoint=True)
        return self.Jhat.hessian(
            x, apply_riesz=self.apply_riesz)

    # @check_rf_action(TLM)
    def _mult_tlm(self, A, x):
        # self.update_tape_values(update_adjoint=False)
        return self.Jhat.tlm(x)

    # @check_rf_action(Adjoint)
    def _mult_adjoint(self, A, x):
        # self.update_tape_values(update_adjoint=False)
        return self.Jhat.derivative(
            adj_input=x, apply_riesz=self.apply_riesz)


def ReducedFunctionalMat(Jhat, action=Hessian, *, comm=PETSc.COMM_WORLD, **kwargs):
    ctx = ReducedFunctionalMatCtx(
        Jhat, action, comm=comm, **kwargs)

    ncol = ctx.xinterface.n
    Ncol = ctx.xinterface.N

    nrow = ctx.yinterface.n
    Nrow = ctx.yinterface.N

    mat = PETSc.Mat().createPython(
        ((nrow, Nrow), (ncol, Ncol)),
        ctx, comm=comm)
    mat.setUp()
    mat.assemble()
    return mat


def EnsembleMat(ctx, row_space, col_space=None):
    if col_space is None:
        col_space = row_space

    # number of columns is row length, and vice-versa
    ncol = row_space.nlocal_rank_dofs
    Ncol = row_space.nglobal_dofs

    nrow = col_space.nlocal_rank_dofs
    Nrow = col_space.nglobal_dofs

    mat = PETSc.Mat().createPython(
        ((nrow, Nrow), (ncol, Ncol)), ctx,
        comm=row_space.ensemble.global_comm)
    mat.setUp()
    mat.assemble()
    return mat


class EnsembleMatCtxBase:
    def __init__(self, row_space, col_space=None):
        if col_space is None:
            col_space = row_space.dual()

        if not isinstance(row_space, fd.EnsembleFunctionSpace):
            raise ValueError(
                f"EnsembleMat row_space must be EnsembleFunctionSpace not {type(row_space).__name__}")
        if not isinstance(col_space, fd.EnsembleFunctionSpace):
            raise ValueError(
                f"EnsembleMat col_space must be EnsembleFunctionSpace not {type(col_space).__name__}")

        self.row_space = row_space
        self.col_space = col_space

        # input/output Vecs will be copied in/out of these
        # so that base classes can implement mult only in
        # terms of Ensemble objects not Vecs.
        self.x = fd.EnsembleFunction(self.row_space)
        self.y = fd.EnsembleFunction(self.col_space)

    def mult(self, A, x, y):
        with self.x.vec_wo() as xvec:
            x.copy(xvec)

        self.mult_impl(A, self.x, self.y)

        with self.y.vec_ro() as yvec:
            yvec.copy(y)


class EnsembleBlockDiagonalMatCtx(EnsembleMatCtxBase):
    def __init__(self, blocks, row_space, col_space=None):
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
                                     self.x.subfunctions,
                                     self.y.subfunctions):
            with xsub.dat.vec_ro as xvec, ysub.dat.vec_wo as yvec:
                block.mult(xvec, yvec)


def EnsembleBlockDiagonalMat(row_space, blocks, **kwargs):
    return EnsembleMat(
        EnsembleBlockDiagonalMatCtx(blocks, row_space, **kwargs),
        row_space, kwargs.get('col_space', None))


# L Mat
class AllAtOnceRFMatCtx(EnsembleMatCtxBase):
    def __init__(self, fdvrf, action, **kwargs):
        super().__init__(fdvrf.solution_space)

        if action not in (TLM, Adjoint):
            raise ValueError(
                f"AllAtOnceRFMat action type must be 'tlm' or 'adjoint', not {action}")

        self.action = action
        self.fdvrf = fdvrf
        self.ensemble = fdvrf.ensemble

        self.models = [ReducedFunctionalMat(M, action, **kwargs)
                       for M in fdvrf.model_rfs]

        if action == Adjoint:
            self.models = reversed(self.models)

        self.xhalo = fd.Function(self.row_space.local_spaces[0])
        self.mx = self.x.copy()

        # Set up list of x_{i-1} functions to propogate.
        # TLM means we propogate forwards, and use halo from previous rank.
        # Adjoint means we propogate backwards, and use halo from next rank.
        # The initial timestep on the initial rank doesn't have a halo.
        if self.action == TLM:
            self.xprevs = [*self.x.subfunctions[:-1]]
        else:
            self.xprevs = [*reversed(self.x.subfunctions[1:])]

        initial_rank = 0 if action == TLM else (self.ensemble.ensemble_size - 1)
        if self.ensemble.ensemble_rank != initial_rank:
            self.xprevs.insert(self.xhalo, 0)

    def update_halos(self, x):
        ensemble_rank = self.ensemble.ensemble_rank
        ensemble_size = self.ensemble.ensemble_size

        # halo swap is a right shift
        next_rank = (ensemble_rank + 1) % ensemble_size
        prev_rank = (ensemble_rank - 1) % ensemble_size

        src = prev_rank if self.action == TLM else next_rank
        dst = next_rank if self.action == TLM else prev_rank

        frecv = self.xhalo
        fsend = self.x.subfunctions[-1 if self.action == TLM else 0]

        self.ensemble.sendrecv(
            fsend=fsend, dest=dst, sendtag=dst,
            frecv=frecv, source=src, recvtag=ensemble_rank)

    def mult_impl(self, A, x, y):
        self.update_halos(x)

        # propogate from last step mx <- M*x
        for M, xi, mxi in zip(self.models, self.xprevs, self.mx.subfunctions):
            mxi.assign(M.mult(xi))

        # diagonal contribution
        # x_{i} <- x_{i} - M*x_{i-1}
        x -= self.mx

        y.assign(x.riesz_representation())


def AllAtOnceRFMat(fdvrf, action, **kwargs):
    return EnsembleMat(
        AllAtOnceRFMatCtx(fdvrf, action, **kwargs),
        fdvrf.solution_space)


# H Mat
def ObservationEnsembleRFMat(fdvrf, action, **kwargs):
    if action == TLM:
        row_space = fdvrf.solution_space
        col_space = fdvrf.observation_space
    elif action == Adjoint:
        row_space = fdvrf.observation_space
        col_space = fdvrf.solution_space
    else:
        raise ValueError(
            f"Unrecognised matrix action type {action}")

    blocks = [ReducedFunctionalMat(Jobs, action, **kwargs)
              for Jobs in fdvrf.observation_rfs]
    return EnsembleBlockDiagonalMat(blocks, row_space, col_space)


# CovarianceMat
def CovarianceMat(covariancerf, action='mult'):
    """
    action='mult' for action of B, or 'inv' for action of B^{-1}
    """
    if not isinstance(covariancerf, CovarianceNormReducedFunctional):
        raise TypeError(
            "CovarianceMat can only be constructed from a CovarianceNormReducedFunctional"
            f" not a {type(covariancerf).__name__}")
    space = covariancerf.controls[0].control.function_space()
    comm = space.mesh().comm
    covariance = covariancerf.covariance

    if action == 'mult':
        weight = float(covariance)
    elif action == 'inv':
        weight = float(1/covariance)
    else:
        raise ValueError(f"Unrecognised action type {action} for CovarianceMat")

    sizes = space.dof_dset.layout_vec.sizes
    shape = (sizes, sizes)
    covmat = PETSc.Mat().createConstantDiagonal(
        shape, covariance, comm=comm)
    covmat.setUp()
    covmat.assemble()
    return covmat


# D Mat
def ModelCovarianceEnsembleRFMat(fdvrf, action, **kwargs):
    blocks = [CovarianceMat(mnorm, action, **kwargs)
              for mnorm in fdvrf.model_norms]
    if fdvrf.ensemble.ensemble_rank == 0:
        blocks.insert(
            0, CovarianceMat(fdvrf.background_norm, action, **kwargs))
    return EnsembleBlockDiagonalMat(blocks, fdvrf.solution_space)


# R Mat
def ObservationCovarianceEnsembleRFMat(fdvrf, action, **kwargs):
    blocks = [CovarianceMat(obs_norm, action, **kwargs)
              for obs_norm in fdvrf.observation_norms]
    return EnsembleBlockDiagonalMat(blocks, fdvrf.observation_space)
