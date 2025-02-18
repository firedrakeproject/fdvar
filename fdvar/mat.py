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
from functools import partial


class ISNest:
    attrs = (
        'getSizes',
        'getSize',
        'getLocalSize',
        'getIndices',
        'getComm',
    )
    def __init__(self, ises):
        self._comm = ises[0].getComm()
        self._ises = ises
        for attr in self.attrs:
            setattr(self, attr,
                    partial(self._getattr, attr))

    def _getattr(self, attr, i, *args, **kwargs):
        return getattr(self.ises[i], attr)(*args, **kwargs)

    @cached_property
    def globalSize(self):
        return sum(self.getSize(i) for i in range(len(self)))

    @cached_property
    def localSize(self):
        return sum(self.getLocalSize(i) for i in range(len(self)))

    @property
    def sizes(self):
        return (self.localSize, self.globalSize)

    @property
    def comm(self):
        return self._comm

    @property
    def ises(self):
        return self._ises

    def __getitem__(self, i):
        return self.ises[i]

    def __len__(self):
        return len(self.ises)

    def __iter__(self):
        return iter(self.ises)

    def createVec(self, i=None, vec_type=PETSc.Vec.Type.MPI):
        vec = PETSc.Vec().create(comm=self.comm)
        vec.setType(vec_type)
        vec.setSizes(self.sizes if i is None else self.getSizes(i))
        return vec

    def createVecs(self, vec_type=PETSc.Vec.Type.MPI):
        return (self.createVec(i, vec_type=vec_type) for i in range(len(self)))

    def createVecNest(self, vecs=None):
        if vecs is None:
            vecs = self.createVecs()
        else:
            if all(vecs[i].getSizes() != self.getSizes(i) for i in range(len(self))):
                raise ValueError("vec sizes must match is sizes")
        return PETSc.Vec().createNest(vecs, self.ises, self.comm)


class RFAction(Enum):
    TLM = 'tlm'
    Adjoint = 'adjoint'
    Hessian = 'hessian'
TLM = RFAction.TLM
Adjoint = RFAction.Adjoint
Hessian = RFAction.Hessian


def copy_controls(controls):
    return controls.delist([c.copy() for c in controls])


def convert_types(overloaded, options):
    overloaded = Enlist(overloaded)
    return overloaded.delist([o._ad_convert_type(o, options=options)
                              for o in overloaded])


def saddle_ises(fdvrf):
    ensemble = fdvrf.ensemble
    rank = ensemble.ensemble_rank
    global_comm = ensemble.global_comm

    Vsol = fdvrf.solution_space
    Vobs = fdvrf.observation_space
    Vs = Vsol.local_spaces[0]
    Vo = Vobs.local_spaces[0]

    # ndofs per (dn, dl, dx) block
    bsol = Vsol.dof_dset.layout_vec.getLocalSize()
    bobs = Vobs.dof_dset.layout_vec.getLocalSize()
    nlocal_blocks = Vsol.nlocal_spaces

    bsize_dn = bsol
    bsize_dl = bobs
    bsize_dx = bsol
    bsize = bsize_dn + bsize_dl + bsize_dx

    # number of blocks on previous ranks
    nprev_blocks = ensemble.ensemble_comm.exscan(nlocal_blocks)
    if rank == 0:  # exscan returns None
        nprev_blocks = 0

    # offset to start of global indices of each field in local block j
    offset = bsize*nprev_blocks
    offset_dn = lambda j: offset + j*bsize
    offset_dl = lambda j: offset_dn(j) + bsize_dn
    offset_dx = lambda j: offset_dl(j) + bsize_dl

    indices_dn = np.concatenate(
        [offset_dn(j) + np.arange(bsize_dn, dtype=np.int32)
         for j in range(nlocal_blocks)])

    indices_dl = np.concatenate(
        [offset_dl(j) + np.arange(bsize_dl, dtype=np.int32)
         for j in range(nlocal_blocks)])

    indices_dx = np.concatenate(
        [offset_dx(j) + np.arange(bsize_dx, dtype=np.int32)
         for j in range(nlocal_blocks)])

    is_dn = PETSc.IS().createGeneral(indices_dn, comm=global_comm)
    is_dl = PETSc.IS().createGeneral(indices_dl, comm=global_comm)
    is_dx = PETSc.IS().createGeneral(indices_dx, comm=global_comm)

    return ISNest((is_dn, is_dl, is_dx))


def FDVarSaddlePointKSP(fdvrf, solver_parameters, options_prefix=None):
    saddlemat = FDVarSaddlePointMat(fdvrf)

    ksp = PETSc.KSP().create(comm=fdvrf.ensemble.global_comm)
    ksp.setOperators(saddlemat, saddlemat)

    options = OptionsManager(solver_parameters, options_prefix)
    options.set_from_options(ksp)

    return ksp, options


# Saddle-point MatNest
def FDVarSaddlePointMat(fdvrf):
    ensemble = fdvrf.ensemble

    isnest = saddle_ises(fdvrf)
    dn_is, dl_is, dx_is = isnest.ises

    # L Mat
    L = AllAtOnceRFMat(fdvrf, action=TLM)
    Lt = AllAtOnceRFMat(fdvrf, action=Adjoint)

    Lrow = dn_is
    Lcol = dx_is

    Ltrow = dx_is
    Ltcol = dn_is

    # H Mat
    H = ObservationEnsembleRFMat(fdvrf, action=TLM)
    Ht = ObservationEnsembleRFMat(fdvrf, action=Adjoint)

    Hrow = dl_is
    Hcol = dx_is

    Htrow = dx_is
    Htcol = dl_is

    # D Mat
    D = ModelCovarianceEnsembleRFMat(fdvrf)

    Drow = dn_is
    Dcol = dn_is

    # R Mat
    R = ObservationCovarianceEnsembleRFMat(fdvrf)

    Rrow = dl_is
    Rcol = dl_is

    fdvmat = PETSc.Mat().createNest(
        mats=[D,     L,               # noqa: E127,E202
                  R, H,               # noqa: E127,E202
              Lt, Ht  ],              # noqa: E127,E202
        isrows=[Drow,        Lrow,    # noqa: E127,E202
                       Rrow, Hrow,    # noqa: E127,E202
                Ltrow, Htrow      ],  # noqa: E127,E202
        iscols=[Dcol,        Lcol,    # noqa: E127,E202
                       Rcol, Hcol,    # noqa: E127,E202
                Ltcol, Htcol     ],   # noqa: E127,E202
        comm=ensemble.global_comm)

    return fdvmat


class ReducedFunctionalMatCtx:
    """
    PythonMat context to apply action of a pyadjoint.ReducedFunctional.

    Parameters
    ----------

        action : RFAction
    """
    def __init__(self, Jhat: ReducedFunctional,
                 action: str = Hessian,
                 options: Optional[dict] = None,
                 input_options: Optional[dict] = None,
                 comm: MPI.Comm = PETSc.COMM_WORLD):
        self.Jhat = Jhat
        self.control_interface = PETScVecInterface(Jhat.controls, comm=comm)
        self.functional_interface = PETScVecInterface(Jhat.functional, comm=comm)

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
        self.input_options = input_options
        self.options = options
        self._shift = 0

    @classmethod
    def update(cls, obj, x, A, P):
        ctx = A.getPythonContext()
        ctx.control_interface.from_petsc(x, ctx._m)
        ctx._shift = 0

    def update_tape_values(self, update_adjoint=True):
        _ = self.Jhat(self._m)
        if update_adjoint:
            _ = self.Jhat.derivative(options=self.options)

    def mult(self, A, x, y):
        self.xinterface.from_petsc(x, self.x)
        if self.input_options is None:
            _x = self.x
        else:
            _x = convert_types(self.x, self.input_options)
        out = self.mult_impl(A, _x)
        self.yinterface.to_petsc(y, out)
        if self._shift != 0:
            y.axpy(self._shift, x)

    def _mult_hessian(self, A, x):
        if self.action != Hessian:
            raise NotImplementedError(
                f'Cannot apply hessian action if {self.action = }')
        self.update_tape_values(update_adjoint=True)
        return self.Jhat.hessian(x, options=self.options)

    def _mult_tlm(self, A, x):
        if self.action != TLM:
            raise NotImplementedError(
                f'Cannot apply tlm action if {self.action = }')
        self.update_tape_values(update_adjoint=False)
        return self.Jhat.tlm(x, options=self.options)

    def _mult_adjoint(self, A, x):
        if self.action != Adjoint:
            raise NotImplementedError(
                f'Cannot apply adjoint action if {self.action = }')
        self.update_tape_values(update_adjoint=False)
        return self.Jhat.derivative(adj_input=x, options=self.options)


def ReducedFunctionalMat(Jhat, action=Hessian,
                         options=None, input_options=None,
                         comm=PETSc.COMM_WORLD):
    ctx = ReducedFunctionalMatCtx(
        Jhat, action,
        options=options,
        input_options=input_options,
        comm=comm)

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
            col_space = row_space

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
        self.y = fd.EnsembleFunction(self.col_space.dual())

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


def EnsembleBlockDiagonalMat(blocks, row_space, col_space=None):
    return EnsembleMat(
        EnsembleBlockDiagonalMatCtx(blocks, row_space, col_space),
        row_space, col_space)


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

        # propogate from last step
        for M, xi, mxi in zip(self.models, self.xprevs, self.mx.subfunctions):
            mxi.assign(M.mult(xi))

        # diagonal contribution
        x -= self.M

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
