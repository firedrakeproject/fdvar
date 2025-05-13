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
            if not all(vecs[i].getSizes() == self.getSizes(i) for i in range(len(self))):
                raise ValueError("vec sizes must match is sizes")
        return PETSc.Vec().createNest(vecs, self.ises, self.comm)


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

    dn_is, dl_is, dx_is = saddle_ises(fdvrf)

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
