import firedrake as fd
from firedrake.petsc import PETSc
from petsctools import OptionsManager
from petsctools import (
    OptionsManager, inserted_options,
    attach_options, set_from_options)
from firedrake.utils import IntType
from firedrake.adjoint import ReducedFunctional
from firedrake.adjoint.fourdvar_reduced_functional import FourDVarReducedFunctional
from pyop2.mpi import MPI
from pyadjoint.optimization.tao_solver import (
    PETScVecInterface, ReducedFunctionalMat, RFOperation)
from pyadjoint.enlisting import Enlist
from fdvar.pc import PCBase
from fdvar.mat import EnsembleBlockDiagonalMat
from fdvar.correlations import CorrelationOperatorMat
from typing import Optional
from enum import Enum
from functools import partial, cached_property, wraps
import numpy as np

__all__ = (
    "WC4DVarSaddlePointPC",
    "WC4DVarSaddlePointMat",
    "WC4DVarSaddlePointKSP",
    "getSubWC4DVarSaddlePointMat",
)


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


def nest_ises(vecs, comm):
    # size of local part of each sub block
    nlocals = [v.local_size for v in vecs]

    # global offset of local part of SPS
    global_offset = comm.exscan(sum(nlocals))
    if comm.rank == 0:  # exscan returns None
        global_offset = 0

    # global offset into local part of each SPS block
    offsets = [0]
    for j in range(1, len(vecs)):
        offsets.append(offsets[j-1] + nlocals[j-1])

    # global indices of DoFs in local part of each SPS block
    idxs = [offsets[j] + np.arange(nlocals[j], dtype=IntType)
            for j in range(len(vecs))]

    # IS for each SPS block
    ises = tuple(PETSc.IS().createGeneral(idx, comm=comm)
                 for idx in idxs)

    return ises


def getSubWC4DVarSaddlePointMat(mat, sub=None):
    """
    Return a sub matrix of the saddle point MatNest.
    Options are 'D', 'R', 'L', 'LT', 'H', 'HT',
    or None to return all sub matrices.
    """
    idx = {
        'D': (0, 0),
        'R': (1, 1),
        'L': (0, 2),
        'LT': (2, 0),
        'H': (1, 2),
        'HT': (2, 1),
    }
    return (
        mat.getNestSubMatrix(*idx[sub])
        if sub is not None else
        tuple(mat.getNestSubMatrix(*i) for i in idx.values())
    )


def WC4DVarSaddlePointKSP(Jhat, Jphat=None, solver_parameters=None, options_prefix=None):
    amat = WC4DVarSaddlePointMat(Jhat)
    pmat = WC4DVarSaddlePointMat(Jphat or Jhat)
    ksp = PETSc.KSP().create(
        comm=Jhat.ensemble.global_comm)
    ksp.setOperators(amat, pmat)

    set_from_options(
        ksp, parameters=solver_parameters,
        options_prefix=options_prefix)

    return ksp


def WC4DVarSaddlePointMat(Jhat):
    if not isinstance(Jhat, FourDVarReducedFunctional):
        raise TypeError(
            "WC4DVarSaddlePointMat must be constructed from a"
            f" FourDVarReducedFunctional, not a {type(Jhat).__name__}")

    ensemble = Jhat.ensemble
    Wc = Jhat.control_space
    Wo = Jhat.observation_space

    vec_dn = Wc.layout_vec.duplicate()
    vec_dl = Wo.layout_vec.duplicate()
    vec_dx = Wc.layout_vec.duplicate()

    Lmat = ReducedFunctionalMat(
        Jhat.JL, action=RFOperation.TLM,
        comm=ensemble.global_comm)

    LTmat = ReducedFunctionalMat(
        Jhat.JL, action=RFOperation.ADJOINT,
        comm=ensemble.global_comm)

    Hmat = ReducedFunctionalMat(
        Jhat.JH, action=RFOperation.TLM,
        comm=ensemble.global_comm)

    HTmat = ReducedFunctionalMat(
        Jhat.JH, action=RFOperation.ADJOINT,
        comm=ensemble.global_comm)

    BQmats = [
        CorrelationOperatorMat(
            rf.covariance, action='apply')
        for rf in Jhat.JD.rfs]

    Dmat = EnsembleBlockDiagonalMat(
        BQmats, col_space=Wc,
        row_space=Wc.dual())

    Rmats = [
        CorrelationOperatorMat(
            rf.covariance, action='apply')
        for rf in Jhat.JR.rfs]

    Rmat = EnsembleBlockDiagonalMat(
        Rmats, col_space=Wo,
        row_space=Wo.dual())

    A22 = PETSc.Mat().createConstantDiagonal(
        (vec_dx.sizes, vec_dx.sizes), 0.,
        comm=ensemble.global_comm)
    A22.setUp()
    A22.assemble()
    A22.setAttr("Jhat", Jhat)

    saddle_mat = PETSc.Mat().createNest(
        mats=[[Dmat,  None,  Lmat],     # noqa: E127,E202
              [None,  Rmat,  Hmat],     # noqa: E127,E202
              [LTmat, HTmat, A22]],     # noqa: E127,E202
        comm=ensemble.global_comm)
    saddle_mat.setUp()
    saddle_mat.assemble()

    return saddle_mat


class WC4DVarSaddlePointPC(PCBase):
    needs_python_amat = True
    needs_python_pmat = True

    prefix = "wcsaddle_"

    def initialize(self, pc):
        super().initialize(pc)

        Jhat = self.amat.rf
        if not isinstance(Jhat, FourDVarReducedFunctional):
            raise TypeError(
                f"{obj_name(self)} expects a FourDVarReducedFunctional not a {obj_name(Jhat)}")

        Jphat = self.pmat.rf
        if not isinstance(Jphat, FourDVarReducedFunctional):
            raise TypeError(
                f"{obj_name(self)} expects a FourDVarReducedFunctional not a {obj_name(Jphat)}")

        self.Jhat = Jhat
        self.Jphat = Jphat
        self.ensemble = Jphat.ensemble

        self.rhs_type = "saddle"

        prefix = pc.getOptionsPrefix() + self.prefix
        self.use_amat = PETSc.Options().getBool(prefix + "use_amat", False)

        if self.use_amat:
            Jhat_a = Jhat
        else:
            Jhat_a = Jphat
        Jhat_p = Jphat

        self.saddle_ksp = WC4DVarSaddlePointKSP(
            Jhat_a, Jhat_p, options_prefix=self.full_prefix)
        self.saddle_mat, _ = self.saddle_ksp.getOperators()

        self.saddle_ksp.incrementTabLevel(1, parent=pc)
        self.saddle_ksp.pc.incrementTabLevel(1, parent=pc)

        self.rhs = self._create_vec()
        self.sol = self._create_vec()

        self.rhs_dn, self.rhs_dl, self.rhs_dx = self.rhs.getNestSubVecs()
        self.sol_dn, self.sol_dl, self.sol_dx = self.sol.getNestSubVecs()

    def _create_vec(self):
        Wc = self.Jphat.control_space
        Wo = self.Jphat.observation_space

        v_dn = Wc.layout_vec.duplicate()
        v_dl = Wo.layout_vec.duplicate()
        v_dx = Wc.layout_vec.duplicate()

        v = PETSc.Vec().createNest(
            vecs=(v_dn, v_dl, v_dx),
            isets=self.saddle_mat.getNestISs()[0],
            comm=self.Jphat.ensemble.global_comm)

        v.setUp()

        return v

    def _build_rhs(self):
        vec = self.rhs

        val = self.Jhat.control.data()
        v_dn, v_dl, v_dx = vec.getNestSubVecs()

        b = self.Jphat.JL(val)
        d = self.Jphat.JH(val)

        with b.vec_ro() as bvec:
            bvec.copy(result=v_dn)

        with d.vec_ro() as dvec:
            dvec.copy(result=v_dl)

        v_dx.zeroEntries()

        return vec

    def apply(self, pc, x, y):
        # self._build_rhs()
        self.sol.zeroEntries()
        self.rhs.zeroEntries()

        if self.rhs_type == "saddle":
            val = self.Jhat.control.data()
            # val = self.val

            with self.Jphat.JL(val).vec_ro() as bvec:
                bvec.copy(result=self.rhs_dn)

            with self.Jphat.JH(val).vec_ro() as dvec:
                dvec.copy(result=self.rhs_dl)

        elif self.rhs_type == "primal":
            x.copy(result=self.rhs_dx)

        with inserted_options(self.saddle_ksp):
            self.saddle_ksp.solve(self.rhs, self.sol)

        self.sol_dx.copy(result=y)

    def update(self, pc):
        val = self.Jhat.control.data()
        self.Jphat(val)
        self.Jphat.derivative(apply_riesz=False)
        self.val = val
