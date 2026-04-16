from functools import cached_property
from abc import abstractmethod
from typing import Sequence
from pyadjoint.reduced_functional import AbstractReducedFunctional
from pyadjoint.enlisting import Enlist
from pyadjoint import (
    Control,
    AdjFloat,
    OverloadedType,
    no_annotations,
)

from pyop2.mpi import MPI
from firedrake import (
    Function,
    Cofunction,
    Ensemble,
    EnsembleFunction,
    EnsembleCofunction,
)

from fdvar.adjoint.ensemble_adjvec import EnsembleAdjVec


LocalType = Function | Cofunction | AdjFloat
EnsembleType = EnsembleFunction | EnsembleCofunction | EnsembleAdjVec

LocalTypeList = LocalType | Sequence[LocalType]
EnsembleTypeList = EnsembleType | Sequence[EnsembleType]

OverloadedTypeList = OverloadedType | Sequence[OverloadedType]
ControlList = Control | Sequence[Control]


def _local_subs(val: EnsembleType | list[LocalType]) -> list[LocalType]:
    if isinstance(val, (EnsembleFunction, EnsembleCofunction)):
        return val.subfunctions
    elif isinstance(val, EnsembleAdjVec):
        return val.subvec
    elif isinstance(val, (list, tuple)):
        return val
    else:
        raise TypeError(
            f"Cannot use {type(val).__name__} as an ensemble type.")


def _local_len(val: EnsembleType) -> int:
    return len(_local_subs(val))


def _global_len(val: EnsembleType) -> int:
    if isinstance(val, (EnsembleFunction, EnsembleCofunction)):
        return val.function_space().nglobal_spaces
    elif isinstance(val, EnsembleAdjVec):
        return val.global_size
    else:
        raise TypeError(
            f"Cannot use {type(val).__name__} as an ensemble type.")


def _set_local_subs(dst: EnsembleType, src: EnsembleType):
    assert _local_len(dst) == _local_len(src)
    dst_subs = _local_subs(dst)
    for i, s in enumerate(src):
        if hasattr(dst_subs[i], 'assign'):
            dst_subs[i].assign(s)
        else:
            dst_subs[i] = s
    return dst


def _ad_add(x: OverloadedType, y: OverloadedType) -> OverloadedType:
    """
    Work around bug in Cofunction._ad_add where
    Cofunction.assign(c, Cofunction + Cofunction) resolves
    to Cofunction.assign(FormSum) which gets rejected.
    """
    if isinstance(x, Cofunction):
        new = x.copy(deepcopy=True)
        new.assign(x + y)
        return new
    else:
        return x._ad_add(y)


def _delist_one(self, v):
    if isinstance(v, list) and len(v) != 1:
        raise ValueError(
            f"{type(self).__name__} only has"
            f" one control, not {len(v)}")
        return v[0]
    else:
        return v


def _ensemble(x: EnsembleType) -> Ensemble:
    if isinstance(x, (EnsembleFunction, EnsembleCofunction)):
        return x.function_space().ensemble
    elif isinstance(x, EnsembleAdjVec):
        return x.ensemble


def reduction(ensemble: Ensemble,
              src: EnsembleType) -> LocalType:
    vals = _local_subs(src)
    local_sum = vals[0]._ad_init_zero()
    for v in vals:
        local_sum = _ad_add(local_sum, v)
    return ensemble.allreduce(local_sum)


def broadcast(ensemble: Ensemble,
              src: LocalType,
              dst: EnsembleType,
              root: int | None = None) -> EnsembleType:
    if root is not None:
        src = ensemble.bcast(src, root=root)
    local_src = [src for _ in range(_local_len(dst))]
    _set_local_subs(dst, local_src)
    return dst


class EnsembleCommunicationBase(AbstractReducedFunctional):
    def __init__(self, ensemble: Ensemble, root: int | None = None):
        self._ensemble = ensemble
        self._root = root

    @cached_property
    def controls(self) -> list[Control]:
        return Enlist(Control(self.src))

    @property
    def functional(self) -> LocalType:
        return self.dst

    @property
    @abstractmethod
    def src(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def dst(self):
        raise NotImplementedError

    @property
    def ensemble(self) -> Ensemble:
        return self._ensemble

    @property
    def root(self) -> int | None:
        return self._root

    @no_annotations
    def __call__(self, values):
        values = _delist_one(self, values)

        # 1. Update input tape values
        self.controls[0].update(values)
        # 2. Communication
        out = self.forward(values)
        # 3. Update output tape values
        self.dst.block_variable.checkpoint = out

        return out

    @no_annotations
    def tlm(self, m_dot):
        m_dot = _delist_one(self, m_dot)

        # 1. Update input tape values
        self.src.block_variable.reset_variables("tlm")
        self.src.block_variable.tlm_value = m_dot

        # 2. Communication
        out = self.forward(m_dot)

        # 3. Update output tape values
        self.dst.block_variable.reset_variables("tlm")
        self.dst.block_variable.add_tlm_output(out)

        return out

    @no_annotations
    def derivative(self, adj_input, apply_riesz: bool = False):
        # 1. Update input tape values
        self.dst.block_variable.reset_variables("adjoint")
        self.dst.block_variable.adj_value = adj_input

        # 2. Communication
        out = self.backward(adj_input)

        # 3. Update output tape values
        self.src.block_variable.reset_variables("adjoint")
        self.src.block_variable.add_adj_output(out)

        if apply_riesz:
            return self.controls[0].control._ad_convert_riesz(
                out, riesz_map=self.controls[0].riesz_map)
        else:
            return out

    @no_annotations
    def hessian(self, m_dot, hessian_input=None,
                evaluate_tlm: bool = True,
                apply_riesz: bool = False):
        if evaluate_tlm:
            self.tlm(m_dot)

        if hessian_input is None:
            hessian_input = self.dst._ad_init_zero(dual=True)

        # 1. Update input tape values
        self.dst.block_variable.reset_variables("hessian")
        self.dst.block_variable.hessian_value = hessian_input

        # 2. Communication
        out = self.backward(hessian_input)

        # 3. Update output tape values
        self.src.block_variable.reset_variables("hessian")
        self.src.block_variable.add_hessian_output(out)

        if apply_riesz:
            return self.controls[0].control._ad_convert_riesz(
                out, riesz_map=self.controls[0].riesz_map)
        else:
            return out


class EnsembleReduce(EnsembleCommunicationBase):
    def __init__(self, src: EnsembleType, root: int | None = None):
        # TODO: some type validation

        super().__init__(_ensemble(src), root)
        self._src = src
        self._dst = _local_subs(src)[0]._ad_init_zero()

    @property
    def src(self) -> EnsembleType:
        return self._src

    @property
    def dst(self) -> LocalType:
        return self._dst

    def forward(self, src: EnsembleType) -> LocalType:
        return reduction(self.ensemble, src)

    def backward(self, src: LocalType) -> EnsembleType:
        out = self.src._ad_init_zero(dual=True)
        return broadcast(
            self.ensemble, src=src, dst=out, root=self.root)


class EnsembleBcast(EnsembleCommunicationBase):
    def __init__(self, dst: EnsembleType, root: int | None = None):
        # TODO: some type validation

        super().__init__(_ensemble(dst), root)
        self._dst = dst
        self._src = _local_subs(dst)[0]._ad_init_zero()

    @property
    def src(self) -> LocalType:
        return self._src

    @property
    def dst(self) -> EnsembleType:
        return self._dst

    def forward(self, src: LocalType) -> EnsembleType:
        out = self.dst._ad_init_zero()
        return broadcast(
            self.ensemble, src=src, dst=out, root=self.root)

    def backward(self, src: EnsembleType) -> LocalType:
        return reduction(self.ensemble, src)


class EnsembleTransform(AbstractReducedFunctional):
    def __init__(self, functional: EnsembleType,
                 controls: ControlList,
                 rfs: list[AbstractReducedFunctional]):
        self._functional = functional
        self._controls = Enlist(controls)
        self._rfs = rfs

    @property
    def controls(self) -> list[Control]:
        return self._controls

    @property
    def functional(self) -> EnsembleType:
        return self._functional

    @property
    def rfs(self) -> list[AbstractReducedFunctional]:
        return self._rfs

    @property
    def ensemble(self) -> Ensemble:
        return self._ensemble

    @no_annotations
    def __call__(self, values: EnsembleTypeList) -> EnsembleType:
        # 1. Update control tape values
        for c, v in zip(self.controls, Enlist(values)):
            c.update(v)

        # 2. Transform
        local_vals = self._global_to_local_data(values)
        local_Js = [
            rf(v)
            for rf, v in zip(self.rfs, local_vals)
        ]

        J = self.functional._ad_init_zero()
        self._local_to_global_data(local_Js, J)

        # 3. Update functional tape value
        self.functional.block_variable.checkpoint = J

        return J

    @no_annotations
    def tlm(self, m_dot: EnsembleTypeList) -> EnsembleType:
        # 1. Update control tape values
        for ci, mi in zip(self.controls, Enlist(m_dot)):
            ci.control.block_variable.reset_variables("tlm")
            ci.control.block_variable.tlm_value = mi

        # 2. Transform
        local_mdot = self._global_to_local_data(m_dot)
        local_tlm = [
            rf.tlm(md)
            for rf, md in zip(self.rfs, local_mdot)
        ]

        tlm = self.functional._ad_init_zero()
        self._local_to_global_data(local_tlm, tlm)

        # 3. Update functional tape value
        self.functional.block_variable.reset_variables("tlm")
        self.functional.block_variable.add_tlm_output(tlm)

        return tlm

    @no_annotations
    def derivative(self, adj_input: EnsembleType,
                   apply_riesz: bool = False) -> EnsembleTypeList:
        # 1. Update functional tape value
        self.functional.block_variable.reset_variables("adjoint")
        self.functional.block_variable.adj_value = adj_input

        # 2. Transform
        local_adj = self._global_to_local_data(adj_input)
        local_dJ = [
            rf.derivative(
                adj_input=adj[0], apply_riesz=apply_riesz)
            for rf, adj in zip(self.rfs, local_adj)
        ]

        dJ = self.controls.delist(
            [c.control._ad_init_zero(dual=not apply_riesz)
             for c in self.controls])

        self._local_to_global_data(local_dJ, dJ)

        # 3. Update control tape values
        for ci, dji in zip(self.controls, Enlist(dJ)):
            ci.control.block_variable.reset_variables("adjoint")
            ci.control.block_variable.add_adj_output(dji)

        return dJ

    @no_annotations
    def hessian(self, m_dot: EnsembleType,
                hessian_input: EnsembleTypeList,
                evaluate_tlm: bool = True,
                apply_riesz: bool = False) -> EnsembleTypeList:
        if evaluate_tlm:
            self.tlm(m_dot)

        # 1. Update functional tape value
        self.functional.block_variable.reset_variables("hessian")
        self.functional.block_variable.hessian_value = hessian_input

        # 2. Transform
        local_hin = self._global_to_local_data(hessian_input)
        local_hess = [
            rf.hessian(
                m_dot=None, evaluate_tlm=False,
                hessian_input=hin[0], apply_riesz=apply_riesz)
            for rf, hin in zip(self.rfs, local_hin)
        ]

        hessian = self.controls.delist(
            [c.control._ad_init_zero(dual=not apply_riesz)
             for c in self.controls])

        self._local_to_global_data(local_hess, hessian)

        # 3. Update control tape values
        for ci, hi in zip(self.controls, Enlist(hessian)):
            ci.control.block_variable.reset_variables("hessian")
            ci.control.block_variable.add_hessian_output(hi)

        return hessian

    def _local_to_global_data(self, local_data, global_data):
        # N local lists of length n -> n global lists of length N
        # [(1,), (2,), (3,)]->  [(1, 2, 3)]
        # [(1, 11), (2, 12), (3, 13)] -> [(1, 2, 3), (11, 12, 13)]

        for j, global_group in enumerate(Enlist(global_data)):
            local_group = [Enlist(local_group)[j]
                           for local_group in local_data]
            _set_local_subs(global_group, local_group)

        return global_data

    def _global_to_local_data(self, global_data):
        # n global lists of length N -> N local lists of length n
        # [(1, 2, 3)] -> [(1,), (2,), (3,)]
        # [(1, 2, 3), (11, 12, 13)] -> [(1, 11), (2, 12), (3, 13)]

        local_groups = [
            ld for ld in zip(*map(_local_subs, Enlist(global_data)))]
        return local_groups


class EnsembleShift(EnsembleCommunicationBase):
    def __init__(self, x: EnsembleType, ensemble: Ensemble):
        super().__init__(_ensemble(x))
        self._src = x._ad_init_zero()
        self._dst = x._ad_init_zero()

    @property
    def src(self) -> EnsembleType:
        return self._src

    @property
    def dst(self) -> EnsembleType:
        return self._dst

    def forward(self, x: EnsembleType) -> EnsembleType:
        return self._shift(x, 'forward')

    def backward(self, x: EnsembleType) -> EnsembleType:
        return self._shift(x, 'backward')

    def _shift(self, x: EnsembleType, direction: str) -> EnsembleType:
        xnew = x._ad_init_zero()
        xold = x
        xn = xnew.subfunctions
        xo = xold.subfunctions

        forward = direction == 'forward'

        rank = self.ensemble.ensemble_rank
        last_rank = self.ensemble.ensemble_size - 1

        # receive/send from next or previous rank?
        src = rank + (-1 if forward else 1)
        dst = rank + (1 if forward else -1)
        # receive into/send from first of last local function?
        send_idx = -1 if forward else 0
        recv_idx = 0 if forward else -1
        # who has to only send or only receive?
        first_rank = 0 if forward else last_rank
        last_rank = last_rank if forward else 0

        # deal with the local shuffle
        if forward:
            for i in range(1, len(xo)):
                xn[i].assign(xo[i-1])
        else:
            for i in range(len(xo)-1):
                xn[i].assign(xo[i+1])

        # send halo
        if rank != last_rank:
            self.ensemble.isend(
                xo[send_idx], dest=dst, tag=dst)

        # receive halo or blank out initial
        if rank == first_rank:  # blank out ics
            xn[recv_idx].assign(0)
        else:
            recv_reqs = self.ensemble.irecv(
                xn[recv_idx], source=src, tag=rank)
            MPI.Request.Waitall(recv_reqs)

        return xnew
