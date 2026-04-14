from typing import Sequence
from pyadjoint.reduced_functional import AbstractReducedFunctional
from pyadjoint.enlisting import Enlist
from pyadjoint import (
    Control,
    AdjFloat,
    OverloadedType,
    no_annotations,
)

from firedrake import (
    Function,
    Cofunction,
    Ensemble,
    EnsembleFunction,
    EnsembleCofunction,
    EnsembleFunctionSpace,
)

from fdvar.adjoint.ensemble_adjvec import EnsembleAdjVec


LocalType = Function | Cofunction | AdjFloat
EnsembleType = EnsembleFunction | EnsembleCofunction | EnsembleAdjVec

LocalTypeList = LocalType | Sequence[LocalType]
EnsembleTypeList = EnsembleType | Sequence[EnsembleType]


def _local_subs(val):
    if isinstance(val, EnsembleFunctionBase):
        return val.subfunctions
    elif isinstance(val, EnsembleAdjVec):
        return val.subvec
    elif isinstance(val, (list, tuple)):
        return val
    else:
        raise TypeError(
            f"Cannot use {type(val).__name__} as an ensemble overloaded type.")


def _local_size(val):
    return len(_local_subs(val))


def _set_local_subs(dst, src):
    assert _local_size(dst) == _local_size(src)
    dst_subs = _local_subs(dst)
    for i, s in enumerate(src):
        if hasattr(dst_subs[i], 'assign'):
            dst_subs[i].assign(s)
        else:
            dst_subs[i] = s
    return dst


def reduction(ensemble: Ensemble,
              src: EnsembleType) -> LocalType:
    vals = _local_subs(src)
    local_sum = vals[0]._ad_init_zero()
    for v in vals:
        local_sum = local_sum._ad_add(v)
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


class EnsembleTransform(AbstractReducedFunctional):
    def __init__(self, functional: EnsembleType,
                 controls: Control | Sequence[Control],
                 rfs: list[AbstractReducedFunctional]):
        pass

    @property
    def controls(self) -> list[Control]:
        return self._controls

    @no_annotations
    def __call__(self, values: EnsembleTypeList) -> EnsembleType:
        pass

    @no_annotations
    def tlm(self, m_dot: EnsembleTypeList) -> EnsembleType:
        pass

    @no_annotations
    def derivative(self, adj_input: EnsembleType,
                   apply_riesz: bool = False) -> EnsembleTypeList:
        pass

    @no_annotations
    def hessian(self, m_dot: EnsembleType,
                hessian_input: EnsembleTypeList,
                evaluate_tlm: bool = True,
                apply_riesz: bool = False) -> EnsembleTypeList:
        pass


class EnsembleReduce(AbstractReducedFunctional):
    def __init__(self, src: EnsembleType,
                 ensemble: Ensemble,
                 root: int | None = None):
        # TODO: some type validation

        self._ensemble = ensemble
        self._src = src
        self._dst = _local_subs(src)[0]._ad_init_zero()
        self._root = root

    @cached_property
    def controls(self) -> list[Control]:
        return Enlist(Control(self.src))

    @property
    def functional(self) -> LocalType:
        return self.dst

    @property
    def src(self) -> EnsembleType:
        return self._src

    @property
    def dst(self) -> EnsembleType:
        return self._dst

    @property
    def ensemble(self) -> Ensemble:
        return self._ensemble

    @property
    def root(self) -> int | None:
        return self._root

    @no_annotations
    def __call__(self, values: EnsembleType) -> LocalType:
        values = self._delist(values)

        self.controls[0].update(values)
        out = reduction(values)
        self.dst.block_variable.checkpoint = out

        return out

    @no_annotations
    def tlm(self, m_dot: EnsembleType) -> LocalType:
        m_dot = self._delist(m_dot)

        src_bv = self.src.block_variable
        dst_bv = self.dst.block_variable

        src_bv.reset_variables("tlm")
        dst_bv.reset_variables("tlm")

        src_bv.tlm_value = m_dot
        out = reduction(values)
        dst_bv.add_tlm_output(out)

        return out

    @no_annotations
    def derivative(self, adj_input: LocalType,
                   apply_riesz: bool = False) -> EnsembleType:

        src_bv = self.src.block_variable
        dst_bv = self.dst.block_variable

        src_bv.reset_variables("adjoint")
        dst_bv.reset_variables("adjoint")

        dst_bv.adj_value = adj_input
        out = self.src._ad_init_zero(dual=True)
        out = broadcast(src=adj_input, dst=out, root=self.root)
        src_bv.add_adj_output(out)

        return self.control[0].get_derivative(apply_riesz=apply_riesz)

    @no_annotations
    def hessian(self, m_dot: EnsembleType,
                hessian_input: LocalType | None = None,
                evaluate_tlm: bool = True,
                apply_riesz: bool = False) -> EnsembleType:
        if evaluate_tlm:
            self.tlm(m_dot)

        src_bv = self.src.block_variable
        dst_bv = self.dst.block_variable

        src_bv.reset_variables("hessian")
        dst_bv.reset_variables("hessian")

        dst_bv.hessian_value = hessian_input
        out = self.src._ad_init_zero(dual=True)
        out = broadcast(src=hessian_input, dst=out, root=self.root)
        src_bv.add_hessian_output(out)

        return self.control[0].get_hessian(apply_riesz=apply_riesz)

    def _delist(self, v):
        if isinstance(v, list) and len(v) != 1:
            raise ValueError(
                f"{type(self).__name__} only has"
                f" one control, not {len(v)}")
            return v[0]
        else:
            return v


class EnsembleBcast(AbstractReducedFunctional):
    def __init__(self, dst: EnsembleType,
                 ensemble: Ensemble,
                 root: int | None = None):
        # TODO: some type validation

        self._ensemble = ensemble
        self._dst = dst
        self._src = _local_subs(dst)[0]._ad_init_zero()
        self._root = root

    @cached_property
    def controls(self) -> list[Control]:
        return Enlist(Control(self.src))

    @property
    def functional(self) -> LocalType:
        return self.dst

    @property
    def src(self) -> EnsembleType:
        return self._src

    @property
    def dst(self) -> EnsembleType:
        return self._dst

    @no_annotations
    def __call__(self, values: LocalType) -> EnsembleType:
        values = self._delist(values)

        self.controls[0].update(values)
        out = broadcast(values)
        self.dst.block_variable.checkpoint = out

        return out

    @no_annotations
    def tlm(self, m_dot: LocalType) -> EnsembleType:
        m_dot = self._delist(m_dot)

        src_bv = self.src.block_variable
        dst_bv = self.dst.block_variable

        src_bv.reset_variables("tlm")
        dst_bv.reset_variables("tlm")

        src_bv.tlm_value = m_dot
        out = reduction(values)
        dst_bv.add_tlm_output(out)

        return out

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        pass

    @no_annotations
    def hessian(self, m_dot, hessian_input=None,
                evaluate_tlm=True, apply_riesz=False):
        pass

    def _delist(self, v):
        if isinstance(v, list) and len(v) != 1:
            raise ValueError(
                f"{type(self).__name__} only has"
                f" one control, not {len(v)}")
            return v[0]
        else:
            return v
