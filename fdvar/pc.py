import firedrake as fd
from firedrake.petsc import PETSc
from petsctools import (
    OptionsManager, flatten_parameters,
    attach_options, set_from_options, inserted_options)
from firedrake.adjoint import FourDVarReducedFunctional
from fdvar.mat import (
    EnsembleMatCtxBase,
    EnsembleBlockDiagonalMat,
    EnsembleBlockDiagonalMatCtx)
from pyadjoint.optimization.tao_solver import (
    ReducedFunctionalMat, TLMAction, AdjointAction)
from fdvar.correlations import CorrelationOperatorMat
from math import pi, sqrt

__all__ = (
    "EnsembleBJacobiPC",
    "WC4DVarSchurPC",
)


def get_default_options(default_prefix, custom_suffixes, options=PETSc.Options()):
    custom_prefixes = [default_prefix + str(suffix)
                       for suffix in custom_suffixes]
    default_options = {
        k.removeprefix(default_prefix): v
        for k, v in options.getAll().items()
        if (k.startswith(default_prefix)
        and not any(k.startswith(prefix) for prefix in custom_prefixes))
    }
    assert not any(k.startswith(str(suf)) for k in default_options.keys() for suf in custom_suffixes)
    return default_options


def obj_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


class PCBase:
    needs_python_amat = False
    needs_python_pmat = False

    def __init__(self):
        self.initialized = False

    def setUp(self, pc):
        if not self.initialized:
            self.initialize(pc)
            self.initialized = True
        self.update(pc)

    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        self.A, self.P = pc.getOperators()
        pcname = obj_name(self)
        if self.needs_python_amat:
            if self.A.getType() != "python":
                raise ValueError(
                    f"PC {pcname} needs a python type amat, not {self.A.getType()}")
            self.amat = self.A.getPythonContext()
        if self.needs_python_pmat:
            if self.P.getType() != "python":
                raise ValueError(
                    f"PC {pcname} needs a python type pmat, not {self.P.getType()}")
            self.pmat = self.P.getPythonContext()

        self.parent_prefix = pc.getOptionsPrefix()
        self.full_prefix = self.parent_prefix + self.prefix

    def update(self, pc):
        pass


class EnsemblePCBase(PCBase):
    needs_python_amat = True
    needs_python_pmat = True

    def initialize(self, pc):
        super().initialize(pc)

        if not isinstance(self.pmat, EnsembleMatCtxBase):
            pcname = obj_name(self)
            matname = obj_name(self.pmat)
            raise TypeError(
                f"PC {pname} needs an EnsembleMatCtxBase pmat, but it is a {matname}")

        self.ensemble = self.pmat.ensemble

        self.row_space = self.pmat.row_space.dual()
        self.col_space = self.pmat.col_space.dual()

        self.x = fd.EnsembleFunction(self.row_space)
        self.y = fd.EnsembleFunction(self.col_space)

    def apply(self, pc, x, y):
        with self.x.vec_wo() as v:
            x.copy(result=v)

        self.apply_impl(pc, self.x, self.y)

        with self.y.vec_ro() as v:
            v.copy(result=y)

    def apply_impl(self, pc, x, y):
        raise NotImplementedError


class EnsembleBJacobiPC(EnsemblePCBase):
    prefix = "ebjacobi_"

    def initialize(self, pc):
        super().initialize(pc)

        if not isinstance(self.pmat, EnsembleBlockDiagonalMatCtx):
            pcname = obj_name(self)
            matname = obj_name(pmat)
            raise TypeError(
                f"PC {pname} needs an EnsembleBlockDiagonalMatCtx pmat, but it is a {matname}")

        # default to behaving like a PC
        default_options = {'ksp_type': 'preonly'}

        default_sub_prefix = self.parent_prefix + "sub_"
        default_sub_options = get_default_options(
            default_sub_prefix, range(self.col_space.nglobal_spaces))
        default_options.update(default_sub_options)

        block_offset = self.ensemble.ensemble_comm.exscan(
            self.col_space.nlocal_spaces) or 0

        sub_ksps = []
        for i, sub_mat in enumerate(self.pmat.blocks):
            sub_ksp = PETSc.KSP().create(
                comm=self.ensemble.comm)
            sub_ksp.setOperators(sub_mat, sub_mat)

            sub_options = OptionsManager(
                parameters=default_options,
                options_prefix=default_sub_prefix + str(block_offset + i))
            sub_options.set_from_options(sub_ksp)

            sub_ksp.incrementTabLevel(1, parent=pc)
            sub_ksp.pc.incrementTabLevel(1, parent=pc)

            sub_ksps.append((sub_ksp, sub_options))

        self.sub_ksps = tuple(sub_ksps)

    def apply_impl(self, pc, x, y):
        sub_vecs = zip(self.x.subfunctions, self.y.subfunctions)
        for (sub_ksp, sub_options), (subx, suby) in zip(self.sub_ksps, sub_vecs):
            with subx.dat.vec_ro as rhs, suby.dat.vec_wo as sol:
                with sub_options.inserted_options():
                    sub_ksp.solve(rhs, sol)


class WC4DVarSchurPC(PCBase):
    prefix = "wcschur_"

    def initialize(self, pc):
        super().initialize(pc)
        if self.P.getType() == "python":
            Jhat = self.P.getPythonContext().rf
        else:
            Jhat = self.P.getAttr("Jhat")

        if not isinstance(Jhat, FourDVarReducedFunctional):
            raise TypeError(
                f"{obj_name(self)} expects a FourDVarReducedFunctional not a {obj_name(Jhat)}")
        self.Jhat = Jhat
        self.ensemble = Jhat.ensemble
        global_comm = self.ensemble.global_comm

        self.col_space = Jhat.control_space
        self.row_space = Jhat.control_space.dual()

        Lmat = ReducedFunctionalMat(
            Jhat.JL, action=TLMAction,
            comm=global_comm)

        LTmat = ReducedFunctionalMat(
            Jhat.JL, action=AdjointAction,
            comm=global_comm)

        BQmats = [
            CorrelationOperatorMat(
                rf.covariance, action='solve')
            for rf in Jhat.JD.rfs]

        Dmat = EnsembleBlockDiagonalMat(
            BQmats, col_space=Jhat.control_space,
            row_space=Jhat.control_space.dual())

        self.Lksp = PETSc.KSP().create(comm=global_comm)
        self.LTksp = PETSc.KSP().create(comm=global_comm)
        self.Dksp = PETSc.KSP().create(comm=global_comm)

        self.Lksp.setOperators(Lmat, Lmat)
        self.LTksp.setOperators(LTmat, LTmat)
        self.Dksp.setOperators(Dmat, Dmat)

        ncontrols = Jhat.control_space.nglobal_spaces

        L_params = {
            'ksp_convergence_test': 'skip',
            'ksp_converged_maxits': None,
            'ksp_type': 'richardson',
            'ksp_max_it': ncontrols,
        }
        D_params = {
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'fdvar.EnsembleBJacobiPC',
            'sub_pc_type': 'python',
            'sub_pc_python_type': 'fdvar.CorrelationOperatorPC',
        }

        default_l_prefix = self.full_prefix + "l_"
        default_l_options = get_default_options(
            default_prefix=default_l_prefix,
            custom_suffixes=("tlm", "adj"))
        L_params.update(flatten_parameters(default_l_options))

        attach_options(
            self.Lksp, parameters=L_params,
            options_prefix=self.full_prefix+"l_tlm")
        set_from_options(self.Lksp)

        attach_options(
            self.LTksp, parameters=L_params,
            options_prefix=self.full_prefix+"l_adj")
        set_from_options(self.LTksp)

        attach_options(
            self.Dksp, parameters=D_params,
            options_prefix=self.full_prefix+"d")
        set_from_options(self.Dksp)

        self.Lksp.incrementTabLevel(1, parent=pc)
        self.Lksp.pc.incrementTabLevel(1, parent=pc)

        self.LTksp.incrementTabLevel(1, parent=pc)
        self.LTksp.pc.incrementTabLevel(1, parent=pc)
        
        self.Dksp.incrementTabLevel(1, parent=pc)
        self.Dksp.pc.incrementTabLevel(1, parent=pc)

    def apply(self, pc, x, y):
        rhs = x.copy()
        sol = y.copy()

        with inserted_options(self.LTksp):
            self.LTksp.solve(rhs, sol)

        sol.copy(result=rhs)
        sol.zeroEntries()

        with inserted_options(self.Dksp):
            self.Dksp.solve(rhs, sol)

        sol.copy(result=rhs)
        sol.zeroEntries()

        with inserted_options(self.Lksp):
            self.Lksp.solve(rhs, sol)

        sol.copy(result=y)
