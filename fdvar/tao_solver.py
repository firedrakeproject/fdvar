import firedrake as fd
from firedrake import PETSc
from pyadjoint.optimization.tao_solver import (
    OptionsManager, TAOConvergenceError, _tao_reasons)
from functools import cached_property
from fdvar.mat import ReducedFunctionalMat

__all__ = ("TAOObjective", "TAOConvergenceError", "TAOSolver")


class TAOObjective:
    def __init__(self, Jhat, apply_riesz=False):
        self.Jhat = Jhat
        self.apply_riesz = apply_riesz
        self._control = Jhat.control.copy()
        self._m = Jhat.control.copy()
        self._mdot = Jhat.control.copy()

        self.n = self._m._vec.getLocalSize()
        self.N = self._m._vec.getSize()
        self.sizes = (self.n, self.N)

    def objective(self, tao, x):
        with self._control.vec_wo() as cvec:
            x.copy(cvec)
        return self.Jhat(self._control)

    def gradient(self, tao, x, g):
        dJ = self.Jhat.derivative()
        with dJ.vec_ro() as dvec:
            dvec.copy(g)

    def objective_gradient(self, tao, x, g):
        with self._control.vec_wo() as cvec:
            x.copy(cvec)
        J = self.Jhat(self._control)
        dJ = self.Jhat.derivative()
        with dJ.vec_ro() as dvec:
            dvec.copy(g)
        return J

    def hessian(self, A, x, y):
        with self._mdot.vec_ro() as mvec:
            x.copy(mvec)
        ddJ = self.Jhat.hessian()
        with ddJ.vec_ro() as dvec:
            dvec.copy(y)
        if self._shift != 0.0:
            y.axpy(self._shift, x)

    @cached_property
    def hessian_mat(self):
        return ReducedFunctionalMat(self.Jhat)

    @cached_property
    def gradient_norm_mat(self):
        ctx = GradientNormCtx(self.Jhat)
        mat = PETSc.Mat().createPython(
            (self.sizes, self.sizes), ctx,
            comm=self.Jhat.ensemble.global_comm)
        mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        mat.setUp()
        mat.assemble()
        return mat


class GradientNormCtx:
    def __init__(self, Jhat):
        self._xfunc = Jhat.control.copy()
        self._ycofunc = self._xfunc.riesz_representation()

        # TODO: Just implement EnsembleFunction._ad_convert_type
        efs = Jhat.control_space
        v = fd.TestFunction(efs._full_local_space)
        self.M = fd.inner(v, self._xfunc._full_local_function)*fd.dx

    def mult(self, mat, x, y):
        with self._xfunc.vec_wo() as xvec:
            x.copy(xvec)

        fd.assemble(self.M, tensor=self._ycofunc._full_local_function)

        # ycofunc = self.Jhat.control._ad_convert_riesz(
        #     self._xfunc, riesz_map=self.Jhat.control.riesz_map)

        with self._ycofunc.vec_ro() as yvec:
            yvec.copy(y)


class TAOSolver:
    def __init__(self, Jhat, *, options_prefix=None,
                 solver_parameters=None):
        self.Jhat = Jhat
        self.ensemble = Jhat.ensemble

        self.tao_objective = TAOObjective(Jhat)

        self.tao = PETSc.TAO().create(
            comm=Jhat.ensemble.global_comm)

        # solution vector
        self._x = Jhat.control._vec.duplicate()
        self.tao.setSolution(self._x)

        # evaluate objective and gradient
        self.tao.setObjective(
            self.tao_objective.objective)

        self.tao.setGradient(
            self.tao_objective.gradient)

        self.tao.setObjectiveGradient(
            self.tao_objective.objective_gradient)

        # evaluate hessian action
        hessian_mat = self.tao_objective.hessian_mat
        self.tao.setHessian(
            hessian_mat.getPythonContext().update,
            hessian_mat)

        # gradient norm in correct space
        self.tao.setGradientNorm(
            self.tao_objective.gradient_norm_mat)

        # solver parameters and finish setup
        self.options = OptionsManager(
            solver_parameters, options_prefix)
        self.options.set_from_options(self.tao)

        self.tao.setUp()

    def solve(self):
        control = self.Jhat.control
        with control.tape_value().vec_ro() as cvec:
            cvec.copy(self._x)

        with self.options.inserted_options():
            self.tao.solve()

        if self.tao.getConvergedReason() <= 0:
            # Using the same format as Firedrake linear solver errors
            raise TAOConvergenceError(
                f"TAOSolver failed to converge after {self.tao.getIterationNumber()} iterations "
                f"with reason: {_tao_reasons[self.tao.getConvergedReason()]}")

        with control.vec_wo() as cvec:
            self._x.copy(cvec)
