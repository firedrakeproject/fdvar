from firedrake import *
from pyadjoint.optimization.tao_solver import get_valid_comm

__all__ = (
    "CorrelationOperatorBase",
    "ExplicitFormCorrelationBase",
    "ImplicitFormCorrelationBase",
    "ExplicitMassCorrelation",
    "ImplicitMassCorrelation",
    "ExplicitDiffusionCorrelation",
    "ImplicitDiffusionCorrelation",
    "CorrelationOperatorPC",
    "CorrelationOperatorMat",
)


def _make_rhs(b):
    if isinstance(b, Function):
        v = TestFunction(b.function_space())
        return inner(b, v)*dx
    elif isinstance(b, Cofunction):
        v = TestFunction(b.function_space().dual())
        return inner(b.riesz_representation(), v)*dx
    else:
        return b


class CorrelationOperatorBase:
    """Correlation weighted norm x^{T}B^{-1}x
    B: V* -> V
    B^{-1}: V -> V*
    """
    def __init__(self, V):
        self.V = V

    def norm(self, x):
        """Return x^{T}B^{-1}x

        Inheriting classes may provide more efficient specialisations.
        """
        return self.solve(x)(x)

    def apply(self, y, x=None):
        """Return x = By
        B: V* -> V
        """
        raise NotImplementedError

    def solve(self, x, y=None):
        """Return y = B^{-1}x
        B^{-1}: V -> V*
        """
        raise NotImplementedError


class FormCorrelationOperatorBase(CorrelationOperatorBase):
    """Correlation operator is the action or inverse of a finite element form m times.
    x^{T}B^{-1}x = ||x||_{B^{-1}}
    B: V* -> V
    B^{-1}: V -> V*
    """
    def __init__(self, V, m=2, solver_parameters=None, bcs=None):
        super().__init__(V)

        self.m = m
        self.bcs = bcs or []
        self.solver_parameters = solver_parameters or {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }

        _x = Function(V)
        _rhs = Function(V)
        self._xaction = _x
        self._rhs = _rhs

        u = TrialFunction(V)
        v = TestFunction(V)

        self.Maction = inner(_x, v)*dx
        self.Msolve = inner(u, v)*dx

        self.Gaction = self.form(_x, v)
        self.Gsolve = self.form(u, v)

        self.rhs = inner(_rhs, v)*dx
        self.solve_eqn = self.Gsolve==self.rhs

    def _form_action(self, x, b=None):
        """Return b = Gx
        G: V -> V*
        """
        b = b or Function(self.V.dual())
        self._xaction.assign(x)
        return assemble(self.Gaction, tensor=b)

    def _form_solve(self, b, x=None):
        """Return x = G^{-1}b
        G^{-1}: V* -> V
        """
        x = x or Function(self.V)
        if isinstance(b, Cofunction):
            b = self.riesz(b)
        self._rhs.assign(b)
        solve(self.solve_eqn, x,
              solver_parameters=self.solver_parameters)
        return x

    def riesz(self, x):
        return x.riesz_representation()

    def form(self, u, v):
        """Return the form defining the correlation operator.

        Inheriting classes must implement this.
        """
        raise NotImplementedError


class ExplicitFormCorrelationBase(FormCorrelationOperatorBase):
    """Correlation operator is the action of a finite element form m times.
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||((G^{-1}M)^(m/2)x)||
    M: V -> V* = <u,v>
    G: V -> V*
    B: V* -> V = Minv*G*Minv*G*Minv (if m=2)
    B^{-1}: V -> V* = (Ginv*M)^{T}M(Ginv*M) = M*Ginv*M*Ginv*M (if m=2)
    """

    def apply(self, y, x=None):
        """Return x = By
        B: V* -> V
        """
        x = x or Function(self.V)
        dual = y
        for _ in range(self.m):
            primal = self.riesz(dual)
            dual = self._form_action(primal)
        return x.assign(self.riesz(dual))

    def solve(self, x, y=None):
        """Return y = B^{-1}x
        B^{-1}: V -> V*
        """
        y = y or Function(self.V.dual())
        primal = x
        for _ in range(self.m):
            primal = self._form_solve(primal)
        return y.assign(self.riesz(primal))

    def norm(self, x):
        """Return x^{T}B^{-1}x
        """
        primal = x
        for _ in range(self.m//2):
            primal = self._form_solve(primal)
        return assemble(inner(primal, primal)*dx)


class ImplicitFormCorrelationBase(CorrelationOperatorBase):
    """Correlation operator is the inverse of a finite element form
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(M^{-1}Gx)|| with:
    G = <sigma*u, v*sigma>
    i.e.
    B^{-1} = (Minv*G)^{T}M(Minv*G) = (G*Minv)*M*(Minv*G)
    B = Ginv*M*Ginv
    """
    def apply(self, y, x=None):
        """Return x = By
        B: V* -> V
        """
        x = x or Function(self.V)
        dual = y
        for _ in range(self.m):
            primal = self._form_solve(dual)
            dual = self.riesz(primal)
        return x.assign(self.riesz(dual))

    def _solve_action(rhs):
        sol = Function(self.V)
        self._xaction.assign(rhs)
        solve(self.Msolve==self.Gaction, sol,
              solver_parameters=solver_parameters)
        return sol

    def solve(self, x, y=None):
        """Return y = B^{-1}x
        B^{-1}: V -> V*
        """
        y = y or Function(self.V.dual())
        primal = x
        for _ in range(self.m):
            primal = self._solve_action(primal)
        return y.assign(self.riesz(primal))

    def norm(self, x):
        """Return x^{T}B^{-1}x
        """
        primal = x
        for _ in range(self.m//2):
            primal = self._solve_action(primal)
        return assemble(inner(primal, primal)*dx)

    def form(self, u, v):
        raise NotImplementedError


class ExplicitMassCorrelation(ExplicitFormCorrelationBase):
    """Correlation operator is the action of a weighted mass matrix
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(G^{-1}Mx)|| with:
    G = sigma^2*(<u, v>)^2
    i.e.
    B^{-1} = (Ginv*M)^{T}M(Ginv*M) = M*Ginv*M*Ginv*M
    B = Minv*G*Minv*G*Minv
    """
    def __init__(self, V, sigma, m=2, solver_parameters=None, bcs=None):
        self.sigma = sigma
        super().__init__(V, m=m, solver_parameters=solver_parameters, bcs=bcs)

    def form(self, u, v):
        weight = self.sigma**(2/self.m)
        return inner(weight*u, v)*dx


class ImplicitMassCorrelation(ImplicitFormCorrelationBase):
    """Correlation operator is the inverse of a finite element form
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(M^{-1}Gx)|| with:
    G = sigma^2*(<u, v)^{-2}
    i.e.
    B^{-1} = (Minv*G)^{T}M(Minv*G) = (G*Minv)*M*(Minv*G)
    B = Ginv*M*Ginv
    """
    def __init__(self, V, sigma, m=2, solver_parameters=None, bcs=None):
        self.sigma = sigma
        super().__init__(V, m=m, solver_parameters=solver_parameters, bcs=bcs)

    def form(self, u, v):
        weight = self.sigma**(2/self.m)
        return inner((1/weight)*u, v)*dx


class ExplicitDiffusionCorrelation(ExplicitFormCorrelationBase):
    """Correlation operator is the action of a diffusion operator
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(G^{-1}Mx)|| with:
    G = scale*sigma^2*(<u, v> - <kappa*grad(u), grad(v)>)^2
    i.e.
    B^{-1} = (Ginv*M)^{T}M(Ginv*M) = M*Ginv*M*Ginv*M
    B = Minv*G*Minv*G*Minv
    """
    def __init__(self, V, sigma, L, m=2, solver_parameters=None, bcs=None):
        self.sigma = sigma
        self.L = L

        kappa = Constant(L*L/(2*m))
        lambda_g = Constant(sqrt(2*pi)*L)
        lg_sqrt = Constant(sqrt(lambda_g))
        scale = Constant((sigma*lg_sqrt)**(2/m))

        self.kappa = kappa
        self.lambda_g = lambda_g
        self.scale = scale

        nx = V.mesh().num_cells()

        cfl_nu = float(kappa*nx*nx)
        PETSc.Sys.Print(f"{float(kappa) = :.3e} | {cfl_nu = :.3e} | {float(scale) = :.3e}")

        super().__init__(V, m=m, solver_parameters=solver_parameters, bcs=bcs)

    def form(self, u, v):
        kappa, scale = self.kappa, self.scale
        return scale*(inner(u, v)*dx - inner(kappa*grad(u), grad(v))*dx)


class ImplicitDiffusionCorrelation(ImplicitFormCorrelationBase):
    """Correlation operator is the inverse of a diffusion operator
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(M^{-1}Gx)|| with:
    G = scale*sigma^2*(<u, v> + <kappa*grad(u), grad(v)>)^{-2}
    i.e.
    B^{-1} = (Minv*G)^{T}M(Minv*G) = (G*Minv)*M*(Minv*G)
    B = Ginv*M*Ginv
    """
    def __init__(self, V, sigma, L, m=2, solver_parameters=None, bcs=None):
        self.sigma = sigma
        self.L = L

        kappa = Constant(L*L/(2*m))
        lambda_g = Constant(sqrt(2*pi)*L)
        lg_sqrt = Constant(sqrt(lambda_g))
        scale = Constant(1/((sigma*lg_sqrt)**(2/m)))

        self.kappa = kappa
        self.lambda_g = lambda_g
        self.scale = scale

        nx = V.mesh().num_cells()

        cfl_nu = float(kappa*nx*nx)
        PETSc.Sys.Print(f"{float(kappa) = :.3e} | {cfl_nu = :.3e} | {float(scale) = :.3e}")

        super().__init__(V, m=m, solver_parameters=solver_parameters, bcs=bcs)

    def form(self, u, v):
        kappa, scale = self.kappa, self.scale
        return scale*(inner(u, v)*dx + inner(kappa*grad(u), grad(v))*dx)


class CorrelationOperatorPC:
    """
    Precondition the inverse correlation operator:
    P = B : V* -> V
    """
    def __init__(self):
        self.initialized = False

    def setUp(self, pc):
        if self.initialized:
            self.update(pc)
        else:
            self.initialize(pc)
            self.initialized = True

    def initialize(self, pc):
        _, P = pc.getOperators()
        correlation_mat = P.getPythonContext()
        if not isinstance(correlation_mat, CorrelationOperatorMatCtx):
            raise TypeError(
                "CorrelationOperatorPC needs a CorrelationOperatorMatCtx")
        correlation = correlation_mat.correlation

        self.correlation = correlation
        self.correlation_mat = correlation_mat

        V = correlation.V
        primal = Function(V)
        dual = Function(V.dual())

        # PC does the opposite of the Mat
        if correlation_mat.action == 'apply':
            self.x = primal
            self.y = dual
            self._apply_op = correlation.solve
        elif correlation_mat.action == 'solve':
            self.x = dual
            self.y = primal
            self._apply_op = correlation.apply

        self.update(pc)

    def apply(self, pc, x, y):
        with self.x.dat.vec_wo as xvec:
            x.copy(result=xvec)

        self._apply_op(self.x, self.y)

        with self.y.dat.vec_ro as yvec:
            yvec.copy(result=y)

    def update(self, pc):
        pass


class CorrelationOperatorMatCtx:
    def __init__(self, correlation, action='solve'):
        self.comm = correlation.V.mesh().comm
        self.correlation = correlation
        self.action = action
        self.V = correlation.V

        primal = Function(self.V)
        dual = Function(self.V.dual())

        if action == 'apply':
            self.x = dual
            self.y = primal
            self._mult_op = correlation.apply
        elif action == 'solve':
            self.x = primal
            self.y = dual
            self._mult_op = correlation.solve
        else:
            raise ValueError(
                f"CorrelationOperatorMatCtx action must be 'solve' or 'apply', not {action}.")

    def mult(self, A, x, y):
        with self.x.dat.vec_wo as v:
            x.copy(result=v)

        self._mult_op(self.x, self.y)

        with self.y.dat.vec_ro as v:
            v.copy(result=y)


def CorrelationOperatorMat(correlation, action='solve'):
    ctx = CorrelationOperatorMatCtx(
        correlation, action=action)

    sizes = correlation.V.dof_dset.layout_vec.getSizes()

    mat = PETSc.Mat().createPython(
        (sizes, sizes), ctx, comm=ctx.comm)

    mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    mat.setUp()
    mat.assemble()
    return mat
