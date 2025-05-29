from firedrake import *
from firedrake.adjoint import *
import numpy as np
from math import pi as pi_m
np.random.seed(6)
Print = PETSc.Sys.Print

correlation_types = (
    "explicit-mass",
    "implicit-mass",
    "implicit-diffusion",
)


class CorrelationOperatorBase:
    """Correlation weighted norm x^{T}B^{-1}x
    """
    def norm(self, x):
        """Return x^{T}B^{-1}x
        """
        raise NotImplementedError

    def apply(self, x)
        """Return B^{-1}x
        """
        raise NotImplementedError

    def solve(self, x):
        """Return Bx
        """
        raise NotImplementedError


class ExplicitMassCorrelation:
    """Correlation operator is the action of a weighted mass matrix
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(M^{-1}Dx)|| with:
    D = <sigma*u, v*sigma>
    i.e.
    B^{-1} = (Minv*D)^{T}M(Minv*D) = D*Minv*D = D*Minv*D
    B = Dinv*M*Dinv
    """
    def __init__(self, V, sigma):
        self.sigma = sigma
        self.V = V

        u = TrialFunction(V)
        v = TestFunction(V)

        x = Function(V)
        y = Function(V.dual()
        self.x = x
        self.y = y

        self.Daction = inner(sigma*x, v*sigma)
        self.Dsolve = inner(sigma*u, v*sigma)

    def norm(self, x):
        """Return x^{T}B^{-1}x
        """
        self.x.assign(x)
        D_x = assemble(self.Daction)
        MinvD_x = Dx.riesz_representation()
        return assemble(inner(MinvDx, MinvDx)*dx)

    def solve(self, y):
        """Return By
        """
        # D^{-1}y
        self.y.assign(y)
        Dinv_y = Function(self.V)
        solve(self.Dsolve==self.y, Dinv_y)

        # M*D^{-1}y
        MDinv_y = Dinv_y.riesz_representation()

        # D^{-1}*M*D^{-1}
        self.y.assign(MDinv_y)
        DinvMDinv_y = Function(self.V)
        solve(self.Dsolve==self.y, DinvMDinv_y)

        return DinvMDinv_y


class ImplicitMassCorrelation:
    """Correlation operator is the inverse of a weighted mass matrix
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(D^{-1}Mx)|| with:
    D = <sigma*u, v*sigma>
    i.e.
    B^{-1} = (Dinv*M)^{T}M(Dinv*M) = M*Dinv*M*Dinv*M
    B = Minv*D*Minv*D*Minv
    """
    def __init__(self, V, sigma):
        pass


class ImplicitDiffusionCorrelation:
    """Correlation operator is the inverse of a weighted mass matrix
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(D^{-1}Mx)|| with:
    D = sigma^2*(M + kappa*<grad(u), grad(v)>)
    i.e.
    B^{-1} = (Dinv*M)^{T}M(Dinv*M) = M*Dinv*M*Dinv*M
    B = Minv*D*Minv*D*Minv
    """
    def __init__(self, V, sigma):
        pass


def diffusion_correlation(V, sigma, L, u=None):
    # number of diffusion steps
    M = 2

    nu = L*L/(2*M)
    lamda_g = sqrt(2*pi_m)*L

    nu_c = Constant(nu)
    # sig_lam = Constant(sigma/sqrt(lamda_g))
    sig_lam = Constant(1/sqrt(lamda_g))

    u = u or TrialFunction(V)
    v = TestFunction(V)

    cfl_nu = nu*nx*nx
    Print(f"{cfl_nu = }")
    Print(f"{sig_lam.values()[0] = }")

    D = sig_lam*(inner(u, v)*dx + inner(nu_c*grad(u), grad(v))*dx)

    return D


class BackgroundPC(PCBase):
    def initialize(self, pc):
        A, _ = pc.getOperators()
        appctx = A.getPythonContext().appctx

        V = appctx["control_space"]
        B2 = appctx["Bsqrt"]
        prefix = pc.getOptionsPrefix()

        action_type = PETSc.Options().getString(
            prefix+"pc_bkg_action_type", "explicit-mass")
        if action_type not in correlation_types:
            raise ValueError(
                "BackgroundPC 'pc_bkg_action_type' should be one of"
                + correlation_types.join(" or ") + f" not {action_type}")
        self.action_type = action_type

        self.x = Function(V.dual())
        self.y = Function(V)

        if action_type == "explicit-mass":
            u = TrialFunction(V)
            v = TestFunction(V)
            w = 1/B2
            M = inner(u*w, w*v)*dx

            self.solver = LinearVariationalSolver(
                LinearVariationalProblem(
                    M, self.x, self.y,
                    constant_jacobian=True),
                options_prefix=prefix+"bkg_")
            ksp = self.solver.snes.ksp
            ksp.incrementTabLevel(1, parent=pc)
            ksp.pc.incrementTabLevel(1, parent=pc)

        elif action_type == "implicit-mass":
            self.riesz_map = RieszMap(V)

            v = TestFunction(V)
            x_D = Function(V)
            self.Dx = inner(x_D*B2, v)*dx

            self.x_D = x_D

        elif action_type == "implicit-diffusion":
            L = appctx["L_B"]
            self.riesz_map = RieszMap(V)

            v = TestFunction(V)
            x_D = Function(V)
            self.Dx = diffusion_correlation(V, B2, L, u=x_D)

            self.x_D = x_D
    
    def apply(self, pc, x, y):
        with self.x.dat.vec_wo as xvec:
            x.copy(xvec)

        if self.action_type == "explicit-mass":
            self.solver.solve()

        elif self.action_type in ("implicit-mass", "implicit-diffusion"):
            Minv_x = self.riesz_map(self.x)

            self.x_D.assign(Minv_x)
            D_Minv_x = assemble(self.Dx)

            Minv_D_Minv_x = self.riesz_map(D_Minv_x)

            self.x_D.assign(Minv_D_Minv_x)
            D_Minv_D_Minv_x = assemble(self.Dx)

            Minv_D_Minv_D_Minv_x = self.riesz_map(D_Minv_D_Minv_x)

            self.y.assign(Minv_D_Minv_D_Minv_x)

        with self.y.dat.vec_wo as yvec:
            yvec.copy(y)

    def update(self, pc, x):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        viewer.printfASCII("PC built from the background covariance operator form\n")
        viewer.printfASCII(f"Form action type: {self.action_type}\n")
        if hasattr(self, "solver"):
            self.solver.snes.ksp.view(viewer)


def correlation_norm(x, sigma, L=None, action_type="explicit-mass"):
    if action_type == "explicit-mass":
        w = 1/sigma
        return assemble(inner(x*w, w*x)*dx)

    elif action_type == "implicit-mass":
        V = x.function_space()

        u = TrialFunction(V)
        v = TestFunction(V)

        D = inner(u*sigma, v)*dx
        Mx = inner(x, v)*dx
        DinvMx = Function(V)

        Dsolver = LinearVariationalSolver(
            LinearVariationalProblem(D, Mx, DinvMx))

        Dsolver.solve()
        return assemble(inner(DinvMx, DinvMx)*dx)

    elif action_type == "implicit-diffusion":
        assert L is not None
        V = x.function_space()

        D = diffusion_correlation(V, sigma, L)

        v = TestFunction(V)
        Mx = inner(x, v)*dx
        DinvMx = Function(V)

        Dsolver = LinearVariationalSolver(
            LinearVariationalProblem(D, Mx, DinvMx))

        Dsolver.solve()
        return assemble(inner(DinvMx, DinvMx)*dx)

    else:
        raise ValueError(
            "action_type should be one of "
            + correlation_types.join(" or ") + f" not {action_type}")


def propagate(stepper, un, un1, nt, ic, final=None):
    un.assign(ic)
    for _ in range(nt):
        un1.assign(un)
        stepper.solve()
        un.assign(un1)
    if final is None:
        final = Function(ic.function_space())
    final.assign(un)
    return final

nx = 50
nt = 25
T = 1.6
dt = T/nt
velocity = 1
cfl = velocity*dt*nx

re = 100
nu = Constant(1/re)

B = 0.01
R = 0.25

L_B = 0.1
L_R = 0.01
action_type = "implicit-mass"

Print(f"{cfl = :.3e}")

mesh = PeriodicUnitIntervalMesh(nx)
x, = SpatialCoordinate(mesh)
V = VectorFunctionSpace(mesh, "CG", 2)

def mass(u, v):
    return inner(u, v)*dx

def tendency(u, v):
    A = inner(dot(u, nabla_grad(u)), v)*dx
    D = inner(nu*grad(u), grad(v))*dx
    return A + D

# midpoint rule
un = Function(V, name="un")
un1 = Function(V, name="un1")
v = TestFunction(V)

uh = Constant(0.5)*(un + un1)
eqn = mass(un1 - un, v) + Constant(dt)*tendency(uh, v)

params = {
    "snes_rtol": 1e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
}

stepper = NonlinearVariationalSolver(
    NonlinearVariationalProblem(eqn, un1),
    solver_parameters=params, options_prefix="M")

ic = Function(V).interpolate(as_vector([1 + sin(2*pi*x)]))
bkg = Function(V).interpolate(as_vector([1 + sin(2*pi*x)]))
target = Function(V).interpolate(as_vector([1 + sin(2*pi*(x - T))]))

Bsqrt = Constant(sqrt(B))
R2 = Constant(sqrt(R))

ic.dat.data[:] += np.random.normal(
    0, B, ic.dat.data.shape)
bkg.dat.data[:] += np.random.normal(
    0, B, bkg.dat.data.shape)
target.dat.data[:] += np.random.normal(
    0, R, target.dat.data.shape)

continue_annotation()
with set_working_tape() as tape:
    J = correlation_norm(Function(V).project(ic - bkg),
                         Bsqrt, L=L_B, action_type=action_type)

    final = propagate(stepper, un, un1, nt, ic)
    J += correlation_norm(Function(V).project(final - target),
                          # R2, L=L_R, action_type=action_type)
                          R2, action_type="implicit-mass")

Jhat = ReducedFunctional(J, [Control(ic)], tape=tape)
pause_annotation()

appctx = {
    "control_space": V,
    "Bsqrt": Bsqrt,
    "L_B": L_B,
}
    
tao_params = {
    "tao_view": ":tao_view.log",
    "tao_monitor": None,
    "tao_converged_reason": None,
    "tao_gttol": 1e-2,
    "tao_type": "nls",
    "tao_nls": {
        "ksp_view": ":ksp_view.log",
        "ksp_monitor_short": None,
        "ksp_converged_rate": None,
        "ksp_converged_maxits": None,
        "ksp_max_it": 10,
        "ksp_rtol": 1e-2,
        "ksp_type": "cg",
        "pc_type": "python",
        "pc_python_type": f"{__name__}.BackgroundPC",
        "pc_bkg_action_type": action_type,
        "bkg": {
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    },
}
tao = TAOSolver(MinimizationProblem(Jhat),
                parameters=tao_params,
                appctx=appctx, options_prefix="")
ic_opt = tao.solve()[0]

un.assign(ic_opt)
for i in range(nt):
    un1.assign(un)
    stepper.solve()
    un.assign(un1)
final_opt = un.copy(deepcopy=True)

PETSc.Sys.Print(f"{errornorm(bkg, ic)           = :.3e}")
PETSc.Sys.Print(f"{errornorm(bkg, ic_opt)       = :.3e}")
PETSc.Sys.Print(f"{errornorm(target, final)     = :.3e}")
PETSc.Sys.Print(f"{errornorm(target, final_opt) = :.3e}")
