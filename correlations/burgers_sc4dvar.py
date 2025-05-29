from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import interpolate
from fdvar import generate_observation_data
np.random.seed(42)
Print = PETSc.Sys.Print


def _make_rhs(b):
    if isinstance(b, Function):
        v = TestFunction(b.function_space())
        return inner(b, v)*dx
    elif isinstance(b, Cofunction):
        v = TestFunction(b.function_space())
        return inner(b.riesz_representation(), v)*dx
    else:
        return b


class CorrelationOperatorBase:
    """Correlation weighted norm x^{T}B^{-1}x
    """
    def norm(self, x):
        """Return x^{T}B^{-1}x
        """
        raise NotImplementedError

    def apply(self, x, y):
        """Return y = B^{-1}x
        """
        raise NotImplementedError

    def solve(self, y, x):
        """Return x = By
        """
        raise NotImplementedError


class ExplicitMassCorrelation(CorrelationOperatorBase):
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
        y = Function(V.dual())
        self.x = x
        self.y = y

        self.Daction = inner(sigma*x, v*sigma)*dx
        self.Dsolve = inner(sigma*u, v*sigma)*dx

    def norm(self, x):
        """Return x^{T}B^{-1}x
        """
        self.x.assign(x)
        D_x = assemble(self.Daction)
        MinvD_x = D_x.riesz_representation()
        return assemble(inner(MinvD_x, MinvD_x)*dx)
        # return assemble(self.Daction(x))

    def solve(self, y, x):
        """Return x = By
        """
        # D^{-1}y
        Dinv_y = x
        solve(self.Dsolve==_make_rhs(y), Dinv_y)

        # M*D^{-1}y
        MDinv_y = Dinv_y.riesz_representation()

        # D^{-1}*M*D^{-1}
        DinvMDinv_y = x
        solve(self.Dsolve==_make_rhs(MDinv_y), DinvMDinv_y)

        return DinvMDinv_y


class ImplicitMassCorrelation(CorrelationOperatorBase):
    """Correlation operator is the inverse of a weighted mass matrix
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(D^{-1}Mx)|| with:
    D = <sigma*u, v*sigma>
    i.e.
    B^{-1} = (Dinv*M)^{T}M(Dinv*M) = M*Dinv*M*Dinv*M
    B = Minv*D*Minv*D*Minv
    """
    def __init__(self, V, sigma):
        self.sigma = sigma
        self.V = V

        u = TrialFunction(V)
        v = TestFunction(V)

        x = Function(V)
        y = Function(V.dual())
        self.x = x
        self.y = y

        D = lambda u, v: inner(sigma*u, v*sigma)*dx
        self.Daction = D(x, v)
        self.Dsolve = D(u, v)
        self.rhs = inner(x, TestFunction(V))*dx

    def norm(self, x):
        """Return x^{T}B^{-1}x
        """
        self.x.assign(x)
        DinvM_x = Function(self.V)
        solve(self.Dsolve==self.rhs, DinvM_x)
        return assemble(inner(DinvM_x, DinvM_x)*dx)

    def solve(self, y, x):
        """Return x = By
        """
        Minv_y = self.x
        Minv_y.assign(y.riesz_representation())

        DMvinv_y = y
        DMinv_y = assemble(self.Daction, tensor=y)

        MinvDMinv_y = self.x
        MinvDMinv_y.assign(DMinv_y.riesz_representation())

        DMinvDMvinv_y = y
        DMinvDMinv_y = assemble(self.Daction, tensor=y)

        MinvDMinvDMinv_y = self.x
        MinvDMinvDMinv_y.assign(DMinv_y.riesz_representation())

        return x.assign(self.x)


class ImplicitDiffusionCorrelation(CorrelationOperatorBase):
    """Correlation operator is the inverse of a diffusion operator
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(D^{-1}Mx)|| with:
    D = <u, v>
    i.e.
    B^{-1} = (Dinv*M)^{T}M(Dinv*M) = M*Dinv*M*Dinv*M
    B = Minv*D*Minv*D*Minv
    """
    def __init__(self, V, sigma, L):
        self.sigma = sigma
        self.V = V
        self.L = L

        M = 2
        nu = Constant(L*L/(2*M))
        lambda_g = Constant(sqrt(2*pi)*L)
        lg_sqrt = Constant(sqrt(lambda_g))

        self.nu = nu
        self.lambda_g = lambda_g

        u = TrialFunction(V)
        v = TestFunction(V)

        x = Function(V)
        y = Function(V.dual())
        self.x = x
        self.y = y

        cfl_nu = float(nu*nx*nx)
        Print(f"{nu.values()[0] = :.4e} | {cfl_nu = :.4e}")

        D = lambda u, v: inner(u, v)*dx + inner(nu*grad(u), grad(v))*dx

        rescale = Constant(sigma/lg_sqrt)
        Print(f"{rescale.values()[0] = :.4e}")

        self.Daction = rescale*D(x, v)
        self.Dsolve = rescale*D(u, v)

        self.rhs = inner(x, TestFunction(V))*dx

    def norm(self, x):
        """Return x^{T}B^{-1}x
        """
        self.x.assign(x)
        DinvM_x = Function(self.V)
        solve(self.Dsolve==self.rhs, DinvM_x)
        return assemble(inner(DinvM_x, DinvM_x)*dx)

    def solve(self, y, x):
        """Return x = By
        """
        Minv_y = self.x
        Minv_y.assign(y.riesz_representation())

        DMvinv_y = y
        DMinv_y = assemble(self.Daction, tensor=y)

        MinvDMinv_y = self.x
        MinvDMinv_y.assign(DMinv_y.riesz_representation())

        DMinvDMvinv_y = y
        DMinvDMinv_y = assemble(self.Daction, tensor=y)

        MinvDMinvDMinv_y = self.x
        MinvDMinvDMinv_y.assign(DMinv_y.riesz_representation())

        return x.assign(self.x)


class CorrelationOperatorPC:
    def __init__(self):
        self.initialized = False

    def setUp(self, pc):
        if self.initialized:
            self.update(pc)
        else:
            self.initialize(pc)
            self.initialized = True

    def initialize(self, pc):
        A, _ = pc.getOperators()
        appctx = A.getPythonContext().appctx
        self.correlation = appctx["correlation"]
        V = self.correlation.V
        self.x = Function(V.dual())
        self.y = Function(V)
        self.update(pc)

    def apply(self, pc, x, y):
        with self.x.dat.vec_wo as xvec:
            x.copy(result=xvec)

        self.correlation.solve(self.x, self.y)

        with self.y.dat.vec_wo as yvec:
            yvec.copy(result=y)

    def update(self, pc):
        pass


# number of observation windows, and steps per window
nw, nt, dt, nu, nx = 20, 5, 1e-2, 1/40, 40

cfl = dt*nx
Print(f"{cfl = }")

# Covariance of background, observation, and model noise
sigma_b = sqrt(1e-2)
sigma_r = sqrt(1e-3)
sigma_q = sqrt(1e-4*(nt*dt))

L_b = 0.25
lim_b = 0.99

def rand_func(V, sigma=0.0, lim=0.5, dist="symmetric"):
    if dist == "symmetric":
        lower = (1 - lim)*sigma
        upper = (1 + lim)*sigma
    elif dist == "normalised":
        lower = lim
        upper = 1
    else:
        raise ValueError(f"Unrecognised {dist = }")
    x = Function(V)
    sample = np.random.random_sample(x.dat.data.shape)
    x.dat.data[:] = lower + (upper - lower)*sample
    return x

# 1D periodic mesh
mesh = PeriodicUnitIntervalMesh(nx)
x, = SpatialCoordinate(mesh)

# Burger's equation with implicit midpoint integration
V = FunctionSpace(mesh, "CG", 2)
R = FunctionSpace(mesh, "R", 0)

t = Function(R).zero()

un, un1 = Function(V), Function(V)
v = TestFunction(V)
uh = (un + un1)/2

# finite element forms
dtc = Constant(dt)
nuc = Constant(nu)
F = (inner(un1 - un, v)*dx
     + dtc*inner(uh, uh.dx(0))*v*dx
     + dtc*inner(nuc*grad(uh), grad(v))*dx
)

params = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_rtol": 1e-12,
    "ksp_type": "preonly",
    "pc_type": "lu",
}

# timestepper solver
# stepper = NonlinearVariationalSolver(
#     NonlinearVariationalProblem(F, un1),
#     solver_parameters=params)

def solve_step():
    un1.assign(un)
    # stepper.solve()
    solve(F==0, un1, solver_parameters=params)
    un.assign(un1)

# "ground truth" reference solution
reference_ic = Function(V).project(1 + 0.5*sin(2*pi*x))

# observations are point evaluations at random locations
vom = VertexOnlyMesh(mesh, np.random.rand(20, 1))
Y = FunctionSpace(vom, "DG", 0)

def H(x):  # operator to take observations
    return assemble(interpolate(x, Y))

# generate "ground-truth" observational data
y, background, target_end = generate_observation_data(
    None, reference_ic, solve_step, un, un1, None,
    H, nw, nt, sigma_b, sigma_r, sigma_q)

# create function evaluating observation error at window i
def observation_error(i):
    return lambda x: Function(Y).assign(H(x) - y[i])

# create distributed control variable for entire timeseries
control = Function(V).assign(background)

# random standard deviation functions
B = rand_func(V, sigma=sigma_b, lim=lim_b, dist="symmetric")
R = rand_func(Y, lim=sigma_r, dist="normalised")

Bop = ImplicitMassCorrelation(V, B)
# Bop = ImplicitDiffusionCorrelation(V, sigma_b, L_b)
Rop = ImplicitMassCorrelation(Y, R)

# tell pyadjoint to start taping operations
continue_annotation()

# This object will construct and solve the 4DVar system
Jhat = FourDVarReducedFunctional(
    Control(control),
    background=background,
    background_covariance=Bop,
    observation_covariance=Rop,
    observation_error=observation_error(0),
    weak_constraint=False)

# loop over each observation stage on the local communicator
with Jhat.recording_stages(nstages=nw) as stages:
    for stage, ctx in stages:
        idx = stage.local_index
        un.assign(stage.control)
        un1.assign(un)

        # let pyadjoint tape the time integration
        for i in range(nt):
            solve_step()

        # tell pyadjoint a) we have finished this stage
        # and b) how to evaluate this observation error
        stage.set_observation(
            state=un,
            observation_error=observation_error(idx),
            observation_covariance=Rop)

# tell pyadjoint to finish taping operations
pause_annotation()

appctx = {"correlation": Bop}

# Solution strategy is controlled via this options dictionary
tao_parameters = {
    'tao_view': ':tao_view.log',
    'tao_monitor': None,
    'tao_ls_type': 'unit',
    # 'tao_ls_monitor': None,
    'tao_converged_reason': None,
    'tao_gttol': 1e-4,
    'tao_gatol': 0,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_monitor_short': None,
        'ksp_converged_maxits': None,
        'ksp_converged_rate': None,
        'ksp_max_it': 10,
        'ksp_rtol': 1e-2,
        'ksp_type': 'cg',
        'pc_type': 'python',
        'pc_python_type': f'{__name__}.CorrelationOperatorPC',
    },
}
tao = TAOSolver(MinimizationProblem(Jhat),
                parameters=tao_parameters,
                options_prefix="",
                appctx=appctx)
xopt = tao.solve()

un.assign(background)
for _ in range(nw*nt):
    solve_step()
bkg_end = un.copy(deepcopy=True)
un.assign(xopt)
for _ in range(nw*nt):
    solve_step()
opt_end = un.copy(deepcopy=True)

bkg = background
Print(f"{errornorm(reference_ic, bkg)   = :.4e}")
Print(f"{errornorm(reference_ic, xopt)  = :.4e}")
Print(f"{errornorm(target_end, bkg_end) = :.4e}")
Print(f"{errornorm(target_end, opt_end) = :.4e}")
