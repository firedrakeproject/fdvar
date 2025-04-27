from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import interpolate
from fdvar import generate_observation_data
from sys import exit
np.random.seed(42)


class BackgroundPC(PCBase):
    def initialize(self, pc):
        A, _ = pc.getOperators()
        scfdv = A.getPythonContext().problem.reduced_functional
        B = scfdv.background_norm.covariance
        Vc = scfdv.controls[0].control.function_space()

        u = Function(Vc)
        b = Cofunction(Vc.dual())
        a = inner(TrialFunction(Vc)*(1/B), TestFunction(Vc))*dx

        solver = LinearVariationalSolver(
            LinearVariationalProblem(
                a, b, u, constant_jacobian=True),
            solver_parameters={'ksp_type': 'preonly',
                               'pc_type': 'lu'})

        self.u, self.b, self.solver = u, b, solver

    def apply(self, pc, x, y):
        with self.b.dat.vec_wo as vb:
            x.copy(vb)
        self.solver.solve()
        with self.u.dat.vec_ro as vu:
            vu.copy(y)

    def update(self, pc, x):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

# number of observation windows, and steps per window
nx, nw, nt, dt, nu = 30, 15, 5, 5e-4, 0.25

T = 0.03
nsw = 50

# Covariance of background, observation, and model noise
sigma_b = 1e-2
sigma_r = 1e-3
sigma_q = (1e-4)*T/nsw

B = sigma_b
R = sigma_r
Q = sigma_q

# 1D periodic mesh
mesh = UnitIntervalMesh(nx)
x, = SpatialCoordinate(mesh)

# Burger's equation with implicit midpoint integration
V = VectorFunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "CG", 1)
Real = FunctionSpace(mesh, "R", 0)

B = Function(V0).project(sigma_b*(1 - 0.9*cos(2*pi*x)*sin(4*pi*(0.5-x))))

t = Function(Real).assign(0)
fdt = Function(Real).assign(dt)

un, un1 = Function(V), Function(V)
v = TestFunction(V)

zero = Constant(0)
bcs = [DirichletBC(V, as_vector([zero]), "on_boundary")]

params = {
    'ksp_type': 'gmres',
    'pc_type': 'ilu',
}

# forcing
k = Constant(0.1)

x1 = 1 - x
t1 = t + 1
g = (
    pi*k*(
        (x + k*t1*sin(pi*x1*t1))
        *cos(pi*x*t1)*sin(pi*x1*t1)

        + (x1 - k*t1*sin(pi*x*t1))
        *sin(pi*x*t1)*cos(pi*x1*t1)
    )

    + (2*nu*(k*pi*t1)**2)
      *(sin(pi*x*t1)*sin(pi*x1*t1)
        + cos(pi*x*t1)*cos(pi*x1*t1))
)
uh = (un + un1)/2

# finite element forms
F = (
    (inner(un1 - un, v)/fdt)*dx
    + inner(dot(uh, nabla_grad(uh)), v)*dx
    + inner(nu*grad(uh), grad(v))*dx
    - inner(as_vector([g]), v)*dx
)

# timestepper solver
stepper = NonlinearVariationalSolver(
    NonlinearVariationalProblem(F, un1, bcs=bcs),
    solver_parameters=params)


def solve_step():
    un1.assign(un)
    stepper.solve()
    un.assign(un1)


# "ground truth" reference solution
reference_ic = Function(V).project(
    as_vector([k*sin(2*pi*x)]))

# observations are point evaluations at random locations
observation_locations = [
    [x] for x in np.random.random_sample(20)]
vom = VertexOnlyMesh(mesh, observation_locations)
Y = VectorFunctionSpace(vom, "DG", 0)
Y0 = FunctionSpace(vom, "DG", 0)

# vary between (sigma_r =< R =< 1)
Rprofile = sin(6*pi*(x + 0.3))
Rexpr = ((1 + sigma_r) + (1 - sigma_r)*Rprofile)/2
R = Function(Y0).interpolate(Rexpr)

def H(x):  # operator to take observations
    return assemble(interpolate(x, Y))

# generate "ground-truth" observational data
y, background = generate_observation_data(
    None, reference_ic, stepper, un, un1,
    H, nw, nt, sigma_b, sigma_r, sigma_q)

# create function evaluating observation error at window i
def observation_error(i):
    return lambda x: Function(Y).assign(H(x) - y[i])

# create distributed control variable for entire timeseries
control = Function(V).assign(background)

# tell pyadjoint to start taping operations
continue_annotation()

# This object will construct and solve the 4DVar system
Jhat = FourDVarReducedFunctional(
    Control(control),
    background=background,
    background_covariance=B,
    observation_covariance=R,
    observation_error=observation_error(0),
    weak_constraint=False)

# loop over each observation stage on the local communicator
t.assign(0.)
with Jhat.recording_stages(nstages=nw, t=t) as stages:
    for stage, ctx in stages:
        idx = stage.local_index
        un.assign(stage.control)
        t.assign(ctx.t)

        # let pyadjoint tape the time integration
        for i in range(nt):
            un1.assign(un)
            stepper.solve()
            un.assign(un1)
            t += dt
        ctx.t.assign(t)

        # tell pyadjoint a) we have finished this stage
        # and b) how to evaluate this observation error
        stage.set_observation(
            state=un,
            observation_error=observation_error(idx),
            observation_covariance=R)


# tell pyadjoint to finish taping operations
pause_annotation()
            

# Solution strategy is controlled via this options dictionary
tao_parameters = {
    'tao_view': ':tao_view.log',
    'tao_monitor': None,
    'tao_converged_reason': None,
    'tao_gttol': 1e-1,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_monitor_short': None,
        'ksp_converged_rate': None,
        'ksp_converged_maxits': None,
        'ksp_max_it': 6,
        'ksp_rtol': 1e-1,
        'ksp_type': 'gmres',
        'pc_type': 'python',
        'pc_python_type': f'{__name__}.BackgroundPC',
    },
}
tao = TAOSolver(MinimizationProblem(Jhat),
                parameters=tao_parameters)
xopt = tao.solve()
PETSc.Sys.Print(f"{errornorm(reference_ic, background) = :.3e}")
PETSc.Sys.Print(f"{errornorm(reference_ic, xopt)       = :.3e}")
