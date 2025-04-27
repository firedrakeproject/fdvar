from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import interpolate
from fdvar import generate_observation_data
from numpy import mean as nmean
from numpy import max as nmax
from sys import exit
from irksome import TimeStepper, GaussLegendre, Dt
np.random.seed(42)

# number of observation windows, and steps per window
nx, nw, nt, dt, nu = 20, 5, 3, 1e-4, 0.25

T = 0.03
nsw = 50

# Covariance of background, observation, and model noise
sigma_b = 1e-2
sigma_r = 1e-3
sigma_q = (1e-4)*T/nsw

# 1D periodic mesh
mesh = UnitIntervalMesh(nx)
x, = SpatialCoordinate(mesh)

# Burger's equation with implicit midpoint integration
V = VectorFunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)
t = Function(R).assign(0)
fdt = Function(R).assign(dt)

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

irk = True

# finite element forms
F = (
    (inner(un1 - un, v)/fdt)*dx
    # + inner(dot(as_vector([uh]), nabla_grad(uh)), v)*dx
    # + inner(nu*grad(uh), grad(v))*dx
    # - inner(g, v)*dx
    + inner(dot(uh, nabla_grad(uh)), v)*dx
    + inner(nu*grad(uh), grad(v))*dx
    - inner(as_vector([g]), v)*dx
)

Firk = (
    inner(Dt(un), v)*dx
    # + inner(dot(as_vector([un]), nabla_grad(un)), v)*dx
    # + inner(nu*grad(un), grad(v))*dx
    # - inner(g, v)*dx
    + inner(dot(un, nabla_grad(un)), v)*dx
    + inner(nu*grad(un), grad(v))*dx
    - inner(as_vector([g]), v)*dx
)
irk_stepper = TimeStepper(
    Firk, GaussLegendre(1), t, fdt, un,
    bcs=bcs, solver_parameters=params)

# timestepper solver
nvs_stepper = NonlinearVariationalSolver(
    NonlinearVariationalProblem(F, un1, bcs=bcs),
    solver_parameters=params)
stepper = nvs_stepper

def solve_step():
    if irk:
        irk_stepper.advance()
    else:
        un1.assign(un)
        nvs_stepper.solve()
        un.assign(un1)

# "ground truth" reference solution
reference_ic = Function(V).project(
    as_vector([k*sin(2*pi*x)]))
    # k*sin(2*pi*x))

# observations are point evaluations at random locations
observation_locations = [
    [x] for x in np.random.random_sample(20)]
vom = VertexOnlyMesh(mesh, observation_locations)
Y = VectorFunctionSpace(vom, "DG", 0)

def H(x):  # operator to take observations
    return assemble(interpolate(x, Y))

# generate "ground-truth" observational data
y, background = generate_observation_data(
    None, reference_ic, solve_step, un,
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
    background_covariance=sigma_b,
    observation_covariance=sigma_r,
    observation_error=observation_error(0),
    weak_constraint=False)

# loop over each observation stage on the local communicator
trajectory = [background.copy(deepcopy=True, annotate=False)]
t.assign(0.)
with Jhat.recording_stages(nstages=nw, t=t) as stages:
    for stage, ctx in stages:
        idx = stage.local_index
        un.assign(stage.control)

        t.assign(ctx.t)

        # let pyadjoint tape the time integration
        for i in range(nt):
            solve_step()
            t += dt
        trajectory.append(un.copy(deepcopy=True, annotate=False))

        # tell pyadjoint a) we have finished this stage
        # and b) how to evaluate this observation error
        stage.set_observation(
            state=un,
            observation_error=observation_error(idx),
            observation_covariance=sigma_r)


# tell pyadjoint to finish taping operations
pause_annotation()

# print(f"{t.dat.data[0] = :.2e}")
# sc4dvar = Jhat.strong_reduced_functional
# print(f"{Jhat._total_functional = }")
# print(f"{sc4dvar.functional = :.2e}")
# print(f"{sc4dvar(background) = :.2e}")
# print(f"{sc4dvar(reference_ic) = :.2e}")
# print(f"{sc4dvar(background) = :.2e}")
# print(f"{sc4dvar(reference_ic) = :.2e}")

# vtk = VTKFile('output/burgers.pvd')
# t = 0
# for j, u in enumerate(trajectory):
#     # ud = u.dat.data
#     # print(f"{j = :>3d} | {norm(u) = :.2e} | {nmean(ud) = :.2e} | {nmax(-ud) = :.2e} | {nmax(ud) = :.2e}")
#     # vtk.write(u, time=t)
#     # t += dt
#     # pass


class CovariancePC(PCBase):
    def initialize(self, pc):
        w = Constant(sigma_b)
        u = Function(V)
        b = Cofunction(V.dual())
        a = (1/w)*inner(TrialFunction(V), TestFunction(V))*dx
        solver = LinearVariationalSolver(
            LinearVariationalProblem(a, b, u),
            solver_parameters={'ksp_type': 'preonly',
                               'pc_type': 'lu'})
        self.u, self.b, self.solver = u, b, solver

    def apply(self, pc, x, y):
        with self.b.dat.vec_wo as vb:
            x.copy(vb)
        self.solver.solve()
        with self.u.dat.vec_ro as vu:
            vu.copy(y)

        # x.copy(y)
        # y.scale(sigma_b)

    def update(self, pc, x):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
            

# Solution strategy is controlled via this options dictionary
tao_parameters = {
    'tao_view': ':tao_view.log',
    'tao_monitor': None,
    'tao_converged_reason': None,
    'tao_gttol': 2e-1,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_monitor_short': None,
        'ksp_converged_rate': None,
        'ksp_converged_maxits': None,
        'ksp_max_it': 10,
        'ksp_rtol': 1e-1,
        'ksp_type': 'gmres',
        'pc_type': 'none',
        # 'pc_type': 'python',
        # 'pc_python_type': f'{__name__}.CovariancePC',
    },
}
tao = TAOSolver(MinimizationProblem(Jhat),
                parameters=tao_parameters)
xopt = tao.solve()
PETSc.Sys.Print(f"{errornorm(reference_ic, background) = :.3e}")
PETSc.Sys.Print(f"{errornorm(reference_ic, xopt)       = :.3e}")
