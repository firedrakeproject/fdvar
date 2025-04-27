from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import interpolate
from fdvar import generate_observation_data
np.random.seed(42)

# number of observation windows, and steps per window
nw, nt, dt, nu = 50, 6, 1e-4, 0.01

# Covariance of background, observation, and model noise
sigma_b = 0.1
sigma_r = 0.03
sigma_q = 0.0002

# 1D periodic mesh
mesh = PeriodicUnitIntervalMesh(100)
x, = SpatialCoordinate(mesh)

# Burger's equation with implicit midpoint integration
V = VectorFunctionSpace(mesh, "CG", 2)

un, un1 = Function(V), Function(V)
v = TestFunction(V)
uh = (un + un1)/2

# finite element forms
F = (inner(un1 - un, v)*dx
     + dt*inner(dot(uh, nabla_grad(uh)), v)*dx
     + dt*inner(nu*grad(uh), grad(v))*dx)

# timestepper solver
stepper = NonlinearVariationalSolver(
    NonlinearVariationalProblem(F, un1))

def solve_step():
    un1.assign(un)
    stepper.solve()
    un.assign(un1)

# "ground truth" reference solution
reference_ic = Function(V).project(
    as_vector([1 + 0.5*sin(2*pi*x)]))

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
with Jhat.recording_stages(nstages=nw) as stages:
    for stage, ctx in stages:
        idx = stage.local_index
        un.assign(stage.control)
        un1.assign(un)

        # let pyadjoint tape the time integration
        for i in range(nt):
            stepper.solve()
            un.assign(un1)

        # tell pyadjoint a) we have finished this stage
        # and b) how to evaluate this observation error
        stage.set_observation(
            state=un,
            observation_error=observation_error(idx),
            observation_covariance=sigma_r)

# tell pyadjoint to finish taping operations
pause_annotation()


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
    'tao_gttol': 1e-2,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_monitor_short': None,
        'ksp_max_it': 10,
        'ksp_converged_maxits': None,
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
