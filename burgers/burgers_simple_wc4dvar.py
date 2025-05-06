from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import interpolate
from fdvar import TAOSolver, generate_observation_data
np.random.seed(42)

# number of observation windows, and steps per window
nw, nt, dt, nu = 32, 6, 1e-4, 0.05

# Covariance of background, observation, and model noise
sigma_b = 0.1
sigma_r = 0.03
sigma_q = 0.0002

# time-parallelism using firedrake's Ensemble
nspatial_ranks = 1
ensemble = Ensemble(COMM_WORLD, nspatial_ranks)
ensemble_size = ensemble.ensemble_size

# 1D periodic mesh
mesh = PeriodicUnitIntervalMesh(
    100, comm=ensemble.comm)
x, = SpatialCoordinate(mesh)

# Burger's equation with implicit midpoint integration
V = VectorFunctionSpace(mesh, "CG", 2)

un, un1 = Function(V), Function(V)
v = TestFunction(V)
uh = (un + un1)/2

# finite element forms
F = (inner(un1 - un, v)/dt
     + inner(dot(uh, grad(uh)), v)
     + inner(nu*grad(uh), grad(v)))*dx

# timestepper solver
stepper = NonlinearVariationalSolver(
    NonlinearVariationalProblem(F, un1))

# "ground truth" reference solution
reference_ic = Function(V).project(
    as_vector([1 + 0.5*sin(2*pi*x)]))

# observations are point evaluations at random locations
vom = VertexOnlyMesh(mesh, np.random.rand(20, 1))
Y = VectorFunctionSpace(vom, "DG", 0)

def H(x):  # operator to take observations
    return assemble(interpolate(x, Y))

# generate "ground-truth" observational data
y, background = generate_observation_data(
    ensemble, reference_ic, stepper, un, un1,
    H, nw, nt, sigma_b, sigma_r, sigma_q)

# create function evaluating observation error at window i
def observation_error(i):
    return lambda x: Function(Y).assign(H(x) - y[i])

# create distributed control variable for entire timeseries
V_ensemble = EnsembleFunctionSpace(
    [V for _ in range(nw//ensemble_size)], ensemble)
control = EnsembleFunction(V_ensemble)

# tell pyadjoint to start taping operations
continue_annotation()

# This object will construct and solve the 4DVar system
Jhat = FourDVarReducedFunctional(
    Control(control),
    background=background,
    background_covariance=sigma_b,
    observation_covariance=sigma_r,
    observation_error=observation_error(0),
    weak_constraint=True)

# loop over each observation stage on the local communicator
with Jhat.recording_stages() as stages:
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
            observation_covariance=sigma_r,
            forward_model_covariance=sigma_q)

# tell pyadjoint to finish taping operations
pause_annotation()

# Solution strategy is controlled via this options dictionary
tao_parameters = {
    'tao_monitor': None,
    'tao_converged_reason': None,
    'tao_gttol': 1e-2,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_monitor_short': None,
        'ksp_converged_maxits': None,
        'ksp_max_it': 8,
        'ksp_rtol': 1e-1,
        'ksp_type': 'gmres'}
}
tao = TAOSolver(Jhat, options_prefix="fdv",
                solver_parameters=tao_parameters)
tao.solve()
