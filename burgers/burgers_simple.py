from firedrake import *
from firedrake.__future__ import interpolate
from firedrake.adjoint import continue_annotation, pause_annotation, Control, FourDVarReducedFunctional
from fdvar import TAOSolver, generate_observation_data
from math import sqrt
np.random.seed(42)

nw, dt, nt, nu = 10, 1e-4, 6, 0.05

sigma_b = 0.1
sigma_r = 0.03
sigma_q = 0.0002

# ensemble parallelism
nspatial_ranks = 1
ensemble = Ensemble(COMM_WORLD, nspatial_ranks)
ensemble_size = ensemble.ensemble_size

# mesh
mesh = PeriodicUnitIntervalMesh(
    100, comm=ensemble.comm)
x, = SpatialCoordinate(mesh)

# finite element forms
V = VectorFunctionSpace(mesh, "CG", 2)

un, un1 = Function(V), Function(V)
v = TestFunction(V)
uh = (un + un1)/2

F = (inner(un1 - un, v)*dx
    + dt*inner(dot(uh, nabla_grad(uh)), v)*dx
    + dt*inner(nu*grad(uh), grad(v))*dx)

# timestepper solver
stepper = NonlinearVariationalSolver(
    NonlinearVariationalProblem(F, un1))

# "ground truth" reference solutions
reference_ic = Function(V).project(
    as_vector([1 + 0.5*sin(2*pi*x)]))

# observation mesh and operator
observation_locations = [
    [x] for x in np.random.random_sample(20)]
vom = VertexOnlyMesh(mesh, observation_locations)
Y = VectorFunctionSpace(vom, "DG", 0)

def H(u):
    return assemble(interpolate(u, Y))

# generate ground-truth observational data
y, background = generate_observation_data(
    ensemble, reference_ic, stepper, un, un1,
    H, nw, nt, sigma_b, sigma_r, sigma_q)

def observation_error(i):
    return lambda x: Function(Y).assign(H(x) - y[i])

# create Ensemble control
V_ensemble = EnsembleFunctionSpace(
    [V for _ in range(nw//ensemble_size)], ensemble)
control = EnsembleFunction(V_ensemble)

# start recording
continue_annotation()

# create 4DVar ReducedFunctional
Jhat = FourDVarReducedFunctional(
    Control(control),
    background=background,
    background_covariance=sigma_b,
    observation_covariance=sigma_r,
    observation_error=observation_error(0),
    weak_constraint=True)

with Jhat.recording_stages() as stages:
    for stage, ctx in stages:
        un.assign(stage.control)
        un1.assign(un)

        for i in range(nt):
            stepper.solve()
            un.assign(un1)

        # record observation at end of each stage
        idx = stage.local_index

        stage.set_observation(
            state=un,
            observation_error=observation_error(idx),
            observation_covariance=sigma_r,
            forward_model_covariance=sigma_q)

# finish recording
pause_annotation()

tao_parameters = {
    'tao_monitor': None,
    'tao_gttol': 1e-1,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_monitor_short': None,
        'ksp_rtol': 2e-1,
        'ksp_type': 'gmres',
        'pc_type': 'none'}
    # 'tao_type': 'cg',
    # 'tao_cg_type': 'fr',  # fr-pr-prp-hs-dy
}
tao = TAOSolver(Jhat, options_prefix="",
                solver_parameters=tao_parameters)
tao.solve()
