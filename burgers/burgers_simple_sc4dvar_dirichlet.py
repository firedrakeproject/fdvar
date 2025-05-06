from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import interpolate
from fdvar import generate_observation_data
from sys import exit
np.random.seed(42)
Print = PETSc.Sys.Print

# number of observation windows, and steps per window
nx, nw, nt, dt, nu = 100, 50, 6, 1e-4, 0.25

# T = 0.03
# nsw = 50
# qscale = T/nsw
qscale = nt*dt

# Covariance of background, observation, and model noise
sigma_b = 1e-2
sigma_r = 1e-3
sigma_q = (1e-4)*qscale

# lengthscales of the background and model covariances
L_b = 0.25
L_q = 0.05

cfl_b = (L_b*L_b/2)*(nx*nx)
Print(f"{cfl_b = }")

# 1D periodic mesh
mesh = UnitIntervalMesh(nx)
x, = SpatialCoordinate(mesh)

# Burger's equation with implicit midpoint integration
degree = 2
V = VectorFunctionSpace(mesh, "CG", degree)
V0 = FunctionSpace(mesh, "CG", degree)
Real = FunctionSpace(mesh, "R", 0)

# vary between (Bmin =< B/sigma_b =< 1)
# Bmin = 0.1
# Bprofile = cos(2*pi*x)*sin(4*pi*(0.5-x))
# Bexpr = ((1 + Bmin) + (1 - Bmin)*Bprofile)/2
# B = Function(V0).project(sigma_b*Bexpr)

t = Function(Real).assign(0.)
fdt = Function(Real).assign(dt)

un, un1 = Function(V), Function(V)
v = TestFunction(V)

zero = Constant(0)
bcs = [DirichletBC(V, as_vector([zero]), "on_boundary")]

# B = (sigma_b, L_b, bcs)
# Q = (sigma_q, L_q, bcs)
# R = sigma_r

lu_params = {'ksp_type': 'preonly', 'pc_type': 'lu'}

B = CovarianceOperator(
    V, form_type="diffusion",
    sigma=sigma_b, L=L_b, bcs=bcs,
    solver_parameters=lu_params)

# B = CovarianceOperator(
#     V, form_type="mass", sigma=sigma_b,
#     bcs=bcs, solver_parameters=lu_params)

params = {
    'snes_rtol': 1e-10,
    **lu_params,
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

# "ground truth" reference solution
reference_ic = Function(V).project(
    as_vector([k*sin(2*pi*x)]))
for bc in bcs:
    bc.apply(reference_ic)

# observations are point evaluations at random locations
observation_locations = [
    [x] for x in np.random.random_sample(20)]
vom = VertexOnlyMesh(mesh, observation_locations)
Y = VectorFunctionSpace(vom, "DG", 0)
Y0 = FunctionSpace(vom, "DG", 0)

# vary between (sigma_r =< R =< 1)
Rprofile = sin(6*pi*(x + 0.3))
Rexpr = ((1 + sigma_r) + (1 - sigma_r)*Rprofile)/2
Rfunc = Function(Y0).interpolate(Rexpr)
R = CovarianceOperator(
    Y, form_type="mass", sigma=Rfunc,
    solver_parameters=lu_params)

def H(x):  # operator to take observations
    return assemble(interpolate(x, Y))

# generate "ground-truth" observational data
y, background = generate_observation_data(
    None, reference_ic, stepper, un, un1, bcs,
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
        'ksp_max_it': 15,
        'ksp_rtol': 1e-1,
        'ksp_type': 'cg',
        'pc_type': 'python',
        'pc_python_type': f'fdvar.SC4DVarBackgroundPC',
    },
}
tao = TAOSolver(MinimizationProblem(Jhat),
                parameters=tao_parameters,
                options_prefix="sc")
xopt = tao.solve()
PETSc.Sys.Print(f"{errornorm(reference_ic, background) = :.3e}")
PETSc.Sys.Print(f"{errornorm(reference_ic, xopt)       = :.3e}")
