from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import interpolate
from fdvar import generate_observation_data
from fdvar.correlations import *
np.random.seed(42)
Print = PETSc.Sys.Print

# number of observation windows, and steps per window
nw, nt, dt = 50, 2, 3.0e-4
re = 40
nx = 32
umax = 0.10
theta = 0.55

# Covariance of background, observation, and model noise
sigma_b = sqrt(1e-3)
sigma_r = sqrt(1e-4)
sigma_q = sqrt(1e-4*(nt*dt))

err_b = sigma_b
err_r = sigma_r
err_q = sigma_q

L_b = 0.03
lim_b = 0.99

taylor_test = True
plot_vtk = True

nu = umax/re
Nt = nw*nt
T = Nt*dt
cfl = umax*dt*nx
Print(f"{nx=:>3d} | {nw=:>2d} | {Nt=:>3d} | {T=:.2e} | {cfl=:.2e} | {re=:.2e}")

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
mesh = UnitIntervalMesh(nx)
x, = SpatialCoordinate(mesh)

# Burger's equation with implicit midpoint integration
degree = 1
V = FunctionSpace(mesh, "CG", degree)
R = FunctionSpace(mesh, "R", 0)

t = Function(R).zero()

un, un1 = Function(V), Function(V)
v = TestFunction(V)
theta_c = Constant(theta)
uh = theta_c*un1 + (1 - theta_c)*un

# finite element forms
dtc = Function(R).assign(dt)
nuc = Constant(nu)
k = Constant(umax)
x1 = 1 - x
t1 = t + 1
g = pi*sqrt(sin(pi*x))*(
    pi*k*(
        (x + k*(1 + sin(t1))*sin(pi*x1*(2+cos(2*t1))))
        *cos(pi*x*sin(t1))*sin(pi*x1*cos(t1))

        + (x1 - k*(1 + cos(t1))*sin(pi*x*cos(t1)))
        *sin(pi*x*cos(t1))*cos(pi*x1*(3+sin(3*t1)))
    )
    + (2*nuc*(k*pi*(1-sin(t1)))**2)
      *(sin(pi*x*cos(t1))*sin(pi*x1*(4+cos(4*t1)))
        + cos(pi*x*sin(t1))*cos(pi*x1*sin(t1)))
)
F = (inner((un1 - un)/dtc, v)*dx
     + inner(uh, uh.dx(0))*v*dx
     + inner(nuc*grad(uh), grad(v))*dx
     - inner(g, v)*dx(degree=2*degree)
)

bcs = [
    DirichletBC(V, 0, "on_boundary")
]

params = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",
    "snes_rtol": 1e-10,
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
    solve(F==0, un1, bcs=bcs, solver_parameters=params)
    un.assign(un1)
    t.assign(t + dt)

# "ground truth" reference solution
reference_ic = Function(V).project(umax*sin(2*pi*x))

# observations are point evaluations at random locations
vom = VertexOnlyMesh(mesh, np.random.rand(20, 1))
Y = FunctionSpace(vom, "DG", 0)

def H(x):  # operator to take observations
    return assemble(interpolate(x, Y))

# generate "ground-truth" observational data
y, background, target_end = generate_observation_data(
    None, reference_ic, solve_step, un, un1, bcs,
    t, H, nw, nt, err_b, err_r, err_q)
bkgdat = background.dat.data
Print(f"{np.min(bkgdat) = :.3e} | {np.max(bkgdat) = :.3e}")

# create function evaluating observation error at window i
def observation_error(i):
    return lambda x: Function(Y).assign(H(x) - y[i])

# create distributed control variable for entire timeseries
control = Function(V).assign(background)

# random standard deviation functions
B = rand_func(V, sigma=sigma_b, lim=lim_b, dist="symmetric")
R = rand_func(Y, lim=sigma_r, dist="normalised")
# R = rand_func(Y, sigma=sigma_r, lim=0, dist="symmetric")

# Bop = ExplicitDiffusionCorrelation(V, sigma_b, L_b, bcs=bcs)
Bop = ExplicitMassCorrelation(V, B)
Rop = ExplicitMassCorrelation(Y, R)

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
t.assign(0.0)
with Jhat.recording_stages(nstages=nw, t=t) as stages:
    for stage, ctx in stages:
        idx = stage.local_index
        un.assign(stage.control)
        un1.assign(un)
        t.assign(ctx.t)

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

if taylor_test:
    from pyadjoint.verification import taylor_to_dict
    from pprint import pprint
    from sys import exit
    taylor = taylor_to_dict(Jhat, reference_ic, background)
    pprint(taylor)
    exit()

appctx = {"correlation": Bop}

# Solution strategy is controlled via this options dictionary
tao_parameters = {
    'tao_view': ':tao_view.log',
    'tao_monitor': None,
    'tao_ls_type': 'unit',
    'tao_converged_reason': None,
    'tao_gttol': 1e-4,
    'tao_gatol': 0,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_monitor': None,
        'ksp_converged_maxits': None,
        'ksp_converged_rate': None,
        'ksp_max_it': 20,
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


un.assign(reference_ic)
t.assign(0)
targets = [un.copy(deepcopy=True)]
for _ in range(Nt):
    solve_step()
    targets.append(un.copy(deepcopy=True))

un.assign(background)
t.assign(0)
priors = [un.copy(deepcopy=True)]
forcing = [Function(V).interpolate(g)]
for _ in range(Nt):
    solve_step()
    priors.append(un.copy(deepcopy=True))
    forcing.append(Function(V).interpolate(g))
bkg_end = un.copy(deepcopy=True)

un.assign(xopt)
t.assign(0)
opts = [un.copy(deepcopy=True)]
for _ in range(Nt):
    solve_step()
    opts.append(un.copy(deepcopy=True))
opt_end = un.copy(deepcopy=True)

bkg = background
Print(f"{norm(reference_ic) = :.3e} | {norm(target_end) = :.3e}")
Print(f"{norm(bkg)          = :.3e} | {norm(bkg_end)    = :.3e}")
Print(f"{norm(xopt)         = :.3e} | {norm(opt_end)    = :.3e}")
Print(f"{errornorm(reference_ic, bkg)   = :.4e}")
Print(f"{errornorm(reference_ic, xopt)  = :.4e}")
Print(f"{errornorm(target_end, bkg_end) = :.4e}")
Print(f"{errornorm(target_end, opt_end) = :.4e}")

if plot_vtk:
    from firedrake.output import VTKFile
    vtk_u = VTKFile("outputs/burgers.pvd")
    vtk_o = VTKFile("outputs/observations.pvd")
    ut = Function(V, name="target")
    up = Function(V, name="prior")
    uo = Function(V, name="opt")
    uf = Function(V, name="forcing")
    for i, (target, prior, opt, force) in enumerate(zip(targets, priors,
                                                        opts, forcing)):
        j = i//nt if i>0 else 0
        time = float(i*dt)
        vtk_u.write(
            ut.assign(target),
            up.assign(prior),
            uo.assign(opt),
            uf.assign(force),
            time=float(i*dt))
