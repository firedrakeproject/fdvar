from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import interpolate, Interpolator
from fdvar import generate_observation_data
from fdvar.correlations import *
import argparse

parser = argparse.ArgumentParser(
    description='Strong constraint 4DVar for the advection diffusion equation.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=50, help='Number of elements.')
parser.add_argument('--dt', type=float, default=1e-2, help='Number of elements.')
parser.add_argument('--umax', type=float, default=0.3, help='Initial scalar variation.')
parser.add_argument('--vel', type=float, default=1.0, help='Mean velocity.')
parser.add_argument('--vprime', type=float, default=0.1, help='velocity perturbation.')
parser.add_argument('--re', type=float, default=50, help='Reynolds number.')
parser.add_argument('--theta', type=float, default=0.5, help='Implicit timestepping parameter.')
parser.add_argument('--sigmab_2', type=float, default=1e-2, help='Background variance.')
parser.add_argument('--sigmar_2', type=float, default=1e-3, help='Observation variance.')
parser.add_argument('--sigmaq_2', type=float, default=1e-4, help='Model variance (times stage duration).')
parser.add_argument('--Btype', type=str, default="diffusion", choices=("mass", "diffusion"), help='Type of background correlation operator.')
parser.add_argument('--Bm', type=int, default=2, help='Number of form applications for Background covariance. Must be even.')
parser.add_argument('--L_b', type=float, default=0.1, help='Background correlation lengthscale.')
parser.add_argument('--lim_b', type=float, default=0.99, help='Background weighting range.')
parser.add_argument('--Qtype', type=str, default="diffusion", choices=("mass", "diffusion"), help='Type of model correlation operator.')
parser.add_argument('--Qm', type=int, default=2, help='Number of form applications for model correlation. Must be even.')
parser.add_argument('--L_q', type=float, default=0.02, help='Model correlation lengthscale.')
parser.add_argument('--lim_q', type=float, default=0.96, help='Model weighting range.')
parser.add_argument('--nw', type=int, default=10, help='Number of observations stages.')
parser.add_argument('--obs_freq', type=int, default=5, help='Frequency of observations in time.')
parser.add_argument('--nx_obs', type=int, default=20, help='Number of observations in space..')
parser.add_argument('--degree', type=int, default=1, help='Degree of CG space.')
parser.add_argument('--seed', type=int, default=13, help='RNG seed.')
parser.add_argument('--reaction', action='store_true', help='Add a nonlinear reaction term. Useful for checking Hessian taylor test.')
parser.add_argument('--taylor_test', action='store_true', help='Run Taylor test instead of optimisation.')
parser.add_argument('--plot_vtk', action='store_true', help='Plot results after optimisation.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

Print = PETSc.Sys.Print
if args.show_args:
    Print()
    Print(args)
    Print()

np.random.seed(args.seed)

# number of observation windows, and steps per window
nw, nt, nx = args.nw, args.obs_freq, args.nx
dt, theta = args.dt, args.theta
re, vel, vprime = args.re, args.vel, args.vprime
umax = args.umax

# Covariance of background, observation, and model noise
sigma_b = sqrt(args.sigmab_2)
sigma_r = sqrt(args.sigmar_2)
sigma_q = sqrt(args.sigmaq_2*(nt*dt))

nu = 1/re
Nt = nw*nt
Tend = Nt*dt
Tobs = nt*dt
cfl = vel*dt*nx
Print(f"{nx=:>3d} | {nw=:>2d} | {Nt=:>3d} | {Tend=:.2e} | {Tobs=:.2e} | {cfl=:.2e}")

def rand_func(V, sigma=0.0, lim=0.5, dist="symmetric", seed=None):
    if seed:
        np.random.seed(seed)
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
V = FunctionSpace(mesh, "CG", args.degree)
Vr = FunctionSpace(mesh, "R", 0)

t = Function(Vr).zero()

un, un1 = Function(V), Function(V)
v = TestFunction(V)
theta_c = Constant(theta)
uh = theta_c*un1 + (1 - theta_c)*un
velocity = Function(V).project(vel + vprime*cos(2*pi*x))

# finite element forms
dtc = Function(Vr).assign(dt)
nuc = Constant(nu)
k = Constant(umax)
x1 = 1 - x
t1 = t + 1
g = pi*sqrt(sin(pi*x))*(
    pi*k*(
        (x + k*(1 + sin(pi*t1))*sin(2*pi*x1*(2+cos(2*pi*t1))))
        *cos(2*pi*x*sin(pi*t1))*sin(2*pi*x1*cos(pi*t1))

        + (x1 - k*(1 + cos(pi*t1))*sin(pi*x*cos(pi*t1)))
        *sin(pi*x*cos(pi*t1))*cos(pi*x1*(3+sin(3*pi*t1)))
    )
    + (2*nuc*(k*pi*(1-sin(pi*t1)))**2)
      *(sin(pi*x*cos(pi*t1))*sin(pi*x1*(4+cos(4*t1)))
        + cos(pi*x*sin(pi*t1))*cos(pi*x1*sin(pi*t1)))
)
F = (inner((un1 - un)/dtc, v)*dx
     + inner(vel, uh.dx(0))*v*dx
     + inner(nuc*grad(uh), grad(v))*dx
     - inner(g, v)*dx(degree=2*args.degree)
)
if args.reaction:
    xi = Constant(0.02)
    F += xi*inner(un1*un1*un1, v)*dx

params = {
    "snes_view": ":propagator_snes_view.log",
    "snes_type": "newtonls" if args.reaction else "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
}

# timestepper solver
# stepper = NonlinearVariationalSolver(
#     NonlinearVariationalProblem(F, un1),
#     solver_parameters=params)

bcs = []

def solve_step():
    un1.assign(un)
    # stepper.solve()
    solve(F==0, un1, bcs=bcs, solver_parameters=params)
    un.assign(un1)
    t.assign(t + dt)

# "ground truth" reference solution
reference_ic = Function(V).project(umax*sin(2*pi*x))

# observations are point evaluations at random locations
np.random.seed(15)
stations = np.random.rand(args.nx_obs, 1)
vom = VertexOnlyMesh(mesh, stations)
Y = FunctionSpace(vom, "DG", 0)
#Print(f"stations=\n{stations.T}")

def H(x):  # operator to take observations
    return assemble(interpolate(x, Y))

# Background correlation operator
seed_b = 2  # 2nd letter
if args.Btype == "mass":
    Bscale = rand_func(V, sigma=sigma_b, lim=args.lim_b, dist="symmetric", seed=seed_b-1)
    B = ExplicitMassCorrelation(V, Bscale, m=args.Bm, seed=seed_b)
elif args.Btype == "diffusion":
    B = ImplicitDiffusionCorrelation(V, sigma_b, args.L_b, m=args.Bm, seed=seed_b)

# Model correlation operator
seed_q = 17  # 17th letter
if args.Qtype == "mass":
    Qscale = rand_func(V, sigma=sigma_q, lim=args.lim_q, dist="symmetric", seed=seed_q-1)
    Q = ExplicitMassCorrelation(V, Qscale, m=args.Qm, seed=seed_q)
elif args.Qtype == "diffusion":
    Q = ImplicitDiffusionCorrelation(V, sigma_q, args.L_q, m=args.Qm, seed=seed_q)

# Observation correlation operator
seed_r = 18  # 18th letter
Rscale = rand_func(Y, lim=sigma_r, dist="normalised", seed=seed_r-1)
R = ExplicitMassCorrelation(Y, Rscale, seed=seed_r)

# generate "ground-truth" observational data
y, background, target_end, ground_truth = generate_observation_data(
    None, reference_ic, solve_step,
    un, un1, bcs, t, H, nw, nt, B, R, Q)
bkgdat = background.dat.data
icdat = reference_ic.dat.data
Print(f"{norm(background) = :.3e} | {norm(reference_ic) = :.3e}")
Print(f"{np.mean(bkgdat) = :.3e} | {np.mean(icdat) = :.3e}")
Print(f"{np.min(bkgdat) = :.3e} | {np.max(bkgdat) = :.3e}")

# create function evaluating observation error at window i
def observation_error(i):
    return lambda x: Function(Y).assign(H(x) - y[i])
    # return lambda x: Function(Y).assign(0.)

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
            observation_covariance=R)

# tell pyadjoint to finish taping operations
pause_annotation()
xorig = control.copy(deepcopy=True)
# Jhat(xorig.assign(xorig + B.correlated_noise()))

if args.taylor_test:
    from pyadjoint.verification import taylor_to_dict
    from pprint import pprint
    from sys import exit
    h0 = Function(V).assign(reference_ic + B.correlated_noise())
    dh = Function(V).assign(B.correlated_noise()/sigma_b)
    taylor = taylor_to_dict(Jhat, h0, dh)
    pprint(taylor)
    Print(f"{min(taylor['R0']['Rate']) = :.4e}")
    Print(f"{min(taylor['R1']['Rate']) = :.4e}")
    Print(f"{min(taylor['R2']['Rate']) = :.4e}")
    exit()

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
        'pc_python_type': 'fdvar.CorrelationOperatorPC',
    },
}
tao = TAOSolver(MinimizationProblem(Jhat),
                parameters=tao_parameters,
                options_prefix="",
                Pmat=CorrelationOperatorMat(B))
xopt = tao.solve()


un.assign(reference_ic)
t.assign(0)
reference = [un.copy(deepcopy=True)]
for _ in range(Nt):
    solve_step()
    reference.append(un.copy(deepcopy=True))

un.assign(background)
t.assign(0)
priors = [un.copy(deepcopy=True)]
forcing = [Function(V).interpolate(g)]
for _ in range(Nt):
    solve_step()
    priors.append(un.copy(deepcopy=True))
    forcing.append(Function(V).interpolate(g))

un.assign(xopt)
t.assign(0)
opts = [un.copy(deepcopy=True)]
for _ in range(Nt):
    solve_step()
    opts.append(un.copy(deepcopy=True))

bkg = background
ref_ic = reference_ic
ref_end = reference[-1]
bkg_end = priors[-1]
opt_end = opts[-1]
truth_end = ground_truth[-1]
Print(f"{norm(reference_ic) = :.3e} | {norm(target_end) = :.3e}")
Print(f"{norm(bkg)          = :.3e} | {norm(bkg_end)    = :.3e}")
Print(f"{norm(xopt)         = :.3e} | {norm(opt_end)    = :.3e}")
Print(f"{errornorm(bkg, xopt)/norm(bkg)       = :.3e}")
Print(f"{errornorm(ref_ic, bkg)/norm(ref_ic)  = :.3e}")
Print(f"{errornorm(ref_ic, xopt)/norm(ref_ic) = :.3e}")
Print(f"{errornorm(truth_end, ref_end)/norm(truth_end) = :.3e}")
Print(f"{errornorm(truth_end, bkg_end)/norm(truth_end) = :.3e}")
Print(f"{errornorm(truth_end, opt_end)/norm(truth_end) = :.3e}")
Print(f"{Jhat(bkg)   = :.3e}")
Print(f"{Jhat(xorig) = :.3e}")
Print(f"{Jhat(xopt)  = :.3e}")

if args.plot_vtk:
    from firedrake.output import VTKFile
    vtk = VTKFile("outputs/advection_sc4dvar.pvd")

    mesh_out = UnitIntervalMesh(args.nx)
    Vout = FunctionSpace(mesh_out, "CG", args.degree)
    usrc = Function(V)
    interp = Interpolator(usrc, Vout)

    ug = Function(Vout, name="ground_truth")
    ur = Function(Vout, name="reference")
    up = Function(Vout, name="prior")
    uo = Function(Vout, name="opt")
    uf = Function(Vout, name="forcing")

    for i, (truth, ref, prior, opt, force) in enumerate(zip(ground_truth, reference,
                                                            priors, opts, forcing)):
        for src, dst in zip((truth, ref, prior, opt, force),
                            (ug, ur, up, uo, uf)):
            usrc.assign(src)
            dst.assign(assemble(interp.interpolate()))
        vtk.write(ug, ur, up, uo, uf, time=float(i*dt))
