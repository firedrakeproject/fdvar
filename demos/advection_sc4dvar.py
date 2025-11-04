from firedrake import *
from firedrake.adjoint import *
from irksome import Dt, TimeStepper, GaussLegendre
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
parser.add_argument('--vprime', type=float, default=0.1, help='velocity perturbation.')
parser.add_argument('--re', type=float, default=50, help='Reynolds number.')
parser.add_argument('--sigmab_2', type=float, default=1e-2, help='Background variance.')
parser.add_argument('--sigmar_2', type=float, default=1e-3, help='Observation variance.')
parser.add_argument('--sigmaq_2', type=float, default=1e-4, help='Model variance (times stage duration).')
parser.add_argument('--Bm', type=int, default=2, help='Number of form applications for Background covariance. Must be even.')
parser.add_argument('--L_b', type=float, default=0.2, help='Background correlation lengthscale.')
parser.add_argument('--Qm', type=int, default=2, help='Number of form applications for model correlation. Must be even.')
parser.add_argument('--L_q', type=float, default=0.05, help='Model correlation lengthscale.')
parser.add_argument('--nw', type=int, default=10, help='Number of observations stages.')
parser.add_argument('--obs_freq', type=int, default=5, help='Frequency of observations in time.')
parser.add_argument('--nx_obs', type=int, default=20, help='Number of observations in space.')
parser.add_argument('--seed', type=int, default=13, help='RNG seed.')
parser.add_argument('--plot_vtk', action='store_true', help='Plot results after optimisation.')
parser.add_argument('--taylor_test', action='store_true', help='Run Taylor test on 4DVar ReducedFunctional.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args, _ = parser.parse_known_args()

Print = PETSc.Sys.Print
np.set_printoptions(legacy='1.25', precision=3, linewidth=200)
if args.show_args:
    Print()
    Print(args)
    Print()

np.random.seed(args.seed)

# number of observation windows, and steps per window
nw, nt, nx = args.nw, args.obs_freq, args.nx
dt = args.dt
re, vprime = args.re, args.vprime
umax = args.umax

# Covariance of background, observation, and model noise
sigma_b = sqrt(args.sigmab_2)
sigma_r = sqrt(args.sigmar_2)
sigma_q = sqrt(args.sigmaq_2*(nt*dt))

nu = 1/re
Nt = nw*nt
Tend = Nt*dt
Tobs = nt*dt
cfl = dt*nx
Print(f"{nx=:>3d} | {nw=:>2d} | {Nt=:>3d} | {Tend=:.2e} | {Tobs=:.2e} | {cfl=:.2e}")

# 1D periodic mesh
mesh = PeriodicUnitIntervalMesh(nx)
x, = SpatialCoordinate(mesh)

# Advection equation with implicit midpoint integration
V = FunctionSpace(mesh, "CG", 1)
Vr = FunctionSpace(mesh, "R", 0)

t = Function(Vr).zero()

un = Function(V)
v = TestFunction(V)
one = Constant(1.0)
velocity = Function(V).project(one + Constant(vprime)*cos(2*pi*x))

# finite element forms
nuc = Constant(nu)
k = Constant(umax)
gscale = Constant(1.0)


def g(tg):
    xp = 2*pi*x
    tp = 2*pi*tg
    kernel = 0.5*(1 - cos(xp))
    return gscale*kernel*2*k*(
        - sin(xp + (0.1*pi*sin(tp)))
        + k*cos(tp+1)*sin(3*xp - 2*tp)
    )


F = (inner(Dt(un), v)*dx
     + inner(velocity, un.dx(0))*v*dx
     + inner(nuc*grad(un), grad(v))*dx
     - inner(g(t), v)*dx(degree=4)
)

solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
}

tableau = GaussLegendre(1)

stepper = TimeStepper(
    F, tableau, t, dt, un,
    solver_parameters=solver_parameters,
    options_prefix="")
un = stepper.u0

# "ground truth" reference solution
reference_ic = Function(V).project(umax*sin(2*pi*x))

# observations are point evaluations at random locations
np.random.seed(15)
stations = np.random.rand(args.nx_obs, 1)
vom = VertexOnlyMesh(mesh, stations)
Y = FunctionSpace(vom, "DG", 0)

def H(x):  # operator to take observations
    return assemble(interpolate(x, Y))

# Background correlation operator
B = ImplicitDiffusionCorrelation(V, sigma_b, args.L_b, m=args.Bm, seed=2)

# Model correlation operator
Q = ImplicitDiffusionCorrelation(V, sigma_q, args.L_q, m=args.Qm, seed=17)

# Observation correlation operator
Rscale = Function(Y)
rdata = Rscale.dat.data
rdata[:] = (
    sigma_r + (1 - sigma_r)*np.random.random_sample(rdata.shape))
R = ExplicitMassCorrelation(Y, Rscale, seed=18)

# Observation noise generator
Rgen = ExplicitMassCorrelation(Y, sigma_r, seed=18-1)

def solve_step():
    stepper.advance()
    t.assign(t + dt)

# generate "ground-truth" observational data
y, background, target_end, ground_truth = generate_observation_data(
    None, reference_ic, solve_step,
    un, un, [], t, H, nw, nt, B, Rgen, Q)

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
t.assign(0.0)
with Jhat.recording_stages(nstages=nw, t=t) as stages:
    for stage, ctx in stages:
        idx = stage.local_index + 1
        stepper.u0.assign(stage.control)
        t.assign(ctx.t)

        # let pyadjoint tape the time integration
        for i in range(nt):
            stepper.advance()
            t += dt

        # tell pyadjoint a) we have finished this stage
        # and b) how to evaluate this observation error
        stage.set_observation(
            state=stepper.u0,
            observation_error=observation_error(idx),
            observation_covariance=R)

# tell pyadjoint to finish taping operations
pause_annotation()
xorig = control.copy(deepcopy=True)

if args.taylor_test:
    from sys import exit
    from pyadjoint import taylor_to_dict
    from pprint import pprint
    m = xorig.copy(deepcopy=True)
    h = m.copy(deepcopy=True)
    h.dat.data[:] = np.random.random_sample(h.dat.data.shape)
    taylor = taylor_to_dict(Jhat, m, h)
    pprint(taylor)
    exit()

# Solution strategy is controlled via this options dictionary
tao_parameters = {
    'tao_view': ':tao_view.log',
    'tao_monitor': None,
    'tao_ls_type': 'unit',
    'tao_converged_reason': None,
    'tao_gttol': 1e-2,
    'tao_gatol': 0,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_monitor': None,
        'ksp_converged_maxits': None,
        'ksp_converged_rate': None,
        'ksp_max_it': 20,
        'ksp_rtol': 1e-1,
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
forcing = [Function(V).interpolate(g(t))]
for _ in range(Nt):
    solve_step()
    priors.append(un.copy(deepcopy=True))
    forcing.append(Function(V).interpolate(g(t)))

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
# Print(f"{Jhat(bkg)   = :.3e}")
# Print(f"{Jhat(xorig) = :.3e}")
# Print(f"{Jhat(xopt)  = :.3e}")

if args.plot_vtk:
    from firedrake.output import VTKFile
    vtk = VTKFile("outputs/advection_sc4dvar.pvd")

    mesh_out = UnitIntervalMesh(args.nx)
    Vout = FunctionSpace(mesh_out, "CG", 1)

    ug = Function(Vout, name="ground_truth")
    ur = Function(Vout, name="reference")
    up = Function(Vout, name="prior")
    uo = Function(Vout, name="opt")
    uf = Function(Vout, name="forcing")

    for i, (truth, ref, prior, opt, force) in enumerate(zip(ground_truth, reference,
                                                            priors, opts, forcing)):
        for src, dst in zip((truth, ref, prior, opt, force),
                            (ug, ur, up, uo, uf)):
            dst.interpolate(src)
        vtk.write(ug, ur, up, uo, uf, time=float(i*dt))
