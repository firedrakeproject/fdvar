from functools import partial

from firedrake import *
from firedrake.adjoint import *
from pyadjoint.optimization.tao_solver import RFAction
from fdvar.correlations import *
from utils.shallow_water import galewsky
from utils import shallow_water as swe
from utils.planets import earth
from utils import units
import argparse

parser = argparse.ArgumentParser(
    description='Strong constraint 4DVar for the shallow water equations.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--ref_level', type=int, default=3, help='Number of mesh refinements.')
parser.add_argument('--init_days', type=int, default=4, help='Number of days to integrate for before starting data assimilation.')
parser.add_argument('--uscale', type=float, default=0.8, help='Scaling for error in velocity.')
parser.add_argument('--dt', type=float, default=1.0, help='Timestep length (hours).')
parser.add_argument('--sigmab_2', type=float, default=1e-2, help='Background variance.')
parser.add_argument('--sigmar_2', type=float, default=1e-3, help='Observation variance.')
parser.add_argument('--nw', type=int, default=1, help='Number of observations stages.')
parser.add_argument('--obs_freq', type=int, default=5, help='Frequency of observations in time.')
parser.add_argument('--hessian_action', type=str, default='hessian', help='Hessian action type.')
parser.add_argument('--plot_vtk', action='store_true', help='Plot results after optimisation.')
parser.add_argument('--taylor_test', action='store_true', help='Run Taylor test on 4DVar ReducedFunctional.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

Print = PETSc.Sys.Print
np.set_printoptions(legacy='1.25', precision=3, linewidth=200)
if args.show_args:
    Print(args)
    Print()

# number of observation windows, and steps per window
nw, nt = args.nw, args.obs_freq
dt = args.dt

# Covariance of background, observation, and model noise
sigma_b = sqrt(args.sigmab_2)
sigma_r = sqrt(args.sigmar_2)

ref = args.ref_level
Nt = nw*nt
Tend = Nt*args.dt
Tobs = nt*args.dt
Print(f"{ref = :>1d} | {Nt = :>3d} | {Tend = :.2e} | {Tobs = :.2e}")

# Icosahedral sphere mesh
mesh = swe.create_mg_globe_mesh(
    ref_level=args.ref_level, coords_degree=1)
x = SpatialCoordinate(mesh)

R = FunctionSpace(mesh, "R", 0)
Vu = FunctionSpace(mesh, "BDM", 2)
Vd = FunctionSpace(mesh, "DG", 1)
W = Vu*Vd
Print(f"{W.dim() = } | {dt = :.2e} | days = {Nt*dt/24:.2f}")
Print()

t = Function(R).assign(0.)
g = earth.Gravity

b = galewsky.topography_expression(*x)
f = swe.earth_coriolis_expression(*x)

ic = Function(W)
icu, ich = ic.subfunctions
icu.project(galewsky.velocity_expression(*x),
            form_compiler_parameters={'quadrature_degree': 6})
ich.interpolate(galewsky.depth_expression(*x))

form_mass = partial(swe.nonlinear.form_mass, mesh)
form_function = partial(swe.nonlinear.form_function, mesh, g, b, f)

wn1 = Function(W).assign(ic)
wn = Function(W).assign(ic)

un1, hn1 = split(wn1)
un, hn = split(wn)
v, q = TestFunctions(W)

uh = 0.5*(un + un1)
hh = 0.5*(hn + hn1)
dT = Constant(dt*units.hour)

F = (
    form_mass(un1-un, hn1-hn, v, q)
    + dT*form_function(uh, hh, v, q, t)
)

solver_parameters = {
    'ksp_error_if_not_converged': None,
    'snes_rtol': 1e-8,
    'ksp_rtol': 1e-5,
    'ksp_type': 'gmres',
    'pc_type': 'ilu',
    'snes_lag_preconditioner': 1000,
}

solver = NonlinearVariationalSolver(
    NonlinearVariationalProblem(F, wn1),
    solver_parameters=solver_parameters,
    options_prefix="")

def solve_step():
    wn1.assign(wn)
    solver.solve()
    wn.assign(wn1)
    t.assign(t + dT)

# "ground truth" reference solution
reference_ic = ic.copy(deepcopy=True)
ur, hr = reference_ic.subfunctions

# Background has smaller velocity perturbation
background = ic.copy(deepcopy=True)
ub, hb = background.subfunctions
ub *= args.uscale

# Start from 4 days in so there's actually some flow features
nt_init = args.init_days*round(24/args.dt)

Print(f"{errornorm(ur, ub)/norm(ur) = :.3e}")
Print(f"{errornorm(hr, hb)/norm(hr) = :.3e}")

# reference
Print("Propagating reference initial condition")
t.assign(0.)
wn.assign(reference_ic)
for _ in range(nt_init):
    solve_step()
reference_ic.assign(wn)

Print("Propagating background initial condition")
# background
t.assign(0.)
wn.assign(background)
for _ in range(nt_init):
    solve_step()
background.assign(wn)

Print(f"{errornorm(ur, ub)/norm(ur) = :.3e}")
Print(f"{errornorm(hr, hb)/norm(hr) = :.3e}")

# dense observations of depth
def H(x):
    return x.subfunctions[1].copy(deepcopy=True)

# Background correlation operator
B = ImplicitMassCorrelation(W, sigma_b)

# Observation correlation operator
R = ExplicitMassCorrelation(Vd, sigma_r)

# generate "ground-truth" observational data
Print("Generating observation data")
t.assign(0.)
wn.assign(reference_ic)
y = [H(wn)]
ground_truth = [wn.copy(deepcopy=True)]
for _ in range(nw):
    for _ in range(nt):
        solve_step()
        ground_truth.append(wn.copy(deepcopy=True))
    y.append(H(wn))

# create function evaluating observation error at window i
def observation_error(i):
    Vo = y[i].function_space()
    return lambda x: Function(Vo).assign(H(x) - y[i])

# create distributed control variable for entire timeseries
control = Function(W).assign(background)

# tell pyadjoint to start taping operations
continue_annotation()

# This object will construct and solve the 4DVar system
Print("Recording ReducedFunctional")
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
        wn.assign(stage.control)
        wn1.assign(wn)
        t.assign(ctx.t)

        # let pyadjoint tape the time integration
        for i in range(nt):
            solve_step()

        # tell pyadjoint a) we have finished this stage
        # and b) how to evaluate this observation error
        stage.set_observation(
            state=wn,
            observation_error=observation_error(idx),
            observation_covariance=R)

# tell pyadjoint to finish taping operations
pause_annotation()
xorig = control.copy(deepcopy=True)

ut, ht = ground_truth[-1].subfunctions
un, hn = wn.subfunctions
Print(f"{errornorm(ut, un)/norm(ut) = :.3e}")
Print(f"{errornorm(ht, hn)/norm(ht) = :.3e}")
Print()

if args.taylor_test:
    from sys import exit
    from pyadjoint import taylor_to_dict
    from pprint import pprint
    m = xorig.copy(deepcopy=True)
    h = m.copy(deepcopy=True)
    for hdat in h.dat:
        hdat.data[:] = np.random.random_sample(hdat.data.shape)
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
    'tao_grtol': 0,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_error_if_not_converged': None,
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'ksp_converged_maxits': None,
        'ksp_max_it': 10,
        'ksp_rtol': 1e-2,
        'ksp_type': 'cg',
        'pc_type': 'python',
        'pc_python_type': 'fdvar.CorrelationOperatorPC',
    },
}

hessian_action = RFAction(args.hessian_action)

Print(f"Optimising with {hessian_action = }\n")
tao = TAOSolver(MinimizationProblem(Jhat),
                parameters=tao_parameters,
                options_prefix="",
                Pmat=CorrelationOperatorMat(B),
                hessian_action=hessian_action)
xopt = tao.solve()

wn.assign(reference_ic)
t.assign(0)
reference = [wn.copy(deepcopy=True)]
for _ in range(Nt):
    solve_step()
    reference.append(wn.copy(deepcopy=True))

wn.assign(background)
t.assign(0)
priors = [wn.copy(deepcopy=True)]
for _ in range(Nt):
    solve_step()
    priors.append(wn.copy(deepcopy=True))

wn.assign(xopt)
t.assign(0)
opts = [wn.copy(deepcopy=True)]
for _ in range(Nt):
    solve_step()
    opts.append(wn.copy(deepcopy=True))

assert len(ground_truth) == len(reference)

xopt_u, xopt_h = xopt.subfunctions
bkg_u, bkg_h = background.subfunctions
ref_ic_u, ref_ic_h = reference_ic.subfunctions

ref_end_u, ref_end_h = reference[-1].subfunctions
bkg_end_u, bkg_end_h = priors[-1].subfunctions
opt_end_u, opt_end_h = opts[-1].subfunctions
truth_end_u, truth_end_h = ground_truth[-1].subfunctions

Print()
Print(f"{norm(ref_ic_u) = :.3e} | {norm(ref_ic_h) = :.3e}")
Print(f"{norm(bkg_u)    = :.3e} | {norm(bkg_h)    = :.3e}")
Print(f"{norm(xopt_u)   = :.3e} | {norm(xopt_h)   = :.3e}")
Print(f"{norm(truth_end_u) = :.3e} | {norm(truth_end_h) = :.3e}")
Print(f"{norm(bkg_end_u)   = :.3e} | {norm(bkg_end_h)   = :.3e}")
Print(f"{norm(opt_end_u)   = :.3e} | {norm(opt_end_h)   = :.3e}")
Print()
Print(f"{errornorm(bkg_u, xopt_u)/norm(bkg_u)       = :.3e}")
Print(f"{errornorm(bkg_h, xopt_h)/norm(bkg_h)       = :.3e}")
Print(f"{errornorm(ref_ic_u, bkg_u)/norm(ref_ic_u)  = :.3e}")
Print(f"{errornorm(ref_ic_h, bkg_h)/norm(ref_ic_h)  = :.3e}")
Print(f"{errornorm(ref_ic_u, xopt_u)/norm(ref_ic_u) = :.3e}")
Print(f"{errornorm(ref_ic_h, xopt_h)/norm(ref_ic_h) = :.3e}")
Print()
Print(f"{errornorm(truth_end_u, ref_end_u)/norm(truth_end_u) = :.3e}")
Print(f"{errornorm(truth_end_u, bkg_end_u)/norm(truth_end_u) = :.3e}")
Print(f"{errornorm(truth_end_u, opt_end_u)/norm(truth_end_u) = :.3e}")
Print()
Print(f"{errornorm(truth_end_h, ref_end_h)/norm(truth_end_h) = :.3e}")
Print(f"{errornorm(truth_end_h, bkg_end_h)/norm(truth_end_h) = :.3e}")
Print(f"{errornorm(truth_end_h, opt_end_h)/norm(truth_end_h) = :.3e}")

if args.plot_vtk:
    from firedrake.output import VTKFile
    vtk = VTKFile("outputs/swe_sc4dvar.pvd")

    wt = Function(W, name="truth")
    wp = Function(W, name="prior")
    wo = Function(W, name="opt")

    for w in (wt, wp, wo):
        u, h = w.subfunctions
        u.rename(f"{w.name()}-velocity")
        h.rename(f"{w.name()}-depth")

    for i, (truth, prior, opt) in enumerate(zip(ground_truth, priors, opts)):
        for src, dst in zip((truth, prior, opt), (wt, wp, wo)):
            dst.assign(src)
            dst.subfunctions[1].assign(dst.subfunctions[1] - galewsky.H0)
        vtk.write(*wt.subfunctions, *wp.subfunctions, *wo.subfunctions, time=float(i*args.dt))
