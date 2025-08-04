from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import interpolate, Interpolator
from fdvar import generate_observation_data
from fdvar.correlations import *
import numpy as np
import argparse
from sys import exit

parser = argparse.ArgumentParser(
    description='Weak constraint 4DVar for the advection diffusion equation.',
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
parser.add_argument('--Bm', type=int, default=2, help='Number of form applications for Background correlation. Must be even.')
parser.add_argument('--L_b', type=float, default=0.1, help='Background correlation lengthscale.')
parser.add_argument('--lim_b', type=float, default=0.99, help='Background weighting range.')
parser.add_argument('--Qm', type=int, default=2, help='Number of form applications for model correlation. Must be even.')
parser.add_argument('--L_q', type=float, default=0.02, help='Model correlation lengthscale.')
parser.add_argument('--lim_q', type=float, default=0.96, help='Model weighting range.')
parser.add_argument('--nw', type=int, default=10, help='Number of observations stages.')
parser.add_argument('--obs_freq', type=int, default=5, help='Frequency of observations in time.')
parser.add_argument('--p_obs_freq', type=int, default=5, help='Number of timesteps per observation in the preconditioner.')
parser.add_argument('--nx_obs', type=int, default=20, help='Number of observations in space..')
parser.add_argument('--degree', type=int, default=1, help='Degree of CG space.')
parser.add_argument('--seed', type=int, default=13, help='RNG seed.')
parser.add_argument('--lits', type=int, default=-1, help='Number of Richardson iterations for L and L^T. Defaults to nw if < 0.')
parser.add_argument('--pc', type=str, default="schur", choices=("schur", "saddle"), help='Type of preconditioning strategy.')
parser.add_argument('--taylor_test', action='store_true', help='Run Taylor test instead of optimisation.')
parser.add_argument('--plot_vtk', action='store_true', help='Plot results after optimisation.')
parser.add_argument('--logdir', type=str, default="logs", help='Directory for log files.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

parsed_args = parser.parse_known_args()
args = parsed_args[0]

Print = PETSc.Sys.Print
np.set_printoptions(legacy='1.25', precision=3, linewidth=200)
if args.show_args:
    Print()
    Print(args)
    Print()
    Print("Other command line options:")
    Print(parsed_args[1])
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
Tob = nt*dt
cfl = vel*dt*nx
Print(f"{nx=:>3d} | {nw=:>2d} | {Nt=:>3d} | {Tend=:.2e} | {Tob=:.2e} | {cfl=:.2e}")

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
# just one ensemble rank
ensemble = Ensemble(COMM_WORLD, 1)
erank = ensemble.ensemble_rank

mesh = PeriodicUnitIntervalMesh(nx, comm=ensemble.comm)
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

params = {
    "snes_view": f":{args.logdir}/propagator_snes_view.log",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
}

bcs = []

def solve_step():
    un1.assign(un)
    solve(F==0, un1, bcs=bcs, solver_parameters=params)
    un.assign(un1)
    t.assign(t + dtc)

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
seed_b = 2 + erank  # 2nd letter
B = ImplicitDiffusionCorrelation(V, sigma_b, args.L_b, m=args.Bm, seed=seed_b)

# Model correlation operator
seed_q = 17 + erank  # 17th letter
Q = ImplicitDiffusionCorrelation(V, sigma_q, args.L_q, m=args.Qm, seed=seed_q)

# Observation correlation operator
seed_r = 18 + erank  # 18th letter
Rscale = rand_func(Y, lim=sigma_r, dist="normalised", seed=seed_r-1)
R = ExplicitMassCorrelation(Y, Rscale, seed=seed_r)
# generate noise with uniform standard deviation
Rgenerator = ExplicitMassCorrelation(Y, sigma=sigma_r, seed=seed_r)

# generate "ground-truth" observational data
y, background, target_end, ground_truth = generate_observation_data(
    ensemble, reference_ic, solve_step,
    un, un1, bcs, t, H, nw, nt, B, Rgenerator, Q)
bkgdat = background.dat.data
icdat = reference_ic.dat.data
Print()
Print(f"{norm(background) = :.3e} | {norm(reference_ic) = :.3e}")
Print(f"{np.mean(bkgdat) = :.3e} | {np.mean(icdat) = :.3e}")
Print(f"{np.min(bkgdat) = :.3e} | {np.max(bkgdat) = :.3e}")

# create function evaluating observation error at window i
def observation_error(i):
    return lambda x: Function(Y).assign(H(x) - y[i])

# create distributed control variable for entire timeseries
nlocal_stages = nw//ensemble.ensemble_size
nlocal_spaces = nlocal_stages + (erank == 0)
# print(f"{erank = } | {len(ground_truth) = } | {nlocal_spaces = }")
W = EnsembleFunctionSpace(
    [V for _ in range(nlocal_spaces)],
    ensemble)
control = EnsembleFunction(W)

truth = EnsembleFunction(W)
if erank == 0:
    truth.subfunctions[0].assign(ground_truth[0])
    for j in range(1, nlocal_stages+1):
        jg = j*nt
        tsub = truth.subfunctions[j]
        tsub.assign(ground_truth[jg])
else:
    for j in range(nlocal_stages):
        jg = (j+1)*nt
        tsub = truth.subfunctions[j]
        tsub.assign(ground_truth[jg])

for u in control.subfunctions:
    u.assign(background)

########
### Now we record the 4DVar system
########

# tell pyadjoint to start taping operations
continue_annotation()

# This object will construct and solve the 4DVar system
Jhat = FourDVarReducedFunctional(
    Control(control),
    background=background,
    background_covariance=B,
    observation_covariance=R,
    observation_error=observation_error(0),
    weak_constraint=True)

# loop over each observation stage on the local communicator
t.assign(0.0)
with Jhat.recording_stages(t=t) as stages:
    for stage, ctx in stages:
        idx = stage.local_index + (erank == 0)
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
            observation_covariance=R,
            forward_model_covariance=Q)

# tell pyadjoint to finish taping operations
pause_annotation()
Print(f"{float(t) = :.2e}")
prior = Jhat.control.control.copy()

########
### Now we record the preconditioner system
########

# take fewer timesteps between each observation
pnt = args.p_obs_freq
dtc.assign(dt*nt/pnt)

pcontrol = control.copy()
pbackground = background.copy(deepcopy=True)
continue_annotation()

# This object will construct and solve the 4DVar system
JPhat = FourDVarReducedFunctional(
    Control(pcontrol),
    background=pbackground,
    background_covariance=B,
    observation_covariance=R,
    observation_error=observation_error(0),
    weak_constraint=True)

# loop over each observation stage on the local communicator
t.assign(0.0)
with JPhat.recording_stages(t=t) as stages:
    for stage, ctx in stages:
        idx = stage.local_index + (erank == 0)
        un.assign(stage.control)
        un1.assign(un)
        t.assign(ctx.t)

        # let pyadjoint tape the time integration
        for i in range(pnt):
            solve_step()

        # tell pyadjoint a) we have finished this stage
        # and b) how to evaluate this observation error
        stage.set_observation(
            state=un,
            observation_error=observation_error(idx),
            observation_covariance=R,
            forward_model_covariance=Q)

# tell pyadjoint to finish taping operations
pause_annotation()
Print(f"{float(t) = :.2e}")
dtc.assign(dt)

# Solution strategy is controlled via this options dictionary
lits = args.lits if args.lits >= 0 else nw+1
ksp_monitor = 'ksp_monitor_true_residual'

tao_parameters = {
    'tao_view': f':{args.logdir}/tao_view.log',
    'tao_monitor': None,
    'tao_max_it': 30,
    'tao_ls_type': 'unit',
    # 'tao_ls_monitor': None,
    'tao_converged_reason': None,
    'tao_gttol': 1e-1,
    'tao_grtol': 1e-6,
    'tao_gatol': 0,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_view': f':{args.logdir}/ksp_view.log',
        ksp_monitor: None,
        'ksp_converged_rate': None,
        'ksp_converged_maxits': None,
        'ksp_max_it': 20,
        'ksp_rtol': 1e-3,
        'ksp_type': 'preonly',
    },
}

schur_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'fdvar.WC4DVarSchurPC',
    'wcschur_l': {
        'ksp_convergence_test': 'skip',
        'ksp_converged_maxits': None,
        'ksp_type': 'richardson',
        'ksp_max_it': lits,
    },
    'wcschur_d': {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'fdvar.EnsembleBJacobiPC',
        'sub_ksp_type': 'preonly',
        'sub_pc_type': 'python',
        'sub_pc_python_type': 'fdvar.CorrelationOperatorPC',
    },
    'wcschur': {  # monitors
        'l_tlm_ksp_view': f':{args.logdir}/wcschur_ltlm_ksp_view.log',
        'l_adj_ksp_view': f':{args.logdir}/wcschur_ladj_ksp_view.log',
        'l_tlm_ksp_monitor': f':{args.logdir}/wcschur_ltlm_ksp_convergence.log',
        'l_adj_ksp_monitor': f':{args.logdir}/wcschur_ladj_ksp_convergence.log',
        'd_ksp_view': f':{args.logdir}/wschur_d_ksp_view.log',
        'd_ksp_monitor': f':{args.logdir}/wcschur_d_ksp_convergence.log',
    },
}

saddle_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'fdvar.WC4DVarSaddlePointPC',
    'pc_wcsaddle_rhs_type': 'saddle',
    'wcsaddle': {
        'ksp_view': f':{args.logdir}/wcsaddle_ksp_view.log',
        ksp_monitor: None,
        'ksp_converged_rate': None,
        'ksp_converged_maxits': None,
        'ksp_max_it': 100,
        'ksp_min_it': 4,
        'ksp_rtol': 1e-2,
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full', # <diag=Pb,upper=Pt>
        'pc_fieldsplit_0_fields': '0,1',
        'pc_fieldsplit_1_fields': '2',
        'fieldsplit_0': {
            'ksp_type': 'preonly',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'additive',
            'fieldsplit': {
                'ksp_type': 'preonly',
                'pc_type': 'python',
                'pc_python_type': 'fdvar.EnsembleBJacobiPC',
                'sub_pc_type': 'python',
                'sub_pc_python_type': 'fdvar.CorrelationOperatorPC',
            },
            'fieldsplit_0_ksp_monitor': f':{args.logdir}/wcsaddle_d_ksp_convergence.log',
            'fieldsplit_1_ksp_monitor': f':{args.logdir}/wcsaddle_r_ksp_convergence.log',
        },
        'fieldsplit_1_ksp_monitor_true_residual': f':{args.logdir}/wcsaddle_s_ksp_convergence.log',
        'fieldsplit_1_ksp_type': 'preonly',
        'fieldsplit_1': schur_parameters
    },
}

if args.pc == "schur":
    tao_parameters["tao_nls"].update(schur_parameters)
    tao_parameters["tao_nls"]["ksp_type"] = "cg"
elif args.pc == "saddle":
    tao_parameters["tao_nls"].update(saddle_parameters)
    tao_parameters["tao_nls"]["ksp_type"] = "preonly"

from pyadjoint.optimization.tao_solver import ReducedFunctionalMat, HessianAction
Pmat = ReducedFunctionalMat(
    JPhat, action=HessianAction,
    comm=ensemble.global_comm)
JPhat(prior)
JPhat.derivative(apply_riesz=False)

tao = TAOSolver(MinimizationProblem(Jhat),
                parameters=tao_parameters,
                options_prefix="", Pmat=Pmat,
                comm=ensemble.global_comm)

Print()
xopt = tao.solve()

prior_ic = prior.subfunctions[0]
prior_end = prior.subfunctions[-1]
xopts_ic = xopt.subfunctions[0]
xopts_end = xopt.subfunctions[-1]
truth_ic = truth.subfunctions[0]
truth_end = truth.subfunctions[-1]
Print()
Print(f"{Jhat.Jmodel(truth) = :.4e}")
Print(f"{Jhat.Jobservations(truth) = :.4e}")
Print(f"{Jhat(truth) = :.4e}")
Print()
Print(f"{Jhat.Jmodel(prior) = :.4e}")
Print(f"{Jhat.Jobservations(prior) = :.4e}")
Print(f"{Jhat(prior) = :.4e}")
Print()
Print(f"{Jhat.Jmodel(xopt) = :.4e}")
Print(f"{Jhat.Jobservations(xopt) = :.4e}")
Print(f"{Jhat(xopt) = :.4e}")
Print()
Print(f"{norm(truth_ic) = :.3e} | {norm(truth_end) = :.3e}")
Print(f"{norm(prior_ic) = :.3e} | {norm(prior_end) = :.3e}")
Print(f"{norm(xopts_ic) = :.3e} | {norm(xopts_end) = :.3e}")
Print(f"{errornorm(prior_ic, xopts_ic)/norm(prior_ic) = :.3e}")
Print(f"{errornorm(truth_ic, prior_ic)/norm(truth_ic) = :.3e}")
Print(f"{errornorm(truth_ic, xopts_ic)/norm(truth_ic) = :.3e}")
Print(f"{errornorm(prior_end, xopts_end)/norm(prior_end) = :.3e}")
Print(f"{errornorm(truth_end, prior_end)/norm(truth_end) = :.3e}")
Print(f"{errornorm(truth_end, xopts_end)/norm(truth_end) = :.3e}")
Print()
