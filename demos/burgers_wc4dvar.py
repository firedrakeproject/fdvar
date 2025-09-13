from firedrake import *
from firedrake.adjoint import *
from firedrake.adjoint.correlation_operators import *
from fdvar import generate_observation_data
import argparse

parser = argparse.ArgumentParser(
    description='Weak constraint 4DVar for the Burgers equation.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=50, help='Number of elements.')
parser.add_argument('--dt', type=float, default=1e-2, help='Number of elements.')
parser.add_argument('--umax', type=float, default=0.3, help='Initial scalar variation.')
parser.add_argument('--re', type=float, default=50, help='Reynolds number.')
parser.add_argument('--sigmab_2', type=float, default=1e-2, help='Background variance.')
parser.add_argument('--sigmar_2', type=float, default=1e-3, help='Observation variance.')
parser.add_argument('--sigmaq_2', type=float, default=1e-4, help='Model variance (times stage duration).')
parser.add_argument('--Bm', type=int, default=2, help='Number of form applications for Background correlation. Must be even.')
parser.add_argument('--L_b', type=float, default=0.2, help='Background correlation lengthscale.')
parser.add_argument('--Qm', type=int, default=2, help='Number of form applications for model correlation. Must be even.')
parser.add_argument('--L_q', type=float, default=0.05, help='Model correlation lengthscale.')
parser.add_argument('--nw', type=int, default=10, help='Number of observations stages.')
parser.add_argument('--obs_freq', type=int, default=5, help='Frequency of observations in time.')
parser.add_argument('--nx_obs', type=int, default=20, help='Number of observations in space..')
parser.add_argument('--seed', type=int, default=13, help='RNG seed.')
parser.add_argument('--lits', type=int, default=-1, help='Number of Richardson iterations for L and L^T. Defaults to nw if < 0.')
parser.add_argument('--pc', type=str, default="schur", choices=("schur", "saddle", "aux"), help='Type of preconditioning strategy.')
parser.add_argument('--taylor_test', action='store_true', help='Run Taylor test instead of optimisation.')
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
re = args.re
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
# just one rank per ensemble member
ensemble = Ensemble(COMM_WORLD, 1)
erank = ensemble.ensemble_rank

mesh = PeriodicUnitIntervalMesh(nx, comm=ensemble.comm)
x, = SpatialCoordinate(mesh)

# Advection equation with implicit midpoint integration
V = FunctionSpace(mesh, "CG", 1)
Vr = FunctionSpace(mesh, "R", 0)

t = Function(Vr).zero()

un, un1 = Function(V), Function(V)
v = TestFunction(V)
one = Constant(1.0)
half = Constant(0.5)
uh = half*(un1 + un)
ic = Function(V).project(one + Constant(umax)*sin(2*pi*x))

# finite element forms
nuc = Constant(nu)
k = Constant(umax)
gscale = Constant(1.0)


def g(tn):
    xp = 2*pi*x
    tp = 2*pi*tn
    kernel = 0.5*(1 - cos(xp))
    return gscale*kernel*2*k*(
        - sin(xp + (0.1*pi*sin(tp)))
        + k*cos(tp+1)*sin(3*xp - 2*tp)
    )


F = (inner((un1 - un)/Constant(dt), v)*dx
     + inner(uh, uh.dx(0))*v*dx
     + inner(nuc*grad(uh), grad(v))*dx
     - inner(g(t+0.5*dt), v)*dx(degree=4)
)

params = {
    "snes_rtol": 1e-8,
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
}

def solve_step():
    un1.assign(un)
    t.assign(t + dt)
    solve(F==0, un1, solver_parameters=params)
    un.assign(un1)

# "ground truth" reference solution
reference_ic = ic.copy(deepcopy=True)

# observations are point evaluations at random locations
np.random.seed(15)
stations = np.random.rand(args.nx_obs, 1)
closest_stations = np.min(np.diff(np.sort(stations.flatten())))
Print(f"{closest_stations = :.2e}")
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

# create distributed control variable for entire timeseries
nlocal_stages = nw//ensemble.ensemble_size
nlocal_spaces = nlocal_stages + (erank == 0)
W = EnsembleFunctionSpace(
    [V for _ in range(nlocal_spaces)], ensemble)

# generate "ground-truth" observational data
y, background, target_end, ground_truth = generate_observation_data(
    W, reference_ic, solve_step,
    un, un1, [], t, H, nw, nt, B, Rgen, Q)

# create function evaluating observation error at window i
def observation_error(i):
    return lambda x: Function(Y).assign(H(x) - y[i])
control = EnsembleFunction(W)

truth = ground_truth
# truth = EnsembleFunction(W)
# if erank == 0:
#     truth.subfunctions[0].assign(ground_truth[0])
#     for j in range(1, nlocal_stages+1):
#         jg = j*nt
#         tsub = truth.subfunctions[j]
#         tsub.assign(ground_truth[jg])
# else:
#     for j in range(nlocal_stages):
#         jg = (j+1)*nt
#         tsub = truth.subfunctions[j]
#         tsub.assign(ground_truth[jg])

for u in control.subfunctions:
    u.assign(background)

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

prior = control.copy()

if args.taylor_test:
    from sys import exit
    from pyadjoint import taylor_to_dict
    from pprint import pprint
    m = prior.copy()
    h = m.copy()
    with h.vec_wo() as v:
        v.array[:] = np.random.random_sample(v.array.shape)
    taylor = taylor_to_dict(Jhat, m, h)
    pprint(taylor)
    exit()

# Solution strategy is controlled via this options dictionary
lits = args.lits if args.lits >= 0 else nw+1

ksp_monitor = 'ksp_monitor_short'

tao_parameters = {
    'tao_view': f':logs/tao_view.log',
    'tao_monitor': None,
    'tao_converged_reason': None,
    'tao_max_it': 30,
    'tao_ls_type': 'more-thuente',
    'tao_gttol': 1e-2,
    'tao_grtol': 0,
    'tao_gatol': 0,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_view': f':logs/ksp_view.log',
        f'{ksp_monitor}': None,
        'ksp_converged_rate': None,
        'ksp_converged_maxits': None,
        'ksp_max_it': 20,
        'ksp_min_it': 3,
        'ksp_rtol': 1e-2,
        'ksp_type': 'preonly',
    }
}

schur_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'fdvar.WC4DVarSchurPC',
    'wcschur_l': {
        'ksp_type': 'richardson',
        'ksp_converged_maxits': None,
        # 'ksp_convergence_test': 'skip',
        'ksp_max_it': lits,
        'ksp_rtol': 1e-5,
        'pc_type': 'none',
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
        'l_tlm_ksp_view': f':logs/wcschur_ltlm_ksp_view.log',
        'l_adj_ksp_view': f':logs/wcschur_ladj_ksp_view.log',
        'l_tlm_ksp_monitor': f':logs/wcschur_ltlm_ksp_convergence.log',
        'l_adj_ksp_monitor': f':logs/wcschur_ladj_ksp_convergence.log',
        'd_ksp_view': f':logs/wschur_d_ksp_view.log',
        'd_ksp_monitor': f':logs/wcschur_d_ksp_convergence.log',
    },
}

saddle_parameters = {
    'pc_type': 'python',
    'pc_python_type': 'fdvar.WC4DVarSaddlePointPC',
    'pc_wcsaddle_rhs_type': 'saddle',
    'wcsaddle': {
        'ksp_view': f':logs/wcsaddle_ksp_view.log',
        ksp_monitor: None,
        'ksp_converged_rate': None,
        'ksp_converged_maxits': None,
        'ksp_max_it': 100,
        'ksp_min_it': 6,
        'ksp_rtol': 1e-3,
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'upper', # <diag=Pb,upper=Pt>
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
            'fieldsplit_0_ksp_monitor': f':logs/wcsaddle_d_ksp_convergence.log',
            'fieldsplit_1_ksp_monitor': f':logs/wcsaddle_r_ksp_convergence.log',
        },
        'fieldsplit_1_ksp_monitor_true_residual': f':logs/wcsaddle_s_ksp_convergence.log',
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
elif args.pc == "aux":
    tao_parameters = aux_parameters

tao = TAOSolver(MinimizationProblem(Jhat),
                parameters=tao_parameters,
                options_prefix="")

Print()
xopt = tao.solve()

prior_ic = ensemble.bcast(prior.subfunctions[0], root=0)
prior_end = ensemble.bcast(prior.subfunctions[-1], root=ensemble.ensemble_size-1)
xopts_ic = ensemble.bcast(xopt.subfunctions[0], root=0)
xopts_end = ensemble.bcast(xopt.subfunctions[-1], root=ensemble.ensemble_size-1)
truth_ic = ensemble.bcast(truth.subfunctions[0], root=0)
truth_end = ensemble.bcast(truth.subfunctions[-1], root=ensemble.ensemble_size-1)
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
