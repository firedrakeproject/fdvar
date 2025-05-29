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
parser.add_argument('--Btype', type=str, default="diffusion", choices=("mass", "diffusion"), help='Type of background correlation operator.')
parser.add_argument('--Bm', type=int, default=2, help='Number of form applications for Background correlation. Must be even.')
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
parser.add_argument('--saddle', action='store_true', help='Run saddle point dev section.')
parser.add_argument('--taylor_test', action='store_true', help='Run Taylor test instead of optimisation.')
parser.add_argument('--plot_vtk', action='store_true', help='Plot results after optimisation.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

Print = PETSc.Sys.Print
np.set_printoptions(legacy='1.25', precision=3, linewidth=200)
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

def H(x):  # operator to take observations
    return assemble(interpolate(x, Y))

# Background correlation operator
seed_b = 2 + erank  # 2nd letter
if args.Btype == "mass":
    Bscale = rand_func(V, sigma=sigma_b, lim=args.lim_b, dist="symmetric", seed=seed_b-1)
    B = ExplicitMassCorrelation(V, Bscale, m=args.Bm, seed=seed_b)
elif args.Btype == "diffusion":
    B = ImplicitDiffusionCorrelation(V, sigma_b, args.L_b, m=args.Bm, seed=seed_b)

# Model correlation operator
seed_q = 17 + erank  # 17th letter
if args.Qtype == "mass":
    Qscale = rand_func(V, sigma=sigma_q, lim=args.lim_q, dist="symmetric", seed=seed_q-1)
    Q = ExplicitMassCorrelation(V, Qscale, m=args.Qm, seed=seed_q)
elif args.Qtype == "diffusion":
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
control = Function(V).assign(background)
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
        # print(f"{erank = } | {j = } | {jg = }")
        tsub = truth.subfunctions[j]
        tsub.assign(ground_truth[jg])
else:
    for j in range(nlocal_stages):
        jg = (j+1)*nt
        # print(f"{erank = } | {j = } | {jg = }")
        tsub = truth.subfunctions[j]
        tsub.assign(ground_truth[jg])

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
prior = Jhat.control.control.copy()

def noisify_efunc(normalise=False, scale=1.0):
    xnoise = EnsembleFunction(W).zero()
    xns = xnoise.subfunctions
    # print(f"{float(B.lamda) = :.3e}")
    # print(f"{float(Q.lamda) = :.3e}")

    if erank == 0:
        bnoise = B.correlated_noise()
        if normalise:
            bnoise /= B.sigma
        bnoise *= scale
        xns[0].assign(bnoise)
        print(f"{norm(bnoise) = :.3e} | {np.mean(bnoise.dat.data) = :.3e}")

    stage_values = xns[1:] if (erank==0) else xns
    for xi in stage_values:
        qnoise = Q.correlated_noise()
        if normalise:
            qnoise /= Q.sigma
        qnoise *= scale
        xi.assign(qnoise)
        print(f"{norm(qnoise) = :.3e} | {np.mean(qnoise.dat.data) = :.3e}")
    return xnoise

# Print(f"{Jhat.Jmodel(truth) = }")
# Print(f"{Jhat.Jobservations(truth) = }")
# Print(f"{Jhat(truth) = }")
# Print(f"{Jhat.Jmodel(prior) = }")
# Print(f"{Jhat.Jobservations(prior) = }")
# Print(f"{Jhat(prior) = }")
# Print(f"{Jhat(truth) = }")
# xorig = noisify_efunc(prior.copy())
# Print(f"{Jhat.Jmodel(xorig) = }")
# Print(f"{Jhat.Jobservations(xorig) = }")
# Print(f"{Jhat(xorig) = }")

if args.taylor_test:
    from pyadjoint.verification import taylor_to_dict
    from pprint import pprint
    # h0 = prior
    h0 = prior.copy() + noisify_efunc(scale=0.01)
    dh = noisify_efunc()

    Print(f"{Jhat.Jmodel(h0) = }")
    Print(f"{Jhat.Jobservations(h0) = }")
    Print(f"{Jhat(h0) = }")

    Print("\nTaylor test: Jhat.Jmodel")
    taylor = taylor_to_dict(Jhat.Jmodel, h0, dh)
    if COMM_WORLD.rank == 0:
        pprint(taylor)

    Print("\nTaylor test: Jhat.Jobservations")
    taylor = taylor_to_dict(Jhat.Jobservations, h0, dh)
    if COMM_WORLD.rank == 0:
        pprint(taylor)

    Print("\nTaylor test: Jhat")
    taylor = taylor_to_dict(Jhat, h0, dh)
    if COMM_WORLD.rank == 0:
        pprint(taylor)

    Print(f"{min(taylor['R0']['Rate']) = :.4e}")
    Print(f"{min(taylor['R1']['Rate']) = :.4e}")
    Print(f"{min(taylor['R2']['Rate']) = :.4e}")
    exit()

if args.saddle:
    # test setting up saddle point system (SPS)
    from fdvar.saddle import *
    rank = COMM_WORLD.rank

    saddle_ksp = WC4DVarSaddlePointKSP(Jhat)
    saddle_mat, _ = saddle_ksp.getOperators()

    Wc = Jhat.control_space
    Wo = Jhat.observation_space

    Dmat, Rmat, Lmat, LTmat, Hmat, HTmat = getSubWC4DVarSaddlePointMat(saddle_mat)

    Lksp = PETSc.KSP().create(comm=ensemble.global_comm)
    Lksp.setOperators(Lmat)
    Lparams = {
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'ksp_rtol': 1e-5,
        'ksp_type': 'richardson',
        'pc_type': 'none',
    }
    attach_options(Lksp, Lparams, options_prefix="L")
    set_from_options(Lksp)

    x, b = Lmat.createVecs()
    b.setRandom()
    # with inserted_options(Lksp):
    #     Lksp.solve(b, x)

    Dksp = PETSc.KSP().create(comm=ensemble.global_comm)
    Dksp.setOperators(Dmat)
    Dparams = {
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'ksp_rtol': 1e-10,
        'ksp_type': 'richardson',
        'pc_type': 'python',
        'pc_python_type': 'fdvar.EnsembleBJacobiPC',
        'sub_ksp_monitor': None,
        'sub_pc_type': 'python',
        'sub_pc_python_type': 'fdvar.CorrelationOperatorPC',
    }
    attach_options(Dksp, Dparams, options_prefix="D")
    set_from_options(Dksp)

    x, b = Dmat.createVecs()
    b.setRandom()
    with inserted_options(Dksp):
        Dksp.solve(b, x)

    exit()

    vec = PETSc.Vec().createNest(
        vecs=(
            Wc.layout_vec.duplicate(),
            Wo.layout_vec.duplicate(),
            Wc.layout_vec.duplicate()),
        isets=saddle_mat.getNestISs()[0],
        comm=ensemble.global_comm)

    x = vec.duplicate()
    yb = vec.duplicate()
    yf = vec.duplicate()
    y0 = vec.duplicate()
    y1 = vec.duplicate()

    x.setRandom()
    yb.zeroEntries()
    yf.zeroEntries()
    y0.zeroEntries()
    y1.zeroEntries()

    x_dn, x_dl, x_dx = x.getNestSubVecs()
    y0_dn, y0_dl, y0_dx = y0.getNestSubVecs()
    y1_dn, y1_dl, y1_dx = y1.getNestSubVecs()

    # top row
    Dmat.mult(x_dn, y0_dn)
    Lmat.mult(x_dx, y1_dn)

    # middle row
    Rmat.mult(x_dl, y0_dl)
    Hmat.mult(x_dx, y1_dl)

    # bottom row
    LTmat.mult(x_dn, y0_dx)
    HTmat.mult(x_dl, y1_dx)

    # sum contributions
    yb_dn, yb_dl, yb_dx = yb.getNestSubVecs()
    yb_dn.waxpy(1, y0_dn, y1_dn)
    yb_dl.waxpy(1, y0_dl, y1_dl)
    yb_dx.waxpy(1, y0_dx, y1_dx)

    # full matrix
    saddle_mat.mult(x, yf)

    assert np.allclose(yf.array, yb.array)
    Print("WC4DVarSaddlePointMat check passed")
    exit()

# Solution strategy is controlled via this options dictionary
none_parameters = {
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
        'pc_type': 'none',
    },
}

lits = nw+1
schur_parameters = {
    'tao_view': ':tao_view.log',
    'tao_monitor': None,
    'tao_max_it': 20,
    # 'tao_ls_type': 'unit',
    # 'tao_ls_monitor': None,
    'tao_converged_reason': None,
    'tao_gttol': 1e-2,
    'tao_grtol': 1e-6,
    'tao_gatol': 0,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_view_eigenvalues': None,
        #'ksp_monitor_singular_value': None,
        'ksp_monitor': None,
        'ksp_converged_rate': None,
        'ksp_converged_maxits': None,
        'ksp_max_it': 20,
        'ksp_rtol': 4e-2,
        'ksp_type': 'cg',
        'pc_type': 'python',
        'pc_python_type': 'fdvar.WC4DVarSchurPC',
        'wcschur_l': {
            'ksp_monitor': ':wcschur_l_ksp_convergence.log',
            'ksp_convergence_test': 'skip',
            'ksp_converged_maxits': None,
            'ksp_type': 'richardson',
            'ksp_max_it': lits,
        },
        'wcschur_lt': {
            'ksp_monitor': ':wcschur_lt_ksp_convergence.log',
            'ksp_convergence_test': 'skip',
            'ksp_converged_maxits': None,
            'ksp_type': 'richardson',
            'ksp_max_it': lits,
        },
        'wcschur_d': {
            'ksp_monitor': ':wcschur_d_ksp_convergence.log',
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'fdvar.EnsembleBJacobiPC',
            'sub_pc_type': 'python',
            'sub_pc_python_type': 'fdvar.CorrelationOperatorPC',
        },
    },
}

saddle_parameters = {
    'tao_view': ':tao_view.log',
    'tao_monitor': None,
    'tao_max_it': 20,
    # 'tao_ls_type': 'unit',
    # 'tao_ls_monitor': None,
    'tao_converged_reason': None,
    'tao_gttol': 1e-2,
    'tao_grtol': 1e-6,
    'tao_gatol': 0,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_monitor': None,
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'fdvar.WC4DVarSaddlePointPC',
        'wcsaddle': {
            'ksp_view': ':wcsaddle_ksp_view.log',
            'ksp_view_eigenvalues': ':wcsaddle_eigenvalues.log',
            'ksp_monitor_singular_value': ':wcsaddle_singular_value.log',
            'ksp_monitor': None,
            'ksp_converged_rate': None,
            'ksp_max_it': 1,
            'ksp_converged_maxits': None,
            'ksp_rtol': 1e-4,
            'ksp_type': 'gmres',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'diag',
            'pc_fieldsplit_0_fields': '0,1',
            'pc_fieldsplit_1_fields': '2',
            'fieldsplit_0': {
                'ksp_monitor': None,
                'ksp_converged_reason': None,
                'ksp_type': 'preonly',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'additive',
                'fieldsplit': {
                    # '0_ksp_monitor': ':wcsaddle_d_ksp_convergence.log',
                    # '1_ksp_monitor': ':wcsaddle_r_ksp_convergence.log',
                    'ksp_monitor': None,
                    'ksp_type': 'preonly',
                    'pc_type': 'python',
                    'pc_python_type': 'fdvar.EnsembleBJacobiPC',
                    'sub_pc_type': 'python',
                    'sub_pc_python_type': 'fdvar.CorrelationOperatorPC',
                },
            },
            'fieldsplit_1': {
                # 'ksp_monitor': ':wcsaddle_s_ksp_convergence.log',
                'ksp_monitor': None,
                'ksp_converged_reason': None,
                'ksp_type': 'preonly',
                'pc_type': 'python',
                'pc_python_type': 'fdvar.WC4DVarSchurPC',
                'wcschur': {
                    'ltlm_ksp_monitor': ':wcschur_l_ksp_convergence.log',
                    'ltlm_ksp_converged_reason': None,
                    'ltlm_ksp_monitor': None,
                    'ladj_ksp_monitor': ':wcschur_lt_ksp_convergence.log',
                    'ladj_ksp_converged_reason': None,
                    'ladj_ksp_monitor': None,
                    'd_ksp_monitor': ':wcschur_d_ksp_convergence.log',
                    'd_ksp_converged_reason': None,
                    'd_ksp_monitor': None,
                },
                'wcschur_ltlm': {
                    'ksp_view': ':wschur_ltlm_ksp_view.log',
                    'ksp_convergence_test': 'skip',
                    'ksp_converged_maxits': None,
                    'ksp_type': 'richardson',
                    'ksp_max_it': lits,
                },
                'wcschur_ladj': {
                    'ksp_view': ':wschur_ladj_ksp_view.log',
                    'ksp_convergence_test': 'skip',
                    'ksp_converged_maxits': None,
                    'ksp_type': 'richardson',
                    'ksp_max_it': lits,
                },
                'wcschur_d': {
                    'ksp_view': ':wschur_d_ksp_view.log',
                    'ksp_type': 'preonly',
                    'pc_type': 'python',
                    'pc_python_type': 'fdvar.EnsembleBJacobiPC',
                    'sub_ksp_type': 'preonly',
                    'sub_pc_type': 'python',
                    'sub_pc_python_type': 'fdvar.CorrelationOperatorPC',
                },
            },
        },
    },
}

tao_parameters = saddle_parameters
tao = TAOSolver(MinimizationProblem(Jhat),
                parameters=tao_parameters,
                options_prefix="")

# from pyadjoint.optimization.tao_solver import ReducedFunctionalMat
# Jhat(truth)
# tao = TAOSolver(MinimizationProblem(Jhat.Jmodel),
#                 parameters=tao_parameters,
#                 options_prefix="",
#                 Pmat=ReducedFunctionalMat(Jhat))

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

exit()

un.assign(reference_ic)
t.assign(0)
reference = [un.copy(deepcopy=True)]
for _ in range(Nt):
    solve_step()
    reference.append(un.copy(deepcopy=True))
Print(f"{len(reference) = }")

un.assign(background)
t.assign(0)
priors = [un.copy(deepcopy=True)]
forcing = [Function(V).interpolate(g)]
for _ in range(Nt):
    solve_step()
    priors.append(un.copy(deepcopy=True))
    forcing.append(Function(V).interpolate(g))
Print(f"{len(priors) = }")
Print(f"{len(forcing) = }")

t.assign(0)
opts = []
for i in range(nw):
    un.assign(xopt.subfunctions[i])
    for j in range(nt):
        opts.append(un.copy(deepcopy=True))
        solve_step()
opts.append(un.copy(deepcopy=True))
Print(f"{len(opts) = }")
Print(f"{len(ground_truth) = }")

prior_ic = priors.subfunctions[0]
prior_end = priors.subfunctions[-1]
xopts_ic = xopt.subfunctions[0]
xopts_end = xopt.subfunctions[-1]
truth_ic = truth.subfunctions[0]
truth_end = truth.subfunctions[-1]
Print(f"{norm(prior_ic) = :.3e} | {norm(prior_end) = :.3e}")
Print(f"{norm(xopts_ic) = :.3e} | {norm(xopts_end) = :.3e}")
Print(f"{norm(truth_ic) = :.3e} | {norm(truth_end) = :.3e}")
Print(f"{errornorm(prior_ic, xopts_ic)/norm(prior_ic) = :.3e}")
Print(f"{errornorm(truth_ic, prior_ic)/norm(truth_ic) = :.3e}")
Print(f"{errornorm(truth_ic, xopts_ic)/norm(truth_ic) = :.3e}")
Print(f"{errornorm(truth_end, ref_end)/norm(truth_end) = :.3e}")
Print(f"{errornorm(truth_end, bkg_end)/norm(truth_end) = :.3e}")
Print(f"{errornorm(truth_end, opt_end)/norm(truth_end) = :.3e}")
Print(f"{Jhat(prior)  = :.3e}")
Print(f"{Jhat(xopt)   = :.3e}")

if args.plot_vtk:
    from firedrake.output import VTKFile
    vtk = VTKFile("outputs/advection_wc4dvar.pvd", comm=ensemble.comm)

    mesh_out = UnitIntervalMesh(args.nx, comm=ensemble.comm)
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
