from firedrake import *
from firedrake.adjoint import *
import numpy as np
np.random.seed(6)
Print = PETSc.Sys.Print

nx = 32
nt = 20
T = 0.8
dt = T/nt
velocity = 1
cfl = velocity*dt*nx
Print(f"{cfl = :.3e}")

mesh = PeriodicUnitIntervalMesh(nx)
x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "DG", 1)

qn = Function(V, name="qn")
qn1 = Function(V, name="qn1")

def mass(q, phi):
    return inner(q, phi)*dx

def tendency(q, phi):
    u = Constant(as_vector([velocity]))
    n = FacetNormal(mesh)
    un = Constant(0.5)*(dot(u, n) + abs(dot(u, n)))
    return (- q*div(phi*u)*dx
            + jump(phi)*jump(un*q)*dS)

# midpoint rule
phi = TestFunction(V)

qh = Constant(0.5)*(qn1 + qn)
eqn = mass(qn1 - qn, phi) + Constant(dt)*tendency(qh, phi)

# stepper = NonlinearVariationalSolver(
#     NonlinearVariationalProblem(eqn, qn1))

def advance():
    solve(eqn==0, qn1)

def correlation_norm(sigma, x):
    return assemble(inner(sigma*x, x*sigma)*dx)

ic = Function(V).project(sin(2*pi*x))
bkg = Function(V).project(sin(2*pi*x))
target = Function(V).project(sin(2*pi*(x - T)))

B = 1e1
R = 2e0

ic.dat.data[:] += np.random.normal(
    0, B, ic.dat.data.shape)
bkg.dat.data[:] += np.random.normal(
    0, B, bkg.dat.data.shape)
target.dat.data[:] += np.random.normal(
    0, R, target.dat.data.shape)

continue_annotation()
with set_working_tape() as tape:
    qn.assign(ic)
    qn1.assign(qn)
    J = correlation_norm(B, qn - bkg)
    for i in range(nt):
        # stepper.solve()
        advance()
        qn.assign(qn1)
    J += correlation_norm(R, qn - target)
    Jhat = ReducedFunctional(J, Control(ic), tape=tape)
pause_annotation()
final = qn.copy(deepcopy=True)

# dq = Function(V).project(cos(2*pi*(x + 1)))
# assert taylor_test(Jhat, ic, dq) > 1.95
# taylor = taylor_to_dict(Jhat, ic, dq)
# from pprint import pprint
# pprint(taylor)
# from sys import exit; exit()

class BackgroundPC(PCBase):
    def initialize(self, pc):
        self.x = Function(V.dual())
        self.y = Function(V)

        u = TrialFunction(V)
        v = TestFunction(V)
        M = inner(B*u, v*B)*dx

        prefix = pc.getOptionsPrefix() + "bkg_"

        self.solver = LinearVariationalSolver(
            LinearVariationalProblem(
                M, self.x, self.y,
                constant_jacobian=True),
            solver_parameters={'ksp_type': 'preonly',
                               'pc_type': 'lu'},
            options_prefix=prefix)
    
    def apply(self, pc, x, y):
        with self.x.dat.vec_wo as xvec:
            x.copy(xvec)

        self.solver.solve()

        with self.y.dat.vec_wo as yvec:
            yvec.copy(y)

    def update(self, pc, x):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
    

tao_params = {
    'tao_view': ':tao_view.log',
    'tao_monitor': None,
    'tao_converged_reason': None,
    'tao_gttol': 1e-4,
    'tao_type': 'nls',
    'tao_nls': {
        'ksp_view': ':ksp_view.log',
        'ksp_monitor_short': None,
        'ksp_converged_rate': None,
        'ksp_converged_maxits': None,
        'ksp_max_it': 10,
        'ksp_rtol': 1e-2,
        'ksp_type': 'cg',
        'pc_type': 'none',
        'pc_type': 'python',
        'pc_python_type': f'{__name__}.BackgroundPC',
    },
}
tao = TAOSolver(MinimizationProblem(Jhat),
                parameters=tao_params,
                options_prefix="")
ic_opt = tao.solve()

qn.assign(ic_opt)
qn1.assign(qn)
for i in range(nt):
    # stepper.solve()
    advance()
    qn.assign(qn1)
final_opt = qn.copy(deepcopy=True)

Print(f"{errornorm(bkg, ic)           = :.3e}")
Print(f"{errornorm(bkg, ic_opt)       = :.3e}")
Print(f"{errornorm(target, final)     = :.3e}")
Print(f"{errornorm(target, final_opt) = :.3e}")
