from firedrake import *
import argparse

parser = argparse.ArgumentParser(
    description='Advection diffusion equation.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=50, help='Number of elements.')
parser.add_argument('--dt', type=float, default=1e-2, help='Number of elements.')
parser.add_argument('--umax', type=float, default=0.3, help='Initial scalar variation.')
parser.add_argument('--vprime', type=float, default=0.1, help='velocity perturbation.')
parser.add_argument('--re', type=float, default=50, help='Reynolds number.')
parser.add_argument('--nw', type=int, default=10, help='Number of observations stages.')
parser.add_argument('--obs_freq', type=int, default=5, help='Frequency of observations in time.')
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

# number of observation windows, and steps per window
nw, nt, nx = args.nw, args.obs_freq, args.nx
dt = args.dt
re, vprime = args.re, args.vprime
umax = args.umax

nu = 1/re
Nt = nw*nt
Tend = Nt*dt
cfl = dt*nx
Print(f"{nx=:>3d} | {nw=:>2d} | {Nt=:>3d} | {Tend=:.2e} | {cfl=:.2e} | {nu=:.2e}")

# 1D periodic mesh
mesh = PeriodicUnitIntervalMesh(nx)
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
velocity = Function(V).project(one + Constant(vprime)*cos(2*pi*x))

# finite element forms
gscale = Constant(1.0)
nuc = Constant(nu)
k = Constant(umax)
x1 = 1 - x
t1 = Constant(0.5)*t + 1

gscale = Constant(1.0)
xp = 2*pi*x
tp = 2*pi*t
kernel = (0.5*(1 - cos(xp)))**2
g = kernel*k*(
    - 2*sin(xp + (0.1*pi*sin(tp)))
    + k*cos(tp+1)*sin(3*xp - 2*tp)
)

F = (inner((un1 - un)/Constant(dt), v)*dx
     + inner(velocity, uh.dx(0))*v*dx
     + inner(nuc*grad(uh), grad(v))*dx
     - inner(gscale*g, v)*dx(degree=4)
)

solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
}

solver = NonlinearVariationalSolver(
    NonlinearVariationalProblem(F, un1),
    solver_parameters=solver_parameters,
    options_prefix="")

def solve_step():
    un1.assign(un)
    solver.solve()
    un.assign(un1)
    t.assign(t + dt)

# "ground truth" reference solution
reference_ic = Function(V).project(umax*sin(2*pi*x))

un.assign(reference_ic)
t.assign(0)
reference = [un.copy(deepcopy=True)]
forcing = [Function(V).interpolate(g)]
source = [assemble(forcing[-1]*dx)]
for _ in range(Nt):
    solve_step()
    reference.append(un.copy(deepcopy=True))
    forcing.append(Function(V).interpolate(g))
    source.append(assemble(forcing[-1]*dx))
    print(f"{float(t) = } | {norm(un) = }")
print(f"{sum(source) = }")

# from sys import exit; exit()

un.assign(reference_ic)
t.assign(0)
gscale.assign(0)
unforced = [un.copy(deepcopy=True)]
for _ in range(Nt):
    solve_step()
    unforced.append(un.copy(deepcopy=True))

from firedrake.output import VTKFile
vtk = VTKFile("outputs/advection.pvd")

mesh_out = UnitIntervalMesh(args.nx)
Vout = FunctionSpace(mesh_out, "CG", 1)

ur = Function(Vout, name="reference")
uf = Function(Vout, name="forcing")
ui = Function(Vout, name="inert")

for i, (ref, force, unf) in enumerate(zip(reference, forcing, unforced)):
    ur.interpolate(ref)
    uf.interpolate(force)
    ui.interpolate(unf)
    vtk.write(ur, uf, ui, time=float(i*dt))
