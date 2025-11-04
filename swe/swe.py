from functools import partial

from firedrake import *
from firedrake.output import VTKFile
from utils.shallow_water import galewsky
from utils import shallow_water as swe
from utils.planets import earth
from utils import units

Print = PETSc.Sys.Print

ref = 4
dt = 1.0  # hours
nt = 240

mesh = swe.create_mg_globe_mesh(
    ref_level=ref, coords_degree=1)
x = SpatialCoordinate(mesh)

R = FunctionSpace(mesh, "R", 0)
Vu = FunctionSpace(mesh, "BDM", 2)
Vd = FunctionSpace(mesh, "DG", 1)
W = Vu*Vd
Print(f"{W.dim() = } | {nt = } | {dt = :.2e} | days = {nt*dt/24:.2f}")

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

w1 = Function(W).assign(ic)
w0 = Function(W).assign(ic)

u1, h1 = split(w1)
u0, h0 = split(w0)
v, q = TestFunctions(W)

uh = 0.5*(u0 + u1)
hh = 0.5*(h0 + h1)
dT = Constant(dt*units.hour)

F = (
    form_mass(u1-u0, h1-h0, v, q)
    + dT*form_function(uh, hh, v, q, t)
)

atol = 1e4
solver_parameters = {
    'snes_rtol': 1e-6,
    'snes_atol': atol,
    'snes_lag_preconditioner': 1000,
    'ksp_rtol': 1e-4,
    'ksp_type': 'gmres',
    'pc_type': 'ilu',
}

solver = NonlinearVariationalSolver(
    NonlinearVariationalProblem(F, w1),
    solver_parameters=solver_parameters,
    options_prefix="")

vtkfile = VTKFile("output/swe.pvd")
wout = Function(W)
uout, hout = wout.subfunctions
uout.rename("velocity")
hout.rename("depth")

def write(t):
    wout.assign(w1)
    hout.assign(hout - galewsky.H0)
    vtkfile.write(uout, hout, time=t)

write(t=0)

for i in ProgressBar("Timestep").iter(range(nt)):
    solver.solve()
    w0.assign(w1)
    t += dT
    write(t=i*dt)
