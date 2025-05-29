from firedrake import *
from firedrake.adjoint import *
import numpy as np
Print = PETSc.Sys.Print

mesh = PeriodicUnitIntervalMesh(10)
x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)

q = Function(V)
f = Function(V)

phi = TestFunction(V)

w = Constant(0.1)
eqn = (
    - w*inner(q*grad(q), grad(phi))*dx
    + inner(q, phi)*dx
    - inner(f, phi)*dx
)
params = {
    'snes_converged_reason': None,
    'snes_rtol': 1e-10,
}

stepper = NonlinearVariationalSolver(
    NonlinearVariationalProblem(eqn, q),
    options_prefix="", solver_parameters=params)

g = Function(V).project(0.6 + sin(2*pi*x))

continue_annotation()
with set_working_tape() as tape:
    q.assign(g)
    for _ in range(2):
        f.assign(q)
        stepper.solve()
        # solve(eqn == 0, q, solver_parameters=params)
    J = assemble((inner(q, q) + inner(g, g))*dx)
    Jhat = ReducedFunctional(J, Control(g), tape=tape)
pause_annotation()

dq = Function(V).project(0.5 + 0.1*cos(2*pi*(x - 1)))
dh = Function(V).project(0.4 + 0.1*sin(2*pi*(3*x - 3)))

taylor = taylor_to_dict(Jhat, dq, dh)
from pprint import pprint
pprint(taylor)
