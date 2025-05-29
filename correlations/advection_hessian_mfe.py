from firedrake import *
from firedrake.adjoint import *

mesh = UnitIntervalMesh(16)
x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 2)

un = Function(V, name="un")
un1 = Function(V, name="un1")

dt = Constant(0.05)
nu = Constant(1/100)

v = TestFunction(V)
F = (
    (un1 - un)*v*dx
    + dt*inner(un, un.dx(0))*v*dx
    + dt*inner(nu*grad(un), grad(v))*dx
)

params = {
    "snes_rtol": 1e-12,
    "ksp_type": "preonly",
    "pc_type": "lu",
}

bcs = [DirichletBC(V, 0, "on_boundary")]

problem = LinearVariationalProblem(
    lhs(F), un1, bcs=bcs)

solver = NonlinearVariationalSolver(
    problem, solver_parameters=params)

ic = Function(V).project(1.0 + 0.1*sin(2*pi*x))
un1.assign(ic)

continue_annotation()
with set_working_tape() as tape:
    un1.assign(ic)

    un.assign(un1)
    solver.solve()
    # solve(F == 0, un1, bcs, solver_parameters=params)

    un.assign(un1)
    solver.solve()
    # solve(F == 0, un1, bcs, solver_parameters=params)

    un.assign(un1)

    J = assemble(inner(un, un)*dx)
    Jhat = ReducedFunctional(J, Control(ic), tape=tape)

pause_annotation()

dq = Function(V).interpolate(1.0 - 0.1*sin(4*pi*x))
dh = Function(V).interpolate(1.0 + 0.1*sin(6*pi*x))
taylor = taylor_to_dict(Jhat, dq, dh)
from pprint import pprint
pprint(taylor)
