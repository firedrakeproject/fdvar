import firedrake as fd
from firedrake.adjoint import (
    ReducedFunctional, Control, set_working_tape,
    continue_annotation, pause_annotation)
from fdvar.mat import ReducedFunctionalMat

mesh = fd.UnitIntervalMesh(3)
x, = fd.SpatialCoordinate(mesh)
expr = 0.5*x*x - 7

V = fd.FunctionSpace(mesh, "CG", 1)

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

A = (u*v + fd.inner(fd.grad(u), fd.grad(v)) + v*u.dx(0))*fd.dx

def tlm(y):
    return fd.assemble(fd.action(A, y))

def adj(y):
    return fd.assemble(fd.action(fd.adjoint(A), y)).riesz_representation()

continue_annotation()
with set_working_tape() as tape:
    x = fd.Function(V)
    Jhat = ReducedFunctional(tlm(x), Control(x), tape=tape)
pause_annotation()

mat_tlm = ReducedFunctionalMat(
    Jhat, action_type='tlm')

mat_adj = ReducedFunctionalMat(
    Jhat, action_type='adjoint',
    input_options={'riesz_representation': 'l2', 'function_space': V})

print("tlm action")

# manual calculation
uin = fd.Function(V).project(expr)
uout = tlm(uin)
print(f"{type(uout) = }")
print(f"{uout.dat.data = }")

# re-evaluate RF
vin = fd.Function(V).project(expr)
vout = Jhat.tlm(vin)
print(f"{type(vout) = }")
print(f"{vout.dat.data = }")

# Mat action
win = fd.Function(V).project(expr)
wout = fd.Function(V.dual())
with win.dat.vec_ro as x, wout.dat.vec_wo as y:
    mat_tlm.mult(x, y)
print(f"{type(wout) = }")
print(f"{wout.dat.data = }")

print()

print("adjoint action")

# manual calculation
uin = fd.Function(V).project(expr)
uout = adj(uin)
print(f"{type(uout) = }")
print(f"{uout.dat.data = }")

# re-evaluate RF
vin = fd.Function(V).project(expr)
vout = Jhat.derivative(adj_input=vin)
print(f"{type(vout) = }")
print(f"{vout.dat.data = }")

# Mat action
win = fd.Function(V).project(expr)
wout = fd.Function(V)
with win.dat.vec_ro as x, wout.dat.vec_wo as y:
    mat_adj.mult(x, y)

print(f"{type(wout) = }")
print(f"{wout.dat.data = }")
