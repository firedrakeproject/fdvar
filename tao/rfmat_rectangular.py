import firedrake as fd
from firedrake.adjoint import (
    ReducedFunctional, Control, set_working_tape,
    continue_annotation, pause_annotation)
from fdvar.mat import ReducedFunctionalMat

mesh = fd.UnitIntervalMesh(3)
x, = fd.SpatialCoordinate(mesh)
expr = 0.5*x*x - 7

Vin = fd.FunctionSpace(mesh, "CG", 1)
Vout = fd.FunctionSpace(mesh, "DG", 1)

u = fd.TrialFunction(Vin)
v = fd.TestFunction(Vout)

A = (u*v + fd.inner(fd.grad(u), fd.grad(v)) + v*u.dx(0))*fd.dx

def tlm(y):
    return fd.assemble(fd.action(A, y)).riesz_representation()

def adj(y):
    return fd.assemble(fd.action(fd.adjoint(A), y)).riesz_representation()

continue_annotation()
with set_working_tape() as tape:
    x = fd.Function(Vin)
    Jhat = ReducedFunctional(tlm(x), Control(x), tape=tape)
pause_annotation()

def riesz(how):
    return {'riesz_representation': how}

mat_tlm = ReducedFunctionalMat(
    Jhat, action_type='tlm',
    options={'riesz_representation': None})

mat_adj = ReducedFunctionalMat(
    Jhat, action_type='adjoint',
    input_options={'riesz_representation': 'l2', 'function_space': Vout.dual()})

ctx_adj = mat_adj.getPythonContext()

print("tlm action")

# manual calculation
uin = fd.Function(Vin).project(expr)
uout = tlm(uin)
print(f"{type(uout) = }")
print(f"{uout.dat.data = }")

# re-evaluate RF
vin = fd.Function(Vin).project(expr)
vout = Jhat.tlm(vin, options={'riesz_representation': None})
print(f"{type(vout) = }")
print(f"{vout.dat.data = }")

# Mat action
win = fd.Function(Vin).project(expr)
wout = fd.Function(Vout)
with win.dat.vec_ro as x, wout.dat.vec_wo as y:
    mat_tlm.mult(x, y)
print(f"{type(wout) = }")
print(f"{wout.dat.data = }")

print()

print("adjoint action")

# manual calculation
uin = fd.Function(Vout).project(expr)
uout = adj(uin)
print(f"{type(uout) = }")
print(f"{uout.dat.data = }")

# re-evaluate RF
vin = fd.Function(Vout).project(expr)
vout = Jhat.derivative(adj_input=vin.riesz_representation())
print(f"{type(vout) = }")
print(f"{vout.dat.data = }")

# Mat action
win = fd.Function(Vout).project(expr).riesz_representation()
wout = fd.Function(Vin)
with win.dat.vec_ro as x, wout.dat.vec_wo as y:
    mat_adj.mult(x, y)

print(f"{type(wout) = }")
print(f"{wout.dat.data = }")

print()
