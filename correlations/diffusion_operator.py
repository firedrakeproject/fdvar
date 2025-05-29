from firedrake import *
import numpy as np
from scipy import sparse
import math

def float_formatter(x):
    return ("    0    " if x == 0
            else "{:+.2e}".format(x))

np.set_printoptions(
    linewidth=200, precision=2,
    formatter={'all': float_formatter})

nx = 32
sigma = sqrt(1e-1)
L = 0.3
M = 2

print(f"{nx=} | {sigma=:.2e} | {L=:.2e}")

mesh = PeriodicUnitIntervalMesh(nx)
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

kappa = (L*L)/(2*M)
# kappa = (L*L)/(4*M)
lambda_g = sqrt(2*math.pi)*L
lg_sqrt = sqrt(lambda_g)
scale = sigma/lg_sqrt

cfl = kappa*nx*nx
print(f"{kappa=:.2e} | {cfl=:.2e}")
print(f"{lambda_g=:.2e} | {lg_sqrt=:.2e} {scale=:.2e}")

kappa_c = Constant(kappa)
lambda_c = Constant(lambda_g)
lg_sqrt_c = Constant(lg_sqrt)
scale_c = Constant(scale)

#Dform = scale_c*(
#Dform = (
Dform = (
    inner(u, v)*dx
    - inner(kappa*grad(u), grad(v))*dx
)
Mform = inner(u, v)*dx

h = 1/nx
# h_c = Constant(h)
# D = (
#     (1/h_c)*inner(u, v)*dx
#     + Constant(h)*inner(grad(u), grad(v))*dx
# )

bcs = [DirichletBC(V, 0, "on_boundary")]

D_pmat = assemble(
    Dform, mat_type="aij",bcs=bcs).petscmat
M_pmat = assemble(
    Mform, mat_type="aij",bcs=bcs).petscmat

D_smat = sparse.csr_matrix(
    D_pmat.getValuesCSR()[::-1],
    shape=D_pmat.getSize())

M_smat = sparse.csr_matrix(
    M_pmat.getValuesCSR()[::-1],
    shape=M_pmat.getSize())

D = D_smat.todense()
# D2 = np.array(np.linalg.matrix_power(D, 2))
Dinv = np.linalg.inv(D)

M = M_smat.todense()
# M2 = np.array(np.linalg.matrix_power(M, 2))
Minv = np.linalg.inv(M)

# B = M @ Dinv @ M @ Dinv @ M
# B = M @ Dinv @ M
#B = lambda_g*(Dinv @ M @ Dinv)
B = lambda_g*((Minv @ D).T @ M @ (Minv @ D))*h*h*h
#B = (lambda_g)*(M @ Dinv @ Dinv @ M)/h
#print(12*kappa/(h*h))

#Binv = Minv @ D @ Minv @ D @ Minv
#Binv = np.linalg.inv(B)

print(np.diag(B))
