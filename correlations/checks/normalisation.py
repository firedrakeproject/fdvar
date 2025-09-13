from firedrake import *
import numpy as np
from scipy import sparse
from scipy import linalg as spla
from numpy import linalg as npla
from math import factorial
from math import sqrt as msqrt
from matplotlib import pyplot as plt
from sys import exit
np.random.seed(31)

np.set_printoptions(
    linewidth=1000, threshold=10000,
    precision=4, legacy='1.25')

nsamples = 2000
n = 50

m = 6
L = 0.2

Bact = "implicit"
#Bact = "explicit"

kappa = L*L/(2*m)
lamda = msqrt(2*pi)*L

# kappa = L*L/m
# lamda = msqrt(4*pi*(m-1))*L

# kappa = L*L/m
# lamda = L*(2**(2*m-1))*(factorial(m-1)**2)/factorial(2*m-2)

lam_sqrt = msqrt(lamda)
cfl = kappa*n*n
print(f"{Bact = } | {m = :>2d}")
print(f"{n = :>4d} | {kappa = :.2e} | {cfl = :.2e}")
print(f"{L = :.2e} | {lamda = :.2e} | L/h={L*n:.2e}")

integral = 1/msqrt(4*pi*(m*kappa))
print(f"{integral = :.2e} | {lamda*integral = :.2e}")
print(f"{nsamples = :>6d}")

mesh = PeriodicUnitIntervalMesh(n)
x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)
# bcs = [DirichletBC(V, 0, "on_boundary")]
bcs = []

v = TestFunction(V)
w = Cofunction(V.dual())

kappa_c = Constant(kappa)
lamda_c = Constant(lamda)
lam_sqrt_c = Constant(sqrt(lamda_c))

u = TrialFunction(V)
M = inner(u, v)*dx
K = inner(kappa_c*grad(u), grad(v))*dx
D = M + K

diagM = assemble(M, diagonal=True).riesz_representation()
diagK = assemble(K, diagonal=True).riesz_representation()
diagD = assemble(D, diagonal=True).riesz_representation()

#Mhat = inner(u/diagM, v)*dx
#Khat = inner((kappa_c/diagK)*grad(u), grad(v))*dx
#
#Dhat = inner(u, v/diagD)*dx + inner(kappa_c*grad(u), grad(v)/diagD)*dx
#Dhat = Mhat + Khat

Mmat = assemble(M, mat_type="aij").petscmat
Mmat = sparse.csr_matrix(
    Mmat.getValuesCSR()[::-1],
    shape=Mmat.getSize()).todense()

Dmat = assemble(D, mat_type="aij").petscmat
Dmat = sparse.csr_matrix(
    Dmat.getValuesCSR()[::-1],
    shape=Dmat.getSize()).todense()

Msqrt = spla.sqrtm(Mmat).real
Msqrt_inv = npla.inv(Msqrt)

dM = np.diag(Mmat)
dD = np.diag(Dmat)
dM2 = np.diag(np.sqrt(dM))
dD2 = np.diag(np.sqrt(dD))
dM2i = np.diag(1/np.sqrt(dM))
dD2i = np.diag(1/np.sqrt(dD))

Minv = npla.inv(Mmat)
Dinv = npla.inv(Dmat)
DinvM = Dinv @ Mmat
MinvD = Minv @ Dmat

Mh = dM2i @ Mmat @ dM2i
Dh = dM2i @ dD2i @ Dmat @ dD2i @ dM2i

#Dinvh = npla.inv(Dh)
#Dinvh = dM2 @ dD2 @ npla.inv(Dmat) @ dD2 @ dM2

#Dfull = Dinvh @ Dinvh
#Dfull = np.eye(n)
Dfull = Minv
for _ in range(m):
    Dfull = DinvM @ Dfull
Dfull *= lamda
Dfinv = npla.inv(Dfull)

Lp = np.eye(n)
for _ in range(m//2):
    Lp = DinvM @ Lp
Lp *= sqrt(lamda)
Linv = npla.inv(Lp)

Lsqrt_inv = np.eye(n)
for _ in range(m//2):
    Lsqrt_inv = MinvD @ Lsqrt_inv
Lsqrt_inv /= sqrt(lamda)

#Dsqrt = Msqrt_inv
Dsqrt = dM2i
#Dsqrt = np.eye(n)
for _ in range(m//2):
    Dsqrt = DinvM @ Dsqrt
Dsqrt *= sqrt(lamda)

issymmetric = lambda A: spla.issymmetric(A, rtol=1e-8)

diagMinv = Function(V).project(1/diagM).riesz_representation()
# print(f"{np.mean(diagM.dat.data) = :.3e}")
# print(f"{np.mean(diagMinv.dat.data) = :.3e}")
# print(f"{np.mean(diagMinv.dat.data*dM) = :.3e}")
# print(f"{np.mean(dM) = :.3e}")
# print()
# print(f"{np.mean(diagD.dat.data) = :.3e}")
# print(f"{np.mean(dD) = :.3e}")
# print(f"{np.mean(np.diag(Dfull))   = :.3e}")
# print(f"{np.mean((1/dD)*dM*(1/dD)) = :.3e}")
# #print(f"{np.mean(np.diag(Mh))   = :.3e}")
# #print(f"{np.mean(np.diag(Dh))   = :.3e}")
# #print(f"{np.mean(np.diag(Dinvh))   = :.3e}")
# print(f"{issymmetric(Mmat) = }")
# print(f"{issymmetric(Dmat) = }")
# print(f"{issymmetric(Dinv) = }")
# print(f"{issymmetric(Dfull) = }")
# print()

# print(f"Dmat  =\n{np.diag(Dmat)}")
# print(f"Dfull =\n{np.diag(Dfull)}")

# print(f"{np.mean(np.diag(Mmat))  = :.3e} | {np.std(np.diag(Mmat))  = :.3e}")
# print(f"{np.mean(np.diag(DinvM)) = :.3e} | {np.std(np.diag(DinvM)) = :.3e}")
# print(f"{np.mean(np.diag(Dmat))  = :.3e} | {np.std(np.diag(Dmat))  = :.3e}")
# print(f"{np.mean(np.diag(Dfull)) = :.3e} | {np.std(np.diag(Dfull)) = :.3e}")
# 

#diagD = assemble(Dhat, diagonal=True)
#print(f"{diagD.dat.data = }")
#print(f"{diagD.riesz_representation().dat.data = }")
Di = M + K
De = M - K

u = Function(V).zero()
total = assemble(v*dx)

udat = u.dat.data
wdat = w.dat.data

ic = Function(V).project(sin(2*pi*x))

generator = Generator(PCG64(seed=31))
Msqrt = Function(V).project(sqrt(diagM))

A = D
rhs = Function(V)
sol = Function(V)
solver = LinearVariationalSolver(
    LinearVariationalProblem(
        A, inner(rhs, v)*dx, sol,
        bcs=bcs, constant_jacobian=True),
    solver_parameters={
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
)

Dact = action(De, rhs)

LiD2 = Linv @ Dsqrt
DfiD2 = Dfinv @ Dsqrt
L2iD2 = Lsqrt_inv @ Dsqrt

dM2inv = Function(V)
dM2inv.dat.data[:] = np.sqrt(1/assemble(M, diagonal=True).dat.data)
xsrc, xdst = Function(V), Function(V)
Ax = Action(A, xsrc)

def generate_samples():
    urand = generator.standard_normal(V)
    ubias = Function(V)
    unorm = Function(V)
    cofunc = Cofunction(V.dual())
    cofunc.dat.data[:] = urand.dat.data

    if Bact == "implicit":
        # sol.assign(cofunc.riesz_representation())

        # sol.dat.data[:] = Dsqrt @ urand.dat.data
        # sol.dat.data[:] = LiD2 @ urand.dat.data
        # sol.dat.data[:] = Linv @ urand.dat.data

        ubias.project(dM2inv*urand)
        #ubias.dat.data[:] = Dsqrt @ ubias.dat.data

        sol.assign(ubias)
        for _ in range(m//2):
            rhs.assign(sol)
            sol.zero()
            solver.solve()
        sol.assign(lam_sqrt_c*sol)
        ubias.assign(sol)

        # unorm.dat.data[:] = Lsqrt_inv @ ubias.dat.data
        xdst.assign(ubias)
        for _ in range(m//2):
            xsrc.assign(xdst)
            xdst.assign(assemble(Ax).riesz_representation())
        xdst.assign((1/lam_sqrt_c)*xdst)
        unorm.assign(xdst)

    elif Bact == "explicit":
        for _ in range(m//2):
            rhs.assign(cofunc.riesz_representation())
            assemble(Dact, tensor=cofunc)
        sol.assign(cofunc.riesz_representation())
    
    #w = Constant(lam_sqrt/sqrt(n))
    #w = diagM
    #w = Msqrt
    #w = Constant(1)
    #ucorr = Function(V).interpolate(w*sol)
    # ucorr = Function(V).assign(sol)
    return [urand, ubias, unorm]

# urand, ucorr = generate_samples()
# print(f"{norm(urand) = :.3e}")
# print(f"{norm(ucorr) = :.3e}")

nx = len(u.dat.data)
noise = np.empty((nx, nsamples))
bias = np.empty((nx, nsamples))
rnorms = np.empty(nsamples)
bnorms = np.empty(nsamples)
nnorms = np.empty(nsamples)
#for i in range(nsamples):
for i in progress_bar.ProgressBar("Sampling").iter(range(nsamples)):
    ur, ub, un = generate_samples()

    noise[:,i] = ur.dat.data
    bias[:,i] = ub.dat.data

    # rnorms[i] = norm(ur)**2
    # bnorms[i] = norm(ub)**2
    # nnorms[i] = norm(un)**2

    rnorms[i] = assemble(inner(ur, ur)*dx)
    bnorms[i] = assemble(inner(ub, ub)*dx)
    nnorms[i] = assemble(inner(un, un)*dx)
print()

print(f"{np.mean(rnorms) = :.2e} | {np.std(rnorms) = :.2e}")
print(f"{np.mean(bnorms) = :.2e} | {np.std(bnorms) = :.2e}")
print(f"{np.mean(nnorms) = :.2e} | {np.std(nnorms) = :.2e}")
print()

noise_cov = np.cov(noise)
bias_cov = np.cov(bias)

print(f"{np.mean(noise) = :.2e} | {np.std(noise) = :.2e}")
print(f"{np.mean(bias)  = :.2e} | {np.std(bias)  = :.2e}")
print(f"{np.mean(np.std(bias, axis=0)) = :.2e}")
nk = max(int(L*n), 1)
pad0 = nk//2
pad1 = nk - pad0

ndiag = np.diag(noise_cov)
bdiag = np.diag(bias_cov)
bdiagk = np.diag(bias_cov, k=nk)
print(f"{np.mean(ndiag) = :.2e} | {np.std(ndiag) = :.2e}")
print()

print(f"{np.mean(bdiagk/bdiag[pad0:-pad1]) = :.2e}")
print(f"{np.std(bdiagk/bdiag[pad0:-pad1]) = :.2e}")
print(f"{np.mean(bdiag) = :.2e} | {np.std(bdiag) = :.2e}")
print()

exit()

#plt.spy(bias_cov)
#plt.imshow(bias_cov, cmap='hot', interpolation='nearest')
#plt.show()

exit()

ic = Function(V).zero()
ic.dat.data[n//2] = 1

delta = Cofunction(V.dual()).zero()
delta.dat.data[n//2] = 1

randf = generator.uniform(V)
delta.dat.data[:] = randf.dat.data[:]
ic.assign(delta.riesz_representation())

ic.assign(randf)
delta = ic.riesz_representation()


print("Implicit step")
u.project(ic)
w.assign(u.riesz_representation())
print(f"{np.max(udat) = :.4e}")
print(f"{np.max(wdat) = :.4e}")
print(f"{np.std(udat) = :.4e}")
print(f"{np.std(wdat) = :.4e}")
print(f"np.cov(udat) = \n{np.cov(udat)}")
print()
solve((1/lamda_c)*Di==w, u)
w.assign(u.riesz_representation())
print(f"{np.max(udat) = :.4e}")
print(f"{np.max(wdat) = :.4e}")
print(f"{np.std(udat) = :.4e}")
print(f"{np.std(wdat) = :.4e}")
print(f"np.cov(udat) = \n{np.cov(udat)}")
print()
print(f"{assemble(total(ic)) = :.2e}")
print(f"{assemble(total(u)) = :.2e}")
print()

# print("Explicit step")
# u.project(ic)
# w.assign(u.riesz_representation())
# print(f"{np.max(udat) = :.4e}")
# print(f"{np.max(wdat) = :.4e}")
# assemble(action(De, u), tensor=w)
# u.assign(w.riesz_representation())
# print(f"{np.max(udat) = :.4e}")
# print(f"{np.max(wdat) = :.4e}")
# print(f"{assemble(total(u)) = :.2e}")
