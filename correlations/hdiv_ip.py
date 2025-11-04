from firedrake import *
Print = PETSc.Sys.Print

def ip_diffusion(u, v, alpha, gamma):
    mesh = u.function_space().mesh()
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg = 0.5*(h('+') + h('-'))
    alpha_h = alpha/h_avg
    gamma_h = gamma/h
    return (
        inner(grad(u), grad(v))*dx
        - inner(jump(u, n), avg(grad(v)))*dS
        - inner(avg(grad(u)), jump(v, n))*dS
        + alpha_h*inner(jump(u, n), jump(v, n))*dS
        - inner(u*n, grad(v))*ds
        - inner(grad(u), v*n)*ds
        + gamma_h*inner(u, v)*ds
    )


def cg_diffusion(u, v):
    return inner(grad(u), grad(v))*dx

nx = 64
mesh = UnitSquareMesh(nx, nx)
x, y = SpatialCoordinate(mesh)

rg = RandomGenerator()

CG = FunctionSpace(mesh, "CG", 1)
DG = FunctionSpace(mesh, "DG", 1)

alpha = Constant(4.0)
gamma = Constant(8.0)

f = -0.5*pi*pi*(4*cos(pi*x) - 5*cos(pi*x*0.5) + 2)*sin(pi*y)
uexact = sin(pi*x)*tan(pi*x*0.25)*sin(pi*y)

dg_coefficients = (TrialFunction(DG), TestFunction(DG))
a_dg = ip_diffusion(*dg_coefficients, alpha, gamma)
L_dg = inner(f, TestFunction(DG))*dx(degree=6)

cg_coefficients = (TrialFunction(CG), TestFunction(CG))
a_cg = cg_diffusion(*cg_coefficients)
L_cg = inner(f, TestFunction(CG))*dx(degree=6)
bc_cg = DirichletBC(CG, 0, "on_boundary")

params_dg = {
    'ksp_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

# solve DG
u_dg = Function(DG)
lvs_dg = LinearVariationalSolver(
    LinearVariationalProblem(a_dg, L_dg, u_dg),
    solver_parameters=params_dg,
    options_prefix="dg")
lvs_dg.solve()

params_cg = {
    'ksp_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

# solve CG
u_cg = Function(CG)
lvs_cg = LinearVariationalSolver(
    LinearVariationalProblem(a_cg, L_cg, u_cg, bcs=bc_cg),
    solver_parameters=params_cg,
    options_prefix="cg")
lvs_cg.solve()

Print(f"{norm(u_cg)/norm(uexact) = :.8e}")
Print(f"{norm(u_dg)/norm(uexact) = :.8e}")
Print(f"{errornorm(uexact, u_cg)/norm(uexact) = :.8e}")
Print(f"{errornorm(uexact, u_dg)/norm(uexact) = :.8e}")
