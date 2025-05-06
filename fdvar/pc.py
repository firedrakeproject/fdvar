import firedrake as fd
from fdvar.mat import EnsembleMatCtxBase
from math import pi, sqrt

__all__ = ("SC4DVarBackgroundPC",)


class PCBase:
    needs_python_amat = False
    needs_python_pmat = False

    def __init__(self):
        self.initialized = False

    def setUp(self, pc):
        if not self.initialized:
            self.initialize(pc)
            self.initialized = True
        self.update(pc)

    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        self.A, self.P = pc.getOperators()
        pcname = f"{type(self).__module__}.{type(self).__name__}"
        if self.needs_python_amat:
            if self.A.getType() != "python":
                raise ValueError(
                    f"PC {pcname} needs a python type amat, not {self.A.getType()}")
            self.amat = self.A.getPythonContext()
        if self.needs_python_pmat:
            if self.P.getType() != "python":
                raise ValueError(
                    f"PC {pcname} needs a python type pmat, not {self.P.getType()}")
            self.pmat = self.P.getPythonContext()

        self.parent_prefix = pc.getOptionsPrefix()
        self.full_prefix = self.parent_prefix + self.prefix

    def update(self, pc):
        pass


class EnsemblePCBase(PCBase):
    requires_python_amat = True
    requires_python_pmat = True

    def __init__(self):
        self.initialized = False

    def initialize(self, pc):
        super().initialize(pc)

        if not isinstance(self.pmat, EnsembleMatCtxBase):
            pcname = f"{type(self).__module__}.{type(self).__name__}"
            matname = f"{type(self.pmat).__module__}.{type(self).pmat.__name__}"
            raise TypeError(
                f"PC {pname} needs an EnsembleMatCtxBase pmat, but it is a {matname}")

        self.row_space = self.pmat.row_space
        self.col_space = self.pmat.col_space

        self.x = fd.EnsembleFunction(self.row_space.dual())
        self.y = fd.EnsembleFunction(self.col_space)


class EnsembleBlockDiagonalPC(PCBase):
    prefix = "ebjacobi_"

    def initialize(self, pc):
        super().initialize(pc)

        ensemble_mat = self.pmat
        self.function_space = ensemble_mat.function_space
        self.ensemble = function_space.ensemble

        submats = ensemble_mat.blocks

        self.x = fd.EnsembleFunction(self.function_space)
        self.y = fd.EnsembleCofunction(self.function_space.dual())

        subksps = []
        for i, submat in enumerate(self.pmat.blocks):
            ksp = PETSc.KSP().create(comm=ensemble.comm)
            ksp.setOperators(submat)

            sub_prefix = self.parent_prefix + f"sub_{i}_"
            # TODO: default options
            options = OptionsManager({}, sub_prefix)
            options.set_from_options(ksp)
            self.subksps.append((ksp, options))

        self.subksps = tuple(subksps)

    def apply(self, pc, x, y):
        with self.x.vec_wo() as xvec:
            x.copy(xvec)

        subvecs = zip(self.x.subfunctions, self.y.subfunctions)
        for (subksp, suboptions), (subx, suby) in zip(self.subksps, subvecs):
            with subx.dat.vec_ro as rhs, suby.dat.vec_wo as sol:
                with suboptions.inserted_options():
                    subksp.solve(rhs, sol)

        with self.y.vec_ro() as yvec:
            yvec.copy(y)


class CovarianceMassPC(PCBase):
    prefix = "covmass_"
    pass


class CovarianceDiffusionPC(PCBase):
    prefix = "covdiff_"
    pass


class SC4DVarBackgroundPC(PCBase):
    prefix = "scbkg_"

    def initialize(self, pc):
        prefix = pc.getOptionsPrefix() or ""
        options_prefix = prefix + self.prefix
        A, _ = pc.getOperators()
        scfdv = A.getPythonContext().problem.reduced_functional

        self.covariance = scfdv.background_norm.covariance
        self.x = fd.Cofunction(self.covariance.V.dual())
        self.y = fd.Function(self.covariance.V)

        # # diffusion coefficient for lengthscale L
        # nu = 0.5*L*L
        # # normalisation for diffusion operator
        # lambda_g = L*sqrt(2*pi)
        # scale = lambda_g/sigma

        # V = scfdv.controls[0].control.function_space()
        # sol = fd.Function(V)
        # b = fd.Cofunction(V.dual())
        # u = fd.TrialFunction(V)
        # v = fd.TestFunction(V)
        # 
        # Binv = scale*(fd.inner(u, v) - nu*fd.inner(fd.grad(u), fd.grad(v)))*fd.dx

        # solver = fd.LinearVariationalSolver(
        #     fd.LinearVariationalProblem(
        #         Binv, b, sol, bcs=bcs,
        #         constant_jacobian=True),
        #     options_prefix=options_prefix,
        #     solver_parameters={'ksp_type': 'preonly',
        #                        'pc_type': 'lu'})

        # self.sol, self.b, self.solver = sol, b, solver

    def apply(self, pc, x, y):
        with self.x.dat.vec_wo as vx:
            x.copy(vx)
        self.covariance.action(self.x, tensor=self.y)
        with self.y.dat.vec_ro as vy:
        # with self.y.riesz_representation().dat.vec_ro as vy:
            vy.copy(y)

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError


class WC4DVarLTDLPC(PCBase):
    prefix = "ltdl_"

    def initialize(self, pc):
        self.fdvrf = self.pmat.fdvrf
        if not isinstance(self.Jhat, FourDVarReducedFunctional):
            raise TypeError(
                "AllAtOnceJacobiPC expects a FourDVarReducedFunctional not a {type(self.Jhat).__name__}")

    def apply(self, pc, x, y):
        with self.x.vec_wo() as xvec:
            x.copy(xvec)

        # P = LL^{T}

        # apply L^{T}
        ltx = self.Jhat.derivative

        with self.y.vec_ro() as yvec:
            yvec.copy(y)
