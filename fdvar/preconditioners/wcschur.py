import petsctools
from pyadjoint.optimization.tao_solver import (
    ReducedFunctionalMat,
    RFOperation
)
from firedrake.adjoint import CovarianceMat
from firedrake import (
    EnsembleBlockDiagonalMat,
    PETSc,
)
from fdvar.wc4dvar_reduced_functional import WC4DVarReducedFunctional


class WC4DVarSchurPC(petsctools.PCBase):
    """
    Preconditioner to approximate the inverse of the Schur complement
    of the saddle point formulation of the weak constraint 4DVar, which is
    equivalent to the Gauss-Newton Hessian of the primal WC4DVar formulation.

    The exact Schur complement :math:`S` and the approximation
    :math:`\\tilde{S}` that this PC applies are:

    .. math::

      S & = L^{T}D^{-1}L + H^{T}R^{-1}H

      \\tilde{S}^{-1} & = \\tilde{L}^{-T}\\tilde{D}\\tilde{L}^{-1}

    where :math:`L` is the all-at-once system, and H, D, and R are the
    block-diagonal matrices with the observation operators, observation
    error covariances, and model error covariances at each observation
    time respectively.

    KSPs are created for; :math:`\\tilde{L}`, for :math:`\\tilde{L}^{-T}` using
    a :func:`~pyadjoint.optimization.tao_solver.ReducedFunctionalMat` for the
    :class:`~firedrake.adjoint.allatonce_reduced_functional.AllAtOnceReducedFunctional`;
    and for :math:`\\tilde{D}` using an :class:`~firedrake.ensemble.ensemble_mat.EnsembleBlockDiagonalMat`
    where each block is a :func:`~firedrake.adjoint.covariance_operator.CovarianceMat`.

    PETSc Options
    -------------
    * ``-wcschur_l`` - Options for solving the :math:`L` and :math:`L^{T}`.
    * ``-wcschur_ltlm`` - Options solely for :math:`L`, e.g. monitors.
    * ``-wcschur_ladj`` - Options solely for :math:`L^{T}`, e.g. monitors.
    * ``-wcschur_d`` - Options for solving the :math:`D^{-1}`

    Notes
    -----
    Identical solver options should be used for :math:`\\tilde{L}` and
    :math:`\\tilde{L}^{T}` to ensure symmetry of :math:`\\tilde{S}^{-1}``.

    References
    ----------
    Fisher M. and Gurol S., 2017: "Parallelization in the time dimension of
    four-dimensional variational data assimilation".
    Q.J.R. Meteorol. Soc. 142: 1136–1147, DOI:10.1002/qj.2997

    See Also
    --------
    ~firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional
    ~firedrake.adjoint.allatonce_reduced_functional.AllAtOnceReducedFunctional
    ~firedrake.ensemble.ensemble_mat.EnsembleBlockDiagonalMat
    ~firedrake.adjoint.covariance_operator.CovarianceMat
    """

    prefix = "wcschur_"

    @PETSc.Log.EventDecorator()
    def initialize(self, pc):
        # TODO: petsctools.cite("Fisher2017")
        super().initialize(pc)

        A, P = pc.getOperators()

        Jhat = self._get_wc4dvar_rf(P)
        self.Jhat = Jhat

        self.ensemble = Jhat.ensemble
        global_comm = self.ensemble.global_comm

        self.col_space = Jhat.control_space
        self.row_space = Jhat.control_space.dual()

        # Create the Mats for each component

        LTmat_p, Dmat_p, Lmat_p = self._schur_comp_mats(Jhat)

        pc_amat_prefix = pc.getOptionsPrefix() + "pc_use_amat"
        self.use_amat = PETSc.Options().getBool(pc_amat_prefix, False)

        if self.use_amat:
            self.Ahat = self._get_wc4dvar_rf(A)
            LTmat, Dmat, Lmat = self._schur_comp_mats(self.Ahat)

        else:
            self.Ahat = Jhat
            (LTmat, Dmat, Lmat) = (LTmat_p, Dmat_p, Lmat_p)

        # Create the KSPs for each component

        self.Lksp = PETSc.KSP().create(comm=global_comm)
        self.LTksp = PETSc.KSP().create(comm=global_comm)
        self.Dksp = PETSc.KSP().create(comm=global_comm)

        self.Lksp.setOperators(Lmat, Lmat_p)
        self.LTksp.setOperators(LTmat, LTmat_p)
        self.Dksp.setOperators(Dmat, Dmat_p)

        # usually will set identical options for L and LT
        default_l_options = petsctools.DefaultOptionSet(
            base_prefix=self.full_prefix + "l_",
            custom_prefix_endings=("tlm", "adj"))

        petsctools.attach_options(
            self.Lksp,
            options_prefix=self.full_prefix+"l_tlm",
            default_options_set=default_l_options)

        petsctools.attach_options(
            self.LTksp,
            options_prefix=self.full_prefix+"l_adj",
            default_options_set=default_l_options)

        petsctools.attach_options(
            self.Dksp,
            options_prefix=self.full_prefix+"d")

        # default to behaving like a set of pcs
        petsctools.set_default_parameter(
            self.Lksp, "ksp_type", "preonly")
        petsctools.set_default_parameter(
            self.LTksp, "ksp_type", "preonly")
        petsctools.set_default_parameter(
            self.Dksp, "ksp_type", "preonly")

        petsctools.set_from_options(self.Lksp)
        petsctools.set_from_options(self.LTksp)
        petsctools.set_from_options(self.Dksp)

        # Make sure we print properly with view
        self.Lksp.incrementTabLevel(1, parent=pc)
        self.Lksp.pc.incrementTabLevel(1, parent=pc)

        self.LTksp.incrementTabLevel(1, parent=pc)
        self.LTksp.pc.incrementTabLevel(1, parent=pc)

        self.Dksp.incrementTabLevel(1, parent=pc)
        self.Dksp.pc.incrementTabLevel(1, parent=pc)

    def _schur_comp_mats(self, Jhat):

        # L and LT: all-at-once system Mats
        Lmat = ReducedFunctionalMat(
            Jhat.JL, action=RFOperation.TLM,
            comm=self.ensemble.global_comm)

        LTmat = ReducedFunctionalMat(
            Jhat.JL, action=RFOperation.ADJOINT,
            comm=self.ensemble.global_comm)

        # D: background and model error covariances
        rank = self.ensemble.ensemble_rank
        BQ = [Jhat.background_covariance] if rank == 0 else []
        BQ.extend(Jhat.model_covariances)

        Dmat = EnsembleBlockDiagonalMat(
            [CovarianceMat(cov, operation='inverse') for cov in BQ],
            col_space=Jhat.control_space,
            row_space=Jhat.control_space.dual())

        return LTmat, Dmat, Lmat

    def _get_wc4dvar_rf(self, mat):
        # 1. If we are using the primal formulation then the mat is the
        #    WC4DVar Hessian and we can grab the RF off the context.
        # 2. If we are using the saddle point formulation then the mat
        #    is the zero (3,3) block of the saddle point MatNest that
        #    we previously stashed the RF on.
        if mat.getType() == "python":
            Jhat = mat.getPythonContext().rf
        else:
            Jhat = mat.getAttr("Jhat")

        if not isinstance(Jhat, WC4DVarReducedFunctional):
            self_name = petsctools.petscobj2str(self)
            Jhat_name = petsctools.petscobj2str(Jhat)
            raise TypeError(
                f"{self_name} expects a WC4DVarReducedFunctional"
                f" not a {Jhat_name}")

        return Jhat

    @PETSc.Log.EventDecorator()
    def apply(self, pc, x, y):
        rhs = x.copy()
        sol = y
        sol.zeroEntries()

        # Just chain the solve for each KSP

        with petsctools.inserted_options(self.LTksp):
            self.LTksp.solve(rhs, sol)

        sol.copy(result=rhs)
        sol.zeroEntries()

        with petsctools.inserted_options(self.Dksp):
            self.Dksp.solve(rhs, sol)

        sol.copy(result=rhs)
        sol.zeroEntries()

        with petsctools.inserted_options(self.Lksp):
            self.Lksp.solve(rhs, sol)

        # y is already sol so no copy needed

    @PETSc.Log.EventDecorator()
    def update(self, pc):
        # The mat should have taken care of updating
        # the Amat but we should check if the Pmat
        # needs updating.
        if self.Jhat is not self.Ahat:
            Adata = self.Ahat.control.data()._ad_to_petsc()
            Jdata = self.Jhat.control.data()._ad_to_petsc()
            if (Adata - Jdata).norm() > 1e-10:
                self.Jhat(self.Ahat.control.data())
        self.LTksp.setUp()
        self.Dksp.setUp()
        self.Lksp.setUp()

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        # Need to view each of the subsolvers as well as ourselves.
        viewer.printfASCII(
            "PC to apply the approximate Weak Constraint 4DVar Schur complement.\n")
        # L ksp
        viewer.printfASCII(
            "The KSP for the all-at-once tangent linear model L is:\n")
        viewer.pushASCIITab()
        self.Lksp.view(viewer)
        viewer.popASCIITab()
        # LT ksp
        viewer.printfASCII(
            "The KSP for the all-at-once adjoint model L^{T} is:\n")
        viewer.pushASCIITab()
        self.LTksp.view(viewer)
        viewer.popASCIITab()
        # D ksp
        viewer.printfASCII(
            "The KSP for the all-at-once model error covariances is:\n")
        viewer.pushASCIITab()
        self.Dksp.view(viewer)
        viewer.popASCIITab()
