import firedrake as fd
from firedrake.petsc import PETSc
import petsctools
from firedrake.adjoint import (
    FourDVarReducedFunctional, CovarianceMat)
from pyadjoint.optimization.tao_solver import (
    ReducedFunctionalMat, RFOperation)


class WC4DVarSchurPC(petsctools.PCBase):
    prefix = "wcschur_"

    def initialize(self, pc):
        super().initialize(pc)

        A, P = pc.getOperators()

        if P.getType() == "python":
            Jhat = P.getPythonContext().rf
        else:
            Jhat = P.getAttr("Jhat")

        if not isinstance(Jhat, FourDVarReducedFunctional):
            self_name = petsctools.petscobj2str(self)
            Jhat_name = petsctools.petscobj2str(Jhat)
            raise TypeError(
                f"{self_name} expects a FourDVarReducedFunctional"
                f" not a {Jhat_name}")

        self.Jhat = Jhat
        self.ensemble = Jhat.ensemble
        global_comm = self.ensemble.global_comm

        self.col_space = Jhat.control_space
        self.row_space = Jhat.control_space.dual()

        Lmat_p = ReducedFunctionalMat(
            Jhat.JL, action=RFOperation.TLM,
            comm=global_comm)

        LTmat_p = ReducedFunctionalMat(
            Jhat.JL, action=RFOperation.ADJOINT,
            comm=global_comm)

        BQmats_p = [
            CovarianceMat(
                rf.covariance, operation='inverse')
            for rf in Jhat.JD.rfs]

        Dmat_p = fd.EnsembleBlockDiagonalMat(
            BQmats_p, col_space=Jhat.control_space,
            row_space=Jhat.control_space.dual())

        pc_prefix = pc.getOptionsPrefix() + "pc_" + self.prefix
        self.use_amat = PETSc.Options().getBool(pc_prefix + "use_amat", False)

        if self.use_amat:
            if A.getType() == "python":
                Ahat = A.getPythonContext().rf
            else:
                Ahat = A.getAttr("Jhat")

            if not isinstance(Ahat, FourDVarReducedFunctional):
                self_name = petsctools.petscobj2str(self)
                Ahat_name = petsctools.petscobj2str(Ahat)
                raise TypeError(
                    f"{self_name} expects a FourDVarReducedFunctional"
                    f" not a {Ahat_name}")

            self.Ahat = Ahat

            Lmat = ReducedFunctionalMat(
                Ahat.JL, action=RFOperation.TLM,
                comm=global_comm)

            LTmat = ReducedFunctionalMat(
                Ahat.JL, action=RFOperation.ADJOINT,
                comm=global_comm)

            BQmats = [
                CovarianceMat(
                    rf.covariance, operation='inverse')
                for rf in Ahat.JD.rfs]

            Dmat = fd.EnsembleBlockDiagonalMat(
                BQmats, col_space=Jhat.control_space,
                row_space=Ahat.control_space.dual())

        else:
            self.Ahat = Jhat
            Lmat = Lmat_p
            LTmat = LTmat_p
            Dmat = Dmat_p

        self.Lksp = PETSc.KSP().create(comm=global_comm)
        self.LTksp = PETSc.KSP().create(comm=global_comm)
        self.Dksp = PETSc.KSP().create(comm=global_comm)

        self.Lksp.setOperators(Lmat, Lmat_p)
        self.LTksp.setOperators(LTmat, LTmat_p)
        self.Dksp.setOperators(Dmat, Dmat_p)

        default_l_options = petsctools.DefaultOptionSet(
            base_prefix=self.full_prefix + "l_",
            custom_prefix_endings=("tlm", "adj"))

        petsctools.set_from_options(
            self.Lksp, parameters={},
            options_prefix=self.full_prefix+"l_tlm",
            default_options_set=default_l_options)

        petsctools.set_from_options(
            self.LTksp, parameters={},
            options_prefix=self.full_prefix+"l_adj",
            default_options_set=default_l_options)

        petsctools.set_from_options(
            self.Dksp, parameters={},
            options_prefix=self.full_prefix+"d")

        self.Lksp.incrementTabLevel(1, parent=pc)
        self.Lksp.pc.incrementTabLevel(1, parent=pc)

        self.LTksp.incrementTabLevel(1, parent=pc)
        self.LTksp.pc.incrementTabLevel(1, parent=pc)

        self.Dksp.incrementTabLevel(1, parent=pc)
        self.Dksp.pc.incrementTabLevel(1, parent=pc)

    def apply(self, pc, x, y):
        rhs = x.copy()
        sol = y.copy()

        # TODO: this should be done by update
        val = self.Ahat.control.data()
        self.Jhat(val)
        self.Jhat.derivative(apply_riesz=False)

        sol.zeroEntries()

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

        sol.copy(result=y)

    def update(self, pc):
        if self.Jhat is not self.Ahat:
            val = self.Ahat.control.data()
            self.Jhat(val)
            # self.Jhat.derivative(apply_riesz=False)

        self.Lksp.setUp()
        self.LTksp.setUp()
        self.Dksp.setUp()


def getSubWC4DVarSaddlePointMat(mat, sub=None):
    """
    Return a sub matrix of the saddle point MatNest.
    Options are 'D', 'R', 'L', 'LT', 'H', 'HT',
    or None to return all sub matrices.
    """
    idx = {
        'D': (0, 0),
        'R': (1, 1),
        'L': (0, 2),
        'LT': (2, 0),
        'H': (1, 2),
        'HT': (2, 1),
    }
    return (
        mat.getNestSubMatrix(*idx[sub])
        if sub is not None else
        tuple(mat.getNestSubMatrix(*i) for i in idx.values())
    )


def WC4DVarSaddlePointKSP(Jhat, Jphat=None,
                          solver_parameters=None,
                          options_prefix=None):
    amat = WC4DVarSaddlePointMat(Jhat)

    if Jphat:
        pmat = WC4DVarSaddlePointMat(Jphat)
    else:
        pmat = amat

    ksp = PETSc.KSP().create(
        comm=Jhat.ensemble.global_comm)
    ksp.setOperators(amat, pmat)

    petsctools.set_from_options(
        ksp, parameters=solver_parameters,
        options_prefix=options_prefix)

    return ksp


def WC4DVarSaddlePointMat(Jhat):
    if not isinstance(Jhat, FourDVarReducedFunctional):
        raise TypeError(
            "WC4DVarSaddlePointMat must be constructed from a"
            f" FourDVarReducedFunctional, not a {type(Jhat).__name__}")

    ensemble = Jhat.ensemble
    Wc = Jhat.control_space
    Wo = Jhat.observation_space

    Lmat = ReducedFunctionalMat(
        Jhat.JL, action=RFOperation.TLM,
        comm=ensemble.global_comm)

    LTmat = ReducedFunctionalMat(
        Jhat.JL, action=RFOperation.ADJOINT,
        comm=ensemble.global_comm)

    Hmat = ReducedFunctionalMat(
        Jhat.JH, action=RFOperation.TLM,
        comm=ensemble.global_comm)

    HTmat = ReducedFunctionalMat(
        Jhat.JH, action=RFOperation.ADJOINT,
        comm=ensemble.global_comm)

    BQmats = [
        CovarianceMat(
            rf.covariance, operation='action')
        for rf in Jhat.JD.rfs]

    Dmat = fd.EnsembleBlockDiagonalMat(
        BQmats, col_space=Wc,
        row_space=Wc.dual())

    Rmats = [
        CovarianceMat(
            rf.covariance, operation='action')
        for rf in Jhat.JR.rfs]

    Rmat = fd.EnsembleBlockDiagonalMat(
        Rmats, col_space=Wo,
        row_space=Wo.dual())

    vec_dx = Wc.layout_vec.duplicate()

    A22 = PETSc.Mat().createConstantDiagonal(
        (vec_dx.sizes, vec_dx.sizes), 0.,
        comm=ensemble.global_comm)
    A22.setUp()
    A22.assemble()
    A22.setAttr("Jhat", Jhat)

    saddle_mat = PETSc.Mat().createNest(
        mats=[[Dmat,  None,  Lmat],     # noqa: E127,E202
              [None,  Rmat,  Hmat],     # noqa: E127,E202
              [LTmat, HTmat, A22]],     # noqa: E127,E202
        comm=ensemble.global_comm)
    saddle_mat.setUp()
    saddle_mat.assemble()

    return saddle_mat


class WC4DVarSaddlePointPC(petsctools.PCBase):
    needs_python_amat = True
    needs_python_pmat = True

    prefix = "wcsaddle_"

    def initialize(self, pc):
        super().initialize(pc)

        Jhat = self.amat.rf
        if not isinstance(Jhat, FourDVarReducedFunctional):
            self_name = petsctools.petscobj2str(self)
            Jhat_name = petsctools.petscobj2str(Jhat)
            raise TypeError(
                f"{self_name} expects a FourDVarReducedFunctional"
                f" not a {Jhat_name}")

        Jphat = self.pmat.rf
        if not isinstance(Jphat, FourDVarReducedFunctional):
            self_name = petsctools.petscobj2str(self)
            Jphat_name = petsctools.petscobj2str(Jphat)
            raise TypeError(
                f"{self_name} expects a FourDVarReducedFunctional"
                f" not a {Jphat_name}")

        self.Jhat = Jhat
        self.Jphat = Jphat
        self.ensemble = Jphat.ensemble

        self.rhs_type = "saddle"

        self.use_amat = PETSc.Options().getBool(
            self.full_prefix + "use_amat", False)

        if self.use_amat:
            Jhat_a = Jhat
        else:
            Jhat_a = Jphat
        Jhat_p = Jphat

        self.saddle_ksp = WC4DVarSaddlePointKSP(
            Jhat_a, Jhat_p, options_prefix=self.full_prefix)
        self.saddle_mat, _ = self.saddle_ksp.getOperators()

        self.saddle_ksp.incrementTabLevel(1, parent=pc)
        self.saddle_ksp.pc.incrementTabLevel(1, parent=pc)

        self.rhs = self._create_vec()
        self.sol = self._create_vec()

        self.rhs_dn, self.rhs_dl, self.rhs_dx = self.rhs.getNestSubVecs()
        self.sol_dn, self.sol_dl, self.sol_dx = self.sol.getNestSubVecs()

    def _create_vec(self):
        Wc = self.Jphat.control_space
        Wo = self.Jphat.observation_space

        v_dn = Wc.layout_vec.duplicate()
        v_dl = Wo.layout_vec.duplicate()
        v_dx = Wc.layout_vec.duplicate()

        v = PETSc.Vec().createNest(
            vecs=(v_dn, v_dl, v_dx),
            isets=self.saddle_mat.getNestISs()[0],
            comm=self.Jphat.ensemble.global_comm)

        v.setUp()

        return v

    def _build_rhs(self):
        vec = self.rhs

        val = self.Jhat.control.data()
        v_dn, v_dl, v_dx = vec.getNestSubVecs()

        b = self.Jphat.JL(val)
        d = self.Jphat.JH(val)

        with b.vec_ro() as bvec:
            bvec.copy(result=v_dn)

        with d.vec_ro() as dvec:
            dvec.copy(result=v_dl)

        v_dx.zeroEntries()

        return vec

    def apply(self, pc, x, y):
        self.sol.zeroEntries()
        self.rhs.zeroEntries()
        # self._build_rhs()

        if self.rhs_type == "saddle":
            val = self.Jhat.control.data()

            with self.Jphat.JL(val).vec_ro() as bvec:
                bvec.copy(result=self.rhs_dn)

            with self.Jphat.JH(val).vec_ro() as dvec:
                dvec.copy(result=self.rhs_dl)

        elif self.rhs_type == "primal":
            x.copy(result=self.rhs_dx)

        with petsctools.inserted_options(self.saddle_ksp):
            self.saddle_ksp.solve(self.rhs, self.sol)

        self.sol_dx.copy(result=y)

    def update(self, pc):
        if self.Jphat is not self.Jhat:
            val = self.Jhat.control.data()
            self.Jphat(val)
            self.Jphat.derivative(apply_riesz=False)
        self.saddle_ksp.setUp()


class AuxiliaryReducedFunctionalPC(petsctools.PCBase):
    """
    Create a preconditioner from an auxiliary ReducedFunctional.

    Builds a KSP from a user provided ReducedFunctional and applies
    this to the residual instead of the original PC operators.

    This class should be subclassed with an implementation of the
    `reduced_functional` method returning two ReducedFunctionals for
    the KSP operators, the type of action (TLM, Adjoint, or Hessian),
    and any other arguments for the ReducedFunctionalMats.
    """

    prefix = "aux_"

    default_options = {
        'ksp_type': 'gmres',
        'pc_type': 'none',
    }

    needs_python_pmat = True
    needs_python_amat = True

    def initialize(self, pc):
        super().initialize(pc)

        self.arf = self.amat.rf

        self.prf, action, mat_kwargs = self.reduced_functional(pc)

        comm = self.pc.comm

        pmat = ReducedFunctionalMat(
            self.prf, action, comm=comm, **mat_kwargs)
        self.aux_mat = pmat

        if self.use_amat:
            amat = ReducedFunctionalMat(
                self.arf, action, comm=comm, **mat_kwargs)
        else:
            amat = pmat

        self.ksp = PETSc.KSP().create(comm=amat.comm)
        self.ksp.setOperators(amat, pmat)
        petsctools.attach_options(
            self.ksp, parameters=self.default_options,
            options_prefix=self.full_prefix)

        for k, v in self.default_options:
            petsctools.set_default_parameter(self.ksp, k, v)

        self.ksp.incrementTabLevel(1, parent=pc)
        self.ksp.pc.incrementTabLevel(1, parent=pc)

    def apply(self, pc, x, y):
        with petsctools.inserted_options(self.ksp):
            self.ksp.solve(x, y)

    def update(self, pc):
        val = self.arf.control.data()
        self.prf(val)
        if self.aux_mat.update_adjoint():
            self.prf.derivative(apply_riesz=False)

    def reduced_functional(self, pc):
        """
        Return (prf, action, **kwargs), where:
            - prf is the ReducedFunctional to use for Pmat.
            - action is the RFOperation for the Mats.
            - kwargs are passed to the ReducedFunctionalMat.
        """
        raise NotImplementedError(
            "User must implement this method to return"
            " an auxiliary ReducedFunctional.")

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        viewer.printfASCII(
            "PC that approximately solves an auxiliary ReducedFunctional\n")
        if hasattr(self, 'ksp'):
            viewer.printfASCII(
                "Inner KSP to solve the auxiliary ReducedFunctional:\n")
            self.ksp.view(viewer)
