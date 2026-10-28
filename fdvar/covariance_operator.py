from firedrake import (
    Cofunction,
    EnsembleCofunction,
    EnsembleFunction,
    Function,
    MixedFunctionSpace,
    PETSc,
)
from firedrake.adjoint import (
    AutoregressiveCovariance as FiredrakeAutoregressiveCovariance,
)
from firedrake.adjoint import (
    MixedCovarianceOperator as FiredrakeMixedCovarianceOperator,
)
from firedrake.adjoint.covariance_operator import CovarianceOperatorBase
from ufl.duals import is_dual


class AutoregressiveCovariance(FiredrakeAutoregressiveCovariance):
    @PETSc.Log.EventDecorator()
    def sqrt_action(self, x: Function, *, tensor: Function | None = None):
        r"""Return :math:`y = \hat{B}^{1/2}x` where :math:`\hat{B}^{1/2}`
        is an approximate square root of the covariance operator satisfying:

        .. math::

        B = \hat{B}^{1/2}M^{-1}\hat{B}^{T/2}, \quad \hat{B}^{1/2}: V \to V

        Parameters
        ----------
        x :
            The :class:`~firedrake.function.Function` to apply the sqrt to.
        tensor :
            Optional location to place the result into.

        Returns
        -------
        firedrake.function.Function :
            The result of :math:`\hat{B}^{1/2}x`
        """
        tensor = tensor or Function(self.function_space())

        if self.iterations == 0:
            return tensor.assign(self.stddev * x)

        self._u.assign(x)

        for i in range(self.iterations // 2):
            self._urhs.assign(self._u)
            self.solver.solve()

        return tensor.assign(self._weight * self._u)

    @PETSc.Log.EventDecorator()
    def sqrt_inverse(self, x: Function, *, tensor: Function | None = None):
        r"""Return :math:`y = \hat{B}^{-1/2}x` where :math:`\hat{B}^{-1/2}`
        is the inverse of an approximate square root of the covariance
        operator satisfying:

        .. math::

        B^{-1} = \hat{B}^{-T/2}M\hat{B}^{-1/2}, \quad \hat{B}^{1/2}: V \to V

        Parameters
        ----------
        x :
            The :class:`~firedrake.function.Function` to apply the sqrt to.
        tensor :
            Optional location to place the result into.

        Returns
        -------
        firedrake.function.Function :
            The result of :math:`\hat{B}^{1/2}x`
        """
        tensor = tensor or Function(self.function_space())

        if self.iterations == 0:
            return tensor.assign((1 / self.stddev) * x)

        lamda1 = 1 / self._weight
        self._u.assign(lamda1 * x)

        for i in range(self.iterations // 2):
            self._urhs.assign(self._u)
            self.mass_solver.solve()

        return tensor.assign(self._u)

    @PETSc.Log.EventDecorator()
    def decorrelated(self):
        return type(self)(
            V=self.function_space(),
            sigma=1.0,
            m=0,
            L=1.0,
            rng=self.rng(),
        )


class MixedCovarianceOperator(FiredrakeMixedCovarianceOperator):
    def __init__(self, W, subcovariances):
        Wflat = MixedFunctionSpace([subi.function_space() for subi in subcovariances])
        if len(Wflat) != len(W.subspaces):
            raise ValueError(
                "Total number of covariance subspaces must match W subspaces."
            )
        if not all(Wfi == Wi for Wfi, Wi in zip(Wflat.subspaces, W.subspaces)):
            raise ValueError("Covariance function spaces must match W subspaces")
        if not all(isinstance(cov, CovarianceOperatorBase) for cov in subcovariances):
            raise TypeError("All covariance operators must be a CovarianceOperatorBase")

        self._W = W
        self._subcovariances = subcovariances
        self._rngs = [cov.rng() for cov in self.subcovariances]

    def _flatten(self, xsubs):
        # TODO: MFE for replaying tape with this:
        # xflattened = [
        #     xisub
        #     for xi in xsubs
        #     for xisub in xi.subfunctions
        # ]
        xflattened = []
        for xi in xsubs:
            if len(xi.subfunctions) == 1:
                xflattened.append(xi)
            else:
                xflattened.extend(xi.subfunctions)
        return xflattened

    def _from_subs(self, xsubs, xfull=None):
        if xfull is None:
            W = self.function_space()
            xfull = Function(W.dual() if is_dual(xsubs[0]) else W)
        for xf, xs in zip(xfull.subfunctions, self._flatten(xsubs)):
            xf.assign(xs)
        return xfull

    def _to_subs(self, xfull, xsubs=None):
        if xsubs is None:
            xsubs = [
                Function(
                    covi.function_space().dual()
                    if is_dual(xfull)
                    else covi.function_space()
                )
                for covi in self.subcovariances
            ]
        flat_subs = self._flatten(xsubs)
        assert len(xfull.subfunctions) == len(flat_subs)
        for xf, xs in zip(xfull.subfunctions, flat_subs):
            xs.assign(xf)
        return xsubs

    @PETSc.Log.EventDecorator()
    def sample(self, rng=None, tensor=None):
        tensor = tensor or Function(self.function_space())
        samples = [
            covi.sample(rng=rngi)
            for covi, rngi in zip(self.subcovariances, rng or self.rng())
        ]
        return self._from_subs(samples, xfull=tensor)

    @PETSc.Log.EventDecorator()
    def norm(self, x):
        xsubs = self._to_subs(x)
        return sum(covi.norm(xi) for covi, xi in zip(self.subcovariances, xsubs))

    def _map(self, method, x, tensor):
        xsubs = self._to_subs(x)
        tsubs = [
            getattr(covi, method)(xi) for covi, xi in zip(self.subcovariances, xsubs)
        ]
        return self._from_subs(tsubs, xfull=tensor)

    @PETSc.Log.EventDecorator()
    def apply_inverse(self, x, tensor=None):
        tensor = tensor or Cofunction(self.function_space().dual())
        return self._map("apply_inverse", x, tensor)

    @PETSc.Log.EventDecorator()
    def apply_action(self, x, tensor=None):
        tensor = tensor or Function(self.function_space())
        return self._map("apply_action", x, tensor)

    @PETSc.Log.EventDecorator()
    def sqrt_action(self, x: Function, *, tensor=None):
        tensor = tensor or Function(self.function_space())
        return self._map("sqrt_action", x, tensor)

    @PETSc.Log.EventDecorator()
    def sqrt_inverse(self, x: Function, *, tensor=None):
        tensor = tensor or Function(self.function_space())
        return self._map("sqrt_inverse", x, tensor)

    @PETSc.Log.EventDecorator()
    def decorrelated(self):
        return type(self)(
            self.function_space(),
            [covi.decorrelated() for covi in self.subcovariances],
        )


class EnsembleCovarianceOperator(CovarianceOperatorBase):
    def __init__(self, W, subcovariances):
        if len(subcovariances) != len(W.local_spaces):
            raise ValueError(
                "Need one covariance operator per component of mixed space"
            )
        if not all(isinstance(cov, CovarianceOperatorBase) for cov in subcovariances):
            raise TypeError("All covariance operators must be a CovarianceOperatorBase")
        if not all(
            cov.function_space() == Wsub
            for cov, Wsub in zip(subcovariances, W.local_spaces)
        ):
            raise ValueError("Covariance function spaces must match W subspaces")

        self._W = W
        self._subcovariances = subcovariances
        self._rngs = [cov.rng() for cov in self.subcovariances]

        self._local_covariance = MixedCovarianceOperator(
            W._full_local_space, subcovariances
        )

    def function_space(self):
        return self._W

    def rng(self):
        return self._rngs

    @property
    def subcovariances(self):
        return self._subcovariances

    @PETSc.Log.EventDecorator()
    def sample(self, rng=None, tensor=None):
        tensor = tensor or EnsembleFunction(self.function_space())
        for covi, rngi, ti in zip(
            self.subcovariances, rng or self.rng(), tensor.subfunctions
        ):
            covi.sample(rngi, tensor=ti)
        return tensor

    @PETSc.Log.EventDecorator()
    def norm(self, x):
        return self.function_space().ensemble_comm.allreduce(
            sum(covi.norm(xi) for covi, xi in zip(self.subcovariances, x.subfunctions))
        )

    def _map(self, method, x, tensor):
        for covi, xi, ti in zip(
            self.subcovariances, x.subfunctions, tensor.subfunctions
        ):
            getattr(covi, method)(xi, tensor=ti)
        return tensor

    @PETSc.Log.EventDecorator()
    def apply_action(self, x, *, tensor=None):
        tensor = tensor or EnsembleFunction(self.function_space())
        return self._map("apply_action", x, tensor)

    @PETSc.Log.EventDecorator()
    def apply_inverse(self, x, *, tensor=None):
        tensor = tensor or EnsembleCofunction(self.function_space().dual())
        return self._map("apply_inverse", x, tensor)

    @PETSc.Log.EventDecorator()
    def sqrt_action(self, x, *, tensor=None):
        tensor = tensor or EnsembleFunction(self.function_space())
        return self._map("sqrt_action", x, tensor)

    @PETSc.Log.EventDecorator()
    def sqrt_inverse(self, x, *, tensor=None):
        tensor = tensor or EnsembleFunction(self.function_space())
        return self._map("sqrt_inverse", x, tensor)

    @PETSc.Log.EventDecorator()
    def decorrelated(self):
        return type(self)(
            self.function_space(),
            [covi.decorrelated() for covi in self.subcovariances],
        )
