from firedrake import EnsembleCofunction, EnsembleFunction, Function
from firedrake.adjoint import (
    AutoregressiveCovariance as FiredrakeAutoregressiveCovariance,
)
from firedrake.adjoint import (
    MixedCovarianceOperator as FiredrakeMixedCovarianceOperator,
)
from firedrake.adjoint.covariance_operator import CovarianceOperatorBase


class AutoregressiveCovariance(FiredrakeAutoregressiveCovariance):
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

    def decorrelated(self):
        return type(self)(
            V=self.function_space(),
            sigma=1.0,
            m=0,
            L=1.0,
            rng=self.rng(),
        )


class MixedCovarianceOperator(FiredrakeMixedCovarianceOperator):
    def sqrt_action(self, x: Function, *, tensor=None):
        tensor = tensor or Function(self.function_space())

        for xi, ti, Bi in zip(x.subfunctions, tensor.subfunctions, self.subcovariances):
            Bi.sqrt_action(xi, tensor=ti)

        return tensor

    def sqrt_inverse(self, x: Function, *, tensor=None):
        tensor = tensor or Function(self.function_space())

        for xi, ti, Bi in zip(x.subfunctions, tensor.subfunctions, self.subcovariances):
            Bi.sqrt_inverse(xi, tensor=ti)

        return tensor

    def decorrelated(self):
        return type(self)(
            self.function_space(), [sub.decorrelated() for sub in self.subcovariances]
        )


class EnsembleCovarianceOperator(MixedCovarianceOperator):
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

    def sample(self, rng=None, tensor=None):
        tensor = tensor or EnsembleFunction(self.function_space())
        return super().sample(rng=rng, tensor=tensor)

    def norm(self, x):
        return self.function_space().ensemble_comm.allreduce(super().norm(x))

    def apply_action(self, x, *, tensor=None):
        tensor = tensor or EnsembleFunction(self.function_space())
        return super().apply_action(x, tensor=tensor)

    def apply_inverse(self, x, *, tensor=None):
        tensor = tensor or EnsembleCofunction(self.function_space().dual())
        return super().apply_inverse(x, tensor=tensor)

    def sqrt_action(self, x, *, tensor=None):
        tensor = tensor or EnsembleFunction(self.function_space())
        return super().sqrt_action(x, tensor=tensor)

    def sqrt_inverse(self, x, *, tensor=None):
        tensor = tensor or EnsembleFunction(self.function_space())
        return super().sqrt_inverse(x, tensor=tensor)
