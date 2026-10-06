from firedrake import Function, Constant, sqrt
from firedrake.adjoint import (
    AutoregressiveCovariance as FiredrakeAutoregressiveCovariance,
    MixedCovarianceOperator as FiredrakeMixedCovarianceOperator,
)


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

        weight = Constant(sqrt(self.lambda_m))
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

        weight = Constant(sqrt(self.lambda_m))
        lamda1 = 1 / self._weight
        self._u.assign(lamda1 * x)

        for i in range(self.iterations // 2):
            self._urhs.assign(self._u)
            self.mass_solver.solve()

        return tensor.assign(self._u)

    def decorrelated(self):
        return FiredrakeAutoregressiveCovariance(
            V=self.function_space(),
            sigma=1.0,
            m=0,
            L=1.0,
            rng=self.rng(),
        )


class MixedCovarianceOperator(FiredrakeMixedCovarianceOperator):
    def decorrelated(self):
        return MixedCovarianceOperator(
            self.function_space(),
            [sub.decorrelated()
             for sub in self.subcovariances]
        )

    def sqrt_action(self, x: Function, *, tensor=None):
        tensor = tensor or Function(self.function_space())

        for xi, ti, Bi in zip(x.subfunctions,
                              tensor.subfunctions,
                              self.subcovariances):
            Bi.sqrt_action(xi, tensor=ti)

        return tensor

    def sqrt_inverse(self, x: Function, *, tensor=None):
        tensor = tensor or Function(self.function_space())

        for xi, ti, Bi in zip(x.subfunctions,
                              tensor.subfunctions,
                              self.subcovariances):
            Bi.sqrt_inverse(xi, tensor=ti)

        return tensor
