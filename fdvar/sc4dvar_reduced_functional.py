from pyadjoint import (
    ReducedFunctional, OverloadedType, Control, Tape, AdjFloat,
    stop_annotating, get_working_tape, annotate_tape)
from pyadjoint.reduced_functional import AbstractReducedFunctional
from pyadjoint.enlisting import Enlist
from firedrake.function import Function
from firedrake.adjoint.covariance_operator import CovarianceOperatorBase
from typing import Callable, Optional
from types import SimpleNamespace
from contextlib import contextmanager
from firedrake.petsc import PETSc


class SC4DVarReducedFunctional(AbstractReducedFunctional):
    """ReducedFunctional for strong constraint 4DVar data assimilation.

    Parameters
    ----------

    control
        The :class:`.Function` for the control x_{0} at the initial condition.

    background_covariance
        The inner product to calculate the background error functional
        from the background error :math:`x_{0} - x_{b}`. Can include the
        error covariance matrix.

    observation_covariance
        The inner product to calculate the observation error functional
        from the observation error :math:`y_{0} - \\mathcal{H}_{0}(x)`.
        Can include the error covariance matrix. Must be provided if
        observation_error is provided.

    observation_error
        Given a state :math:`x`, returns the observations error
        :math:`y_{0} - \\mathcal{H}_{0}(x)` where :math:`y_{0}` are the
        observations at the initial time and :math:`\\mathcal{H}_{0}` is
        the observation operator for the initial time.

    background
        The background (prior) data for the initial condition :math:`x_{b}`.
        If not provided, the value of the control will be used.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    """
    def __init__(self, control: Control,
                 background: OverloadedType,
                 background_covariance: CovarianceOperatorBase,
                 observation_covariance: CovarianceOperatorBase,
                 observation_error: Callable[[OverloadedType], OverloadedType],
                 tape: Tape | None = None):

        if not isinstance(control.control, Function):
            raise TypeError(
                "Control for strong constraint 4DVar must be a Function.")
        if background is control.control:
            raise ValueError(
                "Background must be a different object to the control.")

        self.tape = get_working_tape() if tape is None else tape

        self._controls = Enlist(control)
        self._functional = AdjFloat(0.)
        self._background = background
        x0 = control.control

        self.background_covariance = background_covariance
        self.observation_covariances = [observation_covariance]
        self.observation_errors = [observation_error]

        # penalty for straying from prior
        background_error = x0.copy(deepcopy=True)
        background_error -= background
        self._functional += background_covariance.norm(background_error)

        # penalty for straying from initial observations
        self._functional += observation_covariance.norm(observation_error(x0))

    @property
    def controls(self):
        return self._controls

    @property
    def functional(self):
        return self._functional

    def background(self):
        return self._background

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def __call__(self, values: OverloadedType):
        return self.rf(values)

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def derivative(self, adj_input: float = 1.0, apply_riesz: bool = False):
        return self.rf.derivative(adj_input=adj_input, apply_riesz=apply_riesz)

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def tlm(self, m_dot: OverloadedType):
        return self.rf.tlm(m_dot)

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def hessian(self, m_dot: OverloadedType, hessian_input: Optional[OverloadedType] = None,
                evaluate_tlm: bool = True, apply_riesz: bool = False):
        return self.rf.hessian(m_dot, hessian_input=hessian_input,
                               evaluate_tlm=evaluate_tlm, apply_riesz=apply_riesz)

    @contextmanager
    @PETSc.Log.EventDecorator()
    def recording_stages(self, nstages=None, **stage_kwargs):
        yield SC4DVarObservationStageSequence(
            self.controls[0].control, self,
            stage_kwargs=stage_kwargs, nstages=nstages)

        self.rf = ReducedFunctional(
            self._functional, self.controls.delist(), tape=self.tape)


class SC4DVarObservationStageSequence:
    def __init__(self, control: OverloadedType,
                 rf: SC4DVarReducedFunctional,
                 stage_kwargs: dict,
                 nstages: int):
        self.ctx = SimpleNamespace(**stage_kwargs)
        self.nstages = nstages
        self._index = 0
        self._current_stage = SC4DVarObservationStage(
            control, rf, index=self._index)

    def __iter__(self):
        return self

    @PETSc.Log.EventDecorator()
    def __next__(self):
        if self._index >= self.nstages:
            raise StopIteration

        if self._index > 0:
            self._current_stage = SC4DVarObservationStage(
                control=self._current_stage.state,
                rf=self._current_stage._rf,
                index=self._index)

        self._index += 1
        return self._current_stage, self.ctx


class SC4DVarObservationStage:
    """
    Record an observation for strong constraint 4DVar at the time of `state`.

    Parameters
    ----------

    control :
        The state at the beginning of this stage.
    rf :
        The SC4DVarReducedFunctional.
    index :
        The index of this stage, numbered from 0.
    """

    def __init__(self, control: OverloadedType,
                 rf: SC4DVarReducedFunctional, index: int):
        self._rf = rf
        self.control = control
        self.index = index

    @PETSc.Log.EventDecorator()
    def set_observation(self, state: OverloadedType,
                        observation_error: Callable[[OverloadedType], OverloadedType],
                        observation_covariance: CovarianceOperatorBase):
        """
        Record an observation at the time of `state`.

        Parameters
        ----------

        state
            The state at the current observation time.

        observation_error
            Given a state :math:`x`, returns the observations error
            :math:`y_{i} - \\mathcal{H}_{i}(x)` where :math:`y_{i}` are
            the observations at the current observation time and
            :math:`\\mathcal{H}_{i}` is the observation operator for the
            current observation time.

        observation_covariance
            The inner product to calculate the observation error functional
            from the observation error :math:`y_{i} - \\mathcal{H}_{i}(x)`.
            Can include the error covariance matrix.
        """
        # get the tape used for this stage and make sure its the right one
        if get_working_tape() is not self._rf.tape:
            raise ValueError(
                "Working tape at the end of the observation stage"
                " differs from the tape at the stage beginning."
            )
        if not annotate_tape():
            raise ValueError(
                "Must have annotations switched on whilst"
                " recording the 4DVar observation stages.")

        self._rf._functional += (
            observation_covariance.norm(observation_error(state)))

        self._rf.observation_covariances.append(observation_covariance)
        self._rf.observation_errors.append(observation_error)

        # save the user's state to hand back for beginning of next stage
        self.state = state
