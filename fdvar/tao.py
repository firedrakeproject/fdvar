import numpy as np
from firedrake import PETSc

_ksp_norm_to_enum = {
    k: v
    for k, v in PETSc.KSP.NormType.__dict__.items()
    if isinstance(v, int)
}
_enum_to_ksp_norm = {
    v: k.lower()
    for k, v in _ksp_norm_to_enum.items()
    if 'NORM_' not in k
}


def ksp_objective_monitor(ksp: PETSc.KSP, it: int, prnorm: float, tao: PETSc.TAO | None = None, c: float | None = None):
    """
    KSP monitor for the KSP residual, true residual and quadratic objective.

    This assumes that the linear system :math:`Ax=b` originates from a quadratic
    minimisation problem :math:`\\frac{1}{2}x^{T}Ax - x^{T}b + c`.

    Three alternatives are available for the shift :math:`c`:

    * The default value is :math:`c=0`.
    * A constant value can be set by currying the ``c`` kwarg.
    * If the KSP is solving the linear system from a Newton-type TAO object then
      the shift should be the latest value of the TAO objective. This can be
      achieved by currying the ``tao`` kwarg.

    Standard usage assuming :math:`c=0`:

    .. code-block:: python3

        tao = petsc4py.PETSc.TAO()
        ...
        tao.ksp.setMonitor(ksp_objective_monitor)

    Usage setting the TAO object to use to calculate :math:`c` (similar usage for
    setting a constant :math:`c`):

    .. code-block:: python3

        from functools import partial
        tao.ksp.setMonitor(partial(ksp_objective_monitor, tao=tao))


    Parameters
    ----------
    KSP :
        The KSP object to test for convergence.
    it :
        The current number of KSP iterations.
    prnorm :
        The current KSP residual. The type of this residual will be the default
        norm type for this KSP type, or whatever was set with ``-ksp_norm_type``.
    tao :
        The outer TAO optimisation solver to get the shift :math:`c` from.
        Mutually exclusive with ``c``.
    c :
        The shift :math:`c`.
        Mutually exclusive with ``tao``.
    """
    tab_level = ksp.getTabLevel()
    tab = '    '
    if tao is not None and c is not None:
        raise ValueError("Cannot both set tao and c, choose one.")
    if tao:
        _, shift, _, _, _, _ = tao.getSolutionStatus()
    elif c:
        shift = c
    else:
        shift = 0.
    # obj = 0.5 * (x' * A * x) - (x' * b)
    #     = - 0.5 * x' * (b + r)
    b = ksp.getRhs()
    x = ksp.buildSolution()
    r = ksp.buildResidual()
    rnorm = r.norm()
    bnorm = b.norm()
    obj_val = shift - 0.5*x.dot((r + b))
    ksp.setAttr('_tao_objective', obj_val)
    if it == 0:
        PETSc.Sys.Print(
            tab*tab_level + f'Residual norms for {ksp.prefix} solve.',
            comm=ksp.comm
        )
    PETSc.Sys.Print(
        tab*(tab_level - 1)
        + f'{it: 5d} KSP {_enum_to_ksp_norm[ksp.norm_type]} resid norm {prnorm:8.6e}'
        + f' true resid norm {r.norm():8.6e}'
        + f' ||r(i)||/||b|| {rnorm/bnorm:8.6e}'
        + f' objective {obj_val:9.6e}',
        comm=ksp.comm
    )


def tao_converged_maxits_test(tao: PETSc.TAO):
    """
    Convergence test that allows TAO to declare convergence if
    the maximum number of iterations is reached.

    **PETSc Options**

    * ``-tao_converged_maxits (true|false)`` - Treat reaching the maximum number
        of iterations (``-tao_max_it``) as convergence rather than divergence.
    * See :class:`~petsc4py.PETSc.TAO` tolerance docs for other options.

    Parameters
    ----------
    tao :
        The TAO object to test for convergence.

    Notes
    -----
    All other convergence/divergence checks are the same as with
    TaoDefaultConvergenceTest except the following checks which
    rely on TAO attributes that are not yet exposed in petsc4py:

    * ``-tao_max_funcs`` - sets maximum number of function evaluations.
    * ``-tao_steptol`` - stop if the trust region radius becomes less than tol.
    * ``-tao_ls_failure`` - stop if the linesearch fails.
    """
    its, f, gnorm, cnorm, _step, reason = tao.getSolutionStatus()
    gatol, grtol, gttol = tao.getTolerances()
    catol, crtol = tao.getConstraintTolerances()
    max_it = tao.getMaximumIterations()
    _ = tao.getMaximumFunctionEvaluations()

    prefix = tao.prefix or ""
    opts = PETSc.Options()
    fmin = opts.getReal(f"{prefix}tao_fmin", 0.0)
    converged_maxits = opts.getBool(f"{prefix}tao_converged_maxits", False)

    # stash the initial gradient norm
    if its == 0:
        gnorm0 = gnorm
        tao.setAttr("_fdvar_gnorm0", gnorm0)
    else:
        gnorm0 = tao.getAttr("_fdvar_gnorm0")

    Reasons = PETSc.TAO.ConvergedReason

    # make sure we don't accidentally shortcircuit our own maxits check
    if reason != Reasons.CONTINUE_ITERATING and not (
        reason == Reasons.DIVERGED_MAXITS and converged_maxits
    ):
        return

    if np.isinf(f) or np.isnan(f):
        reason = Reasons.DIVERGED_NAN

    elif (f <= fmin) and (cnorm <= catol):
        reason = Reasons.CONVERGED_MINF

    elif (gnorm <= gatol) and (cnorm <= catol):
        reason = Reasons.CONVERGED_GATOL

    elif (abs(gnorm / f) <= grtol) and (cnorm <= crtol):
        reason = Reasons.CONVERGED_GRTOL

    elif (gnorm / gnorm0 <= gttol) and (cnorm <= crtol):
        reason = Reasons.CONVERGED_GTTOL

    # TODO: petsc4py binding for nfuncs
    # elif nfuncs > max_funcs:
    #     reason = Reasons.DIVERGED_MAXFCN

    # TODO: petsc4py binding for linesearch reason
    # elif tao.getLineSearch().reason < 0:
    #     reason = Reasons.DIVERGED_LS_FAILURE

    # TODO: petsc4py binding for linesearch reason
    # elif (step < steptol) and (niter > 0):
    #     reason = Reasons.CONVERGED_STEPTOL

    elif its >= max_it:
        if converged_maxits:
            reason = Reasons.CONVERGED_USER
            PETSc.Sys.Print(
                f"TAO {prefix} solve converged due to CONVERGED_MAXITS iterations {its}"
            )
        else:
            reason = Reasons.DIVERGED_MAXITS

    else:
        reason = Reasons.CONTINUE_ITERATING

    tao.setConvergedReason(reason)
