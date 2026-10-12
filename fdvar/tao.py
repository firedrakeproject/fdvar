import numpy as np
from firedrake import PETSc


def tao_converged_maxits_test(tao):
    """
    Convergence test that allows TAO to declare convergence if
    the maximum number of iterations is reached.

    Options Database Key
    --------------------
    -tao_converged_maxits (true|false)

    Parameters
    ----------
    tao : petsc4py.PETSc.TAO
        The TAO object to test for convergence.

    Notes
    -----
    All other convergence/divergence checks are the same as with
    TaoDefaultConvergenceTest except the following checks which
    rely on TAO attributes that are not yet exposed in petsc4py:

    -tao_max_funcs - sets maximum number of function evaluations.
    -tao_steptol - stop if the trust region radius becomes less than tol.
    -tao_ls_failure - stop if the linesearch fails.
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
