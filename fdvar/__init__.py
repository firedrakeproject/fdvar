# make sure that PETSc is imported by Firedrake first so cmdline args are passed
from firedrake.petsc import PETSc  # noqa: F401

__all__ = (
    "SC4DVarReducedFunctional",
    "WC4DVarReducedFunctional",
    "AllAtOnceReducedFunctional",
    "AllAtOnceRFGaussSeidelPC",
    "WC4DVarSchurPC",
    "WC4DVarSaddlePC",
    "AutoregressiveCovariance",
    "MixedCovarianceOperator",
    "EnsembleCovarianceOperator",
    "tao_converged_maxits_test",
    "ksp_objective_monitor",
)

from fdvar.allatonce_reduced_functional import (  # noqa: F401
    AllAtOnceReducedFunctional,
)
from fdvar.wc4dvar_reduced_functional import (  # noqa: F401
    WC4DVarReducedFunctional,
)
from fdvar.sc4dvar_reduced_functional import (  # noqa: F401
    SC4DVarReducedFunctional,
)
from fdvar.preconditioners import (  # noqa: F401
    AllAtOnceRFGaussSeidelPC,
    WC4DVarSchurPC,
    WC4DVarSaddlePC,
)
from fdvar.covariance_operator import (  # noqa: F401
    AutoregressiveCovariance,
    MixedCovarianceOperator,
    EnsembleCovarianceOperator,
)
from fdvar.tao import (  # noqa: F401
    tao_converged_maxits_test,
    ksp_objective_monitor,
)
