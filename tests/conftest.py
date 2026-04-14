"""Global test configuration."""

import os

import pytest
from pyadjoint.tape import (
    annotate_tape, get_working_tape, set_working_tape,
    continue_annotation, pause_annotation
)

# Disable warnings for missing options when running with pytest as PETSc does
# not know what to do with the pytest arguments.
os.environ["FIREDRAKE_DISABLE_OPTIONS_LEFT"] = "1"


@pytest.fixture(scope="module", autouse=True)
def check_empty_tape(request):
    """Check that the tape is empty at the end of each module.
    """
    def finalizer():
        # make sure taping is switched off
        assert not annotate_tape()

        # make sure the tape is empty
        tape = get_working_tape()
        if tape is not None:
            assert len(tape.get_blocks()) == 0

    request.addfinalizer(finalizer)


@pytest.fixture
def set_test_tape():
    """Set a new working tape specifically for this test.
    """
    continue_annotation()
    with set_working_tape():
        yield
    pause_annotation()
