from firedrake.petsc import PETSc
from petsctools import OptionsManager
from contextlib import contextmanager

__all__ = (
    "attach_options",
    "get_options",
    "inserted_options",
    "set_from_options",
)


def attach_options(obj, parameters=None,
                   options_prefix=None):
    if "options" in obj.getDict():
        raise ValueError(
            "OptionsManager can only be attached once to each PETSc object")
    options = OptionsManager(
        parameters=parameters,
        options_prefix=options_prefix)
    obj.setAttr("options", options)
    return obj


def get_options(obj):
    return obj.getAttr("options")


@contextmanager
def inserted_options(obj):
    with get_options(obj).inserted_options():
        yield


def set_from_options(obj):
    get_options(obj).set_from_options(obj)
    return obj
