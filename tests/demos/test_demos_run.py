import glob
import importlib
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from os.path import abspath, basename, dirname, join, splitext

import pytest


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


Demo = namedtuple("Demo", ["loc", "requirements"])


# fdvar/demos lives two levels above the fdvar/tests/demos directory
CWD = abspath(dirname(__file__))
DEMO_DIR = join(CWD, "..", "..", "demos")

SERIAL_DEMOS = [
    Demo(('wc4dvar_advection', 'wc4dvar_advection'), [])
]
PARALLEL_DEMOS = [
    Demo(('wc4dvar_advection', 'wc4dvar_advection'), [])
]


@pytest.fixture
def env():
    env = os.environ.copy()
    env["MPLBACKEND"] = "pdf"
    return env


def test_no_missing_demos():
    all_demo_locs = {
        demo.loc
        for demos in [SERIAL_DEMOS, PARALLEL_DEMOS]
        for demo in demos
    }
    for rst_file in glob.glob(f"{DEMO_DIR}/*/*.py.rst"):
        rst_path = Path(rst_file)
        demo_dir = rst_path.parent.name
        demo_name, _, _ = rst_path.name.split(".")
        demo_loc = (demo_dir, demo_name)
        assert demo_loc in all_demo_locs
        all_demo_locs.remove(demo_loc)
    assert not all_demo_locs, f"Unrecognised demos listed: {all_demo_locs}"


def _prepare_demo(demo, monkeypatch, tmpdir):
    # Change to the temporary directory (monkeypatch ensures that this
    # is undone when the fixture usage disappears)
    monkeypatch.chdir(tmpdir)

    demo_dir, demo_name = demo.loc
    rst_file = f"{DEMO_DIR}/{demo_dir}/{demo_name}.py.rst"

    # Get the name of the python file that pylit will make
    name = splitext(basename(rst_file))[0]
    py_file = str(tmpdir.join(name))
    # Convert rst demo to runnable python file
    subprocess.check_call(["pylit", rst_file, py_file])
    return Path(py_file)


def _exec_file(py_file):
    # To execute a file we import it. We therefore need to modify sys.path so the
    # tempdir can be found.
    sys.path.insert(0, str(py_file.parent))
    importlib.import_module(py_file.with_suffix("").name)
    sys.path.pop(0)  # cleanup


@pytest.mark.parametrize("demo", SERIAL_DEMOS, ids=["/".join(d.loc) for d in SERIAL_DEMOS])
def test_serial_demo(demo, env, monkeypatch, tmpdir):
    py_file = _prepare_demo(demo, monkeypatch, tmpdir)
    _exec_file(py_file)


@pytest.mark.parallel(2)
@pytest.mark.parametrize("demo", PARALLEL_DEMOS, ids=["/".join(d.loc) for d in PARALLEL_DEMOS])
def test_parallel_demo(demo, env, monkeypatch, tmpdir):
    py_file = _prepare_demo(demo, monkeypatch, tmpdir)
    _exec_file(py_file)
