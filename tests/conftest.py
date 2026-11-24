import os
import sys
from pathlib import Path

import matplotlib as mpl
import pytest

os.environ["NUMBA_DISABLE_JIT"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def no_usetex():
    """Run each test without LaTeX text rendering.

    Several plotting modules switch on text.usetex when imported, which leaks into
    later tests and needs a LaTeX install (CI has none).
    """
    with mpl.rc_context({"text.usetex": False}):
        yield
