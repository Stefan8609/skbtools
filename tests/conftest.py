import os
import sys
from pathlib import Path

os.environ["NUMBA_DISABLE_JIT"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
