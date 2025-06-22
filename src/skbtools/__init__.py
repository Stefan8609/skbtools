"""Top-level package providing access to project submodules."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__path__.append(str(Path(__file__).resolve().parent.parent))

__all__ = ["geometry", "acoustics", "plotting", "GeigerMethod", "ecco"]
