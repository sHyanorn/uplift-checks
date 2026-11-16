"""Package versions fixed by the registration (Section 12); any other version stops a run (Section 10)."""

import platform
import sys
from importlib.metadata import version

PYTHON = (3, 13)
PINNED = {
    "numpy": "2.5.4",
    "pandas": "3.1.1",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.1",
    "lightgbm": "4.7.0",
    "torch": "2.15.0+cpu",
    "econml": "0.17.0",
    "highspy": "1.16.0",
    "pyarrow": "26.0.1",
    "causalml": "0.18.0",
    "scikit-uplift": "0.5.1",
}


class VersionMismatch(RuntimeError):
    pass


def installed():
    return {name: version(name) for name in PINNED}


def check(found=None, python=None):
    """Return the installed versions, or raise VersionMismatch naming every package that differs."""
    found = installed() if found is None else found
    python = sys.version_info[:2] if python is None else python
    wrong = [f"{name} {found.get(name)} (registered {pinned})" for name, pinned in PINNED.items() if found.get(name) != pinned]
    if tuple(python) != PYTHON:
        wrong.append(f"Python {python[0]}.{python[1]} (registered {PYTHON[0]}.{PYTHON[1]})")
    if wrong:
        raise VersionMismatch("; ".join(wrong))
    return {"python": platform.python_version(), **found}
