"""Base-paper controllers run closed-loop in SUMO.

Imports the base implementation (`qits`, ../base_imp) unchanged and the SUMO network / trips from
../sumo_data. Nothing in those two folders is modified.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO / "base_imp", _REPO / "sumo_data"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
