"""Make the track packages importable from the repo root (pytest + scripts).

Each track lives under ``tracks/`` and keeps its own package name (e.g. ``mammalps_b1``).
Adding the track roots (and ``common/``) to ``sys.path`` here lets ``import mammalps_b1...``
and the future ``mabe22`` / ``common`` packages resolve without installing anything — so the
moved tests keep working unchanged.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _rel in ("tracks/mammalps_baseline", "tracks/mabe22_contact", "common", "."):
    _p = str(_ROOT / _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)
