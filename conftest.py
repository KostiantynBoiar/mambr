"""Put the repo root on ``sys.path`` so ``import zoomtrack`` resolves (pytest + scripts).

ZoomTrack is a single root package; nothing is pip-installed. Tests live under ``tests/`` and import
``from zoomtrack...``; scripts run as ``PYTHONPATH=. python -m zoomtrack.scripts.<name>``.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
