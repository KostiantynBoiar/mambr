"""End-to-end smoke test on the mock backbone (M1).

Builds a synthetic frame with a small blob below the coarse min-area floor; asserts coarse-only
recall < coarse+tiling recall (the small animal is recovered only by zooming a tile). Backbone-free.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="M1: implement geometry/masks/fusion/mock + Pipeline mock path"
)


def test_tiling_recovers_small_animal() -> None:
    """coarse-only recall < coarse+tiling recall on the synthetic small-animal frame."""
    ...
