"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def three():
    """Drei Punkte (0,0), (1,1), (2,3). Handrechnung: x̄ = 1, ȳ = 4/3, Σ(x−x̄)(y−ȳ) = 3, Σ(x−x̄)² = 2, Σ(y−ȳ)² = 14/3, r = 3/√(28/3) = 0,981981."""
    return [0.0, 1.0, 2.0], [0.0, 1.0, 3.0]


@pytest.fixture
def four():
    """Vier Punkte (1,1), (2,3), (3,2), (4,4). Handrechnung: x̄ = ȳ = 2,5, Σ(x−x̄)(y−ȳ) = 4, Σ(x−x̄)² = Σ(y−ȳ)² = 5, also r = 0,8.
    Ohne Punkt 1 oder 4: r = 0,5; ohne Punkt 2 oder 3: r = 13/14 = 0,928571."""
    return [1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0]
