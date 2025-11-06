"""
Characterization tests for boids.py

These tests document the actual current behavior of the boids simulation code.
Following Michael Feathers' "Working Effectively with Legacy Code" methodology.

Tests are written to PASS immediately, documenting what the code DOES,
not what it SHOULD do.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path to import boids
sys.path.insert(0, str(Path(__file__).parent.parent))
from boids import normalize


class TestNormalizeFunction:
    """Characterization tests for the normalize() utility function."""

    def test_characterize_normalize_standard_vector(self):
        """Characterization: normalize() returns unit vector for non-zero input.

        This test documents existing behavior before refactoring.
        Based on Michael Feathers' "Working Effectively with Legacy Code".

        Observed: 2025-11-06
        """
        # Observe: [3, 4, 0] has magnitude 5, should normalize to [0.6, 0.8, 0]
        input_vector = np.array([3.0, 4.0, 0.0])

        result = normalize(input_vector)

        # Document what it ACTUALLY does
        assert np.allclose(result, np.array([0.6, 0.8, 0.0]))
        assert np.isclose(np.linalg.norm(result), 1.0)
