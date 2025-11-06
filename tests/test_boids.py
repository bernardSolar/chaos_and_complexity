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
from boids import normalize, limit_magnitude


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

    def test_characterize_normalize_zero_vector(self):
        """Characterization: normalize() returns zero vector for zero input.

        This documents the defensive zero-handling behavior that prevents
        division by zero errors.

        Observed: 2025-11-06
        """
        zero_vector = np.array([0.0, 0.0, 0.0])

        result = normalize(zero_vector)

        # Document actual behavior: returns zero vector unchanged
        assert np.allclose(result, np.array([0.0, 0.0, 0.0]))
        assert np.linalg.norm(result) == 0.0


class TestLimitMagnitudeFunction:
    """Characterization tests for the limit_magnitude() utility function."""

    def test_characterize_limit_magnitude_exceeds_max(self):
        """Characterization: limit_magnitude() clamps vectors exceeding maximum.

        When a vector's magnitude exceeds the max_magnitude parameter,
        the vector is scaled down to exactly max_magnitude while preserving
        direction.

        Observed: 2025-11-06
        """
        input_vector = np.array([6.0, 8.0, 0.0])  # magnitude = 10.0
        max_magnitude = 5.0

        result = limit_magnitude(input_vector, max_magnitude)

        # Document actual behavior: scaled to max while preserving direction
        assert np.allclose(result, np.array([3.0, 4.0, 0.0]))
        assert np.isclose(np.linalg.norm(result), 5.0)
        # Verify direction preserved (same unit vector)
        expected_direction = input_vector / np.linalg.norm(input_vector)
        actual_direction = result / np.linalg.norm(result)
        assert np.allclose(expected_direction, actual_direction)

    def test_characterize_limit_magnitude_below_max(self):
        """Characterization: limit_magnitude() passes through vectors below max.

        When a vector's magnitude is already below max_magnitude, it is
        returned unchanged (pass-through behavior).

        Observed: 2025-11-06
        """
        input_vector = np.array([2.0, 1.0, 0.0])  # magnitude ≈ 2.236
        max_magnitude = 10.0

        result = limit_magnitude(input_vector, max_magnitude)

        # Document actual behavior: returned unchanged
        assert np.array_equal(result, input_vector)
        assert np.isclose(np.linalg.norm(result), np.linalg.norm(input_vector))
