"""
Characterization tests for mandelbrot function.

These tests document the current behavior of the existing mandelbrot
implementation before refactoring. Following Michael Feathers' approach
from "Working Effectively with Legacy Code".
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Import from new package structure (doesn't exist yet - will fail)
from src.fractals.mandelbrot import mandelbrot


class TestMandelbrotCharacterization:
    """Characterization tests documenting current mandelbrot behavior."""

    def test_returns_array_with_correct_shape(self):
        """The function should return an array matching input dimensions."""
        h, w = 100, 150
        result = mandelbrot(h, w, max_iter=50)

        assert result.shape == (h, w)

    def test_returns_integer_array(self):
        """The function should return integer iteration counts."""
        result = mandelbrot(10, 10, max_iter=50)

        assert result.dtype == int

    def test_values_bounded_by_max_iter(self):
        """All values should be between 0 and max_iter (inclusive)."""
        max_iter = 100
        result = mandelbrot(50, 50, max_iter=max_iter)

        assert np.all(result >= 0)
        assert np.all(result <= max_iter)

    def test_center_point_is_in_set(self):
        """The origin (0+0i) is in the Mandelbrot set.

        Points in the set should return max_iter.
        The center of our coordinate system should map close to origin.
        """
        # Use high resolution to get a point very close to origin
        h, w = 1000, 1500
        max_iter = 100
        result = mandelbrot(h, w, max_iter=max_iter)

        # The coordinate system is [-1.4, 1.4] x [-2, 0.8]
        # Center should be approximately at (-0.6, 0)
        # Let's sample the middle region where we expect points in the set
        center_h = h // 2
        center_w_region = result[center_h, w//3:2*w//3]

        # At least some points near the main body should be in the set
        points_in_set = np.sum(center_w_region == max_iter)
        assert points_in_set > 0, "Expected some points in the set near the center"

    def test_far_points_diverge_quickly(self):
        """Points far from origin should diverge in few iterations."""
        h, w = 100, 150
        max_iter = 100
        result = mandelbrot(h, w, max_iter=max_iter)

        # Check corners (far from the set)
        corners = [
            result[0, 0],      # top-left
            result[0, -1],     # top-right
            result[-1, 0],     # bottom-left
            result[-1, -1],    # bottom-right
        ]

        # All corners should diverge quickly (much less than max_iter)
        for corner_value in corners:
            assert corner_value < max_iter // 2, \
                f"Expected corner to diverge quickly, got {corner_value}"

    def test_consistent_results(self):
        """Running the function twice should give identical results."""
        h, w, max_iter = 50, 75, 80

        result1 = mandelbrot(h, w, max_iter)
        result2 = mandelbrot(h, w, max_iter)

        np.testing.assert_array_equal(result1, result2)

    def test_max_iter_affects_detail(self):
        """Higher max_iter should provide more detail (different values)."""
        h, w = 100, 150

        result_low = mandelbrot(h, w, max_iter=10)
        result_high = mandelbrot(h, w, max_iter=100)

        # The high iteration result should have more unique values
        unique_low = len(np.unique(result_low))
        unique_high = len(np.unique(result_high))

        assert unique_high > unique_low, \
            "Higher max_iter should provide more iteration count variety"
