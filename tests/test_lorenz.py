"""
Characterization tests for Lorenz attractor computation.

Following Michael Feathers' approach - one test at a time.
"""

import numpy as np
import pytest

from src.simulations.lorenz import compute_lorenz_trajectory


class TestLorenzCharacterization:
    """Characterization tests for Lorenz attractor - built incrementally."""

    def test_returns_correct_shape(self):
        """Should return trajectory array with shape (num_steps, 3)."""
        trajectory = compute_lorenz_trajectory(
            initial_state=[1.0, 1.0, 1.0],
            num_steps=100,
            dt=0.01
        )

        assert trajectory.shape == (100, 3)  # 100 steps, 3 dimensions (x,y,z)
