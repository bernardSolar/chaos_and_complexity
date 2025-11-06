"""
Characterization tests for lorenz_attractor_xyz.py

These tests document the actual current behavior of the legacy standalone
Lorenz attractor visualization script.

Following Michael Feathers' "Working Effectively with Legacy Code" methodology.

Tests are written to PASS immediately, documenting what the code DOES,
not what it SHOULD do.
"""
import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path to import lorenz_attractor_xyz
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLorenzAttractorXYZCharacterization:
    """Characterization tests for the legacy lorenz_attractor_xyz.py script."""

    def test_characterize_arrays_have_correct_shape(self):
        """Characterization: Script creates three arrays of length num_steps.

        The script uses np.empty(num_steps) to create xs, ys, and zs arrays,
        then fills them via iteration. All three arrays should have shape (5000,).

        This test documents existing behavior before refactoring.
        Based on Michael Feathers' "Working Effectively with Legacy Code".

        Observed: 2025-11-06
        """
        # Mock plt.show() to prevent blocking during import
        with patch('matplotlib.pyplot.show'):
            # Import triggers all module-level computation
            import lorenz_attractor_xyz

        # Document what it ACTUALLY does: creates arrays with correct shape
        assert lorenz_attractor_xyz.xs.shape == (5000,)
        assert lorenz_attractor_xyz.ys.shape == (5000,)
        assert lorenz_attractor_xyz.zs.shape == (5000,)

        # Verify they are NumPy arrays
        assert isinstance(lorenz_attractor_xyz.xs, np.ndarray)
        assert isinstance(lorenz_attractor_xyz.ys, np.ndarray)
        assert isinstance(lorenz_attractor_xyz.zs, np.ndarray)

    def test_characterize_initial_conditions(self):
        """Characterization: Script uses fixed initial conditions [0.0, 1.0, 1.05].

        The first element of each array (index 0) contains the initial state:
        - xs[0] = 0.0 (convection rate)
        - ys[0] = 1.0 (temperature difference)
        - zs[0] = 1.05 (temperature profile deviation)

        These specific values are hardcoded in the script (line 20).

        This test documents existing behavior before refactoring.
        Based on Michael Feathers' "Working Effectively with Legacy Code".

        Observed: 2025-11-06
        """
        # Mock plt.show() to prevent blocking during import
        with patch('matplotlib.pyplot.show'):
            import lorenz_attractor_xyz

        # Document what it ACTUALLY does: sets specific initial conditions
        assert lorenz_attractor_xyz.xs[0] == 0.0
        assert lorenz_attractor_xyz.ys[0] == 1.0
        assert lorenz_attractor_xyz.zs[0] == 1.05

        # Verify these are the exact float values
        assert isinstance(lorenz_attractor_xyz.xs[0], (float, np.floating))
        assert isinstance(lorenz_attractor_xyz.ys[0], (float, np.floating))
        assert isinstance(lorenz_attractor_xyz.zs[0], (float, np.floating))