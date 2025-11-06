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

    def test_characterize_parameter_values(self):
        """Characterization: Script uses specific Lorenz system parameters.

        The script hardcodes these parameter values:
        - sigma = 10 (Prandtl number)
        - rho = 35 (Rayleigh number) ← NOTE: Different from module version (28)!
        - beta = 8/3 ≈ 2.666... (Geometric factor)
        - dt = 0.01 (time step for Euler integration)
        - num_steps = 5000 (total integration steps)

        The rho=35 value will produce different chaotic dynamics compared to
        the standard rho=28 used in src/simulations/lorenz.py.

        This test documents existing behavior before refactoring.
        Based on Michael Feathers' "Working Effectively with Legacy Code".

        Observed: 2025-11-06
        """
        # Mock plt.show() to prevent blocking during import
        with patch('matplotlib.pyplot.show'):
            import lorenz_attractor_xyz

        # Document what it ACTUALLY does: uses these specific parameter values
        assert lorenz_attractor_xyz.sigma == 10
        assert lorenz_attractor_xyz.rho == 35  # Different from standard 28!
        assert lorenz_attractor_xyz.beta == 8 / 3
        assert np.isclose(lorenz_attractor_xyz.beta, 2.666666666666667)

        # Time integration parameters
        assert lorenz_attractor_xyz.dt == 0.01
        assert lorenz_attractor_xyz.num_steps == 5000

    def test_characterize_first_integration_step(self):
        """Characterization: First step of Euler integration produces specific values.

        Using the Lorenz equations with the hardcoded parameters:
        - Initial: [x=0.0, y=1.0, z=1.05]
        - sigma=10, rho=35, beta=8/3, dt=0.01

        After one Euler step (index 1):
        - dx = sigma * (y - x) * dt = 10 * (1.0 - 0.0) * 0.01 = 0.1
        - dy = (x * (rho - z) - y) * dt = (0.0 * (35 - 1.05) - 1.0) * 0.01 = -0.01
        - dz = (x * y - beta * z) * dt = (0.0 * 1.0 - 2.666... * 1.05) * 0.01 = -0.028

        Result: [x=0.1, y=0.99, z=1.022]

        This test documents existing behavior before refactoring.
        Based on Michael Feathers' "Working Effectively with Legacy Code".

        Observed: 2025-11-06
        """
        # Mock plt.show() to prevent blocking during import
        with patch('matplotlib.pyplot.show'):
            import lorenz_attractor_xyz

        # Document what it ACTUALLY computes at step 1
        assert np.isclose(lorenz_attractor_xyz.xs[1], 0.1)
        assert np.isclose(lorenz_attractor_xyz.ys[1], 0.99)
        assert np.isclose(lorenz_attractor_xyz.zs[1], 1.022)

    def test_characterize_trajectory_after_100_steps(self):
        """Characterization: After 100 steps, trajectory reaches specific values.

        This test locks down the cumulative behavior of the numerical integration
        over 100 time steps. The chaotic nature of the Lorenz system means even
        tiny changes in computation will cause this test to fail.

        After 100 steps (1 second of simulated time at dt=0.01):
        - x ≈ 7.373
        - y ≈ 11.379
        - z ≈ 24.872

        This test documents existing behavior before refactoring.
        Based on Michael Feathers' "Working Effectively with Legacy Code".

        Observed: 2025-11-06
        """
        # Mock plt.show() to prevent blocking during import
        with patch('matplotlib.pyplot.show'):
            import lorenz_attractor_xyz

        # Document what it ACTUALLY computes at step 100
        # Using rtol=1e-10 to catch even tiny numerical changes
        assert np.isclose(lorenz_attractor_xyz.xs[100], 7.373312457204771, rtol=1e-10)
        assert np.isclose(lorenz_attractor_xyz.ys[100], 11.378963072941088, rtol=1e-10)
        assert np.isclose(lorenz_attractor_xyz.zs[100], 24.87171753798129, rtol=1e-10)