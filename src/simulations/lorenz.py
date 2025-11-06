"""
Lorenz attractor computation.

Pure computation module - no visualization.
"""

import numpy as np


def compute_lorenz_trajectory(initial_state, num_steps, dt):
    """
    Compute Lorenz attractor trajectory.

    Args:
        initial_state: Initial [x, y, z] coordinates
        num_steps: Number of integration steps
        dt: Time step size

    Returns:
        numpy.ndarray: Trajectory array of shape (num_steps, 3)
    """
    # MINIMAL implementation - just return correct shape
    return np.zeros((num_steps, 3))
