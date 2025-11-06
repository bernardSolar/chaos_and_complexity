"""
Lorenz attractor computation.

Pure computation module - no visualization.
"""

import numpy as np


def compute_lorenz_trajectory(initial_state, num_steps, dt,
                              sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Compute Lorenz attractor trajectory using Euler integration.

    Args:
        initial_state: Initial [x, y, z] coordinates
        num_steps: Number of integration steps
        dt: Time step size
        sigma: Prandtl number (default: 10.0)
        rho: Rayleigh number (default: 28.0)
        beta: Geometric factor (default: 8/3)

    Returns:
        numpy.ndarray: Trajectory array of shape (num_steps, 3)
    """
    # Create trajectory array
    trajectory = np.zeros((num_steps, 3))

    # Set initial state
    trajectory[0] = initial_state

    # Integrate using Euler method
    for i in range(num_steps - 1):
        x, y, z = trajectory[i]

        # Lorenz equations
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt

        # Update next state
        trajectory[i + 1] = [x + dx, y + dy, z + dz]

    return trajectory
