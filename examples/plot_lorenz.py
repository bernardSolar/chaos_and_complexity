"""
Lorenz attractor visualization with animation.

This script demonstrates how to use the Lorenz computation module
and visualize the results as an animated 3D plot.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from src.simulations.lorenz import compute_lorenz_trajectory


def main():
    """Generate and animate the Lorenz attractor."""

    # Parameters for the Lorenz system
    sigma = 10.0  # Prandtl number
    rho = 28.0    # Rayleigh number
    beta = 8.0 / 3.0  # Geometric factor

    # Time parameters
    dt = 0.01  # Time step
    num_steps = 10000  # Number of steps

    # Initial conditions
    initial_state = [-1.0, -1.0, 1.0]

    # Compute the trajectory
    print(f"Computing Lorenz attractor ({num_steps} steps)...")
    trajectory = compute_lorenz_trajectory(
        initial_state=initial_state,
        num_steps=num_steps,
        dt=dt,
        sigma=sigma,
        rho=rho,
        beta=beta
    )
    print("Computation complete!")

    # Extract x, y, z for easier access
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([0, 50])
    ax.set_title("Lorenz Attractor (Evolving)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Line object to update dynamically
    line, = ax.plot([], [], [], lw=0.5)

    # Update function for the animation
    def update(num):
        line.set_data(x[:num], y[:num])
        line.set_3d_properties(z[:num])
        return line,

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update,
        frames=num_steps,
        interval=1,
        blit=True
    )

    plt.show()


if __name__ == "__main__":
    main()
