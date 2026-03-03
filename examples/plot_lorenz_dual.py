"""
Lorenz attractor dual-panel visualization with animation.

Shows two synchronized views:
- Left: 3D trajectory of the Lorenz attractor
- Right: 2D time evolution of x, y, z variables

This preserves the dual-panel layout from the original lorenz_attractor_xyz.py
while using the clean computation module.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.simulations.lorenz import compute_lorenz_trajectory


def main():
    """Generate and animate the Lorenz attractor with dual-panel display."""

    # Parameters for the Lorenz system
    sigma = 10.0
    rho = 35.0      # Strong chaos regime
    beta = 8.0 / 3.0
    dt = 0.01
    num_steps = 5000
    initial_state = [0.0, 1.0, 1.05]

    # Compute the trajectory
    print(f"Computing Lorenz attractor (rho={rho}, {num_steps} steps)...")
    trajectory = compute_lorenz_trajectory(
        initial_state=initial_state,
        num_steps=num_steps,
        dt=dt,
        sigma=sigma,
        rho=rho,
        beta=beta
    )
    print("Computation complete!")

    xs = trajectory[:, 0]
    ys = trajectory[:, 1]
    zs = trajectory[:, 2]

    # Set up dual-panel figure
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1, projection='3d')
    ax2 = plt.subplot(1, 2, 2)

    # 3D plot setup
    ax1.set_xlim(-30, 30)
    ax1.set_ylim(-30, 30)
    ax1.set_zlim(0, 50)
    ax1.set_xlabel('Convection Rate (x)')
    ax1.set_ylabel('Temperature Difference (y)')
    ax1.set_zlabel('Temperature Profile Deviation (z)')
    ax1.set_title('Lorenz Atmospheric Convection Model')

    # 2D time series setup
    ax2.set_xlim(0, num_steps)
    ax2.set_ylim(-50, 50)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Variable Magnitude')
    ax2.set_title('Evolution of Convection Parameters')

    # Initialize plot lines
    line1, = ax1.plot([], [], [], lw=0.5)
    line2_x, = ax2.plot([], [], label='Convection Rate (x)', color='r')
    line2_y, = ax2.plot([], [], label='Temp Difference (y)', color='g')
    line2_z, = ax2.plot([], [], label='Temp Profile (z)', color='b')
    ax2.legend()

    # Parameter annotation
    fig.text(0.02, 0.02,
             f'\u03c3 (Prandtl) = {sigma}\n\u03c1 (Rayleigh) = {rho}\n\u03b2 = {beta:.4f}',
             fontsize=8, family='monospace')

    def init():
        line1.set_data([], [])
        line1.set_3d_properties([])
        line2_x.set_data([], [])
        line2_y.set_data([], [])
        line2_z.set_data([], [])
        return line1, line2_x, line2_y, line2_z

    def update(frame):
        line1.set_data(xs[:frame], ys[:frame])
        line1.set_3d_properties(zs[:frame])
        line2_x.set_data(range(frame), xs[:frame])
        line2_y.set_data(range(frame), ys[:frame])
        line2_z.set_data(range(frame), zs[:frame])
        return line1, line2_x, line2_y, line2_z

    ani = FuncAnimation(fig, update, frames=num_steps,
                        init_func=init, blit=True, interval=10)

    plt.show()


if __name__ == "__main__":
    main()
