import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Parameters for the Lorenz system
sigma = 10.0  # Prandtl number
rho = 28.0    # Rayleigh number
beta = 8.0 / 3.0  # Geometric factor

# Time parameters
dt = 0.01  # Time step
num_steps = 10000  # Number of steps

# Initialize arrays for x, y, z
x = np.zeros(num_steps)
y = np.zeros(num_steps)
z = np.zeros(num_steps)

# Initial conditions
x[0], y[0], z[0] = -1.0, -1.0, 1.0

# Integrate the Lorenz equations
for i in range(num_steps - 1):
    dx = sigma * (y[i] - x[i]) * dt
    dy = (x[i] * (rho - z[i]) - y[i]) * dt
    dz = (x[i] * y[i] - beta * z[i]) * dt
    x[i + 1] = x[i] + dx
    y[i + 1] = y[i] + dy
    z[i + 1] = z[i] + dz

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
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=1, blit=True)

plt.show()
