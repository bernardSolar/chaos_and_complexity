import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Lorenz system parameters
sigma = 10
rho = 28
beta = 8 / 3

# Time step and duration
dt = 0.01
num_steps = 5000

# Initialize arrays for x, y, z
xs = np.empty(num_steps)
ys = np.empty(num_steps)
zs = np.empty(num_steps)

# Initial conditions
xs[0], ys[0], zs[0] = 0., 1., 1.05

# Compute the Lorenz attractor
for i in range(1, num_steps):
    dx = sigma * (ys[i - 1] - xs[i - 1]) * dt
    dy = (xs[i - 1] * (rho - zs[i - 1]) - ys[i - 1]) * dt
    dz = (xs[i - 1] * ys[i - 1] - beta * zs[i - 1]) * dt
    xs[i] = xs[i - 1] + dx
    ys[i] = ys[i - 1] + dy
    zs[i] = zs[i - 1] + dz

# Set up the figure and axes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 2, 1, projection='3d')  # 3D plot for Lorenz attractor

# Add these lines to set appropriate axis limits
ax1.set_xlim(-30, 30)
ax1.set_ylim(-30, 30)
ax1.set_zlim(0, 50)

# Add labels for clarity
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = plt.subplot(1, 2, 2)                   # 2D plot for x, y, z values

# Initialize the plots
line1, = ax1.plot([], [], [], lw=0.5)
line2_x, = ax2.plot([], [], label='x', color='r')
line2_y, = ax2.plot([], [], label='y', color='g')
line2_z, = ax2.plot([], [], label='z', color='b')

ax2.legend()
ax2.set_xlim(0, num_steps)
ax2.set_ylim(-50, 50)
ax2.set_xlabel("Time step")
ax2.set_ylabel("Values")

# Initialize the animation
def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    line2_x.set_data([], [])
    line2_y.set_data([], [])
    line2_z.set_data([], [])
    return line1, line2_x, line2_y, line2_z

# Update function for the animation
def update(frame):
    # Update the 3D line
    line1.set_data(xs[:frame], ys[:frame])
    line1.set_3d_properties(zs[:frame])
    
    # Update the 2D line plots for x, y, z
    line2_x.set_data(range(frame), xs[:frame])
    line2_y.set_data(range(frame), ys[:frame])
    line2_z.set_data(range(frame), zs[:frame])
    
    return line1, line2_x, line2_y, line2_z

ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=10)

# Show the animation
plt.show()
