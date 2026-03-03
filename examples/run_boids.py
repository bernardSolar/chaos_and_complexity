"""
Boids flocking simulation visualization with PyOpenGL.

This script demonstrates how to use the boids computation module
and visualize the results as an animated 3D flocking simulation.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from src.simulations.boids import (
    Boid, normalize, separation, alignment, cohesion, get_boid_color,
    create_flock,
    DEFAULT_MAX_SPEED, DEFAULT_MAX_FORCE,
    DEFAULT_SEPARATION_RADIUS, DEFAULT_ALIGNMENT_RADIUS, DEFAULT_COHESION_RADIUS,
)


def draw_boid(position, velocity, color=(1.0, 1.0, 1.0), size=1.0):
    """Render a single boid as a triangle in 3D space using OpenGL."""
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])

    # Calculate rotation matrix to align with velocity
    forward = normalize(velocity)
    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    up = np.cross(right, forward)

    rotation_matrix = np.array([
        [right[0], right[1], right[2], 0],
        [up[0], up[1], up[2], 0],
        [forward[0], forward[1], forward[2], 0],
        [0, 0, 0, 1]
    ])

    glMultMatrixf(rotation_matrix)

    # Set boid color
    glColor3f(color[0], color[1], color[2])

    # Draw triangular boid scaled by size
    glBegin(GL_TRIANGLES)
    glVertex3f(2 * size, 0, 0)
    glVertex3f(-2 * size, 1 * size, 0)
    glVertex3f(-2 * size, -1 * size, 0)
    glEnd()

    glPopMatrix()


def main():
    """Run the boids flocking simulation with OpenGL visualization."""

    # Simulation parameters
    WIDTH, HEIGHT = 800, 600
    DEPTH = 800
    NUM_BOIDS = 50
    SPAWN_AREA_SIZE = 200

    # Force weights
    SEPARATION_WEIGHT = 1.5
    ALIGNMENT_WEIGHT = 1.0
    COHESION_WEIGHT = 1.0

    # Initialize Pygame and OpenGL
    pygame.init()
    display = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 2000.0)
    glTranslatef(-WIDTH / 2, -HEIGHT / 2, -1000)

    # Create initial flock
    center = np.array([WIDTH / 2, HEIGHT / 2, DEPTH / 2])
    boids = create_flock(NUM_BOIDS, center, SPAWN_AREA_SIZE)

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Update and draw boids
        for boid in boids:
            # Calculate steering forces
            separation_force = separation(boid, boids, DEFAULT_SEPARATION_RADIUS)
            alignment_force = alignment(boid, boids, DEFAULT_ALIGNMENT_RADIUS)
            cohesion_force = cohesion(boid, boids, DEFAULT_COHESION_RADIUS)

            # Apply forces with weights
            boid.acceleration += separation_force * SEPARATION_WEIGHT
            boid.acceleration += alignment_force * ALIGNMENT_WEIGHT
            boid.acceleration += cohesion_force * COHESION_WEIGHT

            # Update position
            boid.update(np.array([WIDTH, HEIGHT, DEPTH]))

            # Get color based on neighbor proximity
            color = get_boid_color(boid, boids)

            # Draw boid with color at 1.5x size (50% larger)
            draw_boid(boid.position, boid.velocity, color, size=1.5)

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()


if __name__ == "__main__":
    main()
