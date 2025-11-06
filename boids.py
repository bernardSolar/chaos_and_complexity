import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random


class Boid:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)

    def update(self, bounds):
        self.velocity += self.acceleration
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity = self.velocity / speed * MAX_SPEED

        self.position += self.velocity

        # Wrap around bounds
        self.position = np.mod(self.position, bounds)
        self.acceleration = np.zeros(3)


def separation(boid, boids, radius):
    steering = np.zeros(3)
    total = 0

    for other in boids:
        d = np.linalg.norm(boid.position - other.position)
        if 0 < d < radius:
            diff = boid.position - other.position
            diff = diff / d  # Weight by distance
            steering += diff
            total += 1

    if total > 0:
        steering = steering / total
        steering = normalize(steering) * MAX_SPEED
        steering = steering - boid.velocity
        steering = limit_magnitude(steering, MAX_FORCE)

    return steering


def alignment(boid, boids, radius):
    steering = np.zeros(3)
    total = 0

    for other in boids:
        d = np.linalg.norm(boid.position - other.position)
        if 0 < d < radius:
            steering += other.velocity
            total += 1

    if total > 0:
        steering = steering / total
        steering = normalize(steering) * MAX_SPEED
        steering = steering - boid.velocity
        steering = limit_magnitude(steering, MAX_FORCE)

    return steering


def cohesion(boid, boids, radius):
    steering = np.zeros(3)
    total = 0

    for other in boids:
        d = np.linalg.norm(boid.position - other.position)
        if 0 < d < radius:
            steering += other.position
            total += 1

    if total > 0:
        steering = steering / total
        steering = steering - boid.position
        steering = normalize(steering) * MAX_SPEED
        steering = steering - boid.velocity
        steering = limit_magnitude(steering, MAX_FORCE)

    return steering


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def limit_magnitude(v, max_magnitude):
    mag = np.linalg.norm(v)
    if mag > max_magnitude:
        return v / mag * max_magnitude
    return v


def get_boid_color(boid, boids):
    """Determine boid color based on neighbor proximity.

    Returns:
        tuple: RGB color as (r, g, b) where each component is 0.0-1.0
    """
    # Count neighbors within close range (crowded threshold)
    crowded_threshold = SEPARATION_RADIUS / 2

    close_neighbors = 0
    nearby_neighbors = 0

    for other in boids:
        if other is boid:
            continue
        distance = np.linalg.norm(boid.position - other.position)

        if distance < crowded_threshold:
            close_neighbors += 1

        if distance < COHESION_RADIUS:
            nearby_neighbors += 1

    # Red if too crowded
    if close_neighbors > 0:
        return (1.0, 0.0, 0.0)

    # Yellow if isolated (no neighbors within cohesion radius)
    if nearby_neighbors == 0:
        return (1.0, 1.0, 0.0)

    # Green if comfortable spacing (neighbors present but not too close)
    return (0.0, 1.0, 0.0)


def draw_boid(position, velocity, color=(1.0, 1.0, 1.0)):
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

    # Draw triangular boid
    glBegin(GL_TRIANGLES)
    glVertex3f(2, 0, 0)
    glVertex3f(-2, 1, 0)
    glVertex3f(-2, -1, 0)
    glEnd()

    glPopMatrix()


# Simulation parameters
WIDTH, HEIGHT = 800, 600
DEPTH = 800
NUM_BOIDS = 50
MAX_SPEED = 2.0
MAX_FORCE = 0.03
SEPARATION_RADIUS = 25
ALIGNMENT_RADIUS = 50
COHESION_RADIUS = 50

# Initial spawn area size (reduced for tighter grouping)
SPAWN_AREA_SIZE = 200

# Initialize Pygame and OpenGL
pygame.init()
display = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
gluPerspective(45, (WIDTH / HEIGHT), 0.1, 2000.0)
glTranslatef(-WIDTH / 2, -HEIGHT / 2, -1000)

# Create initial boids with tighter grouping
center = np.array([WIDTH / 2, HEIGHT / 2, DEPTH / 2])
boids = []
for _ in range(NUM_BOIDS):
    # Position boids around the center within SPAWN_AREA_SIZE
    position = center + np.array([
        random.uniform(-SPAWN_AREA_SIZE / 2, SPAWN_AREA_SIZE / 2),
        random.uniform(-SPAWN_AREA_SIZE / 2, SPAWN_AREA_SIZE / 2),
        random.uniform(-SPAWN_AREA_SIZE / 2, SPAWN_AREA_SIZE / 2)
    ])
    velocity = np.array([
        random.uniform(-1, 1),
        random.uniform(-1, 1),
        random.uniform(-1, 1)
    ])
    boids.append(Boid(position, velocity))

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
        separation_force = separation(boid, boids, SEPARATION_RADIUS)
        alignment_force = alignment(boid, boids, ALIGNMENT_RADIUS)
        cohesion_force = cohesion(boid, boids, COHESION_RADIUS)

        # Apply forces with weights
        boid.acceleration += separation_force * 1.5
        boid.acceleration += alignment_force * 1.0
        boid.acceleration += cohesion_force * 1.0

        # Update position
        boid.update(np.array([WIDTH, HEIGHT, DEPTH]))

        # Get color based on neighbor proximity
        color = get_boid_color(boid, boids)

        # Draw boid with color
        draw_boid(boid.position, boid.velocity, color)

    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()