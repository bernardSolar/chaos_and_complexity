"""Boids flocking simulation - pure computation, no visualization.

Implements Craig Reynolds' Boids algorithm with three classic behaviors:
separation, alignment, and cohesion. Boids are color-coded by proximity:
red = crowded, green = comfortable spacing, yellow = isolated.
"""

import numpy as np

# Default simulation parameters (matching original boids.py values)
DEFAULT_MAX_SPEED = 2.0
DEFAULT_MAX_FORCE = 0.03
DEFAULT_SEPARATION_RADIUS = 25.0
DEFAULT_ALIGNMENT_RADIUS = 50.0
DEFAULT_COHESION_RADIUS = 50.0


class Boid:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)

    def update(self, bounds, max_speed=DEFAULT_MAX_SPEED):
        self.velocity += self.acceleration
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity / speed * max_speed

        self.position += self.velocity

        # Wrap around bounds
        self.position = np.mod(self.position, bounds)
        self.acceleration = np.zeros(3)


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


def separation(boid, boids, radius,
               max_speed=DEFAULT_MAX_SPEED, max_force=DEFAULT_MAX_FORCE):
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
        steering = normalize(steering) * max_speed
        steering = steering - boid.velocity
        steering = limit_magnitude(steering, max_force)

    return steering


def alignment(boid, boids, radius,
              max_speed=DEFAULT_MAX_SPEED, max_force=DEFAULT_MAX_FORCE):
    steering = np.zeros(3)
    total = 0

    for other in boids:
        d = np.linalg.norm(boid.position - other.position)
        if 0 < d < radius:
            steering += other.velocity
            total += 1

    if total > 0:
        steering = steering / total
        steering = normalize(steering) * max_speed
        steering = steering - boid.velocity
        steering = limit_magnitude(steering, max_force)

    return steering


def cohesion(boid, boids, radius,
             max_speed=DEFAULT_MAX_SPEED, max_force=DEFAULT_MAX_FORCE):
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
        steering = normalize(steering) * max_speed
        steering = steering - boid.velocity
        steering = limit_magnitude(steering, max_force)

    return steering


def get_boid_color(boid, boids,
                   separation_radius=DEFAULT_SEPARATION_RADIUS,
                   cohesion_radius=DEFAULT_COHESION_RADIUS):
    """Determine boid color based on neighbor proximity.

    Returns:
        tuple: RGB color as (r, g, b) where each component is 0.0-1.0
    """
    # Count neighbors within close range (crowded threshold)
    crowded_threshold = separation_radius / 2

    close_neighbors = 0
    nearby_neighbors = 0

    for other in boids:
        if other is boid:
            continue
        distance = np.linalg.norm(boid.position - other.position)

        if distance < crowded_threshold:
            close_neighbors += 1

        if distance < cohesion_radius:
            nearby_neighbors += 1

    # Red if too crowded
    if close_neighbors > 0:
        return (1.0, 0.0, 0.0)

    # Yellow if isolated (no neighbors within cohesion radius)
    if nearby_neighbors == 0:
        return (1.0, 1.0, 0.0)

    # Green if comfortable spacing (neighbors present but not too close)
    return (0.0, 1.0, 0.0)


def create_flock(num_boids, center, spawn_area_size=200):
    """Create a flock of boids clustered around a center point.

    Args:
        num_boids: Number of boids to create.
        center: 3D center point as array-like [x, y, z].
        spawn_area_size: Size of the spawning area around center.

    Returns:
        List of Boid instances.
    """
    import random
    center = np.array(center)
    flock = []
    for _ in range(num_boids):
        position = center + np.array([
            random.uniform(-spawn_area_size / 2, spawn_area_size / 2),
            random.uniform(-spawn_area_size / 2, spawn_area_size / 2),
            random.uniform(-spawn_area_size / 2, spawn_area_size / 2)
        ])
        velocity = np.array([
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ])
        flock.append(Boid(position, velocity))
    return flock
