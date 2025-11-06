"""
Characterization tests for boids.py

These tests document the actual current behavior of the boids simulation code.
Following Michael Feathers' "Working Effectively with Legacy Code" methodology.

Tests are written to PASS immediately, documenting what the code DOES,
not what it SHOULD do.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path to import boids
sys.path.insert(0, str(Path(__file__).parent.parent))
from boids import normalize, limit_magnitude, Boid, separation, alignment, cohesion


class TestNormalizeFunction:
    """Characterization tests for the normalize() utility function."""

    def test_characterize_normalize_standard_vector(self):
        """Characterization: normalize() returns unit vector for non-zero input.

        This test documents existing behavior before refactoring.
        Based on Michael Feathers' "Working Effectively with Legacy Code".

        Observed: 2025-11-06
        """
        # Observe: [3, 4, 0] has magnitude 5, should normalize to [0.6, 0.8, 0]
        input_vector = np.array([3.0, 4.0, 0.0])

        result = normalize(input_vector)

        # Document what it ACTUALLY does
        assert np.allclose(result, np.array([0.6, 0.8, 0.0]))
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_characterize_normalize_zero_vector(self):
        """Characterization: normalize() returns zero vector for zero input.

        This documents the defensive zero-handling behavior that prevents
        division by zero errors.

        Observed: 2025-11-06
        """
        zero_vector = np.array([0.0, 0.0, 0.0])

        result = normalize(zero_vector)

        # Document actual behavior: returns zero vector unchanged
        assert np.allclose(result, np.array([0.0, 0.0, 0.0]))
        assert np.linalg.norm(result) == 0.0


class TestLimitMagnitudeFunction:
    """Characterization tests for the limit_magnitude() utility function."""

    def test_characterize_limit_magnitude_exceeds_max(self):
        """Characterization: limit_magnitude() clamps vectors exceeding maximum.

        When a vector's magnitude exceeds the max_magnitude parameter,
        the vector is scaled down to exactly max_magnitude while preserving
        direction.

        Observed: 2025-11-06
        """
        input_vector = np.array([6.0, 8.0, 0.0])  # magnitude = 10.0
        max_magnitude = 5.0

        result = limit_magnitude(input_vector, max_magnitude)

        # Document actual behavior: scaled to max while preserving direction
        assert np.allclose(result, np.array([3.0, 4.0, 0.0]))
        assert np.isclose(np.linalg.norm(result), 5.0)
        # Verify direction preserved (same unit vector)
        expected_direction = input_vector / np.linalg.norm(input_vector)
        actual_direction = result / np.linalg.norm(result)
        assert np.allclose(expected_direction, actual_direction)

    def test_characterize_limit_magnitude_below_max(self):
        """Characterization: limit_magnitude() passes through vectors below max.

        When a vector's magnitude is already below max_magnitude, it is
        returned unchanged (pass-through behavior).

        Observed: 2025-11-06
        """
        input_vector = np.array([2.0, 1.0, 0.0])  # magnitude ≈ 2.236
        max_magnitude = 10.0

        result = limit_magnitude(input_vector, max_magnitude)

        # Document actual behavior: returned unchanged
        assert np.array_equal(result, input_vector)
        assert np.isclose(np.linalg.norm(result), np.linalg.norm(input_vector))


class TestBoidClass:
    """Characterization tests for the Boid class."""

    def test_characterize_boid_initialization(self):
        """Characterization: Boid.__init__() converts inputs to float32 arrays.

        Position and velocity are converted from lists to numpy arrays with
        float32 dtype. Acceleration is initialized to zeros [0, 0, 0] with
        float32 dtype.

        Observed: 2025-11-06
        """
        position = [10.0, 20.0, 30.0]
        velocity = [1.0, 2.0, 3.0]

        boid = Boid(position, velocity)

        # Document actual behavior: conversion to float32 arrays
        assert isinstance(boid.position, np.ndarray)
        assert boid.position.dtype == np.float32
        assert np.array_equal(boid.position, np.array([10.0, 20.0, 30.0]))

        assert isinstance(boid.velocity, np.ndarray)
        assert boid.velocity.dtype == np.float32
        assert np.array_equal(boid.velocity, np.array([1.0, 2.0, 3.0]))

        # Document acceleration initialization
        assert isinstance(boid.acceleration, np.ndarray)
        assert boid.acceleration.dtype == np.float32
        assert np.array_equal(boid.acceleration, np.array([0.0, 0.0, 0.0]))

    def test_characterize_boid_update_basic_movement(self):
        """Characterization: Boid.update() applies forces and moves position.

        The update cycle: acceleration → velocity → position, then reset
        acceleration to zeros. This is the fundamental movement loop.

        Observed: 2025-11-06
        """
        boid = Boid([100.0, 200.0, 300.0], [1.0, 0.5, 0.0])
        boid.acceleration = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        bounds = np.array([800, 600, 800])

        boid.update(bounds)

        # Document actual behavior: acceleration added to velocity
        assert np.allclose(boid.velocity, np.array([1.1, 0.7, 0.3]))

        # Document: new velocity added to position
        assert np.allclose(boid.position, np.array([101.1, 200.7, 300.3]))

        # Document: acceleration reset to zeros after update
        assert np.array_equal(boid.acceleration, np.array([0.0, 0.0, 0.0]))

    def test_characterize_boid_update_boundary_wrapping(self):
        """Characterization: Boid.update() wraps position at boundaries.

        Uses modulo (%) to wrap positions that exceed bounds back to the
        opposite edge. This creates toroidal (Pac-Man style) wrapping behavior.

        Observed: 2025-11-06
        """
        bounds = np.array([800, 600, 800])

        # Test: crossing right boundary (x exceeds 800)
        boid1 = Boid([799.0, 300.0, 400.0], [2.0, 0.0, 0.0])
        boid1.update(bounds)

        # Document: 799 + 2 = 801, then 801 % 800 = 1
        assert np.allclose(boid1.position, np.array([1.0, 300.0, 400.0]))

        # Test: crossing left boundary (x goes negative)
        boid2 = Boid([1.0, 300.0, 400.0], [-2.0, 0.0, 0.0])
        boid2.update(bounds)

        # Document: 1 + (-2) = -1, then -1 % 800 = 799
        assert np.allclose(boid2.position, np.array([799.0, 300.0, 400.0]))


class TestFlockingBehaviors:
    """Characterization tests for Reynolds' boid flocking behaviors."""

    def test_characterize_separation_with_nearby_boid(self):
        """Characterization: separation() steers away from nearby boids.

        When another boid is within separation radius, generates a steering
        force pointing away from that boid. Force is limited to MAX_FORCE.

        Observed: 2025-11-06
        """
        # Two boids 10 units apart on x-axis (within SEPARATION_RADIUS of 25)
        boid_a = Boid([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boid_b = Boid([10.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boids = [boid_a, boid_b]

        separation_radius = 25.0

        steering = separation(boid_a, boids, separation_radius)

        # Document actual behavior: steers away (negative x direction)
        # Magnitude should be exactly MAX_FORCE (0.03)
        assert steering[0] < 0  # Steering in negative x (away from boid_b)
        assert steering[1] == 0  # No y component
        assert steering[2] == 0  # No z component
        assert np.isclose(np.linalg.norm(steering), 0.03)  # Limited to MAX_FORCE

    def test_characterize_separation_outside_radius(self):
        """Characterization: separation() returns zero when no boids nearby.

        When all other boids are outside the separation radius, returns
        zero steering force (no separation needed).

        Observed: 2025-11-06
        """
        # Two boids 100 units apart (outside SEPARATION_RADIUS of 25)
        boid_a = Boid([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boid_b = Boid([100.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boids = [boid_a, boid_b]

        separation_radius = 25.0

        steering = separation(boid_a, boids, separation_radius)

        # Document actual behavior: zero force when no nearby boids
        assert np.array_equal(steering, np.array([0.0, 0.0, 0.0]))
        assert np.linalg.norm(steering) == 0.0

    def test_characterize_alignment_with_neighbors(self):
        """Characterization: alignment() steers toward average neighbor velocity.

        When neighbors are within alignment radius, generates steering force
        toward the average velocity of those neighbors. Force is limited to
        MAX_FORCE.

        Observed: 2025-11-06
        """
        # Three boids with different velocities, all within alignment radius
        boid_a = Boid([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boid_b = Boid([10.0, 0.0, 0.0], [0.0, 1.0, 0.0])  # 10 units away
        boid_c = Boid([0.0, 10.0, 0.0], [0.0, 0.0, 1.5])  # 10 units away
        boids = [boid_a, boid_b, boid_c]

        alignment_radius = 50.0

        steering = alignment(boid_a, boids, alignment_radius)

        # Document actual behavior: steering toward average neighbor velocity
        # Average of B and C velocities: ([0,1,0] + [0,0,1.5]) / 2 = [0, 0.5, 0.75]
        # Steering magnitude should be limited to MAX_FORCE (0.03)
        assert np.isclose(np.linalg.norm(steering), 0.03)

        # Steering should have components in y and z (toward average velocity)
        assert steering[1] > 0  # Positive y component
        assert steering[2] > 0  # Positive z component

    def test_characterize_alignment_outside_radius(self):
        """Characterization: alignment() returns zero when no neighbors nearby.

        When all other boids are outside the alignment radius, returns
        zero steering force (no velocity matching needed).

        Observed: 2025-11-06
        """
        # Two boids 100 units apart (outside ALIGNMENT_RADIUS of 50)
        boid_a = Boid([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boid_b = Boid([100.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        boids = [boid_a, boid_b]

        alignment_radius = 50.0

        steering = alignment(boid_a, boids, alignment_radius)

        # Document actual behavior: zero force when no nearby neighbors
        assert np.array_equal(steering, np.array([0.0, 0.0, 0.0]))
        assert np.linalg.norm(steering) == 0.0

    def test_characterize_cohesion_with_neighbors(self):
        """Characterization: cohesion() steers toward center of mass of neighbors.

        When neighbors are within cohesion radius, generates steering force
        toward the average position (center of mass) of those neighbors.
        Force is limited to MAX_FORCE.

        Observed: 2025-11-06
        """
        # Three boids forming a triangle, all within cohesion radius
        boid_a = Boid([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boid_b = Boid([20.0, 0.0, 0.0], [0.0, 0.5, 0.0])  # 20 units to the right
        boid_c = Boid([0.0, 20.0, 0.0], [0.0, 0.0, 0.5])  # 20 units up
        boids = [boid_a, boid_b, boid_c]

        cohesion_radius = 50.0

        steering = cohesion(boid_a, boids, cohesion_radius)

        # Document actual behavior: steering toward center of mass
        # Center of mass of B and C: ([20,0,0] + [0,20,0]) / 2 = [10, 10, 0]
        # Steering magnitude should be limited to MAX_FORCE (0.03)
        assert np.isclose(np.linalg.norm(steering), 0.03)

        # Steering should point toward center of mass (positive x and y)
        assert steering[0] > 0  # Positive x component (toward center)
        assert steering[1] > 0  # Positive y component (toward center)
        assert steering[2] == 0  # No z component (center is in xy plane)

    def test_characterize_cohesion_outside_radius(self):
        """Characterization: cohesion() returns zero when no neighbors nearby.

        When all other boids are outside the cohesion radius, returns
        zero steering force (no cohesion needed).

        Observed: 2025-11-06
        """
        # Two boids 100 units apart (outside COHESION_RADIUS of 50)
        boid_a = Boid([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boid_b = Boid([100.0, 0.0, 0.0], [0.0, 0.5, 0.0])
        boids = [boid_a, boid_b]

        cohesion_radius = 50.0

        steering = cohesion(boid_a, boids, cohesion_radius)

        # Document actual behavior: zero force when no nearby neighbors
        assert np.array_equal(steering, np.array([0.0, 0.0, 0.0]))
        assert np.linalg.norm(steering) == 0.0


class TestBoidColorDetermination:
    """Tests for determining boid colors based on neighbor proximity."""

    def test_boid_color_red_when_too_crowded(self):
        """Boid should be red when neighbors are too close (crowded).

        When multiple boids are within very close range (< SEPARATION_RADIUS/2),
        the boid should turn red to indicate it's too crowded.
        """
        # Center boid with neighbors very close (within 10 units)
        boid_a = Boid([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boid_b = Boid([8.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boid_c = Boid([0.0, 8.0, 0.0], [0.5, 0.0, 0.0])
        boids = [boid_a, boid_b, boid_c]

        from boids import get_boid_color

        color = get_boid_color(boid_a, boids)

        # Should be red (1.0, 0.0, 0.0) when too crowded
        assert color == (1.0, 0.0, 0.0)

    def test_boid_color_yellow_when_too_isolated(self):
        """Boid should be yellow when no neighbors nearby (isolated).

        When no other boids are within reasonable range (> COHESION_RADIUS),
        the boid should turn yellow to indicate it's too isolated.
        """
        # Boid with no neighbors within cohesion radius
        boid_a = Boid([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boid_b = Boid([100.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boids = [boid_a, boid_b]

        from boids import get_boid_color

        color = get_boid_color(boid_a, boids)

        # Should be yellow (1.0, 1.0, 0.0) when isolated
        assert color == (1.0, 1.0, 0.0)
