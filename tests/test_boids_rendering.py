"""
Characterization tests for boids rendering (OpenGL-dependent).

These tests require a Pygame/OpenGL context and test the draw_boid() function
which lives in examples/run_boids.py. They are separated from the pure
computation tests because they have heavyweight display dependencies.

NOTE: These tests only work when a display is available. They segfault without
an active OpenGL context. The original tests only passed because importing
the monolithic boids.py triggered pygame.init() as a module-level side effect.
That incidental coupling has been removed by the separation of concerns
refactoring. These tests now properly declare their display dependency.

Run with: python -m pytest tests/test_boids_rendering.py -v
(requires a display / OpenGL context)
"""
import numpy as np
import inspect
import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulations.boids import Boid, get_boid_color

# Import draw_boid from examples/run_boids.py (not a package, so use importlib)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_boids", project_root / "examples" / "run_boids.py"
)
run_boids_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_boids_module)
draw_boid = run_boids_module.draw_boid


def _has_opengl_context():
    """Check if we can safely call OpenGL functions."""
    try:
        import pygame
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.OPENGL | pygame.DOUBLEBUF)
        return True
    except Exception:
        return False


requires_display = pytest.mark.skipif(
    not _has_opengl_context(),
    reason="Requires Pygame/OpenGL display context"
)


@requires_display
class TestBoidRendering:
    """Tests for boid rendering with colors.

    These tests require an active OpenGL context. They verify that draw_boid()
    accepts the expected parameters (color, size) without crashing.
    """

    def test_draw_boid_accepts_color_parameter(self):
        """draw_boid should accept a color parameter for rendering."""
        position = np.array([400.0, 300.0, 400.0])
        velocity = np.array([1.0, 0.0, 0.0])
        color = (1.0, 0.0, 0.0)  # Red

        draw_boid(position, velocity, color)

    def test_render_boid_uses_proximity_based_color(self):
        """get_boid_color() output should be usable as draw_boid() color input."""
        boid_a = Boid([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boid_b = Boid([8.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        boids = [boid_a, boid_b]

        color = get_boid_color(boid_a, boids)
        assert color == (1.0, 0.0, 0.0)

        draw_boid(boid_a.position, boid_a.velocity, color)

    def test_draw_boid_accepts_size_parameter(self):
        """draw_boid should accept a size parameter to scale boid rendering."""
        position = np.array([400.0, 300.0, 400.0])
        velocity = np.array([1.0, 0.0, 0.0])
        color = (0.0, 1.0, 0.0)  # Green
        size = 1.5

        draw_boid(position, velocity, color, size)

    def test_draw_boid_signature(self):
        """draw_boid should accept position, velocity, color, and size parameters."""
        sig = inspect.signature(draw_boid)
        params = list(sig.parameters.keys())
        assert params == ['position', 'velocity', 'color', 'size']
