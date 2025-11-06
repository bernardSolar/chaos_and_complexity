"""
Mandelbrot set computation.

This module provides pure computation functions for generating the Mandelbrot set.
Visualization is handled separately.
"""

import numpy as np


def mandelbrot(h, w, max_iter):
    """
    Compute the Mandelbrot set over a grid.

    Calculates the number of iterations before each point in the complex plane
    diverges (or reaches max_iter if it doesn't diverge).

    Args:
        h (int): Height of the output array (vertical resolution)
        w (int): Width of the output array (horizontal resolution)
        max_iter (int): Maximum number of iterations to test for divergence

    Returns:
        numpy.ndarray: 2D integer array of shape (h, w) where each value
                      represents the iteration count at which that point diverged.
                      Points that don't diverge within max_iter have value max_iter.

    Example:
        >>> result = mandelbrot(100, 150, max_iter=50)
        >>> result.shape
        (100, 150)
        >>> result.dtype
        dtype('int64')
    """
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x + y*1j
    z = c
    divtime = max_iter + np.zeros(z.shape, dtype=int)

    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2

    return divtime
