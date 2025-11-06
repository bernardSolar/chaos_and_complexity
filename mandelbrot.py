"""
DEPRECATED: This module is deprecated. Use src.fractals.mandelbrot instead.

For computation: from src.fractals.mandelbrot import mandelbrot
For visualization: see examples/plot_mandelbrot.py

This file provides backwards compatibility but will be removed in a future version.
"""

import warnings

# Import from new location for backwards compatibility
from src.fractals.mandelbrot import mandelbrot

# Issue deprecation warning
warnings.warn(
    "Importing from root 'mandelbrot' module is deprecated. "
    "Use 'from src.fractals.mandelbrot import mandelbrot' instead.",
    DeprecationWarning,
    stacklevel=2
)

# If run as script, show visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Running deprecated script. Use examples/plot_mandelbrot.py instead.")

    h, w = 1000, 1500
    max_iter = 100
    plt.figure(figsize=(12, 8))
    plt.imshow(mandelbrot(h, w, max_iter), cmap='magma', extent=[-2, 0.8, -1.4, 1.4])
    plt.title("Mandelbrot Set")
    plt.colorbar(label='Iteration count')
    plt.show()
