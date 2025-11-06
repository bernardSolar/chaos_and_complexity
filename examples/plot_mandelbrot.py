"""
Mandelbrot set visualization example.

This script demonstrates how to use the mandelbrot computation module
and visualize the results with matplotlib.
"""

import matplotlib.pyplot as plt
from src.fractals.mandelbrot import mandelbrot


def main():
    """Generate and display the Mandelbrot set."""
    # Computation parameters
    h, w = 1000, 1500
    max_iter = 100

    # Compute the Mandelbrot set
    print(f"Computing Mandelbrot set ({h}x{w}, max_iter={max_iter})...")
    result = mandelbrot(h, w, max_iter)
    print("Computation complete!")

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(
        result,
        cmap='magma',
        extent=[-2, 0.8, -1.4, 1.4],
        origin='lower'
    )
    plt.title("Mandelbrot Set")
    plt.xlabel("Real axis")
    plt.ylabel("Imaginary axis")
    plt.colorbar(label='Iteration count')
    plt.show()


if __name__ == "__main__":
    main()
