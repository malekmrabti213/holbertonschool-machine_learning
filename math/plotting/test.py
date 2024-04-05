#!/usr/bin/env python3
"""Module for generating and saving a scatter plot."""

import numpy as np
import matplotlib.pyplot as plt

def generate_scatter_plot():
    """Generate a scatter plot of men's height vs weight."""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    plt.scatter(x, y, c='magenta', s=10)
    plt.xlabel("Height (in)")
    plt.ylabel("Height (lbs)")
    plt.title("Men's Height vs Weight")
    plt.show()

def main():
    """Generate and save the scatter plot."""
    plt.ion()
    generate_scatter_plot()

    plt.savefig('scatter_plot.png')  # Save the plot as an image file
    plt.close()  # Close the plot without displaying it

if __name__ == "__main__":
    main()

