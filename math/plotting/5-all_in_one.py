#!/usr/bin/env python3
"""
Task 5
"""

import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure(figsize=(6, 6))
    fig.suptitle('All in One')
    spec = fig.add_gridspec(3, 2, wspace=0.4, hspace=0.5)

    ax10 = fig.add_subplot(spec[0, 0])
    ax10.plot(y0, color='red')
    ax10.set_xlim(0, 10)

    ax11 = fig.add_subplot(spec[0, 1])
    ax11.scatter(x1, y1, c='magenta')
    ax11.set_xlabel('Height (in)', size='x-small')
    ax11.set_ylabel('Weight (lbs)', size='x-small')
    ax11.set_title("Men's Height vs Weight", size='x-small')

    ax20 = fig.add_subplot(spec[1, 0])
    ax20.plot(x2, y2)
    ax20.set_xlabel('Time (years)', size='x-small')
    ax20.set_ylabel('Fraction Remaining', size='x-small')
    ax20.set_title("Exponential Decay of C-14", size='x-small')
    ax20.set_xlim(0, 28650)
    ax20.set_yscale("log")

    ax21 = fig.add_subplot(spec[1, 1])
    ax21.plot(x3, y31, c='red', linestyle='dashed', label='C-14')
    ax21.plot(x3, y32, c='green', linestyle='solid', label='Ra-226')
    ax21.set_xlabel('Time (years)', size='x-small')
    ax21.set_ylabel('Fraction Remaining', size='x-small')
    ax21.set_title("Exponential Decay of Radioactive Elements", size='x-small')
    ax21.set_xlim(0, 20000)
    ax21.set_ylim(0, 1)
    ax21.legend()

    ax3 = fig.add_subplot(spec[2, :])
    ax3.hist(student_grades, bins=10, range=(0, 100), edgecolor='black')
    ax3.set_title('Project A', size='x-small')
    ax3.set_xlabel('Grades', size='x-small')
    ax3.set_ylabel('Number of Students', size='x-small')
    ax3.set_ylim(0, 30)
    ax3.set_xlim(0, 100)
    ax3.set_xticks(np.arange(0, 110, 10))

    plt.show()
