#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the function to generate the student's plot
gradient = __import__('100-gradient').gradient

plt.ion()
gradient()
plt.savefig('student.png') 
plt.close()

# Define a function to generate the reference plot
def generate_reference_plot():
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y)) # color of points

    # your code here
    plt.title('Mountain Elevation')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.scatter(x,y,c=z)
    plt.colorbar(label='elevation (m)')
    plt.show()
    
generate_reference_plot()
plt.savefig('reference.png') 
plt.close()

student_plot = mpimg.imread('student.png')
reference_plot = mpimg.imread('reference.png')


# Compare the pixel values of the two images
if np.array_equal(student_plot, reference_plot):
    print("The plot matches the reference.")
else:
    print("The plot does not match the reference.")
