#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

NST = __import__('1-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("C:/Users/CAMPUSNA/Desktop/ML Projects/supervised learnning/12.neural Style Transfer/starry_night.jpg")
    content_image = mpimg.imread("C:/Users/CAMPUSNA/Desktop/ML Projects/supervised learnning/12.neural Style Transfer/golden_gate.jpg")

    nst = NST(style_image, content_image)
    nst.model.summary()