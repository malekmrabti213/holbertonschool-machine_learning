#!/usr/bin/env python3
"""
Task 1
"""
import tensorflow as tf


def crop_image(image, size):
    """
    """
    return tf.image.random_crop(image, size)
