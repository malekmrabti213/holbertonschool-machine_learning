#!/usr/bin/env python3
"""
Advanced Task
"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    """
    h, w, _ = tf.shape(image)
    flat_image = tf.reshape(image, [-1, 3])
    flat_image = tf.cast(flat_image, tf.float32)
    mean = tf.math.reduce_mean(flat_image, axis=0)
    flat_image -= mean
    std = tf.math.reduce_std(flat_image, axis=0)
    flat_image /= std
    # print(flat_image)
    cov = tf.linalg.matmul(flat_image,flat_image, transpose_a=True) / tf.cast(h * w, dtype=tf.float32)
    l, v = tf.linalg.eigh(cov)
    alphas = tf.convert_to_tensor(alphas, dtype=tf.float32)
    # print(alphas)
    l = tf.reshape(l * alphas, [3, 1])
    # print(l)
    deltas = tf.linalg.matmul(v, l)
    # print(deltas)
    flat_image += tf.transpose(deltas)
    flat_image *= std
    flat_image += mean
    flat_image = tf.math.maximum(tf.math.minimum(flat_image, 255), 0)
    flat_image = tf.cast(flat_image, tf.uint8)
    return tf.reshape(flat_image, [h, w, 3])
