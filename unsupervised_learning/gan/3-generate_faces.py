#!/usr/bin/env python3
"""
Task 3
"""
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr() :
    """
    """
    def get_generator() :
        """
        """
        noise = tf.keras.layers.Input( shape=(16) )                     
        x = tf.keras.layers.Dense(
            2048 ,activation = tf.keras.layers.Activation("tanh"))(noise)
        x = tf.keras.layers.Reshape((2, 2, 512))(x)


        x = tf.keras.layers.UpSampling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(
            64, (3,3), strides=(1,1), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("tanh")(x)


        x = tf.keras.layers.UpSampling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(
            16, (3,3), strides=(1,1), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("tanh")(x)


        x = tf.keras.layers.UpSampling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(
            1, (3,3), strides=(1,1), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)


        x = tf.keras.layers.Activation("tanh")(x)


        return keras.models.Model(noise, x, name="generator")

    def get_discriminator():
        """
        """
        img_input = tf.keras.layers.Input(shape=(16,16,1))

        x = tf.keras.layers.Conv2D(
            32, (3,3), strides=(1,1), padding="same")(img_input)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Activation("tanh")(x)
        x = tf.keras.layers.Conv2D(
            64, (3,3), strides=(1,1), padding="same")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Activation("tanh")(x)
        x = tf.keras.layers.Conv2D(
            128, (3,3), strides=(1,1), padding="same")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Activation("tanh")(x)
        x = tf.keras.layers.Conv2D(
            256, (3,3), strides=(1,1), padding="same")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Activation("tanh")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1,activation="tanh")(x)
        return keras.models.Model(img_input, x, name="discriminator")

    return get_generator() , get_discriminator()
