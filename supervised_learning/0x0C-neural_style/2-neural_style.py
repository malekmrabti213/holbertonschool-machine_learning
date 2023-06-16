#!/usr/bin/env python3

"""useless comments"""

import numpy as np
import tensorflow as tf


def check_image_channel_input(img, source):
    """
    Check the channel of given image
    :param img: The image
    :param source: The variable name to error message
    :return:
    """
    if type(img) != np.ndarray or img.shape[-1] != 3:
        raise TypeError(
            "{} must be a numpy.ndarray with shape (h, w, 3)".format(source)
        )


def check_hyperparameter_input(hyperparameter, source):
    """
    Check given hyperparameter
    :param hyperparameter: The hyperparameter
    :param source: The variable name to error message
    :return:
    """
    if type(hyperparameter) not in [float, int] or hyperparameter < 0:
        raise TypeError("{} must be a non-negative number".format(source))


def check_tensor_rank_input(input_layer, source):
    """
    Check the tensor rank
    :param input_layer: The given tensor
    :param source: The variable name to error message
    :return: Nothing
    """
    if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(
            input_layer.shape) != 4:
        raise TypeError("{} must be a tensor of rank 4".format(source))


class NST:
    """Neural style transfer model"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Init function class
        :param style_image: The style_image (?, ?, 3)
        :param content_image: The content image (?, ?, 3)
        :param alpha: The alpha parameter
        :param beta: The beta parameter
        """
        tf.enable_eager_execution()
        check_image_channel_input(style_image, "style_image")
        check_image_channel_input(content_image, "content_image")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        check_hyperparameter_input(alpha, "alpha")
        check_hyperparameter_input(beta, "beta")
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Scale the image to (1, 512 or less, 512 or less, 3)
        :param image: The given image to resize
        :return: The resized image
        """
        check_image_channel_input(image, "image")

        max_dim = max(image.shape[:-1])
        ratio_dims = 512 / max_dim

        new_dims = tuple([int(dim * ratio_dims) for dim in image.shape[:-1]])
        image = tf.expand_dims(image, 0)  # [1, h, w, 3]
        resized_image = tf.image.resize_bicubic(image, new_dims) / 255

        return tf.clip_by_value(resized_image, 0.0, 1.0)

    def load_model(self):
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        x = vgg.input
        model_outputs = []
        content_output = None
        for layer in vgg.layers[1:]:
            if "pool" in layer.name:
                x = tf.keras.layers.AveragePooling2D(pool_size=layer.pool_size, strides=layer.strides, name=layer.name)(x)
            else:
                x = layer(x)
                if layer.name in self.style_layers:
                    model_outputs.append(x)
                if layer.name == self.content_layer:
                    content_output = x
                layer.trainable = False
        model_outputs.append(content_output)
        model = tf.keras.models.Model(vgg.input, model_outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate the gram matrix
        :return: The gram matrix
        """
        check_tensor_rank_input(input_layer, "input_layer")
        # # Checker doesn't like this code

        # coef = 1 / (input_layer.shape[1].value * input_layer.shape[2].value)
        # batch_size, height, width, channels = input_layer.shape
        # flattened_inputs = tf.reshape(
        #     input_layer,
        #     [batch_size, height * width, channels]
        # )
        # gram_matrix = tf.matmul(
        #     flattened_inputs,
        #     flattened_inputs,
        #     transpose_a=True
        # )
        # return gram_matrix * coef

        # Re write the code inspired of github alumni
        batch_size, height, width, channels = input_layer.shape
        flattened_inputs = tf.reshape(
            input_layer,
            [-1, channels]
        )
        gram_matrix = tf.matmul(
            tf.transpose(flattened_inputs),
            flattened_inputs,
        ) / tf.cast(flattened_inputs.shape[0], tf.float32)
        return tf.reshape(gram_matrix, [1, -1, channels])
    
        # flattened_inputs = tf.reshape(
        #     input_layer,
        #     [batch_size * height * width, channels]
        # )
        # gram_matrix = tf.matmul(
        #     tf.transpose(flattened_inputs),
        #     flattened_inputs,
        # ) / tf.cast(batch_size * height * width, tf.float32)
        # return tf.reshape(gram_matrix, [1, -1, channels])