#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gridspec


class Simple_GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_generator, real_examples,
                 batch_size=200, disc_iter=2):
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.training_gen_losses = []  # used to monitor the results
        self.training_discr_losses = []  # used to monitor the results

    def sub_real(self):
        indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(indices)[:self.batch_size]
        return tf.gather(self.real_examples, random_indices)

    # overriding train_step()
    def train_step(self):
        for _ in range(self.disc_iter):
            # compute the loss for the discriminator:
            with tf.GradientTape() as tape:
                # get a real sample:
                real_sample = self.sub_real()
                # get a fake sample:
                fake_sample = self.generator(self.latent_generator(self.batch_size), training=True)
                # compute the loss of the discriminator on real and fake samples
                discr_return_on_fake = self.discriminator(fake_sample, training=True)
                discr_return_on_real = self.discriminator(real_sample, training=True)
                discr_loss = self.discriminator.loss(discr_return_on_real, discr_return_on_fake)
            # apply gradient descent to the discriminator
            discr_gradient = tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradient, self.discriminator.trainable_variables))

        # compute the loss for the generator:
        with tf.GradientTape() as tape:
            fake_sample = self.generator(self.latent_generator(self.batch_size), training=True)
            discr_return_on_fake = self.discriminator(fake_sample, training=True)
            gen_loss = self.generator.loss(discr_return_on_fake)
        # apply gradient descent to the discriminator
        gen_gradient = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
