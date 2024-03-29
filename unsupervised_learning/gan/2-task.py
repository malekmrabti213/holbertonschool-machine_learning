#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras

class WGAN_GP(keras.Model) :
    def __init__( self, generator , discriminator , latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005,lambda_gp=10):
        super().__init__()                         # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter

        self.learning_rate    = learning_rate
        self.beta_1=.3                              # standard value, but can be changed if necessary
        self.beta_2=.9                              # standard value, but can be changed if necessary

        self.lambda_gp        = lambda_gp                                # <---- New !
        self.dims = self.real_examples.shape                             # <---- New !
        self.len_dims=tf.size(self.dims)                                 # <---- New !
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')   # <---- New !
        self.scal_shape=self.dims.as_list()                              # <---- New !
        self.scal_shape[0]=self.batch_size                               # <---- New !
        for i in range(1,self.len_dims):                                 # <---- New !
            self.scal_shape[i]=1                                         # <---- New !
        self.scal_shape=tf.convert_to_tensor(self.scal_shape)            # <---- New !


        # define the generator loss and optimizer:
        self.generator.loss      = lambda x : -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )

        # define the discriminator loss and optimizer:
        self.discriminator.loss      = lambda x , y : -tf.reduce_mean(x)+tf.reduce_mean(y)
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size :
            size= self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size :
            size= self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # generator of interpolating samples of size batch_size              # <---- New !
    def get_interpolated_sample(self,real_sample,fake_sample):

        u = tf.random.uniform(self.scal_shape)
        v=tf.ones(self.scal_shape)-u
        return u*real_sample+v*fake_sample

    # computing the gradient penalty                                     # <---- New !
    def gradient_penalty(self,interpolated_sample):
        with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated_sample)
                pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)


    # overloading train_step()
    def train_step(self,useless_argument):

        # ---> training of the discriminator
        for _ in range(self.disc_iter) :

            with tf.GradientTape() as tape:

                # watch the discriminator's trainable variables
                tape.watch(self.discriminator.trainable_variables)

                # get a real sample :
                real_sample = self.get_real_sample()
                #print(real_sample)

                # get a fake sample :
                fake_sample = self.get_fake_sample(training=True)
                #print(fake_sample)

                # get the interpolated sample :
                interpolated_sample = self.get_interpolated_sample(real_sample,fake_sample)

                # compute the loss of the discriminator on real and fake samples
                discr_return_on_fake = self.discriminator(fake_sample , training=True)
                discr_return_on_real = self.discriminator(real_sample , training=True)
                discr_loss = self.discriminator.loss(discr_return_on_real, discr_return_on_fake)

                # compute the gradient penalty
                gp = self.gradient_penalty(interpolated_sample)

                # compute the new loss
                new_discr_loss = discr_loss + gp * self.lambda_gp

            # apply gradient descent to the discriminator
            discr_gradient = tape.gradient(new_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(discr_gradient, self.discriminator.trainable_variables))


        # ---> training of the generator
        with tf.GradientTape() as tape:
            # watch the discriminator's trainable variables
            tape.watch(self.generator.trainable_variables)

            # get a fake sample :
            fake_sample = self.get_fake_sample(training=True)

            # compute the loss of the generator on this fake samples
            discr_return_on_fake = self.discriminator(fake_sample, training=True)
            gen_loss = self.generator.loss(discr_return_on_fake)


        # apply gradient descent to the discriminator
        gen_gradient = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))


        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp":gp}

    # def hash_value(self) :
    #     weights = self.generator.weights + self.discriminator.weights
    #     return sum([hash_tensor(weight) for weight in weights])%2**30
