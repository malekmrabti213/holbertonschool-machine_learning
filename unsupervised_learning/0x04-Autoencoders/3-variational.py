#!/usr/bin/env python3

import tensorflow.keras as keras

# def autoencoder(input_dims, hidden_layers, latent_dims):
#     if hidden_layers is None:
#         hidden_layers = []
    
#     Ei = keras.layers.Input(shape=(input_dims,))
#     X = Ei
#     for l in hidden_layers:
#         X = keras.layers.Dense(l, activation='relu')(X)
#     mu = keras.layers.Dense(latent_dims)(X)
#     log_sig = keras.layers.Dense(latent_dims)(X)
#     def sample_z(args):
#         _mu, _log_sig = args
#         m = keras.backend.shape(_mu)[0]
#         n = keras.backend.int_shape(_mu)[1]
#         epsilon = keras.backend.random_normal(shape=(m, n), mean=0., stddev=1.)
#         return _mu + keras.backend.exp(log_sig / 2) * epsilon
#     Eo = keras.layers.Lambda(sample_z)([mu, log_sig])
#     encoder = keras.Model(inputs=Ei, outputs=[Eo, mu, log_sig])
#     Di = keras.layers.Input(shape=(latent_dims,))
#     X = Di
#     for l in reversed(hidden_layers):
#         X = keras.layers.Dense(l, activation='relu')(X)
#     Do = keras.layers.Dense(input_dims, activation='sigmoid')(X)
#     decoder = keras.Model(inputs=Di, outputs=Do)
#     auto = keras.Model(inputs=Ei, outputs=decoder(encoder(Ei)[0]))
#     def total_loss(x, x_decoded):
#         content_loss = keras.backend.sum(keras.backend.binary_crossentropy(x, x_decoded), axis=1)
#         kl_loss = 0.5 * keras.backend.sum(keras.backend.exp(log_sig) + keras.backend.square(mu) - 1 - log_sig, axis=1)
#         return content_loss + kl_loss
    
#     auto.compile(optimizer='adam', loss=total_loss)

#     return encoder, decoder, auto

def autoencoder(input_dims, hidden_layers, latent_dims):
    """ Function that creates a variational autoencoder

    Arguments:
        input_dims {int} -- Is the input size
        hidden_layers {list} -- Contain the number of nodes for each hidden
            layer in the encoder, respectively
        latent_dims {int} -- Is the latent space dimensionality

    Returns:
        tuple -- Containing the encoder, decoder and autoencoder models
    """
    ##############################
    # ENCODER MODEL
    ##############################
    inputs = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        units=hidden_layers[0],
        activation='relu'
    )(inputs)

    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(
            units=hidden_layers[i],
            activation='relu'
        )(encoded)

    z_mean = keras.layers.Dense(
        units=latent_dims
    )(encoded)

    z_log_sigma = keras.layers.Dense(
        units=latent_dims
    )(encoded)

    def sampling(args):
        """ Sampling function for the decoder """
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0.0,
            stddev=1.0
        )
        random_sample = z_mean + keras.backend.exp(z_log_sigma) * epsilon

        return random_sample

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    encoder_model = keras.Model(inputs, [z_mean, z_log_sigma, z])

    ##############################
    # DECODER MODEL
    ##############################
    latent_inputs = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(
        units=hidden_layers[-1],
        activation='relu'
    )(latent_inputs)

    for i in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(
            units=hidden_layers[i],
            activation='relu'
        )(decoded)

    decoded = keras.layers.Dense(
        units=input_dims,
        activation='sigmoid'
    )(decoded)

    decoder_model = keras.Model(latent_inputs, decoded)

    ##############################
    # AUTOENCODER MODEL
    ##############################
    outputs = decoder_model(encoder_model(inputs)[0])
    autoencoder_model = keras.Model(inputs, outputs)

    def vae_loss(inputs, outputs):
        """ Loss function """
        reconstruction_loss = keras.losses.binary_crossentropy(
            inputs,
            outputs
        )
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) - \
            keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.mean(kl_loss) * -0.5
        vae_loss = reconstruction_loss + kl_loss

        return vae_loss

    autoencoder_model.compile(
        optimizer='adam',
        loss=vae_loss
    )

    return encoder_model, decoder_model, autoencoder_model