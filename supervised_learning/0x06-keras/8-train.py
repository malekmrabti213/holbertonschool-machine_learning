#!/usr/bin/env python3

import tensorflow.keras as K

def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    callbacks = None
    if validation_data is not None and (early_stopping or learning_rate_decay or save_best):
        callbacks = []
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(patience=patience))
        if learning_rate_decay:
            def step_decay(epoch):
                return alpha / (1 + decay_rate * epoch)
            callbacks.append(K.callbacks.LearningRateScheduler(schedule=step_decay, verbose=1))
        if save_best:
            callbacks.append(K.callbacks.ModelCheckpoint(filepath, save_best_only=True))
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data, callbacks=callbacks,
                       verbose=verbose, shuffle=shuffle)