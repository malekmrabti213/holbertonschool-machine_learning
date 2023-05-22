#!/usr/bin/env python3

import tensorflow.keras as K

def lenet5(X):
    model = K.Sequential()
    model.add(K.layers.Conv2D(6, 5, activation='relu', kernel_initializer='he_normal', padding='same', input_shape=X.shape[1:]))
    model.add(K.layers.MaxPooling2D())
    model.add(K.layers.Conv2D(16, 5, activation='relu', kernel_initializer='he_normal'))
    model.add(K.layers.MaxPooling2D())
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(120, activation='relu', kernel_initializer='he_normal'))
    model.add(K.layers.Dense(84, activation='relu', kernel_initializer='he_normal'))
    model.add(K.layers.Dense(10, activation='softmax', kernel_initializer='he_normal'))

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model