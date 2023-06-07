
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """initializes data for the model"""
    X_p = K.applications.xception.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y)
    return X_p, Y_p

if __name__ == '__main__':
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()

    X_train = np.concatenate([X_train, np.flip(X_train, axis=2)], axis=0)
    Y_train = np.concatenate([Y_train, Y_train], axis=0)

    X_train_p, Y_train_oh = preprocess_data(X_train, Y_train)
    X_valid_p, Y_valid_oh = preprocess_data(X_valid, Y_valid)

    inputs1 = K.layers.Input(shape=(32, 32, 3))
    Y = K.layers.Lambda(lambda image: K.backend.resize_images(image, 299/32, 299/32, "channels_last", interpolation="bilinear"))(inputs1)
    xception = K.applications.xception.Xception(include_top=False, pooling='avg', input_tensor=Y)
    X_t = xception.predict(X_train_p)
    X_v = xception.predict(X_valid_p)

    inputs2 = K.layers.Input(shape=(2048,))
    init = K.initializers.he_uniform()
    X = K.layers.Dense(512, activation=None, kernel_initializer=init)(inputs2)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.LeakyReLU()(X)
    X = K.layers.Dropout(0.45)(X)
    outputs = K.layers.Dense(10, activation='softmax', kernel_initializer=init)(X)

    classifier = K.models.Model(inputs=inputs2, outputs=outputs)
    ckpt = K.callbacks.ModelCheckpoint('cifar10.h5', monitor='val_loss', save_best_only=True, verbose=1)
    classifier.compile(optimizer=K.optimizers.SGD(0.005, 0.9, decay=0.00001, nesterov=True), loss="categorical_crossentropy", metrics=["accuracy"])
    classifier.fit(X_t, Y_train_oh, epochs=30, batch_size=512, validation_data=(X_v, Y_valid_oh), callbacks=[ckpt], shuffle=True)
    del classifier
    model = K.models.load_model('cifar10.h5')
    X = xception.output
    for layer in model.layers[1:]:
        X = layer(X)
    full_model = K.models.Model(inputs=inputs1, outputs=X)
    full_model.compile(optimizer=K.optimizers.SGD(0.005, 0.9, decay=0.00001, nesterov=True), loss="categorical_crossentropy", metrics=["accuracy"])
    full_model.save('cifar10.h5')