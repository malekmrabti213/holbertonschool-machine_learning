# #!/usr/bin/env python3
# """
# Task 4
# """
# from tensorflow import keras as K
# identity_block = __import__('2-identity_block').identity_block
# projection_block = __import__('3-projection_block').projection_block


# def resnet50():
#     """
#     """
#     inputs = K.Input(shape=(224, 224, 3))
#     he_normal = K.initializers.he_normal(seed=0)
#     X = K.layers.Conv2D(64,
#                         (7, 7),
#                         strides=(2, 2),
#                         padding='same',
#                         kernel_initializer=he_normal)(inputs)
#     X = K.layers.BatchNormalization(axis=3)(X)
#     X = K.layers.Activation('relu')(X)
#     # X=K.layers.ReLU()(X)

#     X = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

#     X = projection_block(X, [64, 64, 256], s=1)
#     for _ in range(2):
#         X = identity_block(X, [64, 64, 256])

#     X = projection_block(X, [128, 128, 512])
#     for _ in range(3):
#         X = identity_block(X, [128, 128, 512])

#     X = projection_block(X, [256, 256, 1024])
#     for _ in range(5):
#         X = identity_block(X, [256, 256, 1024])

#     X = projection_block(X, [512, 512, 2048])
#     for _ in range(2):
#         X = identity_block(X, [512, 512, 2048])

#     X = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(X)
#     Y = K.layers.Dense(1000, activation='softmax',
#                        kernel_initializer=he_normal)(X)

#     return K.models.Model(inputs=inputs, outputs=Y)

#!/usr/bin/env python3

"""Useless comment"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Create the ResNet50 Model
    :return: The ResNet50 model
    """
    init = K.initializers.he_normal()
    input = K.Input(shape=(224, 224, 3))

    conv_1 = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             strides=(2, 2),
                             padding="same",
                             kernel_initializer=init)(input)
    norm_1 = K.layers.BatchNormalization(axis=3)(conv_1)
    act_1 = K.layers.ReLU()(norm_1)

    max_pool_1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       strides=(2, 2),
                                       padding="same")(act_1)

    pr_block_1_2x = projection_block(max_pool_1, [64, 64, 256], s=1)
    id_block_1_2x = identity_block(pr_block_1_2x, [64, 64, 256])
    id_block_2_2x = identity_block(id_block_1_2x, [64, 64, 256])

    pr_block_1_3x = projection_block(id_block_2_2x, [128, 128, 512])
    id_block_1_3x = identity_block(pr_block_1_3x, [128, 128, 512])
    id_block_2_3x = identity_block(id_block_1_3x, [128, 128, 512])
    id_block_3_3x = identity_block(id_block_2_3x, [128, 128, 512])

    pr_block_1_4x = projection_block(id_block_3_3x, [256, 256, 1024])
    id_block_1_4x = identity_block(pr_block_1_4x, [256, 256, 1024])
    id_block_2_4x = identity_block(id_block_1_4x, [256, 256, 1024])
    id_block_3_4x = identity_block(id_block_2_4x, [256, 256, 1024])
    id_block_4_4x = identity_block(id_block_3_4x, [256, 256, 1024])
    id_block_5_4x = identity_block(id_block_4_4x, [256, 256, 1024])

    pr_block_1_5x = projection_block(id_block_5_4x, [512, 512, 2048])
    id_block_1_5x = identity_block(pr_block_1_5x, [512, 512, 2048])
    id_block_2_5x = identity_block(id_block_1_5x, [512, 512, 2048])

    max_pool = K.layers.AvgPool2D(pool_size=(7, 7),
                                  strides=(1, 1),
                                  padding="valid")(id_block_2_5x)

    dense_output = K.layers.Dense(units=1000,
                                  activation="softmax",
                                  kernel_initializer=init)(max_pool)

    return K.models.Model(inputs=input, outputs=dense_output)