#!/usr/bin/env python3
"""
    Function create_mini_batches
"""

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Function to create mini-batches from input data X and labels Y.

    :param X: ndarray, input data of shape (m, nx)
    :param Y: ndarray, labels of shape (m, ny)
    :param batch_size: int, size of each mini-batch

    :return: list of mini-batches containing tuples (X_batch, Y_batch)
    """
    # Shuffle the data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    # Partition the shuffled data into mini-batches
    num_batches = X.shape[0] // batch_size
    mini_batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_shuffled[start_idx:end_idx]
        Y_batch = Y_shuffled[start_idx:end_idx]
        mini_batches.append((X_batch, Y_batch))

    # Handling the last mini-batch which may have fewer samples
    if X.shape[0] % batch_size != 0:
        start_idx = num_batches * batch_size
        X_batch = X_shuffled[start_idx:]
        Y_batch = Y_shuffled[start_idx:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
