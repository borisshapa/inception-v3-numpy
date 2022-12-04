import numpy as np


def get_batches(dataset, batch_size):
    x, y = dataset
    n_samples = x.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        yield x[batch_indices], y[batch_indices]
