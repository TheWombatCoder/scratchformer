import numpy as np
from .math import softmax


def cross_entropy(logits, targets):
    probabilities = softmax(logits, axis=-1)
    batch_size, seq_len = (targets.shape)
    batch_indices = np.arange(batch_size)[:, None]
    seq_indices = np.arange(seq_len)[None, :]

    correct_probabilities = probabilities[batch_indices, seq_indices, targets]
    losses = -np.log(correct_probabilities + 1e-12)

    return np.mean(losses)
