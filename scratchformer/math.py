import numpy as np


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
