import numpy as np
from .math import *
from .layers import *


def causal_mask(length):
    return np.tril(np.ones((length, length), dtype=bool))


class SelfAttention:
    def __int__(self, d_model):
        self.d_model = d_model
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)

    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = (Q @ np.swapaxes(K, -1, -2))
        scores = (scores / np.sqrt(self.d_model))

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        weights = softmax(scores, axis=-1)
        output = (weights @ V)

        return output

    def __call__(self, x, mask=None):
        return self.forward(x, mask)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert(d_model % num_heads == 0), ("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = (d_model // num_heads)

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, _ = (x.shape)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        x = np.transpose(x, (0, 2, 1, 3))
        return x

    def combine_heads(self, x):
        (batch_size, num_heads, seq_len, head_dim) = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        x = x.reshape(batch_size, seq_len, num_heads * head_dim)
        return x

    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        scores = (Q @ np.swapaxes(K, -1, -2))
        scores = (scores / np.sqrt(self.head_dim))

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        attention_weights = softmax(scores, axis=-1)
        attention_output = (attention_weights @ V)
        combined = self.combine_heads(attention_output)

        output = self.W_o(combined)
        return output

    def __call__(self, x, mask=None):
        return self.forward(x, mask)
