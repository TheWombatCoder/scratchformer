import numpy as np
from .math import *


class Linear:
    def __init__(self, input_dim, output_dim):
        scale = np.sqrt(2.0 / (input_dim + output_dim))

        self.W = (np.random.randn(input_dim, output_dim) * scale)
        self.b = np.zeros(output_dim)

    def forward(self, x):
        return x @ self.W + self.b

    def __call__(self, x):
        return self.forward(x)


class Embeddings:
    def __init__(self, vocab_size, embedding_dim):
        self.weight = (np.random.randn(vocab_size, embedding_dim) * 0.02)

    def forward(self, token_ids):
        return self.weight[token_ids]

    def __call__(self, token_ids):
        return self.forward(token_ids)


class PositionalEmbeddings:
    def __init__(self, max_length, embedding_dim):
        self.weight = (np.random.randn(max_length, embedding_dim) * 0.02)

    def forward(self, sequence_length):
        return self.weight[:sequence_length]

    def __call__(self, sequence_length):
        return self.forward(sequence_length)


class LayerNorm:
    def __init__(self, dimensions, epsilon=1e-5):
        self.gamma = np.ones(dimensions)
        self.beta = np.zeros(dimensions)
        self.epsilon = epsilon

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.mean((x - mean)**2, axis=-1, keepdims=True)
        normalize = (x - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * normalize + self.beta

    def __call__(self, x):
        return self.forward(x)


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = gelu(x)
        x = self.linear2(x)
        return x

    def __call__(self, x):
        return self.forward(x)
