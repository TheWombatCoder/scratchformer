from .attention import MultiHeadAttention
from .layers import FeedForward, LayerNorm
import numpy as np


class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        print("\nSTART BLOCK")
        print("x max:", np.max(np.abs(x)))
        print("has nan:", np.isnan(x).any())
        print("has inf:", np.isinf(x).any())

        attention_output = self.attention(
            x,
            mask
        )

        print("\nAFTER ATTENTION")
        print("attention max:", np.max(np.abs(attention_output)))
        print("has nan:", np.isnan(attention_output).any())
        print("has inf:", np.isinf(attention_output).any())

        x = x + attention_output

        print("\nAFTER RESIDUAL 1")
        print("x max:", np.max(np.abs(x)))

        x = self.norm1(x)

        print("\nAFTER NORM 1")
        print("x max:", np.max(np.abs(x)))
        print("has nan:", np.isnan(x).any())
        print("has inf:", np.isinf(x).any())

        feed_forward_output = self.feed_forward(x)

        print("\nAFTER FEED FORWARD")
        print("ff max:", np.max(np.abs(feed_forward_output)))
        print("has nan:", np.isnan(feed_forward_output).any())
        print("has inf:", np.isinf(feed_forward_output).any())

        x = x + feed_forward_output

        x = self.norm2(x)

        print("\nEND BLOCK")
        print("x max:", np.max(np.abs(x)))
        print("has nan:", np.isnan(x).any())
        print("has inf:", np.isinf(x).any())

        return x

    def __call__(self, x, mask=None):
        return self.forward(x, mask)
