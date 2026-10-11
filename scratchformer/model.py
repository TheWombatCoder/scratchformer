import numpy as np
from .layers import Embeddings, PositionalEmbeddings, LayerNorm, Linear
from .block import TransformerBlock
from .attention import causal_mask


class Transformer:
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4, d_ff=512, max_len=256):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_len = max_len

        self.token_embeddings = Embeddings(vocab_size=vocab_size, embedding_dim=d_model)
        self.positional_embeddings = (PositionalEmbeddings(max_length=max_len, embedding_dim=d_model))
        self.blocks = [
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff) for _ in range(num_layers)
        ]
        self.final_norm = LayerNorm(d_model)
        self.output_projection = Linear(input_dim=d_model, output_dim=vocab_size)

    def forward(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape(batch_size, seq_len)")

        batch_size, seq_len = (token_ids.shape)

        if seq_len > self.max_len:
            raise ValueError(f"seq_len: {seq_len} exceeds max_length: {self.max_len}")

        token_embeddings = (self.token_embeddings(token_ids))
        positional_embeddings = (self.positional_embeddings(seq_len))

        x = (token_embeddings + positional_embeddings)
        mask = causal_mask(seq_len)
        mask = mask[None, None, :, :]

        for block in self.blocks:
            x = block(x, mask)

        x = self.final_norm(x)
        logits = (self.output_projection(x))
        return logits

    def __call__(self, token_ids):
        return self.forward(token_ids)
