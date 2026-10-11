import numpy as np
from scratchformer.model import Transformer
import warnings

warnings.filterwarnings(
    "error",
    category=RuntimeWarning
)

model = Transformer(vocab_size=100, d_model=64, num_heads=4, num_layers=2, d_ff=256, max_len=32)
tokens = np.array([
    [4, 17, 2, 31, 65],
    [91, 75, 13, 10, 54],
])
logits = model(tokens)

print(f"Input Shape: {tokens.shape}, Output Shape: {logits.shape}")
