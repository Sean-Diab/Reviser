"""
Configuration for Cursor transformer model.
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class CursorConfig:
    """
    Configuration class for CursorTransformer.

    Model specs (embedding-based architecture):
    - 24 attention layers (all self-attention only, no cross-attention)
    - d_model = 512, 8 heads, d_ff = 2048
    - Canvas embedding via attention pooling (added to token embeddings)
    - Tied embedding and unembedding matrices
    - RoPE positional embeddings (applied in attention)
    - Target ~100M parameters
    """

    # Vocabulary
    vocab_size: int = 50291  # GPT-2 (50257) + 34 special tokens

    # Model dimensions
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    head_dim: Optional[int] = None  # Computed from d_model / n_heads if None

    # Layers
    n_layers: int = 24  # All self-attention only
    n_cross_layers: int = 0  # No cross-attention layers

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Positional encoding
    # - "rope": rotary embeddings applied inside attention (current default)
    # - "absolute": GPT-2 style learned absolute positional embeddings added to token embeddings
    #               (and also added to per-canvas token embeddings before pooling).
    positional_encoding: str = "rope"
    # Backward-compatibility knob kept because older configs pass it through.
    # The current standalone model does not vary behavior based on this field.
    canvas_pool_positional_encoding: str = "absolute"
    rope_theta: float = 10000.0
    # Maximum model sequence length (used for RoPE cache and/or absolute position table size).
    max_seq_length: int = 2048
    # Maximum number of tokens in the canvas dimension (used for canvas position table size
    # when positional_encoding="absolute").
    max_canvas_length: int = 512

    # Training constraints
    max_edit_steps: int = 1000

    # Tied embeddings
    tie_embeddings: bool = True

    # Layer norm epsilon
    layer_norm_eps: float = 1e-5

    # Activation function
    activation: str = "gelu"

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        """Compute derived values."""
        if self.head_dim is None:
            assert self.d_model % self.n_heads == 0, \
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            self.head_dim = self.d_model // self.n_heads


    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "head_dim": self.head_dim,
            "n_layers": self.n_layers,
            "n_cross_layers": self.n_cross_layers,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "positional_encoding": self.positional_encoding,
            "canvas_pool_positional_encoding": self.canvas_pool_positional_encoding,
            "rope_theta": self.rope_theta,
            "max_seq_length": self.max_seq_length,
            "max_canvas_length": self.max_canvas_length,
            "max_edit_steps": self.max_edit_steps,
            "tie_embeddings": self.tie_embeddings,
            "layer_norm_eps": self.layer_norm_eps,
            "activation": self.activation,
            "initializer_range": self.initializer_range,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CursorConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CursorConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def estimate_parameters(self) -> int:
        """
        Estimate the number of parameters in the model.

        Returns approximate parameter count.
        """
        # Embedding: vocab_size * d_model (shared with output if tied)
        embedding_params = self.vocab_size * self.d_model

        # Per transformer layer (self-attention only):
        # - Q, K, V projections: 3 * d_model * d_model
        # - Output projection: d_model * d_model
        # - Layer norms: 2 * d_model (before attn and before MLP)
        # - MLP: d_model * d_ff + d_ff * d_model = 2 * d_model * d_ff
        # - MLP biases: d_ff + d_model
        layer_params = (
            4 * self.d_model * self.d_model +  # Q, K, V, O projections
            2 * self.d_model +  # Layer norms
            2 * self.d_model * self.d_ff +  # MLP
            self.d_ff + self.d_model  # MLP biases
        )

        # Canvas embedding (attention pooling only, no projection):
        # - Query vector: d_model
        canvas_embedding_params = self.d_model

        # Absolute positional embeddings (optional).
        abs_pos_params = 0
        if str(self.positional_encoding).lower() == "absolute":
            abs_pos_params = (self.max_seq_length * self.d_model) + (self.max_canvas_length * self.d_model)

        # Final layer norm
        final_ln_params = self.d_model * 2  # weight + bias

        # Output projection (tied with embedding, so don't count again)
        output_params = 0 if self.tie_embeddings else self.vocab_size * self.d_model

        total = (
            embedding_params +
            self.n_layers * layer_params +
            canvas_embedding_params +
            abs_pos_params +
            final_ln_params +
            output_params
        )

        return total


# Default configuration matching the spec
DEFAULT_CONFIG = CursorConfig()
