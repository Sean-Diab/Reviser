"""
Embeddings for Cursor transformer.

Includes token embeddings and Rotary Positional Embeddings (RoPE).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Can be optionally tied with the output projection for weight sharing.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=initializer_range)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input token IDs.

        Args:
            input_ids: (batch_size, seq_len) tensor of token IDs

        Returns:
            (batch_size, seq_len, d_model) tensor of embeddings
        """
        return self.embedding(input_ids)

    @property
    def weight(self) -> torch.Tensor:
        """Return embedding weight for tying with output projection."""
        return self.embedding.weight


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    RoPE encodes position information by rotating the query and key vectors.
    This allows the model to learn relative positions naturally.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_seq_length: int = 2048,
        theta: float = 10000.0,
    ):
        """
        Initialize RoPE.

        Args:
            dim: Dimension of each attention head
            max_seq_length: Maximum sequence length for precomputing frequencies
            theta: Base for frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.theta = theta

        # Precompute frequencies
        self._precompute_freqs(max_seq_length)

    def _precompute_freqs(self, max_seq_length: int):
        """Precompute sin and cos frequencies for efficiency."""
        # IMPORTANT:
        # This method can be called from forward() via _extend_freqs() after the module has
        # already been moved to CUDA. In that case, we must create the new cached buffers
        # on the *current* device, otherwise RoPE will mix CPU cached tensors with CUDA q/k.
        if hasattr(self, "inv_freq") and isinstance(self.inv_freq, torch.Tensor):
            device = self.inv_freq.device
        else:
            device = torch.device("cpu")

        # Compute inverse frequencies
        # freq_i = 1 / (theta^(2i/d)) for i = 0, 1, ..., d/2 - 1
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Precompute position embeddings
        positions = torch.arange(max_seq_length, device=device, dtype=torch.float32)
        # (max_seq_length, dim/2)
        freqs = torch.outer(positions, inv_freq)
        # (max_seq_length, dim)
        freqs = torch.cat([freqs, freqs], dim=-1)

        # Register sin and cos
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def _extend_freqs(self, seq_len: int):
        """Extend cached frequencies if sequence is longer than max."""
        if seq_len <= self.max_seq_length:
            return

        self.max_seq_length = seq_len
        self._precompute_freqs(seq_len)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, n_heads, seq_len, head_dim)
            positions: Optional position indices (batch, seq_len). If None,
                       uses sequential positions 0, 1, 2, ...

        Returns:
            Tuple of rotated (q, k) tensors
        """
        batch_size, n_heads, seq_len, head_dim = q.shape

        # Extend cache if needed
        if seq_len > self.max_seq_length:
            self._extend_freqs(seq_len)

        if positions is None:
            # Use sequential positions
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            # Use provided positions
            cos = self.cos_cached[positions].unsqueeze(1)  # (batch, 1, seq_len, dim)
            sin = self.sin_cached[positions].unsqueeze(1)

        # Ensure cached tensors match q/k device + dtype (especially after _extend_freqs()).
        cos = cos.to(device=q.device, dtype=q.dtype)
        sin = sin.to(device=q.device, dtype=q.dtype)

        # Apply rotation
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)

        return q_rotated, k_rotated

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary embedding to a tensor.

        Uses the rotation formula:
        x_rotated = x * cos + rotate_half(x) * sin
        """
        # Split x into pairs and rotate
        x_rotated = (x * cos) + (self._rotate_half(x) * sin)
        return x_rotated

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dims of x.

        Transforms [x1, x2, x3, x4, ...] to [-x_{d/2+1}, -x_{d/2+2}, ..., x1, x2, ...]
        """
        if not torch.is_tensor(x):
            raise TypeError(f"Expected tensor in _rotate_half, got {type(x)}")
        last_dim = int(x.size(-1))
        half = last_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: RotaryPositionalEmbedding,
    positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to apply RoPE to query and key tensors.

    Args:
        q: Query tensor (batch, n_heads, seq_len, head_dim)
        k: Key tensor (batch, n_heads, seq_len, head_dim)
        rope: RotaryPositionalEmbedding instance
        positions: Optional position indices

    Returns:
        Tuple of rotated (q, k)
    """
    return rope(q, k, positions)


class CanvasEmbedding(nn.Module):
    """
    Canvas embedding using attention pooling (Approach 1).
    
    Encodes the canvas at each timestep into a single d_model vector
    using attention pooling with a learnable query vector.
    No FFN projection - the pooled vector is used directly.
    """

    def __init__(
        self,
        d_model: int,
        initializer_range: float = 0.02,
    ):
        """
        Initialize the canvas embedding.

        Args:
            d_model: Model dimension
            initializer_range: Standard deviation for weight initialization
        """
        super().__init__()
        self.d_model = d_model

        # Learnable query vector for attention pooling
        self.query = nn.Parameter(torch.randn(d_model))

        # Initialize query vector
        nn.init.normal_(self.query, mean=0.0, std=initializer_range)

    def forward(
        self,
        canvas_embeddings: torch.Tensor,
        canvas_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode canvas sequences into summary vectors using attention pooling.

        Args:
            canvas_embeddings: Token embeddings for each canvas
                Shape: (batch, seq_len, max_canvas_len, d_model)
            canvas_mask: Mask for valid tokens in each canvas
                Shape: (batch, seq_len, max_canvas_len) where 1 = valid, 0 = padding

        Returns:
            Canvas summary vectors for each timestep
                Shape: (batch, seq_len, d_model)
        """
        # Legacy/reference implementation (pre-SDPA):
        # scores_{t,i} = <q, x_{t,i}>
        # alpha = softmax(scores)
        # summary_t = sum_i alpha_{t,i} * x_{t,i}
        scores = torch.einsum("d,bscd->bsc", self.query, canvas_embeddings)
        scores = scores.masked_fill(canvas_mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)  # (B, S, C)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        return torch.einsum("bsc,bscd->bsd", attn_weights, canvas_embeddings)
