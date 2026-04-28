"""
Attention mechanisms for Cursor transformer.

Includes Multi-Head Self-Attention and Multi-Head Cross-Attention.
Optimized with Flash Attention 2 when available.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RotaryPositionalEmbedding

# NOTE:
# This repo previously attempted to use the external `flash_attn` package, but the
# implementation never actually called it in forward(). For performance (especially on H100/H200),
# prefer PyTorch's built-in SDPA (`torch.nn.functional.scaled_dot_product_attention`), which
# will automatically dispatch to FlashAttention / memory-efficient kernels when available.


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE.

    Implements causal self-attention for autoregressive generation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        rope: Optional[RotaryPositionalEmbedding] = None,
        use_flash_attn: bool = True,
    ):
        """
        Initialize self-attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            head_dim: Dimension per head (default: d_model // n_heads)
            dropout: Attention dropout probability
            rope: Optional shared RoPE instance
            use_flash_attn: Use Flash Attention if available
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim else d_model // n_heads
        self.dropout = dropout
        # "Flash attention" here means the PyTorch SDPA fastpath (which may use FlashAttention kernels).
        self.use_flash_attn = bool(use_flash_attn)

        assert self.head_dim * n_heads == d_model, \
            f"head_dim ({self.head_dim}) * n_heads ({n_heads}) must equal d_model ({d_model})"

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

        # RoPE (use shared instance if provided)
        self.rope = rope

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # No print here: training scripts often call this many times in multi-process setups.

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for self-attention.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            attention_mask: Optional attention mask (batch, seq_len) where 1 = attend, 0 = mask
            positions: Optional position indices for RoPE

        KV caching:
          - If `past_kv` is provided, it should be (past_k, past_v) with shape
            (batch, n_heads, past_len, head_dim).
          - When `use_cache=True`, returns (output, (present_k, present_v)).

        Returns:
            Output tensor (batch, seq_len, d_model), or (output, present_kv) if use_cache=True
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        if self.rope is not None:
            q, k = self.rope(q, k, positions)

        # KV cache concat (past already has RoPE applied for its positions).
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # Fast path: PyTorch SDPA (will use FlashAttention kernels on supported GPUs/dtypes).
        # We keep RoPE externally (already applied above).
        #
        # - When there is NO padding in the sequence, we can use `is_causal=True` and no mask,
        #   which is the highest-performance path (FlashAttention on H100/H200).
        # - If there IS padding, PyTorch forbids setting both `attn_mask` and `is_causal=True`
        #   at the same time, so we fall back to an explicit combined mask (correct, but may
        #   not hit the FlashAttention kernel depending on PyTorch version).
        attn_output: torch.Tensor
        # On CUDA, SDPA will use FlashAttention/memory-efficient kernels when available.
        # On CPU, some PyTorch builds/versions have exhibited NaNs in SDPA for certain shapes,
        # so we fall back to the explicit attention path for robustness.
        if self.use_flash_attn and x.is_cuda and hasattr(F, "scaled_dot_product_attention"):
            # Cache path: if we have past_kv and seq_len==1, there is no padding and no future tokens,
            # so we can safely use the fastest SDPA kernel without an explicit causal mask.
            if past_kv is not None and seq_len == 1:
                attn_output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
            else:
                has_padding = False
                if attention_mask is not None:
                    # Any zero => padding exists
                    has_padding = bool((attention_mask == 0).any().item())

                if not has_padding:
                    # Best path: causal only, no explicit mask
                    attn_output = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=True,
                    )
                else:
                    # Correctness path: explicit combined mask (padding + causal).
                    key_padding = (attention_mask == 0)  # (B,T) bool
                    pad_mask = key_padding[:, None, None, :]  # (B,1,1,T)

                    causal = torch.triu(
                        torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                        diagonal=1,
                    )  # (T,T) True above diagonal
                    causal = causal[None, None, :, :]  # (1,1,T,T)

                    # True means allow, so invert the combined (pad|future) mask.
                    sdpa_mask = ~(pad_mask | causal)  # (B,1,T,T)

                    attn_output = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=sdpa_mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=False,
                    )
        else:
            # Slow fallback: explicit attention (works on all PyTorch versions).
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # If we are in cache mode with seq_len==1, there is no need for a causal mask.
            if not (past_kv is not None and seq_len == 1):
                causal_mask = self._create_causal_mask(seq_len, x.device)
                attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

            if attention_mask is not None:
                attention_mask_ = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_scores = attn_scores.masked_fill(attention_mask_ == 0, float("-inf"))

            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)

        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.o_proj(attn_output)

        if use_cache:
            return output, (k, v)
        return output

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask.

        Standard autoregressive causal mask:
        Position t can attend to positions [0, 1, ..., t] (including itself).
        When training with shifted labels (predict-next-token), this is the
        typical transformer setup and does not leak the target token.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0)
        return mask


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention with RoPE.

    Used for attending from edit history to canvas summaries.
    Implements causal cross-attention where query position t can only
    attend to source positions 1..t.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        rope: Optional[RotaryPositionalEmbedding] = None,
    ):
        """
        Initialize cross-attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            head_dim: Dimension per head
            dropout: Attention dropout probability
            rope: Optional shared RoPE instance
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim else d_model // n_heads
        self.dropout = dropout

        # Q projection (from query/edit history)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # K, V projections (from source/canvas)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

        # RoPE
        self.rope = rope

        # Scaling
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        query_positions: Optional[torch.Tensor] = None,
        source_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention.

        Args:
            query: Query tensor (batch, query_len, d_model) - edit history
            source: Source tensor (batch, source_len, d_model) - canvas summaries
            source_mask: Optional mask for source (batch, source_len)
            query_positions: Position indices for query
            source_positions: Position indices for source

        Returns:
            Output tensor (batch, query_len, d_model)
        """
        batch_size, query_len, _ = query.shape
        source_len = source.size(1)

        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(source)
        v = self.v_proj(source)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.view(batch_size, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, source_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, source_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.rope is not None:
            # For cross-attention, we apply RoPE separately to q and k
            # using their respective position indices
            q, _ = self.rope(q, q, query_positions)  # Apply to q only
            k, _ = self.rope(k, k, source_positions)  # Apply to k only

        # Compute attention scores
        # (batch, n_heads, query_len, source_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask for cross-attention
        # Query position t can only attend to source positions 0..t
        causal_mask = self._create_cross_causal_mask(query_len, source_len, query.device)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        # Apply source mask if provided
        if source_mask is not None:
            # source_mask: (batch, source_len) -> (batch, 1, 1, source_len)
            source_mask = source_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(source_mask == 0, float("-inf"))

        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Handle NaN from all-masked rows
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, query_len, self.d_model)

        # Output projection
        output = self.o_proj(attn_output)

        return output

    def _create_cross_causal_mask(
        self,
        query_len: int,
        source_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create causal mask for cross-attention.

        Query position t can attend to source positions 0, 1, ..., t.
        This ensures each edit history token only sees the canvas state
        at that timestep and earlier.
        """
        # Create a mask where mask[i, j] = 1 if j <= i
        rows = torch.arange(query_len, device=device).unsqueeze(1)
        cols = torch.arange(source_len, device=device).unsqueeze(0)
        mask = (cols <= rows).float()
        return mask
