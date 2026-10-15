"""
Transformer layers for Cursor model.

Implements transformer blocks with self-attention (Stage 3: no cross-attention).
Includes optional KV-cache support for incremental decoding.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadSelfAttention
from .embeddings import RotaryPositionalEmbedding


class MLP(nn.Module):
    """
    Feed-forward network (MLP) for transformer blocks.

    Structure: Linear -> Activation -> Dropout -> Linear -> Dropout
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize MLP.

        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension
            dropout: Dropout probability
            activation: Activation function ("gelu" or "relu")
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention only.

    Architecture:
    1. x <- x + SelfAttn(LN(x))
    2. x <- x + MLP(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        activation: str = "gelu",
        rope: Optional[RotaryPositionalEmbedding] = None,
    ):
        """
        Initialize transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: FFN intermediate dimension
            head_dim: Dimension per head
            dropout: Dropout probability
            attention_dropout: Attention-specific dropout
            layer_norm_eps: Layer norm epsilon
            activation: Activation function
            rope: Shared RoPE instance
        """
        super().__init__()

        # Layer norms (pre-norm architecture)
        self.ln_self_attn = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln_mlp = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Self-attention
        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=attention_dropout,
            rope=rope,
        )

        # MLP
        self.mlp = MLP(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            attention_mask: Self-attention mask (batch, seq_len)
            positions: Position indices for RoPE

        Returns:
            Output tensor (batch, seq_len, d_model), or (output, present_kv) if use_cache=True
        """
        # Self-attention with residual
        residual = x
        x = self.ln_self_attn(x)
        if use_cache:
            x_attn, present = self.self_attn(
                x,
                attention_mask=attention_mask,
                positions=positions,
                past_kv=past_kv,
                use_cache=True,
            )
            x = x_attn
        else:
            x = self.self_attn(x, attention_mask=attention_mask, positions=positions)
        x = self.dropout(x)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.ln_mlp(x)
        x = self.mlp(x)
        x = residual + x

        if use_cache:
            return x, present
        return x


class TransformerBlockStack(nn.Module):
    """
    Stack of transformer blocks (all self-attention only).
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        activation: str = "gelu",
        rope: Optional[RotaryPositionalEmbedding] = None,
    ):
        """
        Initialize block stack.

        Args:
            n_layers: Total number of layers
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: FFN dimension
            head_dim: Head dimension
            dropout: Dropout probability
            attention_dropout: Attention dropout
            layer_norm_eps: Layer norm epsilon
            activation: Activation function
            rope: Shared RoPE instance
        """
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                head_dim=head_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                layer_norm_eps=layer_norm_eps,
                activation=activation,
                rope=rope,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            attention_mask: Self-attention mask
            positions: Position indices

        Returns:
            Output tensor (batch, seq_len, d_model), or (output, present_key_values) if use_cache=True
        """
        if not use_cache:
            for layer in self.layers:
                x = layer(
                    x,
                    attention_mask=attention_mask,
                    positions=positions,
                )
            return x

        present: List[Tuple[torch.Tensor, torch.Tensor]] = []
        if past_key_values is None:
            past_key_values = [None for _ in range(len(self.layers))]  # type: ignore[list-item]

        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if i < len(past_key_values) else None
            x, kv = layer(
                x,
                attention_mask=attention_mask,
                positions=positions,
                past_kv=pkv,
                use_cache=True,
            )
            present.append(kv)
        return x, present
