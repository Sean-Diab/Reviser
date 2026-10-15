"""
Utility functions for Cursor transformer.

Includes masking helpers and other utilities.
"""

from typing import Optional, Set
import torch


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Lower triangular mask of shape (seq_len, seq_len)
        where 1 = attend, 0 = mask
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create a padding mask from sequence lengths.

    Args:
        lengths: Tensor of sequence lengths (batch_size,)
        max_len: Maximum sequence length

    Returns:
        Mask of shape (batch_size, max_len) where 1 = valid, 0 = padding
    """
    batch_size = lengths.size(0)
    positions = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
    mask = positions < lengths.unsqueeze(1)
    return mask.long()


def create_cross_causal_mask(
    query_len: int,
    source_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a causal mask for cross-attention.

    Query position t can only attend to source positions 0..t.

    Args:
        query_len: Length of query sequence
        source_len: Length of source sequence
        device: Device to create tensor on

    Returns:
        Mask of shape (query_len, source_len)
    """
    rows = torch.arange(query_len, device=device).unsqueeze(1)
    cols = torch.arange(source_len, device=device).unsqueeze(0)
    return (cols <= rows).float()


def mask_invalid_moves(
    logits: torch.Tensor,
    cursor_positions: torch.Tensor,
    canvas_lengths: torch.Tensor,
    move_token_ids: Set[int],
    move_amounts: dict,
) -> torch.Tensor:
    """
    Mask out invalid move tokens during inference.

    A move is invalid if it would move the cursor out of bounds.
    Cursor can be at positions 0 to canvas_length inclusive.

    Args:
        logits: Model output logits (batch, vocab_size)
        cursor_positions: Current cursor position for each item (batch,)
        canvas_lengths: Current canvas length for each item (batch,)
        move_token_ids: Set of all move token IDs
        move_amounts: Dict mapping token_id -> move_amount

    Returns:
        Logits with invalid moves set to -inf
    """
    batch_size = logits.size(0)

    for i in range(batch_size):
        cursor_pos = cursor_positions[i].item()
        canvas_len = canvas_lengths[i].item()

        for token_id in move_token_ids:
            amount = move_amounts[token_id]
            new_pos = cursor_pos + amount

            # Valid positions: 0 to canvas_len (inclusive)
            if new_pos < 0 or new_pos > canvas_len:
                logits[i, token_id] = float("-inf")

    return logits


def mask_invalid_delete(
    logits: torch.Tensor,
    cursor_positions: torch.Tensor,
    delete_token_id: int,
) -> torch.Tensor:
    """
    Mask out delete token when cursor is at position 0.

    Delete removes the token to the left of the cursor,
    so it's invalid when cursor is at the leftmost position.

    Args:
        logits: Model output logits (batch, vocab_size)
        cursor_positions: Current cursor position for each item (batch,)
        delete_token_id: Token ID for delete action

    Returns:
        Logits with invalid deletes set to -inf
    """
    # Mask delete where cursor_pos == 0
    invalid_delete = cursor_positions == 0
    logits[invalid_delete, delete_token_id] = float("-inf")
    return logits


def apply_inference_masks(
    logits: torch.Tensor,
    cursor_positions: torch.Tensor,
    canvas_lengths: torch.Tensor,
    move_token_ids: Set[int],
    move_amounts: dict,
    delete_token_id: int,
) -> torch.Tensor:
    """
    Apply all inference-time masking for valid actions.

    Args:
        logits: Model output logits
        cursor_positions: Current cursor positions
        canvas_lengths: Current canvas lengths
        move_token_ids: Set of move token IDs
        move_amounts: Dict mapping token_id -> move_amount
        delete_token_id: Delete token ID

    Returns:
        Masked logits
    """
    logits = mask_invalid_moves(
        logits, cursor_positions, canvas_lengths, move_token_ids, move_amounts
    )
    logits = mask_invalid_delete(logits, cursor_positions, delete_token_id)
    return logits


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters

    Returns:
        Parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_size(model: torch.nn.Module, name: str = "Model"):
    """Print model size information."""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"{name}:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Size (MB, fp32): {total_params * 4 / 1024 / 1024:.2f}")
    print(f"  Size (MB, fp16): {total_params * 2 / 1024 / 1024:.2f}")
