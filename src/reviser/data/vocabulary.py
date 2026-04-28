"""
Vocabulary constants and helper functions for Cursor transformer.

This module provides easy access to special token IDs and helper functions
for working with the Cursor vocabulary.
"""

from dataclasses import dataclass
from typing import List, Set


# GPT-2 base vocabulary size
GPT2_VOCAB_SIZE = 50257

# GPT-2 end-of-text token ID ("<|endoftext|>").
# NOTE: This lives inside the base GPT-2 vocab range [0, GPT2_VOCAB_SIZE).
GPT2_EOT_TOKEN_ID = GPT2_VOCAB_SIZE - 1

# Special token offsets from GPT2_VOCAB_SIZE
CURSOR_OFFSET = 0
END_OF_INPUT_OFFSET = 1
DELETE_OFFSET = 2
END_OF_RESPONSE_OFFSET = 3

# Move token offsets (4-23): pairs of +/- for each power of 2
# [MOVE_+1], [MOVE_-1], [MOVE_+2], [MOVE_-2], ..., [MOVE_+512], [MOVE_-512]
MOVE_TOKEN_START_OFFSET = 4
MOVE_AMOUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
NUM_MOVE_TOKENS = len(MOVE_AMOUNTS) * 2  # 20 move tokens

# Reserved token offsets (24-33): 10 reserved tokens for future use
RESERVED_TOKEN_START_OFFSET = MOVE_TOKEN_START_OFFSET + NUM_MOVE_TOKENS
NUM_RESERVED_TOKENS = 10

# Total number of special tokens
NUM_SPECIAL_TOKENS = 4 + NUM_MOVE_TOKENS + NUM_RESERVED_TOKENS  # 34 tokens

# Total vocabulary size
VOCAB_SIZE = GPT2_VOCAB_SIZE + NUM_SPECIAL_TOKENS


@dataclass
class SpecialTokens:
    """Container for special token IDs."""
    cursor: int = GPT2_VOCAB_SIZE + CURSOR_OFFSET
    end_of_input: int = GPT2_VOCAB_SIZE + END_OF_INPUT_OFFSET
    delete: int = GPT2_VOCAB_SIZE + DELETE_OFFSET
    end_of_response: int = GPT2_VOCAB_SIZE + END_OF_RESPONSE_OFFSET


# Singleton instance for easy access
SPECIAL_TOKENS = SpecialTokens()


def get_move_token_id(amount: int) -> int:
    """
    Get the token ID for a move operation.

    Args:
        amount: Move amount. Must be in {±1, ±2, ±4, ±8, ±16, ±32, ±64, ±128, ±256, ±512}

    Returns:
        Token ID for the move operation

    Raises:
        ValueError: If amount is not a valid move amount
    """
    abs_amount = abs(amount)
    if abs_amount not in MOVE_AMOUNTS:
        raise ValueError(f"Invalid move amount: {amount}")

    # Find index in MOVE_AMOUNTS
    idx = MOVE_AMOUNTS.index(abs_amount)

    # Each amount has two tokens: positive then negative
    base_offset = GPT2_VOCAB_SIZE + MOVE_TOKEN_START_OFFSET + (idx * 2)

    if amount > 0:
        return base_offset  # Positive move
    else:
        return base_offset + 1  # Negative move


def get_move_amount(token_id: int) -> int:
    """
    Get the move amount from a move token ID.

    Args:
        token_id: Token ID to check

    Returns:
        Move amount (positive or negative)

    Raises:
        ValueError: If token_id is not a move token
    """
    if not is_move_token(token_id):
        raise ValueError(f"Token ID {token_id} is not a move token")

    offset = token_id - GPT2_VOCAB_SIZE - MOVE_TOKEN_START_OFFSET
    amount_idx = offset // 2
    is_negative = offset % 2 == 1

    amount = MOVE_AMOUNTS[amount_idx]
    return -amount if is_negative else amount


def is_move_token(token_id: int) -> bool:
    """Check if a token ID is a move token."""
    offset = token_id - GPT2_VOCAB_SIZE
    return MOVE_TOKEN_START_OFFSET <= offset < MOVE_TOKEN_START_OFFSET + NUM_MOVE_TOKENS


def is_control_token(token_id: int) -> bool:
    """
    Check if a token ID is a control token (move or delete).
    These are tokens that don't insert new content.
    """
    return is_move_token(token_id) or token_id == SPECIAL_TOKENS.delete


def is_insert_token(token_id: int) -> bool:
    """
    Check if a token ID represents an insert operation.
    Insert tokens are regular vocabulary tokens (not special/control tokens).
    """
    return (
        token_id >= 0 and
        token_id < GPT2_VOCAB_SIZE and
        token_id != GPT2_EOT_TOKEN_ID
    )


def is_special_token(token_id: int) -> bool:
    """Check if a token ID is any of our special tokens."""
    return token_id >= GPT2_VOCAB_SIZE and token_id < VOCAB_SIZE


def get_reserved_token_ids() -> List[int]:
    """Get list of reserved token IDs."""
    start = GPT2_VOCAB_SIZE + RESERVED_TOKEN_START_OFFSET
    return list(range(start, start + NUM_RESERVED_TOKENS))


def get_all_move_token_ids() -> List[int]:
    """Get list of all move token IDs."""
    start = GPT2_VOCAB_SIZE + MOVE_TOKEN_START_OFFSET
    return list(range(start, start + NUM_MOVE_TOKENS))


def get_control_token_ids() -> Set[int]:
    """Get set of all control token IDs (moves + delete)."""
    tokens = set(get_all_move_token_ids())
    tokens.add(SPECIAL_TOKENS.delete)
    return tokens


def decompose_move(amount: int) -> List[int]:
    """
    Decompose an arbitrary move amount into a sequence of valid move token IDs.

    Uses greedy decomposition with largest moves first.
    For example: move(5) = [move(4), move(1)]
                 move(-7) = [move(-4), move(-2), move(-1)]

    Args:
        amount: Total amount to move (can be any integer)

    Returns:
        List of move token IDs that sum to the desired amount
    """
    if amount == 0:
        return []

    sign = 1 if amount > 0 else -1
    remaining = abs(amount)
    moves = []

    # Greedily use largest moves first
    for power in reversed(MOVE_AMOUNTS):
        while remaining >= power:
            moves.append(get_move_token_id(sign * power))
            remaining -= power

    return moves
