"""
Tokenizer for Cursor transformer.
Wraps GPT-2 tokenizer and adds special tokens for cursor operations.
"""

import os
from typing import List, Union
from transformers import GPT2Tokenizer


class CursorTokenizer:
    """
    GPT-2 based tokenizer with special tokens for cursor operations.

    Special tokens:
    - [CURSOR]: Cursor position marker
    - [END_OF_INPUT]: Separator between prompt and edit history
    - [DELETE]: Delete token to the left of cursor
    - [END_OF_RESPONSE]: End of model response
    - [MOVE_+N], [MOVE_-N]: Move cursor by N positions (N = 1, 2, 4, 8, ..., 512)
    """

    # Move amounts are powers of 2 up to 512
    MOVE_AMOUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    def __init__(self):
        # Load base GPT-2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Hugging Face GPT-2 tokenizer defaults to model_max_length=1024 and will warn on longer
        # sequences. Cursor regularly exceeds this during RL rollouts (prompt + edit history),
        # so bump the limit to avoid noisy warnings (this does NOT truncate by itself).
        self.tokenizer.model_max_length = int(os.environ.get("CURSOR_TOKENIZER_MAX_LEN", "4096"))

        # Define special tokens
        self.special_tokens = [
            "[CURSOR]",
            "[END_OF_INPUT]",
            "[DELETE]",
            "[END_OF_RESPONSE]",
        ]

        # Add move tokens (positive and negative for each power of 2)
        for amount in self.MOVE_AMOUNTS:
            self.special_tokens.append(f"[MOVE_+{amount}]")
            self.special_tokens.append(f"[MOVE_-{amount}]")

        # Add 10 reserved tokens for future use
        for i in range(10):
            self.special_tokens.append(f"[RESERVED_{i}]")

        # Add special tokens to tokenizer
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": self.special_tokens
        })

        # Set pad token to eos token (GPT-2 doesn't have a pad token by default)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache special token IDs for fast access
        self._cache_special_token_ids()

    def _cache_special_token_ids(self):
        """Cache special token IDs for fast lookup."""
        self.cursor_token_id = self.tokenizer.convert_tokens_to_ids("[CURSOR]")
        self.end_of_input_token_id = self.tokenizer.convert_tokens_to_ids("[END_OF_INPUT]")
        self.delete_token_id = self.tokenizer.convert_tokens_to_ids("[DELETE]")
        self.end_of_response_token_id = self.tokenizer.convert_tokens_to_ids("[END_OF_RESPONSE]")
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # Cache move token IDs
        self.move_token_ids = {}
        for amount in self.MOVE_AMOUNTS:
            pos_token = f"[MOVE_+{amount}]"
            neg_token = f"[MOVE_-{amount}]"
            self.move_token_ids[amount] = self.tokenizer.convert_tokens_to_ids(pos_token)
            self.move_token_ids[-amount] = self.tokenizer.convert_tokens_to_ids(neg_token)

        # Create reverse mapping for decoding move amounts
        self.id_to_move_amount = {v: k for k, v in self.move_token_ids.items()}

        # Cache reserved token IDs
        self.reserved_token_ids = []
        for i in range(10):
            token_id = self.tokenizer.convert_tokens_to_ids(f"[RESERVED_{i}]")
            self.reserved_token_ids.append(token_id)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.tokenizer)

    @property
    def base_vocab_size(self) -> int:
        """Original GPT-2 vocabulary size before special tokens."""
        return 50257

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens (default False)

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_move_token_id(self, amount: int) -> int:
        """
        Get token ID for a move operation.

        Args:
            amount: Move amount (positive or negative). Must be in ±{1, 2, 4, ..., 512}

        Returns:
            Token ID for the move operation

        Raises:
            ValueError: If amount is not a valid move amount
        """
        if amount not in self.move_token_ids:
            raise ValueError(f"Invalid move amount: {amount}. Must be in ±{{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}}")
        return self.move_token_ids[amount]

    def get_move_amount(self, token_id: int) -> int:
        """
        Get move amount from a move token ID.

        Args:
            token_id: Token ID to check

        Returns:
            Move amount (positive or negative)

        Raises:
            ValueError: If token_id is not a move token
        """
        if token_id not in self.id_to_move_amount:
            raise ValueError(f"Token ID {token_id} is not a move token")
        return self.id_to_move_amount[token_id]

    def is_move_token(self, token_id: int) -> bool:
        """Check if token ID is a move token."""
        return token_id in self.id_to_move_amount

    def is_special_token(self, token_id: int) -> bool:
        """Check if token ID is any special token (cursor ops or control)."""
        return (
            token_id == self.cursor_token_id or
            token_id == self.end_of_input_token_id or
            token_id == self.delete_token_id or
            token_id == self.end_of_response_token_id or
            token_id in self.id_to_move_amount
        )

    def is_insert_token(self, token_id: int) -> bool:
        """
        Check if token ID represents an insert operation.
        Insert tokens are all regular vocabulary tokens (not special tokens).
        """
        return not self.is_special_token(token_id) and token_id != self.pad_token_id

    def decompose_move(self, amount: int) -> List[int]:
        """
        Decompose a move amount into a sequence of power-of-2 move token IDs.

        For example, move by +5 = move by +4 then +1

        Args:
            amount: Total amount to move (can be positive or negative)

        Returns:
            List of move token IDs that sum to the desired amount
        """
        if amount == 0:
            return []

        sign = 1 if amount > 0 else -1
        remaining = abs(amount)
        moves = []

        # Greedily use largest moves first
        for power in reversed(self.MOVE_AMOUNTS):
            while remaining >= power:
                moves.append(self.get_move_token_id(sign * power))
                remaining -= power

        return moves

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
