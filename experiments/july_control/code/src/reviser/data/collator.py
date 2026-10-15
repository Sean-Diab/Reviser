"""
Collator for batching Cursor transformer training samples.

Handles padding of variable-length sequences and canvas states.
"""

from typing import Dict, List, Any, Optional
import torch


class CursorCollator:
    """
    Collates samples into batches for Cursor transformer training.

    Handles:
    - Padding input_ids and labels to max length in batch
    - Padding canvas sequences (2D: timesteps x canvas_length)
    - Creating attention masks
    """

    def __init__(
        self,
        pad_token_id: int,
        max_seq_length: Optional[int] = None,
        max_canvas_length: Optional[int] = None,
    ):
        """
        Initialize the collator.

        Args:
            pad_token_id: Token ID to use for padding
            max_seq_length: Optional maximum sequence length (truncates if exceeded)
            max_canvas_length: Optional maximum canvas length per timestep
        """
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.max_canvas_length = max_canvas_length

    def __call__(self, batch: List[Optional[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            batch: List of sample dictionaries (may contain None for failed samples)

        Returns:
            Dictionary with batched tensors:
            - input_ids: (batch_size, max_seq_len)
            - labels: (batch_size, max_seq_len)
            - attention_mask: (batch_size, max_seq_len)
            - canvas_ids: (batch_size, max_seq_len, max_canvas_len)
            - canvas_mask: (batch_size, max_seq_len, max_canvas_len)
            - prompt_lengths: (batch_size,)
        """
        # Filter out None samples
        batch = [s for s in batch if s is not None]

        if len(batch) == 0:
            raise ValueError("Empty batch after filtering None samples")

        # Get max lengths
        max_seq_len = max(len(s["input_ids"]) for s in batch)
        if self.max_seq_length is not None:
            max_seq_len = min(max_seq_len, self.max_seq_length)

        # Find max canvas length across all timesteps and samples
        max_canvas_len = 1  # Minimum of 1 to avoid empty tensors
        for s in batch:
            for canvas in s["canvas_sequences"]:
                if len(canvas) > max_canvas_len:
                    max_canvas_len = len(canvas)

        if self.max_canvas_length is not None:
            max_canvas_len = min(max_canvas_len, self.max_canvas_length)

        batch_size = len(batch)

        # Initialize tensors
        input_ids = torch.full(
            (batch_size, max_seq_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, max_seq_len),
            -100,  # Ignore index for loss
            dtype=torch.long,
        )
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        # Canvas tensors: (batch, seq_len, max_canvas_len)
        canvas_ids = torch.zeros(
            batch_size, max_seq_len, max_canvas_len,
            dtype=torch.long,
        )
        canvas_mask = torch.zeros(
            batch_size, max_seq_len, max_canvas_len,
            dtype=torch.long,
        )

        prompt_lengths = torch.zeros(batch_size, dtype=torch.long)

        # Fill tensors
        for i, sample in enumerate(batch):
            seq_len = min(len(sample["input_ids"]), max_seq_len)

            # Input IDs and labels
            input_ids[i, :seq_len] = torch.tensor(sample["input_ids"][:seq_len])
            labels[i, :seq_len] = torch.tensor(sample["labels"][:seq_len])
            attention_mask[i, :seq_len] = 1

            # Prompt length
            prompt_lengths[i] = min(sample["prompt_length"], max_seq_len)

            # Canvas sequences
            canvas_sequences = sample["canvas_sequences"]
            for t in range(min(len(canvas_sequences), max_seq_len)):
                canvas = canvas_sequences[t]
                canvas_len = min(len(canvas), max_canvas_len)
                if canvas_len > 0:
                    canvas_ids[i, t, :canvas_len] = torch.tensor(canvas[:canvas_len])
                    canvas_mask[i, t, :canvas_len] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "canvas_ids": canvas_ids,
            "canvas_mask": canvas_mask,
            "prompt_lengths": prompt_lengths,
        }


class DynamicBatchCollator(CursorCollator):
    """
    Collator that also handles dynamic batching based on token count.

    Useful for maximizing GPU utilization by packing sequences efficiently.
    """

    def __init__(
        self,
        pad_token_id: int,
        max_tokens_per_batch: int = 8192,
        max_seq_length: Optional[int] = None,
        max_canvas_length: Optional[int] = None,
    ):
        """
        Initialize the dynamic batch collator.

        Args:
            pad_token_id: Token ID for padding
            max_tokens_per_batch: Maximum total tokens in a batch
            max_seq_length: Maximum sequence length
            max_canvas_length: Maximum canvas length
        """
        super().__init__(pad_token_id, max_seq_length, max_canvas_length)
        self.max_tokens_per_batch = max_tokens_per_batch

    def create_batches(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Create batches based on token count limits.

        Args:
            samples: List of samples

        Returns:
            List of batches, where each batch is a list of samples
        """
        # Sort by length for efficient batching
        sorted_samples = sorted(
            samples,
            key=lambda x: len(x["input_ids"]) if x is not None else 0,
        )

        batches = []
        current_batch = []
        current_tokens = 0
        current_max_len = 0

        for sample in sorted_samples:
            if sample is None:
                continue

            seq_len = len(sample["input_ids"])

            # Check if adding this sample would exceed token limit
            new_max_len = max(current_max_len, seq_len)
            new_total = new_max_len * (len(current_batch) + 1)

            if new_total > self.max_tokens_per_batch and len(current_batch) > 0:
                # Start new batch
                batches.append(current_batch)
                current_batch = [sample]
                current_max_len = seq_len
            else:
                current_batch.append(sample)
                current_max_len = new_max_len

        if current_batch:
            batches.append(current_batch)

        return batches
