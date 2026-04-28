"""
Dataset loading and processing for Cursor transformer training.

Loads OpenOrca dataset and generates obfuscation trajectories on-the-fly.
"""

from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset

from .tokenizer import CursorTokenizer
from .obfuscator import Obfuscator, try_generate_sample


class OpenOrcaDataset(Dataset):
    """
    PyTorch Dataset wrapper for OpenOrca with on-the-fly trajectory generation.

    Each sample contains:
    - input_ids: prompt + END_OF_INPUT + restoration trajectory
    - labels: same but prompt masked with -100
    - canvas_sequences: true canvas at each timestep
    - canvas_lengths: length of each canvas
    """

    def __init__(
        self,
        tokenizer: CursorTokenizer,
        split: str = "train",
        max_prompt_length: int = 256,
        max_response_length: int = 256,
        max_edit_steps: int = 1000,
        delete_prob: float = 0.60,
        move_prob: float = 0.25,
        insert_prob: float = 0.15,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the dataset.

        Args:
            tokenizer: CursorTokenizer instance
            split: Dataset split ("train" or "validation")
            max_prompt_length: Maximum prompt length in tokens
            max_response_length: Maximum response length in tokens
            max_edit_steps: Maximum edit steps for obfuscation
            delete_prob: Delete probability for obfuscation
            move_prob: Move probability for obfuscation
            insert_prob: Insert probability for obfuscation
            cache_dir: Directory for caching dataset
            max_samples: Optional limit on number of samples
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

        # Initialize obfuscator
        self.obfuscator = Obfuscator(
            delete_prob=delete_prob,
            move_prob=move_prob,
            insert_prob=insert_prob,
            max_edit_steps=max_edit_steps,
            vocab_size=tokenizer.base_vocab_size,
        )

        # Load OpenOrca dataset
        self.dataset = load_dataset(
            "Open-Orca/OpenOrca",
            split=split,
            cache_dir=cache_dir,
        )

        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a single training sample.

        Returns None if the sample cannot be processed (e.g., obfuscation fails).
        The collator should handle None values.
        """
        item = self.dataset[idx]

        # Extract prompt and response from OpenOrca format
        # OpenOrca has: system_prompt, question, response
        system_prompt = item.get("system_prompt", "")
        question = item.get("question", "")
        response = item.get("response", "")

        # Combine system prompt and question for the prompt
        if system_prompt:
            prompt = f"{system_prompt}\n\n{question}"
        else:
            prompt = question

        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt)
        response_tokens = self.tokenizer.encode(response)

        # Truncate if needed
        if len(prompt_tokens) > self.max_prompt_length:
            prompt_tokens = prompt_tokens[:self.max_prompt_length]
        if len(response_tokens) > self.max_response_length:
            response_tokens = response_tokens[:self.max_response_length]

        # Skip empty responses
        if len(response_tokens) == 0:
            return None

        # Generate training sample with obfuscation
        sample = try_generate_sample(
            self.obfuscator,
            prompt_tokens,
            response_tokens,
            max_attempts=3,
        )

        if sample is None:
            return None

        return {
            "input_ids": sample.input_ids,
            "labels": sample.labels,
            "canvas_sequences": sample.canvas_states,
            "prompt_length": sample.prompt_length,
        }


class StreamingOpenOrcaDataset(IterableDataset):
    """
    Streaming version of OpenOrca dataset for memory-efficient training.

    Generates samples on-the-fly without loading the entire dataset into memory.
    """

    def __init__(
        self,
        tokenizer: CursorTokenizer,
        split: str = "train",
        max_prompt_length: int = 256,
        max_response_length: int = 256,
        max_edit_steps: int = 1000,
        delete_prob: float = 0.60,
        move_prob: float = 0.25,
        insert_prob: float = 0.15,
        cache_dir: Optional[str] = None,
        buffer_size: int = 1000,
    ):
        """
        Initialize the streaming dataset.

        Args:
            tokenizer: CursorTokenizer instance
            split: Dataset split
            max_prompt_length: Maximum prompt length in tokens
            max_response_length: Maximum response length in tokens
            max_edit_steps: Maximum edit steps for obfuscation
            delete_prob: Delete probability for obfuscation
            move_prob: Move probability for obfuscation
            insert_prob: Insert probability for obfuscation
            cache_dir: Directory for caching dataset
            buffer_size: Buffer size for shuffling
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.split = split
        self.cache_dir = cache_dir
        self.buffer_size = buffer_size

        # Obfuscator parameters (created fresh for each worker)
        self.obfuscator_params = {
            "delete_prob": delete_prob,
            "move_prob": move_prob,
            "insert_prob": insert_prob,
            "max_edit_steps": max_edit_steps,
            "vocab_size": tokenizer.base_vocab_size,
        }

    def __iter__(self):
        """Iterate over the dataset, generating samples on-the-fly."""
        # Load dataset in streaming mode
        dataset = load_dataset(
            "Open-Orca/OpenOrca",
            split=self.split,
            streaming=True,
            cache_dir=self.cache_dir,
        )

        # Shuffle with buffer
        dataset = dataset.shuffle(buffer_size=self.buffer_size)

        # Create obfuscator for this worker
        obfuscator = Obfuscator(**self.obfuscator_params)

        for item in dataset:
            sample = self._process_item(item, obfuscator)
            if sample is not None:
                yield sample

    def _process_item(
        self,
        item: Dict[str, str],
        obfuscator: Obfuscator,
    ) -> Optional[Dict[str, Any]]:
        """Process a single item from the dataset."""
        system_prompt = item.get("system_prompt", "")
        question = item.get("question", "")
        response = item.get("response", "")

        if system_prompt:
            prompt = f"{system_prompt}\n\n{question}"
        else:
            prompt = question

        prompt_tokens = self.tokenizer.encode(prompt)
        response_tokens = self.tokenizer.encode(response)

        if len(prompt_tokens) > self.max_prompt_length:
            prompt_tokens = prompt_tokens[:self.max_prompt_length]
        if len(response_tokens) > self.max_response_length:
            response_tokens = response_tokens[:self.max_response_length]

        if len(response_tokens) == 0:
            return None

        sample = try_generate_sample(
            obfuscator,
            prompt_tokens,
            response_tokens,
            max_attempts=3,
        )

        if sample is None:
            return None

        return {
            "input_ids": sample.input_ids,
            "labels": sample.labels,
            "canvas_sequences": sample.canvas_states,
            "prompt_length": sample.prompt_length,
        }


def create_dataset(
    tokenizer: CursorTokenizer,
    split: str = "train",
    streaming: bool = False,
    **kwargs,
) -> Dataset:
    """
    Factory function to create the appropriate dataset.

    Args:
        tokenizer: CursorTokenizer instance
        split: Dataset split
        streaming: Whether to use streaming mode
        **kwargs: Additional arguments passed to dataset constructor

    Returns:
        Dataset instance
    """
    if streaming:
        return StreamingOpenOrcaDataset(tokenizer, split=split, **kwargs)
    else:
        return OpenOrcaDataset(tokenizer, split=split, **kwargs)
