"""
Efficient dataset loader for preprocessed batches.

Loads preprocessed data created by preprocess_efficient.py.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset, IterableDataset
import random


class PreprocessedBatchDataset(Dataset):
    """
    Dataset that loads preprocessed batches from disk.

    This is FAST because all the O(N²) canvas state computation
    was done offline during preprocessing.

    Each sample contains:
    - input_ids: List[int] - prompt + END_OF_INPUT + restoration trajectory
    - labels: List[int] - same as input_ids but prompt masked with -100
    - canvas_states: List[List[int]] - precomputed canvas token IDs at each timestep
    - prompt_length: int - length of prompt including END_OF_INPUT
    """

    def __init__(
        self,
        data_dir: str,
        shuffle_batches: bool = True,
        max_batches: Optional[int] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing preprocessed batch files
            shuffle_batches: Whether to shuffle the order of batches
            max_batches: Maximum number of batches to load (for testing)
        """
        self.data_dir = Path(data_dir)

        # Load summary
        summary_file = self.data_dir / "summary.npz"
        if not summary_file.exists():
            raise FileNotFoundError(
                f"No summary.npz found in {data_dir}. "
                "Run preprocess_efficient.py first!"
            )

        summary = np.load(summary_file, allow_pickle=True)
        self.total_samples = int(summary['total_samples'])
        self.num_batches = int(summary['num_batches'])
        self.examples_per_batch = int(summary['examples_per_batch'])

        print(f"Loading preprocessed dataset from {self.data_dir}")
        print(f"  Total samples: {self.total_samples:,}")
        print(f"  Number of batches: {self.num_batches}")
        print(f"  Examples per batch: {self.examples_per_batch}")

        # Find all batch files
        self.batch_files = sorted(self.data_dir.glob("batch_*.npz"))

        if not self.batch_files:
            raise FileNotFoundError(f"No batch files found in {data_dir}")

        if max_batches:
            self.batch_files = self.batch_files[:max_batches]
            print(f"  Limited to {max_batches} batches")

        if shuffle_batches:
            random.shuffle(self.batch_files)
            print(f"  Batch order shuffled")

        # Load all samples into memory
        # (This is fine because we eliminated the O(N²) bottleneck,
        #  so the data is much smaller and loading is fast)
        print("  Loading batches into memory...")
        self.samples = []

        for batch_file in self.batch_files:
            batch_data = np.load(batch_file, allow_pickle=True)

            # Extract samples from this batch
            input_ids = batch_data['input_ids']
            labels = batch_data['labels']
            canvas_states = batch_data['canvas_states']
            prompt_lengths = batch_data['prompt_lengths']

            num_samples = len(input_ids)
            for i in range(num_samples):
                # Data is already in Python list format (stored as object dtype)
                sample_input_ids = input_ids[i]
                sample_labels = labels[i]
                sample_canvas = canvas_states[i]

                # Convert to list if it's a numpy array, otherwise keep as is
                if hasattr(sample_input_ids, 'tolist'):
                    sample_input_ids = sample_input_ids.tolist()
                if hasattr(sample_labels, 'tolist'):
                    sample_labels = sample_labels.tolist()
                if hasattr(sample_canvas, 'tolist'):
                    sample_canvas = sample_canvas.tolist()

                self.samples.append({
                    'input_ids': sample_input_ids,
                    'labels': sample_labels,
                    'canvas_sequences': sample_canvas,  # Collator expects 'canvas_sequences'
                    'prompt_length': int(prompt_lengths[i]),
                })

        print(f"  Loaded {len(self.samples):,} samples into memory")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a preprocessed sample.

        This is VERY FAST - just a dictionary lookup!
        No O(N^2) computation, no Python loops, no obfuscation simulation.
        """
        return self.samples[idx]


class StreamingPreprocessedDataset(IterableDataset):
    """
    Memory-efficient streaming version that loads batches on-demand.

    Use this if you have limited RAM and many preprocessed batches.
    """

    def __init__(
        self,
        data_dir: str,
        shuffle_batches: bool = True,
        shuffle_within_batch: bool = True,
    ):
        """
        Initialize the streaming dataset.

        Args:
            data_dir: Directory containing preprocessed batch files
            shuffle_batches: Whether to shuffle the order of batches
            shuffle_within_batch: Whether to shuffle samples within each batch
        """
        self.data_dir = Path(data_dir)
        self.shuffle_batches = shuffle_batches
        self.shuffle_within_batch = shuffle_within_batch

        # Load summary
        summary_file = self.data_dir / "summary.npz"
        if not summary_file.exists():
            raise FileNotFoundError(f"No summary.npz found in {data_dir}")

        summary = np.load(summary_file, allow_pickle=True)
        self.total_samples = int(summary['total_samples'])
        self.num_batches = int(summary['num_batches'])

        print(f"Streaming preprocessed dataset from {self.data_dir}")
        print(f"  Total samples: {self.total_samples:,}")
        print(f"  Number of batches: {self.num_batches}")

        # Find all batch files
        self.batch_files = sorted(self.data_dir.glob("batch_*.npz"))

        if not self.batch_files:
            raise FileNotFoundError(f"No batch files found in {data_dir}")

    def __iter__(self):
        """Iterate over samples, loading batches on-demand."""
        batch_files = list(self.batch_files)

        if self.shuffle_batches:
            random.shuffle(batch_files)

        for batch_file in batch_files:
            # Load batch
            batch_data = np.load(batch_file, allow_pickle=True)

            input_ids = batch_data['input_ids']
            labels = batch_data['labels']
            canvas_states = batch_data['canvas_states']
            prompt_lengths = batch_data['prompt_lengths']

            num_samples = len(input_ids)

            # Create sample indices
            indices = list(range(num_samples))
            if self.shuffle_within_batch:
                random.shuffle(indices)

            # Yield samples
            for i in indices:
                # Convert to list if needed
                sample_input_ids = input_ids[i]
                sample_labels = labels[i]
                sample_canvas = canvas_states[i]

                if hasattr(sample_input_ids, 'tolist'):
                    sample_input_ids = sample_input_ids.tolist()
                if hasattr(sample_labels, 'tolist'):
                    sample_labels = sample_labels.tolist()
                if hasattr(sample_canvas, 'tolist'):
                    sample_canvas = sample_canvas.tolist()

                yield {
                    'input_ids': sample_input_ids,
                    'labels': sample_labels,
                    'canvas_sequences': sample_canvas,  # Collator expects 'canvas_sequences'
                    'prompt_length': int(prompt_lengths[i]),
                }


def create_preprocessed_dataset(
    data_dir: str,
    streaming: bool = False,
    **kwargs,
) -> Dataset:
    """
    Factory function to create preprocessed dataset.

    Args:
        data_dir: Directory containing preprocessed batches
        streaming: Whether to use streaming mode (on-demand loading)
        **kwargs: Additional arguments

    Returns:
        Dataset instance
    """
    if streaming:
        return StreamingPreprocessedDataset(data_dir, **kwargs)
    else:
        return PreprocessedBatchDataset(data_dir, **kwargs)
