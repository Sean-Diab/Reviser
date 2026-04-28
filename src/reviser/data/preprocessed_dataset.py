"""
Dataset that loads preprocessed OpenOrca with precomputed canvas states.

This eliminates the O(N^2) bottleneck by loading precomputed canvas states
instead of computing them on-the-fly during training.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset


class PreprocessedOpenOrcaDataset(Dataset):
    """
    PyTorch Dataset for preprocessed OpenOrca data.

    Loads samples with precomputed canvas states, eliminating the
    data loading bottleneck during training.

    Each sample contains:
    - input_ids: prompt + END_OF_INPUT + restoration trajectory
    - labels: same but prompt masked with -100
    - canvas_states: precomputed true canvas at each timestep
    - prompt_length: length of prompt
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the preprocessed dataset.

        Args:
            data_dir: Directory containing preprocessed checkpoint files
            split: Dataset split (for compatibility, not used)
            max_samples: Optional limit on number of samples to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples: List[Dict[str, Any]] = []

        # Load metadata
        metadata_path = self.data_dir / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Loaded metadata: {self.metadata['total_samples']:,} total samples")
        else:
            print("Warning: No metadata.pkl found")
            self.metadata = {}

        # Load all checkpoint files
        checkpoint_files = sorted(self.data_dir.glob("checkpoint_*.pkl"))

        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {self.data_dir}. "
                "Run preprocess_dataset.py first!"
            )

        print(f"Loading {len(checkpoint_files)} checkpoint files from {self.data_dir}...")

        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_samples = pickle.load(f)
                self.samples.extend(checkpoint_samples)

            if max_samples is not None and len(self.samples) >= max_samples:
                self.samples = self.samples[:max_samples]
                break

        print(f"Loaded {len(self.samples):,} preprocessed samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single preprocessed sample.

        This is FAST because canvas_states are already computed!
        No O(N²) Python loops, just a simple dictionary lookup.

        Returns:
            Dictionary with:
            - input_ids: List[int]
            - labels: List[int]
            - canvas_states: List[List[int]] (precomputed!)
            - prompt_length: int
        """
        return self.samples[idx]


class StreamingPreprocessedDataset(Dataset):
    """
    Memory-efficient version that loads checkpoints on-demand.

    Instead of loading all samples into memory at once, this loads
    checkpoint files as needed.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        checkpoint_size: int = 10_000,
    ):
        """
        Initialize the streaming preprocessed dataset.

        Args:
            data_dir: Directory containing preprocessed checkpoint files
            split: Dataset split
            checkpoint_size: Number of samples per checkpoint file
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.checkpoint_size = checkpoint_size

        # Load metadata
        metadata_path = self.data_dir / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            self.total_samples = self.metadata['total_samples']
        else:
            raise FileNotFoundError(f"metadata.pkl not found in {data_dir}")

        # Find all checkpoint files
        self.checkpoint_files = sorted(self.data_dir.glob("checkpoint_*.pkl"))

        if not self.checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {data_dir}")

        print(f"Streaming from {len(self.checkpoint_files)} checkpoint files")
        print(f"Total samples: {self.total_samples:,}")

        # Cache for current checkpoint
        self._current_checkpoint_idx = -1
        self._current_checkpoint_data = []

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single preprocessed sample by loading checkpoint if needed.

        Args:
            idx: Sample index (0 to total_samples-1)

        Returns:
            Dictionary with input_ids, labels, canvas_states, prompt_length
        """
        # Determine which checkpoint file this index belongs to
        checkpoint_idx = idx // self.checkpoint_size
        local_idx = idx % self.checkpoint_size

        # Load checkpoint if not already loaded
        if checkpoint_idx != self._current_checkpoint_idx:
            checkpoint_file = self.checkpoint_files[checkpoint_idx]
            with open(checkpoint_file, 'rb') as f:
                self._current_checkpoint_data = pickle.load(f)
            self._current_checkpoint_idx = checkpoint_idx

        return self._current_checkpoint_data[local_idx]


def create_preprocessed_dataset(
    data_dir: str,
    split: str = "train",
    streaming: bool = False,
    **kwargs,
) -> Dataset:
    """
    Factory function to create the appropriate preprocessed dataset.

    Args:
        data_dir: Directory containing preprocessed data
        split: Dataset split
        streaming: Whether to use streaming mode (loads checkpoints on-demand)
        **kwargs: Additional arguments

    Returns:
        Dataset instance
    """
    if streaming:
        return StreamingPreprocessedDataset(data_dir, split=split, **kwargs)
    else:
        return PreprocessedOpenOrcaDataset(data_dir, split=split, **kwargs)
