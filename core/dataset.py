"""Dataset utilities for sequence prediction tasks using PyTorch.

This module provides the TraceDataset class for loading and preprocessing
sequence data from Parquet files for use in PyTorch models.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TraceDataset(Dataset):
    """A Dataset that emits (X, y) pairs for sequence prediction tasks.

    - Reads a Parquet of shape [sample_id, timestep, throughput]
    - Groups by sample_id, sorts by timestep
    - Emits (X, y) where
        X: FloatTensor [seq_len, 1]  (normalized throughput)
        y: FloatTensor ()           (next-step throughput)
    """

    def __init__(
        self,
        parquet_path: str,
        seq_len: int,
        normalize: bool = True,
        stats: dict = None,
    ):
        """Initialize the TraceDataset.

        Args:
            parquet_path (str): Path to the Parquet file containing the data.
            seq_len (int): Length of the input sequence.
            normalize (bool, optional): Whether to normalize throughput values. Defaults to True.
            stats (dict, optional): Precomputed normalization statistics. Defaults to None.

        """
        # Load DataFrame from Parquet
        self.df = pd.read_parquet(parquet_path)
        self.seq_len = seq_len

        # Compute normalization stats if needed
        if normalize:
            if stats is None:
                vals = self.df["throughput"].values.astype(np.float32)
                self.stats = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=0)) + 1e-6,
                }
            else:
                self.stats = stats
        else:
            self.stats = {"mean": 0.0, "std": 1.0}

        # Unique sample IDs
        self.sample_ids = self.df["sample_id"].unique().tolist()

    def __len__(self):
        """Return the number of unique samples in the dataset."""
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        """Retrieve the (X, y) pair for the given sample index."""
        sid = self.sample_ids[idx]
        block = (
            self.df[self.df["sample_id"] == sid]
            .sort_values("timestep")
            .reset_index(drop=True)
        )

        # Ensure seq_len + 1 rows
        assert (
            len(block) == self.seq_len + 1
        ), f"Sample {sid} has {len(block)} rows, expected {self.seq_len+1}"

        # Extract throughput sequence
        seq = block.loc[: self.seq_len - 1, "throughput"].to_numpy(dtype=np.float32)
        seq = (seq - self.stats["mean"]) / self.stats["std"]
        x = torch.from_numpy(seq).unsqueeze(-1)  # shape: (seq_len, 1)

        # Next-step target
        nxt = float(block.loc[self.seq_len, "throughput"])
        y = torch.tensor(
            (nxt - self.stats["mean"]) / self.stats["std"], dtype=torch.float32
        )

        return x, y
