#!/usr/bin/env python3
"""Prepare data for training.

If run with no args, generates a tiny dummy trace dataset of length `seq_len + 1` per sample.
Later, point `--input-dir` at data/raw/ to process your real traces.
"""

import argparse
import os

import numpy as np
import pandas as pd


def generate_dummy(num_samples: int = 1000, seq_len: int = 10):
    """Create a dummy timeseries dataset for bitrate prediction.

    Each sample_id will have seq_len+1 rows:
      - first seq_len rows: features (throughput)
      - last row: target (we'll store it as throughput and let the Dataset code pick it up)
    """
    total_len = seq_len + 1
    rows = []
    for sid in range(num_samples):
        # random throughput trace of length seq_len+1
        thru = np.random.rand(total_len).astype(np.float32) * 5 + 1
        for t in range(total_len):
            rows.append(
                {
                    "sample_id": sid,
                    "timestep": t,
                    "throughput": thru[t],
                }
            )
    df = pd.DataFrame(rows)
    return df


def main():
    """Parse arguments and prepare training data (dummy or real) for model training."""
    p = argparse.ArgumentParser(description="Prepare training data (dummy or real).")
    p.add_argument(
        "--input-dir", type=str, help="Path to raw trace folders", default=None
    )
    p.add_argument(
        "--output",
        type=str,
        help="Where to write the Parquet",
        default="data/training_data.parquet",
    )
    p.add_argument(
        "--dummy", action="store_true", help="Generate a small dummy dataset"
    )
    p.add_argument(
        "--num-samples", type=int, help="Number of dummy samples", default=100
    )
    p.add_argument(
        "--seq-len",
        type=int,
        help="Number of input timesteps (the script will add 1 for the target)",
        default=10,
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.dummy or args.input_dir is None:
        df = generate_dummy(num_samples=args.num_samples, seq_len=args.seq_len)
        print(
            f"Generated dummy DataFrame with shape {df.shape} "
            f"({args.num_samples} samples × {args.seq_len+1} rows each)"
        )
    else:
        # TODO: implement real-trace parsing into same schema
        print(f"Processing real traces from {args.input_dir} …")
        # fallback to dummy
        df = generate_dummy(num_samples=10, seq_len=args.seq_len)
        print(f"[FALLBACK] Dummy DataFrame with shape {df.shape}")

    df.to_parquet(args.output, index=False)
    print(f"Wrote training data to {args.output}")


if __name__ == "__main__":
    main()
