#!/usr/bin/env python3
"""Entry point for training BitrateLSTM on your prepared Parquet dataset.

Usage:
    python train.py \
        --data data/training_data.parquet \
        --seq-len 10 \
        --batch-size 16 \
        --hidden-size 128 \
        --num-layers 3 \
        --dropout 0.2 \
        --lr 1e-3 \
        --epochs 5 \
        --save-model models/bitrate_lstm.pt
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from core.dataset import TraceDataset
from core.model import BitrateLSTM, quantile_loss


def parse_args():
    """Parse command-line arguments for training the BitrateLSTM model."""
    p = argparse.ArgumentParser(description="Train BitrateLSTM model")
    p.add_argument(
        "--data", type=str, required=True, help="Path to training_data.parquet"
    )
    p.add_argument("--seq-len", type=int, default=10, help="Number of input timesteps")
    p.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    p.add_argument("--hidden-size", type=int, default=128, help="Hidden size of LSTM")
    p.add_argument(
        "--num-layers", type=int, default=3, help="Number of stacked LSTM layers"
    )
    p.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout between LSTM layers"
    )
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p.add_argument(
        "--save-model",
        type=str,
        default="bitrate_lstm.pt",
        help="Where to save the trained model",
    )
    return p.parse_args()


def main():
    """Train the BitrateLSTM model on the specified dataset."""
    args = parse_args()

    # Create output dir if needed
    os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = TraceDataset(parquet_path=args.data, seq_len=args.seq_len, normalize=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Loaded dataset with {len(dataset)} samples")

    # Model, optimizer
    model = BitrateLSTM(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_outputs=3,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)  # (B, seq_len, 1)
            y_batch = y_batch.to(device)  # (B,)

            preds, _ = model(x_batch)  # (B, 3)
            loss = quantile_loss(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{args.epochs} — Avg Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), args.save_model)
    print(f"Model weights saved to {args.save_model}")


if __name__ == "__main__":
    main()
