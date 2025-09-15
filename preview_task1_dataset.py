"""
Task 1 — Dataset Preview Script
================================

Purpose
-------
Compose the Task-1 transform pipeline (undersampling with configurable
acceleration and center fraction, light image-domain augmentation,
normalization, and tensor formatting), instantiate `RandomPhantomDataset`,
and save a few preview images for inspection and the write-up.

Usage
-----
  python preview_task1_dataset.py \
    --num-samples 8 \
    --out-dir report/figs \
    --accel 4 \
    --center-frac 0.08 \
    --rot 10 \
    --shift 2 \
    --seed 123

Outputs
-------
PNG images named: `task1_sample_0.png`, `task1_sample_1.png`, ...
saved to `--out-dir` (default: `report/figs`).

Notes
-----
- Augmentations are applied in image domain and identically to real/imag.
- Preview images use a simple coil-wise max magnitude projection.
- This script is lightweight and does not depend on training code.
"""

import os
import argparse
import torch
import matplotlib
matplotlib.use("Agg")  # ensure headless save
import matplotlib.pyplot as plt

from transforms import Normalize, EquispacedUndersample, Augmentation, ToCompatibleTensor
from datasets import RandomPhantomDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview Task-1 dataset samples.")
    parser.add_argument("--num-samples", type=int, default=8, help="How many samples to render.")
    parser.add_argument("--out-dir", type=str, default="report/figs", help="Directory to save figures.")
    # Required Task-1 undersampling parameters (defaults per brief)
    parser.add_argument("--accel", type=int, default=4, help="Equispaced acceleration factor (default: 4).")
    parser.add_argument("--center-frac", type=float, default=0.08,
                        help="Center fraction for low-freq band (default: 0.08).")
    # Our augmentation choices (document in report)
    parser.add_argument("--rot", type=float, default=10.0,
                        help="Max rotation (degrees) for augmentation (default: 10).")
    parser.add_argument("--shift", type=float, default=2.0,
                        help="Max translation (pixels) for augmentation (default: 2).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    return parser.parse_args()


def show_tensor(chw: torch.Tensor, title: str, path: str) -> None:
    """
    Render a quick visualization from (channels, H, W) where channels = coils*2 (real, imag).
    We reconstruct coil-wise magnitude and take a max-projection across coils.
    """
    C2, H, W = chw.shape
    assert C2 % 2 == 0, "Expected channels = coils*2 (real, imag)."
    C = C2 // 2
    x = chw.reshape(C, 2, H, W)          # (C, 2, H, W)
    real = x[:, 0]
    imag = x[:, 1]
    mag = torch.sqrt(real**2 + imag**2)  # (C, H, W)
    mag_proj = mag.max(dim=0).values     # (H, W)

    plt.figure(figsize=(4, 4))
    plt.imshow(mag_proj.cpu().numpy(), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    train_transforms = [
        EquispacedUndersample(acceleration=args.accel, center_fraction=args.center_frac),
        Augmentation(max_rot_deg=args.rot, max_shift_px=args.shift),  # train-time pipeline
        Normalize(),
        ToCompatibleTensor(),
    ]

    # NOTE: RandomPhantomDataset expects `length=...`, not `num_samples`.
    ds = RandomPhantomDataset(length=args.num_samples, transforms=train_transforms)

    for i in range(min(3, args.num_samples)):
        chw = ds[i]  # (channels, H, W)
        out_path = os.path.join(args.out_dir, f"task1_sample_{i}.png")
        show_tensor(chw, f"Task-1 sample {i}", out_path)

    print(f"Saved preview figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
