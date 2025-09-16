# Foqus ML Task

[![CI](https://github.com/msoltanpour/foqus-ml-task/actions/workflows/ci.yml/badge.svg)](https://github.com/msoltanpour/foqus-ml-task/actions/workflows/ci.yml)

A compact PyTorch repo for the Foqus ML interview tasks:
- **Task 1:** Dataset & transforms for synthetic multi-coil MRI phantoms
- **Task 2:** Small, production-friendly CNN embedding model
- **Task 3:** Contrastive training (triplet loss) with diagnostics
- **Task 4:** Cleanup — packaging, CLI, lint/format, tests, CI

## Quickstart

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate

# Editable install (+ dev tools)
python -m pip install -U pip
pip install -e ".[dev]"

# Smoke test (CPU, 1 epoch)
foqus-train --epochs 1 --batch-size 8 --num-workers 0 --device cpu --out-dir report/figs/exp1
Artifacts are written under report/figs/exp1/:

task3_curves.png, task3_curves.csv

task3_best.pt (best checkpoint)

Package layout
bash
Copy code
foqus-ml-task/
├─ foqus_ml_task/
│  ├─ datasets.py           # datasets + triplet dataset
│  ├─ transforms.py         # Normalize, undersampling, augmentations, tensor adapters
│  ├─ model.py              # MRIEmbeddingModel (CNN → embedding)
│  ├─ train.py              # Task 3 training script (argparse inside)
│  ├─ cli.py                # console entry → foqus-train
│  └─ phantom.py            # phantom generation helpers
├─ tests/
│  └─ test_model_shapes.py  # fast unit tests for model behavior
├─ report/                  # figures, PDFs, and saved artifacts
├─ pyproject.toml           # packaging + tooling config
└─ .github/workflows/ci.yml # CI (lint + tests)
Training
foqus-train forwards to foqus_ml_task/train.py (argparse inside). Typical run:

bash
Copy code
foqus-train \
  --epochs 20 \
  --train-length 800 --val-length 200 \
  --batch-size 16 --num-workers 0 \
  --image-size 128 --n-coils 8 \
  --lr 3e-4 --margin 0.2 \
  --out-dir report/figs/exp1
During training, it prints:

triplet loss and diagnostic stats (d_pos, d_neg, viol%)

Recall@1 on validation

saves curves and best checkpoint under --out-dir

Reproducibility
Deterministic seeds for PyTorch/NumPy in train.py (seed_all)

Validation uses the same transforms as training without augmentation, offset=len(train_ds) and deterministic=True.

Dev workflow
Lint/format: pre-commit runs Ruff + Black

bash
Copy code
pre-commit install
pre-commit run -a
Tests:

bash
Copy code
pytest -q
Results (examples)
See report/figs/exp1/:

task3_curves.png — training/validation loss curves

task3_best.pt — best model checkpoint

task3_curves.csv — CSV log for quick plotting

License
MIT (or update to your preferred license).
