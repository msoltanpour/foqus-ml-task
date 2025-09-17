# Foqus ML Task — Contrastive MRI (k-space)

[![CI](https://github.com/msoltanpour/foqus-ml-task/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/msoltanpour/foqus-ml-task/actions/workflows/ci.yml)




Production-ready repository for the Foqus ML engineering interview tasks.
Implements a complete pipeline for MRI phantom data preparation, embedding learning, and contrastive training.

---

## 📌 Features
- **Dataset generation** with undersampling, augmentation, normalization, and tensor conversion
- **Compact CNN embedding model** for complex multi-coil MRI inputs
- **Triplet-loss contrastive training** with diagnostics and validation metrics
- **CLI entrypoint (`foqus-train`)** for reproducible training runs
- **Pre-commit, linting, and unit tests** for production quality
- **Runbook** for setup, smoke tests, and troubleshooting

---

## 📂 Repository Structure
```text
.
├── foqus_ml_task
│   ├── __init__.py
│   ├── cli.py                 # CLI wrapper (foqus-train)
│   ├── datasets.py            # Phantom dataset + triplet dataset
│   ├── model.py               # CNN embedding model
│   ├── phantom.py             # Phantom MRI generator
│   ├── train.py               # Training script (triplet loss)
│   ├── transforms.py          # Preprocessing transforms
│   └── utils/                 # Helper functions
├── preview_task1_dataset.py   # Dataset preview script
├── pyproject.toml             # Build system + project metadata
├── requirements.txt           # Runtime dependencies
├── requirements-dev.txt       # Dev dependencies
├── instructions.pdf           # Original task instructions
├── README.md                  # Project documentation
└── tmp_run/                   # Example training artifacts
    ├── task3_best.pt
    ├── task3_curves.csv
    └── task3_curves.png
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/msoltanpour/foqus-ml-task.git
cd foqus-ml-task
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3. Install dependencies
```bash
# For runtime
pip install -r requirements.txt

# For development (tests, linting, pre-commit)
pip install -e ".[dev]"
```

---

## 🚀 Usage

### Dataset Preview (Task 1)
Generate and visualize sample phantom MRIs:
```bash
python preview_task1_dataset.py --num-samples 4 --out-dir report/figs
```

### Training (Task 3)
Run a full training job (CPU example):
```bash
foqus-train \
  --epochs 20 \
  --train-length 800 --val-length 200 \
  --batch-size 16 --num-workers 0 \
  --image-size 128 --n-coils 8 \
  --lr 3e-4 --margin 0.2 \
  --out-dir report/exp1
```

---

## 📊 Results & Artifacts

Each training run produces:
- **`task3_curves.png`** — loss curves
- **`task3_curves.csv`** — per-epoch loss log
- **`task3_best.pt`** — best checkpoint (PyTorch state dict)

Example outputs (from `tmp_run/`):
```
tmp_run/
├── task3_best.pt
├── task3_curves.csv
└── task3_curves.png
```

---

## ✅ Testing & CI

Run all tests:
```bash
pytest -q
```

Format and lint:
```bash
pre-commit run -a
```

CI runs **Ruff**, **Black**, and **pytest** on every push (see `.github/workflows/ci.yml`).

---

## 🔧 Troubleshooting
- `ModuleNotFoundError: foqus_ml_task` → Ensure you installed in editable mode:
  `pip install -e ".[dev]"` inside the venv.
- CUDA not available → Use `--device cpu` or install a CUDA build of PyTorch.
- Pre-commit fixes code automatically. Run until no changes remain.

---

## 📜 License
This project is provided for interview evaluation purposes only.



