import argparse
import sys
from . import train as train_mod  # your existing training module

def main():
    p = argparse.ArgumentParser("Foqus ML Task")
    p.add_argument("--config", type=str, default=None,
                   help="Path to config (YAML/JSON) if supported by train.py.")
    p.add_argument("--device", type=str, default=None,
                   help="Override device (e.g., 'cuda' or 'cpu') if supported.")
    args, _ = p.parse_known_args()

    # Try common entrypoints; we’ll wire precisely after seeing train.py
    if hasattr(train_mod, "main") and callable(train_mod.main):
        return train_mod.main(cfg_path=args.config, device=args.device)
    if hasattr(train_mod, "run") and callable(train_mod.run):
        return train_mod.run()
    if hasattr(train_mod, "train") and callable(train_mod.train):
        return train_mod.train()

    print(
        "train.py doesn’t expose main()/run()/train(). "
        "Please paste the first ~60 lines of foqus_ml_task/train.py so I can wire this.",
        file=sys.stderr,
    )
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
