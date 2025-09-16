from . import train as train_mod


def main():
    # Let train.py own argparse and execution
    if hasattr(train_mod, "main") and callable(train_mod.main):
        return train_mod.main()
    # Fallbacks if someone renames entry later
    for fn in ("run", "train"):
        if hasattr(train_mod, fn) and callable(getattr(train_mod, fn)):
            return getattr(train_mod, fn)()
    raise SystemExit(
        "No entrypoint found in foqus_ml_task/train.py (expected main()/run()/train())."
    )


if __name__ == "__main__":
    raise SystemExit(main())
