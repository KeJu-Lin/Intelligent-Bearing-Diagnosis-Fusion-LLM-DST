# evidence/alpaca_loader.py
from pathlib import Path
from datasets import load_dataset, DatasetDict

def load_alpaca_any(root_dir: str, verbose: bool = True) -> DatasetDict:
    """
    递归查找并加载所有 *alpaca*.jsonl，
    自动合并为 DatasetDict(train/validation/test)
    """
    root = Path(root_dir)

    train_files = sorted(set(root.rglob("*train*alpaca*.jsonl")))
    val_files = sorted(set(root.rglob("*val*alpaca*.jsonl"))) \
              + sorted(set(root.rglob("*valid*alpaca*.jsonl"))) \
              + sorted(set(root.rglob("*validation*alpaca*.jsonl")))
    test_files = sorted(set(root.rglob("*test*alpaca*.jsonl")))

    val_files = sorted(set(val_files))

    if verbose:
        print(f"\n[load_alpaca_any] root = {root.resolve()}")
        print(f"  train: {len(train_files)}")
        print(f"  val:   {len(val_files)}")
        print(f"  test:  {len(test_files)}")

    if not train_files and not val_files and not test_files:
        raise FileNotFoundError(f"No *alpaca*.jsonl found under {root.resolve()}")

    data_files = {}
    if train_files: data_files["train"] = [str(p) for p in train_files]
    if val_files: data_files["validation"] = [str(p) for p in val_files]
    if test_files: data_files["test"] = [str(p) for p in test_files]

    ds = load_dataset("json", data_files=data_files)

    need = {"instruction", "input", "output"}
    for split in ds.keys():
        cols = set(ds[split].column_names)
        if not need.issubset(cols):
            raise ValueError(
                f"[{split}] columns mismatch: {cols}\n"
            )
    return ds
