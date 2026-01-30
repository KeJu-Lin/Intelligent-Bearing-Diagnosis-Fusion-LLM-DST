from datasets import load_dataset, DatasetDict
from pathlib import Path
def load_alpaca_any(root_dir: str, verbose: bool = True) -> DatasetDict:
    """
    通用 Alpaca(jsonl) 加载器：
    - 递归查找所有 *alpaca*.jsonl
    - 自动按 train / val(validation) / test 分类
    - 每个 split 支持多个文件，自动合并
    适配：IMS(1st/2nd/3rd)、CWRU、XJTU 等
    """
    root = Path(root_dir)

    # 只匹配 Alpaca jsonl，避免 baseline.json / metadata.csv 被读进去
    train_files = sorted(set(root.rglob("*train*alpaca*.jsonl")))
    val_files   = sorted(set(root.rglob("*val*alpaca*.jsonl"))) \
                + sorted(set(root.rglob("*valid*alpaca*.jsonl"))) \
                + sorted(set(root.rglob("*validation*alpaca*.jsonl")))
    test_files  = sorted(set(root.rglob("*test*alpaca*.jsonl")))

    # 去重（val_files 可能重复）
    val_files = sorted(set(val_files))

    if verbose:
        print(f"\n[load_alpaca_any] root = {root.resolve()}")
        print(f"  train files: {len(train_files)}")
        for p in train_files[:5]: print("   -", p)
        if len(train_files) > 5: print("   ...")
        print(f"  val files:   {len(val_files)}")
        for p in val_files[:5]: print("   -", p)
        if len(val_files) > 5: print("   ...")
        print(f"  test files:  {len(test_files)}")
        for p in test_files[:5]: print("   -", p)
        if len(test_files) > 5: print("   ...")

    if len(train_files) == 0 and len(val_files) == 0 and len(test_files) == 0:
        raise FileNotFoundError(
            f"在 {root.resolve()} 下没有找到任何 *alpaca*.jsonl 文件。\n"
            f"请确认你的文件名包含 train/val/test + alpaca + .jsonl"
        )

    data_files = {}
    if train_files: data_files["train"] = [str(p) for p in train_files]
    if val_files:   data_files["validation"] = [str(p) for p in val_files]
    if test_files:  data_files["test"] = [str(p) for p in test_files]

    ds = load_dataset("json", data_files=data_files)

    # 保险：确保字段齐全
    need = {"instruction", "input", "output"}
    for split in ds.keys():
        cols = set(ds[split].column_names)
        if not need.issubset(cols):
            raise ValueError(
                f"{split} split 字段不完整！当前列={cols}\n"
            )

    return ds