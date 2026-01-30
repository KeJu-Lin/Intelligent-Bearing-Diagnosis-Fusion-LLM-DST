# scripts/train_stage_views_from_alpaca.py
# 训练两个视角小分类器：IMS+XJTU
import sys
from pathlib import Path
project_root = r"./DST_Function/"
sys.path.insert(0, project_root)
import re
import numpy as np
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from datasets import concatenate_datasets

import evidence.alpaca_loader 
from evidence.constants import FRAME_STAGE, TIME_KEYS, FREQ_KEYS

# 解析 z-score
ZLINE = re.compile(r"-\s*([a-zA-Z0-9_]+)=([0-9.eE+-]+)\s*\(z=([+-]?\d+(\.\d+)?)\)")

# 解析标签（你 Alpaca output 里必须有这行）
STAGE_LINE = re.compile(r"健康阶段：\s*(Normal|Degrading|Severe)", re.IGNORECASE)

def extract_z_from_input(text: str):
    feats = {}
    for m in ZLINE.finditer(text or ""):
        feats[m.group(1)] = float(m.group(3))
    return feats

def extract_stage_from_output(text: str):
    m = STAGE_LINE.search(text or "")
    if not m:
        return None
    s = m.group(1)
    s = s[0].upper() + s[1:].lower()
    return s if s in FRAME_STAGE else None

def to_vec(z: dict, keys: list[str]) -> np.ndarray:
    x = np.zeros((len(keys),), dtype=np.float32)
    for i, k in enumerate(keys):
        if k in z:
            x[i] = float(z[k])
    return x

def build_xy(ds_split, keys):
    X, y = [], []
    for ex in ds_split:
        z = extract_z_from_input(ex["input"])
        stage = extract_stage_from_output(ex["output"])
        if stage is None:
            continue
        X.append(to_vec(z, keys))
        y.append(FRAME_STAGE.index(stage))
    if len(X) == 0:
        raise ValueError("No training samples parsed. 请确认 output 里包含 '健康阶段：Normal/Degrading/Severe'")
    return np.stack(X), np.array(y, dtype=np.int64)

def train_one_view(train_ds, val_ds, keys, out_path: Path):
    Xtr, ytr = build_xy(train_ds, keys)
    Xva, yva = build_xy(val_ds, keys)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000,solver="lbfgs"))
    ])
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)

    print("=" * 80)
    print("VIEW KEYS:", keys)
    print(classification_report(yva, pred, target_names=FRAME_STAGE))

    dump({"model": clf, "keys": keys, "classes": FRAME_STAGE}, out_path)
    print("Saved:", out_path)

def main():
    #  改成本地路径
    ims_dir  = r"./Data/IMS/ims_alpaca_48k_out/"
    xjtu_dir = r"./Data/XJTU-SY_Bearing_Datasets/xjtu_c3_alpaca_48k_out/"

    ims = evidence.alpaca_loader.load_alpaca_any(ims_dir)
    xjtu = evidence.alpaca_loader.load_alpaca_any(xjtu_dir)

    train_all = concatenate_datasets([ims["train"], xjtu["train"]]).shuffle(seed=42)

    if "validation" in ims and "validation" in xjtu:
        val_all = concatenate_datasets([ims["validation"], xjtu["validation"]]).shuffle(seed=42)
    else:
        # 没有 val 就切一份
        n = len(train_all)
        val_n = min(2000, max(200, n // 10))
        val_all = train_all.select(range(val_n))
        train_all = train_all.select(range(val_n, n))

    out_dir = Path("./DST_Function/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_one_view(train_all, val_all, TIME_KEYS, out_dir / "stage_time.joblib")
    train_one_view(train_all, val_all, FREQ_KEYS, out_dir / "stage_freq.joblib")

if __name__ == "__main__":
    main()
