# xjtu_c3_to_alpaca_48k.py
# pip install numpy pandas scipy tqdm

from __future__ import annotations
from pathlib import Path
import re, json, random, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample_poly
from scipy.stats import kurtosis

# ======== 数据集固定信息（来自官方PDF）========
FS_IN = 25600        # 原始采样率 25.6 kHz
FS_OUT = 48000       # 目标采样率 48 kHz
UP, DOWN = 15, 8     # 25.6k -> 48k : 48000/25600 = 15/8
RAW_POINTS = 32768   # 每个CSV 32768点（约1.28s）

SEG_LEN = 4096
HOP = 2048

# 基线：每个 bearing 取最早 BASELINE_RATIO 的文件（至少 BASELINE_MIN_FILES）
BASELINE_RATIO = 0.05
BASELINE_MIN_FILES = 30

# 你这次指定的 bearings：3_1~3_5
# 默认 split：train=3_1~3_3, val=3_4, test=3_5（避免同轴承泄漏）
DEFAULT_SPLIT = {
    "Bearing3_1": "train",
    "Bearing3_2": "train",
    "Bearing3_3": "train",
    "Bearing3_4": "val",
    "Bearing3_5": "test",
}

# Condition 3 最终故障元素（来自表2）
FAULT_ELEMENT = {
    "Bearing3_1": "Outer race",
    "Bearing3_2": "Inner race, ball, cage and outer race",
    "Bearing3_3": "Inner race",
    "Bearing3_4": "Inner race",
    "Bearing3_5": "Outer race",
}

SYSTEM = (
    "你是工业旋转机械状态监测助手。输出必须包含四段：结论、关键证据、建议动作、不确定性与下一步。"
    "严禁编造未提供的信息。"
)
TASKS = [
    "请根据输入的水平/垂直振动特征摘要判断健康阶段（Normal/Degrading/Severe）并解释。",
    "你是轴承退化监测助手。依据特征相对基线的偏离，判断当前窗口处于健康/退化/严重，并给出建议。",
    "根据输入的特征与z-score，输出健康阶段与处置建议（四段式）。",
]

random.seed(42)

# ======== 读取 CSV（兼容有无表头）========
def read_xjtu_csv(p: Path) -> np.ndarray:
    """
    CSV 两列：水平、垂直振动
    有的文件首行是表头：Horizontal_vibration_signals, Vertical_vibration_signals
    """
    first = p.open("r", encoding="utf-8", errors="ignore").readline().strip()
    skip = 1 if re.search(r"[A-Za-z]", first) else 0

    df = pd.read_csv(p, header=None, skiprows=skip)
    arr = df.values.astype(np.float32)

    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] > 2:
        arr = arr[:, :2]

    # sanity
    if arr.shape[0] != RAW_POINTS:
        # 不强制报错：有些用户文件可能被裁剪过
        pass
    return arr  # (N,2)

# ======== 文件名解析 ========
# 兼容两种：
# 1) Bearing3_1_1.csv（你这次上传的形式）
# 2) Bearing3_1/1.csv（官方常见目录形式）
pat_flat = re.compile(r"^(Bearing\d+_\d+)_([0-9]+)\.csv$", re.IGNORECASE)

def infer_bearing_and_index(p: Path):
    name = p.name
    m = pat_flat.match(name)
    if m:
        return m.group(1), int(m.group(2))
    # fallback：目录名作为 bearing，文件名数字作为 index
    bearing = p.parent.name
    idx_m = re.match(r"^([0-9]+)\.csv$", name)
    idx = int(idx_m.group(1)) if idx_m else 999999
    return bearing, idx

def list_all_csvs(root: Path):
    files = [p for p in root.rglob("*.csv") if p.is_file() and p.stat().st_size > 0]
    items = []
    for p in files:
        bearing, idx = infer_bearing_and_index(p)
        items.append((bearing, idx, p))
    items.sort(key=lambda t: (t[0], t[1]))
    return items

# ======== 特征提取（与前面 IMS 逻辑一致：时域 + 频域）========
freqs = np.fft.rfftfreq(SEG_LEN, d=1 / FS_OUT).astype(np.float32)
bands = [
    ("band_0_1k", 0, 1000),
    ("band_1_3k", 1000, 3000),
    ("band_3_8k", 3000, 8000),
    ("band_8_20k", 8000, 20000),
]

def feats_window(x48: np.ndarray) -> dict:
    """
    x48: (SEG_LEN,2)  -> 水平/垂直
    """
    x = x48 - x48.mean(axis=0, keepdims=True)

    rms = np.sqrt((x * x).mean(axis=0) + 1e-12)
    peak = np.max(np.abs(x), axis=0) + 1e-12
    crest = peak / (rms + 1e-12)
    k = kurtosis(x, axis=0, fisher=False, bias=False)

    F = np.fft.rfft(x, axis=0)
    pw = (np.abs(F) ** 2).mean(axis=1) + 1e-12
    tot = float(pw.sum() + 1e-12)

    centroid = float((pw * freqs).sum() / tot)
    hf_ratio = float(pw[freqs > 5000].sum() / tot)

    out = {
        "rms_h": float(rms[0]),
        "rms_v": float(rms[1]),
        "rms_mean": float(rms.mean()),
        "crest_h": float(crest[0]),
        "crest_v": float(crest[1]),
        "crest_mean": float(crest.mean()),
        "kurt_h": float(k[0]),
        "kurt_v": float(k[1]),
        "kurt_mean": float(np.mean(k)),
        "centroid_hz": centroid,
        "hf_ratio_gt5k": hf_ratio,
    }
    for name, lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(pw[m].sum() / tot)
    return out

def iter_windows(n: int):
    i = 0
    while i + SEG_LEN <= n:
        yield i, i + SEG_LEN
        i += HOP

def stage_from_zmax(zmax: float) -> str:
    if zmax <= 2.0:
        return "Normal"
    if zmax <= 4.0:
        return "Degrading"
    return "Severe"

# ======== 主转换逻辑 ========
def build_baseline_for_bearing(files: list[Path]) -> dict:
    """
    每个 bearing 用最早的一段数据做基线（默认前5%，至少30个文件）
    """
    if not files:
        return {}

    baseline_n = max(BASELINE_MIN_FILES, int(len(files) * BASELINE_RATIO))
    baseline_n = min(baseline_n, len(files))

    feat_list = []
    for p in tqdm(files[:baseline_n], desc=f"[baseline] {files[0].parent.name}"):
        x = read_xjtu_csv(p)  # (32768,2)
        x48 = resample_poly(x, up=UP, down=DOWN, axis=0).astype(np.float32)  # (61440,2)

        for a, b in iter_windows(len(x48)):
            feat_list.append(feats_window(x48[a:b]))

    bdf = pd.DataFrame(feat_list)
    mu = bdf.mean()
    sd = bdf.std() + 1e-12
    return {k: {"mean": float(mu[k]), "std": float(sd[k])} for k in bdf.columns}

def convert(root: str, out_dir: str):
    root = Path(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) 收集所有 csv，并按 bearing 分组
    items = list_all_csvs(root)
    by_bearing = {}
    for bearing, idx, p in items:
        by_bearing.setdefault(bearing, []).append((idx, p))

    # 只处理 Bearing3_*（如果目录里还有其它 Condition）
    by_bearing = {b: v for b, v in by_bearing.items() if b.lower().startswith("bearing3_")}

    # 2) 每个 bearing 建基线
    baseline_by_bearing = {}
    for bearing, pairs in by_bearing.items():
        pairs.sort(key=lambda t: t[0])
        files = [p for _, p in pairs]
        baseline_by_bearing[bearing] = build_baseline_for_bearing(files)

    (out / "baseline_by_bearing.json").write_text(
        json.dumps(baseline_by_bearing, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3) 逐文件提取窗口 -> Alpaca
    alpaca = {"train": [], "val": [], "test": []}
    meta_rows = []
    global_id = 0

    feat_cols = list(next(iter(baseline_by_bearing.values())).keys())

    for bearing, pairs in by_bearing.items():
        pairs.sort(key=lambda t: t[0])
        split = DEFAULT_SPLIT.get(bearing, "train")
        baseline = baseline_by_bearing[bearing]

        for idx, p in tqdm(pairs, desc=f"[convert] {bearing}"):
            x = read_xjtu_csv(p)
            x48 = resample_poly(x, up=UP, down=DOWN, axis=0).astype(np.float32)

            for w_id, (a, b) in enumerate(iter_windows(len(x48))):
                f = feats_window(x48[a:b])
                z = {k: (f[k] - baseline[k]["mean"]) / baseline[k]["std"] for k in feat_cols}

                zmax = max(abs(v) for v in z.values())
                stage = stage_from_zmax(zmax)

                top = sorted([(k, z[k], f[k]) for k in feat_cols], key=lambda t: abs(t[1]), reverse=True)[:5]
                evidence = "\n".join([f"- {k}={val:.4g}，偏离 {zz:+.2f}σ" for k, zz, val in top])

                instruction = f"{SYSTEM}\n\n任务：{random.choice(TASKS)}"
                input_text = (
                    "数据集：XJTU-SY Bearing Datasets\n"
                    "工况：Condition 3（2400 rpm / 40 Hz，10 kN）\n"
                    f"采样：原始 {FS_IN} Hz（32768点/文件，约1.28s），重采样到 {FS_OUT} Hz（up={UP},down={DOWN}）\n"
                    f"窗口：{SEG_LEN}点（{SEG_LEN/FS_OUT:.3f}s），hop={HOP}\n"
                    f"轴承：{bearing}；文件：{p.name}；窗口编号：{w_id}\n\n"
                    "特征（括号内为相对该轴承早期基线 z-score，单位σ）：\n"
                    + "\n".join([f"- {k}={f[k]:.6g} (z={z[k]:+.2f})" for k in feat_cols])
                )

                output_text = (
                    "结论：\n"
                    f"- 健康阶段：{stage}\n"
                    f"- 是否异常：{'否' if stage == 'Normal' else '是'}\n"
                    f"- 该轴承最终故障元素（数据集标注）：{FAULT_ELEMENT.get(bearing, 'Unknown')}\n\n"
                    f"关键证据：\n{evidence}\n\n"
                    "建议动作：\n"
                    "- 以1分钟采样周期持续追踪多窗口趋势；重点关注 RMS/峭度/高频能量比。\n"
                    "- 若进入 Degrading/Severe：建议提高采样频率或缩短告警周期，并做包络谱/特征频率诊断。\n\n"
                    "不确定性与下一步：\n"
                    "- 当前判断仅基于单窗口特征摘要，可能受噪声与负载扰动影响。\n"
                    "- 建议结合相邻采样点的趋势与水平/垂直通道一致性验证。"
                )

                rec = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                    "id": f"xjtu-c3-{bearing}-{global_id:08d}",
                }
                alpaca[split].append(rec)

                meta_rows.append({
                    "id": rec["id"],
                    "split": split,
                    "bearing": bearing,
                    "file": p.name,
                    "sample_index": int(idx),
                    "w_id": int(w_id),
                    "win_start": int(a),
                    "win_end": int(b),
                })

                global_id += 1

    # 4) 写出文件
    for sp in ["train", "val", "test"]:
        with (out / f"{sp}_alpaca.jsonl").open("w", encoding="utf-8") as f:
            for rec in alpaca[sp]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    all_list = alpaca["train"] + alpaca["val"] + alpaca["test"]
    (out / "alpaca_all.json").write_text(json.dumps(all_list, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(meta_rows).to_csv(out / "segments_metadata.csv", index=False, encoding="utf-8-sig")

    print("Done ->", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Condition3 根目录（包含 Bearing3_1~3_5）")
    ap.add_argument("--out", type=str, default="./xjtu_c3_alpaca_48k_out", help="输出目录")
    args = ap.parse_args()
    convert(args.root, args.out)

if __name__ == "__main__":
    convert(root="./Data/XJTU-SY_Bearing_Datasets/40Hz10kN/", out_dir="./Data/XJTU-SY_Bearing_Datasets/xjtu_c3_alpaca_48k_out")
