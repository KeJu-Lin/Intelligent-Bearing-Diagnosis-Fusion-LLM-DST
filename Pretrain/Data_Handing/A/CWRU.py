"""
CWRU Bearing Data Center -> Stanford Alpaca(jsonl) 数据集构造脚本
- 支持：Normal Baseline Data + 48k Drive End Bearing Fault Data（四类：Normal/Inner/Outer/Ball）
- 输出：alpaca_all.jsonl / train_alpaca.jsonl / val_alpaca.jsonl / test_alpaca.jsonl
- 关键点：按"文件级"划分 train/val/test，避免同一原始 .mat 切片泄漏
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.io as sio


# -----------------------------
# 1) CWRU 48k Drive-End 文件号 -> (label, load_hp, rpm, fault_diam, or_pos)
# 说明：CWRU 48k Drive-End 常用四类：Normal/Inner/Outer/Ball
# Normal Baseline: 97,98,99,100 (load 0..3)
# 48k DE Fault: 按官方表格常用编号映射（0.007/0.014/0.021/0.028）
# -----------------------------
RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def _mk_entries(nums: List[int], label: str, diam_inch: float, or_pos: Optional[str] = None) -> Dict[int, dict]:
    """
    nums: 按 load=0..3 对应的编号列表（长度 4；若某个 load 缺失可用 None 占位）
    """
    m = {}
    for load, num in enumerate(nums):
        if num is None:
            continue
        m[int(num)] = {
            "label": label,
            "load_hp": load,
            "rpm": RPM_BY_LOAD.get(load),
            "fault_diam_inch": diam_inch,
            "or_pos": or_pos,  # 外圈故障位置：'6'/'3'/'12' 或 None
        }
    return m


def build_cwru_48k_de_mapping() -> Dict[int, dict]:
    mp: Dict[int, dict] = {}

    # Normal baseline
    for load, num in enumerate([97, 98, 99, 100]):
        mp[num] = {"label": "Normal", "load_hp": load, "rpm": RPM_BY_LOAD.get(load),
                   "fault_diam_inch": 0.0, "or_pos": None}

    # 0.007"
    mp.update(_mk_entries([109, 110, 111, 112], "Inner", 0.007))
    mp.update(_mk_entries([122, 123, 124, 125], "Ball", 0.007))
    mp.update(_mk_entries([135, 136, 137, 138], "Outer", 0.007, or_pos="6"))
    mp.update(_mk_entries([148, 149, 150, 151], "Outer", 0.007, or_pos="3"))
    mp.update(_mk_entries([161, 162, 163, 164], "Outer", 0.007, or_pos="12"))

    # 0.014"
    mp.update(_mk_entries([174, 175, 176, 177], "Inner", 0.014))
    mp.update(_mk_entries([189, 190, 191, 192], "Ball", 0.014))
    mp.update(_mk_entries([201, 202, 203, 204], "Outer", 0.014, or_pos="6"))

    # 0.021"
    # 官方表中 load=3 对应 217（中间 216 可能不存在/不常用）
    mp.update(_mk_entries([213, 214, 215, 217], "Inner", 0.021))
    mp.update(_mk_entries([226, 227, 228, 229], "Ball", 0.021))
    mp.update(_mk_entries([238, 239, 240, 241], "Outer", 0.021, or_pos="6"))
    mp.update(_mk_entries([250, 251, 252, 253], "Outer", 0.021, or_pos="3"))
    mp.update(_mk_entries([262, 263, 264, 265], "Outer", 0.021, or_pos="12"))

    # 0.028"（48k DE 常见仅提供 IR/B）
    mp.update(_mk_entries([3001, 3002, 3003, 3004], "Inner", 0.028))
    mp.update(_mk_entries([3005, 3006, 3007, 3008], "Ball", 0.028))

    return mp


CWRU_48K_DE_MAP = build_cwru_48k_de_mapping()


# -----------------------------
# 2) 读取 .mat 中 Drive-End 信号（X***_DE_time）
# -----------------------------
def read_de_signal(mat_path: Path) -> np.ndarray:
    mat = sio.loadmat(str(mat_path))
    keys = [k for k in mat.keys() if not k.startswith("__")]

    # 优先匹配 *_DE_time 或包含 DE_time 的键
    cand = [k for k in keys if k.endswith("_DE_time") or ("DE_time" in k)]
    if not cand:
        raise ValueError(f"[{mat_path.name}] 未找到 drive-end 时间序列变量（*_DE_time）。可用键：{keys}")

    # 选"长度最大"的那个（一般就是 DE_time）
    best_k = None
    best_n = -1
    for k in cand:
        v = mat[k]
        if isinstance(v, np.ndarray) and v.size > best_n:
            best_k = k
            best_n = v.size

    x = mat[best_k].squeeze().astype(np.float32)
    if x.ndim != 1:
        x = x.reshape(-1).astype(np.float32)
    return x


# -----------------------------
# 3) 特征提取：时域 + 频域（与项目中常用 key 保持一致）
# -----------------------------
FEATURE_KEYS = [
    "rms_mean", "crest_mean", "kurt_mean",
    "band_0_1k", "band_1_3k", "band_3_8k", "band_8_20k",
    "hf_ratio_gt5k", "centroid_hz"
]


def time_features(seg: np.ndarray) -> Dict[str, float]:
    seg = seg.astype(np.float64)
    mu = float(seg.mean())
    sigma = float(seg.std(ddof=0)) + 1e-12
    rms = float(math.sqrt(float((seg * seg).mean())))
    peak = float(np.max(np.abs(seg)))
    crest = float(peak / (rms + 1e-12))
    kurt = float((((seg - mu) ** 4).mean()) / (sigma ** 4))
    return {"rms_mean": rms, "crest_mean": crest, "kurt_mean": kurt}


def freq_features(seg: np.ndarray, fs: int) -> Dict[str, float]:
    seg = seg.astype(np.float64)
    seg = seg - seg.mean()
    n = seg.shape[0]

    # Hann 窗降低泄漏
    w = np.hanning(n)
    segw = seg * w

    spec = np.fft.rfft(segw)
    pxx = (np.abs(spec) ** 2)  # power
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    total = float(pxx.sum()) + 1e-12

    def band_ratio(f1: float, f2: float) -> float:
        idx = (freqs >= f1) & (freqs < f2)
        return float(pxx[idx].sum() / total)

    hf_ratio = float(pxx[freqs >= 5000].sum() / total)
    centroid = float((freqs * pxx).sum() / total)

    return {
        "band_0_1k": band_ratio(0, 1000),
        "band_1_3k": band_ratio(1000, 3000),
        "band_3_8k": band_ratio(3000, 8000),
        "band_8_20k": band_ratio(8000, 20000),
        "hf_ratio_gt5k": hf_ratio,
        "centroid_hz": centroid,
    }


def extract_features(seg: np.ndarray, fs: int) -> Dict[str, float]:
    out = {}
    out.update(time_features(seg))
    out.update(freq_features(seg, fs))
    return out


# -----------------------------
# 4) 元信息推断（文件号 -> 标签/工况）
# -----------------------------
@dataclass
class FileMeta:
    mat_path: Path
    file_num: int
    label: str           # Normal/Inner/Outer/Ball
    load_hp: Optional[int]
    rpm: Optional[int]
    fault_diam_inch: float
    or_pos: Optional[str]  # '6'/'3'/'12' or None


_INT_RE = re.compile(r"(\d+)")


def infer_file_meta(mat_path: Path) -> Optional[FileMeta]:
    stem = mat_path.stem
    m = _INT_RE.search(stem)
    if not m:
        return None
    num = int(m.group(1))
    if num not in CWRU_48K_DE_MAP:
        # 不在 48k DE 的常用映射里就跳过（你也可以改成 return None 前打印提示）
        return None
    info = CWRU_48K_DE_MAP[num]
    return FileMeta(
        mat_path=mat_path,
        file_num=num,
        label=info["label"],
        load_hp=info.get("load_hp"),
        rpm=info.get("rpm"),
        fault_diam_inch=float(info.get("fault_diam_inch", 0.0)),
        or_pos=info.get("or_pos"),
    )


# -----------------------------
# 5) Alpaca 构造：instruction/input/output
# -----------------------------
INSTRUCTION = (
    "你是工业轴承故障诊断助手。"
    "你只能基于输入的特征与z-score推理，禁止编造未提供的信息。"
    "输出必须严格包含四段：结论、关键证据、建议动作、不确定性与下一步。"
)


def _fmt_pos(or_pos: Optional[str]) -> str:
    if or_pos is None:
        return ""
    # 外圈位置：6/3/12 点钟
    return f"，外圈位置≈{or_pos}:00"


def build_input_text(meta: FileMeta, win_idx: int, seg_len: int, hop: int,
                     feats: Dict[str, float], z: Dict[str, float]) -> str:
    lines = []
    lines.append("数据集：CWRU Bearing Data Center（48 kHz，Drive-End）")
    lines.append("类别：四类（Normal/Inner/Outer/Ball）")
    lines.append(f"原始文件：{meta.file_num}.mat")
    lines.append(f"工况：负载={meta.load_hp} HP，转速≈{meta.rpm} rpm{_fmt_pos(meta.or_pos)}")
    if meta.fault_diam_inch > 0:
        lines.append(f"故障直径：{meta.fault_diam_inch:.3f} inch")
    lines.append(f"窗口：{win_idx}（len={seg_len}, hop={hop}）")
    lines.append("")
    lines.append("特征（z-score）：")
    for k in FEATURE_KEYS:
        v = feats[k]
        zz = z[k]
        if "band_" in k or "ratio" in k:
            lines.append(f"- {k}={v:.6f} (z={zz:+.2f})")
        elif k == "centroid_hz":
            lines.append(f"- {k}={v:.2f} (z={zz:+.2f})")
        else:
            lines.append(f"- {k}={v:.6f} (z={zz:+.2f})")
    lines.append("")
    lines.append("任务：判断故障类型（Normal/Inner/Outer/Ball）并输出四段式报告。")
    return "\n".join(lines)


def _topk_evidence(z: Dict[str, float], k: int = 5) -> List[str]:
    items = sorted(z.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return [name for name, _ in items]


def build_output_text(label: str, feats: Dict[str, float], z: Dict[str, float]) -> str:
    # 选绝对 z 最大的若干个特征作为"证据"
    top = _topk_evidence(z, k=5)

    # 建议动作（尽量通用）
    if label == "Normal":
        action = "建议继续按既定周期采集并记录（RMS/峭度/频带能量等），用于后续趋势建模与阈值自适应更新。"
    elif label == "Inner":
        action = "建议优先检查内圈/滚道表面与润滑状况，必要时结合包络谱与特征频率（BPFI）进一步定位。"
    elif label == "Outer":
        action = "建议优先检查外圈安装座、配合间隙与载荷方向相关的磨损/点蚀，必要时结合包络谱与特征频率（BPFO）验证。"
    else:  # Ball
        action = "建议检查滚动体与保持架状态（磨损/剥落/异物），必要时结合包络谱与特征频率（BSF/FTF）进一步判别。"

    # 关键证据句子
    ev_lines = []
    for k in top:
        v = feats[k]
        zz = z[k]
        if k == "centroid_hz":
            ev_lines.append(f"- {k}={v:.2f} (z={zz:+.2f})")
        else:
            ev_lines.append(f"- {k}={v:.6f} (z={zz:+.2f})")

    return (
        "结论：\n"
        f"- 任务：判断故障类型（Normal/Inner/Outer/Ball）\n"
        f"- 最终结论：{label}\n"
        "\n"
        "关键证据：\n"
        + "\n".join(ev_lines) + "\n"
        "\n"
        "建议动作：\n"
        f"- {action}\n"
        "\n"
        "不确定性与下一步：\n"
        "- 当前结论来自单窗口统计与频域特征，可能受负载扰动、噪声或传感器安装差异影响。\n"
        "- 建议结合相邻窗口的趋势一致性、更多工况数据与更细粒度谱特征进行复核。\n"
    )


# -----------------------------
# 6) 文件级切分：train/val/test
# -----------------------------
@dataclass
class SplitSets:
    train_files: set
    val_files: set
    test_files: set


def split_by_file(records: List[FileMeta], seed: int,
                  train_ratio: float, val_ratio: float) -> SplitSets:
    rng = random.Random(seed)
    by_label: Dict[str, List[FileMeta]] = {}
    for r in records:
        by_label.setdefault(r.label, []).append(r)

    train_files, val_files, test_files = set(), set(), set()
    for label, arr in by_label.items():
        rng.shuffle(arr)
        n = len(arr)
        if n <= 1:
            train_files.add(arr[0].mat_path)
            continue
        if n == 2:
            train_files.add(arr[0].mat_path)
            test_files.add(arr[1].mat_path)
            continue
        # n >= 3
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_val > 0:
                n_val -= 1
            else:
                n_train = max(1, n_train - 1)

        train = arr[:n_train]
        val = arr[n_train:n_train + n_val]
        test = arr[n_train + n_val:]

        for r in train:
            train_files.add(r.mat_path)
        for r in val:
            val_files.add(r.mat_path)
        for r in test:
            test_files.add(r.mat_path)

    # 防御：确保互斥
    val_files -= train_files
    test_files -= train_files
    test_files -= val_files

    return SplitSets(train_files=train_files, val_files=val_files, test_files=test_files)


# -----------------------------
# 7) 主流程：扫描 -> 计算 Normal 基线统计 -> 生成 Alpaca jsonl
# -----------------------------
def iter_mat_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.mat"):
        yield p


def compute_baseline_stats(metas: List[FileMeta], fs: int,
                           seg_len: int, hop: int,
                           max_windows_per_file: Optional[int]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    用 Normal 类别的窗口统计估计 mean/std，用于 z-score
    """
    normals = [m for m in metas if m.label == "Normal"]
    if not normals:
        raise FileNotFoundError("未找到 Normal baseline 文件（通常为 97/98/99/100.mat），无法计算 z-score。")

    all_feat = {k: [] for k in FEATURE_KEYS}
    for fm in normals:
        x = read_de_signal(fm.mat_path)
        nwin = 0
        for st in range(0, len(x) - seg_len + 1, hop):
            seg = x[st:st + seg_len]
            feats = extract_features(seg, fs)
            for k in FEATURE_KEYS:
                all_feat[k].append(feats[k])
            nwin += 1
            if max_windows_per_file is not None and nwin >= max_windows_per_file:
                break

    mu = {k: float(np.mean(all_feat[k])) for k in FEATURE_KEYS}
    sigma = {k: float(np.std(all_feat[k], ddof=0) + 1e-12) for k in FEATURE_KEYS}
    return mu, sigma


def make_samples_for_file(meta: FileMeta, fs: int,
                          seg_len: int, hop: int,
                          mu: Dict[str, float], sigma: Dict[str, float],
                          max_windows_per_file: Optional[int]) -> List[dict]:
    x = read_de_signal(meta.mat_path)

    samples = []
    nwin = 0
    for win_idx, st in enumerate(range(0, len(x) - seg_len + 1, hop)):
        seg = x[st:st + seg_len]
        feats = extract_features(seg, fs)
        z = {k: (feats[k] - mu[k]) / sigma[k] for k in FEATURE_KEYS}

        item = {
            "id": f"cwru48k_{meta.file_num}_w{win_idx}",
            "instruction": INSTRUCTION,
            "input": build_input_text(meta, win_idx, seg_len, hop, feats, z),
            "output": build_output_text(meta.label, feats, z),
        }
        samples.append(item)

        nwin += 1
        if max_windows_per_file is not None and nwin >= max_windows_per_file:
            break

    return samples


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def convert_cwru(root: Path, out_dir: Path,
                 fs: int = 48000,
                 seg_len: int = 4096,
                 hop: int = 2048,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 seed: int = 42,
                 max_windows_per_file: Optional[int] = None,
                 save_stats: bool = True) -> None:
    root = root.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 扫描并识别可用文件
    metas: List[FileMeta] = []
    for p in iter_mat_files(root):
        m = infer_file_meta(p)
        if m is not None:
            metas.append(m)

    if not metas:
        raise FileNotFoundError(
            f"在 {root} 下未找到可识别的 CWRU 48k Drive-End .mat 文件。\n"
            f"请确认包含 97/98/99/100（Normal）以及 109/122/135 等编号的 .mat。"
        )

    # 2) 计算 Normal 基线统计（z-score）
    mu, sigma = compute_baseline_stats(metas, fs, seg_len, hop, max_windows_per_file)

    if save_stats:
        with (out_dir / "baseline_stats_normal.json").open("w", encoding="utf-8") as f:
            json.dump({"mean": mu, "std": sigma, "fs": fs, "seg_len": seg_len, "hop": hop},
                      f, ensure_ascii=False, indent=2)

    # 3) 文件级切分
    splits = split_by_file(metas, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)

    # 4) 生成样本
    all_rows, train_rows, val_rows, test_rows = [], [], [], []
    for meta in metas:
        rows = make_samples_for_file(meta, fs, seg_len, hop, mu, sigma, max_windows_per_file)

        all_rows.extend(rows)
        if meta.mat_path in splits.train_files:
            train_rows.extend(rows)
        elif meta.mat_path in splits.val_files:
            val_rows.extend(rows)
        elif meta.mat_path in splits.test_files:
            test_rows.extend(rows)
        else:
            # 如果某类样本太少导致未分配，默认进 train
            train_rows.extend(rows)

    # 5) 写出 jsonl
    write_jsonl(out_dir / "alpaca_all.jsonl", all_rows)
    write_jsonl(out_dir / "train_alpaca.jsonl", train_rows)
    write_jsonl(out_dir / "val_alpaca.jsonl", val_rows)
    write_jsonl(out_dir / "test_alpaca.jsonl", test_rows)

    # 6) 打印统计
    print("========== CWRU -> Alpaca 生成完成 ==========")
    print(f"root    : {root}")
    print(f"out_dir : {out_dir}")
    print(f"files   : {len(metas)} (Normal={sum(m.label=='Normal' for m in metas)}, "
          f"Inner={sum(m.label=='Inner' for m in metas)}, Outer={sum(m.label=='Outer' for m in metas)}, "
          f"Ball={sum(m.label=='Ball' for m in metas)})")
    print(f"samples : all={len(all_rows)} | train={len(train_rows)} | val={len(val_rows)} | test={len(test_rows)}")
    print(f"stats   : {out_dir / 'baseline_stats_normal.json' if save_stats else '(not saved)'}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="CWRU 数据根目录（包含 Normal Baseline Data 和 48k Drive End Bearing Fault Data）")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录（保存 jsonl）")
    ap.add_argument("--fs", type=int, default=48000, help="采样率（48k Drive-End 默认 48000）")
    ap.add_argument("--seg_len", type=int, default=4096, help="窗口长度")
    ap.add_argument("--hop", type=int, default=2048, help="滑动步长")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_windows_per_file", type=int, default=0,
                    help="每个文件最多取多少窗口（0 表示不限，用于快速调试）")
    ap.add_argument("--no_save_stats", action="store_true", help="不保存 baseline_stats_normal.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    max_w = None if args.max_windows_per_file == 0 else int(args.max_windows_per_file)
    convert_cwru(
        root=Path(args.root),
        out_dir=Path(args.out_dir),
        fs=int(args.fs),
        seg_len=int(args.seg_len),
        hop=int(args.hop),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        max_windows_per_file=max_w,
        save_stats=(not args.no_save_stats),
    )


if __name__ == "__main__":
    config = {
        "root": r"./Data/CWRU Bearing Data Center",  # 修改为你的CWRU数据路径
        "out_dir": r"./Data/CWRU Bearing Data Center/",  # 修改为输出目录
        "fs": 48000,
        "seg_len": 4096,
        "hop": 2048,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "seed": 42,
        "max_windows_per_file": None,  # None表示不限制，100表示每文件最多100个窗口
        "save_stats": True
    }
    
    # 运行转换
    convert_cwru(
        root=Path(config["root"]),
        out_dir=Path(config["out_dir"]),
        fs=config["fs"],
        seg_len=config["seg_len"],
        hop=config["hop"],
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        seed=config["seed"],
        max_windows_per_file=config["max_windows_per_file"],
        save_stats=config["save_stats"]
    )
    