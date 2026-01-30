# ims_to_alpaca_48k.py
# pip install numpy pandas scipy tqdm

from __future__ import annotations
from pathlib import Path
import re, json, random, zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample_poly
from scipy.stats import kurtosis

# ========= 基本参数 =========
FS_IN = 20000
FS_OUT = 48000
UP, DOWN = 12, 5                  # 20k -> 48k
RAW_POINTS = 20480                # 每文件点数（IMS）
SEG_LEN = 4096
HOP = 2048

# 基线：默认取每个Set前 5% 文件（至少 30 个文件）
BASELINE_RATIO = 0.05
BASELINE_MIN_FILES = 30

# 数据集切分：按时间顺序 70/15/15（每个Set独立切）
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15

SEED = 42
random.seed(SEED)

SYSTEM = (
    "你是工业旋转机械状态监测助手。输出必须包含四段：结论、关键证据、建议动作、不确定性与下一步。"
    "严禁编造未提供的信息。"
)
TASKS = [
    "请根据振动多通道特征摘要判断健康阶段（Normal/Degrading/Severe）并解释。",
    "你是轴承退化监测助手。依据特征相对基线的偏离，判断当前窗口处于健康/退化/严重，并给出建议。",
    "根据输入的特征与z-score，输出健康阶段与处置建议（四段式）。"
]

# ========= 工具函数 =========
def parse_timestamp(name: str):
    """
    IMS 文件名通常是：YYYY.MM.DD.HH.MM.SS
    """
    m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})", name)
    if not m:
        return None
    return tuple(map(int, m.groups()))

def infer_set_no(num_cols: int) -> int:
    """
    Set1: 8列（4轴承*2方向）
    Set2/3: 4列
    """
    if num_cols == 8:
        return 1
    if num_cols == 4:
        return 2  # 可能是Set2或Set3，数据层面我们都按 4通道处理
    return 0

def read_ascii_file(p: Path) -> np.ndarray:
    """
    兼容 Windows 编码问题：用二进制读入，再用 ASCII 忽略非法字符。
    """
    raw = p.read_bytes()

    # 关键：强制按 ASCII 解码，遇到非法字节直接丢弃
    text = raw.decode("ascii", errors="ignore")

    # np.fromstring 对“任意空白分隔”的数字很快（空格/换行/Tab都行）
    arr = np.fromstring(text, sep=" ", dtype=np.float32)

    # IMS 正常情况下：arr.size = 20480 * 通道数
    if arr.size % RAW_POINTS == 0:
        C = arr.size // RAW_POINTS
        return arr.reshape(RAW_POINTS, C)

    # 兜底：如果文件里有奇怪的分隔符或行，走慢一点的 genfromtxt
    import io
    arr2 = np.genfromtxt(io.StringIO(text), dtype=np.float32)
    if arr2.ndim == 1:
        arr2 = arr2[:, None]
    return arr2

def iter_windows(n: int):
    i = 0
    while i + SEG_LEN <= n:
        yield i, i + SEG_LEN
        i += HOP

def features_window(x48: np.ndarray, fs: int = FS_OUT) -> dict:
    """
    x48: (SEG_LEN, C)
    输出：一组“窗口级”特征（多通道取均值）
    """
    x = x48 - x48.mean(axis=0, keepdims=True)
    rms = np.sqrt((x * x).mean(axis=0) + 1e-12)
    peak = np.max(np.abs(x), axis=0) + 1e-12
    crest = peak / (rms + 1e-12)
    kurt = kurtosis(x, axis=0, fisher=False, bias=False)

    # 频域：对通道取平均功率谱
    F = np.fft.rfft(x, axis=0)
    pw = (np.abs(F) ** 2).mean(axis=1) + 1e-12
    freqs = np.fft.rfftfreq(SEG_LEN, d=1 / fs)
    tot = float(pw.sum() + 1e-12)

    centroid = float((pw * freqs).sum() / tot)
    hf_ratio = float(pw[freqs > 5000].sum() / tot)

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(pw[m].sum() / tot)

    return {
        "rms_mean": float(rms.mean()),
        "crest_mean": float(crest.mean()),
        "kurt_mean": float(np.mean(kurt)),
        "centroid_hz": centroid,
        "hf_ratio_gt5k": hf_ratio,
        "band_0_1k": band(0, 1000),
        "band_1_3k": band(1000, 3000),
        "band_3_8k": band(3000, 8000),
        "band_8_20k": band(8000, 20000),
    }

def stage_from_zmax(zmax: float) -> str:
    if zmax <= 2.0:
        return "Normal"
    if zmax <= 4.0:
        return "Degrading"
    return "Severe"

# ========= 主流程 =========
def convert_one_set(files: list[Path], out_dir: Path, set_name: str):
    """
    files：该Set目录下所有文件（已排序）
    """
    if not files:
        return

    # ---- 基线文件数量 ----
    baseline_n = max(BASELINE_MIN_FILES, int(len(files) * BASELINE_RATIO))
    baseline_n = min(baseline_n, len(files))

    # ---- 先算基线（取每个文件的所有窗口）----
    feat_list = []
    for p in tqdm(files[:baseline_n], desc=f"[{set_name}] baseline"):
        x = read_ascii_file(p)
        x48 = resample_poly(x, up=UP, down=DOWN, axis=0).astype(np.float32)

        for a, b in iter_windows(len(x48)):
            feat_list.append(features_window(x48[a:b]))

    bdf = pd.DataFrame(feat_list)
    bmean = bdf.mean()
    bstd = bdf.std() + 1e-12

    baseline = {k: {"mean": float(bmean[k]), "std": float(bstd[k])} for k in bdf.columns}
    (out_dir / f"baseline_{set_name}.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 处理全部文件 -> Alpaca 样本 ----
    rows_meta = []
    alpaca_records = {"train": [], "val": [], "test": []}
    idx_global = 0

    for i, p in enumerate(tqdm(files, desc=f"[{set_name}] process")):
        # split 按时间顺序
        r = i / max(1, (len(files) - 1))
        if r < SPLIT_TRAIN:
            split = "train"
        elif r < SPLIT_TRAIN + SPLIT_VAL:
            split = "val"
        else:
            split = "test"

        x = read_ascii_file(p)
        C = x.shape[1]
        set_no = infer_set_no(C)

        x48 = resample_poly(x, up=UP, down=DOWN, axis=0).astype(np.float32)

        for w_id, (a, b) in enumerate(iter_windows(len(x48))):
            feat = features_window(x48[a:b])
            z = {k: (feat[k] - baseline[k]["mean"]) / baseline[k]["std"] for k in feat.keys()}

            zmax = max(abs(v) for v in z.values())
            stage = stage_from_zmax(zmax)
            abnormal = "否" if stage == "Normal" else "是"

            # 证据：取 z-score 绝对值最大的 4 项
            top = sorted([(k, z[k], feat[k]) for k in feat.keys()], key=lambda t: abs(t[1]), reverse=True)[:4]
            evidence = "\n".join([f"- {k}={v:.4g}，偏离 {zz:+.2f}σ" for k, zz, v in top])

            instruction = f"{SYSTEM}\n\n任务：{random.choice(TASKS)}"
            input_text = (
                f"数据集：IMS Bearings Run-to-Failure\n"
                f"Set={set_name}（推断SetNo={set_no}，通道数={C}）\n"
                f"文件={p.name}\n"
                f"原始采样={FS_IN}Hz（20480点/文件），重采样={FS_OUT}Hz（up={UP},down={DOWN}）\n"
                f"窗口={SEG_LEN}点（{SEG_LEN/FS_OUT:.3f}s），hop={HOP}\n"
                f"窗口编号={w_id}\n\n"
                f"特征（括号内为相对该Set早期基线 z-score，单位σ）：\n"
                + "\n".join([f"- {k}={feat[k]:.6g} (z={z[k]:+.2f})" for k in feat.keys()])
            )

            output_text = (
                f"结论：\n- 健康阶段：{stage}\n- 是否异常：{abnormal}\n\n"
                f"关键证据：\n{evidence}\n\n"
                "建议动作：\n"
                "- 建议按时间序列持续监测（RMS/峭度/高频占比）并设置趋势告警。\n"
                "- 若进入 Degrading/Severe，建议加密采样并做包络谱/特征频率检查以定位故障部位。\n\n"
                "不确定性与下一步：\n"
                "- 当前判断基于单窗口特征，可能受噪声或安装条件影响。\n"
                "- 建议结合相邻时间点趋势与多通道一致性验证。"
            )

            rec = {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "id": f"ims-{set_name}-{idx_global:08d}",
            }
            alpaca_records[split].append(rec)

            rows_meta.append({
                "id": rec["id"],
                "split": split,
                "set": set_name,
                "file": p.name,
                "w_id": w_id,
                "channels": C,
                "win_start": a,
                "win_end": b,
            })

            idx_global += 1

    # 保存 jsonl
    for sp in ["train", "val", "test"]:
        with (out_dir / f"{set_name}_{sp}_alpaca.jsonl").open("w", encoding="utf-8") as f:
            for rec in alpaca_records[sp]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 合集
    all_list = alpaca_records["train"] + alpaca_records["val"] + alpaca_records["test"]
    (out_dir / f"{set_name}_alpaca_all.json").write_text(
        json.dumps(all_list, ensure_ascii=False), encoding="utf-8"
    )

    # metadata
    pd.DataFrame(rows_meta).to_csv(out_dir / f"{set_name}_segments_metadata.csv", index=False, encoding="utf-8-sig")


def convert_ims_root(input_root: str, out_dir: str):
    """
    input_root: 目录，里面包含 IMS 的时间戳文件（或包含多个子目录）
    out_dir: 输出目录
    """
    root = Path(input_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 允许 root 下有多个子目录（例如 1st_test、2nd_test、3rd_test）
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        sets = subdirs
    else:
        sets = [root]
    ts_pat = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")
    for sdir in sets:
        files = [p for p in sdir.iterdir() if p.is_file() and ts_pat.match(p.name)]
        files.sort(key=lambda p: parse_timestamp(p.name))
        convert_one_set(files, out, set_name=sdir.name)

    print("Done ->", out)


if __name__ == "__main__":
    # 示例：
    # python ims_to_alpaca_48k.py
    # 默认读取当前目录下的 IMS_DATA
    convert_ims_root(input_root="./Data/IMS/", out_dir="./Data/IMS/ims_alpaca_48k_out")
