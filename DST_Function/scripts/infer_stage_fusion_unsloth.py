# scripts/infer_stage_fusion_unsloth_pycharm.py
import sys
from pathlib import Path

# 让你可以从 scripts/ 里 import 到项目根目录的 evidence/*
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from unsloth import FastLanguageModel

from evidence.text_parse import extract_zscores
from evidence.constants import FRAME_STAGE
from evidence.ds_fusion import fuse_prob_sources, format_fusion_block
from evidence.prob_sources_model import load_view_models, build_stage_prob_sources_from_models


SYSTEM = """你是工业轴承故障诊断助手。
你只能基于输入的特征与z-score以及证据融合结果推理，禁止编造未提供的信息。
输出必须严格包含四段：结论、关键证据、建议动作、不确定性与下一步。
最终诊断结论以“证据融合（D-S）结果”为准（BetP最大者）。"""

ALPACA_PROMPT = """### Instruction:
{}

### Input:
{}

### Response:
{}"""


def load_unsloth_model(model_path: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def build_prompt_with_fusion(
    user_text: str,
    time_model_path: str,
    freq_model_path: str,
    source_alpha=None,
    rule: str = "dempster",
):
    z = extract_zscores(user_text)
    if len(z) == 0:
        raise ValueError("没有从输入中解析到任何 z-score 特征，请检查格式：- xxx=... (z=+1.23)")

    time_model, freq_model = load_view_models(time_model_path, freq_model_path)
    prob_sources = build_stage_prob_sources_from_models(z, time_model, freq_model)

    if source_alpha is None:
        source_alpha = {"time_view": 0.85, "freq_view": 0.75}

    res = fuse_prob_sources(prob_sources, FRAME_STAGE, source_alpha, rule=rule)
    fusion_block = format_fusion_block(res)

    prompt = f"""{user_text}

{fusion_block}

任务：
1) 最终结论以 D-S 融合结果为准（BetP最大者）
2) 输出四段式报告（结论/关键证据/建议动作/不确定性与下一步）
3) 必须引用 m(Θ)、K 或 BetP 的数值解释原因
"""
    return prompt, res


@torch.inference_mode()
def run_once_unsloth(
    model,
    tokenizer,
    user_text: str,
    time_model_path: str,
    freq_model_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
):
    user_prompt, fusion_res = build_prompt_with_fusion(
        user_text=user_text,
        time_model_path=time_model_path,
        freq_model_path=freq_model_path,
        source_alpha={"time_view": 0.85, "freq_view": 0.75},
        rule="dempster",
    )

    eos_token = tokenizer.eos_token or ""
    full_text = ALPACA_PROMPT.format(SYSTEM, user_prompt, "") + eos_token

    inputs = tokenizer([full_text], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    out = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # 只截取 Response 后面的内容（避免把 prompt 也打印出来）
    marker = "### Response:"
    if marker in out:
        out = out.split(marker, 1)[1].strip()

    return out, fusion_res


def main():
    # ======= 1) 这里改成你的路径 =======
    MODEL_PATH = "./lora_model/"  # 你的 Unsloth LoRA 合并后/或可加载的目录
    TIME_MODEL_PATH = "./DST_Function/models/stage_time.joblib"
    FREQ_MODEL_PATH = "./DST_Function/models/stage_freq.joblib"

    # ======= 2) 这里直接写你要推理的文本（PyCharm 运行最方便）=======
    USER_TEXT = """特征（z-score）：
- rms_mean=0.182 (z=+2.60)
- crest_mean=4.85 (z=+2.20)
- kurt_mean=5.40 (z=+3.10)
- hf_ratio_gt5k=0.058 (z=+2.70)
- band_8_20k=0.10 (z=+3.00)

任务：判断健康阶段（Normal/Degrading/Severe）并输出四段式报告。"""

    # ======= 3) 加载模型 =======
    model, tokenizer = load_unsloth_model(
        model_path=MODEL_PATH,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # ======= 4) 推理 =======
    report, fusion_res = run_once_unsloth(
        model=model,
        tokenizer=tokenizer,
        user_text=USER_TEXT,
        time_model_path=TIME_MODEL_PATH,
        freq_model_path=FREQ_MODEL_PATH,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
    )

    print("\n========== 模型输出（四段式） ==========\n")
    print(report)


if __name__ == "__main__":
    main()
