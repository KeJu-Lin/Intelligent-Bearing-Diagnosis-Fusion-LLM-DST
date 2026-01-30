# scripts/infer_stage_fusion_only.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from evidence.text_parse import extract_zscores
from evidence.constants import FRAME_STAGE
from evidence.ds_fusion import fuse_prob_sources, format_fusion_block

#  小分类器证据源
from evidence.prob_sources_model import load_view_models, build_stage_prob_sources_from_models

def main():
    user_text = """数据集：IMS Bearings Run-to-Failure
Set No.1，窗口编号：5

特征（z-score）：
- rms_mean=0.182 (z=+2.60)
- crest_mean=4.85 (z=+2.20)
- kurt_mean=5.40 (z=+3.10)
- hf_ratio_gt5k=0.058 (z=+2.70)
- band_8_20k=0.10 (z=+3.00)
"""

    z = extract_zscores(user_text)
    time_model, freq_model = load_view_models("./DST_Function/models/stage_time.joblib", "./DST_Function/models/stage_freq.joblib")
    prob_sources = build_stage_prob_sources_from_models(z, time_model, freq_model)

    source_alpha = {"time_view": 0.85, "freq_view": 0.75}
    res = fuse_prob_sources(prob_sources, FRAME_STAGE, source_alpha, rule="dempster")

    print(format_fusion_block(res))

if __name__ == "__main__":
    main()
