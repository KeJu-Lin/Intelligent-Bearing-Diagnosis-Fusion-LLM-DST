# evidence/constants.py

# 证据融合的健康阶段类别空间（IMS + XJTU 最适合）
FRAME_STAGE = ["Normal", "Degrading", "Severe"]

# 时域视角（冲击/幅值相关特征）
TIME_KEYS = [
    "rms_mean", "crest_mean", "kurt_mean",
    "rms_h", "rms_v", "crest_h", "crest_v", "kurt_h", "kurt_v"
]

# 频域视角（频带能量/高频比/中心频率）
FREQ_KEYS = [
    "hf_ratio_gt5k",
    "band_0_1k", "band_1_3k", "band_3_8k", "band_8_20k",
    "centroid_hz"
]
