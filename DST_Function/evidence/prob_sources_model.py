# evidence/prob_sources_model.py
#小分类器
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from joblib import load
from evidence.constants import FRAME_STAGE

def _uniform_stage_prob() -> Dict[str, float]:
    return {k: 1.0 / len(FRAME_STAGE) for k in FRAME_STAGE}

def _to_vec(z: Dict[str, float], keys: List[str]) -> np.ndarray:
    x = np.zeros((1, len(keys)), dtype=np.float32)
    for i, k in enumerate(keys):
        if k in z:
            x[0, i] = float(z[k])
    return x

class ViewStageModel:
    """
    joblib里保存 dict：
      {"model": sklearn_pipeline, "keys": feature_keys, "classes": ["Normal","Degrading","Severe"]}
    """
    def __init__(self, joblib_path: str):
        obj = load(joblib_path)
        self.model = obj["model"]
        self.keys: List[str] = list(obj["keys"])
        self.classes: List[str] = list(obj["classes"])

    def predict_proba(self, z: Dict[str, float]) -> Dict[str, float]:
        has_any = any((k in z) for k in self.keys)
        if not has_any:
            return _uniform_stage_prob()

        x = _to_vec(z, self.keys)
        proba = self.model.predict_proba(x)[0]

        out = {c: 0.0 for c in FRAME_STAGE}
        for i, c in enumerate(self.classes):
            out[c] = float(proba[i])

        s = sum(out.values())
        if s <= 0:
            return _uniform_stage_prob()
        return {k: v / s for k, v in out.items()}

def load_view_models(
    time_model_path: str = "./models/stage_time.joblib",
    freq_model_path: str = "./models/stage_freq.joblib",
) -> Tuple[ViewStageModel, ViewStageModel]:
    return ViewStageModel(time_model_path), ViewStageModel(freq_model_path)

def build_stage_prob_sources_from_models(
    z_all: Dict[str, float],
    time_model: ViewStageModel,
    freq_model: ViewStageModel,
) -> Dict[str, Dict[str, float]]:
    p_time = time_model.predict_proba(z_all)
    p_freq = freq_model.predict_proba(z_all)
    return {"time_view": p_time, "freq_view": p_freq}
