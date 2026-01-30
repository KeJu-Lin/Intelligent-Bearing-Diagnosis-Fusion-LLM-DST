# evidence/text_parse.py 解析文本输入的 z-score
import re
from typing import Dict

ZLINE = re.compile(r"-\s*([a-zA-Z0-9_]+)=([0-9.eE+-]+)\s*\(z=([+-]?\d+(\.\d+)?)\)")

def extract_zscores(text: str) -> Dict[str, float]:
    """
    从文本框输入提取 z-score:
    - rms_mean=0.182 (z=+2.60)
    返回 {"rms_mean":2.6, ...}
    """
    feats = {}
    for m in ZLINE.finditer(text):
        name = m.group(1)
        z = float(m.group(3))
        feats[name] = z
    return feats
