# evidence/ds_fusion.py
#证据理论核心：prob→BPA→融合→BetP→证据块
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, FrozenSet, Any

Hypothesis = FrozenSet[str]
MassFunction = Dict[Hypothesis, float]


@dataclass
class FusionResult:
    frame: FrozenSet[str]         # Θ
    m_fused: MassFunction         # fused BPA
    betp: Dict[str, float]        # Pignistic probability
    conflict_K: float             # average conflict K
    rule: str
    details: Dict[str, Any]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize_prob(p: Dict[str, float], frame: Iterable[str]) -> Dict[str, float]:
    frame = list(frame)
    out = {k: float(p.get(k, 0.0)) for k in frame}
    s = sum(out.values())
    if s <= 0:
        return {k: 1.0 / len(frame) for k in frame}
    return {k: v / s for k, v in out.items()}


def _normalize_mass(m: MassFunction) -> MassFunction:
    s = sum(float(v) for v in m.values())
    if s <= 0:
        return {}
    return {h: float(v) / s for h, v in m.items()}


def prob_to_bpa(prob: Dict[str, float], frame: Iterable[str], alpha: float = 0.85, allow_ignorance: bool = True) -> MassFunction:
    """
    概率 -> BPA（质量分配）
    m({label}) = alpha * p(label)
    m(Θ) = 1 - alpha
    """
    frame = frozenset(frame)
    alpha = _clamp(float(alpha), 0.0, 1.0)
    p = _normalize_prob(prob, frame)

    m: MassFunction = {}
    for k, v in p.items():
        m[frozenset({k})] = alpha * float(v)

    if allow_ignorance:
        m[frame] = 1.0 - alpha
    else:
        m = _normalize_mass(m)
    return m


def ds_combine(m1: MassFunction, m2: MassFunction, frame: Iterable[str], rule: str = "dempster") -> Tuple[MassFunction, float]:
    """
    融合两个 BPA
    rule:
      - dempster: 归一化(1-K)
      - yager: 冲突K给Θ（更保守）
      - dubois_prade: 冲突给并集（保守）
    """
    frame = frozenset(frame)
    rule = rule.lower().strip()

    m1 = _normalize_mass(m1)
    m2 = _normalize_mass(m2)

    m: MassFunction = {}
    K = 0.0

    for A, vA in m1.items():
        for B, vB in m2.items():
            inter = A.intersection(B)
            if len(inter) == 0:
                K += vA * vB
                if rule == "dubois_prade":
                    uni = A.union(B)
                    m[uni] = m.get(uni, 0.0) + vA * vB
            else:
                m[inter] = m.get(inter, 0.0) + vA * vB

    if rule == "dempster":
        if K >= 1.0 - 1e-12:
            return {frame: 1.0}, float(K)
        scale = 1.0 / (1.0 - K)
        m = {h: v * scale for h, v in m.items()}
        return _normalize_mass(m), float(K)

    if rule == "yager":
        m[frame] = m.get(frame, 0.0) + K
        return _normalize_mass(m), float(K)

    if rule == "dubois_prade":
        return _normalize_mass(m), float(K)

    raise ValueError(f"Unknown rule: {rule}")


def fuse_many(masses: List[MassFunction], frame: Iterable[str], rule: str = "dempster") -> Tuple[MassFunction, float]:
    """
    多证据源融合，返回 (m_fused, avgK)
    """
    if not masses:
        raise ValueError("masses is empty")
    frame = frozenset(frame)

    m = _normalize_mass(masses[0])
    Ks = []
    for i in range(1, len(masses)):
        m, K = ds_combine(m, masses[i], frame=frame, rule=rule)
        Ks.append(K)
    avgK = float(sum(Ks) / max(1, len(Ks)))
    return m, avgK


def pignistic_transform(m: MassFunction, frame: Iterable[str]) -> Dict[str, float]:
    """
    BetP：把集合质量均分到元素上，用于最终决策
    """
    frame = list(frame)
    bet = {x: 0.0 for x in frame}

    for A, v in m.items():
        if len(A) == 0:
            continue
        share = float(v) / len(A)
        for x in A:
            if x in bet:
                bet[x] += share

    s = sum(bet.values())
    if s <= 0:
        return {x: 1.0 / len(frame) for x in frame}
    return {x: v / s for x, v in bet.items()}


def fuse_prob_sources(
    prob_sources: Dict[str, Dict[str, float]],
    frame: List[str],
    source_alpha: Dict[str, float],
    rule: str = "dempster",
    allow_ignorance: bool = True
) -> FusionResult:
    """
    多证据源概率 -> BPA -> 融合 -> BetP
    """
    frame_set = frozenset(frame)
    masses = []
    debug = {"sources": {}}

    for name, p in prob_sources.items():
        alpha = float(source_alpha.get(name, 0.8))
        m = prob_to_bpa(p, frame_set, alpha=alpha, allow_ignorance=allow_ignorance)
        masses.append(m)
        debug["sources"][name] = {
            "alpha": alpha,
            "prob": dict(p),
            "bpa": {tuple(sorted(list(k))): float(v) for k, v in m.items()}
        }

    m_fused, avgK = fuse_many(masses, frame_set, rule=rule)
    betp = pignistic_transform(m_fused, frame_set)

    return FusionResult(
        frame=frame_set,
        m_fused=m_fused,
        betp=betp,
        conflict_K=avgK,
        rule=rule,
        details=debug
    )


def format_fusion_block(result: FusionResult, topk: int = 4) -> str:
    """
    生成给LLM的证据融合块（让LLM引用m(Θ)、K、BetP写解释）
    """
    def h2s(h: Hypothesis) -> str:
        if h == result.frame:
            return "Θ(不确定)"
        if len(h) == 1:
            return next(iter(h))
        return "{" + ",".join(sorted(h)) + "}"

    items = [(h, v) for h, v in result.m_fused.items() if len(h) > 0]
    items.sort(key=lambda x: x[1], reverse=True)

    m_theta = result.m_fused.get(result.frame, 0.0)
    bet_sorted = sorted(result.betp.items(), key=lambda x: x[1], reverse=True)[:topk]

    lines = []
    lines.append("证据理论融合（D-S）结果：")
    for h, v in items[:topk]:
        lines.append(f"- m({h2s(h)}) = {v:.3f}")
    lines.append(f"- m(Θ) = {m_theta:.3f}（总体不确定性）")
    lines.append(f"- 冲突度K = {result.conflict_K:.3f}（证据是否打架）")
    lines.append("用于决策的BetP（Pignistic概率）：")
    for k, v in bet_sorted:
        lines.append(f"- BetP({k}) = {v:.3f}")

    return "\n".join(lines)
