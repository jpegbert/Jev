# -*- coding: utf-8 -*-
"""
07_confidence_gate.py —— 置信度门控 + LLM/人工降级（决策模型的安全带）
=====================================================================

对应文档「3.3 工程接入」与 PPT《10 个使用场景》MVP 第 3 条：
“低于阈值一律交还人类，绝不自动动作”。

关键认知：
    - Noul 没有独立 confidence 字段，0~1 的 yes 概率本身就是信号；
      接近 0.5 表示“拿不准”，不是“否”。
    - Choice/Score 用 confidence；阈值不要拍脑袋，用数据选（见 16_threshold_jevcal.py）。

运行：python 07_confidence_gate.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from systemone_client import get_client


def llm_fallback(state: str, question: Dict[str, Any]) -> str:
    """占位：真实环境替换为 LLM 调用或人工队列。"""
    return f"<升级LLM/人工：{question.get('instructions')}>"


@dataclass
class GateConfig:
    noul_high: float = 0.90     # noul > 该值 => yes
    noul_low: float = 0.10      # noul < 该值 => no
    choice_conf: float = 0.60   # choice/score 置信度阈值
    score_conf: float = 0.60


def decide(client, state: str, question: Dict[str, Any],
           fallback: Callable = llm_fallback, cfg: GateConfig = GateConfig()):
    """返回 (动作, 载荷)。动作 ∈ {yes, no, choice, score, fallback}。"""
    r = client.ask(state, {"q": question})["answers"]["q"]
    t = question["type"]

    if t == "noul":
        p = r["noul"]
        if p >= cfg.noul_high:
            return "yes", p
        if p <= cfg.noul_low:
            return "no", p
        return "fallback", fallback(state, question)   # 中间带：不瞎猜

    if t == "choice":
        if r.get("confidence", 1) < cfg.choice_conf or r.get("choice") == "other":
            return "fallback", fallback(state, question)
        return "choice", r

    if t == "score":
        if r.get("confidence", 1) < cfg.score_conf:
            return "fallback", fallback(state, question)
        return "score", r

    return "fallback", fallback(state, question)


def main() -> None:
    client = get_client()
    samples = [
        ("这笔重复扣款我要求立刻退款。",
         {"type": "noul", "instructions": "用户是否要求退款？"}),
        ("今天天气不错，想问问营业时间。",
         {"type": "noul", "instructions": "用户是否要求退款？"}),
        ("服务器一直返回 502，订单下不了。",
         {"type": "choice", "instructions": "分给哪个团队？",
          "criteria": {"tech": "故障 bug 502", "billing": "退款 账单",
                       "sales": "购买 续费", "other": "其他"}}),
    ]
    for state, q in samples:
        action, payload = decide(client, state, q)
        print(f"  state={state[:22]!r:26} type={q['type']:<7} -> {action:<9} {payload if not isinstance(payload, dict) else payload.get('choice')}")


if __name__ == "__main__":
    main()
