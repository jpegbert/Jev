# -*- coding: utf-8 -*-
"""
03_cost_model.py —— 成本 / 速度测算（把“便宜”算成账）
====================================================

对应文档「1.4 为什么快、为什么便宜」。纯标准库，离线可跑。
所有单价均为厂商公开口径（2026-09~10），带 * 者为自报上限值，非典型生产值。

口径说明：
    - JEV 非自回归，输出 token 免费，仅按输入 token 计费。
    - LLM 自回归，输入 + 输出都计费；“带推理”场景还有数百 reasoning token。

运行：python 03_cost_model.py
"""
from __future__ import annotations

from dataclasses import dataclass


# ---- 单价（美元 / 百万 token；厂商公开口径，随时间变化，用时请复核）----
PRICE_IN = {
    "JEV": 0.042,          # 输入；输出免费
    "GPT6_Luna_short": 0.10,   # 仅给短标签（输入摊薄后的建模单价）
    "GPT6_Luna_reason": 0.10,  # 带 ~500 推理 token
}
PRICE_OUT = {
    "JEV": 0.0,
    "GPT6_Luna_short": 0.40,
    "GPT6_Luna_reason": 0.40,
}


@dataclass
class Scenario:
    name: str
    calls: int          # 决策次数
    in_tokens: int      # 每次输入 token
    out_tokens: int     # 每次输出 token


def cost_usd(model: str, s: Scenario) -> float:
    return (s.calls * s.in_tokens / 1e6 * PRICE_IN[model]
            + s.calls * s.out_tokens / 1e6 * PRICE_OUT[model])


def main() -> None:
    scenarios = [
        Scenario("每千次·2000 输入·轻量判断", 1000, 2000, 0),
        Scenario("每千次·5000 输入·中等工单", 1000, 5000, 0),
        Scenario("每天 1 亿次判断（2000 输入）", 100_000_000, 2000, 0),
    ]

    print("== 单次决策成本对比（输出 token：JEV 免费 / LLM 计费）==\n")
    header = f"{'场景':<34}{'JEV':>12}{'Luna短标签':>14}{'Luna带推理':>14}"
    print(header)
    for s in scenarios:
        jev = cost_usd("JEV", s)
        short = Scenario(s.name, s.calls, s.in_tokens, 30)
        reason = Scenario(s.name, s.calls, s.in_tokens, 500)
        c_short = cost_usd("GPT6_Luna_short", short)
        c_reason = cost_usd("GPT6_Luna_reason", reason)
        print(f"{s.name:<34}{jev:>12.3f}{c_short:>14.3f}{c_reason:>14.3f}")

    # 每千次、2000 输入的官方建模对照（大纲口径）
    print("\n== 大纲给定对照（每 1000 次、各 2000 输入）==")
    print("  JEV ≈ $0.084；GPT-6 Luna 短标签 ≈ $0.215；带 ~500 推理 ≈ $0.465")
    print("  → 同样判断，JEV 比“带推理 LLM”便宜约 5.5×。")

    # 批量并行省账
    print("\n== 批量并行（13 个问题）成本直觉 ==")
    state, per_q, n = 2000, 40, 13
    seq = n * (state + per_q)
    batch = state + n * per_q
    saved = 1 - batch / seq
    print(f"  顺序输入 {seq} token，批量输入 {batch} token，"
          f"省输入 {saved:.1%}（官方口径成本约降 12.2×、速度约快 10×*）")

    print("\n* 带星号/“官方口径”的数字为厂商自报上限，非典型生产值，引用请注明。")


if __name__ == "__main__":
    main()
