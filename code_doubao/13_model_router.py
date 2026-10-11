# -*- coding: utf-8 -*-
"""
13_model_router.py —— 模型 / 工具路由（PPT 场景 04，成本优化）
============================================================

每个请求先用决策模型判断该走「便宜 / 均衡 / 旗舰」哪一档；
弱模型搞不定时再升级。把“反射动作”和“深度思考”拆开，是 agent
成本结构的关键拐点。

PPT 实测*：12 条 prompt 经路由后 9 条不需要旗舰模型，成本约降 70%。
（* 原作者实测，仅作示例，非通用保证值。）

本文件同时给出一个**确定性的成本账本**，用三档单价测算节省。

运行：python 13_model_router.py
"""
from __future__ import annotations

from dataclasses import dataclass

from systemone_client import get_client

# 三档模型的每百万 token 建模成本（输入+输出粗估，仅用于算账）
TIER_PRICE = {"cheap": 0.10, "balanced": 0.60, "flagship": 3.00}


@dataclass
class Routed:
    prompt: str
    tier: str
    confidence: float
    est_tokens: int


def route_one(client, prompt: str) -> Routed:
    q = {"type": "choice",
         "instructions": "完成该请求需要多强的模型？按任务难度选择档位。",
         "criteria": {
             "cheap": "简单分类、改写、事实性常识、格式转换",
             "balanced": "一般问答、常规代码、多步但直接的任务",
             "flagship": "复杂推理、疑难 debug、长链路规划、高风险决策",
         }}
    r = client.ask(prompt, {"tier": q})["answers"]["tier"]
    est = max(200, len(prompt) * 2)
    return Routed(prompt, r["choice"], r["confidence"], est)


def cost_of(routed: list[Routed], price=TIER_PRICE) -> float:
    return sum(r.est_tokens / 1e6 * price[r.tier] for r in routed)


def main() -> None:
    client = get_client()
    prompts = [
        "把这句话翻译成英文：今天开会。",
        "给这段文字挑错别字。",
        "帮我写一个正则匹配手机号。",
        "总结这篇 2000 字文章的三个要点。",
        "解释一下什么是闭包，并举个例子。",
        "排查这个分布式系统偶发死锁的根因并给出修复方案。",
        "为高并发交易系统设计一套限流+降级+对账方案。",
        "把客户姓名按拼音首字母排序。",
        "判断这条评论是正面还是负面。",
        "从这段堆栈定位空指针并修改代码。",
        "写一首关于春天的短诗。",
        "推导这个概率题并给出严格证明。",
    ]
    routed = [route_one(client, p) for p in prompts]
    print("== 路由结果 ==")
    for r in routed:
        print(f"  [{r.tier:<8}] conf={r.confidence:<5} {r.prompt}")

    smart = cost_of(routed)
    all_flag = sum(r.est_tokens / 1e6 * TIER_PRICE["flagship"] for r in routed)
    all_cheap = sum(r.est_tokens / 1e6 * TIER_PRICE["cheap"] for r in routed)
    n_flag = sum(r.tier == "flagship" for r in routed)
    print(f"\n== 成本账本（建模）==")
    print(f"  全部走旗舰       : ${all_flag:.4f}")
    print(f"  路由后混合档位   : ${smart:.4f}（仅 {n_flag}/{len(routed)} 条走旗舰）")
    print(f"  相比全旗舰节省   : {1 - smart / all_flag:.0%}")
    print(f"  （对照：全走最便宜档 ${all_cheap:.4f}，但难任务质量无保障）")


if __name__ == "__main__":
    main()
