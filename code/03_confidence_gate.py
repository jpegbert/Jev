"""03 · 置信度门控 + LLM 降级骨架。

决策模型拿不准就上 LLM / 转人工，不瞎猜。这是「用好处女座」的核心（见文档 3.3、PPT MVP 第 3 条）。

运行:
  python 03_confidence_gate.py
"""
from __future__ import annotations
import json
from systemone_client import get_client


def llm_fallback(state: str, question: dict) -> str:
    """占位：真实环境里这里调用你的 LLM / 人工队列。"""
    return f"<LLM/人工> state={state[:20]!r} q={question.get('instructions')}"


def decide(client, state: str, question: dict,
           noul_thr: float = 0.9, conf_thr: float = 0.6) -> object:
    r = client.ask(state, {"q": question})["answers"]["q"]
    if question["type"] == "noul":
        p = r["noul"]
        if p > noul_thr:
            return "yes"
        if p < 1 - noul_thr:
            return "no"
        return llm_fallback(state, question)        # 中间带 → 升级
    if r.get("confidence", 1) < conf_thr:
        return llm_fallback(state, question)        # 低置信 → 升级
    return r


def main():
    client = get_client()
    samples = [
        ("这笔订单已退款成功。", {"type": "noul", "instructions": "是否涉及退款？"}),
        ("今天天气真不错。", {"type": "noul", "instructions": "是否要求退款？"}),
        ("帮我归类：服务器 502 报错。",
         {"type": "choice", "instructions": "分给哪个团队？",
          "criteria": {"tech": "技术", "sales": "销售", "billing": "账单"}}),
    ]
    for state, q in samples:
        out = decide(client, state, q)
        print(f"state={state!r:40} -> {out}")


if __name__ == "__main__":
    main()
