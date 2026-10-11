# -*- coding: utf-8 -*-
"""
12_context_compression.py —— 编码 Agent 的上下文压缩（PPT 场景 05）
=================================================================

长会话里，老的工具调用 / 观察结果在“早就帮不上忙”之后仍在吃上下文窗口。
用 Noul 逐条判断历史片段「对当前任务是否仍有用」，保留 / 摘要 / 删除。

PPT 实测*：agent 记忆检索从约 13000 token 压到约 2756 token，单次约 $0.003。
（* 原作者实测，引用请注明，非通用保证值。）

运行：python 12_context_compression.py
"""
from __future__ import annotations

from dataclasses import dataclass

from systemone_client import get_client


@dataclass
class HistoryItem:
    role: str
    content: str
    tokens: int


def approx_tokens(text: str) -> int:
    # 粗估：中文按字数、英文按 4 字符/token，这里统一用 max(1, len//2) 演示
    return max(1, len(text) // 2)


def compress(client, current_goal: str, history: list[HistoryItem],
             keep_thr=0.7):
    """对每条历史并行问一个 Noul：对完成当前目标是否仍有用。"""
    questions = {}
    for i, item in enumerate(history):
        questions[f"h{i}"] = {
            "type": "noul",
            "instructions": f"该历史片段对完成当前目标「{current_goal}」是否仍然有用、需要保留？",
        }
    state = f"当前目标：{current_goal}\n" + "\n".join(
        f"[h{i}] ({it.role}) {it.content}" for i, it in enumerate(history))
    ans = client.ask(state, questions)["answers"]

    kept, dropped = [], []
    for i, it in enumerate(history):
        p = ans[f"h{i}"]["noul"]
        if p >= keep_thr:
            kept.append((it, p))
        else:
            dropped.append((it, p))
    return kept, dropped


def main() -> None:
    client = get_client()
    goal = "修复支付接口的 502 错误"
    history = [
        HistoryItem("tool", "GET /health 返回 200，所有服务正常。", 24),
        HistoryItem("tool", "支付服务日志：上游超时导致 502，错误集中在 db 连接池。", 60),
        HistoryItem("user", "顺便问下今天北京天气怎么样？", 18),
        HistoryItem("tool", "三年前的一次无关部署记录。", 30),
        HistoryItem("tool", "db 连接池配置 max_connections=10，当前已打满。", 40),
        HistoryItem("assistant", "已确认根因：连接池过小，调大到 50 并加重试。", 45),
    ]
    kept, dropped = compress(client, goal, history)
    before = sum(h.tokens for h in history)
    after = sum(h.tokens for h, _ in kept)
    print(f"当前目标：{goal}")
    print(f"\n保留 {len(kept)} 条：")
    for it, p in kept:
        print(f"  keep p={p:<5} ({it.role}) {it.content}")
    print(f"\n压缩 {len(dropped)} 条：")
    for it, p in dropped:
        print(f"  drop p={p:<5} ({it.role}) {it.content}")
    print(f"\n上下文 token：约 {before} -> {after}（下降 {1 - after / before:.0%}）。")


if __name__ == "__main__":
    main()
