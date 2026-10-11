# -*- coding: utf-8 -*-
"""
11_agent_guardrail.py —— Agent 工具安全门 + 完成验证
===================================================

对应文档「3.2.4 Agent」与 PPT 场景 09「智能体护栏」。
核心思想：护栏不应依赖“模型自觉”，而应收口成**可配置的概率阈值**。

两类门：
    1) 调用前 gate：危险/越权工具执行前，用 Noul 判“是否安全且被授权”，否则阻断；
    2) 完成后 verify：子任务结束用 Noul 判“目标是否达成”。

运行：python 11_agent_guardrail.py
"""
from __future__ import annotations

from systemone_client import get_client

SAFE_THR = 0.80


class Guardrail:
    def __init__(self, client, safe_thr=SAFE_THR):
        self.client = client
        self.safe_thr = safe_thr
        self.log = []

    def can_call(self, ctx: str, tool: str, action: str) -> tuple[bool, float]:
        state = f"上下文：{ctx}\n即将调用工具：{tool}\n动作：{action}"
        q = {"type": "noul",
             "instructions": "在当前上下文下，该动作是否安全、合规且已被用户授权执行？"}
        p = self.client.ask(state, {"safe": q})["answers"]["safe"]["noul"]
        ok = p >= self.safe_thr
        self.log.append((tool, action, p, ok))
        return ok, p

    def is_done(self, ctx: str, goal: str) -> tuple[bool, float]:
        state = f"上下文：{ctx}\n目标：{goal}\n当前结果：{ctx}"
        q = {"type": "noul", "instructions": "目标是否已经真正达成？"}
        p = self.client.ask(state, {"done": q})["answers"]["done"]["noul"]
        return p >= self.safe_thr, p


def execute_with_guardrail(g: Guardrail, ctx, tool, action, goal):
    ok, p = g.can_call(ctx, tool, action)
    if not ok:
        print(f"  🛑 阻断 {tool}.{action}（safe_p={p} < {g.safe_thr}）→ 转人工确认")
        return False
    print(f"  ✅ 放行 {tool}.{action}（safe_p={p}）→ 执行")
    done, pd = g.is_done(ctx, goal)
    print(f"     完成验证 done_p={pd} -> {'达成' if done else '未达成，重试/上报'}")
    return done


def main() -> None:
    client = get_client()
    g = Guardrail(client)

    print("== 正常：删除测试环境临时文件 ==")
    execute_with_guardrail(g, "用户请求删除测试环境的临时日志。",
                           "fs", "rm temp.log", "临时文件已删除")

    print("\n== 危险：删除生产数据库（应被阻断）==")
    execute_with_guardrail(g, "用户请求删除生产数据库。",
                           "db", "drop production", "生产库被删除")

    print("\n== 越权：未经确认的大额转账（应被阻断）==")
    execute_with_guardrail(g, "对话中用户只问了余额，并未要求转账。",
                           "pay", "transfer 50000", "完成转账")


if __name__ == "__main__":
    main()
