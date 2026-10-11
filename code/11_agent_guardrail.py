"""11 · Agent 工具安全门 + 完成验证（对应文档 3.2.4 / PPT 场景 09）。

在调用「危险工具」前用 Noul 判意图/权限，危险即阻断；
子任务结束后用 Noul 验证是否达成。护栏收口成可配置概率阈值，而非依赖模型自觉。

运行:
  python 11_agent_guardrail.py
"""
from __future__ import annotations
from systemone_client import get_client


def allowed(client, ctx: str, tool: str, action: str) -> bool:
    """调用前风险门控：Noul 判「该动作在当前上下文是否安全/被授权」。"""
    state = f"上下文：{ctx}\n即将调用的工具：{tool}\n动作：{action}"
    q = {"type": "noul", "instructions": "该动作是否安全且被授权执行？"}
    p = client.ask(state, {"safe": q})["answers"]["safe"]["noul"]
    SAFE_THR = 0.8
    decision = p > SAFE_THR
    print(f"  [门控] {tool}.{action}  safe_p={p}  -> {'放行' if decision else '阻断'}")
    return decision


def verified(client, ctx: str, goal: str) -> bool:
    """完成后验证：Noul 判「目标是否已达成」。"""
    state = f"上下文：{ctx}\n目标：{goal}"
    q = {"type": "noul", "instructions": "目标是否已经达成？"}
    p = client.ask(state, {"done": q})["answers"]["done"]["noul"]
    return p > 0.8


def main():
    client = get_client()
    ctx = "用户请求删除测试环境的临时文件。"
    if allowed(client, ctx, tool="fs", action="rm temp.log"):
        print("  (执行) fs.rm temp.log")
        ok = verified(client, ctx, goal="临时文件已删除")
        print(f"  [验证] 目标达成={ok}")

    ctx2 = "用户请求删除生产数据库。"
    allowed(client, ctx2, tool="db", action="drop production")   # 应阻断


if __name__ == "__main__":
    main()
