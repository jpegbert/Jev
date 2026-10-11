# -*- coding: utf-8 -*-
"""
01_primitives.py —— 三原语最小可运行示例（Choice / Score / Noul）
================================================================

对应分享文档「1.2 核心定位：Decisions, not strings」。
一次调用、同一个 state，并行提出三种结构化问题，返回可直接 if/else 的结果。

运行：
    python 01_primitives.py                      # 离线 Mock（无需 Key）
    SYSTEMONE_BASE_URL=http://localhost:8000 python 01_primitives.py   # 本地 Laya
    TYPESAFE_API_KEY=xxx python 01_primitives.py                       # JEV 云
"""
from __future__ import annotations

import json

from systemone_client import get_client


def main() -> None:
    client = get_client()

    # 1) state：被判断的“情境”，一次调用只传一次（最贵的部分）
    state = "用户说：我被重复扣款了，请立刻退款，这已经是本月第二次了。"

    # 2) questions：三种原语并行
    questions = {
        "refund": {  # Noul：是非判断
            "type": "noul",
            "instructions": "用户是否明确要求退款？",
        },
        "department": {  # Choice：从候选集选一个（建议保留 other 兜底）
            "type": "choice",
            "instructions": "这张工单应分给哪个团队？",
            "criteria": {
                "billing": "账单、发票、退款、重复扣款",
                "tech": "技术故障、bug、崩溃、宕机",
                "sales": "新购、续费、升级、报价",
                "other": "以上都不是",
            },
        },
        "urgency": {  # Score：放到有序量表上
            "type": "score",
            "instructions": "这件事的紧急程度？",
            "criteria": ["不急", "一般", "较急", "紧急", "非常紧急"],
        },
    }

    resp = client.ask(state, questions)
    print(f"[backend] {client.__class__.__name__}  "
          f"[latency] {resp.get('_latency_ms')} ms  "
          f"mock={resp.get('_mock', False)}")
    print(json.dumps(resp["answers"], ensure_ascii=False, indent=2))

    # 3) 业务代码直接分支——不需要再 parse 文本
    a = resp["answers"]
    p_refund = a["refund"]["noul"]
    if p_refund > 0.9 and a["department"]["choice"] == "billing" \
            and a["urgency"]["score"] >= 3:
        print(f"\n→ 自动退款流程：团队={a['department']['choice']}，"
              f"紧急度={a['urgency']['score']}，退款概率={p_refund}")
    elif p_refund < 0.1:
        print("\n→ 非退款诉求，进入普通分流。")
    else:
        print("\n→ 处于不确定区间（0.1~0.9），转人工 / 升级 LLM 复核。")


if __name__ == "__main__":
    main()
