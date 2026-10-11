"""01 · 三原语最小调用示例（Choice / Score / Noul）。

演示一次调用同时问多个原语，并打印结构化响应。
无 API Key 时自动走 Mock（见 systemone_client.get_client）。

运行:
  python 01_basic_primitives.py                 # 离线 Mock
  SYSTEMONE_BASE_URL=http://localhost:8000 python 01_basic_primitives.py   # 本地 Laya
  TYPESAFE_API_KEY=xxx python 01_basic_primitives.py                       # JEV 云
"""
from __future__ import annotations
import json
from systemone_client import get_client


def main():
    client = get_client()
    state = "用户说：我被重复扣款了，请立刻退款，这已经第二次了。"
    questions = {
        "refund": {
            "type": "noul",
            "instructions": "用户是否要求退款？",
        },
        "department": {
            "type": "choice",
            "instructions": "该分给哪个团队处理？",
            "criteria": {
                "billing": "账单、退款、扣款、发票",
                "tech": "技术故障、bug、崩溃",
                "sales": "销售、购买、续费",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "这件事的紧急程度？",
            "criteria": ["不急", "一般", "紧急", "非常紧急"],
        },
    }

    resp = client.ask(state, questions)
    print(f"[latency] {resp.get('_latency_ms')} ms  mock={resp.get('_mock', False)}")
    print(json.dumps(resp["answers"], ensure_ascii=False, indent=2))

    # 程序化分支示例
    a = resp["answers"]
    if a["refund"]["noul"] > 0.8:
        print(f"→ 退款诉求确认，路由至 {a['department']['choice']} 团队，"
              f"紧急度分数 {a['urgency']['score']}")


if __name__ == "__main__":
    main()
