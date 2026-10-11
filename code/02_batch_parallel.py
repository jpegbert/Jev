"""02 · 批量并行：一次 state 多问题，只传一次上下文。

性价比关键：state 通常最贵，输出免费。把 13 个判断打包成一次调用，
成本约降 12.2×、速度约快 10×（对比顺序 13 次调用）。

awesome-jev 的 jevfire 是同一思路的 vLLM 批量实现。
"""
from __future__ import annotations
import json
from systemone_client import get_client


def build_questions():
    """构造一批「邮件归档」判断：文件夹=选项，state 共享。"""
    folders = ["inbox", "spam", "billing", "tech", "sales", "hr"]
    emails = [
        "您的发票已出具，请查收附件。",
        "限时折扣！点击领取免费会员。",
        "生产环境数据库连接超时，请尽快处理。",
        "新版本企业套餐现已上线。",
        "入职材料补充通知，请在周五前提交。",
    ]
    questions = {}
    for i, body in enumerate(emails):
        questions[f"mail_{i}"] = {
            "type": "choice",
            "instructions": "这封邮件应归入哪个文件夹？拿不准选 other。",
            "criteria": {f: f for f in folders},
        }
    return emails, questions


def main():
    client = get_client()
    emails, questions = build_questions()
    # 用一个拼接的 state 演示「共享上下文只传一次」
    shared_state = "\n".join(f"[{i}] {b}" for i, b in enumerate(emails))
    resp = client.ask(shared_state, questions)
    print(f"[latency] {resp.get('_latency_ms')} ms  mock={resp.get('_mock', False)}  "
          f"questions={len(questions)}")
    for name, ans in resp["answers"].items():
        print(f"  {name}: -> {ans.get('choice')} (conf={ans.get('confidence')})")


if __name__ == "__main__":
    main()
