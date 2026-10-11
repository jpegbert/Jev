# -*- coding: utf-8 -*-
"""
02_batch_parallel.py —— 批量并行：一次 state、多个问题
=====================================================

对应文档「1.3.3 硬件感知并行采样」与「1.4 为什么便宜」。
state 通常最贵、输出免费：把 N 个判断打包进一次调用，
通过 packing + tree attention mask 在一次前向里给所有候选打分，
避免把同一段上下文重复传 N 遍。

官方口径*：13 个问题一次调用 vs 顺序 13 次，成本约降 12.2×、速度约快 10×。
（* 厂商自报上限值，非典型生产值。）

运行：python 02_batch_parallel.py
"""
from __future__ import annotations

import time

from systemone_client import get_client

FOLDERS = ["inbox", "spam", "billing", "tech", "sales", "hr"]

EMAILS = [
    "您的发票已出具，请查收附件。",
    "限时免费领取会员！点击链接立即激活。",
    "生产环境数据库连接超时，订单服务 502，请尽快处理。",
    "新版本企业套餐上线，需要我给您报价吗？",
    "请于周五前补充入职材料并回传身份证复印件。",
    "我上周申请的退款至今没有到账，请帮忙催一下。",
]


def build_questions(n: int):
    """每封邮件一个 choice 问题，候选集相同（文件夹）。"""
    qs = {}
    for i in range(n):
        qs[f"mail_{i}"] = {
            "type": "choice",
            "instructions": "这封邮件应自动归入哪个文件夹？拿不准选 other。",
            "criteria": {f: f for f in FOLDERS} | {"other": "无法判断"},
        }
    return qs


def run_batch(client, emails):
    """一次调用：所有邮件拼成一个共享 state，问题并行。"""
    shared_state = "\n".join(f"[{i}] {b}" for i, b in enumerate(emails))
    questions = build_questions(len(emails))
    t0 = time.time()
    resp = client.ask(shared_state, questions)
    elapsed = (time.time() - t0) * 1000
    return resp, elapsed


def run_sequential(client, emails):
    """顺序调用：每封邮件单独传一次 state（反例，用于对比）。"""
    questions = build_questions(1)
    t0 = time.time()
    results = {}
    for i, body in enumerate(emails):
        r = client.ask(body, {"mail_0": questions["mail_0"]})
        results[f"mail_{i}"] = r["answers"]["mail_0"]
    elapsed = (time.time() - t0) * 1000
    return results, elapsed


def main() -> None:
    client = get_client()
    print(f"[backend] {client.__class__.__name__}\n")

    resp, t_batch = run_batch(client, EMAILS)
    print(f"== 批量并行（1 次调用，{len(EMAILS)} 个问题）==")
    for i, body in enumerate(EMAILS):
        ans = resp["answers"][f"mail_{i}"]
        print(f"  [{i}] -> {ans['choice']:<8} conf={ans['confidence']:<5} {body}")
    print(f"  耗时(含 Mock 处理): {t_batch:.2f} ms\n")

    _, t_seq = run_sequential(client, EMAILS)
    print(f"== 顺序调用（{len(EMAILS)} 次调用）==")
    print(f"  耗时(含 Mock 处理): {t_seq:.2f} ms")
    if t_batch > 0:
        print(f"  本地 Mock 墙钟倍速 ≈ {t_seq / max(t_batch, 1e-6):.1f}× "
              f"（真实模型上官方口径约 10×，且 state 只传一次，省输入 token）")

    # token 口径的成本直觉（与墙钟无关，按“输入重复传 N 次”建模）
    state_tokens = 2000
    per_q_tokens = 40
    n = len(EMAILS)
    seq_tokens = n * (state_tokens + per_q_tokens)
    batch_tokens = state_tokens + n * per_q_tokens
    print(f"\n== 输入 token 建模（state={state_tokens}, 每问题={per_q_tokens}）==")
    print(f"  顺序 {seq_tokens} vs 批量 {batch_tokens}，"
          f"输入放大约 {seq_tokens / batch_tokens:.1f}×")


if __name__ == "__main__":
    main()
