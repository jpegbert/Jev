# -*- coding: utf-8 -*-
"""
make_figures.py —— 生成分享文档中的数据类配图（本地 PNG，避免网络热链失效）
============================================================================

生成：
    images/cost_comparison.png   每千次决策成本对比（JEV vs LLM 短标签/带推理）
    images/batch_savings.png     顺序 vs 批量并行的输入 token / 成本直觉

架构图、流程图、时间线等用 Mermaid 在 Markdown 中自绘（见 Jev_doubao.md）。
运行：pip install numpy matplotlib && python make_figures.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
os.makedirs(IMG_DIR, exist_ok=True)


def fig_cost():
    labels = ["JEV\n(输出免费)", "GPT-6 Luna\n短标签", "GPT-6 Luna\n带~500推理"]
    costs = [0.084, 0.215, 0.465]   # 每 1000 次、各 2000 输入，大纲给定建模口径
    colors = ["#2563EB", "#F59E0B", "#EF4444"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    bars = ax.bar(labels, costs, color=colors, width=0.55)
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.01, f"${c:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("每 1000 次决策成本（美元，建模口径）")
    ax.set_title("同样的判断，决策模型 vs 前沿 LLM 成本对比*")
    ax.set_ylim(0, max(costs) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.99, -0.18, "* 厂商公开/自报口径，非典型生产值，仅用于量级对比",
            transform=ax.transAxes, ha="right", fontsize=8, color="#666")
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "cost_comparison.png")
    fig.savefig(out, bbox_inches="tight")
    print("saved:", out)


def fig_batch():
    state, per_q = 2000, 40
    n = np.arange(1, 21)
    seq = n * (state + per_q)
    batch = state + n * per_q
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    ax.plot(n, seq, "o-", color="#EF4444", ms=4, label="顺序调用（state 重复传 N 次）")
    ax.plot(n, batch, "s-", color="#2563EB", ms=4, label="批量并行（state 只传 1 次）")
    ax.axvline(13, ls="--", color="#888", lw=1)
    ax.text(13.2, seq.max() * 0.5, "13 问：官方口径成本约降 12.2×、速度约快 10×*",
            fontsize=9, color="#444")
    ax.set_xlabel("一次决策中的问题数量")
    ax.set_ylabel("输入 token 总量（建模）")
    ax.set_title("批量并行：state 最贵、输出免费")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "batch_savings.png")
    fig.savefig(out, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    fig_cost()
    fig_batch()
    print("图片目录：", IMG_DIR)
