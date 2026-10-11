"""09 · 阈值选择（jevcal 思路）：选满足准确率目标的阈值，并算 LLM 流量。

对应文档 3.3 / awesome-jev jevcal。给定 yes 概率与真值，
扫描阈值：自动决定区间 = p>thr 或 p<1-thr；其余转 LLM/人工。
目标：在自动区间_accuracy >= target 的前提下，让 LLM 流量最小。

运行:
  pip install numpy
  python 09_threshold_selection.py
"""
from __future__ import annotations
import numpy as np


def pick_threshold(probs, labels, target_acc=0.95):
    best, best_flow = None, 1.0
    for thr in np.round(np.arange(0.5, 0.96, 0.05), 2):
        auto = (probs >= thr) | (probs <= 1 - thr)
        if auto.sum() == 0:
            continue
        acc_auto = ((probs[auto] >= 0.5).astype(int) == labels[auto]).mean()
        flow = 1 - auto.mean()                 # 转 LLM/人工 的比例
        if acc_auto >= target_acc and flow < best_flow:
            best, best_flow = thr, flow
    return best, best_flow


def main():
    rng = np.random.default_rng(0)
    probs = np.clip(rng.normal(0.5, 0.25, 400), 0.01, 0.99)
    labels = (probs + rng.normal(0, 0.12, 400) >= 0.5).astype(int)
    thr, flow = pick_threshold(probs, labels, target_acc=0.95)
    print(f"在自动区间准确率>=0.95 目标下：")
    print(f"  推荐阈值 thr = {thr}")
    print(f"  LLM/人工 流量 = {flow:.1%}（其余 {1-flow:.1%} 全自动）")
    print("\n建议：把 thr 写进配置，模型更新后重跑本脚本；若 acc_auto 跌破目标，")
    print("      让 CI 失败（jevcal 的「模型更新悄悄破坏时让 CI 失败」）。")


if __name__ == "__main__":
    main()
