# -*- coding: utf-8 -*-
"""
16_threshold_jevcal.py —— 阈值选择（awesome-jev `jevcal` 思路）
==============================================================

对应文档「3.3 阈值工程 / 3.4 容量规划」。别拍脑袋设阈值：
在你的标注数据上扫描阈值，在“自动处理区间准确率 ≥ 目标”的前提下，
**最小化需要升级 LLM/人工的流量**；并给出模型更新后的 CI 守门思路。

自动区间：p >= thr（自动 yes）或 p <= 1-thr（自动 no）；其余升级。

运行：pip install numpy && python 16_threshold_jevcal.py
"""
from __future__ import annotations

try:
    import numpy as np
except ImportError:
    raise SystemExit("需要 numpy：pip install numpy")


def simulate_data(n=2000, seed=7):
    """模拟一批带真值的决策概率（真实场景换成你评测集的输出）。"""
    rng = np.random.default_rng(seed)
    # 真实标签
    y = rng.integers(0, 2, n).astype(float)
    # 模型概率：正例偏高、负例偏低，但有噪声和一部分“难样本”集中在 0.5 附近
    p = np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    hard = rng.random(n) < 0.15            # 15% 难样本，概率贴近 0.5
    p[hard] = np.clip(0.5 + rng.normal(0, 0.08, hard.sum()), 0.02, 0.98)
    return p, y


def scan_threshold(p, y, target_acc=0.95, grid=None):
    grid = grid or np.round(np.arange(0.5, 0.975, 0.025), 3)
    rows = []
    for thr in grid:
        auto = (p >= thr) | (p <= 1 - thr)
        n_auto = int(auto.sum())
        if n_auto == 0:
            continue
        pred = (p[auto] >= 0.5).astype(int)
        acc = float((pred == y[auto]).mean())
        llm_flow = 1 - n_auto / len(p)
        rows.append((float(thr), acc, llm_flow, n_auto))
    feasible = [r for r in rows if r[1] >= target_acc]
    best = min(feasible, key=lambda r: r[2]) if feasible else None
    return rows, best


def main():
    p, y = simulate_data()
    target = 0.95
    rows, best = scan_threshold(p, y, target_acc=target)
    print(f"目标：自动处理区间准确率 ≥ {target:.0%}，最小化 LLM/人工 流量\n")
    print(f"{'thr':>6}{'自动区间准确率':>16}{'LLM/人工流量':>16}{'自动样本数':>12}")
    for thr, acc, flow, n in rows[::2]:
        mark = "  <== 推荐" if best and abs(thr - best[0]) < 1e-9 else ""
        print(f"{thr:>6.3f}{acc:>16.3f}{flow:>16.1%}{n:>12}{mark}")
    if best:
        thr, acc, flow, n = best
        print(f"\n推荐阈值 thr={thr:.3f}：自动区间准确率 {acc:.3f}，"
              f"{1 - flow:.1%} 流量全自动、仅 {flow:.1%} 升级 LLM/人工。")
    else:
        print("\n没有阈值能满足目标准确率，需提升模型质量或提高目标可行性。")

    print("""
CI 守门（jevcal 做法）：
  1) 把选定 thr 与评测集固化到仓库；
  2) 每次模型/基座更新后重跑本脚本；
  3) 若自动区间准确率跌破目标（如 0.95），让 CI 失败，阻止静默上线。""")


if __name__ == "__main__":
    main()
