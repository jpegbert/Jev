# -*- coding: utf-8 -*-
"""
15_eval.py —— 中文评测集 + 四维指标（准确率 / 底线率 / Brier / ECE）
==================================================================

对应文档「3.4 评测纪律」与 Datawhale jev-cookbook 第 6 章的诚实评测纪律：
无重试、无回退、预算记账；**不只看准确率，还要看校准（ECE / Brier）**。

四维含义（Noul 二分类，yes 概率 p）：
    准确率 Accuracy : 以 0.5 为界判对的比例
    底线率 Baseline : 落在“模棱两可”区间 [0.1,0.9]、需要降级/转人工的比例
                      （模型不强行自动动作的比例；越高越保守、越安全）
    Brier           : 概率预测的均方误差，越小越好（综合校准+分辨）
    ECE             : 期望校准误差，分桶后 |平均置信度 - 桶内真实准确率|，越小越校准

真实使用：把 synthetic_predictions 换成对你的 client 的循环调用即可。
运行：pip install numpy matplotlib && python 15_eval.py
"""
from __future__ import annotations

try:
    import numpy as np
except ImportError:
    raise SystemExit("评测脚本需要 numpy：pip install numpy")


# 最小中文菜品判等评测集（示例；真实应数百~数千条，覆盖错别字/别名/主料差异）
EVAL_SET = [
    ("宫保鸡丁", "宫爆鸡丁", 1), ("西红柿炒鸡蛋", "番茄炒蛋", 1),
    ("麻婆豆腐", "麻辣豆腐", 1), ("酸辣土豆丝", "酸辣土豆片", 1),
    ("宫保鸡丁", "宫保虾球", 0), ("宫保鸡丁", "辣子鸡", 0),
    ("红烧肉", "红烧排骨", 0), ("鱼香肉丝", "鱼香茄子", 0),
    ("水煮鱼", "水煮肉片", 0), ("糖醋里脊", "糖醋排骨", 0),
    ("回锅肉", "盐煎肉", 1), ("白切鸡", "口水鸡", 0),
    ("番茄炒蛋", "西红柿炒鸡蛋", 1), ("黄焖鸡", "黄焖鸡米饭", 0),
]


def synthetic_predictions(n=600, overconfident=False, seed=42):
    """用“潜在正确概率 s”模拟决策模型的 yes 概率，便于干净地展示校准。

    - s ~ Uniform：该样本真实为正例（同一道菜）的概率；标签 y ~ Bernoulli(s)；
    - 校准模型：p ≈ s（加小噪声），说 0.7 时真实约 70%；
    - 过度自信模型：把 s 往 0/1 锐化（sigmoid），判断准确率相近，却系统性地
      “把没把握说成很确定”，ECE 显著变大——这正是 LLM 概率不可门控的原因。
    """
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < s).astype(float)
    if overconfident:
        p = 1 / (1 + np.exp(-9 * (s - 0.5)))     # 锐化 → 过度自信
        p = np.clip(p + rng.normal(0, 0.03, n), 0.01, 0.99)
    else:
        p = np.clip(s + rng.normal(0, 0.05, n), 0.01, 0.99)
    return np.array(p), y


def accuracy(p, y, thr=0.5):
    return float(((p >= thr).astype(int) == y).mean())


def baseline_rate(p, band=(0.1, 0.9)):
    return float(((p >= band[0]) & (p <= band[1])).mean())


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def reliability_bins(p, y, n_bins=10):
    """返回每个置信度桶的 (平均预测概率, 真实正例率, 样本数)。"""
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum():
            bins.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return bins


def ece(p, y, n_bins=10):
    """对 *yes 概率* 直接分桶（标准二分类校准定义）。"""
    edges = np.linspace(0, 1, n_bins + 1)
    total = len(p)
    e = 0.0
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum():
            e += m.sum() / total * abs(p[m].mean() - y[m].mean())
    return float(e)


def report(name, p, y):
    print(f"--- {name} ---")
    print(f"  样本 N={len(y)}")
    print(f"  准确率 Accuracy = {accuracy(p, y):.3f}")
    print(f"  底线率 Baseline = {baseline_rate(p):.3f}（需降级/转人工比例）")
    print(f"  Brier           = {brier(p, y):.3f}（越小越好）")
    print(f"  ECE             = {ece(p, y):.3f}（越小越校准）")
    bins = reliability_bins(p, y)
    print("  可靠性分桶 (预测概率 -> 真实正例率, n):")
    for mp, fy, n in bins:
        print(f"    {mp:.2f} -> {fy:.2f}  (n={n})")
    print()


def maybe_plot(p1, y1, p2, y2, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过校准曲线图生成。")
        return
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=140)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="完美校准")
    for p, y, style, label in [(p1, y1, "o-", "决策模型(校准)"),
                               (p2, y2, "s-", "LLM式(过度自信)")]:
        bins = reliability_bins(p, y)
        xs = [b[0] for b in bins]
        ys = [b[1] for b in bins]
        ax.plot(xs, ys, style, lw=2, ms=6, label=f"{label} ECE={ece(p, y):.2f}")
    ax.set_xlabel("预测概率（说“是同一道菜”的概率）")
    ax.set_ylabel("真实正例率（确实是同一道菜的比例）")
    ax.set_title("可靠性曲线 Reliability Diagram")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"校准曲线图已保存：{out_path}")


def main():
    p1, y1 = synthetic_predictions(overconfident=False)
    p2, y2 = synthetic_predictions(overconfident=True)
    report("决策模型（校准良好）", p1, y1)
    report("LLM 式（过度自信，对照）", p2, y2)
    print(f"注：上方为 {len(y1)} 条模拟样本，用于讲清指标；真实评测请把")
    print("    synthetic_predictions 换成对 client 的循环调用，评测集模板见 EVAL_SET。")
    import os
    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
    os.makedirs(img_dir, exist_ok=True)
    maybe_plot(p1, y1, p2, y2, os.path.join(img_dir, "calibration_reliability.png"))
    print("\n纪律：四项齐报、无重试无回退、预算记账；在自己的中文集上 A/B，勿直接引用厂商榜单。")


if __name__ == "__main__":
    main()
