"""08 · 中文评测集 + 四维指标（准确率 / 底线率 / Brier / ECE）。

对应文档 3.4（cookbook 第 6 章诚实评测纪律）。指标纯 numpy 实现，离线可跑。
真实使用时把 `synthetic_predictions` 换成对你的模型/client 的循环调用即可。

运行:
  pip install numpy
  python 08_eval_chinese.py
"""
from __future__ import annotations
import numpy as np

# ---- 一个最小中文评测集（菜品判等）：样例，真实应更大且带标注 ----
EVAL = [
    # (菜A, 菜B, 是否同一道菜 0/1)
    ("宫保鸡丁", "宫爆鸡丁", 1),
    ("宫保鸡丁", "辣子鸡", 0),
    ("西红柿炒鸡蛋", "番茄炒蛋", 1),
    ("麻婆豆腐", "麻辣豆腐", 1),
    ("红烧肉", "红烧排骨", 0),
    ("鱼香肉丝", "鱼香茄子", 0),
    ("水煮鱼", "水煮肉片", 0),
    ("回锅肉", "椒盐排条", 0),
]


def synthetic_predictions(eval_set):
    """离线伪预测：用简单规则模拟一个「还行但不完美、且校准尚可」的模型。

    返回 (probs, labels)：probs[i] 为「同一道菜」的 yes 概率。
    """
    rng = np.random.default_rng(42)
    probs, labels = [], []
    for a, b, y in eval_set:
        # 相同字越多越可能是同一道菜；加噪声模拟不完美
        overlap = len(set(a) & set(b)) / max(1.0, len(set(a) | set(b)))
        p = 0.15 + 0.7 * overlap + rng.normal(0, 0.08)
        probs.append(float(np.clip(p, 0.01, 0.99)))
        labels.append(float(y))
    return np.array(probs), np.array(labels)


def accuracy(probs, labels, thr=0.5):
    return float((((probs >= thr).astype(int)) == labels).mean())


def baseline_rate(probs, labels, thr=0.5, band=(0.1, 0.9)):
    """底线率 = 被迫降级/转人工的比例（落在「模棱两可」band 内）。

    这是 cookbook 第 6 章「底线」轴的直观近似：模型不强行自动动作的比例。
    """
    return float(((probs >= band[0]) & (probs <= band[1])).mean())


def brier_score(probs, labels):
    return float(np.mean((probs - labels) ** 2))


def ece(probs, labels, n_bins=10):
    """Expected Calibration Error：预测置信度与真实准确率的差距。"""
    bins = np.linspace(0, 1, n_bins + 1)
    conf = np.abs(probs - 0.5) * 2          # 以「偏离 0.5 的程度」作为置信度
    pred = (probs >= 0.5).astype(int)
    correct = (pred == labels).astype(int)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) & (conf > 0)
        if m.sum() == 0:
            continue
        acc = correct[m].mean()
        avg_conf = conf[m].mean()
        ece_val += m.sum() / len(probs) * abs(acc - avg_conf)
    return float(ece_val)


def main():
    probs, labels = synthetic_predictions(EVAL)
    print("中文菜品判等评测（样例集，离线伪预测）")
    print(f"  样本数        : {len(labels)}")
    print(f"  准确率 Accuracy: {accuracy(probs, labels):.3f}")
    print(f"  底线率 Baseline: {baseline_rate(probs, labels):.3f}")
    print(f"  Brier         : {brier_score(probs, labels):.3f}")
    print(f"  ECE           : {ece(probs, labels):.3f}")
    print("\n注：真实评测请复用 cookbook 第 6 章「无重试无回退、预算记账」纪律，")
    print("    四项齐报（准确率+底线+Brier+ECE），并在你的中文集上画 reliability diagram。")


if __name__ == "__main__":
    main()
