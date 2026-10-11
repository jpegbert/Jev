# -*- coding: utf-8 -*-
"""
10_high_cardinality.py —— 高基数 Choice 的“两阶段”做法
=====================================================

对应文档「1.2.2 Choice 高基数技巧」。Choice 候选上限 255，
一次性在很大候选集上 softmax 会被稀释。社区做法是两阶段：

    阶段一（Score/独立打分）：对每个候选独立打分，不相互竞争；
    阶段二（Choice）：只在 top-k 候选里做一次显式 choice。

本文件用“标准菜名库”演示：先用相似度/打分从大库召回 top-k，
再在小集合上 choice，规模可远超 255（阶段一不受 choice 上限约束）。

运行：python 10_high_cardinality.py
"""
from __future__ import annotations

from systemone_client import get_client, _char_overlap

# 构造一个较大的标准菜名库（演示用，真实可达数千条；先用向量/BM25 粗召回）
BIG_MENU = [
    "宫保鸡丁", "宫保虾球", "宫保豆腐", "鱼香肉丝", "鱼香茄子", "鱼香烘蛋",
    "麻婆豆腐", "麻辣豆腐", "红烧肉", "红烧排骨", "红烧鱼块", "糖醋里脊",
    "糖醋排骨", "糖醋鲤鱼", "西红柿炒鸡蛋", "番茄炒蛋", "酸辣土豆丝",
    "水煮鱼", "水煮肉片", "水煮牛肉", "回锅肉", "辣子鸡", "口水鸡",
    "白切鸡", "黄焖鸡米饭", "东坡肉", "梅菜扣肉", "蒜蓉西兰花", "干煸豆角",
]


def stage1_recall(name: str, menu: list[str], k: int = 5) -> list[tuple[str, float]]:
    """阶段一：独立给每个候选打分（这里用字符重叠模拟；真实用向量/BM25/模型 score）。"""
    scored = [(m, _char_overlap(name, m)) for m in menu]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def stage2_choice(client, name: str, candidates: list[str]) -> dict:
    """阶段二：仅在 top-k 上做显式 choice（k 远小于 255，分布不被稀释）。"""
    criteria = {c: c for c in candidates}
    criteria["other"] = "top-k 里仍没有正确菜名"
    q = {"type": "choice",
         "instructions": "在给出的候选标准菜名中，用户菜品最匹配哪一个？",
         "criteria": criteria}
    return client.ask(f"用户下单菜品：{name}", {"cls": q})["answers"]["cls"]


def main() -> None:
    client = get_client()
    for name in ["宫爆鸡丁", "番茄炒蛋", "麻辣豆腐", "黄焖鸡"]:
        topk = stage1_recall(name, BIG_MENU, k=5)
        cand = [m for m, _ in topk]
        decision = stage2_choice(client, name, cand)
        print(f"用户菜品：{name}")
        print(f"  阶段一 top-k: {cand}")
        print(f"  阶段二 choice -> {decision['choice']} "
              f"(conf={decision['confidence']})\n")


if __name__ == "__main__":
    main()
