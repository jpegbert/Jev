# -*- coding: utf-8 -*-
"""
08_dish_matching.py —— 菜品匹配 POC（拉美/中文菜品归一）
=======================================================

对应文档「3.2.2 菜品匹配（天然适配）」。把跨平台菜品匹配拆成三类决策：

    Noul   ：is A == B？同一道菜的不同写法 / 别名 / 错别字（核心）
    Choice ：归属到哪个标准菜名（候选 ≤255，带 other 兜底）
    Score  ：相似度 / 置信分级（分数可落在两级之间）

置信度驱动分流：
    高置信 → 自动合并 / 自动归一
    低置信 → 自动判新 / 拒绝
    中置信 → 升级 LLM 复核或转人工标注

运行：python 08_dish_matching.py
"""
from __future__ import annotations

from dataclasses import dataclass

from systemone_client import get_client

STANDARD_MENU = [
    "宫保鸡丁", "鱼香肉丝", "麻婆豆腐", "红烧肉", "糖醋里脊",
    "西红柿炒鸡蛋", "酸辣土豆丝", "水煮鱼", "回锅肉", "辣子鸡",
]

# (平台A写法, 平台B写法, 真值, 备注)
LABELED_PAIRS = [
    ("宫保鸡丁", "宫爆鸡丁", 1, "错别字等价"),
    ("西红柿炒鸡蛋", "番茄炒蛋", 1, "别名等价"),
    ("麻婆豆腐", "麻辣豆腐", 1, "近义等价"),
    ("酸辣土豆丝", "酸辣土豆片", 1, "刀工/写法差异"),
    ("宫保鸡丁", "辣子鸡", 0, "不同菜"),
    ("红烧肉", "红烧排骨", 0, "主料不同"),
    ("鱼香肉丝", "鱼香茄子", 0, "主料不同"),
    ("水煮鱼", "水煮肉片", 0, "主料不同"),
]


@dataclass
class DishDecision:
    a: str
    b: str
    p_equal: float
    action: str
    truth: int


class DishMatcher:
    def __init__(self, client, high=0.85, low=0.15):
        self.client = client
        self.high = high
        self.low = low

    def is_equal(self, a: str, b: str) -> float:
        state = f"菜A：{a}；菜B：{b}。"
        q = {"type": "noul",
             "instructions": "菜A 与 菜B 是否为同一道菜（不同写法、错别字、别名都算）？"}
        return self.client.ask(state, {"eq": q})["answers"]["eq"]["noul"]

    def similarity_level(self, a: str, b: str) -> float:
        state = f"菜A：{a}；菜B：{b}。"
        q = {"type": "score",
             "instructions": "两道菜的相似程度？",
             "criteria": ["完全不同", "略有相似", "比较相似", "高度相似", "几乎相同"]}
        return self.client.ask(state, {"sim": q})["answers"]["sim"]["score"]

    def classify(self, name: str) -> dict:
        criteria = {m: m for m in STANDARD_MENU}
        criteria["other"] = "标准库中没有这道菜"
        q = {"type": "choice",
             "instructions": "这道菜最接近标准菜名库中的哪一个？都不符合选 other。",
             "criteria": criteria}
        return self.client.ask(f"用户下单菜品：{name}", {"cls": q})["answers"]["cls"]

    def route(self, p: float) -> str:
        if p >= self.high:
            return "AUTO_MERGE"      # 自动合并
        if p <= self.low:
            return "AUTO_NEW"        # 自动判为不同 / 新菜
        return "LLM_REVIEW"          # 中间带升级

    def match(self, a: str, b: str, truth: int = None) -> DishDecision:
        p = self.is_equal(a, b)
        return DishDecision(a, b, p, self.route(p), truth)


def main() -> None:
    client = get_client()
    dm = DishMatcher(client)

    print("== ① 判等 Noul + 置信度分流 ==")
    correct = 0
    n_auto = 0
    n_review = 0
    for a, b, y, note in LABELED_PAIRS:
        d = dm.match(a, b, y)
        if d.action == "LLM_REVIEW":
            flag = "·升级"
            n_review += 1
        else:
            n_auto += 1
            pred = d.action == "AUTO_MERGE"
            if pred == bool(y):
                correct += 1
                flag = "✓"
            else:
                flag = "✗"
        print(f"  {flag} {a:<8} ~ {b:<8} p={d.p_equal:<6} -> {d.action:<11} "
              f"真值={y} ({note})")
    print(f"  自动判定 {n_auto} 条，正确率 {correct}/{n_auto}；"
          f"其余 {n_review} 条进入 LLM/人工复核（不计错，也不自动动作）")

    print("\n== ② 相似度分级 Score ==")
    for a, b in [("宫保鸡丁", "宫爆鸡丁"), ("宫保鸡丁", "辣子鸡"),
                 ("红烧肉", "红烧排骨")]:
        print(f"  {a} ~ {b} -> 相似度分 {dm.similarity_level(a, b)}")

    print("\n== ③ 归属标准菜名 Choice（带 other）==")
    for name in ["宫爆鸡丁", "番茄炒蛋", "麻辣豆腐", "巴西烤肉汉堡(未知)"]:
        c = dm.classify(name)
        print(f"  {name:<16} -> {c['choice']:<8} conf={c['confidence']}")


if __name__ == "__main__":
    main()
