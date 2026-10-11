"""04 · 菜品匹配 POC（Noul 判等 + Choice 归属 + Score 分级）。

场景（文档 3.2.2）：同一道菜不同写法是否等价？归属到标准菜名？相似度分级？
置信度驱动：高置信自动合并、低置信转人工、中置信上 LLM 复核。

运行:
  python 04_dish_matching.py
"""
from __future__ import annotations
import json
from systemone_client import get_client

# 标准菜名库（Choice 候选，≤255）
STANDARD_MENU = [
    "宫保鸡丁", "鱼香肉丝", "麻婆豆腐", "红烧肉", "糖醋里脊",
    "西红柿炒鸡蛋", "酸辣土豆丝", "水煮鱼", "回锅肉", "辣子鸡",
]

# 带标注的决策数据集样例（用于后续 Laya 微调 / 评测，见 08、10）
LABELED_PAIRS = [
    # (菜A, 菜B, 是否同一道菜, 备注)
    ("宫保鸡丁", "宫爆鸡丁", 1, "错别字等价"),
    ("宫保鸡丁", "辣子鸡", 0, "不同菜"),
    ("西红柿炒鸡蛋", "番茄炒蛋", 1, "别名等价"),
    ("麻婆豆腐", "麻辣豆腐", 1, "近义等价"),
    ("红烧肉", "红烧排骨", 0, "不同主料"),
    ("鱼香肉丝", "鱼香茄子", 0, "不同菜"),
]


def equivalence(client, a: str, b: str) -> dict:
    """Noul：A 与 B 是否为同一道菜。"""
    state = f"菜A：{a}；菜B：{b}。"
    q = {"type": "noul", "instructions": "A 与 B 是否为同一道菜（不同写法/别名也算）？"}
    return client.ask(state, {"eq": q})["answers"]["eq"]


def classify(client, name: str) -> dict:
    """Choice：归属到标准菜名库。"""
    criteria = {m: m for m in STANDARD_MENU}
    criteria["other"] = "以上都不是"
    q = {"type": "choice", "instructions": "这道菜最接近标准菜名里的哪一个？",
         "criteria": criteria}
    return client.ask(f"菜品：{name}", {"cls": q})["answers"]["cls"]


def route(a: str, b: str, eq: dict) -> str:
    """置信度驱动的动作。"""
    p = eq["noul"]
    if p > 0.9:
        return "AUTO_MERGE"
    if p < 0.1:
        return "AUTO_SPLIT"
    return "LLM_REVIEW"          # 中间带：上 LLM / 人工


def main():
    client = get_client()
    print("=== 判等 + 置信度路由 ===")
    for a, b, label, note in LABELED_PAIRS:
        eq = equivalence(client, a, b)
        action = route(a, b, eq)
        print(f"  {a!r:10} ~ {b!r:10}  noul={eq['noul']}  "
              f"truth={label}  action={action}  ({note})")

    print("\n=== 归属到标准菜名 ===")
    for name in ["宫爆鸡丁", "番茄炒蛋", "麻辣豆腐", "未知菜品X"]:
        c = classify(client, name)
        print(f"  {name!r:10} -> {c.get('choice')} (conf={c.get('confidence')})")


if __name__ == "__main__":
    main()
