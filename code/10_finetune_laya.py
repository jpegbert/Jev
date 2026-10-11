"""10 · 构造中文决策数据集（Laya RLCD 微调原料）。

对应文档 2.4 微调 / 3.2.2 菜品匹配。本脚本只负责「造数据」这一步——
把标注对转成 Laya 训练所需的 {state, questions, answers} jsonl。
真正的 RLCD 训练在官方 Kaggle Notebook / upstream 项目完成（见 10_finetune_laya.md）。

运行:
  pip install numpy
  python 10_finetune_laya.py            # 生成 dish_match.jsonl
"""
from __future__ import annotations
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def build_dish_dataset(pairs):
    """pairs: [(菜A, 菜B, 是否同一道菜 0/1), ...]"""
    rows = []
    for a, b, y in pairs:
        rows.append({
            "state": f"菜A：{a}；菜B：{b}。",
            "questions": {
                "is_same": {
                    "type": "noul",
                    "instructions": "A 与 B 是否为同一道菜（不同写法/别名也算）？",
                }
            },
            "answers": {"is_same": {"type": "noul", "noul": float(y)}},
        })
    return rows


def main():
    pairs = [
        ("宫保鸡丁", "宫爆鸡丁", 1), ("宫保鸡丁", "辣子鸡", 0),
        ("西红柿炒鸡蛋", "番茄炒蛋", 1), ("麻婆豆腐", "麻辣豆腐", 1),
        ("红烧肉", "红烧排骨", 0), ("鱼香肉丝", "鱼香茄子", 0),
        ("水煮鱼", "水煮肉片", 0), ("回锅肉", "椒盐排条", 0),
        ("糖醋里脊", "糖醋排骨", 0), ("酸辣土豆丝", "酸辣土豆片", 1),
    ]
    rows = build_dish_dataset(pairs)
    out = os.path.join(HERE, "dish_match.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已生成 {len(rows)} 条 → {out}")
    print("下一步：按 10_finetune_laya.md 把该 jsonl 送入 Laya RLCD 训练流程。")


if __name__ == "__main__":
    main()
