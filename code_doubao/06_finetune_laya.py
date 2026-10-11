# -*- coding: utf-8 -*-
"""
06_finetune_laya.py —— 构造领域决策数据集（Laya RLCD 微调原料）
==============================================================

对应文档「2.4.5 微调：把 Laya 变成我们的判断模型」。
本脚本负责“造数据”这一步：把业务标注（如菜品判等、搜索相关性）
转成 Laya / System One 训练所需的 {state, questions, answers} jsonl。

真正的 RLCD 训练 + 温度拟合 + 评估 + 推送 Hub 在官方 Kaggle Notebook
（免费 2×T4，约 4~5 小时）或 upstream 训练仓库完成，详见 06_finetune_laya.md。

运行：python 06_finetune_laya.py
产物：dish_match.jsonl / search_relevance.jsonl
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- 菜品判等（Noul）+ 标准菜名归属（Choice）----------
DISH_PAIRS = [
    # (用户写法, 标准菜名, 是否同一道菜 0/1)
    ("宫爆鸡丁", "宫保鸡丁", 1),
    ("宫保鸡丁", "辣子鸡", 0),
    ("番茄炒蛋", "西红柿炒鸡蛋", 1),
    ("西红柿炒鸡蛋", "番茄炒鸡蛋", 1),
    ("麻辣豆腐", "麻婆豆腐", 1),
    ("红烧肉", "红烧排骨", 0),
    ("鱼香肉丝", "鱼香茄子", 0),
    ("水煮鱼", "水煮肉片", 0),
    ("酸辣土豆片", "酸辣土豆丝", 1),
    ("糖醋排骨", "糖醋里脊", 0),
]

STANDARD_MENU = ["宫保鸡丁", "鱼香肉丝", "麻婆豆腐", "红烧肉", "糖醋里脊",
                 "西红柿炒鸡蛋", "酸辣土豆丝", "水煮鱼", "回锅肉", "辣子鸡"]


def build_dish_dataset(pairs):
    rows = []
    for raw, std, y in pairs:
        criteria = {m: m for m in STANDARD_MENU}
        criteria["other"] = "标准库中没有对应菜"
        rows.append({
            "state": f"菜A：{raw}；菜B：{std}。用户下单菜品：{raw}。",
            "questions": {
                "is_same": {
                    "type": "noul",
                    "instructions": "菜A 与 菜B 是否为同一道菜（错别字、别名也算）？",
                },
                "std_name": {
                    "type": "choice",
                    "instructions": "用户下单菜品应归到哪个标准菜名？",
                    "criteria": criteria,
                },
            },
            "answers": {
                "is_same": {"type": "noul", "noul": float(y)},
                "std_name": {"type": "choice",
                             "choice": std if y == 1 or std in STANDARD_MENU
                             else "other"},
            },
        })
    return rows


# ---------- 搜索相关性（Score，0~3 档）----------
SEARCH_ROWS = [
    # (query, doc, 相关档 0~3)
    ("如何做宫保鸡丁", "宫保鸡丁的家常做法与配料表。", 3),
    ("如何做宫保鸡丁", "宫保鸡丁的历史起源与口味特点。", 2),
    ("如何做宫保鸡丁", "鱼香肉丝的正宗做法。", 1),
    ("如何做宫保鸡丁", "今日 A 股行情综述与板块分析。", 0),
    ("退款多久到账", "退款将在 1-3 个工作日原路退回。", 3),
    ("退款多久到账", "如何修改账户绑定手机号。", 0),
]


def build_search_dataset(rows):
    levels = ["完全无关", "弱相关", "相关", "高度相关"]
    out = []
    for q, d, y in rows:
        out.append({
            "state": f"用户查询：{q}\n候选文档：{d}",
            "questions": {
                "relevance": {
                    "type": "score",
                    "instructions": "候选文档对用户查询的相关程度？",
                    "criteria": levels,
                }
            },
            "answers": {
                "relevance": {"type": "score", "score": float(y)},
            },
        })
    return out


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def main():
    p1 = write_jsonl(build_dish_dataset(DISH_PAIRS),
                     os.path.join(HERE, "dish_match.jsonl"))
    p2 = write_jsonl(build_search_dataset(SEARCH_ROWS),
                     os.path.join(HERE, "search_relevance.jsonl"))
    print(f"已生成菜品决策数据集：{p1}（{len(DISH_PAIRS)} 条）")
    print(f"已生成搜索相关性数据集：{p2}（{len(SEARCH_ROWS)} 条）")
    print("\n下一步：按 06_finetune_laya.md 把 jsonl 送入 Laya RLCD 训练流程，")
    print("      完成温度拟合后，务必在自己数据上画 reliability diagram（见 15_eval.py）。")


if __name__ == "__main__":
    main()
