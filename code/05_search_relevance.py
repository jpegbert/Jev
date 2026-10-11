"""05 · 搜索相关性判定 + 重排序过滤层（RAG 前的「过滤层」，文档 3.2.1 / PPT 场景 07）。

让决策模型坐在「检索」与「生成」之间：对召回 doc 逐条打相关性，
过滤掉无关/过时的，再交给大模型，省钱又稳。

运行:
  python 05_search_relevance.py
"""
from __future__ import annotations
import json
from systemone_client import get_client


def score_relevance(client, query: str, doc: str) -> dict:
    state = f"用户查询：{query}\n候选文档：{doc}"
    q = {"type": "score", "instructions": "文档对查询的相关程度？",
         "criteria": ["完全无关", "弱相关", "相关", "高度相关"]}
    return client.ask(state, {"rel": q})["answers"]["rel"]


def rerank(client, query: str, docs: list[str]) -> list[tuple[str, float]]:
    """对多个 doc 并行打分（一次 state，多个问题）。"""
    questions = {f"d{i}": {"type": "score",
                            "instructions": "文档对查询的相关程度？",
                            "criteria": ["完全无关", "弱相关", "相关", "高度相关"]}
                  for i in range(len(docs))}
    state = "\n".join(f"[{i}] {d}" for i, d in enumerate(docs))
    state = f"用户查询：{query}\n" + state
    ans = client.ask(state, questions)["answers"]
    scored = [(docs[i], ans[f"d{i}"]["score"]) for i in range(len(docs))]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def main():
    client = get_client()
    query = "如何做宫保鸡丁"
    docs = [
        "宫保鸡丁的家常做法与配料表。",
        "今天股市行情综述。",
        "鱼香肉丝的做法。",
        "宫保鸡丁的历史起源与口味特点。",
    ]
    ranked = rerank(client, query, docs)
    print(f"query={query!r}")
    for d, s in ranked:
        flag = "✅保留" if s >= 2.0 else "🗑️过滤"
        print(f"  score={s}  {flag}  {d}")


if __name__ == "__main__":
    main()
