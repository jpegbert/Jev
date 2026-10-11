# -*- coding: utf-8 -*-
"""
09_search_rerank.py —— 搜索相关性判定 + 重排序过滤层（RAG 前的“闸门”）
=====================================================================

对应文档「3.2.1 搜索/NLP」与 PPT 场景 07「搜索结果重排序」。
让决策模型坐在「检索」与「生成」之间：

    召回 docs -> 决策模型并行打相关性分 -> 过滤无关/过时 -> 再交给 LLM

好处：减少进 LLM 的噪声与 token，提升答案质量、降低成本。

运行：python 09_search_rerank.py
"""
from __future__ import annotations

from systemone_client import get_client

LEVELS = ["完全无关", "弱相关", "相关", "高度相关"]
KEEP_THRESHOLD = 2.0     # 低于该分的文档被过滤


class SearchFilter:
    def __init__(self, client):
        self.client = client

    def score_one(self, query: str, doc: str) -> dict:
        state = f"用户查询：{query}\n候选文档：{doc}"
        q = {"type": "score", "instructions": "候选文档对用户查询的相关程度？",
             "criteria": LEVELS}
        return self.client.ask(state, {"rel": q})["answers"]["rel"]

    def rerank_batch(self, query: str, docs: list[str]):
        """一次 state、多问题并行给所有 doc 打分（批量并行，见 02）。"""
        questions = {}
        for i in range(len(docs)):
            questions[f"d{i}"] = {"type": "score",
                                  "instructions": "候选文档对用户查询的相关程度？",
                                  "criteria": LEVELS}
        state = f"用户查询：{query}\n" + "\n".join(f"[{i}] {d}"
                                                   for i, d in enumerate(docs))
        ans = self.client.ask(state, questions)["answers"]
        scored = [(docs[i], ans[f"d{i}"]["score"], ans[f"d{i}"].get("confidence"))
                  for i in range(len(docs))]
        return sorted(scored, key=lambda x: x[1], reverse=True)


def main() -> None:
    client = get_client()
    sf = SearchFilter(client)
    query = "如何做宫保鸡丁"
    docs = [
        "宫保鸡丁的家常做法与配料表。",
        "今天股市行情综述与板块点评。",
        "鱼香肉丝的正宗做法。",
        "宫保鸡丁的历史起源、口味特点与小贴士。",
        "一篇与做菜无关的旅游随笔。",
    ]
    ranked = sf.rerank_batch(query, docs)
    print(f"query = {query!r}\n")
    kept, dropped = [], []
    for d, s, conf in ranked:
        if s >= KEEP_THRESHOLD:
            kept.append(d)
            print(f"  保留 score={s:<4} conf={conf}  {d}")
        else:
            dropped.append(d)
            print(f"  过滤 score={s:<4} conf={conf}  {d}")
    print(f"\n召回 {len(docs)} -> 保留 {len(kept)} / 过滤 {len(dropped)}，"
          f"仅把高相关文档送入 LLM。")


if __name__ == "__main__":
    main()
