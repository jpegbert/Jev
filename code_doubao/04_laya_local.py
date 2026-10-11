# -*- coding: utf-8 -*-
"""
04_laya_local.py —— Laya 本地推理（私有化 / 中文 / 数据不出域）
=============================================================

对应文档「2.4 重点深讲：Laya」。Laya 是 Convai Innovations 的开源
System One 决策引擎（Apache-2.0），与 JEV 同范式（choice/score/noul、
校准概率、单次前向、无文本生成），提供多语言 checkpoint（含中文）。

三种本地用法任选：
  A) Apple Silicon 原生（无 PyTorch）： pip install laya-mlx
  B) 通用 CPU/CUDA（PyTorch 版 laya）： pip install "laya[serve]"
  C) 没装 Laya 时：自动回退到本仓库 Mock，仅演示 schema

checkpoint（HuggingFace）：
  convaiinnovations/laya                ModernBERT-large 421M 英文 512
  convaiinnovations/laya-multilingual   mmBERT-base 322M 多语言(含中文) 1024  ← 中文优先
  convaiinnovations/laya-typed-decisions ModernBERT-large 421M typed 1024
  MLX FP16：aac6fef/laya-mlx、aac6fef/laya-multilingual-mlx 等

运行：python 04_laya_local.py
"""
from __future__ import annotations

import json

from systemone_client import get_client

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "该分给哪个团队处理？",
        "criteria": {"billing": "发票、支付、退款、重复扣款",
                     "technical": "bug、宕机、故障",
                     "sales": "新购、续费",
                     "other": "其他"},
    },
    "urgency": {
        "type": "score",
        "instructions": "紧急程度？",
        "criteria": ["not urgent", "soon", "critical"],
    },
    "refund": {
        "type": "noul",
        "instructions": "客户是否要求退款？",
    },
}


def run_with_laya_mlx() -> bool:
    try:
        import laya_mlx as laya
    except ImportError:
        return False
    agent = laya.load("aac6fef/laya-multilingual-mlx", dtype="float16")
    result = agent.predict(
        "我被重复扣款了，请今天就把多扣的钱退回。",
        QUESTIONS,
        # 进阶：batch_size=16, compile=True, cache_prompts=True
    )
    print("[backend] laya-mlx (Apple Silicon)")
    print(json.dumps(result["answers"], ensure_ascii=False, indent=2))
    return True


def run_with_laya_torch() -> bool:
    try:
        from laya import Laya  # PyTorch 版包名以官方为准
    except ImportError:
        return False
    agent = Laya.load("convaiinnovations/laya-multilingual")
    result = agent.predict("我被重复扣款了，请退款。", QUESTIONS)
    print("[backend] laya (PyTorch CPU/CUDA)")
    print(json.dumps(result["answers"], ensure_ascii=False, indent=2))
    return True


def main() -> None:
    if run_with_laya_mlx() or run_with_laya_torch():
        return
    print("[backend] 未安装 laya / laya-mlx，回退到离线 Mock（仅演示 schema）。")
    print("          Apple Silicon: pip install laya-mlx；"
          "通用平台: pip install \"laya[serve]\"\n")
    client = get_client(force_mock=True)
    resp = client.ask("我被重复扣款了，请立刻退款。", QUESTIONS)
    print(json.dumps(resp["answers"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
