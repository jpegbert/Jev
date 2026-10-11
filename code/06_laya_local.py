"""06 · Laya-MLX 本地推理（Apple Silicon 原生，无 PyTorch）。

文档 2.4 部署方式 4。把决策模型跑在本地笔记本，数据不出域、毫秒级。
需安装: pip install laya-mlx

运行:
  python 06_laya_local.py
"""
from __future__ import annotations
import json


def main():
    try:
        import laya_mlx as laya
    except ImportError:
        raise SystemExit("请先安装: pip install laya-mlx （仅 Apple Silicon 适用；"
                         "其他平台可用 code/07_laya_serve.py 起 CPU/CUDA 服务）")

    # 多语言 checkpoint 含中文，最贴合我们；英文可用 aac6fef/laya-mlx
    agent = laya.load("aac6fef/laya-multilingual-mlx", dtype="float16")
    result = agent.predict(
        "用户说：我被重复扣款了，请退款。",
        {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "invoices, payments, refunds",
                                        "technical": "bugs and outages",
                                        "sales": "new purchases"}},
            "urgency": {"type": "score", "instructions": "How urgent is this request?",
                        "criteria": ["not urgent", "soon", "critical"]},
            "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
        },
    )
    print(json.dumps(result["answers"], ensure_ascii=False, indent=2))

    # 进阶：批量 + 前缀缓存（重复负载提速）
    # agent.predict(state, questions, batch_size=16, compile=True, cache_prompts=True)
    # 多语言 Router：Router(dtype="float16", max_loaded=2).predict(state, questions, task="typed_decisions")


if __name__ == "__main__":
    main()
