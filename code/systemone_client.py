"""通用 System One 决策模型 HTTP 客户端（JEV / Laya 兼容）。

同时兼容 TypeSafe JEV 云 API 与本地 Laya（/v1/systemone 兼容）服务。
请求/响应格式遵循 JEV 官方 /v1/systemone schema。

依赖: requests
环境变量:
  TYPESAFE_API_KEY   JEV 云调用时需要
  SYSTEMONE_BASE_URL 默认 https://api.typesafe.ai；本地 Laya 设为 http://localhost:8000
  SYSTEMONE_API_KEY  本地服务可填空/任意
  JEV_ALLOW_MOCK     设为 1 且无真实配置时，启用离线 Mock 客户端（用于演示/单测）

用法:
  from systemone_client import get_client
  client = get_client()                 # 有 key/base 走真实，否则走 Mock
  r = client.ask(state, questions)
"""
from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None  # 真实 HTTP 才需要；Mock 模式零依赖可跑


DEFAULT_BASE_URL = os.getenv("SYSTEMONE_BASE_URL", "https://api.typesafe.ai")


class SystemOneClient:
    """真实 HTTP 客户端：POST {base_url}/v1/systemone。"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 api_key: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("SYSTEMONE_API_KEY", "")
        self.timeout = timeout

    def ask(self, state: str, questions: Dict[str, Dict[str, Any]],
            **params) -> Dict[str, Any]:
        """发起一次决策调用。

        state: 共享上下文（文本 / JSON 字符串）
        questions: {name: {type: choice|score|noul, instructions, criteria, ...}}
        返回完整响应（含 answers 与 _latency_ms）。
        """
        payload: Dict[str, Any] = {"state": state, "questions": questions}
        if params:
            payload.update(params)
        if requests is None:
            raise RuntimeError("真实 HTTP 需要 requests：pip install requests；"
                               "或无 Key 时设置 JEV_ALLOW_MOCK=1 走 Mock。")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        t0 = time.time()
        resp = requests.post(f"{self.base_url}/v1/systemone", json=payload,
                              headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        data["_latency_ms"] = round((time.time() - t0) * 1000, 1)
        return data

    def ask_many(self, state: str, *named_questions) -> Dict[str, Any]:
        """便捷：把多个 (name, spec) 合并成一次调用（批量并行）。"""
        questions = {name: spec for name, spec in named_questions}
        return self.ask(state, questions)


class MockSystemOneClient(SystemOneClient):
    """离线 Mock：用最简单的关键词启发式伪造结构化答案，仅用于演示数据流。

    绝非模型本身，仅保证代码在无 API Key 时也能端到端跑通、看清 schema。
    """

    @staticmethod
    def _char_overlap(a: str, b: str) -> float:
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def ask(self, state: str, questions: Dict[str, Dict[str, Any]],
            **params) -> Dict[str, Any]:
        t0 = time.time()
        state_l = (state or "").lower()
        answers: Dict[str, Any] = {}
        for name, q in questions.items():
            qtype = q.get("type", "noul")
            instr = q.get("instructions", "")
            if qtype == "noul":
                p = 0.35
                if "同一道菜" in instr:
                    import re
                    m = re.search(r"菜A[:：](.+?)[；;].*?菜B[:：](.+?)[。.]", state)
                    if m:
                        ov = self._char_overlap(m.group(1), m.group(2))
                        p = round(0.1 + 0.8 * ov, 3)
                else:
                    yes_kw = ["退款", "refund", "扣款", "错误", "违规", "是"]
                    p = 0.92 if any(k in state_l for k in yes_kw) else 0.35
                answers[name] = {"type": "noul", "noul": p}
            elif qtype == "choice":
                crit = q.get("criteria", {})
                best, best_n = "other", 0
                for k, desc in crit.items():
                    if k == "other":
                        continue
                    n = self._char_overlap(state, str(k)) + self._char_overlap(state, str(desc))
                    if n > best_n:
                        best, best_n = k, n
                conf = round(min(0.97, 0.5 + 0.12 * best_n), 3)
                probs = {k: round(1.0 / len(crit), 3) for k in crit}
                if best != "other":
                    probs[best] = round(conf, 3)
                answers[name] = {"type": "choice", "choice": best,
                                 "probabilities": probs, "confidence": conf}
            elif qtype == "score":
                answers[name] = {"type": "score", "score": 2.0,
                                 "distribution": {}, "confidence": 0.6}
            else:
                answers[name] = {"type": qtype, "value": None}
        return {"answers": answers, "_latency_ms": round((time.time() - t0) * 1000, 1),
                "_mock": True}


def get_client() -> SystemOneClient:
    """按环境自动选择真实 / Mock 客户端。"""
    has_real = bool(os.getenv("TYPESAFE_API_KEY") or os.getenv("SYSTEMONE_BASE_URL"))
    if has_real and os.getenv("JEV_ALLOW_MOCK") != "1":
        return SystemOneClient()
    return MockSystemOneClient()


if __name__ == "__main__":
    client = get_client()
    demo = client.ask(
        state="用户说：我被重复扣款了，请退款。",
        questions={
            "refund": {"type": "noul", "instructions": "用户是否要求退款？"},
            "department": {"type": "choice", "instructions": "该分给哪个团队？",
                           "criteria": {"billing": "账单 退款 扣款",
                                        "tech": "技术 故障 bug",
                                        "sales": "销售 购买"}},
        },
    )
    print(json.dumps(demo, ensure_ascii=False, indent=2))
