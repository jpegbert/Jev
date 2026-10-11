# -*- coding: utf-8 -*-
"""
05_laya_serve.py —— 把 Laya 包成 JEV 兼容的 HTTP 服务
=====================================================

对应文档「2.4.4 部署方式 2（推荐接入现有链路）」。
启动后暴露 POST /v1/systemone，请求/响应与 JEV 云完全一致，
业务侧只需把请求地址从 https://api.typesafe.ai 换成 http://localhost:8000，
客户端代码（systemone_client.SystemOneClient）零改动。

依赖：
    pip install fastapi uvicorn
    # Apple Silicon: pip install laya-mlx ；通用平台: pip install "laya[serve]"

启动：
    uvicorn 05_laya_serve:app --host 0.0.0.0 --port 8000
环境变量：
    LAYA_CKPT    默认 aac6fef/laya-multilingual-mlx（中文多语言）
    LAYA_DEVICE  cuda | cpu | mlx
    LAYA_PRELOAD 1 启动即加载权重

自检（另开终端）：
    SYSTEMONE_BASE_URL=http://localhost:8000 python 01_primitives.py
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Local SystemOne (Laya) — JEV compatible")


class Question(BaseModel):
    type: str                 # choice | score | noul
    instructions: str
    criteria: Any = None


class DecisionRequest(BaseModel):
    state: str
    questions: Dict[str, Question]


_AGENT = None


def _get_agent():
    """懒加载权重，避免 import 失败阻塞服务启动。"""
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    ckpt = os.getenv("LAYA_CKPT", "aac6fef/laya-multilingual-mlx")
    try:
        import laya_mlx as laya          # Apple Silicon
        _AGENT = laya.load(ckpt, dtype="float16")
    except ImportError:
        from laya import Laya            # 通用 CPU/CUDA（包名以官方为准）
        _AGENT = Laya.load(ckpt.replace("-mlx", ""))
    return _AGENT


@app.post("/v1/systemone")
def systemone(req: DecisionRequest):
    t0 = time.time()
    try:
        agent = _get_agent()
    except Exception as e:  # 权重未就绪
        raise HTTPException(status_code=503, detail=f"Laya 未就绪: {e}")
    questions = {k: v.model_dump() for k, v in req.questions.items()}
    try:
        result = agent.predict(req.state, questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result["_latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _AGENT is not None}


if __name__ == "__main__":
    # 也可直接 python 05_laya_serve.py 启动
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
