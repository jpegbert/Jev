"""07 · 本地 /v1/systemone 兼容服务（FastAPI）。

文档 2.4 部署方式 2。把 Laya 包成与 JEV 同格式的 HTTP 服务，
业务代码把请求地址从 JEV 云换成 http://localhost:8000 即可，客户端无需改动
（code/systemone_client.py 的 SystemOneClient 直接可用）。

依赖: pip install fastapi uvicorn laya-mlx
运行: uvicorn code.07_laya_serve:app --port 8000
测试: python 01_basic_primitives.py   # 设置 SYSTEMONE_BASE_URL=http://localhost:8000
"""
from __future__ import annotations
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Local SystemOne (Laya) compatible server")


class Question(BaseModel):
    type: str                       # choice | score | noul
    instructions: str
    criteria: Any = None


class DecisionRequest(BaseModel):
    state: str
    questions: Dict[str, Question]


# 懒加载 Laya，避免 import 失败阻塞服务启动（非 Apple Silicon 可用 laya ONNX/CPU 版替换）
_LAYA = None


def _get_laya():
    global _LAYA
    if _LAYA is None:
        import laya_mlx as laya
        ckpt = os.getenv("LAYA_CKPT", "aac6fef/laya-multilingual-mlx")
        _LAYA = laya.load(ckpt, dtype="float16")
    return _LAYA


@app.post("/v1/systemone")
def systemone(req: DecisionRequest):
    t0 = time.time()
    try:
        agent = _get_laya()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Laya 未就绪: {e}")
    questions = {k: v.dict() for k, v in req.questions.items()}
    try:
        result = agent.predict(req.state, questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result["_latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
