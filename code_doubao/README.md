# code_doubao · JEV / 决策模型分享配套代码

与分享文档 `../Jev_doubao.md` 配套。**核心示例零第三方依赖、无需 API Key**，
通过内置离线 `MockSystemOneClient` 即可端到端跑通三原语、门控、业务场景与
Agent 护栏，方便在分享现场直接演示；接真实后端时只改环境变量、业务代码不变。

## 一分钟跑通

```bash
# 1) 什么都不装（Python 3.9+），直接跑核心示例（自动离线 Mock）
python 01_primitives.py
python 08_dish_matching.py
python 14_ten_use_cases.py

# 2) 评测 / 阈值 / 配图（需要 numpy、matplotlib）
pip install -r requirements.txt
python 15_eval.py
python 16_threshold_jevcal.py
python make_figures.py

# 3) 接本地 Laya（数据不出域）
pip install "laya[serve]"            # 或 Apple Silicon: pip install laya-mlx
uvicorn 05_laya_serve:app --port 8000
SYSTEMONE_BASE_URL=http://localhost:8000 python 01_primitives.py

# 4) 接 JEV 云（中国大陆暂未开放，且涉数据出境，合规场景勿用）
TYPESAFE_API_KEY=xxx python 01_primitives.py
```

## 文件索引（按文档章节）

| 文件 | 对应文档章节 | 讲什么 |
| --- | --- | --- |
| `systemone_client.py` | 1.2 / 3.3 | JEV/Laya 通用客户端 + 离线 Mock（三后端同一套代码） |
| `01_primitives.py` | 1.2 | Choice / Score / Noul 三原语最小示例 |
| `02_batch_parallel.py` | 1.3.3 / 1.4 | 一次 state 多问题并行，顺序 vs 批量对比 |
| `03_cost_model.py` | 1.4 | 成本/速度测算，把“便宜”算成账 |
| `04_laya_local.py` | 2.4.4 | Laya 本地推理（MLX / PyTorch / Mock 三回退） |
| `05_laya_serve.py` | 2.4.4 | FastAPI 把 Laya 包成 `/v1/systemone` 兼容服务 |
| `06_finetune_laya.py` / `.md` | 2.4.5 | 构造领域决策数据集 + RLCD 微调上手指南 |
| `07_confidence_gate.py` | 3.3 | 置信度门控 + LLM/人工降级骨架 |
| `08_dish_matching.py` | 3.2.2 | 菜品匹配 POC：判等/归属/分级 + 置信分流 |
| `09_search_rerank.py` | 3.2.1 / 场景07 | 搜索相关性 + 重排序过滤层 |
| `10_high_cardinality.py` | 1.2.2 | 高基数 Choice 两阶段（先打分再 top-k choice） |
| `11_agent_guardrail.py` | 3.2.4 / 场景09 | Agent 工具安全门 + 完成验证 |
| `12_context_compression.py` | 场景05 | 编码 Agent 上下文压缩 |
| `13_model_router.py` | 场景04 | 便宜/均衡/旗舰模型路由 + 成本账本 |
| `14_ten_use_cases.py` | 3.2.3 | PPT 十个使用场景综合演示 |
| `15_eval.py` | 3.4 | 中文评测集四维指标 + 可靠性曲线（出图） |
| `16_threshold_jevcal.py` | 3.3 / 3.4 | jevcal 式阈值选择 + LLM 流量 + CI 守门 |
| `make_figures.py` | 1.4 | 生成成本对比、批量并行配图到 `../images/` |

## 三种后端如何切换

| 后端 | 环境变量 | 说明 |
| --- | --- | --- |
| 离线 Mock（默认） | 无 | 确定性关键词启发式，仅演示数据流，非真实模型 |
| 本地 Laya | `SYSTEMONE_BASE_URL=http://localhost:8000` | 数据不出域，中文用 multilingual checkpoint |
| JEV 云 | `TYPESAFE_API_KEY=...` | 闭源托管，中国大陆暂未开放，注意数据合规 |

> Mock 仅用于“跑通结构、看清 schema”。所有准确率/成本结论都必须在**你自己的数据**
> 与**真实后端**上复测（见 `15_eval.py`、`16_threshold_jevcal.py`）。
