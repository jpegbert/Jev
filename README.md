# JEV 决策模型 · 组内分享资料包

> 主题：JEV 与决策模型（Decision Models）——Agent 时代的「快思考」基础设施。
> 配套：分享文档 + 可运行代码 + 现场演示 Notebook。

## 目录

| 文件 | 说明 |
| --- | --- |
| `JEV分享文档.md` | **主文档**（详实版，含图/表/代码引用/参考文献） |
| `code/` | 完整可运行 Python 示例（11 个 .py + `requirements.txt`） |
| `notebooks/JEV_live_demo.ipynb` | 现场演示 Notebook（复用 `code/`，无 Key 走 Mock） |
| `..\2026-10-11-09-41-32\Jev使用场景.pptx` | 组内 PPT《Jev：10 个使用场景》（文档第三部分场景表来源） |

## 快速开始

```bash
# 1) 文档阅读：用支持 Mermaid 的 Markdown 预览器打开 JEV分享文档.md
# 2) 跑代码（无 Key 也能跑通数据结构，自动 Mock）
cd code
pip install -r requirements.txt
python 01_basic_primitives.py        # 三原语
python 04_dish_matching.py            # 菜品匹配 POC
python 08_eval_chinese.py             # 中文评测四维

# 3) 现场演示 Notebook
cd notebooks
jupyter lab JEV_live_demo.ipynb

# 4) 接真实模型（二选一）
SYSTEMONE_BASE_URL=http://localhost:8000 python 01_basic_primitives.py   # 本地 Laya
TYPESAFE_API_KEY=xxx python 01_basic_primitives.py                       # JEV 云
```

## 代码索引（`code/`）

| 文件 | 内容 |
| --- | --- |
| `systemone_client.py` | 通用 `/v1/systemone` 客户端（真实 HTTP + 离线 Mock） |
| `01_basic_primitives.py` | 三原语最小调用 |
| `02_batch_parallel.py` | 一次 state 多问题并行 |
| `03_confidence_gate.py` | 置信度门控 + LLM 降级 |
| `04_dish_matching.py` | 菜品匹配 POC（Noul/Choice/Score） |
| `05_search_relevance.py` | 搜索相关性 + 重排序过滤层 |
| `06_laya_local.py` | Laya-MLX 本地推理 |
| `07_laya_serve.py` | FastAPI `/v1/systemone` 本地服务 |
| `08_eval_chinese.py` | 中文评测四维（准确率/底线/Brier/ECE） |
| `09_threshold_selection.py` | jevcal 式阈值选择 + 流量测算 |
| `10_finetune_laya.py` | 构造中文决策数据集（RLCD 原料） |
| `10_finetune_laya.md` | Laya RLCD 微调端到端配方 |
| `11_agent_guardrail.py` | Agent 工具安全门 + 完成验证 |

> 文档中每段「📎 完整可运行示例」即指向上述文件。
