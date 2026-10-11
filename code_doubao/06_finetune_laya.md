# Laya RLCD 微调上手指南（配套 06_finetune_laya.py）

> 对应分享文档「2.4.5 微调」。本文说明如何把通用 Laya 微调成「我们自己的」
> 中文领域判断模型（菜品匹配 / 搜索相关性），并强调**校准**这一最容易翻车的环节。

## 0. 你会得到什么

- 一个与 JEV 同范式（choice/score/noul）、跑在中文多语言基座上、**数据不出域**的私有决策模型；
- 一条可复现流水线：`决策数据集 → RLCD 训练 → 温度拟合 → 评估(校准曲线) → 推送 Hub → 本地服务`。

## 1. 环境与资源

- 官方提供 **Kaggle Notebook**（免费 2×T4 GPU），全流程约 **4–5 小时**；本地建议 ≥1 张 16GB 显存 GPU（CPU 也能跑 322M 多语言模型，仅慢）。
- 推理 / 权重转换可用 `laya-mlx`（Apple Silicon）或 PyTorch 版 `laya`；**RLCD 训练代码在 upstream 训练仓库**，`laya-mlx` 只负责推理与格式转换。
- 基座选择：中文任务优先 `convaiinnovations/laya-multilingual`（mmBERT-base，322M，1024 token）。

## 2. 构造决策数据集

运行：

```bash
python 06_finetune_laya.py
# 产物：dish_match.jsonl、search_relevance.jsonl
```

每行是一条 `{state, questions, answers}`：

```json
{
  "state": "菜A：宫爆鸡丁；菜B：宫保鸡丁。用户下单菜品：宫爆鸡丁。",
  "questions": {
    "is_same": {"type": "noul", "instructions": "菜A 与 菜B 是否为同一道菜（错别字、别名也算）？"}
  },
  "answers": {"is_same": {"type": "noul", "noul": 1.0}}
}
```

实践建议：

- **Noul 标签尽量用软标签**：确定为真给 0.95~0.99 而非 1.0，模棱两可样本给 0.4~0.6，有助于模型学出「不确定」。
- 三种原语都覆盖：判等用 noul、标准菜名归属用 choice（**务必带 `other`**）、相似度分级用 score。
- 直接把历史人工标注 / 生产日志（脱敏后）转成该格式，规模越大、越贴近线上分布越好。

## 3. RLCD 训练 + 温度拟合（按官方 Notebook）

1. 打开官方 Kaggle / upstream 训练 Notebook，挂载上一步的 jsonl；
2. 切到 multilingual checkpoint，按其训练脚本跑 **RLCD（面向校准决策的强化学习）**；
3. 训练后做 **temperature fitting（温度拟合校准）**：在验证集上分桶学习温度，把 logits 缩放到「概率≈真实正确率」；
4. 评估并导出权重，推送到 HuggingFace Hub（私有仓库亦可）。

> RLCD 与 RLHF/RLVR 不是互斥类别：**奖励来源**（人类偏好 / 可验证结果 / 合成评判 / 生产日志）
> 与**训练目标**（受约束决策 + 校准分布）是两个正交维度，这些奖励都可以喂给 RLCD。

## 4. 校准坑（务必避免）

- v0.3.5 起，官方把拟合温度 **clamp 到 [0.5, 5.0]**。若不 clamp，某个分桶原始温度
  （如 0.1006）会把 logits 锐化约 10×，把「抛硬币（0.5）」报成「近乎确定（0.9+）」——
  这会让门控形同虚设。
- **微调后一定要在自己的数据上画 reliability diagram（可靠性曲线）**，报告 ECE / Brier，
  不要直接假设「输出概率 = 真实准确率」。脚本见 `15_eval.py`。
- 数据分布漂移 / 换基座后，重新做温度拟合与阈值选择（见 `16_threshold_jevcal.py`）。

## 5. 部署与灰度

1. 用 `05_laya_serve.py` 把微调模型包成 `/v1/systemone`；
2. **shadow**：与现有匹配结果并跑，比对一致性，不影响线上；
3. **canary**：小流量让高置信样本自动合并，监控准确率 / 人工复核率 / ECE；
4. **enforce**：高置信自动、中置信升级 LLM、低置信转人工，全量。

## 6. 参考

- Laya 模型与 checkpoint：<https://huggingface.co/convaiinnovations>
- Laya-MLX 权重：<https://huggingface.co/aac6fef/laya-multilingual-mlx>
- Datawhale jev-cookbook 第 10 章「本地模型：Laya RLCD 微调全流程」：
  <https://github.com/datawhalechina/jev-cookbook>
