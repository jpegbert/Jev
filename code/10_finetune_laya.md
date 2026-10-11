# Laya RLCD 微调配方（端到端）

> 目标：把开源 Laya 变成「我们的」判断模型（菜品匹配 / 搜索相关性），
> 数据不出域、可本地服务化。对应文档 2.4 微调、3.2.2。

## 流程总览（约 4–5 小时，免费 2×T4）

```
领域决策数据集(jsonl)
   │  build_dish_dataset() 见 10_finetune_laya.py
   ▼
RLCD 训练（官方 Kaggle Notebook / upstream 项目）
   ▼
温度拟合 temperature fitting（校准）
   ▼
评估 reliability diagram（别直接信 概率=准确率）
   ▼
推送 HuggingFace Hub
   ▼
本地 /v1/systemone 服务（code/07_laya_serve.py）
```

## 步骤

1. **造数据**：用 `10_finetune_laya.py` 生成 `dish_match.jsonl`
   （格式：`{state, questions, answers}`）。真实项目里应扩充到数千条、带人工/日志标注。
2. **打开官方 Kaggle Notebook**（Convai Innovations 提供，免费 2×T4 GPU）。
   把 `dish_match.jsonl` 作为训练集上传；按 Notebook 指引执行
   **RLCD 训练 → 温度拟合 → 评估**。
3. **校准坑（务必检查）**：v0.3.5 起拟合温度被 clamp 到 [0.5, 5.0]。
   微调后一定在你的验证集上画 **reliability diagram**（预测概率 vs 真实准确率），
   确认趋势接近 y=x；若某个 bucket 概率被锐化（如原始温度 0.1006 把 logits 放大 ~10×），
   说明校准失败，需重训或重新拟合。
4. **评估**：复用 `code/08_eval_chinese.py` 的四维指标（准确率/底线/Brier/ECE）
   对比 base checkpoint，确认微调后中文判等显著变好。
5. **上服务**：把微调后的权重推到 Hub，本地用 `code/07_laya_serve.py`
   起 `/v1/systemone` 兼容服务；业务侧把 `SYSTEMONE_BASE_URL` 指向它即可。

## 与现有方案 A/B（见文档 3.4）

- 指标：准确率 + 成本 + 延迟 + 校准（ECE/Brier），四项齐报。
- 对照：原版 Laya-multilingual（零样本） vs 微调后 Laya vs 现有 BERT/GLiNER vs LLM。
- 数据不出域：全程本地/私有 GPU，符合 2.1 的合规要求。

## 备注

- RLCD 训练/微调代码在 Laya 的 **upstream** 项目；`laya-mlx` 仅做推理/转换。
- 训练 Notebook 与最新参数以官方仓库/ cookbook 第 10 章（datawhalechina/jev-cookbook）为准。
