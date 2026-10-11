# JEV 与决策模型（Decision Models）：Agent 时代的「快思考」基础设施

> 组内技术分享大纲 · 适用对象：搜索/NLP/Agent 方向算法工程师  
> 建议时长：75–90 分钟（开场 5 + 第一部分 25–30 + 第二部分 25–30 + 第三部分 25–30 + 收尾 5）  
> 资料来源：网络公开报道、TypeSafe 官方文档、社区复现项目与第三方基准（2026-09~10，详见文末参考链接）。带「*」的性能数字为厂商自报上限值，非典型生产值，使用时应注明。

---

## 开场（5 min）

- **钩子**：一个「不写任何字、只做判断」的模型，发布三周日处理 Token 约 1 万亿，约 1/4 世界 500 强已接入；前 OpenAI 研究员、RLHF/ChatGPT 共同发明者创立，种子轮 4000 万美元。
- **一句话定位**：JEV 不是「更聪明的 LLM」，而是把「判断」从「生成」里剥离出来，做成廉价、高频、可程序化调用的基础设施。
- **本次分享三段结构**：
  1. JEV 本体（是什么、为什么快、技术原理、边界）
  2. 开源生态（闭源同类 + 开源/可本地部署 + 与传统小模型对比 + 评测基准）
  3. **对我们组算法工程师日常有用的东西**（心智模型、可直接套的场景、工程接入、选型、风险）
- **说明**：第一、二部分以通用技术/生态为主，不强行关联组内场景；第三部分重点落到我们的日常工作。

---

## 第一部分：JEV 是什么（重点）

### 1.1 来龙去脉与命名

- **TypeSafe AI**：2026-09-15 从 stealth 发布 JEV（首个公开模型），创始人 Diogo Almeida（前 OpenAI，RLHF/InstructGPT/ChatGPT/GPT-4 核心）、Erik Gafni、Sasha Sheng；4000 万美元种子轮，DCVC 领投，估值约 2 亿美元。
- **名字来源**：取自 19 世纪经济学家 **William Stanley Jevons（杰文斯）**，暗合「杰文斯悖论」——资源使用效率越高、单位成本越低，总消耗量反而越大。TypeSafe 的假设：把「判断」做到极便宜，会催生海量新用法。
- **System One 模型**：借卡尼曼《思考，快与慢》的「系统一/系统二」比喻。JEV = 系统一（快、直觉式判断）；生成式 LLM = 系统二（慢、深思、生成）。

### 1.2 核心定位：Decisions, not strings

- **与传统 LLM 的本质区别**：LLM 逐 token 生成自由文本，应用还要解析/校验；JEV **不生成文本**，直接返回「类型化决策 + 概率 + 置信度」，程序可直接分支。
- **三个原语（primitives）**：
  - **Choice（选择）**：从你给定的候选集（≤255 项）中选一个，返回「选中项 + 每个选项概率分布 + 置信度」。用于无序类目，如「分给哪个团队」。建议保留 an `other` 兜底项。
  - **Score（打分）**：把状态放到有序量表（2–10 级）上，返回「概率加权分数（可落在两级之间）+ 分布 + 置信度」。用于有语义的频谱，如严重度、紧急度。
  - **Noul（是非）**：对一条陈述返回 0–1 的概率（yes 的概率）；接近 1 强 yes、接近 0 强 no、接近 0.5 表示模型无法判断。**无独立 confidence 字段**，这个概率本身就是信号。
- **一个直觉例子**：客服消息 → 问「是否要求退款(Noul) / 不满程度(Score) / 分给哪个团队(Choice)」→ 结构化判断 → 业务代码决定自动处理还是转人工。

### 1.3 技术原理（工程师最关心的部分）

- **非自回归（non-autoregressive）**：不做 token-by-token 生成，单次前向传播直接产出所有结构化值。
- **RLCD（Reinforcement Learning for Calibrated Decisions，面向校准决策的强化学习）**：
  - 训练目标是「校准概率」——模型输出 0.9 时，真实准确率约 90%。这与 LLM 常出现的「过度自信、概率无统计意义」形成对比。
  - RLCD 与 RLHF/RLVR 不是互斥的数学类别：奖励来源（人类偏好/可验证结果）和训练目标（受约束决策+校准分布）是正交维度，RLHF、RLVR、合成评判、生产日志都可喂给 RLCD。
- **硬件感知并行采样（hardware-aware parallel sampler）**：并行评估并一次性返回所有结构化值；高基数选择（high-cardinality）用「两阶段」——先独立给每个选项打分，再做显式选择。
- **社区对「架构创新」的冷静解构**（值得讲，体现批判性思维）：
  - JEV = RLCD + 决策头(decision head) + 类型化 schema + 序列打包(packing) + 树状注意力掩码(attention mask)，让多个问题共享同一 state 前缀、且问题之间/候选之间互不串扰，在**一次前向传播**里完成所有候选打分。
  - 所谓「parallel sampler」是成熟 Transformer 工程（packing + mask + 决策头 + schema 解码）的产品化组合，并非新的注意力算子或采样算法；TypeSafe 的发布稿未给出新的算子、复杂度证明或隔离创新点的消融实验。
- **输入约束**：接受文本、JSON 对象、文本数组；**不支持图像/音频/视频**（多模态需先用其他组件转成文本/结构化观察）。
- **定价**：输入约 $0.042 / 百万 token，输出 token 免费（因非自回归生成、无输出计费）。

### 1.4 为什么快、为什么便宜（性能数据）*

- 官方 workflow 评测：分类决策任务比前沿 LLM 快约 **194–200×**，成本最低约 **445×**（供应商上限数字，非典型值）。
- 单次调用端到端延迟约 **70–500 ms**（官方西海岸服务器实测），官方说法单次约 100 ms 量级。
- **批量并行提问**是性价比关键：state 通常最贵，输出免费。13 个问题一次调用 vs 顺序 13 次调用，成本约降 **12.2×**、速度约快 **10×**（只传一次 state）。
- 与 GPT-6 Luna 对比（每 1000 次决策、各 2000 输入 token，建模成本）：JEV ≈ $0.084（全输入）；GPT-6 Luna 短标签 ≈ $0.215；带约 500 推理 token ≈ $0.465。

### 1.5 JEV 的边界（工程师必须知道，避免误用）

- **做不了**：精确算术/计数、跨日期有序比较、数值邻近、双重否定、长上下文「腐烂」（大状态塞满无关信息会退化）。
- **CJK（中文等）准确率偏低**：TypeSafe 自述在中文等非英文语言上准确率更低——**对中文团队尤其重要**，中文场景应优先测开源中文基座（Qwen 系）。
- **置信度 ≠ 正确性；格式正确 ≠ 语义正确**：JEV 保证从你给的 A/B/C 里返回一个合法候选，但完全可能选错 B。严谨说法：「JEV 不会给出候选范围之外的结果，同时提供一个可被程序控制的概率信号」。
- **不适用**：开放文本生成、长推理链、需要「把想法说出来」的任务。

---

## 第二部分：开源生态（重点）

### 2.1 为什么需要开源替代

- JEV 闭源、无公开权重、不能本地部署、仅托管 API；且**对中国大陆地区尚未开放**。
- 行业在发布后约两周内集体跟进，决策模型新品类从「首发」走到「群战」，速度在大模型行业也少见。

### 2.2 闭源/托管同类（快速跟进者）

- **OpenAI Decisions API**：开发者日推出，结构化决策能力开放；基于自家小模型，约两周做出。
- **Perplexity pplx-decider-v1-27b**：Qwen3.8-27B 微调，Apache 2.0，HuggingFace 公开发布；约 $0.04 / 百万输入；厂商自报 11 基准面板 85.71% vs JEV 84.51%。
- **Cloudflare Clef / Clef-flash**（2026-10-01 开源）：基于 Qwen 后训练，Apache 2.0；Clef 中位 ~209 ms / $0.24/M，Clef-flash ~39 ms / $0.09/M；65k token 上下文。
- **AWS Strands Decider 2B**：Qwen 2B 底座，把 next-word 头换成约百万参数 pointer head；中位 ~115 ms（RTX 3090）/ ~153 ms（M3）；**开源权重 + 训练数据 + 脚本**（Apache 2.0），本地零边际成本。
- **Microsoft-Decision-1**（2026-10，微软 CTO 办公室 VP 发布）：Qwen3.5-9B 后训练，上架 Foundry + OpenRouter；官方称 36 基准 15 万题盲测最高准确率、比 GPT-6 Sol 快约 35×、8 类扰动仅 1.3% 翻转。闭源托管、走网关调用，是继 OpenAI/Perplexity/Cloudflare 之后大厂正式下场的重要标志（cookbook 知识库 #39 全译）。
- **关键观察**：这一轮决策模型的「躯干」大量是**中国开源基座 Qwen**——Clef、Strands Decider、APUS、Microsoft-Decision-1 都跑在 Qwen 上。闭源 JEV 留下的「本地化/数据不出域」缺口，正好被开源中文基座补上。

### 2.3 开源 / 可本地部署（重点列举，附工程要点）

| 项目                                | 基座/路线                 | 协议              | 规模                   | 特点 / 适用                                                                            |
| --------------------------------- | --------------------- | --------------- | -------------------- | ---------------------------------------------------------------------------------- |
| **Kev** (jaredpalmer/kev)         | Qwen3.5               | Apache 2.0（可商用） | 0.8B/4B/9B/27B       | 兼容 `/v1/systemone` 接口；0.8B 无 GPU 可跑；Modal 一键 HTTPS 部署；推荐 4B 起                      |
| **OpenJev** (razorback16/openjev) | DiffusionGemma + vLLM | 开源              | —                    | 同时提供文本生成接口；openjev-sglang：Qwen + SGLang（B200 部署示例）                                 |
| **APUS fast-browser-use**         | Qwen3.5-9B            | MIT             | 单模型                  | 全球首批对 JEV 的开源复现；单 token logits 决策 + KV-Cache 广播 + 并发批量评估；浏览器自动化；mac/Win/Linux 离线运行 |
| **Laya / Laya-MLX / Laya-ONNX**   | 421M 参数               | 开源              | 421M                 | Apple Silicon(MLX) / AMD(ROCm) / x86 CPU；毫秒级本地，~420MB；最小零样本上限低，需微调                 |
| **NanoJev**                       | 0.6B 并行决策             | 开源              | 0.6B                 | 附带训练资料，适合研究小模型如何组织多项决策                                                             |
| **SemIf / jeff / simple-jev**     | 各自开放模型                | 开源              | jeff 400M(GLiFormer) | SemIf 复现接口模式（未复现未公开模型）；simple-jev 用开放模型分数构造分类/评分                                   |
| **Intern-Decision**（上海AI实验室）      | 多模态                   | 开源              | 0.8B/2B/4B           | 看懂图像/界面再做决策；官方称推理比 JEV 快 2–3×                                                      |
| **StartLux-Decision**（陈大年/原点星辉）   | Qwen 后训练              | 开源              | 0.8B–27B（五档）         | 配齐量化；4B Q4 量化 ~2.71GB 与原始权重一致率 98.3%；27B 自测 DecisionIndex 超 JEV（需谨慎看待自测）           |

> 注意：上述开源项目大多「复现接口/范式」，并非复现 JEV 未公开的模型与训练；切换基座后不能把原版 JEV 的成绩直接平移过去，要逐一核对基座、限制与测试方法。

### 2.4 ★ 重点深讲：Laya（部署 + 微调，算法工程师日常可用）

- **为什么选它做重点**：Laya 是 Convai Innovations 出的开源「System One 决策引擎」，Apache-2.0；与 JEV 同范式（choice/score/noul、校准概率、单次前向、无文本生成），且**提供多语言 checkpoint（含中文友好）**、Jev 兼容 `/v1/systemone` 接口、体量小（322–421M）、并附带可复现的微调配方——对「想私有化 + 中文 + 能自己训」的我们最实用。
- **架构（一页纸）**：双向编码器（ModernBERT-large 421M 英文 / mmBERT-base 322M 多语言）→ 决策 Transformer + scoring/action heads → 校准概率输出。无自回归解码；带 fitted temperature calibration（温度拟合校准）。
- **三个官方 checkpoint（HuggingFace）**：
  - `convaiinnovations/laya`：ModernBERT-large，421M，512 token，英文
  - `convaiinnovations/laya-multilingual`：mmBERT-base，322M，1024 token，**多语言（含中文）** ← 我们优先
  - `convaiinnovations/laya-typed-decisions`：ModernBERT-large，421M，1024 token，typed-decisions 工作流
  - MLX 预转换 FP16：`aac6fef/laya-mlx` / `aac6fef/laya-multilingual-mlx` / `aac6fef/laya-typed-decisions-mlx`
- **部署方式（四种，按场景挑）**：
  1. **一行命令 + CLI（最快试用）**：`pip install laya` → `laya "我被重复扣款了，请退款。"`
  2. **HTTP 服务（Jev 兼容，推荐接入现有链路）**：`pip install "laya[serve]"` → `LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve`，起 FastAPI，暴露 `POST /v1/systemone`，与 JEV API 同格式；把现有 JEV 请求地址换成 Laya 本地地址即可。环境变量控设备/预加载/线程数。
  3. **Docker（CPU + CUDA，服务化）**：`docker compose -f compose.yaml -f compose.http.yaml up --build laya-serve`；首次启动下载权重，之后很快。
  4. **Laya-MLX（Apple Silicon 原生，无 PyTorch）**：`pip install laya-mlx`，内存 <1GB，M 系列芯片毫秒级（M3 Max P50 英文 13.4ms / 多语言 7.4ms；batch-50 多语言 395 q/s）。适合本地开发机/笔记本跑决策。
- **Python 调用（工程师可直接抄）**：
  ```python
  import laya_mlx as laya
  agent = laya.load("aac6fef/laya-mlx", dtype="float16")
  result = agent.predict(
      "I was billed twice. Please refund the duplicate today.",
      {
          "department": {"type":"choice","instructions":"Which team should handle this?",
                         "criteria":{"billing":"invoices, payments, refunds",
                                     "technical":"bugs and outages","sales":"new purchases"}},
          "urgency": {"type":"score","instructions":"How urgent is this request?",
                      "criteria":["not urgent","soon","critical"]},
          "refund": {"type":"noul","instructions":"Does the customer ask for money back?"},
      },
  )
  print(result["answers"])  # 同 JEV schema：choice/probabilities/confidence, score, noul
  ```
  - 进阶参数：`device="gpu"|"cpu"`、`batch_size`（每前向问题数，默认 16，超出分块）、`compile=True`+`cache_prompts=True`（重复负载提速，前缀缓存上限 128 问题）；多语言可用 `Router(dtype="float16", max_loaded=2)` + `router.predict(state, questions, task="typed_decisions")`。
- **微调（重点：把 Laya 变成「我们的」判断模型）**：
  - 官方提供 **Kaggle Notebook**，免费 2×T4 GPU 上约 **4–5 小时**跑完整流程：**构建决策数据集 → RLCD 训练 → 温度拟合(temperature fitting) → 评估 → 推送到 Hub**。RLCD 训练/微调代码在 upstream 项目（laya-mlx 仅做推理/转换）。
  - **对我们组的含义**：菜品匹配（is A==B? 用 Noul；候选标准菜名归属用 Choice；相似度分级用 Score）、搜索相关性判定等都可沉淀为「带标注的决策数据集」，在 multilingual/EN checkpoint 上微调出领域模型，本地服务化后比基础版准很多、且数据不出域。
  - **校准坑（务必讲）**：v0.3.5 起把拟合温度 clamp 到 [0.5, 5.0]，否则某个 bucket 原始温度 0.1006 会把 logits 锐化约 10×，把「抛硬币」报成「近乎确定」。微调后一定要在你的数据上画校准曲线（reliability diagram），别直接信「概率=准确率」。

### 2.5 生态索引 & 中文学习入口（两个 GitHub 项目必看）

- **Awesome Jev（fatwang2/awesome-jev，MIT，独立整理）**：JEV 生态的「总目录」，收录大量项目并标注类型与拟合度；建议分享时作为「延伸阅读」贴出。挑几个与日常最相关的：
  - `jevcal`：**别瞎猜置信度阈值**——在你的数据上测决策模型、选满足准确率目标的阈值、算还需多少流量走 LLM、模型更新悄悄破坏时让 CI 失败（直接对应我们的阈值工程）。
  - `jevfire`：把多个有限选择字段批量并行过 vLLM，复用共享指令/上下文前缀。
  - `jevmlx`：给定 schema，一次前向给每个字段算概率、自行拼出合法 JSON（valid by construction）。
  - 开源复现/兼容层：`LitJev`、`WaterSheep`、`openjev`、`Open-Jev`、`Laya`（均实现/兼容 `/v1/systemone`）。
  - 安全/合规评测：`jev-sec-bench`（提示注入、漏洞代码检测）、`jev-secret-detection`（密令识别）、`jev-spam-eval`（零样本分类 + 上下文增强）。
  - Coding Agent 工具（对应我们后续 Agent 方向）：`jev-mcp`、`jev-router`、`pi-jev`、`foreman`、`jev-rules`、`skillbox`（MCP 化判断、按 turn 路由模型/技能、工具前语义防火墙等）。
  - 另外两个聚合版：`awesome-jev.xyz`（由 JEV 自己筛选项目）、`Nedomas/awesome-jev-tanxarx`。
- **Datawhale jev-cookbook（datawhalechina/jev-cookbook，CC BY-NC-SA 4.0，在线阅读 datawhalechina.github.io/jev-cookbook）**：**最适合中文工程师的系统性入口**。11 章可运行中文教程 + 官方文档全站中译 + 40+ 条社区知识库（含 Microsoft-Decision-1、open-annie 等前沿全译）+ 18 篇官方实战配方 + 12 个可运行应用。结构导航：**第一部分 入门**（1 认识 Jev / 2 核心概念 5 册 / 3 架构模式：推测性扇出·置信度门控·复合评分·意图路由）→ **第二部分 实战**（4 18 篇配方 / 5 语音 3D 智能家居 / 6 模型评测：Laya vs Jev 四维基准与「无重试无回退、预算记账」的诚实评测纪律）→ **第三部分 研究与生态**（7 12 个应用 / 8 前沿研究：Jev-Mem 记忆 + JevHarness 控制流 / 9 Agent 集成：Pi 工具调用 gate + DSH 决策协作 / 10 本地模型：Laya RLCD 微调全流程 / 11 知识库）。**与我们的直接关系**：第 4 章 18 篇配方里「重排序 / 语义搜索 / 函数调用 / 引用核查 / 防护栏」几乎就是搜索/NLP 日常模板；第 6 章的 231 题四维（准确率/底线/Brier/ECE）评测纪律可直接套我们的中文评测集；第 9–10 章是后续 Agent 与本地微调的现成路径。无需 API Key 可离线跑（`./setup_env.sh` + `JEV_RUN_MODE=offline`）。

### 2.6 与传统小模型对比：决策模型 vs BERT/DeBERTa/GLiNER

- **传统分类器（BERT/DeBERTa/GLiNER）**：需持续标注、单任务训练管线、监控/重训、数据漂移、每类目一个独立模型。
- **决策模型**：零样本迁移（问题空间推理时可变）、基础模型的泛化能力、同一模型多任务复用。
- **GLiNER 等开源 NER/信息抽取**：低延迟、可本地，但**无校准概率保证**——这是与 JEV 系的核心差异（可信自动化依赖校准概率）。
- **结论**：决策模型不是替代 BERT，而是把「判断」做成可复用、可校准、可阈值化控制的**基础设施层**；高频固定类目+有标注数据+极致延迟的场景，现有小模型仍能打，先别急着换。

### 2.7 评测基准：怎么选、怎么评估（避免被榜单带偏）

- **JevBench**（Benchmark Heaven，v1.5.5）：4 轴等权——智能/校准/速度/成本；109 系统，1624 决策/系统，720 密封题防训练泄露。v1.5.5 中 JEV 综合 #3(72.1)，前两名是 Gemma-4-12B 社区微调；GPT-6 Luna 智能 96.2/校准 95.6 远高于 JEV，但因速度/成本轴崩只排 #37。
- **Decision Index**（HuggingFace，multimodalart，0.2.1）：43 基准、120340 请求/系统，单卡 RTX PRO 6000，单独报校准(ECE)。0.2.1 快照 JEV 1.13 居首(57.91)，但身后开源 27B 仅差半分；JEV 校准 ECE 0.074 并非最优（Jebadiah 0.014、pplx-decider 0.018 更好）。
- **其他**：S1MB（23 模型/137 任务）、sysone-bench（密封 manifest，同输入对比）、Cua-Bench-S1（computer-use 动作选择）、Bespoke Nimble 公开基准、StartLux eval 套件（可 `--endpoint` 拉通评测任意 /v1/systemone 服务）。
- **关键认知**：厂商榜单数字是「选出来的上限」不是典型值；**决策模型买的不是「比最强 LLM 更聪明」，而是「够用的判断 + 极低代价与延迟」**。


## 第三部分：对我们组算法工程师日常有用的东西（重点，25–30 min）

> 这一段把前两部分「通用技术/生态」落到我们的日常工作：心智模型 → 可套场景 → 工程接入 → 评测纪律 → 选型 → 风险。cookbook 的 18 篇配方是现成场景模板，awesome-jev 的工具是现成工程件。

### 3.1 心智模型：把「判断」从「生成」里剥出来

- **决策模型 = 独立的一层「判断」**：系统二（LLM）负责「想/生成」，系统一（决策模型）负责「判」。做法——把 pipeline 里所有「先让 LLM 生成文本、再 parse 成 bool/类目/分数」的环节，改成「直接问决策模型拿结构化判断」。
- **收益**：更便宜（输出免费、非自回归）、更快（单次前向）、可程序化分支、可阈值化控制、带校准概率。代价：放弃自由文本与长推理——这正是它该待的位置（高频、固定形态的判断）。

### 3.2 直接可套的场景（按我们组日常排序）

- **搜索 / NLP（最现成，cookbook 第 4 章 18 篇配方即模板）**：
  - 相关性判定 / 重排序（rerank）：query–doc 相关性用 Noul/Score，多档重排用 Choice/Score → 对应配方「重排序」「语义搜索」。
  - 语义路由：把请求路由到正确召回链路/模型，Choice。
  - 蕴含 / NLI：两句话是否矛盾/蕴含，Noul。
  - 防护栏 / 安全门：输入是否违规/越权，Noul + 阈值 → 对应配方「防护栏」。
  - 函数/工具调用 gate：给 LLM 前先判「该不该调、调哪个」，Choice → 对应配方「函数调用」「引用核查」。
  - 上下文压缩：哪些历史片段该保留，Noul 逐条筛。
- **菜品匹配（你主做，天然适配，Noul 判等是核心）**：
  - is A==B（同一道菜不同写法）？→ **Noul**（接近 0.5 转人工/上 LLM）。
  - 候选标准菜名归属 → **Choice**（≤255 项标准库）。
  - 相似度/置信分级 → **Score**（可落两级之间）。
  - 置信度驱动：高置信自动合并、低置信转人工、中置信上 LLM 复核 → 降本提速。
  - 沉淀带标注决策数据集 → Laya-multilingual 上 RLCD 微调（接 2.4）。
- **Agent（后续方向，cookbook 第 9 章 + awesome-jev 工具件）**：
  - 工具安全门：调危险工具前用 Noul 判意图/权限；完成验证：子任务是否达成，Noul。
  - 按 turn 路由模型/技能：awesome-jev 的 `jev-router` / `jev-mcp` / `foreman` / `jev-rules` / `skillbox`。
  - 控制流：JevHarness（cookbook 第 8 章前沿研究）把「判断」当控制流节点。

### 3.3 工程接入（可直接抄）

- **双栈策略**：线上先用 JEV/托管 API 验证范式；生产切 Laya 本地（见 2.4 四种部署）。
- **置信度门控 + LLM 降级骨架**（决策模型拿不准就上 LLM，不瞎猜）：

  ```python
  from jev import Jev  # 或 laya_mlx（见 2.4）
  def decide(state, question, llm_fallback):
      r = Jev().ask(state, question)          # Choice/Score/Noul + 概率 + 置信度
      if question["type"] == "noul":
          p = r["noul"]                        # yes 概率
          if p > 0.9:  return "yes"
          if p < 0.1:  return "no"
          return llm_fallback(state, question) # 0.1–0.9 转 LLM/人工
      if r["confidence"] < THRESHOLD:          # Choice/Score 用 jevcal 选的阈值
          return llm_fallback(state, question)
      return r
  ```
- **批量并行**：一次调用塞多个问题（state 最贵、输出免费），awesome-jev `jevfire` 思路。
- **阈值工程**：awesome-jev `jevcal`——在你的数据上选满足准确率目标的阈值、算还需多少流量走 LLM、模型更新破坏时让 CI 失败。

### 3.4 评测纪律（别被厂商榜带偏）

- **复用 cookbook 第 6 章诚实评测**：231 道公开题四维（**准确率 / 底线 / Brier / ECE**），无重试、无回退、预算记账。关键：**不只看准确率，还要看校准（ECE/Brier）**——决策模型买的正是校准概率。
- **自建中文评测集**（菜品匹配 / 搜索相关性），A/B 对比决策模型 vs 现有 BERT/GLiNER/LLM：四项齐报——准确率 + 成本 + 延迟 + 校准。
- **容量/阈值规划**用 `jevcal`；榜单（DecisionIndex / JevBench）数字审慎引用（见 2.7）。

### 3.5 选型决策树

- 高频固定类目 + 有标注 + 极致延迟 + 数据不出域 → 现有 BERT/GLiNER 或 **Laya 微调**（别急着换）。
- 需零样本泛化、问题空间多变、要校准概率做自动分支 → 决策模型（JEV/托管/Laya）。
- 开放生成 / 长推理 → 仍用 LLM。
- 中文优先 → 测 Qwen 系（Kev / StartLux / Laya-multilingual / Clef / Strands / Microsoft-Decision-1）。

### 3.6 风险与误用（工程红线）

- 置信度 ≠ 正确性；格式正确 ≠ 语义正确 → 必须门控 + 降级 + 人工兜底（见 3.3）。
- CJK 偏弱 → 中文场景先测开源中文基座，别直接信原版 JEV 成绩（见 1.5）。
- 长上下文「腐烂」→ state 要精简，去掉无关信息。
- 闭源 JEV 对中国大陆未开放、数据出境风险 → 优先本地/开源（见 2.1）。

---

## 收尾（5 min）

- **一句话总结**：JEV 不是「会思考的聊天模型」，而是 Agent 架构里的「廉价高频判断层」；系统二负责想、系统一负责判，Harness 负责调度。
- **下一步（详见第三部分）**：① 盘点 pipeline 里所有「判断型」LLM 调用；② 选 POC（**强烈建议菜品匹配 / 搜索相关性判定**）；③ 自建中文评测集 A/B（准确率+成本+延迟+校准）；④ 若做 Agent，先在小接缝（工具安全门、完成验证）shadow → canary → enforce。
- **Q\&A / 讨论**：我们组哪些判断现在最贵、最慢、最不稳？哪个最适合先拿决策模型试？

---

## 附：参考链接（建议分享时附在最后一页）

- TypeSafe 官方：jev-lab.com / jevtypesafe.org / jevplayground.com（概念、guides、what-is-jev）
- 发布报道：央广网、环球网、36氪/极客公园、Business Standard、Physical AI News、CNBC For AI
- 技术解构/冷静分析：HN Today（RLCD+packing+mask 解构）、BestHub（fast/slow 架构 shift、ModernBERT 自托管示例）、EulerFold（架构蓝图）
- 开源项目：jaredpalmer/kev、razorback16/openjev、openjev-sglang、APUS fast-browser-use、Laya、NanoJev、SemIf、jeff、simple-jev、Intern-Decision、StartLux-Decision
- 闭源同类：OpenAI Decisions API、Perplexity pplx-decider-v1-27b、Cloudflare Clef、AWS Strands Decider 2B
- 基准：JevBench（Benchmark Heaven）、Decision Index（HuggingFace/multimodalart）、S1MB、sysone-bench、Cua-Bench-S1、Bespoke Nimble、StartLux eval 套件
- Harness 实践：Workstation「Building a Jev agent harness」、LangChain JEV 集成、腾讯云/社区「快慢分工」实践文
- 接入示例：ToolScout「How to Use Jev」、Refix Python tutorial、dev.to「The Decision Layer AI Agents Have Been Missing」
