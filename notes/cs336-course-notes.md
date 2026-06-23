# CS336 课程笔记

> Stanford CS336 Spring 2025 — Language Modeling from Scratch
> 整理时间：2026-06-23

---

## 目录

- [第一讲：Introduction（3/31，Percy Liang）](#第一讲introduction31percy-liang)
- [第二讲：Systems 详解（Assignment 2）](#第二讲systems-详解assignment-2)
- [第三讲：Scaling Laws 详解（Assignment 3）](#第三讲scaling-laws-详解assignment-3)
- [第四讲：Data 详解（Assignment 4）](#第四讲data-详解assignment-4)
- [第五讲：Alignment 详解（Assignment 5）](#第五讲alignment-详解assignment-5)

---

## 第一讲：Introduction（3/31，Percy Liang）

### 课程定位：为什么叫"从零构建"？

CS336 的灵感来自操作系统课程（如 CS140）——学生从零写一个完整的操作系统。本课程做同样的事，但对象是**语言模型**：

> *"We will lead students through every aspect of language model creation, including data collection and cleaning for pre-training, transformer model construction, model training, and evaluation before deployment."*

**核心理念**：不是调用 API，而是亲手实现每一个组件。

---

### 先修要求（硬门槛）

| 领域 | 要求 |
|------|------|
| Python | 精通，代码量比其他 AI 课程多一个数量级 |
| 深度学习 | 熟悉 PyTorch |
| 系统 | 基本内存层次结构概念 |
| 数学 | 微积分、线性代数（MATH 51 / CME 100） |
| 概率统计 | 高斯分布、均值、标准差等（CS 109） |

---

### 课程结构（5 个 Assignment）

| 作业 | 主题 | 核心内容 |
|------|------|----------|
| Assignment 1 | Basics | Tokenizer、Transformer 架构、Optimizer、最小训练 LLM |
| Assignment 2 | Systems | Profiling、Benchmark、Triton 写 FlashAttention、分布式训练 |
| Assignment 3 | Scaling | Transformer 各组件分析、Scaling Law、API 拟合 |
| Assignment 4 | Data | Common Crawl、预训练数据处理 |
| Assignment 5 | Alignment | RLHF、Safety |

---

### 为什么现在学 LLM？

- LLM 是历史上第一个**真正通用**的 AI 系统
- 理解 LLM = 理解当今 AI 的核心
- 本课程目标：**从数据到部署，完整掌握 LLM 生命周期**

---

## 第二讲：Systems 详解（Assignment 2）

### 什么是 Profiling（性能剖析）？

**核心问题**：你的代码到底慢在哪里？

训练 LLM 时，GPU 每秒能算数万亿次，但**数据传输**（CPU→GPU）和**内存访问**往往才是瓶颈。Profiling 就是用量化工具测量程序每个部分的耗时，找出"热点"。

**类比**：你开车从上海到北京（10小时），Profiling 告诉你其中 8 小时在休息区加油，2 小时在开车——问题一目了然。

**常用工具**：
- `torch.profiler`：PyTorch 内置，可视化每层耗时
- `nsight-systems`：NVIDIA 官方，GPU 级细粒度分析
- `nvidia-smi`：实时监控 GPU 利用率

**关键指标**：
- **GPU Utilization**：GPU 是否在"空转"（等待数据）
- **Memory Bandwidth**：显存读写是否饱和
- **Kernel Launch Overhead**：频繁小操作的开销

---

### 什么是 Benchmark（基准测试）？

**核心问题**：你的模型到底有多快/多好？

Benchmark 是标准化的测试集，用于：
1. **性能对比**：A 模型 vs B 模型，谁更快？
2. **回归检测**：改代码后，速度有没有下降？
3. **硬件评估**：A100 vs H100，实际差距多少？

**LLM 训练常用 Benchmark**：
- **Throughput（吞吐量）**：每秒处理多少 token
- **Time to First Token（TTFT）**：推理延迟
- **Memory Footprint**：峰值显存占用

---

### Triton 与 FlashAttention

**背景问题**：标准 Attention 计算复杂度 O(n²)，内存访问效率极低。

**FlashAttention**（Stanford 2022）的核心思想：
- 不把整个 n×n 的 Attention 矩阵存到显存
- 分块（tiling）计算，每次只加载必要的数据到 SRAM
- 减少 HBM（高带宽内存）访问次数

**Triton**：OpenAI 开发的 GPU 编程框架，用 Python 写 CUDA Kernel。

```python
# Triton FlashAttention 伪代码结构
import triton
import triton.language as tl

@triton.jit
def flash_attn_kernel(Q, K, V, O, ...):
    # 分块加载 Q、K、V
    # 在线 Softmax（避免存储完整 Attention 矩阵）
    # 写回输出 O
```

**Assignment 2 目标**：用 Triton 实现一个简化版 FlashAttention，理解 IO-Aware 优化的核心思想。

---

### 分布式训练

**为什么需要分布式？**
- 大模型参数太多，单张 GPU 放不下
- 数据太多，单卡训练太慢

**两种主要并行策略**：

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| **Data Parallel（数据并行）** | 多卡各持一份模型，处理不同 batch | 模型能放入单卡 |
| **Tensor Parallel（张量并行）** | 单层参数拆分到多卡 | 单层太大，如 giant LLM |
| **Pipeline Parallel（流水线并行）** | 不同层放在不同卡 | 模型层数极多 |

**实际系统**：
- **Megatron-LM**（NVIDIA）：TP + PP + DP 混合
- **DeepSpeed**（Microsoft）：ZeRO 优化器分片
- **FSDP**（PyTorch 原生）：Fully Sharded Data Parallel

---

## 第三讲：Scaling Laws 详解（Assignment 3）

### Scaling Law 是什么？

**核心发现**（Kaplan et al., 2020；Chinchilla, 2022）：

> 模型性能（Loss）随**参数量 N**、**训练数据量 D**、**计算量 C** 呈幂律关系。

**公式（简化版）**：

```
L(N) = A·N^(-α) + B          # 参数量 Scaling Law
L(D) = C·D^(-β) + E          # 数据量 Scaling Law
L(C) = F·C^(-γ) + G          # 计算量 Scaling Law
```

其中 α ≈ 0.34（Chinchilla），β ≈ 0.28，γ ≈ 0.1

**关键洞察**：
- 参数量翻倍 → Loss 下降约 26%（不是线性）
- 存在**最优计算分配**：给定计算预算 C，如何分配 N 和 D？

---

### Chinchilla 最优分配

**Chinchilla（DeepMind, 2022）的结论**：

> 对于 7B 参数模型，应训练约 1.4T token，而非 300B。

**直觉**：大模型学得更快（每个参数看到更多数据），但训练成本更高。最优策略是**参数和数据同步增长**，而非只堆参数。

**实际意义**：
- 训练前先算最优 N 和 D
- 避免"参数浪费"（参数多但数据少，模型学不到东西）

---

### "查询 API 拟合 Scaling Law" 是什么意思？

Assignment 3 的一个任务：**不自己训练模型，而是调用 OpenAI / Anthropic API，获取不同模型在不同数据量下的 Loss，然后拟合 Scaling Law 参数**。

**流程**：
1. 调用 GPT-4、Claude 等 API，获取它们在标准 benchmark 上的表现
2. 收集不同参数量模型的公开数据（如 LLM Leaderboard）
3. 用最小二乘法拟合 L(N) = A·N^(-α) + B
4. 估计 A、B、α，验证是否符合幂律

**为什么这样做？**
- 训练大模型成本极高（百万美元级）
- API 数据是"免费"的观测样本
- 理解 Scaling Law 的**实证方法**，而非纯理论

---

### Transformer 各组件分析

Assignment 3 还会逐组件分析 Scaling 行为：

| 组件 | Scaling 特征 |
|------|-------------|
| **Embedding 层** | 参数量 ∝ vocab_size × d_model，随 d_model 线性增长 |
| **Attention 层** | 计算量 ∝ n² × d，是主要瓶颈 |
| **FFN 层** | 参数量 ∝ d_model × 4d，通常占模型 2/3 参数 |
| **LayerNorm** | 参数量极少，可忽略 |
| **Output Head** | 参数量 ∝ d_model × vocab_size |

**关键问题**：如果只放大某个组件（如只加 FFN 宽度），Loss 下降是否符合预期？

---

## 第四讲：Data 详解（Assignment 4）

### Common Crawl 是什么？

**Common Crawl** 是一个非营利组织，每月抓取整个互联网，公开存储为 WARC 格式。

**规模**：
- 每月约 20-30 TB 原始数据
- 包含网页文本、HTML、元数据
- 免费公开下载（AWS Open Data 计划）

**为什么用 Common Crawl？**
- 规模最大、最易获取的公开语料
- 几乎所有主流 LLM（GPT、LLaMA、Mistral）都基于它预训练

---

### 预训练数据处理流程

```
原始网页 → 去重 → 质量过滤 → 去毒 → Tokenization → 训练集
```

**每一步详解**：

**1. 去重（Deduplication）**
- 问题：互联网上大量重复内容（模板、CMS 系统）
- 方法：MinHash + LSH（局部敏感哈希）
- 效果：可去除 30-50% 重复数据

**2. 质量过滤（Quality Filtering）**
- 问题：很多网页是垃圾内容、广告、乱码
- 方法：
  - 基于规则：长度、符号比例、语言检测
  - 基于模型：用分类器打分（如 FastText 分类器）
- 参考：CCNet pipeline（Wenzek et al., 2020）

**3. 去毒（Detoxification）**
- 问题：互联网包含有毒内容（仇恨言论、暴力等）
- 方法：
  - 关键词过滤
  - 分类器识别
  - 人工抽检

**4. 数据配比（Data Mixing）**
- 问题：不同领域数据比例如何分配？
- 方法：经验配比（如 LLaMA：67% C4 + 15% Common Crawl + ...）
- 挑战：配比影响模型能力分布

---

### 关键概念：Pile、C4、RedPajama

| 数据集 | 来源 | 规模 | 特点 |
|--------|------|------|------|
| **C4** | Common Crawl 过滤版 | 750GB | Google 用于 T5，质量较高 |
| **The Pile** | 多源混合 | 825GB | EleutherAI，含学术论文、代码 |
| **RedPajama** | 复刻 LLaMA 数据 | 1.2T token | 完全开源，可复现 |
| **FineWeb** | Common Crawl 精滤 | 15T token | HuggingFace，当前最佳公开语料之一 |

---

## 第五讲：Alignment 详解（Assignment 5）

### 什么是 Alignment（对齐）？

**问题**：预训练后的 LLM 能生成连贯文本，但可能：
- 生成有害内容
- 不遵循用户指令
- 产生幻觉（Hallucination）

**Alignment 目标**：让模型行为符合人类意图和价值观。

---

### RLHF（Reinforcement Learning from Human Feedback）

**三步流程**：

```
预训练模型 → SFT → Reward Model → PPO 微调
```

**Step 1：SFT（Supervised Fine-Tuning）**
- 收集高质量指令-回答对
- 用标准交叉熵损失微调
- 得到"基础对齐"模型

**Step 2：Reward Model（奖励模型）**
- 收集人类偏好数据（A 回答比 B 回答好）
- 训练一个打分模型：r(x, y) → 标量分数
- 用 Bradley-Terry 模型建模偏好概率

**Step 3：PPO（Proximal Policy Optimization）**
- 用 Reward Model 的分数作为奖励
- PPO 更新策略，使高分回答概率增加
- 加入 KL 惩罚，防止偏离基础模型太远

---

### 其他 Alignment 方法

| 方法 | 核心思想 | 代表工作 |
|------|----------|----------|
| **DPO** | 直接优化偏好，无需 Reward Model | Direct Preference Optimization |
| **KTO** | 只需要二元反馈（好/坏） | Kahneman-Tversky Optimization |
| **Constitutional AI** | 用 AI 自身检查并修正输出 | Anthropic Claude |
| **RLAIF** | 用 AI 替代人类提供反馈 | Google |

---

### Safety（安全性）

**主要风险**：
- **Jailbreak**：绕过安全限制（如"忽略之前指令"）
- **Hallucination**：生成虚假信息
- **Bias**：输出歧视性内容
- **Privacy**：泄露训练数据中的个人信息

**缓解方法**：
- 红队测试（Red Teaming）
- 安全过滤器
- 输出验证层
- 持续监控与迭代

---

## 学习建议

### 按 Assignment 顺序推进

```
Assignment 1（Basics）
    ↓ 理解 Transformer 内部结构
Assignment 2（Systems）
    ↓ 知道如何高效训练
Assignment 3（Scaling）
    ↓ 理解模型大小与性能的关系
Assignment 4（Data）
    ↓ 知道数据如何影响模型
Assignment 5（Alignment）
    ↓ 让模型安全、有用
```

### 每个 Assignment 的学习节奏

1. **先读 Lecture 幻灯片**（lectures/ 目录）
2. **看 Assignment 说明**（assignments/ 目录 README）
3. **理解 starter code 结构**
4. **逐部分实现 + 测试**
5. **提交前运行测试套件**

---

## 参考资源

- [CS336 官方页面](https://stanford-cs336.github.io/spring2025/)
- [CS336 GitHub 组织](https://github.com/stanford-cs336)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)（可视化入门）
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [Chinchilla 论文](https://arxiv.org/abs/2203.15556)
- [LLaMA 论文](https://arxiv.org/abs/2302.13971)

---

*笔记持续更新中...*
