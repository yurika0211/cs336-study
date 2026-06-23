# Stanford CS336 — Language Modeling from Scratch (Spring 2025)

## 课程主页
https://stanford-cs336.github.io/spring2025/

## 课程定位
从零构建语言模型，类比 OS 课程从零写操作系统。覆盖数据采集/清洗、Transformer 构建、训练、评估、部署全流程。

## 授课教师
- Percy Liang
- Tatsu Hashimoto

## 学分
5-unit（非常重，代码量是其他 AI 课的 10 倍以上）

## 前置要求
- **Python 精通**：几乎不给脚手架代码
- **深度学习框架**：熟悉 PyTorch
- **系统优化**：内存层次、GPU 编程
- **数学**：微积分、线性代数（MATH 51 / CME 100）
- **概率统计**：CS 109 或同等水平
- **机器学习**：CS 221/229/230/124/224N 或同等水平

## 作业

### Assignment 1 — Basics
- Repo: https://github.com/stanford-cs336/assignment1-basics/tree/main
- Leaderboard: https://github.com/stanford-cs336/assignment1-basics-leaderboard/tree/master
- 内容：实现 Tokenizer、Transformer 架构、Optimizer，训练一个最小语言模型

### Assignment 2 — Systems
- Repo: https://github.com/stanford-cs336/assignment2-systems/tree/main
- Leaderboard: https://github.com/stanford-cs336/assignment2-systems-leaderboard/tree/main
- 内容：Profiling、Benchmarking、用 Triton 实现 FlashAttention2、分布式训练

### Assignment 3 — Scaling
- Repo: https://github.com/stanford-cs336/assignment3-scaling/tree/main
- 内容：理解 Transformer 各组件功能、拟合 Scaling Law

### Assignment 4 — Data
- Repo: https://github.com/stanford-cs336/assignment4-data/tree/main
- Leaderboard: https://github.com/stanford-cs336/assignment4-data-leaderboard
- 内容：Common Crawl 原始数据清洗、过滤、去重

### Assignment 5 — Alignment
- Repo: https://github.com/stanford-cs336/assignment5-alignment
- 内容：SFT/RLHF 对齐

## 课程表

| 周次 | 日期 | 主题 | 讲师 | 讲义 |
|------|------|------|------|------|
| 1 | 4/1 | Introduction | Percy | lecture_1.pdf |
| 2 | 4/3 | Tokenization | Percy | lecture_2.pdf |
| 3 | 4/8 | Transformer Architecture | Percy | lecture_3.pdf |
| 4 | 4/10 | Optimization | Percy | lecture_4.pdf |
| 5 | 4/15 | Systems I | Tatsu | lecture_5.pdf |
| 6 | 4/17 | Systems II | Tatsu | lecture_6.pdf |
| 7 | 4/22 | Data I | Percy | lecture_7.pdf |
| 8 | 4/24 | Data II | Percy | lecture_8.pdf |
| 9 | 4/29 | Parallelism | Tatsu | lecture_9.pdf |
| 10 | 5/1 | Scaling | Tatsu | lecture_10.pdf |
| 11 | 5/6 | Scaling Details | Tatsu | lecture_11.pdf |
| 12 | 5/8 | Evaluation | Percy | lecture_12.py |
| 13 | 5/13 | Data | Percy | lecture_13.py |
| 14 | 5/15 | Data | Percy | lecture_14.py |
| 15 | 5/20 | Alignment - SFT/RLHF | Tatsu | lecture_15.pdf |
| 16 | 5/22 | Alignment - RL | Tatsu | lecture_16.pdf |
| 17 | 5/27 | Alignment - RL | Percy | lecture_17.py |
| 18 | 5/29 | Guest Lecture: Junyang Lin | - | - |
| 19 | 6/3 | Guest Lecture: Mike Lewis | - | - |

## 讲义 PDF 地址（GitHub）
- Lecture 1: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%201%20-%20Introduction.pdf
- Lecture 2: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%202%20-%20Tokenization.pdf
- Lecture 3: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%203%20-%20Transformer%20Architecture.pdf
- Lecture 4: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%204%20-%20Optimization.pdf
- Lecture 5: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20Systems%20I.pdf
- Lecture 6: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%206%20-%20Systems%20II.pdf
- Lecture 7: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%207%20-%20Data%20I.pdf
- Lecture 8: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%208%20-%20Data%20II.pdf
- Lecture 9: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%209%20-%20Parallelism.pdf
- Lecture 10: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%2010%20-%20Scaling.pdf
- Lecture 11: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf
- Lecture 15: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf
- Lecture 16: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf