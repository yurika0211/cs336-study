# CS336: Language Modeling from Scratch — 学习仓库

Stanford CS336 (Spring 2025) 课程资料整理，用于系统学习。

## 目录结构

```
cs336-study/
├── lectures/                    # 讲课幻灯片 PDF
│   ├── 2025 Lecture 3 - architecture.pdf
│   ├── 2025 Lecture 4 - MoEs.pdf
│   ├── 2025 Lecture 5 - GPUs.pdf
│   ├── 2025 Lecture 7 - Parallelism basics.pdf
│   ├── 2025 Lecture 9 - Scaling laws basics.pdf
│   ├── 2025 Lecture 11 - Scaling details.pdf
│   ├── 2025 Lecture 15 - RLHF Alignment.pdf
│   └── 2025 Lecture 16 - RLVR.pdf
├── assignments/                 # 作业仓库（Git Submodule）
│   ├── assignment1-basics       # 基础：Transformer 实现
│   ├── assignment2-systems      # 系统：训练框架
│   ├── assignment3-scaling      # 扩展：Scaling Laws
│   ├── assignment4-data         # 数据：数据处理与预处理
│   └── assignment5-alignment    # 对齐：RLHF / Safety
├── README.md                    # 课程概览
└── course_page.html             # 原始课程页面
```

## 快速开始

```bash
# 克隆仓库（含 submodule）
git clone --recursive <repo-url> cs336-study
cd cs336-study

# 或已克隆后初始化 submodule
git submodule update --init --recursive

# 安装作业依赖（以 assignment1 为例）
cd assignments/assignment1-basics
uv sync
```

## 课程链接

- 官方页面：https://stanford-cs336.github.io/spring2025/
- GitHub 组织：https://github.com/stanford-cs336

## 学习进度

| 作业 | 主题 | 状态 |
|------|------|------|
| assignment1-basics | Transformer 基础实现 | ⬜ |
| assignment2-systems | 训练系统 | ⬜ |
| assignment3-scaling | Scaling Laws | ⬜ |
| assignment4-data | 数据处理 | ⬜ |
| assignment5-alignment | RLHF / Safety | ⬜ |
