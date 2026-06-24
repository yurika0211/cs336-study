## 一: 线性变换

直觉是对的：y = Σ w_i · x_i，就是每个输入乘一个系数再加起来。

nn.Linear(in_features, out_features) 内部有一个参数 weight，shape 是 [out_features, in_features]。

```text
output = input @ weight.T   # [batch, in] @ [in, out]^T → [batch, out]
```

这里就是说，pytorch自动帮我管理所有的参数，也就是权重，调用linear()函数，就可以直接完成矩阵的乘法，并且返回一个结果。


## 二: 词嵌入

Tokenizer: "你好世界" → [token_id_1, token_id_2, token_id_3]

Embedding: 查表 → [vector_1, vector_2, vector_3]

> Tokenizer 把文本切成"词块"，每个词块对应一个整数 ID，Embedding 矩阵就是一个"词典"：embedding[ID]= 该 token 的向量表示，这个矩阵就是权重，训练过程中会被不断更新。

Shape 说明：
• embedding矩阵：[vocab_size, d_model]，比如[10000, 512]

> 就是说，第一个是这个向量的维度，第二个是模型本身的维度，这里的运算就是将维度做一个换算

• 输入 token IDs：[batch, seq_len]，比如[4, 10]

> 这个部分，给定了对应的batch的就是token的id，和整个字段的长度，就是告诉我们采样的长度

• 输出：[batch, seq_len, d_model]=[4, 10, 512]

> 这是把token对应的id和输出字段的长度都进行加载了吗？我还是有点小疑问在这里，每个参数代表的具体含义是什么？

## 三: SwiGLU 激活函数

> 不是所有的神经元都应该被激活，SwiGLU用内控来控制信息流动

```
SwiGLU(x) = (x @ W₁) ⊙ SiLU(x @ W₃)  然后  @ W₂
• W₁：产生"候选值"（gate 的输入）
• W₃：产生"门控信号"（0~1 之间的权重，由 SiLU 控制）
• ⊙：逐元素相乘（element-wise multiply）
• W₂：把中间结果投影回输出维度
```
> 这里的意义就是，先对输入变量进行W1参数化，然后对输入变量进行w3参数化之后用SiLU函数激活，得到的结果进行点积，最后再对整体进行一下W2参数化。

## 四: 注意力机制

```
Q = X @ W_Q    # 查询矩阵：我要找什么
K = X @ W_K    # 键矩阵：我有什么
V = X @ W_V    # 值矩阵：我的内容是什么

scores = Q @ K.T / sqrt(d_k)    # 缩放点积
scores = masked_softmax(scores)  # 加 mask 后 softmax
output = scores @ V              # 加权求和
```

Mask 是什么？

> 训练时，模型不能"看到"未来的 token。Mask 是一个上三角矩阵（未来位置设为 -inf），softmax 后这些位置权重变为 0，实现因果注意力。

Softmax 公式

> softmax(x_i) = exp(x_i) / Σ exp(x_j),把原始分数变成概率分布（和为 1）。

多头注意力（Multi-Head）

> Q 被拆成 [batch, num_heads, seq_len, d_k_per_head],就是把 d_model 维度拆成 num_heads 份，每份独立做一次 SDPA，最后拼接回来。好处： 不同头可以关注不同的语义模式（比如一个头关注语法，一个头关注语义相似性）。

SDPA（Scaled Dot-Product Attention）

>  就是上面那个完整的 Q@K^T/√d → softmax → @V 的过程。

五: RoPE旋转位置编码

> RoPE 的核心思想：把位置信息"拧"进向量里，而不是加进去。

```
对向量中相邻的一对维度 (x_{2i}, x_{2i+1})，乘以一个旋转矩阵：

[x_{2i}  ]   [cos(m·θ_i)  -sin(m·θ_i)] [x_{2i}  ]
[x_{2i+1}] = [sin(m·θ_i)   cos(m·θ_i)] [x_{2i+1}]
• m= token 的位置（第几个 token）
• θ_i = 1 / (base^(2i/d_model))，base通常取 10000
• i= 维度对的下标（第几对）
```

>base 是什么？ 它是一个"频率控制参数"。base 越大，不同维度对的旋转速度差异越小，模型能编码更长距离的相对位置。

旋转矩阵为什么有这个性质，两个token的Q和K做内积的时候，

> 这里和q相关的就是m位置出的Query向量，在注意力机制里面，这个就是对这里发出查询的信号，k相关的就是在位置n处的key向量，代表第n个token的身份标识的向量，这里的点积就是说两个向量之间的近似度，是标准注意力的注意力分数，值越大，就是两者越相关，然后有个R的旋转矩阵，角度由位置差决定，本身这里是一个正交矩阵，，回忆一下什么是正交矩阵，就是矩阵的转置和逆相同，就是下面的公式：

$$q_m^T k_n = (R_m q_0)^T (R_n k_0) = q_0^T \underbrace{R_m^T R_n}_{R_{m-n}} k_0$$

就是这里变成了和相对位置相关的内积。

## 六: 梯度裁剪

这里裁剪可以防止梯度爆炸：

```
norm = sqrt(Σ g_i²)    # L2 范数，所有参数的梯度平方和开根号
if norm > threshold:
g_i = g_i * (threshold / norm)    # 整体缩放
```

L2范数对应的是欧几里得距离，就是梯度向量在参数空间中的长度，超过max就说明这一步走的太大了。

## 七: Adam优化器和权重衰减

Adam是维护每个参数的动量和方差，一个是一阶矩，一个是二阶矩。可以自适应调整每个参数的学习率，参数两更新大而且稳定的维度，学习率自动变小。

权重衰减（Weight Decay）:

不是简单的L2正则化，而是直接对权重值做衰减

$$w \leftarrow w - \text{lr} \cdot (\underbrace{\nabla_w L}_{\text{grad}} + \underbrace{\lambda w}_{\text{惩罚}})$$

这里是更新的规则，w是当前的参数，lr是学习率，grad是损失函数对于w的梯度，lambda是l2正则化系数，这里存在梯度下降和L2惩罚两重，呀u你是的损失是：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \|w\|^2$$

求导之后得到：

$$\nabla_w \mathcal{L}_{\text{total}} = \underbrace{\nabla_w \mathcal{L}_{\text{data}}}_{\text{grad}} + \underbrace{\lambda w}_{\text{惩罚项}}$$

## 八: 余弦学习率调度

预热阶段：lr = initial_lr * (step / warmup_steps)

余弦退火阶段：lr = final_lr + (initial_lr - final_lr) * 0.5 * (1 + cos(π * (step - warmup_steps) / (total_steps - warmup_steps)))

就是两种学习率的变化，这边还是很重要的，用的是一个余弦的变化函数：

```
• 预热：训练初期梯度不稳定，小学习率起步，线性增长到初始值
• 余弦退火：学习率沿余弦曲线从initial_lr降到final_lr（通常为 0）
• 相比线性衰减，余弦衰减在后期下降更慢，给模型更充分的微调时间
```

## 九: 序列化/Checkpoint

```python
torch.save({
    'model': model.state_dict(),      # 模型权重
    'optimizer': optimizer.state_dict(),  # 优化器状态（动量等）
    'epoch': epoch,                    # 当前训练轮数
    'rng_state': torch.get_rng_state(), # 随机数状态（可复现）
}, 'checkpoint.pt')
```

**加载：**

checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

`state_dict` 就是一个 Python 字典，把每个参数的 tensor 以名字为 key 存起来。

## 十: BPE Tokenizer

1. 把文本拆成单个字符（字节）作为初始词表
2. 统计所有相邻字符对的频率
3. 合并频率最高的字符对，加入 merges 列表
4. 重复步骤 2-3，直到达到预设的 merge 次数
5. 编码时：按 merges 列表的顺序， greedily 合并字符对






