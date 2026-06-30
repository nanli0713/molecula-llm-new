# RL + LLM + EA 分子生成框架

这个项目实现的是一套多目标分子优化方法，核心思想不是只靠单一模型“直接生成更好分子”，而是把下面四个模块串成闭环：

1. `DQN` 负责选择“用什么方式生成分子”
2. `动态知识库` 负责持续维护高分/低分分子与片段
3. `LLM` 负责基于 prompt 提案新分子
4. `遗传算法` 负责交叉、变异和失败时的补充生成

最终，系统通过 `RewardEngine` 把多个评价信号合成 reward，再把这些 reward 送回 DQN 训练。这样不是简单地“生成分子”，而是一个持续自我更新的生成-筛选-学习闭环。

---

## 1. 一句话概括

输入当前分子种群和历史知识，先由 DQN 选择生成策略，再由 LLM 和遗传算子提出候选分子，随后用 oracle、毒性规则、片段统计和新颖性等多项信号计算 reward，最后把结果回写到高/低分知识库和 replay buffer 中，继续训练 DQN。

---

## 2. 代码结构

当前主线代码主要在下面两个目录：

| 路径 | 作用 |
| --- | --- |
| `RL+LLM+EA_before_talk_best_res_task3/main/molleo_multi/` | 主版本，不做 Pareto 前沿选择 |
| `RL+LLM+EA_before_talk_best_res_task3/main/molleo_multi_pareto/` | Pareto 版本，在 population 上额外做 Pareto front 选择 |

这两个目录的主方法论是一样的：`DQN + 动态知识库 + LLM + 遗传算法 + 多项 reward`。  
区别只是在 `molleo_multi_pareto` 里多了一步 `select_pareto_front(...)`。

如果只看一条主线，建议先读：

1. `main/molleo_multi/run.py`
2. `main/molleo_multi/reward.py`
3. `main/molleo_multi/rl.py`
4. `main/molleo_multi/GPT4.py` / `main/molleo_multi/biot5.py`

---

## 3. 整体流程

可以把整个方法看成下面这个循环：

```text
初始化分子池和片段池
    ↓
构建状态（父代 + 聚类中心 + 高/低分池）
    ↓
DQN 选择生成动作（prompt 策略）
    ↓
LLM 依据高/低片段知识生成候选分子
    ↓
遗传算法交叉/变异作为补充和兜底
    ↓
oracle / surrogate / toxicity / novelty / fragment reward 共同打分
    ↓
更新高分分子池、低分分子池、片段池、重复记忆
    ↓
写入 replay buffer
    ↓
训练 DQN
    ↓
进入下一轮
```

这个闭环的目标不是单纯让某一次生成变好，而是让“生成策略本身”在训练过程中越来越会选。

---

## 4. 动态知识库

动态知识库是这套方法里最关键的部分之一。它不是静态字典，而是每一轮都会更新的“分子经验库 + 片段经验库”。

### 4.1 分子级知识库

代码里维护了两类分子池：

- `high_score_mols`：高分分子池
- `low_score_mols`：低分分子池

初始化时通常从数据集中按分数排序，取前 10% 作为高分池，后 10% 作为低分池。  
后续每一轮根据新生成并通过 oracle 评估过的 offspring，再更新这两个池子。

更新逻辑的核心是：

- 高分池保留分数最高的一批
- 低分池保留分数最低的一批
- 去重时按 canonical SMILES 统一表示

同时还有一个全局重复记忆：

- `all_smiles`

它用于重复分子惩罚，防止模型不断产出已经见过的结构。

### 4.2 片段级知识库

分子不是直接拿来喂 prompt 的，而是先被拆成 BRICS 片段，再根据“高分池 / 低分池”中的频率统计形成片段知识库。

片段级知识库会做几件事：

1. 从高分分子中统计“正向片段”
2. 从低分分子中统计“负向片段”
3. 计算片段重要性，常见形式是类似：

   `importance = log((freq_high + eps) / (freq_low + eps))`

4. 对高分和低分片段分别生成上下文描述 `ctx`
5. 将上下文裁剪为前若干条，送入 prompt

在当前代码里，LLM 输入一般会使用：

- `res["high"]["ctx"][:10]`
- `res["low"]["ctx"][:10]`

这样做的目标不是让 prompt 变长，而是让 prompt 带着“当前任务最有效的局部化化学知识”。

### 4.3 历史记忆

除了当前高/低分池，代码还维护了历史片段/历史生成记录，例如：

- `high_frag_history`
- `past_generation_total`
- `past_generation_total_low`

这些历史信息的作用是：

- 减少 prompt 中重复使用同一类片段
- 避免 LLM 一直围绕同样的局部修饰做小变化
- 让知识库随轮次滚动，而不是静态不变

---

## 5. 强化学习是怎么做的

这里的强化学习不是拿 DQN 直接生成分子，而是让 DQN 学会“选哪种 prompt / 生成策略更合适”。

### 5.1 状态

当前状态由四部分拼接而成：

1. 父代分子 fingerprint
2. 父代对应的 cluster center 相关特征
3. 高分池特征
4. 低分池特征

当前实现里，状态向量维度固定为 `20480`。  
状态的含义可以理解成：当前正在什么化学区域里工作、周围有哪些高分/低分经验、应该倾向于哪类生成方式。

### 5.2 动作

当前动作空间是 4 个动作，语义大致是：

- `action = 0`：使用高分片段引导生成
- `action = 1`：使用低分片段约束生成
- `action = 2`：同时利用高分片段和低分片段
- `action = 3`：不强调片段知识，走更通用的生成模板

这几个动作最终会映射到不同的 prompt 模板上。

### 5.3 动作选择

动作选择不是纯随机，也不是单纯贪心，而是一个带探索机制的 `ActionSelector`：

- 先保证每个 cluster 的每个动作都至少尝试一定次数
- 然后使用 epsilon-greedy / UCB 风格的探索
- 选择时带有 cluster id

也就是说，DQN 不是在全局无差别地选动作，而是在“某个分子簇里”学习什么策略更有效。

### 5.4 DQN 结构与训练

当前实现使用的是 `DuelingQNetwork`：

- `Q(s, a)` 被拆成 state value 和 advantage 两部分
- 有 target network
- 有 replay buffer

Replay buffer 里存的是：

- `state`
- `action`
- `reward`
- `next_state`
- `done`

另外，代码对 reward 做了特殊处理：

- 一个 transition 里可能对应多个 offspring reward
- 训练时先用 mask 处理变长 reward
- 再用 top-k mean 聚合成一个标量 reward

这比“只把一个平均分数塞进 DQN”更稳，因为它保留了生成批次内部的高质量候选信号。

### 5.5 训练时机

当前代码里，DQN 训练不是等整个实验结束后再统一做，而是在线进行：

- replay buffer 足够大后就开始训练
- 每隔固定步数更新 target network

这样可以让 prompt 策略尽早跟上当前分子分布的变化。

---

## 6. RewardEngine 是怎么组成的

RewardEngine 的目标不是只看一个分数，而是把“目标分数、毒性、片段先验、新颖性、重复惩罚、相对提升”等信号统一起来。

当前 reward 的主要项包括：

### 6.1 主目标分数

`main_raw = float(main)`

也就是直接把 oracle 给出的主目标分数作为核心优化方向，而不是再转换成距离类惩罚。

### 6.2 毒性项

毒性由 RDKit 的几个规则库计算：

- `PAINS`
- `BRENK`
- `NIH`
- `ZINC`

对于每个分子，得到一个 toxicity probability，再与目标值比较形成惩罚项。

### 6.3 结构正负项

这一项来自动态片段知识库：

- 高分池里常见的片段会形成正向奖励
- 低分池里常见的片段会形成负向奖励

本质上是把 BRICS 片段的高低频差异转成结构先验。

### 6.4 新颖性项

reward 还会看新分子离 cluster center 有多远：

- 既不能太像已有簇中心
- 也不能偏离到完全无意义的区域

因此代码里使用了：

- `cutoff_new_cluster`
- `novelty_band`

来平衡“探索”和“落在合理化学区域内”。

### 6.5 相对提升项

除了绝对分数，还看相对提升：

- 相对父代是否提升
- 是否优于当前 top-k 的平均水平

这能避免 reward 只奖励绝对高分、但忽略了“当前这一步是否真的比父代更好”。

### 6.6 重复惩罚

如果分子已经在 `all_smiles` 中出现过，就会被惩罚。  
这个机制能减少模型反复生成已知高分模板。

### 6.7 最终 reward

所有 term 会先做 running normalization，再加权求和，最后通过 `tanh` 做压缩，得到最终 reward。

直观上可以理解成：

```text
final_reward = tanh(
    w_main * main
  + w_tox * tox
  + w_struct_pos * struct_pos
  - w_struct_neg * struct_neg
  + w_novelty * novelty
  + w_new_cluster * new_cluster
  + w_relative * relative
  - w_duplicate * duplicate
)
```

其中还会叠加 fragment hard penalty 和 novelty hard penalty。

---

## 7. LLM + 遗传算法是怎么结合的

### 7.1 GPT-4 分支

`main/molleo_multi/GPT4.py` 负责根据 prompt 直接生成 SMILES。

prompt 的输入由三部分组成：

1. 父代分子及其分数
2. 高分片段上下文
3. 低分片段上下文

输出解析时会尽量从固定格式中提取 SMILES，如果第一次解析失败，会再 retry 一次，尽量把可用候选挖出来。

### 7.2 BioT5 分支

`main/molleo_multi/biot5.py` 使用 SELFIES 作为输入输出表示。

它的基本流程是：

1. 把 SMILES 转成 SELFIES
2. 把高/低片段上下文拼进 prompt
3. 用 beam search 生成多个候选
4. 再把 SELFIES 解码回 SMILES
5. 只保留合法分子

### 7.3 遗传算法的作用

遗传算法在这里不是替代 LLM，而是作为两层补充：

- `crossover`
- `mutation`

当 LLM 输出失败、输出非法、或者候选不足时，遗传操作可以兜底补足；  
同时，EA 也负责保留“结构连续性”，避免 LLM 每次都跳得太远。

这就是这个项目里所谓的 `LLM + EA` 混合生成方式。

---

## 8. 为什么要做动态知识库 + DQN

这个设计的出发点很直接：

1. 只用纯 prompt，知识利用是静态的，容易重复同类修饰
2. 只用 EA，探索强，但缺少对“当前任务应该用哪类知识”的策略学习
3. 只用 reward 不做策略学习，生成策略不会随任务区域变化

所以这里把三者合并：

- 动态知识库提供“经验”
- DQN 决定“怎么用经验”
- LLM/EA 负责真正产生候选结构

这样方法论上是一个闭环系统，而不是几个互不相干的模块并排放着。

---

## 9. `molleo_multi` 和 `molleo_multi_pareto` 的区别

这两个目录的共同点是：

- 都有 DQN
- 都有动态高/低分分子池
- 都有动态高/低分片段池
- 都有 LLM + EA 混合生成
- 都有 RewardEngine 多项 reward

区别是：

- `molleo_multi`：保留原始 population 选择流程，不做 Pareto front 选择
- `molleo_multi_pareto`：在 population 更新后，会额外做一次 Pareto front selection

所以如果你研究的是“方法论本身”，可以把两个目录看成同一框架的两个实现分支；  
如果你研究的是“是否加入 Pareto front 对结果有何影响”，那就看 `molleo_multi_pareto`。

---

## 10. 常见结果指标怎么理解

日志里常见的指标包括：

- `avg_top1`
- `avg_top10`
- `avg_top100`
- `auc_top1`
- `auc_top10`
- `auc_top100`
- `avg_sa`
- `diversity_top100`
- `n_oracle`

它们的含义大致是：

- `avg_topk`：当前时刻种群里前 k 个分子的平均分
- `auc_topk`：前 k 个分子分数随迭代累积的面积，更看重“持续优化能力”
- `avg_sa`：平均可合成性
- `diversity_top100`：前 100 个候选的多样性
- `n_oracle`：oracle 调用次数

如果你主要关心“训练策略是否真的起作用”，通常要同时看：

1. `auc_top10`
2. `avg_top10`
3. `diversity_top100`
4. `n_oracle`

因为单看某一轮的峰值分数，很容易误判整个方法到底是“真的更强”还是“偶然撞到了高分分子”。

---

## 11. 这份代码最核心的设计点

如果只记住几句话，可以记住下面这几个：

1. 不是纯 prompt，而是 `DQN` 在选 prompt
2. 不是静态知识库，而是高/低分分子池和片段池都在动态更新
3. 不是单一 reward，而是多项 reward 共同决定学习信号
4. 不是只靠 LLM，而是 `LLM + crossover + mutation` 的混合生成
5. 不是一次性优化，而是“生成-评估-记忆-训练”的闭环

这就是这个项目的主要方法论。

