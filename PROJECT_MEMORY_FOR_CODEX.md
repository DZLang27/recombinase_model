# 重组酶–DNA 匹配排序建模：Codex 项目总记忆

> **2026-10-09 最高优先级更新（覆盖下文中尚未同步的旧描述）**
>
> 本项目已按后续讨论从零实现为当前目录中的 **RecRank V1**。权威可执行定义是 `configs/v1.yaml`、`src/rec_rank/` 与 `README.md`。下文仍有少量 2026-10-08 旧状态，仅作为历史背景；若与本段冲突，以本段和当前代码为准。
>
> 1. 训练集不再拆 validation；每个 epoch 直接在测试集完整候选库评估，并以 `Recall@10 → MRR → 更低 Median Rank → 更早 epoch` 选择 `best.pt`。这使测试集承担开发/模型选择功能，结果不表述为未触碰的无偏最终测试估计。
> 2. `gamma` 不再使用 `exp(log_gamma)`、`gamma_max=100` 或硬 clamp。当前为 `softplus(raw_gamma)+epsilon`，另有可学习 bias；`raw_gamma` 使用更小学习率和更高 weight decay，并完整记录 gamma 曲线。
> 3. AMP 训练在 `unscale_` 后、optimizer step 前执行 `clip_grad_norm_(..., 1.0)`，记录裁剪前梯度范数与裁剪比例，并检测 NaN/Inf。
> 4. 每轮保存可恢复的 `last.pt`，按上述检索指标保存 `best.pt`；checkpoint 包含 optimizer、GradScaler、RNG、配置及数据/cache 指纹。
> 5. 报告完整测试集指标，并辅助报告方向无关的新 DNA 子集（当前 3,270 条）。反向互补仅用于该审计；V1 不做序列增强。
> 6. 当前统一 CLI：`prepare / embed / train / evaluate / export / build-index / screen / report`。使用两层 cache：与投影无关的 backbone cache，以及与 checkpoint 和候选库指纹绑定的投影 candidate index。
> 7. V1 仍严格不使用局部识别区域、attP/attB 分类、center_common、CLIP 批内负样本、InfoNCE、ranking loss、hard-negative mining、局部精排或 backbone 微调。

> **文档用途**：这是给 Codex / 新接手的代码助手使用的“从零接管”项目记忆。假设接手者此前完全不知道本项目，必须先读完本文件，再阅读代码，再做任何修改。
>
> **状态日期**：2026-10-09
>
> **当前阶段**：正式模型 V1 已在本项目目录完成重构实现、真实数据 prepare 审计及合成 embedding 全链路自测；尚未在 GPU 计算节点完成真实 ESM-2 + Nucleotide Transformer embedding 预计算和正式 V1 训练。
>
> **权威性原则**：本文件中“当前已确认决策 / 不可擅改规则”优先于历史聊天中更早的设想。历史方案仅作为背景，不得自动恢复。

---

## 0. 一句话理解这个项目

用户要做的不是“判断一条 DNA 属于哪一类”，也不是“设计新蛋白”，而是：

> **输入一条目标 DNA，在一个候选重组酶蛋白数据库中，为每条具体重组酶计算兼容性分数，按分数进行稠密排名（dense rank），最后推荐 TOP10。**

当前项目只做 **检索 / 排序（retrieval / ranking）**。未来模型成熟后才可能扩展到蛋白设计或靶点改造；当前不要把范围扩到设计。

所有模型、数据处理、baseline、评估和代码迭代都必须服务于这个最终目标。

---

# 1. 项目最终目标与任务边界

## 1.1 输入

一条 DNA 序列，代表用户希望寻找可匹配重组酶的目标位点。

DNA 可以与某个重组酶的 **attP** 或 **attB** 之一匹配。并不要求输入 DNA 同时包含或同时匹配 attP 和 attB。

## 1.2 候选库

候选是**具体重组酶蛋白序列**。

最终模型应能够给训练阶段未出现过的候选蛋白打分，只要该蛋白能被 protein backbone 编码。

当前独立测试中，候选库由测试集中的 unique protein sequences 构成，共 **1,287 条**。

## 1.3 输出

对所有候选蛋白计算 compatibility score（兼容性分数），当前正式模型 V1 使用投影后的 **cosine similarity（余弦相似度）**。

然后：

1. 分数从高到低排序；
2. 使用 **dense rank（稠密排名）**；
3. 同分候选共享同一名次；
4. TOP10 的定义是 **rank <= 10**，不是机械截取 10 条；
5. 因此存在并列时，TOP10 可以超过 10 个蛋白。

## 1.4 严格目标层级

最终目标是**具体蛋白序列级（exact protein sequence level）**的重组酶推荐。

`recombinases_clu_id` 等 cluster 信息可以做辅助诊断，但**不能替代具体蛋白级评价作为主结果**。用户之前明确反对“只算 cluster 命中就算成功”。

---

# 2. 用户已经准备好的数据

## 2.1 原始文件

项目已有：

- `train_dataset_with_neg.txt`
- `test_dataset_with_neg.txt`
- `train_DNA.fasta`
- `test_DNA.fasta`

当前正式模型的数据源以两个 TSV/TXT 文件为准；FASTA 主要是历史 baseline / 辅助用途。

用户已经完成 train/test 的划分和验证：

> **不要重新划分 train/test。**

用户进一步明确：

> **不要再从训练集拆 validation set（验证集）。训练集就是训练集，测试集用于测试。**

因此 V1 不做 validation split，不做基于 validation 的 early stopping（早停），也不做 checkpoint selection（检查点选择）。

如果未来因论文严格性需要独立 hold-out，应先与用户讨论；不要擅自创建。

---

# 3. 原始表结构与字段语义

两个文件表头一致：

```text
attB
attP
recombinases
attB_str
attP_str
recombinases_str
center_common
resource
attB_clu_id
attP_clu_id
recombinases_clu_id
attB_neg
attP_neg
```

关键字段解释：

| 字段 | 含义 | 当前 V1 是否使用 |
|---|---|---|
| `attB` | attB 记录 ID | 追溯用途 |
| `attP` | attP 记录 ID | 追溯用途 |
| `recombinases` | 重组酶记录 ID | 追溯用途 |
| `attB_str` | 已观察的 attB DNA 序列 | **正样本** |
| `attP_str` | 已观察的 attP DNA 序列 | **正样本** |
| `recombinases_str` | 重组酶蛋白序列 | **核心输入** |
| `center_common` | 数据注释 | V1 不用于建模 |
| `resource` | 数据来源 | V1 不用于建模 |
| `attB_clu_id` | attB cluster | 可辅助分析 |
| `attP_clu_id` | attP cluster | 可辅助分析 |
| `recombinases_clu_id` | 蛋白 cluster | 可辅助分析，非主评价 |
| `attB_neg` | 已提供的 attB 负 DNA | **负样本** |
| `attP_neg` | 已提供的 attP 负 DNA | **负样本** |

---

# 4. 本项目最重要的数据语义

这是 Codex 最容易犯错的地方，必须严格遵守。

## 4.1 一个重组酶对应两个合法正 DNA

对于同一个蛋白 `P_i`：

```text
P_i + attP_i      -> positive (1)
P_i + attB_i      -> positive (1)
P_i + attP_neg_i  -> negative (0)
P_i + attB_neg_i  -> negative (0)
```

attP 和 attB **不是两个互斥类别**。

它们是同一个重组酶的两个合法识别位点。

正式模型学习的是：

```text
f(protein, DNA) -> compatibility
```

不是：

```text
这是 attP 还是 attB？
```

## 4.2 未标注 protein–DNA pair 不是负样本

如果数据库没有说明某个蛋白和某条 DNA 是正还是负，则该组合是：

```text
unknown / unlabeled
```

它既不能自动当正样本，也**绝不能自动当负样本**。

因此监督关系应理解为稀疏矩阵：

```text
                 Protein1  Protein2  Protein3
attP_1               +        ?         ?
attB_1               +        ?         ?
attP_neg_1           -        ?         ?
attB_neg_1           -        ?         ?

attP_2               ?        +         ?
attB_2               ?        +         ?
attP_neg_2           ?        -         ?
attB_neg_2           ?        -         ?
```

`?` 必须 mask / ignore，不进入 loss。

## 4.3 不能使用普通 CLIP 的“batch 非对角 = negative”假设

普通 CLIP-style contrastive learning（CLIP 式对比学习）通常假设一个 batch 中只有对角线是正样本，其余跨样本组合都是负样本。

本项目不能这样做。

例如：

```text
P1 + attP1 = positive
P2 + attP2 = positive
```

不能因为 P2 和 attP1 恰好在同一 batch 就自动得到：

```text
P2 + attP1 = negative
```

数据库没有这个标签时，它必须保持 unknown。

## 4.4 正负样本当前不做置信度分级

历史上讨论过：teacher negative 是否应该降低权重。

用户已经明确否定 V1 中这样做：

> **数据库中已经给出的正样本和负样本先正常使用，不区分置信度，不做 `w_neg < w_pos`。**

所以当前：

```text
positive weight = 1
negative weight = 1
```

V1 使用未加权的 `BCEWithLogitsLoss`。

## 4.5 数据是多对多关系

必须支持：

- 一条 DNA 可能对应多个蛋白；
- 一个蛋白可能对应多条 attP / attB；
- 同一个 protein sequence 可能在多个原始 row 中重复出现；
- ID 不等价于 sequence；不同 ID 可以出现完全相同 protein sequence。

因此不能把“行号”或“原始 recombinase ID”直接当模型实体主键。

---

# 5. 数据规范化与稳定主键

当前 V1 已实现以下规则。

## 5.1 蛋白序列

```text
去空白 -> 大写
```

允许标准氨基酸和少量 `X`；ESM-2 可处理 `X`。

unique protein identity 以**规范化后的完整蛋白序列**为准。

## 5.2 DNA 序列

顺序很重要：

1. 去空白；
2. 大写；
3. `U -> T`；
4. **先检查是否包含 `FAILED`**；
5. 如果不是失败哨兵，再把 IUPAC 模糊碱基 `RYSWKMBDHV` 映射为 `N`。

原因：早期实现曾先做 IUPAC 映射，导致 `FailedFailed` 中的字母被改变，进而错误地把失败记录当成合法 DNA。该 bug 已修复。

训练集只有 **7 条 DNA** 含 `K/R/S/W` 等模糊碱基；测试 DNA 没有这类问题。

Nucleotide Transformer 词表支持 `N`，所以模糊碱基统一映射到 `N`。

## 5.3 stable key

当前代码对规范化序列使用 SHA1 的前 20 个 hex 字符生成稳定 key：

```text
P_<sha1-prefix>   # protein
D_<sha1-prefix>   # DNA
```

具体实现见 `src/rec_rank/data.py::stable_key()`。

不要用 Python 内置 `hash()` 作为持久 ID，因为它不是跨进程稳定标识。

---

# 6. 数据审计的当前权威结果

## 6.1 训练集

原始训练表：

- **49,912 行数据**（不含 header）。
- 原始数据中约有 **37,262 条 unique protein sequences**。
- 其中 **45 条 unique protein sequence** 只出现在没有可用显式 DNA 监督的失败记录中，因此不进入 V1 sampler。

V1 数据准备后：

- `train_unique_proteins = 37,217`
- `train_unique_dnas = 179,059`
- `train_category_memberships = 198,099`
- `train_unique_labeled_pairs = 197,911`
- `train_conflicting_pairs_removed = 1`

说明：

- category membership 指 pair 在 attP+/attB+/attP-/attB- 某一类别中的成员关系；
- 同一个 protein–DNA pair 可能在不同原始行/类别中重复，所以 category membership 数量可以大于 unique labeled pair 数量；
- 当前 V1 最终用于监督的 unique labeled pair 是 **197,911**。

### 唯一明确正负冲突

完整审计只发现 **1 个相同 protein–DNA pair 同时有正标签和负标签的冲突**。

历史追溯定位到原始：

- `source_row = 14993`：observed positive
- `source_row = 19369`：teacher / explicit negative

当前处理原则：

> **只移除这个冲突 protein–DNA pair 的监督，不删除两条原始 row 中其他合法信息。**

不要自行决定“正样本覆盖负样本”或“负样本覆盖正样本”。

## 6.2 测试集

原始测试文件：

- 2,171 行数据（不含 header）；
- 其中 4 条记录的相关 DNA 为 `FailedFailed` 等无效内容；
- 可形成约 2,167 条有效记录；
- 每条正常记录有 attP 和 attB 两个 query，因此历史上曾表述为 **4,334 个有效 query instances（查询实例）**。

但最终评估不能按重复实例重复加权。

去重后当前权威测试数据为：

- `test_unique_queries = 3,672` 条 unique DNA sequence；
- `test_unique_candidates = 1,287` 条 unique candidate protein sequence；
- `test_truth_pairs = 4,143` 个已知 positive DNA–protein pair；
- candidate coverage = **100%**，即所有测试 truth 中的正蛋白都存在于 1,287 条候选蛋白库。

多正样本结构：

- 3,467 条 DNA 只有 1 个已知正蛋白；
- 205 条 DNA 有多个已知正蛋白；
- 因此评价必须支持 multi-positive retrieval（多正样本检索）。

---

# 7. 评估协议：必须保持一致

## 7.1 候选空间

对每一条测试 DNA，必须在**同一个完整候选蛋白库**中打分。

当前 test：

```text
3672 DNA queries × 1287 candidate proteins
```

形成完整 cosine score matrix。

## 7.2 Dense Rank（稠密排名）

按 cosine score 从高到低：

```text
score: 0.90, 0.90, 0.80, 0.70, 0.70
rank :    1,    1,    2,    3,    3
```

TOP-K：

```text
rank <= K
```

不是 `argsort()[:K]`。

因此 TOP10 可能多于 10 条。

### 已修复的历史 bug

V1 最初的 `evaluation.py` 已按 dense rank 计算指标，但 `rank_test.py` 导出 TOP10 时曾用普通 `argsort[:10]`，导致并列截断和普通序号。

2026-10-08 已修复：

- 导出也使用 dense rank；
- 同分共享名次；
- TOP-K 按 `rank <= K`；
- 新增回归测试；
- 当前自动测试 **6/6 通过**。

Codex 不要把它改回普通 top-k slicing。

## 7.3 Multi-positive 指标

如果一条 query DNA 有多个已知正蛋白：

```text
truth(D) = {P1, P2, ...}
```

当前 Recall/MRR 使用**最佳已知正蛋白的 dense rank**：

```text
best_rank(D) = min(rank(P) for P in truth(D))
```

指标：

- `Recall@1`
- `Recall@5`
- `Recall@10` —— 当前最重要主指标，因为最终产品输出 TOP10
- `MRR`（Mean Reciprocal Rank，平均倒数排名）
- `Median Rank`（中位排名）

其中：

```text
Recall@K = fraction(best_rank <= K)
MRR = mean(1 / best_rank)
```

## 7.4 Cluster-level 评价

cluster-level 可以作为辅助分析，例如判断模型是否至少找到了正确家族，但：

> **不能把 cluster Recall 当成最终主结果。**

最终目标要求准确到具体重组酶 / 蛋白序列。

---

# 8. Baseline 的历史与当前定位

## 8.1 最早 BLASTn baseline

曾实现 / 讨论的 DNA similarity baseline：

1. 训练集 `attP + attB` 合并建 BLASTn 库；
2. 测试集 attP / attB 分别做 query；
3. `-evalue 1000`；
4. `max_target_seqs` 设置到库规模；
5. 每个候选取最大 `bitscore`；
6. 按 score 做 dense rank；
7. 短 DNA 曾考虑使用 `blastn-short`，普通序列用 `blastn`；
8. 历史环境：BLAST+ **2.17.0**。

注意曾踩过 `-outfmt "?"` 报错；正式 outfmt 应明确列出字段，例如：

```text
6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
```

## 8.2 为什么 BLASTn-only 不能作为严格的最终 baseline

最初 BLASTn baseline 的候选本质来自训练集 DNA/recombinase 映射。

如果测试集中的具体重组酶蛋白在训练库中根本不存在，那么 exact-protein Recall 必然是 0，即使 DNA 找到了相似 cluster。

这说明：

> 该 baseline 的候选空间与最终任务“对给定候选蛋白数据库排序”并不完全一致。

所以它最多作为 DNA 最近邻 / sanity check（合理性检查），不能被包装成已经公平解决 exact protein retrieval 的主 baseline。

## 8.3 讨论过但尚未作为当前主线完成的候选对齐 baseline

曾讨论过：

1. 对 candidate protein 用 BLASTp 找训练集中最近的重组酶；
2. 继承最近训练重组酶的 attP/attB；
3. query DNA 再与这些 DNA 做 BLASTn；
4. 得到 candidate-level score。

这可以形成候选空间一致的 reference baseline，但它**不是当前 V1 的前置阻塞项**，也不要误写成已经正式实现并验证的结果。

## 8.4 当前 baseline 原则

本项目是一个相对新颖的具体检索任务。Baseline 的作用是提供可解释参考，而不是为了形式上必须模仿某篇已有论文。

不要为了 baseline 强行改变最终任务定义。

---

# 9. 正式模型路线的演化

## 9.1 早期设想

早期路线考虑过：

- Protein encoder：ESM 系列；
- DNA encoder：DNABERT / Nucleotide Transformer；
- Dual Encoder（双塔模型）；
- CLIP-style InfoNCE 对比学习；
- 后续可能 Cross Encoder（交叉编码器）精排；
- 结构信息未来再作为 refinement（精化）。

## 9.2 为什么放弃普通 CLIP-style loss

因为数据只明确标注有限的 positive / negative pair，而大量跨实体组合没有实验标签。

普通 CLIP 把 batch 内非配对项全部视为 negative，会大量创造：

```text
false negatives（假负样本）
```

即潜在可以匹配、只是没人实验验证的 pair 会被模型强行推远。

因此当前 V1 不用普通 InfoNCE / CLIP 非对角负样本。

## 9.3 讨论过 ranking loss，但 V1 明确不加

曾考虑：

```text
L = compatibility BCE + lambda * pairwise ranking loss
```

用户质疑得很正确：最终排序本身可以直接基于训练后 cosine similarity 完成；第一版没有必要为了“目标是排序”就强行加 ranking loss。

因此当前决定：

> **V1 只学习兼容性空间，不加入 ranking loss。**

以后只有当 BCE 学得不错但 Recall@10 仍明显不理想时，才把 ranking loss 作为一个独立 ablation（消融实验）测试其增益。

## 9.4 暂不使用 hard-negative mining

V1 不做 hard-negative mining（困难负样本挖掘）。

原因：

- 当前先验证最干净的监督定义；
- unknown pair 不能被挖成负样本；
- 即使未来做 hard negative，也只能非常谨慎地从**已有明确负标签**的集合中考虑。

---

# 10. 当前正式模型 V1：权威定义

## 10.1 模型类型

**Frozen Dual Encoder + Trainable Projection Heads**

中文：冻结的双塔基础编码器 + 可训练投影头。

Backbone 不参与 V1 梯度更新。

## 10.2 Protein backbone

```text
facebook/esm2_t30_150M_UR50D
```

- ESM-2 150M；
- hidden dimension = **640**；
- V1 全部冻结；
- 用于生成 protein backbone representation。

## 10.3 DNA backbone

```text
InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
```

- Nucleotide Transformer v2 100M multi-species；
- hidden dimension = **512**；
- V1 全部冻结；
- 当前 Hugging Face 加载需要 `trust_remote_code=True`；
- 用于生成 DNA backbone representation。

## 10.4 Pooling

当前 `precompute_embeddings.py` 使用：

> **special-token-aware mean pooling**

即：

- 对 `last_hidden_state`；
- 排除 tokenizer 标记的 special tokens；
- 对剩余 token 做平均；
- 如果防御性检查发现全部被标 special，则回退到 attention mask。

当前没有使用 CLS token 作为固定 pooling。

不要在没有实验比较的情况下擅自切换 pooling。

## 10.5 Backbone 预计算

因为 backbone 冻结，所以不应每个 epoch 重跑 ESM-2 / Nucleotide Transformer。

流程：

```text
all unique proteins
    -> ESM-2 once
    -> protein backbone embeddings

all unique DNAs
    -> Nucleotide Transformer once
    -> DNA backbone embeddings
```

保存 `.pt` embedding store 后，正式训练只加载这些 frozen representations。

这能：

- 大幅降低训练成本；
- 使 V1 的性能变化只来自 projection learning；
- 让实验解释更干净。

## 10.6 Projection Head

Protein projection：

```text
640
 -> LayerNorm
 -> Linear(640, 512)
 -> GELU
 -> Dropout(0.1)
 -> Linear(512, 256)
 -> L2 normalize
```

DNA projection：

```text
512
 -> LayerNorm
 -> Linear(512, 512)
 -> GELU
 -> Dropout(0.1)
 -> Linear(512, 256)
 -> L2 normalize
```

最终两个模态都进入 **256-dimensional shared embedding space**。

## 10.7 Cosine similarity

投影输出已 L2 normalize，因此：

```text
cosine = z_dna @ z_protein.T
```

或 pair 训练时：

```text
cosine_i = sum(z_protein_i * z_dna_i)
```

范围理论上为 `[-1, 1]`。

## 10.8 Learnable gamma

用户已明确决定：

> **gamma（缩放系数）可学习。**

训练时：

```text
logit = gamma * cosine
```

实现不是直接优化 gamma，而是：

```text
gamma = exp(log_gamma)
```

这样永远保证：

```text
gamma > 0
```

当前：

```text
gamma_init = 5.0
gamma_max  = 100.0
```

`gamma_max` 用 clamp 防止无限爆大。

注意：

- gamma 只服务于 BCE 训练尺度；
- 最终检索排序直接用 cosine；
- sigmoid 和 gamma 都不改变固定 gamma>0 时 cosine 的相对顺序；
- 最终 TOP10 不需要 sigmoid 概率。

## 10.9 Loss

当前唯一正式 loss：

```text
BCEWithLogitsLoss(gamma * cosine, label)
```

等价概念：

```text
sigmoid(gamma * cosine)
 -> binary cross entropy
```

但实现直接使用 `BCEWithLogitsLoss`，数值更稳定，不手工先做 sigmoid。

当前：

- 正样本不加权；
- 负样本不加权；
- unknown 不生成、不进入 loss；
- 无 ranking loss；
- 无 InfoNCE；
- 无 hard-negative loss。

---

# 11. Batch / Sampler：当前已确认方案

## 11.1 为什么不能把所有 pair 简单按 row 权重随机采样

原始训练集有大量相同 protein sequence 重复记录。

审计发现：

- raw unique proteins ~37,262；
- 其中 **6,661 条蛋白在原始表中出现多次**；
- 单个 protein sequence 最多可以对应 **94 条原始记录**。

如果直接把所有 row/pair 均匀打散，大量重复记录的蛋白会天然获得更高训练权重。

用户认可的方案是：

> **以 unique protein sequence 为 sampler 的基本单位。**

## 11.2 每个 unique protein 的四类监督池

每个 protein group 保存：

```text
attP_pos[]
attB_pos[]
attP_neg[]
attB_neg[]
```

## 11.3 每个 epoch 的采样规则

对于一个 protein：

```text
最多取 1 × attP_pos
最多取 1 × attB_pos
最多取 1 × attP_neg
最多取 1 × attB_neg
```

如果某类有多个 DNA：

> **按 epoch 确定性轮换，而不是有放回随机抽样。**

这样：

- 不会因为记录多而一个 epoch 权重暴增；
- 多条合法 DNA 又会随着 epoch 推进逐步被模型看到。

## 11.4 同一 protein–DNA 跨类别重复

同一个显式 protein–DNA pair 偶尔可能因为原始元数据跨 row 出现在不同 attP/attB category。

V1 学的是 compatibility 而不是 site identity，因此 sampler 在同一个 protein 的同一个 epoch 内：

- 尽量选择未使用过的 DNA；
- 若某 category 只有已经被其他 category 选过的同一 DNA，则跳过该重复；
- 避免同一个 pair 在一个 epoch 被重复计权。

## 11.5 缺失类别

如果某 protein 的某类记录是 FAILED 或不存在：

```text
直接少一个 pair
```

不要：

- 从其他 protein 借 DNA；
- 随机制造负样本；
- 用 unknown 填空。

## 11.6 Batch size

当前默认：

```text
proteins_per_batch = 128
```

在常规完整四槽情况下约等于：

```text
128 proteins × 4 pairs = 512 supervised pairs/batch
```

本项目没有 CLIP in-batch negatives，所以没有为了“增加负样本数量”盲目扩大 batch 的必要。

## 11.7 一个完整 epoch 的实际统计

完整真实训练 group 跑一个 epoch：

```text
37,217 protein groups
148,868 supervised pairs
74,434 positive
74,434 negative
```

即当前 sampler 在实际数据中得到精确 1:1 正负。

这不是人为 `class_weight`，而是数据四槽结构产生的结果。

## 11.8 轮换采样覆盖率

累计 unique labeled pair coverage：

| Epoch | 已覆盖 / 197,911 | 覆盖率 |
|---:|---:|---:|
| 1 | 148,868 | 75.22% |
| 2 | 175,176 | 88.51% |
| 5 | 188,856 | 95.42% |
| 10 | 193,202 | 97.62% |
| 20 | 195,694 | 98.88% |
| 94 | 197,911 | 100% |

因此当前 V1 预设 20 epochs 能看到约 **98.88%** 的 unique explicit supervision。

---

# 12. 当前 V1 配置

`configs/v1.yaml`：

```yaml
protein_model: facebook/esm2_t30_150M_UR50D
dna_model: InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
protein_hidden_dim: 640
dna_hidden_dim: 512
projection_dim: 256
projection_hidden_dim: 512
dropout: 0.1
gamma_init: 5.0
gamma_max: 100.0
proteins_per_batch: 128
epochs: 20
learning_rate: 0.001
weight_decay: 0.0001
seed: 42
loss: BCEWithLogitsLoss
unknown_pairs: masked
ranking_loss: false
hard_negative_mining: false
```

当前 optimizer：

```text
AdamW
lr = 1e-3
weight_decay = 1e-4
```

当前没有 scheduler。

当前没有 validation / early stopping。

训练结束默认使用 final checkpoint。

---

# 13. 当前代码结构

用户原项目建议统一放在：

```text
~/program/recombinase_model/
```

用户之前已经建立 / 讨论过：

```text
~/program/recombinase_model/
├── data/
├── src/
├── results/
└── baseline/
```

不要擅自另起一个完全独立的新根目录造成项目分裂。

当前生成的 V1 包结构：

```text
rec_rank_model_v1/
├── src/rec_rank/
│   ├── __init__.py
│   ├── data.py
│   ├── sampler.py
│   ├── model.py
│   ├── embeddings.py
│   └── evaluation.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── precompute_embeddings.py
│   ├── train_v1.py
│   └── rank_test.py
│
├── tests/
│   ├── test_data_and_sampler.py
│   └── test_model.py
│
├── configs/
│   └── v1.yaml
│
├── requirements.txt
├── README.md
└── SELF_TEST.md
```

Codex 接手时应把模块合理并入现有 `~/program/recombinase_model/`，优先保持清晰模块化，而不是把所有功能塞进一个超大脚本。

---

# 14. V1 各文件职责

## `src/rec_rank/data.py`

负责：

- 原始 TSV 读取；
- protein / DNA normalize；
- FAILED 识别；
- IUPAC ambiguity -> N；
- stable sequence key；
- 显式 pair 构建；
- 正负冲突移除；
- unique protein group 聚合；
- test unique query / candidate / multi-positive truth 构建；
- prepared data 写出。

## `src/rec_rank/sampler.py`

负责：

- unique-protein group-aware epoch dataset；
- 每类每 epoch 最多 1 条；
- deterministic rotation；
- 同一 pair 跨 category 重复去重；
- group collate 成监督 pair batch。

## `src/rec_rank/model.py`

负责：

- protein projection head；
- DNA projection head；
- L2 normalization；
- learnable positive gamma；
- cosine/logit；
- `BCEWithLogitsLoss`。

## `src/rec_rank/embeddings.py`

负责：

- `.pt` embedding store 读取；
- key -> tensor row lookup。

## `src/rec_rank/evaluation.py`

负责：

- descending dense rank；
- dense TOP-K tie-aware selection；
- multi-positive Recall@K；
- MRR；
- Median Rank。

## `scripts/prepare_data.py`

原始 train/test -> processed train groups / sequence tables / test truth。

## `scripts/precompute_embeddings.py`

冻结真实 backbone，对 unique protein / DNA 做一次 representation 预计算。

## `scripts/train_v1.py`

读取 frozen embeddings，只训练：

```text
protein projection
DNA projection
log_gamma
```

## `scripts/rank_test.py`

加载 checkpoint，对：

```text
query DNA embeddings × candidate protein embeddings
```

生成完整 cosine matrix，计算指标并输出 dense-rank TOP-K。

---

# 15. Prepared data 输出文件

`prepare_data.py` 当前写出：

```text
train_proteins.tsv
train_dnas.tsv
train_pairs.tsv
train_groups.jsonl
train_conflicts.tsv
summary.json

test_candidates.tsv
test_queries.tsv
test_truth.tsv
```

含义：

### `train_proteins.tsv`

```text
protein_key    sequence
```

V1 可训练 group 中所有 unique proteins。

### `train_dnas.tsv`

```text
dna_key    sequence
```

训练显式监督中所有 unique DNAs。

### `train_pairs.tsv`

```text
protein_key	dna_key	label	category	source_row
```

显式监督成员关系。

### `train_groups.jsonl`

每行一个 unique protein group：

```json
{
  "protein_key": "...",
  "attP_pos": [...],
  "attB_pos": [...],
  "attP_neg": [...],
  "attB_neg": [...]
}
```

### `train_conflicts.tsv`

被自动屏蔽的正负冲突 pair。

### `test_candidates.tsv`

测试候选 unique protein sequences。

### `test_queries.tsv`

测试 unique DNA queries。

### `test_truth.tsv`

每个 DNA 的全部 known positive protein keys：

```text
dna_key    positive_protein_keys
```

一个 query 可以有多个正蛋白。

---

# 16. 当前运行命令

以下是 V1 包中的标准流程。部署到用户服务器后路径应按现有项目目录调整。

## 16.1 数据准备

```bash
python scripts/prepare_data.py \
  --train data/train_dataset_with_neg.txt \
  --test data/test_dataset_with_neg.txt \
  --out data/processed
```

完成后必须先检查 `summary.json` 是否仍符合核心不变量：

```text
train_unique_proteins = 37217
train_unique_dnas = 179059
train_category_memberships = 198099
train_unique_labeled_pairs = 197911
train_conflicting_pairs_removed = 1
test_unique_queries = 3672
test_unique_candidates = 1287
test_truth_pairs = 4143
```

如果这些数字因代码改动意外变化，不要继续训练，应先解释为什么。

## 16.2 Protein embedding 预计算

```bash
python scripts/precompute_embeddings.py \
  --input data/processed/train_proteins.tsv \
  --output embeddings/train_proteins.pt \
  --model facebook/esm2_t30_150M_UR50D \
  --batch-size 16 \
  --device cuda
```

测试候选 protein 同理。

## 16.3 DNA embedding 预计算

```bash
python scripts/precompute_embeddings.py \
  --input data/processed/train_dnas.tsv \
  --output embeddings/train_dnas.pt \
  --model InstaDeepAI/nucleotide-transformer-v2-100m-multi-species \
  --trust-remote-code \
  --batch-size 64 \
  --device cuda
```

测试 query DNA 同理。

## 16.4 Projection V1 训练

```bash
python scripts/train_v1.py \
  --groups data/processed/train_groups.jsonl \
  --protein-embeddings embeddings/train_proteins.pt \
  --dna-embeddings embeddings/train_dnas.pt \
  --output checkpoints/v1.pt \
  --epochs 20 \
  --proteins-per-batch 128 \
  --device cuda
```

## 16.5 测试全库排名

```bash
python scripts/rank_test.py \
  --checkpoint checkpoints/v1.pt \
  --candidate-embeddings embeddings/test_candidates.pt \
  --query-embeddings embeddings/test_queries.pt \
  --truth data/processed/test_truth.tsv \
  --output-ranking results/v1_top10.tsv \
  --top-k 10 \
  --device cuda
```

---

# 17. 当前自测状态

## 17.1 已完成

代码层自动测试当前：

```text
6/6 passed
```

覆盖：

- positive/negative conflict removal；
- epoch rotation；
- IUPAC ambiguity -> N；
- `FailedFailed` 不被误归一化；
- learnable gamma 始终为正；
- compatibility loss 可下降；
- dense rank ties；
- multi-positive evaluation；
- dense TOP-K 并列保留。

## 17.2 完整真实数据工程链路已测

当前 ChatGPT 工作环境中无法依赖用户服务器的真实 Hugging Face backbone 权重，因此使用**维度匹配的 synthetic frozen embeddings**做工程自测。

注意：

> synthetic embedding 结果绝对不能当作模型性能报告。

它只证明以下链路能跑通：

```text
完整真实 train/test 数据
-> group-aware batch
-> projection heads
-> learnable gamma
-> BCE training
-> checkpoint save/load
-> 3672 × 1287 full cosine matrix
-> dense ranking
-> Recall/MRR
-> TOP10 export
```

## 17.3 还没有完成

截至本记忆文件生成时，尚未在用户服务器真实完成：

1. 全部 train proteins 的 ESM-2 embedding；
2. 全部 train DNAs 的 Nucleotide Transformer embedding；
3. test candidates / queries 的真实 backbone embedding；
4. V1 真实 projection training；
5. V1 正式 Recall@1/5/10、MRR、Median Rank；
6. 正式 TOP10 输出分析。

Codex 不要把 synthetic self-test 写成“V1 已训练完成”。

---

# 18. 环境与运行约束

## 18.1 Conda 环境

项目统一环境：

```text
rec_rank
```

历史确认：

```text
Python 3.11.16
BLAST+ 2.17.0
```

不要随意再创建大量新环境；优先在 `rec_rank` 中补齐必要依赖。

当前 V1 `requirements.txt`：

```text
torch>=2.2
transformers>=4.40
safetensors
sentencepiece
pytest
```

实际服务器应先检查已有版本兼容性再决定是否安装，不要无脑升级破坏环境。

## 18.2 登录节点无 GPU

这是用户反复强调的固定事实：

> **登录节点没有 GPU。不要再在登录节点做 GPU 检查，不要用“没检测到 GPU”作为问题。**

需要 GPU 的 embedding 预计算 / 真实模型任务应提交到合适的计算节点 / Slurm GPU 资源。

Codex 如果生成 Slurm 脚本，应基于用户集群现有习惯和队列配置，不要猜分区名、GPU 型号等未知信息。

## 18.3 Hugging Face

用户已经设置过 HF cache 相关路径，但本文件没有可靠的具体绝对路径。

规则：

- 优先读取现有 shell 环境 / 项目脚本中的 `HF_HOME`、`TRANSFORMERS_CACHE` 等；
- 不要凭空猜缓存路径；
- 不要因为不知道路径就重新下载一套模型到随机目录。

---

# 19. 用户对代码与协作方式的明确偏好

Codex 必须遵守：

## 19.1 代码风格

- 代码尽量最小化；
- 模块化，不写单一巨型脚本；
- 中文注释优先；
- 变量与函数名可用清晰英文；
- 不要为“工程感”堆无必要抽象层；
- baseline 与正式模型尽可能复用共同的数据层 / 评估层。

## 19.2 自测责任

用户明确不希望：

```text
助手写脚本 -> 用户跑 -> 报错 -> 用户贴回来 -> 助手再修
```

期望流程是：

```text
助手先检查 / 自测 / 迭代
-> 尽可能发现逻辑 bug
-> 再给用户成熟命令
```

因此 Codex 每次修改核心数据 / sampler / evaluation / model 后：

1. 先运行自动测试；
2. 能跑真实数据 audit 就跑；
3. 对关键不变量做 assert；
4. 再交给用户执行。

## 19.3 解释风格

用户希望技术讨论中：

> 英文缩写或专业术语同时给出中文翻译和一句简短解释。

例如：

```text
MRR（Mean Reciprocal Rank，平均倒数排名）
BCE（Binary Cross-Entropy，二元交叉熵）
```

不要只堆缩写。

## 19.4 最终目标优先

不要为了使用某个流行方法而改变任务。

用户经常会追问：

> “这一步对最终输入 DNA -> 推荐 TOP10 有什么帮助？”

Codex 的每个新增复杂度都应能回答这个问题。

---

# 20. 当前已经明确放弃 / 暂缓的方案

下面这些是历史上讨论过，但**当前 V1 不应自动恢复**的内容。

## 20.1 不做正负样本置信度降权

不要自动设置：

```text
w_neg < w_pos
```

当前 explicit positive / explicit negative 同权。

## 20.2 不加 ranking loss

V1：

```text
ranking_loss = false
```

只有以后出现明确证据表明 compatibility BCE 无法转化为满意排序时，再单独做 ablation。

## 20.3 不做 CLIP-style in-batch negatives

这是硬约束。

## 20.4 不把 unknown 当 negative

这是硬约束。

## 20.5 不做 hard-negative mining

V1 暂缓。

## 20.6 不微调 backbone

V1：

```text
ESM-2 frozen
Nucleotide Transformer frozen
```

未来可讨论 partial fine-tuning / LoRA，但不是当前实现。

## 20.7 不做 validation split

用户明确要求当前不要重新划分。

## 20.8 不把 cluster 命中作为最终成功标准

exact protein sequence retrieval 才是主目标。

## 20.9 不先上 cross-encoder

Cross Encoder 可以作为未来精排，但当前必须先把高效 dual encoder V1 建立起来。

---

# 21. 当前 V1 设计为什么与最终产品一致

最终真实部署希望：

```text
candidate protein database
    -> protein backbone + projection
    -> precompute candidate embeddings once

new query DNA
    -> DNA backbone + projection
    -> one query embedding

query embedding @ candidate embeddings.T
    -> cosine scores
    -> dense rank
    -> TOP10
```

这意味着候选数据库可以较大，而不需要每次 query 都重新对每个 protein 做一个大型 pairwise cross-encoder forward。

这是当前选择 dual encoder 的核心工程原因。

---

# 22. 下一步执行顺序（Codex 接手后不要跳步骤）

## Step 1：把 V1 代码并入真实项目

目标路径以现有：

```text
~/program/recombinase_model/
```

为主。

先检查当前目录，不要覆盖用户已有 baseline / data 文件。

## Step 2：在 `rec_rank` 环境跑测试

至少：

```bash
pytest -q
```

确认 **6/6** 或在并入过程中有对应新增测试全部通过。

## Step 3：对真实原始数据运行 `prepare_data.py`

然后检查 `summary.json` 是否保持：

```text
37217
179059
198099
197911
1
3672
1287
4143
```

顺序对应前文 8 个 summary 字段。

## Step 4：审计真实序列长度

在真正跑 backbone 前，建议只做一次轻量统计：

- protein min/median/max length；
- DNA min/median/max length；
- 是否有超过 backbone/tokenizer 限长的序列。

这是为了确认 `truncation=True` 不会静默截断有意义信息。

注意：该审计**不能改变任务定义**，只是运行安全检查。

## Step 5：在 GPU 计算节点预计算真实 embeddings

先小规模 smoke test（烟雾测试），确认：

- tokenizer 正常；
- hidden dimension 正确；
- pooling 输出无 NaN；
- key 数量和输入行数一致；
- 保存/重载一致。

再跑全量。

## Step 6：训练 V1 projection heads

当前固定：

```text
20 epochs
128 proteins/batch
lr 1e-3
weight_decay 1e-4
seed 42
```

不要因为 test 结果临时修改一堆超参数后仍把 test 称为完全 untouched final test。

## Step 7：全 test 候选库排名

必须：

```text
3672 × 1287
```

得到：

- Recall@1
- Recall@5
- Recall@10
- MRR
- Median Rank
- dense TOP10 TSV

## Step 8：结果诊断

第一版至少查看：

- training loss curve；
- gamma trajectory；
- positive pair cosine distribution；
- negative pair cosine distribution；
- test known-positive ranks distribution；
- Recall@K。

这些是诊断，不改变主评价协议。

## Step 9：只有看到真实 V1 结果后再决定 V2

可能的未来 ablation（不是当前默认）：

- ranking loss；
- pooling 方式比较；
- projection dimension；
- partial backbone fine-tuning；
- LoRA（Low-Rank Adaptation，低秩适配）；
- only-explicit-negative hard-negative mining；
- candidate-aligned classical baseline；
- cross-encoder reranker；
- 结构信息 refinement。

不要在 V1 真实结果出现前一次性全部加入。

---

# 23. 当前开放问题 / 需要未来通过实验回答，而不是靠猜

以下问题没有最终实验答案：

1. ESM-2 150M + Nucleotide Transformer v2 100M 的 frozen representation 是否足够支持这个跨模态对齐任务？
2. special-token-aware mean pooling 是否优于其他 pooling？
3. 256D shared projection 是否最合适？
4. 20 epochs 是否足够？当前依据是覆盖率和简单性，不是独立 validation 最优点。
5. compatibility BCE 是否足以产生满意 Recall@10？
6. 后续 ranking loss 是否有增益？
7. 是否需要 partial fine-tuning / LoRA？
8. teacher-provided negative 的生物学真实性在总体上如何？当前 V1 按用户要求正常使用，不降权。
9. 后续真正部署时的候选蛋白数据库规模和来源是什么？当前测试候选库只是 1,287 条测试 protein。

Codex 遇到这些问题不要擅自宣称已有结论。

---

# 24. 需要特别避免的逻辑错误

任何修改提交前逐项检查：

- [ ] 有没有把未标注 pair 当成负样本？如果有，错误。
- [ ] 有没有恢复 CLIP 非对角 negatives？如果有，错误。
- [ ] 有没有只把 attP 或 attB 其中一个当正样本？如果有，错误。
- [ ] 有没有把 attP/attB 当互斥分类标签？如果有，错误。
- [ ] 有没有按 row 而不是 unique protein 过度加权重复蛋白？需解释。
- [ ] 有没有因为某类别缺失而人工制造 replacement negative？如果有，错误。
- [ ] 有没有把测试重复 DNA 当成多个独立 query 重复计权？当前正式评估应按 3,672 unique DNA。
- [ ] 有没有把 cluster Recall 当主结果？如果有，错误。
- [ ] 有没有用普通 ordinal rank 代替 dense rank？如果有，错误。
- [ ] TOP10 有没有机械截断 10 条而忽略 ties？如果有，错误。
- [ ] gamma 有没有可能变成负数？当前用 `exp(log_gamma)` 防止。
- [ ] inference 排名有没有误用 sigmoid/gamma 作为必要步骤？最终直接 cosine 即可。
- [ ] 有没有重新拆 validation set？当前用户不允许。
- [ ] 有没有在登录节点浪费时间检查 GPU？不要。
- [ ] 有没有未经测试就让用户反复跑报错版本？不要。
- [ ] 有没有把 synthetic embedding self-test 当真实模型性能？绝对不要。

---

# 25. Codex 第一次接手时的建议启动流程

Codex 在一片空白时，应按下面顺序：

```text
1. 读完 PROJECT_MEMORY_FOR_CODEX.md
2. 检查 ~/program/recombinase_model/ 当前目录树
3. 阅读 data/schema 相关代码
4. 阅读 V1 的 README.md 和 SELF_TEST.md
5. 阅读 src/rec_rank/data.py
6. 阅读 src/rec_rank/sampler.py
7. 阅读 src/rec_rank/model.py
8. 阅读 src/rec_rank/evaluation.py
9. 运行 pytest
10. 对 train/test 运行 prepare_data 并核对不变量
11. 再开始真实 backbone embedding
```

如果真实服务器里的代码和本文件存在明显矛盾：

> **先停下来定位是代码旧版、记忆旧版还是用户后来改了决定；不要静默选择一个版本继续。**

---

# 26. 面向用户汇报时应该怎么说

用户通常不想只听“代码写好了”，更关心：

- 为什么这样设计；
- 这个设计与最终 DNA -> TOP10 的关系；
- 当前实际完成到哪里；
- 什么是经过真实数据验证的；
- 什么只是工程 smoke test；
- 下一步具体做什么。

汇报时区分：

```text
事实 / 已运行结果
设计决策
尚未验证的假设
下一步实验
```

不要把假设包装成结果。

---

# 27. 术语速查

- **Dual Encoder（双塔编码器）**：蛋白和 DNA 分别编码到同一向量空间，之后快速计算相似度。
- **Backbone（基础编码器）**：预训练的大模型主体，如 ESM-2 / Nucleotide Transformer。
- **Projection Head（投影头）**：把不同 backbone 的表示映射到同一低维空间的小网络。
- **Embedding（嵌入向量）**：序列经模型编码后的数值表示。
- **Cosine Similarity（余弦相似度）**：比较两个归一化向量方向相似程度的分数。
- **BCE（Binary Cross-Entropy，二元交叉熵）**：正/负标签的二元监督损失。
- **Logit（逻辑值）**：送入 sigmoid/BCE 前的未归一化分数。
- **Sigmoid**：把实数映射到 `(0,1)`；V1 训练由 BCEWithLogits 内部完成，推理排序不需要显式 sigmoid。
- **Gamma / Scale（缩放系数）**：放大 cosine 的 logit 尺度；V1 可学习且保持正值。
- **Dense Rank（稠密排名）**：并列同名次，下一名次连续。
- **Recall@K（前 K 召回率）**：已知正确蛋白是否至少有一个出现在 rank<=K。
- **MRR（Mean Reciprocal Rank，平均倒数排名）**：平均 `1 / best_positive_rank`。
- **Multi-positive（多正样本）**：同一 query 有多个合法正蛋白。
- **Unknown / Unlabeled（未知 / 未标注）**：数据库没有正负证据的 pair，V1 完全忽略。
- **InfoNCE**：常见对比学习损失；V1 不使用普通 CLIP-style InfoNCE。
- **Hard Negative Mining（困难负样本挖掘）**：寻找模型最容易误判的负样本重点训练；V1 暂不使用。
- **LoRA（Low-Rank Adaptation，低秩适配）**：未来可能用于参数高效微调 backbone，当前不使用。

---

# 28. 最终权威摘要（Codex 只记 12 条也至少记这些）

1. **目标：输入 DNA，对具体重组酶蛋白库排名，输出 dense TOP10。**
2. **attP 和 attB 都是同一蛋白的正样本。**
3. **attP_neg / attB_neg 是当前正常使用、同权的负样本。**
4. **未标注 protein–DNA pair 永远不能自动当负样本。**
5. **V1 不使用 CLIP in-batch negatives、InfoNCE、ranking loss、hard-negative mining。**
6. **采样单位是 unique protein；每 epoch 每类最多 1 条，重复记录按 epoch 轮换。**
7. **ESM-2 150M 与 Nucleotide Transformer v2 100M 在 V1 全冻结并预计算 embedding。**
8. **只训练两个 projection heads + 正值 learnable gamma；loss = BCEWithLogits(gamma*cosine, label)。**
9. **正式推理只用 projection 后 cosine 排名；不需要 sigmoid/gamma。**
10. **正式评估：3,672 unique DNA × 1,287 candidate proteins；exact protein-level Recall@10 是主指标。**
11. **dense rank：同分同名次；TOP10 = rank<=10，因此可以超过 10 条。**
12. **不要重新拆 validation；不要在登录节点检查 GPU；修改后先自测，再交用户运行。**

---

# 29. 当前交付物

当前应与本文件一起交给 Codex 的代码包：

```text
rec_rank_model_v1_code.zip
```

包内包括：

- V1 source code；
- scripts；
- tests；
- config；
- README；
- SELF_TEST。

本文件建议直接放到项目根目录：

```text
~/program/recombinase_model/PROJECT_MEMORY_FOR_CODEX.md
```

并要求后续重大设计决策发生变化时同步更新它，避免代码和项目记忆逐渐漂移。

---

**End of project memory — 2026-10-08**
