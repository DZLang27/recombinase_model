# RecRank V1

RecRank V1 是一个重组酶–DNA 双塔检索系统。输入一条目标 DNA，模型在候选重组酶蛋白库中计算投影后余弦相似度，并输出所有 `dense_rank <= 10` 的候选。并列会共享名次，所以结果可能超过 10 条。

## V1 的边界

- `attP` 与 `attB` 都是同一蛋白的合法正 DNA，不做二者分类。
- 只使用数据库明确给出的正负 pair；未标注组合保持 unknown，不自动作为负样本。
- 不分析识别区域、`center_common` 或切割位点。
- 不使用 CLIP 批内非对角负样本、InfoNCE、ranking loss、hard-negative mining 或局部精排器。
- ESM-2 与 Nucleotide Transformer backbone 冻结并预计算；只训练两个投影头、`gamma` 和 bias。
- 推理排序只使用原始 cosine；`gamma` 与 bias 只服务于训练时 BCE 的尺度。

当前按已确认方案，每个 epoch 都使用测试集做完整候选库检索，并用以下顺序选择 best checkpoint：

1. Recall@10 更高；
2. MRR 更高；
3. Median Rank 更低；
4. epoch 更早。

因此这些测试指标承担开发/模型选择功能，不应被描述为“从未触碰的最终无偏测试结果”。

## 环境

项目使用既有 Conda 环境 `rec_rank`：

```bash
conda activate rec_rank
python -m pip install -e '.[dev]'
```

登录节点没有 GPU。真实 backbone embedding 和训练应提交到 Slurm GPU 节点；CPU 仅适合数据准备与小型测试。

## 命令总览

```bash
rec-rank --help
```

统一入口包含：`prepare`、`embed`、`train`、`evaluate`、`evaluate-full-library`、`export`、`build-index`、`screen`、`report`。

### 1. 数据准备

```bash
rec-rank prepare --config configs/v1.yaml
```

命令会校验当前数据集的核心不变量，并输出到 `data/processed/`：

- `train_proteins.tsv`、`train_dnas.tsv`
- `train_pairs.tsv`、`train_groups.jsonl`、`train_conflicts.tsv`
- `test_candidates.tsv`、`test_dnas.tsv`、`test_pairs.tsv`
- `test_queries.tsv`、`test_truth.tsv`、`summary.json`

### 2. 冻结 backbone embedding

四个 cache 均带数据、模型、revision、pooling、dtype、ordered keys 等指纹；不匹配时拒绝静默复用。未完成 cache 可以从已写完的 shard 恢复。

```bash
rec-rank embed --modality protein \
  --input data/processed/train_proteins.tsv \
  --output embeddings/train_proteins --device cuda

rec-rank embed --modality dna \
  --input data/processed/train_dnas.tsv \
  --output embeddings/train_dnas --device cuda

rec-rank embed --modality protein \
  --input data/processed/test_candidates.tsv \
  --output embeddings/test_candidates --device cuda

rec-rank embed --modality dna \
  --input data/processed/test_dnas.tsv \
  --output embeddings/test_dnas --device cuda
```

两种 backbone 都使用 special-token-aware mean pooling。DNA 模型的 `trust_remote_code` 已在配置中启用。

也可提交通用 Slurm 脚本：

```bash
sbatch scripts/slurm_precompute.sh protein data/processed/train_proteins.tsv embeddings/train_proteins
sbatch scripts/slurm_precompute.sh dna data/processed/train_dnas.tsv embeddings/train_dnas
sbatch scripts/slurm_precompute.sh protein data/processed/test_candidates.tsv embeddings/test_candidates
sbatch scripts/slurm_precompute.sh dna data/processed/test_dnas.tsv embeddings/test_dnas
```

### 3. 训练

```bash
rec-rank train --config configs/v1.yaml --device cuda
```

或：

```bash
sbatch --export=ALL,SEED=42 scripts/slurm_train.sh
```

正式比较建议至少运行 3 个预先指定的 seed，例如 42、43、44，不按测试结果挑 seed。恢复中断运行：

```bash
rec-rank train --config configs/v1.yaml --device cuda \
  --run-dir runs/<run_id> --resume
```

每个 run 包含：

```text
runs/<run_id>/
├── manifest.json
├── resolved_config.yaml
├── status.json
├── logs/
│   ├── history.csv
│   ├── events.jsonl
│   └── tensorboard/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── evaluation/
├── figures/
└── exports/
```

训练包含 Epoch 0 和固定 seed 随机排序 baseline、AMP unscale 后梯度裁剪、`gamma` 单独低学习率/高 weight decay 参数组、NaN/Inf 检测以及原子 checkpoint。

### 4. 训练可视化

实时查看：

```bash
tensorboard --logdir runs/<run_id>/logs/tensorboard
```

生成静态图：

```bash
rec-rank report --run-dir runs/<run_id>
```

静态报告包含 loss、Recall@1/5/10、MRR/Median Rank、正负 cosine 分布、pair margin、`gamma`/bias、梯度裁剪、最佳正样本 rank 分布和 Recall@K 曲线。

### 5. 在训练+测试蛋白并集上回顾性评价

该命令不重新训练，也不覆盖原训练报告。它将训练蛋白与测试蛋白按
`protein_key` 去重后作为完整候选库，并同时输出两套 truth 口径：

- `exact_truth`：合并训练集和测试集中完全相同 DNA 序列的正蛋白；
- `direction_invariant_truth`：进一步把反向互补 DNA 视作同一个 query。

为了避免一次保存完整的 query×candidate 分数矩阵，评价按 query batch 做精确
cosine dense ranking。可一次提供多个 checkpoint，最后生成统一的
`comparison.tsv`：

```bash
rec-rank evaluate-full-library \
  --checkpoint runs/<run_id>/checkpoints/best.pt \
  --checkpoint runs/<run_id>/checkpoints/last.pt \
  --output evaluations/full_library_existing_v1 \
  --device cuda
```

也可以通过 Slurm 提交：

```bash
sbatch scripts/slurm_evaluate_full_library.sh \
  runs/<run_id>/checkpoints/best.pt \
  runs/<run_id>/checkpoints/last.pt
```

### 6. checkpoint → candidate index → screening

构建与 checkpoint 和候选库指纹绑定的投影索引：

```bash
rec-rank build-index \
  --checkpoint runs/<run_id>/checkpoints/best.pt \
  --candidate-embeddings embeddings/train_proteins \
  --output candidate_indexes/<index_id> \
  --device cuda
```

筛选单条 DNA：

```bash
rec-rank screen \
  --checkpoint runs/<run_id>/checkpoints/best.pt \
  --candidate-index candidate_indexes/<index_id> \
  --dna ACGT... \
  --output results/query_dense_top10.tsv \
  --device cuda
```

输出字段包括 query、protein key、cosine、dense rank、checkpoint ID 和 candidate-index ID。索引与 checkpoint 不匹配时会直接拒绝执行。

## 测试

```bash
pytest
```

测试覆盖数据清洗与冲突移除、轮换 sampler、正值 `gamma`、梯度可学习性、dense ties、多正样本指标，以及 synthetic embedding 下的训练→best checkpoint→index→screening 全链路。Synthetic 测试只验证工程链路，不能作为模型性能结果。

## V1.1：完整候选库、排序损失与方向增强

V1.1 保留双塔与纯 cosine 检索，不增加局部识别区域或交互精排器。相对 V1
只做四项受控改变：

- 每个 epoch 都在训练集与测试集蛋白去重并集上评价，不再把 1,287 个测试蛋白
  的结果作为主指标；当前完整候选库为 38,502 条蛋白。
- best checkpoint 首先按 `novel_dna_recall_at_10` 选择，再依次比较 novel-DNA
  MRR、novel-DNA median rank、全体 Recall@10 和更早 epoch。
- 训练目标为 `BCE + 0.25 × margin ranking loss`。ranking pair 只来自同一蛋白下
  明确标注的正、负 DNA；unknown 组合不会被构造成负样本。
- 训练时保留全部原始样本，并为每一条样本额外追加反向互补副本；副本继承
  原标签，因此每个 epoch 的训练实例为原来的 2 倍。逐 epoch 检索将正向
  与反向互补的投影向量相加后重新 L2 归一化。主 truth 同样按方向不变口径合并，
  exact truth 指标仍作为诊断项记录。

先重新运行数据准备，生成反向互补映射和仅需补算的序列：

```bash
rec-rank prepare --config configs/v1_1.yaml
```

新增文件为 `train_dnas_rc_map.tsv`、`train_dnas_rc_only.tsv`、
`test_queries_rc_map.tsv` 和 `test_queries_rc_only.tsv`。当前实际需补算 165,991 条
训练 DNA 与 3,002 条测试 query DNA：

```bash
sbatch scripts/slurm_precompute.sh dna \
  data/processed/train_dnas_rc_only.tsv embeddings/train_dnas_rc_only

sbatch scripts/slurm_precompute.sh dna \
  data/processed/test_queries_rc_only.tsv embeddings/test_queries_rc_only
```

两个 embedding 任务完成后训练 V1.1：

```bash
sbatch --export=ALL,SEED=42 scripts/slurm_train_v1_1.sh
```

也可以用依赖关系一次提交，避免训练任务提前启动：

```bash
TRAIN_RC_JOB=$(sbatch --parsable scripts/slurm_precompute.sh dna \
  data/processed/train_dnas_rc_only.tsv embeddings/train_dnas_rc_only)
TEST_RC_JOB=$(sbatch --parsable scripts/slurm_precompute.sh dna \
  data/processed/test_queries_rc_only.tsv embeddings/test_queries_rc_only)
sbatch --dependency=afterok:${TRAIN_RC_JOB}:${TEST_RC_JOB} \
  --export=ALL,SEED=42 scripts/slurm_train_v1_1.sh
```

## V1.2-frozen：面向实际检索方向的显式排序

V1.2-frozen 继续冻结 ESM-2 和 Nucleotide Transformer backbone，沿用 V1.1 的
embedding cache、完整 38,502 蛋白候选库、方向不变 truth 和 novel-DNA
Recall@10 checkpoint 选择规则。它不使用 unknown pair，也不引入局部交互模型。

新增的训练信号为：

- protein-centric smooth margin ranking，并按蛋白先做组内平均；
- DNA-centric smooth margin ranking：对同一 DNA，只比较明确标注的正蛋白和
  负蛋白；
- DNA 与其反向互补序列在投影空间中的 cosine consistency；
- 原始监督 pair 和全部反向互补副本仍共同参与 BCE。

当前真实训练数据可以构造 2,314 个 DNA ranking group、3,979 个显式正 pair、
2,453 个显式负 pair和 4,628 个合法正负 triplet。先重新 prepare 生成审计文件
`train_dna_ranking_groups.jsonl`：

```bash
rec-rank prepare --config configs/v1_2_frozen.yaml
```

这一步不会改变序列 key，也不需要重算现有 backbone embedding。随后提交 seed 42：

```bash
sbatch --export=ALL,SEED=42 scripts/slurm_train_v1_2_frozen.sh
```

V1.2-frozen 的结构化日志额外记录 DNA-centric ranking loss、有效比较数、正负
margin、RC consistency loss 和 DNA/RC embedding cosine。报告同时显示
novel-DNA Recall@10 的比例与实际命中数，避免只用相对提升描述绝对性能。

## V1.3-projection：回到 V1.1，仅增强投影头

V1.3-projection 保留 V1.1 的全部训练与评价定义：相同的 protein-centric hinge
ranking、BCE 权重、batch sampler、全量反向互补追加、完整 38,502 蛋白候选库和
novel-DNA Recall@10 best-checkpoint 规则。它不包含 V1.2-frozen 的 smooth ranking、
DNA-centric ranking、RC consistency 或 cosine scheduler，因此与 V1.1 的主要变量
只有投影头容量。

蛋白和 DNA 使用两个参数相互独立、结构相同且学习率相同的投影头。每个投影头为
`输入 LayerNorm → Linear(输入维度, 512) → 2 个残差 MLP block → LayerNorm → L2`
；每个 block 为 `Pre-LN → Linear(512, 1024) → GELU → Dropout →
Linear(1024, 512) → Dropout → LayerScale(初值 0.1) → residual add`。共同 cosine
空间由 256 维扩为 512 维。两个 backbone 仍冻结，继续复用现有 embedding cache，
不需要重新计算 embedding。

提交 seed 42：

```bash
sbatch --export=ALL,SEED=42 scripts/slurm_train_v1_3_projection.sh
```

三个 seed 训练完成后，可以在同一完整候选库上对每个 DNA–蛋白 cosine score 取
算术平均，再统一执行 dense ranking。ensemble 不平均投影向量，也不使用测试标签
调整权重；三个 checkpoint 的权重固定为 `1/3`：

```bash
sbatch scripts/slurm_evaluate_v1_3_ensemble.sh
```

默认输出到 `evaluations/v1_3_seed_42_43_44_ensemble/`，其中记录三个 checkpoint
的绝对路径、SHA256、seed、epoch、聚合规则、完整库指纹、逐 query rank 与 dense
TOP10，避免把 ensemble 结果与单 checkpoint 结果混淆。

## V1.4-dna-ranking：单变量检索方向监督

V1.4 完整保留 V1.3 的残差投影头及训练、评价设置，只新增权重为 0.1 的
DNA-centric hinge ranking。对同一 DNA，仅在数据中有明确标签的正蛋白和负蛋白
之间构造 `max(0, 0.1 - cosine_pos + cosine_neg)`；unknown protein–DNA pair 永远
不作为负样本。原序列与追加的反向互补序列都参与该约束。

每个 epoch 对 2,314 个合法 DNA group 各使用一次，共包含 3,979 个显式正 pair、
2,453 个显式负 pair和4,628个正负 triplet。每个主训练 step 配8个 DNA group，
恰好覆盖约290个 batch。V1.2 的 smooth loss、protein group balancing、RC
consistency、cosine scheduler及降低学习率均未启用，因此这次实验只检验与实际
`DNA → protein ranking` 方向一致的轻量监督是否有效。

先运行 seed 42：

```bash
sbatch --export=ALL,SEED=42 scripts/slurm_train_v1_4_dna_ranking.sh
```
