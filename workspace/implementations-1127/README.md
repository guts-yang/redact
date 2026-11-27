# implementations-1127 — 第二轮 10 个方法的实验复现代码

对应 `workspace/idea-stage-1127/` 的 10 份 proposal（2026-11-27）。本目录是全部实验复现的算法设计代码：每个方法一个独立、解耦、可单独维护的子包，上传到 GPU 集群后即可依次启动复现。

> proposal 里的分数（7.8 / 7.6 …）都是 **机制分（mechanism score）**，尚待验证。本代码只提供「训练期可微松弛 + 诊断量」的实现骨架，**不产生任何科学结论、不推进任何 ARIS 投稿门控**。真实结论必须来自 GPU 集群上的 T0–T3 决定性实验与独立评审。

## 目录结构（解耦契约）

```
implementations-1127/
  README.md                     本文件
  PROMPT_FOR_CLUSTER.md         给远程服务器/AI 的快速上手提示词
  requirements.txt              core=torch；runtime=transformers/peft
  methods/<slug>/               core.py train.py test_core.py method.yaml __init__.py
    budget pivot consensus drift reverse faith graph attrib layer probe
  runtime/    paths.py data.py model.py runner.py  (通用 GPU 接线, 延迟导入 HF/PEFT)
  scripts/    selftest.py check_env.py run_all.py  (+ gradcheck.py 开发用)
  configs/    repro_order.yaml  (排名顺序 + 核心量 + 冻结设置 + T0-T3)
```

**解耦约束（已核验）**：10 个 `methods/*/core.py` 只 import `torch` + 标准库，**方法之间零交叉 import**；每个把所需最小工具内联，不依赖共享 core 包。每个方法自带 core / train / test_core / method.yaml，改一个不动别的。`runtime/` 是唯一公共层，且不在方法间传递状态。

## 两层设计（为什么能在本机验证）

| 层 | 依赖 | 本机(CPU, 仅 torch) | GPU 集群 |
|---|---|---|---|
| `core.py` 纯算法 | torch | 单测 + 自检通过 | 实跑 |
| `train.py` / `runtime/` | transformers/peft | 可 import（延迟导入）| 实跑 |

`train.py` 的 `compute_loss(ctx)` 只消费 `ctx` 张量 + 可微松弛，不直接调用 generate（采样不可微）。基于真实生成的指标（预算生成、多数投票、层修补扫描、跨语言改写）属于 **eval 侧**，在集群上用 `runtime.model` 工具做，不进训练梯度。因此 `selftest.py` 能用合成 `Context` 在纯 CPU 上验证每个方法「能跑、loss 有限、可反向、对输入敏感」。

## 本机验证结果（已执行）

- `python scripts/selftest.py` → **10/10 PASS**（finite loss + backward + diagnostics）。
- `python methods/<slug>/test_core.py`（10 个）→ **10/10 单测通过**（单调正则、DPC、CM 随 N 非降、漂移分解、可逆半径、Faith、图覆盖、KA、Depth、AUC 公式断言）。
- `python scripts/dryrun_runner.py` → **MOCK-GPU RUNNER SMOKE PASS**：纯 torch 桩件端到端跑通训练循环（采样→forward→loss→backward→optimizer→save adapter→summary），验证 runner 契约与落盘。
- `python scripts/run_all.py --all --dry-run` → CPU 预检 + 打印 10 条 GPU 命令，不占 GPU。
- 开发中自检抓到并修复 5 个真实缺陷：budget 单调正则方向反、layer soft_depth 不够锐利、consensus/drift 死常数 loss、reverse 维度错配、setup 硬编码 cuda。

## GPU 集群三步启动

```bash
cd <repo>/workspace/implementations-1127
# 1) 环境预检（不加载模型，秒级）
python scripts/check_env.py
# 2) CPU 预检 + 查看命令（不占 GPU）
python scripts/run_all.py --all --dry-run
# 3) 依次复现（每个方法独立写 adapter + train_summary.json）
python scripts/run_all.py --only budget     # 先单跑一个验证（默认冒烟 8 步）
python scripts/run_all.py --all            # 按排名跑完；默认两个冻结种子 1001/2002
python scripts/run_all.py --only budget --decisive  # 放大到 decisive_preset（200 步 / 8+8 / 1024）
python -m methods.consensus.train --steps 8  # 也可直接单方法
```

run-id 现在带阶段与种子（`unl1127_<slug>_<tag>_s<seed>`，`tag` = `smoke` / `decisive`），两阶段、两个种子的产物互不覆盖。
`--decisive` 只放大训练规模，仍**不是** T0–T3 结论：真实生成指标需要 eval 侧脚本，禁止拿 `train_summary.json` 的松弛值当结果。
`scripts/queue_1127_gpu1.sh` 会在 GPU1 空闲后按「先 smoke 全量、再 decisive 全量」自动跑两阶段。
`scripts/queue_1127_eval_gpu1.sh` 是独立的 PHASE 3：等训练队列写出 `QUEUE ALL DONE` 且 GPU1 再次空闲后评测 20 个 decisive adapter（**方案 B**）。主种子 1001 的 10 个做完整 4-split + **Hy4** AFE/CFE + MU；副种子 2002 的 10 个只生成 `forget10` 并做 Hy4 AFE/CFE 抽查（无 MU），省掉 10×617 行生成。

**流水线**：每个 run 拆为 `--stage gpu`（merge + 生成，独占 GPU1）与 `--stage cpu`（Hy4 AFE/CFE + MU，`CUDA_VISIBLE_DEVICES=""` 强制走 CPU）。GPU 段一结束就后台启动 CPU 段，GPU1 立刻开始下一个 run 的生成，judge 时间被隐藏；`MAX_CPU_JOBS=2` + 背压防止 Hy4 端点被并发压垮。
注意 `redact/evaluator/cosine_similarity.py` 与 `entailment_score.py` 默认会选 cuda，若不加 `--stage cpu` 的 GPU 屏蔽，MU 会重新占住 GPU1，流水线失效。

它不修改正在后台运行的训练队列脚本，也不改 `scripts/upbd/*`（UPBD 链正在用）。评测产物在 `$UNL1127_RESULTS/eval1127/<run_id>/eval.json`，状态在 `logs/unl1127/queue_1127_eval_gpu1/status.tsv`（含 `mode=` 列区分 full / forget-only）。

每个方法输出到 `$UNL1127_RESULTS/unl1127_<slug>_<tag>_s<seed>/`：LoRA `adapter/`、`train_summary.json`（losses / final_diagnostics / meta）、`train_args.json`。summary 带 `diagnostic_only: true`、`claim_level: non-claim`。

## 路径配置（环境变量覆盖，无硬编码本机路径）

`runtime/paths.py` 默认对齐 R-TOFU 服务器布局，前缀 `UNL1127_` 可覆盖：

| 变量 | 默认 | 含义 |
|---|---|---|
| `UNL1127_ASU_ROOT` | `/jizhicfs/chenyliao` | 数据/模型根 |
| `UNL1127_RTOFU_REPO` | `/usr/local/workspace/R-TOFU` | R-TOFU 仓库（拒答语料所在） |
| `UNL1127_TARGET_MODEL` | `$ASU_ROOT/models/sangyon/LRM-target` | 目标 LRM |
| `UNL1127_FORGET_FILE` | `$ASU_ROOT/data/rtofu/forget10.json` | forget set |
| `UNL1127_RETAIN_FILE` | `$ASU_ROOT/data/rtofu/retain90.json` | retain set |
| `UNL1127_REFUSAL_FILE` | `$RTOFU_REPO/data/idontknow.jsonl` | 拒答语料 |
| `UNL1127_CLAIMS_FILE` | `$REDACT_ROOT/knowledge_banks/forget10/claims.json` | 记忆图 claims |
| `UNL1127_RESULTS` | `$WORKSPACE/results/unl1127` | 输出根 |
| `UNL1127_UNLEARNED_ADAPTER` | 空 | 已遗忘参考 checkpoint（DRIFT/REVERSE/LAYER/PROBE 用 θ₀） |
| `UNL1127_BUDGET_GRID` | `0,64,128,256,512` | BUDGET 预算网格 |
| `UNL1127_LANGUAGES` | `EN,ZH,DE,RU,TR` | PIVOT 语言集 |
| `UNL1127_SEED` | `1001` | 种子（proposal 冻结 1001, 2002） |

数据契约与 R-TOFU 一致：每行 `question / answer / cot / task_id`，ground truth 恒来自数据集；prompt 模板内联在 `runtime/data.py`，不 import 隔壁仓库。

## 每个方法在做什么

| slug | 核心量 | 训练松弛 |
|---|---|---|
| budget | BC=max_b ρ(b)−ρ(0)→0 | per-budget 可微 gate + 单调正则 + worst-budget |
| pivot | DPC→0 | 跨语言步级 scope + 共享层一致约束 |
| consensus | CM→0 | soft-max 共识 CLR 随 N 非降 + min-max |
| drift | R=Δ_rec/‖Δ_free‖ | 自由步回补、承重步锁定 |
| reverse | R_θ(ε)→0 | 最坏 steering 方向 + 去低秩抑制 |
| faith | FaithGap + 残余泄漏 | 忠实度门控（忠实步压 CoT / 不忠实压答案级） |
| graph | Δ_MU≤0.05, SCS≥0.9 | 记忆图采样 + 自生成步级 scope |
| attrib | RR≤0.05, DR≤0.02 | 步级 KA 重定向 + 两难闭合 |
| layer | min_p Depth≤ℓ_shared, DC≤1 | 软擦除深度 + 跨策略一致 |
| probe | UDec→0.5 | GRL 对抗不可解码 + 双证据 Gap |

细节见各自 `idea-stage-1127/IDEA_XX_*/proposal.md` §3 与 `methods/<slug>/method.yaml`。

## 诚实性边界（ARIS）

- 所有 relaxed training objectives 是 **机制分的实现**，不是结论；proposal 自身标注 *pending the validation experiment*。
- T0（基线复现）、T1（完整方法）、T2/T3（消融）需在集群上用真实数据/模型跑，并由独立评审者（不同模型家族）判定；执行者不自评诚信。
- 真实生成类指标走 eval 侧，禁止用可微松弛值充当结论（proposal 明确要求同时报真实值）。
