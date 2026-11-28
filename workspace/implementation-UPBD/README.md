# implementation-UPBD

六个 UPBD proposal 的隔离实现：`TCR`、`BLESS`、`RCBF`、`STREAM`、`TMOT`、`CROWN`，另含 `upbd_base` 参考控制。

**所有输出均为 diagnostic-only / non-claim。** proposal 的分数是机制分；本目录提供可测试的算法与训练/评测接线，不提供遗忘有效性、参数级擦除或投稿结论。

## 1. 解耦结构

```text
implementation-UPBD/
  configs/                         # 共享协议、资源预设、实验控制
  upbd_impl/
    contracts.py                  # 无副作用类型与 Method 生命周期
    registry.py                   # 静态方法注册，延迟 import
    core/                         # 通用纯计算：margin、probe、controls、decision
    runtime/                      # 数据、设备、模型、训练、产物、评测适配
    methods/<slug>/
      core.py                     # 方法独立算法；torch + contracts/core
      train.py                    # runtime 适配入口
      method.yaml                 # 方法独立超参数
      test_core.py                # 方法独立 CPU 不变量测试
  scripts/selftest.py             # base + 六方法合成张量自检
  scripts/check_env.py            # 只读环境/输入预检
  tests/                          # 隔离、数据、产物、评测契约测试
```

依赖方向为 `CLI -> registry -> method adapter -> method core`；runtime 提供训练与 I/O。方法 core 不 import 其他方法或 runtime，不直接生成文本、调用 Hy4、选择 GPU 或写结果。RCBF 的 barrier 参数组、STREAM/CROWN 的请求状态均通过统一 hook 管理。

## 2. 数据盘与独立产物路径

默认输入均位于 `/jizhicfs/chenyliao/`，已逐个检查存在：

| 配置 | 默认值 |
|---|---|
| `UPBD_IMPL_ASU_ROOT` | `/jizhicfs/chenyliao` |
| `UPBD_IMPL_TARGET_MODEL` | `/jizhicfs/chenyliao/models/sangyon/LRM-target` |
| `UPBD_IMPL_UNLEARNED_MODEL` | `/jizhicfs/chenyliao/checkpoints/r2mu-forget10/checkpoint-last` |
| forget CSV | `/jizhicfs/chenyliao/runs/r2mu/forget10/eval/forget10.csv` |
| retain/aux CSV | `/jizhicfs/chenyliao/runs/target/forget10/eval_tf443/` |
| `UPBD_IMPL_RESULTS` | `/jizhicfs/chenyliao/lrm-unlearning/results/upbd_impl` |

模型和 CSV 只读。adapter、merged、split manifest、生成 CSV、Hy4 JSONL、日志和 summary 只能写入新的 run 目录；已存在且非空的 run 默认拒绝覆盖。

### 实体切分

现有 CSV 没有实体列，因此：

- `decisive` 默认 `split_mode=entity`，必须提供 `UPBD_IMPL_ENTITY_MAP_JSON`；JSON 格式为 `{normalized_question: entity_key}`。缺少映射会 fail closed。
- `smoke` 默认 `split_mode=question`，只验证工程链路，manifest 明确写 `scientific_valid=false`。
- `holdout` 仅用于显式 legacy 对照，manifest 标记 calibration/test overlap。

训练只使用 manifest 的 train 分区；接受决策只能使用 calibration；评测在原始生成后，通过 `select_test_csvs` 只保留 test 行。小样本生成若未覆盖 test 行会明确失败，不会用训练行替代。

### 方法专用 JSON 输入

真实实验所需的方法状态均通过显式文件注入，不用合成默认值代替：

| 环境变量 | 内容与使用方 |
|---|---|
| `UPBD_IMPL_BOUNDARY_STATS_JSON` | `representation_sensitivity`（TCR 必需）、`parameter_weights`、`benign_operator_drops`（BLESS 必需）、`aliases`、`allow_concession` |
| `UPBD_IMPL_REFERENCE_HIDDEN_JSON` | 参考表示的嵌套数组，TMOT decisive 必需 |
| `UPBD_IMPL_REQUESTS_JSON` | 请求对象列表，包含 `request_id / forget / protect / aux / risk_budget / min_utility`，STREAM/CROWN decisive 必需 |

这些文件只读。缺少任一 decisive 必需字段都会在模型训练前 fail closed；smoke 的 `torch.roll`、零对偶等仅用于合成测试，不构成方法证据。

## 3. CPU 与 dry-run

```bash
cd /usr/local/workspace/tracks/lrm-unlearning/redact/workspace/implementation-UPBD
python -m upbd_impl.cli list
python scripts/check_env.py --cpu --check-inputs
python scripts/selftest.py
python -m unittest discover -s tests -v
python -m unittest discover -s . -p 'test_core.py' -v
python -m upbd_impl.cli train --method tcr --preset smoke --dry-run
python -m upbd_impl.cli train --method rcbf --preset decisive --dry-run
```

`list`、registry import 和 dry-run 不加载模型、不联网、不初始化 CUDA。

## 4. 独立 GPU 接入

必须显式指定一张**当前空闲**的 GPU。未设置 `CUDA_VISIBLE_DEVICES` 时，CUDA 训练直接失败；实现不会自动选卡、清空显存、启动/停止现有作业或加入既有队列。

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU> UPBD_IMPL_RESULTS=<NEW_RESULTS_ROOT> \
python -m upbd_impl.cli train \
  --method tcr --preset smoke --device cuda \
  --run-id tcr_smoke_s1001 --save-merged
```

评测分为原始生成和 test-only 评分两段：

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU> python -m upbd_impl.cli eval \
  --run-dir <RUN_DIR> --stage gpu --num-examples 4
CUDA_VISIBLE_DEVICES="" python -m upbd_impl.cli eval \
  --run-dir <RUN_DIR> --stage cpu
```

评测需要训练时生成 `merged/`。完整实验建议使用全量生成；若 `num_examples` 未覆盖 test 分区，CPU 段会拒绝评分。Hy4 固定为唯一 judge；judge 覆盖不完整时不允许聚合。CPU 段强制 `CUDA_VISIBLE_DEVICES=""`，避免 MU 组件重新占卡。

## 5. 方法职责

| 方法 | 独立算法资产 | 主要接入点 |
|---|---|---|
| `upbd_base` | 三源行为目标与普通 SFT | 参考控制、C1/C2、legacy parity |
| `tcr` | 敏感方向投影、参数能量、三层契约接受 | `constrain_update`、calibration `accept` |
| `bless` | 良性算子 margin 消耗、组合裕度、超域 `UNKNOWN` | 算子族准备、组合后接受 |
| `rcbf` | 正裕度 barrier、轨迹 violation、策略/屏障参数组 | 交替阶段 loss、独立 barrier 参数 |
| `stream` | 非负对偶、累计违反、不可行状态、可选 proxy regret | 请求快照、对偶更新、历史重验 |
| `tmot` | torch Sinkhorn、禁运支撑、边际误差、unknown 成本 | 参考统计、OT 内化、边际检查 |
| `crown` | 请求规范化、集合组合、冲突检测、最小让步 | 联合请求、`INFEASIBLE`、组合后重验 |

TMOT 不依赖 POT；CROWN 不依赖 SAT。`requirements.txt` 仅为依赖声明，本次没有安装或升级任何包。

## 6. 新旧流程兼容

- 新代码只位于本目录，使用唯一包名 `upbd_impl`。
- 不修改 `scripts/upbd/*`、现有 `config/*`、`implementations-1127/*` 或其队列。
- 不复用固定 GPU1 的旧 shell 脚本；只通过 run-local 适配器调用现有生成、Hy4 与聚合 Python 入口。
- 使用 `UPBD_IMPL_*` 环境变量、独立 run-id 和独立结果根。
- `train_summary.json` 固定包含 `diagnostic_only: true` 与 `claim_level: non-claim`；训练 loss/松弛量不能作为科学结果。

## 7. 已实际执行的验证

截至 2026-11-28，以下检查均已在 `CUDA_VISIBLE_DEVICES=""` 下通过：

- `scripts/selftest.py`：base + 六方法，**7/7 PASS**（有限标量 loss、backward、有限 diagnostics）。
- 方法包 `test_core.py`：**22 项 PASS**，覆盖投影、裕度、barrier、对偶、Sinkhorn、请求冲突等。
- `tests/`：**23 项 PASS**，覆盖 import/依赖隔离、matched controls、实体映射、split manifest、方法专用 JSON 输入、产物防逃逸/防覆盖、test-only 过滤及 legacy parity。
- `python -m compileall`：通过。
- `cli list`、smoke/decisive dry-run、`check_env.py --cpu`：通过；数据盘模型和四个 CSV 均确认存在。
- **未启动 GPU 训练、未生成真实模型输出、未调用 Hy4、未运行科学实验。**

GPU smoke 需要在明确空闲的卡上另行执行；当前文档不将其标记为已完成。
