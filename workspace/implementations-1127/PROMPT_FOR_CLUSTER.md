# 给远程 GPU 服务器的快速上手提示词

把下面框里的内容整段发给远程服务器上的 AI / 助手 / 你自己作为操作 checklist。它假设仓库已拉取、Python 环境已激活。

---

```
你在一台 Linux GPU 集群上，工作区已包含仓库 redact。
任务：熟悉并按顺序复现 workspace/implementations-1127/ 下第二轮 10 个
machine-unlearning 方法的实验代码。请严格按下列步骤，不要跳步、不要改
任何一个方法去迁就另一个方法（10 个方法彼此解耦、独立维护）。

【第 0 步 · 定位与通读】
cd <repo>/workspace/implementations-1127
- 先读 README.md（整体契约），再读 configs/repro_order.yaml（排名顺序、
  每个方法的核心量、冻结设置、T0-T3 含义）。
- 每个方法目录 methods/<slug>/ 读 method.yaml + core.py + train.py；
  数学定义以对应 ../../idea-stage-1127/IDEA_XX_*/proposal.md §3 为准。
- runtime/ 是唯一公共层：paths.py(路径) data.py(数据/prompt) model.py
  (模型/LoRA/forward/生成/激活修补) runner.py(训练循环)。HF/PEFT 全部
  延迟导入，core.py 只依赖 torch + 标准库。

【第 1 步 · 环境预检】
python scripts/check_env.py
- 若有 MISS：按提示设 UNL1127_* 环境变量（尤其 ASU_ROOT / TARGET_MODEL /
  FORGET_FILE / RETAIN_FILE / RESULTS）或 pip install -r requirements.txt。
- 数据/模型默认对齐 R-TOFU：forget10.json / retain90.json / LRM-target。
- 若某方法需要「已遗忘的参考 checkpoint」（DRIFT/REVERSE/LAYER/PROBE 的 θ₀），
  设 UNL1127_UNLEARNED_ADAPTER=<adapter 路径>；没有也能跑（会用合成基底烟测，
  但那不是科学结论）。

【第 2 步 · CPU 预检（不占 GPU）】
python scripts/selftest.py                # 10 方法 compute_loss + backward
python scripts/dryrun_runner.py           # 端到端训练循环冒烟(纯 torch 桩件)
python scripts/run_all.py --all --dry-run  # 上面 + 打印将运行的 GPU 命令
- 期望：10/10 PASS + MOCK-GPU RUNNER SMOKE PASS。失败先修环境/路径，不要上 GPU。
- 可选：python methods/<slug>/test_core.py 跑单方法数学单测。

【第 3 步 · 依次复现】
按 configs/repro_order.yaml 的排名顺序（budget → pivot → consensus → drift
→ reverse → faith → graph → attrib → layer → probe）：
  python scripts/run_all.py --only budget     # 先单跑一个，确认 GPU/显存 OK
  python scripts/run_all.py --all            # 再按排名一次跑完
或每次单独：python -m methods.<slug>.train --steps 8 --seed 1001
- 两个冻结种子都要跑：--seed 1001、--seed 2002（proposal 冻结 fp32/seeds）。
- 每个方法独立写 $UNL1127_RESULTS/unl1127_<slug>/ 下的 adapter/、
  train_summary.json、train_args.json，互不覆盖。

【第 4 步 · 评估与决定性实验】
- 训练侧是可微松弛 + 诊断量；真实指标（预算生成、多数投票、层修补扫描、
  跨语言改写、探针 AUC）必须用 runtime.model 里的工具在 eval 侧单独测，
  禁止用 train 的松弛数值充当结论。
- 对每个方法补 T0(基线复现) / T1(完整方法) / T2(消融核心机制) /
  T3(消融跨条件对齐)；判据见各 proposal §4 表格。
- 汇总每个方法的「核心量」是否达标（如 BC→0、CM→0、DPC→0、UDec→0.5），
  连同 collapse、Retain ROUGE-L 一起报告。

【纪律 / 边界】
- proposal 分数是机制分(pending validation)，不是结果；本代码不产生任何
  科学结论、不推进 ARIS 投稿门控。所有 summary 标 diagnostic_only /
  claim_level: non-claim。
- 执行者不自评诚信；结论需独立评审者（不同模型家族）判定。
- 不编造：引用、数据、结果必须可追溯；多源冲突并列呈现。
- 方法之间零交叉 import，任何修改只落在单个 methods/<slug>/ 目录内。
```

---

## 同步说明（把代码取到远程）

本机（macOS）HTTPS 的 `credential-osxkeychain` 鉴权不可用，推送沿用 SSH 通道；远程服务器拉取用常规 HTTPS 即可：

```bash
# 本机推送（由助手完成，远程已配好 deploy key/HTTPS 的情况直接拉）
git remote set-url origin git@github.com:guts-yang/redact.git
git add -A && git commit -m "feat(1127): 10 decoupled method implementations + runtime + tests"
git push origin fuxian
git remote set-url origin https://github.com/guts-yang/redact.git

# 远程服务器
git pull --ff-only origin fuxian
```
