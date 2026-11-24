# Idea Discovery Report — LRM/CoT Unlearning 自动化科研闭环

**Run ID**: `lrm-loop-20261124`
**Direction**: 面向大型推理模型（LRM）长思维链遗忘的可信评测与解码不变性
**Date**: 2026-11-24
**Inputs**: `/usr/local/workspace/awesome-lrm-unlearning`（领域概况，34 篇）+ `/usr/local/workspace/redact`（REDaCT 方法代码）
**Pipeline**: 问题提炼与验证 → 代码自诊断 → 思路生成 → 10 idea 冻结评分 → 高分方案 GPU 复现 → LLM 评测 → 证据补齐复评
**Canonical deliverable**: 本文件。其余证据为 `idea-stage/*.json`，由 `auto_loop.py` 机械生成。
**Status**: 研究内容已完成；**证据门禁 `BLOCKED`，不得视为 PASS**（见 §9）。

---

## 0. 闭环设计（可重复执行）

```bash
cd /usr/local/workspace/redact
python3 idea-stage/auto_loop.py --stage all
#   verify → diagnose → novelty → pilot → score → gate
```

数据流：

```
本地 GPU 生成记录 (R-TOFU/workspace/results/eval_*.jsonl)
  + Hy4 LLM judge (judge_*.jsonl)
  + redact 源码/产物 + awesome 领域清单 (data/papers.json)
        ↓ auto_loop.py（只读取，不推断、不补数）
  problem_validation.json / code_diagnosis.json / novelty_scan.json / pilot_results.json
        ↓ 冻结 rubric (hash cff0a224b94802cc)
  idea_scores_v1.json（提案态）→ idea_scores.json（证据态）
        ↓ gate_status.json
  BLOCKED / REVIEW_UNAVAILABLE（独立审查缺失时强制）
```

迭代规则：**只能补充证据，不能改动锚点、权重或 8 分达标线**。v1 与 v2 使用同一 rubric 与同一份脚本；分数变化全部来自新增的本地实测。

评分维度（各 0/1/2，总分 10，达标 ≥8）：

| 维度 | 2 分 | 1 分 | 0 分 |
|---|---|---|---|
| D1 问题锚定 | 本地量化验证的缺陷 | 仅文献主张 | 无关联 |
| D2 可证伪 | 预注册指标 + 基线/消融 + 正/负/零三种解释 | 部分 | 无 |
| D3 可复现 | 本轮给出**判别性**实测（对照/界/判定） | 仅描述性或 partial | 缺资产 |
| D4 新颖性 | 外部多源查新 + 跨家族审查回执 | 本地 34 篇无近邻 + 区分点（标注 UNVERIFIED） | 本地清单出现任一近邻 |
| D5 信息量 | 结论两种符号都有价值 | 仅正向有用 | 无决策价值 |

> **完整性说明**：开发期曾把 D3/D4 从宽松**收紧**（发现"有测量即满分""本地查新即满分"有放水嫌疑），并修正 3 处会误导结论的问题（删失上界算出 1.25、预算曲线混样、CoT 通道结构性零值）。首次正式评分前冻结为 `v1.0`，v1→v2 未改任何锚点/权重/阈值。方向始终是**提高严格度**，不是降低标准。

---

## 1. Executive Summary

本轮把"长 CoT 遗忘到底有没有被正确度量"变成了可复现的实验问题。在 H20 上对同一份未修改权重的 `sangyon/LRM-target` 跑了 3 组生成，并用 Hy4 完成 LLM 评测。最关键的发现是：**现有 `entity_leak` 是"回答全文是否含 gold 关键短语"的子串匹配，它度量的是生成状态，而不是遗忘程度**——同一模型、同一 seed、同一批问题，仅因生成预算/状态不同，泄漏率从 `0.025` 摆动到 `1.0`（幅度 0.975）。LLM judge 与字符串指标 agreement = 1.0，说明 judge 继承同一语义，无法纠正该混淆。

10 个候选 idea 在冻结 rubric 下从 v1 的 **0/10 达标（均分 7.0）** 提升到 v2 的 **10/10 达标（均分 8.7，最低 8.0）**，达标完全靠补齐本地实证，不是靠调低标准。最高分被环境上限压在 **9.0**：外部多源查新与跨家族审查后端不可用，D4 全局封顶 1 分。

推荐首选 **I01 生成状态分层泄漏审计**，并把 I04/I05/I06/I07/I08/I10 作为同一诊断框架下的并列高分方案。

---

## 2. Literature Landscape

来源：`awesome-lrm-unlearning/data/papers.json`（34 篇，2024-10 ~ 2026-11），`README.md`、`survey/index.html`。

- **五类方法**：参数更新、推理时免训练轨迹调控、基准与整体评测、攻击/漏洞/记忆审计、逐步遗忘测 CoT 忠实度（`data/papers.json:8-38`）。
- **三类失败模式**（`README.md:58-62`）：Residual CoT leakage、Reasoning collapse、Decoding & reasoning-path bypass。生成分解为 `π(c,a|q)=π(c|q)π(a|q,c)`（`README.md:41-43`）；只约束答案 `a` 覆盖不到思维链 `c`。
- **开放问题**（`survey/index.html:1093-1110`）：Q21 ZeroThink/LessThink 鲁棒、Q22 CFE/AFE 好但 ShadowBench LELR 高（lexical anchoring）、Q23 暴露 thinking 降低攻击成本、Q24 构造式遗忘引入幻觉替代、Q25 六步"可证明遗忘"闭环，明确承认 O1/O5 未解前完整可证明性仍是开放目标。
- **真实记忆 vs 虚构基准**（`survey/index.html:1084-1085`）：虚构注入数据"transfers poorly to reality"。
- **指标现状**：10 基准 / 15 指标 / 10 框架（`data/benchmarks_metrics.json`），核心有 CFE/AFE/RIL/LELR/NFRS/K-Score 等，均为**无条件的率或分数**；没有一条把"生成是否被截断 / 是否坍塌 / 是否复述了问题里的实体"作为分层变量。这正是本轮 I01/I04/I07 的切入点。
- **审查状态自述**：`survey/index.html:1120`（A.3）记录本环境无跨模型审查后端，状态为 `REVIEW_UNAVAILABLE, not PASS`；与本次 `gate_status.json` 一致。

---

## 3. Problem Anchor 与验证

### 3.1 五个核心问题（`problem_validation.json` 全部 confirmed）

| ID | 问题 |
|---|---|
| P0 | **生成状态混淆**：`entity_leak` 是对 response 的子串匹配，同时被答案可达性/截断、推理坍塌、问题回显三种状态污染；低泄漏率不能证明遗忘 |
| P1 | 通道缺口：答案级指标可能漏掉 `<think>` 内的泄漏 |
| P2 | 解码路径绕过：不同策略泄漏率不同，单策略分数不可比 |
| P3 | judge-指标漂移：字符串匹配与 LLM judge 一致性、judge 失败改变分母 |
| P4 | 效用/坍塌权衡：`</think>` 闭合、答案可达性、retain 质量与泄漏联动 |

### 3.2 P0 的决定性实证

| | 历史 `eval_target`（512 tok） | 本轮 `eval_loop24`（192 tok） |
|---|---|---|
| 权重 / seed / 问题 | `sangyon/LRM-target` / 1001 / 同一批 | 完全相同 |
| `leak_rate` | **0.025** | **1.0** |
| `truncated_rate` | 0.975 | 1.0 |
| `coverage`（到达答案） | 0.15 | 0.0 |
| `degenerate_rate` | 0.975 | 0.0 |
| `question_echo_rate` | 0.025 | 1.0 |

样本对照：历史生成的开头是 `"Okay, let's main themes in his work is a AI assistant, AI assistant, right? ..."`（坍塌、不含实体名）；本轮是 `"Alright, let's dive into the themes Nikolai Abilov commonly explores in his works..."`（**只是复述了问题里已有的人名**）。两者 `key_phrase` 都是 `Nikolai Abilov`。

**结论**：泄漏率的 0.975 摆动完全由生成状态驱动。LLM judge 对 `loop24` 32 个样本判 `llm_leak_rate=1.0`、`agreement=1.0`，说明 judge 只是把同一子串语义重新表述了一次，不构成独立证伪。

### 3.3 与领域概况的对应

P0 是 `README.md:60-61` 两类失败模式（residual CoT leakage / reasoning collapse）在**评测端**的镜像：领域已指出模型会坍塌，但没有指出坍塌会让指标"看起来更好"。P2 对应 `survey:1093` Q21；P1 对应 `README:60`；P3 是 judge 作为 metric 的方法学问题（R-TOFU 代码注释亦明确 "LLM judge is a metric, NOT an independent cross-family review"）。

---

## 4. 给定方法代码（REDaCT）自行诊断

`code_diagnosis.json`：7 项全部经文件级复验（`verified_defect_count=7`），不是转述文档。

| ID | 发现 | 证据 |
|---|---|---|
| D1 | 复述（paraphrase）产物整体失败 | `paraphrases/forget10.jsonl` 273/273 `status=error`；deepseek 变体 400/400 `ok` |
| D2 | judge 后端与文档不一致 | `judge_afe_rtofu.py` 走 DeepSeek API，`PIPELINE.md:340-350` 仍写 Claude/GPT-4o |
| D3 | 阈值持久化静默改变判定率 | `PIPELINE.md:221-224` 明写 "silently change the flagging rate, not error" |
| D4 | 泄漏检测器权重未随仓库提供 | README:144-152 要求 `hf download`，本地无 classifier 目录 |
| D5 | R2MU 预生成范围为未实现桩 | `scripts/r2mu/pregen_extent.py:12-15` "Not implemented" |
| D6 | R-TOFU 输入数据未随仓库提供 | `data/rtofu/` 只有 `PROVENANCE.md` |
| D7 | 声明的 FA2 运行时不可用 | 配置 `attn_implementation=flash_attention_2`，`redact-r2mu` venv 无 `flash_attn` |

**诊断结论**：REDaCT 的推理时门控假设本身可测，但当前仓库的**复现契约**失效（检测器权重/阈值/数据/judge 后端均未版本绑定）。这直接生成了 I10，并让 I01/I04 的"评测契约"诉求获得工程依据。数据盘 `/jizhicfs/chenyliao` 已具备模型权重与 R-TOFU 数据，因此本轮改用 R-TOFU 推理引擎完成 GPU 复现，避免依赖缺失的检测器权重。

---

## 5. Ranked Ideas（10 个，按 v2 分数降序）

v1 = 提案态（无实测），v2 = 证据态；rubric 冻结 `cff0a224b94802cc`，达标线 8.0。

| 排名 | ID | Idea | D1-D5 | v1 | v2 | 锚点 | 本轮实测信号 |
|---|---|---|---|---|---|---|---|
| 1 | I01 | 生成状态分层泄漏审计 | 22212 | 7 | **9** | P0 | 32 样本：raw=1.0，coverage=0.75，censored 25%、question-echo 100%、collapsed 0% |
| 2 | I04 | 删失评测区间 | 22212 | 7 | **9** | P0 | 区间 [1.0, 1.0]，width=0（假设被反驳，阴性结果保留） |
| 3 | I05 | judge-指标分歧分层 | 22212 | 7 | **9** | P3 | 32 样本，30 判定成功 / 2 失败，agreement=1.0，llm_leak=1.0 |
| 4 | I06 | 强制答案可达性探针 | 22212 | 7 | **9** | P4 | DefaultCoT 到达率 0.0 vs 强制短思考 1.0，两者泄漏均 1.0 |
| 5 | I07 | retain 效用硬门控 | 22212 | 7 | **9** | P4 | retain 到达率 0.75、ROUGE 0.476，门限 0.5 → PASS |
| 6 | I08 | 可达性控制下的不变性重评 | 22212 | 7 | **9** | P2 | `gamma_controlled=0.0`（DefaultCoT 覆盖 0，其余策略 ARCL=1.0） |
| 7 | I10 | REDaCT 检测器阈值复现审计 | 22212 | 7 | **9** | P3 | 7/7 项复验缺陷，预注册阈值 ≥3 → 阳性 |
| 8 | I02 | 等预算解码对比 | 22112 | 7 | **8** | P2 | 分 tag 预算曲线已得；预注册 `gamma_budget` 未覆盖（网格仅 DefaultCoT）→ D3=1 |
| 9 | I03 | CoT-only 泄漏通道审计 | 22202 | 7 | **8** | P1 | 1024-tok 补跑 8/8 闭合 `</think>`，`channel_gap=0.0`（阴性）；本地清单 9 篇近邻 → D4=0 |
| 10 | I09 | 复述恢复 / lexical anchoring | 22112 | 7 | **8** | P1 | 400 条改写库可用，但无配对生成 → recovery 未测，D3=1 |

**统计**：v1 `0/10` 达标、均分 7.0；v2 `10/10` 达标、均分 8.7、最低 8.0、最高 9.0。

### 5.1 高分方案展开（优先复现）

**I01 — 生成状态分层泄漏审计（推荐首选）**
- **方法**：① 把每条生成分入 answer-reached / censored / collapsed / question-echo 四类；② 每类分别报条件泄漏率、coverage 与删失区间；③ 用分层估计重排各系统，不可判定的层单列而不记 0。
- **假设**：表观遗忘排序由各状态样本占比决定，而非权重级擦除。
- **最小实验**：本轮 48 行 GPU 生成 + 历史 512-token 记录（已完成）。
- **正/负/零**：状态占比跨系统/预算变化且解释泄漏摆动 / 占比稳定则原指标有效 / 覆盖过低无法估计。
- **风险**：状态分类规则本身需人工定义，可能引入新的主观性。
- **复现结果**：POSITIVE（摆动 0.975，且 `question_echo_rate` 1.0 vs 0.025 给出机制解释）。

**I04 — 删失评测区间**
- **方法**：把截断样本作为右删失，发布 `[optimistic, pessimistic]` 而非点估计；上界 clamp 到 1.0 保证是合法概率区间。
- **复现结果**：NEGATIVE（本批 optimistic 已达 1.0，区间宽度 0）。这条阴性结果本身有价值：它说明"当所有样本都泄漏时，删失不再增加不确定性"，界只在中间区间有判别力。

**I05 — judge-指标分歧分层**
- **复现结果**：WEAK NEGATIVE。本轮 agreement=1.0、失败率 2/32；但历史 `judge_target` 失败 24/27、`judge_r2mu01` agreement=0.917，说明**失败率与一致性随样本长度剧烈变化**，分层审计仍必要。Hy4 每回复消耗 reasoning tokens（本任务实测 54 tokens/次），批量成本不可忽略。

**I10 — REDaCT 检测器阈值复现审计**
- **方法**：把检测器阈值、judge 后端、数据版本绑定成可校验契约，在随仓库产物上逐项复验。
- **复现结果**：POSITIVE（7 项缺陷全部文件级验证 ≥3 的预注册阈值）。

### 5.2 未淘汰但受证据限制的方案

- **I02**：预算网格只覆盖 `DefaultCoT`，无法计算预注册的"匹配预算下跨策略 gamma"。分 tag 曲线已保留（不混样），D3 记 1 分。
- **I03**：补跑长预算后 8/8 样本都闭合 `</think>`，`channel_gap=0`，且答案通道泄漏率 1.0，即本模型上未观察到"仅在 CoT 泄漏"。这是**真实阴性结果**，不是先前那个结构性零值。本地清单中有 9 篇近邻（REDaCT、K-Bench 等），D4=0。
- **I09**：仓库 400 条 deepseek 改写库可用，但本批没有把改写探针与 GPU 生成配对，`recovery_measured=null`，状态标 `partial`，D3=1。**没有把"库可用"冒充"恢复率已测"**。

---

## 6. Novelty Verification

- 机械查新范围：`awesome-lrm-unlearning/data/papers.json` 的 **34 篇**本地条目（标题/摘要/分类关键词匹配，`novelty_scan.json`）。
- 结果：9/10 idea 本地 0 近邻；I03 有 9 篇近邻（含 REDaCT、K-Bench），因此 D4=0。
- **状态：`LOCAL-CATALOGUE-ONLY [UNVERIFIED]`**。本环境无 `codex/gemini/claude` CLI，也没有可用的外部文献检索后端去跑 `/novelty-check` 的多源检索与跨家族复核；未对任何 arXiv 编号/DOI 作外部核验，也未声称"已发表工作不存在"。因此 D4 对所有 idea 封顶 1 分，这是本批最高分只能到 9.0 的原因。

---

## 7. External Critical Review

**`REVIEW_UNAVAILABLE`**。`codex`、`gemini`、`claude` CLI 均未挂载（见 `gate_status.json`），无法取得带真实模型身份的跨家族评审回执。

- Hy4 LLM judge 只用于**评测指标**（`llm_judge.py` 与仓库注释均明确 "LLM judge is a metric, NOT an independent cross-family review"）；它不得替代 review receipt。
- 领域概况 `survey/index.html:1120`（A.3）记录完全相同的环境限制，并声明该 draft 未经独立跨模型审查、不得当作 review-passed。本轮沿用同一诚实口径。
- 因此本报告**没有**外部评审分数，也没有用同族模型的自评去凑一个 reviewer score。

---

## 8. Pilot Experiment Results（真实运行）

GPU：1× NVIDIA H20 96GB；venv `/jizhicfs/chenyliao/ops/venvs/r-tofu-ga`（torch 2.5.1+cu121）；fp32（该 host 上 bf16 KV-cache decode 会 SIGFPE）；`do_sample=False`，seed 1001；模型 `/jizhicfs/chenyliao/models/sangyon/LRM-target`（4 个 shard，权重未改动）。

| 运行 | 配置 | 产物 | 耗时 | 关键结果 |
|---|---|---|---|---|
| `loop24` | 8 forget + 4 retain × 4 策略，192 new tokens | `eval_loop24.jsonl`（48 行） | 40.8 s | 全部策略 `leak_rate=1.0`；DefaultCoT truncated=1.0 / reached=0.0；其余策略 truncated=0 / reached=1.0；`gamma=0.0` |
| `loop24_budget` | 8 forget × 预算 64/128/256/512，DefaultCoT | `eval_loop24_budget.jsonl` | — | 该 tag 曲线四个预算均 `leak_rate=1.0`，truncated 由 1.0 降到 0.75 |
| `loop24_cot` | 8 forget × 1024 tokens，DefaultCoT | `eval_loop24_cot.jsonl` | — | 8/8 生成闭合 `</think>`，可测通道；`channel_gap=0.0` |

LLM 评测（Hy4，`https://tokenlink.woa.com/openapi/v1/chat/completions`，`model=hy4`）：

| 运行 | 样本 | 成功/失败 | agreement | llm_leak_rate |
|---|---|---|---|---|
| `judge_loop24` | 32 | 30 / 2 | 1.0 | 1.0 |
| `judge_loop24_cot` | 8 | 8 / 0 | 1.0 | 1.0 |

GPU 预算：三次运行合计约 2 分钟量级，远低于 skill 的 `PILOT_MAX_HOURS=2` / `MAX_TOTAL_GPU_HOURS=8`。

**阴性/受限结果均保留**（按 "let pilots kill, not vibes" 与 "阴性结果同样可发表"）：I04 区间宽度 0、I08 `gamma_controlled=0`、I03 `channel_gap=0`、I02/I09 证据不足降分。

---

## 9. Evidence Gate

| 检查项 | 状态 |
|---|---|
| `codex` / `gemini` / `claude` CLI | 全部 MISSING → `REVIEW_UNAVAILABLE` |
| `run_state.py` / `idea_discovery_gate.py` / `reviewer_route.py` | `.aris/tools/`、`tools/`、`/root/.aris/repo/tools/` 全部未找到 |
| novelty-check / research-review 的跨家族回执 | 无法生成 |
| `idea_discovery_gate.py` 终检 | 无法执行 |

**结论：`BLOCKED`**。按 `idea-discovery` 的 per-stage evidence gate 规则，`done` 状态或章节标题不等于评审证据；没有真实身份的正面评审回执不得写 `accept`，终检非零退出时必须在报告里写出 `BLOCKED`，不得把工作流当作完成。本文件因此明确标注为研究内容已完成、门禁未通过。

HTML 渲染（`/render-html`）本轮跳过：属于非阻断的便利产物，且渲染 helper 同样未挂载。

---

## 10. Refined Proposal 与后续

- 提案：`refine-logs/FINAL_PROPOSAL.md`
- 实验计划：`refine-logs/EXPERIMENT_PLAN.md`
- 追踪表：`refine-logs/EXPERIMENT_TRACKER.md`
- 研究契约：`idea-stage/docs/research_contract.md`

下一批必须保持同一 rubric（`cff0a224b94802cc`）与 8 分线，优先级：

1. **解门禁**：挂载跨家族 reviewer（codex/gemini 或 manual review）并安装 ARIS helper，重跑 `/novelty-check` 与 `/research-review`，取得回执后再跑证据终检。在此之前不得宣称 PASS。
2. **补证据**：I02 跑跨策略 × 匹配预算网格；I09 把改写探针与 GPU 生成配对以真正测恢复率；I01/I04/I07 扩到多 seed、多系统（target / r2mu01 / r2mu05 / ours-inv / ctrl-single），检验状态分层能否区分"真遗忘"与"坍塌/拒答"。
3. **方法落地**：把状态分层做成评测库的默认输出，替换单点 `leak_rate`；为 REDaCT 检测器建立版本绑定契约（I10）。
4. **复算**：`python3 idea-stage/auto_loop.py --stage all` 会自动读取新产物并重排 10 个 idea；分数只随证据变化。

> "每批 10 个均 ≥8 分"是**力争目标而非承诺**：本批达成 10/10，是在冻结标准下靠补齐实证得到的；若后续批次出现证据不足或新颖性近邻，脚本会如实降分，不会为凑数放宽 D1–D5 或 8 分线。
