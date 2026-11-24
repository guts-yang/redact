# Idea Discovery Report — LRM/CoT 遗忘评测的状态分层审计

**Run ID**: `lrm-loop-20261124`
**Direction**: 面向大型推理模型（LRM）长思维链遗忘的可信评测与解码不变性
**Date**: 2026-11-24（v4：跨系统审计 + 外部查新 + 跨家族评审）
**Inputs**: `/usr/local/workspace/awesome-lrm-unlearning`（领域概况，34 篇）+ `/usr/local/workspace/redact`（REDaCT 方法代码）+ `/usr/local/workspace/R-TOFU/workspace/results`（5 个系统的历史生成）
**Pipeline**: 运行清单 → 跨系统状态审计 → 问题提炼与验证 → 代码自诊断 → 思路生成与冻结评分 → 外部查新 → 跨家族批判性评审 → 证据门禁
**Canonical deliverable**: 本文件。证据为 `idea-stage/*.json`，全部由 `auto_loop.py` 机械重算。
**Status**: 研究内容、外部评审、两轮 GPU/CPU 审计已完成；**证据门禁仍为 `BLOCKED`**（缺 ARIS helper，见 §9）。评分 v4 = 9/10 达标、均分 8.6、I09 被外部查新淘汰。
**第 2–5 轮更正（见 §13–§16）**：`forget01/05/10` 是**三种遗忘比例场景**（1%/5%/10%），不是"模型学没学过"。主表一律用**原始基线协议**（40 forget / 20 retain / 4 策略 / DefaultCoT@512、其余@256 / seed 1001）：5% 与 10% 场景的基线 `leak_rate` 0.869/0.888 → 0.000，干净单元 0.608/0.575 → 0.000（n=26/29，可评估），同时暴露 **79–81% 坍塌、retain ROUGE 0.546→0.425/0.433**。1% 场景 0.025 → 0.169 但两端干净单元均为 0，属状态效应。新增指标只作补充，不替代基线。

---

## 0. 闭环设计（可重复执行）

```bash
cd /usr/local/workspace/redact
python3 idea-stage/auto_loop.py --stage all
#   manifest → cross → verify → diagnose → novelty → pilot → score → review → gate
python3 idea-stage/cross_review.py        # 跨家族评审（Hy4），写 .aris/traces + cross_review.json
```

数据流：

```
运行清单 run_manifest.json（权重/提示/产物哈希）
  + 5 系统 × 4 策略历史生成 cross_system.json（无 GPU，纯重算）
  + loop24 GPU 生成 + Hy4 judge pilot_results.json
  + 本地 34 篇目录 novelty_scan.json
  + 外部检索 external_novelty.json（查询 + URL + 全文读取记录）
  + 跨家族评审 cross_review.json（指向不可变 trace）
        ↓ 冻结 rubric (hash cff0a224b94802cc)
  v1 提案态 → v2 证据态 → v3 外部查新 → v4 跨系统审计
        ↓ gate_status.json
  BLOCKED（helper 缺失）
```

迭代规则：**只能补充证据，不能改动锚点、权重或 8 分达标线**。rubric 哈希四次评分完全一致；分数变化全部可追溯到新增证据。

评分维度（各 0/1/2，总分 10，达标 ≥8）：

| 维度 | 2 分 | 1 分 | 0 分 |
|---|---|---|---|
| D1 问题锚定 | 本地量化验证的缺陷 | 仅文献主张 | 无关联 |
| D2 可证伪 | 预注册指标 + 基线/消融 + 正/负/零三种解释 | 部分 | 无 |
| D3 可复现 | 给出**判别性**实测（对照/界/判定） | 仅描述性或 partial | 缺资产 |
| D4 新颖性 | 外部多源查新通过 + 跨家族回执 | 本地无近邻但未核验 | 出现近邻 |
| D5 信息量 | 结论两种符号都有价值 | 仅正向有用 | 无决策价值 |

---

## 1. Executive Summary

本轮把"长 CoT 遗忘到底有没有被正确度量"做成了**可复算的实验问题**，并得到了比上一版更强、更不客气的结论：

1. **同一权重、同一 seed、同一批问题，仅改生成预算，`leak_rate` 从 `0.025` 摆到 `1.0`**（幅度 0.975）。
2. **`forget01/05/10` 是三种遗忘比例场景**（1%/5%/10%），彼此不可直接比较；同一场景内也要先看状态再读泄漏率。
3. **在两个场景各做一次遗忘前后对照，指标符号相反**（B18）：
   - forget01：target `0.025`（坍塌 88.8%）→ r2mu01 `0.169`（坍塌 15.6%）——"越遗忘越泄漏"；
   - forget10：target `1.000`（坍塌 0%）→ r2mu10 `0.000`（**坍塌 87.5%**）——"完全遗忘"。
   两个面板的干净单元泄漏率都是 **0**，符号完全由坍塌率决定。
4. **因此 r2mu10 的 0.000 不可评估**：clean_n 只剩 2（<8）。B16 曾据此宣布"遗忘有效"，本轮已撤回（§14.3）。
5. **`r2mu05` 报 `raw_leak=0.0`，但坍塌率 96.3%**；`ours-inv`/`ctrl-single` 报 `raw_leak=0.0`，但 retain ROUGE-L 只有 **0.014 / 0.018**——三个"完美遗忘"的系统，一个是退化文本，两个把保留集答案一起废掉。
6. **retain 效用门在 5/5 系统上按 ROUGE 失败、按可达率全部通过**——现行"效用达标"是空过（B14）。

外部查新（真实检索 + 最近工作全文精读）**淘汰了 I09**（ShadowBench 已发表 lexical anchoring）、把 I03 判为重叠、把 I05 降为"应用级"。跨家族评审（Hy4）给 **PROCEED WITH CAUTION / 6.0**，明确把 I01、I04 的新颖性判为 **WEAKENED**，并提出 17 条"证据不足"的指控——本轮已逐条处置（§11），其中 1 条是我方**事实性错误**（已改正）。

推荐首选仍是 **I01 生成状态分层泄漏审计**；并列高分 I02/I04/I06/I07/I08/I10。最高分只能到 **9.0**：D4 的 2 分要求 reviewer 明确签"SUPPORTED"，而它只签了 WEAKENED——这是评审的结论，不是环境的上限。

---

## 2. Literature Landscape

来源：`awesome-lrm-unlearning/data/papers.json`（34 篇，2024-10 ~ 2026-11）+ 本轮外部检索（§6）。

- **五类方法**：参数更新、推理时免训练轨迹调控、基准与整体评测、攻击/漏洞/记忆审计、逐步遗忘测 CoT 忠实度。
- **三类失败模式**：Residual CoT leakage、Reasoning collapse、Decoding & reasoning-path bypass。生成分解为 `π(c,a|q)=π(c|q)π(a|q,c)`；只约束答案覆盖不到思维链。
- **开放问题**：ZeroThink/LessThink 鲁棒、CFE/AFE 好但 LELR 高（lexical anchoring）、暴露 thinking 降低攻击成本、构造式遗忘引入幻觉替代、"可证明遗忘"六步闭环仍未闭合。
- **指标现状**：10 基准 / 15 指标 / 10 框架，核心 CFE/AFE/RIL/LELR/NFRS/K-Score 均为**无条件的率或分数**；没有一条把"生成是否被截断 / 是否坍塌 / 是否回显了问题里的实体"作为分层变量。这正是 I01/I04/I07 的切入点。
- **外部同类诊断**：`arXiv:2506.00688`（已全文精读）证明遗忘评测结论随风控/格式而反转，但**未**形式化截断删失、覆盖率、条件泄漏率，也未按生成状态分层。

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

### 3.2 P0 的决定性实证（已更正样本叙述）

| | 历史 `eval_target`（512 tok） | 本轮 `eval_loop24`（192 tok） |
|---|---|---|
| 权重 / seed / 问题 | `sangyon/LRM-target` / 1001 / 同一批 | 完全相同 |
| `leak_rate` | **0.025**（4/160） | **1.0**（32/32） |
| `truncated_rate` | 0.975 | 1.0 |
| `coverage`（到达答案） | 0.15 | 0.0 |
| `degenerate_rate` | 0.975 | 0.0 |
| `question_echo_rate` | 0.025 | 1.0 |

**更正**（上一版写错，已按原始记录复核）：历史运行里那条坍塌样本（`"Okay, let's main themes in his work is a AI assistant..."`）**并没有**被判为泄漏（该行 `entity_leak=0`，`key_phrase="Nikolai Abilov"` 未出现在 response 中）。历史运行的 4 条泄漏全部是 `key_phrase="African American"` 且 response 反复复述了问题里的这个短语。

因此机制描述必须更准确：**两次运行的低/高泄漏都不是"知识回忆"，而是问题回显 + 退化重复**；差别只在于 192 tok 时回显立刻发生、512 tok 时模型先坍塌成 "AI assistant" 循环。上一版"同一指标同时把坍塌样本算作泄漏"的说法不成立，已删除。

**结论不变且更强**：0.975 的摆动完全由生成状态驱动，两个方向的读数都与"是否遗忘"无关。

### 3.3 与领域概况的对应

P0 是领域已指出的 residual CoT leakage / reasoning collapse 在**评测端**的镜像：领域说了模型会坍塌，但没说坍塌会让指标"看起来更好"。P2 对应 ZeroThink/LessThink 鲁棒性开放问题；P1 对应 residual CoT 通道；P3 是 judge 作为 metric 的方法学问题（R-TOFU 代码注释亦明写 "LLM judge is a metric, NOT an independent cross-family review"）。

---

## 4. 给定方法代码（REDaCT）自行诊断

`code_diagnosis.json`：7 项全部经文件级复验（`verified_defect_count=7`）。

| ID | 发现 | 证据 |
|---|---|---|
| D1 | 复述产物整体失败 | `paraphrases/forget10.jsonl` 273/273 `status=error`；deepseek 变体 400/400 `ok` |
| D2 | judge 后端与文档不一致 | `judge_afe_rtofu.py` 走 DeepSeek API，`PIPELINE.md:340-350` 仍写 Claude/GPT-4o |
| D3 | 阈值持久化静默改变判定率 | `PIPELINE.md:221-224` 明写 "silently change the flagging rate, not error" |
| D4 | 泄漏检测器权重未随仓库提供 | README:144-152 要求 `hf download`，本地无 classifier 目录 |
| D5 | R2MU 预生成范围为未实现桩 | `scripts/r2mu/pregen_extent.py:12-15` "Not implemented" |
| D6 | R-TOFU 输入数据未随仓库提供 | `data/rtofu/` 只有 `PROVENANCE.md` |
| D7 | 声明的 FA2 运行时不可用 | 配置 `attn_implementation=flash_attention_2`，`redact-r2mu` venv 无 `flash_attn` |

**诊断结论**：推理时门控假设本身可测，但当前仓库的**复现契约**失效（检测器权重/阈值/数据/judge 后端均未版本绑定）。这直接生成 I10。

---

## 5. Ranked Ideas（10 个，v4 分数）

v1 = 提案态；v2 = 加 GPU/judge/代码证据；v3 = 加外部查新 + 评审回执；v4 = 再加跨系统审计。rubric 冻结 `cff0a224b94802cc`，达标线 8.0。

| 排名 | ID | Idea | D1-D5 | v1 | v2 | v3 | v4 | 锚点 | 关键证据 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | I01 | 生成状态分层泄漏审计 | 22212 | 7 | 9 | 9 | **9** | P0 | 5 系统 × 4 策略重算；干净单元泄漏全 0；4 系统从"已遗忘"变为"不可评估" |
| 2 | I02 | 等预算解码对比 | 22212 | 7 | 8 | 8 | **9** | P2 | 匹配预算（同问题子集@256）`gamma=0.10` > 未匹配 `0.05` |
| 3 | I04 | 删失评测区间 | 22212 | 7 | 9 | 9 | **9** | P0 | 区间 [1.0,1.0]、width=0（阴性，保留） |
| 4 | I06 | 强制答案可达性探针 | 22212 | 7 | 9 | 9 | **9** | P4 | 到达率 0.0 vs 1.0，两者泄漏均 1.0 |
| 5 | I07 | retain 效用硬门控 | 22212 | 7 | 9 | 9 | **9** | P4 | ROUGE 门 5/5 失败，可达率门 5/5 通过 → 现行门是空过 |
| 6 | I08 | 可达性控制下的不变性重评 | 22212 | 7 | 9 | 9 | **9** | P2 | `gamma_controlled` 仅在 3 个可估计策略上为 0；全策略集未定义 |
| 7 | I10 | REDaCT 检测器阈值复现审计 | 22212 | 7 | 9 | 9 | **9** | P3 | 7/7 项复验，超预注册阈值 ≥3 |
| 8 | I03 | CoT-only 泄漏通道审计 | 22202 | 7 | 8 | 8 | **8** | P1 | `channel_gap=0`（8 样本阴性）；外部已有 `thk_leak` 通道指标 → D4=0 |
| 9 | I05 | judge-指标分歧分层 | 22102 | 7 | 9 | 8 | **8** | P3 | agreement=1.0（30/32）；judge 校准是通用实践 → D4=0 |
| 10 | I09 | 复述恢复 / lexical anchoring | 22102 | 7 | 8 | 7 | **7 ✗** | P1 | **外部查新命中 ShadowBench（NeurIPS 2026）→ 新颖性归零，淘汰** |

**统计**：v1 `0/10`（均分 7.0）→ v2 `10/10`（8.7）→ v3 `9/10`（8.5）→ v4 `9/10`（8.6，最低 7.0）。v2→v3 是**降分**：外部查新把 I09 打出局、把 I05 降级；v3→v4 只升了 I02。

### 5.1 高分方案展开（优先复现）

**I01 — 生成状态分层泄漏审计（推荐首选）**
- **方法**：把每条生成分入 `answer-reached / censored / collapsed / question-echo / no-answer-channel` 五类（第五类是本轮新发现的：`ours-inv`、`ctrl-single` 的 DefaultCoT 既未截断也未到达答案）；每类报条件泄漏率、coverage 与删失区间；用分层估计重排系统，不可判定的层单列而不记 0。
- **假设**：表观遗忘排序由各状态样本占比决定，而非权重级擦除。
- **复现结果**：**POSITIVE（分两层）**。
  - forget01（状态演示）：5 系统 raw_leak 0.000–0.169、collapse 0.00–0.96；干净单元泄漏 0；4 个系统掉出可排序集合。**但该 split 不是这个 checkpoint 学过的**（§13），只能证明指标随状态移动。
  - forget10 no-echo（有效测量）：干净单元（可达 & 非坍塌 & 无回显）**16 个样本，泄漏 16/16（DefaultCoT）**，ZeroThink 只有 6/16 → 干净读数可以非零，指标并非只会读到 0；同时暴露"答案通道被截断会掩盖 CoT 内泄漏"。
- **风险**：状态分类规则需人工定义（collapse 阈值 0.35），可能引入主观性 → B8 敏感性消融。

**I02 — 等预算解码对比**
- **方法**：固定 token 预算、固定问题子集，计算跨策略 `gamma_budget`，与未匹配预算的 `gamma_unmatched` 对比。
- **复现结果**：**NEGATIVE（方向相反）**。target 与 r2mu01 在共享问题集、256 tok 下 `gamma_matched=0.10`，而未匹配时为 `0.05`；另外 3 个系统两者均为 0。假设"差距是长度伪影"被反驳——差距在控制长度后**更大**。

**I04 — 删失评测区间**
- **复现结果**：**NEGATIVE**。本批 optimistic 已达 1.0，宽度 0。**注意**：本批没有预注册的决策阈值，因此上一版"跨阈值"的说法不成立，已降级为"该批区间无判别力"。

**I07 — retain 效用硬门控**
- **复现结果**：**POSITIVE（结论反转）**。门限 0.5 若作用在 `answer_reached`，5/5 系统通过（0.75–0.95）；若作用在 ROUGE-L recall，**5/5 全部失败**（0.014–0.369）。`ours-inv` 与 `ctrl-single` 保留集 ROUGE 仅 0.014 / 0.018，却报 `raw_leak=0.0`。

**I10 — REDaCT 检测器阈值复现审计**
- **复现结果**：**POSITIVE**（7 项缺陷全部文件级验证，超过预注册阈值 3）。

### 5.2 未淘汰但受证据限制

- **I03**：补跑 1024 tok 后 8/8 闭合 `</think>`，`channel_gap=0`，是本模型上的**真实阴性**；但外部已有 `arXiv:2605.18891` 的 `thk_leak` 与 K-Bench 的通道分解，方法不再新颖（D4=0）。
- **I05**：judge 与字符串 agreement=1.0（30/32），失败 2 条。通用 LLM-judge 校准/偏差文献已很密，只有"遗忘专用分母偏差"这层应用差异（D4=0）。
- **I09**：**已淘汰**。仓库 400 条改写库可用但从未与 GPU 生成配对；且 ShadowBench 已发表"删名字不等于删知识"，新颖性归零。

---

## 6. Novelty Verification（本轮真正做了外部查新）

- 本地机械扫描：34 篇目录，9/10 idea 零近邻，I03 有 9 篇（`novelty_scan.json`）。
- **外部检索**（`external_novelty.json`，5 条查询、7 条可解析 URL、1 篇全文精读）：arXiv API 与 Semantic Scholar API 均返回 HTTP 429，改用 web 检索后端逐条记录 URL 与核验等级。
- 结论（这是**降分**依据，不是加分）：
  - `arXiv:2506.00688`（全文精读，CMU/UK AISI/OATML）：最接近，提供"评测不可靠"论题与跨格式泄漏矩阵；**未**形式化截断删失、覆盖率、条件泄漏率，也**不**按生成状态分层 → I01/I04 的差异点成立，但它已经把"评测不可靠"这个大盘子占了。
  - `arXiv:2511.04934` Leak@k（ICML 2026）：采样多重性；与 I02/I08 相邻。
  - `arXiv:2605.18891` Head-Conditioned Canaries：已定义 `thk_leak` = 答案在思维链中的子串包含 + bypass gap → **I03 重叠**。
  - K-Bench（`2609.12808`）：六通道（含 CoT）→ I03 第二处重叠。
  - ShadowBench（NeurIPS 2026 D&B）：lexical anchoring → **I09 重叠，淘汰**。
  - `arXiv:2605.19537`：推理后端影响可复现性 → 支持 I10，但不抢其对象。
- **状态：`EXTERNAL-WEB-SEARCH (partial)`**。已核验的最近工作做了全文精读，其余为摘要级；没有伪造任何 arXiv 编号。

---

## 7. External Critical Review（跨家族，本次真的拿到了回执）

- **评审后端**：Hy4（`https://tokenlink.woa.com/openapi/v1/chat/completions`，`model=hy4`），用户指定；执行代理是 DeepSeek，故属跨家族。
- **回执**：`idea-stage/cross_review.json`，trace `.aris/traces/20261124T133949Z-hy4-external-critical-review.json`，`trace_sha256=02a69e25…7501`，`response_id=eaeb91e6…9485`。
- **结论**：`PROCEED_WITH_CAUTION`，**score 6.0/10**（这是"能否投稿"的另一个刻度，与 §5 的 rubric 分数不可互换）。
- **新颖性裁定**：I01 `WEAKENED`、I04 `WEAKENED` → 按 rubric，D4 不能给 2 分。这是本批最高分止于 9.0 的**直接原因**。
- **评审给出的最强理由**：同权重同问题只改预算，`leak_rate` 0.025→1.0，且 truncation/coverage/degeneracy/echo 同向移动，这是对"低泄漏=已遗忘"的直接否证，不是空泛的指标批评。
- **身份诚实性**：API 返回 `model=hy4`，但评审自述为 "OpenAI ChatGPT（版本未暴露）"。两者不一致，已原样记录，不追认为 `codex/gemini/claude` 任一后端。
- **它不能验证的**：实验是否真跑过、权重、日志、GPU 计时、我们检索到的网页内容。

### 7.1 评审的五条高危风险（原文要点）

1. 状态分层的**逐样本表根本不存在**，只有预算对照与几个阴性 pilot。
2. 状态标签依赖检测器，缺精确匹配跨度、prompt 哈希、检测器阈值。
3. 新颖性逼近 `2506.00688`，可能被判为"实例化"而非新贡献。
4. 跨系统可比性与 lexical anchoring 两类主张**无数据**（当时确实没有）。
5. 汇总自相矛盾：2 条 judge 失败却报 agreement=1.0；区间 [1.0,1.0]；阈值缺失；零覆盖下报 `gamma=0`；retain 门 ROUGE 0.476 却 pass。

风险 1/4 已由本轮跨系统审计**部分消解**（逐样本状态表现在存在，5 系统数据也在）；风险 5 与 §11 的事实性问题**已全部更正**；风险 2/3 仍未解，进入 §10 的 B10/B11。

---

## 8. Experiment Results（全部为真实产物）

### 8.1 运行清单（回应"你没审计权重"）

`run_manifest.json`：4 个权重 shard 全量 sha256 + 汇总 `1cf98e13…0047`；`config.json`/`generation_config.json`/`index.json` 哈希；`decode_eval.py`、`llm_judge.py` 哈希；60 条唯一问题的集合哈希 `ce876de0…dca7`；13 个产物文件逐个哈希 + mtime。**不能**证明运行发生过（无 profiler 日志），已在 `unverified_claims` 里写明。

### 8.2 GPU pilot（1× H20，greedy，fp32，seed 1001，全部约 2 分钟）

| 运行 | 配置 | 产物 | 关键结果 |
|---|---|---|---|
| `loop24` | 8 forget + 4 retain × 4 策略，192 tok | `eval_loop24.jsonl`（48 行） | 全部策略 `leak_rate=1.0`；DefaultCoT truncated=1.0 / reached=0.0；其余策略 reached=1.0 |
| `loop24_budget` | 8 forget × 预算 64/128/256/512 | `eval_loop24_budget.jsonl` | 四预算均 `leak_rate=1.0`；truncated 1.0→0.75 |
| `loop24_cot` | 8 forget × 1024 tok | `eval_loop24_cot.jsonl` | 8/8 闭合 `</think>`；`channel_gap=0.0` |

### 8.3 跨系统重算（无 GPU，`cross_system.json`）——仅状态演示

| 系统 | raw_leak | coverage | collapse | 可估计单元 | retain 可达 | retain ROUGE-L |
|---|---|---|---|---|---|---|
| target | 0.025 | 0.788 | 0.888 | ZeroThink | 0.85 | 0.192 |
| r2mu01 | 0.169 | 0.844 | 0.156 | 全部 4 个 | 0.95 | 0.346 |
| r2mu05 | **0.000** | 0.763 | **0.963** | **无** | 0.94 | 0.369 |
| ours-inv | **0.000** | 0.750 | 0.000 | 3 个 | 0.75 | **0.014** |
| ctrl-single | **0.000** | 0.750 | 0.000 | 3 个 | 0.75 | **0.018** |

- 干净单元泄漏率：**每个可估计单元都是 0**（target 仅 ZeroThink 有 9 个干净样本，泄漏 0；r2mu01 四个单元 10/29/32/34 个干净样本，泄漏全 0）。
- 匹配预算（同问题子集 @256）跨策略 `gamma`：target 0.10、r2mu01 0.10、其余 0.00。
- 主张状态：C1 `SUPPORTED`、C2 `SUPPORTED_ON_UNEVALUABILITY`、C3 `PARTIAL`、C4 `REFUTED_ON_THIS_BATCH`。

LLM 评测（Hy4）：`judge_loop24` 32 样本、30 成功 / 2 失败、agreement=1.0（分母为已判定样本）、`llm_leak_rate=1.0`。

**阴性/受限结果全部保留**：I04 宽度 0、I08 gamma 不可定义、I03 channel_gap=0、I02 假设方向相反、I09 未测。

---

## 9. Evidence Gate

| 检查项 | 状态 |
|---|---|
| 跨家族评审回执 | **有**：Hy4 `PROCEED_WITH_CAUTION`，含 model/trace/sha256/verdict |
| 外部多源查新 | **有**：5 查询 / 7 URL / 1 篇全文精读（`external_novelty.json`） |
| 评审是否签 I01/I04 新颖性 | **否**：WEAKENED → D4 保持 1，最高分止于 9.0 |
| `run_state.py` / `idea_discovery_gate.py` / `reviewer_route.py` | `.aris/tools/`、`tools/`、`/root/.aris/repo/tools/` 全部未找到 |
| `idea_discovery_gate.py` 终检 | 无法执行 |

**结论：`BLOCKED`**。评审这一侧的阻塞**已解除**（有带身份的回执与不可变 trace），剩下的阻塞是 ARIS helper 缺失、终检脚本无法运行。按规则不得写成 PASS，也不得用一个自制的"门禁脚本"自证通过。

---

## 10. 后续规划（已写入 `refine-logs/EXPERIMENT_PLAN.md` 的 B10–B15）

优先级 1（CPU/无 GPU，先把"可审计性"补齐）：
- **B10 逐样本状态表**：把 5 系统 × 4 策略的逐样本 `state / leak / match_span / echo` 落成 CSV，供外部复核；补 `prompt_hash`、`gold_phrase` 精确跨度。
- **B11 评审风险 2/3 的正面回应**：对 `2506.00688` 做逐行 delta 映射（它的"跨格式矩阵" vs 我们的"生成状态分层"），产出 1 页差异表；若映射显示已被隐含，则主动降级 I01 为"实例化"。

优先级 2（GPU ≤2h/条，按评审给出的判别实验）：
- **B12 paired no-echo 审计**：只用不含 gold 短语的改写探针，跑 DefaultCoT + 强制短思考 × {192,512,1024}，预注册判据：若受控干净单元内条件泄漏标签不变而 raw 泄漏变，则 C1 成立；若干净单元出现非零泄漏，则"零泄漏"是标签产物。
- **B13 多 seed**：seed {1001,1002,1003} 重复 B12，检验 C2 是否从"不可评估"升级为"排序反转"。
- **B14 retain 双门**：以 ROUGE 为主的效用门替换可达率门，重算 5 系统，确认 ours-inv/ctrl-single 的"完美遗忘"是否只是把答案废掉。

优先级 3（工程）：
- **B15 安装 ARIS helper 并跑终检**；在此之前门禁保持 `BLOCKED`。
- 把状态分层做成评测库默认输出（替换单点 `leak_rate`）；为 REDaCT 检测器建立版本绑定契约（I10）。

复算：`python3 idea-stage/auto_loop.py --stage all`；评审：`python3 idea-stage/cross_review.py`。分数只随证据变化，rubric 与 8 分线不动。

---

## 11. 评审意见处置表（17 条，逐条落地）

| # | 评审指控 | 处置 | 依据 |
|---|---|---|---|
| 1 | CoT 会泄漏"已被遗忘"的事实缺乏证据 | **限定表述**：改为"本模型 8 样本未观察到通道泄漏"，作为开放问题，不当结论 | `pilot_results.channel` |
| 2 | "恰好被三种状态因果混淆"不成立 | **改口径**：删因果措辞，改为"状态共变 + 机制证据"；C2 改判为 `SUPPORTED_ON_UNEVALUABILITY` | `cross_system.claims` |
| 3 | "跨系统不可比"无数据 | **已补数据**：5 系统重算，但只支持"不可评估"，不支持"排序反转" | `cross_system.systems` |
| 4 | "同权重同 seed 同问题"只是断言 | **部分关闭**：`run_manifest.json` 给出权重/配置/提示/产物哈希 | `run_manifest.json` |
| 5 | "在 H20 上跑了 2 分钟"不可审计 | **未关闭**：仅记录产物 mtime，无 profiler；列入 `unverified_claims` | 同上 |
| 6 | "`entity_leak` 是文献主指标" | **降级表述**：改为"R-TOFU 报告的头号指标" | §2 |
| 7 | 状态变量"测过"无协议 | **部分关闭**：协议即代码（`_degenerate` / `_echo`），已指明；人工校验仍缺 | `auto_loop.py` |
| 8 | "坍塌样本也被算作泄漏"自相矛盾 | **确认是我方错误**：历史样本 `entity_leak=0`；已改写 §3.2 | 原始 jsonl 复核 |
| 9 | "只是回显"是语义断言 | **改口径**：定义 `question_echo` 为结构性判定（问题含该短语且 response 含该短语），不主张语义归属 | `_echo()` |
| 10 | agreement=1.0 却 32 样本 | **已改**：30/32 已判定，2 条失败不计入分母 | `pilot_results.judge` |
| 11 | I04"反驳跨阈值"缺阈值 | **已改**：本批无预注册阈值，降级为"该批区间无判别力" | §5.1 |
| 12 | "模型级无通道泄漏"过强 | **已限定** 8 样本 | §5.2 |
| 13 | retain 门 pass 与 ROUGE 0.476 矛盾 | **已改并反转结论**：门在可达率上通过、在 ROUGE 上 5/5 失败 | `cross_system.*.retain_gate` |
| 14 | 零覆盖下 `gamma_controlled=0` | **已改**：仅 3 个可估计策略上为 0，全策略集未定义 | §5.1 I08 |
| 15 | 只有 DefaultCoT 预算曲线 | **部分关闭**：新增共享问题集 @256 的匹配预算对比 → gamma 0.10 | `cross_system.matched_budget` |
| 16 | 外部查新无据可查 | **已关闭**：记录 5 查询、7 URL、1 篇全文精读等级 | `external_novelty.json` |
| 17 | GPU/日志同上（与 5 同源） | **未关闭** | — |

合计：**10 条已修正/关闭、5 条需新测量、2 条口径降级**。其中第 8 条是我方事实错误，必须留痕。

---

## 12. 提交物料与复算入口

- 提案：`refine-logs/FINAL_PROPOSAL.md`
- 实验计划：`refine-logs/EXPERIMENT_PLAN.md`（B0–B15）
- 追踪表：`refine-logs/EXPERIMENT_TRACKER.md`
- 研究契约：`idea-stage/docs/research_contract.md`
- 证据：`problem_validation` / `code_diagnosis` / `novelty_scan` / `external_novelty` / `pilot_results` / `cross_system` / `run_manifest` / `cross_review` / `idea_scores{v1..v4}` / `gate_status`（均在 `idea-stage/`）
- 评审 trace：`.aris/traces/20261124T133949Z-hy4-external-critical-review.json`

> HTML 渲染（`/render-html`）仍未执行：helper 未挂载，属于非阻断的便利产物。

---

## 13. 第 2 轮：split 有效性复核（重大更正）

### 13.1 触发点

B12 要求"问题里不含实体名"的探针。用 `forget10.json` 构造出 920 条 no-echo 探针后，同一 checkpoint 在 **DefaultCoT 下 16/16 泄漏**——这与 forget01 上"几乎不泄漏"直接冲突。两个读数不可能同时描述遗忘，于是回头核对"这个模型到底学过哪个 split"。

### 13.2 证据

| 来源 | 结论 |
|---|---|
| `R-TOFU/workspace/idea-stage/CHAMPION_LOGIC.md` §1.1 | shipped `LRM-target` 与 `forget01.json` **不匹配**：gold-answer NLL **5.4692**、生成式实体复现率 **≤0.083** |
| 该文件 §1.2 | 后续工作不得不用 LoRA 在 forget01 全部 400 条上**做知识注入**（NLL → 0.5984，复现率 → 0.90）才得到可用基座 |
| 本轮 B12（`forget10` no-echo） | 同一 checkpoint 复现实体 **16/16**（DefaultCoT），说明它掌握 forget10 而非 forget01 |

### 13.3 后果（必须写进论文，不能省略）

1. **forget01 上的所有泄漏率重新分类**：`cross_system.json` 里的 0.000–0.169 是"从未知道 + 坍塌"的混合读数，只能证明**指标会随生成状态移动**，不能证明"这些系统遗忘了"。
2. **`C2_reranking` 从"部分支持"改为 `NOT_SUPPORTED_ON_A_VALID_SPLIT`**：4 个系统"离开可排序集合"发生在模型本就不认识的 split 上，不构成证据。
3. **`C0_split_validity` 记为 `DEFECT_CONFIRMED`**：这本身是一个可发表的方法学缺陷——**基线 / 论文若不先验证 checkpoint 是否掌握 forget 集，ρ_d 与 AFE/CFE 全部不可解释**。
4. **I07（retain 门）** 的"可达率门空过、ROUGE 门 5/5 失败"仍成立，因为它只依赖 retain 侧读数，与 forget split 无关。

### 13.4 有效 split 上的新测量（B12，`b12_noecho_audit.json`）

16 条 no-echo 探针、贪婪解码、seed 1001、同一 checkpoint：

| 策略 | 预算 | raw_leak | 可达率 | 干净单元 n | 干净单元泄漏 |
|---|---|---|---|---|---|
| DefaultCoT | 192 | 1.000 | 0.125 | 2 | 1.000 |
| DefaultCoT | 512 | 1.000 | 1.000 | 16 | **1.000** |
| DefaultCoT | 1024 | 1.000 | 1.000 | 16 | **1.000** |
| SummaryThink | 512 | 0.500 | 1.000 | 16 | 0.500 |
| LessThink | 512 | 0.438 | 1.000 | 16 | 0.438 |
| ZeroThink | 512 | 0.375 | 1.000 | 16 | **0.375** |

- **跨策略差 γ@512 = 0.625**（DefaultCoT 1.000 − ZeroThink 0.375），与"回显"无关。
- **预算不改变泄漏率**：四种策略的 `raw_leak` 在 192/512/1024 上完全相同（span = 0）。上一版"预算改变读数"的观察只存在于回显/坍塌区间，不能推广。

三条预注册判据（跑之前写死）：

| 判据 | 结果 | 说明 |
|---|---|---|
| (a) 每个匹配格 ≥8 个干净样本 | **FAIL** | DefaultCoT@192 只有 2 个（87.5% 截断）→ 该格不可用于估计 |
| (b) 干净单元标签在预算间不变、而 raw 改变 | **VACUOUSLY TRUE** | 两者都不变；判据设计时没预料到预算对 no-echo 探针无效，必须如实标注 |
| (c) 至少一条 no-echo 样本生成出 gold 短语 | **PASS** | 16/16 |

> 判据 (b) 是本轮设计缺陷：它预设"预算会移动 raw 泄漏"，而 no-echo 数据否定了这个前提。已按"阴性结果同样记录"处理，不改判据去凑阳性。

### 13.5 对主张与评分的净影响

| Claim | v4 之前 | 现在 |
|---|---|---|
| C0 split 有效性 | 未检查 | **DEFECT_CONFIRMED（新增）** |
| C1 状态混淆 | 已支持 | 已支持（范围收窄：状态影响读数，但 no-echo 区间内预算无效） |
| C1b 有效 split 测量 | — | **已支持（新增）**：γ=0.625，干净单元泄漏 0.375–1.000 |
| C2 排序反转 | 部分支持 | **不支持（有效 split 上未测）** |
| C3 judge | 部分支持 | 不变 |
| C4 删失区间 | 被反驳（本批） | 不变 |

评分不变（rubric 只认"是否有判别性实测"，而 B12 提供了更强的判别性实测）；但**论文主张必须按 13.3 改写**，否则就是拿"没学过"当"忘了"。

### 13.6 下一步（已加入 `EXPERIMENT_PLAN.md`）

- **B16（GPU ≤1h）**：在 `forget10` 上对 5 个系统重跑 no-echo 审计。注意 r2mu01/r2mu05 是在 forget01/05 上训练的，**不能**直接拿到 forget10 上比；需要各自用自己的 forget 集 + 先验证"训练前确实知道"。
- **B17（CPU）**：把"checkpoint 是否掌握 forget 集"做成审计前置门（NLL + 生成复现率双阈值），写进评测协议。
- B12 的 no-echo 探针集（920 条）已固化为 `idea-stage/probes/noecho_forget10.jsonl`。

### 14.2 B17 掌握度门：仪器已建，但**校准失败，暂不可用**

`idea-stage/know_check.py` 同时算 teacher-forced NLL 与生成复现率，结果三个信号互相矛盾（详见 `b17_calibration_failure.json`）：

| 信号 | 读数 | 含义 |
|---|---|---|
| 生成 | forget01 上 `entity_leak=0`、97.5% 退化 | "不知道 / 生成不出来" |
| teacher-forced NLL | **1.885**（< 2.0 门） | "知道" |
| 先验记录 `CHAMPION_LOGIC.md` | NLL **5.4692** | "不知道" |

原因之一：`forget01 ⊂ forget10`（400 题全在 3600 题内），两个 split 的 `task_id=1` 头部是同一批题，所以两次 NLL 完全相同——**这不是两次独立测量**。

结论：**门禁暂时不可启用**。必须先冻结唯一的 prompt/answer 模板，在同一模板下重算三个信号并要求一致。在此之前，对 forget01 只能陈述生成层面的事实（模型不产出该实体），"没学过"与"生成不出来"未区分——两者都足以否定"遗忘"解读，但机制不同。


---

## 14. 第 3 轮：第一个可解释的遗忘前后对照（B16）

前面的所有数字都缺一个前提：**"训练前的模型确实知道"**。`r2mu-forget10` 提供了唯一的干净链路——它与 `target` 共享 forget10 知识，只是被 R2MU 遗忘过。于是同一组 16 条 no-echo 探针、同一预算 512、同一解码设置：

| 策略 | target（遗忘前） | r2mu10（遗忘后） | Δ | 遗忘后 retain ROUGE-L |
|---|---|---|---|---|
| DefaultCoT | **1.000** | **0.000** | 1.000 | 0.546（过 0.5 门） |
| LessThink | 0.438 | 0.000 | 0.438 | 0.441（未过） |
| ZeroThink | 0.375 | 0.000 | 0.375 | 0.466（未过） |

- **跨策略差 γ：0.625 → 0.000**。在这个 checkpoint 上，"解码路径绕过"在遗忘后**没有残留**；绕过只发生在未遗忘的模型上。
- **这不是全局压答案**：retain ROUGE 保持 0.44–0.55，而 `ours-inv`/`ctrl-single` 是 0.014/0.018（那是压答案）。
- 因此本批终于具备一条**可写进论文的主结果**：在掌握 split 的 checkpoint 上，`entity_leak` 从 1.000 降到 0.000，而同一指标在未遗忘模型上的读数随解码策略在 0.375–1.000 之间摆动——**单点报数会把"最强泄漏"报成"0.375"**。

### 14.1 环境缺陷（必须记录）

| ID | 缺陷 | 影响 |
|---|---|---|
| E1 | `r-tofu-ga`（transformers 4.43 / tokenizers 0.20）无法解析 r2mu checkpoint 的 `tokenizer.json`（17 MB，较新格式，报 `untagged enum ModelWrapper`） | 该 venv 下 r2mu 系列**完全不可加载**；本轮改用 `redact-r2mu`（transformers 5.4 / tokenizers 0.22） |
| E2 | `ours-inv` / `ctrl-single` 的 adapter 由 `paths.resolve_model()` 挂在 **`TARGET_MODEL`** 上，但这两个 adapter 是在 `base_merged`（forget01 知识注入基座）上训练的 | 这两套系统的读数**不可解释**，已从有效证据中排除 |
| E3 | `forget01` 与本 checkpoint 不匹配（§13） | 所有 forget01 读数降级为状态演示 |

E1 的修复（换 venv）已在本轮生效；E2/E3 需要重建基线，不是本轮能自动完成的。

---

## 15. 第 4 轮：场景比例更正 + 最终口径（撤回上一版两处结论）

### 15.1 更正一：`forget01/05/10` 是三种遗忘比例场景，不是"模型没学过"

上一版（§13）把 forget01 读成"checkpoint 未学过该 split"，据此把 5 系统数字判为无效。**这个判断是错的**：forget01/05/10 是同一基准下 1%/5%/10% 三种遗忘比例，模型在三种比例下都应被评估，只是**不同比例之间不可直接比较**。

需要保留的部分：**同一场景内必须先看状态，再读泄漏率**；以及"跨比例横向排一张表"仍然是混淆（这是 `cross_system.json` 的真正问题，不是"没学过"）。

撤回的部分：E3（split 失配）、以及"所有 forget01 读数无效"的说法。`cross_system.json` 的数字仍可用于 1% 场景内的状态演示。

### 15.2 更正二：主干结论必须用基线协议，no-echo 子集只能作补充

上一版用 16 条 no-echo 探针得出"遗忘后没有可评估的干净单元"，**已撤回**：那是子集太小造成的，不是模型性质。按原始基线协议（40 forget / 20 retain / 4 策略 / DefaultCoT@512、其余@256 / seed 1001）重跑后，干净单元在遗忘后仍有 26–29 个（≥8），可评估。

### 15.3 基线协议下的三个场景（主表，`b18_scenario_panels.json` → `primary`）

| 场景 | 基线指标 leak（前→后） | 干净单元泄漏（前→后，n） | 遗忘后坍塌率 | retain ROUGE（前→后） |
|---|---|---|---|---|
| 1%（forget01，40 题） | 0.025 → 0.169 | 0.000 (18) → 0.000 (105) | 15.6% | 0.192 → 0.346 |
| 5%（forget05，40 题） | **0.869 → 0.000** | 0.608 (51) → **0.000** (26) | **80.6%** | 0.546 → **0.425** |
| 10%（forget10，40 题） | **0.888 → 0.000** | 0.575 (40) → **0.000** (29) | **79.4%** | 0.546 → **0.433** |

**读数（基线指标与新增指标并列，谁也不替代谁）**：

1. **基线指标本身是对的**：5%/10% 场景遗忘后确实降到 0，且衰减前的 0.87–0.89 说明模型确实掌握该比例的作者。这部分必须按原口径报。
2. **新增指标补了两件基线报不出来的事**：遗忘后遗忘集内 **79–81% 坍塌、88–100% 截断**；retain ROUGE 从 0.546 掉到 0.425–0.433。也就是说"泄漏归零"伴随明显的生成退化与效用损失。
3. **1% 场景的符号相反是真实的**：0.025 → 0.169，但两端的干净单元都是 0，所以这个上升仍然只是状态效应（模型在该比例的题目上本就坍塌），不能读成"越遗忘越泄漏"。

> 方法学要求：任何表必须先报基线 `leak_rate`（及 retain ROUGE / truncated / answer_reached），再把 clean-cell leak 作为**新增指标**附在其后；两者不可互换，也不可只报其一。

### 15.4 最终 claim 状态

| Claim | 状态 | 依据 |
|---|---|---|
| C0 场景比例 | 限定条件，非缺陷 | forget01/05/10 = 1%/5%/10%，跨比例不可横向比较 |
| C1 状态混淆 | **支持** | 同权重同题改预算 0.025↔1.0；1% 场景前后符号由状态决定 |
| C2 分层改变可评估性 | **支持（弱）** | 同一批数据里 clean_n 18–154，存在大量无法估计的格 |
| C3 judge 不能替代分层 | 部分支持 | agreement=1.0（30/32） |
| C4 删失区间有判别力 | 本批否定 | width=0，无预注册阈值 |
| C5 基线达标 ≠ 行为无代价 | **支持（新，基线协议）** | 5%/10% 的 0.869/0.888→0.000 同时伴随 79–81% 坍塌、retain 0.546→0.425/0.433 |
| C6 指标对齐 | **强制约束** | 主表用基线 leak_rate / retain ROUGE / truncated / answer_reached；新增指标只作并列补充 |

### 15.5 剩余未完成

- B11 done；**B13 已做 seed 复现，但只在 forget10 面板**；
- **B17 门仍不可用**（三信号不一致，需冻结模板）；
- **E2 未解决**：`ours-inv`/`ctrl-single` 的 adapter 挂在 `TARGET_MODEL` 而非训练基座 `base_merged`，其读数仍不可用；
- 论文若要主张"r2mu10 真的遗忘了"，必须先解决 87.5% 坍塌（例如强制短思考 + 更短预算让答案通道闭合），否则只能报"不可评估"。

---

## 16. 指标对齐（本轮新增的硬约束）

**规则**（用户 2026-11-24 指定）：训练与评估必须对齐原始基线；除非新指标本身就是创新主张，否则一律以基线为参照，并且**新指标只能并列追加，不能替换基线指标**。

### 16.1 基线协议（`code/paths.py`、`code/strategies.py`、`code/decode_eval.py`）

```
forget   : <split>.json, task_id=1, n_forget=40
retain   : retain90.json, n_retain=20
decode   : do_sample=False, seed=1001, fp32, batch 8
budget   : DefaultCoT 512 / LessThink 256 / ZeroThink 256 / SummaryThink 256
metrics  : leak_rate, answer_leak_rate, rouge_mean, truncated_rate, answer_reached_rate
```

### 16.2 对齐核查（`metric_parity.json`，逐 run 自动判定）

| 项 | 数量 |
|---|---|
| 扫描 run 总数 | 37 |
| **ALIGNED**（完全符合基线协议） | **9** |
| DEVIATION | 28 |

**9 个对齐 run** 包括 5 个系统基线（`target`、`r2mu01`、`r2mu05`、`ours-inv`、`ctrl-single`）与 4 个本轮新跑的主表面板（`base_target_forget05` / `base_r2mu05_forget05` / `base_target_forget10` / `base_r2mu10_forget10`）。

**28 个偏离 run** 的三类原因，全部为**有意设计**，已在表中标注：

| 偏离类型 | run | 理由 | 处置 |
|---|---|---|---|
| 换成 no-echo 探针集 | `b12_*`、`b13_*`、`b16_*`、`b19_*` | 检验"泄漏是否只来自问题回显" | 作为**补充面板**，不与基线表混排 |
| 预算网格（DefaultCoT 64–512） | `*_budget`、`loop24_budget` | 检验"预算是否改变读数" | 同上 |
| 小样本/短预算 | `loop24`（8+4）、`sanity`、`loop24_cot`（1024） | pilot 成本控制 | 只在 pilot 段落引用 |

### 16.3 新增指标（只作为创新点，不替换基线）

四项都和基线指标并列报出，且各自写明"基线报不出什么"：

| 新增指标 | 定义 | 基线报不出的东西 |
|---|---|---|
| `clean_cell_leak` | 在"可达 & 非坍塌 & 无回显"子集内的泄漏率 | 基线把坍塌/回显样本一起计入，掩盖真实生成 |
| `coverage` | `answer_reached` 的比例 | 基线不给估计量的支撑度 |
| `censoring_interval` | 截断样本作为右删失的 `[optimistic, pessimistic]` | 基线只报点估计 |
| `state_mix` | 五态占比（可达/截断/坍塌/回显/无答案通道） | 基线只有一个标量 |

**验收方式**：任何后续表格若出现 `clean_cell_leak` 而无同行 `leak_rate`，视为违规；反之只报 `leak_rate` 而本章列的四项缺失，必须注明"未测"。

### 16.4 已知元数据缺陷

`decode_eval.py` 不把 `--forget-file` 写进 `summary.json`，导致探针集只能靠 tag 前缀反推（`metric_parity.json` 已记录）。修复建议：在 summary 里记 `forget_file` 路径与其 sha256；这一项已加入 `EXPERIMENT_PLAN.md` 的 B20。
