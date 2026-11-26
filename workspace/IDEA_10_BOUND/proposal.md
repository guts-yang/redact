# IDEA 10 · BOUND — 承诺边界可控早停 + 删失评测
## Commitment-Boundary Early Exit with Censoring-Aware Evaluation

| 项 | 值 |
|---|---|
| 排名 | **#10** |
| 五维 (H/F/S/C/B) | 8 / 8 / 7 / 8 / 6 |
| 均分 | **7.4** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU（预算 96/192/512/1024 多档） |
| 基线 | 固定 1024 预算生成、`idea-stage/evidence/i04_baseline_intervals.json`（删失区间） |

---

## 1. Problem Anchor（一句话故事）

> 仓库实测「截断率 0.93–0.95 时，泄漏率 0.000 只是生成被截断的另一种说法」，删失区间宽到 1.0、完全不含信息。BOUND 不再被动接受这个结论：它检测 CoT 的**承诺边界**（哪一步之后再想也不会改变答案），在该边界主动干净退出，让「零泄漏」变成可判定而非删失。

锚定问题：**把「不可判定」变成「可判定」**，同时省算力。这是 `N02 预算鲁棒遗忘` 的方向升级：N02 给删失**区间评测**，BOUND 给**训练/生成机制**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| 仓库 `N02`（WEAKENED, 9/10） | 把截断当右删失，发布区间 $[lo,hi]$，多预算最坏情况**训练** | 区间是**评测侧**；训练仅「多预算压 margin」，无承诺边界检测 / 主动早停 |
| CoT collapse 综述（emergentmind, 二手） | `commitment boundaries`、`dynamic early-exit`、`coupling tax` 概念 | 通用推理效率文献，非遗忘，无删失校正 |
| R-TOFU | 报告截断 / collapse 指标 | 不把截断从泄漏证据里分离 |
| 仓库 `N10` | 单轮 Δleak 状态分解 | 分解，不给早停机制 |

**Delta**：用步级概率/激活检测承诺边界 $\tau^\*$，在其后注入干净退出；并把删失区间宽度作为**主判据**。据现有检索，未见「承诺边界早停 + 删失区间收窄」联用于 LRM 遗忘。

---

## 3. 方法（LaTeX）

**3.1 承诺边界**
设步 $t$ 后答案分布 $p_t(a\mid r_{\le t})$，定义
$$
\tau^\*=\min\Big\{t:\ \underbrace{\mathrm{JS}\big(p_t(\cdot),p_{t+1}(\cdot)\big)}_{\text{后续步不再改变答案}}\le\delta_{\text{JS}},\ \forall t'\ge t\Big\}
$$
实践中用步级 logit 熵 / 激活变化率的单调检测器 $\hat\tau$ 近似，避免提前提交。

**3.2 边界处干净退出**
$$
y_{\text{exit}}=\texttt{SAFE\_COT}(r_{\le \hat\tau})\oplus \texttt{IDK},
\qquad
\mathcal{L}_{\text{exit}}=\mathrm{CE}\big(y_{\text{exit}}\mid x\big)+\mu\,\mathcal{L}_{\text{forget}}(r_{\le\hat\tau})
$$
即：在承诺边界**之前**已把推理导向安全轨迹，边界处显式闭合，避免「还没想完就被截断」。

**3.3 多预算最坏情况（沿用 N02 思路，但作为约束）**
$$
\mathcal{L}_{\text{bound}}=\max_{b\in\{96,192,512,1024\}}
\Big[\mathcal{L}_{\text{leak}}(\theta;b)+\zeta\,\mathcal{L}_{\text{exit}}(\theta;b)\Big]
$$

**3.4 删失评测（核心判据）**
观测 $n$ 条里 $\ell$ 条泄漏、$u$ 条被截断：
$$
[\,lo,\,hi\,]=\Big[\frac{\ell}{n},\ \min\big(1,\frac{\ell+u}{n}\big)\Big],
\qquad
W=hi-lo
$$
BOUND 的成功标准：$W\le 0.2$ **且** $lo\le 0.05$（可判定且安全）。这与 N02 的阈值一致，但由机制而非仅区间达成。

---

## 4. R-TOFU 接口

- 多预算生成复用 `redact/inference.py`（`max_new_tokens` 可调）；步级 JS 需缓存中间 logits（`inference_regen.py` 已有 `</think>` 边界与步切分）。
- 指标：$W$（区间宽度）、$\hat\tau$ 分布、每预算 LEAK、Retain ROUGE-L、collapse、token 节省量。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | 固定 1024 预算 R2MU | 复现 $W\approx 1.0$（N02 实测 `base_r2mu10_forget10`） |
| T1 | BOUND（边界早停 + 多预算约束） | $W\le 0.2$、$lo\le 0.05$、token 用量降 ≥ 30% |
| T2 | 消融：去掉 96 预算项 | $W$ 回到 > 0.5（沿用 N02 的 T2 判据） |
| T3 | 消融：随机 $\hat\tau$（无承诺检测） | $W$ 回弹或 collapse 上升 → 证明边界检测必要 |

**决定性实验**：T1 vs T3（隔离承诺边界机制，区别于 N02 的单纯多预算压 margin）。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 把「不可判定」变「可判定」+ 真零泄漏；待 T1 |
| Faster | 8（机制分） | 边界早停直接省 token（目标 ≥30%），且相对 exact-retrain 快 |
| Stronger | 7（机制分） | 对预算漂移鲁棒，但承诺检测器跨模型稳定性未验证故不给 8 |
| Cheaper | 8（机制分） | 省生成预算 + 复用既有 CSV + 无重训 |
| Broader | 6（机制分） | 删失/早停可迁移，但与「预算/解码协议」强绑定，普适性最弱 |

**裁决：Accept with Revisions**。降档理由：① F1=MAJOR（N02 删失区间、CoT early-exit 文献近邻强，且 N02 本身 9/10 WEAKENED；必须主动划界「机制 vs 评测」）；② 承诺边界检测器是新增工程依赖，需验证跨 split 稳定（范围修订）。机制分 → *pending validation*。

---

## 7. 失败模式与局限（F10）

1. $\hat\tau$ 检测错误（过早提交）会把「还没推理完就退出」误判为遗忘 → 须与 DISCO 的裸 prompt 探针联用，确认参数级也无残留。
2. 早停省了 token 但可能掩盖真实泄漏（模型没机会说） → 这正是 $W$ 判据要防的，必须同时报 bare-prompt 泄漏。
3. 与 `N02 / N10` 相邻：N02 给区间、N10 给分解、BOUND 给早停机制，三者不重复但论文需清晰划界。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/bound/commitment.py`（步级 JS / 熵检测）与 `scripts/bound/censor.py`（区间计算，可复用 `idea-stage/tools/fixes.py i04` 的思路）；生成侧在 `redact/inference.py` 接早停 hook。冻结：fp32 / seeds 1001, 2002 / 预算集预注册。
