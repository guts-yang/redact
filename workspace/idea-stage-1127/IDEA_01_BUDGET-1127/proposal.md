# IDEA 01 · BUDGET — 推理预算单调安全的连续预算 min-max 遗忘
## Budget-Monotone Continuous-Budget Min-Max Unlearning for LRMs

| 项 | 值 |
|---|---|
| 排名 | **#1（第二轮 1127）** |
| 五维 (H/F/S/C/B) | 8 / 7 / 9 / 7 / 8 |
| 均分 | **7.80** |
| 裁决 | **Strong Accept**（机制分，worth pursuing, pending validation） |
| 状态 | provisional · 无跨家族评审 · evidence_gate = BLOCKED |
| 数据集 | R-TOFU `forget01/05/10`，DeepSeek-R1-Distill-Llama-8B |
| 基线 | R-TOFU CoT-only GA、R2MU、STaR-MCS、DISCO（首轮） |

---

## 1. Problem Anchor（一句话故事）

> R-TOFU 发现「少推理 / 零推理（ZeroThink / LessThink）解码会复活已遗忘内容」，但只把它当评测现象；STaR 的 MCS 也只是打分。BUDGET 把「给多少推理预算」从离散策略升级成**连续可控变量**，要求模型在任意预算下都不能把 forget 事实说出来，且预算越大越安全 —— 用序结构约束堵住「少想就漏、多想也漏」的整条复活带。

锚定问题：**LRM 遗忘必须对连续推理预算单调安全，而不是只在某几个离散解码策略上通过评测**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| R-TOFU (EMNLP 2025) | 发现 ZeroThink/LessThink 复活；step-wise 评测 | 发现，不做训练闭环；预算是离散枚举 |
| STaR (AAAI 2026) | MCS 多解码一致性**评分** | 指标，非训练目标；无预算序约束 |
| DISCO（首轮 #4） | 对离散策略集 $\mathcal P$ 做 $\min\max$ | 策略是有限集合；无「预算单调性」这一序结构 |
| GUARD (2026) | 学「自然遗忘轨迹」 | 单条轨迹质量，不跨预算 |
| R2MU (2025) | 全步表示误导 | collapse 0.95+；非预算感知 |

**Delta（一句话可验证）**：把推理预算 $b\in[0,B]$ 连续化，证明并优化一个**泄漏率关于预算的单调非增约束** $\partial_b \rho_b \le 0$，用连续 min-max + 单调正则替代离散策略枚举。据现有检索，未见把「预算连续性 + 单调安全」形式化为 LRM 训练目标的工作。

**与首轮 10 个的划界**：最像 `DISCO`（解码策略 min-max）。不同轴：DISCO 的作用对象是**离散策略集合** $\mathcal P=\{\text{Default, Less, Zero, Summary}\}$，是集合枚举；BUDGET 的作用对象是**连续预算标量** $b$，机制是序结构（单调性），不是策略枚举。二者可叠加（BUDGET 提供连续轴，DISCO 提供策略边界）。

---

## 3. 方法（LaTeX）

设生成轨迹在预算 $b$（允许的 $\max$ 推理 token / 强制 early-exit 步数）下产生 CoT $\mathcal T_b=(r_1,\dots,r_{T(b)}; a_b)$，$e_f$ 为 forget 实体，$\phi_\theta^{(\ell)}(r_t)$ 为步级隐藏表示。

**3.1 连续预算泄漏场**
$$
\rho_\theta(b)\;=\;\mathbb E_{x\in D_f}\Big[\,\mathbb 1\big[\,e_f\in \mathrm{Gen}_\theta(x;b)\,\big]\Big],\qquad b\in[0,B]
$$
其中 $\mathrm{Gen}_\theta(x;b)$ 为预算 $b$ 下的完整生成（含 CoT 与答案）。R-TOFU 的 ZeroThink / LessThink 只是 $b=0$ 与 $b\ll B$ 的采样点。

**3.2 预算单调安全约束（核心）**
$$
\frac{\partial \rho_\theta(b)}{\partial b}\;\le\;0\quad \forall b\in[0,B],
\qquad\Longleftrightarrow\quad
\rho_\theta(b')\le\rho_\theta(b)\ \ \text{for } b'>b .
$$
直觉：模型不允许「想得越多越漏」。用软惩罚实现（在 $K$ 个预算格点 $b_1<\dots<b_K$ 上）：
$$
\mathcal L_{\text{mono}}
=\sum_{k=1}^{K-1}\Big[\,\rho_\theta(b_k)-\rho_\theta(b_{k+1})+\varepsilon\,\Big]_+^{2}.
$$

**3.3 连续 min-max 目标**
$$
\min_\theta\ \underbrace{\int_0^B \rho_\theta(b)\,\mathrm db}_{\text{预算积分泄漏}}
\;+\;\lambda\,\mathcal L_{\text{mono}}
\;+\;\mu\,\underbrace{\mathrm{KL}\big(p_\theta(\cdot\mid x_{\text{ret}})\,\|\,p_{\theta_0}(\cdot\mid x_{\text{ret}})\big)}_{\text{retain 锚}}
\;+\;\nu\,\underbrace{\sum_{t}\|\phi_\theta(r_t)-\phi_{\theta_0}(r_t)\|_2^2}_{\text{非 forget 步结构保持}}
$$
对内层 $\max$ 的实现：在训练每一步从 $[0,B]$ 采样一批预算 $b\sim\mathrm{Unif}$ 并取最坏预算的梯度（等价 $\min_\theta\max_b$ 的随机近似）。

**3.4 预算-深度证书（机制推断，待验证）**
$$
\hat\rho_{\text{worst}}
=\max_{b\in[0,B]}\hat\rho_\theta(b)
\;\le\;\hat\rho_\theta(0)+\int_0^B\big[\partial_b\rho\big]_+\,\mathrm db
\xrightarrow[\mathcal L_{\text{mono}}\to 0]{}\hat\rho_\theta(0),
$$
即最坏预算泄漏被「零预算泄漏」加单调性余项界定；给出可报告的 $\text{BC}=\max_b\rho(b)-\rho(0)$（预算证书宽度）。

参数更新走 LoRA（$r=8$），base $\theta_0$ 冻结。

---

## 4. R-TOFU 接口

- 数据：`forget10/task_id=1` 的 400 forget + 400 protocol-retain + 100 real_authors + 117 world_facts。
- 预算实现：复用 `redact/inference.py` 的 `max_new_tokens` 与 `</think>` 切分，构造预算格点 $b\in\{0,64,128,256,512\}$（含 ZeroThink 前缀）。
- 指标：核心 $\max_b\rho(b)$、$\text{BC}$、R-TOFU step-wise ROUGE-L / Cosine / LLM-as-Judge，加 LEAK、collapse、Retain ROUGE-L。
- 开销：每步 $K$ 次不同预算前向，可用缓存激活复用。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | R-TOFU CoT-only GA / R2MU | $\max_b\rho(b)-\rho(B)$ 显著 $>0$（复活带存在，复现 R-TOFU 发现） |
| T1 | BUDGET（$\mathcal L_{\text{mono}}$ 开） | $\text{BC}\le 0.05$ 且 $\max_b\rho(b)\le 0.05$ 且 collapse 增幅 $<0.20$ |
| T2 | 消融：去掉 $\mathcal L_{\text{mono}}$，只留积分项 | $\text{BC}$ 显著大于 T1 → 证明「单调性」而非「平均压泄漏」是机制 |
| T3 | 消融：$\mathcal L_{\text{mono}}$ 换成 DISCO 的离散 4 策略 min-max | 连续预算中间点（$b=64/256$）泄漏回弹 → 证明「连续轴」必要 |

**决定性实验**：T1 vs T2（隔离单调性）与 T1 vs T3（隔离连续 vs 离散）。若均无差异，核心机制被驳倒 → 转向。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 连续 min-max 把整条复活带纳入优化，机制上压最坏预算泄漏；待 T1 验证 |
| Faster | 7（机制分） | 相对 exact-retrain 少若干数量级梯度步；但每步多 $K$ 次前向 |
| Stronger | 9（机制分） | **主轴**：直接对 R-TOFU ZeroThink 复活做训练闭环 + 序结构鲁棒；待跨 split 验证 |
| Cheaper | 7（机制分） | 复用冻结 target 与既有 CSV，无新标注；预算采样不增数据 |
| Broader | 8（机制分） | 「预算单调安全」可迁移到任意 test-time-compute 可调的推理模型 |

**裁决：Strong Accept**（两维 ≥8、零 CRITICAL）。高分全为机制分 → *worth pursuing, pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. 预算格点太粗，模型在格点间「钻空子」泄漏 → 需报最坏格点间插值上界，并加对抗预算采样。
2. 单调约束可能与 retain 锚冲突，压低正常长链推理 → 用 $\nu$ 结构保持项与 T1 的 Retain ROUGE-L 判据兜底。
3. ZeroThink（$b=0$）本身可能只是「没机会说」而非「已忘」 → 必须同时报 bare-prompt 泄漏（与首轮 BOUND 的教训一致）。
4. 本机无 GPU、checkpoint 在服务器；评审为执行者本地初筛（provisional），`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/budget/budget_sweep.py`（复用 `redact/inference.py` 的预算切分）与 `scripts/budget/mono_loss.py`；训练在 `scripts/upbd/train_upbd.py` / `diagnostic_train_hooked.py` 骨架上替换损失。冻结：fp32 / 1024 max new tokens / seeds 1001, 2002。
