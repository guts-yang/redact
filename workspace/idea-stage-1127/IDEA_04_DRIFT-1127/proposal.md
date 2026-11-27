# IDEA 04 · DRIFT — 可恢复漂移的 CoT 步级分解与效用回补
## Step-Wise Recoverable-Drift Decomposition and Post-Hoc Utility Restoration

| 项 | 值 |
|---|---|
| 排名 | **#4（第二轮 1127）** |
| 五维 (H/F/S/C/B) | 8 / 7 / 7 / 8 / 8 |
| 均分 | **7.60** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU `forget01/05/10`，DeepSeek-R1-Distill-Llama-8B |
| 基线 | Recoverable Drift / PTP-U (arXiv:2610.11915)、首轮 SEQU / CSDA |

---

## 1. Problem Anchor（一句话故事）

> 2026-10 的 Recoverable Drift 工作发现：遗忘之后模型有大量「可恢复漂移」—— 把内部表示往回拉，通用能力显著回升而遗忘程度几乎不变。但它是在**整体表示**上做一次性回拉（PTP-U）。LRM 的漂移其实是不均匀的：承重步该漂、非承重的推理结构步不该漂。DRIFT 把漂移分解到 CoT 步级，只回补「非承重步」的漂移，把效用找回来而不动遗忘本身。

锚定问题：**遗忘带来的效用损伤中，有多大一部分是「步级可恢复漂移」而非必要代价；能否只回补非承重步，做到遗忘不变、效用回升**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| Recoverable Drift / PTP-U (arXiv:2610.11915, 2026-10) | 发现可恢复漂移；整体表示回拉恢复效用 | 整体级回拉；不分解到 CoT 步；不区分承重/非承重 |
| SEQU（首轮 #7） | 多次**请求之间**的漂移状态 $D_n$ 与锚回补 | 跨请求的状态漂移；不是单次遗忘后步级漂移 |
| CSDA（首轮 #1） | 步级因果承重 $c_t$ 选择干预掩码 | 训练期选步；不做遗忘后回补 |
| ORTHO（首轮 #5） | 推理子空间正交保护 | 训练期保护；非事后回补 |

**Delta**：定义「步级可恢复漂移」 $\Delta_t=\phi_{\theta}(r_t)-\phi_{\theta_0}(r_t)$，按承重度加权分解，只对 $c_t$ 小的步做回补，并证明（机制推断）回补后遗忘度不变而效用回升。据现有检索，未见把可恢复漂移下沉到 LRM CoT 步级的工作。

**与首轮 10 个的划界**：最像 `SEQU`（漂移校正）与 `CSDA`（承重步）。不同轴：SEQU 是**跨请求时间序列**的漂移；DRIFT 是**单次遗忘后、同一轨迹内步级**的漂移分解 + 事后回补。CSDA 用 $c_t$ 做训练期干预，DRIFT 用 $c_t$ 做事后回补掩码（同一量、不同阶段、不同用途）。

---

## 3. 方法（LaTeX）

设 $\theta_0$ 为未遗忘模型，$\theta$ 为遗忘后模型，$\mathcal T=(r_1,\dots,r_T;a)$ 为 forget 集上的轨迹，$\phi^{(\ell)}(r_t)$ 为 $\ell$ 层步表示，$c_t$ 为步级因果承重（借 CSDA 的 activation-patching 估计，但此处在 $\theta_0$ 上算、detached）。

**3.1 步级漂移分解**
$$
\Delta_t^{(\ell)}=\phi_\theta^{(\ell)}(r_t)-\phi_{\theta_0}^{(\ell)}(r_t),\qquad
\Delta=\big[\Delta_1,\dots,\Delta_T\big].
$$
把总漂移按承重度分成两部分：
$$
\Delta=\underbrace{\Delta^{\text{load}}}_{\text{承重步：承载遗忘，应保留}}
+\underbrace{\Delta^{\text{free}}}_{\text{非承重步：结构漂移，应回补}},
\qquad
\Delta^{\text{load}}_t=\mathbb 1[c_t\ge\tau]\,\Delta_t .
$$

**3.2 步级回补目标（核心）**
$$
\min_{\delta}\ 
\underbrace{\sum_{t:\,c_t<\tau}\big\|\,\phi_{\theta+\delta}^{(\ell)}(r_t)-\phi_{\theta_0}^{(\ell)}(r_t)\,\big\|_2^2}_{\text{只回补非承重步}}
\;+\;\beta\,\underbrace{\sum_{t:\,c_t\ge\tau}\big\|\,\phi_{\theta+\delta}^{(\ell)}(r_t)-\phi_{\theta}^{(\ell)}(r_t)\,\big\|_2^2}_{\text{承重步不许动}}
\;+\;\gamma\,\|\delta\|_2^2 .
$$
$\delta$ 用一个低秩/LoRA 增量实现（不重训全模型），且**不接触 forget loss**。

**3.3 遗忘不变性判据（机制推断，待验证）**
$$
\big|\,\mathrm{Leak}(\theta+\delta)-\mathrm{Leak}(\theta)\,\big|\le\epsilon,
\qquad
\mathrm{Util}(\theta+\delta)-\mathrm{Util}(\theta)\ \ge\ \Delta_{\text{rec}}>0,
$$
其中承重步被 $\beta$ 项锁住，故泄漏主要由承重步决定、回补不扰动它；效用由非承重步结构支撑、回补后回升。报告「恢复比」$R=\Delta_{\text{rec}}/\|\Delta^{\text{free}}\|$。

---

## 4. R-TOFU 接口

- 数据：`forget10/task_id=1` 400 forget + retain；需 $\theta_0$ 与 $\theta$ 两份 checkpoint（仓库已有 `sangyon/LRM-target` 与各 adapter）。
- 步切分：复用 `redact/dataset/contextual_window.py` 的 `split_sentences` 与 `inference_regen.py` 的 `</think>` 边界（步 = CoT 句子）。
- 指标：Retain ROUGE-L、collapse、R-TOFU step-wise、LEAK（遗忘不变性）、新指标 $R$、$\|\Delta^{\text{free}}\|/\|\Delta\|$（可恢复占比）。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | 遗忘后不回补（基线 $\theta$） | 记录 $\mathrm{Leak}(\theta)$、$\mathrm{Util}(\theta)$ 与漂移分解占比 |
| T1 | DRIFT（步级回补） | $\|\mathrm{Leak}(\theta+\delta)-\mathrm{Leak}(\theta)\|\le 0.02$ 且 $\mathrm{Util}$ 回升 $\ge 0.10$ |
| T2 | 消融：整体回拉（等价 PTP-U，无 $c_t$ 掩码） | 泄漏回弹或效用回升更小 → 证明「步级分解」必要 |
| T3 | 消融：随机选等量步回补 | 效用回升显著劣于 T1 → 证明「承重度掩码」而非「回补步数少」是机制 |

**决定性实验**：T1 vs T3（隔离承重度掩码）与 T1 vs T2（隔离步级 vs 整体）。若均无差异，核心机制被驳倒 → 转向。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 效用显著回升而遗忘不变，机制上直接改善 forget-utility 权衡；待 T1 |
| Faster | 7（机制分） | 事后低秩回补，不重训主遗忘；相对 exact-retrain 快 |
| Stronger | 7（机制分） | 对「遗忘副作用」鲁棒；但需 $\theta_0$ 双 checkpoint，泛化性待验证 |
| Cheaper | 8（机制分） | **主轴之一**：无需新标注、无需重跑遗忘训练，只加一个轻量回补增量 |
| Broader | 8（机制分） | 「步级漂移分解」可迁移到任意带显式 CoT 的推理模型 |

**裁决：Accept with Revisions**。降档理由：① F1=MAJOR（Recoverable Drift / PTP-U 近邻强，须主动划界「步级分解 vs 整体回拉」）；② 依赖 $\theta_0$ 与遗忘 checkpoint 双份、且 $c_t$ 估计对 patching 分布敏感（F7 资源）。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. 承重步与非承重步并非二值，$\tau$ 敏感 → 用连续松弛 $w_t=\sigma((\tau-c_t)/\kappa)$ 并报 $\tau$ 扫描。
2. 回补可能顺带把 forget 知识从非承重步「捎回来」 → 用 T1 的泄漏不变性判据与 bare-prompt 泄漏兜底。
3. 若遗忘本身主要发生在共享语义层而非步级，分解失效 → 这是可证伪点，T1 vs T2 直接裁决。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/drift/step_drift.py`（双 checkpoint 步表示差 + $c_t$ 掩码）与 `scripts/drift/restore.py`（低秩回补）；复用 CSDA 的 `step_attribution.py` 估计 $c_t$。冻结：fp32 / seeds 1001, 2002。
