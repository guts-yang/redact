# IDEA 07 · SEQU — 序列遗忘的 CoT 漂移校正
## Sequential Unlearning with CoT Drift Correction

| 项 | 值 |
|---|---|
| 排名 | **#7** |
| 五维 (H/F/S/C/B) | 8 / 6 / 8 / 7 / 8 |
| 均分 | **7.4** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU 顺序 forget 请求流（forget01→05→10 连续删除） |
| 基线 | 单轮 R2MU / NPO 重复应用、无锚重训（精确但昂贵） |

---

## 1. Problem Anchor（一句话故事）

> 真实系统不是「忘一次」，而是 GDPR 请求一个接一个来；连续近似遗忘会累积误差漂移，第 N 次删除后模型连 retain 都坏了。SEQU 把顺序遗忘建模成带锚的递归校准，用 CoT **漂移量**而非轮数来决定何时回补。

锚定问题：**LRM 顺序遗忘的累积漂移应在推理链层被测量与抑制**。综述（IJERT 2026 / 阿里云综述）把 `Sequential Streaming Unlearning` 列为十三个未解方向之一。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| 综述「Sequential Streaming Unlearning」(二手来源) | 指出累积 error drift，建议 anchor regularization / recursive calibration | 仅列为 future direction，无具体 LRM 方法 |
| 仓库 `N10 状态位移`（单轮） | 单轮内把 collapse 从 Δleak 里扣掉 | 不管**多轮累积漂移** |
| 持续学习 EWC / 回放 | 通用抗遗忘 | 目标是「别忘旧任务」，与「持续删知识」方向相反 |
| 仓库 `N06 状态配对` | 单轮跨解码状态同态 | 无时间轴 |

**Delta**：定义「CoT 漂移状态」 $D_n$，给出递归校正与**漂移触发的锚回补**条件，并以 R-TOFU step-wise 指标量化。据现有检索，未见在 LRM CoT 上做序列遗忘漂移校正的工作。

---

## 3. 方法（LaTeX）

**3.1 请求流与漂移状态**
$$
\mathcal{Q}=(q_1,q_2,\dots,q_N),\quad
\theta_n=\mathcal{U}(\theta_{n-1},q_n),\quad
D_n=\mathbb{E}_{x\in R}\Big[\,\big\|\phi_{\theta_n}(x)-\phi_{\theta_0}(x)\big\|_2\,\Big]
$$
$D_n$ 是 retain 锚上的表示漂移，度量「前 $n$ 次删除把离原始模型多远」。

**3.2 递归校正目标**
$$
\boxed{\;
\theta_n=\arg\min_\theta\ \mathcal{L}_{\text{forget}}(\theta;q_n)
+\lambda_n\,\mathrm{KL}\big(p_\theta(\cdot\mid x_r)\,\|\,p_{\theta_0}(\cdot\mid x_r)\big)
+\rho\,\big\|\theta-\theta_{n-1}^{\text{anchor}}\big\|_2^2
\;}
$$
其中 $\lambda_n=\lambda_0(1+D_{n-1})$：漂移越大，retain 锚越强（与 EAS 的 $\lambda(\kappa)$ 同构，但沿**时间轴**而非样本轴）。

**3.3 漂移触发的回补**
$$
\text{if } D_n>\delta:\quad
\theta_n\leftarrow \theta_n-\eta\,\nabla_\theta\mathcal{L}_{\text{anchor}},\qquad
\mathcal{L}_{\text{anchor}}=\sum_{t\in\text{CoT}_{\text{retain}}}\big\|\phi_{\theta_n}(r_t)-\phi_{\theta_0}(r_t)\big\|_2^2
$$
即只在漂移超阈时做一次 CoT 步级回补，避免每次请求都全量锚定（省算力）。

**3.4 误差累积界（机制推断，待验证）**
$$
D_n\;\le\;\sum_{i=1}^{n}\alpha_i\;\xrightarrow[\text{触发回补}]{}\;D_n\;\le\;\delta+\mathcal{O}(\alpha_{\max})
$$
把无界累积压成有界。该式若实测不成立，机制被驳倒。

---

## 4. R-TOFU 接口

- 构造请求流：从 forget10 里分 $N=5$ 批顺序删除，每批后测 forget/retain/collapse/step-wise。
- 指标：每轮 Leak、Retain ROUGE-L、collapse + 核心 $D_n$ 曲线与「回补触发次数」。
- 对照：重复单轮 R2MU（无漂移控制）与无锚重训（上界，昂贵）。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | 重复单轮 R2MU | $D_n$ 单调上升，第 5 轮 Retain 掉 > 0.15 |
| T1 | SEQU（漂移触发回补） | $D_n\le\delta+0.02$ 有界，且每轮 Leak ≤ 0.05 |
| T2 | 消融：固定 $\lambda$（去掉 $\lambda_n$） | $D_n$ 回弹 → 证明漂移自适应必要 |
| T3 | 消融：去掉 CoT 步级 anchor，只留样本级 KL | 推理链保真度下降 → 证明 LRM 特有步级必要 |

**决定性实验**：T1 vs T2（隔离漂移自适应）。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 多轮后遗忘与效用同时守住；待 T1 |
| Faster | 6（机制分） | 相对「每次全量重训」快，但多轮顺序本身增加总开销 |
| Stronger | 8（机制分） | 对请求数量 / 顺序漂移鲁棒，含可验证上界；待 T2 |
| Cheaper | 7（机制分） | 触发式回补省算力；复用既有 CSV |
| Broader | 8（机制分） | 「漂移状态 + 触发校准」可迁移到任意持续编辑系统 |

**裁决：Accept with Revisions**。降档理由：F1=MAJOR（持续学习/EWC 近邻多，需严格划界「删除方向而非保留方向」）；且需先定义 R-TOFU 的请求流协议（当前基准是单轮），属**范围修订**。机制分 → *pending validation*。

---

## 7. 失败模式与局限（F10）

1. $D_n$ 依赖留住集 $R$ 的选取；若 $R$ 与后续 forget 请求重叠，漂移被低估 → 用固定、冻结的 anchor 集（借鉴 N12 冻结模板纪律）。
2. 触发阈值 $\delta$ 是超参，可能过调 → 预注册 $\delta$ 与回补上限次数，不按结果回调。
3. 与 `N10` 单轮状态位移相邻：SEQU 明确是**跨轮时间轴**，论文需划界。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/sequ/request_stream.py`（构造 5 批删除流 + 漂移统计）；训练在 `scripts/upbd/train_upbd.py` 骨架上加 $\lambda_n$ 与触发回补。冻结：fp32 / seeds 1001, 2002 / 固定 anchor 集。
