# workspace 目录组织规范

## 按 idea-discovery 轮次日期归档

每完成一轮 `/idea-discovery` + `/idea-evaluator`，该轮的 proposals 与报告统一放入
`idea-stage-<MMDD>/`，`<MMDD>` 取该轮报告生成日期的月日（两位）。

| Stage | 日期 | 轮次 | 内容 |
|---|---|---|---|
| `idea-stage-1126/` | 2026-11-26 | 第一轮 | 10 个 `IDEA_XX_*`（CSDA…BOUND，含 drawio/png/proposal）+ `IDEA_REPORT.md` |
| `idea-stage-1127/` | 2026-11-27 | 第二轮 | 10 个 `IDEA_XX_*-1127`（proposal.md）+ `IDEA_REPORT-1127.md` + `IDEA_RANKING-1127.json` |

- 子目录名保留该轮的时间戳后缀（如 `-1127`）；批次区分由父文件夹 `idea-stage-<MMDD>/`
  承担，两批目录名在同一 workspace 下互不冲突。
- `IDEA_RANKING-1127.json` 的 `dir` 是相对 `stage_dir` 的纯目录名，解析为
  `workspace/<stage_dir>/<dir>`。
- 各报告 §0 的「输出根目录」是生成时记录的历史路径；归档后实际位置以本表为准。
- 新增一轮时：新建 `idea-stage-<MMDD>/`，把该轮 `IDEA_*` 目录、`IDEA_REPORT*.md`、
  `IDEA_RANKING*.json` 移入，并按上表补一行。

## 共用实现层（不随批次归档）

`implementations/`（10 个核心算法 + `hooks/` + `evals/` + `tests/`）留在 workspace 根目录，
**不放入任何 `idea-stage-*`**：它被隔壁
`R-TOFU/workspace/code/diagnostic_train_hooked.py` 以
`sys.path.insert(0, ".../redact/workspace")` + `from implementations.hooks.* import ...`
硬编码引用，移动会破坏训练脚本的 import。它对应第一轮 10 个方法；第二轮方法应作为
新模块接入同一个包，而不是另起目录。

## 归档范围

只归档 idea 工件（proposal / 报告 / 排名 / 机制图）。代码、数据资产、配置留在各自
既有位置。本次重组于 2026-11-27 完成，文件未提交 git（可用 git status 查看）。
