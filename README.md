# AbstractTrace
（东南大学大模型智能体与具身智能课程期末作业）A multi-agent system that checks abstract claims against the full paper, suggests evidence-backed revisions, and independently verifies the revised abstract.

## D01 离线演示

D01 使用合成证据和确定性的假模型。CLI 把审稿结果写入本地 SQLite，可在另一进程读取运行状态并导出 UTF-8 JSON。示例中的三个断言分别得到支持、反驳和证据不足。这个切片用于验证工程链路，PDF 解析和真实模型接入在后续任务中实现。

在仓库根目录用 Python 3.12 和 `uv` 运行下列 PowerShell 命令。

```powershell
uv sync --locked
uv run abstract-review doctor

$run = uv run abstract-review review start --fixture examples/d01/inputs.json --sample-id synthetic-001 | ConvertFrom-Json
uv run abstract-review review status $run.run_id
uv run abstract-review review export $run.run_id --version-id $run.report_version_id --format json --output .review-data/report.json

uv run abstract-review bench run --input-manifest examples/d01/inputs.json --output-dir .review-data/bench
uv run abstract-review bench score --predictions .review-data/bench/predictions.json --gold-manifest examples/d01/gold.json --output .review-data/bench/scores.json

uv run pytest -m "not live"
```

`examples/d01/inputs.json` 只包含摘要、待核断言和合成证据。`bench run` 与单篇审稿共用同一服务。只有 `bench score` 读取独立的 `gold.json`。预期标签正确率和完成率为 3/3，有证据金标的断言中，证据组召回率及联合准确率为 2/2。

运行数据库默认保存在 `.review-data/state.sqlite3`。可在子命令之前传 `--data-dir PATH`，或设置 `ABSTRACT_REVIEW_DATA_DIR`，以隔离不同的运行目录。
