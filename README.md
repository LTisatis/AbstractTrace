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

## D02 从本地 PDF 回查证据

```powershell
uv run python examples/d02/make_sample.py .review-data/sample.pdf
$run = uv run abstract-review review start --pdf .review-data/sample.pdf | ConvertFrom-Json
uv run abstract-review review export $run.run_id --version-id $run.report_version_id --output .review-data/pdf-report.json
```

`documents` 记录原文件名、SHA-256 和相对于数据目录的 `blob_path`。`evidence[].doc_id` 对应这份归档 PDF。用 `locator.page_index` 打开零基物理页，按 `locator.bbox` 在未旋转的裁剪页上定位文字区域。边框表格还提供表、行、列索引。`--supplement` 可重复使用，指定本人已确认归属的 PDF 附件。默认选 PDF 内摘要，有另贴摘要时可用 `--abstract-file PATH --abstract-source pasted` 切换。

不指定 `--engine` 时，PDF 入口只解析原始文档和锚点。PDF 解析依赖 PyMuPDF，分发时请参阅其许可说明。 [PyMuPDF 许可](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)

## 审稿、修订与核验

下面的离线样例以同一 PDF 为证据，把初稿里的 `10` 复算并修订为 `8` 个百分点。离线判定适用于可核算的同表比较与能回查的原句。整篇一致性由模型或人工复核。

```powershell
Set-Content .review-data/alternate.txt 'On dataset A, accuracy increased by 10 percentage points.' -Encoding utf8
$run = uv run abstract-review review start --pdf .review-data/sample.pdf --abstract-file .review-data/alternate.txt --abstract-source pasted --engine offline | ConvertFrom-Json
uv run abstract-review review export $run.run_id --format md --output .review-data/review.md
uv run abstract-review serve
```

网页从 `http://127.0.0.1:8000` 打开，可提交 PDF、查看原页和高亮、下载报告、选择改句或提交手改版本。命令行支持 `review submit-selection`。每次提交携带当前版本 ID，旧报告仍可按旧 ID 导出。

`--engine openai` 显式使用 `OPENAI_API_KEY` 和 Responses API。先运行 `uv run abstract-review doctor --online` 验证权限。模型默认 `gpt-6.2-sol`，可用 `--model` 指定。运行暂挂后用 `review resume RUN_ID` 从检查点继续。扩额时使用新 `--request-id` 和 `--add-requests` 或 `--add-seconds`。同一请求 ID 重试沿用原值。

原页没有可读文本时，可用 `--ocr-language eng` 启用本机 Tesseract。需要语言数据时，按安装方式配置 `TESSDATA_PREFIX`。检索默认使用 SQLite FTS5、精确词元和图扩展。`uv sync --locked --extra embeddings` 后，`--embed-model PATH` 读取本地模型目录。跨页关联仅用于有明确续表标记且表头对齐的表格。

外部网络入口使用 `review start --source DOI_OR_ARXIV`。DOI 开放全文查询使用 `UNPAYWALL_EMAIL`。运行审稿引擎时，用 `--check-external` 启用内容核查。含首创性措辞时查询少量书目候选，OpenAlex 支持通过 `OPENALEX_API_KEY` 配置。用 `--check-omissions` 查找有正文依据的潜在遗漏。默认审稿只读取指定材料。

## 独立评分

```powershell
uv run python examples/d11/make_benchmark.py .review-data/d11
uv run abstract-review bench run --engine offline --input-manifest .review-data/d11/inputs.json --output-dir .review-data/d11/out --resume
uv run abstract-review bench score --predictions .review-data/d11/out/predictions.json --gold-manifest .review-data/d11/gold.json --output .review-data/d11/out/scores.json
```

PDF 输入清单列出 `sample_id`、相对于清单的 `pdf`，以及可选 `abstract_file`、`paper_family_id`、`split` 和给定 `claims`。给定断言每项含 `claim_id` 与 `text`，省略时由引擎从摘要抽取。金标从独立清单载入评分器。样例为合成的多个论文族，完整跨学科质量评测需要独立标注的论文留出集。
