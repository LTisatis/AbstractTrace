# AbstractTrace

AbstractTrace 是一个论文摘要证据审阅 Agent。导入论文 PDF 后，它会逐条核对摘要里的断言，把判定链接到正文、图或表格单元格。系统可以提出带证据的改句，并复核用户保存后的摘要版本。

东南大学「大模型智能体与具身智能」课程期末项目。

## 快速开始

```sh
git clone https://github.com/LTisatis/AbstractTrace.git
cd AbstractTrace
```

**Windows（x64 或 ARM64）**。双击仓库里的 [start-web.cmd](start-web.cmd)。首次启动请联网。启动器会按需安装校验过的 `uv`、Python 3.12 和锁定的项目依赖，并用默认浏览器打开首页。[uv 发行包](https://github.com/astral-sh/uv/releases/tag/0.9.18)

**macOS 或 Linux**。先按 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 `uv`，再用下面的命令启动网页。

```sh
uv run --locked --no-dev python -m abstract_review serve --open --port 0
```

服务在本机 `127.0.0.1` 自动选择一个可用端口。如果浏览器没有自动打开，访问终端里显示的地址。使用期间保留终端窗口，按 `Ctrl+C` 停止服务。论文和报告默认保存在仓库的 `.review-data/`。

### 选择审阅方式

- **本地规则**适合先体验 PDF 导入、证据定位和数值复算，无需密钥。
- **OpenAI 模型**用于模型判定、定向补查和暂定修订。启动服务前，在同一个终端设置 `OPENAI_API_KEY`。选用此方式时，审阅所需的摘要和证据会发送给模型 API。

在 Windows PowerShell 中，可用以下命令输入密钥并启动服务。

```powershell
$key = Read-Host 'OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $key).Password
.\start-web.cmd
```

### 在网页上完成一次审阅

1. **导入论文。** 选择有可选中文本的 PDF。选「从 PDF 提取」审阅论文原摘要，或选「使用粘贴内容」审阅自己的版本。选择审阅方式后，点「开始审阅」。
2. **先看概览。** 报告就绪后，点「反驳」或「待确认」的数量查看相关断言。「待确认」汇总待核验、证据不足和证据冲突。
3. **回查原文。** 在「断言与证据」展开一条断言，点击「判定引用」下方的页码链接，右侧会打开并高亮 PDF 原页。用「上一页」「下一页」或输入页码浏览全文。点「返回证据页」恢复引用位置和高亮。
4. **处理改句。** 在「修订与复核」比较原句和建议，点「采用到编辑稿」把通过核验的改句填入当前草稿。使用 OpenAI 模型时，有可引用原文的「待核验」断言还能生成**暂定修订**。点「去编辑器应用」，再点「一键应用」填入暂定改句。检查草稿后点「保存并复核」。暂定修订保留原断言的待核验状态。
5. **查看版本并导出。** 复核完成后，在「版本与导出」查看当前摘要和历史版本，下载当前页面所显示版本的 Markdown 或 JSON 报告。

### 体验一次自动改句

首次启动后，在仓库目录生成样例 PDF。Windows PowerShell 可运行下方命令。macOS 或 Linux 可把 Python 路径替换成 `uv run --locked --no-dev python`。

```powershell
.\.venv\Scripts\python.exe examples/d02/make_sample.py .review-data/sample.pdf
```

把 `.review-data/sample.pdf` 上传到网页，选「使用粘贴内容」和「本地规则」，输入下方测试句后开始审阅。报告会用数据集 A 的 **80%** 和 **88%** 复算，建议把 **10 个百分点** 改为 **8 个百分点**。

```text
On dataset A, accuracy increased by 10 percentage points.
```

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

能从原页确认关系时，数据单元格会保留行表头、多层列表头和脚注的证据 ID。Web 报告的「证据关系图」按断言显示实际走过的路径。节点可打开并高亮同一报告版本的 PDF 原页。「取证路径」可逐步查看每条连线的原文依据。

不指定 `--engine` 时，PDF 入口只解析原始文档和锚点。PDF 解析依赖 PyMuPDF，分发时请参阅其许可说明。 [PyMuPDF 许可](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)

## 审稿、修订与核验

下面的离线样例以同一 PDF 为证据，把初稿里的 `10` 复算并修订为 `8` 个百分点。离线判定适用于可核算的同表比较与能回查的原句。整篇一致性由模型或人工复核。

对能明确拆分的复合断言，审稿器保留各分项的原文位置、共享条件和例外。它分别取证和判定，再按原句的逻辑关系合并结论。每次改稿核验针对该版本重新计算可核实的数字，并在报告中保留复算值和原文证据。结果有歧义时，报告会标记需要确认的数值。

```powershell
Set-Content .review-data/alternate.txt 'On dataset A, accuracy increased by 10 percentage points.' -Encoding utf8
$run = uv run abstract-review review start --pdf .review-data/sample.pdf --abstract-file .review-data/alternate.txt --abstract-source pasted --engine offline | ConvertFrom-Json
uv run abstract-review review export $run.run_id --format md --output .review-data/review.md
uv run abstract-review serve
```

网页从 `http://127.0.0.1:8000` 打开。「最近审阅」可找回本地任务。报告页可切换论文与补充材料、查看证据高亮、下载报告，也可把建议改句填进当前编辑稿，再统一保存并复核。Web 审稿和改后核验进入 SQLite 队列，页面会显示阶段及完成提示。服务重启后会接手续跑任务和租约已过期的任务。命令行支持 `review submit-selection`。每次提交携带当前版本 ID，旧报告仍可按旧 ID 导出。

`--engine openai` 显式使用 `OPENAI_API_KEY` 和 Responses API。`uv run abstract-review doctor --online` 用自拟文本和合成裁图检查结构化响应，并记录模型、耗时和用量。也可以用 `uv run pytest -m live --run-live` 进行显式联调。模型默认 `gpt-6.2-sol`，可用 `--model` 指定。运行暂挂后用 `review resume RUN_ID` 从检查点继续。扩额时使用新 `--request-id` 和 `--add-requests` 或 `--add-seconds`。同一请求 ID 重试沿用原值。

原页没有可读文本时，可用 `--ocr-language eng` 启用本机 Tesseract。需要语言数据时，按安装方式配置 `TESSDATA_PREFIX`。检索默认使用 SQLite FTS5、精确词元和图扩展。报告保留正文每行的原页锚点。跨页表格只有在有明确续表标记且表头对齐时才会关联。

`uv sync --locked --extra embeddings` 后，可在 `review start` 和 PDF `bench run` 使用 `--embed-model PATH` 启用本地混合召回，或用 `--rerank-model PATH` 对候选段落进行本地 CrossEncoder 重排。模型需预先保存在本机，权重使用 safetensors。批跑记录模型文件哈希，恢复时仅复用与模型内容相符的检查点。

外部网络入口使用 `review start --source DOI_OR_ARXIV`。DOI 开放全文查询使用 `UNPAYWALL_EMAIL`。arXiv 官方域名可通过 `HTTPS_PROXY` 中的免认证 HTTP 代理访问，并遵循 `NO_PROXY`。运行审稿引擎时，用 `--check-external` 启用内容核查。含首创性措辞时查询少量书目候选，OpenAlex 支持通过 `OPENALEX_API_KEY` 配置。用 `--check-omissions` 查找有正文依据的潜在遗漏。默认审稿只读取指定材料。

## 独立评分

```powershell
uv run python examples/d11/make_benchmark.py .review-data/d11
uv run abstract-review bench run --engine offline --input-manifest .review-data/d11/inputs.json --output-dir .review-data/d11/out --resume
uv run abstract-review bench score --predictions .review-data/d11/out/predictions.json --gold-manifest .review-data/d11/gold.json --output .review-data/d11/out/scores.json
```

PDF 输入清单列出 `sample_id`、相对于清单的 `pdf`，以及可选 `pdf_sha256`、`abstract_file`、`paper_family_id`、`split` 和给定 `claims`。给定断言每项含 `claim_id` 与 `text`，省略时由引擎从摘要抽取。批跑开始前会计算每个 PDF 的实际 SHA-256 并验证清单中提供的校验值。即使文件名和论文族不同，内容完全相同的 PDF 也不能跨划分。使用相同输入、配置、数据目录和输出目录加上 `--resume`，可复用已完成项，并沿用原有额度恢复已保存的暂停项。

金标从独立清单载入评分器。`run.json` 记录输入和配置指纹、用量与耗时。通用评分和 SciFact 评分可加上 `--run-record`，传入同一轮生成的 `run.json`。评分器按预测文件内容的 SHA-256 校验两者，`run_diagnostics` 会列出未执行、暂停、已执行未决和已判定等结果。评分区分已完成项的分类成绩和所有金标的交付比例，并按实际送审顺序计算完整证据组的 `Recall@K`。样例为合成的多个论文族，跨学科质量评测需要独立标注的全文留出集。

自主抽取的严格评分按同句区间和规范化命题匹配。独立核对等义命题后，可以向 `bench score` 增加 `--semantic-matches equivalents.json`。原严格结果保留在顶层，`semantic_extraction` 记录补充匹配，`semantic_scoring` 记录相应的判定、覆盖和取证指标。

等义清单使用 `schema_version: 1`、`annotation_id`、`annotator_type`（`human` 或 `ai`）、`predictions_sha256`、`gold_sha256` 及 `matches`。每个匹配填写 `sample_id`、`predicted_claim_id` 和 `gold_claim_id`。哈希可用 `abstract_review.scoring.semantic_manifest_sha256` 计算。核对时应保留事实、极性、条件和数量。评分器要求同句区间重叠、一对一对应，并优先采用已有的严格匹配。推理过程只读取输入清单。

旧报告有时只给出整句反驳。若独立审阅确认报告中的反证指向一个合取子命题，可以通过 `--verdict-projections projections.json` 补充诊断。清单沿用等义清单的版本、审阅者和哈希字段，将 `matches` 换成 `projections`。每条记录填写上述三个 ID、`relation`（`conjunct`）、金标子命题的完整 `operand_anchor`、与报告引证相符的 `evidence_ids`，以及说明反证为何指向该子命题的 `review_note`。证据须实际送审、已被引用并覆盖一组完整的金标证据。通过的记录写入 `verdict_projection`，严格评分和等义评分沿用各自口径。

SciFact 适配器读取作者发布的 JSONL，将每条断言与它的给定引用摘要配对，并保留原始文档和句子 ID。这组指标的适用范围是给定引用摘要。 [SciFact 数据和评估协议](https://github.com/allenai/scifact#dataset)

判定证据不足时，系统要求完整摘要和对原要求的显式检查。模型可另外解释缺少哪个结果或条件，这些说明保留在报告里。

反证通过范围和引文校验后，JSON 的 `evidence_ids` 先列直接反驳断言的结果句，再列用来确认范围的句子，其余引文放在后面。

在线裁决逐条引用摘要中的句子，系统校验编号后恢复为原始证据 ID。支持和反驳仍需满足证据与研究范围校验。摘要只送审了一部分时，未找到证据的断言保留为待查。

SciFact 在线批跑默认最多预留 8 次请求和 180 秒，可用 `--max-requests` 与 `--max-seconds` 调整。预算内的预测逐样本保存到输出目录。使用相同输入、模型和输出目录并加上 `--resume` 可以续跑，两项限额均按累计用量计算。`run.json` 记录暂停原因和样本用量，`batch.last_paused_sample` 保留最近一次已捕获暂停的详细诊断。

可用 `--scifact-recheck-model gpt-6.2-astra` 为完整摘要启用一次额外复核。首轮输出证据不足或待核验时才发起复核，两次判定共用原批次的请求和时间额度。只有复核给出有原句引文且通过现有校验的支持或反驳，才替换首轮结果。

```powershell
uv run abstract-review bench prepare-scifact --corpus data/corpus.jsonl --claims data/claims_dev.jsonl --split dev --dataset-version author-release-2020 --output-dir .review-data/scifact
uv run abstract-review bench run --dataset scifact --engine offline --input-manifest .review-data/scifact/inputs.json --output-dir .review-data/scifact/out
uv run abstract-review bench score --dataset scifact --predictions .review-data/scifact/out/predictions.json --gold-manifest .review-data/scifact/gold.json --output .review-data/scifact/out/scores.json
```

CLAIM-BENCH 适配器需要本地标注文件及其对应的 PDF。它把能在单一原页上唯一匹配的正文或表格片段绑定到证据 ID，并单独记录对齐覆盖率。评分衡量原文片段定位。原有的严格 `quoted_passage_recall_at_k` 继续要求前 K 条覆盖全部金标锚点。`diagnostics` 另外报告锚点召回、前 K 条可容纳片段的完整召回，以及完整证据包的召回。 [CLAIM-BENCH 作者数据](https://github.com/shashidharjavaji/CLAIM_BENCH)

审稿器保留八条主候选，并可附上按同页同栏组装的正文。主候选所在的原文范围未被已选上下文覆盖时，系统会补入该范围。启用本地重排时，全文查询命中同一正文范围，可以再选一个不同范围。每个证据包最多附加 24 行、1,800 个原文字符，原始证据 ID 和页坐标可用于回链。

裁决器可以选择要补查的条件和来源。第一次补查优先检查指定页码、章节或表格，并保留此前已核对的关键引证。仍有必要条件待查时，系统只在另一处指定来源找到了尚未送审的证据后发起第二轮补查。两轮沿用同一断言的预算，补查目标和证据包在暂停后可恢复。遇到同节正文在页尾中断且有续接依据时，系统会在现有证据预算内读取相邻页开头，并保留两页各自的引用锚点。

实证主张的支持裁决分别记录观察结果和实际测试范围，二者须来自同一研究并有送审引证。只有总结性表述时，系统可在原有补查轮次内检索原始结果。模型也可以引用一个实际送审正文段的首尾锚点，系统校验连续范围后展开原文引用。裁决请求中的简短编号在写报告前恢复成原始证据 ID。

```powershell
uv run abstract-review bench prepare-claimbench --pdf paper.pdf --annotations annotations.json --paper-family-id paper --split dev --dataset-version local-release --output-dir .review-data/claimbench
uv run abstract-review bench run --engine offline --input-manifest .review-data/claimbench/inputs.json --output-dir .review-data/claimbench/out
uv run abstract-review bench score --dataset claimbench --predictions .review-data/claimbench/out/predictions.json --gold-manifest .review-data/claimbench/gold.json --output .review-data/claimbench/out/scores.json
```

SciTabAlign 适配器读取作者发布的表格和单元格标注。离线默认排列最多 8 格。启用 `--engine openai` 后，表格总共不超过 8 格时，上限为 8 格，其余表格上限为 24 格。模型读取完整表格和表题，选出相关数值及解释这些数值所需的行名、表头和对照单元格。所有坐标都会与原表核对。如果模型选了超过 8 格，系统会结合断言、行名和表头重排这些坐标，让关键数值及其上下文靠前。用 `--table-cell-limit 48` 可以为跨多行、多指标的断言留出更多证据位置，也可固定为 8、16 或 24 格。每条断言使用一次模型调用，`run.json` 记录所用上限。两种模式报告前 1、3、5、8、16、24、48 格的选证指标和完整组命中，并单列空标注和越界金标。 [SciTabAlign 作者数据](https://github.com/Alab-NII/SciTabAlign)

```powershell
uv run abstract-review bench prepare-scitabalign --annotations data/data.json --dataset-version author-release --output-dir .review-data/scitabalign
uv run abstract-review bench run --dataset scitabalign --input-manifest .review-data/scitabalign/inputs.json --output-dir .review-data/scitabalign/out
uv run abstract-review bench score --dataset scitabalign --predictions .review-data/scitabalign/out/predictions.json --gold-manifest .review-data/scitabalign/gold.json --output .review-data/scitabalign/out/scores.json
```

在线模式按批次累计请求和时间，并在 `run.json` 中记录进度和用量。用同一输入、模型及输出目录加上 `--resume` 可继续已有批次。与离线结果做配对比较时，两组使用同一输入清单和金标，分别写入输出目录。

```powershell
uv run abstract-review bench run --dataset scitabalign --engine openai --input-manifest .review-data/scitabalign/inputs.json --output-dir .review-data/scitabalign/online --max-requests 8 --max-seconds 180
uv run abstract-review bench score --dataset scitabalign --predictions .review-data/scitabalign/online/predictions.json --gold-manifest .review-data/scitabalign/gold.json --output .review-data/scitabalign/online/scores.json
```

PDF 批跑的 `run.json` 保留每篇的版本、修订、复算和调用用量。恢复批跑时复用同配置下已保存的记录。人工复核后，可以把标注单独交给评分器。

```powershell
uv run abstract-review bench score-reviews --run-record out/run.json --gold-manifest review-gold.json --output out/review-scores.json
```

`review-gold.json` 使用 `schema_version: 1`。每条人工记录包含 `sample_id` 和 `version_id`。按评测范围提供以下数组。

- `revision_checks` 使用 `revision_id` 定位修改，以布尔值记录 `corrected`、`facts_preserved` 和 `unsupported_added`。人工尚无法判定的项填 `null`。
- `abstract_checks` 使用布尔值 `has_conflict` 记录该版本是否存在整篇冲突。
- `table_checks` 使用 `claim_id`、`operation` 定位复算，并记录有序的 `operand_ids`、`header_ids`、`note_ids`、十进制字符串 `result` 和 `unit`。

如需估算成本，可加 `--pricing prices.json`。价格文件包含 `schema_version: 1`、`pricing_id` 和按实际返回模型名索引的 `models`。每个模型用十进制字符串填写每百万词元的美元费率 `input_per_million`、`cached_input_per_million` 和 `output_per_million`。模型有缓存写入计费时，可填写 `cache_write_30m_per_million` 和 `cache_write_12h_per_million`。费用结果单列缺少用量或费率的请求。
