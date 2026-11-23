"""Render the same versioned report as Markdown."""

from html import escape


def _plain(value: object) -> str:
    return escape(str(value), quote=False).replace("[", "\\[").replace("]", "\\]")


def display_documents(report: dict) -> list[dict]:
    docs = {doc["doc_id"]: doc for doc in report.get("documents", ())}
    for check in report.get("external_checks", ()):
        if doc := check.get("document"):
            docs.setdefault(doc["doc_id"], doc)
    return list(docs.values())


def display_evidence(report: dict) -> list[dict]:
    evidence = {item["evidence_id"]: item for item in report.get("evidence", ())}
    for check in report.get("external_checks", ()):
        if check.get("document"):
            for item in check.get("evidence", ()):
                evidence.setdefault(item["evidence_id"], item)
    return list(evidence.values())


def render_markdown(report: dict) -> str:
    original = report["original_abstract"]
    actual = report["reviewed_abstract"]
    lines = ["# AbstractTrace review", "", f"Run `{report['run_id']}` · version `{actual['version_id']}` · "
             f"{report.get('run_status', 'report_ready')}", "", "## Original abstract", "", _plain(original["text"]),
             "", "## Reviewed abstract", "", _plain(actual["text"]), "",
             f"Whole-abstract verification: **{report['abstract_review']['status']}**", "", "## Findings", ""]
    if not report["findings"]:
        lines.append("The run has no claim verdicts.")
    for item in report["findings"]:
        label = item.get("verdict") or item["review_status"]
        ids = ", ".join(f"`{eid}`" for eid in item["evidence_ids"])
        lines.extend([f"- **{label}** · {_plain(item['text'])}",
                      f"  {_plain(item['reason'])}", f"  {ids}" if ids else ""])
    lines.extend(["", "## Evidence locations", ""])
    docs = {doc["doc_id"]: doc for doc in display_documents(report)}
    for item in display_evidence(report):
        anchor = item["locator"]
        source = docs.get(item.get("doc_id"), {}).get("name", "Synthetic fixture")
        page = anchor.get("page_number", anchor["page_index"] + 1)
        bbox = anchor.get("bbox")
        location = f"{_plain(source)} · page {page} · {_plain(anchor.get('section', 'Body'))}"
        if bbox is not None:
            location += " · box " + ", ".join(f"{value:.2f}" for value in bbox)
        if "row_index" in anchor:
            location += f" · table {anchor['table_index'] + 1}, row {anchor['row_index'] + 1}, column {anchor['column_index'] + 1}"
        lines.extend([f"- `{item['evidence_id']}` · {location}", f"  {_plain(item['text'])}"])
    for key, title, fields in (
        ("calculations", "Calculations", ("operation", "formula", "result", "unit", "evidence_ids")),
        ("revisions", "Revisions", ("original_text", "replacement", "status", "evidence_ids")),
        ("omissions", "Omissions", ("status", "reason", "evidence_ids")),
        ("external_checks", "External checks", ("citation_status", "access_status", "support_status", "reason")),
        ("unresolved_items", "Unresolved items", ("reason",)),
    ):
        if report.get(key):
            lines.extend(["", f"## {title}", ""])
            for item in report[key]:
                lines.append("- " + " · ".join(_plain(item[field]) for field in fields if field in item))
    if report.get("runtime_error"):
        lines.extend(["", "## Runtime status", "", _plain(report["runtime_error"])])
    return "\n".join(lines).rstrip() + "\n"
