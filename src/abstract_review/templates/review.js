function setupIntake() {
  const form = document.querySelector("[data-intake]");
  if (!form) return;

  const file = form.elements.pdf;
  const dropzone = file.closest(".dropzone");
  const name = form.querySelector("[data-file-name]");
  const source = form.elements.abstract_source;
  const pasted = form.elements.pasted_abstract;
  const pastedField = form.querySelector("[data-pasted-field]");
  const engine = form.elements.engine;
  const engineNote = form.querySelector("[data-engine-note]");
  const submit = form.querySelector(".start-review");
  const submitLabel = submit.innerHTML;

  function updateIntake() {
    name.textContent = file.files[0]?.name || "选择或拖入论文 PDF";
    dropzone.classList.toggle("has-file", file.files.length > 0);
    const usesPasted = source.value === "pasted";
    pastedField.hidden = !usesPasted;
    pasted.required = usesPasted;
    engineNote.hidden = engine.value !== "openai";
  }

  for (const input of [file, source, engine]) input.addEventListener("change", updateIntake);
  dropzone.addEventListener("dragover", () => dropzone.classList.add("is-dragging"));
  for (const type of ["dragleave", "drop"]) {
    dropzone.addEventListener(type, () => dropzone.classList.remove("is-dragging"));
  }

  form.addEventListener("submit", () => {
    submit.disabled = true;
    submit.textContent = "正在提交论文…";
  });
  window.addEventListener("pageshow", () => {
    submit.disabled = false;
    submit.innerHTML = submitLabel;
    updateIntake();
  });
  updateIntake();
}

function setupProgress() {
  const progress = document.getElementById("review-progress");
  if (!progress) return;

  const card = document.querySelector(".progress-card");
  const label = card?.querySelector("[data-run-label]");
  const reconnect = card?.querySelector("[data-reconnect]");
  const steps = card?.querySelectorAll("[data-step]");
  const reportState = document.querySelector("[data-current-run-status]");
  let phase = progress.dataset.phase;

  async function refresh() {
    try {
      const response = await fetch(progress.dataset.statusUrl, {cache: "no-store"});
      if (!response.ok) throw new Error("Status unavailable");
      const state = await response.json();
      if (!["queued", "running"].includes(state.run_status) ||
          (state.report_version_id || "") !== progress.dataset.version) {
        window.location.reload();
        return;
      }
      if (reconnect) reconnect.hidden = true;
      if (state.run_status !== phase) {
        phase = state.run_status;
        const text = phase === "queued" ? progress.dataset.queuedLabel : progress.dataset.runningLabel;
        if (reportState) {
          reportState.textContent = text;
          reportState.className = `pill pill-${phase}`;
        }
        if (card) {
          card.dataset.runPhase = phase;
          label.textContent = text;
          for (const step of steps) {
            if (step.dataset.step === phase) step.setAttribute("aria-current", "step");
            else step.removeAttribute("aria-current");
          }
        }
      }
    } catch {
      if (reconnect) reconnect.hidden = false;
    }
    window.setTimeout(refresh, 1200);
  }
  window.setTimeout(refresh, 1200);
}

function setupReport() {
  const pane = document.getElementById("pdf-preview");
  if (!pane) return;

  const endpoint = pane.dataset.previewEndpoint;
  const notice = pane.querySelector("[data-preview-announcement]");
  const failure = pane.querySelector("[data-preview-error]");
  const filters = document.querySelector("[data-filters]");
  const tally = document.querySelector("[data-finding-count]");
  const rows = [...document.querySelectorAll(".finding")];
  const isNarrow = window.matchMedia("(max-width: 970px)");
  const attention = rows.filter(row => !["supported", "refuted"].includes(row.dataset.verdict));
  const attentionVerdicts = new Set(attention.map(row => row.dataset.verdict));
  const attentionFilter = attentionVerdicts.size === 1 ? attention[0].dataset.verdict : "attention";
  let activeFilter = "all";
  let requested = null;
  let origin = null;
  let zoomed = false;

  function alignEvidence(content) {
    const pdf = content.querySelector(".pdf");
    const image = pdf?.querySelector("img");
    if (!image || pdf.dataset.focusY === undefined) return;

    const align = () => {
      if (!image.isConnected || !image.naturalWidth) return;
      pdf.scrollLeft = image.clientWidth * Number(pdf.dataset.focusX) - pdf.clientWidth / 2;
      pdf.scrollTop = image.clientHeight * Number(pdf.dataset.focusY) - pdf.clientHeight / 2;
    };
    if (image.complete) window.requestAnimationFrame(align);
    else image.addEventListener("load", align, {once: true});
  }

  function preparePreview(content) {
    const zoom = content.querySelector("[data-zoom]");
    const pdf = content.querySelector(".pdf");
    if (zoom) {
      zoom.hidden = false;
      pdf.classList.toggle("is-zoomed", zoomed);
      zoom.setAttribute("aria-pressed", String(zoomed));
      zoom.textContent = zoomed ? "适合宽度" : "放大";
    }
    alignEvidence(content);
  }

  function setCurrentEvidence(id) {
    for (const item of document.querySelectorAll("a[data-evidence-id]")) {
      if (id && item.dataset.evidenceId === id) item.setAttribute("aria-current", "true");
      else item.removeAttribute("aria-current");
    }
  }

  function clearPendingPreview() {
    if (requested) requested.abort();
    requested = null;
    pane.removeAttribute("aria-busy");
    failure.hidden = true;
    notice.textContent = "";
  }

  function focusPreview() {
    pane.scrollIntoView({block: "start"});
    pane.focus({preventScroll: true});
  }

  function matchesPreview(content, pageUrl) {
    const query = pageUrl.searchParams;
    const page = query.get("page");
    return (content.dataset.selectedEvidence || "") === (query.get("evidence_id") || "") &&
      content.dataset.requestedDoc === (query.get("doc_id") || "") &&
      content.dataset.requestedPage === (page === null ? "" : String(Number(page)));
  }

  async function updatePreview(pageUrl, {scroll = false, mode = "push", focusControl = ""} = {}) {
    if (!endpoint) return;
    const evidenceId = pageUrl.searchParams.get("evidence_id") || "";
    const previous = pane.querySelector("[data-preview-content]");
    clearPendingPreview();
    if (matchesPreview(previous, pageUrl)) {
      if (scroll && isNarrow.matches) focusPreview();
      return;
    }

    const url = new URL(endpoint, pageUrl);
    url.searchParams.set("version_id", pane.dataset.version);
    for (const param of ["evidence_id", "doc_id", "page"]) {
      if (pageUrl.searchParams.has(param)) url.searchParams.set(param, pageUrl.searchParams.get(param));
    }
    const request = new AbortController();
    requested = request;
    pane.setAttribute("aria-busy", "true");
    notice.textContent = "正在打开 PDF 页";

    try {
      const response = await fetch(url, {signal: request.signal});
      if (!response.ok) throw new Error("Preview unavailable");
      const markup = new DOMParser().parseFromString(await response.text(), "text/html");
      const content = markup.querySelector("[data-preview-content]");
      if (!content || !matchesPreview(content, pageUrl)) throw new Error("Preview mismatch");
      if (request.signal.aborted) return;

      const locationChanged = !matchesPreview(content, new URL(window.location.href));
      const restoreFocus = focusControl && previous.contains(document.activeElement);
      previous.replaceWith(content);
      preparePreview(content);
      setCurrentEvidence(evidenceId);
      if (mode === "replace") history.replaceState(null, "", pageUrl);
      else if (mode === "push" && locationChanged) history.pushState(null, "", pageUrl);
      notice.textContent = content.dataset.pageNumber ? `已打开第 ${content.dataset.pageNumber} 页` : "没有可预览的 PDF 页";
      if (restoreFocus) {
        const control = content.querySelector(`[data-pdf-nav="${focusControl}"]`) ||
          content.querySelector("#pdf-page-number");
        control?.focus({preventScroll: true});
      }
      if (scroll && isNarrow.matches) focusPreview();
    } catch {
      if (!request.signal.aborted) {
        failure.querySelector("a").href = pageUrl;
        failure.hidden = false;
        notice.textContent = "无法更新原页预览";
        if (scroll && isNarrow.matches) focusPreview();
      }
    } finally {
      if (request === requested) {
        pane.removeAttribute("aria-busy");
        requested = null;
      }
    }
  }

  function openRow(row) {
    for (const other of rows) other.open = other === row;
  }

  function showRowEvidence(row, mode = "push") {
    const first = row.querySelector(".claim-sources a[data-evidence-id]");
    if (first) {
      updatePreview(new URL(first.href), {mode});
    } else {
      const pageUrl = new URL(window.location.href);
      for (const param of ["evidence_id", "doc_id", "page"]) pageUrl.searchParams.delete(param);
      pageUrl.hash = "";
      updatePreview(pageUrl, {mode});
    }
  }

  function applyFilter(key, syncPreview) {
    activeFilter = key;
    const matching = rows.filter(row => key === "all" || row.dataset.verdict === key ||
      (key === "attention" && !["supported", "refuted"].includes(row.dataset.verdict)));
    for (const row of rows) row.hidden = !matching.includes(row);
    for (const button of filters.querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.filter === key));
    }
    tally.textContent = key === "all" ? `${rows.length} 条` : `${matching.length} / ${rows.length} 条`;
    if (syncPreview && matching.length && !matching.some(row => row.open)) {
      openRow(matching[0]);
      showRowEvidence(matching[0]);
    }
  }

  if (rows.length) {
    const groups = new Map([["all", {label: "全部", count: rows.length}]]);
    if (attentionVerdicts.size > 1) groups.set("attention", {label: "待确认", count: attention.length});
    for (const row of rows) {
      const key = row.dataset.verdict;
      const group = groups.get(key) || {label: row.querySelector(".pill").textContent, count: 0};
      group.count += 1;
      groups.set(key, group);
    }
    for (const [key, {label, count}] of groups) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "filter-chip";
      chip.dataset.filter = key;
      chip.setAttribute("aria-pressed", String(key === "all"));
      const number = document.createElement("span");
      number.textContent = count;
      chip.append(label, number);
      filters.append(chip);
    }
    filters.hidden = false;
    filters.addEventListener("click", event => {
      const chip = event.target.closest("button[data-filter]");
      if (chip) applyFilter(chip.dataset.filter, true);
    });

    const current = pane.querySelector("[data-preview-content]");
    const url = new URL(window.location.href);
    const finding = url.searchParams.get("finding");
    const requestedRow = finding && rows.find(row => row.id === `finding-${finding}`);
    const initial = requestedRow || (current.dataset.selectedEvidence ? rows.find(row => row.open) || rows[0]
      : rows.find(row => row.dataset.verdict === "refuted") || attention[0] || rows[0]);
    openRow(initial);
    if (!current.dataset.selectedEvidence && !current.dataset.requestedPage && !current.dataset.requestedDoc) {
      showRowEvidence(initial, "replace");
    }
  }

  preparePreview(pane.querySelector("[data-preview-content]"));
  document.addEventListener("click", event => {
    if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;

    const zoom = event.target.closest("[data-zoom]");
    if (zoom) {
      zoomed = !zoomed;
      preparePreview(pane.querySelector("[data-preview-content]"));
      return;
    }
    const overview = event.target.closest("[data-overview-filter]");
    if (overview && rows.length) {
      const key = overview.dataset.overviewFilter;
      applyFilter(key === "attention" ? attentionFilter : key, true);
    }
    const jump = event.target.closest('a[href="#abstract"]');
    if (jump) document.getElementById("abstract").open = true;

    const back = event.target.closest(".mobile-back");
    if (back && origin?.isConnected) {
      event.preventDefault();
      const row = origin.closest(".finding");
      if (row && !row.hidden) row.open = true;
      if (!row?.hidden) {
        origin.focus();
        return;
      }
      document.getElementById("findings").scrollIntoView();
      return;
    }

    const pageLink = event.target.closest("a[data-pdf-nav]");
    if (pageLink && endpoint) {
      event.preventDefault();
      updatePreview(new URL(pageLink.href), {focusControl: pageLink.dataset.pdfNav});
      return;
    }
    const link = event.target.closest("a[data-evidence-id]");
    if (link && endpoint) {
      event.preventDefault();
      if (!pane.contains(link)) origin = link;
      else if (isNarrow.matches) pane.focus({preventScroll: true});
      updatePreview(new URL(link.href), {scroll: !pane.contains(link)});
      return;
    }
    const toggle = event.target.closest(".finding-toggle");
    if (!toggle) return;
    const row = toggle.closest(".finding");
    if (!row.open) {
      event.preventDefault();
      openRow(row);
      showRowEvidence(row);
    }
  });

  const editor = document.querySelector("[data-abstract-edit-form]");
  if (editor) {
    editor.addEventListener("click", event => {
      const button = event.target.closest("[data-provisional-apply]");
      if (!button) return;
      const textarea = editor.elements.edited_text;
      const source = button.dataset.source.replace(/\r\n?/g, "\n");
      const replacement = button.dataset.replacement.replace(/\r\n?/g, "\n");
      const status = editor.querySelector(".provisional-apply-status");
      const start = textarea.value.indexOf(source);
      status.hidden = false;
      if (!source || start < 0 || start !== textarea.value.lastIndexOf(source)) {
        status.textContent = "编辑框里找不到唯一的原句。请核对已有编辑后手动应用这条改句。";
        return;
      }
      textarea.focus();
      textarea.setRangeText(replacement, start, start + source.length, "select");
      textarea.dispatchEvent(new Event("input", {bubbles: true}));
      status.textContent = "已填入暂定改句。检查后可点「保存并复核」。";
    });
  }

  pane.addEventListener("submit", event => {
    if (!endpoint || !event.target.matches("[data-page-jump]")) return;
    event.preventDefault();
    const pageUrl = new URL(event.target.action);
    pageUrl.search = new URLSearchParams(new FormData(event.target));
    updatePreview(pageUrl, {focusControl: "jump"});
  });

  document.addEventListener("submit", async event => {
    const form = event.target.closest("[data-provisional-form]");
    if (!form) return;
    event.preventDefault();
    const button = form.querySelector("button");
    if (button.disabled) return;
    const panel = form.closest(".provisional");
    const status = form.querySelector(".provisional-status");
    button.disabled = true;
    status.textContent = "正在根据原文生成暂定修订…";
    status.hidden = false;
    try {
      const response = await fetch(form.action, {
        method: "POST", body: new FormData(form), headers: {"X-AbstractTrace-Fragment": "provisional"},
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        status.textContent = typeof error.detail === "string" ? error.detail : "暂定修订未能生成，请稍后重试。";
        return;
      }
      const markup = new DOMParser().parseFromString(await response.text(), "text/html");
      const replacement = markup.getElementById(panel.id);
      if (!replacement) throw new Error("Provisional revision is missing");
      const restoreFocus = panel.contains(document.activeElement) ||
        (document.activeElement === document.body && panel.closest(".finding").open);
      panel.replaceWith(replacement);
      const options = markup.querySelector("[data-provisional-options]");
      if (options) editor?.querySelector("[data-provisional-options]")?.replaceWith(options);
      if (restoreFocus) replacement.querySelector("h3").focus({preventScroll: true});
    } catch {
      status.textContent = "暂定修订未能生成，请检查连接后重试。";
    } finally {
      button.disabled = false;
    }
  });

  window.addEventListener("popstate", () => {
    if (!endpoint) return;
    const pageUrl = new URL(window.location.href);
    const id = pageUrl.searchParams.get("evidence_id");
    if (id) {
      const match = rows.find(row => [...row.querySelectorAll("a[data-evidence-id]")]
        .some(link => link.dataset.evidenceId === id));
      if (match) {
        if (activeFilter !== "all" && match.hidden) applyFilter("all", false);
        openRow(match);
      }
    }
    updatePreview(pageUrl, {mode: "none"});
  });
}

setupIntake();
setupProgress();
setupReport();
