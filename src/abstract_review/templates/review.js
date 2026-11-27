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
  const supplements = form.elements.supplements;
  const attachments = form.querySelector("[data-attachments]");
  const error = form.querySelector("[data-file-error]");
  const maxBytes = Number(form.dataset.maxPdfBytes);

  function updateIntake() {
    name.textContent = file.files[0]?.name || "选择或拖入论文 PDF";
    dropzone.classList.toggle("has-file", file.files.length > 0);
    const usesPasted = source.value === "pasted";
    pastedField.hidden = !usesPasted;
    pasted.required = usesPasted;
    pasted.disabled = !usesPasted;
    engineNote.hidden = engine.value !== "openai";
  }

  function updateFiles() {
    const invalid = [...file.files, ...supplements.files].find(item =>
      !item.name.toLowerCase().endsWith(".pdf") || item.size > maxBytes);
    error.hidden = !invalid;
    if (invalid) error.textContent = invalid.name.toLowerCase().endsWith(".pdf")
      ? invalid.name + " 超过 " + maxBytes / (1024 * 1024) + " MiB。"
      : invalid.name + " 需要使用 .pdf 扩展名。";
    attachments.replaceChildren();
    attachments.hidden = supplements.files.length === 0;
    for (const [index, attachment] of [...supplements.files].entries()) {
      const row = document.createElement("li");
      row.textContent = attachment.name + " · " + (attachment.size / (1024 * 1024)).toFixed(2) + " MiB ";
      if (typeof DataTransfer === "function") {
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "button button-small";
        remove.textContent = "移除";
        remove.setAttribute("aria-label", "移除 " + attachment.name);
        remove.addEventListener("click", () => {
          const remaining = new DataTransfer();
          [...supplements.files].forEach((item, position) => {
            if (position !== index) remaining.items.add(item);
          });
          supplements.files = remaining.files;
          updateFiles();
          (attachments.querySelector("button") || supplements).focus();
        });
        row.append(remove);
      }
      attachments.append(row);
    }
    updateIntake();
    return !invalid;
  }

  for (const input of [source, engine]) input.addEventListener("change", updateIntake);
  for (const input of [file, supplements]) input.addEventListener("change", updateFiles);
  dropzone.addEventListener("dragover", () => dropzone.classList.add("is-dragging"));
  for (const type of ["dragleave", "drop"]) {
    dropzone.addEventListener(type, () => dropzone.classList.remove("is-dragging"));
  }

  form.addEventListener("submit", event => {
    if (submit.disabled || !updateFiles()) {
      event.preventDefault();
      return;
    }
    submit.disabled = true;
    submit.textContent = "正在提交论文…";
  });
  window.addEventListener("pageshow", () => {
    submit.disabled = false;
    submit.innerHTML = submitLabel;
    updateFiles();
  });
  updateFiles();
}

function setupProgress() {
  const progress = document.getElementById("review-progress");
  if (!progress) return;

  const card = document.querySelector(".progress-card");
  const label = card?.querySelector("[data-run-label]");
  const steps = card?.querySelectorAll("[data-step]");
  const reportState = document.querySelector("[data-current-run-status]");
  const stageText = document.querySelector("[data-stage-text]");
  const update = document.getElementById("progress-update");
  const message = update.querySelector("[data-progress-message]");
  const link = update.querySelector("[data-progress-link]");
  const stageLabels = {
    parse: "正在解析 PDF 并提取原文证据。",
    review: "正在提取或核对断言。",
    checks: "正在核查摘要遗漏或外部参考文献。",
    revise: "正在起草并核验建议改句。",
    verify: "正在复核当前摘要。",
  };
  const labels = {
    queued: progress.dataset.queuedLabel, running: progress.dataset.runningLabel,
    report_ready: "报告已就绪", paused: "已暂停", failed: "未完成",
  };
  let retryDelay = 1200;

  function notify(text, action = "") {
    if (message.textContent !== text) message.textContent = text;
    if (link.textContent !== action) link.textContent = action;
    link.hidden = !action;
    update.hidden = !text;
  }

  function updatePhase(phase) {
    if (reportState && reportState.textContent !== labels[phase]) {
      reportState.textContent = labels[phase];
      reportState.className = "pill pill-" + phase;
    }
    if (card && card.dataset.runPhase !== phase) {
      card.dataset.runPhase = phase;
      label.textContent = labels[phase];
      for (const step of steps) {
        if (step.dataset.step === phase) step.setAttribute("aria-current", "step");
        else step.removeAttribute("aria-current");
      }
    }
  }

  async function refresh() {
    try {
      const response = await fetch(progress.dataset.statusUrl, {cache: "no-store"});
      if (response.status === 404) {
        notify("这项审阅任务已找不到。");
        return;
      }
      if (!response.ok) throw new Error("Status unavailable");
      const state = await response.json();
      const phase = state.run_status;
      updatePhase(phase);
      if (!["queued", "running"].includes(phase)) {
        notify({report_ready: "审阅已完成。", paused: "审阅已暂停。", failed: "本次审阅未完成。"}[phase],
          phase === "report_ready" ? "查看最新报告" : "查看原因和后续操作");
        return;
      }
      retryDelay = 1200;
      let text = phase === "queued" ? "正在等待调度。" : stageLabels[state.stage] || "正在审阅。";
      if (state.stage === "review" && state.claims_total > 0) {
        text += " 已核对 " + state.claims_done + " / " + state.claims_total + " 条。";
      }
      if (stageText.textContent !== text) stageText.textContent = text;
      if (!progress.dataset.version && state.report_version_id) {
        notify("PDF 已解析。", "查看已解析的 PDF");
      } else notify("");
    } catch {
      notify("连接暂时中断，正在重新检查任务状态。");
      retryDelay = Math.min(retryDelay * 2, 10000);
    }
    window.setTimeout(refresh, retryDelay);
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
  const atomicRows = rows.filter(row => !row.hasAttribute("data-compound"));
  const isNarrow = window.matchMedia("(max-width: 970px)");
  const attention = atomicRows.filter(row => !["supported", "refuted"].includes(row.dataset.verdict));
  const attentionVerdicts = new Set(attention.map(row => row.dataset.verdict));
  const attentionFilter = attentionVerdicts.size === 1 ? attention[0].dataset.verdict : "attention";
  let activeFilter = "all";
  let requested = null;
  let origin = null;
  let zoomPercent = "100";
  let previewSearch = window.location.search;
  let evidenceRequest = null;

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

  function setZoom(content) {
    const pdf = content.querySelector(".pdf");
    pdf.style.setProperty("--pdf-zoom", zoomPercent + "%");
    const image = pdf.querySelector("img");
    if (Number(zoomPercent) >= 150) {
      const url = new URL(image.src);
      if (!url.searchParams.has("detail")) {
        url.searchParams.set("detail", "true");
        image.src = url.href;
      }
    }
  }

  function preparePreview(content) {
    const zoom = content.querySelector("[data-zoom]");
    if (zoom) {
      zoom.hidden = false;
      zoom.querySelector("select").value = zoomPercent;
      setZoom(content);
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
      previewSearch = window.location.search;
      notice.textContent = content.dataset.pageNumber ? `已打开第 ${content.dataset.pageNumber} 页` : "没有可预览的 PDF 页";
      if (restoreFocus) {
        const control = focusControl === "document" ? content.querySelector("#preview-document") :
          content.querySelector(`[data-pdf-nav="${focusControl}"]`) || content.querySelector("#pdf-page-number");
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
    row.open = true;
  }

  function showRowEvidence(row, mode = "push") {
    const first = row.querySelector(".claim-sources a[data-evidence-id]") ||
      row.querySelector(".follow-up a[data-evidence-id]") ||
      row.querySelector(".evidence-graph a[data-evidence-id]");
    if (first) {
      updatePreview(new URL(first.href), {mode});
    }
  }

  function applyFilter(key, syncPreview) {
    activeFilter = key;
    const matching = atomicRows.filter(row => key === "all" || row.dataset.verdict === key ||
      (key === "attention" && !["supported", "refuted"].includes(row.dataset.verdict)));
    const shown = new Set(matching);
    for (const row of rows) row.hidden = key !== "all" && !shown.has(row);
    for (const button of filters.querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.filter === key));
    }
    tally.textContent = key === "all" ? atomicRows.length + " 条" : matching.length + " / " + atomicRows.length + " 条";
    if (syncPreview && matching.length && !matching.some(row => row.open)) {
      openRow(matching[0]);
      showRowEvidence(matching[0]);
    }
  }

  if (rows.length) {
    const groups = new Map([["all", {label: "全部", count: atomicRows.length}]]);
    if (attentionVerdicts.size > 1) groups.set("attention", {label: "待确认", count: attention.length});
    for (const row of atomicRows) {
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

  async function loadEvidenceList(controls, query) {
    evidenceRequest?.abort();
    const pending = new AbortController();
    evidenceRequest = pending;
    const url = new URL(controls.dataset.endpoint);
    url.search = query;
    controls.setAttribute("aria-busy", "true");
    try {
      const response = await fetch(url, {signal: pending.signal});
      if (!response.ok) throw new Error("Evidence list unavailable");
      const markup = new DOMParser().parseFromString(await response.text(), "text/html");
      const replacement = markup.querySelector("[data-evidence-list]");
      if (!replacement) throw new Error("Evidence list missing");
      if (pending.signal.aborted) return;
      controls.replaceWith(replacement);
      setCurrentEvidence(pane.querySelector("[data-preview-content]").dataset.selectedEvidence);
      replacement.querySelector("[data-evidence-result]").focus({preventScroll: true});
    } catch {
      if (!pending.signal.aborted) controls.querySelector("[data-evidence-result]").textContent = "证据列表加载失败。请重试。";
    } finally {
      if (evidenceRequest === pending) {
        controls.removeAttribute("aria-busy");
        evidenceRequest = null;
      }
    }
  }

  preparePreview(pane.querySelector("[data-preview-content]"));
  document.addEventListener("click", event => {
    if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;

    const pagination = event.target.closest("a[data-evidence-page]");
    if (pagination) {
      event.preventDefault();
      loadEvidenceList(pagination.closest("[data-evidence-list]"), new URL(pagination.href).search);
      return;
    }
    const findingLink = event.target.closest("a[data-finding-link]");
    const targetRow = findingLink && document.getElementById(findingLink.hash.slice(1));
    if (targetRow) {
      if (targetRow.hidden) applyFilter("all", false);
      openRow(targetRow);
    }
    if (event.target.closest("[data-abstract-sentence-link]")) {
      document.querySelector("[data-abstract-sentences]").open = true;
    }
    const overview = event.target.closest("[data-overview-filter]");
    if (overview && rows.length) {
      const key = overview.dataset.overviewFilter;
      applyFilter(key === "attention" ? attentionFilter : key, true);
    }
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
    const textarea = editor.elements.edited_text;
    const status = editor.querySelector("[data-editor-status]");
    const edits = new Map();
    const normalize = value => value.replace(/\r\n?/g, "\n");
    const warnUnsaved = event => {
      if (editor.dataset.submitting) return;
      event.preventDefault();
      event.returnValue = true;
    };
    textarea.addEventListener("input", () => {
      window.removeEventListener("beforeunload", warnUnsaved);
      if (textarea.value !== textarea.defaultValue) window.addEventListener("beforeunload", warnUnsaved);
    });

    for (const button of document.querySelectorAll("[data-editor-insert]")) button.hidden = false;
    document.addEventListener("click", event => {
      const button = event.target.closest("[data-provisional-apply], [data-editor-insert], [data-open-editor]");
      if (!button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      editor.closest("[data-editor-details]").open = true;
      textarea.focus();
      if (button.hasAttribute("data-open-editor")) return;

      const source = normalize(button.dataset.source);
      const known = edits.get(source);
      const previous = known && textarea.value.includes(known.text) ? known : {text: source, ranges: []};
      const current = previous?.text || "";
      const start = textarea.value.indexOf(current);
      status.hidden = false;
      status.textContent = "这条改句与当前草稿有冲突。请对照建议手动合并。";
      if (!source || !current || start < 0 || start !== textarea.value.lastIndexOf(current)) {
        const replacement = normalize(button.dataset.replacement);
        if (replacement && textarea.value.includes(replacement)) status.textContent = "编辑稿中已包含这条改句。";
        return;
      }

      const inlineSource = button.dataset.inlineSource;
      const first = inlineSource ? Array.from(source).slice(0, Number(button.dataset.inlineStart)).join("").length : 0;
      const last = inlineSource ? first + normalize(inlineSource).length : source.length;
      if (previous.ranges.some(edit => first < edit.last && edit.first < last)) return;
      const offset = previous.ranges.filter(edit => edit.last <= first).reduce((sum, edit) => sum + edit.delta, 0);
      const from = first + offset;
      const until = last + offset;
      if (inlineSource && current.slice(from, until) !== normalize(inlineSource)) return;
      const replacement = normalize(inlineSource ? button.dataset.inlineReplacement : button.dataset.replacement);
      const updated = current.slice(0, from) + replacement + current.slice(until);
      textarea.setRangeText(replacement, start + from, start + until, "select");
      edits.set(source, {text: updated, ranges: [...previous.ranges, {first, last, delta: replacement.length - (last - first)}]});
      textarea.dispatchEvent(new Event("input", {bubbles: true}));
      status.textContent = "已填入改句。检查后可点「保存并复核」。";
    });
  }

  pane.addEventListener("change", event => {
    if (event.target.matches("[data-document-switch] select") && endpoint) {
      event.target.form.requestSubmit();
    } else if (event.target.matches("[data-zoom] select")) {
      const content = pane.querySelector("[data-preview-content]");
      const pdf = content.querySelector(".pdf");
      const image = pdf.querySelector("img");
      const centerX = (pdf.scrollLeft + pdf.clientWidth / 2) / image.clientWidth;
      const centerY = (pdf.scrollTop + pdf.clientHeight / 2) / image.clientHeight;
      zoomPercent = event.target.value;
      setZoom(content);
      if (pdf.dataset.focusY !== undefined) alignEvidence(content);
      else if (Number.isFinite(centerX) && Number.isFinite(centerY)) {
        pdf.scrollLeft = image.clientWidth * centerX - pdf.clientWidth / 2;
        pdf.scrollTop = image.clientHeight * centerY - pdf.clientHeight / 2;
      }
    }
  });

  pane.addEventListener("submit", event => {
    if (!endpoint || !event.target.matches("[data-page-jump], [data-document-switch]")) return;
    event.preventDefault();
    const pageUrl = new URL(event.target.action);
    pageUrl.search = new URLSearchParams(new FormData(event.target));
    updatePreview(pageUrl, {focusControl: event.target.matches("[data-page-jump]") ? "jump" : "document"});
  });

  document.addEventListener("submit", async event => {
    const lockedForm = event.target.closest("[data-lock-submit]");
    if (lockedForm) {
      if (lockedForm.dataset.submitting) {
        event.preventDefault();
      } else {
        const submit = lockedForm.querySelector("[type='submit']");
        lockedForm.dataset.submitting = "true";
        submit.dataset.idleLabel = submit.textContent;
        submit.disabled = true;
        submit.textContent = "正在提交复核…";
      }
      return;
    }
    const searchForm = event.target.closest("[data-evidence-search]");
    if (searchForm) {
      event.preventDefault();
      loadEvidenceList(searchForm.closest("[data-evidence-list]"), new URLSearchParams(new FormData(searchForm)));
      return;
    }
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

  window.addEventListener("pageshow", event => {
    if (!event.persisted) return;
    for (const form of document.querySelectorAll("[data-lock-submit][data-submitting]")) {
      const submit = form.querySelector("[type='submit']");
      submit.textContent = submit.dataset.idleLabel;
      submit.disabled = false;
      delete form.dataset.submitting;
    }
  });

  window.addEventListener("popstate", () => {
    if (!endpoint) return;
    const pageUrl = new URL(window.location.href);
    if (pageUrl.search === previewSearch) return;
    previewSearch = pageUrl.search;
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
