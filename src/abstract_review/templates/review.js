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

  async function updatePreview(pageUrl, {scroll = false, mode = "push"} = {}) {
    if (!endpoint) return;
    const evidenceId = pageUrl.searchParams.get("evidence_id") || "";
    const previous = pane.querySelector("[data-preview-content]");
    clearPendingPreview();
    if ((previous.dataset.selectedEvidence || "") === evidenceId) {
      if (scroll && isNarrow.matches) focusPreview();
      return;
    }

    const url = new URL(endpoint, pageUrl);
    url.searchParams.set("version_id", pane.dataset.version);
    if (evidenceId) url.searchParams.set("evidence_id", evidenceId);
    const request = new AbortController();
    requested = request;
    pane.setAttribute("aria-busy", "true");
    notice.textContent = "正在打开证据原页";

    try {
      const response = await fetch(url, {signal: request.signal});
      if (!response.ok) throw new Error("Preview unavailable");
      const markup = new DOMParser().parseFromString(await response.text(), "text/html");
      const content = markup.querySelector("[data-preview-content]");
      if (!content || (content.dataset.selectedEvidence || "") !== evidenceId) throw new Error("Preview mismatch");
      if (request.signal.aborted) return;

      previous.replaceWith(content);
      preparePreview(content);
      setCurrentEvidence(evidenceId);
      const locationChanged = pageUrl.searchParams.get("evidence_id") !==
        new URL(window.location.href).searchParams.get("evidence_id");
      if (mode === "replace") history.replaceState(null, "", pageUrl);
      else if (mode === "push" && locationChanged) history.pushState(null, "", pageUrl);
      notice.textContent = evidenceId ? "已定位所选证据" : "已打开论文首页";
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
      pageUrl.searchParams.delete("evidence_id");
      pageUrl.hash = "";
      updatePreview(pageUrl, {mode});
    }
  }

  function applyFilter(key, syncPreview) {
    activeFilter = key;
    const matching = rows.filter(row => key === "all" || row.dataset.verdict === key);
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

    const initial = rows.find(row => row.open) || rows[0];
    openRow(initial);
    if (!pane.querySelector("[data-preview-content]").dataset.selectedEvidence) showRowEvidence(initial, "replace");
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
    const jump = event.target.closest('.report-nav a[href="#abstract"]');
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
