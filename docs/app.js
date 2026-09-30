(() => {
  const data = window.ANESTRACE_DATA;
  const table = document.getElementById("leaderboard-table");
  if (!data || !table) return;

  const tabs = [...document.querySelectorAll(".tab[data-level]")];
  const search = document.getElementById("model-search");
  const filter = document.getElementById("category-filter");
  const context = document.getElementById("table-context");
  const count = document.getElementById("table-count");
  const footnote = document.getElementById("table-footnote");
  const panel = document.getElementById("leaderboard-panel");
  const defaults = { perception: "teeGroundMiou", single: "avg", multi: "turnAvg" };
  const levelDescriptions = {
    perception: "L1 · Raw TEE and physiological inputs. Default sort: TEE grounding mIoU.",
    single: "L2 · One clinical decision from standardized evidence. Default sort: average clinical score.",
    multi: "L3 · Sequential decisions under a shared agent interface. Default sort: turn-level average."
  };
  const notes = {
    perception: "Scores are percentages. ACC = accuracy; mIoU = mean intersection over union. Higher is better for every L1 metric.",
    single: "Clinical scores and major/critical Safety Error are percentages. P95 latency is in seconds per question; lower is better for Safety Error and latency.",
    multi: "Clinical scores, major/critical Safety Error, and micro-averaged Evidence Acquisition are percentages. P95 latency is in seconds per decision turn. Evidence Acquisition is an exploratory automated measure; see the paper for its definition."
  };
  const state = { level: "perception", search: "", category: "all", sortKey: defaults.perception, sortDirection: "desc" };

  function rowsForLevel() {
    return state.level === "perception" ? data.perception : data.decision;
  }

  function formatValue(value, metric) {
    return Number(value).toFixed(metric.digits ?? 1);
  }

  function makeCell(tag, text, className) {
    const cell = document.createElement(tag);
    cell.textContent = text;
    if (className) cell.className = className;
    return cell;
  }

  function renderHeader(metrics) {
    const head = table.tHead;
    head.replaceChildren();

    const groups = document.createElement("tr");
    groups.className = "group-row";
    const modelHeader = makeCell("th", "Model");
    modelHeader.scope = "col";
    modelHeader.rowSpan = 2;
    groups.append(modelHeader);

    for (let index = 0; index < metrics.length;) {
      let end = index + 1;
      while (end < metrics.length && metrics[end].group === metrics[index].group) end++;
      const group = makeCell("th", metrics[index].group);
      group.scope = "colgroup";
      group.colSpan = end - index;
      groups.append(group);
      index = end;
    }
    head.append(groups);

    const labels = document.createElement("tr");
    labels.className = "metric-row";
    for (const metric of metrics) {
      const cell = document.createElement("th");
      cell.scope = "col";
      const active = state.sortKey === metric.key;
      cell.setAttribute("aria-sort", active ? (state.sortDirection === "asc" ? "ascending" : "descending") : "none");
      const button = document.createElement("button");
      button.type = "button";
      button.className = `sort-button${active ? " active" : ""}`;
      button.title = `Sort by ${metric.label} ${metric.unit}, ${metric.better === "high" ? "higher" : "lower"} is better`;
      button.setAttribute("aria-label", button.title);

      const label = makeCell("span", metric.label, "sort-label");
      const arrow = makeCell("span", active ? (state.sortDirection === "asc" ? "↑" : "↓") : (metric.better === "high" ? "↑" : "↓"), "sort-arrow");
      label.append(arrow);
      button.append(label, makeCell("span", metric.unit, "sort-unit"));
      button.addEventListener("click", () => {
        if (state.sortKey === metric.key) {
          state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
        } else {
          state.sortKey = metric.key;
          state.sortDirection = metric.better === "high" ? "desc" : "asc";
        }
        render();
      });
      cell.append(button);
      labels.append(cell);
    }
    head.append(labels);
  }

  function renderBody(metrics) {
    const allRows = rowsForLevel();
    const best = Object.fromEntries(metrics.map(metric => [
      metric.key,
      metric.better === "high"
        ? Math.max(...allRows.map(row => row[metric.key]))
        : Math.min(...allRows.map(row => row[metric.key]))
    ]));
    const visible = allRows
      .filter(row => (state.category === "all" || row.category === state.category) && row.model.toLowerCase().includes(state.search))
      .sort((a, b) => {
        const difference = a[state.sortKey] - b[state.sortKey];
        return (state.sortDirection === "asc" ? difference : -difference) || a.model.localeCompare(b.model);
      });

    const body = table.tBodies[0];
    body.replaceChildren();
    if (!visible.length) {
      const row = document.createElement("tr");
      row.className = "empty-row";
      const cell = makeCell("td", "No models match these filters.");
      cell.colSpan = metrics.length + 1;
      row.append(cell);
      body.append(row);
    }
    for (const model of visible) {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.append(makeCell("span", model.model, "model-name"), makeCell("span", data.categories[model.category], "model-category"));
      row.append(name);
      for (const metric of metrics) {
        row.append(makeCell("td", formatValue(model[metric.key], metric), model[metric.key] === best[metric.key] ? "best-value" : ""));
      }
      body.append(row);
    }
    count.textContent = `${visible.length} of ${allRows.length} models`;
  }

  function render() {
    const metrics = data.metrics[state.level];
    panel.setAttribute("aria-labelledby", `tab-${state.level}`);
    context.textContent = levelDescriptions[state.level];
    footnote.textContent = notes[state.level];
    table.dataset.level = state.level;
    tabs.forEach(tab => {
      const selected = tab.dataset.level === state.level;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    renderHeader(metrics);
    renderBody(metrics);
  }

  function selectLevel(level) {
    state.level = level;
    state.sortKey = defaults[level];
    state.sortDirection = "desc";
    render();
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectLevel(tab.dataset.level));
    tab.addEventListener("keydown", event => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const offset = event.key === "ArrowRight" ? 1 : -1;
      const next = tabs[(index + offset + tabs.length) % tabs.length];
      selectLevel(next.dataset.level);
      next.focus();
    });
  });
  search.addEventListener("input", () => { state.search = search.value.trim().toLowerCase(); renderBody(data.metrics[state.level]); });
  filter.addEventListener("change", () => { state.category = filter.value; renderBody(data.metrics[state.level]); });

  const copy = document.getElementById("copy-citation");
  copy.addEventListener("click", async () => {
    const citation = document.getElementById("bibtex").textContent.trim();
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(citation);
      } else {
        const field = document.createElement("textarea");
        field.value = citation;
        field.style.position = "fixed";
        field.style.opacity = "0";
        document.body.append(field);
        field.select();
        if (!document.execCommand("copy")) throw new Error("Copy unavailable");
        field.remove();
      }
      copy.textContent = "Copied";
      window.setTimeout(() => { copy.textContent = "Copy BibTeX"; }, 2000);
    } catch {
      copy.textContent = "Select BibTeX below";
      window.setTimeout(() => { copy.textContent = "Copy BibTeX"; }, 2500);
    }
  });

  render();
})();
