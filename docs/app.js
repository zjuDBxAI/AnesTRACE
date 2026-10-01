(() => {
  const data = window.ANESTRACE_DATA;
  const table = document.getElementById("leaderboard-table");
  if (!data || !table) return;

  const chinese = document.documentElement.lang.startsWith("zh");
  const translations = {
    Model: "模型", Proprietary: "闭源模型", "General-purpose": "通用模型", "Medical-domain": "医学领域模型",
    "Foundational perception": "基础感知", "Visual grounding": "视觉定位", "Functional assessment": "功能评估",
    "Value extraction": "数值提取", "Trend reasoning": "趋势推理", "Anomaly perception": "异常感知", Waveform: "波形",
    Risk: "风险预测", Diagnosis: "临床诊断", Intervention: "干预决策", Reassessment: "复评计划", Average: "平均分",
    "Safety error": "严重/危急安全错误", "P95 latency": "P95 延迟", "Turn-level average": "当轮平均分",
    "Temporal consistency": "时序一致性", "Evidence acquisition": "证据获取",
    "Task scores": "任务得分", "Aggregate & efficiency": "综合表现与效率", "Clinical quality": "临床决策质量",
    "Safety & agent behavior": "安全性与 Agent 表现", score: "分数", s: "秒"
  };
  const translate = text => chinese ? (translations[text] ?? text) : text;

  const metricUnit = metric => metric.unit === "score" ? "%" : translate(metric.unit);

  const languageSwitch = document.querySelector(".language-switch");
  languageSwitch?.addEventListener("click", () => { languageSwitch.hash = window.location.hash; });

  const tabs = [...document.querySelectorAll(".tab[data-level]")];
  const search = document.getElementById("model-search");
  const filter = document.getElementById("category-filter");
  const context = document.getElementById("table-context");
  const count = document.getElementById("table-count");
  const footnote = document.getElementById("table-footnote");
  const panel = document.getElementById("leaderboard-panel");
  const shortcuts = [...document.querySelectorAll("[data-sort]")];
  const defaults = { perception: "teeGroundMiou", single: "avg", multi: "turnAvg" };
  const sortKeys = {
    perception: { quality: defaults.perception },
    single: { quality: defaults.single, safety: "l2Safety", latency: "l2Latency" },
    multi: { quality: defaults.multi, safety: "l3Safety", latency: "l3Latency" }
  };
  const levelDescriptions = chinese ? {
    perception: "L1 · 原始 TEE 与生理信号输入。",
    single: "L2 · 根据标准化证据完成单点临床决策。",
    multi: "L3 · 在统一 Agent 接口下完成多步决策。"
  } : {
    perception: "L1 · Raw TEE and physiological inputs.",
    single: "L2 · One clinical decision from standardized evidence.",
    multi: "L3 · Sequential decisions under a shared agent interface."
  };
  const notes = chinese ? {
    perception: "得分以百分制显示。ACC 为准确率，mIoU 为平均交并比。L1 所有指标均为越高越好。",
    single: "临床得分为百分制，严重/危急安全错误以百分比表示。P95 延迟的单位为秒/题；安全错误率与延迟越低越好。",
    multi: "临床得分为百分制，严重/危急安全错误与微平均证据获取率以百分比表示。P95 延迟的单位为秒/决策轮次。证据获取率是探索性的自动化指标，定义见下方指标说明。"
  } : {
    perception: "Scores are percentages. ACC = accuracy; mIoU = mean intersection over union. Higher is better for every L1 metric.",
    single: "Clinical scores and major/critical Safety Error are percentages. P95 latency is in seconds per question; lower is better for Safety Error and latency.",
    multi: "Clinical scores, major/critical Safety Error, and micro-averaged Evidence Acquisition are percentages. P95 latency is in seconds per decision turn. Evidence Acquisition is an exploratory automated measure; see the metric definitions below."
  };
  const state = { level: "perception", search: "", category: "all", sortKey: defaults.perception, sortDirection: "desc" };

  function rowsForLevel() {
    return state.level === "perception" ? data.perception : data.decision;
  }

  function metricsForLevel() {
    const order = state.level === "single"
      ? ["avg", "l2Safety", "l2Latency", "risk", "diagnosis", "intervention", "reassessment"]
      : state.level === "multi"
        ? ["turnAvg", "temporal", "l3Safety", "l3Latency", "evidence"]
        : data.metrics.perception.map(metric => metric.key);
    return order.map(key => data.metrics[state.level].find(metric => metric.key === key));
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
    const modelHeader = makeCell("th", translate("Model"));
    modelHeader.scope = "col";
    modelHeader.rowSpan = 2;
    groups.append(modelHeader);

    for (let index = 0; index < metrics.length;) {
      let end = index + 1;
      while (end < metrics.length && metrics[end].group === metrics[index].group) end++;
      const group = makeCell("th", translate(metrics[index].group));
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
      if (active) cell.setAttribute("aria-sort", state.sortDirection === "asc" ? "ascending" : "descending");
      const button = document.createElement("button");
      button.type = "button";
      button.className = `sort-button${active ? " active" : ""}`;
      button.title = chinese
        ? `按${translate(metric.label)}（${metricUnit(metric)}）排序，越${metric.better === "high" ? "高" : "低"}越好`
        : `Sort by ${metric.label} ${metricUnit(metric)}, ${metric.better === "high" ? "higher" : "lower"} is better`;
      button.setAttribute("aria-label", button.title);

      const label = makeCell("span", translate(metric.label), "sort-label");
      const arrow = makeCell("span", active ? (state.sortDirection === "asc" ? "↑" : "↓") : "", "sort-arrow");
      label.append(arrow);
      button.append(label, makeCell("span", metricUnit(metric), "sort-unit"));
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
      const cell = makeCell("td", chinese ? "没有符合筛选条件的模型。" : "No models match these filters.");
      cell.colSpan = metrics.length + 1;
      row.append(cell);
      body.append(row);
    }
    for (const model of visible) {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.append(makeCell("span", model.model, "model-name"), makeCell("span", translate(data.categories[model.category]), "model-category"));
      row.append(name);
      for (const metric of metrics) {
        row.append(makeCell("td", formatValue(model[metric.key], metric), model[metric.key] === best[metric.key] ? "best-value" : ""));
      }
      body.append(row);
    }
    count.textContent = chinese ? `显示 ${visible.length} / ${allRows.length} 个模型` : `${visible.length} of ${allRows.length} models`;
  }

  function render() {
    const metrics = metricsForLevel();
    const selectedMetric = metrics.find(metric => metric.key === state.sortKey);
    panel.setAttribute("aria-labelledby", `tab-${state.level}`);
    context.textContent = `${levelDescriptions[state.level]} ${chinese ? "当前排序：" : "Sorted by: "}${translate(selectedMetric.label)} (${metricUnit(selectedMetric)}) ${state.sortDirection === "asc" ? "↑" : "↓"}`;
    footnote.textContent = notes[state.level];
    table.dataset.level = state.level;
    tabs.forEach(tab => {
      const selected = tab.dataset.level === state.level;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    shortcuts.forEach(button => {
      const key = sortKeys[state.level][button.dataset.sort];
      button.disabled = !key;
      button.setAttribute("aria-pressed", String(key === state.sortKey));
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
  shortcuts.forEach(button => button.addEventListener("click", () => {
    const key = sortKeys[state.level][button.dataset.sort];
    if (!key) return;
    state.sortKey = key;
    state.sortDirection = data.metrics[state.level].find(metric => metric.key === key).better === "high" ? "desc" : "asc";
    render();
  }));
  search.addEventListener("input", () => { state.search = search.value.trim().toLowerCase(); renderBody(metricsForLevel()); });
  filter.addEventListener("change", () => { state.category = filter.value; renderBody(metricsForLevel()); });

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
      copy.textContent = chinese ? "已复制" : "Copied";
      window.setTimeout(() => { copy.textContent = chinese ? "复制 BibTeX" : "Copy BibTeX"; }, 2000);
    } catch {
      copy.textContent = chinese ? "请选中下方 BibTeX" : "Select BibTeX below";
      window.setTimeout(() => { copy.textContent = chinese ? "复制 BibTeX" : "Copy BibTeX"; }, 2500);
    }
  });

  render();
})();
