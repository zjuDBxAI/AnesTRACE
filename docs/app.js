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
    "Safety error": "安全错误", "P95 latency": "P95 延迟", "Turn-level average": "当轮平均分",
    "Temporal consistency": "时序一致性", "Evidence acquisition": "证据获取",
    "Task scores": "任务得分", "Aggregate & efficiency": "综合表现与效率", "Clinical quality": "临床决策质量",
    "Safety & agent behavior": "安全性与 Agent 表现", score: "分数", s: "秒"
  };
  const translate = text => chinese ? (translations[text] ?? text) : text;

  const languageSwitch = document.querySelector(".language-switch");
  languageSwitch?.addEventListener("click", () => { languageSwitch.hash = window.location.hash; });

  const tabs = [...document.querySelectorAll(".tab[data-level]")];
  const search = document.getElementById("model-search");
  const filter = document.getElementById("category-filter");
  const context = document.getElementById("table-context");
  const count = document.getElementById("table-count");
  const footnote = document.getElementById("table-footnote");
  const panel = document.getElementById("leaderboard-panel");
  const defaults = { perception: "teeGroundMiou", single: "avg", multi: "turnAvg" };
  const levelDescriptions = chinese ? {
    perception: "L1 · 原始 TEE 与生理信号输入。默认按 TEE 视觉定位 mIoU 排序。",
    single: "L2 · 根据标准化证据完成单点临床决策。默认按临床平均分排序。",
    multi: "L3 · 在统一 Agent 接口下完成多步决策。默认按当轮平均分排序。"
  } : {
    perception: "L1 · Raw TEE and physiological inputs. Default sort: TEE grounding mIoU.",
    single: "L2 · One clinical decision from standardized evidence. Default sort: average clinical score.",
    multi: "L3 · Sequential decisions under a shared agent interface. Default sort: turn-level average."
  };
  const notes = chinese ? {
    perception: "得分以百分制显示。ACC 为准确率，mIoU 为平均交并比。L1 所有指标均为越高越好。",
    single: "临床得分为百分制，严重/危急安全错误以百分比表示。P95 延迟的单位为秒/题；安全错误率与延迟越低越好。",
    multi: "临床得分为百分制，严重/危急安全错误与微平均证据获取率以百分比表示。P95 延迟的单位为秒/决策轮次。证据获取率是探索性的自动化指标，定义见论文。"
  } : {
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
      cell.setAttribute("aria-sort", active ? (state.sortDirection === "asc" ? "ascending" : "descending") : "none");
      const button = document.createElement("button");
      button.type = "button";
      button.className = `sort-button${active ? " active" : ""}`;
      button.title = chinese
        ? `按${translate(metric.label)}（${translate(metric.unit)}）排序，越${metric.better === "high" ? "高" : "低"}越好`
        : `Sort by ${metric.label} ${metric.unit}, ${metric.better === "high" ? "higher" : "lower"} is better`;
      button.setAttribute("aria-label", button.title);

      const label = makeCell("span", translate(metric.label), "sort-label");
      const arrow = makeCell("span", active ? (state.sortDirection === "asc" ? "↑" : "↓") : (metric.better === "high" ? "↑" : "↓"), "sort-arrow");
      label.append(arrow);
      button.append(label, makeCell("span", translate(metric.unit), "sort-unit"));
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
      copy.textContent = chinese ? "已复制" : "Copied";
      window.setTimeout(() => { copy.textContent = chinese ? "复制 BibTeX" : "Copy BibTeX"; }, 2000);
    } catch {
      copy.textContent = chinese ? "请选中下方 BibTeX" : "Select BibTeX below";
      window.setTimeout(() => { copy.textContent = chinese ? "复制 BibTeX" : "Copy BibTeX"; }, 2500);
    }
  });

  render();
})();
