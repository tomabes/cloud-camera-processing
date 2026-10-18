const refreshMs = 2000;
let currentRun = null;
let availableRuns = [];
let resultRows = [];
let systemRows = [];
let summary = null;

function withCacheBust(path) {
  return `${path}?v=${Date.now()}`;
}

function parseCsv(text) {
  const records = [];
  let row = [];
  let value = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      row.push(value);
      value = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(value);
      if (row.some((item) => item !== "")) records.push(row);
      row = [];
      value = "";
    } else {
      value += character;
    }
  }
  if (value || row.length) {
    row.push(value);
    records.push(row);
  }
  if (records.length < 2) return [];
  const headers = records[0];
  return records.slice(1).map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] || ""]))
  );
}

function numericValues(rows, field) {
  return rows
    .filter((row) => row[field] !== "" && row[field] !== undefined)
    .map((row) => Number(row[field]))
    .filter((value) => Number.isFinite(value));
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function formatMs(value) {
  return value === null || value === undefined || value === "" ? "-" : `${Math.round(Number(value))} ms`;
}

function formatPercent(value) {
  return value === null || value === undefined || value === "" ? "-" : `${Number(value).toFixed(1)}%`;
}

function formatBytes(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (number >= 1024 ** 2) return `${(number / 1024 ** 2).toFixed(2)} MB`;
  if (number >= 1024) return `${(number / 1024).toFixed(1)} KB`;
  return `${number} B`;
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString();
}

function updateRunOptions(latestRun) {
  const select = document.getElementById("runSelect");
  const previous = select.value;
  select.replaceChildren();

  if (!availableRuns.length) {
    select.append(new Option("Waiting for results", ""));
    currentRun = null;
    return;
  }

  availableRuns.forEach((run) => {
    const cameraLabel = `${run.camera_count} camera${Number(run.camera_count) === 1 ? "" : "s"}`;
    select.append(new Option(`${cameraLabel} / ${formatTime(run.started_at_utc)}`, run.run_id));
  });

  const runIds = availableRuns.map((run) => run.run_id);
  if (runIds.includes(previous)) {
    select.value = previous;
  } else if (latestRun && runIds.includes(latestRun.run_id)) {
    select.value = latestRun.run_id;
  } else {
    select.value = runIds[0];
  }
  currentRun = availableRuns.find((run) => run.run_id === select.value) || null;
}

function selectedRows() {
  const camera = document.getElementById("cameraSelect").value;
  return camera === "all" ? resultRows : resultRows.filter((row) => row.camera_id === camera);
}

function updateCameraOptions() {
  const select = document.getElementById("cameraSelect");
  const previous = select.value;
  const cameraIds = [...new Set(resultRows.map((row) => row.camera_id).filter(Boolean))];
  select.replaceChildren(new Option("All cameras", "all"));
  cameraIds.forEach((cameraId) => select.append(new Option(cameraId, cameraId)));
  select.value = cameraIds.includes(previous) || previous === "all" ? previous : "all";
}

function cameraSeries(rows, key) {
  const colors = ["#2563eb", "#16a34a", "#f97316"];
  const cameraIds = [...new Set(rows.map((row) => row.camera_id).filter(Boolean))];
  return cameraIds.map((cameraId, index) => ({
    key,
    cameraId,
    label: cameraId,
    color: colors[index % colors.length],
  }));
}

function drawLineChart(svgId, rows, series, ySuffix = "") {
  const svg = document.getElementById(svgId);
  svg.replaceChildren();
  const width = 960;
  const height = Number(svg.getAttribute("viewBox").split(" ")[3]);
  const wrappedLegend = series.length > 3;
  const margin = { top: wrappedLegend ? 58 : 34, right: 30, bottom: 45, left: 58 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const activeSeries = series.map((item) => ({
    ...item,
    values: rows.map((row) => {
      if (item.cameraId && row.camera_id !== item.cameraId) return null;
      if (row[item.key] === "" || row[item.key] === undefined) return null;
      const value = Number(row[item.key]);
      return Number.isFinite(value) ? value : null;
    }),
  }));
  const allValues = activeSeries.flatMap((item) => item.values.filter((value) => value !== null));
  if (!rows.length || !allValues.length) {
    const message = document.createElementNS("http://www.w3.org/2000/svg", "text");
    message.setAttribute("x", width / 2);
    message.setAttribute("y", height / 2);
    message.setAttribute("text-anchor", "middle");
    message.setAttribute("class", "empty-chart");
    message.textContent = "Waiting for metrics...";
    svg.append(message);
    return;
  }

  const maximum = Math.max(1, ...allValues);
  for (let lineIndex = 0; lineIndex <= 4; lineIndex += 1) {
    const y = margin.top + (innerHeight / 4) * lineIndex;
    const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
    grid.setAttribute("x1", margin.left);
    grid.setAttribute("x2", width - margin.right);
    grid.setAttribute("y1", y);
    grid.setAttribute("y2", y);
    grid.setAttribute("class", "grid-line");
    svg.append(grid);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", margin.left - 9);
    label.setAttribute("y", y + 4);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "axis-label");
    label.textContent = `${Math.round(maximum - (maximum / 4) * lineIndex)}${ySuffix}`;
    svg.append(label);
  }

  activeSeries.forEach((item, seriesIndex) => {
    const points = item.values.map((value, index) => {
      if (value === null) return null;
      const x = margin.left + (rows.length === 1 ? innerWidth / 2 : index / (rows.length - 1) * innerWidth);
      const y = margin.top + innerHeight - value / maximum * innerHeight;
      return `${x},${y}`;
    }).filter(Boolean);
    if (!points.length) return;
    const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    polyline.setAttribute("points", points.join(" "));
    polyline.setAttribute("class", "chart-line");
    polyline.setAttribute("stroke", item.color);
    svg.append(polyline);
    const legendX = wrappedLegend
      ? margin.left + (seriesIndex % 3) * 290
      : margin.left + seriesIndex * 190;
    const legendY = wrappedLegend ? 22 + Math.floor(seriesIndex / 3) * 24 : 22;
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    marker.setAttribute("cx", legendX + 5);
    marker.setAttribute("cy", legendY - 5);
    marker.setAttribute("r", 5);
    marker.setAttribute("fill", item.color);
    const legend = document.createElementNS("http://www.w3.org/2000/svg", "text");
    legend.setAttribute("x", legendX + 17);
    legend.setAttribute("y", legendY);
    legend.setAttribute("class", "axis-label");
    legend.textContent = item.label;
    svg.append(marker, legend);
  });
}

function updateSummary(rows) {
  const successful = rows.filter((row) => row.status === "ok");
  const response = numericValues(successful, "vm_roundtrip_ms");

  document.getElementById("cameraMode").textContent = summary?.camera_count ?? "-";
  document.getElementById("averageResponse").textContent = formatMs(average(response));

  const cpu = numericValues(systemRows, "cpu_percent");
  const memory = numericValues(systemRows, "memory_percent");
  document.getElementById("cpuUsage").textContent = cpu.length
    ? `${formatPercent(average(cpu))} / ${formatPercent(Math.max(...cpu))}` : "-";
  document.getElementById("memoryUsage").textContent = memory.length
    ? `${formatPercent(average(memory))} / ${formatPercent(Math.max(...memory))}` : "-";

  const received = summary?.network_received_bytes;
  const sent = summary?.network_sent_bytes;
  document.getElementById("networkTraffic").textContent = summary
    ? `↓ ${formatBytes(received)} / ↑ ${formatBytes(sent)}` : "-";
}

function updateFrames() {
  const container = document.getElementById("cameraFrames");
  container.replaceChildren();
  if (!currentRun?.latest_images) return;
  const selectedCamera = document.getElementById("cameraSelect").value;
  Object.entries(currentRun.latest_images).forEach(([cameraId, imagePath]) => {
    if (selectedCamera !== "all" && selectedCamera !== cameraId) return;
    const figure = document.createElement("figure");
    figure.className = "camera-card";
    const caption = document.createElement("figcaption");
    caption.textContent = cameraId;
    const image = document.createElement("img");
    image.src = withCacheBust(imagePath);
    image.alt = `Latest processed frame from ${cameraId}`;
    figure.append(caption, image);
    container.append(figure);
  });
}

function updateTable(rows) {
  const table = document.getElementById("resultsTable");
  table.replaceChildren();
  const latestRows = rows.slice().reverse();
  if (!latestRows.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="7">Waiting for the first request...</td>';
    table.append(row);
    return;
  }
  latestRows.forEach((item) => {
    const row = document.createElement("tr");
    const values = [
      formatTime(item.completed_at_utc || item.captured_at_utc), item.camera_id, item.frame_number,
      item.person_count === "" ? "-" : item.person_count,
      item.vehicle_count === "" ? "-" : item.vehicle_count,
      formatMs(item.vm_roundtrip_ms), item.status || "-",
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 6) cell.className = item.status === "ok" ? "status-ok" : "status-error";
      row.append(cell);
    });
    table.append(row);
  });
}

function render() {
  const rows = selectedRows();
  document.getElementById("runStatus").textContent = currentRun?.status || "-";
  document.getElementById("resultCount").textContent = `${rows.length} request rows`;
  updateSummary(rows);
  updateFrames();
  updateTable(rows);
  drawLineChart("peopleChart", rows, cameraSeries(rows, "person_count"));
  drawLineChart("vehicleChart", rows, cameraSeries(rows, "vehicle_count"));
  drawLineChart("latencyChart", rows, [
    { key: "vm_inference_ms", label: "YOLO inference", color: "#2563eb" },
    { key: "vm_total_ms", label: "VM total", color: "#16a34a" },
    { key: "network_overhead_ms", label: "Network overhead", color: "#f97316" },
    { key: "vm_roundtrip_ms", label: "VM round-trip", color: "#7c3aed" },
    { key: "capture_ms", label: "Frame capture", color: "#dc2626" },
    { key: "end_to_end_ms", label: "End-to-end", color: "#0891b2" },
  ]);
  drawLineChart("resourceChart", systemRows, [
    { key: "cpu_percent", label: "CPU", color: "#2563eb" },
    { key: "memory_percent", label: "Memory", color: "#16a34a" },
  ], "%");
}

async function fetchOptionalJson(path) {
  const response = await fetch(withCacheBust(path));
  if (!response.ok) return null;
  return response.json();
}

async function refresh() {
  try {
    const [runsIndex, latestRun] = await Promise.all([
      fetchOptionalJson("runs_index.json"),
      fetchOptionalJson("current_run.json"),
    ]);
    availableRuns = Array.isArray(runsIndex?.runs) ? runsIndex.runs : [];
    if (latestRun && !availableRuns.some((run) => run.run_id === latestRun.run_id)) {
      availableRuns.unshift(latestRun);
    }
    updateRunOptions(latestRun);
    if (!currentRun) return;
    const [resultsResponse, metricsResponse, loadedSummary] = await Promise.all([
      fetch(withCacheBust(currentRun.results_csv)),
      fetch(withCacheBust(currentRun.system_metrics_csv)),
      fetchOptionalJson(currentRun.summary_json),
    ]);
    resultRows = resultsResponse.ok ? parseCsv(await resultsResponse.text()) : [];
    systemRows = metricsResponse.ok ? parseCsv(await metricsResponse.text()) : [];
    summary = loadedSummary;
    updateCameraOptions();
    render();
  } catch (error) {
    console.warn("VM dashboard refresh failed", error);
  }
}

document.getElementById("runSelect").addEventListener("change", refresh);
document.getElementById("cameraSelect").addEventListener("change", render);
refresh();
setInterval(refresh, refreshMs);
