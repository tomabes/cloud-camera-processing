const refreshMs = 2000;
let allRows = [];

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

function deduplicateRows(rows) {
  const byBlob = new Map();
  rows.forEach((row) => {
    const key = row.raw_blob || `${row.run_id}-${row.camera_id}-${row.frame_number}`;
    const previous = byBlob.get(key);
    if (!previous || previous.status !== "ok" || row.status === "ok") {
      byBlob.set(key, row);
    }
  });
  return [...byBlob.values()].sort(
    (left, right) => new Date(left.processed_at_utc || 0) - new Date(right.processed_at_utc || 0)
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

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString();
}

function updateRunOptions() {
  const select = document.getElementById("runSelect");
  const previous = select.value;
  const runsById = new Map();
  allRows.forEach((row) => {
    const timestamp = new Date(row.processed_at_utc || 0).getTime();
    const existing = runsById.get(row.run_id);
    if (!existing || timestamp > existing.timestamp) {
      runsById.set(row.run_id, {
        runId: row.run_id,
        cameraCount: row.camera_count,
        timestamp,
      });
    }
  });
  const runs = [...runsById.values()]
    .filter((run) => run.runId)
    .sort((left, right) => right.timestamp - left.timestamp);
  const runIds = runs.map((run) => run.runId);
  select.replaceChildren();
  if (!runIds.length) {
    select.append(new Option("Waiting for results", ""));
    return;
  }
  runs.forEach((run) => {
    const cameraLabel = `${run.cameraCount} camera${Number(run.cameraCount) === 1 ? "" : "s"}`;
    select.append(new Option(`${cameraLabel} / ${formatTime(run.timestamp)}`, run.runId));
  });
  select.value = runIds.includes(previous) ? previous : runIds[0];
}

function rowsForSelectedRun() {
  const runId = document.getElementById("runSelect").value;
  return allRows.filter((row) => row.run_id === runId);
}

function updateCameraOptions(runRows) {
  const select = document.getElementById("cameraSelect");
  const previous = select.value;
  const cameraIds = [...new Set(runRows.map((row) => row.camera_id).filter(Boolean))];
  select.replaceChildren(new Option("All cameras", "all"));
  cameraIds.forEach((cameraId) => select.append(new Option(cameraId, cameraId)));
  select.value = cameraIds.includes(previous) || previous === "all" ? previous : "all";
}

function selectedRows() {
  const runRows = rowsForSelectedRun();
  const cameraId = document.getElementById("cameraSelect").value;
  return cameraId === "all" ? runRows : runRows.filter((row) => row.camera_id === cameraId);
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

function drawLineChart(svgId, rows, series) {
  const svg = document.getElementById(svgId);
  svg.replaceChildren();
  const width = 960;
  const height = Number(svg.getAttribute("viewBox").split(" ")[3]);
  const margin = { top: 34, right: 30, bottom: 45, left: 58 };
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
    const y = margin.top + innerHeight / 4 * lineIndex;
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
    label.textContent = Math.round(maximum - maximum / 4 * lineIndex);
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
    if (item.dash) polyline.setAttribute("stroke-dasharray", item.dash);
    if (item.linecap) polyline.setAttribute("stroke-linecap", item.linecap);
    svg.append(polyline);
    const legendX = margin.left + seriesIndex * 190;
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    marker.setAttribute("cx", legendX + 5);
    marker.setAttribute("cy", 17);
    marker.setAttribute("r", 5);
    marker.setAttribute("fill", item.color);
    const legend = document.createElementNS("http://www.w3.org/2000/svg", "text");
    legend.setAttribute("x", legendX + 17);
    legend.setAttribute("y", 22);
    legend.setAttribute("class", "axis-label");
    legend.textContent = item.label;
    svg.append(marker, legend);
  });
}

function updateSummary(rows) {
  const successful = rows.filter((row) => row.status === "ok");
  const failed = rows.filter((row) => row.status !== "ok");
  const first = rows[0] || {};
  const response = numericValues(successful, "end_to_end_ms");
  const expected = Number(first.camera_count || 0) * Number(first.expected_frames_per_camera || 0);
  const missing = Math.max(0, expected - rows.length);

  document.getElementById("cameraMode").textContent = first.camera_count || "-";
  document.getElementById("averageResponse").textContent = formatMs(average(response));
  document.getElementById("processedCount").textContent = expected
    ? `${successful.length} / ${expected}` : successful.length;
  document.getElementById("failedCount").textContent = failed.length;
  document.getElementById("missingCount").textContent = expected ? missing : "-";
  document.getElementById("runStatus").textContent = !rows.length
    ? "waiting"
    : expected && rows.length < expected
      ? "processing"
      : failed.length
        ? "finished with errors"
        : "finished";
}

function updateFrames(rows) {
  const container = document.getElementById("cameraFrames");
  container.replaceChildren();
  const latestByCamera = new Map();
  rows.filter((row) => row.status === "ok").forEach((row) => {
    latestByCamera.set(row.camera_id, row);
  });
  latestByCamera.forEach((row, cameraId) => {
    const figure = document.createElement("figure");
    figure.className = "camera-card";
    const caption = document.createElement("figcaption");
    caption.textContent = cameraId;
    const image = document.createElement("img");
    image.src = withCacheBust(row.processed_image);
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
    row.innerHTML = '<td colspan="7">Waiting for the first processed frame...</td>';
    table.append(row);
    return;
  }
  latestRows.forEach((item) => {
    const row = document.createElement("tr");
    const values = [
      formatTime(item.processed_at_utc), item.camera_id, item.frame_number,
      item.person_count === "" ? "-" : item.person_count,
      item.vehicle_count === "" ? "-" : item.vehicle_count,
      formatMs(item.end_to_end_ms), item.status || "-",
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
  const runRows = rowsForSelectedRun();
  const rows = selectedRows();
  document.getElementById("resultCount").textContent = `${rows.length} request rows`;
  updateSummary(runRows);
  updateFrames(rows);
  updateTable(rows);
  drawLineChart("latencyChart", rows, [
    { key: "vision_ms", label: "Azure AI", color: "#2563eb" },
    { key: "response_time_ms", label: "Function", color: "#f97316", dash: "1 7", linecap: "round" },
    { key: "end_to_end_ms", label: "End-to-end", color: "#7c3aed" },
  ]);
  drawLineChart("peopleChart", rows, cameraSeries(rows, "person_count"));
  drawLineChart("vehicleChart", rows, cameraSeries(rows, "vehicle_count"));
}

async function refresh() {
  try {
    const response = await fetch(withCacheBust("runs/results.csv"));
    if (!response.ok) return;
    allRows = deduplicateRows(parseCsv(await response.text())).filter(
      (row) => !row.run_id.startsWith("serverless_stress_")
    );
    updateRunOptions();
    updateCameraOptions(rowsForSelectedRun());
    render();
  } catch (error) {
    console.warn("Serverless dashboard refresh failed", error);
  }
}

document.getElementById("runSelect").addEventListener("change", () => {
  updateCameraOptions(rowsForSelectedRun());
  render();
});
document.getElementById("cameraSelect").addEventListener("change", render);
refresh();
setInterval(refresh, refreshMs);
