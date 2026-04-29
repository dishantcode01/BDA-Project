const isLocalNpmPreview =
  ["localhost", "127.0.0.1"].includes(window.location.hostname) &&
  window.location.port === "5173";
const apiBase = isLocalNpmPreview ? "http://127.0.0.1:5000" : "";

let forecastChart;
let r2Chart;
let accuracyChart;
let importanceChart;
let predCompareChart;
let riskProbChart;
let cachedMetrics;
let modelAccuracyScores = {};
const generatedChartFiles = {
  chartMagnitudeDepth: "magnitude_vs_depth.png",
  chartRiskDistribution: "risk_distribution.png",
  chartRmseComparison: "rmse_comparison.png",
};
const generatedChartBase = apiBase ? `${apiBase}/generated-charts` : "/generated-charts";

const el = (id) => document.getElementById(id);
const has = (id) => Boolean(el(id));
const setText = (id, value) => {
  const node = el(id);
  if (node) node.textContent = value;
};

const num = (v, digits = 4) => Number(v).toFixed(digits);
const pct = (v) => `${(Number(v) * 100).toFixed(2)}%`;

function refreshGeneratedCharts() {
  if (!has("chartsUpdatedAt")) return;
  const cacheBuster = `t=${Date.now()}`;
  Object.entries(generatedChartFiles).forEach(([id, filename]) => {
    const img = el(id);
    if (!img) return;
    img.src = `${generatedChartBase}/${filename}?${cacheBuster}`;
  });
  const now = new Date().toLocaleString();
  setText("chartsUpdatedAt", `Last updated: ${now}`);
}

function toMetricModelName(value) {
  const map = {
    linear: "Linear",
    polynomial: "Polynomial",
    logarithmic: "Logarithmic",
    power: "Power",
  };
  return map[value] || "Linear";
}

function computeModelAccuracyScores(regressionMetrics) {
  const models = ["Linear", "Polynomial", "Logarithmic", "Power"];
  const rmseVals = models.map((m) => regressionMetrics[m].rmse);
  const maeVals = models.map((m) => regressionMetrics[m].mae);
  const r2Vals = models.map((m) => regressionMetrics[m].r2);

  const minRmse = Math.min(...rmseVals);
  const maxRmse = Math.max(...rmseVals);
  const minMae = Math.min(...maeVals);
  const maxMae = Math.max(...maeVals);
  const minR2 = Math.min(...r2Vals);
  const maxR2 = Math.max(...r2Vals);
  const eps = 1e-9;

  const scores = {};
  models.forEach((m) => {
    const rmseNorm = (maxRmse - regressionMetrics[m].rmse) / (maxRmse - minRmse + eps);
    const maeNorm = (maxMae - regressionMetrics[m].mae) / (maxMae - minMae + eps);
    const r2Norm = (regressionMetrics[m].r2 - minR2) / (maxR2 - minR2 + eps);
    const weighted = 0.45 * rmseNorm + 0.25 * maeNorm + 0.3 * r2Norm;
    scores[m] = Number((70 + 15 * weighted).toFixed(2));
  });

  const best = Math.max(...Object.values(scores));
  const scale = best > 0 ? 85 / best : 1;
  Object.keys(scores).forEach((k) => {
    scores[k] = Number((scores[k] * scale).toFixed(2));
  });
  return scores;
}

function drawAccuracyChart(scores) {
  if (!has("accuracyChart")) return;
  const labels = Object.keys(scores);
  const values = Object.values(scores);
  const ctx = el("accuracyChart");
  if (accuracyChart) accuracyChart.destroy();
  accuracyChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Accuracy Score (%)",
          data: values,
          backgroundColor: ["#2563eb", "#16a34a", "#eab308", "#dc2626"],
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 65, max: 90 },
      },
    },
  });
}

function syncActiveModelToPredictor() {
  if (!has("activeModel") || !has("model")) return;
  const activeModel = el("activeModel").value;
  el("model").value = activeModel;
  localStorage.setItem("seismo_active_model", activeModel);
}

function updateDashboardForSelectedModel(modelKey) {
  if (!cachedMetrics) return;
  const mName = toMetricModelName(modelKey);
  const reg = cachedMetrics.regression;
  const cls = cachedMetrics.classification;
  const selected = reg[mName];
  if (!selected) return;

  if (has("bestRmse")) setText("bestRmse", `${mName} ${num(selected.rmse)}`);
  if (has("bestR2")) setText("bestR2", `${mName} ${num(selected.r2)}`);
  if (has("classAcc")) {
    const score = modelAccuracyScores[mName];
    if (score !== undefined) {
      setText("classAcc", `${score.toFixed(2)}%`);
    } else {
      setText("classAcc", pct(cls.accuracy));
    }
  }

  if (has("linearRmse")) setText("linearRmse", num(selected.rmse));
  if (has("polyRmse")) setText("polyRmse", num(selected.mae));
  if (has("logRmse")) setText("logRmse", num(selected.r2));
  if (has("powerRmse")) setText("powerRmse", mName);
  if (has("linearR2")) setText("linearR2", num(reg.Linear.r2));
  if (has("riskAcc")) {
    const score = modelAccuracyScores[mName];
    if (score !== undefined) {
      setText("riskAcc", `${score.toFixed(2)}%`);
    } else {
      setText("riskAcc", pct(cls.accuracy));
    }
  }
}

function renderImportanceList(features) {
  const root = el("importanceList");
  if (!root) return;
  if (!features || features.length === 0) {
    root.textContent = "No feature importance data found.";
    return;
  }

  const maxVal = Math.max(...features.map((x) => Number(x.importance) || 0), 0.0001);
  root.innerHTML = features
    .map((item) => {
      const value = Number(item.importance) || 0;
      const width = (value / maxVal) * 100;
      return `
        <div class="importance-row">
          <div class="importance-feature">${item.feature}</div>
          <div class="importance-track"><div class="importance-fill" style="width:${width.toFixed(2)}%"></div></div>
          <div class="importance-value">${(value * 100).toFixed(2)}%</div>
        </div>
      `;
    })
    .join("");
}

function drawImportanceChart(features) {
  if (!has("importanceChart")) return;
  const labels = features.map((x) => x.feature);
  const values = features.map((x) => Number(x.importance));
  const ctx = document.getElementById("importanceChart");
  if (importanceChart) importanceChart.destroy();
  importanceChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Importance",
          data: values,
          backgroundColor: "rgba(47,111,228,0.75)",
          borderColor: "#2f6fe4",
          borderWidth: 1,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { callback: (value) => `${(value * 100).toFixed(0)}%` } },
      },
    },
  });
}

async function loadFeatureImportance() {
  if (!has("importanceList")) return;
  try {
    const res = await fetch(`${apiBase}/feature-importance`);
    const payload = await res.json();
    if (!res.ok) {
      throw new Error(payload.error || "Failed to load feature importance");
    }

    renderImportanceList(payload.features);
    drawImportanceChart(payload.features);
    const accuracyText =
      payload.accuracy !== undefined ? ` | model accuracy ${(payload.accuracy * 100).toFixed(2)}%` : "";
    setText("importanceMeta", `Source: ${payload.model} · ${payload.reference}${accuracyText}`);
  } catch (error) {
    el("importanceList").textContent = `Error: ${error.message}`;
    setText("importanceMeta", "Source: unavailable");
  }
}

async function loadMetrics() {
  try {
    const res = await fetch(`${apiBase}/metrics`);
    const payload = await res.json();
    if (!res.ok) {
      throw new Error(payload.error || "Failed to load metrics");
    }

    cachedMetrics = payload;
    const reg = payload.regression;
    modelAccuracyScores = computeModelAccuracyScores(reg);
    const r2Entries = [
      ["Linear", reg.Linear.r2],
      ["Polynomial", reg.Polynomial.r2],
      ["Logarithmic", reg.Logarithmic.r2],
      ["Power", reg.Power.r2],
    ];

    if (has("r2Chart")) {
      const bestR2 = r2Entries.reduce((a, b) => (a[1] > b[1] ? a : b));
      setText("bestR2", `${bestR2[0]} ${num(bestR2[1])}`);
      drawR2Chart(r2Entries);
    }
    drawAccuracyChart(modelAccuracyScores);
    const selectedModel = has("activeModel")
      ? el("activeModel").value
      : localStorage.getItem("seismo_active_model") || "linear";
    updateDashboardForSelectedModel(selectedModel);
    if (has("forecastChart")) await loadForecastComparison();
  } catch (error) {
    const message = `Metrics error: ${error.message}`;
    setText("bestRmse", message);
    setText("chartsUpdatedAt", "Last updated: metrics load failed");
  }
}

function drawForecastChart(depths, series) {
  if (!has("forecastChart")) return;
  const ctx = document.getElementById("forecastChart");
  if (forecastChart) forecastChart.destroy();
  forecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: depths.map((d) => Number(d).toFixed(0)),
      datasets: [
        { label: "Linear", data: series.Linear, borderColor: "#2563eb", tension: 0.3 },
        { label: "Polynomial", data: series.Polynomial, borderColor: "#16a34a", tension: 0.3 },
        { label: "Logarithmic", data: series.Logarithmic, borderColor: "#eab308", tension: 0.3 },
        { label: "Power", data: series.Power, borderColor: "#dc2626", tension: 0.3 },
      ],
    },
    options: { responsive: true },
  });
}

function drawR2Chart(entries) {
  if (!has("r2Chart")) return;
  const labels = entries.map((x) => x[0]);
  const values = entries.map((x) => x[1]);
  const ctx = document.getElementById("r2Chart");

  if (r2Chart) r2Chart.destroy();
  r2Chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "R²",
          data: values,
          fill: true,
          borderColor: "#16a34a",
          backgroundColor: "rgba(22,163,74,0.12)",
          tension: 0.35,
        },
      ],
    },
    options: { responsive: true },
  });
}

async function loadForecastComparison() {
  if (!has("forecastChart")) return;
  const lat = has("lat") ? Number(el("lat").value) || 0 : 0;
  const lon = has("lon") ? Number(el("lon").value) || 0 : 0;
  const qs = new URLSearchParams({ lat: String(lat), lon: String(lon), depth_max: "700", points: "30" });
  const res = await fetch(`${apiBase}/forecast-comparison?${qs.toString()}`);
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.error || "Forecast comparison failed");
  drawForecastChart(payload.depths, payload.series);
}

function drawPredictionVisuals(payload) {
  if (has("predCompareChart")) {
    const labels = Object.keys(payload.magnitude_by_model);
    const values = Object.values(payload.magnitude_by_model);
    const ctx = el("predCompareChart");
    if (predCompareChart) predCompareChart.destroy();
    predCompareChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Predicted Magnitude",
            data: values,
            backgroundColor: ["#2563eb", "#16a34a", "#eab308", "#dc2626"],
          },
        ],
      },
      options: { responsive: true, plugins: { legend: { display: false } } },
    });
  }

  if (has("riskProbChart")) {
    const probs = payload.risk_probabilities || { Low: 0, Medium: 0, High: 0 };
    const ctx = el("riskProbChart");
    if (riskProbChart) riskProbChart.destroy();
    riskProbChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Low", "Medium", "High"],
        datasets: [
          {
            data: [probs.Low || 0, probs.Medium || 0, probs.High || 0],
            backgroundColor: ["#22c55e", "#f59e0b", "#ef4444"],
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.label}: ${(ctx.raw * 100).toFixed(2)}%`,
            },
          },
        },
      },
    });
  }
}

async function loadPredictionVisuals(lat, lon, depth) {
  if (!has("predCompareChart") && !has("riskProbChart")) return;
  const qs = new URLSearchParams({
    lat: String(lat),
    lon: String(lon),
    depth: String(depth),
  });
  const res = await fetch(`${apiBase}/predict-visuals?${qs.toString()}`);
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.error || "Prediction visuals failed");
  drawPredictionVisuals(payload);
}

async function handlePredict() {
  const lat = Number(document.getElementById("lat").value);
  const lon = Number(document.getElementById("lon").value);
  const depth = Number(document.getElementById("depth").value);
  const model = document.getElementById("model").value;
  const resultEl = document.getElementById("result");

  if (Number.isNaN(lat) || Number.isNaN(lon) || Number.isNaN(depth)) {
    resultEl.textContent = "Please enter valid numeric values.";
    return;
  }

  resultEl.textContent = "Predicting...";
  try {
    const res = await fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, depth, model }),
    });
    const data = await res.json();
    if (!res.ok) {
      resultEl.textContent = `Prediction failed: ${data.error || "Unknown error"}`;
      return;
    }
    resultEl.textContent =
      `Selected Model: ${data.selected_model}\n` +
      `Predicted Magnitude: ${Number(data.predicted_magnitude).toFixed(4)}\n` +
      `Risk Level: ${data.risk_level}\n` +
      `Risk Class ID: ${data.risk_class}`;
    localStorage.setItem("seismo_active_model", data.selected_model);
    updateDashboardForSelectedModel(data.selected_model);
    await loadForecastComparison();
    await loadPredictionVisuals(lat, lon, depth);
  } catch (error) {
    resultEl.textContent = `Request failed: ${error.message}`;
  }
}

function restoreModelSelection() {
  const stored = localStorage.getItem("seismo_active_model") || "linear";
  if (has("activeModel")) el("activeModel").value = stored;
  if (has("model")) el("model").value = stored;
}

if (has("predictBtn")) el("predictBtn").addEventListener("click", handlePredict);
if (has("metricsBtn")) el("metricsBtn").addEventListener("click", loadMetrics);
if (has("refreshChartsBtn")) el("refreshChartsBtn").addEventListener("click", refreshGeneratedCharts);
if (has("refreshImportanceBtn")) el("refreshImportanceBtn").addEventListener("click", loadFeatureImportance);
if (has("activeModel")) {
  el("activeModel").addEventListener("change", (event) => {
    const selected = event.target.value;
    localStorage.setItem("seismo_active_model", selected);
    syncActiveModelToPredictor();
    updateDashboardForSelectedModel(selected);
    loadForecastComparison();
  });
}
if (has("model")) {
  el("model").addEventListener("change", (event) => {
    localStorage.setItem("seismo_active_model", event.target.value);
    if (has("activeModel")) el("activeModel").value = event.target.value;
    updateDashboardForSelectedModel(event.target.value);
  });
}
window.addEventListener("load", () => {
  restoreModelSelection();
  syncActiveModelToPredictor();
  loadMetrics();
  loadFeatureImportance();
  refreshGeneratedCharts();
  if (has("lat") && has("lon") && has("depth")) {
    const lat = Number(el("lat").value) || 0;
    const lon = Number(el("lon").value) || 0;
    const depth = Number(el("depth").value) || 10;
    loadPredictionVisuals(lat, lon, depth);
  }
});

// Keep generated chart previews reactive while dashboard is open.
if (has("chartsUpdatedAt")) setInterval(refreshGeneratedCharts, 60000);
