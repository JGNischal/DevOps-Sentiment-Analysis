const API_BASE =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? "http://localhost:5000/api"
    : "/api";

// Utility: show loading spinner on a button
function setLoading(btn, loading) {
  const origText = btn.dataset.origText || btn.textContent;
  if (loading) {
    btn.dataset.origText = origText;
    btn.textContent = "Loading...";
    btn.disabled = true;
  } else {
    btn.disabled = false;
    btn.textContent = btn.dataset.origText || origText;
  }
}

// Utility: show error in a container
function showError(container, message) {
  container.innerHTML = `<div class="error">${message}</div>`;
}

// Utility: clear error / result
function clear(container) {
  container.innerHTML = "";
}

// ---------- Charts & Metrics ----------

async function fetchJson(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return await res.json();
}

let metricsChart = null;
let rocChart = null;

function renderMetricsChart(metrics) {
  const ctx = document.getElementById("metricsChart").getContext("2d");

  // Metrics we want to show (all must be present in your `metrics.json`)
  const metricKeys = ["accuracy", "precision", "recall", "f1", "roc_auc"];
  const metricLabels = {
    accuracy: "Accuracy",
    precision: "Precision",
    recall: "Recall",
    f1: "F1 Score",
    roc_auc: "ROC‑AUC",
  };

  // Collect data for each model
  const modelNames = Object.keys(metrics);
  const datasets = metricKeys.map((key) => ({
    label: metricLabels[key],
    data: modelNames.map((m) => metrics[m][key]),
    backgroundColor: [
      "rgba(59, 130, 246, 0.7)",
      "rgba(16, 185, 129, 0.7)",
      "rgba(239, 68, 68, 0.7)",
    ],
    borderColor: [
      "rgba(59, 130, 246, 1)",
      "rgba(16, 185, 129, 1)",
      "rgba(239, 68, 68, 1)",
    ],
    borderWidth: 1,
  }));

  if (metricsChart) metricsChart.destroy();
  metricsChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: modelNames,
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        x: { stacked: false },
        y: {
          stacked: false,
          beginAtZero: true,
          max: 1.0,
        },
      },
    },
  });
}

function renderRocChart(rocCurves) {
  const ctx = document.getElementById("rocChart").getContext("2d");

  const modelNames = Object.keys(rocCurves);
  const colors = ["#3b82f6", "#10b981", "#ef4444"];

  const datasets = modelNames.map((m, i) => ({
    label: rocCurves[m].label,
    data: rocCurves[m].fpr.map((fpr, j) => ({
      x: fpr,
      y: rocCurves[m].tpr[j],
    })),
    borderColor: colors[i % colors.length],
    backgroundColor: colors[i % colors.length] + "40", // with alpha
    borderWidth: 2,
    fill: false,
    pointRadius: 0,
  }));

  if (rocChart) rocChart.destroy();
  rocChart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        x: {
          type: "linear",
          position: "bottom",
          min: 0,
          max: 1,
          ticks: {
            beginAtZero: true,
          },
          title: {
            display: true,
            text: "False Positive Rate (FPR)",
          },
        },
        y: {
          type: "linear",
          min: 0,
          max: 1,
          ticks: {
            beginAtZero: true,
          },
          title: {
            display: true,
            text: "True Positive Rate (TPR)",
          },
        },
      },
    },
  });
}

function renderConfusionMatrices(metrics) {
  const container = document.getElementById("confusionMatrices");
  container.innerHTML = "";

  const modelNames = Object.keys(metrics);
  const colors = ["#3b82f6", "#10b981", "#ef4444"];

  modelNames.forEach((m, i) => {
    const cm = metrics[m].confusion_matrix;

    const card = document.createElement("div");
    card.className = "cm-card";

    const h4 = document.createElement("h4");
    h4.textContent = m;
    h4.style.color = colors[i % colors.length];
    card.appendChild(h4);

    const table = document.createElement("table");
    table.className = "cm-table";

    // Create confusion matrix table: [[TN, FP], [FN, TP]]
    const matrix = [
      [cm.true_negative, cm.false_positive],
      [cm.false_negative, cm.true_positive],
    ];

    for (let r = 0; r < matrix.length; r++) {
      const tr = document.createElement("tr");
      for (let c = 0; c < matrix[r].length; c++) {
        const td = document.createElement("td");
        td.textContent = matrix[r][c].toLocaleString();
        tr.appendChild(td);
      }
      table.appendChild(tr);
    }

    card.appendChild(table);
    container.appendChild(card);
  });
}

// ---------- Model Dropdown ----------

async function populateModelSelect() {
  const select = document.getElementById("modelSelect");
  try {
    const response = await fetchJson("/models");
    const models = response.models;
    select.innerHTML = "";
    models.forEach((model) => {
      const opt = document.createElement("option");
      opt.value = model.name;
      opt.textContent = model.label;
      select.appendChild(opt);
    });
    select.disabled = false;
  } catch (err) {
    console.error("Failed to fetch models:", err);
    select.innerHTML = `<option value="">(Failed to load models)</option>`;
  }
}

// ---------- Live Prediction ----------

async function handlePrediction() {
  const reviewEl = document.getElementById("reviewText");
  const modelEl = document.getElementById("modelSelect");
  const btn = document.getElementById("predictBtn");
  const resultContainer = document.getElementById("resultContainer");

  const review = reviewEl.value.trim();
  const model = modelEl.value;

  if (!review) {
    showError(resultContainer, "Please enter a review.");
    return;
  }
  if (!model) {
    showError(resultContainer, "Please select a model.");
    return;
  }

  clear(resultContainer);
  setLoading(btn, true);

  try {
    const res = await fetch(API_BASE + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review, model }),
    });

    if (!res.ok) {
      const body = await res.text();
      throw new Error(
        `Prediction error: ${res.status} ${body.slice(0, 100)}...`,
      );
    }

    const data = await res.json();

    // Expecting: { sentiment: "positive"/"negative", confidence: 0..1, model_used: "..." }

    const isPos = data.sentiment === "positive";
    const badgeClass = isPos ? "sentiment-pos" : "sentiment-neg";
    const colorClass = isPos ? "success" : "danger";

    const resultHtml = `
      <div class="result-card">
        <div class="result-row">
          <span>Sentiment</span>
          <span class="sentiment-badge ${badgeClass}">
            ${data.sentiment.toUpperCase()}
          </span>
        </div>
        <div class="result-row">
          <span>Confidence</span>
          <span>${Math.round(data.confidence * 100)}%</span>
        </div>
        <div class="result-row">
          <span>Model</span>
          <span>${data.model_used}</span>
        </div>
        <div class="progress-container">
          <div class="progress-bar" id="confidenceBar"></div>
        </div>
      </div>
    `;
    resultContainer.innerHTML = resultHtml;

    // Animate confidence bar
    const bar = document.getElementById("confidenceBar");
    bar.style.width = "0%";
    setTimeout(() => {
      bar.style.width = `${Math.round(data.confidence * 100)}%`;
    }, 50);
  } catch (err) {
    console.error("Prediction failed:", err);
    showError(resultContainer, `Prediction failed: ${err.message}`);
  } finally {
    setLoading(btn, false);
  }
}

// ---------- Init on page load ----------

document.addEventListener("DOMContentLoaded", async () => {
  const predictBtn = document.getElementById("predictBtn");

  try {
    const metrics = await fetchJson("/metrics");
    renderMetricsChart(metrics);
    renderConfusionMatrices(metrics);
  } catch (err) {
    console.error("Failed to load metrics:", err);
    showError(
      document.getElementById("metricsChart").parentElement,
      "Failed to load metrics.",
    );
  }

  try {
    const rocCurves = await fetchJson("/roc-curves");
    renderRocChart(rocCurves);
  } catch (err) {
    console.error("Failed to load ROC curves:", err);
    showError(
      document.getElementById("rocChart").parentElement,
      "Failed to load ROC curves.",
    );
  }

  try {
    await populateModelSelect();
  } catch (err) {
    console.error("Failed to load models for dropdown:", err);
  }

  predictBtn.addEventListener("click", handlePrediction);
});
