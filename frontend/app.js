/**
 * PharmaLens — Frontend Application Logic
 *
 * Handles API communication, prediction, screening,
 * and dynamic UI updates.
 */

const API_BASE = "http://localhost:8000";

// ─── Server Health Check ────────────────────────────────────────

async function checkServerHealth() {
    const statusDot = document.querySelector(".status-dot");
    const statusText = document.querySelector(".status-text");

    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        const data = await res.json();

        statusDot.className = "status-dot status-dot--online";
        statusText.textContent = data.model_loaded ? "Model Ready" : "No Model";

        if (data.model_loaded) {
            loadDatasetStats();
        }
    } catch {
        statusDot.className = "status-dot status-dot--offline";
        statusText.textContent = "API Offline";
    }
}

// ─── Prediction ─────────────────────────────────────────────────

async function runPrediction() {
    const smiles = document.getElementById("smiles-input").value.trim();
    const sequence = document.getElementById("sequence-input").value.trim();
    const explain = document.getElementById("explain-checkbox").checked;

    if (!smiles || !sequence) {
        showToast("Please enter both SMILES and protein sequence", "error");
        return;
    }

    const btn = document.getElementById("predict-btn");
    const resultPanel = document.getElementById("result-panel");

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Predicting...';

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ smiles, sequence, explain }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Prediction failed");
        }

        const data = await res.json();
        displayResult(data);
        resultPanel.style.display = "block";
        resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
        showToast(err.message, "error");
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn__icon">⚡</span> Predict Interaction Score';
    }
}

function displayResult(data) {
    const score = data.score;

    document.getElementById("result-score").textContent = score.toFixed(4);
    document.getElementById("result-time").textContent = `${(data.inference_time * 1000).toFixed(0)}ms`;
    document.getElementById("result-smiles").textContent = data.smiles || "—";
    document.getElementById("result-seqlen").textContent = `${data.sequence_length} amino acids`;

    // Binding strength classification (KIBA: lower = stronger)
    const bindingEl = document.getElementById("result-binding");
    if (score < 3) {
        bindingEl.textContent = "Strong Binder";
        bindingEl.className = "result-binding result-binding--strong";
    } else if (score < 10) {
        bindingEl.textContent = "Moderate Binder";
        bindingEl.className = "result-binding result-binding--moderate";
    } else {
        bindingEl.textContent = "Weak / Non-Binder";
        bindingEl.className = "result-binding result-binding--weak";
    }

    // SHAP explanation
    const shapSection = document.getElementById("shap-section");
    if (data.shap_values) {
        shapSection.style.display = "block";
        displaySHAP(data.shap_values);
    } else {
        shapSection.style.display = "none";
    }
}

function displaySHAP(shapData) {
    const baseEl = document.getElementById("shap-base");
    baseEl.textContent = `Base value: ${shapData.base_value.toFixed(4)}`;

    const container = document.getElementById("shap-features");
    container.innerHTML = "";

    const features = shapData.top_features || [];
    const maxAbs = Math.max(...features.map((f) => Math.abs(f.shap_value)), 0.01);

    features.forEach((f) => {
        const isPositive = f.shap_value > 0;
        const pct = (Math.abs(f.shap_value) / maxAbs) * 50;

        const bar = document.createElement("div");
        bar.className = "shap-bar";
        bar.innerHTML = `
            <span class="shap-bar__name" title="${f.feature}">${f.feature}</span>
            <span class="shap-bar__value ${isPositive ? "shap-bar__value--positive" : "shap-bar__value--negative"}">
                ${f.shap_value > 0 ? "+" : ""}${f.shap_value.toFixed(4)}
            </span>
            <div class="shap-bar__visual">
                <div class="shap-bar__fill ${isPositive ? "shap-bar__fill--positive" : "shap-bar__fill--negative"}"
                     style="width: ${pct}%"></div>
            </div>
        `;
        container.appendChild(bar);
    });
}

// ─── Screening ──────────────────────────────────────────────────

async function runScreening() {
    const drugsText = document.getElementById("screen-drugs").value.trim();
    const targetsText = document.getElementById("screen-targets").value.trim();

    if (!drugsText || !targetsText) {
        showToast("Please enter drug SMILES and protein sequences", "error");
        return;
    }

    const smilesList = drugsText.split("\n").map((s) => s.trim()).filter(Boolean);
    const sequenceList = targetsText.split("\n").map((s) => s.trim()).filter(Boolean);

    const btn = document.getElementById("screen-btn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Screening...';

    try {
        const res = await fetch(`${API_BASE}/screen`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                smiles_list: smilesList,
                sequence_list: sequenceList,
                drug_ids: smilesList.map((_, i) => `Drug_${i + 1}`),
                target_ids: sequenceList.map((_, i) => `Target_${i + 1}`),
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Screening failed");
        }

        const data = await res.json();
        displayScreeningResults(data);
    } catch (err) {
        showToast(err.message, "error");
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn__icon">📊</span> Run Screening';
    }
}

function displayScreeningResults(data) {
    const container = document.getElementById("screen-results");
    container.style.display = "block";

    document.getElementById("screen-count").textContent = `${data.total_combinations} combinations`;

    const tbody = document.querySelector("#screen-table tbody");
    tbody.innerHTML = "";

    data.results.forEach((r, idx) => {
        const score = r.predicted_score;
        let bindingClass = "";
        let bindingText = "";

        if (score !== null) {
            if (score < 3) {
                bindingClass = "result-binding--strong";
                bindingText = "Strong";
            } else if (score < 10) {
                bindingClass = "result-binding--moderate";
                bindingText = "Moderate";
            } else {
                bindingClass = "result-binding--weak";
                bindingText = "Weak";
            }
        }

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${r.rank || idx + 1}</td>
            <td>${r.drug_id}</td>
            <td>${r.target_id}</td>
            <td>${score !== null ? score.toFixed(4) : "Error"}</td>
            <td><span class="${bindingClass}" style="font-weight:600">${bindingText}</span></td>
        `;
        tbody.appendChild(tr);
    });

    container.scrollIntoView({ behavior: "smooth" });
}

// ─── Dataset Stats ──────────────────────────────────────────────

async function loadDatasetStats() {
    try {
        const res = await fetch(`${API_BASE}/dataset/stats`);
        if (!res.ok) return;

        const data = await res.json();

        // Hero stats
        animateNumber("stat-drugs", data.unique_drugs);
        animateNumber("stat-targets", data.unique_targets);
        animateNumber("stat-interactions", data.total_interactions);

        // Explore section
        document.getElementById("ds-drugs").textContent = data.unique_drugs.toLocaleString();
        document.getElementById("ds-targets").textContent = data.unique_targets.toLocaleString();
        document.getElementById("ds-interactions").textContent = data.total_interactions.toLocaleString();
        document.getElementById("ds-score-range").textContent = `${data.score_range[0].toFixed(1)} – ${data.score_range[1].toFixed(1)}`;

        // Split info
        if (data.split_sizes) {
            displaySplitBars(data.split_sizes, data.total_interactions);
        }
    } catch {
        // API offline — silently ignore
    }
}

function displaySplitBars(splits, total) {
    const container = document.getElementById("split-info");
    container.innerHTML = "";

    const colors = ["#6366F1", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"];

    Object.entries(splits).forEach(([name, count], idx) => {
        const pct = (count / total) * 100;
        const color = colors[idx % colors.length];

        const item = document.createElement("div");
        item.className = "split-bar-item";
        item.innerHTML = `
            <span class="split-bar-item__label">${name.replace(/_/g, " ")}</span>
            <div class="split-bar-item__bar">
                <div class="split-bar-item__fill" style="width: ${pct}%; background: ${color}"></div>
            </div>
            <span class="split-bar-item__value">${count.toLocaleString()}</span>
        `;
        container.appendChild(item);
    });
}

// ─── Model Comparison ───────────────────────────────────────────

async function loadModelComparison() {
    try {
        const res = await fetch(`${API_BASE}/models/comparison`);
        if (!res.ok) throw new Error("Failed to load");

        const data = await res.json();
        displayModelTable(data.models || []);
    } catch (err) {
        showToast("Could not load model results. Is the API running?", "error");
    }
}

function displayModelTable(models) {
    const tbody = document.getElementById("model-table-body");
    tbody.innerHTML = "";

    if (models.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="placeholder-text">No experiments logged yet. Run the research notebooks first.</td></tr>';
        return;
    }

    // Sort by RMSE
    models.sort((a, b) => (a.rmse || 999) - (b.rmse || 999));

    models.forEach((m, idx) => {
        const tr = document.createElement("tr");
        if (idx === 0) tr.style.background = "rgba(16, 185, 129, 0.05)";

        tr.innerHTML = `
            <td style="font-weight: 600">${m.model}${idx === 0 ? " 🏆" : ""}</td>
            <td>${fmt(m.rmse)}</td>
            <td>${fmt(m.mae)}</td>
            <td>${fmt(m.pearson)}</td>
            <td>${fmt(m.r2)}</td>
            <td>${fmt(m.ci)}</td>
            <td>${m.training_time ? m.training_time.toFixed(1) + "s" : "—"}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ─── Utilities ──────────────────────────────────────────────────

function fmt(val) {
    return val !== null && val !== undefined ? val.toFixed(4) : "—";
}

function animateNumber(elementId, target) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const duration = 1200;
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (target - start) * eased);

        el.textContent = current.toLocaleString();

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function showToast(message, type = "info") {
    // Remove existing toast
    const existing = document.querySelector(".toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        padding: 14px 24px;
        border-radius: 10px;
        font-size: 14px;
        font-weight: 500;
        color: white;
        z-index: 10000;
        animation: fadeInUp 0.3s ease-out;
        max-width: 400px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        background: ${type === "error" ? "#EF4444" : type === "success" ? "#10B981" : "#6366F1"};
    `;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transition = "opacity 0.3s";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ─── Navigation ─────────────────────────────────────────────────

document.querySelectorAll(".nav__link").forEach((link) => {
    link.addEventListener("click", (e) => {
        document.querySelectorAll(".nav__link").forEach((l) => l.classList.remove("nav__link--active"));
        e.currentTarget.classList.add("nav__link--active");
    });
});

// Active section observer
const observer = new IntersectionObserver(
    (entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                const id = entry.target.id;
                document.querySelectorAll(".nav__link").forEach((link) => {
                    link.classList.toggle("nav__link--active", link.dataset.section === id);
                });
            }
        });
    },
    { threshold: 0.3 }
);

document.querySelectorAll(".section").forEach((section) => observer.observe(section));

// ─── Initialize ─────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    checkServerHealth();
    setInterval(checkServerHealth, 15000);
});
