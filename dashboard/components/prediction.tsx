"use client";

import { useState } from "react";
import { ArrowRight, Clock3, Download, Loader2 } from "lucide-react";
import { api, number, Prediction, scope } from "@/lib/api";
import {
  Empty,
  ModelPicker,
  PageHeading,
  useWorkspace,
} from "@/components/workspace";

export function PredictionForm() {
  const { models, loading } = useWorkspace();
  const [modelId, setModelId] = useState("");
  const [smiles, setSmiles] = useState("");
  const [sequence, setSequence] = useState("");
  const [explain, setExplain] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const model = models?.find((candidate) => candidate.name === modelId);
  async function predict(event: React.FormEvent) {
    event.preventDefault();
    if (!model) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const data = await api<Prediction>("/predict", {
        method: "POST",
        body: JSON.stringify({
          ...scope(model),
          smiles: smiles.trim(),
          sequence: sequence.trim(),
          explain,
        }),
      });
      setResult(data);
    } catch (issue) {
      setError(issue instanceof Error ? issue.message : "Prediction failed.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeading
        title="Predict an interaction"
        description="Evaluate one molecule–protein pair using a registered, compatible model."
      />
      {!loading && !models?.length ? (
        <div className="notice warning">
          No model artifacts are registered. Dataset and experiment views remain
          available.
        </div>
      ) : null}
      <div className="prediction-grid">
        <form className="panel form-panel" onSubmit={predict} aria-busy={busy}>
          <div className="panel-heading">
            <div>
              <h2>Prediction inputs</h2>
              <p>Output units follow the selected model’s training task.</p>
            </div>
          </div>
          <ModelPicker
            models={models || []}
            value={modelId}
            onChange={(value) => {
              setModelId(value);
              setResult(null);
            }}
            disabled={busy}
          />
          <label htmlFor="smiles">
            Molecule SMILES
            <textarea
              id="smiles"
              className="mono"
              rows={3}
              placeholder="Paste a canonical SMILES string"
              required
              maxLength={10000}
              value={smiles}
              onChange={(event) => setSmiles(event.target.value)}
              disabled={busy}
            />
          </label>
          <div className="field-hint">
            {smiles.length.toLocaleString()} characters · validated before
            inference
          </div>
          <label htmlFor="sequence">
            Protein sequence
            <textarea
              id="sequence"
              className="mono sequence-input"
              rows={7}
              placeholder="Paste the amino acid sequence"
              required
              minLength={10}
              maxLength={10000}
              value={sequence}
              onChange={(event) => setSequence(event.target.value)}
              disabled={busy}
            />
          </label>
          <div className="field-hint">
            {sequence.replace(/\s/g, "").length.toLocaleString()} residues
            entered
          </div>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={explain}
              disabled={busy}
              onChange={(event) => setExplain(event.target.checked)}
            />
            Include SHAP explanation{" "}
            <span className="muted">Tree features or neural representation groups</span>
          </label>
          {error ? (
            <div className="notice warning" role="alert">
              {error}
            </div>
          ) : null}
          <div className="form-actions">
            <button
              className="button primary"
              disabled={busy || !model?.available}
            >
              {busy ? (
                <Loader2 size={16} className="spin" />
              ) : (
                <ArrowRight size={16} />
              )}{" "}
              {busy ? "Evaluating pair…" : "Run prediction"}
            </button>
            <span className="muted">Research prioritization only</span>
          </div>
        </form>
        <section
          className="panel result-panel"
          aria-live="polite"
          aria-label="Prediction result"
        >
          {result ? (
            <>
              <div className="panel-heading">
                <div>
                  <h2>Prediction result</h2>
                  <p>
                    {result.label.label_name} · {result.label.unit}
                  </p>
                </div>
                <span className="badge">Completed</span>
              </div>
              <div className="score-block">
                <span>Predicted {result.label.label_name}</span>
                <strong>{number(result.score)}</strong>
                <p>
                  {result.label.higher_is_stronger
                    ? "Higher scores indicate stronger predicted binding."
                    : "Lower scores indicate stronger predicted binding."}
                </p>
              </div>
              <dl className="result-metadata">
                <div>
                  <dt>Model</dt>
                  <dd>{result.model_id}</dd>
                </div>
                <div>
                  <dt>Dataset release</dt>
                  <dd>
                    {result.dataset} · {result.dataset_version}
                  </dd>
                </div>
                <div>
                  <dt>Protein length</dt>
                  <dd>{result.sequence_length.toLocaleString()} residues</dd>
                </div>
                <div>
                  <dt>Inference</dt>
                  <dd>
                    <Clock3 size={13} />
                    {(result.inference_time * 1000).toFixed(1)} ms
                  </dd>
                </div>
              </dl>
              {result.shap_values ? (
                <div className="explanation">
                  <h3>{result.shap_values.method || "SHAP feature contributions"}</h3>
                  <p>
                    Contributions describe model behavior, not biological
                    causality.
                  </p>
                  <p>{result.shap_values.scope}</p>
                  <p className="muted">Baseline: {result.shap_values.baseline}</p>
                  <p>Base score {number(result.shap_values.base_value)} + contributions = {number(result.shap_values.reconstructed_score)}. Reconstruction error: {number(result.shap_values.additivity_error, 7)}.</p>
                  {result.shap_values.top_features.map((feature) => (
                    <div key={feature.feature} className="feature-row">
                      <span className="mono">{feature.feature}</span>
                      <strong
                        className={
                          feature.shap_value >= 0 ? "text-success mono" : "mono"
                        }
                      >
                        {feature.shap_value > 0 ? "+" : ""}
                        {number(feature.shap_value)}
                      </strong>
                    </div>
                  ))}
                  {result.shap_values.other_features_contribution !== 0 ? <p>Other features: {number(result.shap_values.other_features_contribution)}</p> : null}
                  <ul>{result.shap_values.limitations?.map((text) => <li key={text}>{text}</li>)}</ul>
                </div>
              ) : null}
              {result.storage_warning ? <p role="alert" className="notice warning">{result.storage_warning}</p> : null}
              {result.history_id ? <p>Saved to your private prediction history.</p> : null}
              <button
                type="button"
                className="button secondary"
                onClick={() => {
                  const blob = new Blob(
                    [JSON.stringify({ ...result, smiles, sequence }, null, 2)],
                    { type: "application/json" },
                  );
                  const url = URL.createObjectURL(blob);
                  const link = document.createElement("a");
                  link.href = url;
                  link.download = "pharmalens-prediction.json";
                  link.click();
                  setTimeout(() => URL.revokeObjectURL(url), 1000);
                }}
              >
                <Download size={15} />
                Export prediction
              </button>
            </>
          ) : (
            <Empty
              title={
                busy ? "Evaluating interaction" : "Your result will appear here"
              }
            >
              {busy
                ? "Computing the score with the selected artifact."
                : "Select a model and provide a molecule and protein to begin."}
            </Empty>
          )}
        </section>
      </div>
    </>
  );
}
