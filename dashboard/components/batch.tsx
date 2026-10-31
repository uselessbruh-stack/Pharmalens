"use client";

import { useEffect, useState } from "react";
import Papa from "papaparse";
import { Download, FileUp, Loader2, Play } from "lucide-react";
import { api, Model, number, scope } from "@/lib/api";
import {
  Empty,
  ModelPicker,
  PageHeading,
  useWorkspace,
} from "@/components/workspace";

type Pair = { smiles: string; sequence: string; id?: string };
type BatchRow = { row_id: number; score: number | null; error?: string };
type BatchResponse = {
  results: BatchRow[];
  total: number;
  model_id: string;
  dataset_version: string;
  dataset: string;
  artifact_sha256: string;
  inputs: Pair[];
  label: { label_name: string; unit: string };
};
type Job = {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  completed_rows: number;
  total_rows: number;
  model_id: string;
  cancel_requested: boolean;
  error: string | null;
  attempts: number;
};
const terminal = (job: Job) =>
  ["completed", "failed", "cancelled"].includes(job.status);
function remember(id: string) {
  try {
    localStorage.setItem("pharmalens.lastBatchJob", id);
  } catch {
    /* Server history still preserves the job. */
  }
}
function download(text: string, filename: string) {
  const url = URL.createObjectURL(
    new Blob([text], { type: "text/csv;charset=utf-8" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function BatchPredictions() {
  const { models } = useWorkspace();
  const [modelId, setModelId] = useState("");
  const [rows, setRows] = useState<Pair[]>([]);
  const [filename, setFilename] = useState("");
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [page, setPage] = useState(0);
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [recent, setRecent] = useState<Job[]>([]);
  const [pollVersion, setPollVersion] = useState(0);
  const model = models?.find((candidate) => candidate.name === modelId);
  useEffect(() => {
    try {
      const saved = localStorage.getItem("pharmalens.lastBatchJob");
      if (saved && /^[a-f0-9]{32}$/.test(saved)) setJobId(saved);
    } catch {
      /* Saved server jobs remain available below. */
    }
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    api<{ jobs: Job[] }>("/jobs?limit=10", { signal: controller.signal })
      .then((data) => setRecent(data.jobs))
      .catch(() => {});
    return () => controller.abort();
  }, [job?.status]);
  useEffect(() => {
    if (!jobId) return;
    const controller = new AbortController();
    let timer: number | undefined;
    setBusy(true);
    async function poll() {
      try {
        const current = await api<Job>(`/jobs/${encodeURIComponent(jobId)}`, {
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        setJob(current);
        if (current.status === "completed") {
          const data = await api<BatchResponse>(
            `/jobs/${encodeURIComponent(jobId)}/result`,
            { signal: controller.signal },
          );
          if (controller.signal.aborted) return;
          if (
            !Array.isArray(data.inputs) ||
            data.results.length !== data.inputs.length
          )
            throw new Error("The saved batch result is incomplete.");
          setRows(data.inputs);
          setFilename("Saved batch inputs");
          setModelId(data.model_id);
          setResult(data);
          setPage(0);
          setBusy(false);
          setError("");
        } else if (terminal(current)) {
          setBusy(false);
          setError(
            current.status === "cancelled"
              ? "Batch cancelled. You can retry the saved inputs."
              : current.error || "Batch failed. Review the model and retry.",
          );
        } else {
          timer = window.setTimeout(() => void poll(), 1500);
        }
      } catch (issue) {
        if (!controller.signal.aborted) {
          setBusy(false);
          setError(
            issue instanceof Error
              ? issue.message
              : "Job updates are unavailable. The saved job may still be processing.",
          );
        }
      }
    }
    void poll();
    return () => {
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [jobId, pollVersion]);
  async function readFile(file: File | undefined) {
    if (!file) return;
    setRows([]);
    setJobId("");
    setJob(null);
    setResult(null);
    setPage(0);
    setError("");
    setFilename(file.name);
    if (file.size > 2 * 1024 * 1024) {
      setError(
        "Upload a CSV smaller than 2 MB. Split larger files into batches.",
      );
      return;
    }
    try {
      const parsed = Papa.parse<Pair>(await file.text(), {
        header: true,
        skipEmptyLines: "greedy",
        transformHeader: (header) => header.trim().toLowerCase(),
      });
      if (parsed.errors.length)
        throw new Error(`CSV could not be parsed: ${parsed.errors[0].message}`);
      const fields = parsed.meta.fields || [];
      if (!fields.includes("smiles") || !fields.includes("sequence"))
        throw new Error(
          "The CSV needs smiles and sequence columns. An id column is optional.",
        );
      if (fields.some((field) => !["smiles", "sequence", "id"].includes(field)))
        throw new Error(
          "Use only smiles, sequence, and optional id columns. Check for duplicate headers.",
        );
      if (!parsed.data.length || parsed.data.length > 1000)
        throw new Error("Each batch must contain between 1 and 1,000 pairs.");
      if (
        parsed.data.some((row) => !row.smiles?.trim() || !row.sequence?.trim())
      )
        throw new Error("Every row needs a molecule and protein sequence.");
      if (parsed.data.some((row) => (row.id?.length || 0) > 200))
        throw new Error("Identifiers must be at most 200 characters.");
      setRows(
        parsed.data.map((row) => ({
          smiles: row.smiles.trim(),
          sequence: row.sequence.trim(),
          id: row.id,
        })),
      );
    } catch (issue) {
      setError(
        issue instanceof Error ? issue.message : "File could not be read.",
      );
    }
  }
  async function run(selected: Model) {
    setBusy(true);
    setError("");
    setResult(null);
    setPage(0);
    setJobId("");
    setJob(null);
    try {
      const created = await api<Job>("/jobs", {
        method: "POST",
        body: JSON.stringify({
          ...scope(selected),
          pairs: rows.map(({ smiles, sequence }) => ({ smiles, sequence })),
          ids: rows.map((row) => row.id || ""),
          idempotency_key: crypto.randomUUID(),
        }),
      });
      remember(created.id);
      setJob(created);
      setJobId(created.id);
    } catch (issue) {
      setError(
        issue instanceof Error ? issue.message : "Batch prediction failed.",
      );
      setBusy(false);
    }
  }
  async function cancelJob() {
    if (!jobId) return;
    try {
      setJob(await api<Job>(`/jobs/${jobId}/cancel`, { method: "POST" }));
      setPollVersion((value) => value + 1);
    } catch (issue) {
      setError(issue instanceof Error ? issue.message : "Cancellation failed.");
    }
  }
  async function retryJob() {
    if (!jobId) return;
    setBusy(true);
    try {
      const created = await api<Job>(`/jobs/${jobId}/retry`, {
        method: "POST",
      });
      remember(created.id);
      setError("");
      setResult(null);
      setJob(created);
      setJobId(created.id);
    } catch (issue) {
      setError(issue instanceof Error ? issue.message : "Retry failed.");
      setBusy(false);
    }
  }
  const start = page * 25;
  const display = result
    ? result.results.slice(start, start + 25)
    : rows.slice(start, start + 25);
  const failures =
    result?.results.filter((row) => row.score === null).length || 0;
  return (
    <>
      <PageHeading
        title="Batch predictions"
        description="Upload paired molecule–protein inputs, inspect each result, and export the full record."
      />
      <section className="panel form-panel">
        <div className="panel-heading">
          <div>
            <h2>Prepare a batch</h2>
            <p>
              CSV with smiles, sequence, and optional id columns · up to 1,000
              pairs
            </p>
          </div>
          <button
            className="button secondary"
            onClick={() =>
              download("id,smiles,sequence\n", "pharmalens-batch-template.csv")
            }
          >
            <Download size={14} />
            CSV template
          </button>
        </div>
        <ModelPicker
          models={models || []}
          value={modelId}
          onChange={(value) => {
            setModelId(value);
            setResult(null);
            setJobId("");
            setJob(null);
          }}
          disabled={busy}
        />
        <label className="upload-zone">
          <FileUp size={22} />
          <strong>{filename || "Choose an input CSV"}</strong>
          <span>Paired rows · maximum file size 2 MB</span>
          <input
            aria-label="Upload molecule protein CSV"
            type="file"
            accept=".csv,text/csv"
            disabled={busy}
            onChange={(event) => void readFile(event.target.files?.[0])}
          />
        </label>
        {error ? (
          <div role="alert" className="notice warning">
            {error}
          </div>
        ) : null}
        <div className="form-actions">
          <button
            className="button primary"
            disabled={busy || !model?.available || !rows.length}
            onClick={() => model && void run(model)}
          >
            {busy ? <Loader2 size={15} className="spin" /> : <Play size={15} />}{" "}
            {busy ? "Processing saved batch…" : "Run batch"}
          </button>
          {job && !terminal(job) ? (
            <button
              className="button secondary"
              onClick={() => void cancelJob()}
              disabled={job.cancel_requested}
            >
              Cancel batch
            </button>
          ) : null}
          {job &&
          ["failed", "cancelled"].includes(job.status) &&
          job.attempts < 3 ? (
            <button
              className="button secondary"
              onClick={() => void retryJob()}
              disabled={busy}
            >
              Retry saved batch
            </button>
          ) : null}
          {job && !terminal(job) && !busy ? (
            <button
              className="button secondary"
              onClick={() => setPollVersion((value) => value + 1)}
            >
              Resume updates
            </button>
          ) : null}
          <span className="muted">
            {rows.length.toLocaleString()} input pairs
          </span>
        </div>
        {job ? (
          <div role="status" className="notice">
            Batch {job.id.slice(0, 8)} ·{" "}
            {job.cancel_requested && !terminal(job)
              ? "Cancellation requested"
              : job.status}{" "}
            · {job.completed_rows} / {job.total_rows} rows processed. Saved jobs
            continue when you leave this page.
          </div>
        ) : null}
      </section>
      {recent.length ? (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Recent batches</h2>
              <p>Open saved progress or restore a completed result.</p>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Batch</th>
                  <th>Model</th>
                  <th>Status</th>
                  <th>Rows</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {recent.map((saved) => (
                  <tr key={saved.id}>
                    <td className="mono">{saved.id.slice(0, 8)}</td>
                    <td>{saved.model_id}</td>
                    <td>{saved.status}</td>
                    <td>
                      {saved.completed_rows} / {saved.total_rows}
                    </td>
                    <td>
                      <button
                        className="button secondary"
                        onClick={() => {
                          remember(saved.id);
                          setJobId(saved.id);
                          setJob(saved);
                          setResult(null);
                          setRows([]);
                          setError("");
                          setPollVersion((value) => value + 1);
                        }}
                      >
                        Open batch
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>{result ? "Batch results" : "Input preview"}</h2>
            <p>
              {result
                ? `${result.total - failures} successful · ${failures} failed · ${result.label.label_name} · ${result.model_id}`
                : "Verify the paired rows before submitting."}
            </p>
          </div>
          {result ? (
            <button
              className="button secondary"
              onClick={() =>
                download(
                  Papa.unparse(
                    result.results.map((row) => ({
                      input_row: row.row_id + 1,
                      id: rows[row.row_id]?.id || "",
                      smiles: rows[row.row_id]?.smiles,
                      sequence: rows[row.row_id]?.sequence,
                      score: row.score,
                      error: row.error || "",
                      model_id: result.model_id,
                      dataset_version: result.dataset_version,
                      dataset: result.dataset,
                      artifact_sha256: result.artifact_sha256,
                      label: result.label.label_name,
                      unit: result.label.unit,
                    })),
                    { escapeFormulae: true },
                  ),
                  "pharmalens-batch-results.csv",
                )
              }
            >
              <Download size={14} />
              Export all rows
            </button>
          ) : null}
        </div>
        {display.length ? (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Row</th>
                    <th>Identifier</th>
                    <th>Molecule</th>
                    <th>Protein</th>
                    <th>{result ? "Prediction" : "State"}</th>
                  </tr>
                </thead>
                <tbody>
                  {display.map((item, index) => {
                    const record = result ? (item as BatchRow) : null;
                    const position = record?.row_id ?? start + index;
                    const input = rows[position];
                    return (
                      <tr key={position}>
                        <td className="mono">{position + 1}</td>
                        <td>{input?.id || "—"}</td>
                        <td className="mono truncate" title={input?.smiles}>
                          {input?.smiles}
                        </td>
                        <td>{input?.sequence.length} residues</td>
                        <td>
                          {record ? (
                            record.score === null ? (
                              <span className="text-danger">
                                {record.error || "Failed"}
                              </span>
                            ) : (
                              <span className="mono">
                                {number(record.score)}
                              </span>
                            )
                          ) : (
                            <span className="badge neutral">Pending</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <span>
                {start + 1}–{Math.min(start + 25, rows.length)} of {rows.length}
              </span>
              <div>
                <button
                  className="button secondary"
                  disabled={!page}
                  onClick={() => setPage(page - 1)}
                >
                  Previous
                </button>
                <button
                  className="button secondary"
                  disabled={start + 25 >= rows.length}
                  onClick={() => setPage(page + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        ) : (
          <Empty title="No input pairs yet">
            Upload the CSV template filled with your molecule–protein pairs.
          </Empty>
        )}
      </section>
      <div className="notice">
        Rows with invalid chemical or protein inputs are reported individually.
        Batch results retain the selected model and output scale.
      </div>
    </>
  );
}
