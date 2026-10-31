"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  ArrowUpRight,
  Blocks,
  Database,
  FlaskConical,
  LayoutDashboard,
  RefreshCw,
  Rows3,
} from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  api,
  count,
  Dataset,
  familyNames,
  Model,
  number,
  Run,
  scopeKey,
  TrainingCheckpoint,
  MatrixStatus,
} from "@/lib/api";

type Resources = {
  models: Model[] | null;
  datasets: Dataset[] | null;
  runs: Run[] | null;
  checkpoints: TrainingCheckpoint[];
  matrices: MatrixStatus[];
  connected: boolean;
  loading: boolean;
  error: string;
  refresh: () => void;
};
const Context = createContext<Resources | null>(null);
export function useWorkspace() {
  const value = useContext(Context);
  if (!value) throw new Error("Workspace missing");
  return value;
}

const navigation = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/predict", label: "Predict interaction", icon: Activity },
  { href: "/batch", label: "Batch predictions", icon: Rows3 },
  { href: "/models", label: "Model registry", icon: Blocks },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
];

export function Workspace({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const controller = useRef<AbortController | null>(null);
  const [state, setState] = useState<Omit<Resources, "refresh">>({
    models: null,
    datasets: null,
    runs: null,
    checkpoints: [],
    matrices: [],
    connected: false,
    loading: true,
    error: "",
  });
  const refresh = useCallback(async () => {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setState((previous) => ({ ...previous, loading: true, error: "" }));
    const results = await Promise.allSettled([
      api<{ status: string }>("/health", { signal: current.signal }),
      api<{ models: Model[] }>("/models/list", { signal: current.signal }),
      api<{ datasets: Dataset[] }>("/datasets", { signal: current.signal }),
      api<{
        runs: Run[];
        checkpoints: TrainingCheckpoint[];
        matrices: MatrixStatus[];
      }>("/experiments", { signal: current.signal }),
    ]);
    if (current.signal.aborted) return;
    setState((previous) => ({
      connected: results[0].status === "fulfilled",
      loading: false,
      models:
        results[1].status === "fulfilled"
          ? results[1].value.models
          : previous.models,
      datasets:
        results[2].status === "fulfilled"
          ? results[2].value.datasets
          : previous.datasets,
      runs:
        results[3].status === "fulfilled"
          ? results[3].value.runs
          : previous.runs,
      checkpoints:
        results[3].status === "fulfilled"
          ? results[3].value.checkpoints || []
          : previous.checkpoints,
      matrices:
        results[3].status === "fulfilled"
          ? results[3].value.matrices || []
          : previous.matrices,
      error: results.some((result) => result.status === "rejected")
        ? "Some workspace data could not be loaded. Check the API connection and refresh."
        : "",
    }));
  }, []);
  useEffect(() => {
    void refresh();
    return () => controller.current?.abort();
  }, [refresh]);
  const current =
    navigation.find((item) => item.href === path)?.label || "Workspace";
  return (
    <Context.Provider value={{ ...state, refresh }}>
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <header className="sidebar">
        <Link href="/" className="brand">
          <span className="brand-symbol">
            PL<span className="brand-cross">+</span>
          </span>
          <span>
            PharmaLens<small>DRUG–TARGET INTELLIGENCE</small>
          </span>
        </Link>
        <div className="workspace-label">
          RESEARCH
          <br />
          WORKSPACE / 0.2
        </div>
        <nav aria-label="Main navigation">
          {navigation.map((item, index) => (
            <Link
              key={item.href}
              href={item.href}
              className={path === item.href ? "nav-link active" : "nav-link"}
              aria-current={path === item.href ? "page" : undefined}
            >
              <span className="nav-index">0{index + 1}</span>
              {item.label}
            </Link>
          ))}
        </nav>
      </header>
      <div className="main-shell">
        <header className="topbar">
          <div>
            <span className="topbar-label">LAB NOTEBOOK</span>
            <span className="divider">/</span>
            {current}
          </div>
          <div className="topbar-actions">
            <span className="connection">
              <i className={state.connected ? "dot online" : "dot"} />
              {state.connected
                ? "API connected"
                : state.loading
                  ? "Connecting"
                  : "API unavailable"}
            </span>
            <button
              className="icon-button"
              aria-label="Refresh workspace data"
              onClick={() => void refresh()}
              disabled={state.loading}
            >
              <RefreshCw size={16} className={state.loading ? "spin" : ""} />
            </button>
          </div>
        </header>
        <main id="main">
          {state.error ? (
            <div className="notice warning" role="alert">
              {state.error}
            </div>
          ) : null}
          {children}
        </main>
        <footer className="page-footer">
          <span>
            PHARMALENS <span className="divider">/</span> Research workspace
          </span>
          <span>
            Research prioritization only · Predictions require experimental
            confirmation.
          </span>
        </footer>
      </div>
    </Context.Provider>
  );
}

export function PageHeading({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <p className="eyebrow">PHARMALENS / RESEARCH</p>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}
export function Empty({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="empty">
      <Database size={24} />
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
export function ModelPicker({
  models,
  value,
  onChange,
  disabled = false,
}: {
  models: Model[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  const [chosenScope, setChosenScope] = useState("");
  const selected = models.find((model) => model.name === value);
  const groups = [...new Set(models.map(scopeKey))].sort();
  const current =
    (selected ? scopeKey(selected) : chosenScope || groups[0]) || "";
  const compatible = models.filter((model) => scopeKey(model) === current);
  useEffect(() => {
    if (disabled || value || !current) return;
    const available = models.filter(
      (model) => scopeKey(model) === current && model.available,
    );
    const preferred = available.find((model) => model.is_best) || available[0];
    if (preferred) onChange(preferred.name);
  }, [models, current, disabled, value, onChange]);
  return (
    <div className="form-grid">
      <label>
        Dataset · release · endpoint
        <select
          value={current}
          disabled={disabled || !groups.length}
          onChange={(event) => {
            setChosenScope(event.target.value);
            onChange("");
          }}
        >
          {groups.length ? (
            groups.map((group) => <option key={group}>{group}</option>)
          ) : (
            <option>No registered tasks</option>
          )}
        </select>
      </label>
      <label>
        Model
        <select
          value={value}
          disabled={disabled || !compatible.length}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">
            {compatible.some((model) => model.available)
              ? "Use task default"
              : "No available models"}
          </option>
          {compatible.map((model) => (
            <option
              key={model.name}
              value={model.name}
              disabled={!model.available}
            >
              {model.name}
              {model.is_best ? " · default" : ""}
              {!model.available ? " · artifact unavailable" : ""}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

export function Overview() {
  const { models, datasets, runs, loading } = useWorkspace();
  const panel = datasets
    ?.filter(
      (dataset) =>
        dataset.non_kinase_only &&
        dataset.endpoints.length === 1 &&
        dataset.endpoints.some(
          (endpoint) =>
            endpoint.endpoint === "Ki" &&
            endpoint.available &&
            endpoint.rows > 0,
        ),
    )
    .sort((left, right) => right.endpoints[0].rows - left.endpoints[0].rows)[0];
  const completed = runs?.filter((run) => !run.smoke);
  const families = Object.entries(panel?.family_counts || {}).filter(
    ([key]) => key !== "unclassified",
  );
  const maximum = Math.max(1, ...families.map(([, value]) => value));
  const metrics = [
    [
      "Prepared datasets",
      datasets
        ? count(
            datasets.filter(
              (d) =>
                !d.source_sample &&
                d.endpoints.some((e) => e.available && e.rows > 0),
            ).length,
          )
        : "—",
      "Versioned & traceable",
    ],
    [
      "Available models",
      models ? count(models.filter((m) => m.available).length) : "—",
      "Ready for inference",
    ],
    [
      "Validation runs",
      completed ? count(completed.length) : "—",
      "Smoke checks excluded",
    ],
  ];
  return (
    <div className="research-overview">
      <div className="overview-title">
        <div>
          <p className="eyebrow">THE RESEARCH DESK / OVERVIEW</p>
          <h1>
            A clearer view of
            <br />
            <em>what comes next.</em>
          </h1>
        </div>
        <p className="overview-intro">
          From molecular questions to measurable evidence. Explore your data,
          evaluate models, and prioritize the next interaction.
        </p>
      </div>

      <div className="desk-grid">
        <section className="research-launch" aria-labelledby="launch-title">
          <div className="launch-top">
            <span className="eyebrow">01 / INVESTIGATE</span>
            <FlaskConical size={21} strokeWidth={1.3} />
          </div>
          <h2 id="launch-title">
            Every discovery starts
            <br />
            with an interaction.
          </h2>
          <p>
            Bring a molecule and a protein sequence.
            <br />
            Find an affinity estimate with traceable model evidence.
          </p>
          <Link className="button primary launch-button" href="/predict">
            Start a prediction <ArrowUpRight size={17} />
          </Link>
          <div className="launch-path">
            <span>SMILES</span>
            <span aria-hidden="true">+</span>
            <span>Protein sequence</span>
            <span aria-hidden="true">→</span>
            <span>Affinity estimate</span>
          </div>
        </section>
        <section className="workspace-ledger" aria-labelledby="ledger-title">
          <div className="ledger-heading">
            <h2 id="ledger-title">Workspace at a glance</h2>
            <span className="mono">LIVE CATALOG</span>
          </div>
          {metrics.map(([label, value, note]) => (
            <div className="ledger-row" key={label}>
              <div>
                <span>{label}</span>
                <small>{note}</small>
              </div>
              <strong>{value}</strong>
            </div>
          ))}
          <Link href="/datasets" className="text-link">
            Explore the catalog <ArrowUpRight size={14} />
          </Link>
        </section>
      </div>

      <nav className="workflow-strip" aria-label="Research actions">
        {[
          [
            "02",
            "Screen a collection",
            "Run and recover CSV batch jobs",
            "/batch",
          ],
          [
            "03",
            "Inspect the models",
            "Compare compatible artifacts",
            "/models",
          ],
          [
            "04",
            "Follow the evidence",
            "Review experiments and held-out results",
            "/experiments",
          ],
        ].map(([index, title, description, href]) => (
          <Link href={href} key={href}>
            <span className="workflow-number">{index}</span>
            <div>
              <strong>{title}</strong>
              <small>{description}</small>
            </div>
            <ArrowUpRight size={18} />
          </Link>
        ))}
      </nav>

      <div className="evidence-grid">
        <section className="coverage-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">THE DATA LANDSCAPE</p>
              <h2>Beyond kinase targets</h2>
            </div>
            <span className="badge neutral">{families.length} families</span>
          </div>
          <div className="coverage-summary">
            <strong>
              {panel
                ? count(panel.endpoints.find((e) => e.endpoint === "Ki")?.rows)
                : "—"}
            </strong>
            <span>
              Ki observations
              <br />
              in the reference panel
            </span>
          </div>
          {families.length ? (
            <div className="family-chart">
              {families
                .toSorted((a, b) => b[1] - a[1])
                .map(([family, rows]) => (
                  <div className="family-row" key={family}>
                    <span>{familyNames[family] || family}</span>
                    <div className="bar-track">
                      <div style={{ width: `${(rows / maximum) * 100}%` }} />
                    </div>
                    <strong>{count(rows)}</strong>
                  </div>
                ))}
            </div>
          ) : (
            <Empty
              title={loading ? "Loading coverage" : "No verified family panel"}
            >
              Prepared, family-annotated datasets will appear here.
            </Empty>
          )}
          <p className="coverage-note">
            Family views may share molecules, targets, and source studies.
            Coverage does not imply independent validation.
          </p>
          <Link href="/datasets" className="text-link">
            Inspect data provenance <ArrowUpRight size={14} />
          </Link>
        </section>
        <section className="validation-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">THE EXPERIMENT LOG</p>
              <h2>Recent validation runs</h2>
            </div>
            <Link href="/experiments" className="text-link">
              All records <ArrowUpRight size={14} />
            </Link>
          </div>
          <p className="section-description">
            Results belong to their stated task and split protocol.
          </p>
          {completed?.length ? (
            <RunTable runs={completed.slice(0, 5)} />
          ) : (
            <Empty
              title={
                loading ? "Loading experiments" : "No completed experiments"
              }
            >
              Completed, versioned experiments will appear here with their
              validation evidence.
            </Empty>
          )}
          <div className="evidence-note">
            <span className="evidence-mark">i</span>
            <p>
              <strong>Evidence before selection.</strong> Compare within one
              dataset, release, and endpoint. Validation guides selection;
              held-out tests remain separate.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}

export function RunTable({ runs }: { runs: Run[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Experiment</th>
            <th>Task</th>
            <th>Split · seed</th>
            <th className="numeric">Val. RMSE ↓</th>
            <th className="numeric">Val. MAE ↓</th>
            <th className="numeric">Test RMSE ↓</th>
            <th>Checks</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id}>
              <td>
                <strong>{run.architecture.toUpperCase()}</strong>
                <small className="mono">{run.id}</small>
              </td>
              <td>
                {run.dataset}
                <small>
                  {run.endpoint} · {run.dataset_version}
                </small>
              </td>
              <td>
                {run.split}
                <small>Seed {run.seed}</small>
              </td>
              <td className="numeric mono">
                {number(run.validation_metrics.rmse)}
              </td>
              <td className="numeric mono">
                {number(run.validation_metrics.mae)}
              </td>
              <td className="numeric mono">{number(run.test_metrics?.rmse)}</td>
              <td>
                <span className={run.smoke ? "badge neutral" : "badge"}>
                  {run.smoke ? "Smoke only" : "Validation"}
                </span>
                <small>
                  {run.serving_parity
                    ? "Reload verified"
                    : "Reload not verified"}
                </small>
                {run.test_evaluated ? (
                  <small>Held-out test recorded</small>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Experiments() {
  const { runs, loading, checkpoints, matrices, refresh } = useWorkspace();
  useEffect(() => {
    const timer = window.setInterval(refresh, 15000);
    return () => window.clearInterval(timer);
  }, [refresh]);
  const [includeSmoke, setIncludeSmoke] = useState(false);
  const [dataset, setDataset] = useState("");
  const visible =
    runs?.filter(
      (run) =>
        (includeSmoke || !run.smoke) && (!dataset || run.dataset === dataset),
    ) || [];
  return (
    <>
      <PageHeading
        title="Experiment records"
        description="Inspect validation results, split protocols, and serving checks."
      />
      {matrices.length ? (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Experiment matrices</h2>
              <p>Saved execution state · refreshes every 15 seconds</p>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Plan</th>
                  <th>Completed</th>
                  <th>Failed</th>
                  <th>Registration</th>
                </tr>
              </thead>
              <tbody>
                {matrices.map((matrix) => (
                  <tr key={matrix.id}>
                    <td className="mono">{matrix.id}</td>
                    <td>
                      {matrix.completed_jobs} / {matrix.total_jobs}
                    </td>
                    <td>{matrix.failed_jobs}</td>
                    <td>
                      {matrix.registration_errors
                        ? `${matrix.registration_errors} issues to review`
                        : "No recorded issues"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
      {checkpoints.length ? (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Latest training checkpoints</h2>
              <p>Intermediate losses are not completed validation results.</p>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Experiment</th>
                  <th>Epoch</th>
                  <th>Validation loss</th>
                  <th>Last update</th>
                </tr>
              </thead>
              <tbody>
                {checkpoints.map((checkpoint) => (
                  <tr key={checkpoint.id}>
                    <td>
                      <strong className="mono">{checkpoint.id}</strong>
                      <small>
                        {checkpoint.reported_status}
                        {checkpoint.stale
                          ? " · stale record"
                          : " · recently updated"}
                      </small>
                    </td>
                    <td>
                      {checkpoint.epoch} / {checkpoint.max_epochs ?? "—"}
                    </td>
                    <td className="mono">
                      {number(checkpoint.validation_loss)}
                      <small>
                        {checkpoint.loss_space === "normalized_target"
                          ? "Normalized target MSE"
                          : checkpoint.loss_space}
                      </small>
                    </td>
                    <td>{Math.floor(checkpoint.age_seconds / 60)} min ago</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
      <div className="toolbar">
        <label className="inline-label">
          Dataset
          <select
            value={dataset}
            onChange={(event) => setDataset(event.target.value)}
          >
            <option value="">All datasets</option>
            {[...new Set(runs?.map((run) => run.dataset))].map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={includeSmoke}
            onChange={(event) => setIncludeSmoke(event.target.checked)}
          />
          Include smoke checks
        </label>
        <span className="muted">{visible.length} records</span>
      </div>
      <section className="panel">
        {visible.length ? (
          <RunTable runs={visible} />
        ) : (
          <Empty title={loading ? "Loading records" : "No matching runs"}>
            Completed experiments appear after their artifact and validation
            evidence are saved.
          </Empty>
        )}
      </section>
      <div className="notice">
        Smoke checks verify the training path. Their scores are excluded from
        model selection. Results from different tasks or split protocols are not
        a shared leaderboard.
      </div>
    </>
  );
}

export function Datasets() {
  const { datasets, loading } = useWorkspace();
  return (
    <>
      <PageHeading
        title="Dataset catalog"
        description="Prepared versions, verified file integrity, and biological coverage."
      />
      <section className="panel">
        {datasets?.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Dataset version</th>
                  <th>Endpoints · observations</th>
                  <th>Family coverage</th>
                  <th>State</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((dataset) => (
                  <tr key={dataset.id}>
                    <td>
                      <strong className="mono">{dataset.id}</strong>
                      <small>
                        {dataset.non_kinase_only
                          ? "Verified non-kinase subset"
                          : "Prepared source dataset"}
                      </small>
                    </td>
                    <td>
                      {dataset.endpoints.map((endpoint) => (
                        <div key={endpoint.endpoint}>
                          {endpoint.endpoint}{" "}
                          <strong>{count(endpoint.rows)}</strong>
                          <small
                            className={
                              endpoint.available
                                ? "text-success"
                                : "text-danger"
                            }
                          >
                            {endpoint.available
                              ? "Checksum verified"
                              : "Artifact unavailable or changed"}
                          </small>
                        </div>
                      ))}
                    </td>
                    <td>
                      {Object.entries(dataset.family_counts)
                        .filter(([, rows]) => rows > 0)
                        .map(([family]) => familyNames[family] || family)
                        .join(", ") || "Not annotated"}
                    </td>
                    <td>
                      <span
                        className={
                          dataset.source_sample ? "badge neutral" : "badge"
                        }
                      >
                        {dataset.source_sample ? "Sample" : "Prepared"}
                      </span>
                      {dataset.unique_pairs ? (
                        <small>
                          {count(dataset.unique_pairs)} unique pairs
                        </small>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty title={loading ? "Loading catalog" : "No prepared datasets"}>
            Acquisition preserves raw source files. Preparation publishes
            versioned, endpoint-specific datasets.
          </Empty>
        )}
      </section>
      <div className="notice">
        Shared source observations are not independent evidence. Unknown protein
        families remain unclassified, and Ki and Kd labels stay separate.
      </div>
    </>
  );
}

export function Models() {
  const { models, loading } = useWorkspace();
  const [chosen, setChosen] = useState("");
  const tasks = [...new Set(models?.map(scopeKey))].sort();
  const current = chosen || tasks[0] || "";
  const visible =
    models
      ?.filter((model) => scopeKey(model) === current)
      .toSorted(
        (a, b) =>
          (a.validation_metrics.rmse ?? Infinity) -
          (b.validation_metrics.rmse ?? Infinity),
      ) || [];
  return (
    <>
      <PageHeading
        title="Model registry"
        description="Choose among registered artifacts within a compatible prediction task."
      />
      {tasks.length ? (
        <div className="toolbar">
          <label className="inline-label">
            Prediction task
            <select
              value={current}
              onChange={(event) => setChosen(event.target.value)}
            >
              {tasks.map((task) => (
                <option key={task}>{task}</option>
              ))}
            </select>
          </label>
        </div>
      ) : null}
      <section className="panel">
        {visible.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Architecture</th>
                  <th className="numeric">Val. RMSE ↓</th>
                  <th className="numeric">Val. MAE ↓</th>
                  <th>Artifact</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((model) => (
                  <tr key={model.name}>
                    <td>
                      <strong>{model.name}</strong>
                      {model.is_best ? (
                        <span className="badge">Default</span>
                      ) : null}
                      <small>{model.description}</small>
                    </td>
                    <td>
                      {model.architecture || model.model_type.toUpperCase()}
                    </td>
                    <td className="numeric mono">
                      {number(model.validation_metrics.rmse)}
                    </td>
                    <td className="numeric mono">
                      {number(model.validation_metrics.mae)}
                    </td>
                    <td>
                      <span
                        className={model.available ? "badge" : "badge neutral"}
                      >
                        {model.available ? "Available" : "Unavailable"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty title={loading ? "Loading models" : "No registered artifacts"}>
            Train and verify an artifact, then register it with its dataset,
            output scale, and validation evidence. Historical experiment rows
            are not automatically runnable models.
          </Empty>
        )}
      </section>
      <div className="notice">
        Model files are verified when loaded. A default is scoped to its
        dataset, release, and endpoint; it is not a claim of universal
        superiority.
      </div>
    </>
  );
}
