import { authorizationHeaders } from "@/lib/firebase";

export type Model = {
  name: string;
  model_type: string;
  architecture?: string;
  dataset: string;
  dataset_version: string;
  endpoint: string;
  is_best: boolean;
  available: boolean;
  description: string;
  validation_metrics: Record<string, number | null>;
};
export type Dataset = {
  id: string;
  endpoints: { endpoint: string; rows: number; available: boolean }[];
  family_counts: Record<string, number>;
  source_sample: boolean;
  non_kinase_only: boolean;
  unique_pairs: number | null;
  limitations: string[];
};
export type Run = {
  id: string;
  model: string;
  architecture: string;
  dataset: string;
  dataset_version: string;
  endpoint: string;
  split: string;
  seed: number;
  smoke: boolean;
  completed_at: string;
  validation_metrics: Record<string, number | null>;
  test_evaluated: boolean;
  test_metrics?: Record<string, number | null> | null;
  seconds: number;
  serving_parity: boolean;
  iteration_budget_reached: boolean;
};
export type TrainingCheckpoint = {
  id: string;
  epoch: number;
  max_epochs: number | null;
  reported_status: string;
  age_seconds: number;
  stale: boolean;
  validation_loss: number | null;
  loss_space: string;
};
export type MatrixStatus = {
  id: string;
  total_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  registration_errors: number;
  updated_at: string | null;
};
export type Prediction = {
  score: number;
  model_id: string;
  dataset: string;
  dataset_version: string;
  label: {
    label_name: string;
    unit: string;
    endpoint: string;
    higher_is_stronger: boolean;
  };
  sequence_length: number;
  inference_time: number;
  history_id?: string | null;
  storage_warning?: string | null;
  shap_values?: {
    base_value: number;
    other_features_contribution: number;
    method?: string;
    scope?: string;
    baseline?: string;
    reconstructed_score?: number;
    additivity_error?: number;
    limitations?: string[];
    top_features: { feature: string; shap_value: number }[];
  };
};
export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`/api${path}`, {
    cache: "no-store",
    ...options,
    headers: { "Content-Type": "application/json", ...await authorizationHeaders(), ...options.headers },
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail
              .map((issue: { msg?: string }) => issue.msg || "Invalid input")
              .join("; ")
          : "The service could not complete this request.";
    throw new Error(message);
  }
  return data as T;
}
export const scope = (model: Model) => ({
  model_id: model.name,
  dataset: model.dataset,
  dataset_version: model.dataset_version,
  endpoint: model.endpoint,
});
export const scopeKey = (model: Model) =>
  [model.dataset, model.dataset_version, model.endpoint].join(" · ");
export const number = (value: number | null | undefined, digits = 4) =>
  value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
export const count = (value: number | undefined) =>
  value == null ? "—" : value.toLocaleString();
export const familyNames: Record<string, string> = {
  gpcr: "GPCRs",
  protease: "Proteases",
  nuclear_receptor: "Nuclear receptors",
  ion_channel: "Ion channels",
  transporter: "Transporters",
  other_enzyme: "Other enzymes",
  unclassified: "Unclassified",
};
