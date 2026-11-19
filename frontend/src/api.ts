// Typed client for the FastAPI backend. Types mirror backend/app/routers/*.py.

export type User = { id: number; email: string; role: "analyst" | "reviewer" | "admin" };
export type Prediction = { score: number; threshold: number; review_recommended: boolean; model: string };
export type Anomaly = { score: number; flagged: boolean; model: string };
export type Claim = {
  claim_id: string; customer_id: string; region: string | null; policy_start_date: string; claim_date: string;
  claim_category: string; claim_amount: number; document_count: number | null; manual_review_required: number | null;
  prediction: Prediction | null; anomaly: Anomaly | null;
};
export type ClaimDetail = Claim & { features: Record<string, number | string | null> | null; history: Claim[] };
export type Contribution = { feature: string; label: string; value: number | string | null; contribution: number };
export type Explanation = {
  claim_id: string; model: string; score: number; threshold: number; review_recommended: boolean;
  base_value: number; risk_log_odds: number; contributions: Contribution[]; note: string;
};
export type Drift = {
  checked: boolean; reason?: string; model?: string; rows?: number; psi?: Record<string, number>;
  levels?: Record<string, "stable" | "moderate" | "large">; document_count_missing?: { reference: number; batch: number };
  worst?: number; warnings?: string[];
};
export type ImportResult = {
  import_id: number; rows_total: number; rows_imported: number; rows_rejected: number;
  errors: string[]; ignored_columns: string[]; drift: Drift | null;
};
export type Stats = {
  model: string | null; anomaly_model: string | null; threshold: number | null;
  claims: number; scored: number; review_recommended: number; unusual: number;
  histogram: { from: number; to: number; count: number }[];
  by_category: { category: string; claims: number; recommended_rate: number; unusual_rate: number; historical_review_rate: number | null }[];
  recent_imports: { id: number; filename: string; created_at: string; rows_imported: number; rows_rejected: number;
    drift_checked: boolean; drift_worst: number | null; drift_warnings: string[] }[];
};
export type Metrics = Record<string, number | null>;
export type ModelVersion = {
  id: number; name: string; version: string; estimator: string; status: "candidate" | "active" | "retired";
  threshold: number; created_at: string; activated_at: string | null;
  validation: Metrics; test: Metrics; fairness_by_region_test: Record<string, Metrics>;
};
export type CaseStatus = "queued" | "investigating" | "awaiting_decision" | "closed" | "failed";
export type CaseSummary = {
  id: number; claim_id: string; status: CaseStatus; priority: "low" | "medium" | "high"; current_step: string | null;
  opened_by: string; decision: "escalate" | "clear" | null; created_at: string; updated_at: string;
};
export type Evidence = { id: string; tool: string; title: string; source: Record<string, string | number>; facts: { label: string; value: string | number }[] };
export type Finding = { text: string; evidence: string[]; problems: string[] };
export type CaseReport = {
  claim_id: string; summary: string; findings: Finding[]; reviewer_questions: string[]; unverified: Finding[];
  retries: number; limitations: string[];
};
export type CaseDetail = CaseSummary & {
  evidence: Evidence[] | null; report: CaseReport | null; error: string | null; decision_reason: string | null;
  decided_by: string | null; decided_at: string | null; can_decide: boolean;
  events: { step: string; message: string; created_at: string }[];
};
export type ClaimFilters = { search?: string; category?: string; recommended?: boolean; unusual?: boolean; sort: "score" | "date" | "amount"; offset: number };

export const CATEGORIES = ["vehicle", "home", "health", "travel", "device"];
export const PAGE_SIZE = 50;

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

// The login token lives in memory and sessionStorage (cleared when the tab closes).
let token: string | null = (() => { try { return sessionStorage.getItem("token"); } catch { return null; } })();
let onUnauthorized = () => {};

export function setToken(t: string | null) {
  token = t;
  try { t ? sessionStorage.setItem("token", t) : sessionStorage.removeItem("token"); } catch { /* storage blocked */ }
}
export const hasToken = () => token !== null;
export const setUnauthorizedHandler = (fn: () => void) => { onUnauthorized = fn; };

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(path, { ...init, headers });
  if (!res.ok) {
    if (res.status === 401 && token) { setToken(null); onUnauthorized(); }
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* not JSON */ }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export const api = {
  async login(email: string, password: string) {
    const r = await request<{ access_token: string }>("/api/auth/login", {
      method: "POST", body: new URLSearchParams({ username: email, password }),
    });
    setToken(r.access_token);
  },
  me: () => request<User>("/api/auth/me"),
  stats: () => request<Stats>("/api/stats"),
  claims(f: ClaimFilters) {
    const q = new URLSearchParams({ sort: f.sort, limit: String(PAGE_SIZE), offset: String(f.offset) });
    if (f.category) q.set("category", f.category);
    if (f.search?.trim()) q.set("search", f.search.trim());
    if (f.recommended !== undefined) q.set("recommended", String(f.recommended));
    if (f.unusual !== undefined) q.set("unusual", String(f.unusual));
    return request<{ total: number; items: Claim[] }>(`/api/claims?${q}`);
  },
  claim: (id: string) => request<ClaimDetail>(`/api/claims/${encodeURIComponent(id)}`),
  explanation: (id: string) => request<Explanation>(`/api/claims/${encodeURIComponent(id)}/explanation`),
  importCsv(file: File) {
    const form = new FormData();
    form.append("file", file);
    return request<ImportResult>("/api/claims/import", { method: "POST", body: form });
  },
  score: () => request<{ scored: number; review_recommended: number; unusual: number }>("/api/predictions", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
  }),
  models: () => request<ModelVersion[]>("/api/models"),
  activate: (id: number) => request<ModelVersion>(`/api/models/${id}/activate`, { method: "POST" }),
  cases(f: { status?: CaseStatus; claim_id?: string } = {}) {
    const q = new URLSearchParams(Object.entries(f).filter(([, v]) => v) as [string, string][]);
    return request<CaseSummary[]>(`/api/cases?${q}`);
  },
  case: (id: number) => request<CaseDetail>(`/api/cases/${id}`),
  openCase: (claim_id: string) => request<CaseDetail>("/api/cases", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ claim_id }),
  }),
  retryCase: (id: number) => request<CaseDetail>(`/api/cases/${id}/retry`, { method: "POST" }),
  decide: (id: number, decision: "escalate" | "clear", reason: string) => request<CaseDetail>(`/api/cases/${id}/decision`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision, reason }),
  }),
};

export const pct = (v: number | null | undefined, digits = 1) => v == null ? "–" : `${(v * 100).toFixed(digits)}%`;
export const money = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 });   // the data has no currency
export const num = (v: number) => v.toLocaleString();
