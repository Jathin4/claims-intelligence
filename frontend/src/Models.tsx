import { useEffect, useState } from "react";
import { api, pct, type Metrics, type ModelVersion, type User } from "./api";

const ROWS: [string, string, (v: number) => string][] = [
  ["pr_auc", "PR-AUC", (v) => v.toFixed(3)], ["roc_auc", "ROC-AUC", (v) => v.toFixed(3)], ["brier", "Brier score (lower is better)", (v) => v.toFixed(3)],
  ["review_rate", "Review rate", (v) => pct(v)], ["precision", "Precision at threshold", (v) => pct(v)], ["recall", "Recall at threshold", (v) => pct(v)],
  ["flag_rate", "Share flagged unusual", (v) => pct(v)], ["detection_rate", "Planted anomalies caught", (v) => pct(v)],
  ["precision_vs_planted", "Flags that were planted anomalies", (v) => pct(v)], ["random_precision", "…same, if flagging at random", (v) => pct(v)],
];

export default function Models({ user }: { user: User }) {
  const [models, setModels] = useState<ModelVersion[] | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<number | null>(null);
  const load = () => api.models().then(setModels).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  async function activate(m: ModelVersion) {
    if (!confirm(`Activate ${m.name} v${m.version}? The current active version will be retired. Existing scores stay; new claims are scored with this version.`)) return;
    setBusy(m.id); setError("");
    try { await api.activate(m.id); await load(); } catch (e) { setError(e instanceof Error ? e.message : "Activation failed"); } finally { setBusy(null); }
  }

  if (!models) return <div className="page-pad">{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</div>;
  return (
    <div className="page-pad">
      {error && <p className="error" role="alert">{error}</p>}
      {models.map((m) => (
        <section key={m.id} className="card">
          <div className="row spread">
            <h2>{m.name} v{m.version} <span className="muted small">· {m.estimator}</span></h2>
            <span className="row">
              <span className={`pill ${m.status}`}>{m.status}</span>
              {user.role === "admin" && m.status !== "active" && (
                <button className="small" disabled={busy !== null} onClick={() => activate(m)}>{busy === m.id ? "Activating…" : "Activate"}</button>
              )}
            </span>
          </div>
          <p className="muted small">Threshold {m.threshold.toFixed(4)} · registered {new Date(m.created_at).toLocaleDateString()}
            {m.activated_at && ` · activated ${new Date(m.activated_at).toLocaleDateString()}`}</p>
          <MetricTable validation={m.validation} test={m.test} />
        </section>
      ))}
    </div>
  );
}

function MetricTable({ validation, test }: { validation: Metrics; test: Metrics }) {
  const rows = ROWS.filter(([k]) => typeof validation[k] === "number" || typeof test[k] === "number");
  if (!rows.length) return <p className="muted small">No metrics recorded.</p>;
  const cell = (m: Metrics, k: string, f: (v: number) => string) => (typeof m[k] === "number" ? f(m[k] as number) : "–");
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Metric</th><th className="num">Validation</th><th className="num">Test (held out)</th></tr></thead>
        <tbody>{rows.map(([k, label, f]) => <tr key={k}><td>{label}</td><td className="num">{cell(validation, k, f)}</td><td className="num">{cell(test, k, f)}</td></tr>)}</tbody>
      </table>
    </div>
  );
}
