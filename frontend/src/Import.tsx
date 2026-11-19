import { useState, type FormEvent } from "react";
import { api, num, pct, type Drift, type ImportResult } from "./api";

const LABELS: Record<string, string> = {
  claim_amount: "Claim amount", policy_age_days: "Policy age", claim_history_count: "Claims in past year",
  days_since_previous_claim: "Days since previous claim", document_count: "Documents", amount_vs_category_median: "Amount vs category median",
  claim_category: "Category mix",
};

export default function Import() {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState("");
  const [result, setResult] = useState<ImportResult | null>(null);
  const [scored, setScored] = useState<{ scored: number; review_recommended: number; unusual: number } | null>(null);
  const [error, setError] = useState("");

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(""); setResult(null); setScored(null);
    try {
      setBusy("Importing…");
      const r = await api.importCsv(file);
      setResult(r);
      if (r.rows_imported > 0) { setBusy("Scoring new claims…"); setScored(await api.score()); }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Import failed");
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="page-pad">
      <form className="card upload" onSubmit={submit}>
        <h2>Import claims (CSV)</h2>
        <p className="muted small">Columns: claim_id, customer_id, region, policy_start_date, claim_date, claim_category, claim_amount, document_count. Valid rows are imported and scored; invalid rows are listed with the reason. New data is compared with the training data (drift check).</p>
        <div className="row">
          <label>CSV file<input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></label>
          <button className="primary" disabled={!file || !!busy}>{busy || "Import"}</button>
        </div>
        {error && <p className="error" role="alert">{error}</p>}
      </form>

      {result && (
        <section className="card">
          <h2>Result</h2>
          <div className="metrics">
            <div className="metric"><div className="metric-v">{num(result.rows_imported)}</div><div className="metric-l">imported of {num(result.rows_total)}</div></div>
            <div className="metric"><div className="metric-v">{num(result.rows_rejected)}</div><div className="metric-l">rejected</div></div>
            {scored && <div className="metric"><div className="metric-v">{num(scored.review_recommended)}</div><div className="metric-l">recommended for review of {num(scored.scored)} scored</div></div>}
            {scored && <div className="metric"><div className="metric-v">{num(scored.unusual)}</div><div className="metric-l">flagged unusual</div></div>}
          </div>
          {result.ignored_columns.length > 0 && <p className="muted small">Ignored columns: {result.ignored_columns.join(", ")}</p>}
          {result.errors.length > 0 && (
            <details>
              <summary>{result.errors.length} row error{result.errors.length > 1 ? "s" : ""}{result.errors.length < result.rows_rejected ? " (first ones shown)" : ""}</summary>
              <ul className="errors small">{result.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
            </details>
          )}
        </section>
      )}
      {result?.drift && <DriftCard drift={result.drift} />}
    </div>
  );
}

function DriftCard({ drift }: { drift: Drift }) {
  if (!drift.checked) return <section className="card"><h2>Drift check</h2><p className="muted">Not checked: {drift.reason}</p></section>;
  return (
    <section className="card">
      <h2>Drift check <span className="muted small">· {num(drift.rows ?? 0)} rows vs the data {drift.model} was trained on</span></h2>
      {drift.warnings?.length ? (
        <div className="callout-warn small"><b>⚠ This batch differs from the training data.</b> Scores may be less reliable and the review rate may jump.
          <ul>{drift.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></div>
      ) : <p className="small"><span className="pill stable">✓ stable</span> No large shifts.</p>}
      <div className="table-wrap">
        <table>
          <thead><tr><th>Feature</th><th className="num">PSI</th><th>Level</th></tr></thead>
          <tbody>
            {Object.entries(drift.psi ?? {}).sort((a, b) => b[1] - a[1]).map(([f, v]) => (
              <tr key={f}>
                <td>{LABELS[f] ?? f}</td>
                <td className="num">{v.toFixed(3)}</td>
                <td><span className={`pill ${drift.levels?.[f]}`}>{drift.levels?.[f] === "large" ? "▲ large" : drift.levels?.[f] === "moderate" ? "△ moderate" : "✓ stable"}</span></td>
              </tr>
            ))}
            {drift.document_count_missing && (
              <tr><td>Documents missing</td><td className="num" colSpan={2}>{pct(drift.document_count_missing.batch)} of rows (training {pct(drift.document_count_missing.reference)})</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="muted small">PSI (population stability index): under 0.1 stable, 0.1–0.25 moderate, 0.25 or more large.</p>
    </section>
  );
}
