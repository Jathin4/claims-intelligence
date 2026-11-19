import { useEffect, useState } from "react";
import { api, CATEGORIES, money, num, PAGE_SIZE, type Claim, type ClaimFilters, type User } from "./api";
import ClaimDetail from "./ClaimDetail";

const tri = (v: string) => (v === "" ? undefined : v === "yes");

export default function Claims({ user, onCase }: { user: User; onCase: (id: number) => void }) {
  const [filters, setFilters] = useState<ClaimFilters>({ sort: "score", offset: 0 });
  const [page, setPage] = useState<{ total: number; items: Claim[] } | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setError("");
    api.claims(filters).then((p) => live && setPage(p)).catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [filters]);

  const set = (patch: Partial<ClaimFilters>) => setFilters((f) => ({ ...f, ...patch, offset: patch.offset ?? 0 }));
  if (selected) return <ClaimDetail id={selected} user={user} onCase={onCase} onBack={() => setSelected(null)} onOpen={setSelected} />;

  return (
    <div className="page-pad wide">
      <section className="card">
        <div className="row filters">
          <label>Claim or customer ID
            <input type="search" placeholder="CLM-019910 or CUS-04947" value={filters.search ?? ""} maxLength={20}
                   onChange={(e) => set({ search: e.target.value })} />
          </label>
          <label>Category
            <select value={filters.category ?? ""} onChange={(e) => set({ category: e.target.value || undefined })}>
              <option value="">All</option>
              {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label>Recommended for review
            <select value={filters.recommended === undefined ? "" : filters.recommended ? "yes" : "no"}
                    onChange={(e) => set({ recommended: tri(e.target.value) })}>
              <option value="">Any</option><option value="yes">Yes</option><option value="no">No</option>
            </select>
          </label>
          <label>Flagged unusual
            <select value={filters.unusual === undefined ? "" : filters.unusual ? "yes" : "no"}
                    onChange={(e) => set({ unusual: tri(e.target.value) })}>
              <option value="">Any</option><option value="yes">Yes</option><option value="no">No</option>
            </select>
          </label>
          <label>Sort by
            <select value={filters.sort} onChange={(e) => set({ sort: e.target.value as ClaimFilters["sort"] })}>
              <option value="score">Risk score (highest first)</option>
              <option value="date">Claim date (newest first)</option>
              <option value="amount">Amount (largest first)</option>
            </select>
          </label>
        </div>
      </section>

      {error && <p className="error" role="alert">{error}</p>}
      {page && (
        <section className="card">
          <div className="row spread">
            <span className="muted small">{num(page.total)} claims · showing {page.total ? filters.offset + 1 : 0}–{Math.min(filters.offset + PAGE_SIZE, page.total)}</span>
            <span className="row">
              <button className="small" disabled={filters.offset === 0} onClick={() => set({ offset: filters.offset - PAGE_SIZE })}>← Previous</button>
              <button className="small" disabled={filters.offset + PAGE_SIZE >= page.total} onClick={() => set({ offset: filters.offset + PAGE_SIZE })}>Next →</button>
            </span>
          </div>
          <div className="table-wrap">
            <table className="clickable">
              <thead><tr><th>Claim</th><th>Date</th><th>Category</th><th className="num">Amount</th><th className="num">Docs</th><th>Risk score</th><th>Flags</th></tr></thead>
              <tbody>
                {page.items.map((c) => (
                  <tr key={c.claim_id} tabIndex={0} onClick={() => setSelected(c.claim_id)}
                      onKeyDown={(e) => e.key === "Enter" && setSelected(c.claim_id)}>
                    <td><span className="link">{c.claim_id}</span><div className="muted small">{c.customer_id}</div></td>
                    <td className="nowrap">{c.claim_date}</td>
                    <td className="cap">{c.claim_category}</td>
                    <td className="num">{money(c.claim_amount)}</td>
                    <td className="num">{c.document_count ?? "–"}</td>
                    <td>{c.prediction ? <ScoreBar score={c.prediction.score} threshold={c.prediction.threshold} /> : <span className="muted">not scored</span>}</td>
                    <td><Flags claim={c} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

export function Flags({ claim }: { claim: Claim }) {
  return (
    <span className="flags">
      {claim.prediction?.review_recommended && <span className="pill review">● Review</span>}
      {claim.anomaly?.flagged && <span className="pill unusual">◆ Unusual</span>}
    </span>
  );
}

export function ScoreBar({ score, threshold }: { score: number; threshold: number }) {
  return (
    <span className="score-bar" title={`score ${score.toFixed(3)} · threshold ${threshold.toFixed(3)}`}>
      <span className="bar-track" aria-hidden="true">
        <span className="bar-fill" style={{ width: `${score * 100}%` }} />
        <span className="bar-ref" style={{ left: `${threshold * 100}%` }} />
      </span>
      <span className="num">{score.toFixed(2)}</span>
    </span>
  );
}
