import { useEffect, useState } from "react";
import { api, money, type CaseSummary, type ClaimDetail as Detail, type Contribution, type Explanation, type User } from "./api";
import { Flags, ScoreBar } from "./Claims";

export default function ClaimDetail({ id, user, onBack, onOpen, onCase }: {
  id: string; user: User; onBack: () => void; onOpen: (id: string) => void; onCase: (id: number) => void;
}) {
  const [claim, setClaim] = useState<Detail | null>(null);
  const [why, setWhy] = useState<Explanation | null>(null);
  const [error, setError] = useState("");
  const [whyError, setWhyError] = useState("");

  useEffect(() => {
    let live = true;
    setClaim(null); setWhy(null); setError(""); setWhyError("");
    api.claim(id).then((c) => live && setClaim(c)).catch((e) => live && setError(e.message));
    api.explanation(id).then((w) => live && setWhy(w)).catch((e) => live && setWhyError(e.message));
    return () => { live = false; };
  }, [id]);

  return (
    <div className="page-pad">
      <div><button className="ghost small" onClick={onBack}>← Back to claims</button></div>
      {error && <p className="error" role="alert">{error}</p>}
      {claim && (
        <>
          <section className="card">
            <div className="row spread">
              <h1>{claim.claim_id}</h1>
              <span className="row"><Flags claim={claim} /><Investigate claimId={claim.claim_id} user={user} onCase={onCase} /></span>
            </div>
            <dl className="facts">
              <div><dt>Customer</dt><dd>{claim.customer_id}</dd></div>
              <div><dt>Region</dt><dd className="cap">{claim.region ?? "–"}</dd></div>
              <div><dt>Category</dt><dd className="cap">{claim.claim_category}</dd></div>
              <div><dt>Amount</dt><dd>{money(claim.claim_amount)}</dd></div>
              <div><dt>Claim date</dt><dd>{claim.claim_date}</dd></div>
              <div><dt>Policy start</dt><dd>{claim.policy_start_date}</dd></div>
              <div><dt>Documents</dt><dd>{claim.document_count ?? "missing"}</dd></div>
              <div><dt>Known outcome</dt><dd>{claim.manual_review_required == null ? "not yet known" : claim.manual_review_required ? "was reviewed" : "not reviewed"}</dd></div>
            </dl>
          </section>

          <section className="card two-col">
            <div>
              <h2>Risk score</h2>
              {claim.prediction ? (
                <>
                  <ScoreBar score={claim.prediction.score} threshold={claim.prediction.threshold} />
                  <p className="small">{claim.prediction.review_recommended
                    ? <>Score <b>{claim.prediction.score.toFixed(3)}</b> is above the review threshold <b>{claim.prediction.threshold.toFixed(3)}</b>: <b>recommended for manual review</b>.</>
                    : <>Score <b>{claim.prediction.score.toFixed(3)}</b> is below the review threshold <b>{claim.prediction.threshold.toFixed(3)}</b>.</>}</p>
                  <p className="muted small">{claim.prediction.model}</p>
                </>
              ) : <p className="muted">Not scored by the active model yet.</p>}
            </div>
            <div>
              <h2>Unusual-claim check</h2>
              {claim.anomaly ? (
                <>
                  <p className="small">{claim.anomaly.flagged
                    ? <><span className="pill unusual">◆ Unusual</span> This claim looks unlike typical claims (anomaly score {claim.anomaly.score.toFixed(3)}).</>
                    : <>Looks typical (anomaly score {claim.anomaly.score.toFixed(3)}).</>}</p>
                  <p className="muted small">{claim.anomaly.model} · Isolation Forest; it flags the most unusual 2%, not wrongdoing.</p>
                </>
              ) : <p className="muted">No anomaly result.</p>}
            </div>
          </section>

          <section className="card">
            <h2>Why this score</h2>
            {whyError ? <p className="muted">{whyError}</p> : !why ? <p className="muted">Loading…</p> : (
              <>
                <div className="legend small">
                  <span><i className="sw up" /> raises risk</span>
                  <span><i className="sw down" /> lowers risk</span>
                </div>
                <ShapBars items={why.contributions} />
                <p className="muted small">{why.note}</p>
              </>
            )}
          </section>

          <section className="card">
            <h2>Earlier claims by this customer</h2>
            {claim.history.length === 0 ? <p className="muted">No earlier claims.</p> : (
              <div className="table-wrap">
                <table className="clickable">
                  <thead><tr><th>Claim</th><th>Date</th><th>Category</th><th className="num">Amount</th><th className="num">Docs</th></tr></thead>
                  <tbody>
                    {claim.history.map((h) => (
                      <tr key={h.claim_id} tabIndex={0} onClick={() => onOpen(h.claim_id)} onKeyDown={(e) => e.key === "Enter" && onOpen(h.claim_id)}>
                        <td><span className="link">{h.claim_id}</span></td>
                        <td className="nowrap">{h.claim_date}</td>
                        <td className="cap">{h.claim_category}</td>
                        <td className="num">{money(h.claim_amount)}</td>
                        <td className="num">{h.document_count ?? "–"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}

const show = (v: Contribution["value"]) =>
  v == null ? "missing" : typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : v;

// Diverging bars around a zero line: right/red raises the risk, left/blue lowers it.
function ShapBars({ items }: { items: Contribution[] }) {
  const max = Math.max(...items.map((c) => Math.abs(c.contribution)), 0.01);
  return (
    <ul className="shap">
      {items.map((c) => {
        const w = `${(Math.abs(c.contribution) / max) * 50}%`;
        const up = c.contribution >= 0;
        return (
          <li key={c.feature} title={`${c.label} = ${show(c.value)} · ${up ? "+" : ""}${c.contribution.toFixed(3)} log-odds`}>
            <span className="shap-label">{c.label} <span className="muted">= {show(c.value)}</span></span>
            <span className="shap-track">
              <span className={up ? "shap-bar up" : "shap-bar down"} style={up ? { left: "50%", width: w } : { right: "50%", width: w }} />
              <span className="shap-zero" />
            </span>
            <span className="num shap-val">{up ? "+" : "−"}{Math.abs(c.contribution).toFixed(2)}</span>
          </li>
        );
      })}
    </ul>
  );
}

// Opens an investigation case, or links to the open one. Only analysts and admins may open cases.
function Investigate({ claimId, user, onCase }: { claimId: string; user: User; onCase: (id: number) => void }) {
  const [open, setOpen] = useState<CaseSummary | null | undefined>(undefined);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    api.cases({ claim_id: claimId }).then((cs) => setOpen(cs.find((c) => c.status !== "closed") ?? null)).catch(() => setOpen(null));
  }, [claimId]);
  if (open === undefined) return null;
  if (open) return <button className="small" onClick={() => onCase(open.id)}>Case #{open.id} →</button>;
  if (user.role === "reviewer") return null;
  return (
    <>
      <button className="primary small" disabled={busy} onClick={() => {
        setBusy(true); setError("");
        api.openCase(claimId).then((c) => onCase(c.id)).catch((e) => { setError(e.message); setBusy(false); });
      }}>{busy ? "Opening…" : "Investigate"}</button>
      {error && <span className="error small" role="alert">{error}</span>}
    </>
  );
}