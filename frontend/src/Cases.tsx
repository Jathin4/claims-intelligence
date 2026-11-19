import { useEffect, useState, type FormEvent } from "react";
import { api, type CaseDetail, type CaseStatus, type CaseSummary, type Finding, type User } from "./api";

const STATUS_LABEL: Record<CaseStatus, string> = {
  queued: "Queued", investigating: "Investigating", awaiting_decision: "Awaiting decision", closed: "Closed", failed: "Failed",
};
const STEPS: [string, string][] = [["plan", "Plan"], ["gather_evidence", "Evidence"], ["analysis", "AI analysis"],
  ["verification", "Number check"], ["report", "Report"], ["decision", "Decision"]];
const DECISION_LABEL = { escalate: "▲ Escalated for full review", clear: "✓ Cleared" };
const when = (s: string) => new Date(s).toLocaleString();

export default function Cases({ user, openId, onOpen }: { user: User; openId: number | null; onOpen: (id: number | null) => void }) {
  return openId === null ? <CaseList onOpen={onOpen} /> : <CaseView id={openId} user={user} onBack={() => onOpen(null)} />;
}

function CaseList({ onOpen }: { onOpen: (id: number) => void }) {
  const [status, setStatus] = useState<CaseStatus | "">("");
  const [rows, setRows] = useState<CaseSummary[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    const load = () => api.cases({ status: status || undefined }).then((r) => live && setRows(r)).catch((e) => live && setError(e.message));
    load();
    const t = setInterval(load, 5000);            // cases change in the background while agents work
    return () => { live = false; clearInterval(t); };
  }, [status]);

  return (
    <div className="page-pad">
      <section className="card">
        <div className="row spread">
          <h2>Investigation cases</h2>
          <label className="inline">Status
            <select value={status} onChange={(e) => setStatus(e.target.value as CaseStatus | "")}>
              <option value="">All</option>
              {Object.entries(STATUS_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </label>
        </div>
        <p className="muted small">Open a case from a claim's page ("Investigate"). Agents collect evidence and draft a report; a reviewer who didn't open the case makes the decision.</p>
        {error && <p className="error" role="alert">{error}</p>}
        {rows && (rows.length === 0 ? <p className="muted">No cases yet.</p> : (
          <div className="table-wrap">
            <table className="clickable">
              <thead><tr><th>Case</th><th>Claim</th><th>Priority</th><th>Status</th><th>Opened by</th><th>Updated</th></tr></thead>
              <tbody>
                {rows.map((c) => (
                  <tr key={c.id} tabIndex={0} onClick={() => onOpen(c.id)} onKeyDown={(e) => e.key === "Enter" && onOpen(c.id)}>
                    <td><span className="link">#{c.id}</span></td>
                    <td>{c.claim_id}</td>
                    <td><Priority p={c.priority} /></td>
                    <td><StatusPill c={c} /></td>
                    <td className="muted">{c.opened_by}</td>
                    <td className="nowrap muted">{when(c.updated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </section>
    </div>
  );
}

function Priority({ p }: { p: CaseSummary["priority"] }) {
  return <span className={`pill prio-${p}`}>{p === "high" ? "▲▲ high" : p === "medium" ? "▲ medium" : "low"}</span>;
}

function StatusPill({ c }: { c: CaseSummary }) {
  const label = c.status === "closed" && c.decision ? DECISION_LABEL[c.decision] : STATUS_LABEL[c.status];
  return <span className={`pill st-${c.status}${c.decision ? ` dec-${c.decision}` : ""}`}>{label}</span>;
}

function CaseView({ id, user, onBack }: { id: number; user: User; onBack: () => void }) {
  const [c, setC] = useState<CaseDetail | null>(null);
  const [error, setError] = useState("");
  const [focus, setFocus] = useState<string[]>([]);
  const running = c?.status === "queued" || c?.status === "investigating";

  useEffect(() => {
    let live = true;
    api.case(id).then((r) => live && setC(r)).catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [id]);
  useEffect(() => {                              // poll only while the agents are working
    if (!running) return;
    const t = setInterval(() => api.case(id).then(setC).catch((e) => setError(e.message)), 2500);
    return () => clearInterval(t);
  }, [id, running]);

  if (!c) return <div className="page-pad">{error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>}</div>;
  const reached = c.status === "closed" ? STEPS.length : c.status === "awaiting_decision" ? STEPS.length - 1
    : Math.max(0, STEPS.findIndex(([k]) => k === c.current_step) + 1);

  return (
    <div className="page-pad wide">
      <div><button className="ghost small" onClick={onBack}>← All cases</button></div>
      <section className="card">
        <div className="row spread">
          <h1>Case #{c.id} <span className="muted">· {c.claim_id}</span></h1>
          <span className="row"><Priority p={c.priority} /><StatusPill c={c} /></span>
        </div>
        <p className="muted small">Opened by {c.opened_by} · {when(c.created_at)}</p>
        <ol className="steps-bar" aria-label="Progress">
          {STEPS.map(([k, label], i) => (
            <li key={k} className={i < reached ? "reached" : i === reached && running ? "active" : ""}>{label}</li>
          ))}
        </ol>
        {running && <p className="muted small" aria-live="polite">The agents are working; the AI analysis takes a few minutes on this laptop's CPU. This page updates by itself.</p>}
        {c.status === "failed" && (
          <div className="callout-warn small">
            <b>The investigation stopped:</b> {c.error}
            {user.role !== "reviewer" && <div><button className="small" onClick={() => api.retryCase(c.id).then(setC).catch((e) => setError(e.message))}>Retry</button></div>}
          </div>
        )}
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      {c.report && c.evidence && (
        <div className="case-grid">
          <div className="case-main">
            <section className="card">
              <h2>Summary</h2>
              <p>{c.report.summary}</p>
              <p className="muted small">Written by code from the evidence, not by the AI.</p>
            </section>
            <section className="card">
              <h2>Findings <span className="pill stable small">✓ numbers checked</span></h2>
              {c.report.findings.length === 0 ? <p className="muted">No AI finding passed the checks; rely on the evidence.</p> : (
                <ol className="findings">
                  {c.report.findings.map((f, i) => <FindingItem key={i} f={f} onFocus={setFocus} focus={focus} />)}
                </ol>
              )}
              <p className="muted small">AI-written (qwen3:8b). Every number, date and ID was checked in code against the evidence each finding cites; the reasoning was not. Click an evidence tag to see it.</p>
            </section>
            {c.report.reviewer_questions.length > 0 && (
              <section className="card">
                <h2>Questions to check in the claim documents</h2>
                <ul>{c.report.reviewer_questions.map((q, i) => <li key={i}>{q}</li>)}</ul>
              </section>
            )}
            {c.report.unverified.length > 0 && (
              <section className="card unverified">
                <h2>Did not pass the checks <span className="muted small">· not findings</span></h2>
                <ul className="findings">
                  {c.report.unverified.map((f, i) => (
                    <li key={i}><s className="muted">{f.text}</s><div className="warn small">{f.problems.join("; ")}</div></li>
                  ))}
                </ul>
              </section>
            )}
            <section className="card">
              <h2>Limitations</h2>
              <ul className="limits">{c.report.limitations.map((l, i) => <li key={i}>{l}</li>)}</ul>
            </section>
            <DecisionCard c={c} user={user} onDone={setC} />
          </div>
          <aside className="card evidence">
            <h2>Evidence</h2>
            <p className="muted small">Copied from the database when the agents ran.</p>
            {c.evidence.map((e) => (
              <div key={e.id} id={`ev-${e.id}`} className={focus.includes(e.id) ? "ev focus" : "ev"}>
                <h3><span className="cite">{e.id}</span> {e.title}</h3>
                <div className="muted small">source: {Object.entries(e.source).map(([k, v]) => `${k} ${v}`).join(" · ")}</div>
                <dl>{e.facts.map((f, i) => <div key={i}><dt>{f.label}</dt><dd>{String(f.value)}</dd></div>)}</dl>
              </div>
            ))}
          </aside>
        </div>
      )}

      <section className="card">
        <h2>Activity</h2>
        <ul className="events">{c.events.map((e, i) => <li key={i}><span className="muted nowrap">{new Date(e.created_at).toLocaleTimeString()}</span> <b>{e.step}</b> · {e.message}</li>)}</ul>
      </section>
    </div>
  );
}

function FindingItem({ f, focus, onFocus }: { f: Finding; focus: string[]; onFocus: (ids: string[]) => void }) {
  const show = () => { onFocus(f.evidence); document.getElementById(`ev-${f.evidence[0]}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }); };
  const active = f.evidence.length > 0 && f.evidence.every((e) => focus.includes(e));
  return (
    <li className={active ? "active" : ""}>
      {f.text}{" "}
      {f.evidence.map((e) => <button key={e} className="cite" onClick={show} title="Show this evidence">{e}</button>)}
    </li>
  );
}

function DecisionCard({ c, user, onDone }: { c: CaseDetail; user: User; onDone: (c: CaseDetail) => void }) {
  const [decision, setDecision] = useState<"escalate" | "clear" | "">("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  if (c.status === "closed" && c.decision) {
    return (
      <section className="card">
        <h2>Decision</h2>
        <p><StatusPill c={c} /> by <b>{c.decided_by}</b> · {c.decided_at && when(c.decided_at)}</p>
        <p className="small"><b>Reason:</b> {c.decision_reason}</p>
      </section>
    );
  }
  if (c.status !== "awaiting_decision") return null;
  if (!c.can_decide) {
    return <section className="card"><h2>Decision</h2><p className="muted">
      {user.email === c.opened_by ? "You opened this case, so another reviewer must decide it (four-eyes rule)."
        : "Waiting for a reviewer's decision."}</p></section>;
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!decision) return;
    setBusy(true); setError("");
    try { onDone(await api.decide(c.id, decision, reason)); } catch (err) { setError(err instanceof Error ? err.message : "Failed"); } finally { setBusy(false); }
  }
  return (
    <form className="card decision" onSubmit={submit}>
      <h2>Your decision</h2>
      <div className="row">
        <label className="inline"><input type="radio" name="d" checked={decision === "escalate"} onChange={() => setDecision("escalate")} /> Escalate for full manual review</label>
        <label className="inline"><input type="radio" name="d" checked={decision === "clear"} onChange={() => setDecision("clear")} /> Clear: no further review</label>
      </div>
      <label>Reason (required, saved in the audit log)
        <textarea rows={3} value={reason} onChange={(e) => setReason(e.target.value)} minLength={10} maxLength={2000} required
                  placeholder="What in the evidence or documents led to this decision?" />
      </label>
      {error && <p className="error" role="alert">{error}</p>}
      <div><button className="primary" disabled={busy || !decision || reason.trim().length < 10}>{busy ? "Saving…" : "Record decision"}</button></div>
    </form>
  );
}
