import { useEffect, useState } from "react";
import { api, num, pct, type Stats } from "./api";

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState("");
  useEffect(() => { api.stats().then(setStats).catch((e) => setError(e.message)); }, []);

  if (error) return <div className="page-pad"><p className="error" role="alert">{error}</p></div>;
  if (!stats) return <div className="page-pad muted">Loading…</div>;
  const share = (n: number) => (stats.scored ? n / stats.scored : 0);
  const maxRate = Math.max(...stats.by_category.map((c) => c.recommended_rate), 0.01);
  const maxUnusual = Math.max(...stats.by_category.map((c) => c.unusual_rate), 0.01);

  return (
    <div className="page-pad">
      <div className="metrics">
        <Tile value={num(stats.claims)} label="Claims in the system" />
        <Tile value={pct(share(stats.review_recommended))} label={`Recommended for review (${num(stats.review_recommended)})`} />
        <Tile value={pct(share(stats.unusual))} label={`Flagged unusual (${num(stats.unusual)})`} />
        <Tile value={stats.threshold?.toFixed(3) ?? "–"} label={`Review threshold · ${stats.model ?? "no active model"}`} />
      </div>
      {stats.scored < stats.claims && (
        <p className="warn small">{num(stats.claims - stats.scored)} claims have no score from the active model yet.</p>
      )}

      <section className="card">
        <h2>Risk score distribution</h2>
        <p className="muted small">Scored claims by model risk score (probability that the claim needs manual review). Claims right of the line are recommended for review.</p>
        <Histogram bins={stats.histogram} threshold={stats.threshold} />
      </section>

      <section className="card">
        <h2>By category</h2>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Category</th><th className="num">Claims</th><th>Recommended for review</th><th>Flagged unusual</th><th className="num">Historical review rate</th></tr></thead>
            <tbody>
              {stats.by_category.map((c) => (
                <tr key={c.category}>
                  <td className="cap">{c.category}</td>
                  <td className="num">{num(c.claims)}</td>
                  <td><InlineBar value={c.recommended_rate} max={maxRate} /></td>
                  <td><InlineBar value={c.unusual_rate} max={maxUnusual} /></td>
                  <td className="num">{pct(c.historical_review_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">Historical review rate covers only claims with a known outcome; new imports have none yet.</p>
      </section>

      <section className="card">
        <h2>Recent imports</h2>
        {stats.recent_imports.length === 0 ? <p className="muted">No imports yet.</p> : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>File</th><th>When</th><th className="num">Imported</th><th className="num">Rejected</th><th>Drift</th></tr></thead>
              <tbody>
                {stats.recent_imports.map((i) => (
                  <tr key={i.id}>
                    <td>{i.filename}</td>
                    <td className="nowrap">{new Date(i.created_at).toLocaleString()}</td>
                    <td className="num">{num(i.rows_imported)}</td>
                    <td className="num">{num(i.rows_rejected)}</td>
                    <td>{!i.drift_checked ? <span className="pill">not checked</span>
                      : i.drift_warnings.length ? <span className="pill large" title={i.drift_warnings.join("\n")}>⚠ {i.drift_warnings.length} warning{i.drift_warnings.length > 1 ? "s" : ""} · worst PSI {i.drift_worst?.toFixed(2)}</span>
                      : <span className="pill stable">✓ stable</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function Tile({ value, label }: { value: string; label: string }) {
  return <div className="metric"><div className="metric-v">{value}</div><div className="metric-l">{label}</div></div>;
}

export function InlineBar({ value, max }: { value: number; max: number }) {
  return (
    <span className="inline-bar">
      <span className="bar-track" aria-hidden="true"><span className="bar-fill" style={{ width: `${(value / max) * 100}%` }} /></span>
      <span className="num">{pct(value)}</span>
    </span>
  );
}

function Histogram({ bins, threshold }: { bins: Stats["histogram"]; threshold: number | null }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 720, H = 220, L = 52, B = 26, T = 12;
  const max = Math.max(...bins.map((b) => b.count), 1);
  const step = Math.pow(10, Math.floor(Math.log10(max)));
  const top = Math.ceil(max / step) * step;
  const ticks = [0, top / 2, top];
  const bw = (W - L) / bins.length;
  const y = (v: number) => T + (H - T - B) * (1 - v / top);
  const x = (score: number) => L + (score / 1) * (W - L);
  const h = hover === null ? null : bins[hover];
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Histogram of risk scores" onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W} y1={y(t)} y2={y(t)} className="grid" />
            <text x={L - 6} y={y(t) + 4} className="tick" textAnchor="end">{num(t)}</text>
          </g>
        ))}
        {bins.map((b, i) => {
          const bh = (H - T - B) * (b.count / top);
          return (
            <g key={i} onMouseEnter={() => setHover(i)}>
              <rect x={L + i * bw} y={T} width={bw} height={H - T - B} fill="transparent" />
              {b.count > 0 && <path className={hover === i ? "bar hot" : "bar"}
                d={roundedTop(L + i * bw + 1, H - B - Math.max(bh, 2), bw - 2, Math.max(bh, 2))} />}
            </g>
          );
        })}
        <line x1={L} x2={W} y1={H - B} y2={H - B} className="axis" />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => <text key={t} x={x(t)} y={H - 8} className="tick" textAnchor={t === 1 ? "end" : "middle"}>{t}</text>)}
        {threshold !== null && (
          <g>
            <line x1={x(threshold)} x2={x(threshold)} y1={T - 4} y2={H - B} className="ref" />
            <text x={x(threshold) + 6} y={T + 8} className="ref-label">review threshold {threshold.toFixed(2)}</text>
          </g>
        )}
      </svg>
      {h && (
        <div className="tooltip" style={{ left: `${((L + (hover! + 0.5) * bw) / W) * 100}%` }}>
          <b>{num(h.count)}</b> claims<br /><span className="muted">score {h.from.toFixed(2)}–{h.to.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}

// A bar with 4px rounded top corners and a square base on the axis.
function roundedTop(x: number, y: number, w: number, h: number) {
  const r = Math.min(4, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}
