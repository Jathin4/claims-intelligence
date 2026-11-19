import { useEffect, useState } from "react";
import { api, hasToken, setToken, setUnauthorizedHandler, type User } from "./api";
import Cases from "./Cases";
import Claims from "./Claims";
import Dashboard from "./Dashboard";
import Import from "./Import";
import Login from "./Login";
import Models from "./Models";

type Tab = "dashboard" | "claims" | "cases" | "import" | "models";
const LABELS: Record<Tab, string> = { dashboard: "Dashboard", claims: "Claims", cases: "Cases", import: "Import", models: "Models" };

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(hasToken());
  const [tab, setTab] = useState<Tab>("dashboard");
  const [openCase, setOpenCase] = useState<number | null>(null);
  const showCase = (id: number) => { setOpenCase(id); setTab("cases"); };

  useEffect(() => {
    setUnauthorizedHandler(() => setUser(null));   // expired token → back to login
    if (hasToken()) api.me().then(setUser).catch(() => setToken(null)).finally(() => setChecking(false));
  }, []);

  if (checking) return <div className="center muted">Loading…</div>;
  if (!user) return <Login onLogin={setUser} />;

  // Reviewers read and investigate; importing needs the analyst (or admin) role, as the API enforces.
  const tabs: Tab[] = user.role === "reviewer" ? ["dashboard", "claims", "cases", "models"] : ["dashboard", "claims", "cases", "import", "models"];
  return (
    <div className="app">
      <header className="topbar">
        <strong>Claims Intelligence</strong>
        <nav className="tabs" aria-label="Sections">
          {tabs.map((t) => (
            <button key={t} className={tab === t ? "tab active" : "tab"} aria-current={tab === t} onClick={() => { setTab(t); if (t === "cases") setOpenCase(null); }}>
              {LABELS[t]}
            </button>
          ))}
        </nav>
        <span className="muted user">{user.email} · {user.role}</span>
        <button className="ghost" onClick={() => { setToken(null); setUser(null); }}>Log out</button>
      </header>
      <main className="content">
        {tab === "dashboard" ? <Dashboard /> : tab === "claims" ? <Claims user={user} onCase={showCase} />
          : tab === "cases" ? <Cases user={user} openId={openCase} onOpen={setOpenCase} />
          : tab === "import" ? <Import /> : <Models user={user} />}
      </main>
    </div>
  );
}
