"""Data drift: compare a new batch's feature distributions with the model's training data (PSI).

PSI = Σ over bins (new% − reference%) × ln(new% / reference%)
Readings (rules of thumb): < 0.1 stable · 0.1–0.25 moderate shift · ≥ 0.25 large shift.
"""
import numpy as np
import pandas as pd

try:
    from features import NUMERIC            # run as a script from ml/
except ImportError:
    from ml.features import NUMERIC         # imported by the backend

MIN_ROWS = 200          # below this, PSI is mostly noise
EPS = 1e-4              # empty bins would make ln(0); use a tiny share instead


def level(psi: float) -> str:
    return "stable" if psi < 0.1 else "moderate" if psi < 0.25 else "large"


def psi_from_shares(ref: np.ndarray, new: np.ndarray) -> float:
    ref, new = np.clip(ref, EPS, None), np.clip(new, EPS, None)
    return float(np.sum((new - ref) * np.log(new / ref)))


def numeric_psi(ref_values: pd.Series, new_values: pd.Series, bins: int = 10) -> float:
    ref, new = ref_values.dropna().to_numpy(float), new_values.dropna().to_numpy(float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))   # bins from the REFERENCE data
    if len(edges) < 3:                                                 # (almost) constant feature: compare value shares
        return categorical_psi(pd.Series(ref), pd.Series(new))
    ref_share = np.histogram(np.clip(ref, edges[0], edges[-1]), edges)[0] / len(ref)
    new_share = np.histogram(np.clip(new, edges[0], edges[-1]), edges)[0] / len(new)
    return psi_from_shares(ref_share, new_share)


def categorical_psi(ref_values: pd.Series, new_values: pd.Series) -> float:
    cats = sorted(set(ref_values.dropna()) | set(new_values.dropna()))
    ref = ref_values.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    new = new_values.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return psi_from_shares(ref, new)


def drift_report(reference: pd.DataFrame, batch: pd.DataFrame) -> dict:
    """PSI per feature, category mix and missing documents. Both inputs are feature tables."""
    if len(batch) < MIN_ROWS:
        return {"checked": False, "reason": f"only {len(batch)} rows; at least {MIN_ROWS} are needed for a drift check"}
    features = {f: round(numeric_psi(reference[f], batch[f]), 4) for f in NUMERIC}
    features["claim_category"] = round(categorical_psi(reference["claim_category"], batch["claim_category"]), 4)
    missing = {"reference": round(float(reference["document_count"].isna().mean()), 4),
               "batch": round(float(batch["document_count"].isna().mean()), 4)}
    warnings = [f"{f}: PSI {v} ({level(v)} shift)" for f, v in sorted(features.items(), key=lambda t: -t[1]) if v >= 0.1]
    if missing["batch"] - missing["reference"] >= 0.05:
        warnings.append(f"document_count missing in {missing['batch']:.0%} of rows (training: {missing['reference']:.0%})")
    return {"checked": True, "rows": len(batch), "psi": features,
            "levels": {f: level(v) for f, v in features.items()}, "document_count_missing": missing,
            "worst": max(features.values()), "warnings": warnings}
