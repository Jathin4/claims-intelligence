r"""One-off check: scores stored by the API equal the training script's scores (no training-serving skew).

    ..\.venv\Scripts\python.exe ..\tests\check_serving.py      (from backend/, after `manage.py score`)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sqlalchemy import select

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))
import train  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import RiskPrediction  # noqa: E402
from features import FEATURES  # noqa: E402

parts, _, _ = train.load_split()
model = joblib.load(ROOT / "model-artifacts" / "claims-review-v1.joblib")
test = parts["test"]
expected = dict(zip(test["claim_id"], model.predict_proba(test[FEATURES])[:, 1]))
with SessionLocal() as db:
    stored = dict(db.execute(select(RiskPrediction.claim_id, RiskPrediction.score)
                             .where(RiskPrediction.claim_id.in_(list(expected)))).all())
diff = np.array([abs(stored[k] - expected[k]) for k in expected])
print(f"test claims compared: {len(diff):,} · max difference {diff.max():.5f} (scores are stored rounded to 4 decimals)")
assert diff.max() <= 0.00005 + 1e-9, "training and serving disagree"
print("OK: the API scores exactly what training evaluated")
