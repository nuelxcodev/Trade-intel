"""Directional models with calibrated probabilities and an honest evaluation.

One binary model per direction answers: "if I enter now, does price reach the target before the
stop within the horizon?" Probabilities are calibrated on a validation window, the decision
threshold is chosen on that same window, and everything is reported on a later, untouched test
window. A model that shows no net edge on test abstains instead of producing signals.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from .features import Dataset, make_features, purged_split
from .outcomes import OutcomeSpec

MIN_TEST_SIGNALS = 30
MIN_AUC = 0.52


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [{"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int((idx == b).sum()),
             "mean_pred": float(p[idx == b].mean()), "observed": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def fingerprint(df: pd.DataFrame) -> str:
    """Identifies the exact candles a model was trained on."""
    h = hashlib.sha256()
    h.update(f"{len(df)}|{df['open_time'].iat[0]}|{df['open_time'].iat[-1]}".encode())
    h.update(np.ascontiguousarray(df[["open", "high", "low", "close", "volume"]].to_numpy()).tobytes())
    return h.hexdigest()[:16]


@dataclass
class DirectionalModel:
    direction: int
    features: list[str]
    model: HistGradientBoostingClassifier
    calibrator: IsotonicRegression
    threshold: float | None
    usable: bool
    metrics: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(X[self.features])[:, 1]
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)


def _select_threshold(p: np.ndarray, r: np.ndarray, min_n: int) -> float | None:
    best, best_mean = None, 0.0
    for t in np.arange(0.30, 0.86, 0.01):
        m = p >= t
        if m.sum() < min_n:
            continue
        mean_r = r[m].mean()
        if mean_r > best_mean:
            best, best_mean = float(round(t, 2)), mean_r
    return best


def train_direction(ds: Dataset, direction: int, train_frac: float = 0.6, val_frac: float = 0.2,
                    seed: int = 0) -> DirectionalModel:
    X, y, r = ds.X, ds.y[direction].loc[ds.X.index], ds.r[direction].loc[ds.X.index]
    tr, va, te = purged_split(X.index.to_numpy(), ds.spec, train_frac, val_frac)
    feats = list(X.columns)
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                         l2_regularization=1.0, min_samples_leaf=40,
                                         random_state=seed)
    clf.fit(X[tr], y[tr])
    raw_val = clf.predict_proba(X[va])[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_val, y[va])
    p_val = cal.predict(raw_val)
    thr = _select_threshold(p_val, r[va].to_numpy(), max(MIN_TEST_SIGNALS, int(0.05 * va.sum())))

    p_te = np.clip(cal.predict(clf.predict_proba(X[te])[:, 1]), 0, 1)
    y_te, r_te = y[te].to_numpy(), r[te].to_numpy()
    auc = float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te)) > 1 else float("nan")
    sel = p_te >= thr if thr is not None else np.zeros(len(p_te), bool)
    metrics = {
        "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "bar_ranges": {k: [int(X.index[m].min()), int(X.index[m].max())]
                       for k, m in (("train", tr), ("val", va), ("test", te)) if m.any()},
        "base_rate_test": float(y_te.mean()),
        "auc_test": auc,
        "brier_test": float(brier_score_loss(y_te, p_te)),
        "brier_base_rate": float(brier_score_loss(y_te, np.full_like(p_te, y[tr].mean()))),
        "ece_test": expected_calibration_error(y_te, p_te),
        "reliability_test": reliability_table(y_te, p_te),
        "threshold": thr,
        "mean_r_all_test": float(r_te.mean()),
        "signals_test": int(sel.sum()),
    }
    if sel.sum():
        yhat = sel.astype(int)
        metrics.update({
            "precision_test": float(precision_score(y_te, yhat, zero_division=0)),
            "recall_test": float(recall_score(y_te, yhat, zero_division=0)),
            "f1_test": float(f1_score(y_te, yhat, zero_division=0)),
            "mean_r_signals_test": float(r_te[sel].mean()),
            "win_rate_signals_test": float(y_te[sel].mean()),
        })
    usable = bool(thr is not None and sel.sum() >= MIN_TEST_SIGNALS
                  and metrics.get("mean_r_signals_test", -1) > 0 and auc > MIN_AUC)
    return DirectionalModel(direction, feats, clf, cal, thr, usable, metrics)


def train_models(ds: Dataset, **kw) -> dict[int, DirectionalModel]:
    return {d: train_direction(ds, d, **kw) for d in (1, -1)}


# ---------- registry: every prediction must trace back to a model and its data ----------

def save_models(models: dict[int, DirectionalModel], df: pd.DataFrame, symbol: str, tf: str,
                spec: OutcomeSpec, directory: str = "models") -> str:
    fp = fingerprint(df)
    model_id = f"{symbol}-{tf}-{fp}"
    path = Path(directory) / model_id
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path / "models.joblib")
    meta = {"model_id": model_id, "symbol": symbol, "tf": tf, "data_fingerprint": fp,
            "trained_through_open_time": int(df["open_time"].iat[-1]), "n_bars": len(df),
            "spec": spec.__dict__, "created_at": int(time.time()), "sklearn": sklearn.__version__,
            "features": models[1].features,
            "models": {("long" if d == 1 else "short"): {"usable": m.usable, "threshold": m.threshold,
                                                            "metrics": m.metrics} for d, m in models.items()}}
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return model_id


def load_models(model_id: str, directory: str = "models") -> tuple[dict[int, DirectionalModel], dict]:
    path = Path(directory) / model_id
    return joblib.load(path / "models.joblib"), json.loads((path / "meta.json").read_text())


def predict_latest(models: dict[int, DirectionalModel], df: pd.DataFrame, model_id: str,
                   i: int | None = None) -> dict:
    """Calibrated probabilities for bar i (default: last closed bar), tagged with the model id."""
    i = len(df) - 1 if i is None else i
    X = make_features(df).iloc[[i]]
    out = {"model_id": model_id, "bar_open_time": int(df["open_time"].iat[i])}
    for d, name in ((1, "long"), (-1, "short")):
        m = models[d]
        p = float(m.predict_proba(X)[0]) if not X[m.features].isna().any(axis=None) else None
        out[name] = {"probability": p, "threshold": m.threshold, "usable": m.usable,
                     "qualifies": bool(m.usable and p is not None and m.threshold is not None and p >= m.threshold)}
    return out


def needs_retrain(meta: dict, df: pd.DataFrame, min_new_bars: int = 500, max_age_days: int = 30) -> bool:
    """Retrain when enough new validated bars exist or the model is old."""
    horizon = meta["spec"]["horizon"]
    resolved_new = int((df["open_time"] > meta["trained_through_open_time"]).sum()) - horizon
    age_days = (time.time() - meta["created_at"]) / 86_400
    return resolved_new >= min_new_bars or age_days > max_age_days
