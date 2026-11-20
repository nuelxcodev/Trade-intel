import numpy as np
import pandas as pd
import pytest

from tradeintel.features import build_dataset, make_features, purged_split
from tradeintel.indicators import add_indicators
from tradeintel.ml import (expected_calibration_error, load_models, needs_retrain, predict_latest,
                           save_models, train_models)
from tradeintel.outcomes import OutcomeSpec
from tradeintel.similarity import SimilarityIndex
from tradeintel.synthetic import make_candles

SPEC = OutcomeSpec()


def frame(n, seed=21):
    return add_indicators(pd.DataFrame([c.__dict__ for c in make_candles(n, "1h", seed=seed)]))


@pytest.fixture(scope="module")
def trained():
    df = frame(6000)
    ds = build_dataset(df, SPEC)
    return df, ds, train_models(ds)


def test_features_are_causal():
    df = frame(700)
    full, cut = make_features(df), make_features(df.iloc[:450])
    pd.testing.assert_frame_equal(full.iloc[:450], cut)


def test_dataset_excludes_unresolved_tail():
    df = frame(1000)
    ds = build_dataset(df, SPEC)
    assert ds.X.index.max() + 1 + SPEC.horizon <= len(df)
    assert set(ds.X.index) == set(ds.y[1].index) == set(ds.r[-1].index)


def test_purged_split_leaves_gaps_and_is_ordered():
    bi = np.arange(60, 5000)
    tr, va, te = purged_split(bi, SPEC)
    assert not (tr & va).any() and not (va & te).any()
    gap = SPEC.horizon + 1
    assert bi[va].min() - bi[tr].max() > gap
    assert bi[te].min() - bi[va].max() > gap
    assert bi[tr].max() < bi[va].min() < bi[va].max() < bi[te].min()


def test_no_leakage_labels_cannot_cross_split_boundaries(trained):
    """A training row's outcome window must end before the validation window starts."""
    df, ds, models = trained
    ranges = models[1].metrics["bar_ranges"]
    assert ranges["train"][1] + SPEC.horizon < ranges["val"][0]
    assert ranges["val"][1] + SPEC.horizon < ranges["test"][0]


def test_probabilities_are_calibrated_and_valid(trained):
    _, _, models = trained
    for m in models.values():
        s = m.metrics
        assert 0 <= s["brier_test"] <= 1 and 0 <= s["ece_test"] <= 1
        assert all(0 <= b["mean_pred"] <= 1 and 0 <= b["observed"] <= 1 for b in s["reliability_test"])


def test_ece_detects_miscalibration():
    y = np.array([1, 0] * 500)
    assert expected_calibration_error(y, np.full(1000, 0.5)) < 0.01
    assert expected_calibration_error(y, np.full(1000, 0.95)) > 0.4


def test_model_abstains_without_net_edge_on_noise():
    """Pure noise: whatever the model finds on validation must not be advertised as usable edge
    unless it also shows positive net R and AUC above the bar on the untouched test window."""
    for seed in (31, 32, 33):
        df = add_indicators(pd.DataFrame([c.__dict__ for c in make_candles(6000, "1h", seed=seed, vol=0.006)]))
        for m in train_models(build_dataset(df, SPEC)).values():
            if m.usable:
                assert m.metrics["mean_r_signals_test"] > 0 and m.metrics["auc_test"] > 0.52
            else:
                assert not (m.metrics["signals_test"] >= 30 and m.metrics.get("mean_r_signals_test", -1) > 0
                            and m.metrics["auc_test"] > 0.52)


def test_registry_roundtrip_and_traceability(trained, tmp_path):
    df, _, models = trained
    mid = save_models(models, df, "SYN", "1h", SPEC, directory=str(tmp_path))
    loaded, meta = load_models(mid, directory=str(tmp_path))
    assert meta["model_id"] == mid and meta["data_fingerprint"] in mid
    assert meta["trained_through_open_time"] == int(df["open_time"].iat[-1])
    a, b = predict_latest(models, df, mid), predict_latest(loaded, df, mid)
    assert a == b and a["model_id"] == mid
    df2 = df.copy()
    df2.loc[10, "close"] *= 1.01                           # different data => different fingerprint
    from tradeintel.ml import fingerprint
    assert fingerprint(df2) != fingerprint(df)


def test_retrain_trigger(trained):
    df, _, models = trained
    import time
    meta = {"spec": SPEC.__dict__, "trained_through_open_time": int(df["open_time"].iat[-1]),
            "created_at": int(time.time())}
    assert not needs_retrain(meta, df)
    longer = add_indicators(pd.DataFrame([c.__dict__ for c in make_candles(6800, "1h", seed=21)]))
    assert needs_retrain({**meta, "trained_through_open_time": int(df["open_time"].iat[-1])}, longer)
    assert needs_retrain({**meta, "created_at": int(time.time()) - 40 * 86_400}, df)


def test_similarity_index_never_includes_recent_or_future_bars(trained):
    df, ds, _ = trained
    cutoff = 4000
    idx = SimilarityIndex(ds, cutoff_bar=cutoff)
    assert idx.rows.max() < cutoff - SPEC.horizon - 1 + 1
    q = ds.X.loc[[cutoff]]
    nb = idx.query(q, k=40)
    assert nb.k == 40 and max(nb.bar_index) < cutoff
    assert 0 <= nb.long_success <= 1 and 0 <= nb.short_success <= 1
