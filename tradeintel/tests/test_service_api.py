import json
import threading
import urllib.error
import urllib.request

import pytest

from tradeintel.api import Config, clean, make_server
from tradeintel.config import Settings
from tradeintel.service import Pipeline, env_list, env_settings, validate_timeframes
from tradeintel.synthetic import FakeExchange, make_candles
from tradeintel.timeframes import tf_ms

STEP = tf_ms("1h")


@pytest.fixture(scope="module")
def running(tmp_path_factory):
    d = tmp_path_factory.mktemp("svc")
    candles = make_candles(2200, "1h", seed=51)
    now = candles[-1].open_time + STEP + 1
    ex = FakeExchange({("AAAUSDT", "1h"): candles}, now)
    db, models = str(d / "t.db"), str(d / "models")
    pipe = Pipeline(ex, ["AAAUSDT", "BBBUSDT"], ["1h"], db, models, Settings(min_quote_volume_24h=1.0),
                    history_days=3650)
    summary = pipe.cycle()
    cfg = Config(db, models, api_key="secret", exchange="fake", symbols=["AAAUSDT", "BBBUSDT"], tfs=["1h"])
    srv = make_server(cfg, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield pipe, summary, f"http://127.0.0.1:{srv.server_address[1]}", d
    srv.shutdown()


def get(url, key="secret"):
    req = urllib.request.Request(url, headers={"X-API-Key": key} if key else {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_cycle_syncs_trains_scans_and_isolates_the_broken_pair(running):
    pipe, summary, _, d = running
    tf = summary["1h"]
    assert "BBBUSDT" not in tf["sync_errors"]                       # no candles is not an error, just empty
    assert pipe.store.count("fake", "AAAUSDT", "1h") == 2200
    assert any((d / "models").glob("AAAUSDT-1h-*/meta.json"))
    assert "BBBUSDT" in tf["model_notes"] and "need 1500" in tf["model_notes"]["BBBUSDT"]


def test_second_cycle_is_cheap_and_does_not_retrain(running):
    pipe, _, _, d = running
    before = len(list((d / "models").glob("AAAUSDT-1h-*")))
    pipe.exchange.requests.clear()
    pipe.cycle()
    assert len(list((d / "models").glob("AAAUSDT-1h-*"))) == before
    assert all(sym == "BBBUSDT" for sym, _, _ in pipe.exchange.requests)   # AAA is up to date: zero requests


def test_health_needs_no_key_and_api_does(running):
    _, _, base, _ = running
    assert get(base + "/health", key=None)[0] == 200
    assert get(base + "/api/overview", key=None)[0] == 401
    assert get(base + "/api/overview", key="wrong")[0] == 401


def test_overview_and_models(running):
    _, _, base, _ = running
    code, ov = get(base + "/api/overview")
    assert code == 200 and ov["monitored_pairs"] == 2 and ov["models"] >= 1
    code, models = get(base + "/api/models")
    m = models["models"][0]
    assert m["symbol"] == "AAAUSDT" and set(m["usable"]) == {"long", "short"}


def test_opportunities_report_explicit_none_when_no_edge(running):
    _, _, base, _ = running
    code, rep = get(base + "/api/opportunities?tf=1h")
    assert code == 200 and rep["tf"] == "1h"
    if not rep["opportunities"]:
        assert rep["message"].startswith("No qualified")
        assert "AAAUSDT" in rep["rejected"] and "BBBUSDT" in rep["rejected"]
    code, rep = get(base + "/api/opportunities?tf=4h")
    assert rep["message"] == "No scan has completed yet"


def test_signals_performance_and_notifications_endpoints(running):
    pipe, _, base, _ = running
    assert get(base + "/api/signals")[1] == {"signals": []} or "signals" in get(base + "/api/signals")[1]
    assert get(base + "/api/signals/nope")[0] == 404
    code, perf = get(base + "/api/performance")
    assert code == 200 and perf["overall"]["signals"] == len(get(base + "/api/signals")[1]["signals"])
    code, inbox = get(base + "/api/notifications?user=default")
    assert code == 200 and inbox["notifications"] == []
    assert get(base + "/api/nothing")[0] == 404


def test_signal_flows_through_api(running):
    pipe, _, base, _ = running
    sid = pipe.signals.create(dict(bar_open_time=0, symbol="AAAUSDT", tf="1h", direction="LONG", entry=100.0,
                                   entry_lo=99.8, entry_hi=100.2, sl=96.0, tp1=103.0, tp2=108.0, tp3=112.0, rr=2.0,
                                   qty=2.5, risk_amount=10.0, score=75.0, probability=0.6, model_id="m1",
                                   expiry_bars=4, hold_bars=6, payload={"evidence": {}}))
    pipe.notifier.notify_signal(sid)
    code, one = get(base + f"/api/signals/{sid}")
    assert code == 200 and one["state"]["status"] == "WAITING FOR ENTRY" and len(one["history"]) == 2
    assert [s["id"] for s in get(base + "/api/signals?status=open")[1]["signals"]] == [sid]
    inbox = get(base + "/api/notifications?user=default")[1]["notifications"]
    assert len(inbox) == 1 and inbox[0]["signal_id"] == sid


def post(url, data, key="secret"):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **({"X-API-Key": key} if key else {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_prefs_endpoint_validates_and_requires_key(running):
    pipe, _, base, _ = running
    assert post(base + "/api/prefs/ada", {"min_score": 70}, key=None)[0] == 401
    assert post(base + "/api/prefs/ada", {"hack": 1})[0] == 400
    assert post(base + "/api/prefs/ada", {"channels": ["pigeon"]})[0] == 400
    code, out = post(base + "/api/prefs/ada", {"min_score": 70, "timeframes": ["1h"], "channels": ["in_app"]})
    assert code == 200 and pipe.notifier.get_prefs("ada").min_score == 70


def test_writes_are_disabled_without_an_api_key(tmp_path):
    srv = make_server(Config(str(tmp_path / "x.db")), "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        code, out = post(f"http://127.0.0.1:{srv.server_address[1]}/api/prefs/u", {}, key=None)
        assert code == 403 and "TI_API_KEY" in out["error"]
    finally:
        srv.shutdown()


def test_json_is_always_valid():
    import numpy as np
    assert clean({"a": float("nan"), "b": np.float64(1.5), "c": [float("inf"), np.int64(2)]}) == \
        {"a": None, "b": 1.5, "c": [None, 2]}


def test_env_config(monkeypatch):
    monkeypatch.setenv("TI_MIN_SCORE", "72.5")
    monkeypatch.setenv("TI_CAPITAL", "2500")
    s = env_settings()
    assert s.min_score == 72.5 and s.capital == 2500.0 and s.risk_per_trade == Settings().risk_per_trade
    monkeypatch.setenv("TI_SYMBOLS", " BTCUSDT , ETHUSDT,,")
    assert env_list("TI_SYMBOLS", "x") == ["BTCUSDT", "ETHUSDT"]
    with pytest.raises(ValueError):
        validate_timeframes(["1h", "3h"])
