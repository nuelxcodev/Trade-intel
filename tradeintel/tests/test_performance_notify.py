import pytest

from tradeintel.config import Settings
from tradeintel.exchange.base import Candle
from tradeintel.notify import (EmailChannel, InAppChannel, Notifier, Prefs, TelegramChannel, format_signal)
from tradeintel.performance import build_report, signal_outcome
from tradeintel.signals import SignalStore
from tradeintel.store import CandleStore
from tradeintel.timeframes import tf_ms

STEP = tf_ms("1h")
COST = 2 * (0.001 + 0.0005) * 100 / 4          # round-trip cost in R for entry 100, risk 4


def sig(**kw):
    base = dict(bar_open_time=0, symbol="AAAUSDT", tf="1h", direction="LONG", entry=100.0, entry_lo=99.8,
                entry_hi=100.2, sl=96.0, tp1=103.0, tp2=108.0, tp3=112.0, rr=2.0, qty=2.5, risk_amount=10.0,
                score=75.0, probability=0.6, model_id="m1", expiry_bars=4, hold_bars=6,
                payload={"components": {"model": 0.9, "similarity": 0.4, "structure": 0.7},
                         "invalidation": ["a candle trades through the stop loss at 96 before entry"],
                         "evidence": {"patterns_now": {"engulfing": 1}, "base_rate": 0.4,
                                      "neighbours": {"k": 50, "long_success": 0.52, "short_success": 0.3}}})
    return {**base, **kw}


def c(k, o, h, l, cl):
    return {"open_time": k * STEP, "open": o, "high": h, "low": l, "close": cl}


ENTER = [c(1, 100, 100.3, 99.9, 100.1)]


def run(ss, path, **kw):
    sid = ss.create(sig(**kw))
    ss.process(sid, path)
    return sid


# ---------------- performance ----------------

def outcome(ss, sid):
    s = ss.get(sid); s["id"] = sid
    return signal_outcome(s, ss.history(sid), Settings())


def test_outcome_stop_before_tp1_is_minus_one_r_net_of_costs():
    ss = SignalStore(CandleStore())
    sid = run(ss, ENTER + [c(2, 100, 100.2, 95, 96)])
    o = outcome(ss, sid)
    assert o["state"] == "closed" and o["r_gross"] == pytest.approx(-1.0)
    assert o["r_net"] == pytest.approx(-1.0 - COST)


def test_outcome_tp2_then_breakeven():
    ss = SignalStore(CandleStore())
    sid = run(ss, ENTER + [c(2, 100.1, 108.5, 100.05, 108), c(3, 108, 108.2, 99.5, 100)])
    o = outcome(ss, sid)
    assert o["tp_hit"] == 2 and o["r_gross"] == pytest.approx((0.75 + 2.0) / 3)
    assert o["r_net"] == pytest.approx((0.75 + 2.0) / 3 - COST)


def test_outcome_full_tp3_and_short_and_expiry():
    ss = SignalStore(CandleStore())
    a = run(ss, ENTER + [c(2, 100.1, 113, 100.05, 112.5)], id="a")
    assert outcome(ss, a)["r_gross"] == pytest.approx((0.75 + 2.0 + 3.0) / 3)
    b = run(ss, [c(1, 99, 100.1, 98.5, 99.7), c(2, 99.7, 99.9, 96.5, 97.2), c(3, 97, 97.2, 91.5, 92), c(4, 92, 88, 87, 88)],
            id="b", direction="SHORT", sl=104, tp1=97, tp2=92, tp3=88)
    assert outcome(ss, b)["tp_hit"] == 3 and outcome(ss, b)["r_gross"] > 0
    e = run(ss, ENTER + [c(k, 100.5, 101, 100.2, 101) for k in (2, 3, 4, 5, 6, 7)], id="e")
    o = outcome(ss, e)                                       # expired active, 2/3... all 3 thirds exit at 101
    assert o["final_status"] == "EXPIRED" and o["r_gross"] == pytest.approx(0.25)


def test_no_entry_and_open_signals_are_not_trades():
    ss = SignalStore(CandleStore())
    n = run(ss, [c(1, 98, 98.5, 95, 96.5)], id="n")
    assert outcome(ss, n)["state"] == "no_entry"
    o = run(ss, ENTER, id="o")
    assert outcome(ss, o)["state"] == "open"
    rep = build_report(ss)
    assert rep["overall"]["signals"] == 2 and rep["overall"]["trades"] == 0
    assert rep["overall"]["no_entry"] == 1 and rep["overall"]["open"] == 1


def test_report_metrics_and_breakdowns():
    cs = CandleStore()
    ss = SignalStore(cs)
    run(ss, ENTER + [c(2, 100.1, 113, 100.05, 112.5)], id="w", symbol="AAAUSDT")        # big win
    run(ss, ENTER + [c(2, 100, 100.2, 95, 96)], id="l", symbol="BBBUSDT", tf="4h")      # loss
    rep = build_report(ss)
    ov = rep["overall"]
    assert ov["trades"] == 2 and ov["wins"] == 1 and ov["losses"] == 1 and ov["win_rate"] == 0.5
    wins, losses = (5.75 / 3 - COST), (1 + COST)
    assert ov["profit_factor"] == pytest.approx(wins / losses)
    assert ov["max_drawdown_r"] >= 0
    assert set(rep["by_symbol"]) == {"AAAUSDT", "BBBUSDT"} and set(rep["by_timeframe"]) == {"1h", "4h"}
    assert rep["by_pattern"]["engulfing"]["trades"] == 2 and rep["by_model"]["m1"]["trades"] == 2
    assert rep["by_symbol"]["AAAUSDT"]["win_rate"] == 1.0 and rep["by_symbol"]["BBBUSDT"]["win_rate"] == 0.0


def test_drawdown_is_measured_in_exit_order():
    ss = SignalStore(CandleStore())
    run(ss, ENTER + [c(2, 100, 100.2, 95, 96)], id="l1")
    run(ss, ENTER + [c(2, 100, 100.2, 95, 96), ], id="l2")
    dd = build_report(ss)["overall"]["max_drawdown_r"]
    assert dd == pytest.approx(2 * (1 + COST))


def test_excursions_use_stored_candles():
    cs = CandleStore()
    ss = SignalStore(cs)
    cs.upsert("fake", "AAAUSDT", "1h", [Candle(1 * STEP, 100, 100.3, 99.9, 100.1, 1), Candle(2 * STEP, 100.1, 104, 98, 103, 1),
                                       Candle(3 * STEP, 103, 103.2, 99.5, 100, 1)])
    sid = ss.create(sig())
    ss.process(sid, [c(1, 100, 100.3, 99.9, 100.1), c(2, 100.1, 104, 98, 103), c(3, 103, 103.2, 99.5, 100)])
    row = build_report(ss, candles=cs, exchange="fake")["signals"][0]
    assert row["mfe_r"] == pytest.approx(1.0) and row["mae_r"] == pytest.approx(-0.5)


# ---------------- notifications ----------------

def stored(ss, **kw):
    sid = ss.create(sig(**kw))
    s = ss.get(sid); s["id"] = sid
    return sid, s


def test_message_contains_required_fields_from_stored_numbers_only():
    ss = SignalStore(CandleStore())
    sid, s = stored(ss)
    title, body = format_signal(s)
    for needle in ("AAAUSDT", "LONG", "1h", "Entry 100.00", "Stop loss 96.00", "103.00 / 108.00 / 112.00",
                   "1 : 2.00", "60%", "not a guarantee", "engulfing", "52%", "Abandon if", "m1", sid):
        assert needle in body or needle in title, needle


class Recorder:
    def __init__(self, name, fail_times=0):
        self.name, self.fail_times, self.sent = name, fail_times, []

    def send(self, user_id, prefs, title, body, signal_id):
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("boom")
        self.sent.append((user_id, signal_id))


def notifier(channels, **kw):
    cs = CandleStore()
    ss = SignalStore(cs)
    return Notifier(cs, ss, channels, sleep=lambda s: None, **kw), ss


def test_preferences_filter_score_timeframe_and_probability():
    tg = Recorder("telegram")
    n, ss = notifier([tg])
    n.set_prefs("hi", Prefs(min_score=80, channels=["telegram"]))
    n.set_prefs("tf", Prefs(timeframes=["4h"], channels=["telegram"]))
    n.set_prefs("p", Prefs(min_probability=0.7, channels=["telegram"]))
    n.set_prefs("ok", Prefs(min_score=70, timeframes=["1h"], channels=["telegram"]))
    sid, _ = stored(ss)
    res = n.notify_signal(sid)
    assert list(res) == ["ok"] and tg.sent == [("ok", sid)]


def test_never_double_sends_even_across_scans():
    tg = Recorder("telegram")
    n, ss = notifier([tg])
    n.set_prefs("u", Prefs(channels=["telegram"]))
    sid, _ = stored(ss)
    n.notify_signal(sid)
    n.notify_signal(sid)
    n.notify_new([sid])
    assert len(tg.sent) == 1


def test_retry_then_success_and_failure_isolation():
    flaky, ok = Recorder("telegram", fail_times=2), Recorder("email")
    n, ss = notifier([flaky, ok], retries=3)
    n.set_prefs("u", Prefs(channels=["telegram", "email"]))
    sid, _ = stored(ss)
    assert n.notify_signal(sid) == {"u": {"telegram": "sent", "email": "sent"}}

    dead, ok2 = Recorder("telegram", fail_times=99), Recorder("email")
    n2, ss2 = notifier([dead, ok2], retries=2)
    n2.set_prefs("u", Prefs(channels=["telegram", "email"]))
    sid2, _ = stored(ss2)
    assert n2.notify_signal(sid2) == {"u": {"telegram": "failed", "email": "sent"}}
    status = n2.conn.execute("SELECT status, attempts FROM delivery_log WHERE channel='telegram'").fetchone()
    assert status == ("failed", 2)
    n2.notify_signal(sid2)                                   # failed sends may be retried later; sent ones are not
    assert len(ok2.sent) == 1


def test_in_app_inbox():
    cs = CandleStore()
    ss = SignalStore(cs)
    n = Notifier(cs, ss, [InAppChannel(cs)])
    n.set_prefs("u", Prefs(channels=["in_app"]))
    sid, _ = stored(ss)
    n.notify_signal(sid)
    inbox = n.inbox("u")
    assert len(inbox) == 1 and inbox[0]["signal_id"] == sid and "AAAUSDT" in inbox[0]["title"]


def test_telegram_and_email_adapters_use_their_transports():
    class Resp:
        status_code = 200

    class Sess:
        def post(self, url, json, timeout):
            self.url, self.json = url, json
            return Resp()

    sess = Sess()
    tg = TelegramChannel(token="TKN", session=sess)
    tg.send("u", Prefs(telegram_chat_id="42"), "T", "B", "s1")
    assert sess.url == "https://api.telegram.org/botTKN/sendMessage" and sess.json["chat_id"] == "42"
    with pytest.raises(RuntimeError):
        TelegramChannel(token="TKN", session=sess).send("u", Prefs(), "T", "B", "s1")      # no chat id

    sent = {}

    class SMTP:
        def __init__(self, host, port, timeout):
            sent["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            sent["tls"] = True

        def login(self, u, p):
            sent["login"] = u

        def send_message(self, msg):
            sent["to"], sent["subject"] = msg["To"], msg["Subject"]

    em = EmailChannel(host="smtp.test", user="bot@x.io", password="pw", smtp_factory=SMTP)
    em.send("u", Prefs(email="me@x.io"), "Subject", "Body", "s1")
    assert sent == {"host": "smtp.test", "tls": True, "login": "bot@x.io", "to": "me@x.io", "subject": "Subject"}
    with pytest.raises(RuntimeError):
        EmailChannel(host=None, sender=None).send("u", Prefs(email="me@x.io"), "T", "B", "s1")
