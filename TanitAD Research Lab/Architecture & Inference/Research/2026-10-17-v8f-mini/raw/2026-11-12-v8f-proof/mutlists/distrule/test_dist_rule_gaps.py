"""test_dist_rule_gaps.py -- tests that close the DISTRULE mutations left STILL-GREEN by the first list (w1, w5, w8, w9, q5, q8).
Synthetic data only; the planner is driven on fakes (base planner, forward-kinematics stub, head stub, a recorder in place of dist_rule.distance_rule), so the
checks read the arguments the wrapper hands to the rule. Needs torch (a one-line tensor in the batch); run on the pod CPU through minipytest or pytest.
Import style as test_margin_qcal.py: the folder of the distrule tool files comes from env DISTRULE_SRC."""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

HERE = os.environ.get("DISTRULE_SRC", os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import margin_calib as MC  # noqa: E402
import qcal_fit as QF  # noqa: E402
import v8f_t1_eval_rule as W  # noqa: E402


# ---------------------------------------------------------------------------------------------- fakes for the RulePlanner
class _FK:
    """Ego position at step kk: (kk - 0.5, 0) on the x axis (the logged path below is the x axis)."""
    def gt_waypoints(self, track, t0, kk):
        return np.array([[[float(kk[0]) - 0.5, 0.0]]])


def _state(ci, t0, j, logged_v):
    g = np.stack([np.arange(1.0, 31.0), np.zeros(30)], 1)          # logged path (1,0) .. (30,0): the arc of logged point k is k
    return types.SimpleNamespace(ci=ci, t0=t0, j=j, d={"gt": {t0: g}}, track=np.zeros(8), logged_v=logged_v)


def _planner(monkeypatch, states, head, rec):
    import torch
    B = len(states)
    base = lambda batch: np.zeros((B, 30, 2))
    model = types.SimpleNamespace(wm=types.SimpleNamespace(lead_head=object()))
    rp = W.RulePlanner(base, model, "head", 10, _FK())
    rp.bind(states)

    def fake_rule(plan, v0, present, g_src, g0, v_L, m=10, **kw):
        rec.update(present=np.array(present), g_src=np.array(g_src), g0=np.array(g0), v_L=np.array(v_L))
        z = np.zeros(B)
        return types.SimpleNamespace(plan=plan, active=z.astype(bool), n_sat=z, n_sat_exec=z, n_lift=z, changed=z.astype(bool))
    monkeypatch.setattr(W.R, "distance_rule", fake_rule)
    monkeypatch.setattr(W, "head_values", lambda model, feats, frame_ok=None, hist=None: head)
    batch = {"p10f": torch.zeros(B, 3, 4), "feats": None, "frame_ok": None}
    return rp, batch


def test_w1_source_gap_carries_the_progress_offset_a_l_minus_a_m(monkeypatch):
    """Closes DISTRULE w1 (progress offset dropped from the source gap).  Hand value: tick k=3 on the x-axis path: logged arc a_l(3) = 3.0, ego at x = 2.5 so a_m = 2.5,
    g0 = g_src + a_l - a_m = 10 + 3.0 - 2.5 = 10.5; at k=0 both offsets are 0 by definition, g0 = g_src = 10.  The mutation hands 10.0 to the rule on the k=3 row."""
    lv = np.full(100, 5.0)
    st = [_state(0, 0, 3, lv), _state(1, 0, 0, lv)]
    head = (np.array([0.9, 0.9], np.float32), np.array([10.0, 10.0], np.float32), np.zeros(2, np.float32), np.ones(2, bool))
    rec = {}
    rp, batch = _planner(monkeypatch, st, head, rec)
    rp(batch)
    assert np.allclose(rec["g0"], [10.5, 10.0], atol=1e-9), rec["g0"]
    assert np.allclose(rec["g_src"], [10.0, 10.0])
    z = rp.log.arrays()
    assert np.allclose(z["a_l"], [3.0, 0.0], atol=1e-9) and np.allclose(z["a_m"], [2.5, 0.0], atol=1e-9) and np.allclose(z["g0"], [10.5, 10.0], atol=1e-9)


def test_w5_head_presence_threshold_is_half(monkeypatch):
    """Closes DISTRULE w5 (threshold 0.5 changed to 0.0).  Presence probabilities 0.3 and 0.49 are below 0.5 (not present), 0.5 and 0.9 are present (>=), 0.0 is not.
    With the threshold at 0.0 the first two rows would be present."""
    lv = np.full(100, 5.0)
    st = [_state(i, 0, 0, lv) for i in range(5)]
    p = np.array([0.3, 0.49, 0.5, 0.9, 0.0], np.float32)
    head = (p, np.full(5, 10.0, np.float32), np.zeros(5, np.float32), np.ones(5, bool))
    rec = {}
    assert W.HEAD_P_MIN == 0.5
    rp, batch = _planner(monkeypatch, st, head, rec)
    rp(batch)
    assert rec["present"].tolist() == [False, False, True, True, False], rec["present"]
    assert rp.log.arrays()["present"].tolist() == [False, False, True, True, False]


def test_w8_present_tick_with_a_non_finite_source_value_is_marked_bad_and_not_present(monkeypatch):
    """Closes DISTRULE w8 (source-bad ticks stay present).  Rows: all finite; gap NaN; closing speed NaN; logged speed row missing (v_l NaN).  Only the first stays present;
    the other three are src_bad.  The mutation leaves them present (the rule would then receive NaN inputs as a present lead)."""
    full, short = np.full(100, 5.0), np.full(3, 5.0)
    st = [_state(0, 0, 0, full), _state(1, 0, 0, full), _state(2, 0, 0, full), _state(3, 5, 5, short)]
    head = (np.full(4, 0.9, np.float32), np.array([10.0, np.nan, 10.0, 10.0], np.float32), np.array([0.0, 0.0, np.nan, 0.0], np.float32), np.ones(4, bool))
    rec = {}
    rp, batch = _planner(monkeypatch, st, head, rec)
    rp(batch)
    assert rec["present"].tolist() == [True, False, False, False], rec["present"]
    assert rp.log.arrays()["src_bad"].tolist() == [False, True, True, True]
    assert rp.log.arrays()["present"].tolist() == [True, False, False, False]


# ---------------------------------------------------------------------------------------------- the CLI refusal
def test_w9_dist_rule_with_a_non_run_mode_is_refused(monkeypatch):
    """Closes DISTRULE w9 (refusal of --dist-rule with a non-run mode removed).  The pinned tool is replaced by a fake whose parser says mode=report and whose main returns 77;
    every non-off rule must be refused with the message, and off must hand over to the pinned main (77, the control of the fake)."""
    fake = types.SimpleNamespace(build_parser=lambda: types.SimpleNamespace(parse_args=lambda rest: types.SimpleNamespace(mode="report")), main=lambda rest: 77)
    monkeypatch.setattr(W, "load_pinned", lambda tree: (fake, "fake"))
    assert W.main(["--tree", "x", "--dist-rule", "off", "report"]) == 77
    for rule in ("label", "head", "never", "push", "plumb"):
        with pytest.raises(SystemExit, match="REFUSED.*applies to the `run` mode"):
            W.main(["--tree", "x", "--dist-rule", rule, "report"])


# ---------------------------------------------------------------------------------------------- quantile map and margin calibration
def _files(tmp, gl, gh, logit=3.0):
    n = len(gl)
    np.savez(os.path.join(tmp, "A_train_lead.npz"), gap_m=np.asarray(gh, float), has_history=np.ones(n, bool), presence_logit=np.full(n, logit))
    np.savez(os.path.join(tmp, "A_train.npz"), clips=np.array(["c0"]), clip_idx=np.zeros(n, np.int64), t=np.arange(n))
    return {"index": {("c0", int(i)): int(i) for i in range(n)}, "gap": np.asarray(gl, float), "pres": np.ones(n)}


def test_q5_fit_pools_a_non_monotone_quantile_sequence(tmp_path):
    """Closes DISTRULE q5 (PAV removed from the fit).  Three bins of 300 rows with head gap 7, 12, 17 m and a constant label gap 6, 4, 15 m: the 20th percentiles are 6, 4, 15
    (a violation between the first two).  Pooled with equal weights: (6+4)/2 = 5 for both, so ys = [5, 5, 15] (all <= xs).  Without the pooling ys = [6, 4, 15]."""
    gh = np.repeat([7.0, 12.0, 17.0], 300)
    gl = np.repeat([6.0, 4.0, 15.0], 300)
    pack = _files(str(tmp_path), gl, gh)
    r = QF.fit_arm("A", str(tmp_path), str(tmp_path), pack, {})
    assert r["xs"] == [7.0, 12.0, 17.0] and r["raw_q20"] == [6.0, 4.0, 15.0]
    assert np.allclose(r["ys"], [5.0, 5.0, 15.0], atol=1e-9), r["ys"]
    assert np.all(np.diff(r["ys"]) >= 0)


def test_q8_margin_calib_p95_is_the_95th_percentile_not_the_90th(tmp_path):
    """Closes DISTRULE q8 (p95 read as p90).  Near rows (label gap 5..35 m): 1850 of 2000 over-read by exactly 1.0 m, 150 by exactly 10.0 m.  Sorted over-read: indices 0..1849 = 1, 1850..1999 = 10.
    numpy linear percentile: p90 at index 1799.1 -> 1.0; p95 at index 1899.05 -> 10.0; p80 -> 1.0.  So p95 = 10.0 and M95 = 10.0 while the 90th percentile would give 1.0 (tolerance 1e-6)."""
    n = 2000
    gl = np.linspace(5.0, 35.0, n)
    over = np.concatenate([np.full(1850, 1.0), np.full(150, 10.0)])
    pack = _files(str(tmp_path), gl, gl + over)
    r = MC.calib_arm("A", str(tmp_path), str(tmp_path), pack, {})
    pc = r["over_read_percentiles_m"]
    assert abs(pc["p95"] - 10.0) < 1e-6 and abs(pc["p80"] - 1.0) < 1e-6 and abs(pc["p50"] - 1.0) < 1e-6, pc
    assert r["M95"] == 10.0 and r["M80"] == 1.0
