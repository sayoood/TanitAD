"""Tests for tactical_detector.py (AMENDMENT_10 DRAFT). SYNTHETIC trajectories only; CPU only; no simulator.
Run:  CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_tactical_detector.py     (plain functions, no fixtures)
Sections: A positive controls (each of the five decisions read from a synthetic case),
          B deliberate-regression arms (mutate one threshold or rule; the matching case must go RED),
          C identity (same trajectory as human and driven => purely diagonal table),
          D reporting rules (named classes, small n, no events), E hygiene.
"""
import ast
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import tactical_detector as td  # noqa: E402

T = 170
DT = 0.1
TT = np.arange(T) * DT


def traj(v, y=None, lane=None, x0=0.0):
    v = np.broadcast_to(np.asarray(v, float), (T,)).copy()
    x = x0 + np.concatenate([[0.0], np.cumsum(v[:-1] * DT)])
    o = dict(t=TT, x=x, y=np.zeros(T) if y is None else np.asarray(y, float), yaw=np.zeros(T), v=v)
    if lane is not None:
        o["lane_offset"] = np.asarray(lane, float)
    return o


def empty_scene(**kw):
    s = dict(scenario_id="s", log_id="s", agent_id=np.zeros(0, np.int64), agent_present=np.zeros((0, T), bool),
             agent_x=np.zeros((0, T)), agent_y=np.zeros((0, T)), agent_yaw=np.zeros((0, T)), agent_v=np.zeros((0, T)),
             agent_l=np.zeros(0), agent_w=np.zeros(0), agent_cls=np.zeros(0, np.int8))
    s.update(kw)
    return s


def with_agents(scene, rows):
    """rows: list of (id, x(T), y(T), yaw, v(T))."""
    A = len(rows)
    scene = dict(scene)
    scene["agent_id"] = np.array([r[0] for r in rows], np.int64)
    scene["agent_present"] = np.ones((A, T), bool)
    scene["agent_x"] = np.array([r[1] for r in rows], float)
    scene["agent_y"] = np.array([r[2] for r in rows], float)
    scene["agent_yaw"] = np.array([np.full(T, r[3]) for r in rows], float)
    scene["agent_v"] = np.array([np.broadcast_to(r[4], (T,)) for r in rows], float)
    scene["agent_l"] = np.full(A, 4.8)
    scene["agent_w"] = np.full(A, 1.9)
    scene["agent_cls"] = np.zeros(A, np.int8)
    return scene


def with_signal(scene, state, line_s):
    scene = dict(scene)
    scene["signal_state"] = np.full((T, 1), state, np.int8)
    scene["signal_governs_ego"] = np.array([True])
    scene["stop_line_s"] = float(line_s)
    return scene


# ---------------------------------------------------------------------------------------------- cases
def case_stop():   # human stops before a red line; driven runs it
    v = np.where(TT < 3.0, 8.0, np.maximum(8.0 - 2.0 * (TT - 3.0), 0.0))
    return traj(v), traj(8.0), with_signal(empty_scene(), 1, 60.0)


def case_go():     # green line, human crosses; driven stands still
    return traj(8.0), traj(0.0), with_signal(empty_scene(), 3, 35.0)


def case_yield():  # crossing agent; human arrives after it cleared; driven is fast and enters during the conflict
    ag = (7, np.full(T, 43.0), -20.0 + 5.0 * TT, np.pi / 2, 5.0)
    sc = with_agents(empty_scene(), [ag])
    sc["conflict_zone_xy"] = np.array([[40., -2.], [46., -2.], [46., 2.], [40., 2.]])
    sc["conflict_agent_ids"] = np.array([7], np.int64)
    return traj(5.0), traj(12.0), sc


def case_follow():
    ag = (3, 30.0 + 8.0 * TT, np.zeros(T), 0.0, 8.0)
    lane_off = np.where(np.arange(T) < 50, 0.0, np.minimum((np.arange(T) - 50) * 0.5, 3.5))
    h = traj(8.0, lane=np.zeros(T))
    d_stop = traj(0.0, lane=np.zeros(T))
    d_lane = traj(8.0, y=lane_off, lane=lane_off)
    return h, d_stop, d_lane, with_agents(empty_scene(), [ag])


def case_merge():
    t = TT
    y = np.where(t < 3.0, 4.0, np.maximum(4.0 - 2.0 * (t - 3.0), 0.0))
    ag = (9, 20.0 + 8.0 * t, y, 0.0, 8.0)
    return traj(8.0), traj(14.0), traj(58.4 / 4.8), with_agents(empty_scene(), [ag])


def case_free():
    return traj(10.0), traj(0.0), traj(np.where(np.arange(T) % 2 == 0, 0.5, 2.0)), empty_scene()


def case_lane():
    off = np.clip((np.arange(T) - 60) * 0.5, 0.0, 3.5)
    return traj(10.0, y=off, lane=off), traj(10.0, lane=np.zeros(T)), traj(10.0, y=-off, lane=-off), empty_scene()


def labels(h, d, sc, cls, P=td.P0):
    rows, _ = td.score_scenario(h, d, sc, P)
    return [(r["human"], r["driven"]) for r in rows if r["cls"] == cls]


def only(lst):
    assert len(lst) == 1, lst
    return lst[0]


# ================================================================================ A  positive controls
def test_A1_line_red_stop_vs_go():
    h, d, sc = case_stop()
    assert only(labels(h, d, sc, "LINE_R")) == ("stop", "go")


def test_A2_line_green_go_vs_stop():
    h, d, sc = case_go()
    assert only(labels(h, d, sc, "LINE_G")) == ("go", "stop")


def test_A3_conflict_yield_vs_go():
    h, d, sc = case_yield()
    assert only(labels(h, d, sc, "CONFLICT")) == ("yield", "go")


def test_A4_lead_follow_vs_stop_vs_changelane():
    h, ds, dl, sc = case_follow()
    assert only(labels(h, ds, sc, "LEAD")) == ("follow", "stop")
    assert only(labels(h, dl, sc, "LEAD")) == ("follow", "change-lane")


def test_A5_merge_yield_go_contested():
    h, dgo, dcon, sc = case_merge()
    assert only(labels(h, dgo, sc, "MERGE")) == ("yield", "go")
    assert only(labels(h, dcon, sc, "MERGE")) == ("yield", "AMBIG_CONTESTED")


def test_A6_free_go_stop_slow():
    h, ds, dslow, sc = case_free()
    assert only(labels(h, ds, sc, "FREE")) == ("go", "stop")
    assert only(labels(h, dslow, sc, "FREE")) == ("go", "AMBIG_SLOW")


def test_A7_lane_change_same_wrong_none():
    h, dnone, dwrong, sc = case_lane()
    assert only(labels(h, h, sc, "LANE")) == ("change-lane", "change-lane")
    assert only(labels(h, dnone, sc, "LANE")) == ("change-lane", "go")
    assert only(labels(h, dwrong, sc, "LANE")) == ("change-lane", "AMBIG_WRONG_SIDE")


def test_A8_all_five_decisions_occur_as_human_labels():
    seen = set()
    for h, d, sc in (case_stop(), case_go(), case_yield()):
        seen |= {r[0] for c in td.CLASSES for r in labels(h, d, sc, c)}
    h, _, _, sc = case_follow(); seen |= {r[0] for r in labels(h, h, sc, "LEAD")}
    h, _, _, sc = case_lane(); seen |= {r[0] for r in labels(h, h, sc, "LANE")}
    assert set(td.DECISIONS) <= seen, seen


def test_A9_not_engaged_when_driven_never_gets_within_40m_of_the_line():
    h = traj(8.0)
    sc = with_signal(empty_scene(), 3, 100.0)
    assert only(labels(h, traj(0.0), sc, "LINE_G")) == ("go", "NOT_ENGAGED")


def test_A10_ref_invalid_cuts_station_based_events():
    h, d, sc = case_go()
    assert len(labels(h, d, sc, "LINE_G")) == 1
    sc2 = dict(sc, human_ref_valid_until_s=1.5)       # before the 2.0 s warm-up and before any trigger
    rows, audit = td.score_scenario(h, d, sc2)
    assert [r for r in rows if r["cls"] == "LINE_G"] == [] and audit["ref_invalid_ticks"] > 0


def test_A11_lane_class_not_computable_without_lane_offset():
    h, _, _, sc = case_lane()
    h2 = {k: v for k, v in h.items() if k != "lane_offset"}
    ev, audit = td.find_events(h2, sc)
    assert not any(e["cls"] == "LANE" for e in ev) and audit["lane_change_available"] is False
    e = dict(cls="LANE", i0=30, i1=100, key="LANE:30:", meta=dict(sign=1))
    assert td.decide(e, h2, sc, td.P0, td.RefPath(h["x"], h["y"], h["v"])) == "NOT_COMPUTABLE"


# ================================================================================ B  deliberate-regression arms
def mut(**kw):
    P = dict(td.P0)
    P.update(kw)
    return P


def _red(case_fn, cls, expect, P):
    """True if the case still reads `expect` under P."""
    got = labels(*case_fn(), cls, P)
    return got == [expect]


def test_B01_stop_v_never_stops():
    assert _red(case_stop, "LINE_R", ("stop", "go"), td.P0)
    assert not _red(case_stop, "LINE_R", ("stop", "go"), mut(STOP_V=-1.0))


def test_B02_stop_min_ticks_unreachable():
    assert not _red(case_stop, "LINE_R", ("stop", "go"), mut(STOP_MIN_TICKS=10000))


def test_B03_near_m_zero_kills_line_trigger():
    assert not _red(case_stop, "LINE_R", ("stop", "go"), mut(NEAR_M=0.0))


def test_B04_crossing_ignored():
    assert not _red(case_go, "LINE_G", ("go", "stop"), mut(MUT_CROSS_IGNORED=True))


def test_B05_zone_ttz_negative():
    assert not _red(case_yield, "CONFLICT", ("yield", "go"), mut(ZONE_TTZ_S=-1.0))


def test_B06_follow_frac_unreachable():
    h, ds, dl, sc = case_follow()
    assert labels(h, ds, sc, "LEAD") == [("follow", "stop")]
    assert labels(h, ds, sc, "LEAD", mut(FOLLOW_FRAC=1.5)) != [("follow", "stop")]


def test_B07_go_frac_unreachable():
    h, ds, dslow, sc = case_free()
    assert labels(h, ds, sc, "FREE", mut(GO_FRAC=1.5)) != [("go", "stop")]


def test_B08_lane_half_huge_kills_lane_and_lead_lane():
    h, dn, dw, sc = case_lane()
    assert labels(h, h, sc, "LANE", mut(LANE_HALF_M=100.0)) != [("change-lane", "change-lane")]
    h, ds, dl, sc = case_follow()
    assert labels(h, dl, sc, "LEAD", mut(LANE_HALF_M=100.0)) != [("follow", "change-lane")]


def test_B09_merge_order_flip():
    h, dgo, dcon, sc = case_merge()
    assert labels(h, dgo, sc, "MERGE", mut(MUT_MERGE_FLIP=True)) != [("yield", "go")]


def test_B10_stop_precedence_over_lane():
    h, ds, dl, sc = case_follow()
    off = np.where(np.arange(T) < 40, 0.0, 3.5)
    both = traj(np.where(np.arange(T) < 40, 8.0, 0.0), y=off, lane=off)
    assert labels(h, both, sc, "LEAD") == [("follow", "stop")]
    assert labels(h, both, sc, "LEAD", mut(MUT_LANE_BEFORE_STOP=True)) == [("follow", "change-lane")]


def test_B11_merge_thresholds():
    h, dgo, dcon, sc = case_merge()
    for kw in (dict(MERGE_GAP=0.1), dict(MERGE_OFF_ENTER=-1.0), dict(MERGE_OFF_START=100.0)):
        assert labels(h, dgo, sc, "MERGE", mut(**kw)) == [], kw


def test_B12_lead_thresholds():
    h, ds, dl, sc = case_follow()
    for kw in (dict(LEAD_STEADY_TICKS=1000), dict(LEAD_GAP_MAX=10.0), dict(WARMUP_S=100.0)):
        assert labels(h, ds, sc, "LEAD", mut(**kw)) == [], kw


def test_B13_free_min_and_warmup():
    h, ds, dslow, sc = case_free()
    for kw in (dict(FREE_MIN_TICKS=1000), dict(WARMUP_S=100.0)):
        assert labels(h, ds, sc, "FREE", mut(**kw)) == [], kw


def test_B14_persistence_kills_lane_runs():
    h, dn, dw, sc = case_lane()
    assert labels(h, h, sc, "LANE", mut(PERSIST_TICKS=10000)) == []


def test_B15_lane_tolerance_window_matters():
    # human departs the lane for ticks 64..80 only (window = 24..120 with the 4.0 s tolerance); a driven change at 130+ is
    # outside it and must not match; one at 90 is inside; a 1-tick tolerance shrinks the window away from it.
    i = np.arange(T)
    hoff = np.where(i < 80, np.clip((i - 60) * 0.5, 0.0, 3.5), 0.0)
    h = traj(10.0, y=hoff, lane=hoff)
    sc = empty_scene()
    base = labels(h, h, sc, "LANE")
    assert base == [("change-lane", "change-lane")]
    late = np.where(i >= 130, 3.5, 0.0)
    assert labels(h, traj(10.0, y=late, lane=late), sc, "LANE") == [("change-lane", "go")]
    ok = np.where((i >= 90) & (i < 110), 3.5, 0.0)
    assert labels(h, traj(10.0, y=ok, lane=ok), sc, "LANE") == [("change-lane", "change-lane")]
    assert labels(h, traj(10.0, y=ok, lane=ok), sc, "LANE", mut(LANE_TOL_TICKS=1)) == [("change-lane", "go")]


# ================================================================================ C  identity
def _all_cases():
    out = []
    out.append(case_stop())
    out.append(case_go())
    out.append(case_yield())
    h, ds, dl, sc = case_follow(); out.append((h, h, sc))
    h, dg, dc, sc = case_merge(); out.append((h, h, sc))
    h, ds, dsl, sc = case_free(); out.append((h, h, sc))
    h, dn, dw, sc = case_lane(); out.append((h, h, sc))
    return out


def test_C1_identity_is_purely_diagonal():
    rows = []
    for k, (h, d, sc) in enumerate(_all_cases()):
        sc = dict(sc, log_id=f"c{k}")
        r, _ = td.score_scenario(h, h, sc)          # the SAME trajectory as human and driven
        rows += r
    assert len(rows) >= 7
    tab = td.confusion(rows)
    M = np.array(tab["counts"])
    assert M.sum() == len(rows) and M.sum() == np.trace(M), "off-diagonal mass in the identity table"
    assert tab["agreement"] == 1.0
    for c in td.CLASSES:
        Mc = np.array(td.confusion(rows, c)["counts"])
        assert Mc.sum() == np.trace(Mc)


def test_C2_identity_arm_catches_a_nondeterministic_rule():
    # deliberate regression: if the driven run were scored with different thresholds the identity table would break.
    h, d, sc = case_stop()
    ev, _ = td.find_events(h, sc)
    path = td.RefPath(h["x"], h["y"], h["v"])
    a = [td.decide(e, h, sc, td.P0, path) for e in ev]
    b = [td.decide(e, h, sc, mut(STOP_V=-1.0), path) for e in ev]
    assert a != b


# ================================================================================ D  reporting rules
def _row(c, k, cls, h, d):
    return dict(cluster=c, key=f"{c}|{k}", cls=cls, human=h, driven=d)


def test_D1_named_classes_are_counted_never_dropped():
    rows = [_row("a", 1, "FREE", "go", "AMBIG_SLOW"), _row("a", 2, "FREE", "go", "go"),
            _row("b", 1, "LEAD", "AMBIG_LEAD_LOST", "follow"), _row("b", 2, "LINE_G", "go", "NOT_ENGAGED")]
    t = td.confusion(rows)
    assert t["n"] == 4 and np.array(t["counts"]).sum() == 4
    assert t["n_resolved"] == 1 and t["n_diag"] == 1
    assert t["n_human_named"] == 1 and t["n_driven_named_given_human_resolved"] == 2


def test_D2_small_n_reports_counts_only():
    rows = [_row(f"c{i}", 1, "FREE", "go", "go") for i in range(5)]
    rep = td.report(rows)
    assert rep["classes"]["FREE"]["interval"] is None
    assert rep["classes"]["FREE"]["status"].startswith("INSUFFICIENT_N")
    assert rep["status"] == "INSUFFICIENT_N" and rep["pooled"]["n"] == 5


def test_D3_no_events_is_not_a_pass():
    rep = td.report([])
    assert rep["status"].startswith("NO_EVENTS") and rep["classes"] == {}


def test_D4_bootstrap_interval_when_n_allows_and_is_deterministic():
    rows = [_row(f"c{i}", 1, "FREE", "go", "go" if i % 4 else "stop") for i in range(40)]
    r1 = td.bootstrap_agreement(rows)
    r2 = td.bootstrap_agreement(rows)
    assert r1 == r2 and r1["n_clusters"] == 40 and r1["lo"] <= 0.75 <= r1["hi"]
    rep = td.report(rows)
    assert rep["classes"]["FREE"]["status"] == "OK"


def test_D5_paired_bootstrap_separates_a_real_difference_and_pairs_on_event_key():
    base = [_row(f"c{i}", 1, "FREE", "go", "go" if i % 2 else "stop") for i in range(30)]
    cand = [_row(f"c{i}", 1, "FREE", "go", "go") for i in range(30)]
    r = td.paired_bootstrap_diff(cand, base)
    assert r["status"] == "OK" and r["point"] > 0.4 and r["lo"] > 0 and r["n_unpaired"] == 0
    r2 = td.paired_bootstrap_diff(cand[:5], base[:5])
    assert r2["status"] == "INSUFFICIENT_N"
    r3 = td.paired_bootstrap_diff(cand, base[:20])
    assert r3["n_unpaired"] == 10


def test_D6_loader_downsamples_20hz_to_10hz():
    n = 2 * T
    t20 = np.arange(n) * 0.05
    z = np.zeros(n)
    ro = dict(scenario_id="x", t=t20, ego_x=t20 * 8, ego_y=z, ego_yaw=z, ego_v=z + 8, agent_id=np.array([1]),
              agent_present=np.ones((1, n), bool), agent_x=np.ones((1, n)) * 30, agent_y=np.zeros((1, n)),
              agent_yaw=np.zeros((1, n)), agent_v=np.zeros((1, n)), agent_l=np.array([4.8]), agent_w=np.array([1.9]),
              signal_state=np.full((n, 1), 3, np.int8), signal_governs_ego=np.array([True]), stop_line_s=np.array(35.0))
    h, d, sc = td.from_rollout(ro, ro)
    assert len(h["t"]) == T and np.allclose(np.diff(h["t"]), 0.1) and sc["agent_x"].shape == (1, T) and sc["signal_state"].shape == (T, 1)


# ================================================================================ E  hygiene
def test_E1_no_simulator_import_and_scorer_is_pinned():  # changed for the MM ruling of 2026-11-09 (was: no scorer import)
    src = (Path(__file__).resolve().parent / "tactical_detector.py").read_text(encoding="utf-8")
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert mods <= {"__future__", "math", "numpy", "behaviour_score", "hashlib", "os", "sys", "pathlib"}, mods
    assert td.SCORER_SHA256 == "69d13d712ca6e6edd42ea7d91f8ac13ce0ddb815f480eed4280c320bdde3192a"


def test_E2_inputs_not_mutated():
    h, d, sc = case_stop()
    h0, d0 = copy.deepcopy(h), copy.deepcopy(d)
    td.score_scenario(h, d, sc)
    assert all(np.array_equal(h[k], h0[k]) for k in h) and all(np.array_equal(d[k], d0[k]) for k in d)


def test_E3_events_do_not_depend_on_the_driven_run():
    h, d1, sc = case_stop()
    e1, _ = td.find_events(h, sc)
    e2, _ = td.find_events(h, sc)
    assert [x["key"] for x in e1] == [x["key"] for x in e2]
    import inspect
    assert "driven" not in inspect.signature(td.find_events).parameters
