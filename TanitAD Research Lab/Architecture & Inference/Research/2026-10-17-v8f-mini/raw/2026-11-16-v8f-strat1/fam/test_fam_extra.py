"""test_fam_extra.py -- synthetic inputs reach every branch of fam_extra; a mutation per formula goes RED; zero controls read exactly 0.0.
CPU, numpy only (no torch, no dumps).  Run:  python -m pytest test_fam_extra.py -q --basetemp F:/TanitAD-scratch/strat1/fam/tmp/pt
"""
import os
import re
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
READ = "F:/TanitAD-scratch/strat1/read"
sys.path.insert(0, HERE)
sys.path.insert(0, READ)
sys.path.insert(0, READ + "/tree/taniteval")
import fam_extra as FE  # noqa: E402
from taniteval import ci as CI  # noqa: E402
from taniteval import lead_metrics as LM  # noqa: E402

SRC = open(os.path.join(HERE, "fam_extra.py"), encoding="utf-8").read()
NB = 200


# ---------------------------------------------------------------------------------------------------------------------------------------------- fixtures
def straight(n, speed_step, k=30):
    """[n, k, 2] straight path along +x, speed_step metres per step"""
    x = speed_step * np.arange(1, k + 1, dtype=np.float64)
    return np.tile(np.stack([x, np.zeros(k)], 1)[None], (n, 1, 1))


def world(n=6, lead_x=100.0, with_lead=True):
    """n windows in 3 episodes, a STATIONARY lead at x = lead_x (size 4 m), ego v0 = 10 m/s"""
    leads = np.tile(np.array([lead_x, 0.0])[None, None], (n, 30, 1)) if with_lead else np.full((n, 30, 2), np.nan)
    J = {"leads": leads, "lead_lens": np.full(n, 4.0 if with_lead else np.nan), "speeds": np.full(n, 10.0), "is_lead": np.full(n, with_lead)}
    return J, np.repeat(np.arange(3), n // 3)


def dk_known(m):
    """known values: ego 2 m/step (10 m/s, dt 0.2), lead at 100 m, size 4 -> min gap at the last knot 100 - 60 - 2 = 38 m, time gap 3.8 s, min-TTC 4.0 s (last closing step starts at gap 40, closes at 10 m/s)"""
    J, eid = world()
    g = straight(6, 2.0)
    out, raw = m.dk_arm(LM, CI, g, g, J, eid, (0, 30), NB)
    lv = out["levels"]
    assert lv["headway_min_m"]["mean"] == pytest.approx(38.0) and lv["time_gap_min_s"]["mean"] == pytest.approx(3.8) and lv["min_ttc_s"]["mean"] == pytest.approx(4.0)
    assert out["n_headway"] == 6 and out["n_time_gap"] == 6 and out["n_closing"] == 6 and out["n_lead_windows"] == 6
    assert out["levels"]["headway_min_m"]["n_episodes"] == 3
    return out, raw, J, eid, g


def tac_known(m):
    gt = np.array([0, 0, 1, 1, 2, 2, 2, -100])
    pr = np.array([0, 1, 1, 1, 2, 0, 2, 1])
    cm = m.confusion(gt, pr, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 2]]
    s = m.conf_stats(cm)
    assert s["n"] == 7 and s["accuracy"] == pytest.approx(5 / 7) and s["kappa"] == pytest.approx(19 / 33) and s["macro_f1"] == pytest.approx(0.7)
    assert s["recall"].tolist() == pytest.approx([0.5, 1.0, 2 / 3]) and s["precision"].tolist() == pytest.approx([0.5, 2 / 3, 1.0])
    assert s["f1"].tolist() == pytest.approx([0.5, 0.8, 0.8])
    # an absent class is NaN, not 0, and does not enter the macro mean
    s2 = m.conf_stats(m.confusion(np.array([0, 0, 1]), np.array([0, 1, 1]), 3))
    assert np.isnan(s2["recall"][2]) and np.isnan(s2["f1"][2]) and s2["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 2)


def goal_known(m):
    P = np.zeros((4, 22), bool)
    Y = np.zeros((4, 22), bool)
    W = np.zeros((4, 22), bool)
    W[:, :2] = True
    Y[0, 0] = Y[1, 0] = Y[2, 1] = True
    P[0, 0] = P[2, 0] = P[2, 1] = True
    f1, tp, fp, fn = m._goal_f1(P, Y, W)
    assert f1[0] == pytest.approx(0.5) and f1[1] == pytest.approx(1.0) and np.isnan(f1[2:]).all()
    bits = (P.astype(np.int64) << np.arange(22)).sum(1)
    st = m.goal_stat_fn(Y, W)([(i << m.SHIFT) | int(b) for i, b in enumerate(bits)])
    assert st["goal_macro_f1"] == pytest.approx(0.75) and st["goal_micro_token_accuracy"] == pytest.approx(0.75)
    gp = np.zeros((4, 22))
    gp[0, :2] = [0.9, 0.1]
    gp[1, :2] = [0.2, 0.8]
    gp[2, :2] = [0.3, 0.7]
    gp[3, :2] = [0.6, 0.4]
    gp[0, 5] = 0.99                                                             # an UNWEIGHTED token with the top probability must not be selected
    hit = m._top1(gp, Y, W)
    assert hit[:3].tolist() == [1.0, 0.0, 1.0] and np.isnan(hit[3])


def selexe_known(m):
    agree, info = m.selexe_lat(np.array([0, 1, 2, 3, 4, 1]), np.array([0, 1, 1, 0, 0, 2]))
    assert info["n_mapped"] == 4 and info["n_excluded_lane_change"] == 2
    assert np.nanmean(agree) == pytest.approx(0.5) and np.isnan(agree[3]) and np.isnan(agree[4]) and agree[:3].tolist() == [1.0, 1.0, 0.0]
    t = m.selexe_lon_table(np.array([0, 2, 2, 6]), np.array([1, 0, 0, 2]))
    assert t["counts"][2][0] == 2 and t["counts"][0][1] == 1 and t["counts"][6][2] == 1 and t["n"] == 4


def code_known(m):
    """the item code round-trips: a block built through the bootstrap path reads the closed-form value on the full set"""
    gt = np.array([0, 0, 1, 1, 2, 2, 2, 0])
    pr = np.array([0, 1, 1, 1, 2, 0, 2, 0])
    eid = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    b = m.conf_block(CI, gt, pr, 3, ("a", "b", "c"), eid, 50)
    assert b["accuracy"]["point"] == pytest.approx(6 / 8) and b["n"] == 8 and b["recall"]["b"]["point"] == pytest.approx(1.0)
    assert b["confusion_rows_GT_cols_pred"] == [[2, 1, 0], [0, 2, 0], [1, 0, 2]]
    assert b["majority_baseline"]["class"] == "a" or b["majority_baseline"]["class"] == "c"


def human_known(m):
    J, eid = world()
    g = straight(6, 2.0)
    r = m.human_control(LM, g, J, eid, (0, 30))
    assert r["PASS"] and r["n_viol"] == 0 and r["lead_progress_ratio"] == 1.0
    # a plan twice as fast closes inside 1 s of the lead while the human keeps 3.8 s -> every at-risk window violates
    fast = straight(6, 4.0)
    out, raw = m.dk_arm(LM, CI, fast, g, J, eid, (0, 30), NB)
    assert out["DK-VIOL"]["n_viol"] == 6 and out["DK-VIOL"]["n_at_risk"] == 6 and out["DK-VIOL"]["share_of_at_risk"] == pytest.approx(1.0)


def lost_known(m):
    J, eid = world()
    g = straight(6, 2.0)
    plan = g.copy()
    plan[:, :, 1] = 30.0                                                       # the arm leaves the corridor: the lead is lost, never scored as free flow
    out, _ = m.dk_arm(LM, CI, plan, g, J, eid, (0, 30), NB)
    assert out["n_lead_lost"] == 6 and out["levels"]["headway_min_m"]["status"] == "NOT COMPUTABLE"
    ok, _ = m.dk_arm(LM, CI, g, g, J, eid, (0, 30), NB)
    assert ok["n_lead_lost"] == 0


def pair_known(m):
    J, eid = world()
    g = straight(6, 2.0)
    _, a = m.dk_arm(LM, CI, straight(6, 3.0), g, J, eid, (0, 30), NB)        # faster -> tighter
    _, b = m.dk_arm(LM, CI, g, g, J, eid, (0, 30), NB)
    p = m.dk_pair(LM, CI, a, b, eid, ("fast", "human"), {"headway_min_m": 1.0, "time_gap_min_s": 100.0, "min_ttc_s": 0.0, "dk_viol_share_of_at_risk": 0.0}, NB)
    h = p["levels"]["headway_min_m"]
    assert h["status"] == "OK" and h["n_windows"] == 6 and h["delta"] < 0 and h["above_floor"] is True
    assert p["levels"]["time_gap_min_s"]["above_floor"] is False
    assert p["DK-VIOL"]["above_floor"] is not None


def human_stub_known(m):
    """the control's PASS rule, read through a stub instrument (the real one cannot return n_viol > 0 or a ratio below 1 for human-vs-human)"""
    def stub(n_viol, ratio):
        return types.SimpleNamespace(distance_keeping=lambda *a, **k: {}, path_arc_length=lambda x: np.ones(len(x)),
                                     lead_viol=lambda *a, **k: {"n_viol": n_viol, "lead_progress_ratio": ratio, "n_at_risk": 3})
    J, eid = world()
    g = straight(6, 2.0)
    assert m.human_control(stub(0, 1.0), g, J, eid, (0, 30))["PASS"] is True
    assert m.human_control(stub(3, 1.0), g, J, eid, (0, 30))["PASS"] is False
    assert m.human_control(stub(0, 0.9), g, J, eid, (0, 30))["PASS"] is False


CHECKS = (dk_known, tac_known, goal_known, selexe_known, code_known, human_known)


# ---------------------------------------------------------------------------------------------------------------------------------------------- tests
def test_dk_levels_known_values():
    dk_known(FE)


def test_dk_viol_and_human_control():
    human_known(FE)


def test_tac_confusion_known_values():
    tac_known(FE)


def test_goal_known_values():
    goal_known(FE)


def test_selexe_known_values():
    selexe_known(FE)


def test_item_code_roundtrip():
    code_known(FE)


def test_zero_control_dk_self_pair_reads_exactly_zero():
    out, raw, J, eid, g = dk_known(FE)
    p = FE.dk_pair(LM, CI, raw, raw, eid, ("A", "A"), None, NB)
    for k, v in p["levels"].items():
        assert v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0, k
    assert p["DK-VIOL"]["at_risk"]["delta"] == 0.0


def test_dk_pair_direction_n_used_and_floor():
    J, eid = world()
    g = straight(6, 2.0)
    _, a = FE.dk_arm(LM, CI, straight(6, 3.0), g, J, eid, (0, 30), NB)        # faster -> tighter
    _, b = FE.dk_arm(LM, CI, g, g, J, eid, (0, 30), NB)
    p = FE.dk_pair(LM, CI, a, b, eid, ("fast", "human"), {"headway_min_m": 1.0, "time_gap_min_s": 100.0, "min_ttc_s": 0.0, "dk_viol_share_of_at_risk": 0.0}, NB)
    h = p["levels"]["headway_min_m"]
    assert h["status"] == "OK" and h["n_windows"] == 6 and h["delta"] < 0 and h["above_floor"] is True
    assert p["levels"]["time_gap_min_s"]["above_floor"] is False                # |delta| < floor 100
    assert p["levels"]["min_ttc_s"]["above_floor"] is True                      # floor 0.0: any delta counts (only a zero delta would not)
    assert p["DK-VIOL"]["above_floor"] is not None


def test_dk_not_computable_without_lead():
    J, eid = world(with_lead=False)
    g = straight(6, 2.0)
    out, raw = FE.dk_arm(LM, CI, g, g, J, eid, (0, 30), NB)
    assert out["levels"]["headway_min_m"]["status"] == "NOT COMPUTABLE" and out["levels"]["headway_min_m"]["n_windows"] == 0
    assert out["n_lead_windows"] == 0 and out["n_headway"] == 0 and out["n_closing"] == 0
    assert out["DK-VIOL"].get("status") in ("NOT-APPLICABLE", "NOT COMPUTABLE") or out["DK-VIOL"].get("n_lead_windows") == 0
    p = FE.dk_pair(LM, CI, raw, raw, eid, ("A", "B"), None, NB)
    assert p["levels"]["headway_min_m"]["status"] == "NOT-APPLICABLE"


def test_dk_lead_lost_is_counted():
    J, eid = world()
    g = straight(6, 2.0)
    plan = g.copy()
    plan[:, :, 1] = 30.0                                                       # the arm leaves the corridor: the lead is lost, never scored as free flow
    out, _ = FE.dk_arm(LM, CI, plan, g, J, eid, (0, 30), NB)
    assert out["n_lead_lost"] == 6 and out["levels"]["headway_min_m"]["status"] == "NOT COMPUTABLE"


def test_conf_block_branches_and_pair_zero_control():
    gt = np.array([0, 0, 1, 1, 2, 2, 2, 0, 1, 2, 0, 0])
    pa = np.array([0, 1, 1, 1, 2, 0, 2, 0, 1, 2, 0, 1])
    pb = np.array([0, 0, 1, 0, 2, 0, 2, 0, 1, 1, 0, 1])
    eid = np.repeat(np.arange(4), 3)
    names = ["a", "b", "c"]
    z = FE.conf_pair(CI, gt, pa, pa, 3, names, eid, None, NB)
    for k, v in z.items():
        assert v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0, k
    d = FE.conf_pair(CI, gt, pa, pb, 3, names, eid, {"accuracy": 0.01}, NB)
    assert d["accuracy"]["delta"] == pytest.approx(9 / 12 - 8 / 12, abs=1e-4) and d["accuracy"]["above_floor"] is True and d["kappa"]["above_floor"] is None
    # nothing labelled -> NOT COMPUTABLE with n
    nb = FE.conf_block(CI, np.full(4, -100), np.zeros(4, int), 3, names, np.arange(4), NB)
    assert nb["status"] == "NOT COMPUTABLE" and nb["n"] == 0
    # a class absent from the labels carries no recall cell and no pair cell
    b2 = FE.conf_block(CI, np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 3, names, np.array([0, 0, 1, 1]), NB)
    assert b2["recall"]["c"] is None
    p2 = FE.conf_pair(CI, np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]), 3, names, np.array([0, 0, 1, 1]), None, NB)
    assert "recall_c" not in p2


def test_goal_block_and_pair_and_not_computable():
    N = 12
    rng = np.random.default_rng(0)
    gp = rng.random((N, 22))
    y = np.zeros(N, np.int64)
    w = np.zeros(N, np.int64)
    y[:6] = 1
    w[:] = 3
    has = np.ones(N, bool)
    eid = np.repeat(np.arange(4), 3)
    blk, (P, Y, W) = FE.goal_block(CI, gp, y, w, has, eid, NB)
    assert blk["n_weighted_cells"] == 2 * N and blk["n_positive_per_token"]["FOLLOW_LANE"] == 6 and blk["f1_per_token"]["MERGE"] is None
    assert blk["top1_in_positives"]["n_windows"] == 6
    bits = (P.astype(np.int64) << np.arange(22)).sum(1)
    z = FE.goal_pair(CI, bits, bits, Y, W, eid, None, NB)
    for k, v in z.items():
        assert v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0, k
    nc = FE.goal_block(CI, gp, np.zeros(N, np.int64), w, has, eid, NB)
    assert nc[0]["status"] == "NOT COMPUTABLE"
    nh = FE.goal_block(CI, gp, y, w, np.zeros(N, bool), eid, NB)                # has = False masks every cell
    assert nh[0]["status"] == "NOT COMPUTABLE"


def test_tick0_heads_order_and_refusals(tmp_path):
    n = 5
    ci = np.array([0, 0, 1, 1, 2])
    ws = np.array([4, 8, 4, 8, 4])
    tk = np.tile(np.array([0, 1, 2]), n)
    rows_ci, rows_t0 = np.repeat(ci, 3), np.repeat(ws, 3)
    perm = np.random.default_rng(1).permutation(len(tk))                       # rows are not in window order
    lat = np.repeat(np.arange(n)[:, None] * 1.0, 5, 1)
    lat = np.repeat(lat, 3, 0)
    lat[tk == 1] += 100.0                                                       # a later tick must never be picked
    p = tmp_path / "t.npz"
    np.savez(p, tick=tk[perm], ci=rows_ci[perm], t0=rows_t0[perm], t_lat_logits=lat[perm], t_lon_logits=np.zeros((15, 7)), t_goal_prob=np.zeros((15, 22)))
    h = FE.tick0_heads(str(p), ci, ws)
    assert h["lat"][:, 0].tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(RuntimeError):
        FE.tick0_heads(str(p), ci, np.array([4, 8, 4, 8, 12]))                 # a window with no tick-0 row
    with pytest.raises(RuntimeError):
        FE.tick0_heads(str(p), ci[:4], ws[:4])                                 # more tick-0 rows than windows


def test_v9_rows_masks_unlabelled_windows():
    join = {"off": np.array([0, 3]), "has": np.array([1, 1, 0, 1, 1, 1], bool), "lat_cls": np.array([0, 1, 2, 2, -100, 0]), "lon_cls": np.array([3, 4, 5, 6, 0, 1]),
            "goal_y": np.arange(6), "goal_w": np.arange(6) + 10}
    v = FE.v9_rows(join, np.array([0, 0, 1, 1]), np.array([0, 2, 0, 1]))
    assert v["rows"].tolist() == [0, 2, 3, 4]
    assert v["lat"].tolist() == [0, FE.IGNORE, 2, FE.IGNORE] and v["lon"].tolist() == [3, FE.IGNORE, 6, 0]
    assert v["has"].tolist() == [True, False, True, True]


def test_arms_present_requires_dump_ticks_and_done(tmp_path):
    d = str(tmp_path)
    assert FE.arms_present(d) == []
    os.makedirs(FE.dump_dir(d, "C0_s0"))
    assert FE.arms_present(d) == []
    os.makedirs(d + "/ticks", exist_ok=True)
    open(FE.ticks_path(d, "C0_s0"), "wb").close()
    assert FE.arms_present(d) == []
    open(f"{d}/done_STRAT1-C0_s0_m10_causal_heads.txt", "w").close()
    assert FE.arms_present(d) == ["C0_s0"]


def test_load_dump_refuses_wrong_episode_count(tmp_path):
    with pytest.raises(RuntimeError):
        FE.load_dump(str(tmp_path), 141)


def test_reader_rows_lists_every_family_cell_unpooled():
    res = {"pairs": {"A-B": {"DK": {"operative_0_2s": {"levels": {"headway_min_m": {"status": "OK", "delta": -1.0, "lo": -2.0, "hi": 0.5, "above_floor": True, "n_windows": 9},
                                                                    "min_ttc_s": {"status": "NOT-APPLICABLE"}},
                                                      "DK-VIOL": {"at_risk": {"delta": 0.1, "lo": 0.0, "hi": 0.2, "n_at_risk": 7}, "above_floor": False}}},
                             "TAC": {"TAC-SEL-LAT": {"accuracy": {"delta": 0.02, "lo": -0.01, "hi": 0.04, "above_floor": None, "n_windows": 100}, "recall_TURN_L": {"delta": float("nan")}},
                                     "TAC-SELEXE-LAT": {"status": "OK", "delta": 0.0, "lo": 0.0, "hi": 0.0, "n_windows": 50}}}}}
    rows = FE.reader_rows(res, "A-B")
    fams = {r[0] for r in rows}
    assert fams == {"LONGITUDINAL-DK", "TACTICAL"} and len(rows) == 4
    assert ("LONGITUDINAL-DK", "operative_0_2s:DK-VIOL share_of_at_risk", 0.1, 0.0, 0.2, False, 7) in rows


# ---------------------------------------------------------------------------------------------------------------------------------------------- mutations: one per formula
MUTANTS = [
    ("accuracy denominator", "float(tp.sum() / n) if n", "float(tp.sum() / (n + 1)) if n"),
    ("kappa numerator", "kap = float((acc - pe) / (1 - pe))", "kap = float((acc + pe) / (1 - pe))"),
    ("expected agreement", "pe = float((sup * pred).sum() / (n * n))", "pe = float((sup * sup).sum() / (n * n))"),
    ("recall", "rec = np.where(sup > 0, tp / sup, np.nan)", "rec = np.where(sup > 0, tp / np.maximum(pred, 1), np.nan)"),
    ("precision", "prec = np.where(pred > 0, tp / pred, np.nan)", "prec = np.where(pred > 0, tp / np.maximum(sup, 1), np.nan)"),
    ("F1", "2.0 * tp / (sup + pred)", "1.0 * tp / (sup + pred)"),
    ("macro over present classes only", "np.where(sup > 0, np.where(tp == 0, 0.0, 2.0 * tp / (sup + pred)), np.nan)", "np.where(sup >= 0, np.where(tp == 0, 0.0, 2.0 * tp / (sup + pred + (sup == 0))), np.nan)"),
    ("confusion orientation", "yt[m] * K + yp[m]", "yp[m] * K + yt[m]"),
    ("ignore mask", "m = (yt != ignore) & (yt >= 0) & (yt < K) & (yp >= 0) & (yp < K)", "m = (yt >= -1000) & (yt < K) & (yp >= 0) & (yp < K)"),
    ("goal F1", "2 * tp / (2 * tp + fp + fn)", "tp / (2 * tp + fp + fn)"),
    ("goal fn", "fn = (~P & Y & W).sum(0)", "fn = (~P & ~Y & W).sum(0)"),
    ("goal micro accuracy", "((P == Y[idx]) & W[idx]).sum() / wsum", "((P == Y[idx]) & W[idx]).sum() / (wsum + 1)"),
    ("goal top-1", "hit = np.where(has_pos, (Y & W)[np.arange(len(top)), top].astype(np.float64), np.nan)", "hit = np.where(has_pos, (~Y & W)[np.arange(len(top)), top].astype(np.float64), np.nan)"),
    ("goal weighting", "gp = np.where(W, np.asarray(goal_prob, np.float64), -np.inf)", "gp = np.asarray(goal_prob, np.float64)"),
    ("lateral agreement", "agree[m] = (mapped == exe[m]).astype(np.float64)", "agree[m] = (mapped != exe[m]).astype(np.float64)"),
    ("lateral mapping", "LAT_HEAD_TO_KIN = {0: 0, 1: 1, 2: 2}", "LAT_HEAD_TO_KIN = {0: 0, 1: 2, 2: 1}"),
    ("lane-change exclusion", "m = np.isin(sel, list(LAT_HEAD_TO_KIN))", "m = np.isin(sel, list(range(5)))"),
    ("item decode", "idx, p = v >> SHIFT, v & MASK", "idx, p = v & MASK, v >> SHIFT"),
    ("dk horizon band", "p = np.asarray(track, np.float64)[:, a:b]", "p = np.asarray(track, np.float64)[:, a:b + 0][:, ::-1]"),
    ("dk human path", "h = np.asarray(g, np.float64)[:, a:b]\n    dk =", "h = np.asarray(track, np.float64)[:, a:b]\n    dk ="),
    ("dk lead windows", "np.sum(J[\"is_lead\"])", "np.sum(~J[\"is_lead\"])"),
    ("dk lead lost", "J[\"is_lead\"] & ~np.isfinite(", "J[\"is_lead\"] & np.isfinite("),
    ("floor flag", "r[\"above_floor\"] = None if f is None else bool(abs(r[\"delta\"]) > f)", "r[\"above_floor\"] = None if f is None else bool(abs(r[\"delta\"]) < f)"),
    ("human control ratio", "bool(lv.get(\"n_viol\") == 0 and lv.get(\"lead_progress_ratio\") == 1.0)", "bool(lv.get(\"n_viol\") == 0)"),
    ("human control n_viol", "bool(lv.get(\"n_viol\") == 0 and lv.get(\"lead_progress_ratio\") == 1.0)", "bool(lv.get(\"lead_progress_ratio\") == 1.0)"),
    ("tick-0 pick", "m = np.asarray(z[\"tick\"]) == 0", "m = np.asarray(z[\"tick\"]) == 1"),
    ("v9 row", "r = L.rows_of(join[\"off\"], clip_index, ws)", "r = L.rows_of(join[\"off\"], clip_index, ws) + 1"),
    ("v9 has mask", "np.where(has & (lat >= 0), lat, IGNORE)", "np.where(lat >= 0, lat, IGNORE)"),
]

ALL_CHECKS = (dk_known, tac_known, goal_known, selexe_known, code_known, human_known, lost_known, pair_known, human_stub_known)


def _mutated_module(old, new):
    assert SRC.count(old) == 1, f"mutation anchor not unique ({SRC.count(old)}): {old[:60]}"
    mod = types.ModuleType("fam_extra_mut")
    mod.__file__ = os.path.join(HERE, "fam_extra.py")
    exec(compile(SRC.replace(old, new), "fam_extra_mut.py", "exec"), mod.__dict__)
    return mod


def _extra_checks(m):
    """branch checks that need no fixture but still read formulas the mutants touch"""
    v = m.v9_rows({"off": np.array([0, 3]), "has": np.array([1, 1, 0, 1, 1, 1], bool), "lat_cls": np.array([0, 1, 2, 2, -100, 0]), "lon_cls": np.array([3, 4, 5, 6, 0, 1]),
                   "goal_y": np.arange(6), "goal_w": np.arange(6) + 10}, np.array([0, 0, 1, 1]), np.array([0, 2, 0, 1]))
    assert v["rows"].tolist() == [0, 2, 3, 4] and v["lat"].tolist() == [0, m.IGNORE, 2, m.IGNORE]


@pytest.mark.parametrize("name,old,new", MUTANTS, ids=[x[0] for x in MUTANTS])
def test_every_formula_mutation_goes_red(name, old, new):
    mut = _mutated_module(old, new)
    # the unmutated module must pass every check first (so a RED can only come from the mutation)
    for chk in ALL_CHECKS:
        chk(FE)
    _extra_checks(FE)
    red = False
    for chk in ALL_CHECKS + (_extra_checks,):
        try:
            chk(mut)
        except Exception:  # noqa: BLE001  (any failure, incl. an exception from the mutated code, is a RED)
            red = True
            break
    if not red:
        # tick-0 pick lives in a tmp_path test; exercise it directly
        red = _tick0_red(mut)
    assert red, f"mutation '{name}' stayed GREEN: no check reads that formula"


def _tick0_red(m):
    import tempfile
    base = os.environ.get("FAM_TMP", "F:/TanitAD-scratch/strat1/fam/tmp")
    os.makedirs(base, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as d:
        tk = np.array([0, 1, 0, 1])
        p = os.path.join(d, "t.npz")
        lat = np.array([[0.0] * 5, [100.0] * 5, [1.0] * 5, [101.0] * 5])
        np.savez(p, tick=tk, ci=np.array([0, 0, 1, 1]), t0=np.array([4, 4, 4, 4]), t_lat_logits=lat, t_lon_logits=np.zeros((4, 7)), t_goal_prob=np.zeros((4, 22)))
        try:
            h = m.tick0_heads(p, np.array([0, 1]), np.array([4, 4]))
            return h["lat"][:, 0].tolist() != [0.0, 1.0]
        except Exception:  # noqa: BLE001
            return True
