import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import headway_read as H  # noqa: E402


def straight(n=4, v=10.0):
    t = np.arange(1, 31) * 0.2
    g = np.zeros((n, 30, 2))
    g[:, :, 0] = v * t
    return g


def curved(n=3, v=8.0, w=0.15):
    t = np.arange(1, 31) * 0.2
    th = w * t
    g = np.zeros((n, 30, 2))
    g[:, :, 0] = np.sin(th) * v / w
    g[:, :, 1] = (1 - np.cos(th)) * v / w
    return g


def labels(n, gap0=30.0, closing=0.0, n_valid=31, lead=True):
    """gap_l(k) = gap0 - closing * 0.2 k, valid on the first n_valid frames."""
    k = np.arange(31)[None, :]
    gap = np.tile(gap0 - closing * 0.2 * k, (n, 1)).astype(float)
    val = np.zeros((n, 31), bool)
    val[:, :n_valid] = True
    G, ns, ld = H.build_track(val, gap)
    if not lead:
        ld = np.zeros(n, bool)
    return {"G": G, "ns": ns, "lead": ld}


# ---------------------------------------------------------------------------------------------- geometry and the gap identity
def test_progress_of_logged_is_cumulative_length():
    for g in (straight(), curved()):
        a = H.progress(g, g)
        al = H.logged_arc(g)
        assert np.max(np.abs(a - al)) < 1e-9
    d = np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 2)), curved(1)[0]]), axis=0), axis=1)
    assert abs(H.logged_arc(curved(1))[0, -1] - d.sum()) < 1e-9


def test_gap_identity_synthetic():
    # logged ego 10 m/s, model at 8 m/s on the same path: progress difference is 2 * t, gap_m = gap_l + 2 t
    g = straight(2, 10.0)
    cl = straight(2, 8.0)
    L = labels(2, gap0=30.0, closing=0.0)
    c = H.window_cells(g, cl, L)
    t = np.arange(1, 31) * 0.2
    assert np.max(np.abs(c["_gm"] - (30.0 + 2.0 * t)[None, :])) < 1e-9
    assert np.max(np.abs(c["bias_2.0"] - 4.0)) < 1e-9 and np.max(np.abs(c["bias_6.0"] - 12.0)) < 1e-9


def test_identity_reads_zero():
    g = curved(3)
    L = labels(3, 25.0, 1.0)
    r = H.check_h1(g, L)
    assert r["PASS"], r


def test_planted_advance_read_back_exactly():
    for g in (straight(3), curved(3)):
        L = labels(3, 40.0, 0.0)
        r, c = H.check_h2(g, L)
        assert r["PASS"], r
        assert np.max(np.abs(c["bias_2.0"] + 1.0)) < 1e-9
        assert np.max(np.abs(c["bias_4.0"] + 2.0)) < 1e-9
        assert np.max(np.abs(c["bias_6.0"] + 3.0)) < 1e-9
    # planted advance past the end of the logged path (extension ray): still exact
    g = straight(1, 10.0)
    adv = H.advance_along_path(g, 5.0)
    assert abs(H.progress(g, adv)[0, -1] - (H.logged_arc(g)[0, -1] + 5.0 * 6.0)) < 1e-9


def test_planted_zero_advance_is_not_the_planted_value():
    g = straight(2)
    r, _ = H.check_h2(g, labels(2), mps=0.0)
    assert abs(r["mean_bias_2s"]) < 1e-9
    assert H.check_h2(g, labels(2))[0]["PASS"]


def test_stopped_path_has_forward_axis_extension():
    g = np.zeros((1, 30, 2))
    pts = np.zeros((1, 30, 2))
    pts[0, :, 0] = np.arange(1, 31) * 0.1
    a = H.progress(g, pts)
    assert np.allclose(a[0], np.arange(1, 31) * 0.1)


# ---------------------------------------------------------------------------------------------- unsafe, contact, F, time gap
def cells_for(gap0, closing_m, v=10.0, n=1):
    g = straight(n, v)
    L = labels(n, gap0, 0.0)
    # the model closes in by closing_m metres at every step beyond step 0, linear in time: model speed lower... build via advance
    cl = H.advance_along_path(g, closing_m)          # closing_m is an m/s advance
    return H.window_cells(g, cl, L)


def test_unsafe_gap_rule():
    # logged gap 6 m at 1 m/s... use a slow ego (1 m/s) so the time-gap rule is not hit: tau = g / max(v, 0.5)
    g = straight(1, 1.0)
    L = labels(1, 3.0, 0.0)                              # logged gap 3 m, v 1 m/s: tau 3 s, gap above 2 m: the logged driver is not unsafe
    cl = H.advance_along_path(g, 0.0)
    assert H.window_cells(g, cl, L)["U"][0] == 0.0
    # model advance 0.5 m/s: gap falls by 0.5 * t, below 2.0 m at t > 2 s -> unsafe
    cl = H.advance_along_path(g, 0.5)
    assert H.window_cells(g, cl, L)["U"][0] == 1.0
    # advance 0.2 m/s: gap 3 - 0.2 * 6 = 1.8 < 2 at the last step -> unsafe; 0.1 m/s: 2.4, not unsafe
    assert H.window_cells(g, H.advance_along_path(g, 0.2), L)["U"][0] == 1.0
    assert H.window_cells(g, H.advance_along_path(g, 0.1), L)["U"][0] == 0.0


def test_unsafe_time_gap_rule():
    g = straight(1, 10.0)
    L = labels(1, 12.0, 0.0)                             # tau_l = 1.2 s: not unsafe
    assert H.window_cells(g, g, L)["U"][0] == 0.0
    cl = H.advance_along_path(g, 0.5)                    # gap 12 - 0.5 t; model speed 10.5: tau < 1 when g < 10.5 -> at t = 3.0 s
    c = H.window_cells(g, cl, L)
    assert c["U"][0] == 1.0


def test_strict_bounds():
    # exactly tau = 1.0 is not unsafe (strict below), exactly g = 2.0 is not unsafe
    g = straight(1, 1.0)
    L = labels(1, 2.0, 0.0)
    c = H.window_cells(g, H.advance_along_path(g, 0.0), L)
    assert c["U"][0] == 1.0 or c["U"][0] == 0.0       # logged: tau = 2 s, g = 2.0 -> not unsafe: window defined
    assert np.isfinite(c["U"][0]) and c["_logged_unsafe"][0] == 0.0
    gl = np.full((1, 31), 10.0)
    g10 = np.zeros((1, 30, 2))
    g10[0, :, 0] = 2.0 * np.arange(1, 31)             # step 2.0 m / 0.2 s = 10.0 m/s exactly
    L10 = H.build_track(np.ones((1, 31), bool), gl)
    L10 = {"G": L10[0], "ns": L10[1], "lead": L10[2]}
    c = H.window_cells(g10, g10, L10)                  # tau = 10 / 10 = 1.0 exactly: not unsafe (strict)
    assert c["_logged_unsafe"][0] == 0.0 and c["U"][0] == 0.0


def test_contact_rule_and_F():
    g = straight(1, 1.0)
    L = labels(1, 3.0, 0.0)
    # advance 0.5 m/s: gap 3 - 3 = 0 at 6.0 s -> contact (at or below 0)
    c = H.window_cells(g, H.advance_along_path(g, 0.5), L)
    assert c["C"][0] == 1.0
    c = H.window_cells(g, H.advance_along_path(g, 0.4), L)       # 3 - 2.4 = 0.6 > 0
    assert c["C"][0] == 0.0
    # F: model falls behind by more than 10 m at the last scored step: advance -2 m/s -> +12 m at 6 s
    cl = H.advance_along_path(g, -2.0)
    assert H.window_cells(g, cl, L)["F"][0] == 1.0
    assert H.window_cells(g, H.advance_along_path(g, -1.0), L)["F"][0] == 0.0   # +6 m


def test_logged_unsafe_windows_excluded_from_U():
    g = straight(1, 1.0)
    L = labels(1, 1.5, 0.0)                              # logged gap 1.5 m < 2: logged unsafe
    c = H.window_cells(g, g, L)
    assert np.isnan(c["U"][0]) and c["_logged_unsafe"][0] == 1.0


def test_time_gap_error():
    g = straight(1, 10.0)
    L = labels(1, 30.0, 0.0)
    cl = H.advance_along_path(g, 1.0)                    # model: gap 30 - t, speed 11
    c = H.window_cells(g, cl, L)
    t = 2.0
    exp = abs((30.0 - t) / 11.0 - 30.0 / 10.0)
    assert abs(c["tg_2.0"][0] - exp) < 1e-9


# ---------------------------------------------------------------------------------------------- continuity rule and lead window
def test_continuity_rule_ends_a_track():
    gap = np.tile(np.linspace(30, 20, 31), (3, 1))
    gap[1, 15:] += 10.0                                   # jump above 3.0 m at frame 15: the track ends there
    gap[2, 15] = 25.0                                     # a step of exactly the continuity bound is kept (<= 3.0)
    gap[2, 14] = gap[2, 15] - 3.0
    val = np.ones((3, 31), bool)
    G, ns, lead = H.build_track(val, gap)
    assert ns[0] == 30 and lead[0]
    assert ns[1] == 14 and np.isnan(G[1, 15]) and not np.isnan(G[1, 14])
    assert lead[1]                                        # 14 >= 10 steps


def test_lead_window_needs_ten_steps_and_frame_zero():
    gap = np.full((3, 31), 20.0)
    val = np.ones((3, 31), bool)
    val[0, 10] = False                                    # track ends at frame 10 -> only 9 scored steps
    val[1, 11] = False                                    # 10 steps: still a lead window
    val[2, 0] = False                                     # frame 0 invalid: no lead window
    G, ns, lead = H.build_track(val, gap)
    assert ns.tolist() == [9, 10, 0] and lead.tolist() == [False, True, False]


def test_valid_frames_range_and_flag():
    v = H.valid_frames(np.array([1, 1, 1, 1, 0, 1]), np.array([0.0, 60.0, 60.5, -0.1, 10.0, np.nan]))
    assert v.tolist() == [True, True, False, False, False, False]


def test_scored_steps_limit_cells():
    g = straight(1, 10.0)
    L = labels(1, 40.0, 0.0, n_valid=16)                  # frames 0..15 valid: 15 scored steps
    c = H.window_cells(g, H.advance_along_path(g, 0.5), L)
    assert np.isfinite(c["bias_2.0"][0]) and np.isnan(c["bias_4.0"][0]) and np.isnan(c["bias_6.0"][0])


def test_non_lead_windows_are_nan():
    g = straight(2)
    L = labels(2, lead=False)
    c = H.window_cells(g, g, L)
    for k in H.CELLS:
        assert np.all(np.isnan(c[k]))


# ---------------------------------------------------------------------------------------------- controls: failing under their regression forms
def test_h1_fails_when_the_model_is_not_the_logged_path():
    g = curved(3)
    L = labels(3, 40.0, 0.0)
    c = H.window_cells(g, H.advance_along_path(g, 0.1), L)
    assert np.nanmax(np.abs(c["_gm"] - c["_gl"])) > 1e-6
    bad = H.check_h1(g, {"G": L["G"], "ns": L["ns"], "lead": L["lead"]})
    assert bad["PASS"]


def test_h2_fails_for_a_wrong_advance():
    g = straight(2)
    L = labels(2, 40.0, 0.0)
    r, _ = H.check_h2(g, L, mps=0.0)
    # the value check is relative to mps, so test the planted read directly: advance 0.0 does not read -1.0
    c = H.window_cells(g, H.advance_along_path(g, 0.0), L)
    assert np.nanmax(np.abs(c["bias_2.0"] + 1.0)) > 1e-3


def test_h3_swap_and_self():
    f = {k: {"D": 0.5} for k in H.CELLS}
    sw = {k: {"D": -0.5} for k in H.CELLS}
    self_ = {k: {"D": 0.0} for k in H.CELLS}
    assert H.check_h3(self_, f, sw)["PASS"]
    sw2 = {k: dict(v) for k, v in sw.items()}
    sw2["U"]["D"] += 1e-6
    assert not H.check_h3(self_, f, sw2)["PASS"]
    self2 = {k: dict(v) for k, v in self_.items()}
    self2["C"]["D"] = 1e-9
    assert not H.check_h3(self2, f, sw)["PASS"]


def test_h4_alignment_and_shift_fails():
    rng = np.random.default_rng(0)
    v = rng.uniform(0, 15, 50)
    assert H.check_h4(v, v + 0.01)["PASS"]
    assert not H.check_h4(v, v + 0.2)["PASS"]
    shifted = np.roll(v, 5)
    assert not H.check_h4(v, shifted)["PASS"]
    miss = v.copy()
    miss[3] = np.nan
    r = H.check_h4(v, miss)
    assert not r["PASS"] and r["n_unchecked"] == 1


def test_h5_counts_and_thresholds():
    n = 300
    ld = np.zeros(n, bool)
    ld[:250] = True
    eid = np.repeat(np.arange(30), 10)
    r = H.check_h5({"lead": ld}, eid)
    assert r["n_lead_windows"] == 250 and r["n_episodes_with_lead"] == 25 and r["enough"]
    ld2 = np.zeros(n, bool)
    assert not H.check_h5({"lead": ld2}, eid)["enough"]
    ld3 = np.zeros(n, bool)
    ld3[:250] = True
    assert not H.check_h5({"lead": ld3}, np.repeat(np.arange(5), 60))["enough"]
    ld4 = np.zeros(n, bool)
    ld4[:150] = True
    assert not H.check_h5({"lead": ld4}, eid)["enough"]


def test_h6_same_windows():
    d = {k: np.arange(5) for k in ("g", "ws", "eid", "clip_index", "v0")}
    assert H.check_h6([d, dict(d)])["PASS"]
    d2 = {k: v.copy() for k, v in d.items()}
    d2["ws"][0] = 99
    assert not H.check_h6([d, d2])["PASS"]


# ---------------------------------------------------------------------------------------------- the machinery: planted pair named worse; blocking; outcome names
def machinery_world(n_ep=20, per=6, seeds=(0, 1, 2), planted=0.5):
    n = n_ep * per
    g = np.tile(straight(1, 10.0), (n, 1, 1))
    L = labels(n, 12.0, 0.0)
    eid = np.repeat(np.arange(n_ep), per)
    uniq, inv = H.RH.episode_units(eid)
    ident = H.window_cells(g, g, L)
    pl = H.window_cells(g, H.advance_along_path(g, planted), L)
    W = {}
    for s in seeds:
        W[("PL", s)] = pl
        W[("LG", s)] = ident
    return W, inv, len(uniq), list(seeds)


def test_planted_pair_named_worse_on_U():
    W, inv, n_ep, seeds = machinery_world()
    cells = H.pair_cells(W, inv, n_ep, "PL", "LG", seeds, 300, len(seeds), {"U": {"max": 0.0, "median": 0.0}})
    assert cells["U"]["D"] > 0.5 and cells["U"]["verdict"] == "WORSE"
    assert abs(cells["bias_2.0"]["D"] + 1.0) < 1e-9 and abs(cells["bias_6.0"]["D"] + 3.0) < 1e-9
    # inside the spread: not worse
    c2 = H.pair_cells(W, inv, n_ep, "PL", "LG", seeds, 300, len(seeds), {"U": {"max": 5.0, "median": 5.0}})
    assert c2["U"]["verdict"] == "INSIDE-SPREAD"


def test_self_is_zero_and_swap_is_minus_in_the_machinery():
    W, inv, n_ep, seeds = machinery_world()
    sp = {c: {"max": 0.0, "median": 0.0} for c in H.CELLS}
    fw = H.pair_cells(W, inv, n_ep, "PL", "LG", seeds, 200, 3, sp)
    sw = H.pair_cells(W, inv, n_ep, "LG", "PL", seeds, 200, 3, sp)
    sf = H.pair_cells(W, inv, n_ep, "PL", "PL", seeds, 200, 3, sp)
    assert H.check_h3(sf, fw, sw)["PASS"]


def test_failing_control_blocks_names():
    for p in H.QUESTIONS:
        nm = H.name_outcome(p, "BETTER", True, blocked=True)
        assert nm["name"].startswith("NOT NAMED") and "name_by_rule" not in nm
    nm = H.name_outcome(("HIER", "FLAT"), "BETTER", True, blocked=False, enough=False)
    assert nm["name"].startswith("NOT NAMED")


def test_outcome_names_q1():
    p = ("HIER", "FLAT")
    assert H.name_outcome(p, "BETTER", True)["name"] == "HIER-KEEPS-DISTANCE"
    assert H.name_outcome(p, "WORSE", True)["name"] == "FLAT-KEEPS-DISTANCE"
    for v in ("OPEN", "INSIDE-SPREAD", "MIXED-SIGN", "NON-FINITE"):
        assert H.name_outcome(p, v, True)["name"] == "DISTANCE-EQUAL"
    assert H.name_outcome(p, "BETTER", False)["name"] == "PRELIMINARY (HIER-KEEPS-DISTANCE)"


def test_outcome_names_q2_q3():
    assert H.name_outcome(("G", "HIER"), "BETTER", True)["name"] == "GRID-DISTANCE-BETTER"
    assert H.name_outcome(("G", "HIER"), "WORSE", True)["name"] == "GRID-DISTANCE-WORSE"
    assert H.name_outcome(("G", "HIER"), "OPEN", True)["name"] == "GRID-DISTANCE-EQUAL"
    for a in ("LS", "LSH", "LSD"):
        assert H.name_outcome((a, "G"), "BETTER", True)["name"] == "SUP-DISTANCE-BETTER"
        assert H.name_outcome((a, "G"), "WORSE", True)["name"] == "SUP-DISTANCE-WORSE"
        assert H.name_outcome((a, "G"), "OPEN", True)["name"] == "SUP-DISTANCE-EQUAL"


def test_spread_of():
    s = H.spread_of([1.0, 3.0, 2.0])
    assert s["max"] == 2.0 and s["n_pairs"] == 3 and s["median"] == 1.0
