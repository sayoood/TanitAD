"""PREREG_V8F_VOE1B control 4 and the instrument's unit tests (CPU, no real data). VOE1's 13 tests are kept (copy of test_voe1_synth.py; the only edit: eval_tag's VOE1-style label
is the key ``label_ab`` here); the tests after the line "VOE1B additions" are new.

Control 4 (card): a toy latent world model that predicts constant velocity, given clips with a planted freeze and a planted teleport, must give NOTICES on both
before the real run; an action-blind toy must give BLIND on ACTION-SWAP. The manipulations are planted by the SAME index-map builders and the SAME driver
(``voe1b_core.eval_tag``) the real run uses; only the model and the bank are synthetic.

Toy latent (4 dims): position (2, constant velocity) and a rotating texture (2, radius 4, constant angular velocity 2.5 rad per step, close to the largest chord),
plus Gaussian noise 0.3 on every dim. The texture makes the Copy-Last null unable to see a 1.6 s teleport (consecutive frames already differ by nearly the
largest chord) while a model that extrapolates the phase can. The toy predicts constant velocity in (position, phase). Parameters were fixed by this reasoning
before the first run.
Run: pytest -q test_voe1b_synth.py  (from this directory; the optional real-stack test needs WMACT_CODE_ROOT)"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voe1b_core as C  # noqa: E402


# ------------------------------------------------------------------------------------------------ the toy
class ToyModel:
    levels = (1,)

    def __init__(self, use_action: bool, B: float = 1.0):
        self.use_action, self.B = use_action, B

    def traces(self, z1, a_sc):
        z, a = z1.double().numpy(), a_sc.double().numpy()
        N, T, _ = z.shape
        null = ((z[:, 1:] - z[:, :-1]) ** 2).sum(-1)
        sur = np.zeros((N, T - 1))
        for j in range(1, T):
            zi = z[:, j - 1]
            if self.use_action:
                pos = zi[:, :2] + self.B * a[:, j - 1, :]
            else:
                pos = 2 * zi[:, :2] - z[:, j - 2, :2] if j >= 2 else zi[:, :2]
            a1 = np.arctan2(zi[:, 3], zi[:, 2])
            if j >= 2:
                a0 = np.arctan2(z[:, j - 2, 3], z[:, j - 2, 2])
                a1 = a1 + (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
            r = np.hypot(zi[:, 2], zi[:, 3])
            pred = np.concatenate([pos, (r * np.cos(a1))[:, None], (r * np.sin(a1))[:, None]], 1)
            sur[:, j - 1] = ((pred - z[:, j]) ** 2).sum(-1)
        return {1: {"sur": sur, "null": null}}


def toy_world(action_driven: bool, n_clips: int = 24, T: int = 80, seed: int = 0, sigma: float = 0.3):
    rng = np.random.default_rng(seed)
    Z, ACT, SPD, off, Tn, eid = [], [], [], [], [], []
    pos0 = 0
    for k in range(n_clips):
        a_mean, y_mean = rng.uniform(-3, 3), rng.uniform(-0.3, 0.3)
        act = np.stack([a_mean + 0.2 * rng.standard_normal(T), y_mean + 0.02 * rng.standard_normal(T)], 1)
        th = rng.uniform(0, 2 * np.pi)
        if action_driven:
            step = act.copy()
            p = np.concatenate([[[0.0, 0.0]], np.cumsum(step[:-1], 0)])
        else:
            p = 0.3 * np.arange(T)[:, None] * np.array([[np.cos(th), np.sin(th)]])
        om, ph = 2.5 + 0.1 * rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi)
        tex = 4.0 * np.stack([np.cos(ph + om * np.arange(T)), np.sin(ph + om * np.arange(T))], 1)
        z = np.concatenate([p, tex], 1) + sigma * rng.standard_normal((T, 4))
        Z.append(z.astype(np.float32)); ACT.append(act.astype(np.float32))
        SPD.append(np.full(T, rng.uniform(1.0, 15.0), np.float32))
        off.append(pos0); Tn.append(T); eid += [f"c{k:02d}"] * 1
        pos0 += T
    g0 = np.concatenate([np.arange(0, T - 21, 4) for _ in range(n_clips)])
    ci = np.concatenate([np.full(len(np.arange(0, T - 21, 4)), k) for k in range(n_clips)])
    base = np.array(off)[ci] + g0
    eids = np.array([f"c{k:02d}" for k in ci])
    return {"z1": torch.from_numpy(np.concatenate(Z)), "act": np.concatenate(ACT), "speed": np.concatenate(SPD), "base": base, "g0": g0, "T": np.array(Tn)[ci], "eid": eids}


def run_toy(model, W):
    ep, n_ep = C.episode_codes(W["eid"])
    wt = C.window_table(W["act"], W["speed"], W["base"], W["g0"], W["T"], ep)
    a_sc = torch.from_numpy(W["act"])
    return C.eval_tag(model, W["z1"], a_sc, W["base"], W["eid"], wt, "toy", cal=True), wt


# ------------------------------------------------------------------------------------------------ control 4
def test_control4_cv_toy_notices_freeze_and_teleport_blind_on_action_swap():
    res, wt = run_toy(ToyModel(use_action=False), toy_world(action_driven=False))
    M = res["manips"]
    fz, tp, sw = M["FREEZE"]["levels"]["1"], M["TELEPORT"]["levels"]["1"], M["ACTION-SWAP"]["levels"]["1"]
    assert fz["label_ab"] == "NOTICES", fz["acc"]
    assert tp["label_ab"] == "NOTICES", (tp["acc"], tp["null_acc"], tp["diff"])
    assert tp["null_acc"]["point"] < tp["acc"]["point"]
    assert fz["null_acc"]["point"] <= 0.5 + 1e-12                               # the Copy-Last surprise of a frozen picture cannot exceed the unmodified copy's
    assert sw["label_ab"] == "BLIND"                                                # an action-blind toy: scores identical -> accuracy exactly 0.5
    assert sw["acc"]["point"] == 0.5 and sw["acc"]["lo"] == 0.5 and sw["acc"]["hi"] == 0.5
    assert sw["null_acc"]["point"] == 0.5                                        # the frames are the real ones: the null cannot see the swap
    for l, r in res["rvr"]["levels"].items():                                    # control 1 on the toy: nothing is separated either way
        assert not r["separated_any"], r


def test_control4_action_toy_notices_action_swap_and_freeze():
    res, wt = run_toy(ToyModel(use_action=True), toy_world(action_driven=True))
    M = res["manips"]
    assert M["ACTION-SWAP"]["levels"]["1"]["label_ab"] == "NOTICES", M["ACTION-SWAP"]["levels"]["1"]["acc"]
    assert M["ACTION-SWAP"]["levels"]["1"]["null_acc"]["point"] == 0.5
    assert M["FREEZE"]["levels"]["1"]["label_ab"] == "NOTICES"


def test_control4_toy_calibration_draws_are_not_separated():
    res, _ = run_toy(ToyModel(use_action=False), toy_world(action_driven=False))
    nsep = tot = 0
    for rec in res["manips"].values():
        for r in rec["levels"].values():
            for k in ("acc", "null_acc", "diff"):
                c = r[k]["cal"]
                nsep += sum(c["sep_above"]) + sum(c["sep_below"]); tot += 2 * len(c["sep_above"])
    assert tot > 0 and nsep / tot <= 0.05, (nsep, tot)


# ------------------------------------------------------------------------------------------------ instrument unit tests
def test_index_maps():
    b = np.array([100]); d = np.array([500])
    fr, ac = C.idx_unmod(b)
    assert fr[0].tolist() == list(range(100, 113)) and ac[0].tolist() == list(range(100, 112))
    assert C.idx_unmod(b, 1)[0][0].tolist() == list(range(101, 114))
    fr, ac = C.idx_manip("TELEPORT", b)
    assert fr[0].tolist() == list(range(100, 105)) + list(range(113, 121)) and ac[0].tolist() == list(range(100, 112))
    fr, ac = C.idx_manip("FREEZE", b)
    assert fr[0].tolist() == [100, 101, 102, 103, 104, 104, 104, 104, 104, 105, 106, 107, 108]
    fr, ac = C.idx_manip("REVERSE", b)
    assert fr[0].tolist() == list(range(100, 105)) + [112, 111, 110, 109, 108, 107, 106, 105]
    fr, ac = C.idx_manip("SPLICE", b, d)
    assert fr[0].tolist() == list(range(100, 105)) + list(range(505, 513)) and ac[0].tolist() == list(range(100, 112))
    fr, ac = C.idx_manip("ACTION-SWAP", b, d)
    assert fr[0].tolist() == list(range(100, 113)) and ac[0].tolist() == list(range(100, 104)) + list(range(504, 512))


def test_score_cols_and_step_frames():
    assert C.score_cols(1) == [4, 5, 6, 7, 8, 9, 10, 11]
    assert C.score_cols(2) == [2, 3, 4, 5] and (C.step_frames(2, 6)[C.score_cols(2)] == np.array([6, 8, 10, 12])).all()
    assert C.score_cols(3) == [1, 2] and (C.step_frames(3, 3)[C.score_cols(3)] == np.array([8, 12])).all()


def test_pair_x_ties():
    assert C.pair_x(np.array([2.0, 1.0, 1.0]), np.array([1.0, 1.0, 2.0])).tolist() == [1.0, 0.5, 0.0]


def test_donors_are_cross_episode_and_deterministic():
    ep = np.repeat(np.arange(10), 5)
    d = C.donor_derangement(ep, 0)
    assert (ep[d] != ep).all() and (d == C.donor_derangement(ep, 0)).all()
    lon = np.random.default_rng(1).uniform(-3, 3, 50); yaw = np.zeros(50)
    don, nc = C.swap_donors(lon, yaw, ep, 0)
    ok = don >= 0
    assert ok.any() and (ep[don[ok]] != ep[ok]).all() and (np.abs(lon[don[ok]] - lon[ok]) >= C.SWAP_LON).all()
    assert (don == C.swap_donors(lon, yaw, ep, 0)[0]).all()


def test_ap_matches_sklearn_and_bootstrap_weights_one():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    s = rng.standard_normal(500); y = (rng.uniform(size=500) < 0.2)
    assert abs(C.ap_point(s, y) - average_precision_score(y, s)) < 1e-12
    o = np.argsort(-s, kind="stable")
    ap, prev = C.ap_boot(y[o].astype(np.float32), np.zeros(500, np.int64), np.ones((3, 1)))
    assert np.allclose(ap, C.ap_point(s, y), atol=1e-6) and np.allclose(prev, y.mean())
    assert C.ap_point(y.astype(float) + 1e-9 * s, y) > 0.999                    # a perfect ranking


def test_window_perm_is_a_bijection_to_other_episodes():
    ep = np.repeat(np.arange(20), 14)
    for s in range(5):
        src = C.window_perm(ep, 20, np.random.default_rng(s))
        assert sorted(src.tolist()) == list(range(len(ep))) and (ep[src] != ep).all()


def test_event_labels_band_and_onset():
    c = np.zeros(40, bool); c[10:20] = True; c[30] = True
    on = C.onset(c)
    assert np.flatnonzero(on).tolist() == [10, 30]                            # 30 follows 10 frames without a True: a new onset
    base = np.array([0]); lab = C.event_step_labels(on, base, np.array([0]), np.array([40]), 1, 15)
    assert np.flatnonzero(lab[0]).tolist() == [7, 8, 9, 10, 11]                # step j sits at frame j: |j - 10| <= 2 -> j = 8..12 -> columns 7..11; the onset at 30 is outside the 15 steps
    lab3 = C.event_step_labels(on, base, np.array([0]), np.array([40]), 3, 3)  # frames 4, 8, 12: 8 and 12 are within 2 of 10
    assert lab3[0].tolist() == [False, True, True]


def test_two_part_rule_does_not_separate_chance_data():
    rng = np.random.default_rng(3)
    ep = np.repeat(np.arange(30), 6); nsep = 0
    for k in range(40):
        x = (rng.uniform(size=len(ep)) < 0.5).astype(float)
        r = C.two_part(x, ep, 30, 0.5, f"chance/{k}", cal_seeds=())
        nsep += r["sep_above"] or r["sep_below"]
    assert nsep <= 4


def test_level_labels():
    assert C.level_label("NOTICES", "NOTICES") == "NOTICES" and C.level_label("BLIND", "BLIND") == "BLIND"
    assert C.level_label("NOTICES", "NULL-ONLY") == "NULL-ONLY" and C.level_label("BLIND", "NOTICES") == "DISCORDANT"


def test_real_stack_traces_are_causal_and_match_the_teacher_forced_chunk():
    """Optional (needs WMACT_CODE_ROOT): the sliding-context traces equal the output positions of one full-chunk predictor call (the form ``_pred_loss`` trains)."""
    root = os.environ.get("WMACT_CODE_ROOT")
    if not root:
        pytest.skip("WMACT_CODE_ROOT not set")
    sys.path.insert(0, os.path.join(root, "stack"))
    from tanitad.models.hjepa import HJEPAStack, LevelConfig
    torch.manual_seed(0)
    mk = lambda **kw: LevelConfig(latent_dim=16, beta=0.08, gamma=100.0, pred_width=32, pred_depth=2, pred_heads=4, enc_hidden=16, macro_hidden=8, **kw)
    st = HJEPAStack([mk(action_dim=2, act_embed_dim=8), mk(action_dim=8, stride=2), mk(action_dim=8, stride=2)], sigreg_projections=8).eval()
    z1, a = torch.randn(5, 13, 16), torch.randn(5, 12, 2)
    tr = C.HJEPATraces(st).traces(z1, a)
    with torch.no_grad():
        enc = st.encode(z1, a)
    assert tr[1]["sur"].shape == (5, 12) and tr[2]["sur"].shape == (5, 6) and tr[3]["sur"].shape == (5, 3)
    for l in (1, 2, 3):
        z, e = enc.z[l - 1], enc.act[l - 1]
        with torch.no_grad():
            full = st.predictors[l - 1](z[:, 0:3], e[:, 0:3])
        for j in (1, 2, 3):
            want = ((full[:, j - 1] - z[:, j]).double() ** 2).sum(-1).numpy()
            assert np.allclose(tr[l]["sur"][:, j - 1], want, rtol=1e-4, atol=1e-6), (l, j)
        assert np.allclose(tr[l]["null"][:, 0], ((z[:, 0] - z[:, 1]).double() ** 2).sum(-1).numpy())
    # causality: changing a late frame / action does not change an earlier step
    z1b = z1.clone(); z1b[:, 10:] += 1.0
    trb = C.HJEPATraces(st).traces(z1b, a)
    assert np.allclose(tr[1]["sur"][:, :9], trb[1]["sur"][:, :9]) and not np.allclose(tr[1]["sur"][:, 9:], trb[1]["sur"][:, 9:])


# ================================================================================================ VOE1B additions
class ToyGain:
    """An 'untrained-like' family member: predicts z_{j-1} + g * (z_{j-1} - z_{j-2}) with a random gain g (a bias in a random direction: g < 0 is surprised by
    a pause, g > 0 is anti-surprised). Surprise = squared prediction error; Copy-Last null as usual."""
    levels = (1,)

    def __init__(self, g):
        self.g = float(g)

    def traces(self, z1, a_sc):
        z = z1.double().numpy()
        null = ((z[:, 1:] - z[:, :-1]) ** 2).sum(-1)
        N, Tn, _ = z.shape
        sur = np.zeros((N, Tn - 1))
        for j in range(1, Tn):
            pred = z[:, j - 1] + (self.g * (z[:, j - 1] - z[:, j - 2]) if j >= 2 else 0.0)
            sur[:, j - 1] = ((pred - z[:, j]) ** 2).sum(-1)
        return {1: {"sur": sur, "null": null}}


class ToyBias:
    """Untrained-like toy: surprise = ||z_j - z_{j-1} - d||^2 with a fixed random offset d (a prediction biased in a random direction)."""
    levels = (1,)

    def __init__(self, d):
        self.d = np.asarray(d, float)

    def traces(self, z1, a_sc):
        z = z1.double().numpy()
        null = ((z[:, 1:] - z[:, :-1]) ** 2).sum(-1)
        sur = ((z[:, 1:] - z[:, :-1] - self.d[None, None]) ** 2).sum(-1)
        return {1: {"sur": sur, "null": null}}


_TOY = {}


def toy_cv_world():
    if "w" not in _TOY:
        W = toy_world(action_driven=False)
        ep, n_ep = C.episode_codes(W["eid"])
        _TOY["w"] = (W, C.window_table(W["act"], W["speed"], W["base"], W["g0"], W["T"], ep), torch.from_numpy(W["act"]))
    return _TOY["w"]


def toy_eval(model, tag, lean=False, manips=("FREEZE", "TELEPORT")):
    W, wt, a_sc = toy_cv_world()
    r = C.eval_tag(model, W["z1"], a_sc, W["base"], W["eid"], wt, tag, cal=False, lean=lean)
    return {m: r["manips"][m]["levels"]["1"] for m in manips}


def toy_labels(trained_rows: dict, ut_rows: list, manip: str) -> dict:
    """VOE1B label of one 'trained' member's row against the untrained members' rows."""
    row = trained_rows[manip]
    return C.label_cell_b(row["acc"], row.get("diff"), manip in C.NULL_COUNTS, [u[manip]["acc"]["point"] for u in ut_rows])


def test_label_cell_b_rules_on_hand_made_rows():
    sep = lambda pt, s: {"point": pt, "sep_above": s}
    ut = [0.40, 0.55, 0.51]
    assert C.label_cell_b(sep(0.52, False), None, False, ut)["label"] == "BLIND"                       # (a) fails
    assert C.label_cell_b(sep(0.70, True), None, False, ut)["label"] == "NOTICES"                      # FREEZE-type: (a) and (c)
    assert C.label_cell_b(sep(0.54, True), None, False, ut)["label"] == "INIT-LEVEL"                   # (a) holds, (c) fails: an untrained network does this too
    assert C.label_cell_b(sep(0.55, True), None, False, ut)["label"] == "INIT-LEVEL"                   # (c) is STRICT: equal to the maximum is not above it
    assert C.label_cell_b(sep(0.70, True), sep(0.1, True), True, ut)["label"] == "NOTICES"            # TELEPORT-type: (a), (b), (c)
    assert C.label_cell_b(sep(0.70, True), sep(0.1, False), True, ut)["label"] == "NULL-ONLY"           # (b) fails, whatever (c) says
    assert C.label_cell_b(sep(0.54, True), sep(0.1, True), True, ut)["label"] == "INIT-LEVEL"
    assert C.label_cell_b(sep(0.54, True), sep(0.1, False), True, ut)["label"] == "NULL-ONLY"
    assert C.label_cell_b(sep(0.70, False), sep(0.1, True), True, ut)["label"] == "BLIND"
    r = C.label_cell_b(sep(0.70, True), None, False, ut)
    assert (r["ut_min"], r["ut_max"], r["n_ut"]) == (0.40, 0.55, 3)


def test_level_label_b():
    L = C.level_label_b
    assert L("NOTICES", "NOTICES") == "NOTICES" and L("BLIND", "BLIND") == "BLIND" and L("INIT-LEVEL", "INIT-LEVEL") == "INIT-LEVEL" and L("NULL-ONLY", "NULL-ONLY") == "NULL-ONLY"
    assert L("NOTICES", "INIT-LEVEL") == "DISCORDANT" and L("NOTICES", "BLIND") == "DISCORDANT"          # NOTICES needs both seeds
    assert L("NOTICES", "NULL-ONLY") == "NULL-ONLY" and L("INIT-LEVEL", "NULL-ONLY") == "NULL-ONLY"
    assert L("BLIND", "INIT-LEVEL") == "BLIND/INIT-LEVEL" and L("INIT-LEVEL", "BLIND") == "BLIND/INIT-LEVEL"
    assert L("BLIND", "NULL-ONLY") == "DISCORDANT"


def test_swap_donors_m_rule_on_hand_made_windows():
    # window 0: mean|lon| 2.0, mean|yaw| 0.10, signed lon +1.9, signed yaw +0.08; windows 1..9 differ as noted
    ml = np.array([2.0, 2.1, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    my = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.115, 0.10])
    lon = np.array([1.9, -1.9, -1.9, -1.9, 1.9, -0.4, 1.9, 1.9, -1.9, -1.9, 0.3])
    yaw = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, -0.08, -0.04, 0.08, 0.08, 0.03])
    ep = np.array([0, 1, 2, 0, 3, 4, 5, 6, 7, 8, 9])
    # 1: opposite lon, magnitudes equal            -> candidate
    # 2: mean|lon| 2.3 is 15 % off                -> no (magnitude)
    # 3: same episode as window 0                 -> no
    # 4: same signs                               -> no
    # 5: lon opposite but |signed lon| 0.4 < 0.5, yaw same sign -> no (the donor's own mean must reach the threshold)
    # 6: lon same, yaw opposite (-0.08, both >= 0.05) -> candidate
    # 7: yaw opposite but |-0.04| < 0.05          -> no
    # 8: opposite lon, magnitudes equal           -> candidate
    # 9: opposite lon but mean|yaw| 0.115 is 15 % off -> no
    # 10: both signed means below both thresholds -> never a candidate and has none itself
    don, nc = C.swap_donors_m(ml, my, lon, yaw, ep, seed=0)
    assert nc[0] == 3 and don[0] in (1, 6, 8), (nc[0], don[0])
    ok = don >= 0
    assert (ep[don[ok]] != ep[ok]).all()
    assert nc[5] == 1 and don[5] == 6                                                                     # window 5: lon mean below threshold, but its yaw +0.08 is testable; only window 6 has the opposite yaw
    assert nc[10] == 0 and don[10] == -1                                                                  # window 10: both signed means below the thresholds: no sign test possible
    don2, _ = C.swap_donors_m(ml, my, lon, yaw, ep, seed=0)
    assert (don == don2).all()                                                                            # deterministic


def test_action_swap_m_index_map_is_the_action_swap_map_and_window_table_has_the_row():
    b = np.array([100]); d = np.array([500])
    f1, a1 = C.idx_manip("ACTION-SWAP-M", b, d)
    f2, a2 = C.idx_manip("ACTION-SWAP", b, d)
    assert (f1 == f2).all() and (a1 == a2).all() and f1[0].tolist() == list(range(100, 113)) and a1[0].tolist() == list(range(100, 104)) + list(range(504, 512))
    assert "ACTION-SWAP-M" in C.MANIPS and "ACTION-SWAP-M" not in C.NULL_COUNTS
    W, wt, a_sc = toy_cv_world()
    assert "ACTION-SWAP-M" in wt["q"] and "ACTION-SWAP-M" in wt["donor"] and len(wt["donor"]["ACTION-SWAP-M"]) == len(W["base"])
    res, _ = run_toy(ToyModel(use_action=False), W)
    assert "ACTION-SWAP-M" in res["manips"]                                                               # an empty row (n = 0) is handled, not a crash
    assert C.SWM_MIN_N == 300


def test_control4b_planted_toy_notices_with_the_untrained_reference_rule_c():
    """The planted constant-velocity toy still gives NOTICES on FREEZE and TELEPORT once rule (c) is applied against 16 untrained toys (offset-biased in random directions)."""
    rng = np.random.default_rng(5)
    ut = [toy_eval(ToyBias(1.0 * rng.standard_normal(4)), f"ut{k}", lean=True) for k in range(16)]
    cv = toy_eval(ToyModel(use_action=False), "cv")
    for m in ("FREEZE", "TELEPORT"):
        lab = toy_labels(cv, ut, m)
        assert lab["label"] == "NOTICES" and lab["c"] and lab["a"] and (lab["b"] in (None, True)), (m, lab)
        assert lab["ut_max"] < cv[m]["acc"]["point"]
    assert toy_labels(cv, ut, "TELEPORT")["b"] is True                                                    # (b) applies to TELEPORT


def test_control4b_toy_family_biased_in_random_directions_gives_init_level_not_notices():
    """THE ADDED TEST (card, control 4): a family whose untrained members are biased in random directions (random gain g in U(-1, 2): some are surprised by a pause,
    some anti-surprised) gives INIT-LEVEL, not NOTICES, for a 'trained' member drawn from the same family. 17 exchangeable members; each in turn is the 'trained'
    one against the other 16."""
    gs = np.random.default_rng(0).uniform(-1, 2, 17)
    rows = [toy_eval(ToyGain(g), f"g{k}") for k, g in enumerate(gs)]
    for m in ("FREEZE", "TELEPORT"):
        labs = [toy_labels(rows[k], rows[:k] + rows[k + 1:], m) for k in range(17)]
        pts = np.array([r[m]["acc"]["point"] for r in rows])
        notices = [k for k, l in enumerate(labs) if l["label"] == "NOTICES"]
        assert len(notices) <= 1, (m, notices)
        assert all(pts[k] > np.delete(pts, k).max() for k in notices)                                       # a NOTICES member is the strict maximum of the family
        assert sum(l["label"] == "INIT-LEVEL" for l in labs) >= 1, (m, [l["label"] for l in labs])        # members with (a) (and (b)) but not the maximum: INIT-LEVEL
        for k, l in enumerate(labs):
            if l["a"] and l["b"] is not False and not l["c"]:
                assert l["label"] == "INIT-LEVEL"
    pf = np.array([r["FREEZE"]["acc"]["point"] for r in rows])
    assert (pf >= 0.99).sum() >= 4 and pf.min() < 0.3                                                       # the family is bidirectional and saturates: no member is the strict maximum
    assert all(toy_labels(rows[k], rows[:k] + rows[k + 1:], "FREEZE")["label"] != "NOTICES" for k in range(17))


def test_control4b_deliberate_regression_planted_toy_inside_a_family_that_does_it_too_is_init_level():
    """Deliberate regression of the (c) rule: the planted toy (FREEZE accuracy 1.0) is NOT NOTICES against untrained toys that reach 1.0 as well."""
    gs = np.random.default_rng(0).uniform(-1, 2, 17)
    ut = [toy_eval(ToyGain(g), f"g{k}", lean=True) for k, g in enumerate(gs)][:16]
    cv = toy_eval(ToyModel(use_action=False), "cv")
    lab = toy_labels(cv, ut, "FREEZE")
    assert lab["a"] and not lab["c"] and lab["label"] == "INIT-LEVEL", lab


def _synthetic_rvr_cells(planted: bool, seed: int):
    ep = np.repeat(np.arange(30), 6)
    rng = np.random.default_rng(seed)
    cells = []
    for k in range(6):
        x = np.ones(len(ep)) if planted else (rng.uniform(size=len(ep)) < 0.5).astype(float)
        cells.append(C.two_part(x, ep, 30, 0.5, f"rvr_test/{seed}/{k}", cal_seeds=(), ref_flags=True))
    return cells


def test_control1_calibrated_reference_count():
    null_cells = _synthetic_rvr_cells(False, 1)
    for c in null_cells:
        assert len(c["ref_sep_above"]) == C.R_PERM and len(c["ref_sep_below"]) == C.R_PERM
    v = C.control1_verdict(null_cells)
    ref = C.control1_count(null_cells, "ref")
    assert ref.shape == (C.R_PERM,) and v["R"] == C.R_PERM and v["n_cells"] == 6 and v["real_count"] == C.control1_count(null_cells, "real")
    assert abs(v["ref_mean"] - float(ref.mean())) < 1e-12 and sum(v["ref_count_hist"].values()) == C.R_PERM
    assert v["pass"] == bool(v["real_count"] <= np.percentile(ref, 95))
    # a reference draw is a null draw by construction: the share of draw-cells separated is small (the rule is conservative, not above ~10 %)
    assert float(np.mean([np.mean(np.asarray(c["ref_sep_above"]) | np.asarray(c["ref_sep_below"])) for c in null_cells])) < 0.10
    # deliberate regression: six cells that are all strongly separated (every pair x = 1) must FAIL the control
    bad = C.control1_verdict(_synthetic_rvr_cells(True, 2))
    assert bad["real_count"] == 6 and bad["pass"] is False and bad["p95_linear"] < 6


def test_control2_leave_one_out_notices_share():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.4, 0.6, 16)
    allsep = [{"sep_above": True, "point": float(p)} for p in pts]
    r = C.loo_notices(allsep)
    assert sum(r) == 1 and r[int(np.argmax(pts))]                                                          # exchangeable members: exactly one can pass (c)
    nosep = [{"sep_above": False, "point": float(p)} for p in pts]
    assert sum(C.loo_notices(nosep)) == 0                                                                  # (a) is also required
    tied = [{"sep_above": True, "point": 0.7} for _ in range(16)]
    assert sum(C.loo_notices(tied)) == 0                                                                   # strict: ties pass nobody
    top_unsep = [dict(c) for c in allsep]
    top_unsep[int(np.argmax(pts))]["sep_above"] = False
    assert sum(C.loo_notices(top_unsep)) == 0                                                              # the maximum without (a) is not NOTICES, nobody else is the maximum
    shares = [sum(C.loo_notices([{"sep_above": True, "point": float(p)} for p in np.random.default_rng(s).uniform(size=16)])) / 16 for s in range(50)]
    assert max(shares) <= 1 / 16 + 1e-12                                                                   # share over exchangeable families is at most 1/16
