"""PREREG_V8F_VOE1 control 4 and the instrument's unit tests (CPU, no real data).

Control 4 (card): a toy latent world model that predicts constant velocity, given clips with a planted freeze and a planted teleport, must give NOTICES on both
before the real run; an action-blind toy must give BLIND on ACTION-SWAP. The manipulations are planted by the SAME index-map builders and the SAME driver
(``voe1_core.eval_tag``) the real run uses; only the model and the bank are synthetic.

Toy latent (4 dims): position (2, constant velocity) and a rotating texture (2, radius 4, constant angular velocity 2.5 rad per step, close to the largest chord),
plus Gaussian noise 0.3 on every dim. The texture makes the Copy-Last null unable to see a 1.6 s teleport (consecutive frames already differ by nearly the
largest chord) while a model that extrapolates the phase can. The toy predicts constant velocity in (position, phase). Parameters were fixed by this reasoning
before the first run.
Run: pytest -q test_voe1_synth.py  (from this directory; the optional real-stack test needs WMACT_CODE_ROOT)"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voe1_core as C  # noqa: E402


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
    assert fz["label"] == "NOTICES", fz["acc"]
    assert tp["label"] == "NOTICES", (tp["acc"], tp["null_acc"], tp["diff"])
    assert tp["null_acc"]["point"] < tp["acc"]["point"]
    assert fz["null_acc"]["point"] <= 0.5 + 1e-12                               # the Copy-Last surprise of a frozen picture cannot exceed the unmodified copy's
    assert sw["label"] == "BLIND"                                                # an action-blind toy: scores identical -> accuracy exactly 0.5
    assert sw["acc"]["point"] == 0.5 and sw["acc"]["lo"] == 0.5 and sw["acc"]["hi"] == 0.5
    assert sw["null_acc"]["point"] == 0.5                                        # the frames are the real ones: the null cannot see the swap
    for l, r in res["rvr"]["levels"].items():                                    # control 1 on the toy: nothing is separated either way
        assert not r["separated_any"], r


def test_control4_action_toy_notices_action_swap_and_freeze():
    res, wt = run_toy(ToyModel(use_action=True), toy_world(action_driven=True))
    M = res["manips"]
    assert M["ACTION-SWAP"]["levels"]["1"]["label"] == "NOTICES", M["ACTION-SWAP"]["levels"]["1"]["acc"]
    assert M["ACTION-SWAP"]["levels"]["1"]["null_acc"]["point"] == 0.5
    assert M["FREEZE"]["levels"]["1"]["label"] == "NOTICES"


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
