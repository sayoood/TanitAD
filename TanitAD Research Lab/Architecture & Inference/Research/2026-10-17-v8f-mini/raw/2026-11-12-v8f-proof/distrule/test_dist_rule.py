"""test_dist_rule.py -- DISTRULE (card PREREG_V8F_PROOF_ADD26): the projection, the wrapper, the sources, the controls that can run on the dev-box CPU.

Run:  CUDA_VISIBLE_DEVICES=-1 python -m pytest test_dist_rule.py -q --basetemp=F:/TanitAD-scratch/v8proof/distrule/tmp/pt   (the parent of --basetemp must exist)
Tier T1-kin (kinematic rollout on the model's own actions, logged frames); synthetic worlds only: no number here is a result about a real arm.
TREE = the pinned tool's tree (env DISTRULE_TREE, default the dev-box leadsup tree).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import dist_rule as R  # noqa: E402
import v8f_t1_eval_rule as W  # noqa: E402

TREE = os.environ.get("DISTRULE_TREE", "C:/Users/Admin/v8proof_c/leadsup/tree_patch")
HEADWAY_DIR = os.environ.get("HEADWAY_DIR", "F:/TanitAD-scratch/v8proof/headway")
PINNED_SHA = "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397"
DT = 0.2


# ============================================================================================ r5: the projection, hand-built cases and the card-literal reference
def ref_rule(plan, v0, g0, vL, D=3.0, tau=1.2, B=6.0, dt=0.2):
    """The card section 2 recursion written out literally (a'(i-1) form, scalar loop): the reference of the lag-form vectorised implementation."""
    P = np.vstack([np.zeros((1, 2)), np.asarray(plan, np.float64)])
    aP = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    ap, v, arcs, sat = 0.0, float(v0), [], 0
    for i in range(1, len(aP)):
        a_w = ap + (aP[i] - aP[i - 1])
        a_c = g0 + vL * dt * i - max(D, tau * v)
        a_f = ap + max(0.0, v - B * dt) * dt
        a = max(a_f, min(a_w, a_c))
        sat += int(a_c < a_f and a_c <= a_w)
        v = (a - ap) / dt
        ap = a
        arcs.append(a)
    return np.array(arcs), sat


def straight(v, n=30, y=0.0):
    k = np.arange(1, n + 1) * DT
    return np.stack([v * k, np.full(n, y)], 1)


def call(plan, v0, g_src, v_L, present=True, g0=None, **kw):
    plan = np.asarray(plan, np.float32)
    if plan.ndim == 2:
        plan = plan[None]
    n = len(plan)
    rep = lambda x: np.broadcast_to(np.asarray(x, np.float64), (n,)).copy()
    return R.distance_rule(plan, rep(v0), np.broadcast_to(np.asarray(present, bool), (n,)).copy(), rep(g_src), rep(g_src if g0 is None else g0), rep(v_L), **kw)


def test_r5_stopped_lead_10m_gives_the_braking_limited_profile_and_counts_saturated_steps():
    plan = straight(10.0)[None].astype(np.float32)
    out = call(plan, 10.0, 10.0, 0.0)
    assert out.active[0] and out.changed[0]
    v = np.diff(np.concatenate([[0.0], out.plan[0, :, 0].astype(np.float64)])) / DT
    # braking floor 6 m/s2: 10 -> 8.8 -> 7.6 -> ... -> 0 in 8 steps (1.2 m/s per 0.2 s); the cap (10 - D = 7 m) is already below the floor from the first step
    assert np.allclose(v[:8], [8.8, 7.6, 6.4, 5.2, 4.0, 2.8, 1.6, 0.4], atol=1e-4) and np.abs(v[8:]).max() < 1e-4
    assert abs(float(out.plan[0, -1, 0]) - 7.36) < 1e-4                       # the stop point of the braking limited profile: 10*0.2*(0.88 + 0.76 + ... + 0.04) = 7.36 m
    ref, sat = ref_rule(plan[0], 10.0, 10.0, 0.0)
    assert int(out.n_sat[0]) == sat == 30 and int(out.n_sat_exec[0]) == 10
    assert np.abs(out.a_new[0] - ref).max() < 1e-9
    assert np.abs(out.plan[0, :, 1]).max() == 0.0                              # lateral geometry unchanged (a straight path)


def test_r5_a_faster_lead_leaves_the_plan_unchanged_bit_for_bit():
    plan = straight(8.0)[None].astype(np.float32)
    out = call(plan, 8.0, 30.0, 30.0)
    assert out.active[0] and not out.changed[0] and int(out.n_sat[0]) == 0
    assert out.plan is plan or out.plan.tobytes() == plan.tobytes()
    assert out.plan.tobytes() == plan.tobytes()


def test_r5_the_lateral_geometry_is_kept_on_a_curved_plan():
    th = np.linspace(0, 1.2, 30)
    plan = np.stack([25 * np.sin(th), 25 * (1 - np.cos(th))], 1).astype(np.float32)[None]
    out = call(plan, 10.0, 15.0, 0.0)
    P, L, ext = R.plan_path(plan[0])
    d = np.stack([np.interp(out.a_new[0], L, P[:, 0]), np.interp(out.a_new[0], L, P[:, 1])], 1)
    assert out.changed[0] and np.abs(out.plan[0].astype(np.float64) - d).max() < 1e-4        # every new knot lies on the original polyline at the new arc length


def test_r5_progress_never_decreases_the_floor_holds_and_the_lag_form_equals_the_card_recursion():
    rng = np.random.default_rng(0)
    n_ch = 0
    for _ in range(300):
        v0 = rng.uniform(0, 20)
        v = np.clip(v0 + np.cumsum(rng.normal(0, 0.6, 30)) * DT, 0, 25)
        th = np.cumsum(rng.normal(0, 0.02, 30))
        plan = np.cumsum(np.stack([v * DT * np.cos(th), v * DT * np.sin(th)], 1), 0)
        g0, vL = rng.uniform(-5, 60), rng.uniform(0, 25)
        out = call(plan, v0, min(g0, 60.0), vL, g0=g0)
        a = out.a_new[0]
        ref, sat = ref_rule(plan.astype(np.float32), v0, g0, vL)
        assert np.abs(a - ref).max() < 1e-8 and int(out.n_sat[0]) == sat
        assert (np.diff(np.concatenate([[0.0], a])) >= -1e-12).all()               # progress never decreases (no reversing)
        vv = np.diff(np.concatenate([[0.0], a])) / DT
        vprev = np.concatenate([[v0], vv[:-1]])
        assert (vv >= np.maximum(0.0, vprev - R.B_MAX * DT) - 1e-9).all()          # the braking floor
        n_ch += int(out.changed[0])
    assert n_ch > 50                                                                # the checks above were exercised on windows the rule changed


def test_r5_inactive_windows_keep_their_plan_bit_for_bit_and_all_inactive_returns_the_same_object():
    rng = np.random.default_rng(1)
    plan = np.stack([straight(9.0) + rng.normal(0, 0.01, (30, 2)) for _ in range(6)]).astype(np.float32)
    v0 = np.full(6, 9.0)
    present = np.array([True, False, True, True, True, True])
    g_src = np.array([10.0, 10.0, 61.0, np.nan, 10.0, 10.0])
    g0 = np.array([10.0, 10.0, 61.0, np.nan, np.nan, 10.0])
    vL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.inf])
    out = R.distance_rule(plan, v0, present, g_src, g0, vL)
    assert out.active.tolist() == [True, False, False, False, False, False]
    assert out.changed.tolist() == [True, False, False, False, False, False]
    for b in range(1, 6):
        assert out.plan[b].tobytes() == plan[b].tobytes()
    assert out.plan[0].tobytes() != plan[0].tobytes()
    none = R.distance_rule(plan, v0, np.zeros(6, bool), g_src, g0, vL)
    assert none.plan is plan and not none.active.any()


def test_r5_cap_binds_only_where_it_should_a_lead_far_enough_ahead_changes_nothing():
    plan = straight(10.0)[None].astype(np.float32)
    out = call(plan, 10.0, 59.0, 10.0)                       # a lead 59 m ahead at the ego speed: the cap 59 - 12 + 2 i never binds
    assert out.active[0] and not out.changed[0] and out.plan.tobytes() == plan.tobytes()


def test_never_constants_as_written_still_bind_on_contact_and_the_floor_lifts_hard_braking_plans_the_finding_of_DEFS_E5():
    plan = straight(10.0)[None].astype(np.float32)
    kw = dict(D=-1e9, tau=0.0)
    thru = call(plan, 10.0, 20.0, 0.0, **kw)                  # a stopped lead 20 m ahead, plan passes through it: cap = the lead position (zero margin)
    assert thru.changed[0] and abs(float(thru.plan[0, -1, 0]) - 27.36) < 1e-3 and int(thru.n_sat[0]) > 0     # cap = the lead position, 20 m, but the braking floor stops the ego at 27.36 m: saturated
    free = call(plan, 10.0, 59.0, 10.0, **kw)
    assert not free.changed[0] and free.plan.tobytes() == plan.tobytes()
    hard = np.concatenate([straight(10.0, 1), np.full((29, 2), 0.0) + np.array([2.0, 0.0])])[None].astype(np.float32)     # a one-step stop from 10 m/s (50 m/s2)
    lift = call(hard, 10.0, 59.0, 10.0, **kw)
    assert lift.changed[0] and int(lift.n_lift[0]) >= 1 and int(lift.n_sat[0]) == 0
    plumb = call(hard, 10.0, 20.0, 0.0, use_cap=False, use_floor=False)
    assert plumb.active[0] and not plumb.changed[0] and plumb.plan.tobytes() == hard.tobytes()       # the identity probe
    push = call(plan, 10.0, 20.0, 0.0, push_mps=0.5)
    assert np.allclose(push.plan[0, :, 0] - plan[0, :, 0], 0.5 * DT * np.arange(1, 31), atol=1e-4)


def test_floor_cap_only_option_never_lifts_a_plan_and_equals_the_card_form_where_the_floor_does_not_lift():
    hard = np.concatenate([straight(10.0, 1), np.full((29, 2), 0.0) + np.array([2.0, 0.0])])[None].astype(np.float32)
    card = call(hard, 10.0, 59.0, 10.0, D=-1e9, tau=0.0)
    capo = call(hard, 10.0, 59.0, 10.0, D=-1e9, tau=0.0, floor_lifts=False)
    assert card.changed[0] and int(card.n_lift[0]) >= 1                          # the card form lifts a plan that wants to stop in one step
    assert not capo.changed[0] and int(capo.n_lift[0]) == 0 and capo.plan.tobytes() == hard.tobytes()
    rng = np.random.default_rng(3)
    n_eq = n_diff = 0
    for _ in range(300):
        v0 = rng.uniform(0, 20)
        v = np.clip(rng.uniform(0, 25) + np.cumsum(rng.normal(0, 1.5, 30)) * DT, 0, 25)      # a plan speed level independent of v0: a first-step mismatch above 1.2 m/s lets the floor lift it
        plan = np.cumsum(np.stack([v * DT, np.zeros(30)], 1), 0)
        g0, vL = rng.uniform(-5, 60), rng.uniform(0, 25)
        a = call(plan, v0, min(g0, 60.0), vL, g0=g0)
        b = call(plan, v0, min(g0, 60.0), vL, g0=g0, floor_lifts=False)
        p32 = plan.astype(np.float32).astype(np.float64)
        aP = np.cumsum(np.hypot(*np.diff(np.vstack([np.zeros((1, 2)), p32]), axis=0).T))
        assert (b.a_new[0] <= aP + 1e-9).all()                                    # cap-only: never ahead of the plan itself
        if int(a.n_lift[0]) == 0:
            n_eq += 1
            assert np.abs(a.a_new[0] - b.a_new[0]).max() < 1e-9
        else:
            n_diff += 1
    assert n_eq > 50 and n_diff > 5
    with pytest.raises(ValueError):
        W.RulePlanner(stub_planner, None, "label", 10, None, label=object(), floor="x")


# ============================================================================================ the progress geometry copy equals headway_read
def test_geometry_copy_equals_headway_read(tmp_path):
    """headway_read is run in a SUBPROCESS: importing it loads taniteval from another tree, whose stack guard then refuses the pinned tree in this process."""
    import subprocess
    rng = np.random.default_rng(2)
    g = np.cumsum(np.stack([rng.uniform(0.0, 2.2, (40, 30)), rng.normal(0, 0.3, (40, 30))], -1), 1)
    g[3] = 0.0                                                     # a degenerate (stopped) path
    pts = g + rng.normal(0, 1.5, g.shape)
    np.savez(tmp_path / "in.npz", g=g, pts=pts)
    code = ("import sys, numpy as np; sys.path.insert(0, %r); import headway_read as H; z = np.load(%r); "
            "np.savez(%r, arc=H.logged_arc(z['g']), prog=H.progress(z['g'], z['pts']), c=np.array([H.RAY, H.DT, H.STEP_MAX_M, H.GAP_MIN, H.GAP_MAX]))") % (
        HEADWAY_DIR, str(tmp_path / "in.npz"), str(tmp_path / "out.npz"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, encoding="utf-8", env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert r.returncode == 0, r.stderr[-800:]
    z = np.load(tmp_path / "out.npz")
    assert np.array_equal(R.logged_arc(g), z["arc"])
    assert np.array_equal(R.progress(g, pts), z["prog"])
    assert z["c"].tolist() == [R.RAY, R.DT, W.STEP_MAX_M, 0.0, W.GAP_MAX_M]


# ============================================================================================ the label source
def _rand_join(seed=5):
    rng = np.random.default_rng(seed)
    n = np.array([60, 45, 7, 80])
    off = np.concatenate([[0], np.cumsum(n)[:-1]])
    N = int(n.sum())
    valid = (rng.random(N) < 0.85).astype(np.int8)
    gap = 20 + np.cumsum(rng.normal(0, 1.5, N))
    gap[rng.random(N) < 0.05] += 8.0
    gap[rng.random(N) < 0.03] = np.nan
    gap[rng.random(N) < 0.02] = 75.0
    gap[rng.random(N) < 0.02] = -1.0
    return {"off": off, "n": n, "lead_valid": valid, "lead_gap": gap.astype(np.float32), "clip": np.array(["a", "b", "c", "d"])}


def test_label_source_equals_the_training_label_arrays_row_for_row():
    sys.path.insert(0, os.path.join(TREE, "stack"))
    T, _ = W.load_pinned(TREE)
    J = _rand_join()
    L = T.D.lead_sup_label_arrays(J)
    src = W.LabelSource(J)
    n_pres = n_clos = 0
    for ci in range(4):
        for j in range(int(J["n"][ci])):
            r = int(J["off"][ci]) + j
            pres, g, c = src.read(ci, j)
            assert pres == bool(np.isfinite(L["gap"][r]))
            if pres:
                n_pres += 1
                assert abs(g - float(L["gap"][r])) < 1e-4
                if np.isfinite(L["clos"][r]):
                    n_clos += 1
                    assert abs(c - float(L["clos"][r])) < 1e-4
                else:
                    assert c == 0.0                                  # fewer than 3 frames: closing speed 0
    assert n_pres > 100 and n_clos > 50


def test_r4_label_source_reads_no_row_after_the_tick():
    J = _rand_join(7)
    src = W.LabelSource(J)
    for ci in range(4):
        for j in range(int(J["n"][ci])):
            r = int(J["off"][ci]) + j
            J2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in J.items()}
            end = int(J["off"][ci]) + int(J["n"][ci])
            J2["lead_gap"][r + 1:end] = np.nan                       # every frame of this clip after the tick -> NaN
            J2["lead_valid"][r + 1:end] = 0
            a, b = src.read(ci, j), W.LabelSource(J2).read(ci, j)
            assert (a[0] == b[0]) and (a[1] == b[1] or (np.isnan(a[1]) and np.isnan(b[1]))) and a[2] == b[2]
    # the control that must fire: the same swap on a row at or before the tick DOES change the reading
    ci, j = 0, 30
    r = int(J["off"][ci]) + j
    J3 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in J.items()}
    J3["lead_gap"][r - 1] = J3["lead_gap"][r] - 2.5
    J3["lead_valid"][r - 2:r + 1] = 1
    J3["lead_gap"][r - 2] = J3["lead_gap"][r - 1] - 2.0
    J3["lead_gap"][r] = J3["lead_gap"][r - 1] - 2.0
    J4 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in J3.items()}
    J4["lead_gap"][r - 1] = np.nan
    assert W.LabelSource(J3).read(ci, j)[2] != W.LabelSource(J4).read(ci, j)[2]


# ============================================================================================ a synthetic world, a tiny checkpoint, the rollout
@pytest.fixture(scope="module")
def T():
    sys.path.insert(0, os.path.join(TREE, "stack"))
    sys.path.insert(0, os.path.join(TREE, "stack", "scripts"))
    mod, path = W.load_pinned(TREE)
    assert W.sha256_file(path) == PINNED_SHA
    return mod


@pytest.fixture(scope="module")
def world(tmp_path_factory, T):
    root = tmp_path_factory.mktemp("synth")
    w = T.D.make_synthetic_world(root, n_train=4, n_eval=3, T=70, seed=0)
    return {"root": str(root), "names": w["names"]}


def _lead_join(T, root, t_lead=7, gap0=40.0, name="join_lead.npz", const_gap=None):
    """The eval join with a stopped lead (label gap = gap0 - progress of the logged ego since t_lead, valid while 0 <= gap) on the frames t_lead .. , or a constant gap."""
    J = T.D.load_join(os.path.join(root, "join_eval.npz"))
    store = T.D.ClipStore(root, "eval", lru=8, kin_mode="yaw", tracks=T.D.load_native_tracks(os.path.join(root, "native_tracks.pt")))
    J = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in J.items()}
    for ci, cid in enumerate(J["clip"].tolist()):
        pose = store.get(cid)["pose"]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pose[:, :2], axis=0).T))])
        o, n = int(J["off"][ci]), int(J["n"][ci])
        g = (gap0 - (s - s[t_lead])) if const_gap is None else np.full(n, const_gap, np.float64)
        ok = np.zeros(n, bool)
        ok[t_lead:] = (g[t_lead:] >= 0)
        J["lead_valid"][o:o + n] = ok.astype(np.int8)
        J["lead_gap"][o:o + n] = np.where(ok, g, np.nan).astype(np.float32)
    path = os.path.join(root, name)
    T.D.save_join(J, path)
    return J, path


def _env(T, root, join_path, tick_windows=None):
    cfg_kw = {}
    J = T.D.load_join(join_path)
    tracks = T.D.load_native_tracks(os.path.join(root, "native_tracks.pt"))
    store = T.D.ClipStore(root, "eval", lru=8, kin_mode="yaw", tracks=tracks)
    wins = T.D.eval_grid_windows(os.path.join(root, "eval_grid_windows.json"), J)
    if tick_windows is not None:
        wins = [w for w in wins if w[1] in tick_windows]
    return store, tracks, J, wins


def _ds(T, J, cfg):
    proxy = T._OneClipStore()
    ds = T.D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    return proxy, ds


def stub_planner(batch):
    v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()
    k = np.arange(1, 31) * DT
    return np.stack([v0[:, None] * k[None], np.zeros((len(v0), 30))], -1).astype(np.float32)


@pytest.fixture(scope="module")
def tiny(T, world, tmp_path_factory):
    """A tiny model with a lead head (lead_sup on, grid state), a checkpoint of it and the config."""
    import torch
    import v8f_mini_train as TR
    M = T.M
    a = TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", "--wm-state", "grid", "--wm-cell-dim", "16", "--lead-sup", "on", "--w-lead-sup", "1.0"])
    c0 = TR.make_config(a)
    keep = {k: v for k, v in dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6).items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**c0.__dict__, **keep, "d": 32, "ff": 64})
    Jt = T.D.load_join(os.path.join(world["root"], "join_train.npz"))
    mu, sd = M.cons_stats(Jt)
    torch.manual_seed(0)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd).eval()
    with torch.no_grad():                                           # a head that says "present" on about half of the frames, with gaps in the metres range
        h = model.wm.lead_head
        h.mlp[2].bias.copy_(torch.tensor([0.0, 0.5, -0.3]))
        h.mlp[2].weight.mul_(8.0)
    rec = {k: v for k, v in cfg.__dict__.items() if (k not in M.LEADSUP_CFG_DEFAULTS or v != M.LEADSUP_CFG_DEFAULTS[k])}
    path = str(tmp_path_factory.mktemp("ck") / "ck.pt")
    torch.save({"model": model.state_dict(), "cfg": rec, "grid": "pool4x10"}, path)
    return {"cfg": cfg, "model": model, "ckpt": path}


def run_rollout(T, world, join_path, planner, tick_windows=(7, 39), cfg=None, m=10, mutate_store=None):
    store, tracks, J, wins = _env(T, world["root"], join_path, tick_windows)
    if mutate_store is not None:
        mutate_store(store, J)
    proxy, ds = _ds(T, J, cfg)
    states = W.attach_logged(T.make_states(store, tracks, J, wins, "deadreckon"), store, J)
    if isinstance(planner, W.RulePlanner):
        planner.bind(states)
    plan0, exe = T.rollout(planner, states, ds, proxy, m=m, schedule="periodic", device="cpu")
    return plan0, np.asarray(exe), wins, states, J, store


def make_rp(T, mode, J, m=10, model=None):
    label = W.LabelSource(J) if W.MODE_CFG[mode][0] == "label" else None
    return W.RulePlanner(stub_planner, model, mode, m, T.fk, label=label)


def arrays_equal(a, b):
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_e2e_inactive_wrapper_equals_the_unwrapped_rollout_exactly_for_every_mode(T, world, tiny):
    cfg = tiny["cfg"]
    J0, p0 = _lead_join(T, world["root"], name="join_free.npz", const_gap=None)
    for ci in range(3):                                             # no lead anywhere: the label says absent
        o, n = int(J0["off"][ci]), int(J0["n"][ci])
        J0["lead_valid"][o:o + n] = 0
    T.D.save_join(J0, p0)
    base = run_rollout(T, world, p0, stub_planner, cfg=cfg)
    for mode in ("label", "never", "push", "plumb"):
        J = T.D.load_join(p0)
        rp = make_rp(T, mode, J)
        got = run_rollout(T, world, p0, rp, cfg=cfg)
        assert arrays_equal(got[0], base[0]) and arrays_equal(got[1], base[1]), mode
        assert not rp.log.arrays()["active"].any() and len(rp.log.arrays()["ci"]) == 3 * len(got[2])


def test_e2e_label_with_a_stopped_lead_reduces_progress_push_increases_it_plumb_and_a_far_lead_change_nothing(T, world, tiny):
    cfg = tiny["cfg"]
    _, p = _lead_join(T, world["root"], t_lead=7, gap0=40.0, name="join_stop.npz")
    J = T.D.load_join(p)
    base = run_rollout(T, world, p, stub_planner, cfg=cfg)
    wins = base[2]
    is7 = np.array([w[1] == 7 for w in wins])
    assert is7.sum() == 3 and (~is7).sum() == 3
    prog = lambda exe: np.linalg.norm(exe[:, -1] - exe[:, 0], axis=-1)
    lab_rp = make_rp(T, "label", J)
    lab = run_rollout(T, world, p, lab_rp, cfg=cfg)
    push_rp = make_rp(T, "push", J)
    push = run_rollout(T, world, p, push_rp, cfg=cfg)
    plumb_rp = make_rp(T, "plumb", J)
    plumb = run_rollout(T, world, p, plumb_rp, cfg=cfg)
    never_rp = make_rp(T, "never", J)
    never = run_rollout(T, world, p, never_rp, cfg=cfg)
    pb, pl, pp = prog(base[1]), prog(lab[1]), prog(push[1])
    assert (pl[is7] < pb[is7] - 1.0).all(), (pb[is7], pl[is7])               # the label rule with a stopped lead reduces the progress of every lead window
    assert (pp[is7] > pb[is7] + 0.5).all()                                   # the planted push increases it
    assert arrays_equal(lab[1][~is7], base[1][~is7]) and arrays_equal(push[1][~is7], base[1][~is7])   # windows without a valid lead frame are untouched, bit for bit
    lg = lab_rp.log.arrays()
    assert lg["active"].sum() >= 3 and int(lg["sat"].sum()) > 0 and lg["active"][lg["j"] >= 0].dtype == bool
    assert plumb_rp.log.arrays()["active"].sum() == lg["active"].sum() > 0        # the same ticks are active ...
    assert arrays_equal(plumb[1], base[1]) and arrays_equal(plumb[0], base[0])    # ... and the identity probe returns every plan exactly
    assert never_rp.log.arrays()["changed"].sum() >= 1                            # DEFS E5: the card constants of r2 are not never-binding on a lead the plan reaches
    # the logged a_m of the rule equals the progress of the rolled path of the dump (the geometry of the rule is the geometry of the reader)
    from dist_rule import progress
    Jg = [np.asarray(s.d["gt"][s.t0], np.float64) for s in lab[3]]
    for row in range(len(lg["ci"])):
        w = [i for i, s in enumerate(lab[3]) if (s.ci, s.t0) == (int(lg["ci"][row]), int(lg["t0"][row]))][0]
        k = int(lg["k"][row])
        if k > 0:
            am = progress(np.stack([Jg[w]]), lab[1][w][k - 1][None, None, :])[0, 0]
            assert abs(am - lg["a_m"][row]) < 1e-6
        else:
            assert lg["a_m"][row] == 0.0 and lg["a_l"][row] == 0.0


def test_e2e_a_lead_at_the_ego_speed_leaves_every_mode_unchanged_while_active(T, world, tiny):
    cfg = tiny["cfg"]
    _, p = _lead_join(T, world["root"], name="join_const.npz", const_gap=32.0)
    J = T.D.load_join(p)
    base = run_rollout(T, world, p, stub_planner, cfg=cfg)
    for mode in ("label", "never", "plumb"):
        rp = make_rp(T, mode, J)
        got = run_rollout(T, world, p, rp, cfg=cfg)
        lg = rp.log.arrays()
        assert lg["active"].any() and not lg["changed"].any(), mode
        assert arrays_equal(got[1], base[1]) and arrays_equal(got[0], base[0]), mode


def test_mutation_the_line_that_applies_the_rule_removed_turns_the_reduction_check_red(T, world, tiny, monkeypatch):
    cfg = tiny["cfg"]
    _, p = _lead_join(T, world["root"], t_lead=7, gap0=40.0, name="join_stop2.npz")
    J = T.D.load_join(p)

    def reduction_holds():
        base = run_rollout(T, world, p, stub_planner, cfg=cfg)
        lab = run_rollout(T, world, p, make_rp(T, "label", J), cfg=cfg)
        is7 = np.array([w[1] == 7 for w in base[2]])
        prog = lambda exe: np.linalg.norm(exe[:, -1] - exe[:, 0], axis=-1)
        assert (prog(lab[1])[is7] < prog(base[1])[is7] - 1.0).all()

    reduction_holds()                                                          # green as built
    real = R.distance_rule
    monkeypatch.setattr(R, "distance_rule", lambda plan, *a, **k: real(plan, *a, **k)._replace(plan=plan))        # the rule computed but never applied
    with pytest.raises(AssertionError):
        reduction_holds()
    monkeypatch.undo()
    monkeypatch.setattr(R, "distance_rule", lambda plan, v0, present, *a, **k: real(plan, v0, np.zeros_like(present), *a, **k))      # the rule never active
    with pytest.raises(AssertionError):
        reduction_holds()


# ============================================================================================ the head source
def test_r6_head_source_equals_lead_dump_on_the_same_frames_and_r4_no_token_after_the_tick(T, world, tiny, monkeypatch):
    import torch
    import lead_dump as LD
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_head.npz", const_gap=20.0)

    def run(poison=None):
        rp = W.RulePlanner(stub_planner, model, "head", 10, T.fk)
        out = run_rollout(T, world, p, rp, tick_windows=tuple(range(3, 40, 4)), cfg=cfg, mutate_store=poison)
        return rp, out

    rp, out = run()
    lg = rp.log.arrays()
    n = len(lg["ci"])
    assert n >= 64
    store = out[5]
    n_cmp = 0
    for row in range(n):
        ci, j = int(lg["ci"][row]), int(lg["j"][row])
        if j < cfg.lead_hist - 1:
            assert not lg["present"][row]                                      # fewer than hist real frames behind the tick: the head says absent
            continue
        d = store.get(str(out[4]["clip"][ci]))
        f = torch.as_tensor(np.asarray(d["feats"][j - cfg.lead_hist + 1:j + 1]))[None]
        o = LD.head_outputs(model, cfg, f, "feats")[0, 0].numpy()
        assert abs(lg["p_src"][row] - 1.0 / (1.0 + np.exp(-o[0]))) < 1e-4
        assert abs(lg["g_src"][row] - o[1] * 50.0) < 1e-4 and abs(lg["c_src"][row] - o[2] * 10.0) < 1e-4
        n_cmp += 1
    assert n_cmp >= 64 and lg["active"].any() and (~lg["active"]).any() and lg["changed"].any()
    # r4: every camera token after the tick replaced by noise changes no output (the ticks of a window end at t0 + 20)
    last = {}
    for ci_, t0_ in zip(lg["ci"], lg["t0"]):
        last[(int(ci_), int(t0_))] = int(t0_) + 20

    def poison(store_, J_):
        rng = np.random.default_rng(3)
        for ci_ in range(3):
            d = store_.get(str(J_["clip"][ci_]))
            lo = min(v for (c, t), v in last.items() if c == ci_) + 1
            lo_full = max(last[(c, t)] for (c, t) in last if c == ci_) + 1
            d["feats"][lo_full:] = rng.normal(0, 5, d["feats"][lo_full:].shape).astype(d["feats"].dtype)

    rp2, out2 = run(poison)
    lg2 = rp2.log.arrays()
    for k in ("g_src", "c_src", "p_src", "active", "sat", "changed"):
        assert np.array_equal(lg[k], lg2[k], equal_nan=True), k
    assert arrays_equal(out[1], out2[1]) and arrays_equal(out[0], out2[0])
    # the control that fires: noise INTO frames at or before the last tick does change the reading
    def poison_inside(store_, J_):
        rng = np.random.default_rng(4)
        for ci_ in range(3):
            d = store_.get(str(J_["clip"][ci_]))
            d["feats"][10:40] = rng.normal(0, 5, d["feats"][10:40].shape).astype(d["feats"].dtype)

    rp3, out3 = run(poison_inside)
    assert not np.array_equal(lg["g_src"], rp3.log.arrays()["g_src"])


def test_head_source_refuses_a_model_without_a_lead_head(T):
    class Fake:
        wm = type("W", (), {"lead_head": None})()
    with pytest.raises(SystemExit, match="lead_sup"):
        W.RulePlanner(stub_planner, Fake(), "head", 10, T.fk)


# ============================================================================================ the CLI: argparse -> the plan (reachability), the dump layout, off equals the pinned tool
def _cli(T, world, tiny, out_dir, extra, join_path=None, rule=None, tag="t"):
    root = world["root"]
    argv = ["--tree", TREE] + (["--dist-rule", rule] if rule else []) + extra + [
        "run", "--ckpt", tiny["ckpt"], "--tag", tag, "--tracks", os.path.join(root, "native_tracks.pt"), "--eval-cache", root,
        "--eval-grid", os.path.join(root, "eval_grid_windows.json"), "--eval-join", join_path or os.path.join(root, "join_eval.npz"), "--out-dir", str(out_dir),
        "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--batch", "8", "--device", "cpu", "--no-analyze",
        "--dump-ticks", os.path.join(str(out_dir), "ticks"), "--dump-heads"]
    return W.main(argv)


def _dump_bytes(out_dir, tag):
    d = os.path.join(str(out_dir), f"dump_{tag}")
    files = sorted(f for f in os.listdir(d) if f.startswith("ep") and f.endswith(".npz"))
    res = {}
    for f in files:
        z = np.load(os.path.join(d, f))
        res[f] = {k: z[k].tobytes() for k in sorted(z.files)}
    return res


def test_r1_cli_off_equals_the_pinned_tool_dump_and_the_layout_is_the_pinned_layout(T, world, tiny, tmp_path):
    a, b = tmp_path / "off", tmp_path / "pinned"
    assert _cli(T, world, tiny, a, [], rule="off") == 0
    assert _cli(T, world, tiny, b, [], rule=None) == 0                 # the wrapper without the flag: the same call
    da, db = _dump_bytes(a, "t"), _dump_bytes(b, "t")
    assert da == db and len(da) == 3
    # the pinned tool called directly (its own main, no wrapper code on the path)
    c = tmp_path / "direct"
    root = world["root"]
    T.main(["run", "--ckpt", tiny["ckpt"], "--tag", "t", "--tracks", os.path.join(root, "native_tracks.pt"), "--eval-cache", root,
            "--eval-grid", os.path.join(root, "eval_grid_windows.json"), "--eval-join", os.path.join(root, "join_eval.npz"), "--out-dir", str(c),
            "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--batch", "8", "--device", "cpu",
            "--no-analyze", "--dump-ticks", os.path.join(str(c), "ticks"), "--dump-heads"])
    assert _dump_bytes(c, "t") == da
    ta, tc = np.load(a / "ticks" / "ticks_t.npz"), np.load(c / "ticks" / "ticks_t.npz")
    assert sorted(ta.files) == sorted(tc.files) and all(ta[k].tobytes() == tc[k].tobytes() for k in ta.files if ta[k].dtype.kind not in "UO")
    assert sorted(os.listdir(a)) == sorted(os.listdir(c)) == ["dump_t", "t1v8f_t.json", "ticks"]


def test_cli_label_and_head_are_reachable_from_argparse_to_the_plan_and_a_mutation_turns_it_red(T, world, tiny, tmp_path, monkeypatch):
    _, p = _lead_join(T, world["root"], t_lead=7, gap0=40.0, name="join_cli.npz")
    off = tmp_path / "off"
    assert _cli(T, world, tiny, off, [], join_path=p, rule="off") == 0
    lab = tmp_path / "lab"
    assert _cli(T, world, tiny, lab, ["--rule-log", str(tmp_path / "lab" / "ticks" / "rule_log_t.npz")], join_path=p, rule="label") == 0
    z = np.load(tmp_path / "lab" / "ticks" / "rule_log_t.npz")
    assert set(W.LOG_FIELDS) <= set(z.files) and str(z["mode"]) == "label" and str(z["pinned_sha256"]) == PINNED_SHA and len(str(z["join_sha256"])) == 64
    assert z["active"].sum() > 0 and z["changed"].sum() > 0
    assert _dump_bytes(lab, "t") != _dump_bytes(off, "t")
    ticks = np.load(lab / "ticks" / "ticks_t.npz")
    assert len(ticks["plan"]) == len(z["ci"])                                   # one rule-log row per (window, tick) of the model rollout
    hd = tmp_path / "head"
    assert _cli(T, world, tiny, hd, ["--rule-log", str(tmp_path / "head" / "ticks" / "rule_log_t.npz")], join_path=p, rule="head") == 0
    zh = np.load(tmp_path / "head" / "ticks" / "rule_log_t.npz")
    assert str(zh["source"]) == "head" and np.isfinite(zh["p_src"]).all() and zh["active"].sum() > 0
    # the mutation: the planner of the wrapper returns the base plan instead of the ruled one -> the dump equals off and the check above goes red
    real_call = W.RulePlanner.__call__

    def dead(self, batch):
        real_call(self, batch)
        return self.base(batch)

    monkeypatch.setattr(W.RulePlanner, "__call__", dead)
    dead_dir = tmp_path / "dead"
    assert _cli(T, world, tiny, dead_dir, [], join_path=p, rule="label") == 0
    assert _dump_bytes(dead_dir, "t") == _dump_bytes(off, "t")                  # the red form: no effect on the dump
    monkeypatch.undo()
    assert _dump_bytes(lab, "t") != _dump_bytes(dead_dir, "t")


def test_cli_refusals(T, world, tiny, tmp_path):
    with pytest.raises(SystemExit, match="silently writes nothing"):
        _cli(T, world, tiny, tmp_path / "x", ["--rule-log", str(tmp_path / "r.npz")], rule="off")
    with pytest.raises(SystemExit, match="no v8f_t1_eval.py"):
        W.main(["--tree", str(tmp_path), "run"])


# ============================================================================================ the pod script (checks, refusals, layout), run with a fake python and fake cgroup files
def _bash():
    import shutil
    for c in ("C:/Program Files/Git/usr/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe"):
        if os.path.exists(c):
            return c
    return shutil.which("bash")


def _pod_env(tmp_path, free_gb):
    w = tmp_path / "w"
    (w / "runs" / "Q1" / "HIER_s0").mkdir(parents=True)
    (w / "runs" / "Q1" / "HIER_s0" / "ckpt_last.pt").write_bytes(b"x")
    (w / "data" / "inputs").mkdir(parents=True)
    (w / "data" / "cache_pod").mkdir(parents=True)
    (w / "hierrep" / "pins").mkdir(parents=True)
    for f in ("data/inputs/native_tracks.pt", "hierrep/pins/eval_grid_windows.json", "hierrep/pins/join_eval141.npz"):
        (w / f).write_bytes(b"x")
    (tmp_path / "mm").write_text(str(int((free_gb + 5) * 2 ** 30)) + "\n")
    (tmp_path / "ms").write_text("anon %d\nfile 1\n" % int(5 * 2 ** 30))
    fake = tmp_path / "fakepy.sh"
    fake.write_bytes(b"""#!/usr/bin/env bash
# a stand-in for python: parse --out-dir / --tag / --rule-log / --dist-rule, lay out what the wrapper would
out=""; tag=""; rl=""; max=""
while [ $# -gt 0 ]; do case "$1" in --out-dir) out=$2; shift;; --tag) tag=$2; shift;; --rule-log) rl=$2; shift;; --max-windows) max=$2; shift;; esac; shift; done
mkdir -p "$out/dump_$tag" "$out/ticks"; for i in $(seq 0 140); do : > "$out/dump_$tag/ep$(printf %04d $i).npz"; done
: > "$out/t1v8f_$tag.json"; : > "$out/ticks/ticks_$tag.npz"; [ -z "$rl" ] || : > "$rl"
""")
    env = {**os.environ, "DR_W": str(w), "DR_PY": "bash", "DR_MEMMAX_FILE": str(tmp_path / "mm"), "DR_MEMSTAT_FILE": str(tmp_path / "ms"), "DR_TREE": TREE}
    return w, env, fake


def test_pod_script_refusals_layout_and_one_rollout(tmp_path):
    sh = os.path.join(HERE, "pod_distrule_run.sh")
    bash = _bash()
    import subprocess
    run = lambda env, *a: subprocess.run([bash, sh, *a], capture_output=True, encoding="utf-8", env=env)
    w, env, fake = _pod_env(tmp_path, free_gb=30)
    env["DR_PY"] = str(fake).replace("\\", "/")
    # the wrapper must sit next to the script: the script is run from HERE, which holds it
    r = run(env, "HIERRL_s0", "Q1", "HIER_s0", "nonsense")
    assert r.returncode == 2 and "MODE must be" in r.stderr
    r = run(env, "HIER RL", "Q1", "HIER_s0", "label")
    assert r.returncode == 2
    r = run(env, "HIERRL_s0", "Q1", "NOPE_s0", "label")
    assert r.returncode == 2 and "no checkpoint" in r.stderr
    env_low = {**env, "DR_MEMMAX_FILE": str(tmp_path / "mm_low")}
    (tmp_path / "mm_low").write_text(str(int(24 * 2 ** 30)) + "\n")                     # 24 GB cap, 5 GB anon: 19 GB free < 20
    r = run(env_low, "HIERRL_s0", "Q1", "HIER_s0", "label")
    assert r.returncode == 2 and "free container memory" in r.stderr
    assert not (w / "runs" / "distrule").exists()                                         # a refusal creates nothing
    d = run({**env, "DR_DRY": "1"}, "HIERRL_s0", "Q1", "HIER_s0", "label")
    assert d.returncode == 0 and "--dist-rule label" in d.stdout and "--rule-log" in d.stdout and "--replan-steps 10" in d.stdout and "--dump-heads" in d.stdout
    assert not (w / "runs" / "distrule").exists()
    r = run(env, "HIERRL_s0", "Q1", "HIER_s0", "label")
    assert r.returncode == 0, r.stderr
    out = w / "runs" / "distrule" / "t1" / "HIERRL_s0"
    tag = "HIERRL_s0_m10_causal_heads"
    assert (out / "T1_DONE").exists() and len(list((out / f"dump_{tag}").glob("ep*.npz"))) == 141
    assert (out / "ticks" / "rule_log_HIERRL_s0.npz").exists() and (out / f"t1v8f_{tag}.json").exists() and (out / "ticks" / "rule_run_HIERRL_s0.txt").exists()
    rec = (out / "ticks" / "rule_run_HIERRL_s0.txt").read_text()
    assert "mode label floor card" in rec and PINNED_SHA in rec
    r2 = run(env, "HIERRL_s0", "Q1", "HIER_s0", "label")
    assert r2.returncode == 2 and "output exists" in r2.stderr                             # never overwrites
    ro = run(env, "HIERRO_s0", "Q1", "HIER_s0", "off")
    assert ro.returncode == 0 and not (w / "runs" / "distrule" / "t1" / "HIERRO_s0" / "ticks" / "rule_log_HIERRO_s0.npz").exists()    # off writes no rule log, as the wrapper


def test_files_for_the_pod_have_lf_endings_and_the_pinned_tool_is_byte_identical():
    for f in ("dist_rule.py", "v8f_t1_eval_rule.py", "pod_distrule_run.sh"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            assert b"\r" not in open(p, "rb").read(), f
    assert W.sha256_file(W.find_pinned(TREE)) == PINNED_SHA
