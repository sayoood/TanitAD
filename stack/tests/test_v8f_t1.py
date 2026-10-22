"""v8F-mini T1-kin (``stack/scripts/v8f_t1_eval.py``): CPU only, on the synthetic world of ``v8f_mini_data.make_synthetic_world``.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval;<repo>/stack/scripts  CUDA_VISIBLE_DEVICES=-1  pytest -q stack/tests/test_v8f_t1.py

What is pinned, and the control that must go red for each:
  * K    the Hermite fill recovers analytic (v, a, kappa) on a constant-acceleration line and a constant-curvature arc.
  * C1   executor = the LOGGED motion, nav logged, scene logged: every key the model reads is BIT-identical to the open-loop batch of
         window (clip, t_n) at every tick, and the plan equals the open-loop plan. (Sensitive: with the dead-reckoned nav it breaks.)
  * C2   --replan-steps 30: one tick, the whole plan executed => the executed path IS the open-loop plan.
  * C3   perturb the executed action => the NEXT tick's p10f / base / pose_* / nav change (the loop is closed on state); tick 0 does not.
  * C4   poison every logged ego row after t0 (NaN and garbage) => T1 output unchanged bit-for-bit, while the open-loop item builder
         over the same poisoned store changes. A deliberately leaky state class makes the guard FAIL.
  * the model is handed exactly MODEL_KEYS; `act` / `pose` are not read; dead-reckoned nav against hand values; floor == run_o's p10.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_t1_eval as T1  # noqa: E402

torch.set_num_threads(1)
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, kin_mode="yaw")
A_COEF, V_COEF = 1.5, 8.0


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_t1")
    w = D.make_synthetic_world(root, n_train=2, n_eval=3, T=80, seed=3)
    return {"root": root, **w}


def setup(world, hmax=10, lru=8):
    root = Path(world["root"])
    tracks = D.load_native_tracks(root / "native_tracks.pt")
    J = D.load_join(root / "join_eval.npz")
    store = D.ClipStore(root, "eval", lru=lru, kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(root / "eval_grid_windows.json", J)
    return tracks, J, store, wins


def tiny_model(J, seed=0, residual=0.02):
    cfg = M.V8FMiniConfig(**TINY)
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    prior = M.P10Prior.synthetic(0)
    with torch.no_grad():                         # a random prior is an UNSTABLE closed loop (measured: 1e19 m by tick 4); keep it tame
        for q in prior.parameters():
            q.mul_(0.02)
    model = M.V8FMini(cfg, prior, mu, sd).eval()
    with torch.no_grad():                         # head_o is zero-init: give the residual something to do so model != floor
        torch.nn.init.normal_(model.head_o[2].weight, std=residual)
    return model, cfg


def pick(wins, n_per_clip=2, clips=(0, 1)):
    out = []
    for c in clips:
        out += [w for w in wins if w[0] == c][:n_per_clip]
    return out


def ds_for(cfg, J, hmax=None):
    proxy = T1._OneClipStore()
    return T1.D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=hmax or cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s), proxy


def run(model, cfg, store, tracks, J, chunk, *, nav="deadreckon", m=1, planner=None, hook=None, state_cls=None, **kw):
    ds, proxy = ds_for(cfg, J)
    if state_cls is None:
        states = T1.make_states(store, tracks, J, chunk, nav)
    else:
        states = []
        for ci, t0, _, _ in chunk:
            clip = str(J["clip"][ci])
            o, n = int(J["off"][ci]), int(J["n"][ci])
            states.append(state_cls(store.get(clip), tracks[clip], J["nav"][o:o + n], ci, t0, nav))
    planner = planner or T1.model_planner(model, "cpu")
    plan0, exe = T1.rollout(planner, states, ds, proxy, m=m, device="cpu", hook=hook, **kw)
    return plan0, np.asarray(exe), states, planner


# ------------------------------------------------------------------------------------------------ K: the Hermite fill
def test_hermite_recovers_constant_acceleration_line():
    s_of = lambda tau: V_COEF * tau + 0.5 * A_COEF * tau ** 2
    t_knot = (np.arange(0, 9) - 1) * T1.DT                                   # knots j-1 .. j+7; knot j is tau = 0
    knots = np.column_stack([s_of(t_knot), np.zeros(9)])
    xy, psi, v = T1.hermite_frames(knots[0], knots[1:], 4, 0.0)
    tau = (np.arange(8) + 1) * 0.1                                           # raw frames 2j+1 .. 2j+8
    np.testing.assert_allclose(xy[:, 0], s_of(tau), atol=1e-9)
    np.testing.assert_allclose(v, V_COEF + A_COEF * tau, atol=1e-9)
    np.testing.assert_allclose(psi, 0.0, atol=1e-12)
    # through the project's ONE spelling: the track = exact prefix (raw 0..6, tau = -0.6..0) + the Hermite fill (raw 7..14)
    trk = np.zeros((15, 4))
    tau_all = (np.arange(15) - 6) * 0.1
    trk[:, 0], trk[:, 3] = s_of(tau_all), V_COEF + A_COEF * tau_all
    trk[7:, 0], trk[7:, 3] = xy[:, 0], v
    kin, _ = D.yaw_kin_from_track(trk)                                       # rows j = 0..7 = raw 0, 2, ..., 14
    np.testing.assert_allclose(kin[3:, 1], A_COEF, atol=1e-9)                # a0 at every cache tick from j on
    np.testing.assert_allclose(kin[3:, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(kin[3:, 0], V_COEF + A_COEF * tau_all[6::2], atol=1e-9)


def test_hermite_recovers_constant_curvature_arc():
    R, v0 = 30.0, 9.0
    th = lambda tau: v0 * tau / R
    arc = lambda tau: np.column_stack([R * np.sin(th(tau)), R * (1 - np.cos(th(tau)))])
    ti = np.arange(-1, 4) * T1.DT
    knots = arc(ti)
    xy, psi, v = T1.hermite_frames(knots[0], knots[1:], 2, 0.0)
    tau = (np.arange(4) + 1) * 0.1
    np.testing.assert_allclose(v, v0, rtol=2e-3)
    np.testing.assert_allclose(psi, th(tau), atol=2e-3)
    om = np.diff(np.concatenate([[0.0], psi])) / 0.1                         # one-frame backward yaw rate
    np.testing.assert_allclose(om[1:], v0 / R, rtol=2e-2)                   # kappa0 = omega / v = 1/R within 2 %


def test_envelope_leaves_sane_plans_bit_identical_and_bounds_insane_ones():
    sane = np.column_stack([2.0 * np.arange(1, 31), 0.1 * np.arange(1, 31)])
    out, hit = T1.envelope(sane)
    assert hit is False and np.array_equal(out, sane)
    wild = sane.copy()
    wild[7:] += 1e6
    wild[20, 0] = np.nan
    out, hit = T1.envelope(wild)
    assert hit is True and np.isfinite(out).all()
    steps = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), out]), axis=0), axis=1)
    assert steps.max() <= T1.V_ENV_MS * T1.DT + 1e-9
    np.testing.assert_allclose(out[:7], sane[:7])                              # the sane prefix is untouched


def test_hermite_holds_heading_at_standstill():
    knots = np.zeros((4, 2))
    xy, psi, v = T1.hermite_frames(np.zeros(2), knots, 2, 1.234)
    np.testing.assert_allclose(psi, 1.234)
    np.testing.assert_allclose(v, 0.0)


# ------------------------------------------------------------------------------------------------ state construction: the future does not exist
def test_window_state_has_no_logged_ego_after_t0(world):
    tracks, J, store, wins = setup(world)
    ci, t0, _, _ = wins[1]
    clip = str(J["clip"][ci])
    o, n = int(J["off"][ci]), int(J["n"][ci])
    st = T1.WindowState(store.get(clip), tracks[clip], J["nav"][o:o + n], ci, t0, "deadreckon")
    for k in T1.WindowState.EGO:
        assert np.isnan(st.d[k][t0 + 1:]).all(), k
        assert np.isfinite(st.d[k][1:t0 + 1]).all() or k in ("a",), k
        np.testing.assert_array_equal(st.d[k][:t0 + 1], store.get(clip)[k][:t0 + 1])
    assert np.isnan(st.track[2 * t0 + 1:]).all() and np.isfinite(st.track[2 * t0, :4]).all()
    assert np.isnan(st.nav[t0 + 1:]).all()
    assert st.d["pose"] is not store.get(clip)["pose"]                         # a copy: writing the view never touches the store


# ------------------------------------------------------------------------------------------------ the allowlist
def test_model_gets_exactly_the_keys_it_reads(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    seen = {}
    hook = lambda n, b, p, s: seen.setdefault(n, sorted(b))
    _, _, _, planner = run(model, cfg, store, tracks, J, pick(wins, 1), m=15, hook=hook)
    assert set(seen[0]) == set(T1.MODEL_KEYS)
    assert not ({"act", "pose"} & set(seen[0])) and not any(k.startswith("lab") or k in ("gt", "s_h", "lead_gap_h") for k in seen[0])
    assert planner.keys_read <= set(T1.MODEL_KEYS) and planner.keys_read <= M.INFERENCE_INPUT_KEYS
    assert "act" not in planner.keys_read and "pose" not in planner.keys_read
    # and a model that DID read a dropped key would fail loud
    with pytest.raises(KeyError):
        _kg = M._KeyGuard({k: 1 for k in T1.MODEL_KEYS})["act"]


# ------------------------------------------------------------------------------------------------ C1: exact reproduction
def _open_batch(ds_open, ci, t, a_t, a_s):
    full = D.collate([ds_open.item(ci, t, a_t, a_s)])
    return {k: full[k] for k in T1.MODEL_KEYS}


def test_c1_logged_executor_reproduces_open_loop_exactly(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    chunk = pick(wins, 2, clips=(0, 1, 2))
    ds_open = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    rec, worst = [], {"key": 0.0, "plan": 0.0}

    def hook(n, batch, plan, states):
        a_t, a_s = T1.ages_for(n * 1, "periodic")
        full = D.collate([ds_open.item(st.ci, st.j, a_t, a_s) for st in states])        # the open-loop batch of the SAME windows
        ob = {k: full[k] for k in T1.MODEL_KEYS}
        for k in T1.MODEL_KEYS:
            assert torch.equal(batch[k], ob[k]), f"tick {n} key {k}: T1 input differs from the open-loop batch"
        with torch.no_grad():
            op = model.forward_infer(ob)["traj30"].numpy()
        worst["plan"] = max(worst["plan"], float(np.abs(plan - op).max()))
        worst["key"] += 1
        rec.append(n)

    ds, proxy = ds_for(cfg, J)
    states = T1.make_states(store, tracks, J, chunk, "logged")
    T1.rollout(T1.model_planner(model, "cpu"), states, ds, proxy, m=1, executor="logged", logged_tracks=tracks_by_ci(J, tracks), hook=hook)
    assert len(rec) == 30 and worst["key"] == 30
    assert worst["plan"] == 0.0, f"plan differs from the open-loop plan by {worst['plan']}"      # same inputs, same batch => bitwise


def tracks_by_ci(J, tracks):
    return {ci: tracks[str(J["clip"][ci])] for ci in range(len(J["clip"]))}


def test_c1_is_sensitive_the_deadreckoned_nav_breaks_it(world):
    """The control must be able to go RED: with the dead-reckoned route the batch is NOT the open-loop one (synthetic nav is random)."""
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    ds_open = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    diffs = []

    def hook(n, batch, plan, states):
        if n == 5:
            for i, st in enumerate(states):
                ob = _open_batch(ds_open, st.ci, st.j, *T1.ages_for(5, "periodic"))
                diffs.append(not torch.equal(batch["nav"][i:i + 1], ob["nav"]))
                assert torch.equal(batch["p10f"][i:i + 1], ob["p10f"])               # the ego part is still the logged one
    ds, proxy = ds_for(cfg, J)
    states = T1.make_states(store, tracks, J, pick(wins, 2), "deadreckon")
    T1.rollout(T1.model_planner(model, "cpu"), states, ds, proxy, m=1, executor="logged", logged_tracks=tracks_by_ci(J, tracks), hook=hook)
    assert any(diffs)


# ------------------------------------------------------------------------------------------------ C2: m = 30 is the open loop
def test_c2_replan_30_executes_the_open_loop_plan(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    plan0, exe, _, _ = run(model, cfg, store, tracks, J, pick(wins, 2, clips=(0, 1, 2)), m=30)
    assert exe.shape == plan0.shape == (6, 30, 2)
    assert np.abs(exe - plan0.astype(np.float64)).max() < 1e-9, np.abs(exe - plan0).max()


def test_executing_m_knots_commits_exactly_those_knots(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    seen = {}

    def hook(n, b, plan, states):
        if n == 0:
            seen["plan"] = plan.copy()
            seen["pose"] = [st.d["pose"][st.j].copy() for st in states]
    _, exe, states, _ = run(model, cfg, store, tracks, J, pick(wins, 1), m=3, hook=hook)
    for i, st in enumerate(states):
        W = T1.plan_to_world(seen["plan"][i].astype(np.float64), seen["pose"][i])
        j0 = st.t0
        np.testing.assert_allclose(st.track[2 * (j0 + 1):2 * (j0 + 3) + 1:2, :2], W[:3], atol=1e-9)    # knots j0+1..j0+3 = plan knots 1..3
        assert np.isfinite(st.track[2 * j0: 2 * (j0 + 30) + 1]).all()


# ------------------------------------------------------------------------------------------------ C3: the loop is closed on state
def test_c3_perturbing_the_executed_action_changes_the_next_ticks_inputs(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    chunk = pick(wins, 2)
    snaps = {"base": {}, "pert": {}}

    def mk_hook(tag):
        def hook(n, batch, plan, states):
            if n in (0, 1, 4):
                snaps[tag][n] = {k: batch[k].clone() for k in ("p10f", "base", "pose_now", "pose_t", "pose_s", "nav")}
                snaps[tag][(n, "plan")] = plan.copy()
        return hook

    class Shifted:
        """The same planner, but the plan it executes at tick 0 is moved 0.6 m sideways and 0.5 m ahead."""
        def __init__(self, inner):
            self.inner, self.k = inner, 0
        def __call__(self, b):
            p = self.inner(b)
            if self.k == 0:
                p = p + np.array([0.5, 0.6], np.float32)
            self.k += 1
            return p
    run(model, cfg, store, tracks, J, chunk, hook=mk_hook("base"))
    run(model, cfg, store, tracks, J, chunk, hook=mk_hook("pert"), planner=Shifted(T1.model_planner(model, "cpu")))
    for k in ("p10f", "base", "pose_now", "pose_t", "pose_s", "nav"):
        assert torch.equal(snaps["base"][0][k], snaps["pert"][0][k]), f"tick 0 must be untouched ({k})"
    for k in ("p10f", "base", "pose_now", "nav"):
        assert not torch.equal(snaps["base"][1][k], snaps["pert"][1][k]), f"tick 1 {k} must react to the executed action"
    assert not np.array_equal(snaps["base"][(1, "plan")], snaps["pert"][(1, "plan")])
    assert not np.array_equal(snaps["base"][(4, "plan")], snaps["pert"][(4, "plan")])


# ------------------------------------------------------------------------------------------------ C4: the leak guard
class LeakyState(T1.WindowState):
    """THE MUTATION: forgets to replace the ego rows -- after each commit it copies the LOGGED rows of the new steps back in."""

    def __init__(self, base, track, nav_rows, ci, t0, nav_mode="deadreckon"):
        super().__init__(base, track, nav_rows, ci, t0, nav_mode)
        self._logged = base

    def _finish(self, j_old, j_new):
        super()._finish(j_old, j_new)
        for k in ("p10f", "base", "pose", "kin3"):
            self.d[k][j_old + 1:j_new + 1] = self._logged[k][j_old + 1:j_new + 1]


def _poison(store, tracks, J, chunk, mode):
    """Overwrite every logged ego row after the LAST t0 of each clip in ``chunk`` (so no window's own history is touched) in the store
    and in the tracks. Returns an undo closure."""
    saved = []
    rng = np.random.default_rng(0)
    last_t0 = {}
    for ci, t0, _, _ in chunk:
        last_t0[ci] = max(t0, last_t0.get(ci, -1))
    for ci, t0 in last_t0.items():
        clip = str(J["clip"][ci])
        d = store.get(clip)
        for k in T1.WindowState.EGO:
            saved.append((d[k], d[k][t0 + 1:].copy(), t0 + 1))
            d[k][t0 + 1:] = np.nan if mode == "nan" else (rng.normal(size=d[k][t0 + 1:].shape) * 1e3).astype(d[k].dtype)
        tr = tracks[clip]
        saved.append((tr, tr[2 * t0 + 1:].copy(), 2 * t0 + 1))
        tr[2 * t0 + 1:] = np.nan if mode == "nan" else rng.normal(size=tr[2 * t0 + 1:].shape) * 1e3

    def undo():
        for arr, old, s in saved:
            arr[s:] = old
    return undo


def _leak_guard(world, state_cls):
    """True iff poisoning the logged ego after t0 leaves the T1 output untouched bit-for-bit."""
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    chunk = pick(wins, 1, clips=(0, 1, 2))                                       # ONE window per clip: every logged row after its t0 is poisoned
    _, clean, _, _ = run(model, cfg, store, tracks, J, chunk, state_cls=state_cls)
    same = True
    for mode in ("nan", "garbage"):
        undo = _poison(store, tracks, J, chunk, mode)
        try:
            _, dirty, _, _ = run(model, cfg, store, tracks, J, chunk, state_cls=state_cls)
        finally:
            undo()
        same &= bool(np.array_equal(clean, dirty, equal_nan=False)) and bool(np.isfinite(dirty).all())
    return same


def test_c4_poisoned_logged_ego_after_t0_changes_nothing(world):
    assert _leak_guard(world, T1.WindowState) is True


def test_c4_mutation_a_leaky_state_makes_the_guard_fail(world):
    assert _leak_guard(world, LeakyState) is False


def test_c4_the_open_loop_item_builder_does_read_those_rows(world):
    """Positive control: the same poison DOES change the open-loop batch at a later step -- so 'unchanged' above means something."""
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    ds_open = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    ci, t0, _, _ = wins[1]
    before = D.collate([ds_open.item(ci, t0 + 7, 0, 0)])
    undo = _poison(store, tracks, J, [wins[1]], "garbage")
    try:
        after = D.collate([ds_open.item(ci, t0 + 7, 0, 0)])
    finally:
        undo()
    assert not torch.equal(before["p10f"], after["p10f"]) and not torch.equal(before["base"], after["base"])


# ------------------------------------------------------------------------------------------------ dead-reckoned nav, hand values
def _state_with_nav(world, nav0):
    tracks, J, store, wins = setup(world)
    ci, t0, _, _ = wins[1]
    clip = str(J["clip"][ci])
    o, n = int(J["off"][ci]), int(J["n"][ci])
    rows = J["nav"][o:o + n].copy()
    rows[t0] = np.asarray(nav0, np.float32)
    return T1.WindowState(store.get(clip), tracks[clip], rows, ci, t0, "deadreckon"), t0


def test_deadreckoned_nav_counts_down_by_the_executed_arc_and_clamps_at_zero(world):
    st, t0 = _state_with_nav(world, [0, 1, 0, 0.50, 1])                         # TURN_L, 50 m away, args valid
    v = 10.0
    plan = np.column_stack([v * T1.DT * np.arange(1, 31), np.zeros(30)])         # straight, 2.0 m per step
    for step in range(1, 4):
        st.commit_plan(plan, 1)
        # arc of one step = |pose[j] - pose[j-1]|: the first step starts at the logged pose, so it is exactly 2.0 m
    assert st.arc[t0 + 1] == pytest.approx(2.0, abs=1e-9)
    np.testing.assert_allclose(st.nav[t0 + 1, 3], 0.50 - 2.0 / 100.0, atol=1e-6)
    np.testing.assert_allclose(st.nav[t0 + 3, 3], 0.50 - st.arc[t0 + 3] / 100.0, atol=1e-6)
    np.testing.assert_array_equal(st.nav[t0 + 3, :3], [0, 1, 0])                  # token held
    assert st.nav[t0 + 3, 4] == 1.0
    big, t1 = _state_with_nav(world, [0, 0, 1, 0.02, 1])                          # 2 m away: passed after one step, clamped at 0, token held
    big.commit_plan(plan, 1)
    big.commit_plan(plan, 1)
    assert big.nav[t1 + 2, 3] == 0.0 and big.nav[t1 + 2, 2] == 1.0
    inv, t2 = _state_with_nav(world, [1, 0, 0, 0.0, 0])                           # args invalid: nothing to count down
    inv.commit_plan(plan, 1)
    np.testing.assert_array_equal(inv.nav[t2 + 1], [1, 0, 0, 0, 0])


def test_nav_rows_up_to_t0_stay_logged_and_future_rows_are_nan_until_executed(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    ds, proxy = ds_for(cfg, J)
    states = T1.make_states(store, tracks, J, [wins[1]], "deadreckon")
    st = states[0]
    o = int(J["off"][st.ci])
    np.testing.assert_array_equal(st.nav[:st.t0 + 1], J["nav"][o:o + st.t0 + 1])
    assert np.isnan(st.nav[st.t0 + 1:]).all()


# ------------------------------------------------------------------------------------------------ floor, scene, limit
def test_floor_planner_is_the_prior_run_o_uses(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    ds, proxy = ds_for(cfg, J)
    st = T1.make_states(store, tracks, J, pick(wins, 2), "deadreckon")
    full = D.collate([T1.build_item(ds, proxy, s, 0, 0, 0, "logged", "logged") for s in st])
    batch = {k: full[k] for k in T1.MODEL_KEYS}
    with torch.no_grad():
        ref = model.forward_infer(batch)["p10"].numpy()
    np.testing.assert_array_equal(T1.floor_planner(model, "cpu")(batch), ref)


def test_scene_frozen_and_limit_off_knobs(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    seen = {}

    def hook(n, b, plan, states):
        seen[n] = {k: b[k].clone() for k in ("feats", "frame_ok", "limit", "limit_ok")}
    run(model, cfg, store, tracks, J, pick(wins, 1), m=5, hook=hook, scene="frozen", limit="off")
    for n in range(1, 6):
        assert torch.equal(seen[n]["feats"], seen[0]["feats"]) and torch.equal(seen[n]["frame_ok"], seen[0]["frame_ok"])
    assert float(seen[0]["limit"].abs().sum()) == 0.0 and not bool(seen[0]["limit_ok"].any())
    seen2 = {}
    run(model, cfg, store, tracks, J, pick(wins, 1), m=5, hook=lambda n, b, p, s: seen2.setdefault(n, b["feats"].clone()))
    assert not torch.equal(seen2[1], seen2[0])                                    # logged scene advances


def test_ages_schedule_is_nested_and_inside_the_trained_support():
    cfg = M.V8FMiniConfig(**TINY)
    for s in range(0, 31):
        a_t, a_s = T1.ages_for(s, "periodic")
        assert 0 <= a_t <= cfg.max_age_t and a_t <= a_s <= cfg.max_age_s
    assert T1.ages_for(17, "fresh") == (0, 0)
    with pytest.raises(ValueError):
        T1.ages_for(1, "periodic", k_t=4, k_s=6)


def test_replan_steps_must_divide_the_horizon(world):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    with pytest.raises(ValueError):
        run(model, cfg, store, tracks, J, pick(wins, 1), m=7)


# ------------------------------------------------------------------------------------------------ end to end: dumps -> t1_eval.analyze
def test_cli_run_end_to_end_writes_tier_stamped_scores(world, tmp_path):
    tracks, J, store, wins = setup(world)
    model, cfg = tiny_model(J)
    ck = {"cfg": dataclasses.asdict(cfg), "model": model.state_dict(), "grid": "pool4x10"}
    torch.save(ck, tmp_path / "ck.pt")
    root = Path(world["root"])
    rc = T1.main(["run", "--ckpt", str(tmp_path / "ck.pt"), "--tag", "syn", "--tracks", str(root / "native_tracks.pt"),
                  "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-join", str(root / "join_eval.npz"),
                  "--out-dir", str(tmp_path / "out"), "--replan-steps", "5", "--batch", "6", "--max-windows", "12", "--n-boot", "20"])
    assert rc == 0
    import json
    res = json.load(open(tmp_path / "out" / "t1v8f_syn.json"))
    assert res["arms"]["cl"]["tier"] == "T1" and res["arms"]["fl"]["tier"] == "T1" and res["arms"]["ol"]["tier"] == "T0"
    assert "closed" not in res["_tier_doctrine"].lower().replace("only closed loop", "").replace("the only closed", "") or "OPEN" in res["_tier_doctrine"]
    h = res["t1v8f"]
    assert h["replan_steps"] == 5 and h["nav"] == "deadreckon" and h["scene"] == "logged" and h["limit"] == "logged"
    assert h["strict"] == {"kinematics": True, "route": True, "camera": False, "speed_bucket": False}
    assert set(h["model_keys_read"]) <= set(T1.MODEL_KEYS)
    assert "paired_t1_minus_open" in res["paired_decision_grade"] and "paired_model_minus_floor_t1" in res["paired_decision_grade"]
    assert res["horizon_steps"] == 30 and res["dt_s"] == 0.2 and res["n_windows"] == 12
    for fam in ("longitudinal", "lateral", "tactical", "strategic"):
        assert fam in res["arms"]["cl"]["four_families"]


def test_refuses_a_steer_mode_checkpoint(world, tmp_path):
    tracks, J, store, wins = setup(world)
    cfg = dataclasses.asdict(M.V8FMiniConfig(**{**TINY, "kin_mode": "steer"}))
    torch.save({"cfg": cfg, "model": {}}, tmp_path / "ck.pt")
    root = Path(world["root"])
    with pytest.raises(SystemExit, match="REFUSED"):
        T1.main(["run", "--ckpt", str(tmp_path / "ck.pt"), "--tag", "x", "--tracks", str(root / "native_tracks.pt"),
                 "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(tmp_path)])


REAL_CKPT = Path("C:/Users/Admin/box2_ship/logs/v8f_mini/runs/M2b_s0/ckpt_last.pt")


@pytest.mark.skipif(not REAL_CKPT.exists(), reason="local M2b_s0 checkpoint absent")
def test_real_m2b_checkpoint_rolls_on_the_synthetic_world(world, tmp_path):
    """PLUMBING ONLY: the real weights (hmax 16, WM aux 'b', frozen P10) load into the loop and roll finite outputs on synthetic
    features. The numbers mean nothing -- the real eval cache is not on this machine."""
    ck = torch.load(REAL_CKPT, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    tracks, J, store, wins = setup(world)
    plan0, exe, states, planner = run(model, cfg, store, tracks, J, pick(wins, 1, clips=(0, 1)), m=5)
    assert exe.shape == (2, 30, 2) and np.isfinite(exe).all() and np.isfinite(plan0).all()
    assert planner.keys_read <= set(T1.MODEL_KEYS)
