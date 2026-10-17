"""v8F-mini: model, data join, penalties, launch gate, end-to-end smoke. CPU only, < 90 s.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_v8f_mini.py

What is pinned, and the control that must go red for each:
  * P10 prior == floors_kin.predict_p10; a ZERO-INIT residual reproduces P10 EXACTLY (and a perturbed head does not).
  * sample-and-hold: T reads frame F-1-a_t, S frame F-1-a_s, O frame F-1; scheduled sampling p_gt 1 / 0 picks GT / predicted.
  * navonly BUILDS no scene path into S and T (outputs invariant to the features), M1 does not.
  * the inference path reads exactly INFERENCE_INPUT_KEYS and no label; an injected label read makes the check fail.
  * the torch hinge penalties AGREE with the numpy frozen rules (hinge > 0  <=>  violated) on 400 random cases.
  * the hard speed/Kamm projection: known values.
  * the v9 -> cache join (k = 2 j), disjointness, nested data fractions, the 5 hand-checked real windows (skipped if the
    local release is absent).
  * the launch gate: declared-vs-built refuses a lie, refuses an unchecked flag, refuses a lever that does nothing;
    live-check passes for every arm and reads the frozen prior at exactly 0.0.
  * end to end: train 8 steps -> resume after a crash == uninterrupted -> eval -> summary.json {"done": true}.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from taniteval import floors_kin as fk  # noqa: E402
from taniteval import v8f_consistency as C  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
import v8f_mini_eval as EV  # noqa: E402

torch.set_num_threads(1)
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    return {"root": root, **w}


def make_batch(world, n=8, seed=0, hmax=10, a_t=None, a_s=None, split="train"):
    J = D.load_join(Path(world["root"]) / f"join_{split}.npz")
    store = D.ClipStore(world["root"], split, lru=8)
    wins = D.train_windows(store, J, world["names"][split])
    ds = D.V8FMiniWindows(store, J, wins, hmax=hmax)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(wins), n, replace=False)
    items = []
    for i in idx:
        ci, t, _, _ = wins[i]
        items.append(ds.item(ci, t, 1 if a_t is None else a_t, 3 if a_s is None else a_s))
    return D.collate(items), J, ds


def tiny_model(J, **kw):
    cfg = M.V8FMiniConfig(**{**TINY, **kw})
    mu, sd = M.cons_stats(J)
    return M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd), cfg


# ------------------------------------------------------------------------------------------------ the P10 prior
def test_p10prior_equals_floors_kin_predict():
    rng = np.random.default_rng(0)
    net = fk._make_mlp(8, 60)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape) * 0.1)
    xm, xs, ym = rng.normal(size=8), rng.uniform(0.5, 2, 8), rng.normal(size=60)
    kf, base = rng.normal(size=(17, 8)), rng.normal(size=(17, 30, 2))
    ref = fk.predict_p10((net, (xm, xs, ym), (0, 1e-4, 60)), kf, base)
    pri = M.P10Prior.from_fit((net, (xm, xs, ym), (0, 1e-4, 60)))
    got = pri(torch.as_tensor(kf, dtype=torch.float32), torch.as_tensor(base, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(got, ref, atol=1e-4)
    assert not any(p.requires_grad for p in pri.parameters())


def test_upsample_prior_known_values():
    p30 = torch.arange(1, 31, dtype=torch.float32)[None, :, None].expand(1, 30, 2).clone()        # x = j at knot j
    u = M.upsample_prior(p30)
    assert u.shape == (1, 60, 2)
    np.testing.assert_allclose(u[0, :, 0].numpy(), 0.5 * np.arange(1, 61), atol=1e-6)            # tau 0.1 i -> 0.5 i
    assert torch.equal(u[:, 1::2], p30)                                                           # odd == the knots


def test_zero_init_residual_reproduces_p10_exactly(world):
    b, J, _ = make_batch(world)
    model, _ = tiny_model(J)
    assert model.built_levers()["residual_zero_init"]
    model.eval()
    with torch.no_grad():
        o = model.forward_infer(b)
        want = model.prior(b["p10f"][:, -1], b["base"])
    assert torch.equal(o["traj30"], want)                                   # BIT-exact at the 0.2 s knots
    assert torch.equal(o["p10"], want)
    np.testing.assert_allclose(o["traj60"][:, 0::2][:, 1:].numpy(), (0.5 * (want[:, :-1] + want[:, 1:])).numpy()[:, :], atol=1e-5)
    # CONTROL: a non-zero head moves the output
    with torch.no_grad():
        model.head_o[2].weight.add_(0.05)
        o2 = model.forward_infer(b)
    assert (o2["traj30"] - want).abs().max() > 1e-3


# ------------------------------------------------------------------------------------------------ cycles + sampling
def test_sample_and_hold_ages_select_the_tick_frames(world):
    b, J, _ = make_batch(world, a_t=1, a_s=4)
    model, _ = tiny_model(J)
    model.eval()
    seen = {}
    s0, t0, o0 = model.run_s, model.run_t, model.run_o
    model.run_s = lambda bb, ad, last, *a, **k: (seen.__setitem__("s", last.clone()), s0(bb, ad, last, *a, **k))[1]
    model.run_t = lambda bb, ad, last, *a, **k: (seen.__setitem__("t", last.clone()), t0(bb, ad, last, *a, **k))[1]
    model.run_o = lambda bb, ad, last, *a, **k: (seen.__setitem__("o", last.clone()), o0(bb, ad, last, *a, **k))[1]
    with torch.no_grad():
        model.forward_infer(b)
    F = b["feats"].shape[1]
    assert (seen["o"] == F - 1).all()
    assert (seen["t"] == F - 1 - b["a_t"]).all() and (seen["s"] == F - 1 - b["a_s"]).all()
    assert (b["a_s"] >= b["a_t"]).all()


def test_scheduled_sampling_picks_gt_or_predicted(world):
    b, J, _ = make_batch(world, n=16, seed=1)
    model, _ = tiny_model(J)
    model.train()
    cap = {}
    t0 = model.run_t
    model.run_t = lambda bb, ad, last, ck_t, ck_ok: (cap.__setitem__("ck", ck_t.detach().clone()), t0(bb, ad, last, ck_t, ck_ok))[1]
    ls = b["lab_s"]
    use = (ls["rc_ok"] & ls["has"][:, None])
    # p_gt = 1: T's checkpoint input is the GT RC (transformed S-tick -> T-tick) wherever it is valid
    out = model(b, p_gt=1.0)
    gt_t = M.transform_ckpt(ls["rc"], b["pose_s"], b["pose_t"])
    pred_t = M.transform_ckpt(out["ckpt"].detach(), b["pose_s"], b["pose_t"])
    assert torch.allclose(cap["ck"][use], gt_t[use], atol=1e-5) and use.any()
    # p_gt = 0: it is the predicted one everywhere
    model(b, p_gt=0.0)
    assert torch.allclose(cap["ck"], M.transform_ckpt(model(b, p_gt=0.0)["ckpt"].detach(), b["pose_s"], b["pose_t"]), atol=1e-5)
    assert M.p_gt_schedule(0, 100) == 1.0 and M.p_gt_schedule(100, 100) == pytest.approx(0.2)
    assert M.p_gt_schedule(50, 100) == pytest.approx(0.6) and M.p_gt_schedule(500, 100) == pytest.approx(0.2)


def test_transform_ckpt_known_value():
    ck = torch.tensor([[[10.0, 0.0, 0.0]]])
    src = torch.tensor([[0.0, 0.0, 0.0]])
    dst = torch.tensor([[2.0, 0.0, math.pi / 2]])                           # dst is 2 m ahead, facing left
    out = M.transform_ckpt(ck, src, dst)
    np.testing.assert_allclose(out[0, 0].numpy(), [0.0, -8.0, -math.pi / 2], atol=1e-6)
    assert torch.allclose(M.transform_ckpt(ck, src, src), ck)


def test_navonly_builds_no_scene_path_in_S_and_T(world):
    b, J, _ = make_batch(world)
    b2 = dict(b)
    b2["feats"] = (b["feats"].float() + torch.randn(b["feats"].shape, generator=torch.Generator().manual_seed(0))).half()   # LayerNorm hides a constant shift
    for navonly in (True, False):
        model, _ = tiny_model(J, navonly=navonly)
        model.eval()
        with torch.no_grad():
            o1, o2 = model.forward_infer(b), model.forward_infer(b2)
        same_s = torch.allclose(o1["ckpt"], o2["ckpt"])
        same_t = torch.allclose(o1["goal_logits"], o2["goal_logits"])
        assert (same_s and same_t) == navonly, f"navonly={navonly}: S/T invariance {same_s, same_t}"
        assert model.built_levers()["navonly"] == navonly
        with torch.no_grad():
            model.head_o[2].weight.add_(0.05)
            q1, q2 = model.forward_infer(b), model.forward_infer(b2)
        assert not torch.allclose(q1["traj30"], q2["traj30"])               # O always sees the scene


# ------------------------------------------------------------------------------------------------ admissibility
def test_no_situation_classifier_input(world):
    b, J, _ = make_batch(world)
    model, _ = tiny_model(J)
    r = M.assert_no_situation_classifier_input(model, b)
    assert set(r["read"]) <= M.INFERENCE_INPUT_KEYS and not any(k.startswith("lab") for k in r["read"])
    # inference must not need labels at all: delete every label / evaluator-side key and run
    stripped = {k: v for k, v in b.items() if k in M.INFERENCE_INPUT_KEYS}
    model.eval()
    with torch.no_grad():
        o = model.forward_infer(stripped)
    assert torch.isfinite(o["traj30"]).all()
    # CONTROL 1: a model that peeks at a label at inference is caught
    class Peeker(M.V8FMini):
        def _forward(self, b, p_gt, use_labels, gen=None):
            _ = b["lab"]["goal_y"]
            return super()._forward(b, p_gt, use_labels, gen)
    pk = Peeker(model.cfg, model.prior)
    with pytest.raises(AssertionError):
        M.assert_no_situation_classifier_input(pk, b)
    # CONTROL 2: the source scan sees a situation-classifier identifier
    assert M.scan_source_for_forbidden("def f(x):\n    sitclf_logits = x\n") == ["sitclf"]
    assert M.scan_source_for_forbidden("def f(x):\n    '''situation'''\n    return x\n") == []       # a docstring is not code


# ------------------------------------------------------------------------------------------------ penalties == rules
def _random_cases(N=400, seed=0):
    r = np.random.default_rng(seed)
    v = np.clip(r.uniform(0, 20, (N, 1)) + np.cumsum(r.normal(0, 0.6, (N, 30)), 1), 0.0, None)
    scale = np.where(r.random((N, 1)) < 0.4, 0.02, 1.0)                    # 40 %: nearly straight -> heading change ~ the 1 deg band
    yaw = np.cumsum(r.normal(0, 0.05, (N, 30)) + r.normal(0, 0.15, (N, 1)), 1) * 0.5 * scale
    x = np.cumsum(v * DT_ * np.cos(yaw), 1)
    y = np.cumsum(v * DT_ * np.sin(yaw), 1)
    plan = np.stack([x, y], -1)
    nan = lambda p: np.where(r.random(N) < p, np.nan, 1.0)
    t0 = r.uniform(-3, 7, N)
    c = {"limit_ms": r.choice([np.nan, 8.3, 13.9, 27.8, 33.3], N), "v_hi_ms": r.uniform(3, 25, N) * nan(0.2),
         "stop_active": r.random(N) < 0.35, "d_stop": r.uniform(0, 70, N) * nan(0.1), "t_stop": r.uniform(0, 8, N),
         "turn_side": r.choice([-1, 0, 0, 1], N).astype(np.int64), "turn_t_start": t0 * nan(0.05),
         "turn_t_end": (t0 + r.uniform(0.2, 6, N)) * nan(0.05), "lead_tg_min": r.uniform(0.5, 6, N) * nan(0.4)}
    gap = r.uniform(3, 80, (N, 1)) + np.cumsum(r.normal(0, 0.5, (N, 30)), 1)
    c["lead_gap_h"] = np.where(r.random((N, 30)) < 0.1, np.nan, gap)
    c["lead_gap_h"][r.random(N) < 0.25] = np.nan
    c["s_h"] = np.cumsum(np.clip(v + r.normal(0, 1, (N, 30)), 0, None) * DT_, 1)
    # BOUNDARY cases: put each constraint within +-margin of the plan's own value, so both sides of every threshold are hit
    V0 = C.violations(plan, c)
    kin = C.plan_kinematics(plan)
    m = r.random(N) < 0.6
    c["limit_ms"] = np.where(m, V0["SPEED"]["vmax"] - r.uniform(-0.2, 1.2, N), c["limit_ms"])      # cap +- the 0.5 tolerance
    m = (r.random(N) < 0.7) & c["stop_active"]
    c["d_stop"] = np.where(m, kin["s"][:, -1] - r.uniform(1.0, 3.5, N), c["d_stop"])                # passes by 1 .. 3.5 m
    c["t_stop"] = np.where(m, r.uniform(0, 6, N), c["t_stop"])
    m = (r.random(N) < 0.7) & np.isfinite(V0["LEAD"]["tg_min_plan"])
    c["lead_tg_min"] = np.where(m, V0["LEAD"]["tg_min_plan"] + r.uniform(-0.1, 0.5, N), c["lead_tg_min"])      # +- the 0.2 tolerance
    return plan, c


DT_ = 0.2


def test_torch_penalties_agree_with_the_numpy_rules():
    plan, c = _random_cases()
    V = C.violations(plan, c)
    ct = {k: torch.as_tensor(v) for k, v in c.items()}
    P = M.consistency_penalties(torch.as_tensor(plan), ct)
    n_viol = {}
    for r in C.RULE_NAMES:
        np.testing.assert_array_equal(P["active"][r].numpy(), V[r]["active"], err_msg=f"{r}: active differs")
        np.testing.assert_array_equal(P["viol"][r].numpy(), V[r]["violated"], err_msg=f"{r}: violated differs")
        assert (P["pen"][r].numpy() > 0).tolist() == V[r]["violated"].tolist()
        n_viol[r] = int(V[r]["violated"].sum())
    assert all(0 < n < 400 for n in n_viol.values()), f"degenerate random cases: {n_viol}"          # each rule both fires and holds


def test_penalty_is_differentiable_and_zero_when_consistent():
    plan = torch.tensor(np.stack([np.cumsum(np.full(30, 1.0), 0), np.zeros(30)], -1)[None], requires_grad=True)      # 5 m/s straight
    nan = torch.full((1,), float("nan"), dtype=torch.float64)
    c = {"limit_ms": torch.tensor([4.0], dtype=torch.float64), "v_hi_ms": nan.clone(), "stop_active": torch.zeros(1, dtype=torch.bool),
         "d_stop": nan.clone(), "t_stop": nan.clone(), "turn_side": torch.zeros(1, dtype=torch.long),
         "turn_t_start": nan.clone(), "turn_t_end": nan.clone(), "lead_tg_min": nan.clone(),
         "lead_gap_h": torch.full((1, 30), float("nan"), dtype=torch.float64), "s_h": torch.zeros(1, 30, dtype=torch.float64)}
    P = M.consistency_penalties(plan, c)
    assert float(P["pen"]["SPEED"][0]) == pytest.approx(0.5, abs=1e-6)             # 5 - 4 - tol 0.5 = 0.5 over
    P["pen"]["SPEED"].sum().backward()
    assert plan.grad.abs().sum() > 0
    c["limit_ms"] = torch.tensor([9.0], dtype=torch.float64)
    assert float(M.consistency_penalties(plan.detach(), c)["pen"]["SPEED"][0]) == 0.0


# ------------------------------------------------------------------------------------------------ projection
def test_project_speed_cap_and_kamm_known_values():
    N = 60
    straight = torch.stack([torch.arange(1, N + 1) * 1.0, torch.zeros(N)], -1)[None]            # 10 m/s at dt 0.1
    out = M.project_speed(straight, torch.tensor([6.0]), mu=0.7)
    sp = torch.diff(torch.cat([torch.zeros(1, 1, 2), out], 1), dim=1).norm(dim=-1) / 0.1
    np.testing.assert_allclose(sp[0].numpy(), 6.0, atol=1e-4)                                   # exactly the cap
    assert torch.allclose(out[0, :, 1], torch.zeros(N), atol=1e-6)                              # geometry preserved
    same = M.project_speed(straight, torch.tensor([20.0]), mu=0.7)
    assert torch.allclose(same, straight, atol=1e-4)                                            # cap above: identity
    # Kamm: R = 10 m at 15 m/s -> v_max = sqrt(0.7 g R) = 8.29 m/s
    th = torch.cumsum(torch.full((N,), 15.0 * 0.1 / 10.0), 0)
    circ = torch.stack([10 * torch.sin(th), 10 * (1 - torch.cos(th))], -1)[None]
    out = M.project_speed(circ, None, mu=0.7)
    sp = torch.diff(torch.cat([torch.zeros(1, 1, 2), out], 1), dim=1).norm(dim=-1)[0] / 0.1
    assert sp[2:].max() <= math.sqrt(0.7 * 9.81 * 10.0) * 1.08 and sp[2:].max() < 15.0
    on_circle = (out[0] - torch.tensor([0.0, 10.0])).norm(dim=-1)
    assert (on_circle - 10.0).abs().max() < 0.05                                                # still on the circle
    # CONTROL: with Kamm effectively off the speed stays 15
    out2 = M.project_speed(circ, None, mu=1e6)
    sp2 = torch.diff(torch.cat([torch.zeros(1, 1, 2), out2], 1), dim=1).norm(dim=-1)[0] / 0.1
    assert sp2[2:].mean() > 14.0


# ------------------------------------------------------------------------------------------------ the data join
def _fake_release(spec):
    """spec: {clip_id: (k0, n_rows)}; row r of a clip holds k = k0 + r, with values that encode (clip, k)."""
    rows = {}
    clips = {}
    cols = {k: [] for k in ["k", "now_s", "nav_token", "nav_args_valid", "nav_d_next_m", "goal_y", "goal_w", "lat_cls_a",
                            "lat_allowed_a", "lon_cls", "lon_allowed", "lead_valid"] + list(D.CONS_FIELDS)}
    for v in D.RC_VARIANTS:
        for c in ("x", "y", "psi", "valid"):
            cols[f"rc_{v}_{c}"] = []
    r0 = 0
    for ci, (cid, (k0, n)) in enumerate(spec.items()):
        k = np.arange(n) + k0
        clips[D.stable_episode_id(cid)] = (r0, n, k0)
        cols["k"].append(k.astype(np.int16)); cols["now_s"].append(0.1 * k + 0.01)
        cols["nav_token"].append((k % 3).astype(float)); cols["nav_args_valid"].append((k % 2 == 0).astype(float))
        cols["nav_d_next_m"].append(np.where(k % 4 == 0, np.nan, 10.0 * (k % 5)))
        cols["goal_y"].append(np.full(n, 0b110, np.uint32)); cols["goal_w"].append(np.full(n, 0b111, np.uint32))
        cols["lat_cls_a"].append((k % 5).astype(np.int16)); cols["lat_allowed_a"].append((1 << (k % 5)).astype(np.uint8))
        cols["lon_cls"].append((k % 7).astype(np.int16)); cols["lon_allowed"].append((1 << (k % 7)).astype(np.uint8))
        cols["lead_valid"].append(np.zeros(n, np.int8))
        for f in D.CONS_FIELDS:
            cols[f].append(k.astype(float) if f == "v_hi_ms" else np.full(n, np.nan))
        for v in D.RC_VARIANTS:
            cols[f"rc_{v}_x"].append(float(v[1:]) * np.ones(n)); cols[f"rc_{v}_y"].append(0.1 * k)
            cols[f"rc_{v}_psi"].append(np.full(n, 90.0)); cols[f"rc_{v}_valid"].append((k % 6 != 0).astype(float))
        r0 += n
    from tanitad.data.v9_labels import V9Release
    ro = lambda a: (lambda x: (x.setflags(write=False), x)[1])(np.concatenate(a))
    return V9Release("mem", "0", "t", "{}", MappingProxyType({k: ro(v) for k, v in cols.items()}), MappingProxyType(clips))


def test_join_maps_cache_step_j_to_v9_frame_2j():
    rel = _fake_release({"clipA": (2, 30), "clipB": (2, 10)})                 # A: k = 2..31 ; B: k = 2..11
    speed = {"clipA": (50, np.array([0, 1, 0, 0], np.float32), 1)}
    J = D.build_label_join([("clipA", 20), ("clipB", 20), ("clipC", 5)], rel, speed=speed)
    a = slice(int(J["off"][0]), int(J["off"][0]) + 20)
    j = np.arange(20)
    inside = (2 * j >= 2) & (2 * j <= 31)
    np.testing.assert_array_equal(J["has"][a], inside)                         # j = 1..15 only
    np.testing.assert_allclose(J["cons"][a][inside, D.CONS_IDX["v_hi_ms"]], 2.0 * j[inside])        # row k = 2 j
    assert np.isnan(J["cons"][a][~inside, D.CONS_IDX["v_hi_ms"]]).all()
    np.testing.assert_allclose(J["now_s"][a][inside], 0.1 * 2 * j[inside] + 0.01)
    # nav: one-hot of (k % 3); distance only where args are valid and finite
    k = 2 * j[inside]
    np.testing.assert_array_equal(J["nav"][a][inside][:, :3].argmax(1), k % 3)
    ok = (k % 2 == 0) & (k % 4 != 0)
    np.testing.assert_array_equal(J["nav"][a][inside][:, 4] == 1, ok)
    np.testing.assert_allclose(J["nav"][a][inside][ok, 3], np.clip(10.0 * (k[ok] % 5) / 100.0, 0, 3))
    # RC: degrees -> radians, invalid -> zeros + not ok
    assert np.allclose(J["rc"][a][inside][J["rc_ok"][a][inside][:, 0]][:, 0, 2], math.pi / 2)
    bad = inside & (2 * j % 6 == 0)
    assert not J["rc_ok"][a][bad].any() and (J["rc"][a][bad] == 0).all()
    # clip B only has k = 2..11 -> cache steps 1..5 ; clip C is absent from the release
    b = slice(int(J["off"][1]), int(J["off"][1]) + 20)
    assert J["has"][b].sum() == 5
    assert J["n_clips_missing_labels"] == 1 and not J["has"][int(J["off"][2]):].any()
    # goal bits survive, speed sidecar one-hot lands on the right clip
    assert (J["goal_y"][a][inside] == 0b110).all() and J["limit_kmh"][0] == 50 and J["limit"][0].tolist() == [0, 1, 0, 0]
    assert J["limit_ok"].tolist() == [True, False, False]
    with pytest.raises(ValueError):
        D.build_label_join([("clipC", 5)], rel, require_labels=True)


def test_disjoint_and_nested_fractions():
    assert D.assert_disjoint(["a", "b"], ["c"])["overlap"] == 0
    with pytest.raises(AssertionError):
        D.assert_disjoint(["a", "b"], ["b", "c"])
    clips = [f"clip{i:03d}" for i in range(100)]
    s25, s50, s100 = (D.select_train_clips(clips, f) for f in (0.25, 0.5, 1.0))
    assert (len(s25), len(s50), len(s100)) == (25, 50, 100) and set(s25) < set(s50) < set(s100)


def test_eval_and_train_cache_clips_are_disjoint_in_the_synthetic_world(world):
    tr = [c for c, _ in D.read_manifest(world["root"], "train")]
    ev = [c for c, _ in D.read_manifest(world["root"], "eval")]
    assert D.assert_disjoint(tr, ev)["n_train"] == 6


def test_window_item_matches_an_independent_recomputation(world):
    root = Path(world["root"])
    J = D.load_join(root / "join_train.npz")
    store = D.ClipStore(root, "train", lru=4)
    ds = D.V8FMiniWindows(store, J, [], hmax=10)
    cid = world["names"]["train"][2]
    d = store.get(cid)
    pose = d["pose"]
    t, a_t, a_s = 17, 1, 3
    it = ds.item(2, t, a_t, a_s)
    # GT: future positions in the ego frame at t (independent numpy)
    f = pose[t + 1:t + 31, :2] - pose[t, :2]
    c, s = np.cos(-pose[t, 2]), np.sin(-pose[t, 2])
    np.testing.assert_allclose(it["gt"], np.stack([f[:, 0] * c - f[:, 1] * s, f[:, 0] * s + f[:, 1] * c], 1), atol=1e-4)
    # kin3 = (v_t, (v_t - v_{t-1}) / 0.2, steer_t)
    np.testing.assert_allclose(it["kin3"], [pose[t, 3], (pose[t, 3] - pose[t - 1, 3]) / 0.2, d["steer"][t]], atol=1e-4)
    # the base is kd_x in the CORRECT unit
    kin = d["kin3"][t:t + 1]
    want = fk.kd_x(fk.ha0_paths(kin[:, 0], 0.2, 30), fk.ha0_ext_paths(kin, 0.2, 30, action_units="steer"))
    np.testing.assert_allclose(it["base"], want[0], atol=1e-5)
    legacy = fk.kd_x(fk.ha0_paths(kin[:, 0], 0.2, 30), fk.ha0_ext_paths(kin, 0.2, 30, action_units="kappa"))
    assert np.abs(want - legacy).max() > 1e-3 or abs(kin[0, 2]) < 1e-4            # the unit matters (control)
    # frames: the LAST frame is NOW; label packs sit at the tick frames
    assert it["feats"].shape[0] == 10 and it["frame_ok"].all()
    np.testing.assert_allclose(it["pose_now"], pose[t, :3], atol=1e-5)
    np.testing.assert_allclose(it["pose_t"], pose[t - a_t, :3], atol=1e-5)
    np.testing.assert_allclose(it["pose_s"], pose[t - a_s, :3], atol=1e-5)
    o = int(J["off"][2])
    np.testing.assert_allclose(it["lab_t"]["cons"], J["cons"][o + t - a_t], equal_nan=True)
    np.testing.assert_allclose(it["lab_s"]["rc"], J["rc"][o + t - a_s])
    # ages are clamped so that a tick frame always has `hist` real frames behind it
    early = ds.item(2, 3, 2, 6)
    assert early["a_t"] == 0 and early["a_s"] == 0


def test_sampler_is_seeded_and_ages_are_nested(world):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    store = D.ClipStore(world["root"], "train", lru=4)
    wins = D.train_windows(store, J, world["names"]["train"])
    S = D.ChunkedSampler(wins, 16, seed=3, chunk=2)
    e0a, e0b, e1 = list(S.epoch(0)), list(S.epoch(0)), list(S.epoch(1))
    assert e0a == e0b and e0a != e1
    flat = [w for b in e0a for w in b]
    assert sorted((w[0], w[1]) for w in flat) == sorted((w[0], w[1]) for w in wins)          # every window exactly once
    assert all(0 <= w[2] <= 2 and w[2] <= w[3] <= w[2] + 4 for w in flat)
    assert max(w[3] for w in flat) > 2 and len({w[2] for w in flat}) == 3


@pytest.mark.skipif(not os.path.exists("C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_eval139.npz"),
                    reason="local v9 eval139 release not present")
def test_five_hand_checked_real_windows():
    """Pinned from raw/join_check.json (the release rows, re-derived by build_v9_labels.rc_fields from the ego log, and
    cross-read against the v2ep poses): clock exact, RC bit-identical to the builder, speed matches the poses."""
    from tanitad.data.v9_labels import load_v9_release
    rel = load_v9_release("C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_eval139.npz",
                          expect_md5="6b5c7f207cffc3b7eb3cd527fd433599")
    pins = [("10804db8", 35, 7.081525, (29.9699, 0.3773, -0.748), (49.9225, -0.2784, 0.1574), 4.9823),
            ("480d7910", 47, 9.627721, (30.0002, -0.0252, -0.0543), (50.0002, -0.052, -0.1078), 31.4377),
            ("91929cea", 59, 11.994035, (29.9743, -1.118, -3.4838), (49.9086, -2.6974, -5.4043), 7.6617),
            ("b99058dc", 23, 4.808529, (29.9978, -0.3473, -0.9017), (49.995, -0.5664, 0.1245), 8.17),
            ("f1a94fee", 39, 7.977197, (29.8471, 2.7474, 7.1904), (49.7253, 4.8528, 4.6743), 3.4963)]
    grid = json.load(open(PKG / "raw" / "eval_grid_windows.json"))
    by_prefix = {c[:8]: c for c in grid["order"]}
    for pre, t, now, a30, a50, vnow in pins:
        sid = D.stable_episode_id(by_prefix[pre])
        r0, n, k0 = rel.clips[sid]
        row = r0 + 2 * t - k0
        R = rel.rows
        assert float(R["now_s"][row]) == pytest.approx(now, abs=1e-6)
        assert [float(R[f"rc_A30_{c}"][row]) for c in ("x", "y", "psi")] == pytest.approx(list(a30), abs=1e-4)
        assert [float(R[f"rc_A50_{c}"][row]) for c in ("x", "y", "psi")] == pytest.approx(list(a50), abs=1e-4)
        assert float(R["v_now_ms"][row]) == pytest.approx(vnow, abs=1e-4)
        assert int(R["k"][row]) == 2 * t


# ------------------------------------------------------------------------------------------------ the launch gate
def _args(extra=()):
    return TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", *extra])


def _built(a, J, **over):
    cfg = TR.make_config(a)
    keep = {k: v for k, v in TINY.items() if k not in ("nb", "n_q", "d", "ff")}
    cfg = M.V8FMiniConfig(**{**cfg.__dict__, **keep, "d": 32, "ff": 64, **over})
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    info = {"n_clips": 6, "n_clips_expected": 6, "batch": a.batch, "md5_overridden": True}
    return cfg, model, opt, info


def test_every_flag_has_a_declared_vs_built_entry():
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert dests <= set(TR.FLAG_CHECKS), sorted(dests - set(TR.FLAG_CHECKS))
    assert set(TR.FLAG_CHECKS) <= dests, sorted(set(TR.FLAG_CHECKS) - dests)


def test_declared_vs_built_passes_for_every_arm_and_refuses_lies(world):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    for arm in TR.ARM_SPECS:
        a = _args(["--arm", arm])
        cfg, model, opt, info = _built(a, J)
        assert TR.declared_vs_built(a, cfg, model, opt, info) == [], arm
    a = _args(["--arm", "M1-navonly"])
    cfg, model, opt, info = _built(a, J, navonly=False)                      # built a scene-reading S/T but declared navonly
    assert any("navonly" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    a = _args(["--arm", "M1-nocons"])
    cfg, model, opt, info = _built(a, J, consistency=True)
    assert any("consistency" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    a = _args(["--arm", "M2a"])
    cfg, model, opt, info = _built(a, J, wm="off")
    assert any("wm" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    # a lever that is declared and does nothing: penalties ON with weight 0
    a = _args(["--arm", "M1", "--w-pen", "0"])
    cfg, model, opt, info = _built(a, J)
    assert any("does nothing" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))
    # a data fraction that was not applied
    a = _args(["--arm", "M1", "--data-frac", "0.5"])
    cfg, model, opt, info = _built(a, J)
    info["n_clips"], info["n_clips_expected"] = 6, 3
    assert any("clips built" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))


def test_a_flag_without_an_entry_is_refused(world, monkeypatch):
    J = D.load_join(Path(world["root"]) / "join_train.npz")
    a = _args()
    cfg, model, opt, info = _built(a, J)
    orig = TR.build_parser

    def with_new_flag():
        p = orig()
        p.add_argument("--brand-new-lever", type=float, default=0.0)
        return p
    monkeypatch.setattr(TR, "build_parser", with_new_flag)
    assert any("brand_new_lever" in m for m in TR.declared_vs_built(a, cfg, model, opt, info))


@pytest.mark.parametrize("arm", ["M1", "M1-navonly", "M1-nocons", "M2a", "M2b"])
def test_live_check_every_term_finite_every_group_gets_gradient_control_is_zero(world, arm):
    a = _args(["--arm", arm])
    lv = TR.declared_levers(a)
    hmax = 16 if lv["wm"] != "off" else 10
    b, J, _ = make_batch(world, n=8, hmax=hmax)
    cfg = M.V8FMiniConfig(**{**TINY, "hmax": hmax, "navonly": lv["navonly"], "consistency": lv["consistency"], "wm": lv["wm"]})
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    res = TR.live_check(model, b, cfg)
    assert res["pass"] and res["control_prior_grad_sum"] == 0.0 and res["head_o_grad_step0"] > 0.0
    assert not res["missing_terms"] and not res["unexpected_dead"]
    if arm == "M2a":                                                          # without IDM: the IDM heads are DECLARED dead
        assert res["declared_dead"] and not res["declared_dead_but_alive"]
    if arm == "M2b":
        assert not res["declared_dead"] and res["groups_grad"]["wm"] > 0
    if arm == "M1-nocons":
        assert not any(k.startswith("pen") for k in res["terms"])


def test_live_check_goes_red_on_a_deliberately_broken_model(world):
    b, J, _ = make_batch(world, n=8)
    cfg = M.V8FMiniConfig(**TINY)
    mu, sd = M.cons_stats(J)

    class NoOGradient(M.V8FMini):                                             # the O head is never reached by any loss
        def run_o(self, *a, **k):
            traj, p30 = super().run_o(*a, **k)
            return traj.detach(), p30
    broken = NoOGradient(cfg, M.P10Prior.synthetic(0), mu, sd)
    with pytest.raises(AssertionError, match="G-LIVE"):
        TR.live_check(broken, b, cfg)


# ------------------------------------------------------------------------------------------------ end to end
def test_end_to_end_train_resume_eval(world, tmp_path, monkeypatch):
    common = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--steps", "8", "--log-every", "4", "--warmup", "2",
              "--ckpt-every", "4", "--threads", "1", "--seed", "5"]
    full = tmp_path / "full"
    assert TR.main(["--out", str(full), *common]) == 0
    s = json.load(open(full / "summary.json"))
    assert s["done"] is True and s["steps"] == 8
    sc = json.load(open(full / "selfcheck.json"))
    assert sc["declared_vs_built"] == "PASS" and sc["live"]["pass"] and sc["live"]["control_prior_grad_sum"] == 0.0
    # crash at step 6, resume from the step-4 checkpoint: the SAME run
    res = tmp_path / "res"
    monkeypatch.setenv("V8F_TRAIN_CRASH_AT", "6")
    with pytest.raises(RuntimeError, match="V8F_TRAIN_CRASH_AT"):
        TR.main(["--out", str(res), *common])
    assert not (res / "summary.json").exists()                                # no done-marker on an unfinished run
    monkeypatch.delenv("V8F_TRAIN_CRASH_AT")
    assert TR.main(["--out", str(res), *common]) == 0
    assert json.load(open(res / "summary.json"))["done"] is True
    A = torch.load(full / "ckpt_last.pt", weights_only=False)["model"]
    B = torch.load(res / "ckpt_last.pt", weights_only=False)["model"]
    for k in A:
        assert torch.allclose(A[k].float(), B[k].float(), atol=1e-5), f"resume diverged at {k}"
    # a checkpoint from a different argv is refused
    with pytest.raises(SystemExit, match="RESUME-REFUSED"):
        TR.main(["--out", str(full), *common[:-2], "--seed", "6"])
    # the eval path on the synthetic eval world
    root = Path(world["root"])
    out = tmp_path / "eval"
    rc = EV.main(["run", "--ckpt", str(full / "ckpt_last.pt"), "--tag", "t", "--eval-cache", str(root),
                  "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-kin", str(root / "eval_kin.npz"),
                  "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(out), "--n-boot", "50", "--batch", "16",
                  "--lru", "4", "--project"])
    assert rc == 0
    r = json.load(open(out / "eval_t.json"))["raw"]
    z = np.load(out / "eval_t.npz")
    assert np.isfinite(z["pred"]).all() and z["pred"].shape[1:] == (30, 2)
    assert r["traj"]["ade6_m"]["mean"] > 0 and 0.0 <= r["consistency_a"]["any"]["rate_all"] <= 1.0
    assert "self_consistency_b" in r and "strategic" in r and "tactical" in r
    assert (out / "eval_t_proj.npz").exists()
    # compare: two "seeds" of one arm + a fake floors file built from the same windows (smoke of the PREREG bar plumbing)
    grid = json.load(open(root / "eval_grid_windows.json"))
    fake = {"P10_refit__ade6": z["ade6"] * 1.3, "P10_refit__viol_any": z["viol_any"], "eid": z["eid"],
            "nav_baseline_ckpt": np.zeros((len(z["eid"]), 3, 3))}
    np.savez(out / "floors.npz", **fake)
    cmp_out = out / "compare.json"
    assert EV.main(["compare", "--runs", f"M1_s0={out / 'eval_t.npz'}", f"M1_s1={out / 'eval_t.npz'}",
                    "--floors-npz", str(out / "floors.npz"), "--out", str(cmp_out), "--n-boot", "50",
                    "--eval-grid", str(root / "eval_grid_windows.json"), "--eval-kin", str(root / "eval_kin.npz"),
                    "--eval-join", str(root / "join_eval.npz")]) == 0
    cj = json.load(open(cmp_out))
    assert cj["arms"]["M1"]["seed_floor_F_m"] == 0.0 and "tactical_lat" in cj and "strat" in cj["arms"]["M1"]
