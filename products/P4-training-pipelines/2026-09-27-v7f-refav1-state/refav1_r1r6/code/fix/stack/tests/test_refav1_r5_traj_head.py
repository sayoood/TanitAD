"""R5 (PI 2026-09-27): ONE combined tactical+operative trajectory to 6 s, as a RESIDUAL over a
model-free kinematic prior (`kdx` default; `damp50` / `cv` selectable), read from 4 x 10 REGION
pooling + t0 kinematics + nav + max speed + the tactical decision (R4).

The three proofs the brief asks for, each with its control:
  (a) KNOWN VALUE -- with the zero-initialised output layer the predicted trajectory IS the
      chosen prior, bit for bit, for EVERY prior; and the prior IS the banked reference
      (`build_explore.py::retime`, copied verbatim below) on synthetic edge cases. The
      real-data version (2,399 eval windows, max |delta| 0.0) is banked in the package's
      `raw/checks/kdx_reference_check.*`.
  (b) GRADIENT -- at step 0 only the output layer is reachable (the zero-init property,
      asserted as a known value: body gradient EXACTLY 0); after one optimizer step the traj
      loss ALONE reaches the region projection, the MLP and the ADAPTER through the pooling.
  (c) COUNTERFACTUAL -- changing the tactical decision input changes NOTHING at init (exactly)
      and CHANGES the prediction after a few training steps.
Plus: region pooling pinned to the probe's layout, the loss NaN-safe under masks, the R1 cap
the identity when it does not bind, and the loader's GT / (a0, kappa0) equal to the arm's.
"""
import copy

import numpy as np
import pytest
import torch

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import RefAV1, RefAV1Config
from tanitad.refs.refav1_traj import (TRAJ_PRIORS, cap_path_speed, kinematic_floor_paths,
                                      kinematic_prior, path_speeds, region_pool, retime_paths,
                                      traj_loss)


# ---- VERBATIM from `.../2026-09-27-refav1-fullgrid-loncomb3/raw/analysis/explore/build_explore.py`
def _arclen(p):
    P = np.concatenate([np.zeros((1, 2), p.dtype), p], 0)
    seg = np.diff(P, axis=0)
    L = np.linalg.norm(seg, axis=-1)
    return P, seg, L, np.concatenate([[0.0], np.cumsum(L)])


def _retime_ref(path, lon_src):
    P, seg, L, S = _arclen(path.astype(np.float64))
    s_src = _arclen(lon_src.astype(np.float64))[3][1:]
    out = np.zeros((len(s_src), 2))
    nz = np.nonzero(L > 1e-9)[0]
    for i, s in enumerate(s_src):
        if S[-1] < 1e-6 or len(nz) == 0:
            out[i] = (s, 0.0); continue
        if s <= S[-1]:
            j = int(np.clip(np.searchsorted(S, s, side="right") - 1, 0, len(L) - 1))
            while L[j] <= 1e-9 and j + 1 < len(L):
                j += 1
            out[i] = P[j] + (s - S[j]) / max(L[j], 1e-12) * seg[j] if L[j] > 1e-9 else P[j]
        else:
            jj = nz[-1]
            out[i] = P[-1] + (s - S[-1]) * seg[jj] / L[jj]
    return out.astype(np.float32)
# -------------------------------------------------------------------------------------------


def _cfg(**kw) -> RefAV1Config:
    base = dict(d_enc=16, d_state=16, n_tokens=8, op_layers=1, op_heads=2, op_window=2,
                tac_queries=4, tac_layers=1, str_dim=8, str_layers=1, w_traj=1.0,
                traj_grid=(2, 4), traj_regions=(1, 2), traj_d_region=8, traj_hidden=32,
                strategic_cfg=StrategicPolicyConfig(d_model=16, depth=1, n_heads=2, d_ctx=8,
                                                    d_cmd=8),
                tactical_cfg=TacticalPolicyConfig(d_model=16, depth=1, n_heads=2, d_intent=8))
    base.update(kw)
    return RefAV1Config(**base)


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    m = RefAV1(_cfg(**kw))
    with torch.no_grad():
        m.std.fit(torch.randn(64, 16, generator=torch.Generator().manual_seed(9)))
    return m


#: the kinematic edge cases the reference's branches exist for
_V0 = torch.tensor([0.0, 0.0, 12.0, 25.0, 8.0, 3.0, 15.0, 0.2])
_A0 = torch.tensor([0.0, 1.5, -2.0, 0.8, -4.0, 0.0, 3.0, -1.0])
_K0 = torch.tensor([0.0, 0.3, 0.01, -0.02, 0.5, -0.4, 0.0, 0.2])


def _batch(m, seed=4, bsz=8):
    c = m.cfg
    g = torch.Generator().manual_seed(seed)
    return dict(feats=torch.randn(bsz, c.op_window, c.n_tokens, c.d_enc, generator=g),
                actions=torch.randn(bsz, c.op_steps, c.a_dim, generator=g) * 0.1,
                future_feats=torch.randn(bsz, c.op_steps, c.n_tokens, c.d_enc, generator=g),
                v0=_V0[:bsz].clone(), a0=_A0[:bsz].clone(), kappa0=_K0[:bsz].clone(),
                traj_gt=torch.cumsum(torch.rand(bsz, 30, 2, generator=g) * torch.tensor(
                    [2.0, 0.3]), dim=1),
                traj_mask=torch.ones(bsz, 30, dtype=torch.bool),
                nav_cmd=torch.arange(bsz) % 3)


# =========================================================================== (a) KNOWN VALUE
@pytest.mark.parametrize("steps", [10, 30])
def test_a1_the_batched_prior_is_the_banked_reference_bit_for_bit(steps):
    h0, hx = kinematic_floor_paths(_V0, _A0, _K0, steps, 0.2)
    damp = (0.5 * h0 + 0.5 * hx).numpy()
    ref = np.stack([_retime_ref(damp[i], hx.numpy()[i]) for i in range(len(_V0))])
    assert np.array_equal(retime_paths(torch.from_numpy(damp), hx).numpy(), ref)
    assert np.array_equal(kinematic_prior("kdx", _V0, _A0, _K0, steps=steps).numpy(), ref)
    assert np.array_equal(kinematic_prior("damp50", _V0, _A0, _K0, steps=steps).numpy(), damp)
    assert np.array_equal(kinematic_prior("cv", _V0, _A0, _K0, steps=steps).numpy(), h0.numpy())
    # control: the three priors are different objects on these cases
    assert not np.array_equal(ref, damp) and not np.array_equal(damp, h0.numpy())


@pytest.mark.parametrize("prior", TRAJ_PRIORS)
def test_a2_zero_init_head_output_IS_the_chosen_prior_exactly(prior):
    m = _model(traj_prior=prior)
    b = _batch(m)
    f, a, fut = b.pop("feats"), b.pop("actions"), b.pop("future_feats")
    out = m(f, a, future_feats=fut, **b)
    want = kinematic_prior(prior, b["v0"], b["a0"], b["kappa0"], steps=30, dt=0.2)
    assert torch.equal(out["traj_residual"], torch.zeros_like(want))
    assert torch.equal(out["traj_pred"], want) and torch.equal(out["traj_prior"], want)
    assert out["traj_ade_m"] == out["traj_prior_ade_m"]      # the in-log control
    # control: one optimizer step and the prediction LEAVES the prior
    out["loss"].backward()
    torch.optim.SGD(m.parameters(), lr=0.05).step()
    out2 = m(f, a, future_feats=fut, **b)
    assert not torch.equal(out2["traj_pred"], want)


def test_a3_the_prior_reads_nothing_after_t0():
    """Only (v0, a0, kappa0) enter -- the prior of two windows with the SAME t0 state and
    DIFFERENT futures is identical (and `cv` ignores a0 / kappa0 entirely)."""
    p1 = kinematic_prior("kdx", _V0, _A0, _K0)
    p2 = kinematic_prior("kdx", _V0.clone(), _A0.clone(), _K0.clone())
    assert torch.equal(p1, p2)
    assert torch.equal(kinematic_prior("cv", _V0, _A0, _K0),
                       kinematic_prior("cv", _V0, _A0 * 0 + 3.0, _K0 * 0 - 0.1))
    assert not torch.equal(kinematic_prior("kdx", _V0, _A0 + 0.5, _K0), p1)   # control


# =========================================================================== (b) GRADIENT
def _traj_grads(m, b):
    f, a, fut = b["feats"], b["actions"], b["future_feats"]
    kw = {k: v for k, v in b.items() if k not in ("feats", "actions", "future_feats")}
    m.zero_grad(set_to_none=True)
    out = m(f, a, future_feats=fut, **kw)
    out["loss_traj"].backward()                 # the traj term ALONE
    g = lambda mod: sum(float(p.grad.abs().sum()) for p in mod.parameters()
                        if p.grad is not None)
    return {"out": g(m.traj_head.out), "reg_proj": g(m.traj_head.reg_proj),
            "mlp": g(m.traj_head.mlp), "reg_pos": float(m.traj_head.reg_pos.grad.abs().sum()),
            "adapter": g(m.adapter), "tac_policy": g(m.tactical_policy)}


def test_b_gradient_reaches_the_head_the_region_pooling_and_the_adapter():
    m = _model()
    b = _batch(m)
    g0 = _traj_grads(m, b)
    assert g0["out"] > 0
    # the zero-init property, as a KNOWN VALUE: nothing below `out` gets gradient at step 0
    assert g0["reg_proj"] == 0.0 and g0["mlp"] == 0.0 and g0["adapter"] == 0.0
    torch.optim.SGD(m.parameters(), lr=0.05).step()
    g1 = _traj_grads(m, b)
    for k in ("out", "reg_proj", "mlp", "reg_pos", "adapter"):
        assert g1[k] > 0.0, (k, g1)
    # the decision is DETACHED (R4 conditions, it is not rewritten by imitation)
    assert g1["tac_policy"] == 0.0


def test_b2_region_pooling_carries_spatial_information_the_global_mean_would_not():
    """Two fields with the SAME global mean -- a bright column on the LEFT vs on the RIGHT of
    a row-major 2 x 4 grid -- pool to DIFFERENT regions (left = cols 0-1, right = cols 2-3)."""
    x = torch.zeros(1, 8, 4)
    x[0, [0, 4]] = 1.0                         # column 0 of both rows -> left region
    y = torch.zeros(1, 8, 4)
    y[0, [2, 6]] = 1.0                         # column 2 of both rows -> right region
    assert torch.equal(x.mean(1), y.mean(1))
    rx, ry = region_pool(x, (2, 4), (1, 2)), region_pool(y, (2, 4), (1, 2))
    assert torch.equal(rx[0, :, 0], torch.tensor([0.5, 0.0]))
    assert torch.equal(ry[0, :, 0], torch.tensor([0.0, 0.5]))


# =========================================================================== (c) COUNTERFACTUAL
def test_c_changing_the_tactical_decision_changes_the_trajectory_only_once_trained():
    m = _model()
    b = _batch(m, bsz=6)
    field = m.encode(b["feats"])[:, -1]
    n_dec = m.n_lat + m.n_lon
    turn = torch.zeros(6, n_dec)
    keep = torch.zeros(6, n_dec)
    keep[:, 0] = 1.0                           # LANE_KEEP
    turn[:, 6] = 1.0                           # TURN_L (v7.0 lat index 6)
    keep[:, m.n_lat + 1] = turn[:, m.n_lat + 1] = 1.0      # lon CRUISE in both

    def pred(tac):
        with torch.no_grad():
            return m.traj_readout(field, b["v0"], b["a0"], b["kappa0"],
                                  nav_cmd=b["nav_cmd"], tac=tac)["traj_pred"]
    assert torch.equal(pred(turn), pred(keep))           # init: exactly the prior either way
    opt = torch.optim.Adam(m.traj_head.parameters(), lr=1e-2)
    kw = {k: v for k, v in b.items() if k not in ("feats", "actions", "future_feats")}
    for _ in range(3):
        opt.zero_grad()
        out = m(b["feats"], b["actions"], future_feats=b["future_feats"], **kw)
        out["loss_traj"].backward()
        opt.step()
    d = (pred(turn) - pred(keep)).abs().max()
    assert d > 1e-4, float(d)


# =========================================================================== the pieces
def test_d_region_pooling_is_the_probes_layout_at_the_real_geometry():
    ramp = torch.arange(640, dtype=torch.float32).reshape(1, 640, 1)
    r = region_pool(ramp, (16, 40), (4, 10))
    # region (a=1, e=7): rows 4..7, cols 28..31 of the row-major 16 x 40 grid
    assert float(r[0, 1 * 10 + 7, 0]) == 249.5
    probe = ramp[0].reshape(4, 4, 10, 4, -1).mean(dim=(1, 3)).reshape(40, -1)
    assert torch.equal(r[0], probe)
    assert torch.allclose(r.mean(1), ramp.mean(1))
    with pytest.raises(ValueError, match="grid"):
        region_pool(torch.zeros(1, 639, 2), (16, 40), (4, 10))


def test_e_the_loss_is_nan_safe_under_masks_and_a_real_zero_when_nothing_is_valid():
    pred = torch.zeros(2, 30, 2, requires_grad=True)
    gt = torch.full((2, 30, 2), float("nan"))
    gt[0, :10] = 1.0
    m = torch.zeros(2, 30, dtype=torch.bool)
    m[0, :10] = True
    loss, n = traj_loss(pred, gt, m)
    loss.backward()
    assert n == 10 and torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert pred.grad[1].abs().sum() == 0 and pred.grad[0, :10].abs().sum() > 0
    loss0, n0 = traj_loss(pred, gt, torch.zeros_like(m))
    assert n0 == 0 and float(loss0.detach()) == 0.0 and loss0.requires_grad


def test_f_the_R1_cap_is_the_identity_when_it_does_not_bind_and_holds_when_it_does():
    p = kinematic_prior("kdx", _V0, _A0, _K0)
    free, rec = cap_path_speed(p, torch.full((8,), 60.0), 0.2, v0=_V0)
    assert torch.equal(free, p) and rec["n_capped_steps"] == 0
    capped, rec = cap_path_speed(p, torch.full((8,), 10.0), 0.2, v0=_V0)
    sp = path_speeds(capped, 0.2)
    within = _V0 <= 10.0
    assert float(sp[within].max()) <= 10.0 + 1e-4 and rec["n_rows_over_limit"] == 0
    assert rec["n_rows_v0_over_limit"] == int((~within).sum())
    # a row starting above the limit sheds speed no faster than a_max (the planner's rule)
    allow = torch.clamp(_V0[~within, None] - 4.0 * torch.arange(30) * 0.2, min=10.0)
    assert bool((sp[~within] <= allow + 1e-3).all())
    # the capped path never gets AHEAD of the original along its own geometry
    assert bool((torch.cumsum(sp, 1) <= torch.cumsum(path_speeds(p, 0.2), 1) + 1e-3).all())


def test_g_trajectory_inference_applies_the_cap_and_reports_it():
    m = _model(vmax_input=True).eval()
    b = _batch(m, bsz=4)
    r = m.trajectory(b["feats"], v0=b["v0"], a0=b["a0"], kappa0=b["kappa0"],
                     nav_cmd=b["nav_cmd"], speed_max_ms=torch.tensor([5.0, 5.0, 30.0, 5.0]))
    assert r["vmax_cap"] is not None and r["vmax_cap"]["n_rows_over_limit"] == 0
    assert r["traj"].shape == (4, 30, 2)
    r2 = m.trajectory(b["feats"], v0=b["v0"], a0=b["a0"], kappa0=b["kappa0"],
                      nav_cmd=b["nav_cmd"])
    assert r2["vmax_cap"] is None and torch.equal(r2["traj"], r2["traj_uncapped"])


@pytest.mark.parametrize("kw,msg", [(dict(traj_prior="ha0_ext"), "traj_prior"),
                                    (dict(traj_grid=(16, 40)), "n_tokens"),
                                    (dict(strategic_cfg=None, tactical_cfg=None), "hierarchy"),
                                    (dict(traj_kappa_source="pose_past",
                                          traj_prior_units="steer"), "GEOMETRIC"),
                                    (dict(traj_kappa_source="yaw"), "traj_kappa_source")])
def test_h_sanity_refuses_a_misbuilt_head(kw, msg):
    with pytest.raises(ValueError, match=msg):
        RefAV1(_cfg(**kw))


def test_i_training_forward_without_its_target_or_t0_state_is_refused():
    m = _model()
    b = _batch(m)
    f, a, fut = b.pop("feats"), b.pop("actions"), b.pop("future_feats")
    kw = dict(b)
    kw.pop("traj_gt")
    with pytest.raises(ValueError, match="advertised and inert"):
        m(f, a, future_feats=fut, **kw)
    kw = dict(b)
    kw.pop("a0")
    with pytest.raises(ValueError, match="needs a0"):
        m(f, a, future_feats=fut, **kw)


#: the dev-box eval kit, overridable -- an absolute home path only as an env default
_EVAL = __import__("os").environ.get("TANITAD_REFAV1_EVAL_ROOT",
                                     "C:/Users/Admin/tanitad-data/refav1-eval141")


@pytest.mark.skipif(not __import__("os").path.isdir(_EVAL + "/refav1-fp8-eval"),
                    reason="the dev-box refav1 eval cache is not on this host")
def test_j_loader_gt_and_t0_kinematics_are_the_arms_own_and_future_free():
    import importlib.util
    import pathlib
    from tanitad.data.refav1_loader import RefAV1Windows
    arm_p = pathlib.Path(__file__).resolve().parents[2] / "taniteval" / "tools" / "refav1_arm.py"
    if not arm_p.is_file():
        pytest.skip("taniteval/tools/refav1_arm.py is not in this tree")
    spec = importlib.util.spec_from_file_location("_refav1_arm_r5", arm_p)
    arm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arm)
    C, E = pathlib.Path(_EVAL) / "refav1-fp8-eval", pathlib.Path(_EVAL) / "eps"
    names = sorted(p.stem for p in C.glob("*.pt") if p.stem != "index")[:2]
    ld = RefAV1Windows(C, E, op_window=4, op_steps=30, str_ext_steps=0, lru=2, seed=0,
                       episodes=names, traj_steps=30)
    for wi in (0, 7, len(ld) - 1):
        ei, t = ld.windows[wi]
        ld._order, ld._cursor = [wi], 0
        b = ld.batch(1)
        nm = ld.names[ei]
        o = torch.load(E / f"{nm}.v2ep.pt", map_location="cpu", weights_only=False)
        poses = o["poses"].float()
        _, v, kap = ld._episode(nm)
        assert torch.equal(b["traj_gt"][0], arm.gt_waypoints(poses, t, 30)[0])
        ext = arm.hold_ext_controls(ld, v, kap, t)
        assert float(b["a0"][0]) == float(ext[0]) and float(b["kappa0"][0]) == float(ext[1])
        assert bool(b["traj_mask"].all())
        # FUTURE-FREE: corrupt every frame after 2t -- a0 / kappa0 do not move; corrupt 2t-2 -- a0 does
        v2, k2 = v.clone(), kap.clone()
        v2[2 * t + 1:] += 5.0
        k2[2 * t + 1:] += 1.0
        g2, m2, a2, kk2 = ld._traj_window(nm, v2, k2, t)
        assert float(a2) == float(b["a0"][0]) and float(kk2) == float(b["kappa0"][0])
        v3 = v.clone()
        v3[2 * t - 2] -= 1.0
        assert float(ld._traj_window(nm, v3, kap, t)[2]) != float(b["a0"][0])


def test_l_the_eval_side_rebuilds_the_trained_model_strictly_and_identically(tmp_path):
    """The G-EVAL analogue: a checkpoint written by the trainer with EVERY new lever on is
    rebuilt by the EVAL tool's own `build_config` (from `ckpt['cfg']` AND from config.json's
    JSON form, where tuples are lists), loads STRICT, and forwards bit-identically."""
    import importlib.util
    import json
    import pathlib
    import sys
    root = pathlib.Path(__file__).resolve().parents[2]
    arm_p = root / "taniteval" / "tools" / "refav1_arm.py"
    if not arm_p.is_file():
        pytest.skip("taniteval/tools/refav1_arm.py is not in this tree")
    spec = importlib.util.spec_from_file_location("_refav1_arm_r5l", arm_p)
    arm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arm)
    tp = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "refa_v1_train.py"
    spec = importlib.util.spec_from_file_location("_refa_v1_train_r5l", tp)
    T = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = T
    spec.loader.exec_module(T)
    out = tmp_path / "run"
    assert T.main(["--smoke", "--steps", "2", "--bs", "2", "--device", "cpu", "--out",
                   str(out), "--strategic-off", "--vmax-input", "--w-goal", "0.1",
                   "--w-speed-band", "0.1", "--w-traj", "1", "--r5-prior", "damp50"]) == 0
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    cfg_json = json.loads((out / "config.json").read_text("utf-8"))["cfg"]
    models = []
    for src in (ck["cfg"], cfg_json):
        m = RefAV1(arm.build_config(src))
        res = m.load_state_dict(ck["model"], strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        models.append(m.eval())
    assert models[0].cfg.traj_prior == "damp50" and models[0].strategic_policy is None
    assert bool(models[0].goal_fitted)                        # the split constants travel
    b = _batch(models[0], bsz=3)
    trajs = []
    for m in models:
        with torch.no_grad():
            r = m.trajectory(b["feats"], v0=b["v0"], a0=b["a0"], kappa0=b["kappa0"],
                             nav_cmd=b["nav_cmd"], speed_max_ms=torch.tensor([9.0, 20.0, 30.0]))
        trajs.append(r["traj"])
    assert torch.equal(trajs[0], trajs[1]) and trajs[0].shape == (3, 30, 2)


@pytest.mark.skipif(not __import__("os").path.isdir(_EVAL + "/refav1-fp8-eval"),
                    reason="the dev-box refav1 eval cache is not on this host")
def test_k_pose_past_kappa0_is_refcv7s_ha0_ext_pose_and_reads_only_past_poses():
    """The strictly-admissible kappa0 (SPEC_REFCV7 §10 records the steer channel at t0 as
    'unruled at inference'): refcv7's own `prior_controls('ha0_ext_pose')` on the poses at 2t-2
    and 2t; a0 is the SAME backward difference under both sources."""
    import pathlib
    from tanitad.data.refav1_loader import RefAV1Windows
    from tanitad.models.kinematic_prior import POSE_KAPPA_CAP, prior_controls
    C, E = pathlib.Path(_EVAL) / "refav1-fp8-eval", pathlib.Path(_EVAL) / "eps"
    names = sorted(p.stem for p in C.glob("*.pt") if p.stem != "index")[:2]
    kw = dict(op_window=4, op_steps=30, str_ext_steps=0, lru=2, seed=0, episodes=names,
              traj_steps=30)
    st = RefAV1Windows(C, E, **kw)
    ps = RefAV1Windows(C, E, traj_kappa_source="pose_past", **kw)
    n_diff = 0
    for wi in range(0, len(st), 9):
        for ld in (st, ps):
            ld._order, ld._cursor = [wi], 0
        bs, bp = st.batch(1), ps.batch(1)
        ei, t = st.windows[wi]
        P = torch.load(E / f"{st.names[ei]}.v2ep.pt", map_location="cpu",
                       weights_only=False)["poses"].float()
        _a, k_ref = prior_controls("ha0_ext_pose", P[2 * t - 2: 2 * t + 1: 2][None], n_past=2,
                                   dt=0.2)
        assert float(bp["kappa0"][0]) == float(k_ref[0])
        assert abs(float(bp["kappa0"][0])) <= POSE_KAPPA_CAP
        assert float(bp["a0"][0]) == float(bs["a0"][0])          # ONE a0 definition
        assert torch.equal(bp["traj_gt"], bs["traj_gt"])
        n_diff += int(float(bp["kappa0"][0]) != float(bs["kappa0"][0]))
        # past-only: rewriting every pose AFTER 2t leaves the pose kappa0 unchanged
        P2 = P.clone()
        P2[2 * t + 1:, 2] += 0.7
        orig = ps._poses[ps.names[ei]]
        ps._poses[ps.names[ei]] = P2
        try:
            _, _, _, k2 = ps._traj_window(ps.names[ei], P2[:, 3], P2[:, 3] * 0, t)
        finally:
            ps._poses[ps.names[ei]] = orig        # never leave the cache corrupted
        assert float(k2) == float(bp["kappa0"][0])
    assert n_diff > 0              # control: the two sources are different quantities
