"""v8F rung alpha, arm alpha-1 -- the POSE-ONLY P10 (``--residual-prior p10_pose``): the frozen learned floor refit with the
pose track as its only input, inside the refcv decoder path.

alpha-0's prior ``ha0_ext_pose`` reads the past pose window only; the steer-reading ``p10`` would have moved TWO things (the
prior and the input set; SPEC_REFCV7 A5 excluded the recorded steer, NavSim has no steer channel). So alpha-1 is ``p10_pose``:
P10's eight features and its ``kd_x`` base are built from ``ha0_ext_pose``'s own triple (v0, ONE-frame a0, pose kappa0).

What is pinned (letters follow ``test_p10_prior.py``):
  (a) the artifact is the banked one (sha256 LITERAL), trained on TRAIN clips only, DISJOINT from eval139 / eval141; the
      steer artifact can never be loaded under this mode (nor the reverse);
  (b) the torch port equals the fit script's own numpy-float64 predictions on real windows AND an independent numpy
      reference written here, <= 1e-4 m;
  (c) the CLOCK: a window one frame early FAILS the tolerance on > 95 % of windows;
  (c') the features read ONLY the past pose window -- no future row, no steer / action channel;
  (d) P10 is FROZEN (buffers only, not in the state_dict, no gradient);
  (e) the conversion to per-tick controls: ``ADE(P10u, P10)`` small on real windows, Delta == 0 rolls EXACTLY to P10u;
  (g) the mode registry: ``p10_pose`` is a POSE mode and a TICK mode, never an ACTION mode;
  (h) the model: the trainer's own build and loss; REACHABILITY; MUTATIONS that must go RED.
CPU only. Fixtures ship with the stack tree under ``stack/tanitad/configs/v8f_p10/``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_HERE = Path(__file__).resolve().parent
_STACK = _HERE.parent
_REPO = _STACK.parent
for _p in (str(_STACK), str(_STACK / "scripts"), str(_HERE), str(_REPO / "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.models import kinematic_prior as KP  # noqa: E402
from tanitad.models import p10_prior as P10  # noqa: E402
from tanitad.models.kinematic import rollout_unicycle  # noqa: E402
# the tiny-model helpers of the steer-variant tests (one definition of "the tiny P10 arm")
from test_p10_prior import CFG, P10_FILE, P10_SHA256, T, _argv, _arm, _batch, _forward  # noqa: E402,F401

POSE_FILE = CFG / "p10_pose_prior.pt"
POSE_FIXTURE = CFG / "p10_pose_fixture.npz"
#: the banked artifact (raw/p10_pose/p10_pose_prior.pt) -- a LITERAL, not read from the file under test
POSE_SHA256 = "e70a9ae30b4418705442bffdeb62d9810a43b0a6ccce0c177796c72295fe693d"
PKG = _REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-20-v8f-rung-alpha"
TOL_M = 1e-4
W = 8


@pytest.fixture(scope="module")
def fxp():
    z = np.load(POSE_FIXTURE, allow_pickle=False)
    return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def pp():
    return P10.load_frozen_p10(POSE_FILE, expect_file_sha256=POSE_SHA256, variant="pose")


def _kin(fxp, shift: int = 0):
    return KP.pose_kin3(torch.from_numpy(fxp["poses"]).double(), W - shift)


def _arm_pose(T_, **kw):
    kw.setdefault("path", POSE_FILE)
    kw.setdefault("sha", POSE_SHA256)
    return _arm(T_, "p10_pose", **kw)


# =========================================================================== #
# (a) the artifact and its provenance                                         #
# =========================================================================== #
def test_a_the_artifact_is_the_banked_one_and_declares_the_POSE_variant(pp, tmp_path) -> None:
    assert P10.file_sha256(POSE_FILE) == POSE_SHA256 != P10_SHA256
    assert pp.variant == "pose" and pp.meta["units"] == "pose_kappa" and pp.meta["fit_clips"] == 4572
    assert P10.load_frozen_p10(POSE_FILE).variant == "pose"                 # derived from the artifact's own declaration
    with pytest.raises(P10.P10Error, match="sha256"):
        P10.load_frozen_p10(POSE_FILE, expect_file_sha256="0" * 64, variant="pose")
    with pytest.raises(P10.P10Error, match="does not exist"):
        P10.load_frozen_p10(tmp_path / "nope.pt", variant="pose")


def test_a_a_P10_fit_on_the_other_input_channel_is_ANOTHER_prior_and_is_refused_by_name() -> None:
    with pytest.raises(P10.P10Error, match="'steer' variant.*asked for the 'pose'"):
        P10.load_frozen_p10(P10_FILE, variant="pose")                          # the steer artifact under the pose mode
    with pytest.raises(P10.P10Error, match="'pose' variant.*asked for the 'steer'"):
        P10.load_frozen_p10(POSE_FILE, variant="steer")                        # and the reverse
    assert P10.load_frozen_p10(P10_FILE).variant == "steer"                    # the steer artifact is unchanged


def test_a_an_artifact_that_declares_no_unit_is_refused(tmp_path) -> None:
    o = torch.load(POSE_FILE, map_location="cpu", weights_only=True)
    for meta in ({}, {"units": "furlongs"}):
        bad = tmp_path / f"u{len(meta)}.pt"
        torch.save({**o, "meta": meta}, bad)
        with pytest.raises(P10.P10Error, match="feature unit"):
            P10.load_frozen_p10(bad)


def test_a_the_package_copy_and_the_shipped_copy_are_the_same_bytes_and_the_fit_record_says_what_it_fit() -> None:
    pk = PKG / "raw" / "p10_pose" / "p10_pose_prior.pt"
    assert pk.is_file() and P10.file_sha256(pk) == POSE_SHA256
    fit = json.loads((PKG / "raw" / "p10_pose" / "p10_fit.json").read_text(encoding="utf-8"))
    assert fit["artifact_sha256"] == POSE_SHA256 and fit["variant"] == "pose" and fit["units"] == "pose_kappa"
    assert (fit["fit_clips"], fit["n_manifest_clips"], fit["stride"], fit["seed"]) == (4572, 4572, 8, 0)
    assert fit["n_fit_rows"] == 41139 and fit["train_eval_overlap"] == 0
    assert "no steer" in fit["features"] and "POSE" in fit["features"]
    assert fit["disjoint_from_eval"]["eval139"]["overlap"] == fit["disjoint_from_eval"]["eval141"]["overlap"] == 0
    man = json.loads((CFG / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["p10_pose_prior.pt"]["sha256"] == POSE_SHA256 and man["p10_pose_prior.pt"]["units"] == "pose_kappa"


def test_a_fit_clips_are_TRAIN_clips_only_and_disjoint_from_eval139_and_eval141(fxp) -> None:
    fit = [str(c) for c in fxp["fit_clip_ids"]]
    assert len(fit) == 4572 and len(set(fit)) == 4572
    e141 = [str(c) for c in fxp["eval141_clip_ids"]]
    e139 = [str(c) for c in fxp["eval139_clip_ids"]]
    assert (len(e141), len(e139)) == (141, 139)
    assert P10.assert_fit_disjoint(fit, e141)["overlap"] == P10.assert_fit_disjoint(fit, e139)["overlap"] == 0
    with pytest.raises(P10.P10Error, match="overlap"):                        # the control: the check can SEE an overlap
        P10.assert_fit_disjoint(fit, e139[:5] + fit[:1])
    ids = (PKG / "raw" / "p10_pose" / "fit_clip_ids.txt").read_text(encoding="utf-8").split()
    assert sorted(ids) == sorted(fit)                                          # the banked list IS the fixture's list


# =========================================================================== #
# (b) known value                                                             #
# =========================================================================== #
def test_b_port_equals_the_BANKED_predictions_on_real_windows(fxp, pp) -> None:
    pred = pp(_kin(fxp)).numpy()
    banked = fxp["p10_banked"]
    assert pred.shape == banked.shape == (len(fxp["poses"]), 30, 2) and len(banked) >= 500
    assert fxp["poses"].dtype == np.float64
    assert float(np.abs(pred - banked).max()) <= TOL_M, float(np.abs(pred - banked).max())


def _numpy_reference(poses: np.ndarray, pp_) -> np.ndarray:
    """An independent reference written HERE: float64 numpy features, the S1 floors (`floors_kin`), the fit's predict."""
    fk = pytest.importorskip("taniteval.floors_kin")
    v, yaw = poses[:, :, 3], poses[:, :, 2]
    v0, a0 = v[:, -1], (v[:, -1] - v[:, -2]) / 0.1
    om = (((yaw[:, -1] - yaw[:, -2]) + np.pi) % (2 * np.pi) - np.pi) / 0.1
    kap = np.clip(om / np.maximum(v0, 2.0), -0.3, 0.3)
    kin = np.stack([v0, a0, kap], 1)
    base = fk.kd_x(fk.ha0_paths(v0, 0.2, 30),
                   fk.const_control_paths(v0, a0, kap, 0.2, 30, action_units="kappa")).astype(np.float32)
    st = pp_.tensors()
    h = st["net.0.weight"].shape[0]
    net = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.GELU(), torch.nn.Linear(h, 60))
    net.load_state_dict({k[len("net."):]: t for k, t in st.items() if k.startswith("net.")})
    return fk.predict_p10((net, (st["xm"].double().numpy(), st["xs"].double().numpy(), st["ym"].double().numpy()), None),
                          fk.kin_features(kin), base), kin


def test_b_port_equals_an_independent_numpy_reference(fxp, pp) -> None:
    ref, kin = _numpy_reference(fxp["poses"], pp)
    got = pp(_kin(fxp)).numpy()
    assert float(np.abs(got - ref).max()) <= TOL_M, float(np.abs(got - ref).max())
    assert float(np.abs(_kin(fxp).numpy() - kin).max()) <= 1e-12            # the features agree to float64 round-off


def test_b_the_triple_is_ha0_ext_pose_own_derivation(fxp) -> None:
    """P10-pose reads EXACTLY what alpha-0's prior reads: v0, the ONE-frame a0, the clamped pose curvature."""
    po = torch.from_numpy(fxp["poses"])
    k3 = KP.pose_kin3(po.double(), W)
    a0, k0 = KP.prior_controls("ha0_ext_pose", po.float(), W, None)
    assert torch.equal(k3[:, 0].float(), po[:, -1, 3].float())
    # ha0_ext_pose runs in the window's float32; the port in float64 (the fit's precision): equal to float32 round-off
    assert float((k3[:, 1].float() - a0).abs().max()) < 3e-5
    assert float((k3[:, 2].float() - k0).abs().max()) < 1e-5
    # the SHARED bounds: |kappa| <= POSE_KAPPA_CAP, and below POSE_KAPPA_V_FLOOR the yaw rate is attenuated, not amplified
    p = po[:2].clone().double()
    p[:, -1, 2] = p[:, -2, 2] + 0.5                                           # a 5 rad/s turn
    p[0, -1, 3] = 10.0
    p[1, -1, 3] = 0.1                                                         # standing
    k = KP.pose_kin3(p, W)[:, 2]
    assert float(k[0]) == pytest.approx(KP.POSE_KAPPA_CAP)                    # 5 / 10 = 0.5 -> clamped to the cap
    assert float(k[1]) == pytest.approx(KP.POSE_KAPPA_CAP)                    # 5 / max(0.1, 2) = 2.5 -> clamped
    p[:, -1, 2] = p[:, -2, 2] + 0.05                                          # 0.5 rad/s
    p[0, -1, 3], p[1, -1, 3] = 10.0, 0.1
    k = KP.pose_kin3(p, W)[:, 2]
    assert float(k[0]) == pytest.approx(0.05) and float(k[1]) == pytest.approx(0.25)    # 0.5 / 10, 0.5 / floor 2.0


# =========================================================================== #
# (c) the clock                                                               #
# =========================================================================== #
def test_c_a_ONE_TICK_SHIFTED_clock_FAILS_the_known_value_tolerance(fxp, pp) -> None:
    banked = fxp["p10_banked"]
    ok = np.abs(pp(_kin(fxp, 0)).numpy() - banked).max(axis=(1, 2))
    early = np.abs(pp(_kin(fxp, 1)).numpy() - banked).max(axis=(1, 2))
    assert float(ok.max()) <= TOL_M
    assert float((early > TOL_M).mean()) > 0.95                               # the shifted clock moves nearly EVERY window
    assert float(np.median(early)) > 100 * TOL_M
    assert float(np.median(np.abs(pp(_kin(fxp, 2)).numpy() - banked).max(axis=(1, 2)))) >= float(np.median(early))


def test_c_the_features_read_only_the_past_pose_window_and_never_a_steer_channel(fxp, pp) -> None:
    po = torch.from_numpy(fxp["poses"]).double()
    base = KP.pose_kin3(po, 6)
    po2 = po.clone()
    po2[:, 6:] = 1e6                                                          # poison every step after n_past
    assert torch.equal(KP.pose_kin3(po2, 6), base)
    with pytest.raises(KP.ResidualPriorError, match="n_past 1 < 2"):
        KP.pose_kin3(po, 1)                                                   # a backward difference needs t0-1
    ac = torch.zeros(po.shape[0], W, 2)
    with pytest.raises(KP.ResidualPriorError, match="ONLY the pose track"):
        KP.p10_tick_prior(pp, po.float(), W, ac)                              # a supplied action is REFUSED, not ignored
    with pytest.raises(KP.ResidualPriorError, match="ONLY the pose track"):
        KP.prior_controls("p10_pose", po.float(), W, ac, p10=pp)


# =========================================================================== #
# (d) frozen by construction                                                  #
# =========================================================================== #
def test_d_p10_pose_is_FROZEN_buffers_only_not_in_state_dict_and_carries_no_gradient(fxp, pp) -> None:
    assert list(pp.parameters()) == [] and len(list(pp.buffers())) == 7 and all(not b.requires_grad for b in pp.buffers())
    assert len(pp.state_dict()) == 0
    out = pp(_kin(fxp)[:4].clone().requires_grad_(True))
    assert not out.requires_grad and out.grad_fn is None
    assert pp.content_sha() == pp.content_sha_at_load


# =========================================================================== #
# (e) the conversion to per-tick controls                                     #
# =========================================================================== #
def test_e_ADE_P10u_vs_P10_on_the_fixture_windows_is_small_and_Delta_zero_rolls_EXACTLY_to_P10u(fxp, pp) -> None:
    po = torch.from_numpy(fxp["poses"]).float()
    a, k, p30 = KP.p10_tick_prior(pp, po, W, None)
    assert a.shape == k.shape == (len(po), 60) and p30.shape == (len(po), 30, 2)
    st = torch.zeros(len(po), 4)
    st[:, 3] = po[:, -1, 3]
    rolled = rollout_unicycle(st, torch.stack([a, k], -1), dt=0.1)[..., :2]
    ade6 = (rolled[:, 1::2] - p30).norm(dim=-1).mean(1)
    assert float(ade6.mean()) <= 0.05, float(ade6.mean())                     # the brief's bar
    assert float(ade6.median()) < 0.02
    hz = (5, 10, 15, 20, 30, 40, 50, 60)
    r0 = KP.roll_plan(torch.zeros(len(po), 1, len(hz), 2), a, k, po[:, -1, 3], hz, control_units="alat", tick=0.1)[:, 0]
    assert float((r0 - rolled[:, [h - 1 for h in hz]]).abs().max()) == 0.0    # Delta == 0 IS P10u, bit-exact


# =========================================================================== #
# (g) the mode registry and the stamp                                         #
# =========================================================================== #
def test_g_p10_pose_is_a_POSE_mode_and_a_TICK_mode_and_never_an_ACTION_mode(fxp, pp) -> None:
    assert KP.P10_POSE_MODE == "p10_pose" and KP.P10_MODES == ("p10", "p10_pose")
    assert "p10_pose" in KP.POSE_MODES and "p10_pose" in KP.TICK_MODES and "p10_pose" in KP.RESIDUAL_PRIOR_MODES
    assert "p10_pose" not in KP.ACTION_MODES and KP.needs_actions("p10_pose") is False
    assert "p10" in KP.ACTION_MODES and "p10" not in KP.POSE_MODES               # the steer variant is unchanged
    po = torch.from_numpy(fxp["poses"])[:3].float()
    a, k = KP.prior_controls("p10_pose", po, W, None, p10=pp)
    assert a.shape == k.shape == (3, 60)
    with pytest.raises(KP.ResidualPriorError, match="no frozen P10"):
        KP.prior_controls("p10_pose", po, W, None)                               # never another prior, silently
    with pytest.raises(KP.ResidualPriorError, match="POSE-variant"):
        KP.prior_controls("p10_pose", po, W, None, p10=P10.load_frozen_p10(P10_FILE))     # the steer P10 under the pose mode
    with pytest.raises(KP.ResidualPriorError, match="STEER-variant"):
        KP.prior_controls("p10", po, W, torch.zeros(3, W, 2), p10=pp)            # and the reverse
    with pytest.raises(KP.ResidualPriorError, match="does not read it"):
        KP.prior_controls("ha0_ext_pose", po, W, None, p10=pp)


def test_g_the_stamp_names_the_pose_p10_and_its_variant(pp) -> None:
    idn = {"path": "x", "file_sha256": pp.file_sha, "content_sha256": pp.content_sha(), "variant": "pose", "meta": {}}
    st = KP.prior_stamp("p10_pose", p10=idn)
    assert st["residual_prior"] == "p10_pose" and st["p10"]["file_sha256"] == POSE_SHA256
    assert st["reads"] == ["pose window (v[t0], v[t0-1], yaw[t0], yaw[t0-1])"] and "POSE-ONLY" in st["definition"]
    with pytest.raises(KP.ResidualPriorError):
        KP.prior_stamp("p10_pose")                                               # a P10 mode with no identity
    with pytest.raises(KP.ResidualPriorError, match="false provenance"):
        KP.prior_stamp("p10", p10=idn)                                           # a pose identity stamped as the steer mode
    with pytest.raises(KP.ResidualPriorError, match="false provenance"):
        KP.prior_stamp("p10_pose", p10={**idn, "variant": "steer"})


# =========================================================================== #
# (h) the model: the trainer's OWN build and loss                             #
# =========================================================================== #
def test_h_flags_pin_onto_the_decoder_config_and_a_P10_of_the_other_variant_is_refused_at_build(T) -> None:
    args, cfg, model = _arm_pose(T)
    d = cfg.core.decoder
    assert (d.residual_prior, d.p10_prior_path, d.p10_prior_sha256) == ("p10_pose", str(POSE_FILE), POSE_SHA256)
    assert model.core.decoder.p10.variant == "pose" and model.core.decoder.p10.file_sha == POSE_SHA256
    for kw in ({"sha": None}, {"path": None}):
        with pytest.raises(SystemExit, match="BOTH"):
            T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True),
                               T.build_parser().parse_args(_argv("p10_pose", **kw)))
    with pytest.raises(P10.P10Error, match="sha256"):
        _arm_pose(T, sha="1" * 64)
    with pytest.raises(P10.P10Error, match="does not exist"):
        _arm_pose(T, path=_REPO / "no_such_p10.pt")
    # the other variant's artifact (with ITS right sha) under this mode / the pose artifact under `p10`: refused BY NAME
    with pytest.raises(P10.P10Error, match="'steer' variant.*asked for the 'pose'"):
        _arm_pose(T, path=P10_FILE, sha=P10_SHA256)
    with pytest.raises(P10.P10Error, match="'pose' variant.*asked for the 'steer'"):
        _arm(T, "p10", path=POSE_FILE, sha=POSE_SHA256)


def test_h_the_model_emits_the_per_tick_prior_the_zero_anchor_IS_P10u_and_no_action_channel_is_read(T, fxp) -> None:
    _a, cfg, model = _arm_pose(T)
    b = _batch(T, cfg, fxp)
    assert "actions" in b                                                        # the dataset has them; the pose mode ignores them
    out = _forward(T, cfg, model, b)
    assert tuple(out["residual_prior_ctrl"].shape) == (4, 60, 2)
    rep = KP.plan_check(out, model.core.decoder.anchor_controls, decoder=model.core.decoder)
    assert rep["ok"], rep
    assert rep["max_abs_zero_anchor_minus_prior_m"] == 0.0 and rep["rows_v_gt_0_prior_nonzero"] == rep["rows_v_gt_0"] == 4
    # an ego_actions handed to a pose build is REFUSED (never silently dropped)
    ph = b["pose_hist"]
    with torch.no_grad():
        with pytest.raises(ValueError, match="SILENTLY DROPPED"):
            model.core(T.frames_to_device(b["frames"], "cpu"), nav_cmd=b["nav_cmd"], v0=b["pose_last"][:, 3],
                       ego_poses=ph, ego_n_past=int(ph.shape[1]), ego_actions=b["actions"])


def test_h_IN_DECODER_P10_POSE_EQUALS_THE_BANKED_PATH_and_a_shifted_clock_through_the_model_FAILS(T, fxp) -> None:
    """The clock proof through the MODEL route (set_ego_window -> _residual_prior_inputs -> P10). The model's window is
    float32 (as at inference), so the bar here is the float32 round-off of the yaw difference (<= 2e-3 m worst case at 6 s);
    the module-level test above and the gate's G-CLOCK hold the 1e-4 m bar on the float64 fixture."""
    _a, cfg, model = _arm_pose(T)
    seen = []
    h = model.core.decoder.p10.register_forward_hook(lambda m, i, o: seen.append(o.detach().clone()))
    b = _batch(T, cfg, fxp)
    _forward(T, cfg, model, b)
    banked = torch.from_numpy(fxp["p10_banked"][:4])
    assert len(seen) == 1 and float((seen[0] - banked).abs().max()) <= 2e-3
    b2 = dict(b)
    b2["pose_hist"] = torch.cat([b["pose_hist"][:, :1], b["pose_hist"][:, :-1]], 1)         # the clock ONE FRAME EARLY
    b2["pose_last"] = b2["pose_hist"][:, -1].clone()
    seen.clear()
    _forward(T, cfg, model, b2)
    h.remove()
    assert float((seen[0] - banked).abs().max()) > 0.1


def test_h_REACHABILITY_the_flag_reaches_the_loss_and_P10_pose_takes_no_gradient(T, fxp) -> None:
    """flag -> cfg -> decoder.residual_prior -> RefCModel._residual_prior_inputs -> FrozenP10(pose) -> per-tick prior ->
    roll_plan -> traj -> compute_losses_v3 -> backward, each hop observed. The residual heads take a non-zero gradient,
    P10 takes none, and the P10 VALUE reaches the loss."""
    args, cfg, model = _arm_pose(T)
    assert args.residual_prior == cfg.core.decoder.residual_prior == model.core.decoder.residual_prior == "p10_pose"
    b = _batch(T, cfg, fxp)
    calls = []
    hk = model.core.decoder.p10.register_forward_hook(lambda m, i, o: calls.append(1))
    model.train()
    torch.manual_seed(5)
    l1 = T.compute_losses_v3(model, b, "cpu", mode="diffusion")
    assert len(calls) >= 1
    l1["loss"].backward()
    heads = model.core.decoder.cascade.control_heads
    assert all(h_.weight.grad is not None and float(h_.weight.grad.abs().sum()) > 0.0 for h_ in heads)
    assert all(t.grad is None and not t.requires_grad for t in model.core.decoder.p10.buffers())
    assert model.core.decoder.p10.content_sha() == model.core.decoder.p10.content_sha_at_load
    model.zero_grad(set_to_none=True)
    base = float(l1["loss"].detach())
    with torch.no_grad():
        model.core.decoder.p10.ym.add_(torch.linspace(-3.0, 3.0, 60))
    torch.manual_seed(5)
    l2 = T.compute_losses_v3(model, b, "cpu", mode="diffusion")
    assert abs(float(l2["loss"].detach()) - base) > 1e-4
    hk.remove()


def test_h_MUTATION_unfrozen_P10_pose_takes_gradient(T, fxp, monkeypatch) -> None:
    import inspect
    _a, cfg, model = _arm_pose(T)
    b = _batch(T, cfg, fxp)
    p10 = model.core.decoder.p10
    for nm in ("w1", "b1", "w2", "b2", "ym"):
        t = getattr(p10, nm)
        delattr(p10, nm)
        p10.register_parameter(nm, torch.nn.Parameter(t.clone()))
    src = inspect.getsource(KP.p10_tick_prior).replace("with torch.no_grad():", "if True:")
    ns = dict(vars(KP))
    exec(compile(src, "<mutant>", "exec"), ns)
    monkeypatch.setattr(KP, "p10_tick_prior", ns["p10_tick_prior"])
    monkeypatch.setattr(type(p10), "forward", type(p10).forward.__wrapped__)
    model.train()
    T.compute_losses_v3(model, b, "cpu", mode="diffusion")["loss"].backward()
    g = [float(getattr(p10, nm).grad.abs().sum()) if getattr(p10, nm).grad is not None else 0.0
         for nm in ("w1", "b1", "w2", "b2", "ym")]
    assert max(g) > 0.0, g                                                      # the control: an unfrozen P10 DOES read non-zero
    assert len(list(p10.parameters())) == 5


def test_h_MUTATION_a_decoder_that_silently_falls_back_to_ha0_ext_pose_is_caught(T, fxp, monkeypatch) -> None:
    """alpha-1 degrading into alpha-0's OWN prior: the ha0_ext_pose constants broadcast to 60 ticks are shape-valid, so the
    decoder runs; the independent P10 recompute disagrees (the gate's G-LIVE clause)."""
    _a, cfg, model = _arm_pose(T)
    b = _batch(T, cfg, fxp)
    real = KP.prior_controls

    def fallback_tick(mode, poses, n_past=None, actions=None, **kw):
        kw.pop("p10", None)
        a0, k0 = real("ha0_ext_pose", poses, n_past, None, **kw)
        return a0[:, None].expand(-1, 60).contiguous(), k0[:, None].expand(-1, 60).contiguous()

    monkeypatch.setattr(KP, "prior_controls", fallback_tick)
    out = _forward(T, cfg, model, b)
    ref_a, _ref_k, _ = KP.p10_tick_prior(P10.load_frozen_p10(POSE_FILE), b["pose_hist"], b["pose_hist"].shape[1], None)
    assert float((out["residual_prior_ctrl"][..., 0] - ref_a).abs().max()) > 0.05
    monkeypatch.setattr(KP, "prior_controls", real)
    out2 = _forward(T, cfg, model, b)
    assert float((out2["residual_prior_ctrl"][..., 0] - ref_a).abs().max()) < 1e-6


def test_h_the_checkpoint_carries_the_pose_identity_and_the_stamp_names_the_variant(T) -> None:
    args, cfg, model = _arm_pose(T)
    extra = T._p10_ckpt_extra(model)
    assert extra["p10_prior"]["file_sha256"] == POSE_SHA256
    T._resume_p10_guard(dict(extra), model)
    with pytest.raises(SystemExit, match="different experiment"):
        T._resume_p10_guard({}, model)
    # a STEER-variant checkpoint into a pose build: a different P10 -> refused before any state is loaded
    _a2, _c2, steer = _arm(T)
    with pytest.raises(SystemExit, match="file_sha256"):
        T._resume_p10_guard(dict(T._p10_ckpt_extra(steer)), model)
    stamp = T._seam_stamp(cfg, args)
    assert stamp["residual_prior"]["residual_prior"] == "p10_pose" and stamp["residual_prior"]["p10"]["variant"] == "pose"
    T.assert_seams_are_built(model, stamp)
    bad = {**stamp, "residual_prior": {**stamp["residual_prior"], "p10": {**stamp["residual_prior"]["p10"], "variant": "steer"}}}
    with pytest.raises((SystemExit, ValueError)):
        T.assert_seams_are_built(model, bad)


def test_h_G_DVB_reads_the_BUILT_pose_P10_including_its_variant_and_goes_RED_on_each_mutation(T) -> None:
    from tanitad.train import declared_vs_built as D
    args, cfg, model = _arm_pose(T)
    parser = T.build_parser()
    mine = lambda: [str(m) for m in D.check(model, args, parser)                  # noqa: E731
                    if "p10" in str(m).lower() or "residual" in str(m).lower()]
    assert mine() == []
    p10 = model.core.decoder.p10
    # the variant clause: a pose artifact built as the steer variant (the builder forgot what it was fit on)
    p10.variant = "steer"
    assert any("p10.variant" in m for m in mine()), mine()
    p10.variant = "pose"
    assert mine() == []
    keep = p10.w2.clone()
    with torch.no_grad():
        p10.w2.zero_()
    assert any("content" in m for m in mine())
    with torch.no_grad():
        p10.w2.copy_(keep)
    model.core.decoder.p10 = None
    assert mine()
    model.core.decoder.p10 = p10
    p10.register_parameter("w1x", torch.nn.Parameter(torch.zeros(1)))
    assert any("Parameter" in m for m in mine())
    del p10._parameters["w1x"]
    args.p10_prior_sha256 = "2" * 64
    assert mine()
    args.p10_prior_sha256 = POSE_SHA256
    model.core.decoder.residual_prior = "ha0_ext_pose"                          # alpha-1 silently built as alpha-0
    assert mine()
    model.core.decoder.residual_prior = "p10_pose"
    assert mine() == []
    # refcv7's own helper: the profile's ONE admissible prior is p10_pose (a literal in declared_vs_built)
    assert D.V8F_RESIDUAL_PRIOR == "p10_pose"
    assert [m for m in D.check_refcv7_required(model, args, residual_prior=D.V8F_RESIDUAL_PRIOR)
            if "--residual-prior" in str(m)] == []


def test_h_other_builds_are_untouched_by_the_pose_variant(T) -> None:
    _a, cfg, model = _arm(T, "ha0_ext_pose", path=None, sha=None)
    assert model.core.decoder.p10 is None and not any("p10" in k for k in model.state_dict())
    _a2, _c2, m2 = _arm_pose(T)
    assert not any("p10" in k for k in m2.state_dict())
    assert hashlib.sha256(POSE_FILE.read_bytes()).hexdigest() == POSE_SHA256
