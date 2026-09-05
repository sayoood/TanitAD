"""refav1's KINEMATIC CONTRACT and its trivial-profile floor — the two instruments
that would have caught the 2026-09-03 void read on the night it happened.

CPU only, random-init ``RefAV1``, synthetic episodes. ⛔ Never touches Thor, a pod, a
real checkpoint or the banked eval slice.

WHAT IS PINNED

  A. THE ``ha0`` CONTROL ARM (D-REFAV1-HA0-ARM) — a constant-velocity STRAIGHT line at
     the measured ``v0`` (``a = 0, kappa = 0``), tier T1, beside ``ha``:
       A1 ``ha0`` is straight AND constant-speed on EVERY window;
       A2 ``ha`` is NOT straight on a window whose observed ``kappa != 0`` — so the
          two arms are genuinely different controls and ``ha`` is not the trivial floor;
       A3 the trivial-profile instrument reads ``straight_frac == const_speed_frac ==
          trivial_frac == 1.0`` for ``ha0``;
       A4 two arms that ARE equal are REPORTED as equal (``identical_to``) — the read
          that shipped had ``cl`` bit-identical to ``cl_navshuf`` on 122/140 windows
          and nothing said so;
       A5 ``ha0`` reaches the record with tier T1, an ``arm_meaning``, the four family
          rows, and the paired ``cl - ha0`` block;
       A6 no existing arm's semantics move: ``cl``/``ha``/``ol`` are bit-identical to a
          dump built without ``ha0`` in the arm list.

  B. THE KINEMATIC CONTRACT ITSELF — the invariant the corpus's action channel must
     satisfy for ``ol`` to mean anything:
       B1 with a TRUE-curvature action channel the ``ol`` replay reproduces GT to the
          integrator's own discretisation (the contract HOLDS);
       B2 with the channel the REAL corpus stores — ``steer = atan(L * kappa)``,
          `tanitad/data/physicalai.py:620-630` — the SAME replay misses laterally, and
          misses by MORE than a straight line does on a curving path. ⛔ This is a
          DELIBERATE-REGRESSION arm: it must FAIL, or the probe that found the defect
          could not have found it;
       B3 ``kappa = tan(steer) / L`` restores the contract — one variable, and the
          along-track channel does not get worse;
       B4 the defect is INVISIBLE to a correlation check (r > 0.999 between the stored
          channel and the true curvature) — which is exactly why
          ``refav1_loader.py:17-24`` justified the channel with r = 0.995 and could not
          see a 2.9x gain.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
TOOL = _REPO / "taniteval" / "tools" / "refav1_arm.py"
PROBE = _REPO / "taniteval" / "tools" / "refav1_kin_contract_probe.py"

_spec = importlib.util.spec_from_file_location("refav1_arm_kin_under_test", TOOL)
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

_pspec = importlib.util.spec_from_file_location("refav1_kin_probe_under_test", PROBE)
kp = importlib.util.module_from_spec(_pspec)
_pspec.loader.exec_module(kp)

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig  # noqa: E402
from tanitad.data.physicalai import WHEELBASE  # noqa: E402
from tanitad.refs.refa_v1 import RefAV1, RefAV1Config  # noqa: E402

N_TOK, D = 8, 32
T_EP = 101                        # 10 Hz frames -> T_C = 51 cache steps
T_C = math.ceil(T_EP / 2)
W = 4
NAMES = ("ep00", "ep01", "ep02")
DT = 0.2


# --------------------------------------------------------------------------- #
# fixture                                                                      #
# --------------------------------------------------------------------------- #
def _synth(i: int, *, store: str = "kappa", wheelbase: float = WHEELBASE,
           T: int = T_EP):
    """A real 10 Hz unicycle trajectory + the action channel the corpus would store.

    ``store="kappa"``  the channel IS the curvature (what `refav1_loader` assumes).
    ``store="steer"``  the channel is ``atan(L * kappa)`` — what `physicalai.signals_at`
                       ACTUALLY writes (`physicalai.py:620-630`). The trajectory is
                       IDENTICAL; only the recorded channel differs, so any change in
                       the ``ol`` arm is attributable to the channel alone.
    """
    t = torch.arange(T, dtype=torch.float32)
    v = 6.0 + 1.5 * i + 0.8 * torch.sin(2 * math.pi * t / 60.0)
    kap = 0.03 * torch.sin(2 * math.pi * t / 80.0 + i)
    x, y, yaw = 0.0, 0.0, 0.0
    poses = torch.zeros(T, 4)
    for f in range(T):
        poses[f] = torch.tensor([x, y, yaw, float(v[f])])
        x += float(v[f]) * math.cos(yaw) * 0.1
        y += float(v[f]) * math.sin(yaw) * 0.1
        yaw += float(v[f]) * float(kap[f]) * 0.1
    acts = torch.zeros(T, 2)
    acts[:, 0] = kap if store == "kappa" else torch.atan(wheelbase * kap)
    acts[1:, 1] = (v[1:] - v[:-1]) / 0.1
    return poses, acts


def _fixture(root: Path, *, store: str = "kappa"):
    cache, eps = root / "cache", root / "eps"
    cache.mkdir(parents=True, exist_ok=True)
    eps.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    for i, nm in enumerate(NAMES):
        torch.save(torch.randn(T_C, N_TOK, D, generator=g).to(torch.float8_e4m3fn),
                   cache / f"{nm}.pt")
        poses, acts = _synth(i, store=store)
        torch.save({"poses": poses, "actions": acts, "episode_id": nm,
                    "clip_id": f"clip-{nm}"}, eps / f"{nm}.v2ep.pt")
    return cache, eps


def _ckpt(root: Path):
    torch.manual_seed(0)
    cfg = RefAV1Config(
        tac_vocab_version="v7.0", d_enc=D, n_tokens=N_TOK, d_state=D,
        op_layers=1, op_heads=2, op_window=W, tac_layers=1, tac_queries=4,
        str_dim=16, str_layers=1,
        strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1, n_heads=2,
                                            d_ctx=16, d_cmd=8),
        tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1, n_heads=2,
                                          d_intent=16))
    m = RefAV1(cfg)
    with torch.no_grad():
        m.std.fit(torch.randn(64, N_TOK, D))
    p = root / "run" / "ckpt.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 7, "model": m.state_dict(), "opt": {}, "cfg": dict(vars(cfg))}, p)
    return p


def _args(root: Path, ckpt: Path, cache: Path, eps: Path, dump: str, **over):
    base = dict(ckpt=str(ckpt), config=None, cache=str(cache), episodes=str(eps),
                labels=None, nav=None, device="cpu", episodes_n=0,
                window_stride=20, horizon_k=10, wm_k=0, lru=4,
                plan_n_samples=8, plan_n_iters=1, plan_n_elites=2, plan_seed=0,
                nav_shuffle_seed=0, no_navshuf=False, with_nonav_arm=False,
                with_oracle_goal_arm=False, allow_nonstrict=False,
                dump_dir=str(root / dump))
    base.update(over)
    return argparse.Namespace(**base)


def _load(dump: Path):
    out = {}
    for f in sorted(dump.glob("ep*.npz")):
        with np.load(f) as d:
            out[f.stem] = {k: d[k] for k in d.files}
    return out


def _straight(tr):
    return np.abs(tr[..., 1]).max(axis=-1) < ra.TRIVIAL_STRAIGHT_M


def _const_speed(tr):
    n = tr.shape[0]
    p = np.concatenate([np.zeros((n, 1, 2)), tr[..., :2]], axis=1)
    return np.ptp(np.linalg.norm(np.diff(p, axis=1), axis=-1),
                  axis=1) < ra.TRIVIAL_CONST_SPEED_M


@pytest.fixture(scope="module")
def rolled(tmp_path_factory):
    """3 SYNTHETIC windows (one per episode, stride 20) on a random-init RefAV1."""
    root = tmp_path_factory.mktemp("refav1_kin")
    cache, eps = _fixture(root)
    ck = _ckpt(root)
    a = _args(root, ck, cache, eps, "dump")
    manifest = ra.run_dump(a)
    rec = ra.analyze_refav1(a.dump_dir, n_boot=40, seed=0)
    return root, a, manifest, rec, _load(Path(a.dump_dir))


# =========================================================================== #
# A. the ha0 control arm                                                      #
# =========================================================================== #
def test_A1_ha0_is_straight_and_constant_speed_on_every_window(rolled):
    _, _, manifest, _, dump = rolled
    assert manifest["grid"]["n_windows"] == 3, "the fixture must give 3 windows"
    n = 0
    for ep in dump.values():
        tr = ep["ha0"]
        assert _straight(tr).all(), "ha0 must be a STRAIGHT line on every window"
        assert _const_speed(tr).all(), "ha0 must be CONSTANT SPEED on every window"
        # and it must be the v0 line exactly: chord length == v0 * dt
        v0 = ep["v0"]
        step = np.linalg.norm(tr[:, 0, :2], axis=-1)
        assert np.allclose(step, v0 * ra.DT, atol=1e-4)
        n += tr.shape[0]
    assert n == 3


def test_A2_ha_is_not_straight_where_the_observed_kappa_is_nonzero(rolled):
    """`ha` and `ha0` must be genuinely different controls — otherwise the new arm
    adds nothing and `ha` really was the trivial floor after all."""
    _, _, _, _, dump = rolled
    bent = 0
    for nm, ep in dump.items():
        poses, acts = _synth(int(ep["clip_index"][0]))
        for w, t in enumerate(ep["ws"].tolist()):
            # the observed action that CLOSES at t0 — hold_action_controls' own rule
            kap_obs = float(acts[2 * (t - 1), 0])
            if abs(kap_obs) > 1e-6:
                assert not _straight(ep["ha"][w:w + 1])[0], (
                    f"{nm} w{w}: observed kappa {kap_obs} != 0 but ha is straight — "
                    f"ha would then be the same control as ha0")
                assert np.abs(ep["ha"][w] - ep["ha0"][w]).max() > 1e-4
                bent += 1
    assert bent >= 2, ("the fixture must contain windows with a non-zero observed "
                       "kappa, or A2 proves nothing")


def test_A3_trivial_profile_reads_1_for_ha0(rolled):
    _, _, _, rec, _ = rolled
    tp = rec["refav1"]["trivial_profile"]
    r = tp["arms"]["ha0"]
    assert r["straight_frac"] == 1.0
    assert r["const_speed_frac"] == 1.0
    assert r["trivial_frac"] == 1.0
    assert "ha0" in tp["degenerate_arms"], (
        "ha0 IS the constant-velocity profile — the instrument must say so")
    assert rec["refav1"]["trivial_profile"]["n_windows"] == rec["n_windows"]


def test_A4_two_equal_arms_are_reported_as_equal(tmp_path):
    """The instrument must NAME a pair of bit-identical arms. Built here rather than
    hoped for: a dump whose `ha` is overwritten with `ha0` must read identical."""
    files = []
    n = 5
    rng = np.random.default_rng(0)
    a1 = rng.normal(size=(n, 10, 2))
    a2 = a1.copy()                                            # deliberately identical
    a3 = a1 + 0.5
    for e in range(2):
        p = tmp_path / f"ep{e:03d}.npz"
        np.savez_compressed(p, g=a1, cl=a1, ha=a2, ha0=a3,
                            ws=np.arange(n), eid=np.array([e]),
                            clip_index=np.array([e]), v0=np.ones(n))
        files.append(str(p))
    tp = ra.trivial_profile(files, ["cl", "ha", "ha0"])
    assert tp["arms"]["cl"]["identical_to"]["ha"]["frac"] == 1.0
    assert tp["arms"]["ha"]["identical_to"]["cl"]["frac"] == 1.0
    assert "ha0" not in tp["arms"]["cl"]["identical_to"], (
        "an arm that DIFFERS must not be reported as identical")
    assert tp["arms"]["cl"]["identical_to"]["ha"]["n"] == 2 * n


def test_A5_ha0_reaches_the_record_as_a_T1_arm_with_families_and_the_paired_block(rolled):
    _, _, manifest, rec, dump = rolled
    assert "ha0" in manifest["arms"] and manifest["tiers"]["ha0"] == "T1"
    assert "constant" in manifest["arm_meaning"]["ha0"].lower()
    assert manifest["hold_v0_rule"] and "a = 0" in manifest["hold_v0_rule"]
    assert rec["tiers"]["ha0"] == "T1"
    fam = rec["arms"]["ha0"]["four_families"]
    for k in ("longitudinal", "lateral", "tactical", "strategic"):
        f = fam[k]
        assert f.get("tier") == "T1", k
        if f.get("status") == "UNAVAILABLE":
            assert f.get("reason") and f.get("n") is not None, k
    assert "paired_cl_minus_ha0" in rec["paired_decision_grade"]
    assert "paired_cl_minus_ha0" in rec["refav1"]["families_paired"]
    blk = rec["refav1"]["families_paired"]["paired_cl_minus_ha0"]
    assert blk["direction"] == "cl - ha0" and blk["tier"] == "T1 minus T1"
    assert "LATERAL" in blk["families"] or "lateral" in blk["families"]


def test_A6_adding_ha0_moves_no_existing_arm(rolled):
    """The new arm must be strictly ADDITIVE. Rather than diffing two rolls (which
    would only prove the code is deterministic), each pre-existing arm is REDERIVED
    from its own documented rule and required to match the dump bit for bit — so a
    silent change of semantics anywhere in the roll would fail here."""
    _, _, manifest, rec, dump = rolled
    # ⚠️ 2026-09-05: this was an EQUALITY against exactly four arms, and it went
    # RED the moment `ha0_ext` — the M11 integrator floor — was legitimately
    # added. An equality here does not test "no existing arm moved"; it tests
    # "no arm was ever added", which is not the guarantee the test is named for
    # and which every correct addition must break. The subset form asserts the
    # actual invariant and stays true across the next one.
    pre_existing = {"cl": "T1", "ha": "T1", "ha0": "T1", "ol": "T0"}
    assert {k: manifest["tiers"].get(k) for k in pre_existing} == pre_existing, (
        "the pre-existing tier stamps must be untouched")
    # ...and a NEW arm may not arrive unstamped or mis-stamped: every arm in the
    # manifest carries a tier, and every tier is one of the two that exist.
    assert all(manifest["tiers"].get(k) in ("T0", "T1")
               for k in manifest["tiers"]), manifest["tiers"]
    for nm, ep in dump.items():
        poses, acts = _synth(int(ep["clip_index"][0]))
        for w, t in enumerate(ep["ws"].tolist()):
            v0 = float(poses[2 * t, 3])
            # `ol` — the recorded (a, kappa) from the loader's rule
            ol, _ = _replay(poses, acts, t, 10)
            assert np.abs(ol - ep["ol"][w]).max() < 1e-4, (nm, w, "ol moved")
            # `ha` — the action that CLOSES at t0, held
            v = poses[:, 3]
            a_h = float((v[2 * t] - v[2 * t - 2]) / DT)
            k_h = float(acts[2 * t - 2, 0])
            ha = ra.paths_from_controls(
                torch.tensor([[a_h, k_h]] * 10, dtype=torch.float32),
                v0, DT, 10)[0].numpy()
            assert np.abs(ha - ep["ha"][w]).max() < 1e-4, (nm, w, "ha moved")
            # `g` — the GT waypoints
            g = ra.gt_waypoints(poses, t, 10)[0].numpy()
            assert np.abs(g - ep["g"][w]).max() < 1e-4, (nm, w, "g moved")
            assert abs(v0 - float(ep["v0"][w])) < 1e-5
    # and the pre-existing paired blocks are all still there
    assert {"paired_closed_minus_open", "paired_cl_minus_ha"} <= set(
        rec["paired_decision_grade"])


# =========================================================================== #
# B. the kinematic contract                                                   #
# =========================================================================== #
def _replay(poses, acts, t, k, *, transform=None):
    """The ``ol`` arm's own path: `_kin_actions` semantics -> `paths_from_controls`."""
    v = poses[:, 3]
    kapc = acts[:, 0]
    j = torch.arange(t, t + k)
    a = (v[torch.clamp((j + 1) * 2, max=v.shape[0] - 1)] - v[j * 2]) / DT
    kap = kapc[j * 2]
    if transform is not None:
        kap = transform(kap)
    ctrl = torch.stack([a, kap], dim=-1)
    v0 = float(v[2 * t])
    return ra.paths_from_controls(ctrl, v0, DT, k)[0].numpy(), v0


def _lat_lon(p, g):
    return (float(np.mean(np.abs(p[:, 1] - g[:, 1]))),
            float(np.mean(np.abs(p[:, 0] - g[:, 0]))))


#: (window t, episode) pairs spanning straight-ish and strongly curving windows.
_TW = [(4, 0), (4, 2), (14, 1), (14, 2), (24, 0), (24, 1)]
#: the fixture's own DISCRETISATION FLOOR: the poses are integrated at 10 Hz and the
#: contract is replayed at 0.2 s, so even a perfect channel leaves a residual.
#: MEASURED on this fixture over _TW: 0.015 - 0.114 m lateral. Stated, not guessed —
#: a threshold below it would fail for a reason that has nothing to do with kappa.
_FIXTURE_FLOOR_LAT_M = 0.15
_FIXTURE_FLOOR_LON_M = 0.10


def _three_ways(t: int, i: int):
    """(true-curvature, steer-as-kappa, tan/L-repaired, straight line) vs GT."""
    pk, ak = _synth(i, store="kappa")
    ps, asx = _synth(i, store="steer")
    g = ra.gt_waypoints(pk, t, 10)[0].numpy()
    assert np.array_equal(pk.numpy(), ps.numpy()), (
        "the two fixtures must differ ONLY in the recorded channel")
    p_true, v0 = _replay(pk, ak, t, 10)
    p_steer, _ = _replay(ps, asx, t, 10)
    p_fix, _ = _replay(ps, asx, t, 10, transform=lambda s: torch.tan(s) / WHEELBASE)
    line = kp.integrate(np.zeros(10), np.zeros(10), v0, DT)
    return g, _lat_lon(p_true, g), _lat_lon(p_steer, g), _lat_lon(p_fix, g), \
        _lat_lon(line, g)


@pytest.mark.parametrize("t,i", _TW)
def test_B1_the_contract_holds_when_the_channel_IS_curvature(t, i):
    g, true, _, _, line = _three_ways(t, i)
    assert true[0] < _FIXTURE_FLOOR_LAT_M, (
        f"the contract must hold with a TRUE-curvature channel: {true[0]:.4f} m")
    assert true[1] < _FIXTURE_FLOOR_LON_M, true[1]
    assert true[0] < 0.4 * line[0], (
        f"and it must beat the straight line decisively: {true[0]:.4f} vs "
        f"{line[0]:.4f}")


@pytest.mark.parametrize("t,i", _TW)
def test_B2_deliberate_regression_the_steer_channel_breaks_the_contract(t, i):
    """⛔ THIS ARM MUST FAIL, and the test asserts that it does.

    `physicalai.signals_at` (`physicalai.py:620-630`) writes ``atan(L * kappa)`` into
    ``actions[:, 0]``; a loader that reads it AS kappa over-rotates by ~L. On a curving
    path that is worse than driving straight — which is exactly the 0.716-vs-0.496 m
    reading MEASURED on the banked step-1,000 dumps. Without this arm the repair below
    would be an unfalsifiable claim.
    """
    g, true, steer, _, line = _three_ways(t, i)
    assert steer[0] > 1.2 * line[0] > 0, (
        f"the defect must be WORSE than a straight line: {steer[0]:.4f} vs "
        f"{line[0]:.4f}")
    assert steer[0] > 5 * true[0], (
        f"and far worse than the same replay with a true-curvature channel: "
        f"{steer[0]:.4f} vs {true[0]:.4f}")


@pytest.mark.parametrize("t,i", _TW)
def test_B3_one_variable_tan_over_L_restores_the_contract(t, i):
    g, true, steer, fix, _ = _three_ways(t, i)
    assert fix[0] < _FIXTURE_FLOOR_LAT_M and fix[0] < 0.25 * steer[0], (
        f"kappa = tan(steer)/L must restore the contract: {fix[0]:.4f} vs "
        f"{steer[0]:.4f}")
    # tan(atan(L k)) / L == k exactly, so the repair must land ON the true channel
    assert abs(fix[0] - true[0]) < 1e-5 and abs(fix[1] - true[1]) < 1e-5
    assert fix[1] <= steer[1] + 1e-6, (
        f"the repair must not break the ALONG-TRACK channel: {fix[1]:.4f} vs "
        f"{steer[1]:.4f}")


def test_B4_a_correlation_check_cannot_see_the_gain():
    """Why `refav1_loader.py:17-24`'s r = 0.995 justification could not catch this."""
    poses, acts = _synth(0, store="steer")
    _, true_acts = _synth(0, store="kappa")
    stored = acts[:, 0].numpy()
    truth = true_acts[:, 0].numpy()
    r = float(np.corrcoef(stored, truth)[0, 1])
    gain = float(np.polyfit(truth, stored, 1)[0])
    assert r > 0.999, "the two channels are near-perfectly correlated"
    assert gain > 2.5, f"and yet the GAIN is ~L: {gain:.3f}"
    # the instrument that DOES see it: a slope/regression, not a correlation
    assert abs(gain - WHEELBASE) < 0.15 * WHEELBASE


def test_B5_the_probe_integrator_matches_the_programme_integrator():
    """`refav1_kin_contract_probe.integrate` is a re-expression, not a second
    convention: it must reproduce `kinematic.rollout_unicycle` exactly."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1.0, size=10)
    kap = rng.normal(0, 0.05, size=10)
    v0 = 8.0
    mine = kp.integrate(a, kap, v0, DT, "shipped")
    theirs = ra.paths_from_controls(
        torch.tensor(np.stack([a, kap], -1), dtype=torch.float32), v0, DT, 10
    )[0].numpy()
    assert np.abs(mine - theirs).max() < 1e-4
