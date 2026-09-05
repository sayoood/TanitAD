"""``taniteval/tools/refav1_arm.py`` — the refav1 T0/T1 eval adapter, on a
random-init ``RefAV1`` and a SYNTHETIC 3-episode eval slice (fp8 cache, v2ep
poses from a real unicycle trajectory, v7.2 records for 2 of the 3 episodes).
CPU only; ⛔ never touches Thor, a pod, or a real checkpoint.

WHAT IS PINNED
  (1) END TO END: the dump is consumed by the UNTOUCHED ``t1_eval.analyze`` (and
      by the ``t1_eval.py --analyze-only`` CLI) with tier stamps cl/ha=T1, ol=T0;
      every arm carries the four family rows as OK or an explicit
      UNAVAILABLE/REFUSED with reason + n; the refav1 sidecar analysis yields the
      STRATEGIC (route head vs route_label, three nav conditionings, nav_valid
      excluded) and DECLARED-tactical blocks, the T0 WM diagnostic with its
      persist-last-field control, and the paired per-family contrasts.
  (2) NO FUTURE ENTERS T1 / HOLD-ACTION: perturbing the recorded future (v and
      kappa at every frame after the last window's t0) leaves ``ha`` AND ``cl``
      bit-identical, while ``ol`` (T0, recorded actions) and ``g`` CHANGE — so
      the perturbation provably sat in the consumed future.
  (3) SPEED FOLLOWS THE ARM'S OWN ACTIONS: every arm's chord-speed profile is
      ``v0 + sum_{j<k} a_j dt`` of ITS controls (not the GT speeds), and the
      a_dim=3 speed channel is built from the same rule.
  (4) THE KINEMATIC-CONTRACT CONTROL reads near zero: ``ol`` (recorded (a, kappa)
      integrated from v0) reproduces the GT path within 0.3 m ADE over 2 s.
  (5) MEASURED, not assumed: plan() with no supplied goal imagines one
      (``goal_source == "tactical_imagined"``, e609a98) and never brakes by tie
      any more (the pre-e609a98 defect was ``baseline:decel_1.5`` on every
      window); the record names the goal source, the goal space and the
      SELECTED manoeuvre per planning arm, and on this random-init fixture the
      plan is ``baseline:hold_v0`` (an init property, source named).
  (6) Config handling: config.json wins over ckpt['cfg'] when consistent, a
      contradiction is refused by name, an unknown field is refused by name.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
TOOL = _REPO / "taniteval" / "tools" / "refav1_arm.py"
T1_TOOL = _REPO / "taniteval" / "tools" / "t1_eval.py"

_spec = importlib.util.spec_from_file_location("refav1_arm_under_test", TOOL)
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig  # noqa: E402
from tanitad.refs.refa_v1 import RefAV1, RefAV1Config  # noqa: E402

N_TOK, D = 8, 32
T_EP = 101                       # 10 Hz frames -> T_C = 51 cache steps
T_C = math.ceil(T_EP / 2)
W = 4
NAMES = ("ep00", "ep01", "ep02")
CLIP = {"ep00": "clip-aaaa", "ep01": "clip-bbbb", "ep02": "clip-cccc"}
# v7.0 tokens; ep02 has NO record (nav_valid False, labels -100).
REC = {"ep00": dict(lat="NUDGE_L", lon="BRAKE_TO", nav="NAV_TURN_L",
                    t0=2.0, band=(2.0, 6.0)),          # in band for t*0.2 in [0,4]
       "ep01": dict(lat="TURN_R", lon="ACCELERATE", nav="NAV_FOLLOW_ROAD",
                    t0=1.0, band=(2.0, 4.0))}          # in band for t*0.2 in [0,2]


def _synth_poses(i: int, T: int = T_EP):
    """A real unicycle trajectory at 10 Hz, integrated in rollout_unicycle's
    ORDER (pose advances on the start-of-step speed, v updates last)."""
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
    acts[:, 0] = kap                                      # MEASURED order: kappa first
    acts[1:, 1] = (v[1:] - v[:-1]) / 0.1
    return poses, acts


def _record(clip_id, *, lat, lon, nav, t0, band):
    return {"schema_version": "s2-geom-v7", "vocab": "v7", "clip_id": clip_id,
            "t0_s": t0,
            "bands": {"operative_s": [0.0, 2.0], "tactical_s": list(band),
                      "strategic_s": [8.0, 30.0]},
            "a_tac": {"lat": lat, "lon": lon},
            "nav_command": {"token": nav, "provenance": "ego-future", "oracle": True,
                            "args": {"distance_m": 10.0, "time_s": 2.0}}}


def _fixture(root: Path, perturb_after_frame: int | None = None):
    cache, eps = root / "cache", root / "eps"
    cache.mkdir(parents=True, exist_ok=True)
    eps.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    for i, nm in enumerate(NAMES):
        f = torch.randn(T_C, N_TOK, D, generator=g)
        torch.save(f.to(torch.float8_e4m3fn), cache / f"{nm}.pt")
        poses, acts = _synth_poses(i)
        if perturb_after_frame is not None:
            sl = slice(perturb_after_frame, None)
            poses[sl, 3] += 2.0
            acts[sl, 0] += 0.05
        torch.save({"poses": poses, "actions": acts, "episode_id": nm,
                    "clip_id": CLIP[nm]}, eps / f"{nm}.v2ep.pt")
    recs = [_record(CLIP[nm], **REC[nm]) for nm in ("ep00", "ep01")]
    lp = root / "labels.jsonl"
    lp.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    return cache, eps, lp


def _tiny_cfg() -> RefAV1Config:
    return RefAV1Config(
        tac_vocab_version="v7.0", d_enc=D, n_tokens=N_TOK, d_state=D,
        op_layers=1, op_heads=2, op_window=W, tac_layers=1, tac_queries=4,
        str_dim=16, str_layers=1,
        strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1, n_heads=2,
                                            d_ctx=16, d_cmd=8),
        tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1, n_heads=2,
                                          d_intent=16))


def _ckpt(root: Path, cfg: RefAV1Config | None = None, extra_cfg: dict | None = None):
    torch.manual_seed(0)
    cfg = cfg or _tiny_cfg()
    m = RefAV1(cfg)
    with torch.no_grad():
        m.std.fit(torch.randn(64, N_TOK, D))
    d = dict(vars(cfg))
    d.update(extra_cfg or {})
    p = root / "run" / "ckpt.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 7, "model": m.state_dict(), "opt": {}, "cfg": d}, p)
    return p


def _args(root: Path, ckpt: Path, cache: Path, eps: Path, lp: Path | None, **over):
    base = dict(ckpt=str(ckpt), config=None, cache=str(cache), episodes=str(eps),
                labels=str(lp) if lp else None, nav=str(lp) if lp else None,
                device="cpu", episodes_n=0, window_stride=1, horizon_k=10,
                wm_k=0, lru=4, plan_n_samples=8, plan_n_iters=1, plan_n_elites=2,
                plan_seed=0, nav_shuffle_seed=0, no_navshuf=False,
                with_nonav_arm=False, with_oracle_goal_arm=False,
                allow_nonstrict=False, dump_dir=str(root / "dump"))
    base.update(over)
    return argparse.Namespace(**base)


def _load_dump(dump: Path):
    out = {}
    for f in sorted(dump.glob("ep*.npz")):
        with np.load(f) as d:
            out[f.stem] = {k: d[k] for k in d.files}
    return out


def _chord_speeds(path: np.ndarray, dt: float) -> np.ndarray:
    p = np.concatenate([np.zeros((path.shape[0], 1, 2)), path[..., :2]], 1)
    return np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1) / dt


# =========================================================================== #
# (1) end to end                                                              #
# =========================================================================== #
@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    root = tmp_path_factory.mktemp("refav1_e2e")
    cache, eps, lp = _fixture(root)
    ck = _ckpt(root)
    a = _args(root, ck, cache, eps, lp, with_oracle_goal_arm=True)
    manifest = ra.run_dump(a)
    rec = ra.analyze_refav1(a.dump_dir, n_boot=60, seed=0)
    return root, a, manifest, rec


def test_dump_is_the_t1_contract_and_analyze_reads_it(e2e):
    root, a, manifest, rec = e2e
    dump = Path(a.dump_dir)
    files = sorted(dump.glob("ep*.npz"))
    assert len(files) == 3 and (dump / "manifest.json").exists()
    assert len(sorted((dump / "decisions").glob("ep*.npz"))) == 3
    d = _load_dump(dump)["ep000"]
    N = d["g"].shape[0]
    assert d["g"].shape == (N, 10, 2) and d["g"].dtype == np.float32
    # `ha0` (constant velocity, D-REFAV1-HA0-ARM, 2026-09-03) is ADDITIVE: it joins
    # the default arm list; its own semantics are pinned in test_refav1_kin_contract.py.
    for arm in ("cl", "ha", "ha0", "ol", "cl_navshuf", "cl_oraclegoal"):
        assert d[arm].shape == (N, 10, 2), arm
    assert d["v0"].shape == (N,) and d["eid"].tolist() == [0]
    # `ha0_ext` (2026-09-05, D-REFAV1-CCOS-EVAL): the ECHO control — constant
    # (a0, kappa0) of the MEASURED t0 state, T1 like `ha` / `ha0` (no recorded
    # future). ADDITIVE, like `ha0` before it.
    assert set(rec["arms"]) == {"cl", "ha", "ha0", "ha0_ext", "ol", "cl_navshuf",
                                "cl_oraclegoal"}
    assert rec["tiers"] == {"cl": "T1", "ha": "T1", "ha0": "T1", "ha0_ext": "T1",
                            "ol": "T0", "cl_navshuf": "T1", "cl_oraclegoal": "T0"}
    assert rec["n_windows"] == manifest["grid"]["n_windows"] == 3 * N
    for arm, blk in rec["arms"].items():
        fam = blk["four_families"]
        for k in ("longitudinal", "lateral", "tactical", "strategic"):
            f = fam[k]
            assert f.get("tier") == blk["tier"]
            if f.get("status") == "UNAVAILABLE":
                assert f.get("reason") and f.get("n") is not None, (arm, k)
        assert fam["longitudinal"]["anti_echo"]["status"] == "OK"     # v0 travelled
        assert fam["tactical"]["status"] == "OK"                      # from-trajectory
        assert fam["strategic"]["status"] == "UNAVAILABLE"            # by design
        assert "ade_dense_m" in blk["intervals"]["metrics"]
    # the paired decision-grade contrasts t1_eval emits
    assert {"paired_closed_minus_open", "paired_cl_minus_ha",
            "paired_cl_minus_navshuf"} <= set(rec["paired_decision_grade"])


def test_the_t1_eval_CLI_still_reads_the_dump_untouched(e2e):
    root, a, _, _ = e2e
    out = root / "t1_cli.json"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, str(T1_TOOL), "--arm", "x",
                        "--analyze-only", a.dump_dir, "--out", str(out),
                        # ⚠️ `ha0=T1` is passed EXPLICITLY because
                        # `t1_eval.DEFAULT_TIERS` does not know the arm yet — the bare
                        # t1_eval CLI REFUSES an unstamped arm (t1_eval.py:307, and
                        # that guard is correct). The durable fix is one line in
                        # DEFAULT_TIERS; until it lands this flag is required for any
                        # refav1 dump read through the standalone CLI.
                        "--tiers", "cl_navshuf=T1,cl_oraclegoal=T0,ha0=T1",
                        "--n-boot", "30", "--dt", "0.2"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["tiers"]["ol"] == "T0" and rec["tiers"]["cl"] == "T1"


def test_refav1_families_strategic_tactical_wm_and_paired(e2e):
    root, a, manifest, rec = e2e
    r = rec["refav1"]
    N = rec["n_windows"]
    # --- STRATEGIC: route head vs route_label under three conditionings ------
    s = r["strategic"]
    assert s["n_nav_valid"] * 3 == 2 * N                     # ep02 has no record
    assert abs(s["nav_valid_frac"] - 2 / 3) < 1e-3           # (rounded to 4 dp)
    assert s["n_route_labeled"] > 0
    assert s["n_excluded_no_route_label"] == N - s["n_route_labeled"]
    for c in ("nav_true", "nav_shuffled", "nav_zero"):
        blk = s["conditionings"][c]
        assert blk["status"] == "OK" and blk["tier"] == "T1"
        assert blk["n"] == s["n_route_labeled"]              # labeled ∩ nav_valid
        assert "accuracy" in blk and "ci" in blk and "nav_implied_route_agreement" in blk
        assert blk["classes"] == ["route_left", "route_straight", "route_right"]
    assert s["paired_true_minus_shuffled_accuracy"]["estimator"] == \
        "paired_episode_cluster_bootstrap"
    assert s["nav_shuffle"]["n_valid"] == round(2 / 3 * N)
    # --- TACTICAL, declared heads vs v7.0 labels ---------------------------------
    t = r["tactical_declared"]
    assert t["vocabulary"] == "v7.0"
    for hk in ("lat", "lon"):
        blk = t["conditionings"][f"{hk}_nav_true"]
        assert blk["status"] == "OK" and len(blk["classes"]) == 8
        assert blk["n"] + blk["n_excluded_no_label"] == N
        assert 0 < blk["n"] < N                                # mixed in/out of band
        assert f"{hk}_paired_true_minus_shuffled_accuracy" in t
    # --- T0 WM diagnostic with its known-value control ---------------------------
    wm = r["wm_diagnostic_T0"]
    assert wm["tier"] == "T0" and wm["k_wm"] == 30
    assert "feat_mse_model_mean_over_steps" in wm["intervals"]
    assert "feat_mse_const_step30" in wm["intervals"]
    assert wm["paired_const_minus_model"]["mean_over_steps"]["n_windows"] == N
    assert math.isfinite(wm["tgt_std_mean"])
    # the CONSTANT-ONLY control reads its KNOWN VALUE: in frozen space the
    # zero prediction scores the target's per-channel variance (~1.0 when the
    # standardizer describes the corpus — here it was fit on the same
    # distribution the cache was drawn from).
    z = wm["intervals"]["feat_mse_zero_mean_over_steps"]["mean"]
    assert abs(z - wm["tgt_std_mean"] ** 2) < 0.15, (z, wm["tgt_std_mean"])
    assert "paired_zero_minus_model_mean_over_steps" in wm
    # the protocol is DECLARED on every arm's four-families block, not UNDECLARED;
    # goal_source is DERIVED from the per-window PlanResult provenance (e609a98):
    # this tiny model has the hierarchy, so its planning arms imagined their goal
    for arm, blk in rec["arms"].items():
        assert blk["four_families"]["_protocol_undeclared"] == [], arm
        gs = blk["four_families"]["_protocol"]["goal_source"]
        assert gs.startswith("per arm") and "cl: tactical_imagined 100.0 %" in gs, gs
        assert "cl_oraclegoal: supplied 100.0 %" in gs, gs
    # --- paired per-family contrasts on the same windows -------------------------
    fp = r["families_paired"]["paired_cl_minus_ha"]
    assert fp["tier"] == "T1 minus T1"
    for fam in ("longitudinal", "lateral", "tactical", "ADE"):
        assert fam in fp["families"], fam
    lon = fp["families"]["longitudinal"]["LON_speed_mae_mps"]
    assert lon["estimator"] == "paired_episode_cluster_bootstrap" and lon["n_windows"] == N
    gs = r["protocol"]["goal_source"]
    assert gs.startswith("per arm") and "tactical_imagined" in gs and "supplied" in gs
    # the per-arm goal provenance block (the SELECTED manoeuvre behind the plan)
    for arm in ("cl", "cl_navshuf"):
        pl = r["planner"][arm]
        assert pl["goal_source_fractions"] == {"tactical_imagined": 1.0}, arm
        assert pl["goal_space"] == "tactical_query_field"
        assert pl["n_goal_action"] == N and pl["goal_action_lat"] and pl["goal_action_lon"]
        assert abs(sum(pl["goal_action_lat"].values()) - 1.0) < 1e-3
        ag = pl["goal_vs_declared_head_agreement"]
        assert ag["n"] == N and ag["lat"] == 1.0 and ag["lon"] == 1.0    # one decode path
    assert r["planner"]["cl_oraclegoal"]["goal_source_fractions"] == {"supplied": 1.0}
    assert r["planner"]["cl_oraclegoal"]["n_goal_action"] == 0


def test_default_goal_plan_is_no_longer_a_brake_by_tie_and_the_record_names_its_goal(e2e):
    """HISTORY — WHY THIS TEST EXISTED IN ITS OLD FORM. Before e609a98, with
    goal_field=None the cost had no goal term, so EVERY zero-curvature
    constant-accel candidate (cv, hold_v0, decel_1.5) had jerk 0 and scored
    exactly 0; `icem_plan`'s floor loop kept the LAST tie (`<=` over the
    baseline dict), which was `decel_1.5`, and the "T1 plan" was a constant
    −1.5 m/s² brake BY CONSTRUCTION — this test asserted exactly that
    (`source_fractions == {"baseline:decel_1.5": 1.0}` + the −1.5 m/s² speed
    profile) so the defect could not hide. `refa_v1.plan` now imagines its
    default goal (vision + nav + v0 -> `tactical_imagined`) and resolves ties to
    `hold_v0`, so the OLD assertion fails BY DESIGN and is replaced by its
    negation plus the provenance the adapter now records.

    ⚠️ MEASURED on this random-init fixture the plan is `baseline:hold_v0` on
    every window — a property of the INIT (residual heads x1e-3, every rollout
    within ~1e-3 of the start field, smoothness terms dominate), NOT of the
    wiring; `test_refa_v1_plan_goal` pins that the search wins in an
    informative world. What is pinned HERE: no brake by tie, the chord speeds
    equal the held v0, and the record names the goal the plan ran against."""
    root, a, manifest, rec = e2e
    pl = rec["refav1"]["planner"]
    assert "baseline:decel_1.5" not in pl["cl"]["source_fractions"], pl["cl"]
    assert pl["cl"]["goal_source_fractions"] == {"tactical_imagined": 1.0}
    assert pl["cl"]["goal_space"] == "tactical_query_field"
    assert manifest["goal"]["space"] == ["tactical_query_field"]
    d = _load_dump(Path(a.dump_dir))["ep000"]
    sp = _chord_speeds(d["cl"], ra.DT)
    assert np.allclose(sp, d["v0"][:, None], atol=1e-4)       # hold_v0: speed = v0 ...
    assert np.allclose(d["cl"][..., 1], 0.0, atol=1e-6)        # ... straight ahead
    # the oracle-goal arm is stamped T0 and its provenance is recorded too
    assert manifest["tiers"]["cl_oraclegoal"] == "T0"
    assert "cl_oraclegoal" in pl
    assert pl["cl_oraclegoal"]["goal_source_fractions"] == {"supplied": 1.0}


def test_kinematic_contract_control_reproduces_GT(e2e):
    """`ol` = the RECORDED (a, kappa) integrated from v0 by the programme's
    unicycle. Near-zero ADE is the known value that makes every other arm's ADE
    interpretable; a sign flip in kappa or the ego frame would show here."""
    root, a, _, rec = e2e
    ade = rec["arms"]["ol"]["intervals"]["metrics"]["ade_dense_m"]["mean"]
    assert ade < 0.3, ade
    assert rec["arms"]["ha"]["intervals"]["metrics"]["ade_dense_m"]["mean"] > ade


# =========================================================================== #
# (2) no future enters T1 / hold-action                                       #
# =========================================================================== #
def test_perturbing_the_recorded_future_moves_ol_and_g_but_not_ha_or_cl(tmp_path):
    ck = _ckpt(tmp_path)
    ra_root = tmp_path / "A"
    rb_root = tmp_path / "B"
    cache_a, eps_a, lp_a = _fixture(ra_root)
    # first pass to learn the grid's last t0, then perturb strictly AFTER it
    a = _args(ra_root, ck, cache_a, eps_a, lp_a)
    ra.run_dump(a)
    da = _load_dump(Path(a.dump_dir))
    t_last = int(max(int(d["ws"].max()) for d in da.values()))
    cache_b, eps_b, lp_b = _fixture(rb_root, perturb_after_frame=2 * t_last + 1)
    b = _args(rb_root, ck, cache_b, eps_b, lp_b)
    ra.run_dump(b)
    db = _load_dump(Path(b.dump_dir))
    for ep in da:
        A, B = da[ep], db[ep]
        assert np.array_equal(A["ws"], B["ws"]) and np.array_equal(A["v0"], B["v0"])
        # NOTHING after t0 reaches the hold-action or the planning arm:
        assert np.array_equal(A["ha"], B["ha"]), ep
        assert np.array_equal(A["cl"], B["cl"]), ep
        assert np.array_equal(A["cl_navshuf"], B["cl_navshuf"]), ep
    # ... and the perturbation WAS in the consumed future: the T0 arm (recorded
    # future actions) moved on every window whose 2 s horizon reaches the
    # perturbed frames. GT positions were deliberately left untouched (only the
    # recorded v / kappa channels were perturbed), so `g` must NOT move — which
    # is what makes the ol-vs-ha/cl contrast a test of the ACTION channel alone.
    for ep in da:
        assert np.array_equal(da[ep]["g"], db[ep]["g"]), ep
        assert not np.array_equal(da[ep]["ol"][-1], db[ep]["ol"][-1]), ep


def test_hold_action_controls_read_only_frames_up_to_2t():
    from tanitad.data.refav1_loader import RefAV1Windows
    v = torch.linspace(5.0, 15.0, 41)
    kap = torch.linspace(-0.05, 0.05, 41)

    class _L:                       # only _kin_actions / dt are used
        dt = 0.2
        _kin_actions = RefAV1Windows._kin_actions
    t = 7
    h0 = ra.hold_action_controls(_L(), v, kap, t)
    v2, k2 = v.clone(), kap.clone()
    v2[2 * t + 1:] += 3.0
    k2[2 * t + 1:] += 0.5
    h1 = ra.hold_action_controls(_L(), v2, k2, t)
    assert torch.equal(h0, h1)
    assert torch.isclose(h0[0], (v[2 * t] - v[2 * t - 2]) / 0.2)
    assert torch.isclose(h0[1], kap[2 * t - 2])
    with pytest.raises(ValueError, match="t >= 1"):
        ra.hold_action_controls(_L(), v, kap, 0)


# =========================================================================== #
# (3) speed follows the arm's OWN actions                                     #
# =========================================================================== #
def test_every_arm_speed_profile_is_v0_plus_its_own_accel(e2e):
    root, a, _, _ = e2e
    dump = Path(a.dump_dir)
    d = _load_dump(dump)["ep001"]
    with np.load(dump / "decisions" / "ep001.npz") as s:
        ctrls = {arm: s[f"{arm}_controls"] for arm in ("cl", "cl_navshuf",
                                                       "cl_oraclegoal", "ha")}
    g_sp = _chord_speeds(d["g"], ra.DT)
    for arm, c in ctrls.items():
        v_open = d["v0"][:, None] + np.concatenate(
            [np.zeros((c.shape[0], 1)), np.cumsum(c[..., 0], 1)[:, :-1]], 1) * ra.DT
        assert np.allclose(_chord_speeds(d[arm], ra.DT), np.clip(v_open, 0, None),
                           atol=2e-3), arm
    # the hold arm's accel is the closed-at-t0 accel, non-zero, and its speeds are
    # NOT the GT speeds (the fixture accelerates sinusoidally)
    assert np.abs(ctrls["ha"][:, 0, 0]).max() > 0
    assert not np.allclose(_chord_speeds(d["ha"], ra.DT), g_sp, atol=0.05)


def test_a_speed_channel_checkpoint_rolls_through_the_MODELS_own_augment(tmp_path):
    """``speed_channel=True``: the model derives v_k = v0 + sum a_j dt itself
    (``augment_actions``) and REFUSES a widened input, so the adapter must pass
    2-wide controls + v0 to forward()/plan(). Pinned end to end: the roll runs,
    the dumped controls stay 2-wide, and every arm's speed profile still
    follows its OWN accelerations from the measured v0."""
    cfg = _tiny_cfg()
    cfg.speed_channel = True
    ck = _ckpt(tmp_path, cfg=cfg)
    cache, eps, lp = _fixture(tmp_path)
    a = _args(tmp_path, ck, cache, eps, lp)
    man = ra.run_dump(a)
    assert man["speed_channel"]["enabled"] is True
    assert man["model"]["a_dim"] == 2
    d = _load_dump(Path(a.dump_dir))["ep000"]
    with np.load(Path(a.dump_dir) / "decisions" / "ep000.npz") as s:
        c = s["cl_controls"]
        assert c.shape[-1] == 2
        v_open = d["v0"][:, None] + np.concatenate(
            [np.zeros((c.shape[0], 1)), np.cumsum(c[..., 0], 1)[:, :-1]], 1) * ra.DT
        assert np.allclose(_chord_speeds(d["cl"], ra.DT), np.clip(v_open, 0, None),
                           atol=2e-3)
        assert np.isfinite(s["wm_mse_model"]).all()


# =========================================================================== #
# (6) config handling                                                         #
# =========================================================================== #
def test_config_json_wins_when_consistent_and_a_contradiction_is_refused(tmp_path):
    ck = _ckpt(tmp_path)
    cfg = _tiny_cfg()
    side = ck.parent / "config.json"
    side.write_text(json.dumps({"cfg": ra._jsonable(vars(cfg))}), encoding="utf-8")
    m, c, prov = ra.load_model(str(ck), device="cpu")
    assert prov["config_source"] == str(side) and c.op_window == W
    bad = ra._jsonable(vars(cfg))
    bad["op_window"] = 6
    side.write_text(json.dumps({"cfg": bad}), encoding="utf-8")
    with pytest.raises(SystemExit, match="CONTRADICTS"):
        ra.load_model(str(ck), device="cpu")


def test_unknown_config_field_is_refused_by_name(tmp_path):
    ck = _ckpt(tmp_path, extra_cfg={"no_such_field_zzz": 1})
    with pytest.raises(SystemExit, match="no_such_field_zzz"):
        ra.load_model(str(ck), device="cpu")


def test_config_that_disagrees_with_the_weights_shape_is_refused(tmp_path):
    """A config claiming speed_channel=True over weights trained 2-wide is a
    size mismatch in the predictors' action projection — refused by name, not
    surfaced as a raw torch traceback."""
    ck = _ckpt(tmp_path, extra_cfg={"speed_channel": True})
    with pytest.raises(SystemExit, match="DISAGREE ON SHAPE"):
        ra.load_model(str(ck), device="cpu")


def test_unfitted_standardizer_and_nonstrict_load_are_refused(tmp_path):
    cfg = _tiny_cfg()
    m = RefAV1(cfg)                                   # std NOT fitted
    p = tmp_path / "ckpt.pt"
    torch.save({"step": 1, "model": m.state_dict(), "cfg": vars(cfg)}, p)
    with pytest.raises(SystemExit, match="not fitted"):
        ra.load_model(str(p), device="cpu")
    sd = m.state_dict()
    sd.pop("lat_head.1.bias")
    torch.save({"step": 1, "model": sd, "cfg": vars(cfg)}, p)
    with pytest.raises(SystemExit, match="NON-STRICT"):
        ra.load_model(str(p), device="cpu")


def test_tier_table_matches_the_doctrine():
    assert ra.ARM_TIERS["ol"] == "T0"
    assert ra.ARM_TIERS["cl"] == ra.ARM_TIERS["ha"] == ra.ARM_TIERS["cl_navshuf"] == "T1"
    assert ra.ARM_TIERS["cl_oraclegoal"] == "T0"       # a goal from the future
    assert ra.DT == 0.2 and ra.K_TRAJ_DEFAULT * ra.DT == 2.0
    for k in ra.ARM_TIERS:
        assert k in ra.ARM_MEANING


def test_horizon_beyond_the_plan_is_refused(tmp_path):
    ck = _ckpt(tmp_path)
    cache, eps, lp = _fixture(tmp_path)
    a = _args(tmp_path, ck, cache, eps, lp, horizon_k=11)
    with pytest.raises(SystemExit, match="exceeds the plan horizon"):
        ra.run_dump(a)
