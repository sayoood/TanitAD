"""Tests for scripts/compare_arms.py — the Phase-0 three-arm gate harness.

CPU only, no real data / no real checkpoints. Builds tiny real-FORMAT
checkpoints for all three arms (flagship {model,grounding,step}, REF-A
{model,step,step_readout}, REF-B {model,step}) on matched toy val artifacts
(raw frames + synthetic DINO features from the SAME episodes/poses/ids), then
runs the whole comparison and asserts:
  - the comparison table is well-formed for every arm;
  - the metric-identity contract holds (one shared eval grid / baselines / GT);
  - the same-episode fairness guard drops a pose-mismatched clip;
  - per-arch capability gating is correct (REF-B has no imagination/grounded
    rollout; flagship & REF-A do).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_arms as ca  # noqa: E402

from tanitad.config import (PredictorConfig, flagship4b_smoke_config)  # noqa: E402
from tanitad.data.mixing import save_episode  # noqa: E402
from tanitad.data.toy_driving import generate_episode  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.models.metric_dynamics import (HierarchicalGrounding,  # noqa: E402
                                            StepDisplacementReadout)
from tanitad.refs.refa import RefAModel  # noqa: E402
from tanitad.refs.refb import RefBModel, refb_smoke_config  # noqa: E402

DEV = "cpu"
N_TOK, D_DINO = 16, 32          # tiny DINO grid (real is 256/768)
VAL_IDS = [100, 101, 102]
T = 64


def _refa_smoke_pred():
    return PredictorConfig(d_model=64, depth=2, n_heads=2, window=4,
                           horizons=(1, 2, 4), action_dim=2)


def _build_val(tmp: Path):
    """Matched raw-frame + DINO-feature val caches from the SAME episodes."""
    raw = tmp / "raw" / "toy-val"
    feat = tmp / "feat" / "toy-val-dinov2-b14"
    raw.mkdir(parents=True)
    feat.mkdir(parents=True)
    torch.manual_seed(0)
    for i in VAL_IDS:
        ep = generate_episode(episode_id=i, steps=T, size=64)
        save_episode(ep, str(raw / f"ep_{i:05d}.pt"))
        g = torch.Generator().manual_seed(i)
        torch.save({"feats_fp16": torch.randn(T, N_TOK, D_DINO, generator=g).half(),
                    "actions": ep.actions, "poses": ep.poses[:, :3],
                    "episode_id": i}, str(feat / f"ep_{i:05d}.pt"))
    return tmp / "raw", tmp / "feat"


def _save_flagship(tmp: Path) -> str:
    cfg = flagship4b_smoke_config()
    world = WorldModel(cfg)
    gr = HierarchicalGrounding(world.state_dim)
    p = tmp / "ck_flagship" / "ckpt.pt"
    p.parent.mkdir(parents=True)
    torch.save({"model": world.state_dict(), "grounding": gr.state_dict(),
                "step": 7}, str(p))
    return str(p)


def _save_refa(tmp: Path) -> str:
    model = RefAModel(pred_cfg=_refa_smoke_pred(), adapter_kind="grid",
                      n_tokens=N_TOK, d_dino=D_DINO)
    model.standardizer.fit([torch.randn(64, D_DINO)])
    sr = StepDisplacementReadout(model.state_dim)
    p = tmp / "ck_refa" / "ckpt.pt"
    p.parent.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 5,
                "step_readout": sr.state_dict()}, str(p))
    return str(p)


def _save_refb(tmp: Path) -> str:
    model = RefBModel(refb_smoke_config())
    p = tmp / "ck_refb" / "ckpt.pt"
    p.parent.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 5}, str(p))
    return str(p)


def _run(tmp: Path, behavior_epochs: int = 0):
    raw_root, feat_root = _build_val(tmp)
    arms = [
        ca.build_flagship(_save_flagship(tmp), "smoke", DEV),
        ca.build_refa(_save_refa(tmp), "grid", True, DEV, n_tokens=N_TOK,
                      d_dino=D_DINO),
        ca.build_refb(_save_refb(tmp), True, DEV),
    ]
    frame_val = ca.load_frame_val([str(raw_root)], 8)
    feat_eps = ca.load_feature_val(str(feat_root), 8)
    common, feat_by_id, ids = ca.load_common_val(frame_val, feat_eps, True, 1e-2)
    report = ca.compare(arms, common, feat_by_id, DEV, n_splits=3, val_frac=0.2,
                        seed=0, mlp_epochs=6, batch=8, stride=8, git_hash="test",
                        oracle_target=1.65, behavior_epochs=behavior_epochs)
    return report


# --------------------------------------------------------------------------- #
# end-to-end well-formedness                                                   #
# --------------------------------------------------------------------------- #
def test_three_arm_compare_wellformed(tmp_path):
    r = _run(tmp_path)
    assert set(r["arms"]) == {"flagship", "refa", "refb"}
    # same-episode guarantee: exactly the val ids, shared across arms.
    assert r["val"]["common_episode_ids"] == VAL_IDS
    assert r["val"]["n_windows"] > 0
    # shared model-free baselines present and finite.
    import math
    for n in ("constant_velocity", "go_straight", "constant_yaw_rate"):
        assert math.isfinite(r["baselines"][n]["ade_0_2s"])
    # every arm has a finite D1 parity number + a valid gate status.
    for a, ar in r["arms"].items():
        assert math.isfinite(ar["decode"]["d1_ade_0_2s"]), a
        assert ar["decode"]["d1_status"] in {"PASS", "FAIL", "BLOCKED"}, a
        assert math.isfinite(ar["decode"]["oracle_ceiling_ade_0_2s"]), a
        # oracle in-distribution ceiling must not be worse than held-out.
        assert (ar["decode"]["oracle_ceiling_ade_0_2s"]
                <= ar["decode"]["best_heldout_ade_0_2s"] + 1e-6), a


def test_per_arch_capability_gating(tmp_path):
    r = _run(tmp_path)
    # flagship & REF-A own an action-conditioned predictor -> D2/D3 present.
    for a in ("flagship", "refa"):
        assert r["arms"][a]["imagination"] is not None, a
        assert r["arms"][a]["imagination"]["d2_status"] in {"PASS", "FAIL", "BLOCKED"}
        assert r["arms"][a]["grounded"] is not None, a
    # REF-B is the pre-registered no-world-model reference.
    assert r["arms"]["refb"]["imagination"] is None
    # REF-B still exposes a native trajectory (its BC waypoint head).
    assert r["arms"]["refb"]["grounded"] is not None
    assert "waypoint head" in r["arms"]["refb"]["grounded"]["mechanism"]


def test_verdict_resolves(tmp_path):
    r = _run(tmp_path)
    v = r["verdict"]
    pm = v["per_metric"]
    # D1 parity winner is one of the arms (lowest decode ADE).
    assert pm["d1_decode_ade_0_2s"]["winner_lowest"] in r["arms"]
    # hierarchy-edge necessary-condition block is computed for the flagship.
    edge = v["hierarchy_edge_necessary_conditions"]
    assert edge is not None
    assert isinstance(edge["flagship_beats_refs_on_d1_decode"], bool)
    assert "closed-loop" in v["DOCTRINE"].lower()


def test_markdown_renders(tmp_path):
    r = _run(tmp_path)
    md = ca.render_markdown(r)
    assert "Phase-0 three-arm comparison" in md
    assert "Trivial baselines" in md
    for a in ("flagship", "refa", "refb"):
        assert a in md


# --------------------------------------------------------------------------- #
# fairness guard + metric identity (unit)                                      #
# --------------------------------------------------------------------------- #
def test_same_episode_guard_drops_pose_mismatch(tmp_path):
    """A feature episode whose poses do not match the frame clip is dropped —
    the mechanical same-episode fairness guarantee."""
    raw_root, feat_root = _build_val(tmp_path)
    frame_val = ca.load_frame_val([str(raw_root)], 8)
    feat_eps = ca.load_feature_val(str(feat_root), 8)
    # corrupt ONE feature episode's poses (make it a different clip).
    feat_eps[1]["poses"] = feat_eps[1]["poses"] + 5.0
    common, feat_by_id, ids = ca.load_common_val(frame_val, feat_eps, True, 1e-2)
    assert feat_eps[1]["episode_id"] not in ids
    assert len(ids) == len(VAL_IDS) - 1


def test_reference_grid_is_shared_and_deterministic(tmp_path):
    """The GT + baselines are built ONCE from poses and are identical across
    calls — the identity backbone every arm decodes against."""
    raw_root, _ = _build_val(tmp_path)
    frame_val = ca.load_frame_val([str(raw_root)], 8)
    g1 = ca.build_reference_grid(frame_val, window=4, stride=8)
    g2 = ca.build_reference_grid(frame_val, window=4, stride=8)
    assert torch.allclose(g1.gt, g2.gt)
    assert g1.eid == g2.eid
    for n in ca.dd.BASELINES:
        assert torch.allclose(g1.base[n], g2.base[n])
    # GT waypoints are [N, 4, 2] at the 4 waypoint steps.
    assert g1.gt.shape[1:] == (4, 2)


def test_no_feature_arm_keeps_all_frame_episodes(tmp_path):
    """When no REF-A arm is present, the harness keeps every frame val episode
    (no feature intersection needed)."""
    raw_root, _ = _build_val(tmp_path)
    frame_val = ca.load_frame_val([str(raw_root)], 8)
    common, feat_by_id, ids = ca.load_common_val(frame_val, [], False, 1e-2)
    assert ids == VAL_IDS
    assert feat_by_id == {}


# --------------------------------------------------------------------------- #
# TanitResim integration: the SAME checkpoint gated via the compare_arms       #
# builder AND via the resim MainArm -> ArmSpec adapter must reconcile.         #
# --------------------------------------------------------------------------- #
def test_resim_arm_reconciles_with_builder(tmp_path):
    """replay_app.py gates a checkpoint through armspec_from_resim_arm; a
    compare_arms.py run builds it via build_flagship. Same weights + same
    episodes => byte-identical D1 (and grounded) — the DRY-shared gate code."""
    from tanitad.replay.arms import MainArm
    raw_root, _ = _build_val(tmp_path)
    ckpt = _save_flagship(tmp_path)
    frame_val = ca.load_frame_val([str(raw_root)], 8)
    kw = dict(n_splits=3, val_frac=0.2, seed=0, mlp_epochs=6, batch=8,
              stride=8, git_hash="t", oracle_target=1.65)

    arm_builder = ca.build_flagship(ckpt, "smoke", DEV)
    rep_a = ca.compare([arm_builder], frame_val, {}, DEV, **kw)

    main = MainArm(ckpt, cfg=flagship4b_smoke_config(), device=DEV)
    arm_resim = ca.armspec_from_resim_arm(main, DEV)
    rep_b = ca.compare([arm_resim], frame_val, {}, DEV, **kw)

    da = rep_a["arms"]["flagship"]["decode"]["d1_ade_0_2s"]
    db = rep_b["arms"]["main"]["decode"]["d1_ade_0_2s"]
    assert abs(da - db) < 1e-4, f"D1 reconcile failed: {da} vs {db}"
    # grounded rollout also reconciles (both load grounding from the same ckpt)
    ga = rep_a["arms"]["flagship"]["grounded"]["ade_0_2s"]
    gb = rep_b["arms"]["main"]["grounded"]["ade_0_2s"]
    assert abs(ga - gb) < 1e-3, f"grounded reconcile failed: {ga} vs {gb}"


def test_compact_gate_blocks_shape(tmp_path):
    """compact_gate_blocks (fed to stats.json + the UI) has the per-arm gate
    fields the resim panel + GO banner read."""
    r = _run(tmp_path)
    blocks = ca.compact_gate_blocks(r)
    assert set(blocks) >= {"arms", "baselines", "verdict", "n_val_episodes",
                           "n_windows"}
    for name in ("flagship", "refa", "refb"):
        b = blocks["arms"][name]
        assert b["D1"] in {"PASS", "FAIL", "BLOCKED"}
        assert "d1_ade_0_2s" in b and "oracle_ceiling_ade_0_2s" in b
    assert set(blocks["baselines"]) == {"constant_velocity", "go_straight",
                                        "constant_yaw_rate"}


# --------------------------------------------------------------------------- #
# Behavior gate (item 1): arm-agnostic maneuver/route decodability, wired into #
# the unified suite and reconciling with eval_behavior.py.                     #
# --------------------------------------------------------------------------- #
def test_behavior_probe_reconciles_with_eval_behavior(tmp_path):
    """compare_arms._behavior_probe REPLICATES eval_behavior.maneuver_probe_eval
    (encoder_state / _all / linear) byte-for-byte on the same data — so the
    behavior block in the unified suite reconciles with eval_behavior.py."""
    import math
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_behavior as eb
    from tanitad.config import flagship4b_smoke_config
    torch.manual_seed(0)
    world = WorldModel(flagship4b_smoke_config()).eval()
    window = world.predictor.cfg.window
    eps = [generate_episode(i, steps=110, size=64) for i in range(4)]
    corpora = ["c", "c", "p", "p"]
    from tanitad.instruments.numerics import strict_numerics
    with strict_numerics():
        data = eb.collect(world, eps, corpora, DEV, window, math.radians(45.0),
                          math.radians(20.0), stride=6, batch=4, keep_states=True)
        seeds, vf, ep = [0, 1, 2], 0.5, 10
        mine = ca._behavior_probe(data["encoder_state"], data["man"],
                                  data["eid_global"], eb.N_MAN,
                                  eb.MANEUVER_CLASSES, eb.LANE_KEEP, seeds, vf,
                                  ep, DEV)
        theirs = eb.maneuver_probe_eval(data, seeds, vf, DEV, ep)
    cell = theirs["encoder_state"]["_all"]["linear"]
    assert abs(mine["balanced_accuracy"]
               - cell["seed_mean_std"]["balanced_accuracy"][0]) < 1e-4


def test_behavior_flows_into_suite_and_compact_blocks(tmp_path):
    """With behavior_epochs>0 every arm gets a behavior block, and the compact
    block (stats.json / UI) carries maneuver + route bal-acc."""
    r = _run(tmp_path, behavior_epochs=8)
    for name in ("flagship", "refa", "refb"):
        bh = r["arms"][name]["behavior"]
        assert bh is not None
        assert "maneuver_decode" in bh and "route_decode" in bh
    blocks = ca.compact_gate_blocks(r)
    for name in ("flagship", "refa", "refb"):
        assert "maneuver_balacc" in blocks["arms"][name]
        assert "route_balacc" in blocks["arms"][name]


def test_behavior_off_by_default(tmp_path):
    """behavior_epochs=0 (compare default) keeps the block absent — protects the
    other tests' runtime and the decode-only fast path."""
    r = _run(tmp_path)
    assert r["arms"]["flagship"]["behavior"] is None
