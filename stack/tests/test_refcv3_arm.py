"""``taniteval/tools/refcv3_arm.py`` — the refcv3 eval adapter, on a random-init
``RefCV3Model`` at the smallest instantiable config over a SYNTHETIC 3-episode
v2 slice. CPU only; ⛔ never touches Thor, a pod, or a real checkpoint.

WHAT IS PINNED (each line is a failure this suite exists to prevent)

  (1) THE DUMP IS THE ``t1_eval`` CONTRACT and the UNTOUCHED ``t1_eval.analyze``
      reads it; every family is PRESENT (status OK) or REFUSED/UNAVAILABLE WITH A
      REASON AND AN ``n`` — never silently dropped.
  (2) ⭐ ``ha0`` IS STRAIGHT AND CONSTANT-SPEED ON EVERY WINDOW, and the
      trivial-profile instrument reads ``trivial_frac == 1.0`` for it. That
      instrument prints BEFORE any family row; on 2026-09-03 a 2.5 h refav1
      rollout was found VOID because nobody had asked what SHAPE the arm was.
  (3) TWO ARMS THAT ARE EQUAL ARE REPORTED AS EQUAL — the deliberate-regression
      arm: an ``os`` copied from ``ha0`` must come back flagged as degenerate and
      bit-identical, i.e. the read is VOID, not "no difference".
  (4) THE NAV-SHUFFLE ARM DIFFERS FROM THE TRUE-NAV ARM **ONLY** IN NAV — same
      frames, same v0, same grid, same GT; where the token did not change the two
      arms are bit-identical, and where it did they may differ.
  (5) ⛔ THE ARM IS ``os``, NEVER ``cl``; ``ol`` is ABSENT WITH ITS REASON.
      A shared arm name is how two different procedures end up in one table
      (``D-HF-COMPARABILITY``).
  (6) ⛔ THE DEPLOYED SELECTION IS THE MODEL'S OWN (``out['traj']`` /
      ``sel_score_v3``), NOT ``a_star``: wherever ``sel_idx != a_star`` the ``os``
      and ``oracle_sel`` paths differ.
  (7) THE ACTION-UNIT CONVERSION IS APPLIED EXACTLY ONCE: ``ha`` under
      ``--action-units steer`` equals the unicycle driven with
      ``kappa = tan(steer)/L_enc``, and differs from the legacy ``kappa`` reading
      whenever the held channel is non-zero.
  (8) NO FUTURE ENTERS: perturbing every pose/action AFTER the last window's
      origin leaves ``os``, ``ha`` and ``ha0`` BIT-IDENTICAL while ``g`` (the GT
      waypoints) changes — so the perturbation provably sat in the future.
  (9) The tier stamps travel, and the OPEN RULING is carried on every block.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

tvio = pytest.importorskip("torchvision.io")

_REPO = Path(__file__).resolve().parents[2]
TOOL = _REPO / "taniteval" / "tools" / "refcv3_arm.py"
sys.path.insert(0, str(_REPO / "stack" / "scripts"))

_spec = importlib.util.spec_from_file_location("refcv3_arm_under_test", TOOL)
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

from tanitad.models.kinematic import STEER_WHEELBASE_M  # noqa: E402
from tanitad.refs import refc  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402

N_STACK, SIZE, N_RAW = 3, 64, 40          # -> 9 channels, T_out = 38
CLIPS = ("clip-aaaaaaa0", "clip-bbbbbbb1", "clip-ccccccc2")
#: ep0 turns left, ep1 follows, ep2 turns right — so the nav permutation has
#: something to change (a FOLLOW-only marginal makes the control powerless).
NAV = ("NAV_TURN_L", "NAV_FOLLOW_ROAD", "NAV_TURN_R")
#: v7 tactical vocabulary (``vocab_v7.TACTICAL_{LAT,LON}_ACTIONS_V7``, 8 x 8).
TAC = (("NUDGE_L", "BRAKE_TO"), ("LANE_KEEP", "CRUISE"), ("TURN_R", "ACCELERATE"))


# --------------------------------------------------------------------------- #
# the synthetic corpus                                                         #
# --------------------------------------------------------------------------- #
def _synth_poses(i: int, T: int = N_RAW):
    """A real unicycle trajectory at 10 Hz, integrated in ``rollout_unicycle``'s
    ORDER (the pose advances on the START-of-step speed, v updates last).
    ``actions[:, 0]`` is written as a road-wheel STEER ANGLE — the corpus's own
    convention (``physicalai.py:621`` writes ``arctan(L_enc * curvature)``)."""
    t = torch.arange(T, dtype=torch.float32)
    v = 6.0 + 1.5 * i + 0.8 * torch.sin(2 * math.pi * t / 30.0)
    kap = 0.03 * torch.sin(2 * math.pi * t / 40.0 + i) + 0.01 * (i + 1)
    x, y, yaw = 0.0, 0.0, 0.0
    poses = torch.zeros(T, 4)
    for f in range(T):
        poses[f] = torch.tensor([x, y, yaw, float(v[f])])
        x += float(v[f]) * math.cos(yaw) * 0.1
        y += float(v[f]) * math.sin(yaw) * 0.1
        yaw += float(v[f]) * float(kap[f]) * 0.1
    acts = torch.zeros(T, 2)
    acts[:, 0] = torch.atan(STEER_WHEELBASE_M * kap)   # ⭐ STEER, not kappa
    acts[1:, 1] = (v[1:] - v[:-1]) / 0.1
    return poses, acts


def _write_v2ep(path: Path, i: int, *, perturb_after: int | None = None):
    """One clip in the exact ``build_compressed`` payload format
    (``test_v2_dataset._write_v2ep``'s shape: JPEG frames + f32 poses/actions)."""
    g = torch.Generator().manual_seed(100 + i)
    vid = (torch.rand(N_RAW, 3, SIZE, SIZE, generator=g) * 255).to(torch.uint8)
    jpegs = [tvio.encode_jpeg(vid[k].contiguous(), quality=90)
             for k in range(N_RAW)]
    poses, actions = _synth_poses(i)
    if perturb_after is not None:
        sl = slice(perturb_after, None)
        poses[sl, 0] += 7.0
        poses[sl, 1] += 5.0
        poses[sl, 3] += 2.0
        actions[sl, 0] += 0.2
    torch.save({"jpeg_buf": torch.cat(jpegs),
                "jpeg_len": torch.tensor([int(j.numel()) for j in jpegs],
                                         dtype=torch.int64),
                "actions": actions, "poses": poses, "n_stack": N_STACK,
                "image_size": SIZE, "episode_id": i,
                "clip_id": CLIPS[i], "quality": 90}, str(path))


def _record(i: int):
    lat, lon = TAC[i]
    return {"schema_version": "s2-geom-v7", "vocab": "v7", "clip_id": CLIPS[i],
            "t0_s": 0.0,
            "bands": {"operative_s": [0.0, 2.0], "tactical_s": [0.0, 6.0],
                      "strategic_s": [8.0, 30.0]},
            "a_tac": {"lat": lat, "lon": lon},
            "nav_command": {"token": NAV[i], "provenance": "ego-future",
                            "oracle": True,
                            "args": {"distance_m": 10.0, "time_s": 2.0}}}


def _tiny_cfg() -> v3.RefCV3Config:
    """The SMALLEST instantiable v3: the smoke config with the encoder widened to
    the corpus's 9 channels (3-frame stack) at 64 px — the geometry the adapter
    asserts against the episodes."""
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = refc.CNNEncoderConfig(in_channels=3 * N_STACK,
                                             image_size=SIZE, base_width=8,
                                             blocks=(1, 1, 1, 1))
    return cfg


def _fixture(root: Path, *, perturb_after: int | None = None):
    eps = root / "eps"
    eps.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        _write_v2ep(eps / f"{CLIPS[i]}.v2ep.pt", i, perturb_after=perturb_after)
    lp = root / "labels_v72_eval.jsonl"
    lp.write_text("\n".join(json.dumps(_record(i)) for i in range(3)),
                  encoding="utf-8")
    run = root / "run"
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = v3.RefCV3Model(cfg)
    torch.save({"step": 11, "model": model.state_dict(), "opt": {}},
               run / "ckpt.pt")
    # ⭐ the adapter's documented escape hatch: refc_v3_train writes NO model
    # config, so a config.json may carry an explicit RefCV3Config dict. This also
    # exercises `cfg_from_dict` (tuples survive JSON, unknown fields refused).
    (run / "config.json").write_text(json.dumps({
        "arm": "hier",
        "horizons": list(cfg.core.trajectory.horizons),
        "goal_tau_steps": list(cfg.goal_tau_steps),
        "image_hw": list(cfg.core.encoder.image_hw()),
        "tac_vocab_version": cfg.tac_vocab_version,
        "nav_from_v7": True,
        "nav_cmd_derivation": "v7.2 nav_command token (test fixture)",
        "refcv3_arm_model_cfg": json.loads(
            json.dumps(rc.dataclasses.asdict(cfg))),
    }, indent=1), encoding="utf-8")
    return eps, lp, run / "ckpt.pt"


def _args(root: Path, ck: Path, eps: Path, lp: Path, **over):
    base = dict(ckpt=str(ck), config=None, episodes=str(eps), labels=str(lp),
                nav_source="v72", grid="2s", device="cpu", episodes_n=0,
                window_stride=1, lru=4, action_units="steer",
                nav_shuffle_seed=0, no_navshuf=False, with_navzero=False,
                with_oracle_sel=True, allow_nonstrict=False,
                dump_dir=str(root / "dump"))
    base.update(over)
    return argparse.Namespace(**base)


def _load_dump(dump: Path) -> dict:
    out = {}
    for f in sorted(dump.glob("ep*.npz")):
        with np.load(f) as d:
            out[f.stem] = {k: d[k] for k in d.files}
    return out


def _cat(dump: Path, key: str) -> np.ndarray:
    return np.concatenate([v[key] for v in _load_dump(dump).values()])


def _cat_dec(dump: Path, key: str) -> np.ndarray:
    parts = []
    for f in sorted((dump / "decisions").glob("ep*.npz")):
        with np.load(f) as d:
            parts.append(d[key])
    return np.concatenate(parts)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    root = tmp_path_factory.mktemp("refcv3_e2e")
    eps, lp, ck = _fixture(root)
    a = _args(root, ck, eps, lp)
    manifest = rc.run_dump(a)
    rec = rc.analyze_refcv3(a.dump_dir, n_boot=60, seed=0)
    return root, a, manifest, rec


# =========================================================================== #
# (1) the dump IS the t1_eval contract, and analyze() reads it                 #
# =========================================================================== #
def test_dump_is_the_t1_contract_and_analyze_reads_it(e2e):
    root, a, manifest, rec = e2e
    dump = Path(a.dump_dir)
    files = sorted(dump.glob("ep*.npz"))
    assert len(files) == 3 and (dump / "manifest.json").exists()
    assert len(sorted((dump / "decisions").glob("ep*.npz"))) == 3
    d = _load_dump(dump)["ep000"]
    N = d["g"].shape[0]
    assert N > 0
    assert d["g"].shape == (N, 4, 2) and d["g"].dtype == np.float32
    for arm in ("os", "ha", "ha0", "os_navshuf", "oracle_sel"):
        assert d[arm].shape == (N, 4, 2), arm
    # ⛔ the dump's key space IS the arm space: metadata must not be an arm
    assert d["v0"].shape == (N,) and d["eid"].tolist() == [0]
    assert set(rec["arms"]) == {"os", "ha", "ha0", "os_navshuf", "oracle_sel"}
    assert rec["tiers"] == {"os": "T1", "ha": "T1", "ha0": "T1",
                            "os_navshuf": "T1", "oracle_sel": "T0"}
    assert rec["n_windows"] == manifest["grid"]["n_windows"]
    assert manifest["grid"]["instants_s"] == [0.5, 1.0, 1.5, 2.0]
    assert manifest["grid"]["slots"] == [0, 1, 2, 3]


def test_every_family_is_present_or_refused_with_a_reason(e2e):
    _, _, _, rec = e2e
    #: the family blocks that carry values do NOT set a `status` key (only the
    #: refusal shape does), so PRESENT is "no status, and the family's own
    #: headline metric is finite".
    HEADLINE = {"longitudinal": "speed_mae_mps", "lateral": "cross_mae_m",
                "tactical": None, "strategic": None}
    for arm, blk in rec["arms"].items():
        fam = blk["four_families"]
        for k in ("longitudinal", "lateral", "tactical", "strategic"):
            f = fam[k]
            assert f.get("tier") == blk["tier"], (arm, k)
            st = f.get("status")
            assert st in (None, "OK", "UNAVAILABLE", "REFUSED"), (arm, k, st)
            if st in ("UNAVAILABLE", "REFUSED"):
                assert f.get("reason"), (arm, k)
                assert f.get("n") is not None, (arm, k)
            else:                                   # PRESENT
                assert f.get("n") is not None, (arm, k)
                if HEADLINE[k]:
                    assert np.isfinite(f[HEADLINE[k]]), (arm, k)
        # ⛔ the trajectory-only STRATEGIC row is UNAVAILABLE BY DESIGN — a route
        # class cannot be read off a 2 s path; the real one is in rec['refcv3'].
        assert fam["strategic"]["status"] == "UNAVAILABLE"
        assert fam["tactical"]["status"] == "OK"
        assert fam["_ci_coverage"]["longitudinal"]["complete"] is True
        assert fam["_ci_coverage"]["lateral"]["complete"] is True
        assert "ade_dense_m" in blk["intervals"]["metrics"]
    ref = rec["refcv3"]
    # the adapter's own blocks obey the same contract
    for key in ("distance_keeping", "law_diagnostic"):
        b = ref[key]
        if b.get("status") in ("REFUSED", "UNAVAILABLE"):
            assert b.get("reason") and b.get("n") is not None, key
    for cname, b in ref["strategic"]["conditionings"].items():
        if b.get("status") in ("REFUSED", "UNAVAILABLE"):
            assert b.get("reason") and b.get("n") is not None, cname
    # ⛔ no lead block on this fixture -> distance-keeping is a WORK ITEM
    assert ref["distance_keeping"]["status"] == "REFUSED"
    assert "WORK ITEM" in ref["distance_keeping"]["reason"]
    # `law` can never enter a T1 row and is refused BY NAME, not dropped
    assert ref["law_diagnostic"]["tier"] == "T0"
    assert "future frame" in ref["law_diagnostic"]["reason"]


def test_paired_blocks_are_the_margin_over_the_shared_floor(e2e):
    _, _, _, rec = e2e
    fp = rec["refcv3"]["families_paired"]
    assert "paired_os_minus_ha0" in fp, "the echo test's REAL bar is os - ha0"
    assert "paired_os_minus_ha" in fp
    assert "paired_os_minus_navshuf" in fp
    assert rec["refcv3"]["headline"]["floor_arm"] == "ha0"
    blk = fp["paired_os_minus_ha0"]
    assert blk["estimator"] == "paired_episode_cluster_bootstrap"
    assert blk["direction"] == "os - ha0"
    for famname in ("ADE", "longitudinal", "lateral", "tactical"):
        assert famname in blk["families"], famname
        for mk, r in blk["families"][famname].items():
            assert ("delta" in r and "lo" in r and "hi" in r) or \
                   r.get("status") in ("REFUSED", "UNAVAILABLE"), (famname, mk)


# =========================================================================== #
# (2) ⭐ ha0 is the constant-velocity floor, and the instrument SEES it        #
# =========================================================================== #
def test_ha0_is_straight_and_constant_speed_on_every_window(e2e):
    _, a, _, _ = e2e
    dump = Path(a.dump_dir)
    P = _cat(dump, "ha0").astype(np.float64)
    v0 = _cat(dump, "v0").astype(np.float64)
    n = P.shape[0]
    assert n > 0
    # straight: no lateral displacement at all
    assert np.abs(P[..., 1]).max() < 1e-6
    # constant speed: the chord length per grid step is v0 * dt on every step
    steps = np.linalg.norm(
        np.diff(np.concatenate([np.zeros((n, 1, 2)), P], axis=1), axis=1),
        axis=-1)
    assert np.ptp(steps, axis=1).max() < 1e-4
    assert np.abs(steps - (v0[:, None] * 0.5)).max() < 1e-3
    # and it is EXACTLY zero in either action unit -> bit-comparable with refav1
    assert np.array_equal(
        rc.hold_v0_controls(20).numpy(),
        np.zeros((20, 2), dtype=np.float32))


def test_trivial_profile_reads_one_for_ha0_and_prints_first(e2e, capsys):
    _, a, _, rec = e2e
    triv = rec["refcv3"]["trivial_profile"]
    assert triv["arms"]["ha0"]["straight_frac"] == 1.0
    assert triv["arms"]["ha0"]["const_speed_frac"] == 1.0
    assert triv["arms"]["ha0"]["trivial_frac"] == 1.0
    assert "ha0" in triv["degenerate_arms"]
    # ⚠️ ...but `ha0` must NOT raise VOID-RISK: it IS the constant-velocity plan
    # by definition, and a warning that fires on every correct run is learned as
    # noise. Only a degeneracy that is a FINDING escalates.
    assert triv["degenerate_arms_excluding_floor"] == []
    # the instrument is banked FIRST in the record, and the roll's own arms are
    # NOT the constant-velocity plan (a random-init model still bends)
    assert triv["arms"]["os"]["trivial_frac"] < 1.0
    # it PRINTS before any family row
    out = []
    rc.ra._print_trivial_profile(triv)
    out = capsys.readouterr().out
    assert "trivial-profile" in out and "CONSTANT-VELOCITY" in out


# =========================================================================== #
# (3) two arms that are EQUAL are reported as EQUAL — the regression arm       #
# =========================================================================== #
def test_selection_profile_sees_what_the_trivial_profile_cannot(e2e, capsys):
    """⛔ THE REFCV3-SPECIFIC DEGENERACY. A one-shot anchor model can pick the
    SAME anchor on every window while its paths bend and change speed — so
    ``trivial_frac`` reads 0.0000 and the trivial profile is blind to it. On this
    random-init fixture the selection IS a constant, and the instrument says so
    BEFORE any family row."""
    _, _, _, rec = e2e
    sp = rec["refcv3"]["selection_profile"]
    triv = rec["refcv3"]["trivial_profile"]
    assert triv["arms"]["os"]["trivial_frac"] == 0.0        # invisible there
    assert sp["n_distinct_selected"] == 1                   # …and visible here
    assert sp["modal_frac"] == 1.0
    assert sp["entropy_nats"] == 0.0 and sp["degenerate"] is True
    rc._print_selection_profile(sp)
    out = capsys.readouterr().out
    assert "selection-profile" in out and "VOID-RISK" in out
    assert "CONSTANT" in out


def test_deliberately_identical_arms_are_reported_identical_and_degenerate(e2e,
                                                                           tmp_path):
    """⛔ THE DELIBERATE REGRESSION. A read whose arm is bit-identical to another
    arm is VOID, not "no difference" (``D-REFAV1-PAIRED-READ-VOID``). Copy `ha0`
    over `os` and the instrument must say so BEFORE any family row."""
    _, a, _, _ = e2e
    src, dst = Path(a.dump_dir), tmp_path / "void_dump"
    (dst / "decisions").mkdir(parents=True)
    for f in sorted(src.glob("ep*.npz")):
        with np.load(f) as d:
            arr = {k: d[k] for k in d.files}
        arr["os"] = arr["ha0"].copy()             # the regression
        arr["os_navshuf"] = arr["ha0"].copy()
        np.savez_compressed(dst / f.name, **arr)
    for f in sorted((src / "decisions").glob("ep*.npz")):
        with np.load(f) as d:
            np.savez_compressed(dst / "decisions" / f.name,
                                **{k: d[k] for k in d.files})
    (dst / "manifest.json").write_text(
        (src / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    rec = rc.analyze_refcv3(str(dst), n_boot=40, seed=0)
    triv = rec["refcv3"]["trivial_profile"]
    n = triv["arms"]["os"]["n"]
    assert triv["arms"]["os"]["identical_to"]["ha0"]["n"] == n
    assert triv["arms"]["os"]["identical_to"]["ha0"]["frac"] == 1.0
    assert triv["arms"]["os"]["trivial_frac"] == 1.0
    assert {"os", "ha0", "os_navshuf"} <= set(triv["degenerate_arms"])
    # ⛔ and THIS one escalates: the degeneracy is a finding, not the floor
    assert set(triv["degenerate_arms_excluding_floor"]) == {"os", "os_navshuf"}
    # and the paired margin over the floor is exactly zero on every family
    fam = rec["refcv3"]["families_paired"]["paired_os_minus_ha0"]["families"]
    assert abs(fam["ADE"]["ade_m"]["delta"]) < 1e-12


# =========================================================================== #
# (4) the nav-shuffle arm differs ONLY in nav                                  #
# =========================================================================== #
def test_navshuf_differs_from_true_nav_only_in_nav(e2e):
    _, a, _, rec = e2e
    dump = Path(a.dump_dir)
    nav = _cat_dec(dump, "nav_cmd").astype(int)
    shuf = _cat_dec(dump, "nav_cmd_shuf").astype(int)
    os_ = _cat(dump, "os").astype(np.float64)
    ns_ = _cat(dump, "os_navshuf").astype(np.float64)
    same_tok = nav == shuf
    assert same_tok.any(), "the fixture must keep some tokens"
    assert (~same_tok).any(), "the fixture must CHANGE some tokens"
    # where the token did NOT change the two forwards are the SAME forward
    d_same = np.abs(os_[same_tok] - ns_[same_tok]).max()
    assert d_same < 1e-9, f"same nav token but the paths differ by {d_same}"
    # everything else the two arms consumed is identical by construction: they
    # are two ROWS of ONE batched forward on the same frames and the same v0
    assert rec["refcv3"]["n_windows"] == len(nav)
    # the nav-shuffle stats are recorded, and the control has power
    st = rec["refcv3"]["strategic"]["nav_shuffle"]
    assert st["n_changed"] == int((~same_tok).sum())
    assert st["n_changed"] > 0


# =========================================================================== #
# (5) ⛔ the arm is `os`, never `cl`; `ol` is ABSENT with its reason           #
# =========================================================================== #
def test_arm_is_named_os_and_ol_is_absent_with_a_reason(e2e):
    _, a, manifest, rec = e2e
    d = _load_dump(Path(a.dump_dir))["ep000"]
    assert "cl" not in d and "cl" not in rec["arms"]
    assert "ol" not in d and "ol" not in rec["arms"]
    for blk in (manifest["absent_arms"], rec["refcv3"]["absent_arms"]):
        ol = blk["ol"]
        assert ol["status"] == "ABSENT"
        assert "CONSUMES NO ACTIONS" in ol["reason"]
        assert "ha0" in ol["for_comparison"]
    # the T1 definition names why roll_closed cannot be ported
    td = manifest["t1_definition"]
    assert "NO action argument" in td["no_closed_loop_because"]
    assert "roll_closed" in td["no_closed_loop_because"]
    assert "a_star" in td["selection"] and "sel_score_v3" in td["selection"]


def test_tier_ruling_is_carried_as_UNRULED(e2e):
    _, _, manifest, rec = e2e
    for tr in (manifest["tier_ruling"], rec["refcv3"]["tier_ruling"]):
        assert tr["arm"] == "os" and tr["stamped"] == "T1"
        assert tr["status"] == "UNRULED"
        assert "PI / Master Mind" in tr["decided_by"]
    assert rec["refcv3"]["tactical_declared"]["tier_ruling"] == "UNRULED"
    assert rec["refcv3"]["strategic"]["tier_ruling"] == "UNRULED"


# =========================================================================== #
# (6) ⛔ the deployed selection is the MODEL'S OWN, never a_star               #
# =========================================================================== #
def test_selection_is_the_models_own_not_the_oracle(e2e):
    _, a, _, rec = e2e
    dump = Path(a.dump_dir)
    sel = _cat_dec(dump, "sel_idx").astype(int)
    star = _cat_dec(dump, "a_star").astype(int)
    os_ = _cat(dump, "os").astype(np.float64)
    orc = _cat(dump, "oracle_sel").astype(np.float64)
    diff = sel != star
    assert diff.any(), ("the fixture must produce at least one window where the "
                        "model's own selection is NOT the GT-nearest anchor, or "
                        "this test cannot tell them apart")
    assert np.abs(os_[diff] - orc[diff]).max() > 1e-9
    # where they agree the two arms ARE the same path — the ceiling touches
    if (~diff).any():
        assert np.abs(os_[~diff] - orc[~diff]).max() < 1e-9
    sp = rec["refcv3"]["selection_profile"]
    assert sp["n_anchors"] == 20 and sp["n_windows"] == len(sel)
    assert sp["agrees_with_oracle_frac"] == pytest.approx(
        float((~diff).mean()), abs=1e-4)
    asel = rec["refcv3"]["tactical_declared"]["anchor_selection"]
    assert asel["chance"] == pytest.approx(1.0 / 20, rel=1e-6)   # smoke: 20 anchors
    assert 0.0 <= asel["anchor_acc"]["mean"] <= 1.0
    assert 0.0 <= asel["deployed_selection_agrees_oracle"]["mean"] <= 1.0
    assert asel["deployed_selection_agrees_oracle"]["mean"] == \
        pytest.approx(float((~diff).mean()), abs=1e-6)


# =========================================================================== #
# (7) the action-unit conversion is applied EXACTLY ONCE                       #
# =========================================================================== #
def test_action_units_conversion_is_applied_once_and_only_to_ha():
    """``kappa = tan(steer)/L_enc`` before integration — once. The two readings
    must DIFFER wherever the held channel is non-zero (this is the defect that
    cost refav1 0.716 m of curved-window lateral error)."""
    steer = 0.20                       # rad -> kappa = tan(.2)/2.9 = 0.07031
    hold = torch.tensor([0.5, steer], dtype=torch.float32)
    grid = rc.grid_slots(v3.V3_HORIZONS, "2s")
    ctl = hold[None].expand(grid["n_frames"], 2)
    as_steer = rc.integrate_select(ctl, 8.0, grid, action_units="steer")
    as_kappa = rc.integrate_select(ctl, 8.0, grid, action_units="kappa")
    assert np.abs(as_steer - as_kappa).max() > 0.05, \
        "the two unit readings must not coincide on a curved hold"
    # exactly once: the steer reading must equal a hold of the CONVERTED value
    # integrated under the kappa convention
    kap = math.tan(steer) / STEER_WHEELBASE_M
    ctl2 = torch.tensor([0.5, kap], dtype=torch.float32)[None].expand(
        grid["n_frames"], 2)
    ref = rc.integrate_select(ctl2, 8.0, grid, action_units="kappa")
    assert np.abs(as_steer - ref).max() < 1e-6
    # ⛔ a DOUBLE application would land here instead, and it does not
    kap2 = math.tan(kap) / STEER_WHEELBASE_M
    ctl3 = torch.tensor([0.5, kap2], dtype=torch.float32)[None].expand(
        grid["n_frames"], 2)
    twice = rc.integrate_select(ctl3, 8.0, grid, action_units="kappa")
    assert np.abs(as_steer - twice).max() > 1e-3


def test_manifest_states_the_unit_and_what_it_applies_to(e2e):
    _, _, manifest, _ = e2e
    au = manifest["action_units"]
    assert au["recorded"] == "steer"
    assert au["L_enc_m"] == STEER_WHEELBASE_M
    assert au["applies_to"] == ["ha"]
    assert any("ha0" in s for s in au["does_not_apply_to"])
    assert "physicalai.py:621" in au["rule"]


# =========================================================================== #
# (8) NO FUTURE ENTERS the model arm or the controls                           #
# =========================================================================== #
def test_no_recorded_future_reaches_os_ha_or_ha0(tmp_path):
    """Perturb every pose and action AFTER the last window's origin. ``os``,
    ``ha`` and ``ha0`` must be BIT-IDENTICAL; ``g`` (the GT waypoints, which ARE
    future) must CHANGE — that is what proves the perturbation landed in the
    consumed future rather than nowhere."""
    base = tmp_path / "base"
    pert = tmp_path / "pert"
    eps_b, lp_b, ck_b = _fixture(base)
    a_b = _args(base, ck_b, eps_b, lp_b, episodes_n=1, with_oracle_sel=False)
    rc.run_dump(a_b)
    ws = _cat(Path(a_b.dump_dir), "ws").astype(int)
    last_t0 = int(ws.max())
    # provider row j is RAW frame j + (n_stack - 1); perturb strictly AFTER t0
    eps_p, lp_p, ck_p = _fixture(pert,
                                 perturb_after=last_t0 + (N_STACK - 1) + 1)
    a_p = _args(pert, ck_p, eps_p, lp_p, episodes_n=1, with_oracle_sel=False)
    rc.run_dump(a_p)
    db, dp = Path(a_b.dump_dir), Path(a_p.dump_dir)
    assert np.array_equal(_cat(db, "ws"), _cat(dp, "ws"))
    for arm in ("os", "ha", "ha0"):
        d = np.abs(_cat(db, arm) - _cat(dp, arm)).max()
        assert d == 0.0, f"{arm} moved by {d} under a FUTURE-only perturbation"
    assert np.abs(_cat(db, "v0") - _cat(dp, "v0")).max() == 0.0
    moved = np.abs(_cat(db, "g") - _cat(dp, "g")).max()
    assert moved > 1.0, ("the perturbation did not reach the GT — the test "
                         f"proves nothing (max |dg| = {moved})")


# =========================================================================== #
# (9) misc contracts: grid refusal, tier pass-through, absent-nav refusal      #
# =========================================================================== #
def test_grid_is_index_select_and_an_unservable_grid_is_refused():
    g2 = rc.grid_slots(v3.V3_HORIZONS, "2s")
    assert g2["slots"] == [0, 1, 2, 3] and g2["k"] == 4 and g2["dt_s"] == 0.5
    g6 = rc.grid_slots(v3.V3_HORIZONS, "6s")
    assert g6["slots"] == [1, 3, 4, 5, 6, 7] and g6["instants_s"] == \
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    with pytest.raises(SystemExit) as e:
        rc.grid_slots((5, 10), "6s")       # a model that cannot serve the grid
    assert "index-select" in str(e.value) and "no interpolation" in str(e.value)


def test_tiers_flag_passes_through_and_ha0_needs_no_workaround():
    """⚠️ BACKLOG R13: ``t1_eval.DEFAULT_TIERS`` gained ``"ha0": "T1"``. This
    adapter declares its OWN ARM_TIERS for every arm it writes, so it works
    whether or not that line is present — and ``--tiers`` still passes through."""
    assert rc.ARM_TIERS["ha0"] == "T1" and rc.ARM_TIERS["os"] == "T1"
    assert rc.ARM_TIERS["oracle_sel"] == "T0"
    assert rc.t1._parse_tiers("ha0=T1,os=T1") == {"ha0": "T1", "os": "T1"}
    # this tool never edits t1_eval, and never needs to
    assert "cl" not in rc.ARM_TIERS


def test_unstamped_arm_is_refused(e2e, tmp_path):
    _, a, _, _ = e2e
    src, dst = Path(a.dump_dir), tmp_path / "unstamped"
    (dst / "decisions").mkdir(parents=True)
    for f in sorted(src.glob("ep*.npz")):
        with np.load(f) as d:
            arr = {k: d[k] for k in d.files}
        arr["mystery"] = arr["ha0"].copy()
        np.savez_compressed(dst / f.name, **arr)
    with pytest.raises(SystemExit) as e:
        rc.analyze_refcv3(str(dst), n_boot=10, seed=0)
    assert "no T0/T1 tier stamp" in str(e.value)


def test_config_rebuild_refuses_an_unknown_field():
    with pytest.raises(SystemExit) as e:
        rc.cfg_from_dict(v3.RefCV3Config, {"hier": True, "not_a_field": 1})
    assert "not_a_field" in str(e.value)
    # and it round-trips a real config, tuples intact
    cfg = _tiny_cfg()
    back = rc.cfg_from_dict(v3.RefCV3Config,
                            json.loads(json.dumps(rc.dataclasses.asdict(cfg))))
    assert back.core.trajectory.horizons == cfg.core.trajectory.horizons
    assert isinstance(back.core.trajectory.horizons, tuple)
    assert back.core.encoder.in_channels == 3 * N_STACK
    assert back.goal_tau_steps == cfg.goal_tau_steps
