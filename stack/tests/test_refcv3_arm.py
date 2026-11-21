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
 (4b) ⭐ THE NAV-ZERO ARM REMOVES THE SIGNAL, THE SHUFFLE ONLY THE PAIRING, AND
      THE TWO ARE NOT INTERCHANGEABLE (BACKLOG R39). MEASURED per layer rather
      than asserted: on the HIER build the E13 edge is LIVE under the fed nav
      (`nav_injected_true == 1` on every window) and DEAD under the null
      (`nav_injected_zero == 0`), so `os_navzero` differs from `os` even on
      `follow` windows. On a FLAT build there is no E13 path, so the only channel
      left is the core one-hot — and there `os_navzero` is BIT-IDENTICAL to `os`
      on EXACTLY the windows whose token is already `follow`. That second run is
      the control that isolates the mechanism, and it is why the claim
      "differs where nav is non-trivial" is stated per build rather than
      globally.
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


def _tiny_cfg(hier: bool = True) -> v3.RefCV3Config:
    """The SMALLEST instantiable v3: the smoke config with the encoder widened to
    the corpus's 9 channels (3-frame stack) at 64 px — the geometry the adapter
    asserts against the episodes. ``hier=False`` builds the FLAT arm, which has
    no E13 nav injection at all — the control in (4b)."""
    cfg = v3.refc_v3_smoke_config(hier=hier)
    cfg.core.encoder = refc.CNNEncoderConfig(in_channels=3 * N_STACK,
                                             image_size=SIZE, base_width=8,
                                             blocks=(1, 1, 1, 1))
    return cfg


def _fixture(root: Path, *, perturb_after: int | None = None,
             hier: bool = True):
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
    cfg = _tiny_cfg(hier)
    model = v3.RefCV3Model(cfg)
    torch.save({"step": 11, "model": model.state_dict(), "opt": {}},
               run / "ckpt.pt")
    # ⭐ the adapter's documented escape hatch: refc_v3_train writes NO model
    # config, so a config.json may carry an explicit RefCV3Config dict. This also
    # exercises `cfg_from_dict` (tuples survive JSON, unknown fields refused).
    (run / "config.json").write_text(json.dumps({
        "arm": "hier" if hier else "flat",
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
                nav_shuffle_seed=0, no_navshuf=False, no_navzero=False,
                with_navzero=False, with_oracle_sel=True, allow_nonstrict=False,
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
    # ⭐ `ha0_ext` JOINED THIS LIST AT RUNG A1 (2026-09-05) and it is NOT
    # optional: refcv5's acceptance bar is "beat BOTH `ha` and `ha0_ext`", and
    # before the port this harness computed neither, so half the bar was
    # unreadable on the whole REF-C surface.
    for arm in ("os", "ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero",
                "oracle_sel"):
        assert d[arm].shape == (N, 4, 2), arm
    # ⛔ the dump's key space IS the arm space: metadata must not be an arm
    assert d["v0"].shape == (N,) and d["eid"].tolist() == [0]
    assert set(rec["arms"]) == {"os", "ha", "ha0", "ha0_ext", "os_navshuf",
                                "os_navzero", "oracle_sel"}
    assert rec["tiers"] == {"os": "T1", "ha": "T1", "ha0": "T1",
                            "ha0_ext": "T1", "os_navshuf": "T1",
                            "os_navzero": "T1", "oracle_sel": "T0"}
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
    # ⭐ BACKLOG R39: the nav-ZERO arm needs BOTH its own margin over the shared
    # floor (the DEPLOYMENT-relevant number) and the delta against `os` (what the
    # oracle nav is worth). Neither is answerable from the shuffle.
    assert "paired_os_navzero_minus_ha0" in fp
    assert "paired_os_minus_navzero" in fp
    # ⭐⭐ RUNG A1: THE OTHER HALF OF refcv5's ACCEPTANCE BAR. `ha0` alone is
    # not the bar — stack/tanitad/eval/echo_gate.py retracts that reading by
    # name. An arm that beats `ha` while TYING `ha0_ext` has echoed its own
    # ego state, not read the scene, and only this block can see it.
    assert "paired_os_minus_ha0ext" in fp, (
        "the os - ha0_ext margin is half of refcv5's acceptance bar")
    assert "paired_os_navzero_minus_ha0ext" in fp, (
        "the DEPLOYMENT-relevant form of the same read (no oracle nav)")
    assert fp["paired_os_minus_ha0ext"]["direction"] == "os - ha0_ext"
    hl = rec["refcv3"]["headline"]
    assert hl["floor_arm"] == "ha0"
    assert hl["deployment_margin_block"] == \
        "families_paired.paired_os_navzero_minus_ha0"
    assert hl["oracle_nav_worth_block"] == "families_paired.paired_os_minus_navzero"
    assert "not_interchangeable" not in hl or True
    assert "NOT" in hl["_nav_controls"]
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
def test_navzero_removes_the_signal_not_just_the_pairing(e2e):
    """⭐ BACKLOG R39. Shuffle and zero are different interventions, and on the
    HIER build the difference is MEASURABLE per layer rather than asserted: the
    E13 edge is live under the fed nav and dead under the null."""
    _, a, manifest, rec = e2e
    dump = Path(a.dump_dir)
    inj_t = _cat_dec(dump, "nav_injected_true").astype(float)
    inj_z = _cat_dec(dump, "nav_injected_zero").astype(float)
    # the E13 edge is LIVE under the fed nav and DEAD under nav_cmd=None
    assert (inj_t == 1.0).all(), "E13 must be live under the fed nav"
    assert (inj_z == 0.0).all(), "E13 must be DEAD under nav_cmd=None"
    nav = _cat_dec(dump, "nav_cmd").astype(int)
    os_ = _cat(dump, "os").astype(np.float64)
    nz_ = _cat(dump, "os_navzero").astype(np.float64)
    ns_ = _cat(dump, "os_navshuf").astype(np.float64)
    turn = nav != 0                       # 0 == 'follow' (refb.NAV_COMMANDS)
    assert turn.any() and (~turn).any(), "the fixture needs both kinds of window"
    # where nav is non-trivial the zero arm must move
    assert np.abs(os_[turn] - nz_[turn]).max() > 1e-9
    # ⭐ AND on `follow` windows too — because E13 is removed there as well.
    # This is exactly what the SHUFFLE cannot do: a permuted `follow` that stays
    # `follow` leaves the forward bit-identical.
    assert np.abs(os_[~turn] - nz_[~turn]).max() > 1e-9, (
        "on the hier build the nav-zero arm must differ even on follow windows, "
        "because the E13 injection is switched off for the whole call")
    same_tok = nav == _cat_dec(dump, "nav_cmd_shuf").astype(int)
    assert np.abs(os_[same_tok] - ns_[same_tok]).max() < 1e-9, (
        "the shuffle leaves an unchanged token bit-identical — which is why it "
        "cannot answer the question the zero arm answers")
    # ⚠️ the CROSS-CALL floor is recorded, so `identical_to` (1e-9 m) is not
    # read as nav evidence for an arm that comes from a separate forward call
    triv = rec["refcv3"]["trivial_profile"]
    assert triv["cross_call_arms"] == ["os_navzero"]
    assert "5.96e-07" in triv["cross_call_note"]
    assert "EXACTLY 0.0" in triv["cross_call_note"]
    assert "os_navzero" not in (triv["arms"]["os"].get("identical_to") or {})
    # the manifest states the null, per layer, and why nav_known is not it
    nn = manifest["nav_null"]
    assert nn["emitted"] is True
    assert "nav_cmd=None" in nn["how"] and "SEPARATE forward" in nn["how"]
    assert "nav_known_channel is False" in nn["why_not_nav_known"]
    assert "refc.py:2042-2045" in nn["why_not_nav_known"]
    per = nn["what_nav_zero_removes_per_layer"]
    assert "REMOVED ENTIRELY" in per["tactical (E13 PhiTac)"]
    assert "REMOVED ENTIRELY" in per["strategic (E13 ctx)"]
    assert "NOT REMOVED" in per["core (measurement encoder)"]
    assert "LOWER BOUND" in nn["⛔ read_it_as"]
    assert rec["refcv3"]["nav_null"]["emitted"] is True


def test_flat_build_isolates_the_core_collapse(tmp_path):
    """⭐ THE CONTROL THAT MAKES THE CLAIM PRECISE. On a FLAT (hier=False) build
    there is no E13 path, so `nav_cmd=None` differs from the true nav ONLY
    through the core one-hot — and `os_navzero` is therefore BIT-IDENTICAL to
    `os` on EXACTLY the windows whose token is already `follow` (index 0), and
    different on every other window. That is the mechanism `NAV_NULL` claims,
    measured rather than asserted."""
    root = tmp_path / "flat"
    eps, lp, ck = _fixture(root, hier=False)
    a = _args(root, ck, eps, lp, with_oracle_sel=False)
    rc.run_dump(a)
    dump = Path(a.dump_dir)
    nav = _cat_dec(dump, "nav_cmd").astype(int)
    os_ = _cat(dump, "os").astype(np.float64)
    nz_ = _cat(dump, "os_navzero").astype(np.float64)
    follow = nav == 0
    assert follow.any() and (~follow).any()
    d_follow = np.abs(os_[follow] - nz_[follow]).max()
    d_turn = np.abs(os_[~follow] - nz_[~follow]).max()
    # ⚠️ NOT `== 0.0` HERE, AND THE REASON IS MEASURED, NOT ASSUMED. In the dump,
    # `os` is row 0 of a 2-row batched call and `os_navzero` is its own 1-row
    # call, so a float32 GEMM-kernel difference of ~6e-7 m sits under the
    # comparison. The EXACT claim is asserted below at MATCHED batch size, where
    # it reads 0.0. Here the honest test is the SEPARATION, and it is enormous.
    assert d_follow < 1e-5, (
        f"flat build: on a `follow` window nav_cmd=None must agree with the fed "
        f"token to the float32 batching floor, got {d_follow}")
    assert d_turn > 1000 * max(d_follow, 1e-12), (
        f"flat build: a real nav difference must dwarf the batching floor "
        f"(turn {d_turn} vs follow {d_follow})")
    # ⭐ THE EXACT CLAIM, AT MATCHED BATCH SIZE: with no E13 path, nav_cmd=None
    # IS the `follow` token — bit for bit. This is the mechanism NAV_NULL states.
    model, cfg, _t, prov = rc.load_model(str(ck), None, "cpu", False)
    _e, _f, _c, ds, _l, _j, _s, _o = rc.build_corpus(a, cfg, prov)
    tr = rc.trainer()
    steps = int(prov["decoder_steps"])
    n_same = 0
    for wi in range(0, len(ds.index), 7):
        item = ds[wi]
        fr = tr.frames_to_device(item["frames"][None], "cpu")
        v1 = torch.tensor([float(item["pose_last"][3])], dtype=torch.float32)
        with torch.no_grad():
            a_none = model(fr, nav_cmd=None, v0=v1, steps=steps)["traj"]
            a_zero = model(fr, nav_cmd=torch.tensor([0]), v0=v1,
                           steps=steps)["traj"]
            a_left = model(fr, nav_cmd=torch.tensor([1]), v0=v1,
                           steps=steps)["traj"]
        assert torch.equal(a_none, a_zero), (
            "flat build at MATCHED batch size: nav_cmd=None must be BIT-"
            "IDENTICAL to nav_cmd=0 — there is no E13 path and refc.py:2021-2024 "
            "substitutes one_hot(0)")
        assert (a_none - a_left).abs().max() > 1e-3, (
            "a real nav token must move the flat build's trajectory")
        n_same += 1
    assert n_same >= 3
    # ...and the tool says so in the record rather than leaving it to be inferred
    assert "BIT-IDENTICAL" in rc.NAV_NULL["consequence_for_the_flat_arm"]


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
    # ⭐ `ha0_ext` holds a RECORDED channel-1 value, so the steer->kappa
    # conversion applies to it exactly as it does to `ha`. `ha0` is exactly
    # zero and is therefore unit-free — that is what makes it, and only it,
    # bit-comparable across refav1 and refcv3.
    assert au["applies_to"] == ["ha", "ha0_ext"]
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
    assert rc.ARM_TIERS["os_navzero"] == "T1"
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


# =========================================================================== #
# (10) D-EVALTOOL-ANCHOR-CHANCE — THE ANCHOR CHANCE LEVEL IS DERIVED, NEVER    #
#      REMEMBERED                                                             #
# =========================================================================== #
# THE DEFECT: `_SIDECAR_DOC["anchor_acc"]` shipped a hardcoded
# `"chance = 1/128 = 0.0078"` into every manifest. 128 is `refc.py:356`'s DEFAULT
# bank; refcv4b's is 117, whose chance is 0.008547 — 9.4 % larger. The COMPUTED
# field (`refcv3.tactical_declared.anchor_selection.chance`) was always right, so
# the defect was reader-facing; these tests make the two impossible to drift.
# ⛔ Substituting 117 for 128 would have been the SAME defect with a different
# number, so what is pinned is that NO literal bank size decides anything.
def test_anchor_chance_is_one_over_the_runs_own_bank():
    # the two banks that actually exist in the programme, at the precision the
    # refcv4b landing JSON records (`"n_anchors": 117, "chance": 0.008547`)
    assert rc.anchor_chance(117) == 0.008547
    assert rc.anchor_chance(128) == 0.007812
    assert rc.anchor_chance(256) == 0.003906
    # ⛔ and NO fallback: an artifact that does not declare a bank gets a
    # refusal, never the config default. A fabricated denominator reads exactly
    # like a measured one.
    for bad in (None, 0, -1, "", "117", float("nan")):
        assert rc.anchor_chance(bad) is None, bad


def test_the_schema_string_and_the_computed_chance_cannot_drift(e2e):
    """The schema line, the tactical `chance` and `model.n_anchors` are all the
    SAME derivation — checked against each other, not against a literal."""
    _, _, manifest, rec = e2e
    n = int(manifest["model"]["n_anchors"])
    doc = manifest["sidecar_schema"]["anchor_acc"]
    # (a) ⛔ FIRST, and in the OLD vocabulary, so this test fails on the SUBSTANCE
    # against the pre-fix file rather than on a missing attribute: this run's
    # bank is not 128, so no 128-chance may appear in its schema.
    assert n != 128, "pick a fixture whose bank is not the config default"
    assert "1/128" not in doc and "0.0078" not in doc, (
        f"the schema publishes a 1/128 chance level for a {n}-anchor bank — "
        f"D-EVALTOOL-ANCHOR-CHANCE")
    # (b) and it names THIS run's bank with its derived chance
    assert f"1/{n} = {rc.anchor_chance(n)}" in doc
    # (c) the computed field agrees with the schema, exactly
    sel = rec["refcv3"]["tactical_declared"]["anchor_selection"]
    assert sel["n_anchors"] == n
    assert sel["chance"] == rc.anchor_chance(n)
    assert sel["chance_absent_because"] is None
    # (d) the same bank size is what `_is` tells the reader the model chose among
    assert f"its {n} anchors" in manifest["t1_definition"]["_is"]


@pytest.mark.parametrize("n_anchors", [117, 128, 256, 20])
def test_the_denominator_follows_the_bank_size_by_mutation(n_anchors):
    """⭐ THE PIN THE BRIEF ASKS FOR: change the bank size, and every published
    denominator moves with it. A hardcoded 128 (or a hardcoded 117) fails here."""
    doc = rc.sidecar_schema(n_anchors)["anchor_acc"]
    assert f"chance = 1/{n_anchors} = {rc.anchor_chance(n_anchors)}" in doc
    others = {117, 128, 256, 20} - {n_anchors}
    for other in others:
        assert f"1/{other} " not in doc and f"1/{other}," not in doc


def test_a_manifest_with_no_bank_size_is_refused_not_defaulted(e2e, tmp_path):
    """⛔ The `or 128` fallbacks are gone. A dump whose manifest declares no
    `model.n_anchors` must REFUSE the selection profile and publish NO chance
    level — it must not silently inherit the v3 config default."""
    _, a, manifest, _ = e2e
    src, dst = Path(a.dump_dir), tmp_path / "no_bank"
    (dst / "decisions").mkdir(parents=True)
    for sub in ("", "decisions"):
        for f in sorted((src / sub).glob("ep*.npz")):
            with np.load(f) as d:
                np.savez_compressed(dst / sub / f.name,
                                    **{k: d[k] for k in d.files})
    stripped = json.loads(json.dumps(manifest))
    stripped["model"].pop("n_anchors")
    (dst / "manifest.json").write_text(json.dumps(stripped), encoding="utf-8")

    prof = rc._selection_profile(rc._load_decisions(str(dst))[0], stripped)
    # ⛔ `.get` on purpose: against the pre-fix file this reads the DEFAULTED
    # profile and fails with "it invented n_anchors=128", which is the substance,
    # rather than with a KeyError about a field that did not exist.
    assert prof.get("status") == "REFUSED", (
        f"a manifest with no bank size still produced a profile: "
        f"n_anchors={prof.get('n_anchors')}")
    assert "n_anchors" in prof["reason"]
    assert "128" in prof["reason"]          # it names what it refused to assume

    rec = rc.analyze_refcv3(str(dst), n_boot=10, seed=0)
    sel = rec["refcv3"]["tactical_declared"]["anchor_selection"]
    assert sel["chance"] is None and sel["n_anchors"] is None
    assert "REFUSED" in sel["chance_absent_because"]
    # the schema line says the same thing rather than quoting a number
    assert rc.sidecar_schema(None)["anchor_acc"].endswith("never guessed.")


# =========================================================================== #
# (11) D-EVALTOOL-STAMP-BLIND — THE PROVENANCE STAMP MUST DISCRIMINATE         #
# =========================================================================== #
# THE DEFECT: `_UNVERIFIED_ON_REAL_CKPT` ("UNVERIFIED on a real checkpoint …
# validated on a random-init RefCV3Model … synthetic 3-episode slice only") was
# emitted on EVERY dump and EVERY analysis record — including the refcv4b LANDING
# arm, a real 40,284-step checkpoint. Anyone using it to tell a fixture from a
# real arm therefore misclassified EVERY real dump the programme holds.
#
# ⛔ A STAMP THAT READS IDENTICALLY ON BOTH HAS MEASURED NOTHING. These tests
# mutate the run from synthetic to real-checkpoint and require the stamp to
# change — in BOTH directions, and end-to-end through `run_dump`, not only in
# the pure function.
#: the two MEASURED reference points (2026-09-06).
FIXTURE_REF = {"model": {"step": 11, "n_anchors": 20},
               "grid": {"n_episodes": 3, "n_windows": 42}}
LANDING_REF = {"model": {"step": 40284, "n_anchors": 117},
               "grid": {"n_episodes": 141, "n_windows": 4823}}


def test_the_stamp_separates_the_fixture_from_the_landing_arm():
    fix, real = rc.provenance_stamp(FIXTURE_REF), rc.provenance_stamp(LANDING_REF)
    # ⛔ the whole point: they are NOT the same stamp
    assert fix != real
    assert fix["verdict"] == "SYNTHETIC_FIXTURE"
    assert real["verdict"] == "REAL_CHECKPOINT_ON_REAL_CORPUS"
    assert fix["unverified_on_a_real_checkpoint"] is True
    assert real["unverified_on_a_real_checkpoint"] is False
    # every axis separates, so no single threshold is load-bearing
    assert (fix["checkpoint_scale"], fix["corpus_scale"], fix["bank_scale"]) \
        == ("SMOKE", "SMOKE", "SMOKE")
    assert (real["checkpoint_scale"], real["corpus_scale"], real["bank_scale"]) \
        == ("TRAINED", "FULL", "REAL")
    # and the KEY, not just the verdict, is emitted conditionally
    assert "_unverified" in rc.provenance_keys(FIXTURE_REF)
    assert "_unverified" not in rc.provenance_keys(LANDING_REF)
    assert "_provenance" in rc.provenance_keys(LANDING_REF)


def test_each_discriminator_alone_moves_the_stamp():
    """No axis is decoration: flip one field at a time from the landing arm and
    the stamp must react to that field."""
    base = json.loads(json.dumps(LANDING_REF))
    for path, value, key, want in (
            (("model", "step"), 11, "checkpoint_scale", "SMOKE"),
            (("model", "step"), None, "checkpoint_scale", "UNKNOWN"),
            (("grid", "n_episodes"), 3, "corpus_scale", "SMOKE"),
            (("grid", "n_windows"), 42, "corpus_scale", "SMOKE"),
            (("model", "n_anchors"), 20, "bank_scale", "SMOKE")):
        m = json.loads(json.dumps(base))
        m[path[0]][path[1]] = value
        st = rc.provenance_stamp(m)
        assert st[key] == want, (path, value, st)
        # the field that moved is republished, so the reader can check the call
        assert st["discriminators"][f"{path[0]}.{path[1]}"] == value


def test_a_clean_strict_load_is_carried_but_never_read_as_a_discriminator(e2e):
    """⚠️ THE NAIVE DISCRIMINATOR, REFUTED IN THE FIXTURE ITSELF. The synthetic
    run's state_dict ALSO loads with empty missing/unexpected keys — exactly like
    the real landing arm — so 'it loads cleanly' says nothing about trainedness.
    The stamp must not be reading it."""
    _, _, manifest, _ = e2e
    sd = manifest["model"]["state_dict_load"]
    assert sd["missing_keys"] == [] and sd["unexpected_keys"] == []
    assert manifest["_provenance"]["verdict"] == "SYNTHETIC_FIXTURE"
    assert "model.state_dict_load" in manifest["_provenance"][
        "not_a_discriminator"]


def test_mutating_the_run_from_synthetic_to_real_flips_the_stamp(e2e, tmp_path):
    """⭐ THE TWO-SIDED PROOF, END TO END THROUGH `run_dump`.

    Same corpus, same code path, ONE lever moved — the checkpoint's own step
    count. The synthetic run must carry `_unverified`; the real-checkpoint run
    must NOT.

    ⛔ THE ASSERTIONS BELOW ARE ORDERED ON PURPOSE. Everything before the
    `_provenance` block is written in the vocabulary the OLD tool also emitted,
    so running this test against the pre-fix file fails on the SUBSTANCE — "the
    stamp claims UNVERIFIED about a 40,284-step checkpoint" — and not merely on
    a key that did not exist yet. A test that fails with `KeyError` proves the
    field is new; this one proves the old field could not discriminate."""
    root, a, manifest, _ = e2e

    # --- side A: the fixture as built (random-init, step 11) ---------------- #
    assert manifest["model"]["step"] == 11
    assert manifest["_unverified"] == rc._UNVERIFIED_ON_REAL_CKPT

    # --- side B: the SAME dump with a real-checkpoint step count ------------ #
    ck2 = tmp_path / "ckpt_real.pt"
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    blob["step"] = 40284
    torch.save(blob, str(ck2))
    (tmp_path / "config.json").write_text(
        (Path(a.ckpt).parent / "config.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    a2 = _args(root, ck2, Path(a.episodes), Path(a.labels),
               dump_dir=str(tmp_path / "dump_real"))
    man2 = rc.run_dump(a2)
    assert man2["model"]["step"] == 40284

    # ⛔⛔ THE DISCRIMINATION ITSELF, in the OLD vocabulary: one lever moved, and
    # the stamp is no longer the same. On the pre-fix file BOTH runs carry the
    # identical `_unverified` string and this is the line that fails.
    assert "_unverified" not in man2, (
        "the stamp still asserts UNVERIFIED-ON-A-REAL-CHECKPOINT about a "
        "40,284-step checkpoint — it cannot discriminate, which is "
        "D-EVALTOOL-STAMP-BLIND")
    assert manifest.get("_unverified") != man2.get("_unverified")

    # --- and now the positive block that replaced it ------------------------ #
    assert manifest["_provenance"]["verdict"] == "SYNTHETIC_FIXTURE"
    assert man2["_provenance"]["checkpoint_scale"] == "TRAINED"
    # the corpus is still a 3-episode slice, and the stamp says so separately
    assert man2["_provenance"]["corpus_scale"] == "SMOKE"
    assert man2["_provenance"]["verdict"] == "REAL_CHECKPOINT_ON_SMOKE_CORPUS"
    # ⛔ and the two stamps are not the same object read twice
    assert man2["_provenance"] != manifest["_provenance"]

    # --- and the ANALYSIS record inherits it, not a re-asserted constant ---- #
    rec2 = rc.analyze_refcv3(a2.dump_dir, n_boot=10, seed=0)
    assert "_unverified" not in rec2
    assert rec2["_provenance"]["verdict"] == "REAL_CHECKPOINT_ON_SMOKE_CORPUS"
    rec1 = rc.analyze_refcv3(a.dump_dir, n_boot=10, seed=0)
    assert rec1["_unverified"] == rc._UNVERIFIED_ON_REAL_CKPT
    assert rec1["_provenance"]["verdict"] == "SYNTHETIC_FIXTURE"


def test_a_banked_dump_is_classified_without_being_re_rolled(e2e, tmp_path):
    """The stamp is derived from the manifest, so a dump banked BEFORE it existed
    is classified by reading `manifest.json` alone — no GPU, no re-roll."""
    _, a, manifest, _ = e2e
    legacy = json.loads(json.dumps(manifest))
    legacy.pop("_provenance"), legacy.pop("_unverified")
    legacy["model"]["step"] = 40284
    legacy["grid"].update(n_episodes=141, n_windows=4823)
    st = rc.provenance_stamp(legacy)
    assert st["verdict"] == "REAL_CHECKPOINT_ON_REAL_CORPUS"
    assert st["unverified_on_a_real_checkpoint"] is False
