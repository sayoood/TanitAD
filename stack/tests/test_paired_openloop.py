"""``taniteval/tools/paired_openloop.py`` — the PAIRED CROSS-MODEL **OPEN-LOOP**
comparison, on synthetic dumps of the two real shapes. CPU only, no model, no GPU;
⛔ it never touches Thor, a pod, or a real checkpoint.

WHAT IS PINNED — every line is a way this comparison has gone, or could go, wrong

  (1) ⭐ THE WINDOW KEY IS `(clip_id, RAW 10 Hz frame)` AND IT IS *PROVEN*, NOT
      ASSUMED. refav1's `ws` is a cache index whose RAW origin is `2*ws`
      (`refav1_arm.gt_waypoints`: `f0 = 2 * t`); refcv3's is a PROVIDER index whose
      RAW origin is `ws + (n_stack - 1)`, read from its own manifest. The tool then
      REFUSES unless the two dumps' GT and `v0` agree on the windows they claim to
      share — the only thing that can actually settle a frame-offset question.
  (2) THE COMMON GRID IS DERIVED IN INTEGER RAW FRAMES, never in seconds. Matching
      `0.2*k` against `0.5*j` in floating point is how two grids silently "agree".
  (3) ⭐ THE FLOOR READS ITS KNOWN VALUE. `ha0` is `x = v0*t, y = 0` exactly; the
      DERIVED floor (for a dump that predates the arm) must reproduce a DUMPED one,
      and the two sides' floors must agree — that is what "bit-comparable across two
      architectures" means, and it is the license for one shared floor array.
  (4) ⛔ A CONSTANT-VELOCITY MODEL ARM MAKES THE READ **VOID, NOT NEGATIVE**
      (`D-REFAV1-PAIRED-READ-VOID`): a straight constant-speed plan beating a noisy
      hold-action control reads as lateral SKILL in any family table.
  (5) ⛔ AN ORACLE ARM IS REFUSED AS A MODEL ARM (`D-HF-COMPARABILITY` cond. 2):
      refcv3's `a_star`/`oracle_sel` is the anchor nearest the GROUND TRUTH.
  (6) ⛔ A CROSS-TIER COMPARISON IS REFUSED (cond. 3).
  (7) AN EMPTY INTERSECTION REFUSES AND NAMES THE STRIDE MISMATCH — the failure a
      stride-5 refcv3 dump against a stride-10 refav1 dump produces, where every
      other check passes and the answer is simply "0 windows".
  (8) ⭐ THE ALGEBRAIC IDENTITY IS TRUE AND STATED: with an identical floor,
      `(B - floor) - (A - floor)` IS `B - A`. The margin framing is what gets
      reported, but the record must not imply two different quantities were computed.
  (9) THE DECISION FAMILIES GET THE RIGHT FLOOR: `ha0` is a trajectory and has no
      head, so the no-information value for a categorical row is the MAJORITY-CLASS
      rate — shared across the two models because it is a property of the LABELS.
 (10) THE DECLARED ROWS ARE SCOPED to windows BOTH sides label, and a genuine label
      DISAGREEMENT refuses the row instead of scoring two different questions.
 (11) THE POWER WARNING FIRES below the episode threshold: an episode-cluster
      bootstrap over 2 clips cannot resample anything else, so "not separated" there
      is UNDERPOWERED, not a null.
 (12) ⛔ THE RECORD NEVER SAYS "CLOSED LOOP" (PI ruling 2026-09-02): every arm here
      is OPEN LOOP, including a planner whose own predictor consumes its actions.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
TOOL = _REPO / "taniteval" / "tools" / "paired_openloop.py"

_spec = importlib.util.spec_from_file_location("paired_openloop_under_test", TOOL)
po = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(po)

CLIPS = ("clip-aaaaaaa0", "clip-bbbbbbb1", "clip-ccccccc2")
#: refav1's grid: dt 0.2 s, K 10 -> RAW offsets 2,4,...,20
A_DT, A_K = 0.2, 10
#: refcv3's `2s` grid: model slots 5,10,15,20 at 10 Hz -> 0.5,1.0,1.5,2.0 s
B_HORIZONS = [5, 10, 15, 20]
#: the only instants the two grids share
COMMON_RAW = [10, 20]
B_RAW_OFFSET = 2                      # n_stack - 1


def _BEND(fi: int, n: int):
    """The GT's curvature, deterministic in (clip, window) so BOTH fixture sides
    produce byte-identical ground truth — as the two real dumps do."""
    return 0.004 * (fi + 1) + 0.0005 * np.arange(n)


def _cv(v0, offs):
    """The constant-velocity path at `v0` on the given RAW offsets — `ha0`."""
    t = np.asarray(offs, dtype=np.float64) * 0.1
    return np.stack([np.outer(v0, t), np.zeros((len(v0), len(offs)))], axis=-1)


def _curved(v0, offs, bend):
    """A path that is neither straight nor constant-speed, so it is not trivial."""
    v0 = np.asarray(v0, dtype=np.float64)
    t = np.asarray(offs, dtype=np.float64) * 0.1
    bend = np.broadcast_to(np.asarray(bend, dtype=np.float64), v0.shape)
    x = np.outer(v0, t) * (1.0 + 0.02 * t[None, :])
    y = bend[:, None] * (t ** 2)[None, :]
    return np.stack([x, y], axis=-1)


def _write(dirp: Path, manifest: dict, per_ep: dict, dec: dict | None = None):
    dirp.mkdir(parents=True, exist_ok=True)
    (dirp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if dec:
        (dirp / "decisions").mkdir(exist_ok=True)
    for fi, arrays in per_ep.items():
        np.savez(dirp / f"ep{fi:03d}.npz", **arrays)
        if dec and fi in dec:
            np.savez(dirp / "decisions" / f"ep{fi:03d}.npz", **dec[fi])
    return dirp


def _mk_a(tmp: Path, *, name="A", ws_by_ep=None, arm_maker=None, tier="T1",
          n_clips=3, v0_scale=1.0, gt_shift=0.0, dec_extra=None,
          arms=("cl", "ha", "ol")):
    """A refav1-shaped dump: `ws` is a cache index, RAW origin = 2*ws, no `ha0`."""
    ws_by_ep = ws_by_ep or {i: [13, 23, 33] for i in range(n_clips)}
    offs = [int(round(A_DT / 0.1)) * (j + 1) for j in range(A_K)]
    eps, per_ep, dec = [], {}, {}
    for fi in range(n_clips):
        ws = np.array(ws_by_ep[fi], dtype=np.int64)
        n = ws.size
        v0 = (8.0 + fi + np.arange(n) * 0.5) * v0_scale
        # ⭐ GT is a DETERMINISTIC function of (clip, window) on both sides — the
        # two real dumps read the SAME corpus, so the fixture must too, or the
        # window-key proof would fire on the fixture rather than on a real defect.
        gt = _curved(v0, offs, _BEND(fi, n)) + gt_shift
        arrays = {"g": gt, "ws": ws, "eid": np.array([fi]),
                  "clip_index": np.array([fi]), "v0": v0}
        for arm in arms:
            arrays[arm] = (arm_maker(arm, v0, offs, fi) if arm_maker
                           else _curved(v0, offs, 0.006 * (fi + 1)))
        per_ep[fi] = arrays
        d = {"ws": ws,
             "route_label": np.array([1, 2, -100][:n] if n <= 3 else [1] * n),
             "route_pred_nav_true": np.array([1, 2, 0][:n] if n <= 3 else [1] * n),
             "route_pred_nav_shuffled": np.array([0, 0, 0][:n] if n <= 3 else [0] * n),
             "route_pred_nav_zero": np.array([1, 0, 0][:n] if n <= 3 else [0] * n),
             "lat_label": np.array([0, 7, -100][:n] if n <= 3 else [0] * n),
             "lat_pred_nav_true": np.array([0, 7, 0][:n] if n <= 3 else [0] * n),
             "lat_pred_nav_shuffled": np.array([0, 0, 0][:n] if n <= 3 else [0] * n),
             "lat_pred_nav_zero": np.array([0, 0, 0][:n] if n <= 3 else [0] * n),
             "lon_label": np.array([6, 7, -100][:n] if n <= 3 else [6] * n),
             "lon_pred_nav_true": np.array([6, 7, 0][:n] if n <= 3 else [6] * n),
             "lon_pred_nav_shuffled": np.array([6, 6, 6][:n] if n <= 3 else [6] * n),
             "lon_pred_nav_zero": np.array([6, 6, 6][:n] if n <= 3 else [6] * n),
             "nav_cmd": np.zeros(n, dtype=np.int64),
             "nav_valid": np.ones(n, dtype=bool)}
        if dec_extra:
            d.update({k: v(n) for k, v in dec_extra.items()})
        dec[fi] = d
        eps.append({"file_index": fi, "episode_index": fi, "name": CLIPS[fi],
                    "clip_id": CLIPS[fi], "n_windows": int(n)})
    mf = {"tool": "refav1_arm.py",
          "grid": {"dt_s": A_DT, "horizon_k": A_K, "window_stride": 10,
                   "n_windows": sum(len(v) for v in ws_by_ep.values()),
                   "n_episodes": n_clips},
          "arms": list(arms), "tiers": {x: ("T0" if x == "ol" else tier) for x in arms},
          "episodes": eps,
          "model": {"ckpt": "synthetic", "step": 1000, "tac_vocab_version": "v7.0"}}
    return _write(tmp / name, mf, per_ep, dec)


def _mk_b(tmp: Path, *, name="B", ws_by_ep=None, arm_maker=None, tier="T1",
          n_clips=3, route_label_override=None,
          arms=("os", "ha", "ha0", "os_navshuf", "os_navzero", "oracle_sel")):
    """A refcv3-shaped dump: `ws` is a PROVIDER index, RAW origin = ws + 2, WITH `ha0`."""
    ws_by_ep = ws_by_ep or {i: [24, 44, 64] for i in range(n_clips)}
    eps, per_ep, dec = [], {}, {}
    for fi in range(n_clips):
        ws = np.array(ws_by_ep[fi], dtype=np.int64)
        n = ws.size
        v0 = 8.0 + fi + np.arange(n) * 0.5
        gt = _curved(v0, B_HORIZONS, _BEND(fi, n))
        arrays = {"g": gt, "ws": ws, "eid": np.array([fi]),
                  "clip_index": np.array([fi]), "v0": v0}
        for arm in arms:
            if arm == "ha0":
                arrays[arm] = _cv(v0, B_HORIZONS)
            elif arm_maker:
                arrays[arm] = arm_maker(arm, v0, B_HORIZONS, fi)
            else:
                arrays[arm] = _curved(v0, B_HORIZONS, 0.0042 * (fi + 1))
        per_ep[fi] = arrays
        rl = (route_label_override(n, fi) if route_label_override
              else np.array([1, 2, 1][:n] if n <= 3 else [1] * n))
        dec[fi] = {"ws": ws, "route_label": rl,
                   "route_pred_nav_true": np.array([1, 2, 1][:n] if n <= 3 else [1] * n),
                   "route_pred_nav_shuffled": np.array([1, 1, 1][:n] if n <= 3 else [1] * n),
                   "route_pred_nav_zero": np.array([1, 1, 1][:n] if n <= 3 else [1] * n),
                   "lat_label": np.array([0, 7, -100][:n] if n <= 3 else [0] * n),
                   "lat_pred_nav_true": np.array([0, 0, 0][:n] if n <= 3 else [0] * n),
                   "lat_pred_nav_shuffled": np.array([0, 0, 0][:n] if n <= 3 else [0] * n),
                   "lat_pred_nav_zero": np.array([0, 0, 0][:n] if n <= 3 else [0] * n),
                   "lon_label": np.array([6, 7, -100][:n] if n <= 3 else [6] * n),
                   "lon_pred_nav_true": np.array([6, 7, 6][:n] if n <= 3 else [6] * n),
                   "lon_pred_nav_shuffled": np.array([6, 6, 6][:n] if n <= 3 else [6] * n),
                   "lon_pred_nav_zero": np.array([6, 6, 6][:n] if n <= 3 else [6] * n),
                   "nav_cmd": np.zeros(n, dtype=np.int64),
                   "nav_valid": np.ones(n, dtype=bool),
                   "sel_idx": np.arange(n) % 4,
                   "a_star": np.arange(n) % 4,
                   "sel_agrees_oracle": np.ones(n, dtype=np.float32)}
        eps.append({"file_index": fi, "episode_index": fi, "clip_id": CLIPS[fi],
                    "episode_id": 1000 + fi, "n_windows": int(n)})
    mf = {"tool": "refcv3_arm.py",
          "grid": {"name": "2s", "dt_s": 0.5, "k": 4, "horizons_steps": B_HORIZONS,
                   "slots": [0, 1, 2, 3], "instants_s": [0.5, 1.0, 1.5, 2.0],
                   "window_stride": 1, "obs_window": 8},
          "corpus": {"frames": {"n_stack": 3,
                                "provider_to_raw_frame_offset": B_RAW_OFFSET}},
          "arms": list(arms),
          "tiers": {x: ("T0" if x == "oracle_sel" else tier) for x in arms},
          "action_units": {"recorded": "steer", "applies_to": ["ha"]},
          "episodes": eps,
          "model": {"ckpt": "synthetic", "step": 30000, "tac_vocab_version": "v7.0",
                    "n_anchors": 128},
          "tier_ruling": {"arm": "os", "stamped": "T1", "status": "UNRULED"}}
    return _write(tmp / name, mf, per_ep, dec)


def _args(a_dump, b_dump, **kw):
    import argparse
    d = dict(a_dump=str(a_dump), a_name="refav1", a_arm="cl", a_extra=[],
             b_dump=str(b_dump), b_name="refcv3", b_arm="os", b_extra=[],
             a_run_config=None, b_run_config=None,
             floor="ha0", n_boot=60, seed=0, out=None, md=None)
    d.update(kw)
    return argparse.Namespace(**d)


# --------------------------------------------------------------------------- #
# (1) + (2) the key and the grid                                               #
# --------------------------------------------------------------------------- #
def test_the_raw_frame_rule_is_2ws_for_refav1_and_ws_plus_offset_for_refcv3(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    da, db = po.Dump(str(A), "a"), po.Dump(str(B), "b")
    assert da._raw(13) == 26, "refav1: gt_waypoints opens with f0 = 2 * t"
    assert db._raw(24) == 24 + B_RAW_OFFSET, "refcv3: RAW = ws + (n_stack - 1)"
    # and the rule is READ from the manifest on the side that publishes it
    assert "provider_to_raw_frame_offset" in db.raw_rule_note
    assert db.raw_rule == ("ws_plus_offset", B_RAW_OFFSET)


def test_the_common_grid_is_derived_in_INTEGER_RAW_FRAMES(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    da, db = po.Dump(str(A), "a"), po.Dump(str(B), "b")
    assert da.raw_offsets == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    assert db.raw_offsets == B_HORIZONS
    assert sorted(set(da.raw_offsets) & set(db.raw_offsets)) == COMMON_RAW
    rec = po.run(_args(A, B))
    g1 = rec["gates"]["G1_common_grid"]
    assert g1["common_raw_frame_offsets"] == COMMON_RAW
    assert g1["common_instants_s"] == [1.0, 2.0]
    # the prepended-origin contract: the first instant must equal the spacing
    assert g1["uniform"] and g1["first_step_equals_spacing"]
    assert g1["common_dt_s"] == pytest.approx(1.0)


def test_the_window_key_is_PROVEN_by_GT_identity_and_refused_when_it_fails(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    wk = rec["controls"]["window_key_proof"]
    assert wk["pass"], wk
    assert wk["measured_max_abs_gt_diff_m"] == pytest.approx(0.0, abs=1e-9)
    assert wk["measured_max_abs_v0_diff_mps"] == pytest.approx(0.0, abs=1e-9)
    # now poison the join: same keys, different GT -> the tool must REFUSE
    bad = _mk_a(tmp_path, name="A_bad", gt_shift=0.5)
    with pytest.raises(SystemExit, match="THE WINDOW KEY IS WRONG"):
        po.run(_args(bad, B))


def test_an_empty_intersection_refuses_and_names_the_stride_mismatch(tmp_path):
    # refav1 origins land on RAW 26/46/66; a refcv3 dump whose origins are all
    # RAW ≡ 2 (mod 5) shares NOTHING with them — the real stride-5 failure.
    A = _mk_a(tmp_path)
    B = _mk_b(tmp_path, ws_by_ep={i: [25, 45, 65] for i in range(3)})
    with pytest.raises(SystemExit, match="share ZERO windows"):
        po.run(_args(A, B))


# --------------------------------------------------------------------------- #
# (3) the floor                                                                #
# --------------------------------------------------------------------------- #
def test_the_derived_floor_reads_the_no_information_value_and_matches_the_dumped(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)          # A has NO ha0 -> derived
    rec = po.run(_args(A, B))
    assert rec["gates"]["G6_shared_floor"]["source"]["A"].startswith("DERIVED")
    assert rec["gates"]["G6_shared_floor"]["source"]["B"].startswith("DUMPED")
    c1 = rec["controls"]["floor_C1_no_information_value"]
    for side in ("A", "B"):
        assert c1[side]["pass"], c1[side]
        # y is EXACTLY zero: a constant-velocity plan does not turn
        assert c1[side]["measured_max_abs_y_m"] == pytest.approx(0.0, abs=1e-12)
        assert c1[side]["measured_max_abs_dx_m"] < po.FLOOR_TOL_M
    assert rec["controls"]["floor_C2_bit_comparable"]["pass"]
    c3 = rec["controls"]["floor_C3_derived_vs_dumped"]
    assert c3["pass"] and c3["dumped_side"] == "B", c3
    # ⛔ C3 must be a SAME-SIDE check, or it is C2 under another name
    assert "SAME SIDE" in c3["control"]


def test_the_floor_derivation_is_the_programmes_own_integrator(tmp_path):
    """`derive_ha0` must call `refav1_arm.hold_v0_controls` + `paths_from_controls`,
    not a local re-implementation — a second copy is a second convention."""
    v0 = np.array([7.0, 12.5, 3.25])
    got = po.derive_ha0(v0, A_DT, A_K, [4, 9])           # 1.0 s and 2.0 s
    want = _cv(v0, COMMON_RAW)
    assert np.abs(got - want).max() < 1e-5
    assert np.abs(got[..., 1]).max() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# (4) VOID, not negative                                                       #
# --------------------------------------------------------------------------- #
def test_a_constant_velocity_model_arm_makes_the_read_VOID_not_negative(tmp_path):
    def flat(arm, v0, offs, fi):
        # ⚠️ built through the PROGRAMME'S OWN float32 integrator, exactly as a real
        # refav1 `cl` plan is. A float64 `_cv` here would sit ~1.9e-06 m from the
        # derived floor and `identical_to` (threshold 1e-9 m) could not resolve the
        # two as identical — the same float32-floor caveat REFCV3_ARM.md records for
        # cross-CALL comparisons, one level down.
        if arm != "cl":
            return _curved(v0, offs, 0.006)
        return po.derive_ha0(np.asarray(v0, dtype=np.float64), A_DT, A_K,
                             list(range(len(offs))))
    A = _mk_a(tmp_path, name="A_flat", arm_maker=flat)
    B = _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    assert rec["void"] is True
    assert any("CONSTANT-VELOCITY" in r for r in rec["void_reasons"])
    tp = rec["profiles"]["trivial"]["arms"]["refav1:cl"]
    assert tp["constant_velocity_frac"] == pytest.approx(1.0)
    assert tp["identical_to"]["shared:ha0"] == rec["intersection"]["n_windows"]
    assert rec["gates"]["G4_profiles_non_degenerate"]["pass"] is False
    # the numbers are still banked, but every cross row carries the VOID stamp
    assert rec["cross_model_difference_of_margins"]["ade_m"]["VOID"] is True
    md = po.render_md({**rec, "_a_arm": "cl", "_b_arm": "os", "_floor": "ha0",
                       "_a_extra": [], "_b_extra": []})
    assert "VOID, NOT NEGATIVE" in md
    assert "IS the floor" in md


def test_a_degenerate_SELECTION_is_caught_where_the_trivial_profile_is_blind(tmp_path):
    """⭐ An anchor model can select ONE anchor on every window — maximally
    degenerate — while every trajectory it emits is distinct, so `trivial_frac`
    reads 0.0000 and the family table gets read as scene understanding."""
    A = _mk_a(tmp_path)
    B = _mk_b(tmp_path, name="B_one_anchor")
    for fi in range(3):
        f = B / "decisions" / f"ep{fi:03d}.npz"
        d = dict(np.load(f))
        d["sel_idx"] = np.full(d["sel_idx"].shape, 7, dtype=np.int64)
        np.savez(f, **d)
    rec = po.run(_args(A, B))
    sp = rec["profiles"]["selection"]["refcv3"]
    assert sp["n_distinct_selected"] == 1 and sp["degenerate"] is True
    # ...and the trivial profile did NOT see it — that is the whole point
    assert rec["profiles"]["trivial"]["arms"]["refcv3:os"]["constant_velocity_frac"] == 0.0
    assert rec["void"] is True


# --------------------------------------------------------------------------- #
# (5) + (6) the register's hard refusals                                       #
# --------------------------------------------------------------------------- #
def test_an_ORACLE_arm_is_refused_as_a_model_arm(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    with pytest.raises(SystemExit, match="GATE G2"):
        po.run(_args(A, B, b_arm="oracle_sel"))
    A2 = _mk_a(tmp_path, name="A_og", arms=("cl", "ha", "ol", "cl_oraclegoal"))
    with pytest.raises(SystemExit, match="GATE G2"):
        po.run(_args(A2, B, a_arm="cl_oraclegoal"))


def test_a_cross_tier_comparison_is_refused(tmp_path):
    A = _mk_a(tmp_path)
    B = _mk_b(tmp_path, name="B_T0", tier="T0")
    with pytest.raises(SystemExit, match="GATE G3"):
        po.run(_args(A, B))


def test_the_open_tier_ruling_travels_onto_the_record(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    assert rec["gates"]["G3_same_tier"]["open_ruling"]["status"] == "UNRULED"


# --------------------------------------------------------------------------- #
# (8) the identity that must be stated, not hidden                             #
# --------------------------------------------------------------------------- #
def test_the_cross_statistic_equals_the_level_difference_when_the_floor_is_shared(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    x = rec["cross_model_difference_of_margins"]["ade_m"]
    a_abs = rec["absolute_pooled_full_set"]["refav1:cl"]["ade_m"]["mean"]
    b_abs = rec["absolute_pooled_full_set"]["refcv3:os"]["ade_m"]["mean"]
    assert x["delta"] == pytest.approx(b_abs - a_abs, abs=2e-4)
    # and the record SAYS so rather than implying two different quantities
    assert "equals (B_arm - A_arm) exactly" in rec["controls"]["floor_used"]
    assert x["statistic"].startswith("(refcv3:os - shared:ha0)")


def test_each_arms_own_margin_over_the_floor_is_reported_beside_the_cross_row(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    for key in ("refav1:cl", "refcv3:os"):
        m = rec["margins_over_floor"][key]["ade_m"]
        assert m["estimator"] == "paired_episode_cluster_bootstrap"
        assert "separated" in m and "verdict" in m
    row = rec["families"]["ADE"]["metrics"]["ade_m"]
    assert row["A_margin"] is not None and row["B_margin"] is not None


# --------------------------------------------------------------------------- #
# (9) + (10) the decision families                                             #
# --------------------------------------------------------------------------- #
def test_the_decision_floor_is_the_MAJORITY_CLASS_and_is_shared(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    df = rec["controls"]["decision_floor"]
    assert "MAJORITY-CLASS" in df["what"]
    assert "route_label" in df["per_key"], df
    fl = rec["absolute_pooled_full_set"]["shared:ha0"]["STR_route_correct"]["mean"]
    assert fl == pytest.approx(df["per_key"]["route_label"]["majority_rate"], abs=1e-9)
    # the STRATEGIC family exists at all — a trajectory-only dump leaves it UNAVAILABLE
    assert rec["families"]["STRATEGIC"]["metrics"]["STR_route_correct"]["delta"] is not None


def test_the_declared_rows_are_SCOPED_to_windows_BOTH_sides_label(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)   # A labels 2/3 routes, B labels 3/3
    rec = po.run(_args(A, B))
    ctl = rec["controls"]["declared_label_identity"]["route_label"]
    assert ctl["n_labelled_A"] < ctl["n_labelled_B"]
    assert ctl["n_labelled_BOTH"] == ctl["n_labelled_A"]
    assert ctl["agree_where_both_labelled"] and ctl["usable"]
    assert "asymmetry_note" in ctl and "SCOPED" in ctl["asymmetry_note"]


def test_a_genuine_label_DISAGREEMENT_refuses_the_row_instead_of_scoring_it(tmp_path):
    """⛔ MEASURED ON REAL DUMPS 2026-09-03: `refav1_arm.py` and `refcv3_arm.py`
    assign DIFFERENT `route_label`s to the same (clip, RAW frame) on the windows
    both label. That is not a coverage difference and it is not a caveat — scoring
    two heads against 'the label' would score two different questions. The family
    must come back REFUSED **with its reason, its n and the confusion**, so the
    reader gets a work item rather than a blank."""
    A = _mk_a(tmp_path)
    B = _mk_b(tmp_path, name="B_badlabel",
              route_label_override=lambda n, fi: np.array([3, 3, 3][:n]))
    rec = po.run(_args(A, B))
    ctl = rec["controls"]["declared_label_identity"]["route_label"]
    assert ctl["agree_where_both_labelled"] is False and ctl["usable"] is False
    dis = ctl["disagreement"]
    assert dis["n_disagreeing"] == dis["of_n_both_labelled"] > 0
    assert "1->3" in dis["A_label_to_B_label_counts"]
    assert "NOT a coverage difference" in dis["meaning"]
    assert dis["work_item"]
    assert "STR_route_correct" not in rec["cross_model_difference_of_margins"]
    fam = rec["families"]["STRATEGIC"]
    assert fam["metrics"] == {} and fam["status"] == "REFUSED"
    assert fam["n"] == rec["intersection"]["n_windows"]
    assert "WORK ITEM, not a pass" in fam["estimator"]
    md = po.render_md({**rec, "_a_arm": "cl", "_b_arm": "os", "_floor": "ha0",
                       "_a_extra": [], "_b_extra": []})
    assert "STRATEGIC — family verdict: **REFUSED**" in md


def test_the_nav_echo_controls_are_emitted_for_every_declared_head(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    echo = rec["nav_echo_controls"]["what"]
    assert "STR_route:true_minus_navshuffled" in echo
    assert "STR_route:true_minus_navzero" in echo
    for per in echo.values():
        assert "cross_B_minus_A" in per
    assert "a shuffle cannot stand in for a zero" in \
        rec["nav_echo_controls"]["rule"].lower()


# --------------------------------------------------------------------------- #
# (11) power                                                                   #
# --------------------------------------------------------------------------- #
def test_the_power_warning_fires_below_the_episode_threshold(tmp_path):
    A = _mk_a(tmp_path, name="A2", n_clips=2)
    B = _mk_b(tmp_path, name="B2", n_clips=2)
    rec = po.run(_args(A, B))
    assert rec["power"]["n_episodes"] == 2
    assert rec["power"]["adequate"] is False
    assert any("not separated" in s and "UNDERPOWERED" in s
               for s in rec["what_this_does_not_establish"])


# --------------------------------------------------------------------------- #
# (12) the vocabulary                                                          #
# --------------------------------------------------------------------------- #
def test_the_record_and_the_markdown_never_say_CLOSED_LOOP(tmp_path):
    """⛔ PI ruling 2026-09-02: every arm here is OPEN LOOP, including a planner
    whose own predictor consumes its actions. The one place the phrase may appear
    is the explicit statement that closed loop is NOT what this measures."""
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    rec.update({"_a_arm": "cl", "_b_arm": "os", "_floor": "ha0",
                "_a_extra": [], "_b_extra": []})
    md = po.render_md(rec)
    blob = json.dumps(rec, default=str).lower() + md.lower()
    for bad in ("closed-loop arm", "closed loop arm", "action-closed"):
        assert bad not in blob
    # every occurrence of the phrase must be a DENIAL that it is measured here
    assert "closed loop" in blob and "open loop" in blob
    assert "NOT MEASURED ANYWHERE IN THIS RECORD" in json.dumps(rec, default=str)
    assert "does not exist in this programme" in md


def test_what_this_does_not_establish_is_never_empty_and_names_the_asymmetry(tmp_path):
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    txt = " ".join(rec["what_this_does_not_establish"])
    assert "one-shot" in txt and "iCEM" in txt or "world model" in txt
    assert "ORACLE" in txt          # refcv3's nav token will not exist at deployment
    assert len(rec["what_this_does_not_establish"]) >= 6


# --------------------------------------------------------------------------- #
# (13) the OTHER half of the PI's question, and the corpus caveat              #
# --------------------------------------------------------------------------- #
def test_the_record_carries_a_CLOSED_LOOP_section_marked_NOT_MEASURED(tmp_path):
    """The PI asked for open AND closed loop. Presenting the open-loop table alone
    as the answer is the 'true but wrong for the reader' failure class. The record
    must carry the closed-loop half explicitly — and must say it is a NOT-YET-RUN
    (the harness exists: `closedloop_drive.py` steps a bicycle from the model's own
    control and RE-RENDERS), not a NOT-POSSIBLE."""
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    cl = rec["closed_loop"]
    assert cl["measured_here"] is False
    assert cl["the_harness_exists"].endswith("closedloop_drive.py")
    assert "re-renders" in cl["what_it_does"].lower()
    for arm in ("refcv3", "refav1"):
        assert cl["for_these_arms"][arm].startswith("NOT YET RUN")
        assert "ESTIMATED" in cl["for_these_arms"][arm]
    md = po.render_md({**rec, "_a_arm": "cl", "_b_arm": "os", "_floor": "ha0",
                       "_a_extra": [], "_b_extra": []})
    assert "CLOSED LOOP — the other half of the question" in md
    assert "open-loop half" in md


def test_the_corpus_identity_control_is_UNVERIFIED_without_a_checked_key(tmp_path):
    """⚠️ Two arms both called 'b1-v72' does NOT establish they saw the same
    episodes. The control must read the runs' OWN configs and come back UNVERIFIED
    rather than implying like-for-like."""
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    ci = rec["controls"]["corpus_identity"]
    assert ci["status"] == "UNVERIFIED"
    assert "cannot be shown identical" in ci["reason"]
    assert "bit-identical on every shared window" in ci["what_is_established_instead"]
    assert "TRAINED on the same episodes" in ci["what_is_NOT_established"]

    # a run config carrying a NON-PARITY record is read and reported
    cfg = tmp_path / "b_config.json"
    cfg.write_text(json.dumps({"v2_parity": {"parity": False, "checked": False,
                                             "corpus_key": None,
                                             "clips_present": 4572,
                                             "cache_dirs": ["/root/data/train"]},
                               "v2_cache": ["/root/data/train"]}), encoding="utf-8")
    rec2 = po.run(_args(A, B, b_run_config=str(cfg)))
    pub = rec2["controls"]["corpus_identity"]["B"]["published"]
    assert pub["parity"] is False and pub["parity_checked"] is False
    assert pub["corpus_key"] is None and pub["clips_present"] == 4572
    assert rec2["controls"]["corpus_identity"]["status"] == "UNVERIFIED"


def test_the_refire_preconditions_name_the_real_final_checkpoint(tmp_path):
    """⛔ `ckpt_40284_FINAL.pt` is NEVER WRITTEN (MILESTONES = 5000/15000/20000/30000).
    The re-fire line must send the reader to `ckpt.pt` AND tell them to assert the
    step, because `ckpt.pt` is also the ROLLING checkpoint."""
    A, B = _mk_a(tmp_path), _mk_b(tmp_path)
    rec = po.run(_args(A, B))
    pre = " ".join(rec["refire_preconditions"])
    assert "ckpt_40284_FINAL.pt` — that file is never written" in pre.replace("`ckpt_40284_FINAL.pt`, NOT", "x") or "never written" in pre
    assert "ckpt['step'] == 40284" in pre
    assert "--window-stride 1" in pre
    assert "os_navzero" in pre
    assert "--a-run-config" in rec["refire_command"]
