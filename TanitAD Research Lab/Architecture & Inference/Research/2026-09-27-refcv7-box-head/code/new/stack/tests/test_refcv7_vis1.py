"""refcv7 A9 R3 -- VIS-1 (camera visibility, IGNORE semantics): the rule, the ONE split function, the sidecar and
its refusals, and the vendored z-buffer.

Every guard here is proven by MUTATION: each property is asserted AND the defect it exists for is reintroduced and
shown to go RED (CLAUDE.md "guards need mutation, not inspection").

The headline pin (SPEC_REFCV7 §14 R3): **a hidden car must NOT be a positive.** It is built from the audit's own
z-buffer (a car fully behind a truck on the same ray), not from a hand-set vis_frac, so the test exercises the
geometry that the sidecar was computed with.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data import vis1 as V                    # noqa: E402
from tanitad.data import vis1_zbuffer as VZ           # noqa: E402

IDENT_CAM = {"qx": 0.5, "qy": -0.5, "qz": 0.5, "qw": -0.5, "x": 0.0, "y": 0.0, "z": 1.5, "chunk": -1}
AUDIT_REL = ("TanitAD Research Lab", "Architecture & Inference", "Research", "2026-09-26-box-head-audit", "code")


def _audit_code_dir(repo: Path = ROOT.parent, env: str | None = None) -> Path:
    """The box-head audit's ``code/`` dir. SKIP only on NO_TREE (the checkout carries no Research Lab tree, e.g. a
    stack-only archive); FAIL on MISSING (the tree is there but the audit package is not, or TANITAD_BOX_AUDIT_DIR
    names a dir without ``vis_zbuf.py``) -- a missing banked artifact must never read as a green skip."""
    env = os.environ.get("TANITAD_BOX_AUDIT_DIR", "") if env is None else env
    if env:
        if not (Path(env) / "vis_zbuf.py").exists():
            pytest.fail(f"MISSING: TANITAD_BOX_AUDIT_DIR={env} has no vis_zbuf.py")
        return Path(env)
    if not (repo / AUDIT_REL[0]).is_dir():
        pytest.skip("NO_TREE: this checkout carries no Research Lab tree (the recorded digests are still pinned)")
    d = repo.joinpath(*AUDIT_REL)
    if not (d / "vis_zbuf.py").exists():
        pytest.fail(f"MISSING: the Research Lab tree is present but the box-head audit package is not: {d}")
    return d


def _camera():
    """A forward camera 1.5 m above the rig origin: camera z (optical axis) = rig +x, camera x = rig -y,
    camera y = rig -z (the canonical cylinder's convention). Image bounds only (no intrinsics file)."""
    R, t, obs, why, src = VZ.clip_camera("synthetic", IDENT_CAM, calib_dirs=[])
    assert why == VZ.MASK_SOURCE_BOUNDS and src is None
    return R, t, obs


def _tgt(rows):
    """rows: list of (cx, cy, l, w) -> a one-window target dict with every row REAL (valid)."""
    a = len(rows)
    box = torch.tensor([list(r) for r in rows], dtype=torch.float32)[None]
    return {"box": box, "yaw": torch.zeros(1, a), "cls": torch.zeros(1, a, dtype=torch.long),
            "valid": torch.ones(1, a, dtype=torch.bool), "occ": torch.zeros(1, a),
            "rates": torch.zeros(1, a, 3), "rates_mask": torch.zeros(1, a, dtype=torch.bool)}


# --------------------------------------------------------------------------------------------------------- #
# the vendored z-buffer                                                                                      #
# --------------------------------------------------------------------------------------------------------- #
def test_the_vendored_zbuffer_is_byte_for_byte_the_audits():
    src = Path(VZ.__file__).read_text(encoding="utf-8")
    for name, dig in VZ.AUDIT_FUNCTION_SHA256.items():
        assert VZ.function_sha256(src, name) == dig, f"{name} drifted from the audit's source"


def test_the_vendored_sha12_resolves_its_globals():
    """The vendored block's ``sha12`` reads the bare name ``hashlib``: the module must import it (the repo-wide
    test_no_unresolved_globals.py caught the first build missing it -- a NameError on first call)."""
    assert VZ.sha12("abc") == hashlib.sha256(b"abc").hexdigest()[:12] == "ba7816bf8f01"


def test_the_digest_check_goes_RED_on_a_one_character_edit():
    src = Path(VZ.__file__).read_text(encoding="utf-8")
    mutated = src.replace("hit = tmax >= np.maximum(tmin, 0.0)", "hit = tmax > np.maximum(tmin, 0.0)")
    assert mutated != src, "the mutation anchor moved -- re-point the red arm"
    assert VZ.function_sha256(mutated, "zbuffer") != VZ.AUDIT_FUNCTION_SHA256["zbuffer"]


def test_the_vendored_zbuffer_equals_the_audit_file_when_reachable():
    """The second probe: re-hash the AUDIT file itself (not only the digests recorded at vendoring)."""
    p = _audit_code_dir() / "vis_zbuf.py"
    asrc = p.read_text(encoding="utf-8")
    assert hashlib.md5(p.read_bytes()).hexdigest() == VZ.AUDIT_SOURCE["md5"]
    for name, dig in VZ.AUDIT_FUNCTION_SHA256.items():
        assert VZ.function_sha256(asrc, name) == dig


def test_the_audit_locator_skips_only_on_NO_TREE_and_fails_on_MISSING(tmp_path):
    """The helper's three outcomes, each reached: NO_TREE -> skip; tree present, package absent -> FAIL (the red arm:
    the first build skipped here too, so a deleted audit package read green); package present -> its dir."""
    with pytest.raises(pytest.skip.Exception, match="NO_TREE"):
        _audit_code_dir(repo=tmp_path, env="")
    (tmp_path / AUDIT_REL[0]).mkdir()
    with pytest.raises(pytest.fail.Exception, match="MISSING"):
        _audit_code_dir(repo=tmp_path, env="")
    d = tmp_path.joinpath(*AUDIT_REL)
    d.mkdir(parents=True)
    (d / "vis_zbuf.py").write_text("# stand-in\n", encoding="utf-8")
    assert _audit_code_dir(repo=tmp_path, env="") == d
    with pytest.raises(pytest.fail.Exception, match="MISSING"):
        _audit_code_dir(repo=tmp_path, env=str(tmp_path / "nowhere"))


def test_zbuffer_reads_known_values_on_a_single_unoccluded_box():
    """A lone car 10 m ahead is fully visible: n_vis == n_img == n_full > 0, vis_frac exactly 1.0."""
    R, t, obs = _camera()
    nf, ni, nv, vr, _ = VZ.zbuffer(np.array([[10.0, 0.0, 0.8, 4.5, 1.9, 1.6, 0.0]]), R, t, obs)
    assert nf[0] > 0 and nv[0] == nf[0] and ni[0] == nf[0] and vr[0] > 0


# --------------------------------------------------------------------------------------------------------- #
# ⭐ THE HEADLINE PIN: a hidden car must NOT be a positive                                                  #
# --------------------------------------------------------------------------------------------------------- #
def _hidden_car_scene():
    """A car at 25 m directly behind a 3.5 m-tall truck at 12 m, on the same ray. Both are trainer targets."""
    R, t, obs = _camera()
    boxes = np.array([[12.0, 0.0, 1.75, 8.0, 2.5, 3.5, 0.0],        # truck (occluder)
                      [25.0, 0.0, 0.75, 4.5, 1.8, 1.5, 0.0]])       # car, hidden behind it
    nf, ni, nv, vr, _ = VZ.zbuffer(boxes, R, t, obs)
    tgt = _tgt([(12.0, 0.0, 8.0, 2.5), (25.0, 0.0, 4.5, 1.8)])
    vis = {"n_full": torch.tensor(nf, dtype=torch.int32)[None], "n_vis": torch.tensor(nv, dtype=torch.int32)[None],
           "known": torch.ones(1, 2, dtype=torch.bool)}
    return tgt, vis, nf, nv


def test_a_hidden_car_is_NOT_a_positive():
    tgt, vis, nf, nv = _hidden_car_scene()
    assert nv[1] == 0 and nf[1] > 0, "the scene no longer hides the car -- the test lost its premise"
    sp = V.vis1_split(tgt, n_full=vis["n_full"], n_vis=vis["n_vis"], vis_known=vis["known"])
    assert bool(sp["pos"]["valid"][0, 0]) is True, "the visible truck must stay a positive"
    assert bool(sp["pos"]["valid"][0, 1]) is False, "a HIDDEN car became a VIS-1 positive"
    assert bool(sp["ignore"][0, 1]) is True, "a hidden car is IGNORE (an existing object is never a negative)"


def test_RED_ARM_the_pre_A9_target_set_makes_the_hidden_car_a_positive():
    """The deliberate regression: the trainer's pre-A9 target set (`visible_target_filter` only) keeps the hidden
    car as a POSITIVE. If this ever reads False, the headline pin above no longer discriminates anything."""
    from tanitad.refs.refc_agents import visible_target_filter
    tgt, _vis, _nf, _nv = _hidden_car_scene()
    assert bool(visible_target_filter(tgt)["valid"][0, 1]) is True


def test_RED_ARM_a_rule_without_the_visibility_cut_goes_red(monkeypatch):
    """Mutation: drop the vis_frac threshold (T_POS 0) -- the hidden car's vis_frac is 0.0, so it would pass a
    0.0 threshold if the pixel floor were also 0. With both removed the pin must fail."""
    tgt, vis, _nf, _nv = _hidden_car_scene()
    monkeypatch.setattr(V, "VIS1_T_POS", 0.0)
    monkeypatch.setattr(V, "VIS1_PX_MIN", 0)
    sp = V.vis1_split(tgt, n_full=vis["n_full"], n_vis=vis["n_vis"], vis_known=vis["known"])
    with pytest.raises(AssertionError):
        assert bool(sp["pos"]["valid"][0, 1]) is False


# --------------------------------------------------------------------------------------------------------- #
# the rule and the split                                                                                    #
# --------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_full,n_vis,want", [
    (1000, 300, True),         # exactly 0.30 and >= 100 px
    (1000, 299, False),        # 0.299
    (300, 100, True),          # 0.333, exactly 100 px
    (200, 99, False),          # 0.495 but 99 px
    (0, 0, False),             # no silhouette pixel: NaN, never positive
])
def test_the_rule_literals(n_full, n_vis, want):
    nf = torch.tensor([[n_full]], dtype=torch.int32)
    nv = torch.tensor([[n_vis]], dtype=torch.int32)
    sp = V.vis1_split(_tgt([(10.0, 0.0, 4.0, 2.0)]), n_full=nf, n_vis=nv, vis_known=torch.ones(1, 1, dtype=torch.bool))
    assert bool(sp["pos"]["valid"][0, 0]) is want
    assert bool(sp["ignore"][0, 0]) is (not want)


def test_the_literals_are_the_registered_ones():
    assert (V.VIS1_T_POS, V.VIS1_PX_MIN, V.VIS1_IGNORE_RADIUS_M) == (0.30, 100, 2.0)


def test_IGNORE_is_the_in_filter_remainder_and_out_of_filter_rows_are_neither():
    """A10 §15.1 (binding): IGNORE = in-filter non-positives (vis < 0.05 and unknown included); rows the filter
    removes stay OUTSIDE the target set -- neither positive nor ignore."""
    rows = [(10.0, 0.0, 4, 2),        # 0 in filter, visible          -> POSITIVE
            (20.0, 0.0, 4, 2),        # 1 in filter, vis 0.02 (< 0.05) -> IGNORE (NOT background)
            (30.0, 0.0, 4, 2),        # 2 in filter, vis unknown       -> IGNORE
            (70.0, 0.0, 4, 2),        # 3 beyond the 60 m decode box   -> neither
            (-8.0, 1.0, 4, 2),        # 4 behind the ego               -> neither
            (15.0, 0.0, 4, 2)]        # 5 padding (not real)           -> neither
    tgt = _tgt(rows)
    tgt["valid"][0, 5] = False
    nf = torch.tensor([[1000, 1000, 1000, 1000, 1000, 1000]], dtype=torch.int32)
    nv = torch.tensor([[900, 20, 900, 900, 900, 900]], dtype=torch.int32)
    kn = torch.tensor([[True, True, False, True, True, True]])
    sp = V.vis1_split(tgt, n_full=nf, n_vis=nv, vis_known=kn)
    assert sp["pos"]["valid"][0].tolist() == [True, False, False, False, False, False]
    assert sp["ignore"][0].tolist() == [False, True, True, False, False, False]
    c = sp["counts"]
    assert (c["n_real"], c["n_in_filter"], c["n_positive"], c["n_ignore"], c["n_outside_filter"],
            c["n_ignore_hidden"], c["n_in_filter_unknown_vis"]) == (5, 3, 1, 2, 2, 1, 1)


def test_RED_ARM_the_first_builds_every_real_row_scope_would_make_out_of_filter_rows_IGNORE():
    """The pre-A10 reading ("every other real GT row") put rows 3 and 4 above into IGNORE; A10 keeps them out.
    Reintroduce it (ignore = valid & ~pos) and the pin above goes red."""
    tgt = _tgt([(10.0, 0.0, 4, 2), (70.0, 0.0, 4, 2)])
    nf = torch.tensor([[1000, 1000]], dtype=torch.int32)
    nv = torch.tensor([[900, 900]], dtype=torch.int32)
    sp = V.vis1_split(tgt, n_full=nf, n_vis=nv, vis_known=torch.ones(1, 2, dtype=torch.bool))
    old_scope = tgt["valid"] & ~sp["pos"]["valid"]
    assert bool(old_scope[0, 1]) is True and bool(sp["ignore"][0, 1]) is False


def test_the_A10_frame_set_counts_113_positive_77_ignore():
    """C3 of G-BOX-OVERFIT under A10: per frame POS and IGN + DROP of gbo_frameset.json -> 113 / 77 (28 + 49)."""
    import json
    p = _audit_code_dir().parent / "raw" / "visibility" / "gbo_frameset.json"
    assert p.exists(), f"MISSING: {p}"
    fs = json.loads(p.read_text(encoding="utf-8"))
    assert hashlib.md5(p.read_bytes()).hexdigest() == "b291404c36f83c3e397e3b90367e8e7b"
    pos = sum(f["positive"] for f in fs["frames"])
    ign = sum(f["ignore"] + f["dropped"] for f in fs["frames"])
    assert (pos, ign, len(fs["frames"])) == (113, 77, 16)
    assert sum(f["ignore"] for f in fs["frames"]) == 28 and sum(f["dropped"] for f in fs["frames"]) == 49


def test_RED_ARM_the_audits_DROPPED_rule_would_make_vis_below_0p05_a_negative():
    """The audit proposed vis < 0.05 = background (DROPPED); A9 deliberately made it IGNORE. Reintroduce the audit
    rule and the row leaves BOTH the positive and the ignore set -- i.e. it would be trained as 'no object'."""
    tgt = _tgt([(20.0, 0.0, 4, 2)])
    nf = torch.tensor([[1000]], dtype=torch.int32)
    nv = torch.tensor([[20]], dtype=torch.int32)
    sp = V.vis1_split(tgt, n_full=nf, n_vis=nv, vis_known=torch.ones(1, 1, dtype=torch.bool))
    assert bool(sp["ignore"][0, 0])
    audit_dropped = (nv.double() / nf.double() < 0.05)
    audit_ignore = sp["ignore"] & ~audit_dropped
    assert not bool(audit_ignore[0, 0]), "the audit rule would drop the row from IGNORE (the negative A9 forbids)"


def test_the_split_shares_every_other_tensor_and_narrows_zh_mask():
    tgt = _tgt([(10.0, 0.0, 4, 2), (20.0, 0.0, 4, 2)])
    tgt["zh_mask"] = torch.ones(1, 2, dtype=torch.bool)
    nf = torch.tensor([[1000, 1000]], dtype=torch.int32)
    nv = torch.tensor([[900, 10]], dtype=torch.int32)
    sp = V.vis1_split(tgt, n_full=nf, n_vis=nv, vis_known=torch.ones(1, 2, dtype=torch.bool))
    assert sp["pos"]["box"] is tgt["box"]
    assert sp["pos"]["zh_mask"][0].tolist() == [True, False]
    assert tgt["valid"][0].tolist() == [True, True], "the caller's dict was mutated"


# --------------------------------------------------------------------------------------------------------- #
# the sidecar: deterministic, verified, and it REFUSES                                                       #
# --------------------------------------------------------------------------------------------------------- #
def _arrays(track0=7, cx0=10.0):
    return {"row": np.array([0, 2], np.int16), "track": np.array([track0, 9], np.int64),
            "n_full": np.array([1000, 500], np.int32), "n_img": np.array([1000, 500], np.int32),
            "n_vis": np.array([800, 40], np.int32), "vis_rows": np.array([30, 5], np.int32),
            "cx": np.array([cx0, 30.0], np.float32), "cy": np.array([0.5, -1.0], np.float32),
            "frame_clip": np.array([0, 0], np.int32), "frame_f": np.array([7, 8], np.int32),
            "frame_ptr": np.array([0, 2, 2], np.int64), "frame_n_zh": np.array([3, 0], np.int32),
            "frame_n_valid": np.array([3, 0], np.int32),
            "clip_sha12": np.array(["abcdef012345"], dtype="<U12"), "clip_frame_ptr": np.array([0, 2], np.int64),
            "clip_mask_ftheta": np.array([True]), "clip_split": np.array(["train"], dtype="<U5")}


def _meta():
    return {"store_scope": dict(V.STORE_SCOPE), "n_stack": 3, "window": 8, "splits": {"train": {}}}


def test_sidecar_round_trip_and_deterministic_bytes(tmp_path):
    d1 = V.write_sidecar(tmp_path / "a.npz", _arrays(), _meta())
    d2 = V.write_sidecar(tmp_path / "b.npz", _arrays(), _meta())
    assert d1["sha256"] == d2["sha256"] and d1["content_sha256"] == d2["content_sha256"]
    sc = V.VIS1Sidecar(tmp_path / "a.npz")
    assert (sc.n_clips, sc.n_frames, sc.n_rows) == (1, 2, 2)
    assert sc.frame("abcdef012345", 7) == (0, 2) and sc.frame("abcdef012345", 8) == (2, 2)
    assert sc.frame("abcdef012345", 9) is None and sc.frame("000000000000", 7) is None
    box = np.array([[10.0, 0.5, 4, 2], [60.5, 30.0, 4, 2], [30.0, -1.0, 4, 2]], np.float32)
    nf, nv, kn = V.vis1_block_for_rows(sc, "abcdef012345", 7, box=box, track_ids=["7", "8", "9"],
                                       zh_mask=np.array([True, True, True]), pad=5)
    assert nf.tolist() == [1000, 0, 500, 0, 0] and nv.tolist() == [800, 0, 40, 0, 0]
    assert kn.tolist() == [True, False, True, False, False]     # row 1 is outside the store scope


def test_sidecar_REFUSES_every_mismatch(tmp_path):
    V.write_sidecar(tmp_path / "s.npz", _arrays(), _meta())
    sc = V.VIS1Sidecar(tmp_path / "s.npz")
    box = np.array([[10.0, 0.5, 4, 2]], np.float32)
    with pytest.raises(V.VIS1SidecarError, match="ABSENT from the"):             # frame absent
        V.vis1_block_for_rows(sc, "abcdef012345", 99, box=box, track_ids=["7"], zh_mask=[True], pad=2)
    with pytest.raises(V.VIS1SidecarError, match="track"):                      # different join
        V.vis1_block_for_rows(sc, "abcdef012345", 7, box=box, track_ids=["8"], zh_mask=[True], pad=2)
    with pytest.raises(V.VIS1SidecarError, match="centre"):                     # different clock / frame
        V.vis1_block_for_rows(sc, "abcdef012345", 7, box=box + np.float32(0.25), track_ids=["7"],
                              zh_mask=[True], pad=2)
    box2 = np.array([[10.0, 0.5, 4, 2], [12.0, 0.0, 4, 2]], np.float32)       # in-scope row 1 not stored
    with pytest.raises(V.VIS1SidecarError, match="ABSENT"):
        V.vis1_block_for_rows(sc, "abcdef012345", 7, box=box2, track_ids=["7", "5"], zh_mask=[True, True], pad=2)
    with pytest.raises(V.VIS1SidecarError, match="not in the VIS-1 sidecar"):
        sc.require_clips(["abcdef012345", "ffffffffffff"], where="test")
    with pytest.raises(V.VIS1SidecarError, match="not found"):
        V.VIS1Sidecar(tmp_path / "missing.npz")
    m = _meta()
    m["store_scope"] = {**V.STORE_SCOPE, "x_max_m": 60.0}
    V.write_sidecar(tmp_path / "narrow.npz", _arrays(), m)
    with pytest.raises(V.VIS1SidecarError, match="scope"):
        V.VIS1Sidecar(tmp_path / "narrow.npz")


def test_a_row_without_a_3d_label_is_unknown_not_guessed(tmp_path):
    V.write_sidecar(tmp_path / "s.npz", _arrays(), _meta())
    sc = V.VIS1Sidecar(tmp_path / "s.npz")
    box = np.array([[10.0, 0.5, 4, 2]], np.float32)
    nf, nv, kn = V.vis1_block_for_rows(sc, "abcdef012345", 7, box=box, track_ids=["7"], zh_mask=[False], pad=1)
    assert kn.tolist() == [False] and nf.tolist() == [0]


def test_store_scope_contains_the_trainer_filter_with_margin():
    """Every row the trainer's float32 filter keeps must be inside the float64 store scope (the margin's job)."""
    from tanitad.refs.refc_agents import visible_target_filter
    g = torch.Generator().manual_seed(0)
    cx = torch.rand(20000, generator=g) * 70 - 5
    cy = torch.rand(20000, generator=g) * 40 - 20
    tgt = {"box": torch.stack([cx, cy, torch.ones_like(cx), torch.ones_like(cx)], -1)[None],
           "valid": torch.ones(1, 20000, dtype=torch.bool)}
    keep = visible_target_filter(tgt)["valid"][0].numpy()
    inside = V.in_store_scope(cx.numpy(), cy.numpy())
    assert keep.sum() > 1000 and not (keep & ~inside).any()
