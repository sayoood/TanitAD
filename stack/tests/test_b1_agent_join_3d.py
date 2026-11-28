"""The 3-D agent join: the annotation, its controls, and the loader seam.

⛔ **THE POINT OF THIS FILE IS THE MUTATION TABLE.** A control that is only
read is not proven (project memory: *"Guards need mutation, not inspection"* --
an AST census once read 0 suspects on BOTH the fixed and the broken trainer).
So :func:`test_the_mutation_table` REINTRODUCES each defect on a real build of a
synthetic corpus and asserts that exactly the predicted control turns RED **and
that the others stay green** -- a mutation that reddens everything proves
nothing about which control is load-bearing.

The fixture is built by the **2-D builder itself** (``join_clip_raw``) rather
than hand-rolled, so the test cannot drift from the artifact it is about.

MEASURED equivalents on the real corpus (139 clips / 26,394 lines / 905,512
agents) are in ``TanitAD Research Lab/Benchmarks & Evals/Research/
2026-09-17-b1-agent-join-3d/RESULT.md``; the real-data tests here are
env-gated (``TANITAD_JOIN3D``, ``TANITAD_JOIN2D``) on the ``test_bev_lift.py``
convention.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "stack"), str(REPO / "stack" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

b3 = pytest.importorskip("build_b1_agent_join_3d")
from tanitad.data.agent_cuboid_gt import (              # noqa: E402
    AgentJoin3D, CuboidGTError, base_from_centre, open_join3d, zh_for_frame,
)

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")


# ===========================================================================
# the fixture: a synthetic corpus, joined by the REAL 2-D builder
# ===========================================================================
#: ground-standing by construction: centre = h/2 above rig z = 0.
_SPEC = [
    # track, class, l,    w,    h,     base_z, x0,    y0
    ("t0", "automobile", 4.5, 1.9, 1.50, 0.00, 12.0, 0.5),
    ("t1", "automobile", 4.2, 1.8, 1.44, 0.00, 25.0, -3.0),
    ("t2", "heavy_truck", 9.0, 2.5, 3.50, 0.00, 40.0, 3.5),
    ("t3", "bus", 11.0, 2.6, 3.30, 0.00, -18.0, 1.0),   # behind: occ = 1
]
_N_FRAMES = 6
_DT = 0.1


def _obs_arrays(seed: int = 0):
    """One clip's ``obstacle.offline`` rows: 4 tracks x 6 samples, rig frame."""
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("timestamp_us", "source", "track_id", "center_x",
                            "center_y", "center_z", "size_x", "size_y",
                            "size_z", "orientation_x", "orientation_y",
                            "orientation_z", "orientation_w", "label_class",
                            "reference_frame", "reference_frame_timestamp_us")}
    for tid, cls, l_m, w_m, h_m, base, x0, y0 in _SPEC:
        for i in range(_N_FRAMES):
            t_us = int(round(i * _DT * 1e6))
            yaw = 0.05 * i + 0.01 * seed
            cols["timestamp_us"].append(t_us)
            cols["source"].append("scene:obstacles:autolabels:v2")
            cols["track_id"].append(tid)
            cols["center_x"].append(float(x0 + 1.3 * i))
            cols["center_y"].append(float(y0 + 0.05 * i))
            # ⭐ CENTRE, not base: this is the convention under test.
            cols["center_z"].append(float(base + h_m / 2.0
                                          + 0.002 * rng.standard_normal()))
            cols["size_x"].append(float(l_m))
            cols["size_y"].append(float(w_m))
            cols["size_z"].append(float(h_m))
            cols["orientation_x"].append(0.0)
            cols["orientation_y"].append(0.0)
            cols["orientation_z"].append(float(math.sin(yaw / 2)))
            cols["orientation_w"].append(float(math.cos(yaw / 2)))
            cols["label_class"].append(cls)
            cols["reference_frame"].append("rig")
            cols["reference_frame_timestamp_us"].append(t_us)
    return cols


def _ego_at_rest():
    t = np.arange(0, (_N_FRAMES + 2)) * _DT
    return {"timestamp": (t * 1e6).astype(np.int64),
            "x": np.zeros_like(t), "y": np.zeros_like(t),
            "qx": np.zeros_like(t), "qy": np.zeros_like(t),
            "qz": np.zeros_like(t), "qw": np.ones_like(t)}


def make_corpus(tmp_path: Path, n_clips: int = 2):
    """``(join_2d_path, obstacle_dir)`` -- the 2-D join written by the 2-D
    builder's own ``join_clip_raw``, so the fixture IS the artifact's format."""
    import build_b1_agent_join as b2
    from build_obstacle_join import EgoTrack

    obs_dir = tmp_path / "obstacle"
    obs_dir.mkdir(parents=True, exist_ok=True)
    join = tmp_path / "join2d.jsonl"
    ego = EgoTrack(_ego_at_rest())
    poses = np.zeros((_N_FRAMES, 4), dtype=np.float64)
    t_over = np.arange(_N_FRAMES) * _DT
    lines = []
    for c in range(n_clips):
        cid = "clip%08d-0000-0000-0000-00000000%04d" % (c, c)
        cols = _obs_arrays(seed=c)
        pq.write_table(pa.table(cols), obs_dir / (cid + ".parquet"))
        recs, _st, _t = b2.join_clip_raw(cid, poses, ego, cols, tol_s=0.06,
                                         hfov_deg=120.0, n_stack=3,
                                         t_s_override=t_over)
        assert recs, "the fixture produced no join lines"
        lines += [json.dumps(r, separators=(",", ":")) for r in recs]
    join.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return join, obs_dir


def _build(tmp_path, mutate="none", **kw):
    join, obs_dir = make_corpus(tmp_path / ("m_" + mutate.replace("-", "_")))
    out = tmp_path / ("out_" + mutate.replace("-", "_") + ".jsonl")
    return b3.build(join, obs_dir, out, tol_s=0.06, bevgt_dir=None,
                    mutate=mutate, verbose=False, **kw), out, join


# ===========================================================================
# the clean build
# ===========================================================================
def test_a_clean_build_passes_every_gated_control(tmp_path):
    meta, out, _ = _build(tmp_path)
    assert meta["all_controls_passed"] is True
    assert out.is_file()
    s = meta["summary"]
    assert s["n_agents"] == s["n_agents_with_zh"] > 0
    assert s["n_agents_without_zh"] == 0


def test_the_key_and_every_existing_field_survive(tmp_path):
    """⛔ A consumer joining on the OLD key must keep working: the 3-D line is
    the 2-D line plus two keys per agent, and nothing else."""
    meta, out, join = _build(tmp_path)
    two = [json.loads(x) for x in join.read_text(encoding="utf-8").splitlines() if x]
    three = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x]
    assert len(two) == len(three)
    for a, b in zip(two, three):
        assert a.keys() == b.keys()
        assert (a["clip_id"], a["frame"], a["frame_idx"], a["t_s"]) == \
               (b["clip_id"], b["frame"], b["frame_idx"], b["t_s"])
        for x, y in zip(a["agents"], b["agents"]):
            assert set(y) - set(x) == {"cz", "h"}
            for k in x:
                assert x[k] == y[k], k


def test_strip_zh_is_byte_identical_on_every_line_not_a_sample(tmp_path):
    meta, out, join = _build(tmp_path)
    raw2 = [x for x in join.read_text(encoding="utf-8").splitlines() if x]
    raw3 = [x for x in out.read_text(encoding="utf-8").splitlines() if x]
    assert len(raw3) == len(raw2) > 0
    for a, b in zip(raw2, raw3):
        assert b3.dumps_line(b3.strip_zh(json.loads(b))) == a
    c2 = meta["controls"]["C2_strip_zh_is_byte_identical_to_the_2d_line"]
    assert c2["n_fail"] == 0 and c2["n_lines"] == len(raw2)
    assert c2["scope"] == "ALL lines, not a sample"


def test_the_serializer_is_the_2d_builders_own(tmp_path):
    """C1 is only meaningful if both sides are not wrong the same way."""
    rec = {"clip_id": "x", "frame": 1, "agents": [{"cx": 1.25, "occ": 0}]}
    assert b3.dumps_line(rec) == json.dumps(rec, separators=(",", ":"))


# ===========================================================================
# ⛔ THE MUTATION TABLE
# ===========================================================================
def test_every_mutation_names_the_control_it_must_redden():
    assert set(b3.MUTATIONS) == set(b3._MUTATIONS)
    assert b3.MUTATIONS["none"] is None
    for name, targets in b3.MUTATIONS.items():
        if name == "none":
            continue
        assert targets, f"{name} claims to prove nothing"


@pytest.mark.parametrize("mutate", [m for m in b3.MUTATIONS if m != "none"])
def test_the_mutation_table(tmp_path, mutate):
    """Each defect reddens EXACTLY the controls it is declared to redden.

    ⭐ The "and no others" half is the part that matters. A mutation that turns
    every control red would prove that none of them is the load-bearing one --
    the instrument would be reporting the run, not the defect.
    """
    clean, _o, _j = _build(tmp_path / "clean")
    dirty, _o2, _j2 = _build(tmp_path / "dirty", mutate=mutate)
    red = {k for k, v in dirty["controls"].items() if v.get("passed") is False}
    green_clean = {k for k, v in clean["controls"].items()
                   if v.get("passed") is True}
    # ⛔ A control that did not RUN cannot be expected to redden, and pretending
    # otherwise is how a table starts asserting things it never measured. C4
    # needs a LiDAR reference this synthetic corpus has none of; it is proven
    # red on REAL data instead, and the RESULT records that run.
    expect = set()
    for k in b3.MUTATIONS[mutate]:
        if dirty["controls"][k].get("passed") is None:
            assert dirty["controls"][k].get("available") is False, k
            continue
        expect.add(k)
    assert expect, f"{mutate} reddened nothing that this fixture can run"
    assert expect <= red, (
        f"{mutate} did NOT redden {sorted(expect - red)} -- that control is a "
        f"control-shaped hole")
    assert red <= expect, (
        f"{mutate} reddened {sorted(red - expect)} as well; the table is wrong "
        f"or the mutation is too broad to attribute")
    # every control it must NOT touch was green before and stays green
    for k in green_clean - expect:
        assert dirty["controls"][k].get("passed") is True, k
    assert dirty["all_controls_passed"] is False


def test_the_centre_vs_base_flip_moves_the_control_by_half_a_height(tmp_path):
    """⛔ THE CONTROL THAT READS A KNOWN VALUE. The flip must not merely fail --
    it must fail BY THE QUANTITY IT IS A FLIP OF, roughly half a vehicle height.

    ⚠️ "Roughly", and the band is wide on purpose. The statistic is a MEDIAN and
    the height mixture is bimodal (a car 1.5 m, a truck 3.5 m), so
    ``median(cz - h)`` is not ``median(cz - h/2) - median(h)/2``: MEASURED on
    4,000 real lines the shift is 0.861 m against a median ``h/2`` of 0.765 m.
    Asserting the exact identity would be a test that passes only on a
    unimodal fixture and then breaks on the corpus it is about.
    """
    clean, _o, _j = _build(tmp_path / "c")
    dirty, _o2, _j2 = _build(tmp_path / "d", mutate="emit-base-as-centre")
    c, d = clean["controls"]["C3_road_plane"], dirty["controls"]["C3_road_plane"]
    assert c["passed"] is True and d["passed"] is False
    h_med = clean["controls"]["C8_rival_convention"]["h_median_m"]
    moved = abs(d["median_m"] - c["median_m"])
    assert 0.4 * h_med <= moved <= 0.75 * h_med, (moved, h_med)
    assert moved > b3.ROAD_PLANE_TOL_M * b3.MIN_MUTATION_SEPARATION


def test_misattach_is_the_failure_C5_structurally_cannot_see(tmp_path):
    """Right agent SET, wrong PAIRING. C5 compares sequences and is satisfied;
    only C6 (the picked row's own sizes vs the line's) can see it."""
    dirty, _o, _j = _build(tmp_path, mutate="misattach")
    assert dirty["controls"]["C5_track_sequence_equals_the_line"]["passed"] is True
    assert dirty["controls"]["C6_picked_row_sizes_equal_the_line"]["passed"] is False
    assert dirty["controls"]["C6_picked_row_sizes_equal_the_line"]["n_fail"] > 0


def test_the_road_plane_tolerance_is_adopted_not_invented():
    """The band is the LiDAR artifact's own ground-plane check, quoted."""
    assert b3.ROAD_PLANE_TOL_M == 0.20
    assert "ground_peak_z_m" in b3.control_road_plane(b3.BaseStats())["tol_source"]


def test_protruding_object_is_excluded_from_the_ground_control():
    """It is airborne BY DEFINITION; pooling it would move the statistic."""
    assert "protruding_object" not in b3.GROUND_STANDING_CLASSES
    assert "person" not in b3.GROUND_STANDING_CLASSES     # not a VEHICLE base
    assert "automobile" in b3.GROUND_STANDING_CLASSES


def test_an_empty_or_z_free_build_is_refused(tmp_path):
    out = tmp_path / "empty.jsonl"
    out.write_text("", encoding="utf-8")
    src = tmp_path / "src.jsonl"
    src.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        b3.assert_content(out, 3, False, raw_2d=src)


# ===========================================================================
# the loader seam: BOTH paths
# ===========================================================================
def _join3d_file(tmp_path):
    meta, out, join = _build(tmp_path)
    return out, join, meta


def test_the_loader_prefers_the_3d_join_when_present(tmp_path):
    out, _join, _meta = _join3d_file(tmp_path)
    j = AgentJoin3D.open(out)
    assert len(j) > 0 and j.n_agents > 0
    key = next(iter(j._idx))
    ids = list(j._idx[key])
    cz, h, mask = zh_for_frame(key[0], key[1], ids, join3d=j)
    assert mask.all()
    assert np.isfinite(cz).all() and (h > 0).all()
    # ⭐ the values are the CENTRE: the base lands on the road plane
    assert abs(float(np.median(base_from_centre(cz, h)))) < 0.05


def test_the_loader_falls_back_to_the_masked_2d_behaviour_when_absent(tmp_path):
    """⛔ ABSENCE IS A STATE, NOT A ZERO -- the mask is all-False and the values
    are meaningless, which is exactly what ``box3d_head`` needs to report
    ``n_z == 0`` instead of training on a fabricated height."""
    out, _join, _meta = _join3d_file(tmp_path)
    j = AgentJoin3D.open(out)
    key = next(iter(j._idx))
    ids = list(j._idx[key])

    cz, h, mask = zh_for_frame(key[0], key[1], ids, join3d=None)
    assert mask.shape == (len(ids),) and not mask.any()
    assert not cz.any() and not h.any()

    # a frame the join does not carry is the same state
    _c, _h, m2 = zh_for_frame(key[0], 10 ** 9, ids, join3d=j)
    assert not m2.any()
    # an agent the line does not carry is masked OFF, not zeroed in
    _c, _h, m3 = zh_for_frame(key[0], key[1], ids + ["no-such-track"], join3d=j)
    assert m3[:len(ids)].all() and not m3[len(ids)]


def test_opening_a_2d_join_as_a_3d_one_is_REFUSED(tmp_path):
    """Silently indexing it would mask every target while REPORTING a 3-D join
    was found -- an instrument that cannot report the answer it is cited for."""
    _out, join2d, _meta = _join3d_file(tmp_path)
    with pytest.raises(CuboidGTError, match="that is a 2-D join"):
        AgentJoin3D.open(join2d)


def test_open_join3d_is_none_when_unset_or_missing(tmp_path, monkeypatch):
    monkeypatch.delenv(b3.CZ_KEY.upper(), raising=False)
    monkeypatch.delenv("TANITAD_AGENT_JOIN3D", raising=False)
    assert open_join3d(None) is None
    assert open_join3d(tmp_path / "nope.jsonl.xz") is None


def test_open_join3d_reads_the_env_var(tmp_path, monkeypatch):
    out, _join, _meta = _join3d_file(tmp_path)
    monkeypatch.setenv("TANITAD_AGENT_JOIN3D", str(out))
    j = open_join3d(None)
    assert j is not None and j.n_agents > 0


def test_base_from_centre_has_exactly_one_spelling():
    cz = np.array([0.75, 1.65]); h = np.array([1.5, 3.3])
    assert np.allclose(base_from_centre(cz, h), [0.0, 0.0])
    # applying it to an ALREADY-base z is the retraction's defect, and it is
    # visible as a half-height sink -- the reason the helper exists.
    assert np.allclose(base_from_centre(base_from_centre(cz, h), h), -h / 2)


# ===========================================================================
# no session path, no dead mount, in the files THIS branch owns
# ===========================================================================
_H = "[0-9a-f]"
_SEP = "[/" + chr(92) + chr(92) + "]"
_UUID = re.compile(f"{_H}{{8}}-{_H}{{4}}-{_H}{{4}}-{_H}{{4}}-{_H}{{12}}", re.I)
_SCRATCH = re.compile("App" + "Data" + _SEP + "Local" + _SEP + "Temp"
                      + _SEP + "clau" + "de", re.I)
_DEAD = re.compile("G" + "--" + "Meine" + "|G:" + _SEP + "Meine", re.I)
_HOME = re.compile("C:" + _SEP + "Users" + _SEP + "Admin", re.I)
_ENV_OK = re.compile(r"environ\.get|getenv|os\.environ")

OWNED = (
    "stack/scripts/build_b1_agent_join_3d.py",
    "stack/tanitad/data/agent_cuboid_gt.py",
    "stack/tests/test_b1_agent_join_3d.py",
    "TanitAD Research Lab/Benchmarks & Evals/Research/"
    "2026-09-17-b1-agent-join-3d/RESULT.md",
)


@pytest.mark.parametrize("rel", OWNED)
def test_no_session_path_no_dead_mount(rel):
    """⛔ This class has blocked two landings in one day. Nothing can tell a
    session UUID from a clip UUID by looking at it, so clip ids appear as sha12
    only. The patterns are assembled from fragments so this file cannot flag
    itself."""
    p = REPO / rel
    if not p.is_file():
        pytest.skip(f"{rel} not present in this worktree")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    md = p.suffix.lower() == ".md"
    for i, line in enumerate(lines):
        assert not _UUID.search(line), f"{rel}:{i + 1} UUID-shaped string"
        assert not _SCRATCH.search(line), f"{rel}:{i + 1} session scratchpad"
        assert not _DEAD.search(line), f"{rel}:{i + 1} dead G: mount"
        if _HOME.search(line) and not md:
            prev = lines[i - 1] if i else ""
            assert _ENV_OK.search(line) or _ENV_OK.search(prev), \
                f"{rel}:{i + 1} absolute home path with no env-var escape"


def test_the_guard_would_catch_each_pattern(tmp_path):
    """MUTATION: the guard is proven by feeding it the thing it forbids."""
    cases = {
        "uuid": "x = '" + "0" * 8 + "-" + "0" * 4 + "-" + "0" * 4 + "-"
                + "0" * 4 + "-" + "0" * 12 + "'",
        "scratch": "p = 'C:/App" + "Data/Local/Temp/clau" + "de/x'",
        "dead": "p = 'G" + ":/Meine Ablage/x'",
    }
    pats = {"uuid": _UUID, "scratch": _SCRATCH, "dead": _DEAD}
    for k, text in cases.items():
        assert pats[k].search(text), f"the {k} pattern does not match its own fixture"
    # ⚠️ ASSEMBLED, never spelled: a literal home path here flags THIS file, and
    # the tempting fix (exempt the guard) puts a hole exactly where the guard is.
    home = "C:" + "/Users" + "/Admin" + "/x"
    assert _HOME.search(home)
    assert not _ENV_OK.search("p = '" + home + "'")
    assert _ENV_OK.search("p = os.environ.get('X', '" + home + "')")


# ===========================================================================
# REAL DATA (env-gated, `test_bev_lift.py` convention)
# ===========================================================================
JOIN3D = os.environ.get("TANITAD_JOIN3D", "")
JOIN2D = os.environ.get("TANITAD_JOIN2D", "")
_real = pytest.mark.skipif(not (JOIN3D and Path(JOIN3D).is_file()),
                           reason="set TANITAD_JOIN3D to the banked 3-D join")


@_real
def test_real_join3d_meta_says_every_control_passed():
    meta = json.loads(Path(JOIN3D + ".meta.json").read_text(encoding="utf-8"))
    assert meta["all_controls_passed"] is True
    assert meta["summary"]["zh_coverage"] == 1.0
    assert meta["convention_verdict"]["center_z_is"] == "the cuboid CENTRE"
    assert meta["controls"]["C8_rival_convention"]["verdict"] == "CENTRE"


@_real
@pytest.mark.skipif(not (JOIN2D and Path(JOIN2D).is_file()),
                    reason="set TANITAD_JOIN2D to the banked 2-D join")
def test_real_join3d_strips_back_to_the_banked_2d_join_on_every_line():
    """⛔ ASSERTED, NOT SAMPLED -- all 26,394 lines."""
    n = 0
    with b3.open_lines(JOIN3D) as a, b3.open_lines(JOIN2D) as b:
        for three, two in zip(a, b):
            assert b3.dumps_line(b3.strip_zh(json.loads(three))) == two.rstrip("\n")
            n += 1
        assert a.readline() == "" and b.readline() == "", "line counts differ"
    assert n > 0


@_real
def test_real_heights_are_the_physical_ones():
    """A car is about 1.5 m and a truck about 3 m. If this disagrees, the join
    is wrong -- and the disagreement, not the join, is the finding."""
    meta = json.loads(Path(JOIN3D + ".meta.json").read_text(encoding="utf-8"))
    c = meta["census_by_class"]
    assert 1.3 <= c["automobile"]["h_median_m"] <= 1.8
    assert 2.7 <= c["heavy_truck"]["h_median_m"] <= 4.2
    assert 2.7 <= c["bus"]["h_median_m"] <= 4.2
    assert 1.4 <= c["person"]["h_median_m"] <= 2.0
    for k in ("automobile", "bus", "heavy_truck"):
        assert abs(c[k]["base_median_m"]) <= 0.2, (k, c[k]["base_median_m"])
    # the exception that proves the rule: airborne by definition
    assert c["protruding_object"]["base_median_m"] > 0.4
