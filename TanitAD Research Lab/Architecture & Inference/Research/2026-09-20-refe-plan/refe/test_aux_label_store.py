"""Tests for refe/aux_label_store.py on SYNTHETIC maps (no nuPlan, no GPU, no network).

Two layers.
  1. `conformance(ALS)` -- an INDEPENDENT oracle for the label (its own rotation maths, its own analytic SDF values, its own
     strict point-in-rectangle in the ego frame). The real module must pass it, and every deliberate regression must make it FAIL:
       y axis flipped | x/y swapped | padding removed | clip changed (smaller and int8-wrapping larger) | degenerate-grid
       guard removed | wrong map for a log.
     A check that cannot be turned RED proves nothing, so test_mutations_each_turn_red asserts the SPECIFIC check that trips.
  2. The store: shard format, resume by key, partial resume, stale map, navtest exclusion (also from a stale shard), --shard
     partition, failure accounting, the memory guard, and a spawn-pool run that must reproduce the single-process store.

Run (driverl-eval venv has shapely + scipy; it has no pytest):
    PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe test_aux_label_store.py   -> ZZTEST_PASS <n>
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sqlite3
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aux_label_store as ALS  # noqa: E402

BACKEND_SPEC = "test_aux_label_store:make_fake_backend"
FAIL_SPEC = "test_aux_label_store:make_failing_backend"


# ------------------------------------------------------------------ a fake map backend ------------------------------------
class FakeBackend:
    """mapA = a road along the global x axis (y in [-3.5, 3.5]); mapB = a road 100 m to the left (y in [100, 107])."""

    def __init__(self, fail_x=None):
        self.maps = {"mapA": [shapely.box(-1e4, -3.5, 1e4, 3.5)], "mapB": [shapely.box(-1e4, 100.0, 1e4, 107.0)]}
        self.fail_x = fail_x

    def polys(self, map_name, cx, cy, radius=100.0):
        if self.fail_x is not None and abs(cx - self.fail_x) < 1e-6:
            raise RuntimeError("synthetic map failure")
        return self.maps[map_name]


def make_fake_backend(maps_root):
    return FakeBackend()


def make_failing_backend(maps_root):
    return FakeBackend(fail_x=ALS.ego_centre((80.0, 1.0, 0.1))[0])


# ------------------------------------------------------------------ the independent oracle -------------------------------
def _rot(px, py, pose):
    x, y, h = pose
    return (x + px * math.cos(h) - py * math.sin(h), y + px * math.sin(h) + py * math.cos(h))


def _centres():
    ix, iy = np.arange(160), np.arange(128)
    return np.meshgrid(-8.0 + (ix + 0.5) * 0.5, -32.0 + (iy + 0.5) * 0.5, indexing="ij")           # ego-frame cell centres


def _ego_rect_polygon(x0, x1, y0, y1, pose):
    return shapely.Polygon([_rot(x0, y0, pose), _rot(x1, y0, pose), _rot(x1, y1, pose), _rot(x0, y1, pose)])


def _expect_mask(x0, x1, y0, y1):
    xe, ye = _centres()
    return (xe > x0) & (xe < x1) & (ye > y0) & (ye < y1)


def _cell(x_e, y_e):
    return int(round((x_e + 8.0) / 0.5 - 0.5)), int(round((y_e + 32.0) / 0.5 - 0.5))


def conformance(M) -> list:
    """Run every label check against module-like `M`; return the NAMES of the checks that failed (empty = conforms)."""
    bad = []

    def build(name, polys, pose):
        try:
            return M.build_label(polys, pose)
        except Exception as e:                                    # e.g. the int8 guard: report it as a failed check, not a crash
            bad.append(f"{name}:build_raised:{type(e).__name__}")
            return np.zeros((160, 128), dtype=bool), np.zeros((160, 128), dtype=np.int8)

    def check(name, fn):
        try:
            if not fn():
                bad.append(name)
        except Exception:
            bad.append(name + ":raised")

    # C1: exact values on a rectangle ahead of the ego (heading 0, so only the SDF maths and the x/y roles matter)
    pose = (500.0, -200.0, 0.0)
    mask, sdf = build('c1', [_ego_rect_polygon(10.0, 30.0, -4.0, 4.0, pose)], pose)
    check("shape_dtype", lambda: mask.shape == (160, 128) and mask.dtype == bool and sdf.shape == (160, 128) and sdf.dtype == np.int8)
    check("mask_rect_x_ahead", lambda: np.array_equal(mask, _expect_mask(10.0, 30.0, -4.0, 4.0)))
    ix, iy = _cell(20.25, 0.25)                                  # deepest direction is y: 3.75 m to the +4 edge
    check("sdf_inside_exact", lambda: abs(sdf[ix, iy] * 0.1 - 3.75) <= 0.05 + 1e-9)
    ix, iy = _cell(5.25, 0.25)                                   # 4.75 m in front of the rectangle's rear edge
    check("sdf_outside_exact", lambda: abs(sdf[ix, iy] * 0.1 + 4.75) <= 0.05 + 1e-9)
    ix, iy = _cell(15.25, 3.75)                                  # a quarter metre inside the +y edge
    check("sdf_near_edge", lambda: 2 <= sdf[ix, iy] <= 3)
    check("sdf_sign_and_nonzero", lambda: bool(((sdf > 0) == mask).all() and (sdf != 0).all()))

    # C2: orientation. Heading pi/2 and a generic heading; a box ahead (+x) and to the LEFT (+y) of the rear axle.
    for hdg in (math.pi / 2, 0.7):
        pose = (100.0, 50.0, hdg)
        mask, _ = build('c2', [_ego_rect_polygon(20.1, 30.1, 5.1, 10.1, pose)], pose)
        check(f"frame_left_ahead_h{hdg:.2f}", lambda: np.array_equal(mask, _expect_mask(20.1, 30.1, 5.1, 10.1)))

    # C3: padding. A strip just outside the window must still shape the SDF of the cells next to it.
    pose = (0.0, 0.0, 0.0)
    _, sdf = build('c3', [_ego_rect_polygon(72.5, 100.0, -30.0, 30.0, pose)], pose)
    for x_e, want in ((71.75, -0.75), (67.25, -5.25), (62.25, -10.25)):
        ix, iy = _cell(x_e, 0.25)
        check(f"padding_{x_e}", lambda: abs(sdf[ix, iy] * 0.1 - want) <= 0.05 + 1e-9)
    ix, iy = _cell(57.25, 0.25)                                  # 15.25 m away > clip
    check("clip_far_outside", lambda: sdf[ix, iy] == -127)

    # C4: clip at +-12.7 m on both sides, and the degenerate grids
    pose = (0.0, 0.0, 0.0)
    _, sdf = build('c4a', [_ego_rect_polygon(-5.0, 200.0, -100.0, 100.0, pose)], pose)
    ix, iy = _cell(40.25, 0.25)
    check("clip_deep_inside", lambda: sdf[ix, iy] == 127)
    ix, iy = _cell(0.25, 0.25)
    check("near_boundary_inside", lambda: abs(sdf[ix, iy] * 0.1 - 5.25) <= 0.05 + 1e-9)
    _, sdf = build('c4b', [_ego_rect_polygon(80.0, 90.0, -20.0, 20.0, pose)], pose)
    check("clip_deep_outside", lambda: bool((sdf[:40, 60:70] == -127).all()))
    _, sdf = build('c4c', [], pose)                              # no drivable polygon at all
    check("degenerate_all_outside", lambda: bool((sdf == -127).all()))
    _, sdf = build('c4d', [_ego_rect_polygon(-100.0, 200.0, -100.0, 100.0, pose)], pose)
    check("degenerate_all_inside", lambda: bool((sdf == 127).all()))

    # C5: the spec constants
    check("pad_is_clip_plus_half_metre", lambda: abs(M.PAD_M - 13.2) < 1e-9 and M.grid_centres(M.PAD_M)[0].shape == (212, 180))
    check("clip_fits_int8", lambda: abs(M.SDF_CLIP_M / M.SDF_UNIT_M - 127.0) < 1e-9)
    return bad


def _label_for_log(M, backend, logmap, log="L1"):
    mask, sdf = M.resolve_label(backend, logmap, log, (50.0, 0.0, 0.0))
    return mask, sdf


# ------------------------------------------------------------------ layer 1: the label --------------------------------------
def test_real_module_conforms(tmp):
    bad = conformance(ALS)
    assert bad == [], f"real module fails its own oracle: {bad}"


class _Patch:
    """Swap module attributes for the duration of a `with` block (a deliberate regression), then restore them."""

    def __init__(self, **kw):
        self.kw, self.old = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = getattr(ALS, k)
            setattr(ALS, k, v)
        return ALS

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(ALS, k, v)


def test_mutations_each_turn_red(tmp):
    orig_tg = ALS.to_global
    orig_sd = ALS.signed_distance
    expect = {
        "y_axis_flipped": (dict(to_global=lambda x, y, pose: orig_tg(x, -y, pose)), "frame_left_ahead"),
        "xy_swapped": (dict(to_global=lambda x, y, pose: orig_tg(y, x, pose)), "mask_rect_x_ahead"),
        "padding_removed": (dict(PAD_M=0.0), "padding_"),
        "clip_smaller": (dict(SDF_CLIP_M=6.0), "clip_"),
        "clip_larger_wraps_int8": (dict(SDF_CLIP_M=25.4), "raised"),
    }

    for name, (patch, must_trip) in expect.items():
        with _Patch(**patch) as M:
            bad = conformance(M)
        assert bad, f"mutation {name!r} was NOT detected -- the oracle cannot see it"
        assert any(must_trip in b for b in bad), f"mutation {name!r} tripped {bad}, expected a check containing {must_trip!r}"
    assert conformance(ALS) == [], "mutations leaked into the module"
    assert ALS.to_global is orig_tg and ALS.signed_distance is orig_sd


def _unguarded(inside_pad):                                       # the prototype's signed_distance, without the degenerate branches
    from scipy.ndimage import distance_transform_edt as edt
    return np.where(inside_pad, edt(inside_pad) - 0.5, -(edt(~inside_pad) - 0.5)) * ALS.RES


def test_degenerate_guard_is_equivalent_at_current_pad(tmp):
    """EQUIVALENT MUTANT, pinned rather than hidden: with PAD_M = CLIP + 0.5 the prototype's unguarded code already reads +-127 on
    a window with no drivable cell (edt of a no-background array = distance to the array border >= pad >= clip). The guard only
    matters if the pad ever shrinks below the clip -- and then it is the guard that keeps the answer right."""
    empty, full = np.zeros((212, 180), bool), np.ones((212, 180), bool)
    for g, v in ((empty, -127), (full, 127)):
        guarded = ALS.quantize_sdf(ALS.signed_distance(g))
        free = ALS.quantize_sdf(_unguarded(g)[26:26 + 160, 26:26 + 128])
        assert (guarded == v).all() and (free == v).all(), "at the current pad guarded and unguarded must agree"
    small = np.zeros((60, 60), bool)                              # a grid whose border is closer than the clip (pad < clip)
    assert (ALS.quantize_sdf(ALS.signed_distance(small)) == -127).all()
    assert not (ALS.quantize_sdf(_unguarded(small)) == -127).all(), "without the guard a small-pad grid reads garbage"


def test_wrong_map_for_a_log_turns_red(tmp):
    be = FakeBackend()
    good = {"L1": {"map": "mapA"}}
    wrong = {"L1": {"map": "mapB"}}
    m_good, s_good = _label_for_log(ALS, be, good)
    ix, iy = _cell(10.25, 0.25)
    assert m_good[ix, iy] and s_good[ix, iy] > 0, "the road of mapA is under the ego"
    expected = _expect_mask(-1e9, 1e9, -3.5, 3.5)                # the road strip in the ego frame (heading 0, origin y = 0)
    assert np.array_equal(m_good, expected)
    m_bad, s_bad = _label_for_log(ALS, be, wrong)
    assert not np.array_equal(m_bad, expected) and not m_bad.any() and (s_bad == -127).all(), "wrong map must read as off-road"
    try:
        _label_for_log(ALS, be, {}, "L1")
        raise AssertionError("a log with no map entry must fail loud")
    except KeyError:
        pass


def test_quantize_refuses_int8_wrap_and_clips(tmp):
    q = ALS.quantize_sdf(np.array([-20.0, -12.7, -3.3, 0.0, 5.0, 12.7, 20.0]))
    assert q.dtype == np.int8 and q.tolist() == [-127, -127, -33, 0, 50, 127, 127], q.tolist()
    with _Patch(SDF_CLIP_M=25.4):
        try:
            ALS.quantize_sdf(np.array([20.0]))
            raise AssertionError("clip 25.4 m / 0.1 m = 254 LSB must be refused, not wrapped")
        except ValueError:
            pass


def test_constants_match_navsim_dac(tmp):
    import navsim_dac as ND
    assert ALS.MAP_RADIUS == ND.MAP_RADIUS and abs(ALS.REAR_AXLE_TO_CENTER - ND.REAR_AXLE_TO_CENTER) < 1e-12


def test_yaw_matches_pyquaternion_not_the_textbook(tmp):
    rng = np.random.default_rng(7)
    worst, differs = 0.0, 0
    for _ in range(200):
        q = rng.normal(size=4)
        q[1:3] *= 0.05                                            # a few degrees of roll / pitch, as on a real vehicle
        qw, qx, qy, qz = q / np.linalg.norm(q)
        mine = ALS.yaw_from_quaternion(qw, qx, qy, qz)
        textbook = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        differs += abs(((mine - textbook) + math.pi) % (2 * math.pi) - math.pi) > 1e-6
        try:
            from pyquaternion import Quaternion
            ref = Quaternion(qw, qx, qy, qz).yaw_pitch_roll[0]
            worst = max(worst, abs(ref - mine))
        except ImportError:
            pass
    assert worst < 1e-12, f"yaw differs from pyquaternion by {worst}"
    assert differs > 0, "the trap this guards (textbook sign of q1*q2) was not exercised"
    assert abs(ALS.yaw_from_quaternion(math.cos(0.15), 0.0, 0.0, math.sin(0.15)) - 0.3) < 1e-12


# ------------------------------------------------------------------ layer 2: the shard format -------------------------------
def test_encode_decode_and_shard_roundtrip(tmp):
    rng = np.random.default_rng(3)
    a = rng.integers(-127, 128, size=(160, 128), dtype=np.int8)
    assert np.array_equal(ALS.decode_sdf(ALS.encode_sdf(a)), a)
    for bad in (a.astype(np.int16), a[:, :64], a.astype(np.float32)):
        try:
            ALS.encode_sdf(bad)
            raise AssertionError("wrong dtype / shape must be refused")
        except ValueError:
            pass
    items = [dict(t=f"tok{i}", blob=ALS.encode_sdf(np.full((160, 128), i, dtype=np.int8)), s="bank", p=[i, 2.0 * i, 0.1], q=[i, 20])
             for i in range(5)]
    n, sha = ALS.write_shard(tmp / "s" / "LOG.aux1", "LOG", "mapA", items)
    assert n == (tmp / "s" / "LOG.aux1").stat().st_size and ALS.sha256_file(tmp / "s" / "LOG.aux1") == sha
    head, blobs = ALS.read_shard(tmp / "s" / "LOG.aux1")
    assert head["log"] == "LOG" and head["map"] == "mapA" and head["raster"] == ALS.raster_spec() and len(head["entries"]) == 5
    assert all(np.array_equal(ALS.decode_sdf(blobs[f"tok{i}"]), np.full((160, 128), i, dtype=np.int8)) for i in range(5))
    raw = (tmp / "s" / "LOG.aux1").read_bytes()
    (tmp / "s" / "cut.aux1").write_bytes(raw[:-10])
    try:
        ALS.read_shard(tmp / "s" / "cut.aux1")
        raise AssertionError("a truncated shard must raise")
    except ValueError:
        pass
    (tmp / "s" / "bad.aux1").write_bytes(b"NOTASHARD" + raw[8:])
    try:
        ALS.read_shard(tmp / "s" / "bad.aux1")
        raise AssertionError("a bad magic must raise")
    except ValueError:
        pass
    assert not list((tmp / "s").glob("*.tmp")), "atomic write left a tmp file"


def test_infer_map_name(tmp):
    be = FakeBackend()
    on_a = [(50.0, 0.0, 0.0), (80.0, 1.0, 0.1), (90.0, -1.0, 0.0)]
    r = ALS.infer_map_name(on_a, be, ("mapA", "mapB"))
    assert r["ok"] and r["map"] == "mapA" and r["scores"] == {"mapA": 1.0, "mapB": 0.0}
    r = ALS.infer_map_name([(50.0, 103.5, 0.0)] * 3, be, ("mapA", "mapB"))
    assert r["ok"] and r["map"] == "mapB"
    r = ALS.infer_map_name([(50.0, 50.0, 0.0)] * 3, be, ("mapA", "mapB"))          # on neither map
    assert not r["ok"] and r["map"] is None
    both = FakeBackend()
    both.maps["mapB"] = both.maps["mapA"]                                           # two maps claim the same place -> ambiguous
    r = ALS.infer_map_name(on_a, both, ("mapA", "mapB"))
    assert not r["ok"] and r["map"] is None


# ------------------------------------------------------------------ layer 2: the builder end to end ------------------------
POSES = {"a1": (50.0, 0.0, 0.0), "a2": (80.0, 1.0, 0.1), "a3": (120.0, -1.0, -0.05), "a4": (150.0, 0.5, 0.02),
         "b1": (50.0, 103.5, 0.0), "b2": (70.0, 104.0, 0.05), "b3": (90.0, 103.0, 0.0), "c1": (30.0, 0.0, 0.0), "n1": (40.0, 0.0, 0.0)}
LOG_OF = {"a1": "L1", "a2": "L1", "a3": "L1", "a4": "L1", "b1": "L2", "b2": "L2", "b3": "L2", "c1": "L3", "n1": "L9"}
TOK = {n: n.encode().hex().ljust(32, "0")[:32] for n in [*POSES, "z9", "t7"]}      # real 32-hex tokens: the DB join reads lower(hex(token))


def _row(name):
    x, y, h = POSES[name]
    return {"log_name": LOG_OF[name], "token": TOK[name], "origin_world": [x, y, h], "rank": 0, "step": 0,
            "traj": [[0.5 + i, 0.2, 0.0] for i in range(20)]}


def _make_inputs(d: Path, bank=("a1", "a2", "a3", "b1", "b2", "c1", "n1"), yaml_names=("a1", "a2", "a3", "a4", "b1", "b2", "b3", "c1", "z9"),
                 navtest=None):
    """bank rows; a logmap; navtrain.yaml; a navtest list holding c1 (a bank token = a deliberate LEAK) and L9's n1; and L1's nuPlan-style DB,
    which holds a1..a4 -- a4 has NO bank row, so its origin must come from the DB's own ego_pose."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "bank.jsonl").write_text("\n".join(json.dumps(_row(t)) for t in bank) + "\n", encoding="utf-8")
    logmap = {"schema": "aux-drv-logmap-v1", "summary": {},
              "logs": {"L1": {"map": "mapA"}, "L2": {"map": "mapB"}, "L3": {"map": "mapA"}, "L9": {"map": "mapA"}}}
    (d / "logmap.json").write_text(json.dumps(logmap), encoding="utf-8")
    (d / "navtrain.yaml").write_text("log_names:\n- L1\n- L2\n- L3\ntokens:\n" + "".join(f"- '{TOK[t]}'\n" for t in yaml_names), encoding="utf-8")
    nt = navtest if navtest is not None else {"c1": "L3", "n1": "L9", "t7": "L7"}
    (d / "navtest.json").write_text(json.dumps({"token_to_log_name": {TOK[k]: v for k, v in nt.items()}}), encoding="utf-8")
    dbr = d / "dbs"
    dbr.mkdir(exist_ok=True)
    if not (dbr / "L1.db").exists():
        con = sqlite3.connect(dbr / "L1.db")
        con.executescript("CREATE TABLE log(map_version TEXT); CREATE TABLE ego_pose(token BLOB, x REAL, y REAL, qw REAL, qx REAL, qy REAL, qz REAL);"
                          "CREATE TABLE lidar_pc(token BLOB, ego_pose_token BLOB);")
        con.execute("INSERT INTO log VALUES ('mapA')")
        for i, t in enumerate(("a1", "a2", "a3", "a4")):
            x, y, h = POSES[t]
            ep = bytes([i + 1]) * 16
            con.execute("INSERT INTO ego_pose VALUES (?,?,?,?,?,?,?)", (ep, x, y, math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)))
            con.execute("INSERT INTO lidar_pc VALUES (?,?)", (bytes.fromhex(TOK[t]), ep))
        con.commit()
        con.close()
    return d


def _run(d: Path, out: Path, *extra, backend=BACKEND_SPEC, with_navtest=True, workers=1):
    argv = ["build", "--bank", str(d / "bank.jsonl"), "--db-root", str(d / "dbs"), "--yaml", str(d / "navtrain.yaml"),
            "--logmap", str(d / "logmap.json"), "--out", str(out), "--workers", str(workers), "--backend", backend, "--maps-root", str(d)]
    if with_navtest:
        argv += ["--navtest", str(d / "navtest.json")]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ALS.main(argv + list(extra))
    return rc, buf.getvalue()


def _keys(store: Path) -> set:
    return {e["t"] for p in (store / "shards").glob("*.aux1") for e in ALS.read_shard(p, False)[0]["entries"]}


def _marker(text: str) -> str:
    return text.strip().splitlines()[-1]


def _shas(store: Path) -> dict:
    return {k: v["sha256"] for k, v in json.load(open(store / "manifest.json"))["files"].items()}


def test_build_end_to_end(tmp):
    d = _make_inputs(tmp / "in")
    out = tmp / "store"
    rc, text = _run(d, out)
    assert rc == 0, text
    # stored: a1 a2 a3 (bank) + a4 (DB-derived) + b1 b2 = 6; c1 is a navtest token (a deliberate leak) and must NOT be written;
    # n1's log L9 is navtest-only and not in the yaml
    assert _marker(text) == "ZZAUXSTORE_DONE 6 0 new=6 skipped=0", _marker(text)
    man = json.load(open(out / "manifest.json"))
    assert man["counts"]["stored_keys"] == 6 and man["counts"]["logs"] == 2 and man["counts"]["by_src"] == {"bank": 5, "db": 1}
    assert man["leakage"] == {"navtest_tokens": 3, "token_intersection": 0, "log_intersection": 0}
    cov = man["coverage_vs_navtrain_yaml"]
    assert cov["yaml_tokens"] == 9 and cov["stored_in_yaml"] == 6 and cov["missing"] == 3 and cov["stored_not_in_yaml"] == 0
    assert sorted((out / "missing_tokens.txt").read_text().split()) == sorted(TOK[t] for t in ("b3", "c1", "z9"))
    xc = man["runs"][0]["origin_xcheck"]
    assert xc["n"] == 3 and xc["max_m"] == 0.0 and xc["max_rad"] < 1e-12, xc            # DB ego_pose == bank origin_world
    assert man["qc_rank0_target_points_inside_label"]["share_ge_0p9"] == 1.0, "the synthetic targets lie on the road"
    for rel, ent in man["files"].items():                                              # sha256 of EVERY output file
        assert ALS.sha256_file(out / rel) == ent["sha256"] and (out / rel).stat().st_size == ent["bytes"]
    be = FakeBackend()
    for name, log, mp in (("a1", "L1", "mapA"), ("a4", "L1", "mapA"), ("b2", "L2", "mapB")):
        _m, want = ALS.resolve_label(be, {log: {"map": mp}}, log, POSES[name])
        assert np.array_equal(ALS.read_label(out, log, TOK[name]), want), f"stored label for {name} differs from the oracle"
    ents = {e["t"]: e for e in ALS.read_shard(out / "shards" / "L1.aux1", with_blobs=False)[0]["entries"]}
    assert ents[TOK["a4"]]["s"] == "db" and ents[TOK["a1"]]["s"] == "bank" and ents[TOK["a1"]]["q"] == [20, 20]
    assert TOK["c1"] not in _keys(out) and TOK["n1"] not in _keys(out)


def test_resume_partial_extend_and_stale_map(tmp):
    d = _make_inputs(tmp / "in")
    out = tmp / "store"
    _run(d, out)
    before = _shas(out)
    rc, text = _run(d, out)                                                            # nothing to do, files byte-identical
    assert _marker(text) == "ZZAUXSTORE_DONE 6 0 new=0 skipped=6" and _shas(out) == before, text
    (out / "shards" / "L2.aux1").unlink()                                              # lose a shard -> only it is regenerated
    rc, text = _run(d, out)
    assert _marker(text) == "ZZAUXSTORE_DONE 6 0 new=2 skipped=4" and _shas(out) == before, text         # ... identically
    with open(d / "bank.jsonl", "a", encoding="utf-8") as f:                           # a new key appears -> shard extended
        f.write(json.dumps(_row("b3")) + "\n")
    rc, text = _run(d, out)
    assert _marker(text) == "ZZAUXSTORE_DONE 7 0 new=1 skipped=6" and len(_keys(out)) == 7, text
    lm = json.load(open(d / "logmap.json"))                                            # the log's map changes -> NOT trusted
    lm["logs"]["L2"]["map"] = "mapA"
    (d / "logmap.json").write_text(json.dumps(lm), encoding="utf-8")
    rc, text = _run(d, out)
    assert _marker(text) == "ZZAUXSTORE_DONE 7 0 new=3 skipped=4", text
    assert ALS.read_shard(out / "shards" / "L2.aux1", False)[0]["map"] == "mapA"
    got = ALS.read_label(out, "L2", TOK["b1"])
    want_a = ALS.resolve_label(FakeBackend(), {"L2": {"map": "mapA"}}, "L2", POSES["b1"])[1]
    want_b = ALS.resolve_label(FakeBackend(), {"L2": {"map": "mapB"}}, "L2", POSES["b1"])[1]
    assert np.array_equal(got, want_a) and not np.array_equal(got, want_b), "a changed map must rebuild the shard from the NEW map"


def test_navtest_frames_in_a_stale_store_are_removed(tmp):
    d = _make_inputs(tmp / "in")
    stale = tmp / "store"
    _run(d, stale, with_navtest=False)                                                 # without the list, c1 (log L3) IS stored: the leak
    assert TOK["c1"] in _keys(stale) and (stale / "shards" / "L3.aux1").exists()
    _run(d, stale)                                                                     # navtest LOG L3 -> its whole shard is removed
    assert TOK["c1"] not in _keys(stale) and not (stale / "shards" / "L3.aux1").exists()
    man = json.load(open(stale / "manifest.json"))
    assert man["leakage"] == {"navtest_tokens": 3, "token_intersection": 0, "log_intersection": 0}
    # TOKEN level: b1 is a navtest token filed under a log (L7) that is not L2 -> only that frame goes, the rest of L2 stays
    other = tmp / "store2"
    _make_inputs(tmp / "in2", navtest={"b1": "L7"})
    _run(tmp / "in2", other, with_navtest=False)
    assert TOK["b1"] in _keys(other) and TOK["b2"] in _keys(other)
    _run(tmp / "in2", other)
    assert TOK["b1"] not in _keys(other) and TOK["b2"] in _keys(other) and TOK["a1"] in _keys(other)
    assert json.load(open(other / "manifest.json"))["leakage"]["token_intersection"] == 0


def test_shard_flag_partitions_and_reproduces(tmp):
    d = _make_inputs(tmp / "in")
    one, sh = tmp / "one", tmp / "sharded"
    _run(d, one)
    for i in (0, 1):
        rc, text = _run(d, sh, "--shard", f"{i}/2")
        assert rc == 0 and "run `finalize`" in text
        assert len(list((sh / "shards").glob("*.aux1"))) == (1 if i == 0 else 2)       # logs sorted L1, L2: shard 0 = L1, shard 1 = L2
    ALS.main(["finalize", "--out", str(sh)])
    assert json.load(open(sh / "manifest.json"))["store_sha256"] == json.load(open(one / "manifest.json"))["store_sha256"]


def test_failures_are_counted_not_fatal(tmp):
    d = _make_inputs(tmp / "in", bank=("a1", "a2", "a3", "b1"))
    out = tmp / "store"
    rc, text = _run(d, out, backend=FAIL_SPEC)
    fails = [json.loads(l) for l in (out / "failures_all.jsonl").read_text().splitlines()]
    assert len(fails) == 1 and fails[0]["token"] == TOK["a2"] and "synthetic map failure" in fails[0]["error"], fails
    assert TOK["a2"] not in _keys(out)
    assert _marker(text).startswith("ZZAUXSTORE_DONE 4 1 "), _marker(text)               # a1 a3 a4(db) b1 stored; a2 failed


def test_memory_guard_refuses_a_multiworker_start(tmp):
    assert ALS.free_physical_gb() > 0
    d = _make_inputs(tmp / "in")
    rc, text = _run(d, tmp / "store", "--min-free-gb", "1e9", workers=2)
    assert rc == 3 and "REFUSED" in text and not (tmp / "store" / "shards").exists()


def test_spawn_pool_reproduces_the_single_process_store(tmp):
    d = _make_inputs(tmp / "in", bank=("a1", "a2", "a3", "b1", "b2", "b3"), yaml_names=("a1", "a2", "a3", "a4", "b1", "b2", "b3"))
    one, two = tmp / "one", tmp / "two"
    rc1, t1 = _run(d, one, workers=1)
    rc2, t2 = _run(d, two, "--min-free-gb", "0", workers=2)
    assert rc1 == 0 and rc2 == 0, t2
    assert _marker(t1) == _marker(t2) == "ZZAUXSTORE_DONE 7 0 new=7 skipped=0", (t1[-300:], t2[-300:])
    assert json.load(open(one / "manifest.json"))["store_sha256"] == json.load(open(two / "manifest.json"))["store_sha256"]


# ------------------------------------------------------------------ runner ---------------------------------------------
if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
                print(f"PASS {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"ZZTEST_PASS {len(tests)}" if fails == 0 else f"ZZTEST_FAIL {fails}/{len(tests)}")
    sys.exit(1 if fails else 0)
