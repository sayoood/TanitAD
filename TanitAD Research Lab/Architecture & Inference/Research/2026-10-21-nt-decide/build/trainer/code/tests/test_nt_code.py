"""H-NT-DECIDE-1 E2/E3 builder code (``code/nt_*.py``): literals pinned against INDEPENDENT references.

* ``devkit_yaw`` vs pyquaternion's own ``yaw_pitch_roll[0]`` (three LITERAL pairs computed by pyquaternion on Thor's
  navsim-cpu venv, 2026-10-21; a 20,000-quaternion sweep there read worst |d| 4.4e-16 vs pyquaternion and 9.2e-3 rad
  for the textbook form). Mutation: the textbook ``+xy`` form must FAIL the literals.
* ``NAVSIM_LEGAL_NAV_CMD`` vs the bridge's SOURCE TEXT (``refcv8_bridge.py``), parsed, not imported.
* ``quat_of_R``: analytic rotations round-trip; the road plane literal; the class map.
* ``nt_clock_reference.fit_token``: a nominal 2 Hz log fits (g, dt) = (0.0, 0.1) on the episode clock.

Run: ``PYTHONPATH=<nt_tree>/stack python -m pytest code/tests -q``.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import nt_common as C  # noqa: E402

BRIDGE8 = CODE.parent / "nt_tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code/refcv8_bridge.py"

#: (w, x, y, z) -> pyquaternion.Quaternion(*q).yaw_pitch_roll[0], computed BY PYQUATERNION (Thor, navsim-cpu venv)
PYQ = (
    ([0.8510539130004127, -0.047136603193878655, 0.00039846379878213933, 0.5229581427156768], 1.1000341185415226),
    ([0.1218697314326551, -0.02928522043627145, -0.04884212683174031, -0.9909109904885925], -2.8944388007399593),
    ([0.9866744935999232, 0.041896737824880785, 0.05681565716704152, 0.14659497992648154], 0.2908323624569356),
)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def test_devkit_yaw_matches_pyquaternion_literals():
    for q, want in PYQ:
        assert abs(_wrap(C.devkit_yaw(q) - want)) < 1e-12


def test_devkit_yaw_analytic_pure_yaw():
    for th in (-3.0, -1.0, 0.0, 0.5, 2.9):
        q = [math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]
        assert abs(_wrap(C.devkit_yaw(q) - th)) < 1e-15
    # an un-normalised quaternion reads the same heading
    assert abs(C.devkit_yaw([2 * math.cos(0.25), 0.0, 0.0, 2 * math.sin(0.25)]) - 0.5) < 1e-15


def test_devkit_yaw_mutation_textbook_form_fails():
    """DELIBERATE REGRESSION: the textbook ZYX ``+xy`` form -- the defect E1's v1.2 and this agent's K1 caught."""
    def textbook(q):
        w, x, y, z = q
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    worst = max(abs(_wrap(textbook(q) - want)) for q, want in PYQ)
    assert worst > 1e-4


def test_nav_literal_equals_bridge_source():
    if not BRIDGE8.is_file():
        pytest.skip(f"refcv8_bridge.py not in this tree ({BRIDGE8})")
    lit = None
    for node in ast.walk(ast.parse(BRIDGE8.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NAVSIM_LEGAL_NAV_CMD" for t in node.targets):
            lit = ast.literal_eval(node.value)
    assert lit == {"left": 1, "forward": 0, "right": 2, "unknown": 0}
    assert C.NAVSIM_LEGAL_NAV_CMD == lit
    assert C.nav_token_of_driving_command([1, 0, 0, 0])["nav_token"] == 1       # left -> TURN_L
    assert C.nav_token_of_driving_command([0, 1, 0, 0])["nav_token"] == 0       # forward -> FOLLOW
    assert C.nav_token_of_driving_command([0, 0, 1, 0])["nav_token"] == 2       # right -> TURN_R
    assert C.nav_token_of_driving_command([0, 0, 0, 1])["nav_token"] == 0       # unknown -> FOLLOW
    with pytest.raises(C.NTRefused):
        C.nav_token_of_driving_command([0.5, 0.5, 0, 0])                         # never arg-maxed


def test_class_map_and_road_plane_literals():
    assert C.CLASS_MAP == {"vehicle": "automobile", "pedestrian": "person", "bicycle": "rider"}
    assert C.ROAD_PLANE_MEDIAN_M == -0.3557311968663991


def _R(yaw, pitch, roll):
    cz, sz, cy, sy, cx, sx = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll),
                              math.sin(roll))
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def _R_of_q(qx, qy, qz, qw):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


@pytest.mark.parametrize("ypr", [(0.0, 0.0, 0.0), (math.pi, 0.0, 0.0), (0.3, -0.2, 0.1), (-1.5, 0.05, 3.0),
                                 (0.0, math.pi / 2 - 1e-3, 0.0), (2.0, 0.0, math.pi)])
def test_quat_of_R_round_trip(ypr):
    pytest.importorskip("torch")
    import nt_joins as J
    M = _R(*ypr)
    q = J.quat_of_R(M)
    assert abs(sum(v * v for v in q) - 1.0) < 1e-12
    assert np.abs(_R_of_q(*q) - M).max() < 1e-12


def test_clock_fit_nominal_log():
    import nt_clock_reference as K
    ts = np.array([1_000_000 + 500_000 * i for i in range(30)], dtype=np.int64)
    f = K.fit_token(ts, 10, 70)
    assert abs(f["grid_start_on_label_timeline_s"] - 0.0) < 1e-9
    assert abs(f["dt_s"] - 0.1) < 1e-12
    assert f["fit_resid_max_ms"] < 1e-6
    assert f["n_knots"] == 14                                  # k = -1 .. 12 (raw rows 4 .. 69 < 70)
    # DELIBERATE REGRESSION: a log on the legacy (t + w - 1) clock is 0.2 s late -> the fit must show the offset
    ts2 = ts + 200_000
    f2 = K.fit_token(ts2, 10, 70)
    assert abs(f2["grid_start_on_label_timeline_s"]) < 1e-9    # relative times: a constant shift cancels...
    ts3 = ts.copy()
    ts3[12:] += 50_000                                         # ...a 50 ms step in the chain does not
    f3 = K.fit_token(ts3, 10, 70)
    assert f3["fit_resid_max_ms"] > 10.0


# ------------------------------------------------------------------------------------------------------------------ #
# E3 gather / merge: every .npz array is read ONCE per file (the 2026-10-21 22:50 Thor global OOM, 122.9 GB anon RSS)
# ------------------------------------------------------------------------------------------------------------------ #
def _joins_module():
    """``nt_joins`` from ``code/`` -- or from ``NT_JOINS_PATH`` (the mutation run points it at the PRE-FIX file)."""
    import importlib.util
    import os
    p = os.environ.get("NT_JOINS_PATH") or str(CODE / "nt_joins.py")
    spec = importlib.util.spec_from_file_location("nt_joins_under_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_shard(d: Path, k: int, n: int, clips: list) -> None:
    import json
    d.mkdir(parents=True)
    (d / "JOINS_REPORT.json").write_text(json.dumps({
        "shard": {"k": k, "n": n}, "n_tokens": len(clips), "n_logs": 1, "n_records": 1, "road_z_m": -1.5,
        "road_plane_source": "literal", "class_map": {}, "census": {"c": 1}}))
    (d / "nt_rig_extrinsics.json").write_text(json.dumps({c: {"k": k} for c in clips}))
    (d / "nt_rigs.json").write_text("{}")
    (d / "nt_agents.jsonl").write_text(json.dumps({"clip_id": clips[0]}) + "\n")
    (d / "nt_agents_3d.jsonl").write_text(json.dumps({"clip_id": clips[0]}) + "\n")
    nrows = np.asarray([1 + (i % 3) for i in range(len(clips))], np.int64)        # 1, 2, 3, 1, 2, 3, ...
    tot = int(nrows.sum())
    base = 1000 * (k + 1)
    arr = {key: (np.arange(tot, dtype=np.int32) + base) for key in
           ("row", "track", "n_full", "n_img", "n_vis", "vis_rows")}
    arr["cx"] = np.arange(tot, dtype=np.float32) + base
    arr["cy"] = -(np.arange(tot, dtype=np.float32) + base)
    np.savez(d / "nt_vis1_parts.npz", clip_id=np.asarray(clips), sha12=np.asarray([c[-12:] for c in clips], "<U12"),
             n_rows_per_clip=nrows, n_zh=np.zeros(len(clips), np.int32), n_valid=np.ones(len(clips), np.int32), **arr)


@pytest.mark.parametrize("n_clips", [12, 600])
def test_gather_reads_each_npz_array_once_and_keeps_rows(tmp_path, monkeypatch, n_clips):
    """The npz read count must NOT grow with the clip count (it is bounded by shards x arrays: 2 x 13 = 26), and the
    gathered VIS-1 rows are LITERALLY the shard rows in sha12 order. Mutation (``NT_JOINS_PATH`` = the pre-fix
    ``nt_joins.py``, whose per-clip ``z[k][lo:hi]`` was OOM-killed at 122.9 GB on Thor): reads = clips x 11 -> RED."""
    import argparse
    J = _joins_module()
    reads = {"n": 0}
    orig = np.lib.npyio.NpzFile.__getitem__

    def counting(self, key):
        reads["n"] += 1
        return orig(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", counting)
    shards = []
    for k in range(2):
        clips = [f"navtrain-{k:01d}{i:015x}" for i in range(n_clips)]
        _write_shard(tmp_path / f"s{k}", k, 2, clips)
        shards.append(str(tmp_path / f"s{k}"))
    out = tmp_path / "gathered"
    J.cmd_gather(argparse.Namespace(shards=shards, out=str(out)))
    assert reads["n"] <= 26, f"{reads['n']} npz reads for {2 * n_clips} clips: an array is re-read per clip"
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", orig)
    z = dict(np.load(out / "nt_vis1_parts.npz"))
    # literal expectation: sha12 = the last 12 chars of the clip id; shard k rows are 1000*(k+1) + 0..tot-1, in order
    want_order = sorted(f"navtrain-{k:01d}{i:015x}" for k in range(2) for i in range(n_clips))
    want_order.sort(key=lambda c: c[-12:])
    assert z["clip_id"].tolist() == want_order
    nr = {f"navtrain-{k:01d}{i:015x}": 1 + (i % 3) for k in range(2) for i in range(n_clips)}
    start = {}
    for k in range(2):
        acc = 0
        for i in range(n_clips):
            c = f"navtrain-{k:01d}{i:015x}"
            start[c] = 1000 * (k + 1) + acc
            acc += nr[c]
    want_row = np.concatenate([np.arange(start[c], start[c] + nr[c]) for c in want_order]).astype(np.int32)
    assert np.array_equal(z["row"], want_row)
    assert np.array_equal(z["cy"], -want_row.astype(np.float32))
    assert z["n_rows_per_clip"].tolist() == [nr[c] for c in want_order]
