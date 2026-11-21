"""Literal tests for ``export_navtrain_inputs.py`` (the navtrain port of W3's ``export_navtest_inputs.py``).

Two layers, both with LITERAL expectations (never an expression over the code under test):

* the pure part (any python with numpy): the record builder, the ego-status dict, the sample gate and the output writer;
* the ``devkit`` part: the REAL script, run in the NAVSIM venv against the REAL v1.1 devkit on the four navtest logs of W3's banked smoke
  export.  The control that matters: its 334 records must equal W3's banked records exactly -- W3's artifact was produced by W3's
  script on 2026-09 and is independent of this port.  NAVTEST inputs only; no navtrain data and no score exists anywhere in this file.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

import export_navtrain_inputs as X
import shards as SH

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.dirname(HERE)
SCRIPT = os.path.join(BUILD, "code", "export_navtrain_inputs.py")

NAVSIM_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
V11_TREE = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
W3_CODE = "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/code"
LOGS_TEST = "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
BANKED = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs_smoke4logs.json.gz"

RECORD_KEYS = ["stage", "frame_type", "log_name", "map_name", "scene_token", "frame_tokens", "timestamps_us", "ego_statuses", "cams",
               "human_future_poses", "human_future_sampling", "cv_poses", "fingerprint"]


def es(pose, vel, acc, cmd, dtype=np.float32):
    return types.SimpleNamespace(ego_pose=np.array(pose, dtype), ego_velocity=np.array(vel, dtype),
                                 ego_acceleration=np.array(acc, dtype), driving_command=np.array(cmd, dtype))


def record(position=5, human=None, nh=4):
    frames = SH.make_frames("a_log", 20)
    win = frames[position - 3:position + 11]
    statuses = [es([0.0, 0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0, 1, 0, 0]) for _ in range(nh)]
    human = np.zeros((8, 3), np.float32) if human is None else human
    return X.build_record(win[nh - 1]["token"], win, nh, statuses, human, (8, 0.5), np.zeros((8, 3)), "fp0")


# ------------------------------------------------------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------------------------------------------------------
def test_constants_are_the_registered_literals():
    assert X.HUMAN_HORIZON_POSES == 8              # W3: sc.get_future_trajectory(8) -- 4 s at 0.5 s
    assert X.CAMS == ("cam_l0", "cam_f0", "cam_r0")


# ------------------------------------------------------------------------------------------------------------------------------
# es_dict
# ------------------------------------------------------------------------------------------------------------------------------
def test_es_dict_widens_float32_exactly_and_keeps_float64():
    d32 = X.es_dict(es([0.1, 0.0, 0.0], [0.1, 0.0], [0.1, 0.0], [0, 0, 0, 1]))
    assert d32["ego_pose"][0] == 0.10000000149011612           # float32(0.1) widened: exact, not re-rounded
    d64 = X.es_dict(es([0.1, 0.0, 0.0], [0.1, 0.0], [0.1, 0.0], [0, 0, 0, 1], dtype=np.float64))
    assert d64["ego_pose"][0] == 0.1                           # a float64 source must NOT pass through float32
    assert d64["ego_velocity"][0] == 0.1 and d64["ego_acceleration"][0] == 0.1


def test_es_dict_keeps_velocity_and_acceleration_apart():
    d = X.es_dict(es([1.0, 2.0, 0.5], [1.5, 0.25], [-0.5, 2.0], [0, 1, 0, 0]))
    assert d["ego_pose"] == [1.0, 2.0, 0.5]
    assert d["ego_velocity"] == [1.5, 0.25]
    assert d["ego_acceleration"] == [-0.5, 2.0]
    assert d["driving_command"] == [0.0, 1.0, 0.0, 0.0]


def test_es_dict_records_the_source_dtypes():
    d = X.es_dict(es([0, 0, 0], [0, 0], [0, 0], [0, 0, 0, 1]))
    assert d["_dtypes"] == ["float32", "float32", "float32", "float32"]
    assert sorted(d) == ["_dtypes", "driving_command", "ego_acceleration", "ego_pose", "ego_velocity"]


# ------------------------------------------------------------------------------------------------------------------------------
# build_record
# ------------------------------------------------------------------------------------------------------------------------------
def test_record_schema_is_w3s():
    r = record()
    assert list(r) == RECORD_KEYS
    assert r["stage"] == 1 and r["frame_type"] == "ORIGINAL"
    assert r["fingerprint"] == "fp0"
    assert len(r["ego_statuses"]) == 4


def test_record_frame_tokens_are_the_four_history_frames():
    r = record(position=5)                                     # window = frames[2:16]; history = positions 2,3,4,5
    assert r["frame_tokens"] == ["02a", "03a", "04a", "05a"]
    assert r["timestamps_us"] == [1600000001000000, 1600000001500000, 1600000002000000, 1600000002500000]


def test_record_cams_are_the_history_frames_of_three_cameras():
    r = record(position=5)
    assert sorted(r["cams"]) == ["cam_f0", "cam_l0", "cam_r0"]
    assert r["cams"]["cam_l0"] == ["a_log/CAM_L0/02a_l0.jpg", "a_log/CAM_L0/03a_l0.jpg", "a_log/CAM_L0/04a_l0.jpg", "a_log/CAM_L0/05a_l0.jpg"]
    assert r["cams"]["cam_f0"] == ["a_log/CAM_F0/02a_f0.jpg", "a_log/CAM_F0/03a_f0.jpg", "a_log/CAM_F0/04a_f0.jpg", "a_log/CAM_F0/05a_f0.jpg"]
    assert r["cams"]["cam_r0"][3] == "a_log/CAM_R0/05a_r0.jpg"


def test_record_identity_comes_from_the_token_frame():
    r = record(position=5)
    assert (r["log_name"], r["map_name"], r["scene_token"]) == ("a_log", "us-nv-las-vegas-strip", "sc_a_log")


def test_record_future_arrays_are_float64_lists_with_the_sampling():
    h = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    r = record(human=h)
    assert r["human_future_sampling"] == [8, 0.5]
    assert r["human_future_poses"][1] == [pytest.approx(0.3, abs=1e-6), pytest.approx(0.4, abs=1e-6), pytest.approx(0.5, abs=1e-6)]
    assert all(isinstance(v, float) for row in r["human_future_poses"] for v in row)
    assert len(r["human_future_poses"]) == 8 and len(r["cv_poses"]) == 8


# ------------------------------------------------------------------------------------------------------------------------------
# the sample gate
# ------------------------------------------------------------------------------------------------------------------------------
def sample_file(tmp_path, **over):
    toks = ["03a", "04a", "03b"]
    doc = SH.make_sample_doc(toks)
    doc["token_log"] = {"03a": "a_log", "04a": "a_log", "03b": "b_log"}
    doc.update(over)
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


def test_a_fixed_sample_loads(tmp_path):
    s = X.load_sample(sample_file(tmp_path), "navtrain")
    assert s["tokens"] == ["03a", "03b", "04a"] and s["count"] == 3


def test_sample_must_be_fixed_for_navtrain(tmp_path):
    with pytest.raises(SystemExit) as e:
        X.load_sample(sample_file(tmp_path, status="NOT_FIXED__NO_READABLE_NAVTRAIN_TOKEN"), "navtrain")
    assert "not FIXED" in str(e.value)


def test_sample_hash_must_rederive(tmp_path):
    with pytest.raises(SystemExit) as e:
        X.load_sample(sample_file(tmp_path, sha256_sorted_tokens="0" * 64), "navtrain")
    assert "re-derive" in str(e.value)
    with pytest.raises(SystemExit):
        X.load_sample(sample_file(tmp_path, count=4), "navtrain")


def test_sample_needs_a_complete_token_log(tmp_path):
    with pytest.raises(SystemExit) as e:
        X.load_sample(sample_file(tmp_path, token_log={"03a": "a_log"}), "navtrain")
    assert "token_log" in str(e.value)


def test_the_navtest_control_does_not_demand_a_fixed_sample(tmp_path):
    s = X.load_sample(sample_file(tmp_path, status="CONTROL_ONLY", sha256_sorted_tokens="x"), "navtest")
    assert s["status"] == "CONTROL_ONLY"


# ------------------------------------------------------------------------------------------------------------------------------
# the writer
# ------------------------------------------------------------------------------------------------------------------------------
def doc_for(n=2, **kw):
    d = {"n_tokens": n, "fingerprint_unique": True, "fingerprint_collisions": {}, "tokens": {f"t{i}": {"x": i} for i in range(n)}}
    d.update(kw)
    return d


def test_a_refused_export_leaves_no_file(tmp_path):
    out = tmp_path / "x" / "inputs.json.gz"
    man = X.write_outputs(doc_for(n_missing_vs_sample=1, n_unexpected_vs_sample=0), str(out), str(tmp_path / "m.json"), True)
    assert not out.exists()
    assert man["refused"] is True and man["sha256"] is None and man["bytes"] is None and man["n_missing_vs_sample"] == 1


def test_a_complete_export_round_trips_and_hashes_its_own_bytes(tmp_path):
    out = tmp_path / "x" / "inputs.json.gz"
    d = doc_for(3)
    man = X.write_outputs(d, str(out), str(tmp_path / "m.json"), False)
    assert out.exists() and man["refused"] is False
    assert json.loads(gzip.open(out).read().decode("utf-8")) == d
    assert man["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert man["bytes"] == out.stat().st_size and man["n_tokens"] == 3


def test_an_existing_export_is_never_overwritten(tmp_path):
    out = tmp_path / "inputs.json.gz"
    out.write_bytes(b"old")
    with pytest.raises(SystemExit) as e:
        X.check_fresh(str(out))
    assert "already exists" in str(e.value)
    X.check_fresh(str(tmp_path / "other.json.gz"))             # a fresh path passes silently


def test_the_navsim_tree_check_is_exact_not_a_prefix(tmp_path):
    t = tmp_path / "tree" / "v11"
    (t / "navsim").mkdir(parents=True)
    init = t / "navsim" / "__init__.py"
    init.write_text("", encoding="utf-8")
    other = tmp_path / "tree" / "v10" / "navsim"
    other.mkdir(parents=True)
    (other / "__init__.py").write_text("", encoding="utf-8")
    assert X.navsim_is_from(str(t), str(init)) is True
    assert X.navsim_is_from(str(t).replace("\\", "/") + "/", str(init)) is True          # slash spelling does not matter
    assert X.navsim_is_from(str(tmp_path / "tree"), str(other / "__init__.py")) is False     # the PARENT of two devkits accepts neither
    assert X.navsim_is_from(str(t), str(other / "__init__.py")) is False
    assert X.navsim_is_from(str(tmp_path), str(init)) is False


@pytest.mark.e2e
def test_main_refuses_while_a_mutation_env_is_set(tmp_path):
    env = dict(os.environ, R8EXP_MUTATION="stage_2", PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, "-X", "utf8", SCRIPT, "--sample", "s", "--logs-root", "l", "--navsim-tree", "t", "--w3-code", "w",
                        "--out", str(tmp_path / "o.gz"), "--manifest", str(tmp_path / "m.json")], capture_output=True, text=True, env=env)
    assert p.returncode != 0 and "R8EXP_MUTATION" in (p.stdout + p.stderr)
    assert not (tmp_path / "o.gz").exists()


# ------------------------------------------------------------------------------------------------------------------------------
# the REAL script on the REAL devkit (navtest control; no navtrain data, no score)
# ------------------------------------------------------------------------------------------------------------------------------
NEEDED = [NAVSIM_PY, V11_TREE, W3_CODE, LOGS_TEST, BANKED]
devkit = pytest.mark.skipif(any(not os.path.exists(p) for p in NEEDED), reason="NAVSIM venv / v1.1 devkit / navtest fixtures absent")


def run_export(tmp, sample_path, extra=(), tree=V11_TREE):
    out, man = str(tmp / "inputs.json.gz"), str(tmp / "manifest.json")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4")
    env.pop("R8EXP_MUTATION", None)
    p = subprocess.run([NAVSIM_PY, "-X", "utf8", SCRIPT, "--sample", sample_path, "--logs-root", LOGS_TEST, "--navsim-tree", tree,
                        "--w3-code", W3_CODE, "--split", "navtest", "--out", out, "--manifest", man, *extra],
                       capture_output=True, text=True, env=env, timeout=1200, cwd=str(tmp))
    return p, out, man


def banked_sample(tmp, phantom=False):
    d = json.load(gzip.open(BANKED))
    toks = sorted(d["tokens"])
    tl = {t: d["tokens"][t]["log_name"] for t in toks}
    if phantom:
        toks = sorted(toks + ["ffffffffffffffff"])
        tl["ffffffffffffffff"] = tl[toks[0]]
    doc = {"status": "CONTROL_ONLY", "count": len(toks), "tokens": toks, "token_log": tl}
    p = tmp / "sample.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p), d


@pytest.mark.devkit
@pytest.mark.e2e
@devkit
def test_four_real_navtest_logs_export_exactly_w3s_banked_records(tmp_path):
    sp, banked = banked_sample(tmp_path)
    p, out, man = run_export(tmp_path, sp, ["--no-maps"])
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    assert "ZZEXPORT_DONE tokens=334 missing=0" in p.stdout
    got = json.loads(gzip.open(out).read().decode("utf-8"))
    assert got["n_tokens"] == 334 and set(got["tokens"]) == set(banked["tokens"])
    differing = [t for t in banked["tokens"] if banked["tokens"][t] != got["tokens"][t]]
    assert differing == []                                      # 334/334 records identical (maps stubbed; W3's were built with maps)
    m = json.load(open(man))
    assert m["n_tokens"] == 334 and m["maps_stubbed"] is True and m["refused"] is False and m["fingerprint_unique"] is True
    assert os.path.normcase(m["navsim_file"]).replace("\\", "/").startswith(os.path.normcase(V11_TREE).replace("\\", "/"))
    assert m["sha256"] == hashlib.sha256(open(out, "rb").read()).hexdigest()


@pytest.mark.devkit
@pytest.mark.e2e
@devkit
def test_a_sample_token_the_devkit_does_not_produce_refuses_and_writes_nothing(tmp_path):
    sp, _ = banked_sample(tmp_path, phantom=True)
    p, out, man = run_export(tmp_path, sp, ["--no-maps"])
    assert p.returncode == 1 and "ZZEXPORT_REFUSED tokens=334 missing=1" in p.stdout
    assert not os.path.exists(out)
    m = json.load(open(man))
    assert m["refused"] is True and m["n_missing_vs_sample"] == 1 and m["sha256"] is None


@pytest.mark.devkit
@pytest.mark.e2e
@devkit
def test_a_parent_directory_of_a_different_devkit_refuses(tmp_path):
    """MEASURED while writing the port: ``--navsim-tree C:/Users/Admin`` is a PREFIX of the venv's v1.0 devkit and used to pass."""
    sp, _ = banked_sample(tmp_path)
    p, out, man = run_export(tmp_path, sp, ["--no-maps"], tree="C:/Users/Admin")
    assert p.returncode != 0 and not os.path.exists(out)
    assert "not the v1.1 tree" in (p.stdout + p.stderr)


@pytest.mark.devkit
@pytest.mark.e2e
@devkit
def test_a_navsim_not_from_the_named_tree_refuses(tmp_path):
    sp, _ = banked_sample(tmp_path)
    p, out, man = run_export(tmp_path, sp, ["--no-maps"], tree=str(tmp_path).replace("\\", "/"))
    assert p.returncode != 0 and not os.path.exists(out)
    assert "not the v1.1 tree" in (p.stdout + p.stderr) or "No module named 'navsim'" in (p.stdout + p.stderr)
