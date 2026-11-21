"""Tests for the frozen UNSEEN-STRATUM S1 token rule.   CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_unseen_token_rule.py
Synthetic DBs prove the rule's arithmetic and its RED arms (a wrong phase / stride / window must CHANGE the answer); the real-DB tests are the
identity check in miniature and skip when the data drive is absent."""
import gzip
import hashlib
import json
import os
import sqlite3
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import unseen_token_rule as R  # noqa: E402

PKG = os.path.dirname(HERE)
DB_TV = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"


def make_db(path, n_lidar=600, f0_first_idx=0, l0_first_idx=0, route_empty_scenes=(), scene_len=100, f0_off_us=-12_000):
    """lidar_pc every 50 ms; F0 and L0 images every 100 ms, the first one `*_first_idx` lidar steps in (+ f0_off_us); other cameras present."""
    c = sqlite3.connect(path)
    c.executescript("""CREATE TABLE lidar_pc (token BLOB, timestamp INTEGER, scene_token BLOB);
    CREATE TABLE camera (token BLOB, channel TEXT);
    CREATE TABLE image (token BLOB, camera_token BLOB, timestamp INTEGER, filename_jpg TEXT);
    CREATE TABLE scene (token BLOB, roadblock_ids TEXT);""")
    t0 = 1_600_000_000_000_000
    for i in range(n_lidar):
        c.execute("INSERT INTO lidar_pc VALUES (?,?,?)", (i.to_bytes(8, "big"), t0 + i * 50_000, (i // scene_len).to_bytes(4, "big")))
    for s in range(n_lidar // scene_len + 1):
        c.execute("INSERT INTO scene VALUES (?,?)", (s.to_bytes(4, "big"), "" if s in route_empty_scenes else "11 22 33"))
    first = {"CAM_F0": f0_first_idx, "CAM_L0": l0_first_idx, "CAM_R0": 0, "CAM_B0": 0}
    for k, (ch, fi) in enumerate(first.items()):
        c.execute("INSERT INTO camera VALUES (?,?)", (bytes([k + 1]), ch))
        j = 0
        while True:
            ts = t0 + (fi * 50_000) + j * 100_000 + (f0_off_us if ch in ("CAM_F0", "CAM_L0") else 7_000)
            if ts > t0 + n_lidar * 50_000:
                break
            c.execute("INSERT INTO image VALUES (?,?,?,?)", ((k * 10_000 + j).to_bytes(8, "big"), bytes([k + 1]), ts, f"LOG/{ch}/{k:02d}{j:06d}.jpg"))
            j += 1
    c.commit()
    c.close()


def tok(i):
    return i.to_bytes(8, "big").hex()


def expected_tokens(phase, n_lidar, stride=10, drop=1, hist=4, fut=10):
    fr = list(range(phase, n_lidar, stride))[drop:]
    return [tok(i) for i in fr[hist - 1: len(fr) - fut]]


@pytest.mark.parametrize("f0,l0,phase", [(0, 0, 0), (1, 1, 1), (0, 2, 2), (2, 0, 2)])
def test_phase_grid_window_arithmetic(tmp_path, f0, l0, phase):
    p = str(tmp_path / "a.db")
    make_db(p, f0_first_idx=f0, l0_first_idx=l0)
    rec = R.read_log(p)
    assert R.first_paired(rec) == phase
    c = R.candidates(rec)
    assert c["tokens"] == expected_tokens(phase, 600)
    # first grid frame dropped, window 3 past + 10 future: first candidate is the 4th frame AFTER the dropped one
    assert c["lp_idx"][0] == phase + 10 * 4 and c["lp_idx"][-1] == list(range(phase, 600, 10))[-11]
    assert np.all(np.diff(c["lp_idx"]) == 10)


def test_route_filter_drops_frames_in_routeless_scenes(tmp_path):
    p = str(tmp_path / "r.db")
    make_db(p, route_empty_scenes=(2, 3))     # lidar idx 200..399 have no route
    c = R.candidates(R.read_log(p))
    base = expected_tokens(0, 600)
    keep = [t for t in base if not (200 <= int(t, 16) < 400)]
    assert c["tokens"] == keep and c["n_dropped_route"] == len(base) - len(keep) > 0
    assert R.candidates(R.read_log(p), has_route=False)["tokens"] == base


def test_RED_wrong_phase_stride_window_change_the_answer(tmp_path):
    p = str(tmp_path / "x.db")
    make_db(p, f0_first_idx=1, l0_first_idx=1)
    rec = R.read_log(p)
    truth = set(R.candidates(rec)["tokens"])
    for delta in (1, -1, 5):                                   # off-by-one cadence: no overlap at all
        assert truth.isdisjoint(R.candidates(rec, phase=1 + delta)["tokens"])
    for st in (9, 11):                                         # wrong stride: nearly nothing in common
        assert len(truth & set(R.candidates(rec, stride=st)["tokens"])) <= 0.2 * len(truth)
    assert len(R.candidates(rec, num_history=5)["tokens"]) == len(truth) - 1      # one more history frame -> exactly one fewer
    assert len(R.candidates(rec, num_history=3)["tokens"]) == len(truth) + 1
    assert len(R.candidates(rec, num_future=11)["tokens"]) == len(truth) - 1
    assert len(R.candidates(rec, drop_first=0)["tokens"]) == len(truth) + 1       # first grid frame kept -> one more
    assert R.first_paired(rec, tol_us=40_000) == 0 != R.first_paired(rec)         # tolerance wide enough pairs every lidar_pc: the phase collapses to 0
    rec2 = R.read_log(p)
    assert R.frame_indices(rec2, tol_us=5_000).size == 0                           # tolerance too tight: no lidar_pc pairs, no frames


def test_module_is_frozen():
    fr = json.load(open(os.path.join(PKG, "raw", "rule_freeze.json")))
    raw = open(os.path.join(HERE, "unseen_token_rule.py"), "rb").read().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == fr["sha256_lf"], "the frozen rule was edited: re-run identity_check on both sets and re-freeze"


PK = "C:/Users/Admin/par18/l1prep/work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/rl_pool/navtrain_token_table.tsv.gz"
real = pytest.mark.skipif(not (os.path.isdir(DB_TV) and os.path.exists(PK)), reason="data drive / token table not mounted")


@real
def test_identity_on_real_logs_and_RED_phase():
    import identity_check as IC
    known = IC.load_known()
    per = json.load(open(os.path.join(PKG, "raw", "identity_214_per_log.json")))
    # one log per grid phase (0, 1, 2) among the 214
    chosen = {}
    for lg in sorted(per):
        rec = R.read_log(f"{DB_TV}/{lg}.db")
        ph = R.first_paired(rec) % 10
        if ph not in chosen:
            chosen[ph] = (lg, rec)
        if len(chosen) == 3:
            break
    assert set(chosen) == {0, 1, 2}
    for ph, (lg, rec) in chosen.items():
        got = set(R.candidates(rec)["tokens"])
        assert known[lg] <= got, f"recall < 1 on {lg}"
        assert len(got) > len(known[lg])                      # a SUPERSET: navtrain is the PDM-filtered subset
        for delta in (1, -1):                                 # RED: off-by-one cadence -> recall collapses to 0
            bad = set(R.candidates(rec, phase=R.first_paired(rec) + delta)["tokens"])
            assert not (known[lg] & bad)


@real
def test_nearest_jpgs_layout_and_four_cameras():
    lg = sorted(f for f in os.listdir(DB_TV) if f.endswith(".db"))[0][:-3]
    rec = R.read_log(f"{DB_TV}/{lg}.db")
    c = R.candidates(rec)
    t = c["tokens"][len(c["tokens"]) // 2]
    ts = int(rec["ts"][c["lp_idx"][len(c["tokens"]) // 2]])
    fr = R.nearest_jpgs(f"{DB_TV}/{lg}.db", ts)
    assert set(fr) == set(R.CAMERAS) == {"CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"}
    for ch, (path, dt) in fr.items():
        assert path.startswith(lg + "/" + ch + "/") and path.endswith(".jpg") and dt <= R.MAX_DT_MS
    assert t in rec["tokens"]


def test_fetch_list_is_the_rules_own_output_and_disjoint_from_navtrain():
    fp, up = os.path.join(PKG, "raw", "fetch_list_s2_s4.json"), os.path.join(PKG, "raw", "U_candidate_tokens.json.gz")
    if not (os.path.exists(fp) and os.path.exists(up)):
        pytest.skip("fetch list not built yet")
    fl = json.load(open(fp))
    toks = json.load(gzip.open(up, "rt"))
    assert fl["n_logs"] == 123 and fl["tokens_per_log"] == 4 and fl["n_tokens"] == 123 * 4 and fl["n_jpgs"] == fl["n_tokens"] * 4
    per = {}
    for it in fl["items"]:
        assert it["token"] in set(toks[it["log"]])            # every listed token is a candidate of the frozen rule
        per[it["log"]] = per.get(it["log"], 0) + 1
        assert set(it["jpgs"]) == set(R.CAMERAS)
        for ch, v in it["jpgs"].items():
            assert v["path"].startswith(it["log"] + "/" + ch + "/") and v["dt_ms"] <= R.MAX_DT_MS
    assert set(per.values()) == {4} and len(per) == 123
    names = json.load(open("C:/Users/Admin/par18/l2fresh/raw/scope_unseen_stratum_U_names.json"))["U_trainval_minus_navtrain"] if os.path.exists("C:/Users/Admin/par18/l2fresh/raw/scope_unseen_stratum_U_names.json") else None
    if names:
        assert set(per) <= set(names)
