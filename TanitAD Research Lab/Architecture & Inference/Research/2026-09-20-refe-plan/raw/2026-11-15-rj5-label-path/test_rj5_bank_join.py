"""Tests for rj5_bank_join.py (synthetic data, tmp dirs only). Run: python -m pytest -q test_rj5_bank_join.py  (needs only the stdlib + pytest)."""
import gzip
import json
import os
import struct
import hashlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rj5_bank_join as J


def traj(seed):
    return [[float(seed) + i * 0.1, 0.01 * i, 0.001 * seed] for i in range(20)]


def ref_sha(t):                                           # independent re-implementation of traj_label_table.traj_sha256
    return hashlib.sha256(b"".join(struct.pack("<d", float(v)) for p in t for v in p)).hexdigest()


def wr(p, rows, gz=False):
    op = gzip.open if gz else open
    with op(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def mk(tmp):
    bank = [{"log_name": "L1", "token": f"t{i}", "rank": 0, "traj": traj(i)} for i in range(4)] + [{"log_name": "L1", "token": "t0", "rank": 1, "traj": traj(99)}]
    wr(tmp / "bank.jsonl.gz", bank, gz=True)
    for d in ("a", "b"):
        os.makedirs(tmp / d)
    k5 = {"positive_jerk_limit_when_nonnegative_acc": 5.0}
    k1 = {"positive_jerk_limit_when_nonnegative_acc": 1.0}
    wr(tmp / "a" / "rj5_x.jsonl", [
        {"log_name": "L1", "token": "t0", "rank": 0, "arm": "RJ5", "traj": traj(10), "knobs": k5},
        {"log_name": "L1", "token": "t1", "rank": 0, "arm": "R0", "traj": traj(1), "knobs": k1, "bank_dmax": 0.0},
        {"log_name": "L1", "token": "t2", "rank": 0, "arm": "RJ5", "error": "boom"}])
    wr(tmp / "b" / "rj5_y.jsonl", [
        {"log_name": "L1", "token": "t0", "rank": 0, "arm": "RJ5", "traj": traj(10), "knobs": k5},      # identical duplicate across sources
        {"log_name": "L1", "token": "t1", "rank": 0, "arm": "RJ5", "traj": traj(1), "knobs": k5},       # RJ5 == bank (inert row)
        {"log_name": "L9", "token": "zz", "rank": 0, "arm": "RJ5", "traj": traj(3), "knobs": k5}])      # not in bank
    return tmp


def run(tmp, extra=()):
    argv = ["join", "--bank", str(tmp / "bank.jsonl.gz"), "--rows", f"A={tmp / 'a'}", "--rows", f"B={tmp / 'b'}", "--out-dir", str(tmp / "out")] + list(extra)
    return J.main(argv)


def test_sha_matches_reference():
    assert J.traj_sha256(traj(7)) == ref_sha(traj(7))


def test_join_coverage_sources_and_control(tmp_path):
    mk(tmp_path)
    assert run(tmp_path) == 0
    m = json.load(open(tmp_path / "out" / "rj5_bank_join_manifest.json"))
    assert m["n_bank_rank0_keys"] == 4 and m["n_bank_rows_other_rank_skipped"] == 1
    assert m["status_counts"] == {"rj5": 2, "missing": 2, "conflict": 0}
    assert m["coverage_by_source"] == {"A": 1, "B": 1}                     # t0 won by A (first listed), t1 by B
    assert m["per_source"]["B"]["dup_identical"] == 1 and m["n_rj5_rows_not_in_bank"] == 1
    assert m["control_R0"]["n"] == 1 and m["control_R0"]["passes_bar"] and m["control_R0"]["n_sha_equal_bank"] == 1
    rows = [json.loads(x) for x in gzip.open(tmp_path / "out" / "rj5_bank_join.jsonl.gz", "rt")]
    r0 = rows[0]
    assert r0["bank_key"] == f"L1|t0|0|{ref_sha(traj(0))}" and r0["rj5_traj_sha256"] == ref_sha(traj(10))
    assert rows[1]["dmax_vs_bank"] == 0.0


def test_deterministic_sha(tmp_path):
    mk(tmp_path)
    run(tmp_path)
    s1 = json.load(open(tmp_path / "out" / "rj5_bank_join_manifest.json"))["output_sha256"]
    run(tmp_path)
    assert s1 == json.load(open(tmp_path / "out" / "rj5_bank_join_manifest.json"))["output_sha256"]


def test_conflicting_duplicate_refused(tmp_path):
    mk(tmp_path)
    wr(tmp_path / "b" / "rj5_z.jsonl", [{"log_name": "L1", "token": "t0", "rank": 0, "arm": "RJ5", "traj": traj(11),
                                         "knobs": {"positive_jerk_limit_when_nonnegative_acc": 5.0}}])
    assert run(tmp_path) == 3
    m = json.load(open(tmp_path / "out" / "rj5_bank_join_manifest.json"))
    assert m["n_conflict_keys"] == 1 and m["status_counts"]["conflict"] == 1


def test_knob_mismatch_refused(tmp_path):
    mk(tmp_path)
    wr(tmp_path / "b" / "rj5_k.jsonl", [{"log_name": "L1", "token": "t3", "rank": 0, "arm": "RJ5", "traj": traj(3),
                                         "knobs": {"positive_jerk_limit_when_nonnegative_acc": 1.0}}])
    assert run(tmp_path) == 3


def test_control_over_bar_is_reported(tmp_path):
    mk(tmp_path)
    t = traj(2)
    t[5][0] += 1e-3
    wr(tmp_path / "b" / "rj5_c.jsonl", [{"log_name": "L1", "token": "t2", "rank": 0, "arm": "R0", "traj": t,
                                         "knobs": {"positive_jerk_limit_when_nonnegative_acc": 1.0}}])
    run(tmp_path)
    m = json.load(open(tmp_path / "out" / "rj5_bank_join_manifest.json"))
    assert m["control_R0"]["n_over_bar"] == 1 and not m["control_R0"]["passes_bar"]


def test_min_coverage_gate(tmp_path):
    mk(tmp_path)
    assert run(tmp_path, ["--min-coverage", "0.9"]) == 3


def test_rekey(tmp_path):
    mk(tmp_path)
    run(tmp_path)
    lab = [{"log": "L1", "token": "t0", "rank": 0, "traj_sha256": ref_sha(traj(10)), "p5r": traj(10)},
           {"log": "L1", "token": "t1", "rank": 0, "traj_sha256": ref_sha(traj(1)), "p5r": traj(1)}]
    wr(tmp_path / "labels_s0of1.jsonl", lab)
    jp = str(tmp_path / "out" / "rj5_bank_join.jsonl.gz")
    assert J.main(["rekey", "--join", jp, "--labels", str(tmp_path / "labels_s0of1.jsonl"), "--out", str(tmp_path / "bl.jsonl")]) == 0
    o = [json.loads(x) for x in open(tmp_path / "bl.jsonl")]
    assert o[0]["traj_sha256"] == ref_sha(traj(0)) and o[0]["rj5_traj_sha256"] == ref_sha(traj(10))
    lab[0]["traj_sha256"] = "0" * 64
    wr(tmp_path / "labels_bad.jsonl", lab)
    assert J.main(["rekey", "--join", jp, "--labels", str(tmp_path / "labels_bad.jsonl"), "--out", str(tmp_path / "bl2.jsonl")]) == 3
