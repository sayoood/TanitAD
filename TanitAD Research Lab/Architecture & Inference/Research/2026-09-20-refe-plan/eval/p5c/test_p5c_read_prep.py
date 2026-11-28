"""Tests for p5c_read_prep.py (B5 + B6): converter forms and controls, identity PASS / RED, routing mutation, determinism.
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest test_p5c_read_prep.py -q   (from the tree that holds PK/refe and PK/eval/instruments)"""
import csv
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5c_read_prep as P   # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "..", "..", "refe", "navsim_dac.py")), reason="PK/refe not in this tree")
TRANSP = os.path.join(HERE, "..", "instruments", "transp_score.py")


def synth_seam(tmp, name, n=30, seed=0, with_dump=True):
    rng = np.random.default_rng(seed)
    p20 = np.cumsum(rng.normal(0.5, 0.05, (n, 20, 3)), axis=1)
    p20[..., 2] = np.cumsum(rng.normal(0, 0.01, (n, 20)), axis=1)
    p20 = p20.astype(np.float32)          # the real seam converts the float32 executed plan, and pick20 stores that same float32
    toks = np.array([f"{seed:02d}{i:06x}" for i in range(n)])
    poses = np.stack([P.to_navsim(p20[i]) for i in range(n)]).astype(np.float32)
    seam = os.path.join(tmp, f"{name}_seam.npz")
    np.savez(seam, token=toks, poses=poses, fingerprint=np.array(["x"] * n))
    dump = None
    if with_dump:
        dump = os.path.join(tmp, f"{name}_dump.npz")
        np.savez(dump, token=toks, pick20=p20.astype(np.float32))
    return seam, dump, toks, p20


def test_pick20_form_used_when_dump_has_it(tmp_path):
    seam, dump, toks, p20 = synth_seam(str(tmp_path), "a")
    plans, rep = P.build_arm(seam, dump, None)
    assert rep["form"] == "pick20+plans8" and rep["pick20_control_bitexact"] is True
    assert plans["plans20"].shape == (30, 20, 3) and plans["plans8"].shape == (30, 8, 3)


def test_plans8_form_when_no_dump(tmp_path):
    seam, _, _, _ = synth_seam(str(tmp_path), "a", with_dump=False)
    plans, rep = P.build_arm(seam, None, None)
    assert rep["form"] == "plans8" and "plans20" not in plans and "NOT the registered" in rep["form_note"]


def test_dump_without_pick20_falls_back_and_says_so(tmp_path):
    seam, dump, toks, _ = synth_seam(str(tmp_path), "a")
    old = os.path.join(str(tmp_path), "old_dump.npz")
    np.savez(old, token=toks, proposals=np.zeros((30, 2, 8, 3)))
    _, rep = P.build_arm(seam, old, None)
    assert rep["form"] == "plans8" and "dump_without_pick20" in rep


def test_RED_pick20_not_matching_the_seam_is_refused(tmp_path):
    seam, dump, toks, p20 = synth_seam(str(tmp_path), "a")
    bad = p20.copy()
    bad[7, 4, 0] += 1e-3                       # pose index 4 (t = 1.0 s) lands exactly on a NAVSIM sample: used by to_navsim
    d2 = os.path.join(str(tmp_path), "bad_dump.npz")
    np.savez(d2, token=toks, pick20=bad.astype(np.float32))
    with pytest.raises(SystemExit, match="bit-for-bit"):
        P.build_arm(seam, d2, None)


def test_control_limit_only_12_of_20_poses_are_checked(tmp_path):
    """KNOWN LIMIT of the B3 control (measured here): to_navsim reads only 12 of the 20 pick20 poses (idx 1,2,4,6,7,9,11,12,14,16,17,19),
    so a pick20 that differs ONLY in the other 8 passes the bit-exact check. The transparent arm DOES read all 20."""
    seam, dump, toks, p20 = synth_seam(str(tmp_path), "a")
    assert P.USED_POSE_IDX == [1, 2, 4, 6, 7, 9, 11, 12, 14, 16, 17, 19]
    other = p20.copy()
    for k in set(range(20)) - set(P.USED_POSE_IDX):
        other[:, k, 1] += 0.5
    z = np.load(seam)
    assert P.pick20_equals_seam(other, z["poses"]) is True
    _, rep = P.build_arm(seam, dump, None)
    assert rep["pick20_unverified_poses"] == [0, 3, 5, 8, 10, 13, 15, 18]


def test_dump_token_order_is_aligned_by_token_not_position(tmp_path):
    seam, dump, toks, p20 = synth_seam(str(tmp_path), "a")
    perm = np.random.default_rng(1).permutation(30)
    d2 = os.path.join(str(tmp_path), "perm_dump.npz")
    np.savez(d2, token=toks[perm], pick20=p20[perm].astype(np.float32))
    plans, rep = P.build_arm(seam, d2, None)
    assert rep["pick20_control_bitexact"] is True


def test_subset_is_stable_seeded_and_sorted():
    toks = [f"{i:016x}" for i in range(2000)]
    a, b = P.subset_tokens(toks, 100, 1), P.subset_tokens(list(reversed(toks)), 100, 1)
    assert a == b and len(a) == 100 and a == sorted(a)
    assert P.subset_tokens(toks, 100, 2) != a


def test_convert_deterministic_and_common_tokens(tmp_path):
    s1, _, _, _ = synth_seam(str(tmp_path), "a", with_dump=False)
    s2, _, _, _ = synth_seam(str(tmp_path), "b", with_dump=False)
    outs = []
    for k in (0, 1):
        sys.argv = ["x", "convert", "--arm", f"C1={s1}", f"C0={s1}", "--outdir", str(tmp_path / f"o{k}"), "--subset", "10"]
        assert P.main() == 0
        outs.append(tmp_path / f"o{k}")
    for nm in ("plans_C1.npz", "plans_C0.npz"):
        assert hashlib.sha256((outs[0] / nm).read_bytes()).hexdigest() == hashlib.sha256((outs[1] / nm).read_bytes()).hexdigest()
    assert len(np.load(outs[0] / "plans_C1.npz")["tokens"]) == 10


def test_mixed_forms_flagged(tmp_path):
    s1, d1, _, _ = synth_seam(str(tmp_path), "a")
    sys.argv = ["x", "convert", "--arm", f"C1={s1},{d1}", f"C0={s1}", "--outdir", str(tmp_path / "o")]
    assert P.main() == 0
    assert json.load(open(tmp_path / "o" / "convert_report.json"))["mixed_forms"] is True


# ------------------------------------------------------------------------------------------------ identity (B6)
def fake_rows_and_csv(n=40):
    toks = [f"tok{i:04d}" for i in range(n)]
    rng = np.random.default_rng(0)
    banked, rows = {}, {}
    for t in toks:
        v = {m: float(rng.uniform(0, 1)) for m in P.COL}
        banked[t] = {P.COL[m]: repr(x) for m, x in v.items()}
        rows[t] = {"log": "l", "token": t, "arms": {"A": {"tracked": dict(v)}}}
    return toks, rows, banked


def test_identity_pass_on_equal():
    toks, rows, banked = fake_rows_and_csv()
    r = P.check_identity(rows, [], "A", banked, toks)
    assert r["verdict"] == "PASS" and r["n_over_bar"] == 0 and r["n_scored"] == 40


def test_RED_identity_fails_on_a_1e4_perturbation():
    toks, rows, banked = fake_rows_and_csv()
    rows[toks[5]]["arms"]["A"]["tracked"]["pdms"] += 1e-4
    r = P.check_identity(rows, [], "A", banked, toks)
    assert r["verdict"] == "FAIL" and r["n_over_bar"] == 1 and r["over_first10"][0]["token"] == toks[5]


def test_identity_bar_is_1e6():
    toks, rows, banked = fake_rows_and_csv()
    rows[toks[0]]["arms"]["A"]["tracked"]["ep"] += 5e-7
    assert P.check_identity(rows, [], "A", banked, toks)["verdict"] == "PASS"
    rows[toks[0]]["arms"]["A"]["tracked"]["ep"] += 1e-6
    assert P.check_identity(rows, [], "A", banked, toks)["verdict"] == "FAIL"


def test_identity_fails_on_missing_token_and_on_error_rows():
    toks, rows, banked = fake_rows_and_csv()
    del rows[toks[3]]
    r = P.check_identity(rows, [], "A", banked, toks)
    assert r["verdict"] == "FAIL" and r["n_missing"] == 1
    toks, rows, banked = fake_rows_and_csv()
    assert P.check_identity(rows, [{"token": "tokX", "error": "x"}], "A", banked, toks)["verdict"] == "FAIL"


def test_identity_reads_every_metric_each_one_can_fail():
    for m in P.COL:
        toks, rows, banked = fake_rows_and_csv()
        rows[toks[1]]["arms"]["A"]["tracked"][m] += 1e-3
        assert P.check_identity(rows, [], "A", banked, toks)["verdict"] == "FAIL", m


def test_read_rows_later_good_row_wins_over_error(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"log": "l", "token": "t1", "error": "x"}) + "\n" + json.dumps({"log": "l", "token": "t1", "arms": {}}) + "\n")
    rows, errs = P.read_rows([str(p)])
    assert "t1" in rows and errs == []


def test_harness_csv_skips_the_average_row(tmp_path):
    p = tmp_path / "h.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "token", "valid"] + list(P.COL.values()))
        w.writerow([0, "abc", True] + [1.0] * 7)
        w.writerow([1, "average", True] + [1.0] * 7)
    assert list(P.read_harness_csv(str(p))) == ["abc"]


# ------------------------------------------------------------------------------------------------ routing: the cache line
def test_score_cmds_always_carry_the_metric_cache_flag(tmp_path):
    seam, _, _, _ = synth_seam(str(tmp_path), "a", with_dump=False)
    sys.argv = ["x", "convert", "--arm", f"C1={seam}", "--outdir", str(tmp_path / "pl")]
    P.main()
    for c in P.build_score_cmds(str(tmp_path / "pl"), str(tmp_path / "o"), 2):
        assert "--metric-cache-root" in c and "--split" in c and c[c.index("--split") + 1] == "navtest" and "--human-arm" in c
    assert len(P.build_score_cmds(str(tmp_path / "pl"), str(tmp_path / "o"), 2)) == 2
    with pytest.raises(SystemExit):
        P.build_score_cmds(str(tmp_path / "pl"), str(tmp_path / "o"), 3)       # at most 2 processes
    # all arms in ONE pass: one --plans entry per arm, same command
    cmd = P.build_score_cmds(str(tmp_path / "pl"), str(tmp_path / "o"), 1)[0]
    assert cmd[cmd.index("--plans") + 1].startswith("C1=")


def test_RED_mutation_removes_the_routing_line(tmp_path):
    dst = str(tmp_path / "red.py")
    P.make_red_transp(TRANSP, dst)
    txt = open(dst, encoding="utf-8").read()
    assert P.ROUTE_LINE not in txt and "parts = PT.metric_cache_parts(sc)" in txt
    src = open(TRANSP, encoding="utf-8").read()
    assert src.count(P.ROUTE_LINE) == 1 and len(src) != len(txt)


def test_mutation_refuses_when_the_line_is_absent(tmp_path):
    f = tmp_path / "t.py"
    f.write_text("print('no routing line here')\n")
    with pytest.raises(SystemExit, match="expected 1"):
        P.make_red_transp(str(f), str(tmp_path / "o.py"))


def test_banked_dry_run_results_if_present():
    """the banked 500-token dry run (hold/dryrun): identity PASS on 3 arms at 500/500 and the RED arm DETECTED."""
    cands = [os.environ.get("P5C_DRYRUN_DIR", ""), os.path.join(HERE, "..", "..", "raw", "2026-11-13-p5c-read", "dryrun")]
    d = next((c for c in cands if c and os.path.exists(os.path.join(c, "red_result.json"))), None)
    if d is None:
        pytest.skip("dry-run artifacts not alongside")
    for arm in ("C1", "C0", "base"):
        r = json.load(open(os.path.join(d, f"identity_{arm}.json")))
        assert r["verdict"] == "PASS" and r["n_scored"] == r["n_expected"] == 500 and r["n_over_bar"] == 0
    red = json.load(open(os.path.join(d, "red_result.json")))
    assert red["red_detected"] is True and red["n_over_bar"] > 0
