"""Tests for p5c_determinism.py (B10). Control: an identical rerun PASSES with 0 tokens over tolerance. RED arms: a perturbed rerun, a copy of pass 1,
a missing token, an error row, a sub-score-only change, an off-subset pass, an edited subset file each must FAIL / REFUSE.
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest test_p5c_determinism.py -q"""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5c_determinism as D   # noqa: E402

ARMS = ["C1", "C0", "base"]


def toks(n):
    return [f"tok{i:05d}" for i in range(n)]


def mk_rows(tokens, seed=0, t0=1.0):
    rng = np.random.default_rng(seed)
    rows = {}
    for i, t in enumerate(tokens):
        r = {"log": f"log{i % 7}", "token": t, "arms": {}, "t_s": t0 + i * 0.01}
        for a in ARMS + ["human"]:
            r["arms"][a] = {s: {m: float(rng.random()) for m in D.METRICS} for s in D.SCORERS}
        rows[t] = r
    return rows


def rerun(rows, subset, t_shift=0.5):
    """a second pass: identical scores, its own timings."""
    out = {}
    for t in subset:
        r = copy.deepcopy(rows[t])
        r["t_s"] = r["t_s"] + t_shift
        out[t] = r
    return out


# --------------------------------------------------------------------------------------------- the subset rule
def test_subset_size_is_the_cards_1215_on_12146():
    assert len(D.choose_subset(toks(12146))) == 1215
    assert D.FRAC == 0.10 and D.SEED == 20261112


def test_subset_is_seeded_sorted_unique_and_stable():
    a, b = D.choose_subset(toks(500)), D.choose_subset(list(reversed(toks(500))))
    assert a == b == sorted(set(a)) and len(a) == 50
    assert D.choose_subset(toks(500), seed=1) != a                  # the seed matters (a constant function would not)


def test_subset_doc_detects_an_edited_file():
    pop = toks(300)
    doc = D.make_subset_doc(pop)
    assert D.check_subset_doc(doc, pop) == []
    bad = copy.deepcopy(doc)
    bad["tokens"] = bad["tokens"][1:] + [pop[0] if pop[0] not in bad["tokens"] else "zzz"]
    assert D.check_subset_doc(bad, pop)                             # RED: tokens edited
    bad2 = copy.deepcopy(doc)
    bad2["seed"] = 7
    assert D.check_subset_doc(bad2, pop)                            # RED: the declared seed no longer gives these tokens
    assert D.check_subset_doc(doc, toks(301))                       # RED: another population


# --------------------------------------------------------------------------------------------- the comparison
def setup(n=200):
    pop = toks(n)
    sub = D.choose_subset(pop)
    r1 = mk_rows(pop)
    return pop, sub, r1, rerun(r1, sub)


def test_CONTROL_identical_rerun_passes_with_zero_tokens_over():
    pop, sub, r1, r2 = setup()
    res = D.compare(r1, r2, sub, ARMS + ["human"])
    assert res["pass"] is True and res["verdict"] == "PASS"
    assert res["n_tokens_over_tolerance"] == 0 and res["max_abs_diff_all"] == 0.0
    assert res["n_compared"] == res["n_subset"] == 20
    assert res["control"] == "B10_determinism" and res["not_counted_in_K"] is True
    for a in ARMS:
        for s in D.SCORERS:
            assert res["arms"][a][s]["n_pdms_over_tolerance"] == 0 and res["arms"][a][s]["n_compared"] == 20


@pytest.mark.parametrize("scorer", ["tracked", "transparent"])
@pytest.mark.parametrize("eps", [1e-12, 1e-9, 1e-3])
def test_RED_a_perturbed_rerun_fails(scorer, eps):
    pop, sub, r1, r2 = setup()
    r2[sub[3]]["arms"]["C1"][scorer]["pdms"] += eps
    res = D.compare(r1, r2, sub, ARMS)
    assert res["pass"] is False and res["n_tokens_over_tolerance"] == 1 and res["over_tolerance_first10"] == [sub[3]]
    assert res["arms"]["C1"][scorer]["n_pdms_over_tolerance"] == 1
    other = "transparent" if scorer == "tracked" else "tracked"
    assert res["arms"]["C1"][other]["n_pdms_over_tolerance"] == 0     # the failing scorer is the one named


def test_RED_a_subscore_only_change_fails_too_Q7():
    pop, sub, r1, r2 = setup()
    r2[sub[0]]["arms"]["base"]["tracked"]["ep"] += 1e-9
    res = D.compare(r1, r2, sub, ARMS)
    assert res["pass"] is False and res["arms"]["base"]["tracked"]["n_pdms_over_tolerance"] == 0
    assert res["arms"]["base"]["tracked"]["n_tokens_over_tolerance_by_metric"]["ep"] == 1


def test_tolerance_edge():
    pop, sub, r1, r2 = setup()
    r2[sub[0]]["arms"]["C0"]["tracked"]["pdms"] += 1e-7
    assert D.compare(r1, r2, sub, ARMS, tol=1e-6)["pass"] is True
    assert D.compare(r1, r2, sub, ARMS, tol=1e-8)["pass"] is False


def test_RED_pass_two_is_a_copy_of_pass_one():
    pop, sub, r1, r2 = setup()
    res = D.compare(r1, {t: r1[t] for t in sub}, sub, ARMS)           # same rows, same t_s: not a second run
    assert res["pass"] is False and any("copy" in p for p in res["problems"])


def test_RED_same_row_files():
    pop, sub, r1, r2 = setup()
    res = D.compare(r1, r2, sub, ARMS, paths1=["/x/rows_s0.jsonl"], paths2=["/x/rows_s0.jsonl"])
    assert res["pass"] is False and any("share row files" in p for p in res["problems"])


def test_RED_missing_token_in_pass_two():
    pop, sub, r1, r2 = setup()
    del r2[sub[5]]
    res = D.compare(r1, r2, sub, ARMS)
    assert res["pass"] is False and res["n_compared"] == len(sub) - 1 and any("missing from pass 2" in p for p in res["problems"])


def test_RED_pass_two_outside_the_subset():
    pop, sub, r1, r2 = setup()
    extra = [t for t in pop if t not in sub][0]
    r2[extra] = rerun(r1, [extra])[extra]
    assert D.compare(r1, r2, sub, ARMS)["pass"] is False


def test_RED_error_rows():
    pop, sub, r1, r2 = setup()
    assert D.compare(r1, r2, sub, ARMS, errs1={}, errs2={sub[0]: "boom"})["pass"] is False


def test_RED_arm_missing_in_a_row():
    pop, sub, r1, r2 = setup()
    del r2[sub[0]]["arms"]["C0"]
    assert D.compare(r1, r2, sub, ARMS)["pass"] is False


def test_nan_in_pass_two_fails():
    pop, sub, r1, r2 = setup()
    r2[sub[0]]["arms"]["C1"]["tracked"]["pdms"] = float("nan")
    assert D.compare(r1, r2, sub, ARMS)["pass"] is False


def test_empty_subset_is_not_a_pass():
    assert D.compare({}, {}, [], ARMS)["pass"] is False


# --------------------------------------------------------------------------------------------- CLI round trip on files
def write_plans(d, tokens):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(1)
    for a in ARMS:
        np.savez(os.path.join(d, f"plans_{a}.npz"), tokens=np.array(tokens), plans8=rng.random((len(tokens), 8, 3)))


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows.values():
            f.write(json.dumps(r) + "\n")


def run_cli(*args):
    return subprocess.run([sys.executable, os.path.join(HERE, "p5c_determinism.py"), *map(str, args)], capture_output=True, text=True, encoding="utf-8")


def test_cli_select_plans_compare_roundtrip(tmp_path):
    pop = toks(120)
    write_plans(tmp_path / "plans", pop)
    assert run_cli("select", "--plans-dir", tmp_path / "plans", "--out", tmp_path / "sub.json").returncode == 0
    doc = json.load(open(tmp_path / "sub.json"))
    assert doc["n_subset"] == 12 and doc["n_total"] == 120
    assert run_cli("plans", "--plans-dir", tmp_path / "plans", "--subset", tmp_path / "sub.json", "--out-dir", tmp_path / "plans2").returncode == 0
    z = np.load(tmp_path / "plans2" / "plans_C1.npz")
    assert sorted(str(t) for t in z["tokens"]) == doc["tokens"] and z["plans8"].shape == (12, 8, 3)
    r1 = mk_rows(pop)
    write_rows(tmp_path / "rows1.jsonl", r1)
    write_rows(tmp_path / "rows2.jsonl", rerun(r1, doc["tokens"]))
    ok = run_cli("compare", "--rows1", tmp_path / "rows1.jsonl", "--rows2", tmp_path / "rows2.jsonl", "--subset", tmp_path / "sub.json", "--arms", *ARMS, "--human", "human", "--out", tmp_path / "ok.json")
    assert ok.returncode == 0, ok.stderr
    assert json.load(open(tmp_path / "ok.json"))["pass"] is True
    r2 = rerun(r1, doc["tokens"])
    r2[doc["tokens"][2]]["arms"]["C0"]["transparent"]["pdms"] += 1e-9
    write_rows(tmp_path / "rows3.jsonl", r2)
    bad = run_cli("compare", "--rows1", tmp_path / "rows1.jsonl", "--rows2", tmp_path / "rows3.jsonl", "--subset", tmp_path / "sub.json", "--arms", *ARMS, "--out", tmp_path / "bad.json")
    assert bad.returncode == 1
    assert json.load(open(tmp_path / "bad.json"))["pass"] is False


def test_cli_refuses_an_edited_subset_file(tmp_path):
    pop = toks(120)
    write_plans(tmp_path / "plans", pop)
    run_cli("select", "--plans-dir", tmp_path / "plans", "--out", tmp_path / "sub.json")
    doc = json.load(open(tmp_path / "sub.json"))
    doc["tokens"] = doc["tokens"][:-1]
    json.dump(doc, open(tmp_path / "sub.json", "w"))
    assert run_cli("plans", "--plans-dir", tmp_path / "plans", "--subset", tmp_path / "sub.json", "--out-dir", tmp_path / "p2").returncode == 2


def test_cli_refuses_arms_with_different_token_sets(tmp_path):
    write_plans(tmp_path / "plans", toks(50))
    np.savez(tmp_path / "plans" / "plans_C0.npz", tokens=np.array(toks(49)), plans8=np.zeros((49, 8, 3)))
    assert run_cli("select", "--plans-dir", tmp_path / "plans", "--out", tmp_path / "s.json").returncode != 0
