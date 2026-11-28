"""Tests for the W-A7-RJ5 input of w_a7_read.py (PREREG_P5C_A3 section 5, Amendment A3.1).  CPU only, no DB, no navtest, no scorer.
Run: CUDA_VISIBLE_DEVICES=-1 TMP=<scratch> python -m pytest -q test_w_a7_rj5.py     (the dev box's navsim-crun venv: numpy + pytest)

Every refusal has a deliberate-regression arm: the module text is MUTATED (the refusal removed) and the SAME scenario must then go the other way (`test_mutant_*`).
The pinned gates module (A1 pin 5c2f49cf) is imported as it is for control (i'), so the control is the gates' own R-TOL-1/2/3 code."""
import gzip
import importlib.util
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w_a7_read as W  # noqa: E402

SRC = os.path.join(HERE, "w_a7_read.py")


def mk(seed, n=20):
    rng = np.random.default_rng(seed)
    t = np.zeros((n, 3))
    t[:, 0] = np.cumsum(rng.uniform(0.5, 1.0, n))
    t[:, 1] = np.cumsum(rng.uniform(-0.05, 0.05, n))
    t[:, 2] = np.cumsum(rng.uniform(-0.02, 0.02, n))
    return t


def moved(t, d=0.3):
    t = np.array(t, copy=True)
    i = np.arange(len(t))
    t[:, 0] += d * np.sin(0.3 * i + 1.0)
    t[:, 1] += 0.5 * d * np.cos(0.2 * i)
    return t


def rows_fx(n=6):
    return [{"log": f"log{i // 2}", "token": f"t{i}", "traj": mk(i), "tc20s_banked": mk(i)} for i in range(n)]


def join_rows(rows, mutate=None, same=()):
    out = []
    for r in rows:
        sha = W.traj_sha256(r["traj"])
        t = r["traj"] if r["token"] in same else moved(r["traj"])
        out.append({"log_name": r["log"], "token": r["token"], "rank": 0, "bank_key": f"{r['log']}|{r['token']}|0|{sha}", "bank_traj_sha256": sha, "status": "rj5", "source": "local",
                    "rj5_traj_sha256": W.traj_sha256(t), "rj5_traj": t.tolist()})
    return mutate(out) if mutate else out


def write_gz(path, objs):
    with gzip.GzipFile(path, "wb", mtime=0) as g:
        for o in objs:
            g.write((json.dumps(o) + "\n").encode("utf-8"))
    return W.sha256(path)


def label_of(t):
    """a stub of the pinned labeller's record: a deterministic function of the traj it READS"""
    t = np.asarray(t, np.float64)
    i = np.arange(len(t))
    p5r = t.copy()
    p5r[:, 0] += 0.02 * np.cos(0.4 * i)
    p5r[:, 1] -= 0.01 * np.sin(0.5 * i)
    return p5r


def label_recs(rows, fed, sim8=False):
    out = {}
    for r, t in zip(rows, fed):
        p = label_of(t)
        out[(r["log"], r["token"])] = {"log": r["log"], "token": r["token"], "rank": 0, "traj_sha256": W.traj_sha256(t), "tc20s": ((np.asarray(t) + p) / 2).tolist(), "p5r": p.tolist(), "p5r_sim8": sim8}
    return out


def write_labels(run, recs):
    os.makedirs(f"{run}/labels", exist_ok=True)
    with open(f"{run}/labels/labels_s0of1.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in recs.values():
            f.write(json.dumps(r) + "\n")


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mutated(old, new, tmp, name):
    text = open(SRC, encoding="utf-8").read()
    assert text.count(old) == 1, (text.count(old), old[:70])
    p = os.path.join(str(tmp), name + ".py")
    open(p, "w", encoding="utf-8", newline="\n").write(text.replace(old, new))
    return load_mod(p, name)


# ------------------------------------------------------------------ scenarios: True = the module behaved CORRECTLY (refused / passed as it must)
def refused(M, fn, *a, match=None):
    try:
        fn(*a)
    except M.Rj5Refused as e:
        return match is None or match in str(e)
    return False


def sc_bank_file(M, tmp):
    rows = rows_fx()
    p = str(tmp / "bank.jsonl.gz")                      # a BANK file: rows carry `traj`, no status
    sha = write_gz(p, [{"log_name": r["log"], "token": r["token"], "rank": 0, "traj": r["traj"].tolist()} for r in rows])
    return refused(M, M.read_rj5_join, p, sha, match="BANK file")


def sc_join_sha(M, tmp):
    rows = rows_fx()
    p = str(tmp / "j.jsonl.gz")
    write_gz(p, join_rows(rows))
    return refused(M, M.read_rj5_join, p, "0" * 64, match="not the pinned") and refused(M, M.read_rj5_join, p, None, match="required")


def _feed_case(M, tmp, mutate=None, same=(), match=None):
    rows = rows_fx()
    p = str(tmp / "j.jsonl.gz")
    sha = write_gz(p, join_rows(rows, mutate, same))
    return refused(M, lambda: M.rj5_feed_trajs(rows, M.read_rj5_join(p, sha)), match=match)


def sc_wrong_key(M, tmp):
    def mut(o):
        o[1]["bank_traj_sha256"] = o[2]["bank_traj_sha256"]
        o[1]["bank_key"] = o[2]["bank_key"].replace(o[2]["token"], o[1]["token"])
        return o
    return _feed_case(M, tmp, mut, match="wrong-key override REFUSED")


def sc_status_missing(M, tmp):
    def mut(o):
        o[0].update(status="missing", source=None)
        o[0].pop("rj5_traj"), o[0].pop("rj5_traj_sha256")
        return o
    return _feed_case(M, tmp, mut, match="not rj5")


def sc_rank1(M, tmp):
    def mut(o):
        o[0]["rank"] = 1
        return o
    return _feed_case(M, tmp, mut, match="rank-0 keys only")


def sc_bad_rj5_sha(M, tmp):
    def mut(o):
        o[0]["rj5_traj"][3][0] += 1e-9
        return o
    return _feed_case(M, tmp, mut, match="recorded rj5_traj_sha256")


def sc_absent_row(M, tmp):
    return _feed_case(M, tmp, lambda o: o[1:], match="not in the traj source")


def sc_all_unmoved(M, tmp):                              # the traj source IS the bank in effect: every RJ5 traj equals its bank traj
    return _feed_case(M, tmp, None, same={r["token"] for r in rows_fx()}, match="the traj source is the bank")


def sc_labels_from_bank(M, tmp):
    rows = rows_fx()
    p = str(tmp / "j.jsonl.gz")
    sha = write_gz(p, join_rows(rows))
    run = str(tmp / "run")
    os.makedirs(run)
    M.prep_rj5(run, rows, p, sha)
    write_labels(run, label_recs(rows, [r["traj"] for r in rows]))             # labelled from the BANK trajs (a re-used bank-run labels directory)
    return refused(M, M.check_labels_fed, run, rows, match="not computed from the fed RJ5 traj")


def sc_labels_fed_ok(M, tmp):
    rows = rows_fx()
    p = str(tmp / "j.jsonl.gz")
    sha = write_gz(p, join_rows(rows))
    run = str(tmp / "run")
    os.makedirs(run)
    M.prep_rj5(run, rows, p, sha)
    fed = [np.asarray(json.loads(ln)["traj"]) for ln in open(f"{run}/bank/targets_rank0.jsonl", encoding="utf-8")]
    write_labels(run, label_recs(rows, fed))
    M.check_labels_fed(run, rows)
    return True


def i_prime_setup(M, tmp, label_shift=0.0, sim8_flip=False, table_bad=False, drop_record=False, err_row=False):
    rows = rows_fx()
    fed = [moved(r["traj"]) for r in rows]
    run = str(tmp / "run")
    os.makedirs(run + "/bank_real")
    with open(run + "/bank_real/targets_rank0.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"log_name": r["log"], "token": r["token"], "rank": 0, "traj": r["traj"].tolist()}) + "\n")
    open(run + "/bank_real/targets_rank1.jsonl", "w").close()
    c1 = label_recs(rows, fed)                                                   # the C1 records are keyed by the BANK sha (the rekey)
    table, recs = {}, []
    for r in rows:
        d = dict(c1[(r["log"], r["token"])], traj_sha256=W.traj_sha256(r["traj"]))
        table[f"{r['log']}|{r['token']}|0|{W.traj_sha256(r['traj'])}"] = [list(p) for p in d["p5r"]]
        recs.append(d)
    if table_bad:
        k = next(iter(table))
        table[k][0][0] += 1e-3
    if drop_record:
        recs = recs[1:]
    rp, tp = str(tmp / "labels_rekeyed.jsonl"), str(tmp / "table.json.gz")
    open(rp, "w", encoding="utf-8", newline="\n").write("".join(json.dumps(d) + "\n" for d in recs))
    with gzip.GzipFile(tp, "wb", mtime=0) as g:
        g.write(json.dumps({"meta": {}, "table": table}).encode("utf-8"))
    mine = label_recs(rows, fed, sim8=sim8_flip)
    for d in mine.values():
        d["p5r"] = (np.asarray(d["p5r"]) + label_shift).tolist()
    if err_row:
        k = next(iter(mine))
        mine[k] = {"log": k[0], "token": k[1], "rank": 0, "traj_sha256": mine[k]["traj_sha256"], "p5r_error": "RuntimeError: stub"}
    return rows, run, mine, tp, rp


def sc_i_prime(M, tmp, expect_pass, **kw):
    rows, run, mine, tp, rp = i_prime_setup(M, tmp, **kw)
    res = M.control_i_prime(run, rows, mine, tp, rp, W.GATES_PY)
    return res["pass"] is expect_pass


# ------------------------------------------------------------------ the tests
def test_gates_module_is_the_pinned_blob():
    assert W.blob(W.GATES_PY).startswith("5c2f49cf") and W.GATES_BLOB.startswith("5c2f49cf")


def test_feed_happy_path_prep_files_and_sha_map(tmp_path):
    rows = rows_fx()
    p = str(tmp_path / "j.jsonl.gz")
    sha = write_gz(p, join_rows(rows, same={"t1"}))
    run = str(tmp_path / "run")
    os.makedirs(run)
    st = W.prep_rj5(run, rows, p, sha)
    assert st["n_rows"] == 6 and st["n_traj_moved"] == 5 and st["n_traj_identical_to_bank"] == 1
    fed = [json.loads(ln) for ln in open(f"{run}/bank/targets_rank0.jsonl", encoding="utf-8")]
    real = [json.loads(ln) for ln in open(f"{run}/bank_real/targets_rank0.jsonl", encoding="utf-8")]
    jr = {(o["log_name"], o["token"]): o for o in join_rows(rows, same={"t1"})}
    for r, f_, b in zip(rows, fed, real):
        assert f_["traj"] == jr[(r["log"], r["token"])]["rj5_traj"]                     # the pinned labeller reads the RJ5 traj
        assert np.array_equal(np.asarray(b["traj"]), r["traj"])                         # the real bank rows keep the bank traj
    fm = json.load(open(f"{run}/feed_sha_map.json", encoding="utf-8"))
    assert all(v[0] != v[1] for k, v in fm.items() if not k.endswith("|t1")) and fm["log0|t1"][0] == fm["log0|t1"][1]
    assert json.load(open(f"{run}/prep.json", encoding="utf-8"))["read_kind"] == "W-A7-RJ5"
    sc_labels_fed_ok(W, tmp_path / "x") if (tmp_path / "x").mkdir() is None else None


@pytest.mark.parametrize("scenario", [sc_bank_file, sc_join_sha, sc_wrong_key, sc_status_missing, sc_rank1, sc_bad_rj5_sha, sc_absent_row, sc_all_unmoved, sc_labels_from_bank])
def test_red_refusals(tmp_path, scenario):
    assert scenario(W, tmp_path) is True


def test_i_prime_passes_on_the_same_labels_and_fails_on_each_deviation(tmp_path):
    def d(n):
        p = tmp_path / n
        p.mkdir()
        return p
    assert sc_i_prime(W, d("ok"), True)
    assert sc_i_prime(W, d("shift"), False, label_shift=0.01)                    # R-TOL-2: median 0.01 > 1e-3
    assert sc_i_prime(W, d("sim8"), False, sim8_flip=True)                        # R-TOL-1: sim8 differs on every row
    assert sc_i_prime(W, d("tab"), False, table_bad=True)                         # the table's label is not its record's p5r
    assert sc_i_prime(W, d("drop"), False, drop_record=True)                      # a population row without a C1 record
    assert sc_i_prime(W, d("err"), False, err_row=True)                           # a p5r_error row in this run: never silently dropped


def _bank_run_result(tmp_path, **over):
    r = json.load(open(os.path.join(W.SG, "p5c_launchkit", "w_a7", "run", "w_a7_result.json"), encoding="utf-8"))
    for k, v in over.items():
        if k == "control_i":
            r["controls"]["i_reproduction_tf1_tc20s_tracked"]["pass"] = v
        else:
            r[k] = v
    p = str(tmp_path / f"bank_run_{len(over)}_{abs(hash(str(over)))}.json")
    json.dump(r, open(p, "w", encoding="utf-8"))
    return p


def test_bank_run_validity_control(tmp_path):
    real = os.path.join(W.SG, "p5c_launchkit", "w_a7", "run", "w_a7_result.json")
    assert W.check_bank_run_result(real)["pass"] is True                          # MEASURED: the bank-traj run's control (i) passed, outcome W-A7-PASS
    assert W.check_bank_run_result(_bank_run_result(tmp_path, control_i=False))["pass"] is False
    assert W.check_bank_run_result(_bank_run_result(tmp_path, smoke=True))["pass"] is False
    assert W.check_bank_run_result(_bank_run_result(tmp_path, outcome_row="INVALID"))["pass"] is False


def synth_scores(rows, delta=0.0):
    sc = []
    for i, r in enumerate(rows):
        base = 0.5 + 0.05 * i
        arm = lambda v: {"nc": 1.0, "pdms": v}                                   # noqa: E731
        a = {"R_noA7": arm(base), "R_A7": arm(base + delta), "T_noA7": arm(base), "T_A7": arm(base + delta), "RED_R_h19_plus05": arm(base - 0.05), "IDN_R_pre": arm(base), "IDN_R_A7": arm(base)}
        sc.append({"log": r["log"], "token": r["token"], "arms": a, "rerun": dict(a), "idn_plan_equal": True, "label_src": {"p5r": "p5r", "tc20s": "tc20s"}, "tc20s_vs_banked_maxabs": 1e-3})
    return sc


def _analyze_rj5(tmp_path, delta=0.0, drop_input=None, **kw):
    rows, run, mine, tp, rp = i_prime_setup(W, tmp_path, **kw)
    write_labels(run, mine)
    with open(run + "/scores.jsonl", "w", encoding="utf-8") as f:
        for r in synth_scores(rows, delta):
            f.write(json.dumps(r) + "\n")
    rj5 = {"traj_source": "j.jsonl.gz", "join_sha256": "ab" * 32, "c1_table": tp, "c1_records": rp, "gates": W.GATES_PY,
           "bank_run_result": os.path.join(W.SG, "p5c_launchkit", "w_a7", "run", "w_a7_result.json")}
    if drop_input:
        rj5[drop_input] = None
    return W.analyze(run, rows, False, None, rj5)


def test_analyze_rj5_replaces_control_i_and_keeps_ii_to_iv(tmp_path):
    res = _analyze_rj5(tmp_path)
    c = res["controls"]
    assert "i_reproduction_tf1_tc20s_tracked" not in c and c["i_prime_c1_table_reproduction"]["pass"] and c["i_bank_run_instrument_validity"]["pass"]
    assert all(k in c for k in ("ii_red_arm", "iii_identity", "iv_determinism", "coverage"))
    assert res["controls_all_pass"] and res["outcome_row"] == "W-A7-PASS" and res["read_kind"] == "W-A7-RJ5"
    assert res["traj_source"]["file"] == "j.jsonl.gz"


def test_analyze_rj5_outcome_rule_is_a2s_unchanged(tmp_path):
    res = _analyze_rj5(tmp_path, delta=-0.01)                                      # -1.0 pp on every row
    assert res["controls_all_pass"] and res["outcome_row"] == "W-A7-REBUILD"


@pytest.mark.parametrize("kw", [{"label_shift": 0.01}, {"sim8_flip": True}, {"table_bad": True}, {"drop_input": "c1_table"}, {"drop_input": "bank_run_result"}])
def test_analyze_rj5_invalid_when_i_prime_or_instrument_control_fails(tmp_path, kw):
    res = _analyze_rj5(tmp_path, delta=-0.01, **kw)
    assert not res["controls_all_pass"] and res["outcome_row"] == "INVALID"


def test_analyze_bank_mode_is_unchanged(tmp_path):
    rows, sc = rows_fx(), synth_scores(rows_fx())
    run = str(tmp_path)
    os.makedirs(run + "/labels")
    open(run + "/labels/labels_s0of1.jsonl", "w").write("x\n")
    with open(run + "/scores.jsonl", "w") as f:
        for r in sc:
            f.write(json.dumps(r) + "\n")
    banked = {(r["log"], r["token"]): r["arms"]["T_noA7"]["pdms"] for r in sc}
    res = W.analyze(run, rows, False, banked)
    assert "i_reproduction_tf1_tc20s_tracked" in res["controls"] and "i_prime_c1_table_reproduction" not in res["controls"] and "read_kind" not in res


def test_cli_refuses_a_bank_traj_source_before_any_label(tmp_path, monkeypatch, capsys):
    """end to end through main(): the real population (npz), a traj source that is a BANK file -> exit 2, ZZW_A7 STOP, no labeller started"""
    monkeypatch.setattr(W, "check_pins", lambda: {})                               # the pin check needs the dev box's instr/ copy; its own test is test_pins_resolve of test_w_a7.py
    pop = W.load_population(8)
    p = str(tmp_path / "bank_as_source.jsonl.gz")
    sha = write_gz(p, [{"log_name": r["log"], "token": r["token"], "rank": 0, "traj": r["traj"].tolist()} for r in pop])
    run = str(tmp_path / "run")
    called = []
    monkeypatch.setattr(W, "run_labeller", lambda *_: called.append(1))
    monkeypatch.setattr(sys, "argv", ["w_a7_read.py", "--run", run, "--limit", "8", "--traj-source", p, "--expect-join-sha256", sha])
    assert W.main() == 2 and called == []
    out = capsys.readouterr().out
    assert "ZZW_A7 STOP W-A7-RJ5 refused" in out and "BANK file" in out and "pinned blob mismatch" not in out
    assert not os.path.exists(run + "/bank")
    # the same population through a join whose RJ5 trajs ARE the bank trajs (nothing moved): refused as well
    p2 = str(tmp_path / "unmoved.jsonl.gz")
    sha2 = write_gz(p2, join_rows([{"log": r["log"], "token": r["token"], "traj": r["traj"]} for r in pop], same={r["token"] for r in pop}))
    monkeypatch.setattr(sys, "argv", ["w_a7_read.py", "--run", str(tmp_path / "run2"), "--limit", "8", "--traj-source", p2, "--expect-join-sha256", sha2])
    assert W.main() == 2 and called == []
    assert "the traj source is the bank" in capsys.readouterr().out


def test_cli_without_traj_source_is_the_bank_traj_read(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(W, "check_pins", lambda: {})
    monkeypatch.setattr(W, "prep", lambda run, rows: seen.setdefault("prep", len(rows)))
    monkeypatch.setattr(sys, "argv", ["w_a7_read.py", "--run", str(tmp_path / "r"), "--limit", "5", "--stage", "prep"])
    assert W.main() == 0 and seen["prep"] == 5


# ------------------------------------------------------------------ MUTATIONS of w_a7_read.py: each refusal removed, the same scenario must go the other way
MUT = [
    ("bank_file_refusal", '                raise Rj5Refused(f"traj source {path}, line {n}: lacks {lack}', '                pass; (f"traj source {path}, line {n}: lacks {lack}', sc_bank_file),
    ("join_sha_check", '    if got != expect_sha256:\n        raise Rj5Refused(f"traj source {path}: sha256', '    if False:\n        raise Rj5Refused(f"traj source {path}: sha256', sc_join_sha),
    ("wrong_key_check", '        if j["bank_traj_sha256"] != bsha or j["bank_key"] != f"{k[0]}|{k[1]}|0|{bsha}":', '        if False:', sc_wrong_key),
    ("status_check", '        if j["status"] != "rj5":', '        if False:', sc_status_missing),
    ("rank_check", '        if int(j["rank"]) != 0:', '        if False:', sc_rank1),
    ("rj5_sha_check", '        if traj_sha256(t) != j.get("rj5_traj_sha256"):', '        if False:', sc_bad_rj5_sha),
    ("absent_row_check", '        if j is None:\n            raise Rj5Refused(f"{where}: not in the traj source")', '        if j is None:\n            continue', sc_absent_row),
    ("unmoved_check", '    if moved == 0:', '    if False:', sc_all_unmoved),
    ("labels_fed_check", '    if bad:\n        raise Rj5Refused(f"{len(bad)} of {len(rows)} label records were not computed', '    if False:\n        raise Rj5Refused(f"{len(bad)} of {len(rows)} label records were not computed', sc_labels_from_bank),
]


@pytest.mark.parametrize("name,old,new,scenario", MUT, ids=[m[0] for m in MUT])
def test_mutant_refusal_removed_goes_the_other_way(tmp_path, name, old, new, scenario):
    M = mutated(old, new, tmp_path, "mut_" + name)
    d = tmp_path / "s"
    d.mkdir()
    assert scenario(M, d) is False, f"MUTANT SURVIVED: {name}"


def test_mutant_i_prime_always_pass_is_caught(tmp_path):
    M = mutated('    passed = bool(ok_p and det["n_common_rows"] == len(ctl)', '    passed = bool(True or ok_p and det["n_common_rows"] == len(ctl)', tmp_path, "mut_iprime")
    d = tmp_path / "s"
    d.mkdir()
    assert sc_i_prime(M, d, False, label_shift=0.01) is False                      # the mutant passes a label that moved 0.01 m: the check is RED
