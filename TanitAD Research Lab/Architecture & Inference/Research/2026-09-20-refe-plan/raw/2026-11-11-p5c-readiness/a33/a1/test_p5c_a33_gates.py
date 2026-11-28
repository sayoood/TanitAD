"""A3.3 gate tests (PI rulings 2026-11-16): TB-0's R-A tolerance 1e-8 separate from TB-1's 1e-9 (--tb0-tol), and the TB-5 label fallback (the ONE ruled row takes its own Tc20s label).

Real inputs: the A1 fixtures (40 real rank-0 bank rows labelled by the pinned dual labeller on the dev box + 40 synthetic rank-1 twins, 33 real P5-R EVAL rows as control).
Every A3.3 behaviour is a named check on a gates module; the REAL module must read all True and every MUTANT (the module text edited) must read at least one False.
Defaults: with no A3.3 argument the gate results are byte-identical to the A3.2 module (base/p5c_table_gates.py)."""
import contextlib
import copy
import io
import json
import os
import shutil
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5c_table_gates as G  # noqa: E402

FIX = HERE + "/fixtures"
TB1 = HERE + "/TB1_control_1018.json"
TMP = os.environ.get("P5C_TMP", "F:/tanitad_scratch/p5c_a33_tmp/gt") + "/a33_gates"
BASE_GATES = os.environ.get("P5C_A32_GATES_BASE", os.path.join(HERE, "..", "..", "base", "p5c_table_gates.py"))
P5C = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c"
R0, R1 = P5C + "/labels/repair_rank0.json", P5C + "/labels/repair_rank1.json"
SHA_A, SHA_B = "ab" * 32, "cd" * 32


def _jl(p):
    return [json.loads(ln) for ln in open(p, encoding="utf-8")]


@pytest.fixture(scope="module")
def W():
    bank = G.load_bank([FIX + "/bank"])
    NR = len(bank["log"])
    recs0 = {}
    for d in _jl(FIX + "/dual_records_40.jsonl"):
        recs0[G.triple(d["log"], d["token"], 0)] = dict(d, rank=0)
        recs0[G.triple(d["log"], d["token"], 1)] = dict(d, rank=1)
    ctl_tc = {}
    for d in _jl(FIX + "/dev_control_tc20s_40.jsonl"):
        ctl_tc[G.triple(d["log"], d["token"], 0)] = d
        ctl_tc[G.triple(d["log"], d["token"], 1)] = dict(d, rank=1)
    ev = G.load_eval_control([FIX + "/eval_control_33.jsonl"], "3.0")
    tb1 = json.load(open(TB1, encoding="utf-8"))
    rep0, rep1 = json.load(open(R0, encoding="utf-8")), json.load(open(R1, encoding="utf-8"))
    for rp in (rep0, rep1):
        for j, x in enumerate(rp["rows"]):
            x["p5r_sim8"] = bool(x["tc20s_sim8"]) and (j % 3 == 0)
    idx = {G.triple(bank["log"][i], bank["token"][i], bank["rank"][i]): i for i in range(NR)}
    ii = [idx[t] for t in ev if t in idx]
    la, lh = G.learn(np.stack([np.asarray(ev[t]["label"], np.float64) for t in ev if t in idx]))
    ta, th = G.learn(bank["traj"][ii])
    REF = {0: {"R_acc": float(np.median(la) / np.median(ta)), "R_hr": float(np.median(lh) / np.median(th))}}
    k0s = sorted(t for t in recs0 if t.endswith("|0"))
    return types.SimpleNamespace(bank=bank, NR=NR, recs0=recs0, ctl=ctl_tc, ev=ev, tb1=tb1, rep0=rep0, rep1=rep1, REF=REF, k0s=k0s)


def fresh(W):
    r = copy.deepcopy(W.recs0)
    for k in r:
        if k.endswith("|1"):                       # the twins share nested lists with rank 0 in the fixture: unshare
            r[k]["tc20s"] = [list(x) for x in r[k]["tc20s"]]
            r[k]["p5r"] = [list(x) for x in r[k]["p5r"]]
    return r


def mark(r, k, reason="tb5_s2_tc20s", sha=SHA_A, p5r_is_tc20s=True, yaw_jump=None):
    """turn record k into a label-fallback record, as `p5c_rj5_path.py fallback` writes it"""
    d = r[k]
    d["p5r_S2_yaw_branch"] = False                 # the failure that earned the fallback
    d["p5r_discarded"] = d["p5r"]
    d["label_traj_sha256"] = sha
    if yaw_jump is not None:
        d["tc20s"][19][2] += yaw_jump
    if p5r_is_tc20s:
        d["p5r"] = copy.deepcopy(d["tc20s"])
    d["p5r_fallback"] = reason
    return d["token"]


def run(Gm, W, recs=None, ctl=None, tb1=None, tc20s_tol=None, **kw):
    tol = Gm.TC20S_TOL if tc20s_tol is None else tc20s_tol
    return Gm.run_all_p5r(W.bank, recs if recs is not None else fresh(W), 0, ctl if ctl is not None else W.ctl, W.ev, tb1 if tb1 is not None else W.tb1, W.rep0, W.rep1, W.NR, None,
                          Gm.BAR_N_TB1, W.REF, 33, tol, **kw)


def st(R, gid):
    return R["gates"][gid]["status"]


def ctl_off(W, d):
    c = copy.deepcopy(W.ctl)
    k = [t for t in c if t.endswith("|0")][3]
    c[k]["tc20s"][3][0] += d
    return c


def tb1_off(W, d):
    t = dict(W.tb1)
    t["max_abs_diff"] = d
    t["bit_identical"] = t["n"] - 5
    return t


def cli_files(W):
    cli = TMP + "/cli"
    shutil.rmtree(cli, ignore_errors=True)
    os.makedirs(cli)
    for nm, obj in (("labels.jsonl", fresh(W)), ("control.jsonl", ctl_off(W, 5e-9))):
        open(f"{cli}/{nm}", "w", encoding="utf-8", newline="\n").write("".join(json.dumps(v) + "\n" for v in obj.values()))
    for nm, obj in (("rep0.json", W.rep0), ("rep1.json", W.rep1)):
        json.dump(obj, open(f"{cli}/{nm}", "w", encoding="utf-8", newline="\n"))
    return cli, ["--label", "p5r", "--bank", FIX + "/bank", "--labels", f"{cli}/labels.jsonl", "--control", f"{cli}/control.jsonl", "--eval-control", FIX + "/eval_control_33.jsonl", "--eval-n", "33",
                 "--tb1", TB1, "--repair0", f"{cli}/rep0.json", "--repair1", f"{cli}/rep1.json", "--expect-rows", str(W.NR), "--tc20s-tol", "1e-9"]


def cli_run(Gm, cli, base_args, extra, tag):
    out = f"{cli}/{tag}"
    with contextlib.redirect_stdout(io.StringIO()):
        rc = Gm.main(base_args + extra + ["--out-dir", out])
    return rc, json.load(open(out + "/gates.json", encoding="utf-8")), json.load(open(out + "/table_meta_fields.json", encoding="utf-8"))


def a33_checks(Gm, W):
    """every A3.3 behaviour on gates module Gm -> {name: bool}"""
    c = {}
    ALLOW = ((W.recs0[W.k0s[0]]["token"], SHA_A),)
    # ---- TB-0 / TB-1 tolerances (ruling 1): 5e-9 sits between 1e-9 and 1e-8, 2e-8 above both
    c["T1 TB-0 at 5e-9 FAILS at the default (R-A 1e-9)"] = st(run(Gm, W, ctl=ctl_off(W, 5e-9)), "TB-0") == "FAIL"
    R = run(Gm, W, ctl=ctl_off(W, 5e-9), tb0_tol=1e-8)
    c["T2 TB-0 at 5e-9 PASSES with tb0_tol 1e-8 (the bar names 1e-08)"] = st(R, "TB-0") == "PASS" and "1e-08" in R["gates"]["TB-0"]["bar"]
    c["T3 TB-0 at 2e-8 still FAILS with tb0_tol 1e-8"] = st(run(Gm, W, ctl=ctl_off(W, 2e-8), tb0_tol=1e-8), "TB-0") == "FAIL"
    R = run(Gm, W, tb1=tb1_off(W, 5e-9), tb0_tol=1e-8)
    c["T4 TB-1 (max_abs_diff 5e-9) STAYS FAIL at 1e-9 when tb0_tol is 1e-8, and TB-0 is untouched by it"] = st(R, "TB-1") == "FAIL" and st(R, "TB-0") == "PASS"
    c["T5 TB-1 (max_abs_diff 5e-10) PASSES at 1e-9"] = st(run(Gm, W, tb1=tb1_off(W, 5e-10), tb0_tol=1e-8), "TB-1") == "PASS"
    c["T6 tb0_tol None == the old behaviour: tc20s_tol 1e-8 alone moves TB-0 (and TB-1) as before"] = st(run(Gm, W, ctl=ctl_off(W, 5e-9), tc20s_tol=1e-8), "TB-0") == "PASS"
    cli, base_args = cli_files(W)
    rc, gj, meta = cli_run(Gm, cli, base_args, ["--tb0-tol", "1e-8", "--tc20s-ranks", "all"], "tb0cli")
    c["T7 CLI --tb0-tol 1e-8 reaches TB-0 (PASS at 5e-9) and is recorded in the meta"] = gj["gates"]["TB-0"]["status"] == "PASS" and meta["p5r"].get("tb0_tol_R_A") == 1e-8
    rc, gj, meta = cli_run(Gm, cli, base_args, ["--tc20s-ranks", "all"], "tb0cli_def")
    c["T8 CLI without --tb0-tol: TB-0 FAILS at 5e-9 and no tb0 key is recorded"] = gj["gates"]["TB-0"]["status"] == "FAIL" and "tb0_tol_R_A" not in meta["p5r"]
    # ---- TB-5 fallback (ruling 2)
    R = run(Gm, W)
    c["F1 no marker, no allow-list: TB-5 PASS and no label_fallback key (pre-A3.3 shape)"] = st(R, "TB-5") == "PASS" and "label_fallback" not in R["gates"]["TB-5"]["detail"]
    r = fresh(W)
    tok = mark(r, W.k0s[0])
    R = run(Gm, W, recs=r, fallback_allow=((tok, SHA_A),))
    d5 = R["gates"]["TB-5"]["detail"]
    c["F2 one allowed marked row: TB-5 PASS, S1/S2/S5 count 0, the fallback is counted and disclosed"] = (st(R, "TB-5") == "PASS" and d5["S1_S2_S5_not_true"] == {"S1": 0, "S2": 0, "S5": 0}
                                                                                                            and d5["label_fallback"]["n"] == 1 and d5["label_fallback"]["violations"] == [])
    c["F3 the row's table label IS its Tc20s label (merged)"] = bool(np.array_equal(R["M"]["label"][next(i for i in range(W.NR) if W.bank["token"][i] == tok and W.bank["rank"][i] == 0)],
                                                                                    np.asarray(r[W.k0s[0]]["tc20s"], np.float64)))
    c["F4 the run's other gates still PASS (TB-2 judges the row by its Tc20s knot deviation)"] = st(R, "TB-2") == "PASS" and "label_fallback_rows_judged_by_tc20s_knot_dev" in R["gates"]["TB-2"]["detail"] and st(R, "TB-4") == "PASS"
    r = fresh(W)
    tok = mark(r, W.k0s[0])
    c["F5 a marked row NOT on the allow-list FAILS TB-5 (empty list)"] = st(run(Gm, W, recs=r), "TB-5") == "FAIL"
    c["F6 ... and under an allow-list naming another row"] = st(run(Gm, W, recs=r, fallback_allow=((tok, SHA_B),)), "TB-5") == "FAIL"
    r = fresh(W)
    t1, t2 = mark(r, W.k0s[0]), mark(r, W.k0s[1])
    c["F7 TWO marked rows, both on the allow-list: TB-5 FAILS (cap 1)"] = st(run(Gm, W, recs=r, fallback_allow=((t1, SHA_A), (t2, SHA_A))), "TB-5") == "FAIL"
    r = fresh(W)
    tok = mark(r, W.k0s[0], p5r_is_tc20s=False)
    c["F8 a marked row whose label is still the P5-R label FAILS TB-5"] = st(run(Gm, W, recs=r, fallback_allow=((tok, SHA_A),)), "TB-5") == "FAIL"
    r = fresh(W)
    tok = mark(r, W.k0s[0], yaw_jump=3.5)
    R = run(Gm, W, recs=r, fallback_allow=((tok, SHA_A),))
    c["F9 a marked row whose NEW label has |dyaw vs traj| >= pi FAILS TB-5 (yaw_branch_ge_pi 1)"] = st(R, "TB-5") == "FAIL" and R["gates"]["TB-5"]["detail"]["yaw_branch_ge_pi"] == 1
    r = fresh(W)
    tok = mark(r, W.k0s[0])
    r[W.k0s[0]]["tc20s"][5][0] = float("nan")
    r[W.k0s[0]]["p5r"] = copy.deepcopy(r[W.k0s[0]]["tc20s"])
    R = run(Gm, W, recs=r, fallback_allow=((tok, SHA_A),))
    c["F10 a marked row whose NEW label is non-finite FAILS TB-5"] = st(R, "TB-5") == "FAIL"
    r = fresh(W)
    tok = mark(r, W.k0s[0], reason="because")
    c["F11 a marker with another reason FAILS TB-5"] = st(run(Gm, W, recs=r, fallback_allow=((tok, SHA_A),)), "TB-5") == "FAIL"
    r = fresh(W)
    r[W.k0s[1]]["p5r_S2_yaw_branch"] = False
    tok = mark(r, W.k0s[0])
    R = run(Gm, W, recs=r, fallback_allow=((tok, SHA_A),))
    c["F12 ANOTHER row failing S2 (unmarked) is still a TB-5 FAIL, even next to the ruled row"] = st(R, "TB-5") == "FAIL" and R["gates"]["TB-5"]["detail"]["S1_S2_S5_not_true"]["S2"] == 1
    r = fresh(W)
    tok = mark(r, W.k0s[0])
    r[W.k0s[0]]["knot_maxdev"] = 1e-3
    c["F13 a marked row whose Tc20s knot deviation is over 1e-9 FAILS TB-2"] = st(run(Gm, W, recs=r, fallback_allow=((tok, SHA_A),)), "TB-2") == "FAIL"
    # CLI: the allow-list flag + the meta
    r = fresh(W)
    tok = mark(r, W.k0s[0])
    open(f"{cli}/labels.jsonl", "w", encoding="utf-8", newline="\n").write("".join(json.dumps(v) + "\n" for v in r.values()))
    rc, gj, meta = cli_run(Gm, cli, base_args, ["--tb0-tol", "1e-8", "--tb5-fallback-allow", f"{tok}:{SHA_A}"], "fbcli")
    c["F14 CLI --tb5-fallback-allow reaches TB-5 (PASS) and the meta counts the fallback (n 1, reason tb5_s2_tc20s)"] = (gj["gates"]["TB-5"]["status"] == "PASS" and meta.get("p5r_label_fallback", {}).get("n") == 1
                                                                                                                       and meta["p5r_label_fallback"]["reasons"] == ["tb5_s2_tc20s"])
    rc, gj, meta = cli_run(Gm, cli, base_args, ["--tb0-tol", "1e-8"], "fbcli_noallow")
    c["F15 CLI without the allow-list: the marked row FAILS TB-5 (a marker alone grants nothing)"] = gj["gates"]["TB-5"]["status"] == "FAIL"
    return c


def test_real_module_reads_all_true(W):
    c = a33_checks(G, W)
    bad = [k for k, v in c.items() if not v]
    assert not bad, bad


MUT = [
    ("tolerance applied to TB-1", "gate_tb1_p5r(bank, M, eval_ctl, tb1, ref, eval_n, tb1_n, tc20s_tol)", "gate_tb1_p5r(bank, M, eval_ctl, tb1, ref, eval_n, tb1_n, tb0_tol if tb0_tol is not None else tc20s_tol)"),
    ("tolerance ignored (TB-0 keeps the TB-1 tolerance)", "g_t = gate_tb0(bank, M_t, tc_control_recs, tol0, recs, tc20s_ranks)", "g_t = gate_tb0(bank, M_t, tc_control_recs, tc20s_tol, recs, tc20s_ranks)"),
    ("tolerance not passed from run_all_p5r to TB-0", "recs, tc20s_ranks, tb0_tol), \"TB-1\"", "recs, tc20s_ranks), \"TB-1\""),
    ("CLI drops --tb0-tol", "a.tc20s_ranks, a.tb0_tol, allow)", "a.tc20s_ranks, None, allow)"),
    ("CLI drops the fallback allow-list", "a.tc20s_ranks, a.tb0_tol, allow)", "a.tc20s_ranks, a.tb0_tol, ())"),
    ("fallback applied to an unlisted row (allow-list not enforced)", "elif (bank[\"token\"][i], M[\"fb_label_traj_sha\"][i]) not in allow:", "elif False:"),
    ("fallback applied to 2 rows (cap lifted)", "MAX_LABEL_FALLBACK_ROWS = 1  ", "MAX_LABEL_FALLBACK_ROWS = 9  "),
    ("fallback leaves the P5-R label in place (label == own Tc20s not required)", "or not np.array_equal(M[\"label\"][i], M[\"fb_tc20s\"][i])", "or False"),
    ("TB-5 skips the new label's dyaw check", "allok = finite & yaw_ok & hz_ok",
     "allok = finite & (yaw_ok | np.array([x is not None for x in M.get('fb_reason', np.full(len(M['source']), None, dtype=object))[rows]])) & hz_ok"),
    ("fallback row still judged by S1/S2/S5 (exclusion dropped)", "and i not in fb_set))", "))"),
    ("TB-5 ignores the marker reason", "if M[\"fb_reason\"][i] not in FALLBACK_LABEL_REASONS:", "if False:"),
    ("TB-2 does not judge the fallback row's knot deviation", "ok = n > 0 and bad == 0 and missing == 0 and not fb_bad_rows.any()", "ok = n > 0 and bad == 0 and missing == 0"),
]


@pytest.mark.parametrize("name,old,new", MUT, ids=[m[0] for m in MUT])
def test_mutant_goes_red(W, name, old, new):
    src = open(G.__file__, encoding="utf-8").read()
    assert src.count(old) == 1, ("mutation target", src.count(old), old[:70])
    Gm = types.ModuleType("gates_mut_a33")
    exec(compile(src.replace(old, new), "gates_mut_a33", "exec"), Gm.__dict__)
    try:
        c = a33_checks(Gm, W)
        red = [k for k, v in c.items() if not v]
    except Exception as e:                                    # a crash is also RED
        red = ["raised " + type(e).__name__]
    assert red, "MUTANT SURVIVED: " + name


def _norm(o, out):
    return json.dumps(o, sort_keys=True, default=str).replace(out.replace("\\", "/"), "<OUT>").replace(out, "<OUT>")


def test_defaults_are_byte_identical_to_the_a32_module(W):
    """no A3.3 argument: gates.json and table_meta_fields.json of the A3.3 module == the A3.2 module's, for the same inputs"""
    assert os.path.exists(BASE_GATES), BASE_GATES
    import importlib.util
    spec = importlib.util.spec_from_file_location("p5c_table_gates_a32", BASE_GATES)
    G0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G0)
    cli, base_args = cli_files(W)
    for extra, tag in (([], "d"), (["--tc20s-ranks", "0"], "d0"), (["--tc20s-ranks", "all"], "da")):
        out_n, out_o = f"{cli}/new_{tag}", f"{cli}/old_{tag}"
        rc_n, gj_n, meta_n = cli_run(G, cli, base_args, extra, f"new_{tag}")
        rc_o, gj_o, meta_o = cli_run(G0, cli, base_args, extra, f"old_{tag}")
        assert rc_n == rc_o
        assert _norm(gj_n, out_n) == _norm(gj_o, out_o).replace("old_" + tag, "new_" + tag), tag
        assert _norm(meta_n, out_n) == _norm(meta_o, out_o), tag
        assert open(f"{out_n}/merged_labels_rank0only.jsonl", "rb").read() == open(f"{out_o}/merged_labels_rank0only.jsonl", "rb").read()


def test_the_table_label_of_a_fallback_row_is_its_tc20s_label_through_the_cli_merge(W):
    cli, base_args = cli_files(W)
    r = fresh(W)
    tok = mark(r, W.k0s[0])
    open(f"{cli}/labels.jsonl", "w", encoding="utf-8", newline="\n").write("".join(json.dumps(v) + "\n" for v in r.values()))
    cli_run(G, cli, base_args, ["--tb0-tol", "1e-8", "--tb5-fallback-allow", f"{tok}:{SHA_A}"], "merge")
    rows = [json.loads(ln) for ln in open(f"{cli}/merge/merged_labels_rank0only.jsonl", encoding="utf-8")]
    row = [x for x in rows if x["token"] == tok and x["rank"] == 0][0]
    assert row["label"] == r[W.k0s[0]]["tc20s"] and row["source"] == "p5r"
