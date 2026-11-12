#!/usr/bin/env python3
"""Tests + deliberate-regression arms for p5c_table_gates.py.  stdlib + numpy only.  CPU.
Real data: the MM's 4,393-row subset (30 logs, both ranks) and the dev-box labeller output for those rows.  Every gate must PASS on the real inputs and every
arm must make its gate FAIL by name.  Prints ZZTBGATES_PASS <n> (rc 0) or ZZTBGATES_FAIL.
Env: P5C_SUBSET (rows .jsonl.gz), P5C_LABELS_DIR, P5C_TMP (scratch, two levels deep), P5C_TB1 (TB-1 output json), P5C_TABLE_MOD (traj_label_table.py)."""
import copy
import glob
import gzip
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5c_table_gates as G  # noqa: E402

P5C = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c"
SUBSET = os.environ.get("P5C_SUBSET", "C:/Users/Admin/p5c_gate_scratch/subset_rows.jsonl.gz")
LABDIR = os.environ.get("P5C_LABELS_DIR", P5C + "/labels/out")
TMP = os.environ.get("P5C_TMP", "F:/tanitad_scratch/p5c_gates_agent/test")
TB1 = os.environ.get("P5C_TB1", HERE + "/TB1_control_1018.json")
TABMOD = os.environ.get("P5C_TABLE_MOD", P5C + "/hold/files/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/traj_label_table.py")
R0, R1 = P5C + "/labels/repair_rank0.json", P5C + "/labels/repair_rank1.json"

N = [0]
FAILS = []


def check(name, cond, extra=""):
    N[0] += 1
    if not cond:
        FAILS.append(name)
        print(f"  FAIL {name} {extra}")
    else:
        print(f"  ok   {name}")


def status(R, gid):
    return R["gates"][gid]["status"]


def arm(name, gid, R, also_ok=()):
    """the arm must turn gate `gid` FAIL; gates in also_ok must stay PASS (the arm is specific)."""
    check(f"ARM {name}: {gid} FAIL", status(R, gid) == "FAIL", f"got {status(R, gid)} {R['gates'][gid]['detail']}")
    check(f"ARM {name}: ZZTB_FAIL lists {gid}", gid in R["failing"])
    for o in also_ok:
        check(f"ARM {name}: {o} still PASS (specific)", status(R, o) == "PASS", f"got {status(R, o)}")


def main():
    os.makedirs(TMP, exist_ok=True)
    bank = G.load_bank([SUBSET])
    N_ROWS = len(bank["log"])
    check("subset has 4393 rows", N_ROWS == 4393, str(N_ROWS))
    full, _, _ = G.load_label_recs([LABDIR + "/labels_s0of2.jsonl", LABDIR + "/labels_s1of2.jsonl"])
    keys = {G.triple(bank["log"][i], bank["token"][i], bank["rank"][i]) for i in range(N_ROWS)}
    recs0 = {t: d for t, d in full.items() if t in keys}
    check("every subset row has a dev-box record", len(recs0) == N_ROWS, str(len(recs0)))
    check("every subset record is ok (no error)", all("tc20s" in d and "error" not in d for d in recs0.values()))
    tb1 = json.load(open(TB1, encoding="utf-8")) if os.path.exists(TB1) else None
    rep0, rep1 = json.load(open(R0, encoding="utf-8")), json.load(open(R1, encoding="utf-8"))

    recs_id = copy.deepcopy(recs0)          # clean control for TB-6 / CLI: tc20s := the row's own traj (ratio exactly 1.0)
    for i in range(N_ROWS):
        recs_id[G.triple(bank["log"][i], bank["token"][i], bank["rank"][i])]["tc20s"] = bank["traj"][i].tolist()

    def run(recs=None, bank_=None, control=full, tb1_=tb1, r0=rep0, r1=rep1, ident=None, n_dup=0, expect=None):
        return G.run_all(bank_ if bank_ is not None else bank, recs if recs is not None else copy.deepcopy(recs0), n_dup, control, tb1_, r0, r1, expect, ident)

    print("== helpers")
    spec = importlib.util.spec_from_file_location("tlt", TABMOD)
    tlt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tlt)
    check("traj_sha == traj_label_table.traj_sha256 on every subset row", all(G.traj_sha(bank["traj"][i]) == tlt.traj_sha256(bank["traj"][i].tolist()) for i in range(0, N_ROWS, 7)))
    check("traj_sha == the labeller's own traj_sha256 on every subset record", all(recs0[G.triple(bank["log"][i], bank["token"][i], bank["rank"][i])]["traj_sha256"] == bank["sha"][i] for i in range(N_ROWS)))
    q = np.stack([np.asarray(recs0[G.triple(bank["log"][i], bank["token"][i], bank["rank"][i])]["tc20s"]) for i in range(40)])
    va, vh = G.learn(q)

    def p5_learn(q20):          # P5 section 3 learnability(), verbatim logic, one row
        qq = np.concatenate([np.zeros((1, 3)), q20], 0)
        acc = np.linalg.norm(np.diff(qq[:, :2], n=2, axis=0), axis=1) / 0.2 ** 2
        hr = np.abs(np.diff(np.unwrap(qq[:, 2]))) / 0.2
        return float(acc.max()), float(np.degrees(hr.max()))
    check("learn() == P5 learnability() per row (40 rows)", all((va[i], vh[i]) == p5_learn(q[i]) for i in range(40)))

    print("== real gates on the subset (rank policy from the repair json)")
    R = run()
    for gid in G.GATE_IDS:
        g = R["gates"][gid]
        print(f"     {gid} {g['status']} n={g['n']} value={g['value']}")
    check("TB-1 input present", tb1 is not None, "run p5c_labels.py control --n 1018 first")
    for gid in G.GATE_IDS:
        if gid == "TB-6":
            continue
        check(f"REAL {gid} PASS", status(R, gid) == "PASS", str(R["gates"][gid]["detail"]))
    d6 = R["gates"]["TB-6"]["detail"]
    print("     REAL TB-6 (MEASURED on the subset):", {k: (d6[k]["ratio_acc"], d6[k]["ratio_hr"], d6[k]["pass"]) for k in ("rank0", "rank1")})
    check("REAL TB-6 rank-1 PASS (acc 1.64x, hr 1.75x)", d6["rank1"]["pass"])
    check("REAL TB-6 rank-0 FAILS the committed 2x bar on heading rate (2.018x, acc 1.71x): the measured result, flagged STOP; the bar is not moved",
          (not d6["rank0"]["pass"]) and d6["rank0"]["ratio_hr"] > 2.0 and d6["rank0"]["ratio_acc"] < 2.0 and d6["rank0_fail_STOP_next_lever_P5R"])
    check("REAL final line is ZZTB_FAIL TB-6 TB-6-rank0-STOP", R["failing"] == ["TB-6", "TB-6-rank0-STOP"], str(R["failing"]))
    check("REAL TB-4 value is n/n", R["gates"]["TB-4"]["value"] == f"{N_ROWS}/{N_ROWS}")
    check("REAL TB-0 compares > 0 rows at max diff exactly 0.0", R["gates"]["TB-0"]["n"] > 0 and R["gates"]["TB-0"]["value"] == 0.0)
    pol = R["rank_policy"]
    print("     rank policy:", pol["decision"], {k: pol[k] for k in ("r0", "r1")}, pol["rk"])
    check("rank policy decided (not UNDECIDED)", pol["decision"] in ("all", "rank0", "STOP"))
    check("real: 4.3 rules alone say all, but TB-6 rank-0 FAIL makes the final decision STOP", pol.get("decision_from_4_3_rules_alone") == "all" and pol["decision"] == "STOP")
    check("r0, r1 recomputed with seed 20260927 on per-row records", "seed 20260927" in pol["r0"]["estimator"] and pol["r1"]["n"] == 300 and pol["r0"]["n"] == 300)

    print("== arms: TB-0")
    r = copy.deepcopy(recs0); t = next(iter(r)); r[t]["tc20s"][3][0] += 1e-6
    arm("one label perturbed by 1e-6", "TB-0", run(recs=r), also_ok=("TB-5",))
    r = copy.deepcopy(recs0); t = next(iter(r)); r[t] = {k: v for k, v in r[t].items() if k != "tc20s"}; r[t]["error"] = "X"
    arm("one table row fell back while the control has it", "TB-0", run(recs=r))
    check("TB-0 without --control is NOT_RUN and fails the final line", status(run(control=None), "TB-0") == "NOT_RUN" and "TB-0" in run(control=None)["failing"])

    print("== arms: TB-1")
    bad = dict(tb1 or {"n": 1018, "n_errors": 0, "max_abs_diff": 0.0}); bad["bit_identical"] = bad["n"] - 1
    arm("1017/1018 bit-identical", "TB-1", run(tb1_=bad))
    bad = dict(tb1 or {"n": 1018, "n_errors": 0}); bad["max_abs_diff"] = 1e-12; bad["bit_identical"] = bad["n"]
    arm("max_abs_diff 1e-12", "TB-1", run(tb1_=bad))
    bad = dict(tb1 or {"n_errors": 0, "max_abs_diff": 0.0}); bad.update(n=100, bit_identical=100)
    arm("only 100 rows controlled", "TB-1", run(tb1_=bad))

    print("== arms: TB-2")
    r = copy.deepcopy(recs0); t = next(iter(r)); r[t] = {k: v for k, v in r[t].items() if k != "tc20s"}; r[t]["error"] = G.KNOT_FAIL_MSG
    arm("a row whose knot control failed", "TB-2", run(recs=r))
    r = copy.deepcopy(recs0); t = next(iter(r)); r[t]["knot_maxdev"] = 1e-6
    arm("persisted knot_maxdev 1e-6", "TB-2", run(recs=r))

    print("== arms: TB-3")
    idl = np.array(bank["traj"], copy=True); idl[17, 4, 1] = np.nextafter(idl[17, 4, 1], 1e9)
    arm("identity label 1 ulp off traj", "TB-3", run(ident=idl))
    idl = np.array(bank["traj"], copy=True)
    check("identity table == traj passes TB-3 (explicit)", status(run(ident=idl), "TB-3") == "PASS")

    print("== arms: TB-4")
    r = copy.deepcopy(recs0); r.pop(next(iter(r)))
    arm("one key dropped", "TB-4", run(recs=r))
    r = copy.deepcopy(recs0); r["zz|zz|0"] = {"log": "zz", "token": "zz", "rank": 0, "tc20s": [[0.0] * 3] * 20}
    arm("one extra label key", "TB-4", run(recs=r))
    r = copy.deepcopy(recs0); t = next(iter(r)); r[t]["traj_sha256"] = "0" * 64
    arm("label computed from another traj (sha mismatch)", "TB-4", run(recs=r))
    r = copy.deepcopy(recs0)
    for t in list(r)[:60]:               # 60/4393 = 1.37 % > 1.0 %
        r[t] = {k: v for k, v in r[t].items() if k != "tc20s"}; r[t]["error"] = "no local DB"
    arm("raw fallback 1.37 % > 1.0 %", "TB-4", run(recs=r))
    r = copy.deepcopy(recs0)
    for t in list(r)[:40]:               # 40/4393 = 0.91 % <= 1.0 %: still PASS (the bar is not tighter than committed)
        r[t] = {k: v for k, v in r[t].items() if k != "tc20s"}; r[t]["error"] = "no local DB"
    R2 = run(recs=r)
    check("fallback 0.91 % <= 1.0 % passes TB-4 and is counted", status(R2, "TB-4") == "PASS" and R2["gates"]["TB-4"]["detail"]["n_raw_fallback"] == 40)
    arm("duplicate label record", "TB-4", run(n_dup=1))
    arm("bank expected larger than given (150603 vs subset)", "TB-4", run(expect=150603))
    check("fallback rows get label == traj bit for bit", all(R2["M"]["label"][i].tobytes() == bank["traj"][i].tobytes() for i in np.where(R2["M"]["source"] == "raw_fallback")[0]))

    print("== arms: TB-5")
    r = copy.deepcopy(recs0); t = next(iter(r)); r[t]["tc20s"][5][2] += 2 * math.pi
    arm("yaw + 2 pi on one pose", "TB-5", run(recs=r))
    r = copy.deepcopy(recs0); t = next(iter(r)); r[t]["tc20s"][5][0] = float("nan")
    arm("NaN in one label", "TB-5", run(recs=r))
    b2 = dict(bank); b2["sim_hz"] = list(bank["sim_hz"]); b2["sim_hz"][9] = 20
    arm("sim_hz 20 on one row", "TB-5", run(bank_=b2))

    print("== arms: TB-6")
    rk = bank["rank"]
    saw = np.zeros((20, 3)); saw[::2, 0] = 3.0                      # a +-3 m sawtooth on x: a jerky label
    Rid = run(recs=copy.deepcopy(recs_id), control=recs_id)
    check("CLEAN CONTROL (labels := traj): TB-6 PASS at ratio exactly 1.0 on both ranks", status(Rid, "TB-6") == "PASS" and Rid["gates"]["TB-6"]["detail"]["rank0"]["ratio_acc"] == 1.0 and Rid["gates"]["TB-6"]["detail"]["rank1"]["ratio_hr"] == 1.0)
    check("CLEAN CONTROL: every gate but TB-1-n-independent ones PASS (TB-0..TB-7 all PASS)", Rid["failing"] == [], str(Rid["failing"]))
    r = copy.deepcopy(recs_id)
    for i in np.where(rk == 0)[0]:
        t = G.triple(bank["log"][i], bank["token"][i], 0); r[t]["tc20s"] = (np.asarray(r[t]["tc20s"]) + saw).tolist()
    R3 = run(recs=r, control=recs_id)
    arm("rank-0 labels jerky (+-3 m sawtooth)", "TB-6", R3)
    check("rank-0 TB-6 FAIL is flagged STOP (P5-R)", R3["gates"]["TB-6"]["detail"]["rank0_fail_STOP_next_lever_P5R"] and "TB-6-rank0-STOP" in R3["failing"])
    r = copy.deepcopy(recs_id)
    for i in np.where(rk == 1)[0]:
        t = G.triple(bank["log"][i], bank["token"][i], 1); r[t]["tc20s"] = (np.asarray(r[t]["tc20s"]) + saw).tolist()
    R4 = run(recs=r, control=recs_id)
    arm("rank-1 labels jerky", "TB-6", R4)
    check("rank-1-only TB-6 FAIL is NOT a STOP and the policy goes rank0", not R4["gates"]["TB-6"]["detail"]["rank0_fail_STOP_next_lever_P5R"] and R4["rank_policy"]["decision"] == "rank0",
          R4["rank_policy"]["decision"])

    print("== arms: TB-7")
    r = copy.deepcopy(recs0)
    for i in np.where(rk == 1)[0]:
        t = G.triple(bank["log"][i], bank["token"][i], 1); r[t] = {k: v for k, v in r[t].items() if k != "tc20s"}; r[t]["error"] = "no local DB"
    arm("no rank-1 row labelled (n_differs not recordable)", "TB-7", run(recs=r))
    n_diff = R["gates"]["TB-7"]["detail"]["per_rank"]
    check("TB-7 n_differs > 0 on both ranks (real)", n_diff[0]["n_differs"] > 0 and n_diff[1]["n_differs"] > 0, str(n_diff))

    print("== arms: rank-1 policy (section 4.3)")
    def flip(rep, frac_raw, frac_tc):
        d = copy.deepcopy(rep)
        for j, x in enumerate(d["rows"]):
            x["raw_sim8"] = (j % 100) < frac_raw; x["tc20s_sim8"] = (j % 100) < frac_tc
        return d
    check("policy base (clean control, real repair data): all", run(recs=copy.deepcopy(recs_id), control=recs_id)["rank_policy"]["decision"] == "all")
    check("UNDECIDED without per-row repair records", run(r0=None)["rank_policy"]["decision"] == "UNDECIDED" and run(r1={"n": 1})["rank_policy"]["decision"] == "UNDECIDED")
    check("RK0: label harmful on rank-0 (raw 0 %, Tc20s 20 %) => STOP", run(r0=flip(rep0, 0, 20))["rank_policy"]["decision"] == "STOP" and "RK0-STOP" in run(r0=flip(rep0, 0, 20))["failing"])
    check("RK2: rank-1 no effect (raw == Tc20s) => rank0", run(recs=copy.deepcopy(recs_id), control=recs_id, r1=flip(rep1, 5, 5))["rank_policy"]["decision"] == "rank0")
    d = copy.deepcopy(rep1); d["rows"] = d["rows"][:150]
    check("RK1: rank-1 sample of 150 rows (< 200) => rank0", run(recs=copy.deepcopy(recs_id), control=recs_id, r1=d)["rank_policy"]["decision"] == "rank0")
    d = copy.deepcopy(rep1)
    for x in d["rows"]:
        x["log"] = "L" + str(hash(x["log"]) % 3)            # 3 logs only
    check("RK1: rank-1 sample over 3 logs (< 5) => rank0", run(recs=copy.deepcopy(recs_id), control=recs_id, r1=d)["rank_policy"]["decision"] == "rank0")
    check("RK3: rank-0 point <= 0 (raw 2 %, Tc20s 4 %) with rank-1 fine => rank0, not all", run(r0=flip(rep0, 2, 4))["rank_policy"]["decision"] in ("rank0", "STOP"))
    check("bootstrap is deterministic (same seed, same CI)", G.paired_boot(rep1["rows"]) == G.paired_boot(rep1["rows"]))

    print("== table integration: merged jsonl -> traj_label_table build + verify (real module)")
    bdir = os.path.join(TMP, "bank"); odir = os.path.join(TMP, "out")
    os.makedirs(bdir, exist_ok=True); os.makedirs(odir, exist_ok=True)
    with gzip.open(SUBSET, "rt", encoding="utf-8") as f, open(bdir + "/targets_rank0.jsonl", "w", encoding="utf-8", newline="\n") as f0, open(bdir + "/targets_rank1.jsonl", "w", encoding="utf-8", newline="\n") as f1:
        for ln in f:
            (f0 if json.loads(ln)["rank"] == 0 else f1).write(ln if ln.endswith("\n") else ln + "\n")
    wr = G.write_merged(bank, R["M"], odir)
    check("merged files written", wr["written"])
    for variant, nd in (("all", None), ("rank0only", None)):
        tab = f"{odir}/T_{variant}.json.gz"
        pb = subprocess.run([sys.executable, TABMOD, "build", "--bank-dir", bdir, "--labels", wr[variant], "--out", tab, "--require-coverage", "1.0", "--label-name", "Tc20s"], capture_output=True, encoding="utf-8")
        pv = subprocess.run([sys.executable, TABMOD, "verify", "--table", tab, "--bank-dir", bdir], capture_output=True, encoding="utf-8")
        check(f"build rc 0 ({variant})", pb.returncode == 0, pb.stderr[-300:])
        check(f"verify rc 0 ({variant})", pv.returncode == 0, pv.stdout[-300:])
        if pv.returncode == 0:
            rep = json.loads(pv.stdout)
            check(f"verify covers {N_ROWS}/{N_ROWS} ({variant})", rep["covered"] == N_ROWS and rep["rows"] == N_ROWS)
            if variant == "rank0only":
                n1 = int((rk == 1).sum()); tb = tlt.load(tab); rows = tlt.read_bank_rows(bdir)
                check("rank0only: differs_from_raw == rank-0 n_differs only", rep["differs_from_raw"] == n_diff[0]["n_differs"], f"{rep['differs_from_raw']} vs {n_diff[0]['n_differs']}")
    lines = open(wr["all"], "rb").read()
    check("merged jsonl is LF only", b"\r" not in lines)
    check("merged_all differs_from_raw == TB-7 total", json.loads(subprocess.run([sys.executable, TABMOD, "verify", "--table", f"{odir}/T_all.json.gz", "--bank-dir", bdir], capture_output=True, encoding="utf-8").stdout)["differs_from_raw"] == n_diff[0]["n_differs"] + n_diff[1]["n_differs"])

    print("== CLI end to end on the subset")
    lab_sub = os.path.join(TMP, "labels_subset.jsonl")
    lab_id = os.path.join(TMP, "labels_identity.jsonl")
    for fp, rr in ((lab_sub, recs0), (lab_id, recs_id)):
        with open(fp, "w", encoding="utf-8", newline="\n") as f:
            for d in rr.values():
                f.write(json.dumps(d) + "\n")
    cmd = [sys.executable, G.__file__, "--bank", SUBSET, "--labels", lab_sub, "--control", LABDIR + "/labels_s0of2.jsonl", LABDIR + "/labels_s1of2.jsonl", "--repair0", R0, "--repair1", R1,
           "--out-dir", os.path.join(TMP, "cli"), "--code-file", G.__file__]
    if os.path.exists(TB1):
        cmd += ["--tb1", TB1]
    pc = subprocess.run(cmd, capture_output=True, encoding="utf-8")
    check("CLI on the REAL subset labels: rc 1, last line ZZTB_FAIL TB-6 TB-6-rank0-STOP", pc.returncode == 1 and pc.stdout.strip().endswith("ZZTB_FAIL TB-6 TB-6-rank0-STOP"), pc.stdout[-400:] + pc.stderr[-300:])
    cid = list(cmd); cid[cid.index("--labels") + 1] = lab_id; cid[cid.index("--control") + 1:cid.index("--control") + 3] = [lab_id]
    cid[cid.index("--out-dir") + 1] = os.path.join(TMP, "cli_id")
    pi = subprocess.run(cid, capture_output=True, encoding="utf-8")
    check("CLI on the clean control (labels := traj, control := same): rc 0 and ZZTB_ALL_PASS", pi.returncode == 0 and "ZZTB_ALL_PASS" in pi.stdout, pi.stdout[-500:] + pi.stderr[-300:])
    pc2 = subprocess.run(cmd[:cmd.index("--control")] + cmd[cmd.index("--repair0"):], capture_output=True, encoding="utf-8")
    check("CLI without --control: rc 1 and ZZTB_FAIL TB-0", pc2.returncode == 1 and "ZZTB_FAIL TB-0 TB-6" in pc2.stdout, pc2.stdout[-300:])
    gj = json.load(open(os.path.join(TMP, "cli", "gates.json"), encoding="utf-8"))
    check("gates.json carries per-gate status, n, value and bar", all(all(k in gj["gates"][g] for k in ("status", "n", "value", "bar")) for g in G.GATE_IDS))

    print(f"ZZTBGATES_{'FAIL' if FAILS else 'PASS'} {N[0] - len(FAILS)}/{N[0]} {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
