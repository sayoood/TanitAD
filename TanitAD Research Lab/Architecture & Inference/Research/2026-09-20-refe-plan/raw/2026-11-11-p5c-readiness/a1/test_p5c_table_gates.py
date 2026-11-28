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


def legacy():
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


FIX = HERE + "/fixtures"


def p5r_tests():
    """PREREG_P5C amendment A1 (--label p5r) gates.  Real inputs: 40 rank-0 bank rows (P5-S npz raw trajs) labelled by p5c_dual_labels.py on the dev box (33 of them are
    P5-R EVAL rows, whose published labels are the TB-0 / TB-1 control); rank-1 twins are SYNTHETIC copies (same log/token/traj, rank 1) because no real rank-1 row is on this box.
    Every arm must turn its gate FAIL by name."""
    n0, f0 = N[0], len(FAILS)
    os.makedirs(TMP, exist_ok=True)
    print("== P5-R mode (A1): real fixtures")
    bank = G.load_bank([FIX + "/bank"])
    NR = len(bank["log"])
    check("fixture bank: 40 rank-0 + 40 synthetic rank-1 twins", NR == 80 and int((bank["rank"] == 0).sum()) == 40)
    d0 = [json.loads(ln) for ln in open(FIX + "/dual_records_40.jsonl", encoding="utf-8")]
    recs0 = {}
    for d in d0:
        recs0[G.triple(d["log"], d["token"], 0)] = dict(d, rank=0)
        recs0[G.triple(d["log"], d["token"], 1)] = dict(d, rank=1)
    ctl_tc = {G.triple(d["log"], d["token"], 0): d for d in map(json.loads, open(FIX + "/dev_control_tc20s_40.jsonl", encoding="utf-8"))}
    ctl_tc.update({G.triple(d["log"], d["token"], 1): dict(d, rank=1) for d in map(json.loads, open(FIX + "/dev_control_tc20s_40.jsonl", encoding="utf-8"))})
    ev = G.load_eval_control([FIX + "/eval_control_33.jsonl"], "3.0")
    check("eval control: 33 real P5-R EVAL rows", len(ev) == 33)
    tb1 = json.load(open(TB1, encoding="utf-8"))
    rep0, rep1 = json.load(open(R0, encoding="utf-8")), json.load(open(R1, encoding="utf-8"))
    for rp in (rep0, rep1):                         # a P5-R repair file: p5r_sim8 := a sim8 that is not simply Tc20s's, deterministic
        for j, x in enumerate(rp["rows"]):
            x["p5r_sim8"] = bool(x["tc20s_sim8"]) and (j % 3 == 0)
    rep0_old, rep1_old = json.load(open(R0, encoding="utf-8")), json.load(open(R1, encoding="utf-8"))   # Tc20s-only repair files (no p5r_sim8)

    def ref_of(bank_, ctl_):
        """the dev-box EVAL TB-6 ratios on the control rows themselves (rank 0 only in the fixtures): what verdict.json holds for the full 1,200 rows per rank."""
        idx = {G.triple(bank_["log"][i], bank_["token"][i], bank_["rank"][i]): i for i in range(len(bank_["log"]))}
        ii = [idx[t] for t in ctl_ if t in idx]
        la, lh = G.learn(np.stack([np.asarray(ctl_[t]["label"], np.float64) for t in ctl_ if t in idx]))
        ta, th = G.learn(bank_["traj"][ii])
        return {0: {"R_acc": float(np.median(la) / np.median(ta)), "R_hr": float(np.median(lh) / np.median(th))}}
    REF = ref_of(bank, ev)
    REF_ID = {0: {"R_acc": 1.0, "R_hr": 1.0}}

    def run(recs=None, bank_=None, ctl=ctl_tc, evc=ev, tb1_=tb1, r0=rep0, r1=rep1, ident=None, n_dup=0, expect=None, ref=REF, eval_n=33, tc20s_tol=G.TC20S_TOL):
        return G.run_all_p5r(bank_ if bank_ is not None else bank, recs if recs is not None else copy.deepcopy(recs0), n_dup, ctl, evc, tb1_, r0, r1, expect, ident, G.BAR_N_TB1, ref, eval_n, tc20s_tol)

    ev_keys = sorted(ev)
    R = run(expect=NR)
    for gid in G.GATE_IDS:
        g = R["gates"][gid]
        print(f"     {gid} {g['status']} n={g['n']} value={g['value']}")
    for gid in G.GATE_IDS:
        if gid == "TB-6":
            continue
        check(f"P5R REAL {gid} PASS", status(R, gid) == "PASS", str(R["gates"][gid]["detail"])[:300])
    d0r = R["gates"]["TB-0"]["detail"]["p5r_vs_eval_labels"]["per_rank"][0]
    check("P5R REAL TB-0: p5r max abs diff vs published EVAL labels exactly 0.0, sim8 identical 33/33, R-TOL-1/2/3 all True, Tc20s warm start 0.0",
          R["gates"]["TB-0"]["detail"]["p5r_vs_eval_labels"]["max_abs_diff"] == 0.0 and d0r["n_sim8_identical"] == 33 and d0r["R-TOL-1"] and d0r["R-TOL-2"] and d0r["R-TOL-3"]
          and R["gates"]["TB-0"]["detail"]["tc20s_warm_start_vs_dev_build"]["value"] == 0.0)
    check("no 1e-6 default remains: the module has no P5R_TOL and no --p5r-tol flag", not hasattr(G, "P5R_TOL") and "--p5r-tol" not in open(G.__file__, encoding="utf-8").read())
    vref = G.load_verdict_ref(P5C + "/../p5r/raw/verdict.json") if os.path.exists(P5C + "/../p5r/raw/verdict.json") else None
    check("load_verdict_ref reads P5-R's verdict.json (rank 0 R_acc 1.1582, R_hr 1.5638; rank 1 R_acc 1.1475, R_hr 1.3437)", vref is None or (
        abs(vref[0]["R_acc"] - 1.1581809446765852) < 1e-12 and abs(vref[1]["R_hr"] - 1.3437289308216207) < 1e-12 and sorted(vref) == [0, 1]))
    check("P5R REAL TB-1 reproduces 33/33 EVAL labels", R["gates"]["TB-1"]["value"] == "33/33")
    check("P5R REAL TB-2 is S3 (max over rows <= 1e-6)", R["gates"]["TB-2"]["name"] == "S3 re-simulation identity" and R["gates"]["TB-2"]["value"] <= 1e-6)
    d6 = R["gates"]["TB-6"]["detail"]
    print("     P5R REAL TB-6 (40 rows per rank; informative only):", {k: (d6[k]["ratio_acc"], d6[k]["ratio_hr"], d6[k]["pass"]) for k in ("rank0", "rank1")})
    check("P5R REAL TB-6 computed on p5r rows (n 40 + 40)", d6["rank0"]["n"] == 40 and d6["rank1"]["n"] == 40 and "p5r" in d6["population"])
    check("P5R REAL TB-6 label differs from the Tc20s TB-6 on the same rows (it reads the p5r field)",
          G.run_all(bank, copy.deepcopy(recs0), 0, ctl_tc, tb1, rep0_old, rep1_old, NR)["gates"]["TB-6"]["detail"]["rank0"]["ratio_acc"] != d6["rank0"]["ratio_acc"])

    print("== Tc20s mode is unchanged")
    import inspect
    spec_old = importlib.util.spec_from_file_location("p5c_table_gates_landed", P5C + "/gates/p5c_table_gates.py")
    Gold = importlib.util.module_from_spec(spec_old)
    spec_old.loader.exec_module(Gold)
    same = [n for n in ("traj_sha", "load_bank", "load_label_recs", "merge", "learn", "gate_tb2", "ident_check", "gate_tb3", "table_keys", "gate_tb4", "gate_tb5", "paired_boot",
                        "rank_policy", "write_merged", "meta_fields")]      # gate_tb0 / gate_tb1 / run_all gained the R-A `tol` parameter (default 0.0 == frozen): checked by behaviour below
    check("legacy functions are source-identical to the landed gates module", all(inspect.getsource(getattr(G, n)) == inspect.getsource(getattr(Gold, n)) for n in same),
          str([n for n in same if inspect.getsource(getattr(G, n)) != inspect.getsource(getattr(Gold, n))]))
    check("legacy gate_tb0 / gate_tb1 / run_all keep tol default 0.0 (the frozen behaviour) and the CLI default is 1e-9",
          inspect.signature(G.gate_tb0).parameters["tol"].default == 0.0 and inspect.signature(G.gate_tb1).parameters["tol"].default == 0.0
          and inspect.signature(G.run_all).parameters["tc20s_tol"].default == 0.0 and G.TC20S_TOL == 1e-9)
    check("legacy constants identical", all(getattr(G, c) == getattr(Gold, c) for c in ("BAR_KNOT", "BAR_LEARN", "BAR_FALLBACK", "BAR_N_TB1", "B_BOOT", "SEED_BOOT", "GATE_IDS")))
    Ra, Rb = G.run_all(bank, copy.deepcopy(recs0), 0, ctl_tc, tb1, rep0_old, rep1_old, NR), Gold.run_all(bank, copy.deepcopy(recs0), 0, ctl_tc, tb1, rep0_old, rep1_old, NR)
    ADDED = ("tol", "n_sim8_compared", "n_sim8_verdict_differs", "bit_identical")      # detail keys the R-A change added to TB-0 / TB-1; every other field must be identical at tol 0.0

    def strip(g):
        return json.dumps(dict(g, detail={k: v for k, v in g["detail"].items() if k not in ADDED}), default=str)
    check("run_all(tol default 0.0) on the same inputs: every gate result identical to the landed module (ignoring the 4 added detail keys)", all(strip(Ra["gates"][g]) == strip(Rb["gates"][g]) for g in G.GATE_IDS)
          and Ra["failing"] == Rb["failing"] and json.dumps(Ra["rank_policy"], default=str) == json.dumps(Rb["rank_policy"], default=str))
    check("run_all ignores the dual records' p5r_* fields: Tc20s labels identical to the Tc20s-only fixture",
          all(Ra["M"]["label"][i].tobytes() == np.asarray(ctl_tc[G.triple(bank["log"][i], bank["token"][i], bank["rank"][i])]["tc20s"], np.float64).tobytes() for i in range(NR)))

    print("== R-A: Tc20s TB-0 / TB-1 tolerance 1e-9 (unconditional); 0 reproduces the frozen gate")
    k = ev_keys[0]
    tc_keys = [t for t in recs0 if t.endswith("|0")]
    kt = tc_keys[0]
    r = copy.deepcopy(recs0); r[kt]["tc20s"][3][0] += 1e-12
    check("Tc20s table 1e-12 off the dev-box build: FAIL at --tc20s-tol 0 (frozen), PASS at the default 1e-9",
          status(run(recs=r, tc20s_tol=0.0), "TB-0") == "FAIL" and status(run(recs=r), "TB-0") == "PASS" and status(G.run_all(bank, r, 0, ctl_tc, tb1, rep0_old, rep1_old, NR), "TB-0") == "FAIL")
    check("Tc20s mode with tol 1e-9: 1e-12 passes TB-0 and TB-1 (run_all(tc20s_tol=1e-9))", status(G.run_all(bank, r, 0, ctl_tc, tb1, rep0_old, rep1_old, NR, None, G.BAR_N_TB1, 1e-9), "TB-0") == "PASS")
    r = copy.deepcopy(recs0); r[kt]["tc20s"][3][0] += 1e-8
    arm("Tc20s table 1e-8 off the dev-box build (> 1e-9)", "TB-0", run(recs=r))
    check("... and the same 1e-8 passes if the tolerance is explicitly 1e-7", status(run(recs=r, tc20s_tol=1e-7), "TB-0") == "PASS")
    r = copy.deepcopy(recs0); r[kt]["tc20s_sim8"] = False; ctl2 = copy.deepcopy(ctl_tc); ctl2[kt]["tc20s_sim8"] = True
    R_s = run(recs=r, ctl=ctl2)
    arm("R-A: Tc20s sim8 verdict differs on one common row (labels identical)", "TB-0", R_s)
    check("R-A sim8 compare counted (n_sim8_compared 1, differs 1)", R_s["gates"]["TB-0"]["detail"]["tc20s_warm_start_vs_dev_build"]["n_sim8_compared"] == 1
          and R_s["gates"]["TB-0"]["detail"]["tc20s_warm_start_vs_dev_build"]["n_sim8_verdict_differs"] == 1)
    r = copy.deepcopy(recs0); r[kt]["tc20s_sim8"] = False
    R_s1 = run(recs=r)
    check("R-A: sim8 only on one side is 'not available': PASS, n_sim8_compared 0", status(R_s1, "TB-0") == "PASS" and R_s1["gates"]["TB-0"]["detail"]["tc20s_warm_start_vs_dev_build"]["n_sim8_compared"] == 0)
    r = copy.deepcopy(recs0); r[kt] = {kk: v for kk, v in r[kt].items() if kk != "tc20s"}; r[kt]["error"] = "X"
    arm("R-A: a control-ok row fell back in the table", "TB-0", run(recs=r))
    ok_b = dict(tb1); ok_b.update(max_abs_diff=1e-12, bit_identical=ok_b["n"] - 5)
    check("R-A TB-1: max_abs_diff 1e-12 with 5 rows not bit-identical PASSES at 1e-9 and FAILS at tol 0", status(run(tb1_=ok_b), "TB-1") == "PASS" and G.gate_tb1(ok_b, 1018, 0.0)["status"] == "FAIL")
    bad = dict(tb1); bad["max_abs_diff"] = 1e-8
    arm("R-A TB-1: max_abs_diff 1e-8", "TB-1", run(tb1_=bad))
    bad = dict(tb1); bad["n_errors"] = 1
    arm("R-A TB-1: one control error", "TB-1", run(tb1_=bad))
    bad = dict(tb1); bad["n"] = 100
    arm("R-A TB-1: only 100 rows controlled", "TB-1", run(tb1_=bad))
    check("TB-0 without --eval-control is NOT_RUN and fails the final line", status(run(evc=None), "TB-0") == "NOT_RUN" and "TB-0" in run(evc=None)["failing"])
    check("TB-0 without the dev-box Tc20s control is NOT_RUN", status(run(ctl=None), "TB-0") == "NOT_RUN")

    print("== R-B arms: R-TOL-1 / R-TOL-2 / R-TOL-3, per rank (33 real rows)")
    def crit(Rr, gid="TB-0"):
        return Rr["gates"][gid]["detail"]["p5r_vs_eval_labels" if gid == "TB-0" else "p5r_eval_reproduction"]["failed_criteria"]
    check("baseline: failed_criteria == []", crit(R) == [], str(crit(R)))
    r = copy.deepcopy(recs0); r[k]["p5r"][3][0] += 2e-6
    check("p5r label off by 2e-6 on one EVAL row now PASSES (R-TOL-2's median/p99 bound, not 1e-6)", status(run(recs=r), "TB-0") == "PASS")
    r = copy.deepcopy(recs0); r[k]["p5r_sim8"] = not r[k]["p5r_sim8"]
    R_f = run(recs=r)
    arm("one sim8 verdict flipped in 33 rows (32/33 = 97.0 % < 99.5 %)", "TB-0", R_f)
    check("... named R-TOL-1 (rank 0), and TB-1 fails too", "rank0:R-TOL-1" in crit(R_f) and status(R_f, "TB-1") == "FAIL")
    r = copy.deepcopy(recs0); r[k]["p5r"][3][0] += 2e-2
    R_g = run(recs=r)
    check("one row 2e-2 off: max recorded (2e-2) but p99 over 33 rows is interpolated: R-TOL-2 reads the max only as a record", R_g["gates"]["TB-0"]["detail"]["p5r_vs_eval_labels"]["per_rank"][0]["rowmax_max"] >= 2e-2 - 1e-12)
    r = copy.deepcopy(recs0); r[k]["tc20s"][2][1] = float(np.nextafter(r[k]["tc20s"][2][1], 1e9))
    arm("Tc20s warm start 1 ulp off the dev-box build (p5r untouched), tol 0", "TB-0", run(recs=r, tc20s_tol=0.0))
    check("... but within the R-A 1e-9 default it passes", status(run(recs=r), "TB-0") == "PASS")
    r = copy.deepcopy(recs0); r[k] = {kk: v for kk, v in r[k].items() if not kk.startswith("p5r") or kk == "p5r_error"}; r[k]["p5r_error"] = "X"
    arm("an EVAL row fell back in the table", "TB-0", run(recs=r))
    check("no verdict.json at all: R-TOL-3 still runs on its row-matched reference and PASSES (verdict.json is context only)", status(run(ref=None), "TB-0") == "PASS" and crit(run(ref=None)) == [])
    vfake = {0: {"R_acc": 5.0, "R_hr": 9.0}}
    Rv = run(ref=vfake)
    d0v = Rv["gates"]["TB-0"]["detail"]["p5r_vs_eval_labels"]["per_rank"][0]
    check("a verdict.json that disagrees wildly is RECORDED (verdict_json_context) and does not gate", status(Rv, "TB-0") == "PASS" and abs(d0v["verdict_json_context"]["ref_minus_verdict_hr"]) > 1.0)
    vreal = G.load_verdict_ref(P5C + "/../p5r/raw/verdict.json") if os.path.exists(P5C + "/../p5r/raw/verdict.json") else None
    if vreal:
        Rm = run(ref=vreal)
        d0m = Rm["gates"]["TB-0"]["detail"]["p5r_vs_eval_labels"]["per_rank"][0]
        check("ROW-MATCHED reference: these 33 rows have hr ratio 1.99 vs verdict.json's 1.56 (1,200 rows); with identical labels TB-0 and TB-1 still PASS (diff 0 against the same-row reference)",
              abs(d0m["tb6_ratio_hr"] - vreal[0]["R_hr"]) > 0.3 and d0m["tb6_diff_hr"] == 0.0 and d0m["tb6_diff_acc"] == 0.0 and status(Rm, "TB-0") == "PASS" and status(Rm, "TB-1") == "PASS", str(d0m))
    evfull = G.load_eval_control([P5C + "/../p5r/raw/eval_r0.jsonl"], "3.0") if os.path.exists(P5C + "/../p5r/raw/eval_r0.jsonl") else None
    if evfull:
        Mx = G.merge_p5r(bank, copy.deepcopy(recs0))
        okx, dx = G._p5r_vs_control(bank, Mx, evfull, None)
        px = dx["per_rank"][0]
        check("with the REAL eval_r0.jsonl records (1,200 control rows, 33 in the bank): reference from the records' learn / raw.learn, equals the recomputed learn exactly, same ratios as the slim fixture",
              px["n_common_rows"] == 33 and "records" in px["tb6_ref_source"] and px["reference_learn_vs_recomputed_max_abs_diff"] == 0.0 and px["tb6_diff_hr"] == 0.0 and abs(px["tb6_ref_hr"] - REF[0]["R_hr"]) < 1e-12, str(px))
    arm("eval_n 34 expected, 33 reproduced", "TB-1", run(eval_n=34))
    check("TB-1 without --eval-control is NOT_RUN", status(run(evc=None), "TB-1") == "NOT_RUN")

    print("== R-B edge arms on 13x-inflated fixtures (429 EVAL rows): threshold boundaries")
    def inflate(m):
        B = {"log": [], "token": [], "rank": [], "traj": [], "sim_hz": [], "sha": [], "files": []}
        Rr, Ee, Cc = {}, {}, {}
        for j in range(m):
            for i in range(NR):
                lg, tk, rk = bank["log"][i], bank["token"][i], int(bank["rank"][i])
                tk2 = f"{tk}#{j}"
                B["log"].append(lg); B["token"].append(tk2); B["rank"].append(rk); B["traj"].append(bank["traj"][i]); B["sim_hz"].append(bank["sim_hz"][i]); B["sha"].append(bank["sha"][i])
                t0, t1 = G.triple(lg, tk, rk), G.triple(lg, tk2, rk)
                Rr[t1] = dict(copy.deepcopy(recs0[t0]), token=tk2)
                Cc[t1] = dict(copy.deepcopy(ctl_tc[t0]), token=tk2)
                if t0 in ev:
                    Ee[t1] = copy.deepcopy(ev[t0])
        B["rank"] = np.asarray(B["rank"], np.int64); B["traj"] = np.stack(B["traj"])
        return B, Rr, Ee, Cc
    bankI, recsI, evI, ctlI = inflate(13)
    NI, NE = len(bankI["log"]), len(evI)
    check("inflated fixtures: 1040 bank rows, 429 EVAL control rows", NI == 1040 and NE == 429, f"{NI} {NE}")
    eI = sorted(evI)

    def runI(recs=None, evc=None, ref=REF):
        return G.run_all_p5r(bankI, recs if recs is not None else copy.deepcopy(recsI), 0, ctlI, evc if evc is not None else copy.deepcopy(evI), tb1, rep0, rep1, NI, None, G.BAR_N_TB1, ref, NE, G.TC20S_TOL)
    RI = runI()
    check("inflated baseline: TB-0 and TB-1 PASS, failed_criteria []", status(RI, "TB-0") == "PASS" and status(RI, "TB-1") == "PASS" and crit(RI) == [], str(crit(RI)))

    def flip(rr, ks, val=True):
        for kk in ks:
            rr[kk]["p5r_sim8"] = val
        return rr
    R1f = runI(recs=flip(copy.deepcopy(recsI), eI[:1]))
    check("R-TOL-1/3 boundary: 1 sim8 verdict flipped in 429 (99.77 % identical, +0.233 pp <= 0.25) PASSES", status(R1f, "TB-0") == "PASS", str(crit(R1f)))
    R2f = runI(recs=flip(copy.deepcopy(recsI), eI[:2]))
    arm("R-TOL-3: 2 sim8 flips in 429 (+0.466 pp > 0.25 pp)", "TB-0", R2f)
    check("... named exactly rank0:R-TOL-3 (R-TOL-1 still 99.53 % >= 99.5 %)", crit(R2f) == ["rank0:R-TOL-3"], str(crit(R2f)))
    arm("R-TOL-3: the same sim8 arm fails TB-1", "TB-1", R2f)
    evb = copy.deepcopy(evI)
    for kk in eI[:3]:
        evb[kk]["sim8"] = True                      # dev side True on rows 0-2 ...
    rb = flip(copy.deepcopy(recsI), eI[3:6])        # ... pod side True on rows 3-5: equal fail RATES, 6 verdicts differ (98.6 % identical)
    R3f = runI(recs=rb, evc=evb)
    arm("R-TOL-1 only: 6 verdicts differ in 429 with equal fail rates (98.6 % < 99.5 %)", "TB-0", R3f)
    check("... named exactly rank0:R-TOL-1 (R-TOL-2 and R-TOL-3 pass)", crit(R3f) == ["rank0:R-TOL-1"], str(crit(R3f)))

    def bump(rr, ks, dx):
        for kk in ks:
            rr[kk]["p5r"][3][0] += dx
        return rr
    R4f = runI(recs=bump(copy.deepcopy(recsI), eI[:4], 1e-2))
    pr4 = R4f["gates"]["TB-0"]["detail"]["p5r_vs_eval_labels"]["per_rank"][0]
    check("R-TOL-2 boundary: 4 of 429 rows 1e-2 off: p99 stays under 5e-3, PASS, and the max (1e-2) is RECORDED, not gated", status(R4f, "TB-0") == "PASS" and pr4["rowmax_max"] >= 1e-2 - 1e-12 and pr4["rowmax_p99"] <= 5e-3, str(pr4))
    R5f = runI(recs=bump(copy.deepcopy(recsI), eI[:5], 1e-2))
    arm("R-TOL-2 p99: 5 of 429 rows 1e-2 off (p99 7.2e-3 > 5e-3)", "TB-0", R5f)
    check("... named exactly rank0:R-TOL-2", crit(R5f) == ["rank0:R-TOL-2"], str(crit(R5f)))
    arm("R-TOL-2 p99: the same arm fails TB-1", "TB-1", R5f)
    R6f = runI(recs=bump(copy.deepcopy(recsI), eI[:215], 2e-3))
    arm("R-TOL-2 median: 215 of 429 rows 2e-3 off (median 2e-3 > 1e-3, p99 <= 5e-3)", "TB-0", R6f)
    check("... named rank0:R-TOL-2", "rank0:R-TOL-2" in crit(R6f), str(crit(R6f)))
    check("a verdict.json 0.02 away no longer gates (context only): the inflated baseline still PASSES", status(runI(ref={0: {"R_acc": REF[0]["R_acc"], "R_hr": REF[0]["R_hr"] + 0.02}}), "TB-0") == "PASS")
    rn = copy.deepcopy(recsI)
    for kk in eI:
        for pi in range(20):
            rn[kk]["p5r"][pi][2] += 9e-4 * (1 if pi % 2 == 0 else -1)      # per-row diff 9e-4 <= R-TOL-2's median bound, but the label's heading rate is noisier
    R7f = runI(recs=rn)
    pr7 = R7f["gates"]["TB-0"]["detail"]["p5r_vs_eval_labels"]["per_rank"][0]
    arm("R-TOL-3 ratio: every label's yaw gets +-9e-4 alternating noise (row diff 9e-4, inside R-TOL-2)", "TB-0", R7f)
    check("... named R-TOL-3 with R-TOL-2 and R-TOL-1 still true (ratio shift beyond +-0.01)", "rank0:R-TOL-3" in crit(R7f) and "rank0:R-TOL-2" not in crit(R7f) and "rank0:R-TOL-1" not in crit(R7f), str(pr7))
    rl = copy.deepcopy(recsI)
    for kk in eI:
        rl[kk]["p5r"] = (np.asarray(rl[kk]["p5r"]) * np.array([1.0, 1.0, 1.0])).tolist()
        rl[kk]["p5r"][10][2] += 0.4                 # a yaw spike: the label's heading rate (and so the TB-6 ratio) moves
    R8f = runI(recs=rl)
    arm("label-induced: every EVAL label gets a 0.4 rad yaw spike", "TB-0", R8f)
    check("... which names R-TOL-2 and R-TOL-3 (the label moved and the decision statistic with it)", {"rank0:R-TOL-2", "rank0:R-TOL-3"} <= set(crit(R8f)), str(crit(R8f)))

    print("== arms: TB-2 = S3")
    r = copy.deepcopy(recs0); r[k]["p5r_S3_dev"] = 2e-6
    arm("S3 2e-6 on one row", "TB-2", run(recs=r))
    r = copy.deepcopy(recs0); r[k].pop("p5r_S3_dev")
    arm("S3 field missing", "TB-2", run(recs=r))
    r = copy.deepcopy(recs0); r[k]["p5r_S3_dev"] = 1e-6
    check("S3 exactly 1e-6 passes (bar is <=)", status(run(recs=r), "TB-2") == "PASS")
    r = copy.deepcopy(recs0); r[k]["knot_maxdev"] = 1e-6
    check("the Tc20s knot field no longer drives TB-2 in p5r mode (replaced by S3)", status(run(recs=r), "TB-2") == "PASS")

    print("== arms: TB-4 (+S4), TB-5 (+S1/S2/S5)")
    r = copy.deepcopy(recs0); t0 = ev_keys[0]
    r[t0] = {kk: v for kk, v in r[t0].items() if not kk.startswith("p5r") or kk == "p5r_error"}; r[t0]["p5r_error"] = "ValueError: boom"
    R2 = run(recs=r)
    arm("one p5r solver fallback in rank 0 (2.5 % of rank 0 > 1 %)", "TB-4", R2)
    check("S4 per-rank detail counts the fallback", R2["gates"]["TB-4"]["detail"]["S4_per_rank"][0]["n_raw_fallback"] == 1 and R2["gates"]["TB-4"]["detail"]["S4_per_rank"][1]["n_raw_fallback"] == 0)
    i0 = [i for i in range(NR) if bank["rank"][i] == 0 and G.triple(bank["log"][i], bank["token"][i], 0) == t0][0]
    check("a fallback row gets label == traj bit for bit and is excluded from TB-6/TB-7", R2["M"]["label"][i0].tobytes() == bank["traj"][i0].tobytes() and R2["M"]["source"][i0] == "raw_fallback")
    r = copy.deepcopy(recs0); r[t0].pop("p5r")
    check("a record with no p5r field and no p5r_error is a raw fallback, never a silent pass", run(recs=r)["M"]["source"][i0] == "raw_fallback")
    r = copy.deepcopy(recs0); r[t0]["p5r"][5][2] += 2 * math.pi
    arm("p5r yaw + 2 pi", "TB-5", run(recs=r))
    r = copy.deepcopy(recs0); r[t0]["p5r"][5][0] = float("nan")
    arm("NaN in a p5r label", "TB-5", run(recs=r))
    r = copy.deepcopy(recs0); r[t0]["p5r_S5_lambda_reached"] = False
    arm("S5 flag false", "TB-5", run(recs=r))
    r = copy.deepcopy(recs0); r[t0]["p5r_S1_finite"] = False
    arm("S1 flag false", "TB-5", run(recs=r))
    r = copy.deepcopy(recs0); r[t0]["p5r_S2_yaw_branch"] = False
    arm("S2 flag false", "TB-5", run(recs=r))
    r = copy.deepcopy(recs0); r[t0]["traj_sha256"] = "0" * 64
    arm("p5r computed from another traj (sha mismatch)", "TB-4", run(recs=r))

    print("== arms: TB-6 (binding), TB-7")
    rk = bank["rank"]
    saw = np.zeros((20, 3)); saw[::2, 0] = 3.0
    recs_id = copy.deepcopy(recs0)
    for i in range(NR):
        t = G.triple(bank["log"][i], bank["token"][i], bank["rank"][i]); recs_id[t]["p5r"] = bank["traj"][i].tolist()
    ctl_id = {t: dict(d, tc20s=d["p5r"]) for t, d in recs_id.items()}
    for t in recs_id:
        recs_id[t]["tc20s"] = recs_id[t]["p5r"]
    ev_id = {t: {"label": recs_id[t]["p5r"], "sim8": recs_id[t].get("p5r_sim8")} for t in ev}
    Rid = run(recs=copy.deepcopy(recs_id), ctl=ctl_id, evc=ev_id, expect=NR, r0=rep0, r1=rep1, ref=REF_ID)
    check("CLEAN CONTROL (p5r := traj): TB-6 ratio exactly 1.0 on both ranks and PASS", status(Rid, "TB-6") == "PASS" and Rid["gates"]["TB-6"]["detail"]["rank0"]["ratio_acc"] == 1.0
          and Rid["gates"]["TB-6"]["detail"]["rank1"]["ratio_hr"] == 1.0)
    check("CLEAN CONTROL: all of TB-0..TB-7 PASS", Rid["failing"] == [], str(Rid["failing"]) + str({g: Rid["gates"][g]["status"] for g in G.GATE_IDS}))
    r = copy.deepcopy(recs_id)
    for t in r:
        if t.endswith("|0"):
            r[t]["p5r"] = (np.asarray(r[t]["p5r"]) + saw).tolist()
    R3 = run(recs=r, ctl=ctl_id, evc=ev_id, expect=NR, ref=REF_ID)
    arm("rank-0 p5r labels jerky", "TB-6", R3)
    check("rank-0 TB-6 FAIL is flagged STOP with the P5-T next lever named", R3["gates"]["TB-6"]["detail"]["rank0_fail_STOP_next_lever_P5R"] and "TB-6-rank0-STOP" in R3["failing"]
          and "P5-T" in R3["gates"]["TB-6"]["bar"] and "P5-T" in R3["gates"]["TB-6"]["detail"]["rank0_fail_next_lever"])
    r = copy.deepcopy(recs_id)
    for t in r:
        if t.endswith("|1"):
            r[t]["p5r"] = (np.asarray(r[t]["p5r"]) + saw).tolist()
    R4 = run(recs=r, ctl=ctl_id, evc=ev_id, expect=NR, ref=REF_ID)
    arm("rank-1 p5r labels jerky", "TB-6", R4)
    check("rank-1-only TB-6 FAIL is not a STOP and the policy goes rank0", not R4["gates"]["TB-6"]["detail"]["rank0_fail_STOP_next_lever_P5R"] and R4["rank_policy"]["decision"] == "rank0", R4["rank_policy"]["decision"])
    d64 = R4["gates"]["TB-6"]["detail"]
    check("R-D end to end: a rank-1-only TB-6 FAIL gives failing == ['TB-6'] exactly, rank 0 pass / rank 1 fail, decision rank0 (the runner's rank1only condition)",
          R4["failing"] == ["TB-6"] and d64["rank0"]["pass"] is True and d64["rank1"]["pass"] is False and R4["rank_policy"]["decision"] == "rank0", str(R4["failing"]))
    check("R-D end to end: a rank-0 TB-6 FAIL gives failing containing TB-6-rank0-STOP (the trigger token)", "TB-6-rank0-STOP" in R3["failing"])
    r = copy.deepcopy(recs0)
    for t in r:
        if t.endswith("|1"):
            r[t].pop("p5r"); r[t]["p5r_error"] = "no tc20s warm start"
    arm("no rank-1 row has a p5r label", "TB-7", run(recs=r))
    n_diff = R["gates"]["TB-7"]["detail"]["per_rank"]
    check("TB-7 n_differs > 0 on both ranks (real)", n_diff[0]["n_differs"] > 0 and n_diff[1]["n_differs"] > 0, str(n_diff))

    print("== rank policy in p5r mode reads p5r_sim8, never tc20s_sim8")
    pol_old = run(r0=rep0_old, r1=rep1_old)["rank_policy"]
    check("repair files WITHOUT p5r_sim8 => UNDECIDED (a Tc20s sim8 never stands in for P5-R's)", pol_old["decision"] == "UNDECIDED" and "p5r_sim8" in " ".join(pol_old["missing_inputs"]), str(pol_old))
    pol = Rid["rank_policy"]
    check("repair files WITH p5r_sim8 => decided; the estimator is 4.3's (seed 20260927)", pol["decision"] in ("all", "rank0", "STOP") and "seed 20260927" in pol["r0"]["estimator"])
    d = copy.deepcopy(rep0)
    for x in d["rows"]:
        x["raw_sim8"] = False; x["p5r_sim8"] = True; x["tc20s_sim8"] = False
    pr0 = run(r0=d)["rank_policy"]
    check("RK0 on P5-R: p5r worse than raw on rank 0 => STOP, and it ignores a harmless Tc20s column", pr0["decision"] == "STOP" and pr0["rk"]["RK0_stop_rule_r0_hi_lt_0"])

    print("== table integration + CLI (p5r)")
    odir = os.path.join(TMP, "p5r_out")
    os.makedirs(odir, exist_ok=True)
    wr = G.write_merged(bank, R["M"], odir)
    check("p5r merged files written", wr["written"])
    tab = f"{odir}/T_p5r.json.gz"
    pb = subprocess.run([sys.executable, TABMOD, "build", "--bank-dir", FIX + "/bank", "--labels", wr["all"], "--out", tab, "--require-coverage", "1.0", "--label-name", "P5R"], capture_output=True, encoding="utf-8")
    pv = subprocess.run([sys.executable, TABMOD, "verify", "--table", tab, "--bank-dir", FIX + "/bank"], capture_output=True, encoding="utf-8")
    check("traj_label_table build rc 0 (P5R)", pb.returncode == 0, pb.stderr[-300:])
    check("traj_label_table verify rc 0 and covers 80/80 with differs_from_raw == TB-7 total", pv.returncode == 0 and json.loads(pv.stdout)["covered"] == NR
          and json.loads(pv.stdout)["differs_from_raw"] == n_diff[0]["n_differs"] + n_diff[1]["n_differs"], pv.stdout[-300:])
    check("merged p5r source is p5r (not tc20s)", {json.loads(ln)["source"] for ln in open(wr["all"], encoding="utf-8")} == {"p5r"})
    lab = os.path.join(TMP, "p5r_labels.jsonl")
    with open(lab, "w", encoding="utf-8", newline="\n") as f:
        for dd in recs0.values():
            f.write(json.dumps(dd) + "\n")
    ctl_fp = os.path.join(TMP, "p5r_ctl_tc.jsonl")
    with open(ctl_fp, "w", encoding="utf-8", newline="\n") as f:
        for dd in ctl_tc.values():
            f.write(json.dumps(dd) + "\n")
    rp0, rp1 = os.path.join(TMP, "rep0.json"), os.path.join(TMP, "rep1.json")
    json.dump(rep0, open(rp0, "w", newline="\n")); json.dump(rep1, open(rp1, "w", newline="\n"))
    lam_fp = os.path.join(TMP, "lam.json")
    json.dump({"lambda_star": 3.0, "status": "OK"}, open(lam_fp, "w", newline="\n"))
    ver_fp = os.path.join(TMP, "verdict.json")
    json.dump({"per_rank": {"0": {"TB6_p5r": REF[0]}}}, open(ver_fp, "w", newline="\n"))
    cmd = [sys.executable, G.__file__, "--label", "p5r", "--bank", FIX + "/bank", "--labels", lab, "--control", ctl_fp, "--eval-control", FIX + "/eval_control_33.jsonl", "--eval-n", "33",
           "--lambda-star", lam_fp, "--p5r-verdict", ver_fp, "--tb1", TB1, "--repair0", rp0, "--repair1", rp1, "--expect-rows", str(NR), "--out-dir", os.path.join(TMP, "p5r_cli"), "--code-file", G.__file__]
    pc = subprocess.run(cmd, capture_output=True, encoding="utf-8")
    last = pc.stdout.strip().splitlines()[-1] if pc.stdout.strip() else ""
    check("CLI --label p5r on the real fixtures: prints LABEL p5r, TB-0..TB-7 lines, and a ZZTB final line", "LABEL p5r" in pc.stdout and last.startswith(("ZZTB_ALL_PASS", "ZZTB_FAIL")), pc.stdout[-400:] + pc.stderr[-300:])
    check("CLI: the only possible failures are the TB-6 learnability read on 40 rows (informative), never TB-0/1/2/3/4/5/7", all(g in ("TB-6", "TB-6-rank0-STOP", "RK0-STOP") for g in last.replace("ZZTB_FAIL", "").replace("ZZTB_ALL_PASS", "").split()), last)
    gj = json.load(open(os.path.join(TMP, "p5r_cli", "gates.json"), encoding="utf-8"))
    mj = json.load(open(os.path.join(TMP, "p5r_cli", "table_meta_fields.json"), encoding="utf-8"))
    check("meta records label_name P5R and lambda_star 3.0 and the PREREG_P5R sha", mj["label_name"] == "P5R" and mj["p5r"]["lambda_star"] == 3.0 and mj["p5r"]["prereg"].endswith("cfd23e0651d66e497238a0708c717ba00acbcb2189295efcfb15dad657922a42"))
    check("gates.json carries status/n/value/bar per gate", all(all(kk in gj["gates"][g] for kk in ("status", "n", "value", "bar")) for g in G.GATE_IDS))
    check("CLI: --tc20s-tol default 1e-9 is in the TB-0 bar, and meta records R_B + tc20s_tol_R_A", "1e-09" in gj["gates"]["TB-0"]["bar"] and mj["p5r"]["tc20s_tol_R_A"] == 1e-9 and mj["p5r"]["R_B"]["R-TOL-2_p99_max"] == 5e-3)
    cmd_nv = list(cmd); cmd_nv.pop(cmd_nv.index("--p5r-verdict") + 1); cmd_nv.pop(cmd_nv.index("--p5r-verdict"))
    pcv = subprocess.run(cmd_nv, capture_output=True, encoding="utf-8")
    check("CLI without --p5r-verdict: TB-0 and TB-1 not in the ZZTB_FAIL line (verdict.json is context only)", "TB-0" not in pcv.stdout.splitlines()[-1] and "TB-1" not in pcv.stdout.splitlines()[-1], pcv.stdout[-300:])
    cmd0 = list(cmd) + ["--tc20s-tol", "0"]
    pc0 = subprocess.run(cmd0, capture_output=True, encoding="utf-8")
    check("CLI --tc20s-tol 0 reproduces the frozen bar text (== 0.0 / bit identity) for the warm start", pc0.returncode in (0, 1) and "TB-0" in pc0.stdout)
    cmd2 = list(cmd); cmd2.pop(cmd2.index("--eval-control") + 1); cmd2.pop(cmd2.index("--eval-control"))
    pc2 = subprocess.run(cmd2, capture_output=True, encoding="utf-8")
    check("CLI without --eval-control: rc 1 and TB-0 TB-1 in the ZZTB_FAIL line", pc2.returncode == 1 and "TB-0" in pc2.stdout.splitlines()[-1] and "TB-1" in pc2.stdout.splitlines()[-1], pc2.stdout[-300:])
    cmd3 = [c for c in cmd]; cmd3[cmd3.index("--label") + 1] = "tc20s"
    pc3 = subprocess.run(cmd3[:cmd3.index("--eval-control")] + cmd3[cmd3.index("--lambda-star") + 2:], capture_output=True, encoding="utf-8")
    check("CLI --label tc20s on the dual records still runs (Tc20s mode reads only tc20s)", "LABEL" not in pc3.stdout and "TB-0" in pc3.stdout, pc3.stdout[-300:] + pc3.stderr[-300:])
    n, f = N[0] - n0, len(FAILS) - f0
    print(f"ZZTBGATES_P5R_{'FAIL' if f else 'PASS'} {n - f}/{n}")
    return 1 if f else 0


RUNNER = HERE + "/p5c_pod_label.sh"
GITBASH = "C:/Program Files/Git/usr/bin/bash.exe"      # never the bare `bash` (System32 WSL bash on this box)


def thread_tests():
    """MM ruling R-C: threadpoolctl is optional; without it the labeller REQUIRES OPENBLAS/MKL/OMP_NUM_THREADS == 1 and refuses otherwise; the mechanism is recorded."""
    n0, f0 = N[0], len(FAILS)
    print("== R-C: thread pin without threadpoolctl")
    code = ("import sys, json; sys.modules['threadpoolctl'] = None; sys.path.insert(0, %r); import p5c_dual_labels as D\n"
            "out = {'mode': D.BLAS_LIMIT_MODE, 'has': D.threadpool_limits is not None}\n"
            "def t(env):\n"
            "    try: return D.thread_pin(env)['mechanism']\n"
            "    except RuntimeError as e: return 'REFUSE'\n"
            "one = {'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}\n"
            "out['all_one'] = t(one)\n"
            "for v in one:\n"
            "    out['miss_' + v] = t({k: x for k, x in one.items() if k != v})\n"
            "    out['two_' + v] = t(dict(one, **{v: '2'}))\n"
            "out['with_tpc'] = D.thread_pin(dict(one, OMP_NUM_THREADS='4'), True)['mechanism']\n"
            "print(json.dumps(out))\n") % HERE
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, encoding="utf-8", env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
    try:
        o = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        o = {}
    check("with threadpoolctl blocked: import works and the mode is env", o.get("mode") == "env" and o.get("has") is False, p.stderr[-300:])
    check("all three env vars == 1: mechanism env", o.get("all_one") == "env")
    check("each of the three missing is refused", all(o.get("miss_" + v) == "REFUSE" for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")), str(o))
    check("each of the three == '2' is refused", all(o.get("two_" + v) == "REFUSE" for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")), str(o))
    check("with threadpoolctl present the env is not required (mechanism threadpoolctl)", o.get("with_tpc") == "threadpoolctl")
    code2 = ("import sys; sys.modules['threadpoolctl'] = None; sys.path.insert(0, %r); sys.argv = ['x', 'label', '--out', %r]; import p5c_dual_labels as D; sys.exit(D.main())\n") % (HERE, TMP + "/thr_out")
    env2 = {k: v for k, v in os.environ.items() if k not in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}
    env2["CUDA_VISIBLE_DEVICES"] = "-1"
    p2 = subprocess.run([sys.executable, "-c", code2], capture_output=True, encoding="utf-8", env=env2)
    check("main() with no BLAS env and no threadpoolctl: rc 2 and ZZDUAL_REFUSE thread_pin naming the three vars", p2.returncode == 2 and "ZZDUAL_REFUSE thread_pin" in p2.stdout and "OPENBLAS_NUM_THREADS" in p2.stdout, p2.stdout[-300:] + p2.stderr[-300:])
    src = open(HERE + "/p5c_dual_labels.py", encoding="utf-8").read()
    check("the DONE record and the repair record carry thread_pin", src.count('"thread_pin": PIN["mechanism"]') == 2)
    run_src = open(RUNNER, encoding="utf-8").read()
    check("runner exports all three BLAS vars = 1 and asserts it", "export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1" in run_src and "fail thread_env" in run_src)
    check("runner's import probe no longer requires threadpoolctl (try/except ImportError)", "import threadpoolctl; print('ZZPROBE thread_pin=threadpoolctl')" in run_src and "except ImportError" in run_src
          and 'import threadpoolctl" \\' not in run_src)
    n, f = N[0] - n0, len(FAILS) - f0
    print(f"ZZTBGATES_THREADS_{'FAIL' if f else 'PASS'} {n - f}/{n}")
    return 1 if f else 0


def selection_tests():
    """MM ruling R-D: the runner's 6b selection block (extracted from p5c_pod_label.sh and run under Git bash with fake gate lines + fake gates.json)."""
    n0, f0 = N[0], len(FAILS)
    print("== R-D: runner selection block (6b) with fake gate lines")
    import re
    txt = open(RUNNER, encoding="utf-8").read()
    block = txt[txt.index("# ---- 6b."):txt.index("# ---- 7.")]
    disp = [ln for ln in txt.splitlines() if ln.startswith("case \"$TC_CLASS\" in pass|rank1only) build_table") or ln.startswith("case \"$P5R_CLASS\" in pass|rank1only) build_table")]
    check("section 7 builds a table for class pass AND rank1only, for both labels", len(disp) == 2, str(disp))
    py = sys.executable.replace("\\", "/")

    def gj(path, r0, r1, dec):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"gates": {"TB-6": {"detail": {"rank0": {"pass": r0}, "rank1": {"pass": r1}}}}, "rank_policy": {"decision": dec}}, open(path, "w", newline="\n"))

    def go(name, tc, p5, tcj=(True, False, "rank0"), p5j=(True, False, "rank0")):
        g = f"{TMP}/sel_{abs(hash(name)) % 10**8}"
        gj(g + "/tc20s/gates.json", *tcj)
        gj(g + "/p5r/gates.json", *p5j)
        script = (f'set -u\nPY="{py}"\nGATES="{g}"\nfail() {{ echo "ZZP5CL_FAIL $1 ${{2:-}}"; exit 1; }}\nbuild_table() {{ echo "BUILD $1 $2"; }}\n'
                  f'TC_LINE="{tc}"\nP5R_LINE="{p5}"\n{block}\nTABLE_SHA_TC=""\n' + "\n".join(disp) + '\necho "END SEL=$SEL"\n')
        sp = g + "/sel.sh"
        open(sp, "w", encoding="utf-8", newline="\n").write(script)
        pr = subprocess.run([GITBASH, sp], capture_output=True, encoding="utf-8")
        return pr.returncode, pr.stdout + pr.stderr

    PASS = "ZZTB_ALL_PASS"
    rc, o = go("a", PASS, "ZZTB_FAIL TB-0")
    check("Tc20s PASS: label tc20s, build tc20s only", rc == 0 and "ZZP5CL_LABEL tc20s" in o and "BUILD tc20s Tc20s" in o and "BUILD p5r" not in o, o)
    rc, o = go("b", "ZZTB_FAIL TB-6 TB-6-rank0-STOP", PASS, tcj=(False, False, "STOP"))
    check("rank-0 TB-6 FAIL (rank 1 also fails) is the trigger/STOP path: Tc20s class trigger, P5-R PASS => label p5r, p5r table built, no Tc20s table", rc == 0 and "tc20s=trigger" in o
          and "ZZP5CL_LABEL p5r" in o and "BUILD p5r P5R" in o and "BUILD tc20s" not in o, o)
    rc, o = go("c", "ZZTB_FAIL TB-6 TB-6-rank0-STOP", "ZZTB_FAIL TB-6 TB-6-rank0-STOP", tcj=(False, False, "STOP"), p5j=(False, False, "STOP"))
    check("trigger and P5-R rank-0 TB-6 also fails: label p5t, exit 3, no build", rc == 3 and "ZZP5CL_LABEL p5t" in o and "BUILD" not in o, o)
    rc, o = go("d", "ZZTB_FAIL RK0-STOP", PASS, tcj=(True, True, "STOP"))
    check("RK0-STOP alone is the trigger => p5r", rc == 0 and "tc20s=trigger" in o and "ZZP5CL_LABEL p5r" in o, o)
    rc, o = go("e", "ZZTB_FAIL TB-0", PASS)
    check("Tc20s fails TB-0: ZZP5CL_FAIL gates, rc 1", rc == 1 and "ZZP5CL_FAIL gates" in o, o)
    rc, o = go("f", "ZZTB_FAIL TB-6 TB-0", PASS)
    check("Tc20s fails TB-6 AND TB-0: a fault, never rank1only", rc == 1 and "ZZP5CL_FAIL gates" in o and "tc20s=fault" in o, o)
    rc, o = go("g", "ZZTB_FAIL TB-6", PASS)
    check("R-D: Tc20s line 'ZZTB_FAIL TB-6' alone, gates.json rank0 PASS / rank1 FAIL / decision rank0: PASSES with policy rank0 => label tc20s and build_table runs for it",
          rc == 0 and "tc20s=rank1only" in o and "ZZP5CL_LABEL tc20s" in o and "BUILD tc20s Tc20s" in o and "ZZP5CL_FAIL" not in o, o)
    rc, o = go("h", "ZZTB_FAIL TB-6", PASS, tcj=(False, False, "rank0"))
    check("'ZZTB_FAIL TB-6' but gates.json shows rank 0 FAIL: not rank1only (fault)", rc == 1 and "tc20s=fault" in o, o)
    rc, o = go("i", "ZZTB_FAIL TB-6", PASS, tcj=(True, False, "UNDECIDED"))
    check("'ZZTB_FAIL TB-6' but policy UNDECIDED: fault", rc == 1 and "tc20s=fault" in o, o)
    rc, o = go("j", "ZZTB_FAIL TB-6", PASS, tcj=(True, False, "STOP"))
    check("'ZZTB_FAIL TB-6' but policy STOP: fault", rc == 1 and "tc20s=fault" in o, o)
    rc, o = go("k", "ZZTB_FAIL TB-6", PASS, tcj=(True, True, "rank0"))
    check("'ZZTB_FAIL TB-6' but rank 1 actually PASSES in gates.json (inconsistent): fault", rc == 1 and "tc20s=fault" in o, o)
    rc, o = go("l", "ZZTB_FAIL TB-6 TB-6-rank0-STOP", "ZZTB_FAIL TB-6")
    check("R-D, P5-R line: trigger and P5-R 'ZZTB_FAIL TB-6' with rank0 PASS / rank1 FAIL / decision rank0: P5-R passes with policy rank0 => label p5r, build p5r", rc == 0 and "p5r=rank1only" in o
          and "ZZP5CL_LABEL p5r" in o and "BUILD p5r P5R" in o, o)
    rc, o = go("m", "ZZTB_FAIL TB-6 TB-6-rank0-STOP", "ZZTB_FAIL TB-6", p5j=(False, False, "STOP"))
    check("P5-R 'ZZTB_FAIL TB-6' but its json shows rank 0 FAIL: not passing => p5t, exit 3", rc == 3 and "ZZP5CL_LABEL p5t" in o, o)
    rc, o = go("n", PASS, "ZZTB_FAIL TB-6")
    check("both: Tc20s PASS and P5-R rank1only => label tc20s (P5-R is only a contingency), p5r table also built", rc == 0 and "ZZP5CL_LABEL tc20s" in o and "BUILD tc20s Tc20s" in o and "BUILD p5r P5R" in o, o)
    rc, o = go("o", "ZZTB_FAIL TB-6", "ZZTB_FAIL TB-6", tcj=(True, False, "rank0"), p5j=(True, False, "rank0"))
    check("both rank1only: label tc20s, both tables built", rc == 0 and "BUILD tc20s Tc20s" in o and "BUILD p5r P5R" in o, o)
    n, f = N[0] - n0, len(FAILS) - f0
    print(f"ZZTBGATES_SELECTION_{'FAIL' if f else 'PASS'} {n - f}/{n}")
    return 1 if f else 0


def main():
    only = "--p5r-only" in sys.argv
    rc = 0
    if not only and os.path.exists(SUBSET):
        rc |= legacy()
    else:
        print(f"ZZTBGATES_LEGACY_NOT_RUN subset {SUBSET} is absent (deleted under the 2026-11-07 auto-delete rule): the 94 Tc20s checks need the 4,393-row bank subset; "
              "the p5r section below includes source-identity checks of every legacy function instead")
    rc |= p5r_tests()
    rc |= thread_tests()
    rc |= selection_tests()
    print(f"ZZTBGATES_ALL_{'FAIL' if FAILS else 'PASS'} {N[0] - len(FAILS)}/{N[0]} {FAILS}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
