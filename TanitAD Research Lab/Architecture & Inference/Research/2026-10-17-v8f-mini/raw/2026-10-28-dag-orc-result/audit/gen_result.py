"""gen_result.py -- builds RESULT_V8F_DAG_ORC.md, REGISTER_ROW.md and NUMBERS_AUDIT.md for the DAG-ORC read.

Every number in the RESULT is read here from DAGORC_READ.json / RF_REP.json (or parsed from the banked logs and run records), formatted once, and
logged to NUMBERS_AUDIT.md with its source path. Derived numbers carry their arithmetic. The reader's decisions (R-MOVE, R-CRAWL, R-NEW, R-NOBASE,
R-ADE, R-T0, R-KEEP) are recomputed from the reader's own deltas and the script stops if they disagree.
run: python -I -X utf8 gen_result.py     (one process, no GPU, no threads)
"""
import hashlib
import json
import os
import re
import sys

sys.dont_write_bytecode = True

OUT = "E:/Projects/TanitAD-artifacts/dagland/dagorc_result/"
RD = "F:/TanitAD-scratch/dagorc_t1_out/read/"
T1L = "E:/Projects/TanitAD-artifacts/dagorc_t1/logs/"
T1O = "F:/TanitAD-scratch/dagorc_t1_out/"
RUNS = "E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/"
LG = "C:/Users/Admin/box2_ship/logs/"
NOTES = "E:/Projects/TanitAD-artifacts/dagland/dagorc_r2a/"
KB = "E:/Projects/TanitAD-artifacts/dagorc_t1/kbuild_r2_launch_8c24823f.json"
PKG = "raw/2026-10-28-dag-orc-result/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"

rdb = lambda p: open(p, "rb").read()
sha = lambda p: hashlib.sha256(rdb(p)).hexdigest()
R = json.loads(rdb(RD + "DAGORC_READ.json").decode("utf-8"))
RFJ = json.loads(rdb(RD + "RF_REP.json").decode("utf-8"))
assert sha(RD + "DAGORC_READ.json") == "70a7a3a46e520311ade0cf2e1c71203010a25185ef5eb82b9d34b988b9548112"
assert sha(RD + "RF_REP.json") == "64c9444d486939be8dbbf61ef67028c35aa56b13c5d4bdbbc71d6e17e6fc497a"

MINUS = "\u2212"
AUD = []          # (section, rendered, source, note)
SEC = [""]


def aud(rendered, src, note=""):
    AUD.append((SEC[0], rendered, src, note))


def sg(x, nd=4):
    s = f"{abs(x):.{nd}f}"
    return (MINUS if (x < 0 and float(s) != 0.0) else "+") + s


def un(x, nd=4):
    return f"{x:.{nd}f}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f} %"


def kp(*keys):
    out = ""
    for k in keys:
        out += ("." + k if out else k) if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) else f'["{k}"]'
    return out


def get(root, *keys):
    for k in keys:
        root = root[k]
    return root


def tag(e):
    return "B" if e["separated_better"] else ("W" if e["separated_worse"] else "\u2013")


def tg_ff(e, direction):
    """tag for a four_families leaf ({delta, lo, hi, separated}) using the row's direction"""
    if not e["separated"]:
        return "\u2013"
    better = (e["delta"] < 0) if direction == "lower" else (e["delta"] > 0)
    return "B" if better else "W"


def cellstr(e, t):
    s = f"{sg(e['delta'])} [{sg(e['lo'])}, {sg(e['hi'])}] {t}"
    return f"**{s}**" if t == "W" else s


RA = R["read_a"]
FL = R["floors"]
ROWS = list(FL["bar"].keys())
assert len(ROWS) == 19 and list(RFJ["RF"].keys()) == ROWS
SEEDS = ("s0", "s1")
CRAWL = R["rows"]["A"]["s0"]["crawl_set"]
LONG_ROWS = ["speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG"]
assert CRAWL == R["rows"]["A"]["s1"]["crawl_set"] == ["yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"]
HALF_F = FL["F"]["ADE6"] / 2          # 0.06035; the PREREG writes 0.0604
assert f"{HALF_F:.4f}" == "0.0604"
DIRN = {r: RA["A_CTRL"]["s0"][r]["direction"] for r in ROWS}


def rc(contrast, seed, row):
    """a read_a cell, rendered and audited"""
    e = RA[contrast][seed][row]
    t = tag(e)
    s = cellstr(e, t)
    aud(s, f"read_a.{contrast}.{seed}[\"{row}\"].{{delta,lo,hi,separated_better,separated_worse}}")
    return s


def rcm(contrast, seed, row, unit):
    e = RA[contrast][seed][row]
    word = {"B": "separated-better", "W": "separated-worse", "–": "not separated"}[tag(e)]
    out = f"{sg(e['delta'])} [{sg(e['lo'])}, {sg(e['hi'])}] {unit}, {word}"
    aud(out, f"read_a.{contrast}.{seed}[\"{row}\"].{{delta,lo,hi,separated_better,separated_worse}}")
    return out


# ---------------------------------------------------------------------------------------------------------------- consistency: the reader recomputed
assert RA["A_DAG"] == RA["A_CTRL"] and RA["DAG_P10"] == RA["CTRL_P10"]      # Amendment 1: DAG means DAG-OACC and CTRL means DAG-OACC
assert FL["RF"] == RFJ["RF"]
for r in ROWS:
    mx = max(FL["F"][r], FL["RF0"][r], FL["RF"][r])
    assert abs(FL["bar"][r] - 2 * mx) < 1e-9, r
    for s in SEEDS:
        assert abs(max(abs(RFJ["REP_minus_CTRL"][q][r]["delta"]) for q in SEEDS) - FL["RF"][r]) < 1e-9
        assert RFJ["REP_minus_CTRL"][s][r]["delta"] == RA["REP_CTRL"][s][r]["delta"], (s, r)
F_ADE = FL["F"]["ADE6"]
for s in SEEDS:
    X = R["rows"]["A"][s]
    bar = FL["bar"]
    # R-MOVE / R-CRAWL per row of C
    for r in CRAWL:
        a_c, a_p = RA["A_CTRL"][s][r], RA["A_P10"][s][r]
        move = bool(a_c["separated_better"] and (-a_c["delta"]) >= bar[r])
        assert move == X["R-MOVE_rows"][r], (s, r)
        assert (not a_p["separated_worse"]) == X["R-CRAWL_rows"][r], (s, r)
    assert X["flags"]["R-MOVE"] == all(X["R-MOVE_rows"][r] for r in CRAWL)
    assert X["flags"]["R-CRAWL"] == all(X["R-CRAWL_rows"][r] for r in CRAWL)
    # R-NEW and R-NOBASE per row of D_s (DAG = CTRL = DAG-OACC, Amendment 1)
    for r in X["domain"]:
        a_c, a_p, c_p = RA["A_CTRL"][s][r], RA["A_P10"][s][r], RA["CTRL_P10"][s][r]
        c3 = bool(a_c["separated_worse"] and a_c["delta"] * (1 if DIRN[r] == "lower" else -1) >= bar[r])
        assert (a_p["separated_worse"] and (not c_p["separated_worse"]) and c3) == X["R-NEW_fires"][r], (s, r)
        assert (a_p["separated_worse"] and c3) == X["R-NOBASE_violations"][r], (s, r)
    assert X["flags"]["R-NOBASE"] == (not any(X["R-NOBASE_violations"].values()))
    g = X["guards"]
    assert g["R-NEW"] == (not any(X["R-NEW_fires"].values()))
    assert g["R-ADE"] == (not RA["A_CTRL"][s]["ADE6"]["separated_worse"])
    assert g["R-T0"] == (X["T0_point"] <= HALF_F)
    assert X["flags"]["R-KEEP"] == (X["KEEP_point"] <= HALF_F)
    assert abs(X["KEEP_point"] - RA["A_DAG"][s]["ADE6"]["delta"]) < 6e-5
    ff = R["four_families"]["A"][s]
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        assert ff[fam]["tier"] == R["tier"]["T1"] and ff[fam]["estimator"].startswith("paired episode-cluster bootstrap (B=2000, seed 0)")
        for r, leaf in ff[fam]["rows"].items():
            kc = [k for k in leaf if k.startswith("X-CTRL")][0]
            for k, c in ((kc, "A_CTRL"), ("X-P10", "A_P10")):
                e = RA[c][s][r]
                assert (leaf[k]["delta"], leaf[k]["lo"], leaf[k]["hi"], leaf[k]["separated"]) == (e["delta"], e["lo"], e["hi"], e["separated"]), (s, r, k)
                assert tg_ff(leaf[k], DIRN[r]) == tag(e), (s, r, k)
assert R["controls_PASS"] and all(v is True for v in R["controls_PASS"].values())
NCTRL = len(R["controls_PASS"])
assert NCTRL == 162 and R["OUTCOME"]["OUTCOME"] == "HARMS" and R["A"]["OUTCOME"] == "HARMS" and R["B"] is None and R["A"]["adopts"] is False
assert R["strategic_family"].startswith("NOT PAUSED") and R["t0_four_families"]["status"] == "NOT RUN"

# ---------------------------------------------------------------------------------------------------------------- logs and run records
LOGS = {
    "r1": LG + "chain_v8f_dagoaccorc.out",
    "r2a": LG + "chain_v8f_dagoaccorc_r2.out_r2a_marker_stop",
    "r2b": LG + "chain_v8f_dagoaccorc_r2.out_r2b_queue_edit_stop",
    "r2d": LG + "chain_v8f_dagoaccorc_r2.out_r2c_diskfloor_stop",
    "r2c": LG + "chain_v8f_dagoaccorc_r2.out",
}
LL = {k: rdb(v).decode("utf-8").split("\n") for k, v in LOGS.items()}


def stamps(k):
    ts = []
    for i, l in enumerate(LL[k], 1):
        m = re.match(r"^10-28 (\d\d):(\d\d) ", l)
        if m:
            ts.append((i, int(m[1]) * 60 + int(m[2]), f"{m[1]}:{m[2]}"))
    return ts[0], ts[-1]


def jobs(k):
    out, last = [], None
    for i, l in enumerate(LL[k], 1):
        m = re.match(r"^10-28 (\d\d):(\d\d) launched pid=\d+ rc=\d+ log=D:\\tanitad\\logs\\\w+?_(t|e|gate)_(M2b_s\d_dagoaccorc(?:rep)?)\.log", l)
        if m:
            last = (int(m[1]) * 60 + int(m[2]), f"{m[1]}:{m[2]}", m[3], m[4], i)
        m2 = re.match(r"^===== ZZJOB-END (\d\d)\.(\d\d)\.(\d{4})\s+(\d+):(\d\d):(\d\d),(\d\d) rc=(\d+)", l)
        if m2 and last:
            end = int(m2[4]) * 60 + int(m2[5]) + int(m2[6]) / 60
            out.append((last[3], last[4], last[1], f"{int(m2[4]):02d}:{m2[5]}", round(end - last[0]), last[2], i))
            last = None
    return out


SPAN = {}
for k in ("r1", "r2a", "r2b", "r2c"):
    (l0, m0, t0), (l1, m1, t1) = stamps(k)
    SPAN[k] = (t0, t1, m1 - m0, l0, l1)
assert [SPAN[k][:3] for k in ("r1", "r2a", "r2b", "r2c")] == [("01:16", "02:09", 53), ("03:27", "03:39", 12), ("03:48", "06:04", 136), ("06:09", "07:44", 95)]
TOT_MIN = sum(SPAN[k][2] for k in SPAN)
assert TOT_MIN == 296

RUNNAMES = ["M2b_s0_dagoaccorc", "M2b_s0_dagoaccorcrep", "M2b_s1_dagoaccorc", "M2b_s1_dagoaccorcrep"]
ARMOF = {"M2b_s0_dagoaccorc": ("A", "s0"), "M2b_s0_dagoaccorcrep": ("REP", "s0"), "M2b_s1_dagoaccorc": ("A", "s1"), "M2b_s1_dagoaccorcrep": ("REP", "s1")}
RUNREC = {}
for n in RUNNAMES:
    d = RUNS + n + "/"
    sm = json.loads(rdb(d + "summary.json").decode("utf-8"))
    av = json.loads(rdb(d + "argv.json").decode("utf-8"))
    tk = json.loads(rdb(d + "launch_gate_token.json").decode("utf-8"))
    sums = {}
    for l in rdb(d + "SHA256SUMS.txt").decode("utf-8").splitlines():
        h, f = l.split(" *")
        sums[f] = h
    for f, h in sums.items():
        if f != "ckpt_last.pt" and not f.endswith(".npz"):
            assert sha(d + f) == h, (n, f)
    RUNREC[n] = dict(sm=sm, av=av["argv"], tk=tk, sums=sums, dir=d)

# the chain log's own lines for the four final runs
FINAL = LL["r2c"]
pulled = {}
for i, l in enumerate(FINAL, 1):
    m = re.match(r"^10-28 (\d\d:\d\d) (M2b_s\d_dagoaccorc(?:rep)?) pulled to \S+ \(checkpoint sha256 ([0-9a-f]{64})\)", l)
    if m:
        pulled[m[2]] = (i, m[1], m[3])
assert set(pulled) == set(RUNNAMES)
for n in RUNNAMES:
    assert pulled[n][2] == RUNREC[n]["sums"]["ckpt_last.pt"], n
summ_line = [(i, l) for i, l in enumerate(FINAL, 1) if " summary: " in l][0]
SUMM = dict(re.findall(r"(M2b_s\d_dagoaccorc(?:rep)?)=([0-9a-f]{12})", summ_line[1]))
assert all(pulled[n][2].startswith(SUMM[n]) for n in RUNNAMES)
KREACH = {}
for i, l in enumerate(FINAL, 1):
    if "K-REACH:" in l:
        m = re.search(r"K-REACH: (PASS|FAIL) \| (\w+) \| swap (\d+)/(\d+) = ([\d.]+) .*? \| ego (\d+)/(\d+)", l)
        KREACH.setdefault(m[2], []).append((i, m[1], int(m[3]), int(m[4]), float(m[5]), int(m[6]), int(m[7]), l))
RECLINE = {}
for i, l in enumerate(FINAL, 1):
    m = re.match(r"^DAG-ORC \(DAG-OACC base\) run record: (\S+) \| (\w+) s(\d) \| trainer seed (\d+) \| argv sha16 (\w+) \| steps (\d+) \| ego_perturb_p ([\d.]+) \| w_acc ([\d.]+)", l)
    if m:
        RECLINE[(m[2], int(m[3]))] = (i, m)
assert len(RECLINE) == 4 and all(v[1][1] == "PASS" for v in RECLINE.values())
# rollouts: the done markers' checkpoint sha equals the run record's
TAGOF = {"M2b_s0_dagoaccorc": "DAGORC_s0", "M2b_s0_dagoaccorcrep": "DAGORCREP_s0", "M2b_s1_dagoaccorc": "DAGORC_s1", "M2b_s1_dagoaccorcrep": "DAGORCREP_s1"}
for n in RUNNAMES:
    for lim in ("causal", "logged"):
        dn = json.loads(rdb(f"{T1O}done_{TAGOF[n]}_m10_{lim}_heads.txt").decode("utf-8"))
        assert dn["ckpt_sha256"] == dn["ckpt_sha256_before"] == dn["ckpt_sha256_after"] == RUNREC[n]["sums"]["ckpt_last.pt"], (n, lim)
ctrl_ck = {s: [l.split(" *")[0] for l in rdb(f"E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_{s}_dagoacc/SHA256SUMS.txt").decode().splitlines() if l.endswith("*ckpt_last.pt")][0] for s in SEEDS}
assert ctrl_ck["s0"].startswith("67265ec1") and ctrl_ck["s1"].startswith("90b01474")

# r1 K-REACH line
r1_kr = [l for l in LL["r1"] if "K-REACH: FAIL" in l][0]
m = re.search(r"ego (\d+)/(\d+) = ([\d.]+)", r1_kr)
R1 = (int(m[1]), int(m[2]), float(m[3]))
assert R1 == (19161, 76898, 0.2492)
r2a_line = [l for l in LL["r2a"] if "run record carries no levers.ego_perturb_stream" in l and l.startswith("DAG-ORC")][0]
r2a_msg = re.search(r"the run record carries no levers\.ego_perturb_stream[^'\]]*", r2a_line)[0]
r2b_line = [l for l in LL["r2b"] if "init_ckpt not applied" in l and l.startswith("DAG-ORC")][0]
assert "False / None" in r2b_line
r2b_pull = [l for l in LL["r2b"] if "M2b_s1_dagoaccorc pulled to" in l][0]
r2b_a1_sha = re.search(r"checkpoint sha256 ([0-9a-f]{64})", r2b_pull)[1]
assert r2b_a1_sha.startswith("b245c8e7")
r2d_line = [l for l in LL["r2d"] if l.startswith("ZZDISK-FLOOR-STOP")][0]
md = re.search(r"\((\d+) bytes\).*?free=(\d+)", r2d_line)
DISK = (int(md[1]), int(md[2]))
assert DISK == (805306368, 1858076672)
rjobs = {k: jobs(k) for k in ("r1", "r2a", "r2b", "r2c")}

A_SM = {s: RUNREC[f"M2b_{s}_dagoaccorc"]["sm"] for s in SEEDS}
REP_SM = {s: RUNREC[f"M2b_{s}_dagoaccorcrep"]["sm"] for s in SEEDS}
train_sec = sum(RUNREC[n]["sm"]["seconds"] for n in RUNNAMES)

# ---------------------------------------------------------------------------------------------------------------- the document
L = []


def P(*xs):
    L.extend(xs)


def hdr(t):
    SEC[0] = t
    P("", t, "")


tier1, tier0, estim = R["tier"]["T1"], R["tier"]["T0"], R["estimator"]
cnt_move = {s: sum(R["rows"]["A"][s]["R-MOVE_rows"].values()) for s in SEEDS}
cnt_crawl = {s: sum(R["rows"]["A"][s]["R-CRAWL_rows"].values()) for s in SEEDS}
cnt_new = {s: sum(R["rows"]["A"][s]["R-NEW_fires"].values()) for s in SEEDS}
cnt_nobase = {s: sum(R["rows"]["A"][s]["R-NOBASE_violations"].values()) for s in SEEDS}
ndom = {s: len(R["rows"]["A"][s]["domain"]) for s in SEEDS}
assert cnt_move == {"s0": 0, "s1": 0} and cnt_crawl == {"s0": 1, "s1": 0} and cnt_new == {"s0": 0, "s1": 0} and cnt_nobase == {"s0": 0, "s1": 0} and ndom == {"s0": 15, "s1": 15}
SEC[0] = "header"
sha_read, sha_rf = sha(RD + "DAGORC_READ.json"), sha(RD + "RF_REP.json")
rdr_sha = R["provenance"]["reader_sha256"]
assert rdr_sha.startswith("2e5f9f17")
P("# RESULT_V8F_DAG_ORC: O-RECOVER's launch-recovery perturbation composed into DAG-OACC's recipe. Outcome **HARMS** (arm A, p = 0.3, not adopted)")
P("")
P("Claim row `H-V8F-DAG-ORC-1`. Pre-registration `Project Steering/PREREG_V8F_DAG_ORC.md` with **Amendment 1** (`A1-2026-10-27`, re-based on DAG-OACC,"
  " landed `6cd741b6`, before any DAG-ORC run) and **Amendment 2** (`A2-2026-10-28`, the composed perturbation's own random stream, landed `8c24823f`,"
  " before any DAG-ORC row). Read 2026-10-28 07:58 by the pre-registered reader"
  f" (`raw/2026-10-28-dag-orc-build/rebase/read/dag_orc_read.py`, sha256 `{rdr_sha[:8]}…`; started 07:54:18, rc 0 at 07:58:54, {R['provenance']['wall_s']} s),"
  f" all **{NCTRL} reader controls PASS**. Output `{PKG}read/DAGORC_READ.json` (sha256 `{sha_read[:8]}…`) and `{PKG}read/RF_REP.json` (sha256"
  f" `{sha_rf[:8]}…`, written before any A contrast was computed).")
aud(str(NCTRL), "len(controls_PASS), all True")
aud("2026-10-27 / 2026-10-28", "amendment ids A1-2026-10-27 and A2-2026-10-28 (PREREG headings); read date 2026-10-28 07:58 = DAGORC_READ.json write time, chain.out 'reader rc=0' stamp 07:58:54")
aud("2026-10-17", "four_families.A.s0.STRATEGIC.ruling (quoted)")
aud(str(R["provenance"]["wall_s"]), "provenance.wall_s")
aud("07:54:18 / 07:58:54", "E:/Projects/TanitAD-artifacts/dagorc_t1/logs/chain.out, lines '== reader:' and 'reader rc=0'")
P("")
P(f"* **Tier.** {tier1} R-T0 is at T0: {tier0}")
aud("tier.T1 / tier.T0", "tier.T1, tier.T0 (quoted)")
P("* **Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed, paired, on the same windows and matched cells. It answers *\"would another draw of EPISODES say this?\"*."
  " The control is a different training run (DAG-OACC), so the training-run question is carried by the two seeds and the replicate floors RF0 / RF (`H-ESTIM-SEED-1`). The planner is deterministic, so no inference-seed question arises.")
assert "B=2000, seed 0, 141 episodes / 2,399 windows" in estim and "the planner is deterministic" in estim
aud("B = 2,000; seed 0; 141 episodes / 2,399 windows", "estimator (string; numbers re-typed with thousands separators)")
ck = {n: RUNREC[n]["sums"]["ckpt_last.pt"] for n in RUNNAMES}
P("* **Arms (box 2, r2c chain pinned at `8c24823f`; four gates PASS, four run records PASS, four K-REACH PASS).** A_s = DAG-OACC_s's training argv"
  " verbatim plus the six perturbation flags at the registered law (`--ego-perturb-p 0.3`) and `--ego-perturb-with-dagger`. REP_s = DAG-OACC_s's"
  " argv with `--seed 10` / `--seed 11` (same init M2b_s). CTRL = DAG-OACC_s, banked (no control was retrained). Checkpoints, sha256 prefixes:"
  f" A `{ck['M2b_s0_dagoaccorc'][:8]}…` (s0) / `{ck['M2b_s1_dagoaccorc'][:8]}…` (s1); REP `{ck['M2b_s0_dagoaccorcrep'][:8]}…` / `{ck['M2b_s1_dagoaccorcrep'][:8]}…`;"
  f" CTRL `{ctrl_ck['s0'][:8]}…` / `{ctrl_ck['s1'][:8]}…`. **Arm B (p = 0.1) was not trained and is not read.**")
for n in RUNNAMES:
    aud(ck[n][:8], f"{n}/SHA256SUMS.txt ckpt_last.pt")
aud(ctrl_ck["s0"][:8] + " / " + ctrl_ck["s1"][:8], "E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s{0,1}_dagoacc/SHA256SUMS.txt ckpt_last.pt")
aud("p = 0.3 (A) / 0.1 (B); trainer seeds 10 / 11", "PREREG_V8F_DAG_ORC.md Amendment 1 section 1; M2b_s0_dagoaccorc/argv.json --ego-perturb-p 0.3; M2b_s{0,1}_dagoaccorcrep/argv.json --seed 10 / 11")
aud("6cd741b6; 8c24823f; 4c412181; d8b8885b", "git log --format=%h -- 'Project Steering/PREREG_V8F_DAG_ORC.md' at refs/heads/agent/arch-inf-20260803 (mirror C:/Users/Admin/tanitad-push); sha256 of the file at 4c412181")
P("* **Coverage.** The four families at T1-kin are in section 3.1. **The T0 four families are NOT RUN** (section 3.2), and three registered beside rows are"
  " NOT BUILT (section 3.3). They are work items, and the outcome above does not use them.")

# ---------------------------------------------------------------------------------------------------------------- 1
hdr("## 1. The outcome, as the table decides it")
g0, g1 = R["rows"]["A"]["s0"], R["rows"]["A"]["s1"]
ade0, ade1 = RA["A_CTRL"]["s0"]["ADE6"], RA["A_CTRL"]["s1"]["ADE6"]
SEC[0] = "1"
t0p0, t0p1 = g0["T0_point"], g1["T0_point"]
aud(sg(t0p0), "rows.A.s0.T0_point")
aud(sg(t0p1), "rows.A.s1.T0_point")
aud("+0.0604", "PREREG_V8F_DAG_ORC.md section 4 (R-T0 bar = F/2); F = floors.F.ADE6", f"F/2 = {HALF_F}")
aud(sg(g0["KEEP_point"]), "rows.A.s0.KEEP_point")
aud(sg(g1["KEEP_point"]), "rows.A.s1.KEEP_point")
nv = lambda b: "held" if b else "**FAILS**"
def yes(b): return "true" if b else "false"
P("| condition (PREREG section 7, as amended) | s0 | s1 |", "|---|---|---|")
P(f"| controls (reader, {NCTRL}) | PASS | PASS |")
for gname in ("R-TLAT", "R-TLON"):
    P(f"| guard {gname} (T head macro-F1, ticks ≥ 1, A − CTRL not separated-worse) | {nv(g0['guards'][gname])} | {nv(g1['guards'][gname])} |")
    aud(f"{gname} held/held", f"A.flags.guards_held_per_seed.s0/s1[\"{gname}\"]")
P(f"| guard R-ADE (T1-kin ADE6, A − CTRL, not separated-worse) | **FAILS**: {rcm('A_CTRL','s0','ADE6','m')} | held: {rcm('A_CTRL','s1','ADE6','m')} |")
assert not g0["guards"]["R-ADE"] and g1["guards"]["R-ADE"]
P(f"| guard R-T0 (T0 own-code ADE6, A − CTRL, point ≤ +0.0604 m) | **FAILS**: {sg(t0p0)} m | **FAILS**: {sg(t0p1)} m |")
assert not g0["guards"]["R-T0"] and not g1["guards"]["R-T0"]
P(f"| guard R-NEW (no row of D_s fires) | held: {cnt_new['s0']} of {ndom['s0']} rows | held: {cnt_new['s1']} of {ndom['s1']} rows |")
aud(f"{cnt_new['s0']} of {ndom['s0']}; {cnt_new['s1']} of {ndom['s1']}", "count of true in rows.A.sX.R-NEW_fires; len(rows.A.sX.domain)")
P(f"| decisive R-MOVE (every row of C clears its bar) | {yes(R['A']['flags']['R-MOVE']['s0'])}: {cnt_move['s0']} of 4 rows clear | {yes(R['A']['flags']['R-MOVE']['s1'])}: {cnt_move['s1']} of 4 rows clear |")
aud("0 of 4; 0 of 4", "count of true in rows.A.sX.R-MOVE_rows")
P(f"| decisive R-CRAWL (every row of C not separated-worse vs P10) | {yes(R['A']['flags']['R-CRAWL']['s0'])}: {cnt_crawl['s0']} of 4 rows (yaw-rate only) | {yes(R['A']['flags']['R-CRAWL']['s1'])}: {cnt_crawl['s1']} of 4 rows |")
aud("1 of 4; 0 of 4", "count of true in rows.A.sX.R-CRAWL_rows")
assert [r for r in CRAWL if g0["R-CRAWL_rows"][r]] == ["yaw-rate MAE"]
P(f"| adoption R-KEEP (T1-kin ADE6, A − DAG-OACC, point ≤ +0.0604 m) | {yes(R['A']['flags']['R-KEEP']['s0'])}: {sg(g0['KEEP_point'])} m | {yes(R['A']['flags']['R-KEEP']['s1'])}: {sg(g1['KEEP_point'])} m |")
P(f"| adoption R-NOBASE | {yes(R['A']['flags']['R-NOBASE']['s0'])}: {cnt_nobase['s0']} of {ndom['s0']} rows violate | {yes(R['A']['flags']['R-NOBASE']['s1'])}: {cnt_nobase['s1']} of {ndom['s1']} rows violate |")
aud("0 of 15; 0 of 15", "count of true in rows.A.sX.R-NOBASE_violations")
P("")
P("R-ADE and R-KEEP read one contrast, A − DAG-OACC (Amendment 1 section 2: DAG = CTRL = DAG-OACC), so the two rows of ADE6 above are the same cells.")
P("")
P(f"**{R['A']['OUTCOME']}.** The table's condition, verbatim: *\"{R['A']['condition']}\"*. Three guard cells fail: R-ADE at s0, and R-T0 at s0 and at s1."
  " REFUSED is decided first and HARMS second (PREREG section 7, *Precedence*), so the R-MOVE flags do not change the word."
  f" Committed next, verbatim from the JSON: *\"{R['A']['committed_next']}\"*")
aud("OUTCOME / condition / committed_next", "A.OUTCOME, A.condition, A.committed_next (quoted)")
P("")
P("⚠ **Reported as written, beside the word.** R-MOVE is false at both seeds, with 0 of 4 crawl rows clearing the bar at either seed (section 2.4). R-CRAWL is"
  " false at both seeds. The guard failures are the registered conditions; the table does not test whether the cost is the lever's (section 4 (c)).")
P("")
P("The word is the reader's output (`matching_rows_in_table_order`: one row, `HARMS`). The script that built this document recomputed R-MOVE, R-CRAWL, R-NEW, R-NOBASE, R-ADE, R-T0"
  " and R-KEEP per seed from the reader's own deltas and floors, and all of them agree with the reader's flags (`audit/gen_result.py`).")

# ---------------------------------------------------------------------------------------------------------------- 2
hdr("## 2. The rows (T1-kin; point [95 % CI]; B / W = separated-better / separated-worse)")
SEC[0] = "2.1"
P("The contrasts are the reader's own `read_a`. **A − DAG is identical to A − CTRL, cell for cell**, because Amendment 1 sets DAG = DAG-OACC = CTRL"
  " (`read_a.A_DAG` equals `read_a.A_CTRL` and `read_a.DAG_P10` equals `read_a.CTRL_P10` in the JSON). P10 is the kinematic floor. Row units are the reader's own."
  " The last column of 2.1 is the `n` field (windows / episodes) of the A − CTRL cell at s0 and s1. For the three S-POS rows the P10 contrast is the run's own S − CTRV, and the JSON labels its `n` *\"ticks>=1, unpaired vs CTRV (DAG_READ.json)\"*: those cells are not paired across runs.")
assert RA["A_P10"]["s0"]["S-POS@30"]["n"] == "ticks>=1, unpaired vs CTRV (DAG_READ.json)"
aud("S-POS P10-contrast n label", "read_a.A_P10.s0[\"S-POS@30\"].n (quoted; same on every S-POS P10 cell)")
P("")
P("### 2.1 The floors and the bar (`floors`)")
P("")
P("`bar_r = 2 × max(F_r, RF0_r, RF_r)`, one value per row, applied at both seeds, inclusive. F is 0.1207 m on ADE6 and 0 elsewhere. RF0 is from the banked DAG-0 / ORC-0 pair (frozen"
  f" `P0_RF0.json`, sha256 `{R['provenance']['rf0_sha256'][:8]}…`). RF is max over s of |REP − CTRL| (`RF_REP.json`, written before any A contrast).")
P("")
P("| row | better | F | RF0 | RF | bar | the max term | RF set by | n, A − CTRL (s0 ; s1) |", "|---|---|---|---|---|---|---|---|---|")
setby, TERM = {}, {}
for r in ROWS:
    f_, r0, rr, b = FL["F"][r], FL["RF0"][r], FL["RF"][r], FL["bar"][r]
    mx = max(f_, r0, rr)
    term = "/".join(nm for nm, v in (("F", f_), ("RF0", r0), ("RF", rr)) if v == mx and v > 0)
    TERM[r] = term
    d0, d1 = abs(RA["REP_CTRL"]["s0"][r]["delta"]), abs(RA["REP_CTRL"]["s1"][r]["delta"])
    setby[r] = "s0" if d0 >= d1 else "s1"
    n0, n1 = RA["A_CTRL"]["s0"][r]["n"], RA["A_CTRL"]["s1"][r]["n"]
    nn = n0.replace("/", " / ") if n0 == n1 else f"{n0.replace('/', ' / ')} ; {n1.replace('/', ' / ')}"
    P(f"| {r} | {DIRN[r]} | {un(f_)} | {un(r0)} | {un(rr)} | {un(b)} | {term} | REP {setby[r]} | {nn} |")
    aud(nn, f"read_a.A_CTRL.s0/s1[\"{r}\"].n")
    aud(f"{r}: F={un(f_)} RF0={un(r0)} RF={un(rr)} bar={un(b)} term={term} setby={setby[r]}", f"floors.F/RF0/RF/bar[\"{r}\"]; setby = argmax_s |read_a.REP_CTRL.s[\"{r}\"].delta|")
n_rf_row = sum(1 for r in ROWS if TERM[r] == "RF")
n_rf0_row = sum(1 for r in ROWS if TERM[r] == "RF0")
assert n_rf_row + n_rf0_row == 19
n_s0 = sum(1 for r in ROWS if setby[r] == "s0")
assert [r for r in ROWS if setby[r] == "s1"] == ["cross-track MAE", "S-POS@80"]
P("")
P(f"The REP pair sets the bar on {n_rf_row} of 19 rows (every row of C among them); RF0 sets it on {n_rf0_row} rows. REP s0 sets RF on {n_s0} of 19 rows; REP s1 sets it on cross-track MAE and S-POS@80.")
aud(f"{n_rf_row}; {n_rf0_row}; {n_s0}", "count over floors: bar == 2*RF; bar == 2*RF0 with RF0 > RF; argmax_s |REP-CTRL| == s0")
assert all(FL["bar"][r] == 2 * FL["RF"][r] for r in CRAWL)

SEC[0] = "2.2"
P("")
P("### 2.2 A − CTRL (the one-lever contrast; = A − DAG-OACC) and REP − CTRL (the replicate)")
P("")
P("| row | A − CTRL s0 | A − CTRL s1 | REP − CTRL s0 | REP − CTRL s1 |", "|---|---|---|---|---|")
for r in ROWS:
    P(f"| {r} | {rc('A_CTRL','s0',r)} | {rc('A_CTRL','s1',r)} | {rc('REP_CTRL','s0',r)} | {rc('REP_CTRL','s1',r)} |")
nrep = {s: sum(1 for r in ROWS if RA["REP_CTRL"][s][r]["separated"]) for s in SEEDS}
assert nrep == {"s0": 15, "s1": 2}
aud(f"{nrep['s0']} of 19; {nrep['s1']} of 19", "count of read_a.REP_CTRL.sX[row].separated")
P("")
P(f"REP differs from CTRL in the trainer seed alone. Even so, REP − CTRL reads separated on {nrep['s0']} of 19 rows at s0 and {nrep['s1']} of 19 at s1.")

SEC[0] = "2.3"
P("")
P("### 2.3 A − P10 and CTRL − P10 (DAG-OACC − P10, the banked control)")
P("")
P("| row | A − P10 s0 | A − P10 s1 | CTRL − P10 s0 | CTRL − P10 s1 |", "|---|---|---|---|---|")
for r in ROWS:
    P(f"| {r} | {rc('A_P10','s0',r)} | {rc('A_P10','s1',r)} | {rc('CTRL_P10','s0',r)} | {rc('CTRL_P10','s1',r)} |")

SEC[0] = "2.4"
P("")
P("### 2.4 The decisive rows: the crawl set C (yaw-rate, curvature, L2, M1 at both seeds)")
P("")
P("R-MOVE: A − CTRL separated-better AND the improvement (−delta) ≥ bar. R-CRAWL: A − P10 not separated-worse.")
P("")
P("| seed | row | A − CTRL | improvement | bar | improvement / bar | clears (R-MOVE row) | A − P10 | not separated-worse (R-CRAWL row) |", "|---|---|---|---|---|---|---|---|---|")
imp = {}
for s in SEEDS:
    X = R["rows"]["A"][s]
    for r in CRAWL:
        e = RA["A_CTRL"][s][r]
        imp[(s, r)] = -e["delta"]
        ratio = -e["delta"] / FL["bar"][r]
        P(f"| {s} | {r} | {rc('A_CTRL', s, r)} | {un(-e['delta'])} | {un(FL['bar'][r])} | {pct(ratio)} | {yes(X['R-MOVE_rows'][r])} | {rc('A_P10', s, r)} | {yes(X['R-CRAWL_rows'][r])} |")
        aud(f"{un(-e['delta'])}; {pct(ratio)}", f"-read_a.A_CTRL.{s}[\"{r}\"].delta ; /floors.bar[\"{r}\"]", f"{-e['delta']}/{FL['bar'][r]}")
ratios = {k: v / FL["bar"][k[1]] for k, v in imp.items()}
rmin, rmax = min(ratios.values()), max(ratios.values())
P("")
P(f"Every one of the eight cells is separated-better against CTRL, and every one is below its bar: the largest is {pct(rmax)} of the bar (M1 at s0) and the smallest {pct(rmin)} (yaw-rate at s0).")
assert max(ratios, key=ratios.get) == ("s0", "M1 crawl share") and min(ratios, key=ratios.get) == ("s0", "yaw-rate MAE")
aud(f"{pct(rmax)}; {pct(rmin)}", "max / min over the 8 cells of improvement / floors.bar")

SEC[0] = "2.5"
P("")
P("### 2.5 The guard rows R-NEW and R-NOBASE (domain D_s: 15 rows at each seed, no exempt rows)")
P("")
P("R-NEW fires on a row when (1) A − P10 is separated-worse, (2) DAG-OACC − P10 is not, and (3) A − CTRL is separated-worse with the worsening ≥ bar. R-NOBASE: (1) and (3).")
P("")
P("| seed | rows with A − CTRL separated-worse | of those, worsening ≥ bar (3) | of those, A − P10 separated-worse (1) | fires |", "|---|---|---|---|---|")
near = []
for s in SEEDS:
    X = R["rows"]["A"][s]
    sw = [r for r in X["domain"] if RA["A_CTRL"][s][r]["separated_worse"]]
    c3 = [r for r in sw if abs(RA["A_CTRL"][s][r]["delta"]) >= FL["bar"][r]]
    c1 = [r for r in c3 if RA["A_P10"][s][r]["separated_worse"]]
    near += [(s, r) for r in c3]
    P(f"| {s} | {', '.join(sw)} | {', '.join(c3) or 'none'} | {', '.join(c1) or 'none'} | none |")
    aud(f"{s}: sw={sw} c3={c3} c1={c1}", f"read_a.A_CTRL.{s}[row].separated_worse / delta vs floors.bar; read_a.A_P10.{s}[row].separated_worse")
assert near == [("s0", "DK-VIOL")]
dk = RA["A_CTRL"]["s0"]["DK-VIOL"]
dkp = RA["A_P10"]["s0"]["DK-VIOL"]
P("")
P(f"The closest approach is DK-VIOL at s0: A − CTRL {sg(dk['delta'])} is separated-worse and above its bar ({un(FL['bar']['DK-VIOL'])}). R-NEW does not fire there"
  f" because A − P10 on DK-VIOL is not separated-worse ({rc('A_P10','s0','DK-VIOL')}).")
aud(f"{sg(dk['delta'])}; {un(FL['bar']['DK-VIOL'])}", "read_a.A_CTRL.s0[\"DK-VIOL\"].delta; floors.bar[\"DK-VIOL\"]")

# ---------------------------------------------------------------------------------------------------------------- 3
hdr("## 3. The four families")
SEC[0] = "3.1"
FFA = R["four_families"]["A"]
P("### 3.1 T1-kin (the primary tier), per seed, per family, never pooled")
P("")
P(f"Tier: {tier1} Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0, per seed. Source `four_families.A.<seed>.<family>.rows` (each family block carries this same tier and estimator string, and its leaves equal `read_a` cell for cell; the build script asserts both).")
P("")
P("| family | row | A − CTRL s0 | A − CTRL s1 | A − P10 s0 | A − P10 s1 |", "|---|---|---|---|---|---|")
for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
    rows_ = list(FFA["s0"][fam]["rows"].keys())
    assert rows_ == list(FFA["s1"][fam]["rows"].keys())
    for i, r in enumerate(rows_):
        cells = []
        for k_, s in (("X-CTRL", "s0"), ("X-CTRL", "s1"), ("X-P10", "s0"), ("X-P10", "s1")):
            leaf = FFA[s][fam]["rows"][r]
            kk = [k for k in leaf if k.startswith(k_)][0]
            t = tg_ff(leaf[kk], DIRN[r])
            cells.append(cellstr(leaf[kk], t))
            aud(cells[-1], f"four_families.A.{s}.{fam}.rows[\"{r}\"][\"{kk}\"].{{delta,lo,hi,separated}}")
        P(f"| {fam if i == 0 else ''} | {r} | " + " | ".join(cells) + " |")
    if fam == "TACTICAL":
        P(f"| | R-TLAT / R-TLON (T-head macro-F1, A − CTRL) | held | held | n/a (P10 has no T head) | n/a (P10 has no T head) |")
        aud("R-TLAT / R-TLON held", "A.flags.guards_held_per_seed.sX; the macro-F1 values themselves are not stored in DAGORC_READ.json (UNVERIFIED)")
P("")
st = FFA["s0"]["STRATEGIC"]
assert st["paused"] is False
P(f"STRATEGIC is reported, not paused, for v8F (`four_families.A.s0.STRATEGIC.ruling`: *\"{st['ruling']}\"*). Its rows are S-POS@30 / @50 / @80 (lower is better); the A − CTRL leaf is the difference of the two runs' (S − CTRV, ticks ≥ 1) contrasts on the same windows"
  f" (`read_a.A_CTRL.s0[\"S-POS@30\"].kind`: *\"{RA['A_CTRL']['s0']['S-POS@30']['kind']}\"*). The T-head macro-F1 values behind R-TLAT / R-TLON are not carried in the JSON, only the booleans (UNVERIFIED as numbers).")
aud("STRATEGIC ruling, kind", "four_families.A.s0.STRATEGIC.ruling; read_a.A_CTRL.s0[\"S-POS@30\"].kind (quoted)")

SEC[0] = "3.2"
P("")
P("### 3.2 T0 (box 2's standard eval): the four families are NOT RUN")
P("")
P(f"`t0_four_families.status` reads **{R['t0_four_families']['status']}**. The PREREG (section 5) registered the T0 families as a build item of the reader (*\"DAGGER2 left the T0 families NOT RUN. Here they are a build item of the reader\"*);"
  f" the reader did not carry it. Work item, verbatim from the JSON: *\"{R['t0_four_families']['work_item']}\"*. The only T0 numbers in this read are the R-T0 guard points,"
  f" T0 own-code ADE6 A − CTRL {sg(t0p0)} m (s0) and {sg(t0p1)} m (s1), with no interval in the JSON. Inputs exist: the pulled T0 evals (`eval_M2b-dagoaccorc_s*.json` / `.npz`) of A and of CTRL, on E:. **Nothing is claimed for T0 LONGITUDINAL, LATERAL or TACTICAL here.**")
aud(f"{sg(t0p0)}; {sg(t0p1)}", "rows.A.s0.T0_point; rows.A.s1.T0_point")
aud("NOT RUN + work_item", "t0_four_families.status / work_item (quoted)")

SEC[0] = "3.3"
P("")
P("### 3.3 Registered beside rows that are NOT BUILT")
P("")
bn = R["beside_not_built"]
for k, v in bn.items():
    P(f"* `{k}`: {v}")
aud("beside_not_built (3 entries)", "beside_not_built.* (quoted)")
P("")
P("Not computable by design (PREREG section 5): T1-kin goal-head F1 and head accuracy (the paired tool reads eval `run` dumps, not tick dumps); anchor selection (one path per window).")

# ---------------------------------------------------------------------------------------------------------------- 4
hdr("## 4. Beside, deciding nothing")
SEC[0] = "4a"
P("### (a) A − CTRL is separated-better on every crawl row and on heading, at both seeds. Re-verified from the JSON:")
P("")
P("| row | s0 | s1 |", "|---|---|---|")
all_b = True
for r in CRAWL + ["heading MAE"]:
    a0, a1 = RA["A_CTRL"]["s0"][r], RA["A_CTRL"]["s1"][r]
    all_b &= bool(a0["separated_better"] and a1["separated_better"])
    P(f"| {r} | {rc('A_CTRL','s0',r)} | {rc('A_CTRL','s1',r)} |")
assert all_b and all(RA["A_CTRL"][s_][r_]["hi"] < 0 for s_ in SEEDS for r_ in CRAWL + ["heading MAE"])
P("")
P("All 10 cells are separated-better (CI upper bound below 0). The sizes against the bar are in section 2.4: the effect is real at the episode level and small against the replicate floor (c).")

SEC[0] = "4b"
P("")
P("### (b) The share of the CTRL − P10 crawl gap that A closes: (CTRL−P10 − (A−P10)) / (CTRL−P10), from the reader's deltas")
P("")
P("| seed | row | CTRL − P10 | A − P10 | share closed | matched-cell form −(A−CTRL) / (CTRL−P10) |", "|---|---|---|---|---|---|")
share = {}
for s in SEEDS:
    for r in CRAWL:
        cp, ap, ac = RA["CTRL_P10"][s][r]["delta"], RA["A_P10"][s][r]["delta"], RA["A_CTRL"][s][r]["delta"]
        sh, sh2 = (cp - ap) / cp, -ac / cp
        share[(s, r)] = sh
        P(f"| {s} | {r} | {sg(cp)} | {sg(ap)} | {pct(sh)} | {pct(sh2)} |")
        aud(f"{pct(sh)}; {pct(sh2)}", f"read_a.CTRL_P10.{s}[\"{r}\"].delta, read_a.A_P10.{s}[\"{r}\"].delta, read_a.A_CTRL.{s}[\"{r}\"].delta", f"({cp} - {ap}) / {cp} = {sh}; -({ac}) / {cp} = {sh2}")
smin, smax = min(share.values()), max(share.values())
P("")
ndiff = [(s_, r_) for s_ in SEEDS for r_ in CRAWL if RA["A_P10"][s_][r_]["n"] != RA["CTRL_P10"][s_][r_]["n"]]
mdiff = max(abs(share[(s_, r_)] + RA["A_CTRL"][s_][r_]["delta"] / RA["CTRL_P10"][s_][r_]["delta"]) for s_ in SEEDS for r_ in CRAWL) * 100
nsw = sum(1 for s_ in SEEDS for r_ in CRAWL if RA["A_P10"][s_][r_]["separated_worse"])
assert (len(ndiff), nsw) == (6, 7) and not RA["A_P10"]["s0"]["yaw-rate MAE"]["separated_worse"]
P("")
P(f"A closes {pct(smin)} to {pct(smax)} of the gap that DAG-OACC leaves to P10 on the crawl rows (largest: yaw-rate at s0; smallest: M1 at s1). The two forms differ by at most {mdiff:.1f} points,"
  f" because the CTRL − P10 and A − P10 cells differ by one window on {len(ndiff)} of the 8 rows (the `n` fields). At s0 the yaw-rate row ends not separated-worse against P10; the other {nsw} rows stay separated-worse.")
aud(f"{len(ndiff)}; {mdiff:.1f}; {nsw}", "count of rows with read_a.A_P10.sX[row].n != read_a.CTRL_P10.sX[row].n; max |share - alt form| x 100; count of separated_worse among the 8 A_P10 cells")
assert min(share, key=share.get) == ("s1", "M1 crawl share") and max(share, key=share.get) == ("s0", "yaw-rate MAE")
aud(f"{pct(smin)}; {pct(smax)}", "min / max of the 8 shares")

SEC[0] = "4c"
P("")
P("### (c) The replicate floor is set by REP s0; REP differs from CTRL only in the trainer seed")
P("")
P("K-PAIR (reader control) PASS at both seeds: argv(REP_s) equals argv(DAG-OACC_s) except `--out` and `--seed` (10 vs 0, 11 vs 1). So REP − CTRL is a measured draw of **training variance** on this rig"
  " (`H-ESTIM-SEED-1`), the one thing the episode-cluster bootstrap does not see.")
P("")
P("| row | REP − CTRL s0 | REP − CTRL s1 | RF (max over s of the absolute REP − CTRL) | set by | RF0 | bar (2 × max) |", "|---|---|---|---|---|---|---|")
for r in ["ADE6", "curvature MAE", "yaw-rate MAE", "M1 crawl share"]:
    P(f"| {r} | {rc('REP_CTRL','s0',r)} | {rc('REP_CTRL','s1',r)} | {un(FL['RF'][r])} | REP {setby[r]} | {un(FL['RF0'][r])} | {un(FL['bar'][r])} |")
    aud(f"{un(FL['RF'][r])}; {un(FL['RF0'][r])}; {un(FL['bar'][r])}", f"floors.RF/RF0/bar[\"{r}\"]")
P("")
P("The A − CTRL effects, set beside the floor they had to clear (size = |A − CTRL|; the bar is 2 × RF on all four crawl rows):")
P("")
P("| row | A − CTRL s0 | as share of RF | of bar | A − CTRL s1 | as share of RF | of bar |", "|---|---|---|---|---|---|---|")
for r in ["yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share", "ADE6"]:
    cs = []
    for s in SEEDS:
        e = RA["A_CTRL"][s][r]
        cs += [sg(e["delta"]), pct(abs(e["delta"]) / FL["RF"][r]), pct(abs(e["delta"]) / FL["bar"][r])]
        aud(f"{cs[-3]}; {cs[-2]}; {cs[-1]}", f"read_a.A_CTRL.{s}[\"{r}\"].delta / floors.RF[\"{r}\"] / floors.bar[\"{r}\"]", f"{abs(e['delta'])}/{FL['RF'][r]}; {abs(e['delta'])}/{FL['bar'][r]}")
    P(f"| {r} | " + " | ".join(cs) + " |")
repade = RA["REP_CTRL"]["s0"]["ADE6"]["delta"]
adec = RA["A_CTRL"]["s0"]["ADE6"]["delta"]
P("")
P(f"Two readings, arithmetic only. (i) On the four crawl rows A − CTRL is {pct(min(abs(RA['A_CTRL'][s][r]['delta'])/FL['RF'][r] for s in SEEDS for r in CRAWL))} to {pct(max(abs(RA['A_CTRL'][s][r]['delta'])/FL['RF'][r] for s in SEEDS for r in CRAWL))} of RF, so each is smaller"
  " than the REP s0 draw on the same row. (ii) REP s0 − CTRL on ADE6 is "
  f"{sg(repade)} m, {un(repade / F_ADE, 2)} × F (0.1207 m); the s0 cost A − CTRL on ADE6 that failed R-ADE, {sg(adec)} m, is {pct(adec / repade, 0)} of that replicate draw. REP s1 − CTRL on ADE6 is"
  f" {sg(RA['REP_CTRL']['s1']['ADE6']['delta'])} m. The bar is one value per row from the larger draw by registration (PREREG section 3, *Two choices*), so a small s1 draw does not lower it.")
aud(f"{un(repade / F_ADE, 2)}; {pct(adec / repade, 0)}", "read_a.REP_CTRL.s0.ADE6.delta / floors.F.ADE6; read_a.A_CTRL.s0.ADE6.delta / read_a.REP_CTRL.s0.ADE6.delta", f"{repade}/{F_ADE}; {adec}/{repade}")
aud(f"{pct(min(abs(RA['A_CTRL'][s][r]['delta'])/FL['RF'][r] for s in SEEDS for r in CRAWL))}; {pct(max(abs(RA['A_CTRL'][s][r]['delta'])/FL['RF'][r] for s in SEEDS for r in CRAWL))}", "min / max over s, crawl rows of |read_a.A_CTRL.s[row].delta| / floors.RF[row]")

SEC[0] = "4d"
P("")
P("### (d) The longitudinal cost at s0 (A − CTRL)")
P("")
P("| row | A − CTRL s0 | A − CTRL s1 | RF | bar | means s0 (A vs CTRL) |", "|---|---|---|---|---|---|")
for r in ["ADE6", "along-track MAE", "DK-VIOL"]:
    e0 = RA["A_CTRL"]["s0"][r]
    mm = f"{e0['mean_arm']:.4f} vs {e0['mean_ref']:.4f}" if e0["mean_arm"] is not None else "n/a"
    P(f"| {r} | {rc('A_CTRL','s0',r)} | {rc('A_CTRL','s1',r)} | {un(FL['RF'][r])} | {un(FL['bar'][r])} | {mm} |")
    aud(mm, f"read_a.A_CTRL.s0[\"{r}\"].mean_arm / mean_ref")
    aud(f"{un(FL['RF'][r])}; {un(FL['bar'][r])}", f"floors.RF/bar[\"{r}\"]")
P("")
ea, ealt = RA["A_CTRL"]["s0"]["ADE6"], RA["A_CTRL"]["s0"]["along-track MAE"]
assert all(RA["A_CTRL"]["s0"][r_]["separated_worse"] for r_ in ("ADE6", "along-track MAE", "DK-VIOL")) and not any(RA["A_CTRL"]["s1"][r_]["separated_worse"] for r_ in ("ADE6", "along-track MAE", "DK-VIOL"))
assert RA["A_CTRL"]["s1"]["accel MAE"]["separated_worse"] and [r_ for r_ in LONG_ROWS if RA["A_CTRL"]["s1"][r_]["separated_worse"]] == ["accel MAE"]
assert all(abs(RA["A_CTRL"][s_][r_]["delta"]) < abs(RA["REP_CTRL"]["s0"][r_]["delta"]) for s_ in SEEDS for r_ in CRAWL)
P(f"At s0 all three are separated-worse against CTRL; at s1 none is. Against the floors: ADE6 {sg(ea['delta'])} is below its RF ({un(FL['RF']['ADE6'])}) and its bar ({un(FL['bar']['ADE6'])}); along-track {sg(ealt['delta'])} is below its RF ({un(FL['RF']['along-track MAE'])});"
  f" DK-VIOL {sg(dk['delta'])} is above its RF ({un(FL['RF']['DK-VIOL'])}) and above its bar ({un(FL['bar']['DK-VIOL'])}, which RF0 sets), and R-NEW did not fire on it (section 2.5)."
  f" At s1 the longitudinal family carries one separated-worse row, accel MAE ({rc('A_CTRL','s1','accel MAE')}).")
aud("ADE6 / along-track / DK-VIOL vs RF and bar", "read_a.A_CTRL.s0[row].delta against floors.RF[row], floors.bar[row] (same cells as the table above)")
P("")
P("### What this read cannot claim")
P("")
P("* **It is not closed loop.** T1-kin is a self-action open loop: the lead follows its logged future and the cameras are logged (PREREG section 9).")
P("* **One training run per arm and seed.** The two seeds and the REP pair carry the training-run question; no row here is an effect of the lever on its own (`H-ESTIM-SEED-1`).")
P("* **No lateral / longitudinal split.** The perturbation has two components and neither half is credited.")
P("* **B is not read.** Nothing here says what p = 0.1 does.")

# ---------------------------------------------------------------------------------------------------------------- 5
hdr("## 5. The run history (facts; the notes and logs are in the package)")
SEC[0] = "5"
rk = lambda n: RUNREC[n]
P("**Launches and stops, from the box-2 chain logs (first to last timestamped line, minute resolution).**")
P("")
P("| launch | log | first → last stamp | span | what ended it |", "|---|---|---|---|---|")
JT = {(k_, j_[0]): (j_[2], j_[3]) for k_ in ("r2b", "r2c") for j_ in rjobs[k_] if j_[5] == "t" and not (k_ == "r2c" and j_[0].startswith("M2b_s0"))}
FLOOR_B = int(re.search(r"floor=(\d+)", [l for l in LL["r2d"] if l.startswith("ZZV8FMINI-STOP disk floor")][0])[1])
assert FLOOR_B == 1073741824
r1_pull = re.search(r"checkpoint sha256 ([0-9a-f]{64})", [l for l in LL["r1"] if "M2b_s0_dagoaccorc pulled to" in l][0])[1]
r2a_pull = re.search(r"checkpoint sha256 ([0-9a-f]{64})", [l for l in LL["r2a"] if "M2b_s0_dagoaccorc pulled to" in l][0])[1]
assert r1_pull == r2a_pull and r1_pull.startswith("6ea4d6ba")
for who in ("dagoaccorc s0", "dagoaccorcrep s0"):
    assert any(l.startswith("DAG-ORC (DAG-OACC base) run record: PASS") and who in l for l in LL["r2b"]), who
a1_log = rdb(LG + "v8fdagoaccorc10280348_t_M2b_s1_dagoaccorc.log").decode("utf-8", "replace")
assert a1_log.count("train M2b_s1_dagoaccorc") == 2 and "5:59:36" in a1_log
P(f"| r1 | `chain_v8f_dagoaccorc.out` | {SPAN['r1'][0]} → {SPAN['r1'][1]} | {SPAN['r1'][2]} min | four gates PASS, A s0 trained, **K-REACH FAIL at the run record**: perturbation {R1[0]:,} / {R1[1]:,} = {R1[2]:.4f}, outside 0.3 ± 0.02. Chain stopped (`ZZDAGOACCORC-CONTROL-FAIL`). Cause and fix: Amendment 2 (the swap and the perturbation drew from one random stream) |")
P(f"| r2, first launch | `…r2.out_r2a_marker_stop` | {SPAN['r2a'][0]} → {SPAN['r2a'][1]} | {SPAN['r2a'][2]} min | **refused at its first run record**: *\"{r2a_msg}\"*. A shared aside marker (`r2_aside_done.txt`, written by DAG-OACC's r2 chain into the same runs root) made the aside a no-op, the queue skipped A s0's training because r1's `summary.json` was still there, and r1's checkpoint (`{r1_pull[:8]}…`, the same sha256 as r1's pull) was pulled again. Note: `raw/2026-10-28-dag-orc-r2a/R2A_NOTE.md` |")
P(f"| r2, second launch | `…r2.out_r2b_queue_edit_stop` | {SPAN['r2b'][0]} → {SPAN['r2b'][1]} | {SPAN['r2b'][2]} min | A s0 and REP s0 trained and passed their run records; A s1 trained (checkpoint `{r2b_a1_sha[:8]}…`), then was **refused at its run record**: *\"summary.json init_ckpt not applied / read back: False / None\"*. Per `R2C_NOTE.md`, a self-test run by another agent reached the real box 2 at 05:56:16 and replaced the trainer, the model file and the queue `.cmd` while A s1 was training; cmd.exe resumed the new bytes at the old offset and ran the trainer again (the box-2 train log shows a second `train M2b_s1_dagoaccorc` at 05:59:36). Note: `notes/R2C_NOTE.md` |")
P(f"| disk floor | `…r2.out_r2c_diskfloor_stop` | no timestamp | 0 min | the first relaunch stopped before any job on the E: floor: pulling {DISK[0]:,} B would leave less than {FLOOR_B:,} B (free {DISK[1]:,} B). Per `R2C_NOTE.md`, two aside dirs and two dead lander holds (414 MB) were moved to F: |")
P(f"| r2c (final) | `chain_v8f_dagoaccorc_r2.out` | {SPAN['r2c'][0]} → {SPAN['r2c'][1]} | {SPAN['r2c'][2]} min | **`ZZV8FMINI-DONE`**. Pinned to `8c24823f` through `OAF_BR=agent/dagorc-r2-pin` (`run_chain_v8f_dagoaccorc_r2c_pin.cmd`; per `R2C_NOTE.md` the remote tip had moved to `94a47f4b`, a SPEEDSUB commit with no trainer, model, data or queue change). Four gates PASS; A s0 and REP s0 were skipped by the queue (already trained, jobs ended {JT[('r2b','M2b_s0_dagoaccorc')][1]} and {JT[('r2b','M2b_s0_dagoaccorcrep')][1]} in r2b, checkpoint sha256 re-verified); A s1 and REP s1 trained {JT[('r2c','M2b_s1_dagoaccorc')][0]}–{JT[('r2c','M2b_s1_dagoaccorc')][1]} and {JT[('r2c','M2b_s1_dagoaccorcrep')][0]}–{JT[('r2c','M2b_s1_dagoaccorcrep')][1]} |")
aud("; ".join(f"{k} {SPAN[k][0]} to {SPAN[k][1]} = {SPAN[k][2]} min" for k in ("r1", "r2a", "r2b", "r2c")), "first and last '10-28 HH:MM' lines of chain_v8f_dagoaccorc.out (L%d..L%d), ..._r2a_marker_stop (L%d..L%d), ..._r2b_queue_edit_stop (L%d..L%d), chain_v8f_dagoaccorc_r2.out (L%d..L%d)" % tuple(x for k in ("r1", "r2a", "r2b", "r2c") for x in (SPAN[k][3], SPAN[k][4])))
aud(f"{R1[0]:,} / {R1[1]:,} = {R1[2]:.4f}", "chain_v8f_dagoaccorc.out, the 'K-REACH: FAIL' line")
aud("3.9 h; 0.2 h; 0.7 h", "R2A_NOTE.md 'Cost' (total now about 3.9 h); R2C_NOTE.md 'Cost' (05:56-06:09 lost; A s1 trained twice, 0.7 h), quoted")
aud("05:56:16; 414 MB; about 4.8 h; 94a47f4b", "R2C_NOTE.md ('What happened' first bullet; 'What was done'; 'Cost'), quoted, not re-derived")
aud("05:59:36", "v8fdagoaccorc10280348_t_M2b_s1_dagoaccorc.log, second 'train M2b_s1_dagoaccorc' line stamp [28.10.2026  5:59:36,04]")
aud("~3 h; 2.8 h", "PREREG_V8F_DAG_ORC.md Amendment 1: 'It was asked as ~3 h of box 2' and section 4 'A + REP is 4 runs, ≈ 2.8 h'")
aud("805,306,368; 1,073,741,824; 1,858,076,672", "chain_v8f_dagoaccorc_r2.out_r2c_diskfloor_stop (ZZDISK-FLOOR-STOP line)")
P("")
P(f"**Box-2 cost.** Summed over the four launches the logs span **{TOT_MIN} min = {TOT_MIN / 60:.2f} h** ({SPAN['r1'][2]} + {SPAN['r2a'][2]} + {SPAN['r2b'][2]} + {SPAN['r2c'][2]}), against the **~3 h** the PI's go was asked as"
  " (Amendment 1 section 4 planned 2.8 h) and against the **about 4.8 h** that `R2C_NOTE.md` states (an estimate that adds up the notes' own figures: `R2A_NOTE.md` 3.9 h after the first r2 launch, plus the 0.2 h and the 0.7 h that `R2C_NOTE.md` adds). The overrun beyond the ~3 h is"
  f" {(TOT_MIN - 180) / 60:.1f} h by the logs. The four kept trainings account for {train_sec / 3600:.2f} h of it ({', '.join(f'{ARMOF[n][0]} {ARMOF[n][1]} {RUNREC[n]['sm']['seconds']:.0f} s' for n in RUNNAMES)}; `summary.json`)."
  " The rest is the refused r1 A s0 training, the r2 first launch, the refused A s1 training and the gates, evals and pulls of each launch.")
aud(f"{TOT_MIN} min = {TOT_MIN / 60:.2f} h; overrun {(TOT_MIN - 180) / 60:.1f} h; {train_sec / 3600:.2f} h", "sum of the four spans; (sum - 180)/60; sum of summary.json seconds of the four kept runs", f"{TOT_MIN}; {train_sec}")
for n in RUNNAMES:
    aud(f"{RUNREC[n]['sm']['seconds']:.0f} s", f"{n}/summary.json seconds")
P("")
P("**Training jobs (log minutes; launch stamp to `ZZJOB-END`).**")
P("")
P("| run | launch | start → end | span | kept in the read |", "|---|---|---|---|---|")
for k, kn in (("r1", "r1"), ("r2b", "r2b"), ("r2c", "r2c")):
    for (run, ln, sm_, em_, span_, kind, le) in rjobs[k]:
        if kind != "t":
            continue
        keep = "yes" if (k == "r2b" and run in ("M2b_s0_dagoaccorc", "M2b_s0_dagoaccorcrep")) or (k == "r2c" and run in ("M2b_s1_dagoaccorc", "M2b_s1_dagoaccorcrep")) else ("skipped job (checkpoint already trained)" if k == "r2c" else "no: refused, never rolled out")
        P(f"| {run} | {kn} | {sm_} → {em_} | {span_} min | {keep} |")
        aud(f"{run} {kn}: {sm_} to {em_} = {span_} min", f"{LOGS[k].split('/')[-1]} L{ln} (launch) .. L{le} (ZZJOB-END)")
P("")
P("**The final four runs** (box-2 chain log `chain_v8f_dagoaccorc_r2.out`, the `summary:` line and the per-run lines; checkpoint sha256 from `SHA256SUMS.txt`, verified by `sha256sum -c`; the T1-kin rollout done-markers carry the same sha256).")
P("")
P(f"> `{summ_line[1][:summ_line[1].index('summary:') + len('summary:')]} " + " ".join(f"{n}={SUMM[n]}" for n in RUNNAMES) + f"` (L{summ_line[0]})")
aud(" ".join(f"{n}={SUMM[n]}" for n in RUNNAMES), f"chain_v8f_dagoaccorc_r2.out L{summ_line[0]}")
P("")
P("| run | trainer seed | argv sha16 | checkpoint sha256 | run record | K-REACH: swap n / eligible | K-REACH: perturbation n / eligible | overlap | stream | gate token (commit, argv sha256 prefix) |", "|---|---|---|---|---|---|---|---|---|---|")
for n in RUNNAMES:
    arm, s = ARMOF[n]
    sm_, tk = RUNREC[n]["sm"], RUNREC[n]["tk"]
    key = ("dagoaccorc" if arm == "A" else "dagoaccorcrep", int(s[1]))
    li, mm = RECLINE[key]
    kr = KREACH["dagoaccorc" if arm == "A" else "dagoaccorcrep"]
    krl = [x for x in kr if x[2] == sm_["dagger_n"] and x[3] == sm_["dagger_eligible"]][0]
    assert krl[1] == "PASS" and krl[5] == sm_["ego_perturb_n"] and krl[6] == sm_["ego_perturb_eligible"] and mm[5] == sm_["argv_sha"]
    sw_share, eg_share = sm_["dagger_n"] / sm_["dagger_eligible"], sm_["ego_perturb_n"] / sm_["ego_perturb_eligible"]
    ovl = sm_.get("ego_dagger_overlap_n")
    P(f"| {n} | {sm_['seed']} | `{sm_['argv_sha']}` | `{RUNREC[n]['sums']['ckpt_last.pt']}` | PASS (L{li}) | {sm_['dagger_n']:,} / {sm_['dagger_eligible']:,} = {sw_share:.4f} | {sm_['ego_perturb_n']:,} / {sm_['ego_perturb_eligible']:,} = {eg_share:.4f} | {ovl if ovl is not None else 'n/a (flag off)'} | {sm_['levers'].get('ego_perturb_stream')} | `{tk['binding']['commit'][:8]}`, `{tk['binding']['argv_sha256'][:16]}` |")
    aud(f"{n}: {sm_['dagger_n']:,}/{sm_['dagger_eligible']:,}={sw_share:.4f}; {sm_['ego_perturb_n']:,}/{sm_['ego_perturb_eligible']:,}={eg_share:.4f}",
        f"{n}/summary.json dagger_n, dagger_eligible, ego_perturb_n, ego_perturb_eligible, ego_dagger_overlap_n, levers.ego_perturb_stream; chain log K-REACH line; JSON perturbation_counts.{'A' if arm=='A' else 'REP'}_{s}", f"shares recomputed: {sw_share}, {eg_share}")
    pc = R["perturbation_counts"][f"{'A' if arm == 'A' else 'REP'}_{s}"]
    assert (pc["ego_perturb_n"], pc["ego_perturb_eligible"], pc["dagger_n"], pc["dagger_eligible"]) == (sm_["ego_perturb_n"], sm_["ego_perturb_eligible"], sm_["dagger_n"], sm_["dagger_eligible"])
P("")
P("Run-record and K-REACH lines, verbatim from `chain_v8f_dagoaccorc_r2.out`:")
P("")
P("```")
_vl = []
for n in RUNNAMES:
    arm, s_ = ARMOF[n]
    key = ("dagoaccorc" if arm == "A" else "dagoaccorcrep", int(s_[1]))
    li = RECLINE[key][0]
    assert "K-REACH" in FINAL[li] and FINAL[li - 1].startswith("DAG-ORC (DAG-OACC base) run record")
    for q in (li, li + 1):
        P(FINAL[q - 1].rstrip("\r"))
        aud(FINAL[q - 1].rstrip("\r"), f"chain_v8f_dagoaccorc_r2.out L{q} (verbatim; the source line ends in a carriage return)")
P("```")
P("")
P("The K-REACH bands are 0.3 ± 0.02 for A, exactly 0 for REP, and 0.1 ± 0.02 for the swap; the overlap of swapped and perturbed samples is exactly 0 on A. All four fall inside. The same gate tokens (same commit, same argv sha256) were written in the second launch (r2b) and the final one (r2c).")
bk = R["k_build"]
nch = len(bk["changed"])
assert nch == 37 and all(h["default_off"] and h["bit_identity_test"] == "PASS" and h["behind_flag"] == "--ego-perturb-with-dagger" for h in bk["changed"]) and len(bk["identical_files"]) == 2
P("")
P(f"**K-BUILD** (launch commit `{bk['launch_commit'][:8]}` against `{bk['base_commit'][:8]}`, the commit DAG-OACC trained at): {nch} changed hunks in 2 files (`v8f_mini_train.py` 25, `v8f_mini_data.py` 12), every one behind `--ego-perturb-with-dagger`,"
  " default off, bit-identity test PASS; `v8f_mini.py` and `v8f_dagger.py` byte-identical. Record: `kbuild/kbuild_r2_launch_8c24823f.json` (sha256 `" + sha(KB)[:8] + "…`).")
aud(f"{nch}; 25; 12; 2", "k_build.changed (len; by file), k_build.identical_files")
assert sum(1 for h in bk["changed"] if h["file"].endswith("v8f_mini_train.py")) == 25 and sum(1 for h in bk["changed"] if h["file"].endswith("v8f_mini_data.py")) == 12

# ---------------------------------------------------------------------------------------------------------------- 6
hdr("## 6. Next (Rule Zero): A is not adopted; the committed next is B, blocked on the PI's go")
SEC[0] = "6"
P(f"* **Committed next (the table, verbatim from the JSON):** *\"{R['A']['committed_next']}\"* `cascade.b_status`: *\"{R['cascade']['b_status']}\"*.")
aud("committed_next, b_status", "A.committed_next; cascade.b_status (quoted)")
est = (A_SM["s0"]["seconds"] + A_SM["s1"]["seconds"]) / 3600 + 2 * 4 / 60
P("* **Blocked on:** the PI's go for B (A1 section 4: *B is not queued*). What unblocks it: the go, then box 2 for two runs through the launch gate. "
  f"ESTIMATED ≈ {est:.1f} h of box 2: A's two trainings took {A_SM['s0']['seconds']:.0f} s and {A_SM['s1']['seconds']:.0f} s (`summary.json`), plus about 4 min per run for gate and eval (log minutes). The T1-kin rollouts and the read are dev-box CPU.")
aud(f"{est:.1f} h", "(A s0 seconds + A s1 seconds)/3600 + 2*4/60 (the 4 min is an ESTIMATE from the gate and eval stamps of the r2c log)", "ESTIMATED")
P("")
P("**The Master Mind's ranking, beside the table (arithmetic, not a forecast).**")
P("")
bfac = {k: FL["bar"][k[1]] / v for k, v in imp.items()}
P("1. **What B must clear.** R-MOVE needs, on every row of C at a seed, an improvement ≥ bar. The bars are defined from RF0 and the REP pair (PREREG section 3), neither of which involves the dose, and the committed next reads B *\"under this same table\"*; the factors below assume the same bars"
  " (PREREG section 7 also says B's claim *\"carries B's own floors and seeds\"*; if the reader recomputes floors for B, the factors change). A's crawl improvements stayed below RF and below the bar (section 4 (c)); the bar divided by A's improvement is the factor B's improvement would have to exceed A's by:")
P("")
P("| row | s0: bar / A's improvement | s1: bar / A's improvement |", "|---|---|---|")
for r in CRAWL:
    P(f"| {r} | {bfac[('s0', r)]:.1f} × | {bfac[('s1', r)]:.1f} × |")
    aud(f"{bfac[('s0', r)]:.1f}; {bfac[('s1', r)]:.1f}", f"floors.bar[\"{r}\"] / -read_a.A_CTRL.sX[\"{r}\"].delta", f"{FL['bar'][r]}/{imp[('s0', r)]}; {FL['bar'][r]}/{imp[('s1', r)]}")
fmin, fmax = min(bfac.values()), max(bfac.values())
P("")
P(f"   B's dose is a third of A's (p = 0.1 against 0.3). **So B cannot reach R-MOVE unless its crawl effect exceeds A's by {fmin:.1f}–{fmax:.1f} ×,** at a third of the dose.")
aud(f"{fmin:.1f}-{fmax:.1f}", "min / max of the 8 factors above")
P("2. **Where B's outcomes lead.** Under PREREG section 7 and A1 section 3, HARMS of B, NO EFFECT and PARTIAL / ONE SEED all route to NEXT-CRAWL, which on DAG-OACC's states is LAUNCH-INTENT"
  " (N0 read NOT NAVMOVE; COMPOSE is not met, since DAG-OACC − DAG is not separated-better on the rows of C, `RESULT_V8F_DAG_OACC.md` section 3). Only R-MOVE at both seeds routes elsewhere. So B changes the route only if it clears the factors above.")
P("3. **The registered sequel is NEXT-CRAWL → LAUNCH-INTENT** (draft: `dagland/launch_intent/PREREG_V8F_LAUNCH_INTENT.md`, on E:). Its draft reuses this REP pair, and therefore these floors: the bars in section 2.1 are the"
  " bars a LAUNCH-INTENT crawl effect has to clear unless its PREREG sets its own, and A's measured effects (section 2.4) are the reference size. Its REP-reuse licence is conditional (its section 4); this read shows the REP run records PASS"
  " (swap share 0.1013 / 0.0999, perturbation exactly 0, composition flag absent, stream 0). Its other conditions were not checked here (UNVERIFIED).")
P("")
P("**Unblocked now, 0 GPU, each a registered item of this PREREG that did not get built (work items, not done in this package):**")
P("")
P("* **M-LAUNCH** (PREREG section 4): A − CTRL of the mean (v_own − v_GT) at ticks 1–2 in the GT-launch windows, from the tick dumps on F: (`F:/TanitAD-scratch/dagorc_t1_out/ticks/`). It is the cheapest discriminating row for the ranking: *\"If R-MOVE fails while M-LAUNCH moves, the speed deficit was repaired and the crawl rows did not follow. That outcome points away from the state channel.\"*")
P("* **The T0 four families** (`orc_families.py` on the pulled `eval_*.npz` of A and CTRL) and the **anti-echo block** and **M2 / L2 on the GT-launch subset** (section 3.3).")
P("* **Single copy.** The per-episode dumps and tick arrays are on F: scratch only (`F:/TanitAD-scratch/dagorc_t1_out/`); the checkpoints and T0 evals are on E:. None is in the repo.")

# ---------------------------------------------------------------------------------------------------------------- 7
hdr("## 7. Provenance (artifact path, sha256)")
SEC[0] = "7"
P("Repo paths are under `" + V8 + "` and are staged by the hold (`landing/mk_hold_dagorc_result.py`). Source paths are where the bytes were read from.")
P("")
P("| role | source path | repo path | sha256 |", "|---|---|---|---|")
prov = []


def pr(role, src, repo):
    h = sha(src)
    prov.append((role, src, repo, h))
    P(f"| {role} | `{src}` | `{repo}` | `{h}` |")


pr("read JSON (reader output)", RD + "DAGORC_READ.json", PKG + "read/DAGORC_READ.json")
pr("replicate floors RF", RD + "RF_REP.json", PKG + "read/RF_REP.json")
pr("T1-kin chain log", T1L + "chain.out", PKG + "logs/chain.out")
pr("reader log", T1L + "dagorc_read.out", PKG + "logs/dagorc_read.out")
pr("box-2 chain log, r2c (final; the `summary:` line)", LOGS["r2c"], PKG + "box2/logs/chain_v8f_dagoaccorc_r2.out")
pr("box-2 chain log, r1 (K-REACH stop)", LOGS["r1"], PKG + "box2/logs/chain_v8f_dagoaccorc.out")
pr("box-2 chain log, r2 first launch (marker stop)", LOGS["r2a"], PKG + "box2/logs/chain_v8f_dagoaccorc_r2.out_r2a_marker_stop")
pr("box-2 chain log, r2 second launch (queue-edit stop)", LOGS["r2b"], PKG + "box2/logs/chain_v8f_dagoaccorc_r2.out_r2b_queue_edit_stop")
pr("box-2 chain log, disk-floor stop", LOGS["r2d"], PKG + "box2/logs/chain_v8f_dagoaccorc_r2.out_r2c_diskfloor_stop")
pr("box-2 train log of the A s1 incident (second `train` at 05:59:36)", LG + "v8fdagoaccorc10280348_t_M2b_s1_dagoaccorc.log", PKG + "box2/logs/v8fdagoaccorc10280348_t_M2b_s1_dagoaccorc.log")
pr("pin launcher of the final launch", "C:/Users/Admin/box2_ship/run_chain_v8f_dagoaccorc_r2c_pin.cmd", PKG + "box2/run_chain_v8f_dagoaccorc_r2c_pin.cmd")
pr("note: A s1 refused (self-test replaced the queue)", NOTES + "R2C_NOTE.md", PKG + "notes/R2C_NOTE.md")
pr("note: r2 first launch refused (shared aside marker)", NOTES + "R2A_NOTE.md", "raw/2026-10-28-dag-orc-r2a/R2A_NOTE.md (already in the repo)")
pr("K-BUILD record, launch commit `8c24823f`", KB, PKG + "kbuild/kbuild_r2_launch_8c24823f.json")
pr("frozen RF0 (P-0)", "E:/Projects/TanitAD-artifacts/dagorc_rebase/read/P0_RF0.json", "raw/2026-10-28-dag-orc-build/rebase/read/P0_RF0.json (already in the repo)")
pr("reader", "E:/Projects/TanitAD-artifacts/dagorc_rebase/read/dag_orc_read.py", "raw/2026-10-28-dag-orc-build/rebase/read/dag_orc_read.py (already in the repo)")
for n in RUNNAMES:
    for f in ("summary.json", "selfcheck.json", "argv.json", "launch_gate_token.json", "SHA256SUMS.txt"):
        pr(f"run record {n}", RUNS + n + "/" + f, PKG + f"runs/{n}/{f}")
    P(f"| checkpoint {n} (not staged) | `{RUNS}{n}/ckpt_last.pt` | none (E: only) | `{RUNREC[n]['sums']['ckpt_last.pt']}` |")
P("")
P("The reader's pins (`provenance.ref_pins`), the PREREG sha256 it read (`38bc688b…`, the file at `3b7b404e`) and Amendment 1's (`f2f96157…`) are in the JSON under `provenance`; Amendment 2 changes the build and the re-run,"
  " not a row, a bar or a floor (its section 2). The PREREG read for this document is the file at `4c412181` (sha256 `d8b8885b…`, Amendments 1 and 2 included).")
aud("38bc688b / f2f96157 / d8b8885b", "provenance.prereg_sha256, provenance.amend1_sha256; sha256 of PREREG at 4c412181 (git show)")

TXT = "\n".join(L) + "\n"
TXT = TXT.replace("\n\n\n", "\n\n")
for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"):
    assert w not in TXT.lower(), w
assert "\r" not in TXT and not re.search(r"hf_[A-Za-z0-9]{20,}", TXT)
open(OUT + "RESULT_V8F_DAG_ORC.md", "w", encoding="utf-8", newline="\n").write(TXT)

# ---------------------------------------------------------------------------------------------------------------- register row
SEC[0] = "register"
a0, a1 = RA["A_CTRL"]["s0"], RA["A_CTRL"]["s1"]
cr = {s: [sg(RA["A_CTRL"][s][r]["delta"]) for r in CRAWL] for s in SEEDS}
rfv = [un(FL["RF"][r]) for r in CRAWL]
rowtxt = (
    f"**HARMS (MEASURED 2026-10-28, arm A, Amendments 1 and 2 in force; all {NCTRL} reader controls PASS).** A = DAG-OACC_s's argv + O-RECOVER's ego-state perturbation (p 0.3) behind `--ego-perturb-with-dagger`,"
    " against CTRL = DAG-OACC_s (banked) and a same-recipe replicate REP (trainer seed 10 / 11). Guards: R-TLAT and R-TLON held at both seeds; "
    f"R-ADE FAILS at s0 (T1-kin ADE6 A − CTRL {sg(a0['ADE6']['delta'])} [{sg(a0['ADE6']['lo'])}, {sg(a0['ADE6']['hi'])}] m, separated-worse; s1 {sg(a1['ADE6']['delta'])} [{sg(a1['ADE6']['lo'])}, {sg(a1['ADE6']['hi'])}], held); "
    f"R-T0 FAILS at both seeds (T0 ADE6 A − CTRL {sg(t0p0)} / {sg(t0p1)} m against ≤ +0.0604 m); R-NEW held (0 of 15 rows at each seed). "
    f"Decisive rows: R-MOVE false at both seeds, 0 of 4 crawl rows clear 2 × max(F, RF0, RF) (yaw-rate / curvature / L2 / M1 A − CTRL {' / '.join(cr['s0'])} at s0 and {' / '.join(cr['s1'])} at s1, each separated-better,"
    f" {pct(rmin)} to {pct(rmax)} of its bar; the REP pair sets every crawl bar, RF {' / '.join(rfv)} from REP s0); "
    "R-CRAWL false at both seeds (A − P10 still separated-worse on 3 of 4 rows at s0 and 4 of 4 at s1); "
    f"adoption: R-KEEP false at s0 ({sg(g0['KEEP_point'])} > +0.0604) and true at s1 ({sg(g1['KEEP_point'])}), R-NOBASE true at both. "
    f"Cost at s0 (A − CTRL): ADE6 {sg(a0['ADE6']['delta'])} m, along-track MAE {sg(a0['along-track MAE']['delta'])}, DK-VIOL {sg(a0['DK-VIOL']['delta'])}, all separated-worse, none at s1. "
    "T1-kin self-action open loop (R-T0 at T0); paired episode-cluster bootstrap, B = 2,000, 141 episodes, per seed: it answers the episode question, and the training-run question is carried by REP, whose ADE6 difference from CTRL is "
    f"{sg(RA['REP_CTRL']['s0']['ADE6']['delta'])} (s0) and {sg(RA['REP_CTRL']['s1']['ADE6']['delta'])} (s1) (`H-ESTIM-SEED-1`). "
    "Four families at T1-kin in the RESULT; the T0 four families are NOT RUN, and M-LAUNCH, the anti-echo block and M2 / L2 on the launch subset are NOT BUILT (work items). "
    "Committed next: A is not adopted; read B (p 0.1) under the same table, which needs its own PI go (not queued); the registered sequel is NEXT-CRAWL, LAUNCH-INTENT. "
    "`RESULT_V8F_DAG_ORC.md`, `raw/2026-10-28-dag-orc-result/read/DAGORC_READ.json`. Was:"
)
for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"):
    assert w not in rowtxt.lower(), w
assert "|" not in rowtxt and "\n" not in rowtxt
aud("register row status prefix", "same cells as the RESULT sections 1, 2.4, 4(c), 4(d); rows.A.sX.KEEP_point; rows.A.sX.T0_point; read_a.REP_CTRL.sX.ADE6")
EVID_OLD, EVID_NEW = "MEASURED (N0); HYPOTHESIS (A)", "MEASURED (N0, A); HYPOTHESIS (B)"
rr = [
    "# H-V8F-DAG-ORC-1: replacement text for the register row (staged, not landed)",
    "",
    "Generated by `gen_result.py` from `DAGORC_READ.json` (sha256 `" + sha_read[:8] + "…`). The hold builder `mk_hold_dagorc_result.py` reads the paragraph between the two markers, inserts it at the start of the row's status cell, keeps the previous status text after `Was: `, and",
    "changes the evidence cell. The claim cell and the source cell stay. The file's line count does not change (the row is one line).",
    "",
    "Row: the line of `Project Steering/GOALS_AND_CLAIMS.md` that starts `| **H-V8F-DAG-ORC-1** (2026-10-27) |`. As `H-V8F-SPEEDREAD-1` was promoted: new result first, `Was:` then the earlier text unchanged.",
    "",
    "## Status cell (new prefix)",
    "",
    "<!-- STATUS-BEGIN -->",
    rowtxt,
    "<!-- STATUS-END -->",
    "",
    "## Evidence cell",
    "",
    f"`{EVID_OLD}` becomes `{EVID_NEW}`. A is MEASURED (HARMS); B is untested.",
    "",
    "<!-- EVID-OLD: " + EVID_OLD + " -->",
    "<!-- EVID-NEW: " + EVID_NEW + " -->",
    "",
]
open(OUT + "REGISTER_ROW.md", "w", encoding="utf-8", newline="\n").write("\n".join(rr))

# ---------------------------------------------------------------------------------------------------------------- audit
A_ = ["# NUMBERS_AUDIT: every number in RESULT_V8F_DAG_ORC.md and REGISTER_ROW.md, with the source it was read from", "",
      "Generated by `gen_result.py` in the same run that wrote the RESULT, so each rendered value below is the value that was printed. Paths are JSON key paths into `DAGORC_READ.json`"
      f" (sha256 `{sha_read}`) or `RF_REP.json` (sha256 `{sha_rf}`) unless a file is named. `[\"row\"]` is a key; `{{a,b}}` means each of those keys. Derived numbers show the arithmetic.", "",
      "Cross-checks the script asserts before it writes (a failure stops the build):", "",
      "* `read_a.A_DAG` equals `read_a.A_CTRL`, and `read_a.DAG_P10` equals `read_a.CTRL_P10` (Amendment 1).",
      "* `floors.RF` equals `RF_REP.json` `RF`; `RF_REP.json` `REP_minus_CTRL` equals `read_a.REP_CTRL` on all 19 rows and both seeds; `floors.bar` = 2 × max(F, RF0, RF) on all 19 rows.",
      "* R-MOVE, R-CRAWL, R-NEW, R-NOBASE, R-ADE, R-T0, R-KEEP are recomputed per seed from the reader's deltas and floors and equal the reader's flags.",
      "* Every `four_families` leaf equals the matching `read_a` cell (delta, lo, hi, separated) and its B / W tag.",
      "* Run-record files match `SHA256SUMS.txt`; the checkpoint sha256 in `SHA256SUMS.txt` equals the chain log's `pulled` line, its `summary:` prefix, and the T1-kin rollout done-marker `ckpt_sha256`; K-REACH counts in the chain log equal `summary.json` and `perturbation_counts`.",
      "* Coverage (`audit_cover.py`, run at the end of the build and asserted to report 0 uncovered): every numeric token of the RESULT and of the register row text has a matching token in the table below."
      " Not counted as data: section references, list and heading numerals, hex digests, dates inside repo paths.",
      "", "| section | rendered | source | arithmetic / note |", "|---|---|---|---|"]
for sec_, ren, src, note in AUD:
    A_.append("| " + " | ".join(x.replace("|", "\\|").replace("\n", " ") for x in (sec_, ren, src, note)) + " |")
open(OUT + "NUMBERS_AUDIT.md", "w", encoding="utf-8", newline="\n").write("\n".join(A_) + "\n")
print("wrote RESULT", len(TXT), "chars,", len(AUD), "audit entries, provenance files", len(prov))
import importlib.util
_spec = importlib.util.spec_from_file_location("audit_cover", OUT + "audit_cover.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_un = _mod.main()
assert all(len(v) == 0 for v in _un.values()), _un
