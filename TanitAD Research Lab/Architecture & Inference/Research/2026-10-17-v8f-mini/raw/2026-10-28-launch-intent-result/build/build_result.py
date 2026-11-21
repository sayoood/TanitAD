#!/usr/bin/env python
"""build_result.py: render RESULT_V8F_LAUNCH_INTENT.md from the banked read.

Every number in the document is copied from a banked file by this script and
never retyped: the JSON of record, the box-2 chain log, the T1-kin chain logs,
the PREREG text.  The prose lives in TEMPLATE below; [[expr]] is replaced by the
value of a Python expression over the helpers defined here.

Sources are read only from the pinned commit with
    git -C <mirror> -c safe.directory=* show d7dc9f11:<path>
or from a directory that holds the same repo-relative layout (--root DIR).

Usage:  python build_result.py [--root DIR] [--out PATH]
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys

MIRROR = "C:/Users/Admin/tanitad-push"
COMMIT = "d7dc9f11"
RESEARCH = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = RESEARCH + "/raw/2026-10-28-launch-intent-read"
CHAIN = RESEARCH + "/raw/2026-10-28-launch-intent-chain"
PREREG_PATH = "Project Steering/PREREG_V8F_LAUNCH_INTENT.md"

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--out", default="RESULT_V8F_LAUNCH_INTENT.md")
ARGS = ap.parse_args()


def get(path):
    if ARGS.root:
        with open(ARGS.root.rstrip("/") + "/" + path, "rb") as f:
            return f.read()
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "-c", "core.autocrlf=false",
                        "show", COMMIT + ":" + path], capture_output=True)
    if r.returncode != 0:
        raise SystemExit("git show failed for %s: %s" % (path, r.stderr.decode("utf-8", "replace")))
    return r.stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def text(path):
    return get(path).decode("utf-8").replace(chr(13) + chr(10), chr(10))


JSON_PATH = RAW + "/read_A1/LAUNCH_INTENT_READ.json"
JSON_BYTES = get(JSON_PATH)
D = json.loads(JSON_BYTES.decode("utf-8"))
BOX2 = text(RAW + "/box2/chain_v8f_launchintent.out")
REFUSED = text(RAW + "/lit1_logs/launch_intent_read_REFUSED.out")
CHAIN_OUT = text(RAW + "/lit1_logs/chain.out")
CHAIN_PREV = text(RAW + "/lit1_logs/chain.out.prev_1028151251")
PREREG = text(PREREG_PATH)

RA = D["read_a"]
ROWS = D["rows"]["A"]
FL = D["floors"]
ALL_ROWS = list(RA["A_CTRL"]["s0"].keys())
CRAWL = ["yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"]
FAMS = ["LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"]
SEEDS = ["s0", "s1"]
PAIRS = ["main", "rep"]
XY = {"main": "A_CTRL", "rep": "Arep_REP"}
XP = {"main": "A_P10", "rep": "Arep_P10"}
YP = {"main": "CTRL_P10", "rep": "REP_P10"}
XNAME = {"main": "A", "rep": "A_rep"}
YNAME = {"main": "CTRL", "rep": "REP"}
PAIRWORD = {"main": "main", "rep": "replicate"}
TRAINER_SEED = {("s0", "main"): 0, ("s0", "rep"): 10, ("s1", "main"): 1, ("s1", "rep"): 11}
MINUS = "\u2212"
DASH = "\u2013"
F_HALF = 0.0604  # PREREG section 7: R-T0 and R-KEEP read the point against +F/2


# ----------------------------------------------------------------- formatting
def sg(x):
    x = 0.0 if x == 0 else x
    return "%+.4f" % x


def un(x):
    return "%.4f" % x


def flag(c):
    return "B" if c["separated_better"] else ("W" if c["separated_worse"] else DASH)


def cell(c, bold=False):
    t = "%s [%s, %s] %s" % (sg(c["delta"]), sg(c["lo"]), sg(c["hi"]), flag(c))
    return "**" + t + "**" if (bold and flag(c) != DASH) else t


def C(kind, s, p, row):
    """the read_a cell: kind xy / xp / yp"""
    key = {"xy": XY, "xp": XP, "yp": YP}[kind][p]
    return RA[key][s][row]


def cellt(kind, s, p, row, bold=False):
    return cell(C(kind, s, p, row), bold)


def delta(kind, s, p, row):
    return sg(C(kind, s, p, row)["delta"])


def bar(row):
    return un(FL["bar"][row])


def pct(x):
    return "%.1f %%" % (100.0 * x)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def improvement(s, p, row):
    return -C("xy", s, p, row)["delta"]


def clears(s, p, row):
    c = C("xy", s, p, row)
    return c["separated_better"] and improvement(s, p, row) >= FL["bar"][row]


# ------------------------------------------------------------ log parsing
def parse_box2():
    out = {"gate": {}, "rec": {}, "share": {}, "ckpt": {}, "jobs": {}}
    for ln in BOX2.splitlines():
        m = re.match(r"LAUNCH-INTENT gate: PASS \| (\w+) s(\d) \| launch W ([\d.]+) \| trainer seed (\d+) \| clips (\d+) steps (\d+) windows (\d+) \| init (\w+) \| swap probe grad diff ([\d.]+) \| launch probe contribution ([\d.]+) head_o grad diff ([\d.]+) \| probe batch launch (\d+) / (\d+)", ln)
        if m:
            out["gate"][(m.group(1), m.group(2))] = dict(W=m.group(3), seed=m.group(4), swap=m.group(9), contrib=m.group(10), grad=m.group(11), pbl=m.group(12), pbn=m.group(13))
        m = re.match(r"LAUNCH-INTENT run record: PASS \| (A_[\w,]+) \| (\w+) s(\d) \| trainer seed (\d+) \| argv sha16 (\w+) \| steps (\d+) \| launch_intent_w ([\d.]+) \| w_acc ([\d.]+)", ln)
        if m:
            out["rec"][m.group(1)] = dict(name=m.group(2), s=m.group(3), seed=m.group(4), argv=m.group(5), steps=m.group(6), W=m.group(7), wacc=m.group(8))
        m = re.match(r"LAUNCH-INTENT K-SHARE: PASS \| (A_[\w,]+) \| launch (\d+) / (\d+) = ([\d.]+) \(K-MASK: f ([\d.]+) \+- ([\d.]+)\) \| weight share ([\d.]+) \(must lie in 1/3 \+- ([\d.]+)\) \| rows (\d+) \(== eligible\)", ln)
        if m:
            out["share"][m.group(1)] = dict(n=m.group(2), rows=m.group(3), frac=m.group(4), f=m.group(5), share=m.group(7), band=m.group(8))
        m = re.match(r"(\w{64}) \*ckpt_last\.pt", ln)
        if m:
            out.setdefault("_ck", []).append(m.group(1))
    # checkpoint shas appear once per run, in run order
    order = ["A_0", "A_1", "A_rep,0", "A_rep,1"]
    for r, ck in zip(order, out.get("_ck", [])):
        out["ckpt"][r] = ck
    # job spans: the 'launched ... t_<run>.log' line stamp and the next ZZJOB-END
    lines = BOX2.splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"10-28 (\d\d:\d\d) launched pid=\d+ rc=0 log=D:\\tanitad\\logs\\v8flaunchintent\d+_(t|e)_(M2b_s\d_launchintent(?:rep)?)\.log", ln)
        if m:
            for j in range(i + 1, len(lines)):
                m2 = re.match(r"===== ZZJOB-END 28\.10\.2026 (\d\d:\d\d:\d\d)", lines[j])
                if m2:
                    out["jobs"][(m.group(3), m.group(2))] = (m.group(1), m2.group(1))
                    break
    return out


B2 = parse_box2()
RUNORDER = [("A_0", "s0", "main", "M2b_s0_launchintent"), ("A_1", "s1", "main", "M2b_s1_launchintent"),
            ("A_rep,0", "s0", "rep", "M2b_s0_launchintentrep"), ("A_rep,1", "s1", "rep", "M2b_s1_launchintentrep")]


def parse_chain_walls():
    out = {}
    for ln in CHAIN_OUT.splitlines():
        m = re.match(r"10-28 (\d\d:\d\d:\d\d) end (LI|LIREP)_s(\d)_m10_(causal|logged)_heads rc=0 wall=(\d+)s", ln)
        if m:
            out[(m.group(2), m.group(3), m.group(4))] = (m.group(5), m.group(1))
        m = re.match(r"10-28 (\d\d:\d\d:\d\d) end fresh-T0 eval M2b-launchintent(rep)?_s(\d)-fresh rc=0 wall=(\d+)s", ln)
        if m:
            out[("LIREP" if m.group(2) else "LI", m.group(3), "t0")] = (m.group(4), m.group(1))
    return out


WALL = parse_chain_walls()


# ------------------------------------------------------------------ tables
def t_outcome():
    hdr = ["predicate (PREREG section 10)"]
    for s in SEEDS:
        hdr += ["%s main (A %s CTRL)" % (s, MINUS), "%s replicate (A_rep %s REP)" % (s, MINUS), "**%s, both pairs**" % s]
    rows = []
    rows.append(["controls (reader, %d)" % len(D["controls_PASS"])] + ["PASS"] * 6)
    for g in ["R-TLAT", "R-TLON"]:
        r = [g + " (T head macro-F1, ticks >= 1, X %s Y not separated-worse)" % MINUS]
        for s in SEEDS:
            for p in PAIRS:
                r.append("held" if ROWS[s]["pairs"][p]["guards"][g] else "FAILS")
            r.append("**held**" if ROWS[s]["guards"][g] else "**FAILS**")
        rows.append(r)
    r = ["R-ADE (T1-kin ADE6, X %s Y, not separated-worse)" % MINUS]
    for s in SEEDS:
        for p in PAIRS:
            c = C("xy", s, p, "ADE6")
            ok = ROWS[s]["pairs"][p]["guards"]["R-ADE"]
            r.append("%s: %s %s" % ("held" if ok else "FAILS", sg(c["delta"]), flag(c)))
        r.append("**held**" if ROWS[s]["guards"]["R-ADE"] else "**FAILS**")
    rows.append(r)
    r = ["R-T0 (T0 own-code ADE6, X %s Y, point <= +%s m)" % (MINUS, un(F_HALF))]
    for s in SEEDS:
        for p in PAIRS:
            pr = ROWS[s]["pairs"][p]
            r.append("%s: %s" % ("held" if pr["guards"]["R-T0"] else "FAILS", sg(pr["T0_point"])))
        r.append("**held**" if ROWS[s]["guards"]["R-T0"] else "**FAILS**")
    rows.append(r)
    r = ["R-NEW (no row of D fires)"]
    for s in SEEDS:
        for p in PAIRS:
            pr = ROWS[s]["pairs"][p]
            nf = sum(1 for v in pr["R-NEW_fires"].values() if v)
            r.append("%s: %d of %d rows" % ("held" if pr["guards"]["R-NEW"] else "FAILS", nf, len(pr["domain"])))
        r.append("**held**" if ROWS[s]["guards"]["R-NEW"] else "**FAILS**")
    rows.append(r)
    r = ["R-MOVE (every crawl row clears its bar)"]
    for s in SEEDS:
        for p in PAIRS:
            pr = ROWS[s]["pairs"][p]
            n = sum(1 for v in pr["R-MOVE_rows"].values() if v)
            r.append("%s: %d of %d rows clear" % ("true" if pr["flags"]["R-MOVE"] else "false", n, len(pr["R-MOVE_rows"])))
        r.append("**true**" if ROWS[s]["flags"]["R-MOVE"] else "**false**")
    rows.append(r)
    r = ["R-CRAWL (every crawl row of X %s P10 not separated-worse)" % MINUS]
    for s in SEEDS:
        for p in PAIRS:
            pr = ROWS[s]["pairs"][p]
            n = sum(1 for v in pr["R-CRAWL_rows"].values() if v)
            r.append("%s: %d of %d rows" % ("true" if pr["flags"]["R-CRAWL"] else "false", n, len(pr["R-CRAWL_rows"])))
        r.append("**true**" if ROWS[s]["flags"]["R-CRAWL"] else "**false**")
    rows.append(r)
    r = ["R-KEEP (T1-kin ADE6, X %s Y, point <= +%s m)" % (MINUS, un(F_HALF))]
    for s in SEEDS:
        for p in PAIRS:
            pr = ROWS[s]["pairs"][p]
            r.append("%s: %s" % ("true" if pr["flags"]["R-KEEP"] else "false", sg(pr["KEEP_point"])))
        r.append("**true**" if ROWS[s]["flags"]["R-KEEP"] else "**false**")
    rows.append(r)
    r = ["R-NOBASE"]
    for s in SEEDS:
        for p in PAIRS:
            pr = ROWS[s]["pairs"][p]
            nv = sum(1 for v in pr["R-NOBASE_violations"].values() if v)
            r.append("%s: %d of %d rows violate" % ("true" if pr["flags"]["R-NOBASE"] else "false", nv, len(pr["domain"])))
        r.append("**true**" if ROWS[s]["flags"]["R-NOBASE"] else "**false**")
    rows.append(r)
    return table(hdr, rows)


def t_floors():
    hdr = ["row", "better", "F", "RF0", "**bar** = 2 × max(F, RF0)", "RF (beside, not a bar term)", "RF / bar"]
    rows = []
    for r in ALL_ROWS:
        rows.append([r, RA["A_CTRL"]["s0"][r]["direction"], un(FL["F"][r]), un(FL["RF0"][r]), "**%s**" % un(FL["bar"][r]),
                     un(FL["RF"][r]), "%.2f" % (FL["RF"][r] / FL["bar"][r])])
    return table(hdr, rows)


def t_ade():
    hdr = ["seed", "pair", "row", "X %s Y" % MINUS, "mean X", "mean Y", "X %s P10" % MINUS, "Y %s P10" % MINUS, "n (windows / episodes)"]
    rows = []
    for s in SEEDS:
        for p in PAIRS:
            c = C("xy", s, p, "ADE6")
            rows.append([s, "%s (%s %s %s)" % (PAIRWORD[p], XNAME[p], MINUS, YNAME[p]), "ADE6", cell(c, True), un(c["mean_arm"]), un(c["mean_ref"]),
                         cellt("xp", s, p, "ADE6"), cellt("yp", s, p, "ADE6"), c["n"]])
    return table(hdr, rows)


def t_move():
    hdr = ["seed", "pair", "crawl row", "X %s Y" % MINUS, "improvement", "bar", "improvement / bar", "clears (R-MOVE row)"]
    rows = []
    for s in SEEDS:
        for p in PAIRS:
            for r in CRAWL:
                c = C("xy", s, p, r)
                rows.append([s, "%s (%s %s %s)" % (PAIRWORD[p], XNAME[p], MINUS, YNAME[p]), r, cell(c, True), sg(improvement(s, p, r)),
                             un(FL["bar"][r]), pct(improvement(s, p, r) / FL["bar"][r]), "true" if clears(s, p, r) else "false"])
    return table(hdr, rows)


def t_crawl():
    hdr = ["seed", "pair", "crawl row", "X %s P10" % MINUS, "Y %s P10" % MINUS, "X %s P10 not separated-worse (R-CRAWL row)" % MINUS, "share of the Y %s P10 gap closed" % MINUS]
    rows = []
    for s in SEEDS:
        for p in PAIRS:
            for r in CRAWL:
                x = C("xp", s, p, r)
                y = C("yp", s, p, r)
                share = (y["delta"] - x["delta"]) / y["delta"]
                rows.append([s, "%s (%s %s %s)" % (PAIRWORD[p], XNAME[p], MINUS, YNAME[p]), r, cell(x, True), cell(y, True),
                             "false" if x["separated_worse"] else "true", pct(share)])
    return table(hdr, rows)


def new_sets(s, p):
    pr = ROWS[s]["pairs"][p]
    dom = pr["domain"]
    w = [r for r in dom if C("xy", s, p, r)["separated_worse"]]
    w3 = [r for r in w if C("xy", s, p, r)["delta"] >= FL["bar"][r]]
    w13 = [r for r in w3 if C("xp", s, p, r)["separated_worse"]]
    return dom, w, w3, w13


def t_new():
    hdr = ["seed", "pair", "control Y", "rows in D", "X %s Y separated-worse" % MINUS, "of those, worsening >= bar", "of those, X %s P10 separated-worse (R-NEW fires)" % MINUS]
    rows = []
    for s in SEEDS:
        for p in PAIRS:
            dom, w, w3, w13 = new_sets(s, p)
            rows.append([s, "%s (%s %s %s)" % (PAIRWORD[p], XNAME[p], MINUS, YNAME[p]), "%s %s P10 not separated-worse on D" % (YNAME[p], MINUS), len(dom),
                         ", ".join(w) or "none", ", ".join(w3) or "none", ", ".join(w13) or "none"])
    return table(hdr, rows)


def t_t0keep():
    hdr = ["seed", "pair", "T0 own-code ADE6, X %s Y (point; no interval in the JSON)" % MINUS, "R-T0 (<= +%s m)" % un(F_HALF),
           "T1-kin ADE6, X %s Y (point)" % MINUS, "R-KEEP (<= +%s m)" % un(F_HALF)]
    rows = []
    for s in SEEDS:
        for p in PAIRS:
            pr = ROWS[s]["pairs"][p]
            rows.append([s, "%s (%s %s %s)" % (PAIRWORD[p], XNAME[p], MINUS, YNAME[p]), sg(pr["T0_point"]),
                         "held" if pr["guards"]["R-T0"] else "FAILS", sg(pr["KEEP_point"]), "true" if pr["flags"]["R-KEEP"] else "false"])
    return table(hdr, rows)


def ncell(row, s):
    ns = [C("xy", s, p, row)["n"] for p in PAIRS]
    return ns[0] if len(set(ns)) == 1 else " ; ".join(ns)


def t_family(fam):
    rows_f = list(D["four_families"]["A"]["s0"]["main"][fam]["rows"].keys())
    hdr = ["row", "better"]
    for s in SEEDS:
        hdr += ["A %s CTRL %s" % (MINUS, s), "A_rep %s REP %s" % (MINUS, s)]
    hdr += ["n, s0 main / s0 replicate / s1 main / s1 replicate"]
    out = []
    for r in rows_f:
        row = [r, RA["A_CTRL"]["s0"][r]["direction"]]
        for s in SEEDS:
            for p in PAIRS:
                row.append(cellt("xy", s, p, r, True))
        row.append(" ; ".join(C("xy", s, p, r)["n"] for s in SEEDS for p in PAIRS))
        out.append(row)
    return table(hdr, out)


def t_famcount():
    hdr = ["family (rows)"]
    for s in SEEDS:
        for p in PAIRS:
            hdr.append("%s %s: B / W / not separated" % (s, PAIRWORD[p]))
    rows = []
    for fam in FAMS:
        rows_f = list(D["four_families"]["A"]["s0"]["main"][fam]["rows"].keys())
        row = ["%s (%d)" % (fam, len(rows_f))]
        for s in SEEDS:
            for p in PAIRS:
                cs = [C("xy", s, p, r) for r in rows_f]
                b = sum(1 for c in cs if c["separated_better"])
                w = sum(1 for c in cs if c["separated_worse"])
                row.append("%d / %d / %d" % (b, w, len(cs) - b - w))
        rows.append(row)
    return table(hdr, rows)


def t_p10(s):
    hdr = ["row", "better", "A %s P10 %s" % (MINUS, s), "CTRL %s P10 %s" % (MINUS, s), "A_rep %s P10 %s" % (MINUS, s), "REP %s P10 %s" % (MINUS, s)]
    out = []
    for r in ALL_ROWS:
        out.append([r, RA["A_CTRL"]["s0"][r]["direction"], cellt("xp", s, "main", r, True), cellt("yp", s, "main", r, True),
                    cellt("xp", s, "rep", r, True), cellt("yp", s, "rep", r, True)])
    return table(hdr, out)


def flips(s):
    out = []
    for r in ALL_ROWS:
        a, b = C("xy", s, "main", r), C("xy", s, "rep", r)
        if (a["separated_better"] and b["separated_worse"]) or (a["separated_worse"] and b["separated_better"]):
            out.append(r)
    return out


def t_flip():
    hdr = ["seed", "trainer seed (main pair / replicate pair)", "main pair, A %s CTRL" % MINUS, "replicate pair, A_rep %s REP" % MINUS]
    rows = []
    for s in SEEDS:
        rows.append([s, "%d / %d" % (TRAINER_SEED[(s, "main")], TRAINER_SEED[(s, "rep")]), cellt("xy", s, "main", "ADE6", True), cellt("xy", s, "rep", "ADE6", True)])
    return table(hdr, rows)


def t_crawlcells():
    hdr = ["seed", "pair", "crawl row", "X %s Y" % MINUS, "improvement / bar", "clears (R-MOVE row)"]
    rows = []
    for s in SEEDS:
        for p in PAIRS:
            for r in CRAWL:
                c = C("xy", s, p, r)
                if c["separated_better"]:
                    rows.append([s, "%s (%s %s %s)" % (PAIRWORD[p], XNAME[p], MINUS, YNAME[p]), r, cell(c, True), pct(improvement(s, p, r) / FL["bar"][r]),
                                 "true" if clears(s, p, r) else "false"])
    return table(hdr, rows)


SLOW = D["beside"]["A"]["slow_draw"]
CELLS5 = ["lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms"]
SLOWKEYS = [("main:A-CTRL:s0", "s0 main (A %s CTRL)" % MINUS), ("rep:Arep-REP:s0", "s0 replicate (A_rep %s REP)" % MINUS),
            ("main:A-CTRL:s1", "s1 main (A %s CTRL)" % MINUS), ("rep:Arep-REP:s1", "s1 replicate (A_rep %s REP)" % MINUS)]


def t_slow():
    hdr = ["arm and pair (T0, m/s)"] + CELLS5 + ["label"]
    rows = []
    for k, lab in SLOWKEYS:
        v = SLOW[k]
        row = [lab]
        for c in CELLS5:
            x = v["cells"][c]
            row.append("%s [%s, %s]" % (sg(x["mean"]), sg(x["lo"]), sg(x["hi"])))
        row.append("none" if v["label"] is None else v["label"])
        rows.append(row)
    return table(hdr, rows)


def slow_stats():
    ms = [abs(SLOW[k]["cells"][c]["mean"]) for k, _ in SLOWKEYS for c in CELLS5]
    sep = sum(1 for k, _ in SLOWKEYS for c in CELLS5 if SLOW[k]["cells"][c]["lo"] > 0 or SLOW[k]["cells"][c]["hi"] < 0)
    return un(max(ms)), sep, len(ms)


def t_share():
    hdr = ["run", "trainer seed", "launch rows / rows", "launch-row fraction (K-MASK: 0.1619 +- 0.02)", "weight share (K-SHARE: 1/3 +- 0.02)", "argv sha16", "checkpoint sha256 (12 hex)"]
    rows = []
    for r, s, p, nm in RUNORDER:
        sh = B2["share"][r]
        rc = B2["rec"][r]
        rows.append([r + " (" + nm + ")", rc["seed"], "%s / %s" % (sh["n"], sh["rows"]), sh["frac"], sh["share"], "`" + rc["argv"] + "`", "`" + B2["ckpt"][r][:12] + "`"])
    return table(hdr, rows)


def t_gate():
    hdr = ["run", "W", "trainer seed", "probe batch launch rows", "launch probe contribution", "head_o grad diff", "swap probe grad diff"]
    rows = []
    for r, s, p, nm in RUNORDER:
        g = B2["gate"][(("launchintentrep" if p == "rep" else "launchintent"), s[1])]
        rows.append([r, g["W"], g["seed"], "%s / %s" % (g["pbl"], g["pbn"]), g["contrib"], g["grad"], g["swap"]])
    return table(hdr, rows)


def t_jobs():
    hdr = ["run", "train: launch minute", "train: ZZJOB-END", "T0 eval: launch minute", "T0 eval: ZZJOB-END"]
    rows = []
    for r, s, p, nm in RUNORDER:
        t = B2["jobs"][(nm, "t")]
        e = B2["jobs"][(nm, "e")]
        rows.append([r + " (" + nm + ")", t[0], t[1], e[0], e[1]])
    return table(hdr, rows)


def t_walls():
    hdr = ["run", "causal rollout (s)", "logged rollout (s)", "fresh-T0 eval (s)"]
    rows = []
    for r, s, p, nm in RUNORDER:
        tag = "LIREP" if p == "rep" else "LI"
        rows.append([r, WALL[(tag, s[1], "causal")][0], WALL[(tag, s[1], "logged")][0], WALL[(tag, s[1], "t0")][0]])
    return table(hdr, rows)


def sum_walls():
    return sum(int(v[0]) for v in WALL.values())


def mins(a, b):
    ha, ma = map(int, a.split(":")[:2])
    hb, mb = map(int, b.split(":")[:2])
    return (hb * 60 + mb) - (ha * 60 + ma)


def first_last():
    stamps = re.findall(r"^10-28 (\d\d:\d\d) ", BOX2, flags=re.M)
    return stamps[0], stamps[-1]


FIRST, LAST = first_last()
SPAN_MIN = mins(FIRST, LAST)


def t_notrun():
    hdr = ["family", "what the JSON names", "status", "reason"]
    rows = [
        ["LONGITUDINAL", "the T0 LONGITUDINAL rows and distance keeping at T0", "NOT RUN", "a build item the reader did not carry (`not_built.t0_four_families`)"],
        ["LATERAL", "the T0 LATERAL rows", "NOT RUN", "same"],
        ["TACTICAL", "the T0 TACTICAL rows", "NOT RUN", "same"],
        ["STRATEGIC", "not named in the JSON; PREREG section 8 puts the box-2 families at T0 (`orc_families.py`, `v8f_paired_families.py`)", "NOT RUN", "same; the family is not paused for v8F"],
    ]
    return table(hdr, rows)


# ----------------------------------------------------------- git metadata
def gitmeta(c):
    if ARGS.root:
        return {}
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", "-s", "--format=%H|%cI|%s", c], capture_output=True)
    h, t, s = r.stdout.decode("utf-8").strip().split("|", 2)
    return {"full": h, "time": t[11:19], "subject": s}


GM = {c: gitmeta(c) for c in ["77234276", "d8c9f7bf", "e28516e8", "d7dc9f11", "b09ac89f"]}


def gtime(c):
    return GM[c].get("time", "??:??:??")


def file_sha(path):
    return sha(get(path))


def t_prov():
    hdr = ["role", "repo path (under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-28-launch-intent-read/`)", "sha256"]
    base = RAW + "/"
    items = [("read of record (reader output)", "read_A1/LAUNCH_INTENT_READ.json"),
             ("read of record, console", "read_A1/read_A1.out"),
             ("read of record, launcher", "read_A1/run_read_A1.sh"),
             ("T1-kin chain's own read, REFUSED (launch-time reader)", "lit1_logs/launch_intent_read_REFUSED.out"),
             ("T1-kin chain log (relaunch)", "lit1_logs/chain.out"),
             ("T1-kin chain log (stopped launch)", "lit1_logs/chain.out.prev_1028151251"),
             ("T1-kin chain launch log (both launches)", "lit1_logs/lit1_chain.launch.out"),
             ("manifest the reader ran on", "runs.json"),
             ("box-2 chain log", "box2/chain_v8f_launchintent.out"),
             ("box-2 pins", "box2/PINS.txt"),
             ("K-BUILD record", "box2/K-BUILD.txt"),
             ("launch commit", "LAUNCH_COMMIT.txt"),
             ("chain-file pins", "PINS_BOX2.txt")]
    rows = []
    for role, p in items:
        rows.append([role, "`" + p + "`", "`" + file_sha(base + p) + "`"])
    rows.append(["pin-fill note", "`../2026-10-28-launch-intent-chain/PINS_FILLED.txt`", "`" + file_sha(CHAIN + "/PINS_FILLED.txt") + "`"])
    rows.append(["PREREG as amended (read by the reader of record)", "`Project Steering/PREREG_V8F_LAUNCH_INTENT.md` at `e28516e8` and `d7dc9f11`", "`" + sha(PREREG_BYTES) + "`"])
    for r, s, p, nm in RUNORDER:
        ev = "eval_M2b-%s_s%s.json" % ("launchintentrep" if p == "rep" else "launchintent", s[1])
        for fn in ["summary.json", "argv.json", "launch_gate_token.json", "SHA256SUMS.txt", ev]:
            path = "box2/%s/%s" % (nm, fn)
            rows.append(["run record %s" % r, "`" + path + "`", "`" + file_sha(base + path) + "`"])
    for r, s, p, nm in RUNORDER:
        rows.append(["checkpoint %s (not staged; E: only)" % r, "`E:/Projects/TanitAD-artifacts/v8f-launch-intent/%s/ckpt_last.pt`" % nm, "`" + B2["ckpt"][r] + "`"])
    return table(hdr, rows)


def rd_refused_failed():
    m = re.search(r"failed controls:\n((?:  - .*\n?)+)", REFUSED)
    return [x.strip()[2:] for x in m.group(1).strip().splitlines()]


def controls_text():
    return "%d of %d" % (sum(1 for v in D["controls_PASS"].values() if v is True), len(D["controls_PASS"]))


# ------------------------------------------------------------- extra values
READ_OUT = text(RAW + "/read_A1/read_A1.out")
PINS_FILLED = text(CHAIN + "/PINS_FILLED.txt")
PREREG_BYTES = get(PREREG_PATH)

CRAWL_CELLS = [(s, p, r) for s in SEEDS for p in PAIRS for r in CRAWL]
n_crawl_cells = len(CRAWL_CELLS)
n_crawl_better = sum(1 for s, p, r in CRAWL_CELLS if C("xy", s, p, r)["separated_better"])
CLEARING = [(s, p, r) for s, p, r in CRAWL_CELLS if clears(s, p, r)]
n_crawl_clear = len(CLEARING)
n_xp_worse = sum(1 for s, p, r in CRAWL_CELLS if C("xp", s, p, r)["separated_worse"])
n_fam_cells = sum(len(blk["rows"]) * 3 for s in SEEDS for p in PAIRS for blk in D["four_families"]["A"][s][p].values())
L2 = "L2 curvature MAE (GT>=5.2)"


def flips_text(s):
    return ", ".join(flips(s))


def stamp_of(pattern, text_, which=0):
    ms = [m for m in re.finditer(r"^10-28 (\d\d:\d\d:\d\d) " + pattern, text_, flags=re.M)]
    return ms[which].group(1)


a0_first_start = stamp_of(r"start LI_s0_m10_causal_heads", CHAIN_OUT)
a0_last_end = stamp_of(r"end fresh-T0 eval M2b-launchintent_s0-fresh", CHAIN_OUT)
relaunch_start = stamp_of(r"=== lit1_chain start", CHAIN_OUT)
prev_last_stamp = re.findall(r"^10-28 (\d\d:\d\d:\d\d) ", CHAIN_PREV, flags=re.M)[-1]
reader_start = stamp_of(r"== reader:", CHAIN_OUT)
refused_time = re.search(r"^10-28 (\d\d:\d\d:\d\d) reader rc=2: LAUNCH-INTENT READ REFUSED", CHAIN_OUT, flags=re.M).group(1)
refused_msg = re.search(r"  - K-BUILD \['(.*)'\]", REFUSED).group(1)
ram_msg = re.search(r"waiting, a guard is LOW: (LOW RAM available \d+ MB < \d+ MB)", CHAIN_PREV).group(1)
read_first_stamp = re.search(r"^(\d\d:\d\d:\d\d) \[LI_s0_m10_causal_heads\] ticks loaded", READ_OUT, flags=re.M).group(1)
read_done = re.search(r"^(ZZLAUNCH-INTENT-READ-DONE HARMS -> .*)$", READ_OUT, flags=re.M).group(1)
read_rc = re.search(r"^(ZZREAD-A1-RC \d+)$", READ_OUT, flags=re.M).group(1)
zzdone_full = re.search(r"^ZZV8FMINI-DONE .*$", BOX2, flags=re.M).group(0)
zzdone = "B is not queued" if "B is not queued" in zzdone_full else None
ZZEND = re.findall(r"^===== ZZJOB-END 28\.10\.2026 (\d\d:\d\d:\d\d)", BOX2, flags=re.M)
gate_launch = re.findall(r"^10-28 (\d\d:\d\d) launched pid=\d+ rc=0 log=D:\\tanitad\\logs\\v8flaunchintent\d+_gate_", BOX2, flags=re.M)
gate_span = "%s to %s" % (gate_launch[0], ZZEND[4][:5])

m = re.search(r"\*\*arm B\*\* \(W_B 0\.726175, target 1/4\) ([\d.]+) / ([\d.]+) / ([\d.]+) / ([\d.]+)", PREREG)
est_B = " / ".join(m.groups())
EST_A = re.findall(r"^\| (?:0|1|10|11) \| [\d,]+ \| [\d,]+ \| [\d.]+ \| ([\d.]+) \|$", PREREG, flags=re.M)
est_A = ", ".join(EST_A)
prereg_est_h = re.search(r"\*\*A \(4 runs: A_0, A_1, A_rep,0, A_rep,1\)\*\* \| \| \*\*≈ ([\d.]+) h\*\*", PREREG).group(1)
pins_filled_time = re.search(r"2026-10-28 (~\d\d:\d\d)", PINS_FILLED).group(1)
written_time = re.search(r"Written 2026-10-28 (~\d\d:\d\d)", PREREG).group(1)


# ------------------------------------------------- assertions behind the prose
def req(cond, msg):
    if not cond:
        raise SystemExit("BUILD ASSERTION FAILED: " + msg)


req(D["OUTCOME"]["OUTCOME"] == "HARMS" and D["OUTCOME"]["adopts"] is False, "outcome is HARMS, not adopted")
req(all(v is True for v in D["controls_PASS"].values()), "all controls PASS")
req(CLEARING == [("s0", "rep", L2)], "the only clearing crawl cell is L2 at s0 replicate: %r" % CLEARING)
req(n_xp_worse == n_crawl_cells, "all X-P10 crawl cells separated-worse")
req(all(C("yp", s, p, r)["separated_worse"] for s, p, r in CRAWL_CELLS), "all Y-P10 crawl cells separated-worse")
req(not any(clears(s, "main", r) for s in SEEDS for r in CRAWL), "no main-pair crawl cell clears")
req(all(sum(1 for r in CRAWL if clears(s, p, r)) < 4 for s in SEEDS for p in PAIRS), "no pair clears all four rows")
req(new_sets("s1", "main")[2] == ["speed MAE", "accel MAE", "cross-track MAE", "L1 lateral-accel MAE"], "s1 main (3) rows")
req(new_sets("s0", "rep")[2] == ["along-track MAE"], "s0 rep (3) rows")
req(new_sets("s0", "main")[2] == [] and new_sets("s1", "rep")[2] == [], "no (3) rows at s0 main, s1 rep")
req(all(not new_sets(s, p)[3] for s in SEEDS for p in PAIRS), "no row meets (1) and (3)")
req(all(len(ROWS[s]["pairs"][p]["domain"]) == 15 for s in SEEDS for p in PAIRS), "domain 15 rows")
req([ROWS[s]["pairs"][p]["guards"]["R-ADE"] for s in SEEDS for p in PAIRS] == [True, False, False, True], "R-ADE fails in s0 rep and s1 main only")
ade = [C("xy", s, p, "ADE6") for s in SEEDS for p in PAIRS]
req(all(c["separated"] for c in ade), "all four ADE6 cells separated")
req(ade[0]["separated_better"] and ade[1]["separated_worse"] and ade[2]["separated_worse"] and ade[3]["separated_better"], "ADE6 flips")
req(max(abs(c["delta"]) for c in ade) < F_HALF, "ADE6 points inside F/2")
req(all(ROWS[s]["pairs"][p]["flags"]["R-KEEP"] for s in SEEDS for p in PAIRS), "R-KEEP true in all pairs")
req("ADE6" in flips("s0") and "ADE6" in flips("s1") and "accel MAE" in flips("s1"), "flip rows")
req(all(ROWS[s]["pairs"][p]["guards"][g] for s in SEEDS for p in PAIRS for g in ["R-TLAT", "R-TLON", "R-T0", "R-NEW"]), "other guards held")
req(all(v["label"] is None for v in SLOW.values()), "no slow-draw label")
req(all(float(B2["gate"][k]["grad"]) > 0 for k in B2["gate"]) and len(B2["gate"]) == 4, "head_o grad diff non-zero")
req(["%.4f" % float(B2["share"][r]["share"]) for r, *_ in RUNORDER] == EST_A, "measured shares equal the ESTIMATED shares at four decimals: %r vs %r" % (["%.4f" % float(B2["share"][r]["share"]) for r, *_ in RUNORDER], EST_A))
req(len(re.findall(r"^10-28 \d\d:\d\d:\d\d start (?:LI|LIREP)_s\d_m10_(?:causal|logged)_heads \(attempt 1;", CHAIN_OUT, flags=re.M)) == 8, "eight rollouts, attempt 1")
req(len(re.findall(r"start (?:LI|LIREP)_", CHAIN_PREV)) == 0, "stopped launch has no rollout start line")
req("template proof 1:" in CHAIN_PREV and "template proof 1:" in CHAIN_OUT and "template proofs 2-4:" in CHAIN_PREV and "template proofs 2-4:" in CHAIN_OUT, "template proofs in both launches")
req(rd_refused_failed() == ["K-BUILD ['" + refused_msg + "']"], "REFUSED read failed K-BUILD alone")
req(len(D["controls_PASS"]) == 206, "206 controls")
req(len(WALL) == 12, "twelve chain steps")
req(len(ZZEND) == 13, "thirteen ZZJOB-END lines in the box-2 log (refresh + 4 gates + 4 trains + 4 evals): %d" % len(ZZEND))
if not ARGS.root:
    ls = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "ls-tree", "-r", "--name-only", COMMIT, "--", RAW], capture_output=True).stdout.decode("utf-8")
    req("tick" not in ls.lower(), "no tick arrays in the repo listing")
    req(len([x for x in ls.splitlines() if "/t1/t1v8f_" in x]) == 8, "eight rollout JSONs in t1/")


# ----------------------------------------------------------------- render
NS = dict(globals())
NS.update(dict(
    ncontrols=len(D["controls_PASS"]),
    wall_s=D["provenance"]["wall_s"],
    W_A="1.58926" if "`--launch-intent-w 1.58926`" in PREREG else None,
    W_B="0.726175" if "W_B = 0.726175" in PREREG else None,
    W_HI="4.17852" if "W_HI = 4.17852" in PREREG else None,
    tier1=D["tier"]["T1"], tier0=D["tier"]["T0"],
))
req(NS["W_A"] and NS["W_B"] and NS["W_HI"], "W values found in the PREREG")
req(all(B2["rec"][r]["W"] == NS["W_A"] for r, *_ in RUNORDER), "W in run records")

tmpl = open(__file__.replace("build_result.py", "RESULT_TEMPLATE.md"), encoding="utf-8").read()


def render(t):
    out = []
    i = 0
    while i < len(t):
        j = t.find("[[", i)
        if j < 0:
            out.append(t[i:])
            break
        out.append(t[i:j])
        k = j + 2
        depth = 0
        while True:
            ch = t[k]
            if ch == "[":
                depth += 1
            elif ch == "]":
                if depth == 0 and t[k + 1] == "]":
                    break
                depth -= 1
            k += 1
        expr = t[j + 2:k]
        try:
            val = eval(expr, NS)
        except Exception as e:
            raise SystemExit("template expression failed: %r: %s" % (expr, e))
        out.append(str(val))
        i = k + 2
    return "".join(out)


doc = render(tmpl)
doc = doc.replace("\r\n", "\n")
with open(ARGS.out, "w", encoding="utf-8", newline="\n") as f:
    f.write(doc)
print("wrote %s: %d bytes, %d lines, sha256 %s" % (ARGS.out, len(doc.encode("utf-8")), doc.count("\n"), sha(doc.encode("utf-8"))))
