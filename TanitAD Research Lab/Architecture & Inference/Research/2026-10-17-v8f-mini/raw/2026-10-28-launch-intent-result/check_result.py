#!/usr/bin/env python
"""check_result.py: re-find every number RESULT_V8F_LAUNCH_INTENT.md quotes.

Exits non-zero on any miss.  Independent of build_result.py: it parses the
finished markdown and looks each table cell, each prose number, each quoted
string and each hash prefix up in the banked files.

What it does
  1. loads the banked files read-only from the pinned commit
        git -C <mirror> -c safe.directory=* show d7dc9f11:<path>
     (or from --root DIR with the same repo-relative layout; commit-time checks
     are then skipped) and pins the JSON of record by sha256;
  2. recomputes the decision (R-ADE, R-MOVE, R-CRAWL, R-NEW, R-NOBASE, R-T0,
     R-KEEP per pair and per seed, then the outcome word) from the JSON's cells,
     floors and points, and compares it with the reader's flags;
  3. asserts the four-family blocks equal the read_a cells, cell for cell;
  4. parses every table of the document and compares each cell with the
     source: JSON cells, floors, points, the box-2 chain log, the T1-kin chain
     logs, file hashes;
  5. looks up every quoted string, every number in the prose, every time and
     every hash prefix;
  6. checks the writing rules: banned words, LF only, one open slot, tier
     discipline for the phrases 'closed loop' and 'driving performance'.

What it does NOT do: recompute the cells from the per-window dumps or re-run
the bootstrap (the dumps are a single copy on F:).

Usage:  python check_result.py [--doc PATH] [--root DIR] [--require-slot-filled]
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--doc", default=os.path.join(HERE, "RESULT_V8F_LAUNCH_INTENT.md"))
ap.add_argument("--root", default=None)
ap.add_argument("--require-slot-filled", action="store_true")
ap.add_argument("--mlaunch", default="E:/Projects/TanitAD-artifacts/dagland/launch_intent_mlaunch/MLAUNCH.json",
                help="the M-LAUNCH result that fills the slot (pinned by sha256; Master Mind addition 2026-10-28)")
ARGS = ap.parse_args()

MIRROR = "C:/Users/Admin/tanitad-push"
COMMIT = "d7dc9f11"
RESEARCH = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = RESEARCH + "/raw/2026-10-28-launch-intent-read"
CHAIN = RESEARCH + "/raw/2026-10-28-launch-intent-chain"
PREREG_PATH = "Project Steering/PREREG_V8F_LAUNCH_INTENT.md"
JSON_SHA_PIN = "e451e1a6ba44e1d30e97561456476dc4223cd916bd3be96163cec4b02844bf23"
MINUS, DASH = "\u2212", "\u2013"
F_HALF = 0.0604
BANNED = ["void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"]

FAILS = []
PASSES = []
STATS = {}


def ok(name, detail=""):
    PASSES.append(name)
    print("PASS %s%s" % (name, (": " + detail) if detail else ""))


def bad(name, detail):
    FAILS.append((name, detail))
    print("FAIL %s: %s" % (name, detail))


def check(cond, name, detail=""):
    if cond:
        ok(name, detail if cond is True else "")
    else:
        bad(name, detail)
    return bool(cond)


def git(*a):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "-c", "core.autocrlf=false"] + list(a), capture_output=True)
    return r


def get(path):
    if ARGS.root:
        with open(ARGS.root.rstrip("/") + "/" + path, "rb") as f:
            return f.read()
    r = git("show", COMMIT + ":" + path)
    if r.returncode != 0:
        raise SystemExit("git show failed for %s: %s" % (path, r.stderr.decode("utf-8", "replace")))
    return r.stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def text(path):
    return get(path).decode("utf-8").replace("\r\n", "\n")


# ------------------------------------------------------------------ sources
JSON_BYTES = get(RAW + "/read_A1/LAUNCH_INTENT_READ.json")
check(sha(JSON_BYTES) == JSON_SHA_PIN, "JSON of record equals its pinned sha256", sha(JSON_BYTES)[:16])
D = json.loads(JSON_BYTES.decode("utf-8"))
BOX2 = text(RAW + "/box2/chain_v8f_launchintent.out")
REFUSED = text(RAW + "/lit1_logs/launch_intent_read_REFUSED.out")
CHAIN_OUT = text(RAW + "/lit1_logs/chain.out")
CHAIN_PREV = text(RAW + "/lit1_logs/chain.out.prev_1028151251")
LAUNCH_OUT = text(RAW + "/lit1_logs/lit1_chain.launch.out")
READ_OUT = text(RAW + "/read_A1/read_A1.out")
RUN_SH = text(RAW + "/read_A1/run_read_A1.sh")
PINS_BOX2 = text(RAW + "/box2/PINS.txt")
PINS_CHAIN = text(RAW + "/PINS_BOX2.txt")
PINS_FILLED = text(CHAIN + "/PINS_FILLED.txt")
LAUNCH_COMMIT = text(RAW + "/LAUNCH_COMMIT.txt")
PREREG_BYTES = get(PREREG_PATH)
PREREG = PREREG_BYTES.decode("utf-8").replace("\r\n", "\n")
DAGORC_RESULT = text(RESEARCH + "/RESULT_V8F_DAG_ORC.md")
ML_BYTES = open(ARGS.mlaunch, "rb").read()
ML_SHA_PIN = "f5ebb4b209a5072e8b8171d4103d026a682cf2af84c14c891c1169a7e11aaaa2"
check(sha(ML_BYTES) == ML_SHA_PIN, "MLAUNCH.json equals its pinned sha256", sha(ML_BYTES)[:16])
ML = json.loads(ML_BYTES.decode("utf-8"))
DOC = open(ARGS.doc, "rb").read()
DOCT = DOC.decode("utf-8")

RA = D["read_a"]
ROWS = D["rows"]["A"]
FL = D["floors"]
ALL_ROWS = list(RA["A_CTRL"]["s0"].keys())
CRAWL = ["yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"]
FAMROWS = {f: list(D["four_families"]["A"]["s0"]["main"][f]["rows"].keys()) for f in ["LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"]}
SEEDS = ["s0", "s1"]
PAIRS = ["main", "rep"]
XY = {"main": "A_CTRL", "rep": "Arep_REP"}
XP = {"main": "A_P10", "rep": "Arep_P10"}
YP = {"main": "CTRL_P10", "rep": "REP_P10"}
PAIRWORD = {"main": "main", "rep": "replicate"}


def cellj(kind, s, p, r):
    return RA[{"xy": XY, "xp": XP, "yp": YP}[kind][p]][s][r]


def f4(x):
    x = 0.0 if x == 0 else x
    return "%.4f" % x


def s4(x):
    x = 0.0 if x == 0 else x
    return "%+.4f" % x


def flag_of(c):
    sep = c["lo"] > 0 or c["hi"] < 0
    if not sep:
        return DASH
    better = (c["delta"] < 0) if c["direction"] == "lower" else (c["delta"] > 0)
    return "B" if better else "W"


# ------------------------------------------------- 1. JSON self-consistency
bad_flags = [(k, s, r) for k, v in RA.items() for s, rows in v.items() for r, c in rows.items()
             if (flag_of(c) == "B") != c["separated_better"] or (flag_of(c) == "W") != c["separated_worse"] or (flag_of(c) != DASH) != c["separated"]]
check(not bad_flags, "every read_a cell's separated / better / worse flags follow from its bounds and direction", "%d cells" % sum(len(r) for v in RA.values() for r in v.values()))
mp = {"A - CTRL": "A_CTRL", "A - P10": "A_P10", "CTRL - P10": "CTRL_P10", "Arep - REP": "Arep_REP", "Arep - P10": "Arep_P10", "REP - P10": "REP_P10"}
nf = 0
mism = []
for s in SEEDS:
    for p in PAIRS:
        for fam, blk in D["four_families"]["A"][s][p].items():
            for r, cs in blk["rows"].items():
                for lab, c in cs.items():
                    nf += 1
                    ref = RA[mp[lab]][s][r]
                    if (c["delta"], c["lo"], c["hi"], c["separated"]) != (ref["delta"], ref["lo"], ref["hi"], ref["separated"]):
                        mism.append((s, p, fam, r, lab))
            if "closed loop" in blk["tier"] and "Never 'closed loop'" not in blk["tier"]:
                mism.append(("tier", s, p, fam))
check(not mism and nf == 204, "four_families leaves equal read_a cell for cell", "%d cells" % nf)
STATS["family_cells"] = nf
check(all(v is True for v in D["controls_PASS"].values()) and len(D["controls_PASS"]) == 206, "all 206 reader controls PASS in the JSON")

# ------------------------------------------------- 2. recompute the decision
BAR = FL["bar"]
check(all(abs(BAR[r] - 2 * max(FL["F"][r], FL["RF0"][r])) < 1e-9 for r in ALL_ROWS), "bar = 2 x max(F, RF0) on all 19 rows")
check(abs(FL["F"]["ADE6"] / 2 - 0.06035) < 1e-9 and F_HALF == round(FL["F"]["ADE6"] / 2 + 1e-12, 4), "F/2 = 0.06035 reads 0.0604 at four decimals")


def pair_calc(s, p):
    c_ade = cellj("xy", s, p, "ADE6")
    dom = [r for r in ALL_ROWS if not cellj("yp", s, p, r)["separated_worse"]]
    w = [r for r in dom if cellj("xy", s, p, r)["separated_worse"]]
    w3 = [r for r in w if cellj("xy", s, p, r)["delta"] >= BAR[r]]
    w13 = [r for r in w3 if cellj("xp", s, p, r)["separated_worse"]]
    move = {r: (cellj("xy", s, p, r)["separated_better"] and -cellj("xy", s, p, r)["delta"] >= BAR[r]) for r in CRAWL}
    crawl = {r: (not cellj("xp", s, p, r)["separated_worse"]) for r in CRAWL}
    pj = ROWS[s]["pairs"][p]
    return dict(ade=c_ade, ade_ok=not c_ade["separated_worse"], dom=dom, w=w, w3=w3, fires=w13, move=move, crawl=crawl,
                t0=pj["T0_point"], keep=pj["KEEP_point"], t0_ok=pj["T0_point"] <= F_HALF, keep_ok=pj["KEEP_point"] <= F_HALF,
                tlat=pj["guards"]["R-TLAT"], tlon=pj["guards"]["R-TLON"])


PC = {(s, p): pair_calc(s, p) for s in SEEDS for p in PAIRS}
diffs = []
for s in SEEDS:
    for p in PAIRS:
        c = PC[(s, p)]
        pj = ROWS[s]["pairs"][p]
        exp = {
            ("guards", "R-ADE"): c["ade_ok"], ("guards", "R-T0"): c["t0_ok"], ("guards", "R-NEW"): not c["fires"],
            ("flags", "R-MOVE"): all(c["move"].values()), ("flags", "R-CRAWL"): all(c["crawl"].values()),
            ("flags", "R-KEEP"): c["keep_ok"], ("flags", "R-NOBASE"): not c["fires"],
        }
        for (blk, k), v in exp.items():
            if pj[blk][k] != v:
                diffs.append((s, p, k, pj[blk][k], v))
        if sorted(pj["domain"]) != sorted(c["dom"]):
            diffs.append((s, p, "domain"))
        if [r for r, v in pj["R-MOVE_rows"].items() if v] != [r for r, v in c["move"].items() if v]:
            diffs.append((s, p, "R-MOVE_rows"))
        if {r for r, v in pj["R-CRAWL_rows"].items() if v} != {r for r, v in c["crawl"].items() if v}:
            diffs.append((s, p, "R-CRAWL_rows"))
check(not diffs, "per-pair guards and flags recomputed from the cells equal the reader's", "16 values x 4 pairs; diffs=%r" % diffs)

SEEDPRED = {}
for s in SEEDS:
    sp = {
        "R-ADE": all(PC[(s, p)]["ade_ok"] for p in PAIRS), "R-T0": all(PC[(s, p)]["t0_ok"] for p in PAIRS),
        "R-NEW": all(not PC[(s, p)]["fires"] for p in PAIRS), "R-TLAT": all(PC[(s, p)]["tlat"] for p in PAIRS),
        "R-TLON": all(PC[(s, p)]["tlon"] for p in PAIRS), "R-MOVE": all(all(PC[(s, p)]["move"].values()) for p in PAIRS),
        "R-CRAWL": all(all(PC[(s, p)]["crawl"].values()) for p in PAIRS), "R-KEEP": all(PC[(s, p)]["keep_ok"] for p in PAIRS),
        "R-NOBASE": all(not PC[(s, p)]["fires"] for p in PAIRS),
    }
    SEEDPRED[s] = sp
    for g in ["R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW"]:
        check(ROWS[s]["guards"][g] == sp[g], "seed predicate %s %s recomputed" % (s, g))
    for g in ["R-MOVE", "R-CRAWL", "R-KEEP", "R-NOBASE"]:
        check(ROWS[s]["flags"][g] == sp[g], "seed predicate %s %s recomputed" % (s, g))


def outcome_word():
    if not all(D["controls_PASS"].values()):
        return "REFUSED"
    guards = ["R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW"]
    if any(not SEEDPRED[s][g] for s in SEEDS for g in guards):
        return "HARMS"
    nmove = sum(SEEDPRED[s]["R-MOVE"] for s in SEEDS)
    crawl_all = all(SEEDPRED[s]["R-CRAWL"] for s in SEEDS)
    adopt = all(SEEDPRED[s]["R-KEEP"] and SEEDPRED[s]["R-NOBASE"] for s in SEEDS)
    if nmove == 2:
        if crawl_all:
            return "FIXED / ADOPTED" if adopt else "FIXED / NOT ADOPTED"
        return "PARTIAL / ADOPTED" if adopt else "PARTIAL / NOT ADOPTED"
    return "PARTIAL / ONE SEED" if nmove == 1 else "NO EFFECT"


WORD = outcome_word()
check(WORD == D["OUTCOME"]["OUTCOME"] == "HARMS" and D["OUTCOME"]["adopts"] is False, "the outcome word derived from the recomputed predicates is HARMS, not adopted", WORD)
check(D["OUTCOME"]["committed_next"] in DOCT, "the committed next of the JSON is quoted verbatim")

# ------------------------------------------------- 3. the document: writing rules
check(b"\r" not in DOC, "LF only")
check(DOC.endswith(b"\n") and not DOC.endswith(b"\n\n"), "ends with one newline")
hits = [(w, m.start()) for w in BANNED for m in re.finditer(r"\b" + re.escape(w) + r"\b", DOCT, flags=re.I)]
check(not hits, "no banned word (%s)" % ", ".join(BANNED), repr(hits))
nslot = DOCT.count("@@MLAUNCH@@")
check(nslot == (0 if ARGS.require_slot_filled else 1), "exactly one open slot token (none with --require-slot-filled)", "count=%d" % nslot)
if ARGS.require_slot_filled:
    check(ML["M_LAUNCH"] == "not moved" and "**M-LAUNCH: NOT MOVED**" in DOCT, "the filled slot states the M-LAUNCH label of MLAUNCH.json")
    for k in ("main_s0", "main_s1", "replicate_s0", "replicate_s1"):
        c = ML["primary"]["cells"][k]["x_minus_y"]
        fm = lambda v: ("+" if v >= 0 else MINUS) + "%.4f" % abs(v)
        check("%s [%s, %s]" % (fm(c["delta"]), fm(c["lo"]), fm(c["hi"])) in DOCT, "M-LAUNCH cell %s is quoted from MLAUNCH.json" % k)
if nslot == 1:
    print("OPEN @@MLAUNCH@@ is present: the M-LAUNCH result is not yet inserted (expected until it lands)")
    if ARGS.require_slot_filled:
        bad("slot filled", "@@MLAUNCH@@ still present")
for m in re.finditer(r"closed loop", DOCT, flags=re.I):
    pre = DOCT[max(0, m.start() - 14):m.start()]
    if not re.search(r"(Never '|not |is not |Not )$", pre):
        bad("tier discipline", "'closed loop' without a negation near char %d: %r" % (m.start(), pre))
for m in re.finditer(r"driving performance", DOCT, flags=re.I):
    pre = DOCT[max(0, m.start() - 14):m.start()]
    if not re.search(r"(Never '|not |is not )$", pre):
        bad("tier discipline", "'driving performance' without a negation near char %d: %r" % (m.start(), pre))
ok("tier discipline: every 'closed loop' / 'driving performance' is negated") if not [f for f in FAILS if f[0] == "tier discipline"] else None
check(DOCT.count("T1-kin") >= 10 and "T0" in DOCT, "tier stamps present")
check("paired episode-cluster bootstrap" in DOCT and "B = 2,000" in DOCT and "141 episodes" in DOCT, "estimator stamp present")
est = D["estimator"]
check("B=2000" in est and "141 episodes / 2,399 windows" in est and "2,399" in DOCT, "estimator numbers (B, 141, 2,399) are in the JSON estimator string")
check("[95 % CI]" in DAGORC_RESULT or "95 % CI" in DAGORC_RESULT, "the 95 % interval convention is in RESULT_V8F_DAG_ORC.md (INHERITED)")

# sections: evidence class words
secs = re.split(r"^## ", DOCT, flags=re.M)[1:]
for sec in secs:
    title = sec.split("\n", 1)[0]
    has_num = re.search(r"\d\.\d{2,}", sec) is not None
    has_cls = re.search(r"MEASURED|INHERITED|ESTIMATED|HYPOTHESIS|PUBLISHED", sec) is not None
    check((not has_num) or has_cls, "section carries an evidence class: %s" % title[:50])

# ------------------------------------------------------- 4. tables: parse
lines = DOCT.split("\n")


def split_row(ln):
    return [c.strip().replace("**", "") for c in ln.strip()[1:-1].split(" | ")]


TABLES = []
cur_head = ""
i = 0
while i < len(lines):
    if lines[i].startswith("#"):
        cur_head = lines[i]
    if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|(-+\|)+$", lines[i + 1]):
        hdr = split_row(lines[i])
        j = i + 2
        rows = []
        while j < len(lines) and lines[j].startswith("|"):
            rows.append(split_row(lines[j]))
            j += 1
        TABLES.append(dict(line=i + 1, hdr=hdr, rows=rows, head=cur_head, i0=i, i1=j))
        i = j
        continue
    i += 1

CELL = re.compile(r"^([+-]\d+\.\d{4}) \[([+-]\d+\.\d{4}), ([+-]\d+\.\d{4})\] (B|W|" + DASH + r")$")


def cell_ok(txt, c):
    m = CELL.match(txt)
    if not m:
        return False
    return (m.group(1) == s4(c["delta"]) and m.group(2) == s4(c["lo"]) and m.group(3) == s4(c["hi"]) and m.group(4) == flag_of(c))


def cell_expect(c):
    return "%s [%s, %s] %s" % (s4(c["delta"]), s4(c["lo"]), s4(c["hi"]), flag_of(c))


KINDS = {}


def pair_of(txt):
    return "main" if txt.startswith("main") else "rep"


def verify_rows(name, table, expected_rows):
    """expected_rows: list of lists of strings, compared cell for cell"""
    KINDS[name] = KINDS.get(name, 0) + 1
    got = table["rows"]
    errs = []
    if len(got) != len(expected_rows):
        errs.append("row count %d != %d" % (len(got), len(expected_rows)))
    for k, (g, e) in enumerate(zip(got, expected_rows)):
        if len(g) != len(e):
            errs.append("row %d width %d != %d" % (k, len(g), len(e)))
            continue
        for c, (a, b) in enumerate(zip(g, e)):
            if a != b:
                errs.append("row %d col %d (%s): doc %r != source %r" % (k, c, table["hdr"][c][:30], a, b))
    check(not errs, "table '%s' (line %d) equals the source" % (name, table["line"]), "; ".join(errs[:4]))
    STATS["cells_" + name] = sum(len(r) for r in expected_rows)


def pct(x):
    return "%.1f %%" % (100.0 * x)


def pw(s, p):
    return "%s (%s %s %s)" % (PAIRWORD[p], {"main": "A", "rep": "A_rep"}[p], MINUS, {"main": "CTRL", "rep": "REP"}[p])


def exp_outcome():
    nctl = len(D["controls_PASS"])
    out = [["controls (reader, %d)" % nctl] + ["PASS"] * 6]
    for g in ["R-TLAT", "R-TLON"]:
        r = ["%s (T head macro-F1, ticks >= 1, X %s Y not separated-worse)" % (g, MINUS)]
        for s in SEEDS:
            for p in PAIRS:
                r.append("held" if PC[(s, p)]["tlat" if g == "R-TLAT" else "tlon"] else "FAILS")
            r.append("held" if SEEDPRED[s][g] else "FAILS")
        out.append(r)
    r = ["R-ADE (T1-kin ADE6, X %s Y, not separated-worse)" % MINUS]
    for s in SEEDS:
        for p in PAIRS:
            c = PC[(s, p)]["ade"]
            r.append("%s: %s %s" % ("held" if PC[(s, p)]["ade_ok"] else "FAILS", s4(c["delta"]), flag_of(c)))
        r.append("held" if SEEDPRED[s]["R-ADE"] else "FAILS")
    out.append(r)
    r = ["R-T0 (T0 own-code ADE6, X %s Y, point <= +0.0604 m)" % MINUS]
    for s in SEEDS:
        for p in PAIRS:
            r.append("%s: %s" % ("held" if PC[(s, p)]["t0_ok"] else "FAILS", s4(PC[(s, p)]["t0"])))
        r.append("held" if SEEDPRED[s]["R-T0"] else "FAILS")
    out.append(r)
    r = ["R-NEW (no row of D fires)"]
    for s in SEEDS:
        for p in PAIRS:
            r.append("%s: %d of %d rows" % ("held" if not PC[(s, p)]["fires"] else "FAILS", len(PC[(s, p)]["fires"]), len(PC[(s, p)]["dom"])))
        r.append("held" if SEEDPRED[s]["R-NEW"] else "FAILS")
    out.append(r)
    r = ["R-MOVE (every crawl row clears its bar)"]
    for s in SEEDS:
        for p in PAIRS:
            k = sum(PC[(s, p)]["move"].values())
            r.append("%s: %d of 4 rows clear" % ("true" if k == 4 else "false", k))
        r.append("true" if SEEDPRED[s]["R-MOVE"] else "false")
    out.append(r)
    r = ["R-CRAWL (every crawl row of X %s P10 not separated-worse)" % MINUS]
    for s in SEEDS:
        for p in PAIRS:
            k = sum(PC[(s, p)]["crawl"].values())
            r.append("%s: %d of 4 rows" % ("true" if k == 4 else "false", k))
        r.append("true" if SEEDPRED[s]["R-CRAWL"] else "false")
    out.append(r)
    r = ["R-KEEP (T1-kin ADE6, X %s Y, point <= +0.0604 m)" % MINUS]
    for s in SEEDS:
        for p in PAIRS:
            r.append("%s: %s" % ("true" if PC[(s, p)]["keep_ok"] else "false", s4(PC[(s, p)]["keep"])))
        r.append("true" if SEEDPRED[s]["R-KEEP"] else "false")
    out.append(r)
    r = ["R-NOBASE"]
    for s in SEEDS:
        for p in PAIRS:
            r.append("%s: %d of %d rows violate" % ("true" if not PC[(s, p)]["fires"] else "false", len(PC[(s, p)]["fires"]), len(PC[(s, p)]["dom"])))
        r.append("true" if SEEDPRED[s]["R-NOBASE"] else "false")
    out.append(r)
    return out


def exp_floors():
    out = []
    for r in ALL_ROWS:
        c = RA["A_CTRL"]["s0"][r]
        out.append([r, c["direction"], f4(FL["F"][r]), f4(FL["RF0"][r]), f4(BAR[r]), f4(FL["RF"][r]), "%.2f" % (FL["RF"][r] / BAR[r])])
    return out


def exp_ade():
    out = []
    for s in SEEDS:
        for p in PAIRS:
            c = cellj("xy", s, p, "ADE6")
            out.append([s, pw(s, p), "ADE6", cell_expect(c), f4(c["mean_arm"]), f4(c["mean_ref"]), cell_expect(cellj("xp", s, p, "ADE6")),
                        cell_expect(cellj("yp", s, p, "ADE6")), c["n"]])
    return out


def exp_move():
    out = []
    for s in SEEDS:
        for p in PAIRS:
            for r in CRAWL:
                c = cellj("xy", s, p, r)
                imp = -c["delta"]
                out.append([s, pw(s, p), r, cell_expect(c), s4(imp), f4(BAR[r]), pct(imp / BAR[r]), "true" if PC[(s, p)]["move"][r] else "false"])
    return out


def exp_crawl():
    out = []
    for s in SEEDS:
        for p in PAIRS:
            for r in CRAWL:
                x, y = cellj("xp", s, p, r), cellj("yp", s, p, r)
                out.append([s, pw(s, p), r, cell_expect(x), cell_expect(y), "true" if PC[(s, p)]["crawl"][r] else "false", pct((y["delta"] - x["delta"]) / y["delta"])])
    return out


def exp_new():
    out = []
    for s in SEEDS:
        for p in PAIRS:
            c = PC[(s, p)]
            out.append([s, pw(s, p), "%s %s P10 not separated-worse on D" % ({"main": "CTRL", "rep": "REP"}[p], MINUS), str(len(c["dom"])),
                        ", ".join(c["w"]) or "none", ", ".join(c["w3"]) or "none", ", ".join(c["fires"]) or "none"])
    return out


def exp_t0keep():
    out = []
    for s in SEEDS:
        for p in PAIRS:
            c = PC[(s, p)]
            out.append([s, pw(s, p), s4(c["t0"]), "held" if c["t0_ok"] else "FAILS", s4(c["keep"]), "true" if c["keep_ok"] else "false"])
    return out


def exp_famcount():
    out = []
    for fam, rows in FAMROWS.items():
        r = ["%s (%d)" % (fam, len(rows))]
        for s in SEEDS:
            for p in PAIRS:
                cs = [cellj("xy", s, p, x) for x in rows]
                b = sum(1 for c in cs if flag_of(c) == "B")
                w = sum(1 for c in cs if flag_of(c) == "W")
                r.append("%d / %d / %d" % (b, w, len(cs) - b - w))
        out.append(r)
    return out


def exp_family(fam):
    out = []
    for r in FAMROWS[fam]:
        row = [r, RA["A_CTRL"]["s0"][r]["direction"]]
        for s in SEEDS:
            for p in PAIRS:
                row.append(cell_expect(cellj("xy", s, p, r)))
        row.append(" ; ".join(cellj("xy", s, p, r)["n"] for s in SEEDS for p in PAIRS))
        out.append(row)
    return out


def exp_p10(s):
    out = []
    for r in ALL_ROWS:
        out.append([r, RA["A_CTRL"]["s0"][r]["direction"], cell_expect(cellj("xp", s, "main", r)), cell_expect(cellj("yp", s, "main", r)),
                    cell_expect(cellj("xp", s, "rep", r)), cell_expect(cellj("yp", s, "rep", r))])
    return out


def flips(s):
    out = []
    for r in ALL_ROWS:
        a, b = flag_of(cellj("xy", s, "main", r)), flag_of(cellj("xy", s, "rep", r))
        if {a, b} == {"B", "W"}:
            out.append(r)
    return out


TS = {("s0", "main"): 0, ("s0", "rep"): 10, ("s1", "main"): 1, ("s1", "rep"): 11}


def exp_flip():
    return [[s, "%d / %d" % (TS[(s, "main")], TS[(s, "rep")]), cell_expect(cellj("xy", s, "main", "ADE6")), cell_expect(cellj("xy", s, "rep", "ADE6"))] for s in SEEDS]


def exp_crawlcells():
    out = []
    for s in SEEDS:
        for p in PAIRS:
            for r in CRAWL:
                c = cellj("xy", s, p, r)
                if flag_of(c) == "B":
                    out.append([s, pw(s, p), r, cell_expect(c), pct(-c["delta"] / BAR[r]), "true" if PC[(s, p)]["move"][r] else "false"])
    return out


SLOW = D["beside"]["A"]["slow_draw"]
CELLS5 = ["lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms"]
SLOWKEYS = [("main:A-CTRL:s0", "s0 main (A %s CTRL)" % MINUS), ("rep:Arep-REP:s0", "s0 replicate (A_rep %s REP)" % MINUS),
            ("main:A-CTRL:s1", "s1 main (A %s CTRL)" % MINUS), ("rep:Arep-REP:s1", "s1 replicate (A_rep %s REP)" % MINUS)]


def exp_slow():
    out = []
    for k, lab in SLOWKEYS:
        v = SLOW[k]
        out.append([lab] + ["%s [%s, %s]" % (s4(v["cells"][c]["mean"]), s4(v["cells"][c]["lo"]), s4(v["cells"][c]["hi"])) for c in CELLS5] + ["none" if v["label"] is None else v["label"]])
    return out


# box-2 log, own parse
RUNS = [("A_0", "s0", "main", "M2b_s0_launchintent", "launchintent"), ("A_1", "s1", "main", "M2b_s1_launchintent", "launchintent"),
        ("A_rep,0", "s0", "rep", "M2b_s0_launchintentrep", "launchintentrep"), ("A_rep,1", "s1", "rep", "M2b_s1_launchintentrep", "launchintentrep")]
B2REC, B2SH, B2GATE, B2CK = {}, {}, {}, []
for ln in BOX2.split("\n"):
    m = re.match(r"LAUNCH-INTENT run record: PASS \| (A_[\w,]+) \| \w+ s(\d) \| trainer seed (\d+) \| argv sha16 (\w+) \| steps (\d+) \| launch_intent_w ([\d.]+)", ln)
    if m:
        B2REC[m.group(1)] = dict(seed=m.group(3), argv=m.group(4), steps=m.group(5), W=m.group(6))
    m = re.match(r"LAUNCH-INTENT K-SHARE: PASS \| (A_[\w,]+) \| launch (\d+) / (\d+) = ([\d.]+) \(K-MASK: f ([\d.]+) \+- ([\d.]+)\) \| weight share ([\d.]+) \(must lie in 1/3 \+- ([\d.]+)\) \| rows (\d+) \(== eligible\)", ln)
    if m:
        B2SH[m.group(1)] = dict(n=m.group(2), rows=m.group(3), frac=m.group(4), share=m.group(7), rows2=m.group(9), f=m.group(5), band=m.group(8))
    m = re.match(r"LAUNCH-INTENT gate: PASS \| (\w+) s(\d) \| launch W ([\d.]+) \| trainer seed (\d+) \| .*? \| swap probe grad diff ([\d.]+) \| launch probe contribution ([\d.]+) head_o grad diff ([\d.]+) \| probe batch launch (\d+) / (\d+)", ln)
    if m:
        B2GATE[(m.group(1), m.group(2))] = dict(W=m.group(3), seed=m.group(4), swap=m.group(5), contrib=m.group(6), grad=m.group(7), pbl=m.group(8), pbn=m.group(9))
    m = re.match(r"(\w{64}) \*ckpt_last\.pt", ln)
    if m:
        B2CK.append(m.group(1))
B2CKPT = dict(zip([r[0] for r in RUNS], B2CK))
check(len(B2REC) == 4 and len(B2SH) == 4 and len(B2GATE) == 4 and len(B2CK) == 4, "box-2 log: four run records, four K-SHARE, four gates, four checkpoints")
check(all(B2SH[r]["rows"] == B2SH[r]["rows2"] and abs(float(B2SH[r]["frac"]) - 0.1619) <= 0.02 and abs(float(B2SH[r]["share"]) - 1 / 3.0) <= 0.02 for r in B2SH), "K-SHARE and K-MASK values lie in their bands; rows equal eligible")
# recompute the weight share from the counts and the flag
for r, *_ in RUNS:
    n, rows, W = int(B2SH[r]["n"]), int(B2SH[r]["rows"]), float(B2REC[r]["W"])
    sh = (1 + W) * n / ((1 + W) * n + (rows - n))
    check(abs(sh - float(B2SH[r]["share"])) < 5e-7, "weight share of %s recomputed from launch rows, rows and W" % r, "%.6f" % sh)


def exp_share():
    out = []
    for r, s, p, nm, tag in RUNS:
        out.append(["%s (%s)" % (r, nm), B2REC[r]["seed"], "%s / %s" % (B2SH[r]["n"], B2SH[r]["rows"]), B2SH[r]["frac"], B2SH[r]["share"], "`" + B2REC[r]["argv"] + "`", "`" + B2CKPT[r][:12] + "`"])
    return out


def exp_gate():
    out = []
    for r, s, p, nm, tag in RUNS:
        g = B2GATE[(tag, s[1])]
        out.append([r, g["W"], g["seed"], "%s / %s" % (g["pbl"], g["pbn"]), g["contrib"], g["grad"], g["swap"]])
    return out


JOBS = {}
bl = BOX2.split("\n")
for k, ln in enumerate(bl):
    m = re.match(r"10-28 (\d\d:\d\d) launched pid=\d+ rc=0 log=D:\\tanitad\\logs\\v8flaunchintent\d+_(t|e)_(M2b_s\d_launchintent(?:rep)?)\.log", ln)
    if m:
        for j in range(k + 1, len(bl)):
            m2 = re.match(r"===== ZZJOB-END 28\.10\.2026 (\d\d:\d\d:\d\d)", bl[j])
            if m2:
                JOBS[(m.group(3), m.group(2))] = (m.group(1), m2.group(1))
                break


def exp_jobs():
    return [["%s (%s)" % (r, nm), JOBS[(nm, "t")][0], JOBS[(nm, "t")][1], JOBS[(nm, "e")][0], JOBS[(nm, "e")][1]] for r, s, p, nm, tag in RUNS]


WALL = {}
for ln in CHAIN_OUT.split("\n"):
    m = re.match(r"10-28 \d\d:\d\d:\d\d end (LI|LIREP)_s(\d)_m10_(causal|logged)_heads rc=0 wall=(\d+)s", ln)
    if m:
        WALL[(m.group(1), m.group(2), m.group(3))] = m.group(4)
    m = re.match(r"10-28 \d\d:\d\d:\d\d end fresh-T0 eval M2b-launchintent(rep)?_s(\d)-fresh rc=0 wall=(\d+)s", ln)
    if m:
        WALL[("LIREP" if m.group(1) else "LI", m.group(2), "t0")] = m.group(3)


def exp_walls():
    out = []
    for r, s, p, nm, tag in RUNS:
        t = "LIREP" if p == "rep" else "LI"
        out.append([r, WALL[(t, s[1], "causal")], WALL[(t, s[1], "logged")], WALL[(t, s[1], "t0")]])
    return out


def exp_t0notrun():
    return None


# ---------------------------------------------------------- 4b. dispatch
SIGHDR = re.compile(r"^(A|A_rep|CTRL|REP) " + MINUS + r" (CTRL|REP|P10) (s[01])$")
for t in TABLES:
    h = t["hdr"]
    if h[0].startswith("predicate (PREREG section 10)"):
        verify_rows("outcome", t, exp_outcome())
    elif h[:3] == ["row", "better", "F"]:
        verify_rows("floors", t, exp_floors())
    elif h[0] == "seed" and len(h) > 4 and h[4] == "mean X":
        verify_rows("ade", t, exp_ade())
    elif h[0] == "seed" and len(h) > 4 and h[2] == "crawl row" and h[4] == "improvement":
        verify_rows("move", t, exp_move())
    elif h[0] == "seed" and len(h) > 3 and h[2] == "crawl row" and h[3] == "X " + MINUS + " P10":
        verify_rows("crawl", t, exp_crawl())
    elif h[0] == "seed" and len(h) > 2 and h[2] == "control Y":
        verify_rows("new", t, exp_new())
    elif h[0] == "seed" and len(h) > 2 and h[2].startswith("T0 own-code"):
        verify_rows("t0keep", t, exp_t0keep())
    elif h[0].startswith("family (rows)"):
        verify_rows("famcount", t, exp_famcount())
    elif h[0] == "row" and h[1] == "better" and len(h) == 7 and h[2] == "A " + MINUS + " CTRL s0":
        fam = [f for f, rows in FAMROWS.items() if [r[0] for r in t["rows"]] == rows]
        if len(fam) == 1:
            verify_rows("family:" + fam[0], t, exp_family(fam[0]))
        else:
            bad("family table", "rows do not match one family at line %d" % t["line"])
    elif h[0] == "row" and h[1] == "better" and len(h) == 6 and SIGHDR.match(h[2]):
        s = SIGHDR.match(h[2]).group(3)
        verify_rows("p10:" + s, t, exp_p10(s))
    elif h[0] == "seed" and len(h) > 1 and h[1].startswith("trainer seed (main pair"):
        verify_rows("flip", t, exp_flip())
    elif h[0] == "seed" and len(h) > 4 and h[2] == "crawl row" and h[4] == "improvement / bar":
        verify_rows("crawlcells", t, exp_crawlcells())
    elif h[0].startswith("arm and pair (T0"):
        verify_rows("slow", t, exp_slow())
    elif h[0] == "run" and len(h) > 2 and h[1] == "trainer seed":
        verify_rows("share", t, exp_share())
    elif h[0] == "run" and len(h) > 1 and h[1] == "W":
        verify_rows("gate", t, exp_gate())
    elif h[0] == "run" and len(h) > 1 and h[1] == "train: launch minute":
        verify_rows("jobs", t, exp_jobs())
    elif h[0] == "run" and len(h) > 1 and h[1].startswith("causal rollout"):
        verify_rows("walls", t, exp_walls())
    elif h[0] == "family" and h[1] == "what the JSON names":
        KINDS["t0notrun"] = KINDS.get("t0notrun", 0) + 1
        check(all(r[2] == "NOT RUN" for r in t["rows"]) and [r[0] for r in t["rows"]] == list(FAMROWS.keys()), "T0 table: all four families NOT RUN, per family")
        check(D["not_built"]["t0_four_families"].endswith("NOT RUN, a build item, never a silent omission"), "JSON says the T0 four families are NOT RUN")
    elif h[0] == "role":
        KINDS["prov"] = KINDS.get("prov", 0) + 1
    else:
        KINDS["text:" + h[0][:20]] = KINDS.get("text:" + h[0][:20], 0) + 1

want = {"outcome": 1, "floors": 1, "ade": 1, "move": 1, "crawl": 1, "new": 1, "t0keep": 1, "famcount": 1, "family:LONGITUDINAL": 1, "family:LATERAL": 1,
        "family:TACTICAL": 1, "family:STRATEGIC": 1, "p10:s0": 1, "p10:s1": 1, "flip": 1, "crawlcells": 1, "slow": 1, "share": 1, "gate": 1, "jobs": 1,
        "walls": 1, "t0notrun": 1, "prov": 1}
miss = {k: (KINDS.get(k, 0), v) for k, v in want.items() if KINDS.get(k, 0) != v}
check(not miss, "every expected table kind is present exactly once", repr(miss))
print("INFO unrecognised (prose) tables: %r" % {k: v for k, v in KINDS.items() if k.startswith("text:")})

# provenance table
prov = [t for t in TABLES if t["hdr"][0] == "role"][0]
bad_prov = []
nprov = 0
PROV_SHAS = set()
for role, path, hx in prov["rows"]:
    path = path.strip("`")
    hx = hx.strip("`")
    if path.startswith("E:/"):
        rn = [r for r, s, p, nm, tag in RUNS if "/" + nm + "/" in path][0]
        if B2CKPT[rn] != hx:
            bad_prov.append(path)
        nprov += 1
        continue
    if path.startswith("Project Steering/"):
        real = sha(PREREG_BYTES)
    elif path.startswith("../2026-10-28-launch-intent-chain/"):
        real = sha(get(CHAIN + "/" + path.split("/")[-1]))
    else:
        real = sha(get(RAW + "/" + path))
    nprov += 1
    PROV_SHAS.add(real)
    if real != hx:
        bad_prov.append(path)
check(not bad_prov and nprov == len(prov["rows"]), "provenance table: every sha256 re-measured", "%d rows; bad=%r" % (nprov, bad_prov))
STATS["prov_rows"] = nprov

# --------------------------------------------- 5. prose: cells, quotes, hex
# strip tables out of the prose, keep everything else
tbl_lines = set()
for t in TABLES:
    for k in range(t["i0"], t["i1"]):
        tbl_lines.add(k)
PROSE_LINES = [ln for k, ln in enumerate(lines) if k not in tbl_lines]
PROSE = "\n".join(PROSE_LINES).replace("**", "")

CELL_ANY = re.compile(r"([+-]\d+\.\d{4}) \[([+-]\d+\.\d{4}), ([+-]\d+\.\d{4})\](?: (B|W|" + DASH + r"))?")
leaf = set()
for k, v in RA.items():
    for s, rows in v.items():
        for r, c in rows.items():
            leaf.add((s4(c["delta"]), s4(c["lo"]), s4(c["hi"]), flag_of(c)))
slowleaf = set()
for v in SLOW.values():
    for c in v["cells"].values():
        slowleaf.add((s4(c["mean"]), s4(c["lo"]), s4(c["hi"])))
nprose = 0
badc = []
for m in CELL_ANY.finditer(PROSE):
    nprose += 1
    d_, lo_, hi_, fl_ = m.groups()
    if fl_ is None:
        if (d_, lo_, hi_) not in slowleaf and not any((d_, lo_, hi_) == x[:3] for x in leaf):
            badc.append(m.group(0))
    elif (d_, lo_, hi_, fl_) not in leaf:
        badc.append(m.group(0))
check(not badc, "every cell quoted in the prose is a cell of the JSON, with its flag", "%d prose cells; bad=%r" % (nprose, badc))
STATS["prose_cells"] = nprose

# quotes: *"..."*
corpus_json = []


def collect(o):
    if isinstance(o, str):
        corpus_json.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            collect(v)
    elif isinstance(o, list):
        for v in o:
            collect(v)


collect(D)
CORPUS = "\n".join(corpus_json) + "\n" + BOX2 + "\n" + REFUSED + "\n" + CHAIN_OUT + "\n" + CHAIN_PREV + "\n" + READ_OUT + "\n" + PREREG + "\n" + PINS_FILLED
quotes = re.findall(r'\*"(.+?)"\*', PROSE.replace("\n", " ")) + re.findall(r'\*"(.+?)"\*', DOCT.replace("**", ""))
badq = []
for q in set(quotes):
    qq = q.rstrip(".")
    if qq not in CORPUS and q not in CORPUS and qq.replace("\u2019", "'") not in CORPUS:
        badq.append(q[:90])
check(not badq, "every quoted string is found verbatim in the JSON, a log or the PREREG", "%d quotes; bad=%r" % (len(set(quotes)), badq))
STATS["quotes"] = len(set(quotes))

# hex prefixes in code spans
known_hex = set(re.findall(r"[0-9a-f]{8,64}", CORPUS + PINS_BOX2 + PINS_CHAIN + RUN_SH + LAUNCH_COMMIT + LAUNCH_OUT + json.dumps(D)))
known_hex |= {sha(JSON_BYTES), sha(PREREG_BYTES), sha(ML_BYTES)} | set(B2CKPT.values()) | PROV_SHAS
for path in ["read_A1/read_A1.out", "read_A1/run_read_A1.sh", "runs.json", "lit1_logs/chain.out", "box2/chain_v8f_launchintent.out"]:
    known_hex.add(sha(get(RAW + "/" + path)))
if not ARGS.root:
    for c in ["77234276", "e28516e8", "d8c9f7bf", "d7dc9f11", "b09ac89f", "aeb54468", "d7208328"]:
        r = git("rev-parse", "--verify", c + "^{commit}")
        if r.returncode == 0:
            known_hex.add(r.stdout.decode().strip())
    # reader / PREREG blobs at the promotion and amendment commits
    known_hex.add(sha(git("show", "77234276:" + PREREG_PATH).stdout))
    known_hex.add(sha(git("show", "77234276:" + RESEARCH + "/raw/2026-10-28-launch-intent-v2/read/launch_intent_read.py").stdout))
    known_hex.add(sha(git("show", "e28516e8:" + RESEARCH + "/raw/2026-10-28-launch-intent-amend1/read/launch_intent_read.py").stdout))
codes = re.findall(r"`([0-9a-f]{7,64})…?`", DOCT)
badh = [c for c in codes if not any(h.startswith(c) for h in known_hex)]
check(not badh, "every hash or commit prefix in a code span is a prefix of a hash found in a source", "%d prefixes; bad=%r" % (len(codes), badh))
STATS["hex_prefixes"] = len(codes)
if not ARGS.root:
    p77 = sha(git("show", "77234276:" + PREREG_PATH).stdout)
    r77 = sha(git("show", "77234276:" + RESEARCH + "/raw/2026-10-28-launch-intent-v2/read/launch_intent_read.py").stdout)
    check(p77.startswith("469bac1f") and r77.startswith("5e950dcf") and "469bac1f3918b31b875e7c06b637980f6acb194a69ded3294c180aec95b1f65a" in PINS_BOX2 and "5e950dcff2993653dffbe9b4f01ea17eb4d7b1721b7c41a669e0dc43c559fc35" in PINS_BOX2,
          "PREREG and reader blobs at 77234276 hash to the pins the chain carries (469bac1f, 5e950dcf)")
    ra = sha(git("show", "e28516e8:" + RESEARCH + "/raw/2026-10-28-launch-intent-amend1/read/launch_intent_read.py").stdout)
    check(ra.startswith("fb3d2d7d") and D["provenance"]["reader_sha256"] == ra and sha(PREREG_BYTES).startswith("6ae79f2f") and sha(PREREG_BYTES) in RUN_SH,
          "reader of record (fb3d2d7d) and amended PREREG (6ae79f2f) equal the pins in the read launcher and the JSON")
    check(sha(get(RAW + "/runs.json")) == D["provenance"]["manifest_sha256"], "manifest sha256 equals the JSON's manifest_sha256 (98d2cd3d)")

# ------------------------------------------------------- 6. times
def commit_times():
    out = {}
    if ARGS.root:
        return out
    for c in ["77234276", "e28516e8", "d8c9f7bf", "d7dc9f11"]:
        r = git("show", "-s", "--format=%cI", c)
        out[c] = r.stdout.decode().strip()[11:19]
    return out


CT = commit_times()
time_pool = set(re.findall(r"\b(\d\d:\d\d:\d\d)\b", CORPUS + BOX2 + LAUNCH_OUT)) | set(CT.values())
time_pool_m = {t[:5] for t in time_pool} | set(re.findall(r"\b(\d\d:\d\d)\b", CORPUS + BOX2 + LAUNCH_OUT))
BRIEF_TIMES = {"15:11": "the stop of the T1-kin chain on its RAM guard (brief)", "13:50": "the pin fill (brief)"}
bt = []
for m in re.finditer(r"(~?)(\d\d):(\d\d)(?::(\d\d))?", re.sub(r"`[^`]*`", "", DOCT)):
    full = m.group(0).lstrip("~")
    hhmm = full[:5]
    if m.group(4):
        if full not in time_pool:
            bt.append(full)
    elif hhmm in BRIEF_TIMES:
        pass
    elif hhmm not in time_pool_m:
        bt.append(full)
check(not bt, "every time in the document is in a log, a commit time or a named brief constant", "bad=%r" % bt)
if CT:
    check(CT["77234276"] < "13:50:00" < CT["d8c9f7bf"], "the ~13:50 pin fill lies between the promotion commit and the launch commit", "%s < 13:50 < %s" % (CT["77234276"], CT["d8c9f7bf"]))
    first_read = re.search(r"^(\d\d:\d\d:\d\d) \[", READ_OUT, flags=re.M).group(1)
    check(CT["e28516e8"] < first_read < CT["d7dc9f11"], "e28516e8 landed before the read of record's first stamp and d7dc9f11 after it", "%s < %s < %s" % (CT["e28516e8"], first_read, CT["d7dc9f11"]))
prev_last = re.findall(r"^10-28 (\d\d:\d\d:\d\d) ", CHAIN_PREV, flags=re.M)[-1]
relaunch = re.search(r"^10-28 (\d\d:\d\d:\d\d) === lit1_chain start", CHAIN_OUT, flags=re.M).group(1)
check(prev_last < "15:11" < relaunch, "the 15:11 stop lies between the stopped launch's last line and the relaunch", "%s < 15:11 < %s" % (prev_last, relaunch))
check(len(re.findall(r"start (?:LI|LIREP)_", CHAIN_PREV)) == 0 and "LOW RAM available 3040 MB < 4096 MB" in CHAIN_PREV, "stopped launch: waiting on the RAM guard, no rollout started")
check(len(re.findall(r"^10-28 \d\d:\d\d:\d\d start (?:LI|LIREP)_s\d_m10_(?:causal|logged)_heads \(attempt 1;", CHAIN_OUT, flags=re.M)) == 8, "relaunch: eight rollouts, all attempt 1")
failed_ctrl = re.search(r"failed controls:\n((?:  - .*\n?)+)", REFUSED).group(1).strip().splitlines()
check(len(failed_ctrl) == 1 and failed_ctrl[0].strip().startswith("- K-BUILD"), "the T1-kin chain's own read REFUSED on K-BUILD alone")
check("ZZREAD-A1-RC 0" in READ_OUT and "HARMS" in READ_OUT, "the read of record ended rc 0 with HARMS")
check(len(re.findall(r"^===== ZZJOB-END", BOX2, flags=re.M)) == 13, "box-2 log: thirteen jobs ended (refresh, 4 gates, 4 trains, 4 evals)")
stamps = re.findall(r"^10-28 (\d\d:\d\d) ", BOX2, flags=re.M)
span = (int(stamps[-1][:2]) * 60 + int(stamps[-1][3:])) - (int(stamps[0][:2]) * 60 + int(stamps[0][3:]))
check(span == 174 and "%.1f" % (span / 60.0) == "2.9", "box-2 span: first stamp %s, last %s = %d min = %.1f h" % (stamps[0], stamps[-1], span, span / 60.0))
wsum = sum(int(v) for v in WALL.values())
check(len(WALL) == 12 and wsum == 1756, "twelve chain steps sum to %d s" % wsum)
m = re.search(r"\*\*A \(4 runs: A_0, A_1, A_rep,0, A_rep,1\)\*\* \| \| \*\*≈ ([\d.]+) h\*\*", PREREG)
check(m and m.group(1) == "2.8", "PREREG section 13 estimates A at 2.8 h")
check(re.search(r"Written 2026-10-28 ~15:10", PREREG) is not None, "Amendment 1 says it was written ~15:10")
check(re.search(r"2026-10-28 ~14:10", PINS_FILLED) is not None, "PINS_FILLED.txt says ~14:10")
check("B is not queued" in BOX2, "box-2 closing line says B is not queued")
check(D["not_built"]["M-LAUNCH"] in DOCT and D["not_built"]["anti_echo_block"] in DOCT and D["not_built"]["M2_and_L2_launch_subset"] in DOCT, "the three NOT BUILT strings are quoted from the JSON")

# ------------------------------------------------- 7. prose phrases (derived)
n_cr = [(s, p, r) for s in SEEDS for p in PAIRS for r in CRAWL]
n_better = sum(1 for s, p, r in n_cr if flag_of(cellj("xy", s, p, r)) == "B")
n_clear = sum(1 for s, p, r in n_cr if PC[(s, p)]["move"][r])
clear_cells = [(s, p, r) for s, p, r in n_cr if PC[(s, p)]["move"][r]]
L2 = "L2 curvature MAE (GT>=5.2)"
check(clear_cells == [("s0", "rep", L2)], "the only crawl cell that clears its bar is L2 at s0, replicate pair", repr(clear_cells))
ade = {(s, p): cellj("xy", s, p, "ADE6") for s in SEEDS for p in PAIRS}
slow_abs = [abs(SLOW[k]["cells"][c]["mean"]) for k, _ in SLOWKEYS for c in CELLS5]
slow_sep = sum(1 for k, _ in SLOWKEYS for c in CELLS5 if SLOW[k]["cells"][c]["lo"] > 0 or SLOW[k]["cells"][c]["hi"] < 0)
PHRASES = [
    "all **%d reader controls PASS**, wall %s s" % (len(D["controls_PASS"]), D["provenance"]["wall_s"]),
    "R-ADE fails at s0 in the replicate pair (ADE6, in m: %s) and at s1 in the main pair (ADE6, in m: %s)" % (cell_expect(ade[("s0", "rep")]), cell_expect(ade[("s1", "main")])),
    "%d of the %d crawl cells are separated-better and %d clears its bar: L2 curvature in the replicate pair at s0, %s against a bar of %s. Its main-pair partner reads %s." % (
        n_better, len(n_cr), n_clear, cell_expect(cellj("xy", "s0", "rep", L2)), f4(BAR[L2]), cell_expect(cellj("xy", "s0", "main", L2))),
    "All %d X %s P10 crawl cells" % (sum(1 for s, p, r in n_cr if cellj("xp", s, p, r)["separated_worse"]), MINUS),
    "In the main pair at s1, four rows meet (3): speed MAE %s, accel MAE %s, cross-track MAE %s and L1 lateral-accel MAE %s." % tuple(
        cell_expect(cellj("xy", "s1", "main", r)) for r in ["speed MAE", "accel MAE", "cross-track MAE", "L1 lateral-accel MAE"]),
    "In the replicate pair at s0, along-track MAE meets (3): %s." % cell_expect(cellj("xy", "s0", "rep", "along-track MAE")),
    "at s0, %s; at s1, %s." % (", ".join(flips("s0")), ", ".join(flips("s1"))),
    "main pair %s, replicate pair %s." % (cell_expect(cellj("xy", "s1", "main", "accel MAE")), cell_expect(cellj("xy", "s1", "rep", "accel MAE"))),
    "The four ADE6 points lie between %s and %s m, all inside ±%s m" % (f4(min(abs(c["delta"]) for c in ade.values())), f4(max(abs(c["delta"]) for c in ade.values())), f4(F_HALF)),
    "The largest shift in magnitude over the 20 cells is %s m/s, and %d of the 20 cell intervals exclude 0." % (f4(max(slow_abs)), slow_sep),
    "(%d min, %.1f h; MEASURED" % (span, span / 60.0),
    "The twelve steps sum to %d s." % wsum,
    "First stamped line %s, last %s: %d min" % (stamps[0], stamps[-1], span),
    "(%d cells, asserted" % nf,
    "R-KEEP reads the point against +%s m and holds in all four pairs" % f4(F_HALF),
    "F is %s m on ADE6" % f4(FL["F"]["ADE6"]),
    "Its main-pair partner is %s, not separated." % cell_expect(cellj("xy", "s0", "main", L2)),
]
for ph in PHRASES:
    check(ph in DOCT.replace("**", "") or ph in DOCT, "phrase re-derived from the sources is in the document", ph[:100])

# every pair/seed R-ADE statement in prose matches
check(flag_of(ade[("s0", "main")]) == "B" and flag_of(ade[("s0", "rep")]) == "W" and flag_of(ade[("s1", "main")]) == "W" and flag_of(ade[("s1", "rep")]) == "B",
      "ADE6 cells flip sign between the pairs at each seed (B/W at s0, W/B at s1), all separated")
check(all(flag_of(cellj("xp", s, p, r)) == "W" and flag_of(cellj("yp", s, p, r)) == "W" for s, p, r in n_cr), "every crawl cell is separated-worse against P10 for X and for Y")
check(all(float(B2GATE[k]["grad"]) > 0 for k in B2GATE), "head_o grad diff is non-zero in all four gates")
est_a = re.findall(r"^\| (?:0|1|10|11) \| [\d,]+ \| [\d,]+ \| [\d.]+ \| ([\d.]+) \|$", PREREG, flags=re.M)
check(["%.4f" % float(B2SH[r]["share"]) for r, *_ in RUNS] == est_a, "measured shares equal the PREREG's ESTIMATED shares at four decimals", repr(est_a))
check(est_a and (", ".join(est_a) in DOCT), "the ESTIMATED shares are quoted from the PREREG")
mB = re.search(r"\*\*arm B\*\* \(W_B 0\.726175, target 1/4\) ([\d.]+) / ([\d.]+) / ([\d.]+) / ([\d.]+)", PREREG)
check(mB and " / ".join(mB.groups()) in DOCT, "B's ESTIMATED shares are quoted from the PREREG")
check(all(B2REC[r]["W"] == "1.58926" for r in B2REC) and "W 1.58926" in DOCT, "W 1.58926 in all four run records and in the document")
check("W_B = 0.726175" in PREREG and "W_HI = 4.17852" in PREREG, "W_B and W_HI are in the PREREG")

# ------------------------------------------------- 8. numeric coverage of prose
ev_vals = []  # (value, decimals) pool


def add_num(x):
    ev_vals.append(float(x))


def walk(o):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        add_num(o)
    elif isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)


walk(D)
walk(ML)
NUMRE = r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])"
txt_pool = CORPUS + "\n" + "\n".join(corpus_json) + "\n" + LAUNCH_OUT + "\n" + PINS_BOX2 + "\n" + PINS_CHAIN
for m in re.finditer(NUMRE, txt_pool):
    try:
        ev_vals.append(float(m.group(0).replace(",", "")))
    except ValueError:
        pass
EV = ev_vals
# values the checker has itself verified above (derived) -- each is asserted somewhere in this script
DERIVED = {
    "controls": len(D["controls_PASS"]), "family cells": nf, "crawl cells": len(n_cr), "separated-better crawl cells": n_better, "clears": n_clear,
    "domain": len(PC[("s0", "main")]["dom"]), "rows": len(ALL_ROWS), "span min": span, "span h": round(span / 60.0, 1), "wall sum": wsum,
    "max slow": max(slow_abs), "slow sep": slow_sep, "slow cells": len(slow_abs), "ade min": min(abs(c["delta"]) for c in ade.values()), "ade max": max(abs(c["delta"]) for c in ade.values()),
    "F/2": F_HALF, "95": 95.0,
}
EV += [float(v) for v in DERIVED.values()]
# row names and other labels contain digits; remove them before scanning
scan = PROSE
for r in sorted(ALL_ROWS, key=len, reverse=True):
    scan = scan.replace(r, " ")
scan = re.sub(r"`[^`]*`", " ", scan)  # code spans: paths, flags, hashes (hash prefixes checked above)
scan = CELL_ANY.sub(" ", scan)
scan = re.sub(r"\d{4}-\d\d-\d\d", " ", scan)
scan = re.sub(r"~?\d\d:\d\d(?::\d\d)?", " ", scan)
scan = re.sub(r"\b[Ss]ections? \d+(?:\.\d+)?(?:(?:,| and) \d+(?:\.\d+)?)*", " ", scan)
scan = re.sub(r"\b(?:item|Fact|disclosure|step) \d+", " ", scan)
scan = re.sub(r"^#+ \d+(?:\.\d+)*\.? ", " ", scan, flags=re.M)
scan = re.sub(r"^\s*\d+\. ", " ", scan, flags=re.M)
scan = re.sub(r"\[\d+ %", " ", scan)
BRIEF_NUMS = {2048.0: "2,048 MB RAM guard of the relaunch (brief, INHERITED)"}
unmatched = []
ncheck = 0
for m in re.finditer(NUMRE, scan):
    tok = m.group(0)
    s_ = tok.replace(",", "")
    try:
        v = float(s_)
    except ValueError:
        continue
    dec = len(s_.split(".")[1]) if "." in s_ else 0
    ncheck += 1
    if v in BRIEF_NUMS:
        continue
    if any(abs(round(abs(e), dec) - v) < 1e-9 for e in EV):
        continue
    unmatched.append(tok)
check(not unmatched, "every number in the prose is found in the JSON, a log, the PREREG or a verified derived value", "%d numbers checked; unmatched=%r" % (ncheck, unmatched))
STATS["prose_numbers"] = ncheck
check(re.search(r"2,048 MB", DOCT) is not None and "Master Mind's brief (INHERITED)" in DOCT, "the 2,048 MB guard and the 15:11 stop are labelled INHERITED from the brief")

# ------------------------------------------------------------------ summary
print()
print("SUMMARY cells verified by table:", {k: v for k, v in STATS.items() if k.startswith("cells_")})
print("SUMMARY prose cells %d | quotes %d | hash prefixes %d | prose numbers %d | provenance rows %d | family cells %d" % (
    STATS.get("prose_cells", 0), STATS.get("quotes", 0), STATS.get("hex_prefixes", 0), STATS.get("prose_numbers", 0), STATS.get("prov_rows", 0), STATS.get("family_cells", 0)))
print("SUMMARY %d checks passed, %d failed" % (len(PASSES), len(FAILS)))
if FAILS:
    for n, d in FAILS:
        print("  FAILED: %s: %s" % (n, d[:300]))
    sys.exit(1)
print("ALL CHECKS PASS")
sys.exit(0)
