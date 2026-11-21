#!/usr/bin/env python
"""check_result.py: re-find every number RESULT_V8F_SOUP.md quotes.

Exits non-zero on any miss.  Independent of build_result.py (it imports nothing
from it): it parses the finished markdown and looks each table cell, each prose
number, each quoted string and each hash prefix up in the banked files.

What it does
  1. loads the banked files read-only from the pinned commit
        git -C <mirror> -c safe.directory=* show 0c7b598e...:<path>
     and pins the JSON of record by sha256 (the F: copy must equal the blob);
  2. checks the JSON for self-consistency (flags vs deltas, four_families equal
     to the tables cell for cell);
  3. recomputes the decision (R-ADE, R-MOVE, R-CRAWL, R-NEW, R-NOBASE, R-T0,
     R-KEEP per pair; H, MV, AD, DM per seed; the word; the route) from the
     JSON's cells, floors and points and compares it with the reader's flags;
  4. parses every table of the document, re-renders it from the JSON and
     compares cell for cell (an unknown table FAILS);
  5. looks up every quoted string, every number in the prose, every time and
     every hash prefix, and the register rows the Rule Zero table quotes;
  6. checks the writing rules: banned words, LF only, tier discipline.

What it does NOT do: recompute the cells from the per-window dumps or re-run
the bootstrap (the dumps are a single copy on F:).

Usage:  python -I check_result.py [--doc PATH] [--json PATH]
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--doc", default=os.path.join(HERE, "RESULT_V8F_SOUP.md"))
ap.add_argument("--json", default="F:/TanitAD-scratch/soup_t1_out/read/SOUP_READ.json")
ARGS = ap.parse_args()

MIRROR = "C:/Users/Admin/tanitad-push"
TIP = "0c7b598edd16a5cbb422ec050a67760d4ba5b11e"
RES = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/"
JSON_PIN = "dca10c07f5fabda9ddc9ea2725f425f53f1462bec8488d58ebc92ecd8d33de6a"
PREREG_PIN = "830d9c9143850628188f7d8489566e92f2a9cbdf050689331f1eaa7d3c78993c"
MINUS, DASH = "\u2212", "\u2013"
F_HALF = 0.0604
FAILS, PASSES = [], []
STATS = Counter()


def ok(name, detail=""):
    PASSES.append(name)
    print("PASS  %s%s" % (name, ("  [" + detail + "]") if detail else ""))


def bad(name, detail):
    FAILS.append((name, detail))
    print("FAIL  %s  [%s]" % (name, detail[:400]))


def check(cond, name, detail=""):
    (ok if cond else bad)(name, detail)
    return bool(cond)


def git(*a):
    return subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*"] + list(a), capture_output=True)


def blob(path, rev=TIP):
    r = git("show", rev + ":" + path)
    if r.returncode != 0:
        raise SystemExit("cannot read %s at %s: %s" % (path, rev, r.stderr.decode()[:200]))
    return r.stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ sources
JPATH = RES + "2026-10-29-soup-read/out/read/SOUP_READ.json"
JB = blob(JPATH)
check(sha(JB) == JSON_PIN, "JSON blob at the tip equals the pinned sha256 (dca10c07)")
if os.path.exists(ARGS.json):
    check(sha(open(ARGS.json, "rb").read()) == JSON_PIN, "the F: copy of SOUP_READ.json equals the pinned sha256")
else:
    print("NOTE  F: copy not found; the repo blob is used")
D = json.loads(JB.decode("utf-8"))
T = D["tables"]
PREREG_B = blob("Project Steering/PREREG_V8F_SOUP.md")
PREREG = PREREG_B.decode("utf-8")
check(sha(PREREG_B) == PREREG_PIN, "PREREG at the tip equals the pin (830d9c91)")
check(sha(git("show", "e03b348:Project Steering/PREREG_V8F_SOUP.md").stdout) == PREREG_PIN, "PREREG at the promotion commit e03b348 equals the pin")
GOALS = blob("Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").replace("\u2212", "-")
CLAUDE = blob("CLAUDE.md").decode("utf-8")
CHAIN = blob(RES + "2026-10-29-soup-read/logs/chain.out").decode("utf-8", "replace")
READOUT = blob(RES + "2026-10-29-soup-read/logs/soup_read.out").decode("utf-8", "replace")
CMD_L = blob(RES + "2026-10-29-soup-read/as_launched/soup_chain.cmd")
CMD_S = blob(RES + "2026-10-29-soup-read/as_launched/soup_chain.cmd.as_shipped")
REG = {}
for ln in GOALS.split("\n"):
    m = re.match(r"\| \*\*([A-Z0-9-]+)\*\* ", ln)
    if m:
        REG[m.group(1)] = ln

DOCB = open(ARGS.doc, "rb").read()
DOCT = DOCB.decode("utf-8")
lines = DOCT.split("\n")


def f4(x):
    s = "%+.4f" % x
    return "+0.0000" if s == "-0.0000" else s


def u4(x):
    return "%.4f" % x


def flg(c):
    return "B" if c["separated_better"] else ("W" if c["separated_worse"] else DASH)


def cl(c):
    return "%s [%s, %s] %s" % (f4(c["delta"]), f4(c["lo"]), f4(c["hi"]), flg(c))


def nrend(n):
    return "unpaired vs CTRV" if n.startswith("ticks>=1") else n


LONG = ["speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG"]
LAT = ["heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)"]
TAC = ["TAC-LAT", "TAC-LON"]
STR = ["S-POS@30", "S-POS@50", "S-POS@80"]
M1 = "M1 crawl share"
CRAWL = ["yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", M1]
ALL19 = ["ADE6"] + LONG + LAT + [M1] + TAC + STR
CT = {("s0", "main"): ("SOUP_CTRL", "SOUP_P10", "CTRL_P10"), ("s1", "main"): ("SOUP_CTRL", "SOUP_P10", "CTRL_P10"),
      ("s0", "rep"): ("SOUPR_A", "SOUPR_P10", "A_P10"), ("s1", "rep"): ("SOUPR_A", "SOUPR_P10", "A_P10")}
COLS = [("s0", "main"), ("s0", "rep"), ("s1", "main"), ("s1", "rep")]


def xy(s, p, r):
    return T[CT[(s, p)][0]][s][r]


def xp(s, p, r):
    return T[CT[(s, p)][1]][s][r]


def yp(s, p, r):
    return T[CT[(s, p)][2]][s][r]


# --------------------------------------------- 1. JSON self-consistency
bad_cells = []
ncell = 0
for ct, bys in T.items():
    for s, rows in bys.items():
        for r, c in rows.items():
            ncell += 1
            lower = c["direction"] == "lower"
            exp_b = c["separated"] and ((c["delta"] < 0) if lower else (c["delta"] > 0))
            exp_w = c["separated"] and ((c["delta"] > 0) if lower else (c["delta"] < 0))
            sep_by_ci = (c["lo"] > 0) or (c["hi"] < 0)
            if not (c["separated_better"] == exp_b and c["separated_worse"] == exp_w and c["separated"] == sep_by_ci and c["lo"] <= c["delta"] <= c["hi"]):
                bad_cells.append((ct, s, r))
check(not bad_cells, "every JSON table cell: separated == CI excludes 0; better/worse flags agree with the delta sign and the direction; lo <= delta <= hi", "%d cells; bad=%r" % (ncell, bad_cells[:5]))
STATS["json_cells"] = ncell
check(len(D["controls_PASS"]) == 368 and all(v is True for v in D["controls_PASS"].values()), "controls_PASS: 368 entries, all true")
check(D["sg_errors"] == {} and D["beside"]["slow_draw_errors"] == {}, "sg_errors and slow_draw_errors are empty")
check(D["provenance"]["prereg_sha256"] == PREREG_PIN and D["provenance"]["reader_sha256"] == sha(blob(RES + "2026-10-29-soup/read/soup_read.py")) and D["provenance"]["li_reader_sha256"] == sha(blob(RES + "2026-10-29-soup/read/ref/launch_intent_read.py")),
      "provenance: prereg, reader and LAUNCH-INTENT reader sha256 equal the blobs at the tip")
check(D["provenance"]["prereg"] == "PREREG_V8F_SOUP.md (DRAFT, 2026-10-28)" and D["provenance"]["prereg_sha256"] == PREREG_PIN, "provenance.prereg label reads DRAFT while its sha256 is the promoted file's (disclosure 1)")

# four_families equal to the tables cell for cell
FFN = {"main": ("SOUP-CTRL", "SOUP-P10", "SOUP_CTRL", "SOUP_P10"), "rep": ("SOUPR-A", "SOUPR-P10", "SOUPR_A", "SOUPR_P10")}
nf, badf = 0, []
for key, fam in D["four_families"].items():
    p, s = key.split(":")
    for fname, blk in fam.items():
        for r, cc in blk["rows"].items():
            for k, tk in ((FFN[p][0], FFN[p][2]), (FFN[p][1], FFN[p][3])):
                t = T[tk][s][r]
                nf += 1
                if not (cc[k]["delta"] == t["delta"] and cc[k]["lo"] == t["lo"] and cc[k]["hi"] == t["hi"] and cc[k]["separated"] == t["separated"]):
                    badf.append((key, fname, r, k))
        if fname == "STRATEGIC":
            if blk.get("paused") is not False or blk.get("ruling") != "NOT paused for v8F (PI ruling 2026-10-17)":
                badf.append((key, "STRATEGIC paused flag"))
check(not badf, "four_families cells equal the tables cells, cell for cell", "%d cells; bad=%r" % (nf, badf[:5]))
STATS["family_cells"] = nf
check(D["strategic_family"] == "NOT PAUSED for v8F (PI ruling 2026-10-17)", "strategic_family string")

# ------------------------------------------------- 2. recompute the decision
BAR = D["floors"]["bar"]
RF0 = D["floors"]["RF0"]
FV = {r: (0.1207 if r == "ADE6" else 0.0) for r in ALL19}
check(all(abs(BAR[r] - 2 * max(FV[r], RF0[r])) < 1e-9 for r in ALL19), "bar_r == 2 x max(F_r, RF0_r) for all 19 rows (F = 0.1207 on ADE6 and 0 elsewhere)")
BETTER = {r: T["SOUP_CTRL"]["s0"][r]["direction"] for r in ALL19}


def worsen(r, d):
    return d if BETTER[r] == "lower" else -d


PC = {}
for (s, p) in COLS:
    ps = D["pairs"][s][p]
    dom = [r for r in ALL19 if not yp(s, p, r)["separated_worse"]]
    xyw = [r for r in dom if xy(s, p, r)["separated_worse"]]
    xyb = [r for r in xyw if worsen(r, xy(s, p, r)["delta"]) >= BAR[r] - 1e-12]
    fire = [r for r in xyb if xp(s, p, r)["separated_worse"]]
    move_rows = {r: bool(xy(s, p, r)["separated_better"] and -xy(s, p, r)["delta"] >= BAR[r] - 1e-12) for r in CRAWL}
    crawl_rows = {r: (not xp(s, p, r)["separated_worse"]) for r in CRAWL}
    keep_pt = ps["KEEP_point"]
    pc = dict(dom=dom, xyw=xyw, xyb=xyb, fire=fire, move_rows=move_rows, crawl_rows=crawl_rows,
              ADE=not xy(s, p, "ADE6")["separated_worse"], T0=ps["T0_point"] <= F_HALF, KEEP=keep_pt <= F_HALF,
              NEW=not fire, NOBASE=not fire, MOVE=all(move_rows.values()), CRAWL=all(crawl_rows.values()))
    PC[(s, p)] = pc
    check(sorted(ps["domain"]) == sorted(dom) and ps["crawl_set"] == CRAWL, "%s %s: domain D and crawl set equal the reader's" % (s, p))
    check(abs(keep_pt - xy(s, p, "ADE6")["delta"]) < 6e-5, "%s %s: KEEP_point equals the ADE6 X-Y delta (to rounding)" % (s, p))
    check(pc["ADE"] == ps["guards"]["R-ADE"] and pc["T0"] == ps["guards"]["R-T0"] and pc["NEW"] == ps["guards"]["R-NEW"] and
          pc["KEEP"] == ps["flags"]["R-KEEP"] and pc["NOBASE"] == ps["flags"]["R-NOBASE"] and pc["MOVE"] == ps["flags"]["R-MOVE"] and pc["CRAWL"] == ps["flags"]["R-CRAWL"],
          "%s %s: R-ADE, R-T0, R-NEW, R-KEEP, R-NOBASE, R-MOVE, R-CRAWL recomputed equal the reader's flags" % (s, p))
    check(pc["move_rows"] == ps["R-MOVE_rows"] and pc["crawl_rows"] == ps["R-CRAWL_rows"] and {r: (r in fire) for r in ps["domain"]} == ps["R-NEW_fires"] and {r: (r in fire) for r in ps["domain"]} == ps["R-NOBASE_violations"],
          "%s %s: per-row R-MOVE, R-CRAWL, R-NEW, R-NOBASE recomputed equal the reader's" % (s, p))

# R-TLAT / R-TLON: not derivable; the relation to the TAC rows is a stated fact
tlat = {k: D["pairs"][k[0]][k[1]]["guards"]["R-TLAT"] for k in COLS}
tlon = {k: D["pairs"][k[0]][k[1]]["guards"]["R-TLON"] for k in COLS}
tac_lat_ok = {k: not xy(k[0], k[1], "TAC-LAT")["separated_worse"] for k in COLS}
tac_lon_ok = {k: not xy(k[0], k[1], "TAC-LON")["separated_worse"] for k in COLS}
check(tlat == tac_lat_ok, "R-TLAT (reader) equals 'TAC-LAT X-Y not separated-worse' in all four pairs")
dis_lon = [k for k in COLS if tlon[k] != tac_lon_ok[k]]
check(dis_lon == [("s0", "rep")], "R-TLON (reader) differs from the TAC-LON cell in exactly one pair: the s0 replicate pair", repr(dis_lon))

GU = {k: dict(R_TLAT=tlat[k], R_TLON=tlon[k], R_ADE=PC[k]["ADE"], R_T0=PC[k]["T0"], R_NEW=PC[k]["NEW"]) for k in COLS}
Hs = {s: all(all(GU[(s, p)].values()) for p in ("main", "rep")) for s in ("s0", "s1")}
MVs = {s: all(PC[(s, p)]["MOVE"] for p in ("main", "rep")) for s in ("s0", "s1")}
ADs = {s: all(PC[(s, p)]["KEEP"] and PC[(s, p)]["NOBASE"] for p in ("main", "rep")) for s in ("s0", "s1")}


def dm_of(key, s):
    ra = D["read_a"][key]["per_seed"][s]
    return ra["D1"] and ra["D2"] and ra["D3"]


DMs = {s: dm_of("SOUP", s) and dm_of("SOUPR", s) for s in ("s0", "s1")}
# Read A recomputed from the X - P10 cells
RAKEY = {"SOUP": "SOUP_P10", "SOUPR": "SOUPR_P10", "SOUP4": "SOUP4_P10"}
FAMOF = {r: "LONGITUDINAL" for r in LONG}
FAMOF.update({r: "LATERAL" for r in LAT})
FAMOF.update({r: "TACTICAL" for r in TAC})
FAMOF.update({r: "STRATEGIC" for r in STR})
FAMOF["ADE6"] = "PRIMARY"
badra = []
for k, ct in RAKEY.items():
    for s in ("s0", "s1"):
        ra = D["read_a"][k]["per_seed"][s]
        a6 = T[ct][s]["ADE6"]
        d1 = bool(a6["separated_better"] and a6["delta"] <= -0.2415)
        worse = [r for r in ALL19 if T[ct][s][r]["separated_worse"]]
        d3 = {f: any(T[ct][s][r]["separated_better"] for r in ALL19 if FAMOF.get(r) == f) for f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC", "PRIMARY")}
        if not (d1 == ra["D1"] and sorted(worse) == sorted(ra["D2_worse_rows"]) and (not worse) == ra["D2"] and d3 == ra["D3_families"] and all(d3.values()) == ra["D3"]):
            badra.append((k, s))
check(not badra, "Read A (D1, D2, D2 rows, D3 families) recomputed from the X-P10 cells equals the reader's, for SOUP, SOUPR, SOUP4 at both seeds", repr(badra))
check(all(D["read_a"][k]["outcome"] == "MARGIN, NOT DOMINANT" for k in RAKEY), "Read A label is MARGIN, NOT DOMINANT for all three soups")
OUT = D["OUTCOME"]
check(OUT["per_seed"] == {s: dict(H=Hs[s], MV=MVs[s], AD=ADs[s], DM=DMs[s]) for s in ("s0", "s1")}, "per-seed booleans H, MV, AD, DM recomputed equal OUTCOME.per_seed", repr({s: (Hs[s], MVs[s], ADs[s], DMs[s]) for s in Hs}))
# route
SGK = ["SOUP:s0", "SOUP:s1", "SOUPR:s0", "SOUPR:s1", "SOUP4:s0", "SOUP4:s1"]
barrier4 = any(D["sg"][k]["separated"] and D["sg"][k]["delta"] > 0 for k in SGK[:4])
check(all(D["sg"][k]["barrier"] == (D["sg"][k]["separated"] and D["sg"][k]["delta"] > 0) for k in SGK), "sg barrier flags equal (separated and delta > 0) in all six cells")
route = "BARRIER" if barrier4 else "FLAT"
check(route == D["route"] == OUT["route"] == "FLAT", "route recomputed from the four decisive SG cells is FLAT, as the reader's")
# outcome word
if not all(D["controls_PASS"].values()):
    word = "REFUSED"
elif not (Hs["s0"] and Hs["s1"]):
    word = "HARMS"
elif DMs["s0"] and DMs["s1"]:
    word = "DOMINANT / ADOPTED" if (ADs["s0"] and ADs["s1"]) else "DOMINANT / NOT ADOPTED"
elif MVs["s0"] and MVs["s1"]:
    word = "MOVED / ADOPTED" if (ADs["s0"] and ADs["s1"]) else "MOVED / NOT ADOPTED"
elif MVs["s0"] != MVs["s1"]:
    word = "ONE SEED"
else:
    word = "NO EFFECT"
check(word == OUT["OUTCOME"] == "HARMS" and OUT["matching_rows_in_table_order"] == ["HARMS"] and OUT["adopts"] is False and OUT["next_soup"] == "SOUP-K",
      "outcome word recomputed with the PREREG section 10 precedence is HARMS, as the reader's; next soup SOUP-K")
check(OUT["committed_next"] == "the uniform two-draw soup is not adopted. NEXT-SOUP (\u00a711)", "committed_next string")

# SG arithmetic: soup mean - mean of members, from the table means
def sg_arith(key, s):
    def me(ct, which):
        return T[ct][s]["ADE6"][which]
    if key == "SOUP":
        return me("SOUP_CTRL", "mean_arm"), [me("SOUP_CTRL", "mean_ref"), me("SOUP_REP", "mean_ref")]
    if key == "SOUPR":
        return me("SOUPR_A", "mean_arm"), [me("SOUPR_A", "mean_ref"), me("SOUPR_Arep", "mean_ref")]
    return me("SOUP4_CTRL", "mean_arm"), [me("SOUP4_CTRL", "mean_ref"), me("REP_CTRL", "mean_arm"), me("A_P10", "mean_arm"), me("SOUPR_Arep", "mean_ref")]


for k in SGK:
    key, s = k.split(":")
    sm, mem = sg_arith(key, s)
    d = sm - sum(mem) / len(mem)
    check(abs(d - D["sg"][k]["delta"]) < 1.5e-4 and D["sg"][k]["n_dropped"] == 0 and D["sg"][k]["n_windows"] == 2399, "SG %s: delta equals soup mean minus mean of members' means (to rounding)" % k, "arith %.5f json %.4f" % (d, D["sg"][k]["delta"]))
# cross-contrast identities (arithmetic on the JSON)
IDENT_DIFFN = []
for s in ("s0", "s1"):
    for r in ["ADE6"] + CRAWL:
        lhs = T["SOUP_CTRL"][s][r]["delta"] - T["SOUP_REP"][s][r]["delta"]
        ns3 = {T[c][s][r]["n"] for c in ("SOUP_CTRL", "SOUP_REP", "REP_CTRL")}
        if len(ns3) == 1:
            check(abs(lhs - T["REP_CTRL"][s][r]["delta"]) < 3e-4 + 2e-4 * abs(T["REP_CTRL"][s][r]["delta"]), "identity %s %s: (SOUP-CTRL) - (SOUP-REP) = REP-CTRL (same window set in the three contrasts)" % (s, r), "%.4f vs %.4f" % (lhs, T["REP_CTRL"][s][r]["delta"]))
        else:
            IDENT_DIFFN.append((s, r, sorted(ns3), round(lhs, 4), T["REP_CTRL"][s][r]["delta"]))

# --------------------------------------------------- 3. writing rules
check(b"\r" not in DOCB, "the document has LF line endings only (no CR byte)")
BAN = re.compile(r"\b(void|superseded|retracted|formerly|no longer|correction|withdrawn)\b", re.I)
hits = BAN.findall(DOCT)
check(not hits, "no banned word (void, superseded, retracted, formerly, no longer, correction, withdrawn)", repr(hits[:5]))
check("void" not in DOCT.lower(), "the substring 'void' does not occur at all (covers 'avoid', 'devoid')")
bad_tier = []
for m in re.finditer(r"closed[- ]loop|driving performance", DOCT, re.I):
    pre = DOCT[max(0, m.start() - 60):m.start()].lower()
    if not re.search(r"never|not |isn't|is not", pre):
        bad_tier.append(DOCT[max(0, m.start() - 60):m.end() + 10].replace("\n", " "))
check(not bad_tier, "'closed loop' and 'driving performance' occur only in a negation", repr(bad_tier[:3]))
check(DOCT.startswith("# RESULT_V8F_SOUP:") and "**HARMS**" in lines[0], "title line names the result and the word HARMS")
check(DOCT.count("T1-kin") >= 10 and "Never 'closed loop'" in DOCT and "Never 'driving performance'" in DOCT, "tier strings present: T1-kin, Never 'closed loop', Never 'driving performance'")

# --------------------------------------------------- 4. parse tables
TABLES = []
i = 0
while i < len(lines):
    if lines[i].startswith("|"):
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            j += 1
        blk = lines[i:j]
        assert re.match(r"^\|(---\|)+$", blk[1]), "no separator row at line %d" % (i + 2)
        rows = []
        for ln in [blk[0]] + blk[2:]:
            assert ln.startswith("| ") and ln.endswith(" |"), "bad table line: " + ln[:80]
            rows.append(ln[2:-2].split(" | "))
        hdr, body = rows[0], rows[1:]
        assert len(blk[1].strip("|").split("|")) == len(hdr), "separator width at line %d" % (i + 2)
        assert all(len(r) == len(hdr) for r in body), "ragged table at line %d" % (i + 1)
        TABLES.append(dict(i0=i, i1=j, hdr=[c.replace("**", "") for c in hdr], rows=[[c.strip() for c in r] for r in body]))
        i = j
    else:
        i += 1
STATS["tables"] = len(TABLES)


def cc(x):
    return x.replace("**", "").strip()


CELLRE = re.compile(r"^(\*\*)?([+-]\d+\.\d{4}) \[([+-]\d+\.\d{4}), ([+-]\d+\.\d{4})\] (B|W|" + DASH + r")(\*\*)?$")
badbold = []
for t in TABLES:
    for r in t["rows"]:
        for c in r:
            m = CELLRE.match(c)
            if m:
                STATS["delta_cells_in_tables"] += 1
                if bool(m.group(1)) != (m.group(5) != DASH) or bool(m.group(1)) != bool(m.group(6)):
                    badbold.append(c)
check(not badbold, "bold marks exactly the separated cells (flag B or W) in every table", "%d cells; bad=%r" % (STATS["delta_cells_in_tables"], badbold[:3]))

HANDLED = set()
NTAB = Counter()


def verify(t, exp_rows, name, fuzzy_cols=()):
    """compare parsed rows with expected rows, cell for cell"""
    HANDLED.add(t["i0"])
    got = [[cc(c) for c in r] for r in t["rows"]]
    errs = []
    if len(got) != len(exp_rows):
        errs.append("row count %d vs %d" % (len(got), len(exp_rows)))
    for k, (g, e) in enumerate(zip(got, exp_rows)):
        if len(g) != len(e):
            errs.append("row %d width %d vs %d" % (k, len(g), len(e)))
            continue
        for ci, (a, b) in enumerate(zip(g, e)):
            if ci in fuzzy_cols:
                if set(x.strip() for x in a.split(",")) != set(x.strip() for x in b.split(",")):
                    errs.append("row %d col %d: %r vs %r" % (k, ci, a[:60], b[:60]))
            elif a != b:
                errs.append("row %d col %d: doc %r vs source %r" % (k, ci, a[:70], b[:70]))
    NTAB[name] += sum(len(r) for r in got)
    check(not errs, "table %s: %d rows x cells equal the source" % (name, len(got)), "; ".join(errs[:4]))


def find(pred):
    out = []
    for t in TABLES:
        try:
            if pred(t):
                out.append(t)
        except IndexError:
            pass
    return out


def tb(name, pred, n=1):
    ts = find(pred)
    check(len(ts) == n, "found %d table(s) for %s" % (n, name), "got %d" % len(ts))
    return ts


PAIRLAB = {"main": "main (SOUP %s DAG-OACC)" % MINUS, "rep": "replicate (SOUP' %s A)" % MINUS}
YNAME = {"main": "DAG-OACC", "rep": "A"}


def hf(b):
    return "held" if b else "FAILS"


def tf(b):
    return "true" if b else "false"


# ---- T1: outcome predicates
for t in tb("outcome predicates", lambda t: t["hdr"][0].startswith("predicate (PREREG section 10)")):
    exp = []
    exp.append(["controls (reader, 368)", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS"])

    def line(label, per, kind):
        cells, both = [], {}
        for s in ("s0", "s1"):
            vs = []
            for p in ("main", "rep"):
                b, txt = per(s, p)
                vs.append(b)
                cells.append(txt)
            both[s] = all(vs)
        f = (lambda b: "held" if b else "FAILS") if kind == "hf" else (lambda b: "true" if b else "false")
        return [label, cells[0], cells[1], f(both["s0"]), cells[2], cells[3], f(both["s1"])]

    got_lbl = [r[0] for r in t["rows"]]
    el = lambda pre: [x for x in got_lbl if x.startswith(pre)]
    for pre in ("R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW", "R-MOVE", "R-CRAWL", "R-KEEP", "R-NOBASE"):
        check(len(el(pre)) == 1, "outcome table has one row labelled %s" % pre)
    exp.append(line(el("R-TLAT")[0], lambda s, p: (tlat[(s, p)], hf(tlat[(s, p)])), "hf"))
    exp.append(line(el("R-TLON")[0], lambda s, p: (tlon[(s, p)], hf(tlon[(s, p)])), "hf"))
    exp.append(line(el("R-ADE")[0], lambda s, p: (PC[(s, p)]["ADE"], "%s: %s %s" % (hf(PC[(s, p)]["ADE"]), f4(xy(s, p, "ADE6")["delta"]), flg(xy(s, p, "ADE6")))), "hf"))
    exp.append(line(el("R-T0")[0], lambda s, p: (PC[(s, p)]["T0"], "%s: %s" % (hf(PC[(s, p)]["T0"]), f4(D["pairs"][s][p]["T0_point"]))), "hf"))
    exp.append(line(el("R-NEW")[0], lambda s, p: (PC[(s, p)]["NEW"], "%s: %d of %d rows" % (hf(PC[(s, p)]["NEW"]), len(PC[(s, p)]["fire"]), len(PC[(s, p)]["dom"]))), "hf"))
    exp.append(line(el("R-MOVE")[0], lambda s, p: (PC[(s, p)]["MOVE"], "%s: %d of 4 rows clear" % (tf(PC[(s, p)]["MOVE"]), sum(PC[(s, p)]["move_rows"].values()))), "tf"))
    exp.append(line(el("R-CRAWL")[0], lambda s, p: (PC[(s, p)]["CRAWL"], "%s: %d of 4 rows" % (tf(PC[(s, p)]["CRAWL"]), sum(PC[(s, p)]["crawl_rows"].values()))), "tf"))
    exp.append(line(el("R-KEEP")[0], lambda s, p: (PC[(s, p)]["KEEP"], "%s: %s" % (tf(PC[(s, p)]["KEEP"]), f4(D["pairs"][s][p]["KEEP_point"]))), "tf"))
    exp.append(line(el("R-NOBASE")[0], lambda s, p: (PC[(s, p)]["NOBASE"], "%s: %d of %d rows violate" % (tf(PC[(s, p)]["NOBASE"]), len(PC[(s, p)]["fire"]), len(PC[(s, p)]["dom"]))), "tf"))
    # the predicate header columns
    check(t["hdr"][1:] == ["s0 main (SOUP %s DAG-OACC)" % MINUS, "s0 replicate (SOUP' %s A)" % MINUS, "s0, both pairs", "s1 main (SOUP %s DAG-OACC)" % MINUS, "s1 replicate (SOUP' %s A)" % MINUS, "s1, both pairs"], "outcome table header columns")
    # row order of the doc
    order = [r[0] for r in t["rows"]]
    exp_sorted = []
    for r in t["rows"]:
        for e in exp:
            if e[0] == cc(r[0]):
                exp_sorted.append(e)
                break
    verify(t, exp_sorted, "outcome")

# ---- T2: per-seed booleans
for t in tb("per-seed booleans", lambda t: t["hdr"][0] == "seed" and t["hdr"][1].startswith("H (R-TLAT")):
    verify(t, [[s, tf(Hs[s]), tf(MVs[s]), tf(ADs[s]), tf(DMs[s])] for s in ("s0", "s1")], "per-seed")

# ---- T3: floors
for t in tb("floors", lambda t: t["hdr"][0] == "row" and t["hdr"][2] == "F"):
    exp = []
    for r in [x[0] for x in t["rows"]]:
        exp.append([r, BETTER[r], u4(FV[r]), u4(RF0[r]), u4(BAR[r]), u4(D["floors"]["RF"][r]), "%.2f" % (D["floors"]["RF"][r] / BAR[r])])
    check(sorted(x[0] for x in t["rows"]) == sorted(ALL19), "floors table lists the 19 rows once each")
    verify(t, exp, "floors")

# ---- T4: ADE6
for t in tb("ADE6", lambda t: t["hdr"][:3] == ["seed", "pair", "row"] and t["hdr"][3] == "X %s Y" % MINUS):
    exp = []
    for s in ("s0", "s1"):
        for p in ("main", "rep"):
            c = xy(s, p, "ADE6")
            exp.append([s, PAIRLAB[p], "ADE6", cl(c), u4(c["mean_arm"]), u4(c["mean_ref"]), cl(xp(s, p, "ADE6")), cl(yp(s, p, "ADE6")), c["n"]])
    verify(t, exp, "ade6")

# ---- T5: TAC / guards
for t in tb("TAC and guards", lambda t: t["hdr"][2].startswith("TAC-LAT X")):
    exp = []
    for s in ("s0", "s1"):
        for p in ("main", "rep"):
            exp.append([s, PAIRLAB[p], cl(xy(s, p, "TAC-LAT")), hf(tlat[(s, p)]), cl(xy(s, p, "TAC-LON")), hf(tlon[(s, p)]), f4(D["pairs"][s][p]["T0_point"]), hf(PC[(s, p)]["T0"]), f4(D["pairs"][s][p]["KEEP_point"]), tf(PC[(s, p)]["KEEP"])])
    verify(t, exp, "tac_guards")

# ---- T6: R-MOVE
n_cr_b = n_cr_w = n_clear = 0
for t in tb("R-MOVE", lambda t: t["hdr"][2] == "crawl row" and t["hdr"][4] == "improvement"):
    exp = []
    for s in ("s0", "s1"):
        for p in ("main", "rep"):
            for r in CRAWL:
                c = xy(s, p, r)
                imp = -c["delta"]
                n_cr_b += c["separated_better"]
                n_cr_w += c["separated_worse"]
                n_clear += PC[(s, p)]["move_rows"][r]
                exp.append([s, PAIRLAB[p], r, cl(c), f4(imp), u4(BAR[r]), "%.1f %%" % (100.0 * imp / BAR[r]), tf(PC[(s, p)]["move_rows"][r])])
    verify(t, exp, "r_move")
n_cr_w_s0 = sum(1 for p in ("main", "rep") for r in CRAWL if xy("s0", p, r)["separated_worse"])

# ---- T7: R-CRAWL
for t in tb("R-CRAWL", lambda t: t["hdr"][2] == "crawl row" and t["hdr"][3] == "X %s P10" % MINUS):
    exp = []
    for s in ("s0", "s1"):
        for p in ("main", "rep"):
            for r in CRAWL:
                a, b = xp(s, p, r), yp(s, p, r)
                exp.append([s, PAIRLAB[p], r, cl(a), cl(b), tf(PC[(s, p)]["crawl_rows"][r]), "%.1f %%" % (100.0 * (b["delta"] - a["delta"]) / b["delta"])])
    verify(t, exp, "r_crawl")

# ---- T8: R-NEW
for t in tb("R-NEW", lambda t: t["hdr"][3] == "rows in D"):
    exp = []
    for s in ("s0", "s1"):
        for p in ("main", "rep"):
            q = PC[(s, p)]
            nm = lambda l: ", ".join(l) if l else "none"
            exp.append([s, PAIRLAB[p], "%s %s P10 not separated-worse on D" % (YNAME[p], MINUS), str(len(q["dom"])), nm(q["xyw"]), nm(q["xyb"]), nm(q["fire"])])
    verify(t, exp, "r_new", fuzzy_cols=(4, 5, 6))

# ---- T9: Read A
for t in tb("Read A", lambda t: t["hdr"][0] == "soup" and t["hdr"][2] == "ADE6 X %s P10" % MINUS):
    exp = []
    for key, ct, lab in (("SOUP", "SOUP_P10", "SOUP"), ("SOUPR", "SOUPR_P10", "SOUP'"), ("SOUP4", "SOUP4_P10", "SOUP4 (beside)")):
        for s in ("s0", "s1"):
            ra = D["read_a"][key]["per_seed"][s]
            exp.append([lab, s, cl(T[ct][s]["ADE6"]), tf(ra["D1"]), "%s (%d rows: %s)" % (tf(ra["D2"]), len(ra["D2_worse_rows"]), ", ".join(ra["D2_worse_rows"])), tf(ra["D3"]), D["read_a"][key]["outcome"]])
    verify(t, exp, "read_a")

# ---- T10: SG
for t in tb("SG", lambda t: t["hdr"][0] == "soup" and t["hdr"][3] == "SG (JSON)"):
    exp = []
    for key, lab, mems in (("SOUP", "SOUP (decisive)", "DAG-OACC, REP"), ("SOUPR", "SOUP' (decisive)", "A, A_rep"), ("SOUP4", "SOUP4 (beside)", "DAG-OACC, REP, A, A_rep")):
        for s in ("s0", "s1"):
            g = D["sg"]["%s:%s" % (key, s)]
            sm, mem = sg_arith(key, s)
            mm = sum(mem) / len(mem)
            fl = "B" if (g["separated"] and g["delta"] < 0) else ("W" if g["separated"] else DASH)
            exp.append([lab, s, mems, "%s [%s, %s] %s" % (f4(g["delta"]), f4(g["lo"]), f4(g["hi"]), fl), tf(g["barrier"]), u4(sm), ", ".join(u4(m) for m in mem), u4(mm), f4(sm - mm), str(g["n_windows"])])
    verify(t, exp, "sg")

# ---- T11: separated counts per family
for t in tb("family counts", lambda t: t["hdr"][0] == "family (rows)"):
    exp = []
    for fam, rl in (("LONGITUDINAL", LONG), ("LATERAL", LAT), ("TACTICAL", TAC), ("STRATEGIC", STR), ("crawl row outside the family blocks (M1)", [M1])):
        row = ["%s (%d)" % (fam, len(rl))]
        for (s, p) in COLS:
            b = sum(xy(s, p, r)["separated_better"] for r in rl)
            w_ = sum(xy(s, p, r)["separated_worse"] for r in rl)
            row.append("%d / %d / %d" % (b, w_, len(rl) - b - w_))
        exp.append(row)
    verify(t, exp, "family_counts")

# ---- T12: family X - Y tables
seen_rows = Counter()
ftabs = find(lambda t: t["hdr"][0] == "row" and t["hdr"][1] == "better" and t["hdr"][2].startswith("s0 main (SOUP"))
check(len(ftabs) == 5, "five X-Y row tables (LONGITUDINAL, LATERAL, M1, TACTICAL, STRATEGIC) found", "got %d" % len(ftabs))
for t in ftabs:
    exp = []
    for r in [x[0] for x in t["rows"]]:
        seen_rows[r] += 1
        exp.append([r, BETTER[r]] + [cl(xy(s, p, r)) for (s, p) in COLS] + [" ; ".join(nrend(xy(s, p, r)["n"]) for (s, p) in COLS)])
    verify(t, exp, "family_xy")
STATS["family_xy_rows"] = sum(seen_rows.values())
check(set(seen_rows) == set(LONG + LAT + [M1] + TAC + STR) and all(v == 1 for v in seen_rows.values()), "the X-Y family tables list each of the 18 non-ADE6 rows exactly once")
# which rows appear under which heading
def rows_under(head_pat):
    for k, ln in enumerate(lines):
        if re.match(head_pat, ln):
            for t in ftabs:
                if t["i0"] > k:
                    return [x[0] for x in t["rows"]]
    return None


check(rows_under(r"### 4\.2 LONGITUDINAL") == LONG and rows_under(r"### 4\.3 LATERAL") == LAT and rows_under(r"### 4\.4 TACTICAL") == TAC and rows_under(r"### 4\.5 STRATEGIC") == STR, "each family heading is followed by its own rows")

# ---- T13: P10 tables
for t in tb("X-P10 every row", lambda t: t["hdr"][0] == "row" and t["hdr"][2] == "SOUP %s P10, s0" % MINUS):
    exp = []
    for r in [x[0] for x in t["rows"]]:
        exp.append([r, BETTER[r]] + [cl(xp(s, p, r)) for (s, p) in COLS] + [" ; ".join(nrend(xp(s, p, r)["n"]) for (s, p) in COLS)])
    check(sorted(x[0] for x in t["rows"]) == sorted(ALL19), "X-P10 table lists the 19 rows once each")
    verify(t, exp, "x_p10")
for t in tb("Y-P10 every row", lambda t: t["hdr"][0] == "row" and t["hdr"][2] == "DAG-OACC %s P10, s0" % MINUS):
    exp = []
    for r in [x[0] for x in t["rows"]]:
        cs_ = [T["CTRL_P10"]["s0"][r], T["A_P10"]["s0"][r], T["CTRL_P10"]["s1"][r], T["A_P10"]["s1"][r]]
        exp.append([r, BETTER[r]] + [cl(c) for c in cs_] + [" ; ".join(nrend(c["n"]) for c in cs_)])
    check(sorted(x[0] for x in t["rows"]) == sorted(ALL19), "Y-P10 table lists the 19 rows once each")
    verify(t, exp, "y_p10")

# ---- T14: coverage table (static text; n-ranges recomputed)
def nset(r):
    ns = set()
    for ct in T:
        for s in T[ct]:
            n = T[ct][s][r]["n"]
            if "/" in n and not n.startswith("ticks"):
                ns.add(n)
    return ns


def nrange(rows_):
    ns = set()
    for r in rows_:
        ns |= nset(r)
    w_ = sorted(set(int(n.split("/")[0]) for n in ns))
    e_ = sorted(set(int(n.split("/")[1]) for n in ns))
    return w_, e_


for t in tb("coverage", lambda t: t["hdr"][0] == "family" and t["hdr"][2].startswith("n (windows")):
    HANDLED.add(t["i0"])
    txt = "\n".join(" | ".join(r) for r in t["rows"])
    w1, e1 = nrange(["speed MAE", "along-track MAE", "accel MAE"])
    w2, e2 = nrange(["DK-VIOL"])
    w3, e3 = nrange(["DK-HW"])
    w4, e4 = nrange(["DK-TG"])
    w5, e5 = nrange(["heading MAE"])
    w6, e6 = nrange(["yaw-rate MAE", "curvature MAE", "L1 lateral-accel MAE"])
    w7, e7 = nrange(["cross-track MAE"])
    w8, e8 = nrange(["L2 curvature MAE (GT>=5.2)", M1])
    w9, e9 = nrange(["TAC-LAT", "TAC-LON"])
    sp = [nset(r) for r in STR]
    expect = [
        "2399/141 (speed, along-track, accel)" if (w1, e1) == ([2399], [141]) else None,
        "595/70 (DK-VIOL)" if (w2, e2) == ([595], [70]) else None,
        "%d to %d / %s (DK-HW)" % (w3[0], w3[-1], " or ".join(str(x) for x in e3)),
        "%d to %d / %d (DK-TG)" % (w4[0], w4[-1], e4[0]) if len(e4) == 1 else None,
        "2373/141 (heading)" if (w5, e5) == ([2373], [141]) else None,
        "%d to %d / %d (yaw-rate, curvature, L1)" % (w6[0], w6[-1], e6[0]) if len(e6) == 1 else None,
        "2399/141 (cross-track)" if (w7, e7) == ([2399], [141]) else None,
        "%d to %d / %d (L2, M1)" % (w8[0], w8[-1], e8[0]) if len(e8) == 1 else None,
        "2399/141" if (w9, e9) == ([2399], [141]) else None,
        "2321/141 (@30); 2313/141 (@50); 2290/141 (@80)" if sp == [{"2321/141"}, {"2313/141"}, {"2290/141"}] else None,
    ]
    missing = [e for e in expect if e is None or e not in txt]
    check(not missing, "coverage table: the n values and ranges equal the JSON's", repr(missing))
    check("PREREG section 8" in txt and "NOT BUILT" in txt and "NOT COMPUTABLE" in txt, "coverage table names the not-built / not-computable items")
    NTAB["coverage"] += sum(len(r) for r in t["rows"])

# ---- T15: beside contrasts
BES = {"REP %s DAG-OACC" % MINUS: "REP_CTRL", "SOUP %s REP" % MINUS: "SOUP_REP", "SOUP' %s A_rep" % MINUS: "SOUPR_Arep", "SOUP' %s DAG-OACC" % MINUS: "SOUPR_CTRL", "SOUP4 %s DAG-OACC" % MINUS: "SOUP4_CTRL"}
for t in tb("beside contrasts", lambda t: t["hdr"] == ["contrast", "row", "s0", "s1"]):
    exp = [[r[0], r[1], cl(T[BES[r[0]]]["s0"][r[1]]), cl(T[BES[r[0]]]["s1"][r[1]])] for r in t["rows"]]
    check([(r[0], r[1]) for r in t["rows"]] == [(c, r) for c in BES for r in ["ADE6"] + CRAWL], "beside table: five contrasts x (ADE6 + 4 crawl rows), in order")
    verify(t, exp, "beside")

# ---- T16: SOUP4 every row
for t in tb("SOUP4 every row", lambda t: t["hdr"][0] == "row" and t["hdr"][2].startswith("SOUP4 %s DAG-OACC, s0" % MINUS)):
    exp = [[r[0], BETTER[r[0]], cl(T["SOUP4_CTRL"]["s0"][r[0]]), cl(T["SOUP4_CTRL"]["s1"][r[0]]), cl(T["SOUP4_P10"]["s0"][r[0]]), cl(T["SOUP4_P10"]["s1"][r[0]])] for r in t["rows"]]
    check(sorted(x[0] for x in t["rows"]) == sorted(ALL19), "SOUP4 table lists the 19 rows once each")
    verify(t, exp, "soup4")

# ---- T17: slow draw
SD = D["beside"]["slow_draw"]
SDK = {"SOUP_0 %s DAG-OACC_0" % MINUS: "main:SOUP-CTRL:s0", "SOUP_1 %s DAG-OACC_1" % MINUS: "main:SOUP-CTRL:s1", "SOUP'_0 %s A_0" % MINUS: "rep:SOUPR-A:s0", "SOUP'_1 %s A_1" % MINUS: "rep:SOUPR-A:s1"}
SCELLS = ["lon_v_target_ms", "v_a_ms", "v_end_ms", "v_hi_ms", "v_lo_ms"]
for t in tb("slow draw", lambda t: t["hdr"][0] == "contrast (T0)"):
    check(t["hdr"][1:6] == SCELLS, "slow-draw header lists the five JSON speed cells")
    exp = []
    for r in t["rows"]:
        k = SDK[cc(r[0])]
        cells = SD[k]["cells"]
        exp.append([cc(r[0])] + ["%s [%s, %s]" % (f4(cells[x]["mean"]), f4(cells[x]["lo"]), f4(cells[x]["hi"])) for x in SCELLS] + ["null" if SD[k]["label"] is None else str(SD[k]["label"]), "%.1f" % SD[k]["threshold_m_s"]])
        for x in SCELLS:
            c = cells[x]
            want_bold = c["lo"] > 0 or c["hi"] < 0
            idx = SCELLS.index(x) + 1
            if bool(r[idx].startswith("**")) != want_bold:
                bad("slow-draw bold rule", "%s %s" % (k, x))
    check([cc(r[0]) for r in t["rows"]] == list(SDK), "slow-draw rows in order")
    verify(t, exp, "slow_draw")

# ---- T18: controls by family
cn = Counter(k.split(":")[0] for k in D["controls_PASS"])
for t in tb("control families", lambda t: t["hdr"] == ["control family", "entries", "all true"]):
    exp = [[k, str(v), "PASS"] for k, v in sorted(cn.items())]
    verify(t, exp, "controls")

# ---- T19/T20: Fact 2 and 2b
for t in tb("Fact 2", lambda t: t["hdr"][0] == "seed 0, ADE6 (m, T1-kin)"):
    gap = T["REP_CTRL"]["s0"]["ADE6"]["delta"]
    sl_m, go_m = T["SOUP_REP"]["s0"]["ADE6"], T["SOUP_CTRL"]["s0"]["ADE6"]
    sl_r, go_r = T["SOUPR_Arep"]["s0"]["ADE6"], T["SOUPR_A"]["s0"]["ADE6"]
    gap_r = go_r["delta"] - sl_r["delta"]
    exp = [
        ["slow member %s good member (the member spread)" % MINUS, "REP_0 %s DAG-OACC_0 %s" % (MINUS, cl(T["REP_CTRL"]["s0"]["ADE6"])), "A_rep,0 %s A_0 %s (arithmetic: the A %s soup delta minus the A_rep %s soup delta)" % (MINUS, f4(gap_r), MINUS, MINUS)],
        ["soup %s slow member" % MINUS, "SOUP_0 %s REP_0 %s" % (MINUS, cl(sl_m)), "SOUP'_0 %s A_rep,0 %s" % (MINUS, cl(sl_r))],
        ["soup %s good member (the control)" % MINUS, "SOUP_0 %s DAG-OACC_0 %s" % (MINUS, cl(go_m)), "SOUP'_0 %s A_0 %s" % (MINUS, cl(go_r))],
        ["share of the member spread recovered (%sdelta vs slow member / spread)" % MINUS, "%.1f %%" % (100.0 * -sl_m["delta"] / gap), "%.1f %%" % (100.0 * -sl_r["delta"] / gap_r)],
        ["position of the soup between the good and the slow member (delta vs good member / spread)", "%.1f %%" % (100.0 * go_m["delta"] / gap), "%.1f %%" % (100.0 * go_r["delta"] / gap_r)],
    ]
    verify(t, exp, "fact2")
    check(abs(gap_r - (T["SOUPR_A"]["s0"]["ADE6"]["mean_ref"] * -1 + T["SOUPR_Arep"]["s0"]["ADE6"]["mean_ref"])) < 3e-4, "A_rep,0 - A_0 spread equals the difference of the two table means")
POS, REC = [], []
for t in tb("Fact 2b", lambda t: t["hdr"][0] == "pair (seed 0, T1-kin)"):
    exp = []
    for p, lab in (("main", "main (DAG-OACC_0, REP_0)"), ("rep", "replicate (A_0, A_rep,0)")):
        for r in CRAWL:
            if p == "main":
                sl, go = T["SOUP_REP"]["s0"][r], T["SOUP_CTRL"]["s0"][r]
            else:
                sl, go = T["SOUPR_Arep"]["s0"][r], T["SOUPR_A"]["s0"][r]
            spread = go["delta"] - sl["delta"]
            POS.append(100.0 * go["delta"] / spread)
            REC.append(100.0 * -sl["delta"] / spread)
            exp.append([lab, r, f4(spread), cl(sl), "%.1f %%" % REC[-1], cl(go), "%.1f %%" % POS[-1]])
    verify(t, exp, "fact2b")

# ---- T21: register rows quoted in section 7.2
def toks(sx):
    sx = sx.replace(MINUS, "-").replace("\u2026", "")
    return re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?!\w)", sx)


REGIDS = ["H-CRAWL-01", "H-CRAWL-02", "H-CRAWL-03", "H-ST-REGRET-PRE", "H-ST-RECOVERYLABEL-1", "H-DD-FITGAP-M2B", "H-CRAWL-06-PROBE", "H-CRAWL-06-PROBE2"]
check(all(i in REG for i in REGIDS), "the eight register rows exist at the tip")
for t in tb("Rule Zero register table", lambda t: t["hdr"][0] == "register row"):
    HANDLED.add(t["i0"])
    got_ids = [re.sub(r"[`]", "", r[0]) for r in t["rows"]]
    check(got_ids == REGIDS, "the register table lists the eight rows in the order of the brief")
    nbad = []
    for r in t["rows"]:
        rid = r[0].replace("`", "")
        rowtxt = REG[rid].replace("\u2212", "-").replace("\u2026", "")
        rowtok = set(x.replace(",", "") for x in toks(rowtxt))
        for tok in toks(r[1]):
            if tok.replace(",", "") not in rowtok:
                nbad.append((rid, tok))
    check(not nbad, "every number in the 'what it measured' column is a number of its register row", repr(nbad[:6]))
    NTAB["register_table"] += sum(len(r) for r in t["rows"])
    # statuses of the rows
    check("NOT EVALUABLE" in REG["H-CRAWL-03"] and "PARTIAL" in REG["H-DD-FITGAP-M2B"] and "MIDDLE BAND" in REG["H-ST-REGRET-PRE"], "register statuses quoted: H-CRAWL-03 NOT EVALUABLE, H-DD-FITGAP-M2B PARTIAL, H-ST-REGRET-PRE MIDDLE BAND")

# ---- T22: blocked levers (static)
for t in tb("blocked levers", lambda t: t["hdr"][0] == "lever"):
    HANDLED.add(t["i0"])
    txt = " ".join(" ".join(r) for r in t["rows"])
    need = [("about 2.8 h", PREREG), ("about 0.17 h each on a 10 % clip subset", REG["H-DD-FITGAP-M2B"]), ("a trainer option that keeps the last k checkpoints", REG["H-CRAWL-03"])]
    miss = [n for n, src in need if n.split(" each")[0] not in txt or n.replace("about ", "") .split(" each")[0] not in src.replace("\u2212", "-")]
    check(not miss, "blocked-lever table: the cost and unblock statements are in the PREREG / register rows", repr(miss))
    NTAB["blocked"] += sum(len(r) for r in t["rows"])
    check("ESTIMATED" in txt and "PROPOSED, not registered" in txt, "blocked-lever table: ESTIMATED and PROPOSED labels present")

# ---- T23: run record
WALLS = {m.group(2): int(m.group(3)) for m in re.finditer(r"(\d\d:\d\d:\d\d) end (\S+) rc=0 wall=(\d+)s", CHAIN)}
check(len(WALLS) == 28 and sum(WALLS.values()) == 2074, "chain log: 28 steps with rc=0 and walls summing to 2074 s", "%d steps, sum %d" % (len(WALLS), sum(WALLS.values())))
for t in tb("run record", lambda t: t["hdr"][0] == "soup" and t["hdr"][1] == "causal rollout (s)"):
    S_ = {"SOUP_0": "s0", "SOUP_1": "s1", "SOUP'_0": "r_s0", "SOUP'_1": "r_s1", "SOUP4_0": "4_s0", "SOUP4_1": "4_s1"}
    name = {"SOUP_0": ("SOUP_s0", "M2b-soup_s0"), "SOUP_1": ("SOUP_s1", "M2b-soup_s1"), "SOUP'_0": ("SOUPR_s0", "M2b-soupr_s0"), "SOUP'_1": ("SOUPR_s1", "M2b-soupr_s1"), "SOUP4_0": ("SOUP4_s0", "M2b-soup4_s0"), "SOUP4_1": ("SOUP4_s1", "M2b-soup4_s1")}
    exp = []
    for r in t["rows"]:
        a, b = name[r[0]]
        exp.append([r[0], str(WALLS[a + "_m10_causal_heads"]), str(WALLS[a + "_m10_logged_heads"]), str(WALLS[b + "-fresh"]), str(WALLS[b])])
    verify(t, exp, "run_record")

# ---- T24: provenance
PROVP = {}
for t in tb("provenance", lambda t: t["hdr"] == ["role", "repo path", "sha256"]):
    HANDLED.add(t["i0"])
    errs = []
    for r in t["rows"]:
        path = r[1].strip("`")
        full = path if path.startswith("Project Steering/") else RES + path
        h = sha(blob(full))
        PROVP[path] = h
        if r[2].strip("`") != h:
            errs.append(path)
    check(not errs, "provenance: every sha256 equals the git blob at the tip", "%d rows; bad=%r" % (len(t["rows"]), errs))
    STATS["prov_rows"] = len(t["rows"])
    NTAB["provenance"] += sum(len(r) for r in t["rows"])

unh = [t["hdr"][:2] for t in TABLES if t["i0"] not in HANDLED]
check(not unh, "every table of the document was verified (none is unknown)", repr(unh[:3]))

# --------------------------------------------- 5. prose: cells, quotes, hex
tbl_lines = set()
for t in TABLES:
    for k in range(t["i0"], t["i1"]):
        tbl_lines.add(k)
PROSE = "\n".join(ln for k, ln in enumerate(lines) if k not in tbl_lines).replace("**", "")

CELL_ANY = re.compile(r"([+-]\d+\.\d{4}) \[([+-]\d+\.\d{4}), ([+-]\d+\.\d{4})\](?: (B|W|" + DASH + r"))?")
leaf = set()
for ct, bys in T.items():
    for s, rows in bys.items():
        for r, c in rows.items():
            leaf.add((f4(c["delta"]), f4(c["lo"]), f4(c["hi"]), flg(c)))
for k, g in D["sg"].items():
    fl = "B" if (g["separated"] and g["delta"] < 0) else ("W" if g["separated"] else DASH)
    leaf.add((f4(g["delta"]), f4(g["lo"]), f4(g["hi"]), fl))
slowleaf = set()
for v in SD.values():
    for c in v["cells"].values():
        slowleaf.add((f4(c["mean"]), f4(c["lo"]), f4(c["hi"])))
nprose, badc = 0, []
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


def collect(o, acc):
    if isinstance(o, str):
        acc.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            collect(v, acc)
    elif isinstance(o, list):
        for v in o:
            collect(v, acc)
    return acc


CORPUS = "\n".join(collect(D, [])) + "\n" + PREREG + "\n" + "\n".join(REG.values()) + "\n" + CHAIN + "\n" + READOUT + "\n" + CLAUDE
CORPUS_N = CORPUS.replace("\u2212", "-").lower()
quotes = re.findall(r'\*"(.+?)"\*', PROSE.replace("\n", " "))
badq = []
for q in set(quotes):
    qq = q.rstrip(".,;").replace("\u2212", "-").lower()
    if qq not in CORPUS_N and q.lower() not in CORPUS_N:
        badq.append(q[:90])
check(not badq, "every quoted string is found verbatim in the JSON, the PREREG, a register row, a log or CLAUDE.md", "%d quotes; bad=%r" % (len(set(quotes)), badq))
STATS["quotes"] = len(set(quotes))

known_hex = set(re.findall(r"[0-9a-f]{8,64}", CORPUS + "\n" + json.dumps(D)))
known_hex |= set(PROVP.values()) | {sha(JB), sha(PREREG_B)}
for c in ("0c7b598", "e03b348"):
    r = git("rev-parse", "--verify", c + "^{commit}")
    if r.returncode == 0:
        known_hex.add(r.stdout.decode().strip())
known_hex |= {D["provenance"]["manifest_sha256"], D["provenance"]["reader_sha256"], D["provenance"]["li_reader_sha256"]}
codes = re.findall(r"`([0-9a-f]{7,64})(?:\u2026)?`", DOCT)
badh = [c for c in codes if not any(h.startswith(c) for h in known_hex)]
check(not badh, "every hash or commit prefix in a code span is a prefix of a hash found in a source", "%d prefixes; bad=%r" % (len(codes), badh))
STATS["hex_prefixes"] = len(codes)
check(git("rev-parse", "0c7b598").stdout.decode().strip() == TIP, "0c7b598 is the pinned tip")

# ------------------------------------------------------- 6. times, commits, disclosures
def ci(rev):
    return git("show", "-s", "--format=%ci", rev).stdout.decode().strip()


check(ci("e03b348").startswith("2026-10-29 02:00:06") and ci("0c7b598").startswith("2026-10-29 03:10:30"), "commit times: e03b348 at 02:00:06 and 0c7b598 at 03:10:30 on 2026-10-29", ci("e03b348") + " / " + ci("0c7b598"))
check("promoted at `e03b348` (2026-10-29 02:00:06;" in DOCT and "banked at `0c7b598` (2026-10-29 03:10:30)" in DOCT, "the header states those commit times")
first = re.search(r"10-29 (\d\d:\d\d:\d\d) === soup_chain start", CHAIN)
lastend = max(re.findall(r"(\d\d:\d\d:\d\d) end \S+ rc=0", CHAIN))
rdr = re.search(r"(\d\d:\d\d:\d\d) == reader:", CHAIN)
done = re.search(r"(\d\d:\d\d:\d\d) reader rc=0", CHAIN)
check((first.group(1), lastend, rdr.group(1), done.group(1)) == ("02:19:45", "02:56:08", "02:56:28", "03:07:03"), "chain times: start 02:19:45, last step end 02:56:08, reader start 02:56:28, reader rc=0 at 03:07:03")
check("The chain started 02:19:45 and its last step ended 02:56:08." in DOCT and "The reader started 02:56:28 and the chain closed at 03:07:03" in DOCT, "run-record prose states those times")
check(len(re.findall(r"\(attempt 1;", CHAIN)) == 28 and not re.search(r"\(attempt [2-9]", CHAIN), "chain log: 28 steps, all at attempt 1")
check("ZZSOUP-DONE 28/28 steps, reader rc 0" in CHAIN and "C: gate: 34097 MB free >= 2048 MB" in CHAIN, "chain log carries the done line and the C: gate line")
check(D["provenance"]["wall_s"] == 634.9 and "reader wall 634.9 s" in DOCT, "reader wall 634.9 s")
# launcher diff
cl_l, cl_s = CMD_L.split(b"\n"), CMD_S.split(b"\n")
diff_idx = [k for k in range(max(len(cl_l), len(cl_s))) if (cl_l[k] if k < len(cl_l) else None) != (cl_s[k] if k < len(cl_s) else None)]
check(len(cl_l) == len(cl_s) and diff_idx == [9, 10] and all(x.endswith(b"\r") for x in cl_l[:-1]) and all(x.endswith(b"\r") for x in cl_s[:-1]),
      "launcher: the two banked copies differ in exactly lines 10 and 11, and both are CRLF", repr(diff_idx))
check(b"SOUP_PREREG_SHA=" + PREREG_PIN.encode() in CMD_L and b"rem   set SOUP_PREREG_SHA" in CMD_S, "launcher: lines 10 and 11 set the PREREG path and the pinned sha256 in the launched copy and are remarks in the shipped copy")
check("the promotion changed this line and section 0 item 5 only" in PREREG, "PREREG status line says the promotion changed the status line and section 0 item 5 only")
check("STATIC" in D["controls_PASS"] and not any("PREREG" in k for k in D["controls_PASS"]), "controls_PASS has STATIC and no control named PREREG-TEXT (disclosure 1)")
REREAD = blob("Project Steering/PREREG_V8F_DAG_REREAD.md").decode("utf-8")
check("**MARGIN, NOT DOMINANT** | D1 at both seeds; D2 fails at ≥ 1 seed" in REREAD and "DAgger iteration 2" in REREAD, "PREREG_V8F_DAG_REREAD section 5 defines MARGIN, NOT DOMINANT as D1 at both seeds and D2 failing at >= 1 seed, naming DAgger iteration 2")
for lab, phrase in (("SOUP-K", "**SOUP-K**: K = 4 pure-recipe draws per seed"),):
    check(phrase in PREREG.replace("\n", " ") or "SOUP-K**: K = 4 pure-recipe draws per seed" in PREREG, "PREREG section 11 FLAT row: SOUP-K, K = 4 pure-recipe draws per seed")
check("4 new box-2 runs, \u2248 2.8 h ESTIMATED" in PREREG and "its own PI go" in PREREG, "PREREG section 11: 4 new box-2 runs, about 2.8 h ESTIMATED, its own PI go")
check("SOUP'_0 \u2212 A_0" not in DOCT or True, "-")

# ----- phrase checks (derived claims)
cnt_sg_b = sum(1 for k in SGK if D["sg"][k]["separated"] and D["sg"][k]["delta"] < 0)
check(cnt_sg_b == 5 and "Five of six cells are separated-better" in DOCT, "'Five of six' SG cells are separated-better", str(cnt_sg_b))
check(not D["sg"]["SOUP:s1"]["separated"] and "SOUP at s1 is not separated" in DOCT, "SOUP at s1 SG is not separated")
check(all(not PC[k]["MOVE"] for k in COLS) and all(not PC[k]["CRAWL"] for k in COLS) and "R-MOVE is false in all four pairs and R-CRAWL is false in all four pairs" in DOCT, "R-MOVE and R-CRAWL false in all four pairs")
check(not PC[("s0", "main")]["ADE"] and not PC[("s0", "rep")]["ADE"] and PC[("s1", "main")]["ADE"] and PC[("s1", "rep")]["ADE"], "R-ADE fails in both pairs at s0 and holds in both at s1")
check(not PC[("s0", "main")]["KEEP"] and not PC[("s0", "rep")]["KEEP"] and PC[("s1", "main")]["KEEP"] and PC[("s1", "rep")]["KEEP"], "R-KEEP fails in both pairs at s0 and holds in both at s1")
check(tlat[("s0", "main")] is False and all(v for k, v in tlat.items() if k != ("s0", "main")), "R-TLAT fails only in the s0 main pair")
check(xy("s1", "rep", "ADE6")["separated_better"], "at s1 the replicate pair ADE6 X-Y is separated-better")
check(n_cr_b == 0 and n_clear == 0 and n_cr_w == 9 and n_cr_w_s0 == 8, "crawl cells X-Y: 0 separated-better, 0 clear, 9 separated-worse (8 at s0)", "b=%d clear=%d w=%d w_s0=%d" % (n_cr_b, n_clear, n_cr_w, n_cr_w_s0))
check(all(xp(s, p, r)["separated_worse"] for s in ("s0", "s1") for p in ("main", "rep") for r in CRAWL), "all 16 X-P10 crawl cells (decisive soups) are separated-worse")
check(all(T[ct][s][r]["separated_worse"] for ct in ("SOUP4_P10",) for s in ("s0", "s1") for r in CRAWL), "SOUP4 - P10 crawl cells are separated-worse at both seeds")
check(all(T[ct][s][r]["separated_worse"] for ct in ("CTRL_P10", "A_P10") for s in ("s0", "s1") for r in CRAWL), "DAG-OACC - P10 and A - P10 are separated-worse on the four crawl rows at both seeds")
check(all(not T["REP_CTRL"]["s1"][r]["separated"] for r in ["ADE6"] + CRAWL), "REP_1 - DAG-OACC_1 is not separated on ADE6 or any crawl row")
check(all(T["REP_CTRL"]["s0"][r]["separated_worse"] for r in ["ADE6"] + CRAWL), "REP_0 - DAG-OACC_0 is separated-worse on ADE6 and all four crawl rows")
check(all(xy("s0", p, r)["separated_worse"] for p in ("main", "rep") for r in CRAWL) and all(0 < x < 100 for x in POS), "s0: the soup is separated-worse than its control on all four crawl rows in both pairs, and lies between its members (0 < position < 100 %)", "min %.1f max %.1f" % (min(POS), max(POS)))
check(all(x > 0 for x in REC), "s0: the soup recovers a positive share of the slow member's spread on all eight crawl cells", "min %.1f max %.1f" % (min(REC), max(REC)))
check(all(SD[k]["cells"][x]["hi"] < 0 for k in SD if k.endswith("s0") for x in SCELLS), "slow-draw: at s0 all ten speed cells have intervals below 0")
check(max(abs(SD[k]["cells"][x]["mean"]) for k in SD if k.endswith("s1") for x in SCELLS) < 0.15 and all(SD[k]["label"] is None and SD[k]["threshold_m_s"] == 0.5 for k in SD), "slow-draw: at s1 all speed means are within 0.15 m/s of 0; labels null; threshold 0.5")
check(abs(D["floors"]["RF"]["ADE6"] - T["REP_CTRL"]["s0"]["ADE6"]["delta"]) < 1e-9, "RF on ADE6 equals the REP - DAG-OACC s0 delta")

# the register row H-V8F-SOUP-1 against the JSON: every cell it quotes is in the JSON
regrow = REG["H-V8F-SOUP-1"]
badreg = []
nreg = 0
for m in re.finditer(r"([+-]?\d+\.\d{4}) \[([+-]?\d+\.\d{4}), ([+-]?\d+\.\d{4})\]", regrow):
    nreg += 1
    d_, lo_, hi_ = [("%+.4f" % float(x)) for x in m.groups()]
    if (d_, lo_, hi_) not in {x[:3] for x in leaf}:
        badreg.append(m.group(0))
check(not badreg, "every cell quoted in the register row H-V8F-SOUP-1 is a cell of the JSON", "%d cells; bad=%r" % (nreg, badreg))
check("TAC-LAT macro-F1" in regrow and "dca10c07" in regrow, "register row labels the -0.0125 cell 'TAC-LAT macro-F1' (disclosure 6) and cites dca10c07")

# ------------------------------------------------- 8. numeric coverage of prose
EV = []
INTS = set()


def walk(o):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        EV.append(float(o))
        if float(o) == int(o):
            INTS.add(int(o))
    elif isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)


walk(D)
SRC_TXT = PREREG + "\n" + "\n".join(REG.values()) + "\n" + CHAIN + "\n" + READOUT + "\n" + "\n".join(collect(D, []))
for m in re.finditer(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])", SRC_TXT.replace(MINUS, "-")):
    t_ = m.group(0).replace(",", "")
    try:
        v = float(t_)
    except ValueError:
        continue
    EV.append(v)
    if "." not in t_:
        INTS.add(int(v))
DERIVED = {
    "controls": len(D["controls_PASS"]), "families": len(cn), "tables": len(TABLES), "crawl cells": 16, "crawl w": n_cr_w, "crawl w s0": n_cr_w_s0,
    "crawl w s1": n_cr_w - n_cr_w_s0, "crawl b": n_cr_b, "clear": n_clear, "pos min": min(POS), "pos max": max(POS), "rec min": min(REC), "rec max": max(REC),
    "wall sum": sum(WALLS.values()), "steps": len(WALLS), "95": 95.0, "F/2": F_HALF, "x2": 2.0, "half": 50.0,
    "rows": 19, "dom": len(PC[("s0", "main")]["dom"]), "soup cells": 6, "arith": 0.0708,
}
for k, v in cn.items():
    DERIVED["ctl " + k] = v
gap_r_s1 = abs(T["SOUPR_A"]["s1"]["ADE6"]["delta"] - T["SOUPR_Arep"]["s1"]["ADE6"]["delta"])
DERIVED["gap_r_s1"] = gap_r_s1
EV += [float(v) for v in DERIVED.values()]
INTS |= {int(v) for v in DERIVED.values() if float(v) == int(v)}
BRIEF_NUMS = {307: "free MB at hand-back", 886: "free MB at hand-back", 46: "scratchpad folders moved", 32: "GB moved"}
scan = PROSE
for r in sorted(ALL19 + ["ADE6"], key=len, reverse=True):
    scan = scan.replace(r, " ")
scan = re.sub(r"`[^`]*`", " ", scan)
scan = CELL_ANY.sub(" ", scan)
scan = re.sub(r"\d{4}-\d\d-\d\d", " ", scan)
scan = re.sub(r"~?\d\d:\d\d(?::\d\d)?", " ", scan)
scan = re.sub(r"\b[Ss]ections? \d+(?:\.\d+)?(?:(?:,| and) \d+(?:\.\d+)?)*", " ", scan)
scan = re.sub(r"\b(?:item|Fact|disclosure|step|lines?) \d+(?: and \d+)?", " ", scan)
scan = re.sub(r"^#+ \d+(?:\.\d+)*\.? ", " ", scan, flags=re.M)
scan = re.sub(r"^\s*\d+\. ", " ", scan, flags=re.M)
NUMRE = r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])"
unmatched, ncheck = [], 0
for m in re.finditer(NUMRE, scan):
    tok = m.group(0)
    s_ = tok.replace(",", "")
    try:
        v = float(s_)
    except ValueError:
        continue
    ncheck += 1
    if "." not in s_:
        iv = int(v)
        if iv in INTS or iv in BRIEF_NUMS:
            continue
        unmatched.append(tok)
        continue
    dec = len(s_.split(".")[1])
    if any(abs(round(abs(e), dec) - v) < 1e-9 for e in EV):
        continue
    unmatched.append(tok)
check(not unmatched, "every number in the prose is found in the JSON, a log, the PREREG, a register row or a verified derived value", "%d numbers checked; unmatched=%r" % (ncheck, unmatched))
STATS["prose_numbers"] = ncheck
check(all(x in DOCT for x in ("(INHERITED from the brief)",)) and DOCT.count("INHERITED from the brief") >= 4, "brief-sourced numbers (307, 886, 46, 32) sit in sentences labelled INHERITED from the brief")
sent = [s_ for s_ in re.split(r"(?<=[.;])\s", PROSE) if re.search(r"\b(307|886|46|32)\b", s_)]
check(sent and all("INHERITED" in s_ or "MOVED" in s_ or "free at hand-back" in s_ for s_ in sent), "each sentence carrying a brief-sourced number names INHERITED", "%d sentences" % len(sent))

# ------------------------------------------------------------------ summary
print()
print("SUMMARY tables %d; table cells verified by table: %s" % (STATS["tables"], dict(NTAB)))
print("SUMMARY JSON cells self-checked %d | four_families cells %d | delta cells in tables %d | family X-Y rows %d" % (STATS["json_cells"], STATS["family_cells"], STATS["delta_cells_in_tables"], STATS["family_xy_rows"]))
print("SUMMARY prose cells %d | quotes %d | hash prefixes %d | prose numbers %d | provenance rows %d" % (STATS["prose_cells"], STATS["quotes"], STATS["hex_prefixes"], STATS["prose_numbers"], STATS["prov_rows"]))
print("SUMMARY %d checks passed, %d failed" % (len(PASSES), len(FAILS)))
if FAILS:
    for n, d in FAILS:
        print("  FAILED: %s: %s" % (n, d[:300]))
    sys.exit(1)
print("ALL CHECKS PASS")
sys.exit(0)
