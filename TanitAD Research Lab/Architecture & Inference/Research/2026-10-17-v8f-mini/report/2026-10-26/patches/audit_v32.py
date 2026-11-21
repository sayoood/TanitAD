"""Numbers audit for report v32 (mechanical; mirrors the v31 audit).
python audit_v32.py [tip [out_dir [audit_path]]]  ->  writes v32_NUMBERS_AUDIT.md next to this script (or to audit_path).
out_dir is the folder holding V8F_SCIENTIFIC_REPORT.md (default: out/ beside this script, else the folder above it, which is where the landed copy keeps the report).

Method: the v32 md (out/) is diffed against the v31 md (the blob at <tip>); every added or changed line is tokenised for numbers (sign kept,
the Unicode minus read as '-', thousands commas removed, single digits 0-9 unsigned excluded as enumerations); each distinct signed token is looked up
line by line in the source files below.  A = found with the same sign in a source line; B = found only with a different or absent sign; C = found only in
the v31 report; D = found nowhere.  Every B, C and D token must carry a hand review in REVIEWED, or this script fails (a new unexplained token cannot pass)."""
import difflib
import hashlib
import os
import re
import subprocess
import sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1] if len(sys.argv) > 1 else "ea5e3d3b"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"

def _out_dir():
    if len(sys.argv) > 2:
        return sys.argv[2]
    up = os.path.dirname(HERE.rstrip("/"))
    if os.path.isdir(HERE + "out"):
        return HERE + "out"
    return up if os.path.exists(up + "/V8F_SCIENTIFIC_REPORT.md") else HERE + "out"


OUT = _out_dir().rstrip("/\\") + "/"
AUDIT_PATH = sys.argv[3] if len(sys.argv) > 3 else HERE + "v32_NUMBERS_AUDIT.md"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RP = R + "report/2026-10-26/"


def git(args):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout


def blob(path):
    return git(["show", f"{TIP}:{path}"]).decode("utf-8").replace("\r\n", "\n")


SOURCES = OrderedDict([
    ("RESULT_V8F_SPEEDREAD.md", blob(R + "RESULT_V8F_SPEEDREAD.md")),
    ("RESULT_V8F_DAGGER2.md", blob(R + "RESULT_V8F_DAGGER2.md")),
    ("PREREG_V8F_SPEEDREAD.md", blob("Project Steering/PREREG_V8F_SPEEDREAD.md")),
    ("PREREG_V8F_DAGGER2.md", blob("Project Steering/PREREG_V8F_DAGGER2.md")),
    ("PREREG_V8F_DAG_ORC.md", blob("Project Steering/PREREG_V8F_DAG_ORC.md")),
    ("R2A_NOTE.md", blob(R + "raw/2026-10-28-dag-orc-r2a/R2A_NOTE.md")),
    ("r1_refused/summary.json", blob(R + "raw/2026-10-28-dag-orc-a2/r1_refused/M2b_s0_dagoaccorc/summary.json")),
    ("r2/selftest_chain_r2.out", blob(R + "raw/2026-10-28-dag-orc-r2a/r2/logs/selftest_chain_r2.out")),
    ("t1/SELFTEST.out", blob(R + "raw/2026-10-28-dag-orc-r2a/t1/selftest/SELFTEST.out")),
    ("relaunch/box2_aside_10280348.log", blob(R + "raw/2026-10-28-dag-orc-r2a/relaunch/box2_aside_10280348.log")),
    ("t1/kbuild_r2_launch_8c24823f.json", blob(R + "raw/2026-10-28-dag-orc-r2a/t1/kbuild_r2_launch_8c24823f.json")),
    ("RESULT_V8F_DAG_ORC.md", blob(R + "RESULT_V8F_DAG_ORC.md")),
    ("R2C_NOTE.md", blob(R + "raw/2026-10-28-dag-orc-result/notes/R2C_NOTE.md")),
    ("git log (c7963d72, 8c24823f, 01f3d34e)", git(["log", "--no-walk=sorted", "--format=%h %ad %s", "--date=iso", "c7963d72", "8c24823f", "01f3d34e"]).decode("utf-8")),
    ("MM_STATE_2026-10-28.md (E:, not in the repo)", open("E:/Projects/TanitAD-artifacts/dagland/MM_STATE_2026-10-28.md", encoding="utf-8").read()),
])
V31 = blob(RP + "V8F_SCIENTIFIC_REPORT.md")
V32 = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8").read()
assert "\r" not in V32

NUM = re.compile(r"(?<![\w.])[-+\u2212]?\d[\d,]*(?:\.\d+)?(?:[eE][-+\u2212]?\d+)?(?![\w])")


def norm(tok):
    t = tok.replace("\u2212", "-").replace(",", "")
    return t


def toks(line):
    out = []
    for m in NUM.finditer(line):
        t = norm(m.group(0))
        if re.fullmatch(r"\d", t):
            continue
        out.append(t)
    return out


def unsigned(t):
    return t.lstrip("+-")


a, b = V31.split("\n"), V32.split("\n")
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
added = []  # (v32 line number, text)
removed = []
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op in ("replace", "insert"):
        added += [(j + 1, b[j]) for j in range(j1, j2)]
    if op in ("replace", "delete"):
        removed += [(i + 1, a[i]) for i in range(i1, i2)]
changed_v31 = [t for _, t in removed]

tok_lines = OrderedDict()  # token -> [v32 line numbers]
for ln, text in added:
    for t in toks(text):
        tok_lines.setdefault(t, []).append(ln)

src_idx = OrderedDict()  # token -> {source: [lines]}
src_unsigned = {}
for name, text in SOURCES.items():
    for k, line in enumerate(text.split("\n"), 1):
        for t in toks(line):
            src_idx.setdefault(t, OrderedDict()).setdefault(name, []).append(k)
            src_unsigned.setdefault(unsigned(t), OrderedDict()).setdefault(name, []).append(k)
v31_unsigned = {}
for k, line in enumerate(a, 1):
    if (k, line) in removed:
        continue
    for t in toks(line):
        v31_unsigned.setdefault(unsigned(t), []).append(k)


def span(d, cap=6):
    parts = []
    for name, lines in d.items():
        short = lines[:cap]
        parts.append(name + ":" + ",".join(str(x) for x in short) + ("..." if len(lines) > cap else ""))
    return "; ".join(parts)


A, B, C, D = [], [], [], []
for t, lns in tok_lines.items():
    if t in src_idx:
        A.append((t, lns, src_idx[t]))
    elif unsigned(t) in src_unsigned:
        B.append((t, lns, src_unsigned[unsigned(t)]))
    elif unsigned(t) in v31_unsigned:
        C.append((t, lns, v31_unsigned[unsigned(t)]))
    else:
        D.append((t, lns))

# ---- hand review: every B, C and D token is explained here; the script fails on an unexplained one --------------------------
REVIEWED = {}
SCOPE_OK = {}
exec(open(HERE + "audit_reviewed.py", encoding="utf-8").read()) if os.path.exists(HERE + "audit_reviewed.py") else None
need = [t for t, *_ in B + C + D]
missing = [t for t in need if t not in REVIEWED]
if missing:
    print("UNREVIEWED TOKENS (%d):" % len(missing))
    for kind, lst in (("B", B), ("C", C), ("D", D)):
        for row in lst:
            if row[0] in missing:
                print(kind, row[0], "v32 lines", row[1][:5], "|", (span(row[2]) if kind == "B" else row[2][:5] if kind == "C" else ""))
    sys.exit(2)
stale = [t for t in REVIEWED if t not in need]
assert not stale, ("reviewed tokens no longer in B/C/D", stale)

# ---- section-scoped check: a number of section 7.24 / 7.25 / 7.26 must be found in THAT section's own sources ----------------------
SECTION_SOURCES = {
    "7.24": ["RESULT_V8F_SPEEDREAD.md", "PREREG_V8F_SPEEDREAD.md"],
    "7.25": ["RESULT_V8F_DAGGER2.md", "PREREG_V8F_DAGGER2.md"],
    "7.26": ["RESULT_V8F_DAG_ORC.md", "R2C_NOTE.md", "PREREG_V8F_DAG_ORC.md", "R2A_NOTE.md", "r1_refused/summary.json", "r2/selftest_chain_r2.out", "t1/SELFTEST.out",
             "relaunch/box2_aside_10280348.log", "t1/kbuild_r2_launch_8c24823f.json", "git log (c7963d72, 8c24823f, 01f3d34e)",
             "MM_STATE_2026-10-28.md (E:, not in the repo)"],
}
_i24, _i25, _i26 = V32.index("### 7.24 "), V32.index("### 7.25 "), V32.index("### 7.26 ")
_iend = V32.index("\n---\n\n## 8. Discussion")
SECTION_TEXT = {"7.24": V32[_i24:_i25], "7.25": V32[_i25:_i26], "7.26": V32[_i26:_iend]}
scope_rows, scope_missing = [], []
for sec, text in SECTION_TEXT.items():
    own = {}
    for name in SECTION_SOURCES[sec]:
        for t in {tt for line in SOURCES[name].split("\n") for tt in toks(line)}:
            own.setdefault(t, []).append(name)
    seen = set()
    for line in text.split("\n"):
        if line.startswith("|") and sec in ("7.24", "7.25"):
            continue  # verbatim copies of RESULT lines, checked as such
        for t in toks(line):
            if t in seen:
                continue
            seen.add(t)
            if t not in own:
                scope_rows.append((sec, t))
                if (sec + ":" + t) not in SCOPE_OK:
                    scope_missing.append(sec + ":" + t)
if scope_missing:
    print("TOKENS NOT FOUND IN THEIR OWN SECTION'S SOURCES, UNREVIEWED (%d):" % len(scope_missing), scope_missing)
    sys.exit(3)
stale_scope = [k for k in SCOPE_OK if tuple(k.split(":", 1)) not in set(scope_rows)]
assert not stale_scope, stale_scope

# ---- claims taken from a PREREG (or other non-RESULT source), resolved to lines ------------------------------------------------
CLAIMS = [
    ("7.24: 'two readings were left', the 'no training of the driving model' framing, the read-out recipe, PCA-512 / 14,336 dims, folds, PRIMARY rule",
     "PREREG_V8F_SPEEDREAD.md", ["Two readings are left", "**The vector (fixed now, never tuned).**", "PRIMARY** = the one of LIN / MLP"]),
    ("7.24: substitution hook, the seven modes, tier 'T1-class open loop', estimator", "PREREG_V8F_SPEEDREAD.md",
     ["**The substitution read (the decision instrument).**", "Modes: `own, identity, gt, read, alt, const, perm`.", "Tier: **T1-class open loop**"]),
    ("7.24: Amendment 1 (22 dims, instrument sha256 change, nothing else moves)", "PREREG_V8F_SPEEDREAD.md",
     ["**The defect.**", "### 2. What does not change", "The instrument's sha256 changes from"]),
    ("7.24: self-test 27 passed and 19 / 19 mutants killed (INHERITED)", "PREREG_V8F_SPEEDREAD.md", ["**27 passed**", "**19 / 19 KILLED**"]),
    ("7.25: setup, promotion, ROBUST-row clause, Read A reference", "PREREG_V8F_DAGGER2.md",
     ["Promoted 2026-10-27 by the Master Mind from the draft banked at `5eaa0232`", "**Why it runs.**", "Read A measured M1 and L2 separated-worse"]),
    ("7.25: arms, crawl set C, R-CRAWL, R-MOVE, guards, exemption, outcome table", "PREREG_V8F_DAGGER2.md",
     ["**R-CRAWL:**", "**R-MOVE:**", "R-T0 point ≤ +F/2 = 0.0604 m", "Accel and DK-VIOL are exempt from this", "| **HARMS** |"]),
    ("7.25: Amendment 1 (4 unplaced cells of 16, 12 placed keep their outcome, one-seed R-MOVE is PARTIAL)", "PREREG_V8F_DAGGER2.md",
     ["**4 cells match no row**", "Each of the 12 cells that were placed keeps its outcome", "A one-seed R-MOVE is reported as PARTIAL (one seed)"]),
    ("7.26: why it runs, the PREREG's case (INHERITED), arms, composition semantics", "PREREG_V8F_DAG_ORC.md",
     ["**It is the only training lever measured to move two crawl rows at both seeds.**", "| **A** | DAG-OACC_s's training argv **verbatim**",
      "the swap is drawn first;", "no sample is ever both swapped and perturbed"]),
    ("7.26: floors RF0 / RF, bar_r, clears / fires, R rows and guards, D_s, outcome table, enumeration", "PREREG_V8F_DAG_ORC.md",
     ["| **RF0_r** |", "| **bar_r** |", "**clears (R-MOVE, per row):**", "| **R-KEEP** |", "**The domain D_s** is every Read-A row", "**The table is complete and exclusive (MEASURED"]),
    ("7.26: Amendment 1 re-base (DAG-OACC read LONGITUDINAL FIXED; open rows; CTRL banked; REP new; B only on HARMS with its own PI go)", "PREREG_V8F_DAG_ORC.md",
     ["`PREREG_V8F_DAG_OACC` read **LONGITUDINAL FIXED**", "yaw-rate MAE, curvature MAE, L2, M1", "| **REP** | DAG-OACC_s's argv with `--seed 10`", "Trained **only on A's HARMS, with its own PI go**"]),
    ("7.26: NEXT-CRAWL / LAUNCH-INTENT, N0 NOT NAVMOVE, COMPOSE done", "PREREG_V8F_DAG_ORC.md",
     ["N0 already read NOT NAVMOVE.", "COMPOSE is done: DAG-OACC − DAG is not separated-better on the rows of C.", "**LAUNCH-INTENT:**"]),
    ("7.26: r1 K-REACH stop, the defect, Amendment 2 fix and proofs, cost", "PREREG_V8F_DAG_ORC.md",
     ["**K-REACH failed:**", "**The defect.**", "On the composed path only, the perturbation's generator takes a domain-separated seed", "r1 used box 2 from 01:16 to 02:10"]),
    ("7.26: r2's first launch refused at its run record; the marker fix; cost; class", "R2A_NOTE.md",
     ["## r2's first launch (2026-10-28 03:27–03:40)", "**The fix.**", "**Cost.**", "**The class.**"]),
    ("7.26: the relaunch time 03:48 and the T1-kin chain launch", "git log (c7963d72, 8c24823f, 01f3d34e)", ["relaunched 03:48", "The T1-kin chain (two-step K-BUILD, 153/153) launched."]),
    ("7.26: the T1-kin chain launch at 03:50 (a Master Mind note outside the repo, INHERITED)", "MM_STATE_2026-10-28.md (E:, not in the repo)", ["T1-kin chain LAUNCHED 03:50"]),
    ("7.26: r2b's A s1 refused at its run record (a self-test replaced the box-2 queue), the repair, the pinned relaunch r2c, the two classes (the note is INHERITED, tagged so in the text)", "R2C_NOTE.md",
     ["05:56:16: a dry-run self-test", "Of 65 paths, 62 were byte-identical", "cmd.exe then resumed the queue", "The run-record check refused A s1", "06:04:09: r2's own tar", "agent/dagorc-r2-pin",
      "r2c relaunched at 06:09", "**Cost.**", "**The classes.**"]),
    ("7.26: the controls K-GATE, K-PAIR, K-INIT, K-REACH, K-BUILD, K-P0, K-ZERO, K-RC as registered", "PREREG_V8F_DAG_ORC.md",
     ["* **K-GATE.** Each new run", "* **K-PAIR.**", "* **K-INIT.** Every new run", "* **K-REACH.** `ego_perturb_n", "* **K-BUILD.** The launch commit", "* **K-P0.** RF0's inputs", "* **K-ZERO.** Two structural zeros", "* **K-RC.** The reader, run on DAG2"]),
    ("7.26: M-LAUNCH, the lever's mechanism row, and its reading", "PREREG_V8F_DAG_ORC.md", ["* **M-LAUNCH**, the lever's mechanism row", "If R-MOVE fails while M-LAUNCH moves"]),
]
claim_rows = []
for what, src, phrases in CLAIMS:
    text = SOURCES[src].split("\n")
    refs = []
    for ph in phrases:
        hit = [k for k, line in enumerate(text, 1) if ph in line]
        assert hit, (src, ph)
        refs.append(f"{hit[0]}")
    claim_rows.append(f"| {what} | `{src}`:{','.join(refs)} |")

# ---- verbatim table lines ----------------------------------------------------------------------------------------------------
i24, i25, i26 = V32.index("### 7.24 "), V32.index("### 7.25 "), V32.index("### 7.26 ")
iend = V32.index("\n---\n\n## 8. Discussion")
sr_lines, d2_lines = set(SOURCES["RESULT_V8F_SPEEDREAD.md"].split("\n")), set(SOURCES["RESULT_V8F_DAGGER2.md"].split("\n"))
t24 = [l for l in V32[i24:i25].split("\n") if l.startswith("|")]
t25 = [l for l in V32[i25:i26].split("\n") if l.startswith("|")]
t26 = [l for l in V32[i26:iend].split("\n") if l.startswith("|")]
or_lines = set(SOURCES["RESULT_V8F_DAG_ORC.md"].split("\n"))
t26_res = [l for l in t26 if l in or_lines]
t26_auth = [l for l in t26 if l not in or_lines]
assert all(l in sr_lines for l in t24) and all(l in d2_lines for l in t25)
assert len(t26_res) >= 129, len(t26_res)  # the 12 RESULT tables are copied line for line: 12+21+21+10+4+20+10+7+5+6+7+6 = 129 lines (a few separator lines of authored tables coincide)

# ---- JSON cross-check: the decisive and the derived DAG-ORC numbers, recomputed from the read JSON and found in the text ---------------
import json

RESJ = R + "raw/2026-10-28-dag-orc-result/read/"
J = json.loads(blob(RESJ + "DAGORC_READ.json"))
RFJ = json.loads(blob(RESJ + "RF_REP.json"))
RA, FL = J["read_a"], J["floors"]
ROWS19 = list(RA["A_CTRL"]["s0"].keys())
CRAWL = ["yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"]
FAM = {"LONGITUDINAL": ["speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG"],
       "LATERAL": ["heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)"],
       "TACTICAL": ["TAC-LAT", "TAC-LON"], "STRATEGIC": ["S-POS@30", "S-POS@50", "S-POS@80"]}
assert len(ROWS19) == 19 and all(len(FAM_rows) for FAM_rows in FAM.values())


def sg(x):
    return ("+" if x >= 0 else "−") + "%.4f" % abs(x)


def cis(c):
    return "%s [%s, %s]" % (sg(c["delta"]), sg(c["lo"]), sg(c["hi"]))


def region(name):
    if name == "abstract":
        return V32[V32.index("## 1. Abstract"):V32.index("## 2. Introduction")]
    if name == "7.26":
        return V32[i26:iend]
    if name == "8.1":
        return V32[V32.index("### 8.1 "):V32.index("### 8.2 ")]
    if name == "8.3":
        return V32[V32.index("### 8.3 "):V32.index("### 8.4 ")]
    if name == "row 1w":
        return "\n".join(l for l in V32.split("\n") if l.startswith("| 1w |"))
    if name == "status":
        return V32.split("\n")[2]
    raise KeyError(name)


JSON_ROWS = []  # (what, JSON key path, value in the JSON, snippet, where found)


def jchk(what, path, value, snippet, where, must=True):
    ok = [w for w in where if snippet in region(w)]
    if must:
        assert len(ok) == len(where), ("JSON cross-check failed", what, path, value, snippet, [w for w in where if w not in ok])
    JSON_ROWS.append((what, path, value, snippet, ", ".join(ok)))


# the outcome and the flags
assert J["OUTCOME"]["OUTCOME"] == "HARMS" and J["A"]["OUTCOME"] == "HARMS" and J["A"]["adopts"] is False and J["cascade"]["final"] == "HARMS"
G = J["A"]["flags"]["guards_held_per_seed"]
assert G["s0"] == {"R-TLAT": True, "R-TLON": True, "R-ADE": False, "R-T0": False, "R-NEW": True} and G["s1"] == {"R-TLAT": True, "R-TLON": True, "R-ADE": True, "R-T0": False, "R-NEW": True}
assert J["A"]["flags"]["R-MOVE"] == {"s0": False, "s1": False} and J["A"]["flags"]["R-CRAWL"] == {"s0": False, "s1": False}
assert J["A"]["flags"]["R-KEEP"] == {"s0": False, "s1": True} and J["A"]["flags"]["R-NOBASE"] == {"s0": True, "s1": True}
JSON_ROWS.append(("outcome word, adopts false, guard flags per seed (R-ADE s0 false, R-T0 false at both seeds, the rest held), R-MOVE and R-CRAWL false at both seeds",
                  "OUTCOME.OUTCOME, A.adopts, A.flags.guards_held_per_seed, A.flags.R-MOVE, A.flags.R-CRAWL, A.flags.R-KEEP, A.flags.R-NOBASE", "HARMS; False; as listed", "**HARMS: A is not adopted**", "7.26"))
assert "**HARMS: A is not adopted**" in region("7.26")
for sd in ("s0", "s1"):
    r = J["rows"]["A"][sd]
    assert not any(r["R-NEW_fires"].values()) and len(r["domain"]) == 15 and not any(r["R-MOVE_rows"].values())
assert [k for k, v in J["rows"]["A"]["s0"]["R-CRAWL_rows"].items() if v] == ["yaw-rate MAE"] and not any(J["rows"]["A"]["s1"]["R-CRAWL_rows"].values())
JSON_ROWS.append(("R-NEW fires on 0 of 15 domain rows at each seed; R-MOVE rows all false (0 of 4); R-CRAWL rows true: yaw-rate at s0 only",
                  "rows.A.<seed>.R-NEW_fires, .domain, .R-MOVE_rows, .R-CRAWL_rows", "0 of 15; 0 of 4; 1 of 4 / 0 of 4", "held: 0 of 15 rows", "7.26"))
# ADE6 guards, the T0 points
jchk("R-ADE s0, A - DAG-OACC ADE6", 'read_a.A_CTRL.s0["ADE6"].{delta,lo,hi}', cis(RA["A_CTRL"]["s0"]["ADE6"]), "+0.1434 [+0.0538, +0.2342]", ["7.26", "abstract", "8.1", "8.3", "row 1w"])
jchk("R-ADE s1, A - DAG-OACC ADE6", 'read_a.A_CTRL.s1["ADE6"].{delta,lo,hi}', cis(RA["A_CTRL"]["s1"]["ADE6"]), "+0.0565 [−0.0115, +0.1255]", ["7.26", "abstract"])
jchk("R-T0 s0 point", "rows.A.s0.T0_point", "%.4f" % J["rows"]["A"]["s0"]["T0_point"], "+0.1106", ["7.26", "abstract", "8.1", "8.3", "row 1w"])
jchk("R-T0 s1 point", "rows.A.s1.T0_point", "%.4f" % J["rows"]["A"]["s1"]["T0_point"], "+0.0722", ["7.26", "abstract", "8.1", "8.3", "row 1w"])
# floors and bars of the crawl rows, the ADE6 replicate draw
for r, b, rf in (("yaw-rate MAE", "31.7106", "15.8553"), ("curvature MAE", "0.2648", "0.1324"), ("L2 curvature MAE (GT>=5.2)", "0.1740", "0.0870"), ("M1 crawl share", "0.0776", "0.0388")):
    assert "%.4f" % FL["bar"][r] == b and "%.4f" % FL["RF"][r] == rf and "%.4f" % RFJ["RF"][r] == rf and abs(FL["bar"][r] - 2 * FL["RF"][r]) < 1e-3, r
    jchk("bar of the crawl row " + r, 'floors.bar["%s"] (= 2 x floors.RF["%s"])' % (r, r), b, b + " on", ["7.26"])
    jchk("RF of the crawl row " + r, 'floors.RF["%s"], RF_REP.json RF["%s"]' % (r, r), rf, "(RF %s)" % rf, ["7.26"])
assert "%.4f" % RFJ["REP_minus_CTRL"]["s0"]["ADE6"]["delta"] == "0.4025" and abs(FL["RF"]["ADE6"] / FL["F"]["ADE6"] - 3.335) < 0.01
jchk("REP s0 - CTRL ADE6 (the RF of ADE6) and its multiple of F", 'RF_REP.json REP_minus_CTRL.s0["ADE6"].delta; floors.RF["ADE6"] / floors.F["ADE6"]', "0.4025; %.2f x F" % (FL["RF"]["ADE6"] / FL["F"]["ADE6"]), "+0.4025 m", ["7.26", "abstract"])
jchk("REP s0 - CTRL ADE6 as 3.33 x F", "floors.RF[ADE6] / floors.F[ADE6]", "%.2f" % (FL["RF"]["ADE6"] / FL["F"]["ADE6"]), "3.33 × F", ["7.26", "abstract"])
jchk("the s0 R-ADE cost as a share of the replicate draw", 'read_a.A_CTRL.s0["ADE6"].delta / floors.RF["ADE6"]', "%.1f %%" % (100 * RA["A_CTRL"]["s0"]["ADE6"]["delta"] / FL["RF"]["ADE6"]), "36 % of that replicate draw", ["7.26"])
# the crawl effects against the bar and the floor
imp = {(sd, r): -RA["A_CTRL"][sd][r]["delta"] for sd in ("s0", "s1") for r in CRAWL}
assert all(RA["A_CTRL"][sd][r]["separated_better"] for sd in ("s0", "s1") for r in CRAWL)
pb = {k: 100 * v / FL["bar"][k[1]] for k, v in imp.items()}
pr = {k: 100 * v / FL["RF"][k[1]] for k, v in imp.items()}
assert "%.1f" % min(pb.values()) == "8.8" and "%.1f" % max(pb.values()) == "33.8" and "%.1f" % min(pr.values()) == "17.5" and "%.1f" % max(pr.values()) == "67.5"
assert pb[("s0", "yaw-rate MAE")] < 9 and max(pb, key=pb.get) == ("s0", "M1 crawl share") and all(v < 100 for v in pb.values())
jchk("improvement / bar over the 8 crawl cells: min 8.8 %, max 33.8 % (M1 at s0)", 'read_a.A_CTRL.<seed>[<crawl row>].delta / floors.bar[<crawl row>]', "%.1f to %.1f %%" % (min(pb.values()), max(pb.values())), "8.8–33.8 %", ["7.26", "abstract", "8.1", "8.3", "row 1w"])
jchk("|A - CTRL| / RF over the 8 crawl cells", 'read_a.A_CTRL.<seed>[<crawl row>].delta / floors.RF[<crawl row>]', "%.1f to %.1f %%" % (min(pr.values()), max(pr.values())), "17.5 % to 67.5 % of RF", ["7.26"])
# B must exceed A by bar / improvement
fac = {k: FL["bar"][k[1]] / v for k, v in imp.items()}
assert "%.1f" % min(fac.values()) == "3.0" and "%.1f" % max(fac.values()) == "11.4"
jchk("bar / A's improvement over the 8 crawl cells (the factor B must exceed A by)", 'floors.bar[<crawl row>] / -read_a.A_CTRL.<seed>[<crawl row>].delta', "%.1f to %.1f x" % (min(fac.values()), max(fac.values())), "3.0–11.4 ×", ["7.26", "abstract"])
# the share of the CTRL - P10 gap closed
sh = {}
for sd in ("s0", "s1"):
    for r in CRAWL:
        c, a_ = RA["CTRL_P10"][sd][r]["delta"], RA["A_P10"][sd][r]["delta"]
        sh[(sd, r)] = 100 * (c - a_) / c
assert "%.1f" % min(sh.values()) == "27.6" and "%.1f" % max(sh.values()) == "79.9"
jchk("share of the CTRL - P10 crawl gap that A closes over the 8 crawl cells", 'read_a.CTRL_P10 and read_a.A_P10, <seed>[<crawl row>].delta: (CTRL-P10 - (A-P10)) / (CTRL-P10)', "%.1f to %.1f %%" % (min(sh.values()), max(sh.values())), "27.6 % to 79.9 %", ["7.26"])
jchk("the same range in the 8.1 item", "(same)", "%.1f to %.1f %%" % (min(sh.values()), max(sh.values())), "27.6–79.9 %", ["8.1"])
sw = sum(1 for sd in ("s0", "s1") for r in CRAWL if RA["A_P10"][sd][r]["separated_worse"])
assert sw == 7
jchk("crawl cells A - P10 separated-worse: 7 of 8 (yaw-rate at s0 is the one that is not)", 'read_a.A_P10.<seed>[<crawl row>].separated_worse', "%d of 8" % sw, "seven of the eight", ["7.26", "8.1"])
# the replicate: separated counts, which seed sets RF, which term sets the bar
RC = RA["REP_CTRL"]
n0, n1 = sum(RC["s0"][r]["separated"] for r in ROWS19), sum(RC["s1"][r]["separated"] for r in ROWS19)
assert (n0, n1) == (15, 2)
jchk("REP - CTRL separated rows at s0 and at s1 (of 19)", 'read_a.REP_CTRL.<seed>[<row>].separated', "%d and %d" % (n0, n1), "15 of 19 rows at s0 and 2 of 19 at s1", ["7.26"])
jchk("REP - CTRL separated rows at s0 (the 8.1 item 14 wording)", 'read_a.REP_CTRL.s0[<row>].separated', str(n0), "separated on 15 of 19 rows", ["8.1"])
s0set = sum(1 for r in ROWS19 if abs(RC["s0"][r]["delta"]) >= abs(RC["s1"][r]["delta"]))
assert s0set == 17
jchk("REP s0 sets RF (the larger |REP - CTRL| of the two seeds) on 17 of 19 rows", 'read_a.REP_CTRL.<seed>[<row>].delta', "%d of 19" % s0set, "REP s0 sets RF on 17 of 19 rows", ["7.26"])
jchk("REP s0 sets the floor on 17 of 19 rows (abstract)", 'read_a.REP_CTRL.<seed>[<row>].delta', "%d of 19" % s0set, "sets the floor on 17 of 19 rows", ["abstract"])
rf_bar = sum(1 for r in ROWS19 if abs(FL["bar"][r] - 2 * FL["RF"][r]) < 5e-4 and 2 * FL["RF"][r] >= 2 * FL["RF0"][r] - 1e-9 and 2 * FL["RF"][r] >= 2 * FL["F"][r] - 1e-9)
rf0_bar = [r for r in ROWS19 if abs(FL["bar"][r] - 2 * FL["RF0"][r]) < 5e-4 and 2 * FL["RF0"][r] > 2 * FL["RF"][r]]
assert rf_bar == 15 and sorted(rf0_bar) == sorted(["DK-VIOL", "cross-track MAE", "S-POS@50", "S-POS@80"])
jchk("the bar is 2 x RF on 15 rows and 2 x RF0 on DK-VIOL, cross-track MAE, S-POS@50, S-POS@80", "floors.bar, floors.RF, floors.RF0, floors.F", "15 and 4", "sets the bar on 15 of 19 rows", ["7.26"])
# the controls, by name
CP = J["controls_PASS"]
cnt = {}
for k in CP:
    cnt[k.split(":")[0]] = cnt.get(k.split(":")[0], 0) + 1
named = {"K-GATE": 6, "K-INIT": 6, "K-PAIR": 4, "K-REACH": 4, "K-ZERO": 2, "K-BUILD": 1, "K-P0": 1}
assert len(CP) == 162 and all(CP.values()) and all(cnt.get(k) == v for k, v in named.items()) and cnt["K-RC-DAG2"] == 1 and cnt["K-RC-OACC"] == 1
rest = len(CP) - sum(named.values()) - 2
assert rest == 136
jchk("controls_PASS: 162 keys, all true; K-GATE 6, K-INIT 6, K-PAIR 4, K-REACH 4, K-ZERO 2, K-BUILD 1, K-P0 1, K-RC 2, the other 136",
     "controls_PASS (counted by the key prefix before ':')", "162; " + ", ".join("%s %d" % kv for kv in named.items()) + ", K-RC 2, other %d" % rest,
     "K-GATE 6, K-INIT 6, K-PAIR 4, K-REACH 4, K-ZERO 2, K-BUILD 1, K-P0 1 and K-RC 2", ["7.26"])
jchk("the 136 further groups", "controls_PASS", str(rest), "136 further groups", ["7.26"])
# K-REACH shares
PC = J["perturbation_counts"]
for key, num, den, share in (("A_s0", 22963, 76898, "0.2986"), ("A_s1", 22970, 76435, "0.3005")):
    assert PC[key]["ego_perturb_n"] == num and PC[key]["ego_perturb_eligible"] == den and "%.4f" % (num / den) == share and PC[key]["ego_dagger_overlap_n"] == 0
assert PC["REP_s0"]["ego_perturb_n"] == 0 and PC["REP_s1"]["ego_perturb_n"] == 0
sw_ = [PC[k]["dagger_n"] / PC[k]["dagger_eligible"] for k in ("A_s0", "REP_s0", "A_s1", "REP_s1")]
assert "%.4f" % min(sw_) == "0.0999" and "%.4f" % max(sw_) == "0.1013"
jchk("K-REACH: perturbation share A s0 0.2986, A s1 0.3005, REP exactly 0; swap share 0.0999 to 0.1013; overlap 0", "perturbation_counts.<arm_seed>.{ego_perturb_n, ego_perturb_eligible, dagger_n, dagger_eligible, ego_dagger_overlap_n}",
     "0.2986, 0.3005, 0, 0; %.4f to %.4f; 0" % (min(sw_), max(sw_)), "0.2986 (A s0) and 0.3005 (A s1)", ["7.26"])
jchk("K-REACH swap share range", "perturbation_counts.<arm_seed>.dagger_n / dagger_eligible", "%.4f to %.4f" % (min(sw_), max(sw_)), "between 0.0999 and 0.1013", ["7.26"])
# not run / not built, the tier strings, the ruling
assert J["t0_four_families"]["status"] == "NOT RUN" and all(v.endswith("NOT BUILT") for v in J["beside_not_built"].values()) and set(J["beside_not_built"]) == {"M-LAUNCH", "anti_echo_block", "M2_and_L2_launch_subset"}
JSON_ROWS.append(("t0_four_families.status NOT RUN; beside_not_built: M-LAUNCH, anti_echo_block, M2_and_L2_launch_subset all NOT BUILT", "t0_four_families.status, beside_not_built", "NOT RUN; NOT BUILT x3", "`t0_four_families.status` reads **NOT RUN**", "7.26"))
assert "`t0_four_families.status` reads **NOT RUN**" in region("7.26") and "`M2_and_L2_launch_subset`" in region("7.26") and "`M-LAUNCH`" in region("7.26") and "`longitudinal_claim_admissible`" in region("7.26")
assert J["four_families"]["A"]["s0"]["STRATEGIC"]["ruling"] == "NOT paused for v8F (PI ruling 2026-10-17)" and "NOT paused for v8F (PI ruling 2026-10-17)" in region("7.26")
JSON_ROWS.append(("the STRATEGIC ruling string", "four_families.A.s0.STRATEGIC.ruling", J["four_families"]["A"]["s0"]["STRATEGIC"]["ruling"], "NOT paused for v8F (PI ruling 2026-10-17)", "7.26"))
jchk("reader wall time and the K-BUILD record", "provenance.wall_s; k_build.changed (37 hunks); k_build.launch_commit; k_build.base_commit", "%s s; %d; %s; %s" % (J["provenance"]["wall_s"], len(J["k_build"]["changed"]), J["k_build"]["launch_commit"][:8], J["k_build"]["base_commit"][:8]),
     "275.4 s", ["7.26"])
assert len(J["k_build"]["changed"]) == 37 and J["k_build"]["launch_commit"].startswith("8c24823f") and J["k_build"]["base_commit"].startswith("aeb54468")
assert J["provenance"]["reader_sha256"].startswith("2e5f9f17") and J["provenance"]["rf0_sha256"].startswith("1608d70b") and J["floors"]["RF_file"]["sha256"].startswith("64c9444d")
JSON_ROWS.append(("reader, RF0 and RF_REP sha256 prefixes", "provenance.reader_sha256, provenance.rf0_sha256, floors.RF_file.sha256", "2e5f9f17, 1608d70b, 64c9444d", "sha256 `2e5f9f17…`", "7.26"))
# by family (item 5 of 'What the numbers say'): the separated cells of A - CTRL, derived from the JSON
def sepcells(fam, kind):
    return sorted((r, sd) for r in FAM[fam] for sd in ("s0", "s1") if RA["A_CTRL"][sd][r][kind])
assert sepcells("LONGITUDINAL", "separated_worse") == sorted([("along-track MAE", "s0"), ("DK-VIOL", "s0"), ("accel MAE", "s1")]) and not sepcells("LONGITUDINAL", "separated_better")
assert sepcells("LATERAL", "separated_better") == sorted((r, sd) for r in ("heading MAE", "yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)") for sd in ("s0", "s1"))
assert sepcells("LATERAL", "separated_worse") == [("cross-track MAE", "s0"), ("cross-track MAE", "s1")]
assert sepcells("TACTICAL", "separated_better") == [("TAC-LAT", "s0"), ("TAC-LAT", "s1")] and not sepcells("TACTICAL", "separated_worse")
assert sepcells("STRATEGIC", "separated_worse") == [("S-POS@30", "s0")] and not sepcells("STRATEGIC", "separated_better")
JSON_ROWS.append(("item 5, by family, A - CTRL: the separated cells (LONGITUDINAL worse: along-track s0, DK-VIOL s0, accel s1; LATERAL better: heading, yaw-rate, curvature, L2 at both seeds, worse: cross-track at both; "
                  "TACTICAL better: TAC-LAT at both; STRATEGIC worse: S-POS@30 at s0)", 'read_a.A_CTRL.<seed>[<row>].{separated_better, separated_worse}', "as listed (asserted)", "**By family, A − CTRL at T1-kin**", "7.26"))
assert "**By family, A − CTRL at T1-kin**" in region("7.26")

md = []

md.append("# NUMBERS_AUDIT: report v32 (v31 + section 7.24 SPEEDREAD, 7.25 DAGGER2, 7.26 DAG-ORC)\n")
md.append("Audited deliverable (sha256): `V8F_SCIENTIFIC_REPORT.md` %s, built on source tip `%s`.\n" % (hashlib.sha256(open(OUT + "V8F_SCIENTIFIC_REPORT.md", "rb").read()).hexdigest(), TIP[:8]))
md.append("Method (mechanical, `audit_v32.py`): the v32 md was diffed against the v31 md on the tip (`%s`); every added or changed line was tokenised for numbers"
          " (sign kept, the Unicode minus read as `-`, thousands commas removed); each distinct signed token was looked up line by line in the source files below."
          " `v32:N` are line numbers in `out/V8F_SCIENTIFIC_REPORT.md`. Sources are read from the blobs at the tip, except the one marked as outside the repo.\n" % TIP)
md.append("| source | role |\n|---|---|")
roles = {
    "RESULT_V8F_SPEEDREAD.md": "the SPEEDREAD result (primary source for 7.24, the abstract, 8.1 item 12, 8.3, row 1x); its tables are copied line for line",
    "RESULT_V8F_DAGGER2.md": "the DAGGER2 result (primary source for 7.25, 8.1 item 13, row 1y); its tables are copied line for line",
    "PREREG_V8F_SPEEDREAD.md": "the design and Amendment 1 of SPEEDREAD (wording and constants, not results)",
    "PREREG_V8F_DAGGER2.md": "the design and Amendment 1 of DAGGER2 (wording and constants, not results)",
    "PREREG_V8F_DAG_ORC.md": "the DAG-ORC design, Amendment 1 (re-base) and Amendment 2 (the shared stream, with its proofs and the r1 stop); wording and constants, not results",
    "RESULT_V8F_DAG_ORC.md": "the DAG-ORC result, landed `ea5e3d3b` (primary source for 7.26, the abstract, 8.1 item 14, 8.3, row 1w); its tables are copied line for line",
    "R2C_NOTE.md": "the A s1 incident of r2b (a self-test replaced the box-2 queue) and the pinned relaunch r2c",
    "R2A_NOTE.md": "r2's first launch, refused at its run record",
    "r1_refused/summary.json": "r1's refused run record (`ego_perturb_n` 19161, `ego_perturb_eligible` 76898, overlap 0)",
    "r2/selftest_chain_r2.out": "the r2 chain self-test (66 / 66 tests, 224 / 224 mutants)",
    "t1/SELFTEST.out": "the dev-box T1-kin chain self-test (153 / 153 checks)",
    "relaunch/box2_aside_10280348.log": "the relaunch's box-2 aside (22 items moved, job end 03:48:22)",
    "t1/kbuild_r2_launch_8c24823f.json": "the K-BUILD record of the r2 launch commit (base commit `aeb54468`)",
    "git log (c7963d72, 8c24823f, 01f3d34e)": "commit messages: the relaunch time 03:48 and the T1-kin chain launch (INHERITED)",
    "MM_STATE_2026-10-28.md (E:, not in the repo)": "the Master Mind's state note: the T1-kin chain launch at 03:50 (INHERITED; single location)",
}
for name in SOURCES:
    md.append(f"| `{name}` | {roles[name]} |")
n_added = len(added)
md.append("\n**Totals.** %d added or changed lines (%d v31 lines replaced or removed); %d distinct signed numeric tokens (unsigned single digits 0-9 excluded as enumerations);"
          " %d found with the same sign in a source line (section A); %d found only with a different or absent sign (section B, each reviewed by hand);"
          " %d found only in the v31 report (section C); %d found nowhere (section D).\n" % (n_added, len(removed), len(tok_lines), len(A), len(B), len(C), len(D)))
md.append("## A. Tokens found with the same sign in a source (token -> v32 lines -> source:lines)\n")
md.append("| token | v32 lines | source lines |\n|---|---|---|")
for t, lns, sd in A:
    md.append("| `%s` | %s | %s |" % (t, ",".join(str(x) for x in lns[:5]) + ("..." if len(lns) > 5 else ""), span(sd)))
md.append("\n## B. Found only with a different sign or none (sign is a property of the sentence, reviewed by hand in E)\n")
md.append("| token | v32 lines | unsigned form found at |\n|---|---|---|")
for t, lns, sd in B:
    md.append("| `%s` | %s | %s |" % (t, ",".join(str(x) for x in lns[:5]), span(sd, 3)))
md.append("\n## C. Found only in the v31 report (carried context, not a v32 source claim)\n")
md.append("| token | v32 lines | v31 lines |\n|---|---|---|")
for t, lns, v in C:
    md.append("| `%s` | %s | %s |" % (t, ",".join(str(x) for x in lns[:5]), ",".join(str(x) for x in v[:5])))
md.append("\n## D. Found nowhere (must be structural: section numbers, dates, identifiers, derived values)\n")
md.append("| token | v32 lines |\n|---|---|")
for t, lns in D:
    md.append("| `%s` | %s |" % (t, ",".join(str(x) for x in lns[:5])))
md.append("\n## E. Hand review of sections B, C and D\n")
md.append("| token | review |\n|---|---|")
for t in need:
    md.append("| `%s` | %s |" % (t, REVIEWED[t]))
md.append("\n## F. Claims taken from a PREREG, a log or a note, not from a RESULT (wording and constants, no result numbers)\n")
md.append("| claim in the report | source:lines |\n|---|---|")
md.extend(claim_rows)
md.append("\nThe DAG-ORC result numbers (section 7.26, the abstract, 8.1 item 14, 8.3, row 1w) are taken from `RESULT_V8F_DAG_ORC.md` (tables copied line for line; prose numbers are tokens of the RESULT, section A)"
          " and the decisive and derived ones are recomputed from the read JSON in section H. The remaining digits attached to DAG-ORC are the PREREG's committed constants (0.3, 0.1, 0.0604, 0.1207, 2,"
          " 524,288 and its row counts), the r1 run record (19,161 / 76,898 = 0.2492), the Amendment 2 proofs, hashes and amendment ids, and the run-history times from the notes.\n")
md.append("## G. Section-scoped check (a number in the prose of 7.24 / 7.25 / 7.26 must occur in that section's own sources)\n")
md.append("Section 7.24 is checked against `RESULT_V8F_SPEEDREAD.md` and `PREREG_V8F_SPEEDREAD.md`; 7.25 against `RESULT_V8F_DAGGER2.md` and `PREREG_V8F_DAGGER2.md`;"
          " 7.26 against the DAG-ORC RESULT and PREREG, the r2a and r2c notes, the r1 run record, the two self-test logs, the aside log, the K-BUILD record, the commit messages and the Master Mind's note."
          " Table lines of 7.24 and 7.25 are verbatim copies and are not re-tested here. %d tokens are not found in their own section's sources; each is reviewed:\n" % len(scope_rows))
md.append("| section:token | review |\n|---|---|")
for sec, tk in scope_rows:
    md.append("| `%s:%s` | %s |" % (sec, tk, SCOPE_OK[sec + ":" + tk]))
md.append("")
md.append("## H. JSON cross-check of the decisive and the derived DAG-ORC numbers\n")
md.append("`DAGORC_READ.json` (sha256 `70a7a3a4…`) and `RF_REP.json` (`64c9444d…`) are read from the tip (`raw/2026-10-28-dag-orc-result/read/`). Each row below is recomputed from the JSON by `audit_v32.py`"
          " (an assertion fails the script on a mismatch) and the quoted snippet is found in the named part of v32. Derived values (shares, ranges, counts) are computed here, not copied: this closes the"
          " gap that a token match in section A cannot see (a derived number can coincide with an unrelated number of a source).\n")
md.append("| what | JSON key path | value in the JSON | snippet in v32 | found in |\n|---|---|---|---|---|")
for what, path, val, snip, where in JSON_ROWS:
    md.append("| %s | `%s` | %s | `%s` | %s |" % (what.replace("|", "/"), path.replace("|", "/"), str(val).replace("|", "/"), snip.replace("|", "/"), where))
md.append("")
md.append("## I. Mechanical checks run\n")
md.append("* Every table line inside 7.24 (%d lines) is a line of `RESULT_V8F_SPEEDREAD.md`, and inside 7.25 (%d lines) of `RESULT_V8F_DAGGER2.md`, byte for byte"
          " (`patch_v32.py` extracts each table by an anchored heading and asserts membership in the source). Of the %d table lines in 7.26, %d are lines of `RESULT_V8F_DAG_ORC.md`, byte for byte"
          " (12 tables, extracted the same way), and %d are authored tables (arms, outcome table, families), traced in F to the PREREG." % (len(t24), len(t25), len(t26), len(t26_res), len(t26_auth)))
md.append("* Derived values written in the report, not found in a source as such: the mean over the five cells of the fold-0 hold-out RMSE (0.2724, 0.2920, 0.2807, 0.2475; the source"
          " prints 0.2724 and 0.2920 and 0.2475 in prose, and the arithmetic mean of the five table cells reproduces each of the four), see E.")
md.append("* Banned words, closed-loop wording and tier stamps: see `v32_checks.txt`.")
open(AUDIT_PATH, "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
print("audit written:", len(tok_lines), "tokens; A", len(A), "B", len(B), "C", len(C), "D", len(D))
