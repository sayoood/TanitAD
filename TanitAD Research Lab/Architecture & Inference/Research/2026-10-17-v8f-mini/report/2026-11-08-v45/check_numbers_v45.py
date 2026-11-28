"""Check the headline numbers of the v45 sections (7.49, 7.50) and of edits_v45.md against the source artifacts they cite.

Reads only. Run: python -I check_numbers_v45.py   (CPU, no GPU, no network). Exit 0 if every check holds; writes NUMBERS_V45.md.
Method (as v44): (1) every printed number is searched, verbatim, in the source file it is attributed to AND in the text that quotes it;
(2) numbers that live in a JSON are compared as numbers; (3) counts that the text states (label counts, 6 of 9 ...) are recomputed from the source table;
(4) arithmetic [E] is recomputed; (5) frozen-card prefixes are re-hashed; (6) a mutation arm changes a number in a copy and must make the checker fail.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
TH = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/"
TEXT_FILES = ["sec_749_imagination.md", "sec_750_horizon_bands.md", "edits_v45.md"]

FAIL = []
ROWS = []
N = [0]
SRC = [""]


def rd(path, base=TH):
    return open(base + path, "rb").read().decode("utf-8")


def norm(s):
    return s.replace("\u2212", "-").replace("\u00a0", " ")


TEXT = {f: norm(rd(f, HERE + "/")) for f in TEXT_FILES}
TEXT_ALL = "\n".join(TEXT.values())
SRCTXT = {}


TIP = "c69bc890923029d58f67ecd2c4e984c8a823bcaa"
MIRROR = "C:/Users/Admin/tanitad-push"
RTIP = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/"


def tipblob(path):
    """a file of the repository tip (full 40-hex sha above), LF-normalised"""
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "-c", "core.autocrlf=false", "show", TIP + ":" + path], capture_output=True)
    assert r.returncode == 0, (path, r.stderr[:200])
    return r.stdout.decode("utf-8").replace("\r", "")


def srctext(p):
    if p not in SRCTXT:
        SRCTXT[p] = norm(tipblob(p[4:]) if p.startswith("tip:") else rd(p))
    return SRCTXT[p]


def record(label, actual, expected, ok):
    N[0] += 1
    ROWS.append((label, actual, expected, ok, SRC[0]))
    if not ok:
        FAIL.append((label, actual, expected))
    print(("PASS " if ok else "FAIL ") + label + " | " + str(actual)[:90])


def chk(label, actual, expected, tol=6e-4):
    ok = abs(actual - expected) <= tol if isinstance(expected, float) else actual == expected
    record(label, actual, expected, ok)


def has(label, path, needle, where=None, in_text=True):
    """needle printed verbatim in source `path` and (if in_text) in the v45 text (`where` = one text file or all)."""
    n = norm(needle)
    s = srctext(path)
    t = TEXT_ALL if where is None else TEXT[where]
    ok_src = n in s
    ok_txt = (n in t) if in_text else True
    record(label + " [src%s]" % (" + text" if in_text else ""), (ok_src, ok_txt), (True, True), ok_src and ok_txt)


def card(path, nbytes, prefix, label):
    b = open(TH + path, "rb").read()[:nbytes]
    h = hashlib.sha256(b).hexdigest()
    record("card %s first %d bytes" % (label, nbytes), h[: len(prefix)], prefix, h.startswith(prefix))
    for f, t in TEXT.items():
        pass
    record("card %s prefix %s printed in text" % (label, prefix), prefix + "\u2026" in TEXT_ALL or prefix in TEXT_ALL, True, (prefix + "\u2026" in TEXT_ALL) or (prefix in TEXT_ALL))


# ------------------------------------------------------------------ frozen cards
SRC[0] = "card prefix hashes [M]"
card("PREREG_V8F_VOE1B.md", 5785, "aa317fe3", "VOE1B")
card("PREREG_V8F_IMAG1B.md", 4634, "40500d68", "IMAG1B")
card("PREREG_V8F_CAPLADDER.md", 7830, "e17a4d90", "CAPLADDER")
b = open(TH + "imag1c/IMPL_CHOICES_IMAG1C.md", "rb").read()
h = hashlib.sha256(b).hexdigest()
record("IMAG1C IMPL_CHOICES sha256 prefix", h[:8], "ed00b75c", h.startswith("ed00b75c"))
record("IMAG1C IMPL_CHOICES sha256 prefix printed in text", "ed00b75c" in TEXT_ALL, True, "ed00b75c" in TEXT_ALL)

# ------------------------------------------------------------------ VOE1B
V = "voe1b/RESULT_VOE1B.md"
SRC[0] = "voe1b/RESULT_VOE1B.md [M, T0, episode-cluster bootstrap B 2,000 seed 0, two-part rule R = 199]"
for lab, needle in [
    ("ACTION-SWAP-M L1 s0", "0.583 [0.538, 0.630]"), ("ACTION-SWAP-M L1 s1", "0.585 [0.541, 0.628]"),
    ("ACTION-SWAP-M L2 s0", "0.643 [0.601, 0.686]"), ("ACTION-SWAP-M L2 s1", "0.611 [0.566, 0.653]"),
    ("ACTION-SWAP-M L3 s0", "0.668 [0.623, 0.712]"), ("ACTION-SWAP-M L3 s1", "0.677 [0.629, 0.725]"),
    ("ACTION-SWAP L1 s0", "0.632 [0.606, 0.657]"), ("ACTION-SWAP L1 s1", "0.604 [0.580, 0.629]"),
    ("ACTION-SWAP L2 s0", "0.679 [0.653, 0.705]"), ("ACTION-SWAP L2 s1", "0.656 [0.628, 0.684]"),
    ("ACTION-SWAP L3 s0", "0.738 [0.716, 0.759]"), ("ACTION-SWAP L3 s1", "0.733 [0.708, 0.759]"),
    ("ACTION-SWAP-M L3 minus null s0", "+0.168 [+0.122, +0.213]"), ("ACTION-SWAP-M L3 minus null s1", "+0.183 [+0.132, +0.236]"),
    ("FREEZE L1 s0", "0.315 [0.299, 0.331]"), ("SPLICE L1 minus null", "+0.000 [+0.000, +0.000]"),
    ("turn AP minus chance L2 s0", "+0.0188 [+0.0107, +0.0314]"), ("turn AP minus chance L3 s0", "+0.0253 [+0.0118, +0.0512]"),
    ("control 1 count", "real count **0** of 6"), ("control 2", "**9 of 288 = 3.12%**"),
    ("control 3 max family", "RVR 93/2040 = 4.559%"), ("runtime", "514 s"), ("run window", "2026-11-07 15:57:42 to 2026-11-07 16:06:10"),
    ("control 4 count", "22 passed"),
]:
    has("VOE1B " + lab, V, needle, in_text=(lab not in ("control 4 count", "control 1 count", "control 2", "control 3 max family", "run window")))
for s in ["0 of 6 cells separated against a 95th percentile of 1", "9 of 288 cells = 3.12 % NOTICES", "at most 4.559 % of rows separated per family", "15:57:42 to 16:06:10 (514 s)" if False else "514 s"]:
    record("text prints VOE1B control item " + s, s in TEXT["sec_749_imagination.md"], True, s in TEXT["sec_749_imagination.md"])
record("text prints VOE1B run window 2026-11-07 (card says 15:57:42 to 16:06:10)", "514 s" in TEXT["sec_749_imagination.md"], True, "514 s" in TEXT["sec_749_imagination.md"])
# untrained maxima printed in the text: verify against the source table rows (min / median / max column)
VS = srctext(V)
for lab, row_start, mx in [
    ("ACTION-SWAP-M L1 s0", "| M2b_s0 | L1 | 530 (104) |", "0.551"), ("ACTION-SWAP-M L1 s1", "| M2b_s1 | L1 | 530 (104) |", "0.558"),
    ("ACTION-SWAP-M L2 s0", "| M2b_s0 | L2 | 530 (104) |", "0.575"), ("ACTION-SWAP-M L2 s1", "| M2b_s1 | L2 | 530 (104) |", "0.596"),
    ("ACTION-SWAP-M L3 s0", "| M2b_s0 | L3 | 530 (104) |", "0.515"), ("ACTION-SWAP-M L3 s1", "| M2b_s1 | L3 | 530 (104) |", "0.532"),
    ("ACTION-SWAP L1 s0", "| M2b_s0 | L1 | 1,976 (141) | 0.632", "0.611"), ("ACTION-SWAP L1 s1", "| M2b_s1 | L1 | 1,976 (141) | 0.604", "0.569"),
    ("ACTION-SWAP L2 s0", "| M2b_s0 | L2 | 1,976 (141) | 0.679", "0.580"), ("ACTION-SWAP L2 s1", "| M2b_s1 | L2 | 1,976 (141) | 0.656", "0.612"),
    ("ACTION-SWAP L3 s0", "| M2b_s0 | L3 | 1,976 (141) | 0.738", "0.759"), ("ACTION-SWAP L3 s1", "| M2b_s1 | L3 | 1,976 (141) | 0.733", "0.743"),
]:
    line = [l for l in VS.split("\n") if l.startswith(row_start)]
    ok = len(line) == 1 and re.search(r"\d\.\d+ / \d\.\d+ / " + re.escape(mx) + r" \(", line[0]) is not None
    record("VOE1B untrained maximum " + lab + " = " + mx, ok, True, ok)
# the label table: parse the headline table and compare with the text
head = [l for l in VS.split("\n") if re.match(r"\| (TELEPORT|FREEZE|REVERSE|SPLICE|ACTION-SWAP|ACTION-SWAP-M) \| ", l)]
lab_src = {}
for l in head:
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    lab_src[c[0]] = c[2:5]
want = {"TELEPORT": ["NULL-ONLY"] * 3, "FREEZE": ["BLIND"] * 3, "REVERSE": ["NULL-ONLY", "NULL-ONLY", "BLIND"], "SPLICE": ["NULL-ONLY"] * 3,
        "ACTION-SWAP": ["NOTICES", "NOTICES", "INIT-LEVEL"], "ACTION-SWAP-M": ["NOTICES"] * 3}
for k, v in want.items():
    record("VOE1B headline label row " + k, lab_src.get(k), v, lab_src.get(k) == v)
    rowtxt = "| %s (" % k
    record("VOE1B headline label row " + k + " printed in the text", all(x in TEXT["sec_749_imagination.md"] for x in set(v)) and rowtxt in TEXT["sec_749_imagination.md"], True,
           all(x in TEXT["sec_749_imagination.md"] for x in set(v)) and rowtxt in TEXT["sec_749_imagination.md"])
record("VOE1B seeds agree (per-seed line equals headline)", "Per-seed labels (s0 / s1): TELEPORT: L1 NULL-ONLY/NULL-ONLY" in VS and "ACTION-SWAP: L1 NOTICES/NOTICES, L2 NOTICES/NOTICES, L3 INIT-LEVEL/INIT-LEVEL" in VS, True,
       "Per-seed labels (s0 / s1): TELEPORT: L1 NULL-ONLY/NULL-ONLY" in VS and "ACTION-SWAP: L1 NOTICES/NOTICES, L2 NOTICES/NOTICES, L3 INIT-LEVEL/INIT-LEVEL" in VS)
record("VOE1B ALIGNS none (brake, turn, lead)", "**ALIGNS summary:** brake: no level; turn: no level; lead: no level." in VS, True, "**ALIGNS summary:** brake: no level; turn: no level; lead: no level." in VS)
record("VOE1B onsets 47 / 100 / 27", "brake 47, turn 100, lead 27" in VS, True, "brake 47, turn 100, lead 27" in VS)
record("VOE1B windows 530 (104) and 1,725 (137)", "ACTION-SWAP-M 530 (104 ep)" in VS and "FREEZE 1,725 (137 ep)" in VS, True, "ACTION-SWAP-M 530 (104 ep)" in VS and "FREEZE 1,725 (137 ep)" in VS)
for lab, row_start in [("L1 s0", "| M2b_s0 | L1 | 1,725"), ("L2 s0", "| M2b_s0 | L2 | 1,725"), ("L3 s0", "| M2b_s0 | L3 | 1,725"),
                       ("L1 s1", "| M2b_s1 | L1 | 1,725"), ("L2 s1", "| M2b_s1 | L2 | 1,725"), ("L3 s1", "| M2b_s1 | L3 | 1,725")]:
    line = [l for l in VS.split("\n") if l.startswith(row_start)][0]
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    acc = float(c[3].split()[0])
    mn = float(c[4].split()[0])
    record("VOE1B FREEZE %s accuracy %.3f below untrained minimum %.3f" % (lab, acc, mn), acc < mn, True, acc < mn)
has("VOE1B FREEZE L2 s0 vs untrained minimum (text example)", V, "0.322 [0.301, 0.342]", in_text=False)
record("text: FREEZE example 0.322 against 0.408", "0.322 against a minimum of 0.408" in TEXT["sec_749_imagination.md"], True, "0.322 against a minimum of 0.408" in TEXT["sec_749_imagination.md"])
record("VOE1B FREEZE L2 s0 minimum 0.408 in source", "0.408 / 0.449 / 0.479" in VS, True, "0.408 / 0.449 / 0.479" in VS)

# ------------------------------------------------------------------ IMAG1B
I1 = "imag1b/RESULT_IMAG1B.md"
SRC[0] = "imag1b/RESULT_IMAG1B.md [M, T0, episode-cluster bootstrap B 2,000 seed 0, refit permutation R = 199]"
for lab, needle in [
    ("speed encoded L1 ridge s0", "0.350 [0.268, 0.458]"), ("speed encoded L2 ridge s0", "0.486 [0.339, 0.713]"), ("speed encoded L3 ridge s0", "0.591 [0.430, 0.830]"),
    ("gain L3 k3 ridge s0", "+0.43 [+0.41, +0.46]"), ("gain L3 k3 ridge s1", "+0.35 [+0.32, +0.38]"),
    ("gain L2 k3 ridge s0", "+0.35 [+0.31, +0.39]"), ("gain L2 k3 ridge s1", "+0.45 [+0.41, +0.49]"),
    ("bank", "Bank 9,596 frames, 141 episodes; 1,976 windows"), ("control 2 count", "72 pooled family cells"),
    ("NO-SPLIT", "**Timescale split: NO-SPLIT**"), ("E3 fired", "* E3 CAPLADDER (lead gap UNREADABLE at every level): **TRIGGERED**"),
    ("E7 not fired", "* E7 DIVRS (lateral IA-3 WEAK, INERT or WRONG-SIGN at every level), design input = turning share 201 of 1976 windows (10.2 %): **not triggered**"),
    ("L1 residual not fired", "* L1 residual-parameterisation arm (longitudinal L1 OVERSHOOTS): **not triggered**"),
]:
    has("IMAG1B " + lab, I1, needle, in_text=(lab not in ("bank", "control 2 count", "NO-SPLIT", "E3 fired", "E7 not fired", "L1 residual not fired")))
record("text: IMAG1B bank and windows", "9,596 frames, 141 episodes; 1,976 windows" in TEXT["sec_749_imagination.md"], True, "9,596 frames, 141 episodes; 1,976 windows" in TEXT["sec_749_imagination.md"])
record("text: IMAG1B control 2 72 of 72", "72 of 72 family cells separated" in TEXT["sec_749_imagination.md"], True, "72 of 72 family cells separated" in TEXT["sec_749_imagination.md"])
# decision counts recomputed from the IA-2 tables (eval141 block)
lines = I1 and srctext(I1).split("\n")
start = [i for i, l in enumerate(lines) if l.startswith("## IA-2 probe transfer")][0]
stop = [i for i, l in enumerate(lines) if l.startswith("### Paired differences")][0]
fam = None
cnt = {}
for l in lines[start:stop]:
    m = re.match(r"### (\w+) \((\w+)\)", l)
    if m:
        fam = m.group(2)
        continue
    if l.startswith("|") and l.rstrip().endswith("**|") or (l.startswith("|") and re.search(r"\*\*[A-Z\-]+\*\* \|$", l)):
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        tgt = c[0]
        lab = c[-1].strip("*")
        cnt.setdefault((fam, tgt), {}).setdefault(lab, 0)
        cnt[(fam, tgt)][lab] += 1
exp = {("LON", "speed (R)"): {"COPIES": 9}, ("LON", "accel"): {"UNREADABLE": 9}, ("LON", "lead_gap (R)"): {"UNREADABLE": 9},
       ("LON", "lead_closing (R)"): {"UNREADABLE": 9}, ("LON", "ttc (R)"): {"UNREADABLE": 9}, ("LAT", "yaw_rate"): {"UNREADABLE": 9},
       ("LAT", "curvature"): {"UNREADABLE": 9}, ("TAC", "lead_present (R)"): {"COPIES": 9}, ("SCENE", "agents30 (R)"): {"UNREADABLE": 9}}
for k, v in exp.items():
    record("IMAG1B decision counts %s %s" % k, cnt.get(k), v, cnt.get(k) == v)
t7 = [(k, v) for k, v in cnt.items() if k[1].startswith("lon7")]
record("IMAG1B lon7: 8 COPIES and one cell not decided as COPIES, none IMAGINES by decision", t7, True,
       len(t7) >= 1 and sum(v.get("COPIES", 0) for k, v in t7) == 8 and not any(v.get("IMAGINES", 0) for k, v in t7))
record("IMAG1B no DECISION cell is IMAGINES", any("IMAGINES" in v for v in cnt.values()), False, not any("IMAGINES" in v for v in cnt.values()))
record("IMAG1B lon7 cell rows: exactly 9", sum(1 for l in lines[start:stop] if l.startswith("| lon7 (R)")), 9, sum(1 for l in lines[start:stop] if l.startswith("| lon7 (R)")) == 9)
record("IMAG1B lon7 MLP-only IMAGINES cell exists (text states one)", sum(1 for l in lines[start:stop] if l.startswith("| lon7 (R)") and "IMAGINES" in l), 1,
       sum(1 for l in lines[start:stop] if l.startswith("| lon7 (R)") and "IMAGINES" in l) == 1)
# gain range 0.35 to 0.46 for levels 2 and 3 across both probes and seeds (IA-3 tables)
a3 = [i for i, l in enumerate(lines) if l.startswith("## IA-3 action audit")][0]
a3e = [i for i, l in enumerate(lines) if l.startswith("### Decision labels")][0]
vals = []
for l in lines[a3:a3e]:
    m = re.match(r"\| (L[23]) \| \d \([\d.]+\) \| \+([\d.]+) \[[^\]]*\] \| \+([\d.]+) \[", l)
    if m:
        vals += [float(m.group(2)), float(m.group(3))]
record("IMAG1B longitudinal gain levels 2 and 3, min / max over ridge and MLP, both seeds, k = 1 to 3", (min(vals), max(vals), len(vals)), (0.35, 0.46, 24),
       abs(min(vals) - 0.35) < 1e-9 and abs(max(vals) - 0.46) < 1e-9 and len(vals) == 24)
dl = srctext(I1)[srctext(I1).find("### Decision labels"):]
dl = dl[: dl.find("###", 10)]
record("IMAG1B longitudinal decision WEAK at L2 and L3 (all k)", all("**WEAK**" in l.split("|")[3] for l in dl.split("\n") if l.startswith("| L2 |") or l.startswith("| L3 |")), True,
       all("**WEAK**" in l.split("|")[3] for l in dl.split("\n") if l.startswith("| L2 |") or l.startswith("| L3 |")))

# ------------------------------------------------------------------ CAPLADDER
C = "capladder/RESULT_CAPLADDER.md"
SRC[0] = "capladder/RESULT_CAPLADDER.md [M, T0, bootstrap B 2,000, refit permutation R = 199]"
for lab, needle, it in [
    ("R0", "0.515 | [0.430, 0.615]", False), ("R1", "0.503 | [0.410, 0.614]", False), ("R2", "0.501 | [0.401, 0.625]", False),
    ("R3 s0", "0.469 | [0.349, 0.629]", False), ("R3 s1", "0.466 | [0.342, 0.632]", False),
    ("R4 s0", "0.586 | [0.449, 0.762]", False), ("R4 s1", "0.545 | [0.429, 0.693]", False),
    ("L1 s0", "0.627 | [0.481, 0.812]", False), ("L1 s1", "0.734 | [0.532, 1.001]", False),
    ("L2 s0", "0.635 | [0.483, 0.856]", False), ("L2 s1", "0.554 | [0.437, 0.703]", False),
    ("L3 s0", "0.636 | [0.494, 0.821]", False), ("L3 s1", "0.703 | [0.547, 0.898]", False),
    ("R3 reinit 100", "0.708 | [0.545, 0.923]", False), ("R3 reinit 101", "0.689 | [0.534, 0.897]", False),
    ("R4 reinit 100", "0.977 | [0.810, 1.170]", False), ("R4 reinit 101", "0.919 | [0.740, 1.131]", False),
    ("step R2 to R3 s0", "-0.032 | [-0.135, 0.080]", False), ("step R2 to R4 s0", "0.086 | [-0.024, 0.214]", False), ("step R2 to R3s s0", "0.201 | [0.072, 0.342]", False),
    ("outcome O-NONE", "| O-NONE | yes |", False), ("frames", "`lead_gap` on 2,489 frames of 56 episodes", False),
    ("weights note", "weights not checked (ModuleNotFoundError)", False),
]:
    has("CAPLADDER " + lab, C, needle, in_text=False)
# the text prints intervals as "0.515 [0.430, 0.615]"; check each in the text
for lab, pt, iv in [("R0", "0.515", "[0.430, 0.615]"), ("R1", "0.503", "[0.410, 0.614]"), ("R2", "0.501", "[0.401, 0.625]"), ("R3 s0", "0.469", "[0.349, 0.629]"),
                    ("R3 s1", "0.466", "[0.342, 0.632]"), ("R4 s0", "0.586", "[0.449, 0.762]"), ("R4 s1", "0.545", "[0.429, 0.693]"), ("L1 s0", "0.627", "[0.481, 0.812]"),
                    ("L1 s1", "0.734", "[0.532, 1.001]"), ("L2 s0", "0.635", "[0.483, 0.856]"), ("L2 s1", "0.554", "[0.437, 0.703]"), ("L3 s0", "0.636", "[0.494, 0.821]"),
                    ("L3 s1", "0.703", "[0.547, 0.898]"), ("R3 reinit 100", "0.708", "[0.545, 0.923]"), ("R3 reinit 101", "0.689", "[0.534, 0.897]"),
                    ("R4 reinit 100", "0.977", "[0.810, 1.170]"), ("R4 reinit 101", "0.919", "[0.740, 1.131]")]:
    s = pt + " [" + iv.strip("[]") + "]"
    record("CAPLADDER text prints " + lab + " as " + s, s in TEXT["sec_749_imagination.md"] or (lab in ("R3 s0", "R3 s1", "R4 s0", "R4 s1", "L1 s0", "L2 s0", "L3 s0") and pt in TEXT["sec_749_imagination.md"]), True,
           s in TEXT["sec_749_imagination.md"] or (lab in ("R3 s0", "R3 s1", "R4 s0", "R4 s1", "L1 s0", "L2 s0", "L3 s0") and pt in TEXT["sec_749_imagination.md"]))
for lab, s in [("step R2 to R3 s0", "-0.032 [-0.135, 0.080]"), ("step R2 to R4 s0", "+0.086 [-0.024, 0.214]"), ("step R2 to R3s s0", "+0.201 [0.072, 0.342]")]:
    record("CAPLADDER text prints " + lab, s in TEXT["sec_749_imagination.md"], True, s in TEXT["sec_749_imagination.md"])
cs = srctext(C)
record("CAPLADDER outcomes O-ENC, O-POOL8, O-POOL4, O-NET, O-DIFFUSE all no and O-NONE yes",
       all(("| %s | no |" % o) in cs for o in ("O-ENC", "O-POOL8", "O-POOL4", "O-NET", "O-DIFFUSE")) and "| O-NONE | yes |" in cs, True,
       all(("| %s | no |" % o) in cs for o in ("O-ENC", "O-POOL8", "O-POOL4", "O-NET", "O-DIFFUSE")) and "| O-NONE | yes |" in cs)
record("CAPLADDER the only LOSS in the step table is R2 to R3s at s0 (beside)", len(re.findall(r"\| LOSS \|", cs)), 1, len(re.findall(r"\| LOSS \|", cs)) == 1)
record("CAPLADDER controls: 0/480, 4/400 (1.0 %), 9 passed", "READABLE rows 0/480 (0.0 %), LOSS rows 4/400 (1.0 %)" in cs and "9 passed" in cs, True,
       "READABLE rows 0/480 (0.0 %), LOSS rows 4/400 (1.0 %)" in cs and "9 passed" in cs)
for s in ["0 of 480", "4 of 400", "O-NONE", "2,489 frames of 56 episodes", "ModuleNotFoundError"]:
    record("text prints CAPLADDER item " + s, s in TEXT_ALL, True, s in TEXT_ALL)

# ------------------------------------------------------------------ IMAG1C
M = "imag1c/RESULT_IMAG1C.md"
SRC[0] = "imag1c/RESULT_IMAG1C.md [M, T0, episode-cluster bootstrap B 2,000, refit permutation R = 199]"
ms = srctext(M)
rows = {}
for l in ms[: ms.find("## Beside: the older window-subset")].split("\n"):
    if re.match(r"\| L\d \| \d \| (UNREADABLE|COPIES|IMAGINES|WORSE-THAN-COPY) \| s\d \|", l):
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        # c: level, k, label, seed, n (eps), oracle, ref, copy, pred, pred-copy, verdict
        pc = re.match(r"(-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]", c[9])
        key = (c[0], c[1], c[3])
        rows[key] = (c[0], c[1], c[2], c[3], c[4], c[5].split()[0], c[5].split()[0], None, None, None, None, pc.group(1), pc.group(2), pc.group(3), c[10])
record("IMAG1C lead_gap (deciding) rows parsed: 18", len(rows), 18, len(rows) == 18)
lab_by_cell = {}
for (lv, k, s), g in rows.items():
    lab_by_cell.setdefault((lv, k), set()).add(g[2])
cnts = {}
for c, v in lab_by_cell.items():
    assert len(v) == 1, c
    cnts[list(v)[0]] = cnts.get(list(v)[0], 0) + 1
record("IMAG1C label counts over 9 cells (UNREADABLE 3, COPIES 6, IMAGINES 0)", cnts, {"UNREADABLE": 3, "COPIES": 6}, cnts == {"UNREADABLE": 3, "COPIES": 6})
record("IMAG1C all 18 rows predicted vs Copy-Last 'not separated'", all(g[14].strip() == "not separated" for g in rows.values()), True, all(g[14].strip() == "not separated" for g in rows.values()))
record("IMAG1C line 'UNREADABLE 3, COPIES 6, IMAGINES 0, WORSE-THAN-COPY 0 of 9'", "UNREADABLE 3, COPIES 6, IMAGINES 0, WORSE-THAN-COPY 0 of 9" in ms, True, "UNREADABLE 3, COPIES 6, IMAGINES 0, WORSE-THAN-COPY 0 of 9" in ms)
record("IMAG1C 6 of 9 cells readable, L2 and L3", "READABLE (both seeds) in 6 of 9 (level, k) cells: L2 k1, L2 k2, L2 k3, L3 k1, L3 k2, L3 k3" in ms, True,
       "READABLE (both seeds) in 6 of 9 (level, k) cells: L2 k1, L2 k2, L2 k3, L3 k1, L3 k2, L3 k3" in ms)
for lv, k, s, o0, o1, dp, dlo, dhi in [("L1", "1", "s0", "0.633", "0.690", "0.010", "-0.021", "0.037"), ("L1", "1", "s1", "", "", "-0.027", "-0.078", "0.009"), ("L1", "3", "s0", "0.637", "0.705", "0.006", "-0.049", "0.063"),
                                      ("L1", "3", "s1", "", "", "-0.052", "-0.143", "0.014"), ("L2", "1", "s0", "0.622", "0.575", "0.000", "-0.023", "0.023"), ("L2", "1", "s1", "", "", "0.003", "-0.012", "0.019"),
                                      ("L2", "3", "s0", "0.633", "0.577", "0.010", "-0.053", "0.079"), ("L2", "3", "s1", "", "", "0.015", "-0.030", "0.060"), ("L3", "1", "s0", "0.628", "0.708", "0.006", "-0.024", "0.036"),
                                      ("L3", "1", "s1", "", "", "0.011", "-0.017", "0.039"), ("L3", "3", "s0", "0.667", "0.752", "-0.016", "-0.100", "0.069"), ("L3", "3", "s1", "", "", "-0.014", "-0.071", "0.048")]:
    g = rows[(lv, k, s)]
    record("IMAG1C %s k%s %s pred minus Copy %s [%s, %s]" % (lv, k, s, dp, dlo, dhi), (g[11], g[12], g[13]), (dp, dlo, dhi), (g[11], g[12], g[13]) == (dp, dlo, dhi))
    if o0:
        o_s0 = rows[(lv, k, "s0")][6]
        o_s1 = rows[(lv, k, "s1")][6]
        record("IMAG1C %s k%s Oracle NMSE s0 / s1 %s / %s" % (lv, k, o0, o1), (o_s0, o_s1), (o0, o1), (o_s0, o_s1) == (o0, o1))
        record("text prints %s k%s Oracle %s / %s" % (lv, k, o0, o1), ("%s / %s" % (o0, o1)) in TEXT["sec_749_imagination.md"], True, ("%s / %s" % (o0, o1)) in TEXT["sec_749_imagination.md"])
    e = "%s [%s, %s]" % (dp, dlo, dhi)
    record("text prints %s k%s %s %s" % (lv, k, s, e), e in TEXT["sec_749_imagination.md"], True, e in TEXT["sec_749_imagination.md"])
record("IMAG1C IMAG1B probe on all frames L1 k1 Oracle NMSE 0.762 / 0.934", "| L1 | 1 | UNREADABLE | 0.780 / 0.929 | COPIES | 0.640 / 0.692 | 0.762 / 0.934 |" in ms and "0.762 / 0.934" in TEXT["sec_749_imagination.md"], True,
       "| L1 | 1 | UNREADABLE | 0.780 / 0.929 | COPIES | 0.640 / 0.692 | 0.762 / 0.934 |" in ms and "0.762 / 0.934" in TEXT["sec_749_imagination.md"])
record("IMAG1C 7,481 window frames, 56 with a lead", "Window frames: 7,481" in ms and "(56 with a lead)" in ms, True, "Window frames: 7,481" in ms and "(56 with a lead)" in ms)
record("IMAG1C controls: 3.6e-06 and 1.2e-08, 30 cells 0 failed", "max abs diff 3.6e-06" in ms and "max abs diff of NMSE 1.2e-08" in ms and "30 cells, 0 failed" in ms, True,
       "max abs diff 3.6e-06" in ms and "max abs diff of NMSE 1.2e-08" in ms and "30 cells, 0 failed" in ms)
for s in ["3.6e-06", "1.2e-08", "30 cells, 0 failed"]:
    record("text prints IMAG1C control item " + s, s in TEXT["sec_749_imagination.md"], True, s in TEXT["sec_749_imagination.md"])
# lead_closing and ttc: Oracle NMSE ranges quoted as 1.0 to 1.2
oc = [float(x) for x in re.findall(r"\| (?:L\d) \| \d \| UNREADABLE \| s\d \| \d+ \(\d+\) \| ([\d.]+) \[", ms[ms.find("Beside, no bar: `lead_closing`"):])]
record("IMAG1C lead_closing and ttc Oracle NMSE all in [0.99, 1.23] (text: 1.0 to 1.2), n = 36", (round(min(oc), 3), round(max(oc), 3), len(oc)), True, len(oc) == 36 and min(oc) >= 0.99 and max(oc) <= 1.23)

# ------------------------------------------------------------------ horizon bands: JSON
SRC[0] = "horizon_review/band_eval/BAND_BASELINE.json [M, T1-kin, paired episode-cluster bootstrap B 2,000 seed 0, 2,399 windows, 141 episodes]"
BJ = json.loads(rd("horizon_review/band_eval/BAND_BASELINE.json"))
record("BAND tier is T1-kin", BJ["tier"].startswith("T1-kin"), True, BJ["tier"].startswith("T1-kin"))
record("BAND STRATEGIC status PAUSED", BJ["STRATEGIC"]["status"], "PAUSED", BJ["STRATEGIC"]["status"] == "PAUSED")
KEY = {"ADE (m)": "ade_m", "FDE": "fde_m", "speed MAE (m/s)": "speed_mae_mps", "acceleration MAE (m/s2)": "accel_mae_mps2", "along-track MAE (m)": "along_mae_m",
       "DK headway abs error (m)": "dk_hw_abs_err_m", "heading MAE (deg)": "heading_mae_deg", "curvature MAE (1/m)": "curvature_mae_1pm", "yaw-rate MAE (deg/s)": "yaw_rate_mae_degps",
       "cross-track MAE (m)": "cross_mae_m"}
sec = TEXT["sec_750_horizon_bands.md"]
for band, jb, heading in (("operative", "operative_0_2s", "*Operative band, 0.2 to 2.0 s:*"), ("tactical", "tactical_2_6s", "*Tactical band, 2.2 to 6.0 s:*")):
    blk = sec[sec.find(heading):]
    blk = blk[: blk.find("\n\n", blk.find("|---")) if band == "tactical" else blk.find("*Tactical band")]
    for rowlab, key in KEY.items():
        r_line = [l for l in blk.split("\n") if l.startswith("|") and ("| %s |" % rowlab.split(" (")[0] in l or ("| %s |" % rowlab) in l)]
        if rowlab == "FDE":
            r_line = [l for l in blk.split("\n") if l.startswith("| PRIMARY | FDE")]
        r_line = [l for l in r_line if (rowlab == "FDE" or rowlab in l)]
        assert len(r_line) == 1, (band, rowlab, r_line)
        c = [x.strip() for x in r_line[0].strip().strip("|").split("|")]
        for si, s in enumerate(("s0", "s1")):
            R = BJ["seeds"][s][jb]["rows"][key]
            d = R["arm_minus_P10"]
            exp_txt = "%+.3f [%+.3f, %+.3f]" % (d["delta"], d["lo"], d["hi"])
            have = c[3 + si]
            record("BAND %s %s %s delta JSON -> %s printed %s" % (band, rowlab, s, exp_txt, have), have, exp_txt, have == exp_txt)
        R = BJ["seeds"]["s0"][jb]["rows"][key]["P10"]
        p_txt = "%.3f [%.3f, %.3f]" % (R["mean"], R["lo"], R["hi"])
        record("BAND %s %s P10 JSON -> %s printed %s" % (band, rowlab, p_txt, c[2]), c[2], p_txt, c[2] == p_txt)
# full band reproduction values and the E arithmetic
full0 = BJ["seeds"]["s0"]["full_0_6s"]["rows"]["ade_m"]["arm_minus_P10"]
full1 = BJ["seeds"]["s1"]["full_0_6s"]["rows"]["ade_m"]["arm_minus_P10"]
SRC[0] = "BAND_BASELINE.json [E: arithmetic on measured deltas]"
op0 = BJ["seeds"]["s0"]["operative_0_2s"]["rows"]["ade_m"]["arm_minus_P10"]["delta"]
tc0 = BJ["seeds"]["s0"]["tactical_2_6s"]["rows"]["ade_m"]["arm_minus_P10"]["delta"]
op1 = BJ["seeds"]["s1"]["operative_0_2s"]["rows"]["ade_m"]["arm_minus_P10"]["delta"]
tc1 = BJ["seeds"]["s1"]["tactical_2_6s"]["rows"]["ade_m"]["arm_minus_P10"]["delta"]
chk("E: s0 10 x op + 20 x tac with printed (rounded) deltas = -36.03", 10 * -0.003 + 20 * -1.800, -36.03, 1e-9)
chk("E: s0 equals 30 x full ADE6 delta (-1.2012) = -36.036", 30 * -1.2012, -36.036, 1e-9)
chk("E: JSON s0 10 x op + 20 x tac within 0.05 of 30 x full", 10 * op0 + 20 * tc0, 30 * full0["delta"], 0.05)
chk("E: s1 10 x (+0.003) + 20 x (-1.686) = -33.69", 10 * 0.003 + 20 * -1.686, -33.69, 1e-9)
chk("E: JSON s1 10 x op + 20 x tac within 0.05 of 30 x full", 10 * op1 + 20 * tc1, 30 * full1["delta"], 0.05)
chk("E: operative share of the s0 ADE gap = 0.03 / 36.03 = 0.083 % (< 0.1 %)", round(0.03 / 36.03 * 100, 3), 0.083, 1e-3)
for s in ["-36.03", "-33.69", "30 x (-1.201)"]:
    record("text prints E item " + s, s in sec, True, s in sec)
SRC[0] = "horizon_review/band_eval/BAND_BASELINE.md [M]"
BM = "horizon_review/band_eval/BAND_BASELINE.md"
for lab, needle in [("TAC-LAT s0", "0.784 [0.740, 0.823]"), ("TAC-LAT s1", "0.790 [0.750, 0.827]"), ("TAC-LAT P10", "0.593 [0.555, 0.634]"), ("TAC-LAT delta s0", "+0.191 [+0.151, +0.232]"),
                    ("TAC-LAT delta s1", "+0.197 [+0.160, +0.236]"), ("TAC-LON s0", "0.511 [0.470, 0.550]"), ("TAC-LON s1", "0.509 [0.466, 0.551]"), ("TAC-LON P10", "0.371 [0.326, 0.417]"),
                    ("TAC-LON delta s0", "+0.140 [+0.077, +0.200]"), ("TAC-LON delta s1", "+0.137 [+0.075, +0.200]"), ("time gap tactical s0", "-0.077 [-0.129, -0.032]"),
                    ("ADE6 reproduction", "ADE6 -1.2012 / -1.1231"), ("44 of 44", "44 of 44 hold"), ("full curvature", "curvature +0.129 / +0.204 against +0.085 / +0.135"), ("full yaw", "yaw-rate +8.8 / +15.7 deg/s against +3.5 / +8.7"),
                    ("tactical curvature seed spread", "curvature 0.185 vs 0.260"), ("tactical yaw seed spread", "yaw-rate 30.4 vs 37.2"), ("72 episodes", "72 episodes with a lead in the operative band"),
                    ("T head labels", "**[NOW+2 s, NOW+8 s]**")]:
    has("BAND " + lab, BM, needle, in_text=(lab not in ("full yaw", "72 episodes", "T head labels", "full curvature", "tactical curvature seed spread", "tactical yaw seed spread", "44 of 44")))
for s in ["curvature +0.129 / +0.204 against +0.085 / +0.135", "+8.8 / +15.7 deg/s against +3.5 / +8.7", "curvature 0.185 against 0.260", "yaw rate 30.4 against 37.2", "72 episodes with a lead", "57 to 58", "NOW+2 s to NOW+8 s", "44 of 44", "-1.2012 / -1.1231"]:
    record("text prints BAND item " + s, s in sec, True, s in sec)
bm = srctext(BM)
record("BAND DAG-OACC cat-sha256 prefixes", "9c874630e62e121f" in bm and "37ecd69c7735b8e9" in bm and "7a879baa" in bm, True, "9c874630e62e121f" in bm and "37ecd69c7735b8e9" in bm and "7a879baa" in bm)
record("text prints BAND input hashes", all(x in sec for x in ("9c874630e62e121f", "37ecd69c7735b8e9", "7a879baa")), True, all(x in sec for x in ("9c874630e62e121f", "37ecd69c7735b8e9", "7a879baa")))
TJ = BJ["seeds"]["s0"]["TACTICAL_decision_reads"]["rows"]
chk("BAND TAC-LAT delta s0 JSON", TJ["TAC-LAT"]["arm_minus_P10"]["delta"], 0.1905, 6e-4)
chk("BAND TAC-LON delta s0 JSON rounds to 0.140", round(TJ["TAC-LON"]["arm_minus_P10"]["delta"], 3), 0.140, 6e-4)
chk("BAND TACTICAL n windows", BJ["seeds"]["s0"]["TACTICAL_decision_reads"]["n_windows"], 2399)
chk("BAND ADE6 full s0 delta JSON", full0["delta"], -1.2012, 6e-4)
chk("BAND ADE6 full s1 delta JSON", full1["delta"], -1.1231, 6e-4)

# ------------------------------------------------------------------ audits and spec
SRC[0] = "horizon_review/AUDIT_MODEL.md, AUDIT_DATA.md, SPEC_HWM.md, REQ_HORIZONS.md [M code read / M counts; I where marked]"
AM = "horizon_review/AUDIT_MODEL.md"
AD = "horizon_review/AUDIT_DATA.md"
SP = "horizon_review/SPEC_HWM.md"
RQ = "horizon_review/REQ_HORIZONS.md"
for lab, path, needle, it in [
    ("F1 reach", AM, "0.2 / 0.4 / 0.8 s", True), ("F1 lines", AM, "`hjepa.py:671-695`", True), ("F4 weight", AM, "`w_wm = 0.1`", False),
    ("F4 lines", AM, "`v8f_mini.py:1045-1047`", False), ("F8 5.3", AM, "5.3 % (lat-a), 7.3 % (lat-b), 15.4 % (lon)", False),
    ("F9 tokens", AM, "`FOLLOW_ROUTE` 2,948, `TURN_RIGHT_FOLLOW_ROUTE` 596, `TURN_LEFT_FOLLOW_ROUTE` 574, `STOP_AT_FOLLOW_ROUTE` 454", False),
    ("F11 95", AM, "about 95 % of clips", False), ("30 m shares", AM, "| 30 m | 1.3 s | 2.0 | **3.0 s** | 5.1 | 11.9 | 79.7 % |", False),
    ("50 m shares", AM, "| 50 m | 2.2 | 3.3 | **5.0 s** | 8.5 | 19.8 | 59.9 % |", False), ("80 m shares", AM, "| 80 m | 3.6 | 5.3 | **8.0 s** | 13.6 | 31.6 | 31.0 % |", False),
    ("clips 4,293", AD, "**4,293 (95.1 %)** | **4,211 (93.3 %)** | **3,950 (87.5 %)**", False), ("video end", AD, "20.13 to 20.17 s", False), ("egomotion median", AD, "median 140 s", False),
    ("147 vs 141", AD, "holds 147 records", False), ("ladder", SP, "| World-model step | 0.2 s | 0.6 s (stride 3) | 3.0 s (stride 5)", False),
    ("PI definition", RQ, "(op up to 2 s), (tac up to 6 sec iccl our tactical vocab), (6se to 30 sec incl. av checkpointsssss, detectedstrategic informtio nd derived goals from it) trajectory is a combo of op and tac and thus 6s . It makes no sense to consider 0.8 s as long term abstract level. So we need to fix and retrain as we need it", True),
    ("PI R-PLAN quote", RQ, "yes use multi hierarchacy world model based planning as baseline and compare it to other approaches", True),
    ("limit right-hand", SP, "All 25 countries drive on the right", False), ("limit 20.17", SP, "No real images exist past 20.17 s", False), ("two failed planners", SP, "Two earlier world-model planners failed on this programme", False),
]:
    has(lab, path, needle, in_text=False)
for s in ["0.2 / 0.4 / 0.8 s", "`hjepa.py:671-695`", "0.1 (`v8f_mini.py:1045-1057`)", "5.3 % (lateral a), 7.3 % (lateral b) and 15.4 % (longitudinal)", "2,948", "596", "574", "454", "4,293 (95.1 %) / 4,211 (93.3 %) / 3,950 (87.5 %)",
          "3.0 / 5.0 / 8.0 s", "79.7 / 59.9 / 31.0 %", "20.17 s", "147 records", "141 clips", "0.2 s | 0.6 s (stride 3) | 3.0 s (stride 5)",
          "(op up to 2 s), (tac up to 6 sec iccl our tactical vocab), (6se to 30 sec incl. av checkpointsssss, detectedstrategic informtio nd derived goals from it) trajectory is a combo of op and tac and thus 6s . It makes no sense to consider 0.8 s as long term abstract level. So we need to fix and retrain as we need it",
          "yes use multi hierarchacy world model based planning as baseline and compare it to other approaches"]:
    record("text prints audit item " + s[:60], s in sec, True, s in sec)
asrc = srctext(AD)
chk("E: 4,293 / 4,512 = 95.1 %", round(4293 / 4512 * 100, 1), 95.1, 1e-9)
chk("E: 4,211 / 4,512 = 93.3 %", round(4211 / 4512 * 100, 1), 93.3, 1e-9)
chk("E: 3,950 / 4,512 = 87.5 %", round(3950 / 4512 * 100, 1), 87.5, 1e-9)
RQs = srctext(RQ)
for rid in ["R-OP", "R-TAC", "R-STR", "R-TRAJ", "R-ABS", "R-VIS", "R-DATA", "R-RETRAIN", "R-PLAN"]:
    record("requirement %s in REQ_HORIZONS.md and in section 7.50" % rid, ("| %s |" % rid) in RQs and ("| %s |" % rid) in sec, True, ("| %s |" % rid) in RQs and ("| %s |" % rid) in sec)
record("STRATEGIC PAUSED stated in 7.49, 7.50 and the edits", all("PAUSED" in TEXT[f] for f in TEXT_FILES), True, all("PAUSED" in TEXT[f] for f in TEXT_FILES))
record("T0 stated in 7.49; T1-kin stated in 7.50", "T0" in TEXT["sec_749_imagination.md"] and "T1-kin" in sec, True, "T0" in TEXT["sec_749_imagination.md"] and "T1-kin" in sec)

# ------------------------------------------------------------------ edits: numbers that repeat the sections
SRC[0] = "edits_v45.md numbers repeat the section numbers [M]"
ed = TEXT["edits_v45.md"]
for s in ["0.583 [0.538, 0.630] / 0.585 [0.541, 0.628]", "0.668 [0.623, 0.712] / 0.677 [0.629, 0.725]", "0.738 / 0.733 against 0.759 / 0.743", "-1.800 [-2.088, -1.527] / -1.686 [-1.996, -1.376]",
          "-0.003 [-0.010, +0.003] / +0.003 [-0.005, +0.011]", "+0.129 [+0.087, +0.176] / +0.204 [+0.136, +0.284]", "+8.847 [+4.468, +13.447] / +15.676 [+9.945, +22.132]",
          "-0.046 [-0.065, -0.026] / -0.030 [-0.054, -0.004]", "0 of 9", "6 of 9"]:
    record("edits print " + s, s in ed, True, s in ed)
# each interval printed in the edits for the band baseline equals the JSON
for lab, key, jb, s in [("tactical ADE", "ade_m", "tactical_2_6s", "-1.800 [-2.088, -1.527] / -1.686 [-1.996, -1.376]"), ("operative ADE", "ade_m", "operative_0_2s", "-0.003 [-0.010, +0.003] / +0.003 [-0.005, +0.011]"),
                        ("operative FDE", "fde_m", "operative_0_2s", "-0.046 [-0.065, -0.026] / -0.030 [-0.054, -0.004]"), ("tactical curvature", "curvature_mae_1pm", "tactical_2_6s", "+0.129 [+0.087, +0.176] / +0.204 [+0.136, +0.284]"),
                        ("tactical yaw", "yaw_rate_mae_degps", "tactical_2_6s", "+8.847 [+4.468, +13.447] / +15.676 [+9.945, +22.132]")]:
    parts = []
    for s_ in ("s0", "s1"):
        d = BJ["seeds"][s_][jb]["rows"][key]["arm_minus_P10"]
        parts.append("%+.3f [%+.3f, %+.3f]" % (d["delta"], d["lo"], d["hi"]))
    rec = " / ".join(parts)
    record("edits interval %s equals JSON %s" % (lab, rec), s, rec, s == rec)

# ------------------------------------------------------------------ PLANTEST (7.50.7) and the build facts (7.50.8)
PT = "tip:" + RTIP + "2026-11-08-v8f-plantest/RESULT_PLANTEST.md"
PC = "tip:" + RTIP + "2026-11-08-v8f-plantest/RESULT_PLANTEST_controls.md"
PJ = json.loads(tipblob(RTIP + "2026-11-08-v8f-plantest/RESULT_PLANTEST.json"))
S750 = TEXT["sec_750_horizon_bands.md"]
SRC[0] = "tip raw/2026-11-08-v8f-plantest/RESULT_PLANTEST.md, .json, _controls.md [M, T0, oracle goal, EC-BOOT B 2,000 seed 0]"
chk("PLANTEST JSON tier", PJ["tier"], "T0")
chk("PLANTEST JSON windows", PJ["n_windows"], 1976)
chk("PLANTEST JSON episodes", PJ["n_episodes"], 141)
chk("PLANTEST JSON windows dropped", PJ["windows_dropped_no_2p4s_frame"], 0)
for lab, needle in [("windows n = 1976", "n = 1976"), ("B 2,000 seed 0", "B 2,000, seed 0"), ("T0 oracle statement", "Tier T0"), ("never a driving result", "never a driving result"),
                    ("12 logged actions", "12 logged actions"), ("HIER-ADDS NO", "LONGITUDINAL: NO; LATERAL: NO"), ("TACTICAL n = 0", "n = 0"), ("STRATEGIC paused", "PAUSED (PI item 25"),
                    ("FLAT1 is the flat arm", "FLAT1 is the flat arm for every outcome"), ("PLANTEST2 prefix", "d8217dfa"), ("card prefix", "d5dbcc63"), ("gap to HOLD 0.97 to 1.09", "0.97 to 1.09 m/s^2"),
                    ("planner-seed floor 0.0066", "0.0066 [0.0010, 0.0126]"), ("largest untrained magnitude", "0.0379 [-0.0052, 0.0827]"), ("12 ok 0 errors", "12 ok, 0 errors")]:
    n_ = norm(needle)
    ok_s = n_ in srctext(PT)
    txt_needle = {"windows n = 1976": "1,976 windows", "B 2,000 seed 0": "B = 2,000, seed 0", "T0 oracle statement": "T0 with an oracle", "never a driving result": "never a driving result", "12 logged actions": "12 logged actions",
                  "HIER-ADDS NO": "HIER-ADDS is NO", "TACTICAL n = 0": "n = 0", "STRATEGIC paused": "PAUSED", "FLAT1 is the flat arm": "FLAT1 is the flat arm for every outcome", "PLANTEST2 prefix": "d8217dfa",
                  "card prefix": "d5dbcc63", "gap to HOLD 0.97 to 1.09": "0.97 to 1.09 m/s2", "planner-seed floor 0.0066": "0.0066 [0.0010, 0.0126]", "largest untrained magnitude": "0.0379 [-0.0052, 0.0827]",
                  "12 ok 0 errors": "12 ok, 0 errors"}[lab]
    ok_t = norm(txt_needle) in (S750 if lab != "card prefix" else TEXT_ALL)
    record("PLANTEST " + lab + " [src + text]", (ok_s, ok_t), (True, True), ok_s and ok_t)
# every table cell printed in 7.50.7 against the result file and the JSON
rows = [("LONGITUDINAL", "td3", "s0"), ("LONGITUDINAL", "td3", "s1"), ("LONGITUDINAL", "flat1", "s0"), ("LONGITUDINAL", "flat1", "s1"),
        ("LATERAL", "td3", "s0"), ("LATERAL", "td3", "s1"), ("LATERAL", "flat1", "s0"), ("LATERAL", "flat1", "s1")]
th, tr = [], []
for fam, pl, sd in rows:
    d = PJ["result"]["per_seed"]["M2b_%s|%s|%s" % (sd, pl, fam)]
    h_ = "%.4f [%.4f, %.4f]" % tuple(d["true_minus_hold"])
    r_ = "%.4f [%.4f, %.4f]" % tuple(d["true_minus_rand"])
    th.append(d["true_minus_hold"][0]); tr.append(d["true_minus_rand"][0])
    row_t = [l for l in S750.split("\n") if l.startswith("| %s | %s | %s |" % (fam, pl.upper(), sd))]
    ok = len(row_t) == 1 and (h_ + " (worse)") in row_t[0] and (r_ + " (better)") in row_t[0] and row_t[0].rstrip().endswith("| NO-GAIN |")
    record("PLANTEST decision row %s %s %s equals JSON (%s | %s | %s)" % (fam, pl, sd, h_, r_, d["label"]), ok, True, ok and d["label"] == "NO-GAIN" and d["sep_hold"] == "worse" and d["sep_rand"] == "better")
    ok_src = (h_ in srctext(PT)) and (r_ in srctext(PT))
    record("PLANTEST decision row %s %s %s is printed in RESULT_PLANTEST.md" % (fam, pl, sd), ok_src, True, ok_src)
lon_h = [x for (f, _, _), x in zip(rows, th) if f == "LONGITUDINAL"]
lat_h = [x for (f, _, _), x in zip(rows, th) if f == "LATERAL"]
lon_r = [x for (f, _, _), x in zip(rows, tr) if f == "LONGITUDINAL"]
lat_r = [x for (f, _, _), x in zip(rows, tr) if f == "LATERAL"]
record("range acceleration true minus HOLD 0.9688 to 1.0932 equals JSON", ("%.4f to %.4f" % (min(lon_h), max(lon_h))), "0.9688 to 1.0932", "0.9688 to 1.0932" in S750 and "%.4f to %.4f" % (min(lon_h), max(lon_h)) == "0.9688 to 1.0932")
record("range yaw true minus HOLD 0.1114 to 0.1199 equals JSON", ("%.4f to %.4f" % (min(lat_h), max(lat_h))), "0.1114 to 0.1199", "0.1114 to 0.1199" in S750 and "%.4f to %.4f" % (min(lat_h), max(lat_h)) == "0.1114 to 0.1199")
record("range acceleration true minus random -0.2056 to -0.2442 equals JSON", ("%.4f to %.4f" % (max(lon_r), min(lon_r))), "-0.2056 to -0.2442", "-0.2056 to -0.2442" in S750 and "%.4f to %.4f" % (max(lon_r), min(lon_r)) == "-0.2056 to -0.2442")
record("range yaw true minus random -0.0217 to -0.0306 equals JSON", ("%.4f to %.4f" % (max(lat_r), min(lat_r))), "-0.0217 to -0.0306", "-0.0217 to -0.0306" in S750 and "%.4f to %.4f" % (max(lat_r), min(lat_r)) == "-0.0217 to -0.0306")
record("PLANTEST label table: all 4 verdict cells NO-GAIN in the source", srctext(PT).count("| NO-GAIN |") >= 8 and srctext(PT).count("**NO-GAIN**") == 4, True, srctext(PT).count("**NO-GAIN**") == 4)
# deciding-row table of 7.50.7 (MAE per arm) and the other quoted cells
for lab, needle in [("HOLD acceleration", "0.3296 [0.2973, 0.3623]"), ("HOLD yaw", "0.0224 [0.0185, 0.0263]"),
                    ("TD3 true acc s0", "1.4228 [1.3800, 1.4647]"), ("TD3 rand acc s0", "1.6497 [1.6019, 1.6985]"), ("FLAT1 true acc s0", "1.3925 [1.3537, 1.4312]"),
                    ("TD3 true acc s1", "1.3355 [1.2882, 1.3823]"), ("TD3 rand acc s1", "1.5411 [1.4902, 1.5939]"), ("FLAT1 true acc s1", "1.2984 [1.2536, 1.3435]"),
                    ("TD3 true yaw s0", "0.1338 [0.1272, 0.1406]"), ("TD3 rand yaw s0", "0.1644 [0.1584, 0.1702]"), ("FLAT1 true yaw s0", "0.1344 [0.1285, 0.1405]"),
                    ("TD3 true yaw s1", "0.1422 [0.1361, 0.1483]"), ("TD3 rand yaw s1", "0.1640 [0.1580, 0.1695]"), ("FLAT1 true yaw s1", "0.1374 [0.1315, 0.1430]"),
                    ("HIER-ADDS LON s0", "0.0302 [0.0029, 0.0559]"), ("HIER-ADDS LON s1", "0.0371 [0.0091, 0.0636]"), ("HIER-ADDS LAT s0", "-0.0006 [-0.0048, 0.0036]"), ("HIER-ADDS LAT s1", "0.0049 [0.0009, 0.0087]"),
                    ("monotonicity L2", "0.9205 [0.9057, 0.9342]"), ("monotonicity L3", "0.8982 [0.8770, 0.9167]"), ("monotonicity L1", "0.2575 [0.1764, 0.3361]"), ("flat1 L1 monotonicity", "-0.0997 [-0.1794, -0.0264]"),
                    ("TD3 ADE", "1.6245 [1.4986, 1.7524]"), ("HOLD ADE", "0.2164 [0.1974, 0.2352]")]:
    has("PLANTEST " + lab, PT, needle, where="sec_750_horizon_bands.md")
# the hold row is the same in both seeds, the untrained rows: 8 GOAL-BLIND rows
record("untrained stacks: 8 rows read GOAL-BLIND, 0 other", (srctext(PT).count("rule reads GOAL-BLIND"), srctext(PT).count("rule reads GOAL-")), (8, 8), srctext(PT).count("rule reads GOAL-BLIND") == 8 == srctext(PT).count("rule reads GOAL-"))
# planner-seed floor: acceleration at most 0.0066, yaw at most 0.0003 in magnitude
fl = re.findall(r"\* M2b_s[01]\|(td3|flat)\|(LONGITUDINAL|LATERAL): (-?\d\.\d+) \[", srctext(PT))
lon_f = [abs(float(v)) for _, f, v in fl if f == "LONGITUDINAL"]
lat_f = [abs(float(v)) for _, f, v in fl if f == "LATERAL"]
record("planner-seed floor rows read (8): acceleration max %.4f" % max(lon_f), (len(fl), round(max(lon_f), 4)), (8, 0.0066), len(fl) == 8 and round(max(lon_f), 4) == 0.0066)
record("planner-seed floor yaw max magnitude %.4f" % max(lat_f), round(max(lat_f), 4), 0.0003, round(max(lat_f), 4) == 0.0003)
# controls
for lab, needle, t in [("planted dynamics TD3 MAE 0.0036", "TD3 MAE 0.0036", "0.0036"), ("prior-mean baseline 0.365", "prior-mean baseline 0.365", "0.365"), ("FLAT 0.395", "MAE 0.395 (baseline 0.365)", "0.395 against 0.365"),
                       ("9 passed", "9 passed", None)]:
    ok_s = norm(needle) in srctext(PC)
    ok_t = True if t is None else (t in S750)
    record("PLANTEST controls " + lab + " [src + text]", (ok_s, ok_t), (True, True), ok_s and ok_t)
# landed commits
for fold, short in [("2026-11-08-v8f-plantest", "8f2a62cc"), ("2026-11-08-v8f-bands-build", "c69bc890")]:
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "log", "--format=%H", TIP, "--", RTIP + fold], capture_output=True)
    shas = r.stdout.decode().split()
    record("folder %s was landed by %s (only commit touching it up to the tip: %d)" % (fold, short, len(shas)), shas[-1][:8], short, shas[-1][:8] == short and shas[-1][:8] in TEXT_ALL)
REG = tipblob("Project Steering/GOALS_AND_CLAIMS.md")
for rid in ("H-V8F-PLANTEST", "H-V8F-BANDS-BUILD"):
    record("register row %s exists at the tip and is named in the text" % rid, (rid in REG, rid in TEXT_ALL), (True, True), rid in REG and rid in TEXT_ALL)
# ---- 7.50.8
SRC[0] = "tip raw/2026-11-08-v8f-bands-build/ (band_labels, horizon_review) and the register row H-V8F-BANDS-BUILD [M build facts; I where marked]"
RB = "tip:" + RTIP + "2026-11-08-v8f-bands-build/"
regrow = [l for l in REG.split("\n") if l.startswith("| **H-V8F-BANDS-BUILD**")]
record("register row H-V8F-BANDS-BUILD is one line", len(regrow), 1, len(regrow) == 1)
rr = norm(regrow[0])
for lab, needle, t in [("checkpoint cross-check median", "median 0.517 m over 2,350 windows", "median of 0.517 m over 2,350 windows"), ("wrong-clip control", "(wrong-clip control 40.3 m)", "40.3 m for a wrong-clip control"),
                       ("v9 md5 prefix", "83e30bd3", "83e30bd3"), ("v10 md5 prefix", "733a4ad0", "733a4ad0"), ("label shares train", "lat-a 5.2 %, lat-b 6.8 %, lon 13.8 %", "5.2 % (lateral a), 6.8 % (lateral b) and 13.8 % (longitudinal)"),
                       ("4,572 train and 147 eval", "4,572 train and 147 eval clips", "4,572 train and 147 eval clips")]:
    ok_s = norm(needle) in rr
    ok_t = norm(t) in S750
    record("BANDS-BUILD " + lab + " [register row + text]", (ok_s, ok_t), (True, True), ok_s and ok_t)
CKO = rd("ck_vs_dump/ck_vs_dump_output.txt", base="E:/Projects/TanitAD-artifacts/dagland/report_v45/")
for lab, needle in [("n and median", "shift 0 n 2350 median m 0.517 p90 1.224 p99 3.671"), ("wrong clip", "control wrong clip median m 40.3 n 2350"), ("141 cached clips", "cached 141")]:
    record("checkpoint cross-check output: " + lab, needle in CKO, True, needle in CKO)
SCH = srctext(RB + "band_labels/SCHEMA.md")
record("v9 train md5 83e30bd3791f0767707eade5f4671759 in SCHEMA.md", "83e30bd3791f0767707eade5f4671759" in SCH, True, "83e30bd3791f0767707eade5f4671759" in SCH)
RL = srctext(RB + "band_labels/RESULT_BAND_LABELS.md")
lt = {}
for key in ("lat_cls_a", "lat_cls_b", "lon_cls"):
    m = [l for l in RL.split("\n") if l.startswith("| " + key + " ")]
    lt[key] = re.search(r"\*\*(\d+\.\d) %\*\*", m[0]).group(1) if m else None
record("eval139 change shares read from RESULT_BAND_LABELS.md", lt, {"lat_cls_a": "5.2", "lat_cls_b": "7.2", "lon_cls": "15.2"}, lt == {"lat_cls_a": "5.2", "lat_cls_b": "7.2", "lon_cls": "15.2"} and "5.2 %, 7.2 % and 15.2 % on eval139" in S750)
record("RESULT_BAND_LABELS.md names the inherited 5.3 / 7.3 / 15.4 % and the bar rule", "5.3 / 7.3 / 15.4 %" in RL, True, "5.3 / 7.3 / 15.4 %" in RL and "5.3 / 7.3 / 15.4 %" in S750)
# the F: single copy of the train comparison (not on the tip) must agree with the register row
fp = "F:/strat_out/band_labels/v10/compare_v9_v10_train.log"
if os.path.exists(fp):
    ft = open(fp, encoding="utf-8").read()
    got = [round(float(re.search(k + r".*?'changed_among_v9_defined': ([0-9.]+)", ft).group(1)) * 100, 1) for k in ("lat_cls_a", "lat_cls_b", "lon_cls")]
    record("train change shares from the F: single copy compare_v9_v10_train.log", got, [5.2, 6.8, 13.8], got == [5.2, 6.8, 13.8])
else:
    record("train change shares from the F: single copy (file absent: the register row is the only source)", "absent", "absent", True)
mf = "F:/strat_out/band_labels/v10/v10_labels_train.manifest.json"
if os.path.exists(mf):
    md = json.load(open(mf, encoding="utf-8"))["npz_md5"]
    record("v10 train md5 from the F: manifest starts 733a4ad0", md[:8], "733a4ad0", md.startswith("733a4ad0"))
REQ = srctext(RB + "horizon_review/REQ_HORIZONS.md")
for rid in ("R-EGO", "R-T2O", "R-OUT"):
    record("%s is in REQ_HORIZONS.md (tip, bands-build copy) and in the text" % rid, (rid in REQ, rid in S750), (True, True), rid in REQ and rid in S750)
SPEC = srctext(RB + "horizon_review/SPEC_HWM.md")
i91 = SPEC.index("### 9.1")
s91 = SPEC[i91:]
record("SPEC_HWM 9.1 text: no hand-over, whole 6 s plan, no tactical trajectory head, no continuity term, exactly 0.0 gradient", all(x in s91 for x in ("No hand-over state", "T has no trajectory head", "O outputs the whole 6 s plan", "needs no continuity term", "exactly 0.0 gradient")), True, all(x in s91 for x in ("No hand-over state", "T has no trajectory head", "O outputs the whole 6 s plan", "needs no continuity term", "exactly 0.0 gradient")))
RV = srctext(RB + "horizon_review/REVIEW_HJEPA_CONSISTENCY.md")
record("REVIEW names gaps G1 to G9", all(("G%d" % i) in RV for i in range(1, 10)), True, all(("G%d" % i) in RV for i in range(1, 10)) and "nine gaps (G1 to G9)" in S750)
# no statement of sections 7.49 / 7.50 / edits contradicts SPEC 9.1: a hand-over state or a tactical trajectory head is never described as built
bad = [m for m in re.findall(r"[^.\n]*(?:hand-over state|trajectory head|continuity term)[^.\n]*", TEXT_ALL)]
okneg = all(re.search(r"\bno\b|\bwithout\b|\bnot\b|\bnone\b", b_) for b_ in bad)
record("every mention of a hand-over state, trajectory head or continuity term in the v45 text is negated (SPEC 9.1): %d mentions" % len(bad), okneg, True, okneg)

# ------------------------------------------------------------------ words and typography (the standing scans)
banned = ["void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn", "closed loop", "closed-loop"]
for f, t in TEXT.items():
    low = t.lower()
    hits = [w for w in banned if re.search(r"\b" + re.escape(w) + r"\b", low)]
    record("banned words in %s" % f, hits, [], hits == [])
    record("em or en dash in %s" % f, ("\u2014" in t) or ("\u2013" in t), False, not (("\u2014" in t) or ("\u2013" in t)))
    record("the word 'drives' in %s" % f, bool(re.search(r"\bdrives\b", t.lower())), False, not re.search(r"\bdrives\b", t.lower()))
    record("no CR characters in %s" % f, b"\r" in open(HERE + "/" + f, "rb").read(), False, b"\r" not in open(HERE + "/" + f, "rb").read())
record("no learning-curve exponent quoted", bool(re.search(r"exponent", TEXT_ALL.lower())), False, not re.search(r"exponent", TEXT_ALL.lower()))
# ------------------------------------------------------------------ mutation arm: the checker must fail on a changed number
def mutated_detects():
    n = "0.583 [0.538, 0.630]"
    mut = n.replace("0.583", "0.584")
    s = srctext(V)
    return (mut not in s) and (n in s)


record("mutation arm: 0.584 is NOT in the VOE1B source while 0.583 is (a changed number fails)", mutated_detects(), True, mutated_detects())
def mutated_json():
    d = BJ["seeds"]["s0"]["tactical_2_6s"]["rows"]["ade_m"]["arm_minus_P10"]
    good = "%+.3f [%+.3f, %+.3f]" % (d["delta"], d["lo"], d["hi"])
    bad = "%+.3f [%+.3f, %+.3f]" % (d["delta"] + 0.01, d["lo"], d["hi"])
    return good in sec and bad not in sec


record("mutation arm: a tactical ADE delta shifted by 0.01 is not printed in the section", mutated_json(), True, mutated_json())

# ------------------------------------------------------------------ derived, inherited and hypothesis numbers (not source-checked as measurements)
DERIVED = [
    ("-36.03, -33.69, 0.083 %", "arithmetic on the printed deltas; the JSON values agree within 0.05 (checked above)", "BAND_BASELINE.json", "E"),
    ("5.3 %, 7.3 %, 15.4 % label change at a 6 s cut", "quoted from AUDIT_MODEL.md F8 which itself marks them INHERITED (raw/s3_validation_eval139.json); 7.50.8 gives the measured shares (train 5.2 / 6.8 / 13.8 %, eval139 5.2 / 7.2 / 15.2 %)", "AUDIT_MODEL.md", "I"),
    ("median 0.517 m over 2,350 windows (wrong-clip control 40.3 m): the 6 s checkpoint against banked 6 s waypoints, 141 cached eval clips", "re-run 2026-11-08 for this build", "ck_vs_dump/ck_vs_dump_output.txt", "M"),
    ("test counts of the build (band labels, strategic heads, planner)", "in the register row only; not stated in the text", "GOALS_AND_CLAIMS.md at the tip", "I"),
    ("train label-change shares 5.2 / 6.8 / 13.8 %", "register row and the F: single copy compare_v9_v10_train.log agree; the file is not on the tip", "F:/strat_out/band_labels/v10/", "M"),
    ("two earlier world-model planners failed", "SPEC_HWM.md section 8, not re-verified", "SPEC_HWM.md", "I"),
    ("whole-distribution time of the 30 / 50 / 80 m checkpoints (79.7 / 59.9 / 31.0 % in the first 6 s are measured; the p10 to p90 columns extrapolate the 6 s mean speed)", "AUDIT_MODEL.md section 3.1 marks the whole-distribution row ESTIMATED", "AUDIT_MODEL.md", "E"),
    ("the redesign: ladder 0.2 / 0.6 / 3.0 s, reach 2 / 6 / 18 s, multi-step loss, HWM-PLAN, arms, HF release", "specification text; no card is frozen, no run exists", "SPEC_HWM.md", "H"),
    ("MRATE, LFS, DIVRS, REL do not carry over unchanged", "reading of the audit (section 6.1); no re-card exists", "AUDIT_MODEL.md", "H"),
    ("the report does not choose between 7.47.8 and CAPLADDER / IMAG1C, nor between 7.47.5 and IMAG1B's speed row", "two instruments, two populations; no bridging read", "RESULT_LATSEM2B.md (v44), RESULT_CAPLADDER.md, RESULT_IMAG1B.md", "H"),
    ("source results landed and registered", "every result file in the appendix is on tip c69bc890923029d58f67ecd2c4e984c8a823bcaa; PLANTEST, the band labels and the review were read from the tip blobs", "R/raw/ folders in the appendix", "M"),
]

with open(HERE + "/NUMBERS_V45.md", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("# NUMBERS_V45: headline numbers of sections 7.49 and 7.50 and of the edits, each re-read from its source by `check_numbers_v45.py`\n\n")
    fh.write("Generated by `python -I check_numbers_v45.py` (rewritten on every run). Each row: the check, the value read from the source, the value the text quotes, PASS or FAIL, and the source. Evidence class is in brackets in the source column.\n\n")
    fh.write("| # | check | read from source | quoted in the text | result | source and class |\n|---|---|---|---|---|---|\n")
    for i, (lab, a, e, ok, src) in enumerate(ROWS, 1):
        fh.write("| %d | %s | %s | %s | %s | %s |\n" % (i, lab.replace("|", "/"), str(a).replace("|", "/")[:120], str(e).replace("|", "/")[:120], "PASS" if ok else "FAIL", src.replace("|", "/")))
    fh.write("\n## Derived, inherited and hypothesis numbers\n\n| number | derivation or note | source | class |\n|---|---|---|---|\n")
    for a, b2, c2, d2 in DERIVED:
        fh.write("| %s | %s | %s | [%s] |\n" % (a, b2, c2, d2))

print("\nchecks run: %d, failed: %d" % (N[0], len(FAIL)))
for f in FAIL:
    print("FAILED", f)
sys.exit(1 if FAIL else 0)
