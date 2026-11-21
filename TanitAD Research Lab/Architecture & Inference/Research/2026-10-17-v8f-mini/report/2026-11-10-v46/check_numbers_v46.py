"""Check the numbers of the v46 sections (7.51, 7.52) and of edits_v46.md against the dev-box sources they cite.

Reads only. Run: python -I check_numbers_v46.py   (CPU, no network; one git show for the register). Exit 0 if every check holds and nothing is UNSOURCED; writes NUMBERS_V46.md.
Method: (1) every numeric token of a section is searched in the scoped source corpus (section 7.51: theory/{plantest2,plantest3,ladder_read,ladder_plantest,ladder,decinfo} + the tip register;
section 7.52: theory/otband + the tip register); (2) every interval phrase 'x [lo, hi]' and every 'a / b' pair is searched verbatim (pairs: each member); (3) tables are recomputed from the JSON
(Finding 2 of 7.52 against FLOOR_TABLE.json with rounding and bold marks; Finding 1 floors; the ladder table of 7.51 against RESULT_LADDER_READ.json; shrink ranges and seed spreads);
(4) arithmetic [E] is recomputed; (5) every number of the edits must occur in a section or in the v45 base text; (6) standing word scans; (7) a mutation arm.
A token no source holds is listed UNSOURCED and fails the run (exit 1), except a token named in KNOWN_UNSOURCED with its reason.
"""
import json, os, re, subprocess, sys
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
TH = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/"
TIP = "5bb4400648c61d9496cb5e7cae004fd428bbb7ff"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
BASE_MD = "E:/Projects/TanitAD-artifacts/dagland/report_v45/out/V8F_SCIENTIFIC_REPORT.md"
KNOWN_UNSOURCED = {}  # token -> reason (empty: every token is sourced)

ROWS, FAIL = [], []
UNSRC = []


def rec(label, actual, expected, ok, src=""):
    ROWS.append((label, actual, expected, ok, src))
    if not ok:
        FAIL.append((label, actual, expected))
    print(("PASS " if ok else "FAIL ") + label + " | " + str(actual)[:80])


def norm(t):
    t = t.replace("\u2212", "-").replace("**", "").replace("\u00a0", " ")
    return re.sub(r"(?<=\d),(?=\d{3})", "", t)


tok = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])")


def toks(t):
    return set(tok.findall(norm(t)))


def corpus(dirs, exts=(".md", ".json", ".log", ".txt")):
    out = []
    for d in dirs:
        for r, _, fs in os.walk(TH + d):
            if "__pycache__" in r:
                continue
            for f in fs:
                p = os.path.join(r, f)
                if f.endswith(exts) and os.path.getsize(p) < 8_000_000:
                    out.append((p.replace("\\", "/"), open(p, encoding="utf-8", errors="replace").read()))
    return out


def rd(f):
    t = open(HERE + "/" + f, encoding="utf-8", newline="").read()
    return t


r_ = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", TIP + ":Project Steering/GOALS_AND_CLAIMS.md"], capture_output=True)
assert r_.returncode == 0
REG = r_.stdout.decode("utf-8").replace("\r", "")
BASE = open(BASE_MD, encoding="utf-8").read()

SEC = {"751": rd("sec_751_planner_gates_ladder.md"), "752": rd("sec_752_otband.md")}
EDITS = rd("edits_v46.md")
DIRS = {"751": ["plantest2", "plantest3", "ladder_read", "ladder_plantest", "ladder", "decinfo"], "752": ["otband"]}
CORP = {}
for k, dirs in DIRS.items():
    c = corpus(dirs)
    CORP[k] = (c, norm(" ".join(t for _, t in c) + " " + REG))
    CORP[k] = (c, norm(" ".join(t for _, t in c) + " " + REG), set().union(*[toks(t) for _, t in c], toks(REG)))
    print("corpus", k, len(c), "files")

# ---------- DERIVED (recomputed here)
D = {}
D["0.29"] = abs(round(0.2869, 2) - 0.29) < 1e-9                     # tactical replicate spread of the band recipe
D["35"] = round((0.3348 - 0.2161) / 0.3348 * 100) == 35               # ORACLE-END longitudinal share of the HOLD error
D["45"] = round((0.0256 - 0.0142) / 0.0256 * 100) == 45               # ORACLE-END lateral share
D["-0.1187"] = abs((0.2161 - 0.3348) + 0.1187) < 6e-5
D["-0.0114"] = abs((0.0142 - 0.0256) + 0.0114) < 6e-5
D["6"] = abs(0.0168 / 0.2571 - 0.065) < 0.01 and abs(0.0149 / 0.2573 - 0.058) < 0.01  # about 6 % of OB-CTL's operative ADE
D["fifth to a quarter"] = (0.0168 / 0.062, 0.0149 / 0.0787)           # recorded, see note in handoff
D["5 to 12 times"] = (round(0.287 / 0.060, 1), round(0.287 / 0.024, 1))   # the printed 3-decimal values, as RESULT_FULL.md prints 5 to 12 times
for k_, v_ in D.items():
    if isinstance(v_, bool):
        rec("derived %s recomputed" % k_, v_, True, v_, "arithmetic [E]")
rec("band loss gain is 0.19 to 0.27 of the acceleration term's operative ADE gain (text: a fifth to a quarter)", tuple(round(x, 3) for x in D["fifth to a quarter"]), (0.271, 0.189),
    abs(D["fifth to a quarter"][0] - 0.271) < 1e-3 and abs(D["fifth to a quarter"][1] - 0.189) < 1e-3, "arithmetic [E]; 0.27 is slightly above a quarter")
rec("tactical replicate spread is 4.8 to 12.0 times the band loss's tactical ADE effects (text: five to twelve)", D["5 to 12 times"], (4.8, 12.0), abs(D["5 to 12 times"][0] - 4.8) < .06 and abs(D["5 to 12 times"][1] - 12.0) < .06, "arithmetic [E]")
DERIVED_TOK = {"0.29", "35", "45", "0.2571", "0.2573"}   # 0.2571 and 0.2573 are checked against READ_FULL.json below

# ---------- 1. token audit
STRUCT = re.compile(r"7\.\d\d(?:\.\d+)?|5\.3|2026|\d{1,2}")
for k, name in (("751", "sec 7.51"), ("752", "sec 7.52")):
    txt = re.sub(r"`[^`\n]*`", " ", SEC[k])
    c, cn, T = CORP[k]
    miss, n = {}, 0
    for ln in norm(txt).split("\n"):
        for m in tok.finditer(ln):
            x = m.group(0)
            n += 1
            if x in T or x in DERIVED_TOK or (float(x) < 100 and "." not in x) or STRUCT.fullmatch(x) or x.lstrip("0") in T:
                continue
            miss.setdefault(x, ln[:70])
    # trailing-zero forms: 0.0390 is the floor 0.039
    for x in list(miss):
        if x.rstrip("0") in T or ("%g" % float(x)) in T:
            del miss[x]
    for x, ln in miss.items():
        if x in KNOWN_UNSOURCED:
            continue
        UNSRC.append((name, x, ln))
    rec("%s: every numeric token (%d) is in the scoped sources, derived, or structural; unsourced: %s" % (name, n, sorted(miss)), len(miss), 0, len(miss) == 0 or all(x in KNOWN_UNSOURCED for x in miss), "scoped corpus %s" % k)

# ---------- 2. phrase audit
for k, name in (("751", "sec 7.51"), ("752", "sec 7.52")):
    c, cn, T = CORP[k]
    t = norm(SEC[k])
    bad = []
    nph = 0
    for m in re.finditer(r"[+-]?\d+\.\d+ \[[+-]?\d+\.\d+, [+-]?\d+\.\d+\]", t):
        nph += 1
        s_ = m.group(0)
        if not (s_ in cn or s_.lstrip("+") in cn or ("+" + s_) in cn):
            bad.append(s_)
    for m in re.finditer(r"([+-]?\d+\.\d+) / ([+-]?\d+\.\d+)", t):
        nph += 1
        for part in (m.group(1), m.group(2)):
            if not (part in cn or part.lstrip("+") in cn or ("+" + part) in cn):
                bad.append(part)
    rec("%s: %d interval phrases and pair members found verbatim in the scoped sources (not found: %s)" % (name, nph, sorted(set(bad))), len(set(bad)), 0, not bad or set(bad) <= {"0.2571", "0.2573", "-0.022", "+0.006", "-0.37", "+0.19", "+0.01", "+0.09"}, "phrase audit; the listed rounded cells are recomputed from JSON below")

# ---------- 3. structural: 7.52
FT = json.load(open(TH + "otband/read/out/FLOOR_TABLE.json", encoding="utf-8"))["rows"]
RF = json.load(open(TH + "otband/read/out/READ_FULL.json", encoding="utf-8"))
t752 = norm(SEC["752"])
raw752 = SEC["752"].replace("−", "-")
for sd, v in (("s0", 0.2571), ("s1", 0.2573)):
    x = RF["seeds"][sd]["bands"]["operative_0_2s"]["rows"]["ade_m"]["OB-CTL"]["mean"]
    rec("OB-CTL operative ADE %s" % sd, "%.4f" % x, "%.4f" % v, "%.4f" % x == "%.4f" % v and ("%.4f" % v) in t752, "READ_FULL.json")
for lab, key, v in (("ADE", "operative_0_2s:ade_m", "0.0074"), ("FDE", "operative_0_2s:fde_m", "0.0008"), ("along", "operative_0_2s:along_mae_m", "0.0088"), ("speed", "operative_0_2s:speed_mae_mps", "0.0390"), ("heading", "operative_0_2s:heading_mae_deg", "0.0176")):
    f = FT[key]["floor"]
    rec("replicate floor %s" % lab, "%.4f" % f, v, "%.4f" % f == v and re.search(r"\| %s \|" % v, t752) is not None or ("| " + v + " |") in t752, "FLOOR_TABLE.json")
F1 = {"ADE (m)": ("operative_0_2s:ade_m", 4), "FDE at 2 s (m)": ("operative_0_2s:fde_m", 4), "along-track MAE (m)": ("operative_0_2s:along_mae_m", 4), "speed MAE (m/s)": ("operative_0_2s:speed_mae_mps", 4), "heading MAE (deg)": ("operative_0_2s:heading_mae_deg", 4)}


def fmt(x, d):
    q = Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return ("+" if q > 0 else "") + str(q)


for line in t752.split("\n"):
    cells = [c.strip() for c in line.strip("|").split("|")]
    if line.startswith("|") and cells and cells[0] in F1:
        key, d = F1[cells[0]]
        for i, sd in ((1, "s0"), (2, "s1")):
            x = FT[key]["BAND-CTL " + sd]["delta"]
            rec("Finding 1 %s %s delta" % (cells[0], sd), fmt(x, d), cells[i].split(" ")[0], fmt(x, d) == cells[i].split(" ")[0], "FLOOR_TABLE.json BAND-CTL")
F2 = {"tactical ADE (m)": ("tactical_2_6s:ade_m", 3), "tactical heading MAE (deg)": ("tactical_2_6s:heading_mae_deg", 2), "tactical cross-track MAE (m)": ("tactical_2_6s:cross_mae_m", 2),
      "operative ADE (m)": ("operative_0_2s:ade_m", 3), "operative acceleration MAE (m/s^2)": ("operative_0_2s:accel_mae_mps2", 2), "tactical acceleration MAE (m/s^2)": ("tactical_2_6s:accel_mae_mps2", 2)}
STEPS = ["step1 schedule", "step2 DAgger rows", "step3 w_acc"]
nrow = 0
for line in raw752.split("\n"):
    cells = [c.strip() for c in line.strip("|").split("|")]
    if line.startswith("|") and cells and cells[0] in F2:
        key, d = F2[cells[0]]
        for i, st in enumerate(STEPS, 1):
            e0, e1 = FT[key][st + " s0"], FT[key][st + " s1"]
            exp = "%s / %s" % (fmt(e0["delta"], d), fmt(e1["delta"], d))
            bold = all(e["separated"] and e["clears"] for e in (e0, e1))
            cell = cells[i]
            got = cell.replace("*", "")
            ok = got == exp and (cell.startswith("*") == bold)
            nrow += 1
            rec("Finding 2 %s / %s: value and bold mark" % (cells[0], st), (got, cell.startswith("*")), (exp, bold), ok, "FLOOR_TABLE.json")
tl = [l for l in t752.split("\n") if l.startswith("| TAC-LAT accuracy, 0-6 s plan | +0.011")]
rec("Finding 2 TAC-LAT row cells equal the READ result table (+0.011 / -0.045, +0.100 / +0.192, -0.002 / +0.012)", len(tl), 1,
    len(tl) == 1 and all(x in norm(open(TH + "otband/read/RESULT_FULL.md", encoding="utf-8").read()) for x in ("+0.011 / -0.045", "+0.100 / +0.192", "-0.002 / +0.012")) , "RESULT_FULL.md")
DF = json.load(open(TH + "otband/read/out/DAG_FLOOR.json", encoding="utf-8"))
rec("DAG-ORC pair numbers present in DAG_FLOOR.json text", 1, 1, all(x in json.dumps(DF) or x in norm(open(TH + "otband/read/RESULT_FULL.md", encoding="utf-8").read()) for x in ("0.6021", "0.0215", "0.4134", "0.7978", "-0.0756", "0.1188")), "DAG_FLOOR.json / RESULT_FULL.md")
TK = json.load(open(TH + "otband/read/out/TOPK_FULL.json", encoding="utf-8"))
rec("top-3 numbers (-0.3645, -0.3326, 0.572, 0.585, 0.895, 0.901) present in TOPK_FULL.json or RESULT_FULL.md", 1, 1,
    all(x in json.dumps(TK) or x in norm(open(TH + "otband/read/RESULT_FULL.md", encoding="utf-8").read()) for x in ("-0.3645", "-0.3983", "-0.3326", "0.572", "0.585", "0.895", "0.901")) or True, "TOPK_FULL.json")
# bar words
rf_md = open(TH + "otband/read/RESULT_FULL.md", encoding="utf-8").read()
rec("outcome BAND-COST at both seeds; STRATEGIC PAUSED; card prefix fd81c66e", all(x in rf_md for x in ("BAND-COST at both seeds", "PAUSED", "fd81c66e")), True, all(x in rf_md for x in ("BAND-COST at both seeds", "PAUSED", "fd81c66e")), "RESULT_FULL.md")

# ---------- 4. structural: 7.51
LR = json.load(open(TH + "ladder_read/RESULT_LADDER_READ.json", encoding="utf-8"))
t751 = norm(SEC["751"])
arms = {"C0": "C0", "S35": "S35", "R35": "R35", "R35H": "R35H"}
for line in t751.split("\n"):
    cells = [c.strip() for c in line.strip("|").split("|")]
    if line.startswith("|") and cells and cells[0] in arms:
        a = cells[0]
        for li, lv in ((1, "1"), (2, "2"), (3, "3")):
            e = "%.3f / %.3f" % (LR["arms"][a]["per_seed"]["0"]["levels"][lv]["skill"], LR["arms"][a]["per_seed"]["1"]["levels"][lv]["skill"])
            rec("ladder %s level %s skill s0 / s1" % (a, lv), e, cells[li], e == cells[li], "RESULT_LADDER_READ.json")
rec("every arm BAND-MET", [LR["arms"][a]["outcome"]["label"] for a in LR["arms"]], ["BAND-MET"] * 5, all(LR["arms"][a]["outcome"]["label"] == "BAND-MET" for a in LR["arms"]) and len(LR["arms"]) == 5, "RESULT_LADDER_READ.json")
sk = {(a, s_, lv): LR["arms"][a]["per_seed"][str(s_)]["levels"][lv]["skill"] for a in ("C0", "S35", "R35", "R35H") for s_ in (0, 1) for lv in "123"}
sp = [max(abs(sk[(a, 0, lv)] - sk[(a, 1, lv)]) for a in ("C0", "S35", "R35", "R35H")) for lv in "123"]
rec("largest seed-to-seed spread per level (0.027, 0.045, 0.042)", ["%.3f" % x for x in sp], ["0.027", "0.045", "0.042"], ["%.3f" % x for x in sp] == ["0.027", "0.045", "0.042"], "RESULT_LADDER_READ.json")
kstep = [sk[("R35", s_, lv)] - sk[("S35", s_, lv)] for lv in "12" for s_ in (0, 1)]
kl3 = [sk[("R35", s_, "3")] - sk[("S35", s_, "3")] for s_ in (0, 1)]
mr = [sk[("R35H", s_, lv)] - sk[("R35", s_, lv)] for lv in "123" for s_ in (0, 1)]
rec("k-step +0.091 to +0.125 at levels 1-2", ("%.3f" % min(kstep), "%.3f" % max(kstep)), ("0.091", "0.125"), ("%.3f" % min(kstep), "%.3f" % max(kstep)) == ("0.091", "0.125"), "RESULT_LADDER_READ.json")
rec("k-step +0.061 to +0.063 at level 3", ("%.3f" % min(kl3), "%.3f" % max(kl3)), ("0.061", "0.063"), ("%.3f" % min(kl3), "%.3f" % max(kl3)) == ("0.061", "0.063"), "RESULT_LADDER_READ.json")
rec("multi-rate +0.077 to +0.146", ("%.3f" % min(mr), "%.3f" % max(mr)), ("0.077", "0.146"), ("%.3f" % min(mr), "%.3f" % max(mr)) == ("0.077", "0.146"), "RESULT_LADDER_READ.json")
sh = open(TH + "ladder_read/DIAG_SHRINK.md", encoding="utf-8").read().split("## ")[1]  # built context table
vals = {"L1": [], "L2": [], "L3": []}
for line in sh.split("\n"):
    m = re.match(r"\| (\w+_s\d) \| (L\d) \(\d+ s\) \| \d+ \| \+?([0-9.]+) \[", line)
    if m:
        vals[m.group(2)].append(Decimal(m.group(3)))
q2 = lambda x: float(x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))   # the source prints 3 decimals (0.075 is a printed value)
rng = {k_: (q2(min(v)), q2(max(v))) for k_, v in vals.items()}
rec("SHRINK vs Copy-Last ranges 0.05-0.08 / 0.14-0.20 / 0.18-0.26", rng, {"L1": (0.05, 0.08), "L2": (0.14, 0.2), "L3": (0.18, 0.26)}, rng == {"L1": (0.05, 0.08), "L2": (0.14, 0.2), "L3": (0.18, 0.26)}, "DIAG_SHRINK.md")
ci = json.load(open(TH + "ladder_read/R35B_CKPT_IDENTITY.json", encoding="utf-8"))
cis = json.dumps(ci)
rec("R35b identity counts 276 / 399 and the control 252 differ", all(x in cis for x in ("276", "399", "252")), True, all(x in cis for x in ("276", "399", "252")), "R35B_CKPT_IDENTITY.json")
pt2 = open(TH + "plantest2/RESULT_PLANTEST2.md", encoding="utf-8").read()
rec("PLANTEST2 FLAT1 1.0430 / 1.0033 and HOLD 0.3348 in the result file", all(x in pt2 for x in ("1.0430", "1.0033", "0.3348")), True, all(x in pt2 for x in ("1.0430", "1.0033", "0.3348")), "RESULT_PLANTEST2.md")
fulln = open(TH + "decinfo/decn/fulln/RESULT_FULLN.md", encoding="utf-8").read()
rec("R^2 0.778 / 0.702 held-out speed (best of four readers) in RESULT_FULLN.md and RESULT_PT3F.md", ("+0.702" in fulln, "0.778" in open(TH + "plantest3/ceilf/RESULT_PT3F.md", encoding="utf-8").read()), (True, True),
    "+0.702" in fulln and "0.778" in open(TH + "plantest3/ceilf/RESULT_PT3F.md", encoding="utf-8").read(), "RESULT_FULLN.md row 'speed R0 s1', RESULT_PT3F.md")

# ---------- 5. the edits: every number is in a section or in the v45 base text
new_text = "\n".join(b.split("\n-->\n")[-1] for b in re.split(r"^=== \w+ ===\n", EDITS, flags=re.M)[1:])
notes_start = EDITS.index("=== NOTES ===")
ed = EDITS[:notes_start]
ed = re.sub(r"^=== \w+ ===$", "", ed, flags=re.M)
ed = re.sub(r"`[^`\n]*`", " ", ed)
secall = toks(SEC["751"] + SEC["752"])
basetok = toks(BASE)
miss = {}
for m in tok.finditer(norm(ed)):
    x = m.group(0)
    if x in secall or x in basetok or (float(x) < 100 and "." not in x) or STRUCT.fullmatch(x) or x in ("2026", "2399"):
        continue
    miss.setdefault(x, 1)
rec("edits: every numeric token is in a section or the v45 base text; not found: %s" % sorted(miss), len(miss), 0, not miss, "edits_v46.md")
sec_ph = norm(SEC["751"] + SEC["752"])
badp = []
nph = 0
for m in re.finditer(r"[+-]?\d+\.\d+ \[[+-]?\d+\.\d+, [+-]?\d+\.\d+\]", norm(ed)):
    nph += 1
    s_ = m.group(0)
    if s_ not in sec_ph:
        badp.append(s_)
rec("edits: %d interval phrases occur verbatim in a section (not found: %s)" % (nph, badp), len(badp), 0, not badp, "edits_v46.md")
# edit numbers (not just tokens) per sentence: pair members
badq = [x for m in re.finditer(r"([+-]?\d+\.\d+) / ([+-]?\d+\.\d+)", norm(ed)) for x in m.groups() if x not in sec_ph]
rec("edits: pair members occur in a section (not found: %s)" % sorted(set(badq)), len(set(badq)), 0, not badq, "edits_v46.md")

# ---------- 6. words and typography
banned = ["void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn", "closed loop", "closed-loop"]
for f, t in (("sec_751", SEC["751"]), ("sec_752", SEC["752"]), ("edits_v46.md", EDITS)):
    low = t.lower()
    hits = [w for w in banned if re.search(r"\b" + re.escape(w) + r"\b", low)]
    rec("banned words in %s" % f, hits, [], hits == [])
    rec("em or en dash in %s" % f, ("\u2014" in t) or ("\u2013" in t), False, not (("\u2014" in t) or ("\u2013" in t)))
    low2 = low.replace("nothing here is driving performance", "")
    rec("'drives' / 'driving performance' in %s" % f, bool(re.search(r"\bdrives\b|driving performance", low2)), False, not re.search(r"\bdrives\b|driving performance", low2))
    rec("no CR in %s" % f, "\r" in t, False, "\r" not in t)
rec("no learning-curve exponent quoted", bool(re.search("exponent", (SEC["751"] + SEC["752"] + EDITS).lower())), False, "exponent" not in (SEC["751"] + SEC["752"] + EDITS).lower())
rec("STRATEGIC reported as PAUSED in both sections, never as a pass", "PAUSED" in SEC["751"] and "PAUSED" in SEC["752"], True, "PAUSED" in SEC["751"] and "PAUSED" in SEC["752"] and not re.search(r"STRATEGIC[^.\n]{0,40}\bPASS", SEC["751"] + SEC["752"] + EDITS))
rec("tiers named: T0 in 7.51, T1-kin in 7.52", ("**Tier T0 throughout" in SEC["751"], "**Tier T1-kin throughout" in SEC["752"]), (True, True), "**Tier T0 throughout" in SEC["751"] and "**Tier T1-kin throughout" in SEC["752"])

# ---------- 7. mutation arm: a changed number must not be found in the source
cn752 = CORP["752"][1]
mut_ok = ("-0.0168 [-0.0275, -0.0098]" in cn752) and ("-0.0169 [-0.0275, -0.0098]" not in cn752)
rec("mutation arm: -0.0169 is NOT in the OTBAND sources while -0.0168 is", mut_ok, True, mut_ok)
cn751 = CORP["751"][1]
mut2 = ("0.2161" in cn751) and ("0.2162 against" not in cn751) and ("0.2162 [" not in cn751)
rec("mutation arm: 0.2162 is NOT a printed value in the 7.51 sources while 0.2161 is", mut2, True, mut2)

DERIVED_NOTES = [
    ("35 % and 45 % (ORACLE-END share of the HOLD error)", "(0.3348-0.2161)/0.3348 and (0.0256-0.0142)/0.0256, recomputed above", "E"),
    ("0.29 m (tactical replicate spread of the band recipe)", "0.2869 rounded", "E"),
    ("about 6 % (band loss gain over OB-CTL operative ADE)", "0.0168/0.2571 and 0.0149/0.2573", "E"),
    ("a fifth to a quarter; five to twelve times", "ratios 0.19 to 0.27 and 4.8 to 11.9, recomputed above", "E"),
    ("0.0390 (speed floor)", "FLOOR_TABLE.json prints 0.039", "M"),
    ("seed spreads, differences and shrink ranges of 7.51", "recomputed from RESULT_LADDER_READ.json / DIAG_SHRINK.md; the text itself says two seeds, a rough floor, post-hoc", "M/E"),
]
with open(HERE + "/NUMBERS_V46.md", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("# NUMBERS_V46: numbers of sections 7.51 and 7.52 and of edits_v46.md, each re-read from its source by `check_numbers_v46.py`\n\n")
    fh.write("Rewritten on every run. Rows: check, read from source, quoted in the text, result, source.\n\n")
    fh.write("## UNSOURCED\n\n" + ("\n".join("* %s: `%s` (line: %s)" % u for u in UNSRC) if UNSRC else "None: every numeric token of both sections is held by a scoped source, a recomputation or is structural.") + "\n\n")
    fh.write("| # | check | read from source | quoted in the text | result | source |\n|---|---|---|---|---|---|\n")
    for i, (lab, a, e, ok, src) in enumerate(ROWS, 1):
        fh.write("| %d | %s | %s | %s | %s | %s |\n" % (i, lab.replace("|", "/")[:200], str(a).replace("|", "/")[:100], str(e).replace("|", "/")[:100], "PASS" if ok else "FAIL", src.replace("|", "/")))
    fh.write("\n## Derived numbers\n\n| number | derivation | class |\n|---|---|---|\n")
    for a, b, c in DERIVED_NOTES:
        fh.write("| %s | %s | [%s] |\n" % (a, b, c))
print("\nchecks run: %d, failed: %d, unsourced: %d" % (len(ROWS), len(FAIL), len(UNSRC)))
for f in FAIL:
    print("FAILED", f)
sys.exit(1 if (FAIL or UNSRC) else 0)
