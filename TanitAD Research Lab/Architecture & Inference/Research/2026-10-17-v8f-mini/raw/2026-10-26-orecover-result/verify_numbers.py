#!/usr/bin/env python3
"""Independent re-read of every number in RESULT_V8F_ORECOVER.md from the JSONs.  Exit 0 only if every check passes.

It does NOT import build_result.py. It has its own JSON loader, its own formatter and its own tag logic, and it checks:
  A. the primary file's sha256 (pinned literal);
  B. every manifest entry: the text the builder wrote == the text recomputed here from the JSON;
  C. every manifest text occurs in the document, in order (consumed left to right);
  D. every numeric token left in the document after C is on the whitelist below (dates, section numbers, bars quoted from the PREREG, ...);
  E. semantic checks recomputed from the raw rows (the outcome HARMS, the guard flags, controls, cross-instrument agreement, reproduction of the reader,
     identity, the directional sentences of sections 3.5 and 5);
  F. register_cell.md avoids the lander's retraction-context words.
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "RESULT_V8F_ORECOVER.md")
MAN = os.path.join(HERE, "numbers_manifest.json")
READ_SHA = "a50dbba9fee93828b17b8b1b71335b2db73e249fd7ab9e78c9b4ea4e3577afd6"
fails, passes = [], []


def ok(name, cond, detail=""):
    (passes if cond else fails).append((name, detail))
    if not cond:
        print("FAIL", name, detail)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


man = json.load(open(MAN, encoding="utf-8"))
SRC = man["sources"]
_c = {}


def load(k):
    if k not in _c:
        _c[k] = json.load(open(SRC[k], encoding="utf-8"))
    return _c[k]


def dig(src, path):
    o = load(src)
    for part in path.split("|"):
        o = o[int(part)] if isinstance(o, list) else o[part]
    return o


def render(kind, x):
    if kind == "str":
        return "%s" % (x,)
    if kind == "yn":
        return ["no", "yes"][1 if x else 0]
    if kind == "i":
        return str(int(round(x)))
    if kind == "ic":
        return "{:,}".format(int(round(x)))
    m = re.fullmatch(r"([sf])(\d+)", kind)
    if m:
        return format(x, ("+" if m.group(1) == "s" else "") + "." + m.group(2) + "f")
    if kind == "e":
        return "%.1e" % x
    if kind == "g6":
        return "%.6g" % x
    if kind == "sha8":
        return str(x)[:8]
    if kind == "sha16":
        return str(x)[:16]
    raise SystemExit("unknown format " + kind)


def tag_text(sep, delta, direction):
    if sep is not True:
        return "not separated"
    if direction is None:
        return "separated (no direction label)"
    if direction == "lowerS":
        return "separated, S nearer than CTRV" if delta < 0 else "separated, S farther than CTRV"
    good = delta < 0 if direction == "lower" else delta > 0
    return "separated, ORC better" if good else "separated, ORC WORSE"


# ---- A ---------------------------------------------------------------------------------------------------------------------------------------------
ok("A1 primary file sha256", sha(SRC["READ"]) == READ_SHA, sha(SRC["READ"]))
doc = open(MD, encoding="utf-8").read()
REG = os.path.join(HERE, "register_cell.md")
reg = open(REG, encoding="utf-8").read()
entries_all = man["entries"]
split = man["split"]                      # entries[:split] belong to RESULT, entries[split:] to register_cell.md


def recompute(e):
    op = e["op"]
    if op == "V":
        return render(e["fmt"], dig(e["src"], e["path"]))
    if op.startswith("D:"):
        xa, xb = dig(*e["a"]), dig(*e["b"])
        x = {"D:div": lambda: abs(xa) / abs(xb), "D:sub": lambda: xa - xb, "D:abs": lambda: abs(xa), "D:mean": lambda: (xa + xb) / 2}[op]()
        return render(e["fmt"], x)
    if op == "LEN":
        return str(len(dig(e["src"], e["path"])))
    if op == "CNT":
        return str(len([k for k in dig(e["src"], e["path"]) if k == e["name"] or k.startswith(e["name"] + ":")]))
    if op == "TAG":
        n = dig(e["src"], e["base"])
        return tag_text(n.get("separated"), n.get(e["kd"]), e["direction"])
    raise SystemExit("unknown op " + op)


NUM = re.compile(r"(?<![\w.\x00])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![\w\x00])")
WL = [  # (regex, reason): a numeric token is allowed only if one of these matches a span that CONTAINS the token
    (r"20\d\d-\d\d-\d\d", "a date"),
    (r"^#+ [\d.]+", "a section number in a heading"),
    (r"(?i)sections? [\d.]+(\.\d)?", "a section reference"),
    (r"^\d+\. ", "a list ordinal"),
    (r"30 / 50 / 80 m|@30|@50|@80", "the registered checkpoint arcs"),
    (r"K = 30 \(6 s\)", "the DK horizon K = 30 (6 s)"),
    (r"3,000 (more )?steps", "the registered fine-tune length (PREREG Amendment 1; the run's own steps are a manifest value in section 2)"),
    (r"30 % of samples", "the registered perturbation share (PREREG section 1)"),
    (r"0\.3 \+/- 0\.02", "the registered tolerance of the reached-training control (PREREG section 3)"),
    (r"p = 0\.1", "the committed-next quote"),
    (r"F/2", "the bar name F/2"),
    (r"2,399 (windows|of 2,399)|2,399", "the registered grid count (GRID-COUNTS control: 2,399 / 141, checked in E-section)"),
    (r"141", "the registered episode count (GRID-COUNTS control)"),
    (r"\+0 \[0, 0\]|exactly \+0|exactly 0|0 \[0, 0\]", "the identity-control target"),
    (r"\[0, 1\]|ticks >= 1|>= 1", "a registered inequality / tick range"),
    (r"95 %", "the interval level"),
    (r"n: 0|0 \(not attempted\)", "an n of 0 for a cell that was not computed"),
    (r"22 tokens", "the goal vocabulary size"),
    (r"every 10 steps|m = 10", "the registered replan period (T1-CONFIG control)"),
    (r"1e-3", "K4's registered tolerance"),
    (r"K = 30", "the DK horizon"),
    (r"censored at a 30 s cap", "the TTC cap (taniteval lead_metrics default; also in the instrument's output)"),
    (r"^[|] (30|50|80) m [|]|^[|] s[01] [|] (30|50|80) m [|]", "an arc label in a table row"),
]


def covered(line, m):
    # A single-digit integer with no decimal point or thousands separator is a label or a count in prose (seed 0, tick 0, excludes 0, two seeds, box 2).
    # Every data-bearing number in these documents has a decimal point or a thousands separator, or comes through the manifest.
    if re.fullmatch(r"[-+]?\d", m.group(0)):
        return True
    for p, _ in WL:
        for mm in re.finditer(p, line):
            if mm.start() <= m.start() and m.end() <= mm.end():
                return True
    return False


def find_token(text, t, pos):
    """First occurrence of t at or after pos that is a whole token: not glued to a word character, a sign or a decimal point before it, nor to a word
    character, a decimal fraction or a thousands group after it (so the count '4' cannot be taken from inside '2,399' or '-0.0004')."""
    pre = r"(?<![" + chr(92) + "w.+-])" if not t[:1] in "+-" else r"(?<![" + chr(92) + "w.])"
    post = r"(?![" + chr(92) + "w]|[.,]" + chr(92) + "d)"
    for m in re.finditer(pre + re.escape(t) + post, text[pos:]):
        return pos + m.start()
    return -1


def check_doc(text, entries, label):
    # B: recompute each entry
    bad = 0
    for e in entries:
        t = recompute(e)
        if t != e["text"]:
            bad += 1
            print("B MISMATCH", label, e, "->", repr(t))
    ok(f"B[{label}] every manifest entry recomputed from the JSON equals what was written", bad == 0, f"{len(entries)} entries, {bad} mismatches")
    # C: each numeric text occurs in the document, in order
    pos, work, missing = 0, list(text), 0
    for e in entries:
        i = find_token(text, e["text"], pos)
        if i < 0:
            missing += 1
            print("C MISSING in order:", label, e["text"][:70], e.get("path") or e.get("a"))
            continue
        for j in range(i, i + len(e["text"])):
            work[j] = "\x00"
        pos = i + len(e["text"])
    ok(f"C[{label}] every manifest text found in the document, in order", missing == 0, f"{missing} missing")
    # D: nothing numeric left over except whitelisted tokens
    left = []
    for ln, line in enumerate("".join(work).split("\n"), 1):
        for m in NUM.finditer(line):
            if not covered(line, m):
                left.append((ln, m.group(0), line[max(0, m.start() - 45):m.end() + 45]))
    for r in left:
        print("D UNEXPLAINED numeric token", label, r)
    ok(f"D[{label}] no unexplained numeric token left", not left, f"{len(left)} left")


check_doc(doc, entries_all[:split], "RESULT")
check_doc(reg, entries_all[split:], "register_cell")

# (the register document is checked below in section F with the same B / C / D logic)
# ---- E ---------------------------------------------------------------------------------------------------------------------------------------------
R, F = load("READ"), load("FAM")
ok("E1 controls: all PASS and 104", all(R["controls_PASS"].values()) and len(R["controls_PASS"]) == 104, str(len(R["controls_PASS"])))
ok("E1b every control record also PASS", all(v["PASS"] for v in R["controls"].values()))
H = R["bars"]["F_over_2_t0_point_m"]
ade_worse = [R["rows"][s]["R-ADE"]["separated"] and R["rows"][s]["R-ADE"]["delta"] > 0 for s in ("s0", "s1")]
t0_above = [R["rows"][s]["R-T0"]["point"] > H for s in ("s0", "s1")]
ok("E2 HARMS recomputed from the rows", (any(ade_worse) or any(t0_above)) and R["OUTCOME"]["OUTCOME"] == "HARMS", f"ade_worse {ade_worse} t0_above {t0_above}")
ok("E2b guard flags equal the reader's", ade_worse == [R["OUTCOME"]["flags"]["R-ADE_separated_worse_per_seed"][s] for s in ("s0", "s1")]
   and t0_above == [R["OUTCOME"]["flags"]["R-T0_above_F_over_2_per_seed"][s] for s in ("s0", "s1")])
for hd, key in (("lat", "R-TLAT"), ("lon", "R-TLON")):
    for s in ("s0", "s1"):
        r = R["rows"][s][key]
        want = bool(r["separated"] and r["delta"] > 0 and r["delta"] >= r["two_F"])
        ok(f"E3 {key} {s} meets-2F recomputed", want == r["meets_2F"] == R["OUTCOME"]["flags"]["separated_better_and_ge_2F_per_seed"][hd][s])
ok("E4 R-ADE/R-T0 reproduce in the families driver (delta, lo, hi, separated)", all(
    F["controls"]["reproduces_reader_ade6"][s][t]["here"] == {k: R["rows"][s][rk][k] for k in ("delta", "lo", "hi", "separated")}
    for s in ("s0", "s1") for t, rk in (("T0", "R-T0"), ("T1", "R-ADE"))))
ok("E5 identity: every paired family row ORC vs ORC is exactly 0, DK identity too", all(
    all(F[t][s]["identity_all_exactly_zero"].values()) and all(F[t][s]["distance_keeping"]["identity_control"][k]["delta"] == 0.0 for k in ("headway_min_m", "time_gap_min_s", "min_ttc_s", "lead_viol"))
    for t in ("T0", "T1") for s in ("s0", "s1")))
ok("E6 T1 grid controls", all(all(v for k, v in c.items() if isinstance(v, bool)) and all(
    x["eid_equal_grid"] and x["ws_times2_equals_grid_frame"] and x["v0_max_abs_diff_vs_grid"] == 0.0 and x["g_max_abs_diff_vs_grid_gt_m"] < 1e-4 and x["n_windows"] == 2399
    for kk, x in c.items() if isinstance(x, dict)) for kk0, c in F["controls"].items() if kk0.startswith("T1_grid")))
for s in ("0", "1"):
    P = load("PF" + s)
    ok(f"E7 PF s{s} controls_ok and 2399/141", P["controls_ok"] is True and P["n_windows"] == 2399 and P["n_episodes"] == 141 and P["ages"] == "cycle")
    for a, b in (("lateral_decision", "tac_lat_acc"), ("longitudinal_decision", "tac_lon_acc")):
        x, y = P["tactical"]["plan_derived"][a]["accuracy"], F["T0"]["s" + s]["families_paired_ORC_minus_ORC0"][b]
        ok(f"E8 cross-instrument s{s} {a}", (x["delta"], x["lo"], x["hi"], x["separated"]) == (y["mean"], y["lo"], y["hi"], y["separated"]), f"{x['delta']} {y['mean']}")
ok("E9 lead block sha256 is the registered one", F["inputs"]["lead_block_sha256"] == "7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9")
ok("E10 same estimator and B everywhere", F["n_boot"] == R["n_boot"] == 2000 and load("PF0")["n_boot"] == 2000 and load("PF1")["n_boot"] == 2000)
# the committed next, verbatim, against the PREREG text saved from commit 07dcb513
pre = open(os.path.join(HERE, "_prereg.md"), encoding="utf-8").read().replace("**", "")
nxt = R["OUTCOME"]["committed_next"]
ok("E11 committed next is verbatim in the PREREG HARMS row", nxt in pre, nxt)
ok("E11b committed next is verbatim in the RESULT", doc.count(nxt) >= 2, str(doc.count(nxt)))

# directional sentences of 3.5 and 5, recomputed
def fam(t, s, k):
    return F[t]["s" + s]["families_paired_ORC_minus_ORC0"][k]


def worse(c):
    return bool(c["separated"] and c["mean"] > 0)


def better(c):
    return bool(c["separated"] and c["mean"] < 0)


ok("E12 T0 speed/accel/along separated-worse at both seeds", all(worse(fam("T0", s, k)) for s in "01" for k in ("lon_speed_mae", "lon_accel_mae", "lon_along_mae")))
ok("E12b T1 accel worse at s0 only; speed and along not separated at both seeds",
   worse(fam("T1", "0", "lon_accel_mae")) and not fam("T1", "1", "lon_accel_mae")["separated"]
   and not any(fam("T1", s, k)["separated"] for s in "01" for k in ("lon_speed_mae", "lon_along_mae")))
ok("E13 T1 heading/curvature/yaw-rate separated-better at both seeds", all(better(fam("T1", s, k)) for s in "01" for k in ("lat_heading_mae_deg", "lat_curvature_mae_1pm", "lat_yaw_rate_mae_degps")))
ok("E13b T0 curvature and yaw-rate separated-better at s0 only; cross-track never better",
   better(fam("T0", "0", "lat_curvature_mae_1pm")) and better(fam("T0", "0", "lat_yaw_rate_mae_degps"))
   and not fam("T0", "1", "lat_curvature_mae_1pm")["separated"] and not fam("T0", "1", "lat_yaw_rate_mae_degps")["separated"]
   and not any(better(fam(t, s, "lat_cross_mae")) for t in ("T0", "T1") for s in "01"))
ok("E14 plan-derived lateral agreement: T1 separated-better both seeds; T0 better s0, worse s1",
   all(fam("T1", s, "tac_lat_acc")["separated"] and fam("T1", s, "tac_lat_acc")["mean"] > 0 for s in "01")
   and fam("T0", "0", "tac_lat_acc")["separated"] and fam("T0", "0", "tac_lat_acc")["mean"] > 0
   and fam("T0", "1", "tac_lat_acc")["separated"] and fam("T0", "1", "tac_lat_acc")["mean"] < 0)
ok("E14b head rows: lat n.s. both seeds, lon n.s. s0 and separated-better s1",
   not R["rows"]["s0"]["R-TLAT"]["separated"] and not R["rows"]["s1"]["R-TLAT"]["separated"] and not R["rows"]["s0"]["R-TLON"]["separated"]
   and R["rows"]["s1"]["R-TLON"]["separated"] and R["rows"]["s1"]["R-TLON"]["delta"] > 0)
b = R["beside"]["per_seed"]
ok("E15 tick-0 F1: ORC below ORC-0 on both heads at both seeds", all(b[s]["tick0_f1"]["ORC"][h] < b[s]["tick0_f1"]["ORC0"][h] for s in ("s0", "s1") for h in ("lat", "lon")))
ok("E16 lon F1 sign flips across seeds", R["rows"]["s0"]["R-TLON"]["delta"] < 0 < R["rows"]["s1"]["R-TLON"]["delta"])
ok("E17 lat F1 points both above 2F with neither separated", all(R["rows"][s]["R-TLAT"]["delta"] > R["bars"]["2F"]["lat"] and not R["rows"][s]["R-TLAT"]["separated"] for s in ("s0", "s1")))
ok("E18 own-cells lat s1 separated, s0 not (the disclosure)", R["rows"]["s1"]["R-TLAT_own_cells_beside"]["separated"] and not R["rows"]["s0"]["R-TLAT_own_cells_beside"]["separated"])
ok("E19 T1 anti-echo: not admissible and flagged, all four runs", all(
    load(f"T1J_{a}_s{s}")["arms"]["cl"]["four_families"]["longitudinal"]["anti_echo"]["longitudinal_claim_admissible"] is False for a in ("ORC", "ORC0") for s in (0, 1)))
def dkc(t, s, sub):
    return F[t]["s" + s]["distance_keeping"]["pair"][sub]


ok("E23 DK sentence: LEAD-VIOL never separated; T1 headway separated-lower both seeds; T1 time gap separated-lower s0 (not s1); T0 time gap separated-lower both seeds; T0 headway not separated",
   all(not dkc(t, s, "lead_viol")["at_risk"]["separated"] and not dkc(t, s, "lead_viol")["all_lead_windows"]["separated"] for t in ("T0", "T1") for s in "01")
   and all(dkc("T1", s, "dk")["headway_min_m"]["separated"] and dkc("T1", s, "dk")["headway_min_m"]["delta"] < 0 for s in "01")
   and dkc("T1", "0", "dk")["time_gap_min_s"]["separated"] and dkc("T1", "0", "dk")["time_gap_min_s"]["delta"] < 0 and not dkc("T1", "1", "dk")["time_gap_min_s"]["separated"]
   and all(dkc("T0", s, "dk")["time_gap_min_s"]["separated"] and dkc("T0", s, "dk")["time_gap_min_s"]["delta"] < 0 for s in "01")
   and not any(dkc("T0", s, "dk")["headway_min_m"]["separated"] for s in "01"))
ok("E24 T0 lat head F1 not separated at either seed (the section-5 item-0 sentence)", not load("PF0")["tactical"]["heads"]["lateral"]["macro_f1"]["separated"] and not load("PF1")["tactical"]["heads"]["lateral"]["macro_f1"]["separated"])
ok("E21 T0 STRATEGIC: no checkpoint position or heading cell is separated at either seed", all(
    not load("PF" + s)["strategic"][k][a]["separated"] for s in "01" for k in ("position_error_m", "heading_error_deg") for a in ("30", "50", "80")))
P0, P1 = load("PF0")["tactical"]["heads"], load("PF1")["tactical"]["heads"]
ok("E22 T0 head rows: lat and lon macro-F1 not separated at either seed; goal head F1 separated-worse at s0 and not separated at s1",
   not any(P[h]["macro_f1"]["separated"] for P in (P0, P1) for h in ("lateral", "longitudinal"))
   and P0["goal"]["macro_f1"]["separated"] and P0["goal"]["macro_f1"]["delta"] < 0 and not P1["goal"]["macro_f1"]["separated"])
occ = [m.start() for m in re.finditer("closed loop", doc.lower())]
ok("E20 every 'closed loop' in the RESULT sits in a negation (never / does not occur)", bool(occ) and all(
    ("never" in doc.lower()[max(0, i - 40):i]) or ("does not occur" in doc.lower()[i:i + 60]) for i in occ), str(len(occ)))

# ---- F ---------------------------------------------------------------------------------------------------------------------------------------------
rc = open(os.path.join(HERE, "register_cell.md"), encoding="utf-8").read()
bad = [w for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction") if w in rc.lower()]
ok("F1 register_cell.md has none of the lander's retraction-context words", not bad, str(bad))
for w in ("HARMS",):
    ok(f"F2 register_cell.md names {w}", w in rc)
ok("F3 register_cell.md quotes the committed next verbatim", nxt in rc)

print()
print(f"PASSED {len(passes)}   FAILED {len(fails)}")
sys.exit(1 if fails else 0)
