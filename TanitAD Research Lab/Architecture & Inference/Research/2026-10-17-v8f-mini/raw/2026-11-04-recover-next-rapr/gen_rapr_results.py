"""Write RESULTS.md for the RECOVER-NEXT RAPR arm read (the a-priori deviation law) and its committed next arm, from the read JSONs
(every number is read from a file, none typed).

Inputs: read/raw/RECOVER_READ_RAPR.json (this arm), read/raw/RECOVER_READ_RTL.json and RECOVER_READ_RP01.json (the attribution arms),
read/raw/RECOVER_NEXT.json (the combiner, run after the RAPR read), recover_read/raw/RECOVER_READ.json (R, the reference), the pinned
smooth_t1_read.py for the row -> family map. The next arm: card section 4 (the combiner's `next_arm`), then addendum 1 rows A / B, then
addendum 2 rows E-H (PREREG_RECOVER_NEXT_ADDENDUM2_RAPR.md, frozen 2026-11-04 14:35:47 before any RAPR checkpoint). Output: RESULTS.md
next to this script. Dry run (no RAPR read yet): `gen_rapr_results.py --dry-as-RP01 <out.md>` renders with RP01's read in RAPR's place.
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NX = HERE.parent
DRY = sys.argv[1] if len(sys.argv) > 1 else None   # dry run only: "--dry-as-RP01" reads RP01's read in RAPR's place, output to argv[2]
ARM = NX / "read" / "raw" / ("RECOVER_READ_RP01.json" if DRY else "RECOVER_READ_RAPR.json")
RTL = NX / "read" / "raw" / "RECOVER_READ_RTL.json"
RP1 = NX / "read" / "raw" / "RECOVER_READ_RP01.json"
NEXT = NX / "read" / "raw" / "RECOVER_NEXT.json"
RREF = NX.parent / "recover_read" / "raw" / "RECOVER_READ.json"
SRP = NX.parent / "smooth_t1" / "code" / "smooth_t1_read.py"

spec = importlib.util.spec_from_file_location("smooth_t1_read", SRP)
SR = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SRP.parent))
spec.loader.exec_module(SR)
FAM = SR.FAMILY
_gs = importlib.util.spec_from_file_location("gapdecomp", NX.parent / "gapdecomp" / "gapdecomp.py")
GD = importlib.util.module_from_spec(_gs)
_gs.loader.exec_module(GD)

d = json.load(open(ARM, encoding="utf-8"))
t = json.load(open(RTL, encoding="utf-8"))
p = json.load(open(RP1, encoding="utf-8"))
nx = json.load(open(NEXT, encoding="utf-8"))
r = json.load(open(RREF, encoding="utf-8"))
assert t["r_arm"] == "RTL" and p["r_arm"] == "RP01" and d["r_arm"] == ("RP01" if DRY else "RAPR") and d["mode"] == "production"
OUTF = Path(sys.argv[2]) if DRY else HERE / "RESULTS.md"
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")


def f(x, n=4):
    return f"{x:+.{n}f}"


def row_table(key, title):
    rows = list(d["seeds"]["s0"][key].keys())
    out = [f"**{title}** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)", "",
           "| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |", "|---|---|---|---|---|---|---|"]
    order = ["PRIMARY", *FAMS] + sorted({FAM[k] for k in rows} - {"PRIMARY", *FAMS})
    assert {FAM[k] for k in rows} <= set(order)
    for fam in order:
        for k in rows:
            if FAM[k] != fam:
                continue
            c = []
            for s in ("s0", "s1"):
                v = d["seeds"][s][key][k]
                c += [f"{f(v['delta'])} [{f(v['lo'])}, {f(v['hi'])}]", "yes" if v["separated"] else "no"]
            out.append(f"| {fam} | {k} | {'higher' if k in GD.HIGHER_BETTER else 'lower'} | " + " | ".join(c) + " |")
    assert sum(1 for x in out if x.startswith("| ") and not x.startswith("| family")) == len(rows), "a row was dropped"
    return out


L = []
FFOK = ("ROBUST", "PARTIAL")
J4 = (("R", r), ("RTL", t), ("RP01", p), ("RAPR", d))
SS = ("s0", "s1")
na, q, sec = nx["next_arm"], nx["attribution_q1"], nx["rapr_secondary"]
prof = na["facts"].get("RAPRC_d2_profile") or {}


def counted(J):
    return J["composite"]["status"] == "DOMINANT" and J["frame_freeze"]["reading"] in FFOK


refused = d.get("refused") is not None or bool(d.get("controls_failed"))
dom_both = (not refused) and d["composite"]["status"] == "DOMINANT"
sel = na["selected_row"]
if not refused and (counted(d) or sel == "row1"):
    row = ("card row 1 / addendum 1 row A", "evidence videos + the PI scale signal; a third seed pair of RAPR on fresh inits")
elif refused:
    row = ("addendum 1 row B", "no new box-2 arm: report the refused control, fix the instrument, re-read")
elif sel == "row3" and q["answer"] == "neither":
    row = ("addendum 2 row E", "RAPR-P01: RAPR's frozen argv with `--recover-p 0.1` (2 runs)")
elif sel in ("row2", "row3", "row4"):
    row = (f"card section 4 {sel}", na["selected"] + (f" Setting: {na['row3_setting']}." if sel == "row3" else ""))
elif sec.get("exceeds_at_exactly_one") and prof.get("fails_D2_on_M1_alone"):
    row = ("addendum 2 row F", "card row 2's arm: RAPR at `--recover-p 0.5` (2 runs)")
elif dom_both:
    row = ("addendum 2 row H", "no videos and no scale signal (DOMINANT, not counted: frame-freeze " + d["frame_freeze"]["reading"]
           + "); card row 4's level-from-image auxiliary, a PI-decision item")
elif not d.get("composite") or "no usable read" in (na.get("why_uncovered") or ""):
    row = ("addendum 2 row G (no composite)", "a refused read: addendum 1 row B, fix the instrument, re-read")
else:
    row = ("addendum 2 row G", "card row 4's level-from-image auxiliary, a PI-decision item (RAPRC fails D1 or D3)")

L.append("# RECOVER-NEXT, arm RAPR (`--recover-dev-law apriori`): read result and the committed next arm, 2026-11-04")
L.append("")
L.append(f"Card `PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `{d['cards']['recover_next'][:8]}…`) section 3, the RECOVER-FT section 6 / 7 "
         f"rules with R := RAPR; section 4 (the combiner's `next_arm`), addendum 1 (`PREREG_RECOVER_NEXT_ADDENDUM_RTLRP01.md`, frozen 2026-11-04 01:02) "
         f"and addendum 2 (`PREREG_RECOVER_NEXT_ADDENDUM2_RAPR.md`, frozen 2026-11-04 14:35:47, before any RAPR checkpoint). Tier **{d['tier']}** "
         f"(self-action open loop on logged cameras; not closed loop). Estimator: {d['estimator']}. Reader sha256 `{d['reader_sha256'][:12]}`, "
         f"mode `{d['mode']}`, run clock {d['run_clock']}. Combiner sha256 `{nx['combiner_sha256'][:12]}`, run clock {nx['run_clock']}. "
         f"Evidence class MEASURED (ours): `RECOVER_READ_RAPR.json`, `RECOVER_NEXT_after_RAPR.json` in this folder.")
L.append("")
L.append("## Verdict")
L.append("")
if refused:
    L.append(f"* **READ REFUSED**: refused = {d.get('refused')!r}; failed controls {d.get('controls_failed')}. A refusal is a result, not a label.")
else:
    c = d["composite"]
    L.append(f"* **Label: {d['label']}** (H-RECOVER-NEXT-1, arm RAPR). Controls: {len(d['controls'])} run, {len(d['controls_failed'])} failed; "
             f"KA1 bit-identity rerun of RAPR_s0: PASS={d['controls']['KA1_bit_identity_rerun_of_R_s0']['PASS']}. Guards PASS (both seeds): {d['guards']['PASS']}.")
    L.append(f"* **Composite RAPRC = RAPR + `--smooth sg72 --feasible kbox` vs P10: {c['status']}** (counted: {counted(d)}); "
             f"failing terms s0: {c['by_seed']['s0']['failing_terms']}, s1: {c['by_seed']['s1']['failing_terms']}. "
             f"D1 = {c['by_seed']['s0']['D1']} / {c['by_seed']['s1']['D1']}, D3 (four families, STRATEGIC included) = "
             f"{c['by_seed']['s0']['D3']} / {c['by_seed']['s1']['D3']}; STRATEGIC = {c['by_seed']['s0']['D3_by_family']['STRATEGIC']} / "
             f"{c['by_seed']['s1']['D3_by_family']['STRATEGIC']}.")
    ff = d["frame_freeze"]
    L.append(f"* Frame-freeze reading **{ff['reading']}** (share closed under hold_now s0 {ff['share_closed']['s0']:.4f}, "
             f"s1 {ff['share_closed']['s1']:.4f}; robust bar {ff['robust_bar']}).")
    L.append(f"* **RAPR secondary** ({sec['rule']}): exceeds per seed {sec.get('exceeds_per_seed')}; at both {sec.get('exceeds_at_both')}, "
             f"at neither {sec.get('exceeds_at_neither')}.")
L.append(f"* **Committed next arm: {row[0]}: {row[1]}.** Combiner (card section 4): {na['status']}; matching rows {na['matching_rows']}"
         + (f"; {na['why_uncovered']}" if na.get("why_uncovered") else "") + f". Q1 (from the RTL and RP01 reads): {q['answer']}.")
L.append("")
if not refused:
    L.append("## RAPR against R, RTL and RP01 (each changes one setting of R)")
    L.append("")
    L.append("| quantity | bar | " + " | ".join(f"{n} {s}" for n, _ in J4 for s in SS) + " |")
    L.append("|---|---|" + "---|" * 8)
    L.append(f"| M1 crawl share closed vs C0 | >= {d['primary']['share_bar']} | " + " | ".join(
        f"{J['primary']['share_closed'][s]:.4f}" for _, J in J4 for s in SS) + " |")
    L.append(f"| ADE6 X − C0 (m) | <= +{d['guards']['ade_guard_m']} | " + " | ".join(
        f(J["guards"]["per_seed"][s]["ade6_delta"]) for _, J in J4 for s in SS) + " |")
    L.append("| non-crawl rows separated-worse vs C0 | none | " + " | ".join(
        ", ".join(J["guards"]["per_seed"][s]["non_crawl_separated_worse"]) or "none" for _, J in J4 for s in SS) + " |")
    m = lambda J, s, k: J["seeds"][s][k]["M1 crawl share"]   # noqa: E731
    for k, lab in (("RC_minus_P10", "composite M1 crawl share XC − P10"), ("RC_minus_R", "kbox's own M1 effect XC − X")):
        L.append(f"| {lab} | not separated-worse | " + " | ".join(
            f"{f(m(J, s, k)['delta'])} [{f(m(J, s, k)['lo'])}, {f(m(J, s, k)['hi'])}]" for _, J in J4 for s in SS) + " |")
    L.append("| composite ADE6 XC − P10 (m) | <= -0.2415, separated | " + " | ".join(
        f"{f(J['seeds'][s]['RC_minus_P10']['ADE6']['delta'])}" for _, J in J4 for s in SS) + " |")
    L.append("| floor: abs(X − C0) M1 vs abs(C0rep − C0) | exceeds | " + " | ".join(
        f"{J['floor']['per_seed'][s]['abs_R_minus_C0']:.4f} vs {J['floor']['per_seed'][s]['abs_C0rep_minus_C0']:.4f}"
        for _, J in J4 for s in SS) + " |")
    L.append("| frame-freeze share closed | robust bar | " + " | ".join(
        f"{J['frame_freeze']['share_closed'][s]:.4f}" for _, J in J4 for s in SS) + " |")
    L.append("")
    L += row_table("RC_minus_P10", "Composite RAPRC − P10, all 19 rows (Read A; D2 = no row separated-worse)")
    L.append("")
    L += row_table("R_minus_C0", "RAPR − C0, all 19 rows (the label's guards read the non-crawl rows)")
    L.append("")
    L.append("## What this establishes, and what it cannot")
    L.append("")
    L.append(f"* The a-priori law (dv ~ U[−3, 0] m/s, heavier on the slow side) against the bank law at the same dose p 0.3: M1 crawl share closed "
             f"RAPR {d['primary']['share_closed']['s0']:.4f} / {d['primary']['share_closed']['s1']:.4f}, R "
             f"{r['primary']['share_closed']['s0']:.4f} / {r['primary']['share_closed']['s1']:.4f} (bar {d['primary']['share_bar']}).")
    L.append("* One training run per seed (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only; the "
             "training-run question is the C0rep floor row above. T1-kin is not T2: nothing here is a driving claim beyond T1-kin.")
    L.append("")
L.append("## Next (Rule Zero)")
L.append("")
L.append(f"* {row[0]} decides: {row[1]}." + ("" if row[0].startswith("card row 1") else " Any new box-2 arm needs the PI's go (box 2 runs one job at a time, SAFE MODE)."))
open(OUTF, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("wrote", OUTF, len(L), "lines; next", row[0])
