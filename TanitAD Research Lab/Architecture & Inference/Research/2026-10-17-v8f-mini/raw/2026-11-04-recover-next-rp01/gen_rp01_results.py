"""Write RESULTS.md for the RECOVER-NEXT RP01 arm read (+ Q1 and the addendum row) from the read JSONs (every number is read from a file, none typed).

Inputs: read/raw/RECOVER_READ_RP01.json (this arm), read/raw/RECOVER_READ_RTL.json, read/raw/RECOVER_NEXT.json (the combiner),
recover_read/raw/RECOVER_READ.json (R, the reference), the addendum PREREG_RECOVER_NEXT_ADDENDUM_RTLRP01.md (rows A-D), the pinned
smooth_t1_read.py for the row -> family map. Output: RESULTS.md next to this script.
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NX = HERE.parent
ARM = NX / "read" / "raw" / "RECOVER_READ_RP01.json"
RTL = NX / "read" / "raw" / "RECOVER_READ_RTL.json"
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
nx = json.load(open(NEXT, encoding="utf-8"))
r = json.load(open(RREF, encoding="utf-8"))
assert t["r_arm"] == "RTL" and d["r_arm"] == "RP01" and d["mode"] == "production" and d["controls_failed"] == [] and d["refused"] is None
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
J3 = (("R", r), ("RTL", t), ("RP01", d))
SS = ("s0", "s1")


def counted(J):
    return J["composite"]["status"] == "DOMINANT" and J["frame_freeze"]["reading"] in FFOK


q = nx["attribution_q1"]
if counted(t) or counted(d):
    row = ("A", "evidence videos + the PI scale signal; a third seed pair of that arm on fresh inits")
elif any(J["refused"] is not None or J["controls_failed"] for J in (t, d)):
    row = ("B", "no new box-2 arm: report the refused control, fix the instrument, re-read")
elif q["answer"] in ("blend", "both"):
    row = ("C", "RAPR-TL (RAPR's frozen argv with `--recover-target logged`, without `--recover-rejoin-steps 10`)")
elif q["answer"] in ("dose", "neither"):
    row = ("D", "RAPR as frozen (RAPR_0 / RAPR_1, argv already in `argv/`, unchanged)")
else:
    raise AssertionError(("no addendum row matches", q["answer"]))

L.append("# RECOVER-NEXT, arm RP01 (`--recover-p 0.1`): read result, Q1 and the committed next arm, 2026-11-04")
L.append("")
L.append(f"Card `PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `{d['cards']['recover_next'][:8]}…`) section 3, the RECOVER-FT section 6 / 7 "
         f"rules with R := RP01; addendum `PREREG_RECOVER_NEXT_ADDENDUM_RTLRP01.md` (frozen 2026-11-04 01:02, before any RTL or RP01 "
         f"rollout). Tier **{d['tier']}** (self-action open loop on logged cameras; not closed loop). Estimator: {d['estimator']}. "
         f"Reader sha256 `{d['reader_sha256'][:12]}`, mode `{d['mode']}`, run clock {d['run_clock']}. Combiner sha256 "
         f"`{nx['combiner_sha256'][:12]}`, run clock {nx['run_clock']}. Evidence class MEASURED (ours): `RECOVER_READ_RP01.json`, "
         f"`RECOVER_NEXT_after_RP01.json` in this folder; RTL's read is `../2026-11-04-recover-next-rtl/RECOVER_READ_RTL.json`.")
L.append("")
L.append("## Verdict")
L.append("")
c = d["composite"]
L.append(f"* **Label: {d['label']}** (H-RECOVER-NEXT-1, arm RP01). Controls: {len(d['controls'])} run, "
         f"{len(d['controls_failed'])} failed; KA1 bit-identity rerun of RP01_s0: "
         f"PASS={d['controls']['KA1_bit_identity_rerun_of_R_s0']['PASS']}. Guards PASS (both seeds): {d['guards']['PASS']}.")
L.append(f"* **Composite RP01C = RP01 + `--smooth sg72 --feasible kbox` vs P10: {c['status']}**; "
         f"failing terms s0: {c['by_seed']['s0']['failing_terms']}, s1: {c['by_seed']['s1']['failing_terms']}. "
         f"D1 = {c['by_seed']['s0']['D1']} / {c['by_seed']['s1']['D1']}, D3 (four families, STRATEGIC included) = "
         f"{c['by_seed']['s0']['D3']} / {c['by_seed']['s1']['D3']}.")
ff = d["frame_freeze"]
L.append(f"* Frame-freeze reading **{ff['reading']}** (share closed under hold_now s0 {ff['share_closed']['s0']:.4f}, "
         f"s1 {ff['share_closed']['s1']:.4f}; robust bar {ff['robust_bar']}).")
L.append(f"* Strategic family: reported, D3_by_family STRATEGIC = {c['by_seed']['s0']['D3_by_family']['STRATEGIC']} / "
         f"{c['by_seed']['s1']['D3_by_family']['STRATEGIC']}; for v8F STRATEGIC is reported and counted in D3.")
L.append(f"* **Q1 (did the rejoin blend or the dose harm?): {q['answer']}** (rule: {q['rule']}; blend harmed "
         f"{q['blend_harmed']}, dose harmed {q['dose_harmed']}).")
L.append(f"* **Committed next arm: addendum row {row[0]}: {row[1]}.** Composite DOMINANT and counted: RTLC {counted(t)}, "
         f"RP01C {counted(d)}. The combiner's own card-section-4 table reads: {nx['next_arm']['status']}"
         + (f" ({nx['next_arm']['why_uncovered']})" if nx["next_arm"].get("why_uncovered") else "") + "; the addendum rows decide.")
L.append("")
L.append("## RP01 against R and RTL (each changes one flag of R)")
L.append("")
L.append("| quantity | bar | " + " | ".join(f"{n} {s}" for n, _ in J3 for s in SS) + " |")
L.append("|---|---|" + "---|" * 6)
L.append(f"| M1 crawl share closed vs C0 | >= {d['primary']['share_bar']} | " + " | ".join(
    f"{J['primary']['share_closed'][s]:.4f}" for _, J in J3 for s in SS) + " |")
L.append(f"| ADE6 X − C0 (m) | <= +{d['guards']['ade_guard_m']} | " + " | ".join(
    f(J["guards"]["per_seed"][s]["ade6_delta"]) for _, J in J3 for s in SS) + " |")
L.append("| non-crawl rows separated-worse vs C0 | none | " + " | ".join(
    ", ".join(J["guards"]["per_seed"][s]["non_crawl_separated_worse"]) or "none" for _, J in J3 for s in SS) + " |")
m = lambda J, s, k: J["seeds"][s][k]["M1 crawl share"]   # noqa: E731
for k, lab in (("RC_minus_P10", "composite M1 crawl share XC − P10"), ("RC_minus_R", "kbox's own M1 effect XC − X")):
    L.append(f"| {lab} | not separated-worse | " + " | ".join(
        f"{f(m(J, s, k)['delta'])} [{f(m(J, s, k)['lo'])}, {f(m(J, s, k)['hi'])}]" for _, J in J3 for s in SS) + " |")
L.append("| floor: abs(X − C0) M1 vs abs(C0rep − C0) | exceeds | " + " | ".join(
    f"{J['floor']['per_seed'][s]['abs_R_minus_C0']:.4f} vs {J['floor']['per_seed'][s]['abs_C0rep_minus_C0']:.4f}"
    for _, J in J3 for s in SS) + " |")
L.append("| frame-freeze share closed | robust bar | " + " | ".join(
    f"{J['frame_freeze']['share_closed'][s]:.4f}" for _, J in J3 for s in SS) + " |")
L.append("")
L += row_table("RC_minus_P10", "Composite RP01C − P10, all 19 rows (Read A; D2 = no row separated-worse)")
L.append("")
L += row_table("R_minus_C0", "RP01 − C0, all 19 rows (the label's guards read the non-crawl rows)")
L.append("")
L.append("## What this establishes, and what it cannot")
L.append("")
gp = d["guards"]["per_seed"]
L.append(f"* Lowering the dose to p 0.1 {'removes' if d['guards']['PASS'] else 'does not remove'} RECOVER-FT's harm by the card's "
         f"guards: ADE6 RP01 − C0 {f(gp['s0']['ade6_delta'])} / {f(gp['s1']['ade6_delta'])} m against the "
         f"+{d['guards']['ade_guard_m']} m guard. With RTL's guards failing at both seeds, Q1 = `{q['answer']}`.")
L.append(f"* M1 crawl share closed vs C0: RP01 {d['primary']['share_closed']['s0']:.4f} / {d['primary']['share_closed']['s1']:.4f}, "
         f"RTL {t['primary']['share_closed']['s0']:.4f} / {t['primary']['share_closed']['s1']:.4f}, R "
         f"{r['primary']['share_closed']['s0']:.4f} / {r['primary']['share_closed']['s1']:.4f} (bar {d['primary']['share_bar']}).")
L.append("* One training run per seed (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only; the "
         "training-run question is the C0rep floor row above.")
L.append("")
L.append("## Next (Rule Zero; addendum rows, frozen before any RTL or RP01 rollout)")
L.append("")
L.append(f"* Row {row[0]} decides: {row[1]}." + (" It needs the PI's box-2 go (box 2 runs one job at a time, SAFE MODE)."
                                               if row[0] in ("C", "D") else ""))
open(HERE / "RESULTS.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("wrote", HERE / "RESULTS.md", len(L), "lines; row", row[0], "Q1", q["answer"])
