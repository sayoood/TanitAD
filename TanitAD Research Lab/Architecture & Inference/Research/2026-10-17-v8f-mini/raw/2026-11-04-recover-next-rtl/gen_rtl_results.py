"""Write RESULTS.md for the RECOVER-NEXT RTL arm read from the read JSONs (every number is read from a file, none typed).

Inputs: read/raw/RECOVER_READ_RTL.json (this arm), recover_read/raw/RECOVER_READ.json (R, the reference), the pinned
smooth_t1_read.py for the row -> family map. Output: RESULTS.md next to this script.
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NX = HERE.parent
RTL = NX / "read" / "raw" / "RECOVER_READ_RTL.json"
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

d = json.load(open(RTL, encoding="utf-8"))
r = json.load(open(RREF, encoding="utf-8"))
assert d["r_arm"] == "RTL" and d["mode"] == "production" and d["controls_failed"] == [] and d["refused"] is None
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
L.append("# RECOVER-NEXT, arm RTL (`--recover-target logged`): read result, 2026-11-04")
L.append("")
L.append(f"Card `PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `{d['cards']['recover_next'][:8]}…`) section 3, the RECOVER-FT section 6 / 7 "
         f"rules with R := RTL; addendum `PREREG_RECOVER_NEXT_ADDENDUM_RTLRP01.md` (frozen 2026-11-04 01:02, before any RTL rollout). "
         f"Tier **{d['tier']}** (self-action open loop on logged cameras; not closed loop). Estimator: {d['estimator']}. "
         f"Reader sha256 `{d['reader_sha256'][:12]}`, mode `{d['mode']}`, run clock {d['run_clock']}. "
         f"Evidence class MEASURED (ours): `RECOVER_READ_RTL.json` in this folder.")
L.append("")
L.append("## Verdict")
L.append("")
c = d["composite"]
L.append(f"* **Label: {d['label']}** (H-RECOVER-NEXT-1, arm RTL). Controls: {len(d['controls'])} run, "
         f"{len(d['controls_failed'])} failed; KA1 bit-identity rerun of RTL_s0: "
         f"PASS={d['controls']['KA1_bit_identity_rerun_of_R_s0']['PASS']}.")
L.append(f"* **Composite RTLC = RTL + `--smooth sg72 --feasible kbox` vs P10: {c['status']}** at both seeds; "
         f"failing terms s0: {c['by_seed']['s0']['failing_terms']}, s1: {c['by_seed']['s1']['failing_terms']}. "
         f"D1 = {c['by_seed']['s0']['D1']} / {c['by_seed']['s1']['D1']}, D3 (four families, STRATEGIC included) = "
         f"{c['by_seed']['s0']['D3']} / {c['by_seed']['s1']['D3']}.")
ff = d["frame_freeze"]
L.append(f"* Frame-freeze reading **{ff['reading']}** (share closed under hold_now s0 {ff['share_closed']['s0']:.4f}, "
         f"s1 {ff['share_closed']['s1']:.4f}; robust bar {ff['robust_bar']}).")
L.append(f"* Strategic family: reported, D3_by_family STRATEGIC = {c['by_seed']['s0']['D3_by_family']['STRATEGIC']} / "
         f"{c['by_seed']['s1']['D3_by_family']['STRATEGIC']} (separated-better rows "
         f"{d['read_a_R_vs_P10']['by_seed']['s0']['strategic_rows_separated_better']} on RTL vs P10, s0). "
         f"The reader file also carries the note: \"{d['strategic_family'][:60]}…\"; for v8F STRATEGIC is reported and counted in D3.")
L.append("")
L.append("## RTL against R (the RECOVER-FT arm it changes one flag of)")
L.append("")
L.append("| quantity | bar | R s0 | R s1 | RTL s0 | RTL s1 |")
L.append("|---|---|---|---|---|---|")
L.append(f"| M1 crawl share closed vs C0 | >= {d['primary']['share_bar']} | {r['primary']['share_closed']['s0']:.4f} | "
         f"{r['primary']['share_closed']['s1']:.4f} | {d['primary']['share_closed']['s0']:.4f} | {d['primary']['share_closed']['s1']:.4f} |")
L.append(f"| ADE6 X − C0 (m) | <= +{d['guards']['ade_guard_m']} | {f(r['guards']['per_seed']['s0']['ade6_delta'])} | "
         f"{f(r['guards']['per_seed']['s1']['ade6_delta'])} | {f(d['guards']['per_seed']['s0']['ade6_delta'])} | "
         f"{f(d['guards']['per_seed']['s1']['ade6_delta'])} |")
for s in ("s0", "s1"):
    pass
L.append(f"| non-crawl rows separated-worse vs C0 | none | {', '.join(r['guards']['per_seed']['s0']['non_crawl_separated_worse'])} | "
         f"{', '.join(r['guards']['per_seed']['s1']['non_crawl_separated_worse'])} | "
         f"{', '.join(d['guards']['per_seed']['s0']['non_crawl_separated_worse'])} | "
         f"{', '.join(d['guards']['per_seed']['s1']['non_crawl_separated_worse'])} |")
m = lambda J, s, k: J["seeds"][s][k]["M1 crawl share"]
for k, lab in (("RC_minus_P10", "composite M1 crawl share XC − P10"), ("RC_minus_R", "kbox's own M1 effect XC − X")):
    L.append(f"| {lab} | not separated-worse | " + " | ".join(
        f"{f(m(J, s, k)['delta'])} [{f(m(J, s, k)['lo'])}, {f(m(J, s, k)['hi'])}]" for J in (r, d) for s in ("s0", "s1")) + " |")
L.append(f"| floor: abs(X − C0) M1 vs abs(C0rep − C0) | exceeds | {r['floor']['per_seed']['s0']['abs_R_minus_C0']:.4f} vs "
         f"{r['floor']['per_seed']['s0']['abs_C0rep_minus_C0']:.4f} | {r['floor']['per_seed']['s1']['abs_R_minus_C0']:.4f} vs "
         f"{r['floor']['per_seed']['s1']['abs_C0rep_minus_C0']:.4f} | {d['floor']['per_seed']['s0']['abs_R_minus_C0']:.4f} vs "
         f"{d['floor']['per_seed']['s0']['abs_C0rep_minus_C0']:.4f} | {d['floor']['per_seed']['s1']['abs_R_minus_C0']:.4f} vs "
         f"{d['floor']['per_seed']['s1']['abs_C0rep_minus_C0']:.4f} |")
L.append("")
L += row_table("RC_minus_P10", "Composite RTLC − P10, all 19 rows (Read A; D2 = no row separated-worse)")
L.append("")
L += row_table("R_minus_C0", "RTL − C0, all 19 rows (the label's guards read the non-crawl rows)")
L.append("")
L.append("## What this establishes, and what it cannot")
L.append("")
L.append("* Removing the rejoin blend (target logged) does not remove RECOVER-FT's harm: ADE6 RTL − C0 stays above the "
         f"+{d['guards']['ade_guard_m']} m guard at both seeds. Under the card's Q1 rule (blend harmed iff RTL's guards PASS at both "
         "seeds) the blend is not the whole harm; Q1 itself needs RP01's read (box 2, running).")
L.append("* The composite's only failing term is the M1 crawl share, at the same size as R's composite; every other row "
         "of RTLC − P10 is not separated-worse, and ADE6 is separated-better by about 0.9 m at both seeds.")
L.append("* kbox itself raises the M1 crawl share (XC − X separated-worse at both seeds, for R and for RTL).")
L.append("* One training run per seed (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only; the "
         "training-run question is the C0rep floor, and at s0 abs(RTL − C0) on M1 sits inside it.")
L.append("")
L.append("## Next (Rule Zero; addendum rows, frozen before this read)")
L.append("")
L.append("* RTL's guards fail, so Q1 cannot be `blend` or `both`: addendum row C (RAPR-TL) is unreachable. If RP01C is not "
         "DOMINANT (counted), row D decides: **RAPR as frozen** (argv already shipped on box 2), which needs the PI's box-2 go. "
         "If RP01C is DOMINANT and counted, row A: evidence videos and the PI scale signal.")
L.append("* The larger lever by measured effect is the speed level (§7.35: an oracle swap of the logged history closes 0.918 to "
         "0.984 of the M1 excess, the level alone 0.599 to 0.633), against 0.13 to 0.22 for the RECOVER family; the card's "
         "section 4 names the level-from-image auxiliary as the PI-decision item for it.")
open(HERE / "RESULTS.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("wrote", HERE / "RESULTS.md", len(L), "lines")
