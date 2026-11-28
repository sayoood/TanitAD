"""Write RESULTS.md for the RECOVER-NEXT RAPR-P01 arm read (the a-priori deviation law at `--recover-p 0.1`) and its committed next arm,
from the read JSONs (every number is read from a file, none typed).

Inputs: read/raw/RECOVER_READ_RAPRP01.json (this arm), read/raw/RECOVER_READ_RP01.json (the bank law at the same dose),
read/raw/RECOVER_READ_RAPR.json (the same law at p 0.3), recover_read/raw/RECOVER_READ.json (R, the reference), the pinned
smooth_t1_read.py for the row -> family map. The next arm: addendum 3 rows M, I, N, J, K in that order, the first that matches decides
(PREREG_RECOVER_NEXT_ADDENDUM3_RAPRP01.md, sha256 9bea3fa1..., frozen 2026-11-04 16:29:29 before any RAPR-P01 run). The combiner has no
RAPRP01 row (its ARMS tuple stops at RAPR), so the rows are applied here, from the reader's own booleans:
`composite.DOMINANT` (Read A at both seeds), `composite.counts` (the reader's DOMINANT, COUNTED: frame-freeze ROBUST or PARTIAL and its
controls passed), `guards.per_seed` (ADE6 point <= +0.0604 m and no separated-worse non-crawl row). Output: RESULTS.md next to this script.
Dry run (no RAPR-P01 read yet): `gen_raprp01_results.py --dry-as <RAPR|RP01|RTL> <out.md>` renders another arm's read in RAPR-P01's place.
Mutation test: `gen_raprp01_results.py --json <arm.json> <out.md>` reads that file as the arm (used by test_gen_raprp01.py).
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NX = HERE.parent
RAW = NX / "read" / "raw"
RREF = NX.parent / "recover_read" / "raw" / "RECOVER_READ.json"
SRP = NX.parent / "smooth_t1" / "code" / "smooth_t1_read.py"
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
FFOK = ("ROBUST", "PARTIAL")
SS = ("s0", "s1")
ROWS = {
    "M": ("addendum 3 row M (addendum 1 row B)", "no new box-2 arm: report the refusal, fix the instrument, re-read"),
    "I": ("addendum 3 row I (card row 1 / addendum 1 row A)", "evidence videos and the PI scale signal; a third seed pair of RAPR-P01 on fresh inits"),
    "N": ("addendum 3 row N", "no videos and no scale signal; the frame-freeze reading is the blocker; next as row K: a PI-decision item between "
          "(a) card row 4's level-from-image auxiliary and (b) a per-tick deviation recipe"),
    "J": ("addendum 3 row J", "RAPR-P02: RAPR's frozen argv with `--recover-p 0.2` (2 runs)"),
    "K": ("addendum 3 row K", "the one-offset-per-window recovery recipe is closed at the laws and doses measured (bank law at p 0.3 and 0.1, "
          "a-priori law at p 0.3 and 0.1); next arm: a PI-decision item between (a) card row 4's level-from-image auxiliary (architecture "
          "change) and (b) a per-tick deviation recipe (a new trainer lever with its own launch-gate entry); neither starts without the PI's "
          "decision and box-2 go"),
}


def guard_ok(d, s):
    g = d["guards"]["per_seed"][s]
    return bool(g["ade6_within_guard"] and not g["non_crawl_separated_worse"])


def decide(d):
    """Addendum 3 rows in order; returns (letter, facts)."""
    c = d.get("composite") or {}
    refused = d.get("refused") is not None or bool(d.get("controls_failed")) or c.get("status") is None
    facts = {"refused": refused}
    if refused:
        return "M", facts
    dom = bool(c["DOMINANT"])
    counted = bool(c["counts"])
    assert counted == (dom and c["status"] == "DOMINANT, COUNTED"), "reader counts flag disagrees with its status"
    if counted:
        assert d["frame_freeze"]["reading"] in FFOK, "counted with a frame-freeze reading outside ROBUST / PARTIAL"
    gp = {s: guard_ok(d, s) for s in SS}
    assert all(gp.values()) == bool(d["guards"]["PASS"]), "per-seed guards disagree with the reader's guards.PASS"
    facts.update(dominant_both=dom, counted=counted, guards_pass=gp, status=c["status"])
    if dom and counted:
        return "I", facts
    if dom:
        return "N", facts
    if all(gp.values()):
        return "J", facts
    return "K", facts


def f(x, n=4):
    return f"{x:+.{n}f}"


def main(argv):
    dry = mode = None
    if len(argv) > 1 and argv[1] in ("--dry-as", "--json"):
        mode, dry, outf = argv[1], argv[2], Path(argv[3])
    else:
        outf = HERE / "RESULTS.md"
    arm_path = Path(dry) if mode == "--json" else RAW / (f"RECOVER_READ_{dry}.json" if mode == "--dry-as" else "RECOVER_READ_RAPRP01.json")
    d = json.load(open(arm_path, encoding="utf-8"))
    p = json.load(open(RAW / "RECOVER_READ_RP01.json", encoding="utf-8"))
    a = json.load(open(RAW / "RECOVER_READ_RAPR.json", encoding="utf-8"))
    r = json.load(open(RREF, encoding="utf-8"))
    assert p["r_arm"] == "RP01" and a["r_arm"] == "RAPR" and d["mode"] == "production"
    assert d["r_arm"] == ("RAPRP01" if mode is None else (dry if mode == "--dry-as" else d["r_arm"]))
    spec = importlib.util.spec_from_file_location("smooth_t1_read", SRP)
    SR = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SRP.parent))
    spec.loader.exec_module(SR)
    FAM = SR.FAMILY
    gs = importlib.util.spec_from_file_location("gapdecomp", NX.parent / "gapdecomp" / "gapdecomp.py")
    GD = importlib.util.module_from_spec(gs)
    gs.loader.exec_module(GD)

    def row_table(key, title):
        rows = list(d["seeds"]["s0"][key].keys())
        out = [f"**{title}** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)", "",
               "| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |", "|---|---|---|---|---|---|---|"]
        order = ["PRIMARY", *FAMS] + sorted({FAM[k] for k in rows} - {"PRIMARY", *FAMS})
        for fam in order:
            for k in rows:
                if FAM[k] != fam:
                    continue
                c = []
                for s in SS:
                    v = d["seeds"][s][key][k]
                    c += [f"{f(v['delta'])} [{f(v['lo'])}, {f(v['hi'])}]", "yes" if v["separated"] else "no"]
                out.append(f"| {fam} | {k} | {'higher' if k in GD.HIGHER_BETTER else 'lower'} | " + " | ".join(c) + " |")
        assert sum(1 for x in out if x.startswith("| ") and not x.startswith("| family")) == len(rows), "a row was dropped"
        assert {FAM[k] for k in rows} >= set(FAMS), "a metric family is missing"
        return out

    let, facts = decide(d)
    row = ROWS[let]
    L = ["# RECOVER-NEXT, arm RAPR-P01 (`--recover-dev-law apriori --recover-p 0.1`): read result and the committed next arm, 2026-11-04", ""]
    L.append(f"Card `PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `{d['cards']['recover_next'][:8]}…`) section 3, the RECOVER-FT section 6 / 7 "
             f"rules with R := RAPR-P01; addendum 3 (`PREREG_RECOVER_NEXT_ADDENDUM3_RAPRP01.md`, sha256 `9bea3fa1…`, frozen 2026-11-04 16:29:29, "
             f"before any RAPR-P01 run) rows M, I, N, J, K decide the next arm. Tier **{d['tier']}** (self-action open loop on logged cameras; "
             f"not closed loop). Estimator: {d['estimator']}. Reader sha256 `{d['reader_sha256'][:12]}`, mode `{d['mode']}`, run clock "
             f"{d['run_clock']}. Evidence class MEASURED (ours): `RECOVER_READ_RAPRP01.json` in this folder.")
    if mode:
        L += ["", f"**DRY RUN** ({mode} {dry}): not a RAPR-P01 result."]
    L += ["", "## Verdict", ""]
    if facts["refused"]:
        L.append(f"* **READ REFUSED**: refused = {d.get('refused')!r}; failed controls {d.get('controls_failed')}; composite status "
                 f"{(d.get('composite') or {}).get('status')!r}. A refusal is a result, not a label.")
    else:
        c = d["composite"]
        L.append(f"* **Label: {d['label']}** (H-RECOVER-NEXT-1, arm RAPR-P01). Controls: {len(d['controls'])} run, {len(d['controls_failed'])} "
                 f"failed; KA1 bit-identity rerun: PASS={d['controls']['KA1_bit_identity_rerun_of_R_s0']['PASS']}. Guards per seed: "
                 f"s0 {facts['guards_pass']['s0']}, s1 {facts['guards_pass']['s1']} (reader guards.PASS {d['guards']['PASS']}).")
        L.append(f"* **Composite RAPRP01C = RAPR-P01 + `--smooth sg72 --feasible kbox` vs P10: {c['status']}** (DOMINANT at both seeds: "
                 f"{c['DOMINANT']}; counted: {c['counts']}); failing terms s0: {c['by_seed']['s0']['failing_terms']}, s1: "
                 f"{c['by_seed']['s1']['failing_terms']}. D1 = {c['by_seed']['s0']['D1']} / {c['by_seed']['s1']['D1']}, D2 = "
                 f"{c['by_seed']['s0']['D2']} / {c['by_seed']['s1']['D2']}, D3 (four families, STRATEGIC included) = "
                 f"{c['by_seed']['s0']['D3']} / {c['by_seed']['s1']['D3']}; STRATEGIC = {c['by_seed']['s0']['D3_by_family']['STRATEGIC']} / "
                 f"{c['by_seed']['s1']['D3_by_family']['STRATEGIC']}.")
        ff = d["frame_freeze"]
        L.append(f"* Frame-freeze reading **{ff['reading']}** (share closed under hold_now s0 {ff['share_closed']['s0']:.4f}, "
                 f"s1 {ff['share_closed']['s1']:.4f}; robust bar {ff['robust_bar']}).")
    L.append(f"* **Committed next arm: {row[0]}: {row[1]}.**")
    L.append("")
    if not facts["refused"]:
        J4 = (("R", r), ("RP01", p), ("RAPR", a), ("RAPR-P01", d))
        L += ["## RAPR-P01 against RP01 (bank law, same dose) and RAPR (same law, p 0.3); R for reference (reported, not deciding)", ""]
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
        L += row_table("RC_minus_P10", "Composite RAPRP01C − P10, all 19 rows (Read A; D2 = no row separated-worse)")
        L.append("")
        L += row_table("R_minus_C0", "RAPR-P01 − C0, all 19 rows (the label's guards read the non-crawl rows)")
        L += ["", "## What this establishes, and what it cannot", ""]
        L.append(f"* Dose of the a-priori law: M1 crawl share closed RAPR-P01 {d['primary']['share_closed']['s0']:.4f} / "
                 f"{d['primary']['share_closed']['s1']:.4f} at p 0.1 against RAPR {a['primary']['share_closed']['s0']:.4f} / "
                 f"{a['primary']['share_closed']['s1']:.4f} at p 0.3; the bank law at p 0.1 (RP01) {p['primary']['share_closed']['s0']:.4f} / "
                 f"{p['primary']['share_closed']['s1']:.4f} (bar {d['primary']['share_bar']}).")
        L.append("* One training run per seed (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only; the "
                 "training-run question is the C0rep floor row above. T1-kin is not T2: nothing here is a driving claim beyond T1-kin.")
        L.append("* Instrument note: the RAPR results writer compared the composite status to the string `DOMINANT`, which the reader never "
                 "emits (it writes `DOMINANT, COUNTED` and three other forms). RAPR's composite was NOT DOMINANT at both seeds, so its row "
                 "was not affected; this writer reads the reader's `DOMINANT` and `counts` booleans, and every row M, I, N, J, K is pinned by "
                 "a mutation test (`test_gen_raprp01.py`).")
        L.append("")
    L += ["## Next (Rule Zero)", ""]
    L.append(f"* {row[0]} decides: {row[1]}." + ("" if let == "I" else " Any new box-2 arm needs the PI's go (box 2 runs one job at a time, SAFE MODE)."))
    open(outf, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("wrote", outf, len(L), "lines; row", let, facts)
    return let


if __name__ == "__main__":
    main(sys.argv)
