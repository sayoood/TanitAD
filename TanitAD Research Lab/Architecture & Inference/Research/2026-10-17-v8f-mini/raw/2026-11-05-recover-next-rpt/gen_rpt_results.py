"""Write RESULTS.md for the RECOVER-NEXT RPT arm read (R's bank law at p 0.3 with `--recover-dev-shape pertick`) and its committed next
arm, from the read JSONs and the pulled gate selfchecks (every number is read from a file, none typed).

Inputs: read/raw/RECOVER_READ_RPT.json (this arm), recover_read/raw/RECOVER_READ.json (R: same law, same dose, constant shape),
read/raw/RECOVER_READ_RAPR.json (the largest closure measured before RPT), rowk_decision/d2_closure_need.json (the decision item's
closure-need estimate, reported, not deciding), the pulled gate selfchecks v8f-recover/M2b_s{0,1}_recft_{RPT,R}/gate_selfcheck.json
(the per-tick table's counts and its NOW columns against R's table), the pinned smooth_t1_read.py for the row -> family map. The next
arm: addendum 4 rows M, I, N, J, K1, K2 in that order, the first that matches decides (PREREG_RECOVER_NEXT_ADDENDUM4_RPT.md, frozen
before any RPT run). Rows are applied here from the reader's own booleans: `composite.DOMINANT` (Read A at both seeds),
`composite.counts` (DOMINANT, COUNTED: frame-freeze ROBUST or PARTIAL and its controls passed), `guards.per_seed` (ADE6 point
<= +0.0604 m and no separated-worse non-crawl row), and for K1 / K2 `primary.share_closed` of RPT against R's, read from R's JSON.
Output: RESULTS.md next to this script.
Dry run (no RPT read yet): `gen_rpt_results.py --dry-as <RAPRP01|RAPR|RP01|RTL> <out.md>` renders another arm's read in RPT's place.
Mutation test: `gen_rpt_results.py --json <arm.json> <out.md> [--runs <dir>]` reads that file as the arm (used by test_gen_rpt.py).
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
NEED = NX / "rowk_decision" / "d2_closure_need.json"
RUNS = Path("E:/Projects/TanitAD-artifacts/v8f-recover")
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
FFOK = ("ROBUST", "PARTIAL")
SS = ("s0", "s1")
R_SHARE_ADDENDUM = {"s0": 0.2198, "s1": 0.1318}   # the values addendum 4 row K1 quotes; asserted against R's JSON, never used to decide
NOW_KEYS = ("mean_dv", "mean_da", "mean_dom", "mean_dy", "mean_dpsi")
ROWS = {
    "M": ("addendum 4 row M (addendum 1 row B)", "no new box-2 arm: report the refusal, fix the instrument, re-read"),
    "I": ("addendum 4 row I (card row 1 / addendum 1 row A)", "evidence videos and the PI scale signal; a third seed pair of RPT on fresh inits"),
    "N": ("addendum 4 row N", "no videos and no scale signal; the frame-freeze reading is the blocker; next as row K2"),
    "J": ("addendum 4 row J", "RPT-P05: RPT's argv with `--recover-p 0.5` (2 runs)"),
    "K1": ("addendum 4 row K1", "RPT-P01: RPT's argv with `--recover-p 0.1` (2 runs): the per-tick shape adds closure at R's dose"),
    "K2": ("addendum 4 row K2", "the per-tick shape adds no closure at the dose measured: next arm is option (a), the level-from-image "
           "auxiliary of the decision item, as a PI-decision item with its costs; it does not start without the PI's decision and box-2 go"),
}


def guard_ok(d, s):
    g = d["guards"]["per_seed"][s]
    return bool(g["ade6_within_guard"] and not g["non_crawl_separated_worse"])


def decide(d, r_share):
    """Addendum 4 rows in order; returns (letter, facts). r_share: R's primary.share_closed per seed (from R's JSON)."""
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
    more = {s: float(d["primary"]["share_closed"][s]) > float(r_share[s]) for s in SS}
    facts.update(dominant_both=dom, counted=counted, guards_pass=gp, status=c["status"], share_exceeds_R=more)
    if dom and counted:
        return "I", facts
    if dom:
        return "N", facts
    if all(gp.values()):
        return "J", facts
    return ("K1" if all(more.values()) else "K2"), facts


def pertick_counts(runs, strict):
    """Per seed: the RPT gate's per-tick table record (kept, dropped, ticks) and its NOW columns against R's gate table.
    strict (a production read): an absent or incomplete selfcheck raises; otherwise it is reported as absent."""
    out = {}
    for s in SS:
        try:
            t = json.load(open(runs / f"M2b_{s}_recft_RPT" / "gate_selfcheck.json", encoding="utf-8"))["levers"]["recover_table"]
            r = json.load(open(runs / f"M2b_{s}_recft_R" / "gate_selfcheck.json", encoding="utf-8"))["levers"]["recover_table"]
            ct, pt = t["counts"], t["per_tick"]
            row = {"n": t["n"], "kept": ct["kept"], "dropped_per_tick_row_unreal": ct["dropped_per_tick_row_unreal"], "ticks": pt["ticks"],
                   "R_n": r["n"], "now_equal_R": {k: t[k] == r[k] for k in NOW_KEYS}}
        except (OSError, KeyError, TypeError, ValueError) as e:
            if strict:
                raise AssertionError(f"per-tick table record for {s} unreadable: {type(e).__name__}: {e}") from e
            row = {"absent": f"{type(e).__name__}: {e}"}
        out[s] = row
    return out


def f(x, n=4):
    return f"{x:+.{n}f}"


def main(argv):
    dry = mode = None
    runs = RUNS
    args = list(argv)
    if "--runs" in args:
        i = args.index("--runs"); runs = Path(args[i + 1]); del args[i:i + 2]
    if len(args) > 1 and args[1] in ("--dry-as", "--json"):
        mode, dry, outf = args[1], args[2], Path(args[3])
    else:
        outf = HERE / "RESULTS.md"
    arm_path = Path(dry) if mode == "--json" else RAW / (f"RECOVER_READ_{dry}.json" if mode == "--dry-as" else "RECOVER_READ_RPT.json")
    d = json.load(open(arm_path, encoding="utf-8"))
    a = json.load(open(RAW / "RECOVER_READ_RAPR.json", encoding="utf-8"))
    r = json.load(open(RREF, encoding="utf-8"))
    need = json.load(open(NEED, encoding="utf-8"))
    assert a["r_arm"] == "RAPR" and d["mode"] == "production"
    assert d["r_arm"] == ("RPT" if mode is None else (dry if mode == "--dry-as" else d["r_arm"]))
    r_share = {s: float(r["primary"]["share_closed"][s]) for s in SS}
    assert all(round(r_share[s], 4) == R_SHARE_ADDENDUM[s] for s in SS), f"R's share closed {r_share} is not the addendum's {R_SHARE_ADDENDUM}"
    spec = importlib.util.spec_from_file_location("smooth_t1_read", SRP)
    SR = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SRP.parent))
    spec.loader.exec_module(SR)
    FAM = SR.FAMILY
    gs = importlib.util.spec_from_file_location("gapdecomp", NX.parent / "gapdecomp" / "gapdecomp.py")
    GD = importlib.util.module_from_spec(gs)
    gs.loader.exec_module(GD)

    def ordered(rows):
        order = ["PRIMARY", *FAMS] + sorted({FAM[k] for k in rows} - {"PRIMARY", *FAMS})
        return [(fam, k) for fam in order for k in rows if FAM[k] == fam]

    def cell(v):
        return f"{f(v['delta'])} [{f(v['lo'])}, {f(v['hi'])}]", "yes" if v["separated"] else "no"

    def row_table(key, title):
        rows = list(d["seeds"]["s0"][key].keys())
        out = [f"**{title}** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)", "",
               "| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |", "|---|---|---|---|---|---|---|"]
        for fam, k in ordered(rows):
            out.append(f"| {fam} | {k} | {'higher' if k in GD.HIGHER_BETTER else 'lower'} | "
                       + " | ".join(x for s in SS for x in cell(d["seeds"][s][key][k])) + " |")
        assert sum(1 for x in out if x.startswith("| ") and not x.startswith("| family")) == len(rows), "a row was dropped"
        assert {FAM[k] for k in rows} >= set(FAMS), "a metric family is missing"
        return out

    def side_by_side():
        rows = list(d["seeds"]["s0"]["R_minus_C0"].keys())
        assert rows == list(r["seeds"]["s0"]["R_minus_C0"].keys()), "RPT and R read different rows"
        out = ["**RPT − C0 against R − C0, all rows** (two separate paired reads against C0; there is no paired RPT − R interval, and none "
               "is claimed)", "", "| family | row | better | R s0 | RPT s0 | R s1 | RPT s1 |", "|---|---|---|---|---|---|---|"]
        for fam, k in ordered(rows):
            c = []
            for s in SS:
                for J in (r, d):
                    v = J["seeds"][s]["R_minus_C0"][k]
                    c.append(f"{f(v['delta'])}{' sep' if v['separated'] else ''}")
            out.append(f"| {fam} | {k} | {'higher' if k in GD.HIGHER_BETTER else 'lower'} | " + " | ".join(c) + " |")
        return out

    let, facts = decide(d, r_share)
    row = ROWS[let]
    L = ["# RECOVER-NEXT, arm RPT (R's bank law at p 0.3 with `--recover-dev-shape pertick`): read result and the committed next arm", ""]
    L.append(f"Card `PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `{d['cards']['recover_next'][:8]}…`) section 3, the RECOVER-FT section 6 / 7 "
             f"rules with R := RPT; addendum 4 (`PREREG_RECOVER_NEXT_ADDENDUM4_RPT.md`, frozen before any RPT run) rows M, I, N, J, K1, K2 "
             f"decide the next arm. Tier **{d['tier']}** (self-action open loop on logged cameras: the model's own actions re-integrate the "
             f"ego state). Estimator: {d['estimator']}. Reader sha256 `{d['reader_sha256'][:12]}`, mode `{d['mode']}`, run clock "
             f"{d['run_clock']}. Evidence class MEASURED (ours): `RECOVER_READ_RPT.json` in this folder.")
    if mode:
        L += ["", f"**DRY RUN** ({mode} {dry}): not an RPT result."]
    L += ["", "## Verdict", ""]
    if facts["refused"]:
        L.append(f"* **READ REFUSED**: refused = {d.get('refused')!r}; failed controls {d.get('controls_failed')}; composite status "
                 f"{(d.get('composite') or {}).get('status')!r}. A refusal is a result, not a label.")
    else:
        c = d["composite"]
        L.append(f"* **Label: {d['label']}** (H-RECOVER-NEXT-1, arm RPT). Controls: {len(d['controls'])} run, {len(d['controls_failed'])} "
                 f"failed; KA1 bit-identity rerun: PASS={d['controls']['KA1_bit_identity_rerun_of_R_s0']['PASS']}. Guards per seed: "
                 f"s0 {facts['guards_pass']['s0']}, s1 {facts['guards_pass']['s1']} (reader guards.PASS {d['guards']['PASS']}). M1 share "
                 f"closed RPT {d['primary']['share_closed']['s0']:.4f} / {d['primary']['share_closed']['s1']:.4f} against R "
                 f"{r_share['s0']:.4f} / {r_share['s1']:.4f} (exceeds R: s0 {facts['share_exceeds_R']['s0']}, s1 "
                 f"{facts['share_exceeds_R']['s1']}; bar {d['primary']['share_bar']}).")
        L.append(f"* **Composite RPTC = RPT + `--smooth sg72 --feasible kbox` vs P10: {c['status']}** (DOMINANT at both seeds: "
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
        J3 = (("R", r), ("RAPR", a), ("RPT", d))
        L += ["## RPT against R (same law, same dose, constant shape); RAPR (the largest closure before RPT) for reference "
              "(reported, not deciding)", ""]
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
        L.append("| composite ADE6 XC − P10 (m) | <= -0.2415, separated | " + " | ".join(
            f"{f(J['seeds'][s]['RC_minus_P10']['ADE6']['delta'])}" for _, J in J3 for s in SS) + " |")
        L.append("| floor: abs(X − C0) M1 vs abs(C0rep − C0) | exceeds | " + " | ".join(
            f"{J['floor']['per_seed'][s]['abs_R_minus_C0']:.4f} vs {J['floor']['per_seed'][s]['abs_C0rep_minus_C0']:.4f}"
            for _, J in J3 for s in SS) + " |")
        L.append("| frame-freeze share closed | robust bar | " + " | ".join(
            f"{J['frame_freeze']['share_closed'][s]:.4f}" for _, J in J3 for s in SS) + " |")
        L.append("")
        ns = need["seeds"]
        L.append(f"Closure-need estimate of the decision item (ESTIMATED, post hoc, not a rule; `rowk_decision/d2_closure_need.json`): "
                 f"about {ns['s0']['share_needed']:.2f} (s0) / {ns['s1']['share_needed']:.2f} (s1) of the M1 excess for the composite to "
                 f"stop being separated-worse on M1; RPT closes {d['primary']['share_closed']['s0']:.4f} / "
                 f"{d['primary']['share_closed']['s1']:.4f}.")
        L.append("")
        pc = pertick_counts(runs, strict=mode is None)
        L += ["**The per-tick table** (RPT's launch-gate selfcheck, `levers.recover_table`; NOW columns compared with R's gate table at "
              "the same seed: expected equal when no record is dropped, since the NOW row is R's offset bit for bit)", "",
              "| seed | n | kept | dropped (per-row deviation not finite) | ticks | R's n | NOW columns equal R's |", "|---|---|---|---|---|---|---|"]
        for s in SS:
            v = pc[s]
            if "absent" in v:
                L.append(f"| {s} | absent ({v['absent'][:80]}) | | | | | |")
            else:
                eq = "all five" if all(v["now_equal_R"].values()) else ", ".join(k for k, ok in v["now_equal_R"].items() if not ok) + " differ"
                L.append(f"| {s} | {v['n']} | {v['kept']} | {v['dropped_per_tick_row_unreal']} | {v['ticks']} | {v['R_n']} | {eq} |")
        L.append("")
        L += row_table("RC_minus_P10", "Composite RPTC − P10, all 19 rows (Read A; D2 = no row separated-worse)")
        L.append("")
        L += row_table("R_minus_C0", "RPT − C0, all 19 rows (the label's guards read the non-crawl rows)")
        L.append("")
        L += side_by_side()
        L += ["", "## What this establishes, and what it cannot", ""]
        L.append("* RPT and R share the bank law, the dose p 0.3, the random stream and the drawn record; they differ only in the shape of "
                 "the deviation over the history window. The comparison is two separate paired reads against C0, not a paired RPT − R read.")
        L.append("* One training run per seed (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only; the "
                 "training-run question is the C0rep floor row above. T1-kin is not T2: nothing here is a driving claim beyond T1-kin.")
        L.append("* Every row M, I, N, J, K1, K2 is reached by a mutation test (`test_gen_rpt.py`), written and run before the read.")
        L.append("")
    L += ["## Next (Rule Zero)", ""]
    L.append(f"* {row[0]} decides: {row[1]}." + ("" if let == "I" else " Any new box-2 arm needs the PI's go (box 2 runs one job at a time, SAFE MODE)."))
    open(outf, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("wrote", outf, len(L), "lines; row", let, facts)
    return let


if __name__ == "__main__":
    main(sys.argv)
