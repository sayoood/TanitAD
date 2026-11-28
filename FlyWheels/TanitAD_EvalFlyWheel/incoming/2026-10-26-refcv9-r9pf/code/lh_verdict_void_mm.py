"""lh_verdict_void_mm.py -- R9PF_LH final record on the Master Mind's order (2026-10-26 ~23:45): VOID (K-EQ precondition not completed; the CPU K-EQ
roll was stopped by the MM).  NO compute: reads the provisional run-2 bars (lh_r9pf_run2.py, scratch), the partial roll logs (window COUNTS only, no
clip-id text is kept), the plumbing-test K-EQ diagnostic, and the run-1 VOID record, and writes result/R9PF_LH_VERDICT.{json,md} (the run-1 record is
EMBEDDED verbatim with its sha256, not lost) + copies the provisional JSON into result/.  NEW file (EvalFlyWheel run 2).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

PROV_SRC = RC.SCRATCH / "R9PF_LH_RUN2_PROVISIONAL_NO_KEQ.json"
PROV = RC.RESULT / "R9PF_LH_RUN2_PROVISIONAL_NO_KEQ.json"
OLD_J, OLD_M = RC.RESULT / "R9PF_LH_VERDICT.json", RC.RESULT / "R9PF_LH_VERDICT.md"
OLD_SHA = {"json": "ebc82e313519f57f62b1499e825d3df91ee18cc16f91037c3634010f84dbdee4", "md": "41f247e761a70c97140bbf568102c4b3e943b123273d41159c7f16a0e92f5d62"}
PLUMB = RC.SCRATCH / "KEQ_AMEND2_PLUMBING_TEST.json"


def f(c) -> str:
    if not isinstance(c, dict) or c.get("delta") is None:
        return "n/a"
    return (f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" SEP" if c.get("separated") else "")
            + f" ({c['mean_new']:.4f} vs {c['mean_ref']:.4f}, n={c['n_windows']}/{c['n_episodes']} ep)")


def roll_progress(log: Path) -> dict:
    """Window COUNTS from the roll's '[k/15] <clip> N windows Ts' progress lines; the clip token is discarded, never stored."""
    rx = re.compile(r"\[(\d+)/(\d+)\]\s+\S+\s+(\d+) windows (\d+)s")
    rows = [(int(a), int(b), int(n), int(t)) for a, b, n, t in rx.findall(log.read_text(encoding="utf-8", errors="replace"))] if log.is_file() else []
    return {"clip_chunks_done": len(rows), "clip_chunks_total": rows[0][1] if rows else 15, "windows_done": sum(r[2] for r in rows),
            "elapsed_s_at_last_chunk": rows[-1][3] if rows else None}


def main() -> int:
    for k, p in (("json", OLD_J), ("md", OLD_M)):
        if RC.sha256_file(p) != OLD_SHA[k]:
            raise SystemExit(f"[void] {p} is not the run-1 record ({OLD_SHA[k]}) -- refusing")
    old_j, old_m = json.loads(OLD_J.read_text(encoding="utf-8")), OLD_M.read_text(encoding="utf-8")
    shutil.copyfile(PROV_SRC, PROV)
    if RC.sha256_file(PROV) != RC.sha256_file(PROV_SRC) or len(RC.sha256_file(PROV)) != 64:
        raise SystemExit("[void] provisional copy does not re-hash -- refusing")
    v = json.loads(PROV.read_text(encoding="utf-8"))
    C, B = v["cells"], v["bars"]
    p0 = roll_progress(RC.WORK / "logs" / "a3_keq_amend2_cpu_s0_on.log")
    p1 = roll_progress(RC.WORK / "logs" / "a3_keq_amend2_cpu_s1_on.log")
    rec_s0 = (RC.SCRATCH / "keq_amend2_cpu_s0_on" / "ROLL_RECORD.json").is_file()
    rec_s1 = any((RC.SCRATCH / d / "ROLL_RECORD.json").is_file() for d in ("keq_amend2_cpu_s1_on", "keq_amend2_cpu_s1_on_KILLED_partial_2105"))
    pt = json.loads(PLUMB.read_text(encoding="utf-8")) if PLUMB.is_file() else None
    floor = ({s: x["BESIDE_batch_size_not_registered"] for s, x in pt["K-EQ_AMEND2"]["per_seed"].items()} if pt else None)
    keq = {"status": "NOT COMPLETED -- no K-EQ record exists (the roll writes keq_records.npz / ROLL_RECORD.json only at the end), so no max |d| "
                     "and no decision agreement exist for the CPU in-forward roll",
           "seed0": {**p0, "record_exists": rec_s0, "how_it_ended": "stopped by the Master Mind (explicit PID) at ~23:45 after ~6.6 h"},
           "seed1": {**p1, "record_exists": rec_s1, "how_it_ended": "killed by this agent at ~21:03 (its own PIDs) when the box's commit headroom fell "
                                                                   "to 1.23 GB (C: 0.2 GB free, pagefile could not grow)"},
           "measured_cpu_rate": "~47 min per ~34-window clip chunk alone (~80 s / window, 4 threads); the CUDA roll does a chunk in ~18 s",
           "DIAGNOSTIC_batch_floor_not_registered": {
               "rule": "on the banked CUDA ON roll's own inputs, CPU offline composition batched (515) vs one window at a time -- the SAME function, "
                       "SAME device; the in-forward call is batch 1 (counters batch_gt1 = 0), the registered offline composition is batched",
               "per_seed": floor, "source": str(PLUMB) if pt else None,
               "reading": "the CPU's own batch-size rounding alone exceeds 2 ulp on 13 / 515 (s0) and 14 / 515 (s1) windows (max 1.53e-5 m), so "
                          "K-EQ as written under AMENDMENT2 would very likely have FAILED even device-matched; a batch-matched form (C2: the forward's "
                          "own inputs, batch 1) is the discriminating comparison -- an MM ruling, not run"}}
    # ---- provisional bars beside, every clause per inference seed and per fit seed
    beside = {}
    for a, b in B.items():
        beside[a] = {"class_provisional": b["class"], "arm_bars": {k: b[k] for k in ("PA-1a", "PA-1b", "PA-2", "PA-G", "PA-R", "PASS")}, "per_fit_seed": {}}
        for fs, x in b["per_fit_seed"].items():
            t = f"{a}_{fs}"
            cl = {}
            for s in ("s0", "s1"):
                c = C[t][s]
                cl[s] = {"PA-1a": {"holds": x["PA-1a_per_seed"][s], "MD_vs_H2CN_975_sep_below": f(c["MD_vs_H8_975"]), "FD_vs_H2CN_95_not_sep_above": f(c["FD_vs_H8_95"])},
                         "PA-1b": {"holds": x["PA-1b_per_seed"][s], "FD_vs_FACTORED_975_sep_below": f(c["FD_vs_FX_975"]), "MD_vs_FACTORED_95_not_sep_above": f(c["MD_vs_FX_95"])},
                         "PA-2": {**x["PA-2_per_seed"][s], "S2_ADE_vs_RETIME_975": f(c["ADE_S2_vs_D8_975"]), "S6_ADE_vs_RETIME_975": f(c["ADE_S6_vs_D8_975"]),
                                  "S2_E1_vs_ha0_ext_pose_975": f(c["ADE_S2_vs_HP_975"]), "S6_E3_vs_ha0_ext_pose_975": f(c["ADE_S6_vs_HP_975"])},
                         "rates": {"MD": c["MD_rate"], "FD": c["FD_rate"], "exceedance": c["EXC_rate"], "MD_n": c["MD_n"], "FD_n": c["FD_n"]},
                         "exceedance_vs_H2CN_95": f(c["EXC_vs_H8_95"])}
            beside[a]["per_fit_seed"][fs] = {"clauses": cl, "PA-G_holds": x["PA-G"], "PA-G_worse_rows_95": x["PA-G_worse_rows"]}
    # ---- the five worst PA-G rows (primary arms, fit seed 0, both seeds): ranked by RELATIVE degradation |delta| / |mean_ref|
    cand = []
    for a in ("R9LH-ARG", "R9LH-SOFT"):
        for s in ("s0", "s1"):
            c = C[f"{a}_fs0"][s]
            for sf in ("S2", "S6"):
                for k, x in c[f"fam_{sf}_vs_H8_95"].items():
                    bad = (x["lower_is_better"] and x.get("separated") and x["lo"] > 0) or ((not x["lower_is_better"]) and x.get("separated") and x["hi"] < 0)
                    if bad:
                        cand.append((abs(x["delta"]) / max(abs(x["mean_ref"]), 1e-12), f"{a} fs0 {s} {sf} {x['family']} {k}: {f(x)}"))
            e = c["EXC_vs_H8_95"]
            if e.get("separated") and e["lo"] > 0:
                cand.append((abs(e["delta"]) / max(abs(e["mean_ref"]), 1e-12), f"{a} fs0 {s} exceedance: {f(e)}"))
    cand.sort(key=lambda r: -r[0])
    worst5 = [{"relative_degradation": round(r, 4), "row": t} for r, t in cand[:5]]
    # ---- decision stats: disagreement with the GT regime on GT-decel windows (S6 compliant)
    dstat = {}
    for a in ("R9LH-ARG", "R9LH-SOFT", "R9DER-ARG"):
        for s in ("s0", "s1"):
            d = C[f"{a}_fs0"][s]["TACTICAL_decision_vs_GT_regime_S6_compliant"]
            M = d["rows_GT_regime_cols_decision (0 decel, 1 cruise, 2 accel)"]
            nd = sum(M[0])
            dstat[f"{a}|{s}"] = {"GT_decel_windows": nd, "decision_disagrees": nd - M[0][0], "disagree_rate": (nd - M[0][0]) / nd if nd else None,
                                 "called_cruise": M[0][1], "called_accel": M[0][2], "false_decel_on_GT_cruise": d["false_decel_on_GT_cruise"],
                                 "confusion": M}
    out = {"verdict": "VOID", "reason": "K-EQ precondition not completed; stopped by the MM. PREREG_R9PF_LH s2 needs R9-PF's K-EQ / K-OFF PASSED; K-OFF "
                                        "PASSED (run 1), K-EQ FAILED as written (run 1, CUDA vs CPU) and the device-matched K-EQ under AMENDMENT2 was "
                                        "stopped before any record existed.",
           "stopped_by": "Master Mind, 2026-10-26 ~23:45 (seed-0 CPU roll PID killed by explicit PID); reason given: the provisional bars FAIL on every "
                         "clause including PA-G, so the precondition could only turn VOID into FAIL, at ~13+ further CPU hours",
           "prereg": {"file": str(RC.WORK / "PREREG_R9PF_LH.md"), "sha256": "6de4e4bb0e8abfe0ada03e3e0e401faea9996857ed80744547b923c4f9f2bd59"},
           "amendment2": {"sha256": "865c95692383f78a0c6355003c6d1b0ec63413a5816827babacb28cc5674cc3d"},
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames, not driving performance",
           "estimator": v["estimator"], "interval_rule": v["interval_rule"],
           "K-EQ_AMEND2_partial": keq,
           "BESIDE_provisional_bars_NOT_A_VERDICT": {"word_had_the_precondition_held": "FAIL (s5 'Both FAIL': R9LH-ARG and R9LH-SOFT FAIL every bar)",
                                                     "file": str(PROV), "sha256": RC.sha256_file(PROV), "controls_other_than_precondition":
                                                     {k: c.get("PASS") for k, c in v["controls"].items() if isinstance(c, dict) and "PASS" in c and k != "K-PRECONDITION"},
                                                     "per_arm": beside},
           "worst5_PA-G_rows_primary_fs0": {"rule": "rows separated WORSE than H2CN_DEPLOY8 (95 %), R9LH-ARG / R9LH-SOFT fit seed 0, both inference seeds; "
                                                    "ranked by |delta| / |reference mean|", "rows": worst5, "n_candidates": len(cand)},
           "decision_stats_S6_compliant": dstat, "decision_n": v["decision_stats"],
           "run1_record_embedded": {"json_sha256": OLD_SHA["json"], "md_sha256": OLD_SHA["md"], "json": old_j, "md": old_m},
           "script_sha256": RC.sha256_file(__file__), "generated": time.strftime("%FT%T%z")}
    RC.jdump(out, OLD_J)
    # ---- md
    L = ["# R9PF_LH_VERDICT -- VOID (K-EQ precondition not completed; stopped by the MM)", "",
         f"**tier:** {out['tier']}  ", f"**estimator:** {out['estimator']}  ", f"**intervals (provisional bars):** {out['interval_rule']}", "",
         f"**reason:** {out['reason']}  ", f"**stopped by:** {out['stopped_by']}.", "",
         "## K-EQ under AMENDMENT2 -- partial", "",
         f"* seed 0: {p0['clip_chunks_done']} / {p0['clip_chunks_total']} clip chunks = **{p0['windows_done']} windows** rolled on CPU (of >= 500 required); "
         f"record exists: {rec_s0}. {keq['seed0']['how_it_ended']}.",
         f"* seed 1: {p1['clip_chunks_done']} / {p1['clip_chunks_total']} chunks = **{p1['windows_done']} windows**; record exists: {rec_s1}. {keq['seed1']['how_it_ended']}.",
         f"* **No K-EQ record exists, so no max |d| and no decision agreement exist** for the CPU in-forward roll. Rate MEASURED: {keq['measured_cpu_rate']}.",
         "* DIAGNOSTIC, not registered (CUDA-roll inputs, CPU offline): batched vs one-at-a-time composition of the SAME inputs exceeds 2 ulp on "
         + (", ".join(f"{s}: {x['C3_offline_batched_vs_batch1_same_inputs']['n_over_2ulp']} / {x['C3_offline_batched_vs_batch1_same_inputs']['n']} "
                      f"(max {x['C3_offline_batched_vs_batch1_same_inputs']['max_abs_m']:.3g} m)" for s, x in floor.items()) if floor else "n/a")
         + " -- the tolerance sits below the CPU's own batch-size rounding; K-EQ as written would very likely have failed device-matched too.", "",
         "## Provisional bars -- BESIDE, NOT A VERDICT (word had the precondition held: FAIL, s5 'Both FAIL')", "",
         "| arm | fit seed | PA-1a s0/s1 | PA-1b s0/s1 | PA-2 (vs RETIME; vs ha0_ext_pose) s0 / s1 | PA-G | PA-G rows |", "|---|---|---|---|---|---|---|"]
    for a, b in B.items():
        for fs, x in b["per_fit_seed"].items():
            p2 = x["PA-2_per_seed"]
            L.append(f"| {a} | {fs} | {x['PA-1a_per_seed']['s0']}/{x['PA-1a_per_seed']['s1']} | {x['PA-1b_per_seed']['s0']}/{x['PA-1b_per_seed']['s1']} | "
                     f"({p2['s0']['vs_RETIME']}; {p2['s0']['E1_E3_vs_ha0_ext_pose']}) / ({p2['s1']['vs_RETIME']}; {p2['s1']['E1_E3_vs_ha0_ext_pose']}) | "
                     f"{x['PA-G']} | {len(x['PA-G_worse_rows'])} |")
    for a in B:
        for fs in ("fs0", "fs1", "fs2"):
            cl = beside[a]["per_fit_seed"][fs]["clauses"]
            L += ["", f"### {a} {fs} -- clause cells (S6 compliant unless named)", "", "| clause | seed 0 | seed 1 |", "|---|---|---|"]
            for grp, key in (("PA-1a", "MD_vs_H2CN_975_sep_below"), ("PA-1a", "FD_vs_H2CN_95_not_sep_above"), ("PA-1b", "FD_vs_FACTORED_975_sep_below"),
                             ("PA-1b", "MD_vs_FACTORED_95_not_sep_above"), ("PA-2", "S2_ADE_vs_RETIME_975"), ("PA-2", "S6_ADE_vs_RETIME_975"),
                             ("PA-2", "S2_E1_vs_ha0_ext_pose_975"), ("PA-2", "S6_E3_vs_ha0_ext_pose_975")):
                L.append(f"| {grp} {key} | {cl['s0'][grp][key]} | {cl['s1'][grp][key]} |")
            L.append(f"| PA-G exceedance vs H2CN (95 %) | {cl['s0']['exceedance_vs_H2CN_95']} | {cl['s1']['exceedance_vs_H2CN_95']} |")
            L.append(f"| rates MD / FD / exceedance | {cl['s0']['rates']['MD']:.4f} / {cl['s0']['rates']['FD']:.4f} / {cl['s0']['rates']['exceedance']:.4f} | "
                     f"{cl['s1']['rates']['MD']:.4f} / {cl['s1']['rates']['FD']:.4f} / {cl['s1']['rates']['exceedance']:.4f} |")
            rows = beside[a]["per_fit_seed"][fs]["PA-G_worse_rows_95"]
            L += ["", f"PA-G rows separated WORSE than H2CN_DEPLOY8 ({len(rows)}):"] + [f"* `{r}`" for r in rows]
    L += ["", "## Five worst PA-G rows (R9LH-ARG / R9LH-SOFT fs0, both seeds; ranked by |delta| / |reference mean|)", ""]
    L += [f"* `{r['row']}` (relative {r['relative_degradation']:+.4f})" for r in worst5]
    L += ["", "## Decision stats -- disagreement with the GT regime on GT-decel windows (S6 compliant, fit-seed independent)", "",
          "| arm | seed | GT-decel windows | disagree | rate | called cruise | called accel | false decel on GT cruise |", "|---|---|---|---|---|---|---|---|"]
    for k, d in dstat.items():
        a, s = k.split("|")
        L.append(f"| {a} | {s} | {d['GT_decel_windows']} | {d['decision_disagrees']} | {d['disagree_rate']:.4f} | {d['called_cruise']} | {d['called_accel']} | "
                 f"{d['false_decel_on_GT_cruise']:.4f} |")
    L += ["", "STRATEGIC n/a (refcv8 runs --no-strategic; n = 0). Distance-keeping n/a (no lead-agent join in the eval139 panel; n = 0). The full "
          "four-family tables are in the provisional JSON (`cells.<arm>_fs<k>.s<seed>.fam_{S2,S6}_vs_H8_95`).", "",
          "## Run-1 record (kept verbatim; sha256 json " + OLD_SHA["json"][:12] + "..., md " + OLD_SHA["md"][:12] + "...)", "", "```", old_m.strip(), "```", ""]
    OLD_M.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[void] wrote {OLD_J} / {OLD_M}; worst5 {len(worst5)}; K-EQ partial s0 {p0['windows_done']} windows  ZZVOIDOKZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
