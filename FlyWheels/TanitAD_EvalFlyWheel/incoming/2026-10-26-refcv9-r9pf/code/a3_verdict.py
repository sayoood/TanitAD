"""a3_verdict.py -- PREREG_REFCV9_INMODEL_PA PHASE A: result/R9PF_PA_BARS.json -> result/R9PF_PA_VERDICT.{json,md}, mechanically (no number is computed
here except formatting).  The committed outcome is prereg s5's text for the word a3_bars.py produced.  NEW file (EvalFlyWheel 2026-10-26)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402
from a3_bars import fmt  # noqa: E402

OUTCOME = {
    "PASS": "PASS (A) -> R9PF is refcv9's speed decode; the Phase-B compute request goes to the PI with these numbers. R9PF on refcv8's trunk is also a "
            "refcv8 deliverable candidate, which needs its OWN SHIP-R8 amendment (never assumed from this read).",
    "PARTIAL": "PARTIAL (PA-1a xor PA-1b) -> a new operating point, not a frontier move. Next lever: a larger clean split (V-C, more fresh clips; availability "
               "UNVERIFIED) and the decision's AUC vs the lon head's on the same windows read beside.",
    "INERT": "INERT -> the decision SOURCE adds nothing at 297 clips; the lever moves to training the lon decision in the trunk with richer supervision "
             "(PLAN s12), a Phase-B design item.",
    "FAIL-GUARD": "FAIL-GUARD (PA-G) -> name the rows; next a constrained stage-2 objective, pre-registered first.",
    "FAIL": "FAIL -> none of the prereg s5 named outcomes applies (both PA-1a and PA-1b fail and the INERT condition does not hold); the bars are reported "
            "as written and the next lever is named below.",
    "VOID": "VOID -> a control failed (prereg s3: 'a failed control VOIDS the bars'); the bars are reported BESIDE as unregistered numbers only.",
}


def main() -> int:
    RC.check_prereg()
    RC.check_amendment1()
    RC.check_fix_pin()
    b = json.loads((RC.RESULT / "R9PF_PA_BARS.json").read_text(encoding="utf-8"))
    dev = json.loads((HERE / "deviations_r9pf.json").read_text(encoding="utf-8"))
    nxt = json.loads((HERE / "next_r9pf.json").read_text(encoding="utf-8")) if (HERE / "next_r9pf.json").is_file() else {}
    word, prim, R = b["word"], b["primary_arm"], b["cells"]
    v = {"verdict": word, "committed_outcome": OUTCOME[word], "tier": b["tier"], "estimator": b["estimator"], "primary_arm": prim,
         "bars_primary": b["bars_primary"], "PA-R": b["PA-R"], "PASS": b["PASS"], "failed_controls": b["failed_controls"],
         "controls": {k: (c if isinstance(c, bool) else {kk: vv for kk, vv in c.items() if kk not in ("detail", "pins")}) for k, c in b["controls"].items()},
         "inert_read": b["inert_read"], "inference_seed_floor_primary": b["inference_seed_floor_primary"], "deviations": dev, "next": nxt,
         "bars_file": {"file": str(RC.RESULT / "R9PF_PA_BARS.json"), "sha256": RC.sha256_file(RC.RESULT / "R9PF_PA_BARS.json")},
         "generated": time.strftime("%FT%T%z")}
    RC.jdump(v, RC.RESULT / "R9PF_PA_VERDICT.json")
    L = [f"# R9PF_PA_VERDICT -- {word}", "",
         f"**tier:** {b['tier']}  ", f"**estimator:** {b['estimator']}  ",
         "**questions answered:** another draw of EPISODES (the bootstrap); another TRAINING run = fit seeds 0/1/2 (PA-R); another INFERENCE run = seeds 0 / 1 "
         "reported apart (floor beside).", "",
         f"**amendment:** PREREG_REFCV9_INMODEL_PA_AMENDMENT1 (sha256 {RC.AMEND1_SHA}) -- controls only; every bar, arm, population and estimator "
         "unchanged. Prereg sha256 " + RC.PREREG_SHA + ".", "",
         "**Disclosed (AMENDMENT1 change 5):**", "",
         "* **(a) tree drift (NOT an R9 effect; owned by H-R9-TREEDRIFT, not investigated here).** The tip tree's planner without the R9 head selects "
         "a different path from the fb47a58-rolled bank on 70 / 70 calibration windows (max 7.24 m; z_tac bit-equal; deterministic run-to-run). "
         "So the bars are composed OFFLINE on the BANKED fb47a58 paths (the same path as H2CN_DEPLOY8 / RETIME_DEPLOY8), and the in-forward roll "
         "serves K-OFF / K-EQ only.",
         "* **(b) a DEFECT in the banked R9 code, fixed by one line.** `enable_refcv9` called `torch.manual_seed` inside `fork_rng(devices=[])`, "
         "which re-seeds the CUDA RNG and does not restore it: merely building the head moved the selected path on 70 / 70 calibration windows "
         "(max 8.10 m). Fix: `torch.default_generator.manual_seed` (CPU generator only), pinned in FIX_R9_RNG.sha256; flags-ON == flags-OFF "
         "bit for bit after it on those 70 calibration windows; the registered eval139 check is K-OFF in the controls table.",
         "* **CONSTZ fit-side flag.** On the stage-2 fit (calibration split, out-of-fold CV objective), R9PF-CONSTZ -- z_tac replaced by its "
         "training mean, i.e. NO vision -- scored BETTER than R9PF: 1.83-1.85 vs 1.95-1.97 over fit seeds 0/1/2 (R9PF_FIT_INFO.json). "
         "K-CONSTZ and the INERT branch decide it on eval, as sealed.", "",
         f"**committed outcome (prereg s5):** {OUTCOME[word]}", "",
         "**every bar, as written (primary R9PF_fs0):** " + ", ".join(f"{k} {'HOLDS' if v else 'FAILS'}" for k, v in b["bars_primary"].items())
         + f", PA-R {'HOLDS' if b['PA-R']['holds'] else 'FAILS'} -> PASS = {b['PASS']}.", ""]
    if b["failed_controls"]:
        L += [f"**failed controls:** {', '.join(b['failed_controls'])}", ""]
    L += ["## Bars (primary R9PF_fs0; S6 compliant unless stated; paired on the SAME draw)", "", "| bar | seed 0 | seed 1 | holds |", "|---|---|---|---|"]
    for s_ in ("s0", "s1"):
        pass
    r0, r1 = R[prim]["s0"], R[prim]["s1"]
    bp = b["bars_all_heads"][prim]
    L.append(f"| PA-1a MD vs H2CN_DEPLOY8 (sep below) | {fmt(r0['MD_vs_H8'])} | {fmt(r1['MD_vs_H8'])} | {bp['PA-1a']} |")
    L.append(f"| PA-1a FD vs H2CN_DEPLOY8 (NOT sep above) | {fmt(r0['FD_vs_H8'])} | {fmt(r1['FD_vs_H8'])} | |")
    L.append(f"| PA-1b FD vs FACTORED_DEPLOY8 (sep below) | {fmt(r0['FD_vs_FX'])} | {fmt(r1['FD_vs_FX'])} | {bp['PA-1b']} |")
    L.append(f"| PA-1b MD vs FACTORED_DEPLOY8 (NOT sep above) | {fmt(r0['MD_vs_FX'])} | {fmt(r1['MD_vs_FX'])} | |")
    L.append(f"| PA-2 S2 ADE vs RETIME_DEPLOY8 (sep below) | {fmt(r0['ADE_S2_vs_D8'])} | {fmt(r1['ADE_S2_vs_D8'])} | {bp['PA-2']} |")
    L.append(f"| PA-2 S6 ADE vs RETIME_DEPLOY8 (sep below) | {fmt(r0['ADE_S6_vs_D8'])} | {fmt(r1['ADE_S6_vs_D8'])} | |")
    L.append(f"| PA-2 E1 (S2) vs ha0_ext_pose (sep below) | {fmt(r0['ADE_S2_vs_HP'])} | {fmt(r1['ADE_S2_vs_HP'])} | |")
    L.append(f"| PA-2 E3 (S6) vs ha0_ext_pose (sep below) | {fmt(r0['ADE_S6_vs_HP'])} | {fmt(r1['ADE_S6_vs_HP'])} | |")
    L.append(f"| PA-G exceedance vs H2CN_DEPLOY8 (NOT sep above; full plan, all windows) | {fmt(r0['EXC_vs_H8'])} | {fmt(r1['EXC_vs_H8'])} | {bp['PA-G']} |")
    L.append(f"| PA-R (PA-1a, 1b, 2 at fit seeds 0/1/2) | | | {b['PA-R']['holds']} |")
    L += ["", f"PA-G separated-worse rows (primary): {bp['PA-G_worse_rows'] or 'none'}", "",
          "## Every head: MD / FD (S6 compliant) and the bars", "",
          "| head | MD s0 | FD s0 | MD s1 | FD s1 | PA-1a | PA-1b | PA-2 | PA-G | exceed s0 / s1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in sorted(b["bars_all_heads"]):
        x, y, bb = R[t]["s0"], R[t]["s1"], b["bars_all_heads"][t]
        L.append(f"| {t} | {x['MD_rate']:.4f} | {x['FD_rate']:.4f} | {y['MD_rate']:.4f} | {y['FD_rate']:.4f} | {bb['PA-1a']} | {bb['PA-1b']} | {bb['PA-2']} | "
                 f"{bb['PA-G']} | {x['EXC_rate']:.4f} / {y['EXC_rate']:.4f} |")
    for t, nm in (("H8", "H2CN_DEPLOY8 (beside)"), ("FX", "FACTORED_DEPLOY8 (beside)"), ("D8", "RETIME_DEPLOY8 (beside)"), ("OS", "OSONLY = the battery os-only plan, no re-timing (beside)")):
        x, y = R[t]["s0"], R[t]["s1"]
        L.append(f"| {nm} | {x['MD_rate']:.4f} | {x['FD_rate']:.4f} | {y['MD_rate']:.4f} | {y['FD_rate']:.4f} | | | | | {x['EXC_rate']:.4f} / {y['EXC_rate']:.4f} |")
    L += ["", "GT-oracle pick from the same fan: MD 0.075 / FD 0.057 (INHERITED from PLAN_REFCV9 s12 via the prereg s0; not re-run here).", "",
          f"n: MD on {R[prim]['s0']['MD_n']} GT-decel windows, FD on {R[prim]['s0']['FD_n']} GT-cruise windows (S6 compliant, seed 0).", "",
          "## Controls", "", "| control | PASS | note |", "|---|---|---|"]
    for k, c in b["controls"].items():
        ok = c if isinstance(c, bool) else c.get("PASS")
        note = "" if isinstance(c, bool) else json.dumps({kk: vv for kk, vv in c.items() if kk not in ("PASS", "detail", "pins")}, default=str)[:300]
        L.append(f"| {k} | {ok} | {note} |")
    L += ["", "## Four families -- R9PF_fs0 minus H2CN_DEPLOY8 (compliant sets; worse = separated in the bad direction)", "",
          "| surface | family | metric | seed 0 | seed 1 |", "|---|---|---|---|---|"]
    for sf in ("S2", "S6"):
        for k, c0 in R[prim]["s0"][f"fam_{sf}_vs_H8"].items():
            c1 = R[prim]["s1"][f"fam_{sf}_vs_H8"][k]
            L.append(f"| {sf} | {c0['family']} | {k}{'' if c0['lower_is_better'] else ' (higher better)'} | {fmt(c0)} | {fmt(c1)} |")
    L += ["", "STRATEGIC: n/a (refcv8 runs --no-strategic). Distance-keeping (headway / TTC): not instrumented in this battery (no lead-agent join in the "
          "eval139 panel) -- n/a with reason.", "", "## Four families -- R9PF_fs0 minus RETIME_DEPLOY8 (beside)", "", "| surface | metric | seed 0 | seed 1 |",
          "|---|---|---|---|"]
    for sf in ("S2", "S6"):
        for k, c0 in R[prim]["s0"].get(f"fam_{sf}_vs_D8", {}).items():
            L.append(f"| {sf} | {k} | {fmt(c0)} | {fmt(R[prim]['s1'][f'fam_{sf}_vs_D8'][k])} |")
    L += ["", "## INERT read (R9PF vs R9PF-TL, per fit seed, against the fit-seed floor)", "", "```", json.dumps(b["inert_read"], indent=1), "```", "",
          "## Inference-seed floor beside the primary effects (|mean_new(s0) - mean_new(s1)|)", "", "```",
          json.dumps(b["inference_seed_floor_primary"], indent=1), "```", "", "## Deviations (disclosed)", ""]
    L += [f"* **{d['id']}** {d['text']}" for d in dev]
    if nxt:
        L += ["", "## Next lever", "", nxt.get("text", "")]
    (RC.RESULT / "R9PF_PA_VERDICT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[verdict] {word}  ZZA3VERDICT_{word}ZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
