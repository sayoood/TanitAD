"""IMAG1C report: RESULT_IMAG1C.md + LABELS_IMAG1C.json from RESULT_IMAG1C.json, OLDPROBE_IMAG1C.json, the build report, the test log and ../imag1b/RESULT_IMAG1B.json. No statistic is computed here."""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda p: json.load(open(os.path.join(HERE, p), encoding="utf-8"))
R, OP, BR = J("RESULT_IMAG1C.json"), J("OLDPROBE_IMAG1C.json"), J("build_imag1c.report.json")
OLD = J("../imag1b/RESULT_IMAG1B.json")["datasets"]["eval141"]
TESTS = open(os.path.join(HERE, "TEST_IMAG1C.log"), encoding="utf-8", errors="replace").read()
tests_pass = ("\npassed" in TESTS or TESTS.startswith("passed")) and "FAIL" not in TESTS
TAGS, LV, KS = ("M2b_s0", "M2b_s1"), (1, 2, 3), (1, 2, 3)
SHA = open(os.path.join(HERE, "IMPL_CHOICES_IMAG1C.sha256")).read().split()[0]
f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
ci = lambda d: f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
repro = BR["check_i_pass"] and OP["repro_pass"]
banner = "VALID" if (R["banner"] == "VALID" and tests_pass and repro) else "INVALID"
L = []
w = L.append
w("# RESULT_IMAG1C: is `lead_gap` readable in the encoded true future, and does the imagined gap beat Copy-Last, when IA-2 is read on frames with CAPLADDER's probe?")
w(f"Trigger: CAPLADDER read outcome O-NONE (`../capladder/RESULT_CAPLADDER.md`). Spec `IMPL_CHOICES_IMAG1C.md` (sha256 `{SHA[:8]}...`, in `IMPL_CHOICES_IMAG1C.sha256`, written before any IMAG1C row). Tier **T0** (teacher-forced diagnostic of a world model; never a driving claim). STRATEGIC: **PAUSED** (PI item 25). Evidence class on every number: **MEASURED (ours)**, artifacts `RESULT_IMAG1C.json`, `OLDPROBE_IMAG1C.json` (IMAG1B column: `../imag1b/RESULT_IMAG1B.json`, INHERITED as printed there and reproduced here by check iii). Estimators: episode-cluster bootstrap B 2,000 over {BR['n_episodes']} bank episodes, and the refit-under-permutation reference R = {R['R']} (IMAG1B's two-part rule, unchanged). Window frames: {R['n_windows']:,} (old subset {R['n_old']:,}). Written by `report_imag1c.py`.")
w("")
w(f"## Banner: {banner}")
c2, c3 = R["control2"], R["control3"]
w("| control | result | detail |\n|---|---|---|")
w(f"| 2 not-future (Oracle minus not-future separated in every cell with a non-UNREADABLE label; `speed` is the control-only target) | {'pass' if c2['pass'] else 'FAIL'} | {c2['cells']} cells, {len(c2['failed'])} failed {c2['failed'][:6]} |")
sh = c3["shares"]
w(f"| 3 calibration (20 draws scored as if real, at most 5 % separated per kind) | {'pass' if c3['pass'] else 'FAIL'} | " + "; ".join(f"{k} {v['separated']}/{v['rows']} = {100*v['share']:.1f} %" for k, v in sh.items() if k.startswith(('lead_gap', 'READ3'))) + " |")
w(f"| 4 planted / deranged / dual-equals-primal / regression arm (`TEST_IMAG1C.log`, synthetic) | {'pass' if tests_pass else 'FAIL'} | planted channel READABLE and IMAGINES; Predicted = Copy gives COPIES; deranged and Gaussian worlds give no READABLE; control 2 fails when not-future = Oracle |")
w(f"| reproduction (i) rebuilt Z0 / ORA / PRED of the old windows vs the IMAG1B ctx | {'pass' if BR['check_i_pass'] else 'FAIL'} | max abs diff {max(BR['check_i_max_abs_diff_old_windows_vs_ctx'].values()):.1e} (bound 1e-4); (ii) labels equal exactly |")
w(f"| reproduction (iii) IMAG1B's probe on the old windows vs `RESULT_IMAG1B.json` | {'pass' if OP['repro_pass'] else 'FAIL'} | max abs diff of NMSE {OP['repro_max_abs_diff_nmse_vs_RESULT_IMAG1B']:.1e} (bound 1e-4) |")
w("")
w("IA-0 health (predicted k = 1 retrieves the encoded true future, derangement reference): " + ", ".join(f"L{l} {'healthy' if R['healthy'][str(l)] else 'NOT healthy'}" for l in LV) + ".")
w("")
w("## `lead_gap` on frames (deciding). NMSE: 1 = the scoring-fold mean; READABLE = bootstrap upper bound < 1 AND point below the 2.5th percentile of the 199 refit-permutation NMSEs of that row")
w("Per seed: frames n (episodes), Oracle NMSE [95 % episode-cluster CI], reference 2.5th, Copy-Last NMSE, Predicted NMSE, Predicted minus Copy-Last NMSE [95 % CI] (negative = the imagined gap is closer). Label per IMAG1B's rule.\n")
HDR = ["| level | k | label | seed | n (eps) | Oracle NMSE | ref 2.5th | Copy NMSE | Pred NMSE | Pred - Copy NMSE | pred vs Copy |", "|---|---|---|---|---|---|---|---|---|---|---|"]


def rows_for(q, sub="all"):
    out = []
    for l in LV:
        for k in KS:
            lab = R["labels"][f"{sub}|{l}|{k}|{q}"]["label"]
            for tag in TAGS:
                e = R["rows"][f"{sub}|{tag}|{l}|{k}|{q}"]
                pc = e["pred_minus_copy"]
                out.append(f"| L{l} | {k} | {lab} | {tag[-2:]} | {e['n']} ({e['nep']}) | {ci(e['nmse']['oracle'])} | {f3(e['nmse_ref_p2_5'])} | {f3(e['nmse']['copy']['point'])} | {f3(e['nmse']['pred']['point'])} | {-pc['point']:.3f} [{-pc['hi']:.3f}, {-pc['lo']:.3f}] | {'IMAGINES side' if pc['above'] else ('WORSE side' if pc['below'] else 'not separated')} |")
    return out


L += HDR + rows_for("lead_gap")
labs = {(l, k): R["labels"][f"all|{l}|{k}|lead_gap"]["label"] for l in LV for k in KS}
cnt = {x: sum(v == x for v in labs.values()) for x in ("UNREADABLE", "COPIES", "IMAGINES", "WORSE-THAN-COPY")}
w("")
w("Labels on frames, `lead_gap`: " + ", ".join(f"{k} {v}" for k, v in cnt.items()) + " of 9 (level, k) cells.")
w("")
w("## Beside: the older window-subset result and the attribution (no bar)")
w("| level | k | IMAG1B label (old windows, lambda 10, 2 folds, ridge + MLP) | IMAG1B Oracle NMSE s0 / s1 | CAPLADDER ridge on the old windows: label | Oracle NMSE s0 / s1 | IMAG1B probe on all frames: Oracle NMSE s0 / s1 | CAPLADDER ridge on all frames: label | Oracle NMSE s0 / s1 |\n|---|---|---|---|---|---|---|---|---|")
for l in LV:
    for k in KS:
        ol = OLD["ia2_labels"][f"{l}|{k}|lead_gap"]["decision"]
        on = " / ".join(f3(OLD["ia2"][f"ridge|{t}|{l}|{k}|lead_gap"]["nmse"]["oracle"]["point"]) for t in TAGS)
        no = " / ".join(f3(R["rows"][f"old|{t}|{l}|{k}|lead_gap"]["nmse"]["oracle"]["point"]) for t in TAGS)
        pa = " / ".join(f3(OP["rows"][f"all|{t}|{l}|{k}|oracle"]["nmse"]) for t in TAGS)
        na = " / ".join(f3(R["rows"][f"all|{t}|{l}|{k}|lead_gap"]["nmse"]["oracle"]["point"]) for t in TAGS)
        w(f"| L{l} | {k} | {ol} | {on} | {R['labels'][f'old|{l}|{k}|lead_gap']['label']} | {no} | {pa} | {labs[(l, k)]} | {na} |")
w("")
w("Frames per row (seed s0): old subset " + ", ".join(f"L{l}k{k} {R['rows'][f'old|M2b_s0|{l}|{k}|lead_gap']['n']}" for l in LV for k in KS) + " windows with a finite horizon `lead_gap`; all frames " + ", ".join(f"L{l}k{k} {R['rows'][f'all|M2b_s0|{l}|{k}|lead_gap']['n']}" for l in LV for k in KS) + ". Rows whose horizon frame has no `lead_gap` are dropped; the other windows of the 7,481 are not in the row.")
for q in ("lead_closing", "ttc"):
    w("")
    w(f"## Beside, no bar: `{q}` (labels printed for reading only)")
    L += HDR + rows_for(q)
w("")
w("## Reading")
rd = [(l, k) for (l, k), v in labs.items() if v != "UNREADABLE"]
im = [(l, k) for (l, k), v in labs.items() if v == "IMAGINES"]
wo = [(l, k) for (l, k), v in labs.items() if v == "WORSE-THAN-COPY"]
nold = sum(OLD['ia2_labels'][f'{l}|{k}|lead_gap']['decision'] == 'UNREADABLE' for l in LV for k in KS)
w(f"* **Readable on frames:** the encoded true future's `lead_gap` is READABLE (both seeds) in {len(rd)} of 9 (level, k) cells: {', '.join(f'L{l} k{k}' for l, k in rd) or 'none'}. IMAG1B's rows were UNREADABLE in {nold} of 9.")
if im:
    w(f"* **The imagined lead gap beats Copy-Last** (label IMAGINES, both seeds, two-part rule) at {', '.join(f'L{l} k{k}' for l, k in im)}.")
else:
    cp = ', '.join(f'L{l} k{k}' for l, k in rd if labs[(l, k)] == 'COPIES') or 'none'
    w("* **The imagined lead gap does NOT beat Copy-Last by the two-part rule at any (level, k):** no cell is IMAGINES. " + (f"WORSE-THAN-COPY at {', '.join(f'L{l} k{k}' for l, k in wo)}. " if wo else "") + f"Readable cells that are COPIES: {cp} (the Predicted minus Copy-Last NMSE intervals are in the table).")
w(f"* **What the labels change about IMAG1B's row:** IMAG1B's `lead_gap` row read UNREADABLE in {nold} of 9 cells, which fed its trigger E3 (CAPLADDER). On frames with CAPLADDER's probe the row reads {cnt['UNREADABLE']} UNREADABLE, {cnt['COPIES']} COPIES, {cnt['IMAGINES']} IMAGINES, {cnt['WORSE-THAN-COPY']} WORSE-THAN-COPY. The attribution table separates frames from probe (the IMAG1B probe on all frames against the CAPLADDER ridge on the old windows).")
w("* Tier T0; one probe family (ridge); two seeds of one recipe (M2b); eval-141 frames from 141 episodes (56 with a lead). STRATEGIC: PAUSED. The 16-step windows start at every cache step and overlap, so rows share frames; the episode-cluster bootstrap accounts for that and not for training variance (`H-ESTIM-SEED-1`).")
w("")
w("## Files\nAll under `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/imag1c/`: `IMPL_CHOICES_IMAG1C.md/.sha256`, `imag1c_lib.py`, `build_imag1c.py` (+ `.report.json`, `.log`), `run_imag1c.py`, `oldprobe_imag1c.py`, `report_imag1c.py`, `test_imag1c_controls.py`, `TEST_IMAG1C.log`, `RESULT_IMAG1C.json`, `LABELS_IMAG1C.json`, `OLDPROBE_IMAG1C.json`, `draws_imag1c/` (220 draws), `DELETIONS_IMAG1C.log`, `HANDOFF_IMAG1C.md`. Model-side arrays: `F:/TanitAD-scratch/imag1c_ctx.pkl`.")
open(os.path.join(HERE, "RESULT_IMAG1C.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
lab_out = {"tier": "T0", "strategic": "PAUSED (PI item 25)", "evidence_class": "MEASURED (ours)", "estimator": f"episode-cluster bootstrap B 2000 over {BR['n_episodes']} episodes + refit-under-permutation reference R {R['R']}; IMAG1B two-part rule", "banner": banner,
           "labels": R["labels"], "per_seed_oracle_readable": {k: v["oracle_readable"] for k, v in R["rows"].items()},
           "per_seed_pred_minus_copy": {k: {"above": v["pred_minus_copy"]["above"], "below": v["pred_minus_copy"]["below"]} for k, v in R["rows"].items()}}
json.dump(lab_out, open(os.path.join(HERE, "LABELS_IMAG1C.json"), "w"), indent=1)
print("report written", banner)
