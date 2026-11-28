# HANDOFF: TF-N2 design (TF design agent, 2026-11-12). Nothing staged, committed or pushed. No navtest token scored or opened.

## State
- **EP mechanism found** (navtrain held-out, 469 tokens, 13 logs, MEASURED exploratory; `TFN2_EVIDENCE.md` §1).
  - The converted teacher's EP loss (9.41 points) is 1.28 from DAC-failed rows and 8.13 from passing rows.
  - On those passing rows, the teacher's 4 s progress is a median 0.942 of PDM-Closed's. PDM-Closed's IDM drives toward the speed limit; the teacher's action space is jerk-limited (`driverl_teacher.yaml:23,27`).
  - The loss concentrates below 6 m/s, behind 10–30 m leads, at red lights and at standstill.
  - Tc20s, the tracker, the 8-pose sampling and the 4 s horizon are eliminated (Tc20s − rollout progress +0.015 m).
- **Levers ranked** (§2, paired vs Tc20s):
  1. N2-S = CERT over [RW105 → R → C → T]: **+1.59 [+0.64, +2.66] → 95.46**
  2. RW105: +1.18
  3. CERT[R, C, T]: +0.96
  4. CERT[W105, C]: +0.83
  5. W105: +0.62
  6. CERT N1 form: +0.61
  7. P5-R: +0.55
  - W110 is refuted (−0.40; comfort −3.2).
- **Families (§3).** The warp costs human-likeness: speed MAE +0.17 m/s, min TTC −0.9 s. P5-R improves lateral and tactical.
- `PREREG_TF_N2_DRAFT.md` is written for N2-S: K n = 43, pinned-harness route, the bars kept at 95.8 / 99.9.

## Exact next steps
1. **MM: rule on the warp.** It is admissible by the brief's rule (no score, no future), but it is a label speed bias that a student would inherit. If the MM or PI refuses it, the PRIMARY candidate falls back to CERT over [R → C → T] (navtrain +0.96). The draft then needs only arm R.
2. **Implement** `tf_n2_build.py`, `tf_n2_chain.py`, `tf_n2_analyze.py` and `N2_PINS.json`.
   - Copy them from `../n1/a1/` and `../n1/tf_n1_build.py`.
   - The warp is `tfn2_probe.warp`, verbatim.
   - P5-R is `p5r_labels.solve(·, ·, [3.0], track, ND.to_navsim)` with `track = p4_tracker_comp.tracked8(ND, ·, ego)`.
   - Write a mutation-red test per control.
   - Run C-DERIV on navtrain against `raw/*_rows*.jsonl.xz`.
   - Smoke `transp_score_a1fix.py --metric-cache-root` on the human sub200 seam.
3. **Freeze → K entry n = 43 → launch** (§11). This is about 12 to 14 h wall on CPU.
4. **In parallel (Rule Zero, the honest replacement for the warp): the teacher re-roll levers** (`TFN2_EVIDENCE.md` §4).
   - Arms: positive-jerk limit 5.0, goal minimum speed 8 m/s, 5 Hz policy.
   - Roll with `../tf2b/tf2b_roll.py` (driverl-eval venv; the engine knobs need a small override in `build_teacher_rollouts.build_planner/build_controller`), then score with `tfn2_probe.py`.
   - About 2 h wall detached. Blocked here only by the 4-minute agent window.

## Notes
- **Missing DBs.** The 11 TF-2b logs' DBs copied from the pod on 11-12 are no longer under `E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits`. 362 of the 1,705 tokens are therefore unscorable locally; re-copy them for full coverage.
- **Reproducing the probe.**
  - Export `refe/`, `eval/` and `p5s_cubic.npz` from tip `be6567aa`.
  - Set `PYTHONPATH=<navsim devkit 3e8291b>;<w>/refe` and run `tfn2_probe.py --w <w> --sample 600 --shard i/2` (add `--extra` for RW105).
  - Then `tfn2_analyze.py --rows … --extra … --sel-rows-out …`, and `families_run.py --tanitad-repo <taniteval + stack/tanitad export>`.
- `tfn2_ep_rows.py` runs from `teacher_fidelity/` on the banked TF-2/TF-2b rows (the 1,705-row standstill table).

## Deletions (CLAUDE.md rule 4)
`F:/tanitad_scratch/tfn2/`, 387,577,085 bytes, deleted by literal path at the end. It held:
- the git exports `w/` (~29 MB) and `repo/` (~331 MB), re-creatable from `be6567aa`;
- the probe rows, which are banked as `.xz` and verified byte-equal by `cmp` before the delete;
- the analysis and family outputs, which are banked;
- `sel_rows.jsonl`, the smoke and the logs, which are re-creatable.

## Deliverable manifest (all in `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n2/`)
| file | content |
|---|---|
| `TFN2_EVIDENCE.md` | mechanism, ranking, families, blockers |
| `PREREG_TF_N2_DRAFT.md` | the draft pre-registration |
| `HANDOFF.md` | this file |
| `tfn2_probe.py` | `df7a1f5c` |
| `tfn2_analyze.py` | `b95b810b` |
| `tfn2_ep_rows.py` | `b072bb61` |
| `raw/probe_rows.jsonl.xz` | `6fb235cf` |
| `raw/extra_rows_RW105.jsonl.xz` | `3650abb0` |
| `raw/tfn2_probe_read.json` | `3f80daf2` |
| `raw/fam/fam_{C,R,W105,RW105,SEL,human}.json` | the families |
| `raw/{probe,extra}_s{0,1}.log` | run logs |
