* Amendments: one, below.

## Amendment 1 (`A1-2026-10-28`; appended before the smoke and before any rollout of this experiment)

Written 2026-10-28 by the Master Mind. By then the instrument had been built and `mlp_ego_c` fitted: sha256 `799f65ed…`, controls C-FIT,
C-CAUSAL, C-COVER, C-DISJ and C-PARITY PASS, TRAIN hold-out 0.2793 SD units, which decides nothing. No rollout of this experiment had run,
and no T1-kin substitution number existed. The amendment adds one guard and settles four readings of the text. No bar, row, arm,
permutation or table entry above changes, and every tolerance stays as written.

1. **Guard G-VIOL (added).**
   * *Reason.* `RESULT_V8F_SPEEDSUB.md` section 4.2 (landed `af2dee4a`) found, post-hoc, that the plan rule-violation rate rose under `ego`
     at both seeds, separated-worse, while ADE6 fell.
   * *Where the measure comes from.* The T1-kin tool carries no violation measure: four probe locations were searched (the tool, the dump
     keys, the tool's JSON, `taniteval/tools/t1_eval.py`), and the build notes list them. The reader computes the measure from the dumps
     with the T0 eval's own rule function (`taniteval/taniteval/v8f_consistency.py` `violations`; rules SPEED / STOP / TURN / LEAD), against
     the GT constraints of `eval_context`.
   * *The row.* Row VIOL-ANY is the per-window any-rule violation of the executed `cl` path. It is paired `egoc − identity`, per run, with
     the estimator of section 3.
   * **G-VIOL: `VIOL-ANY` is not separated-worse at ≥ 3 of the 4 runs.** A firing is a guard firing, so the HARMS row of section 6 reads
     *"a guard fires (G-ADE, G-WORSE, G-NEW or G-VIOL)"*.
   * *Beside.* VIOL-ANY-OL (the tick-0 plan, comparable with SPEEDSUB's T0 finding) is reported and decides nothing.
   * *Known-value check, MEASURED on the banked runs.* DAG-OACC CTRL `cl` 0.3618 (s0) / 0.3760 (s1); P10 floor 0.7999 / 0.7999. The row
     separates the floor from the model by +0.4381 [+0.3953, +0.4821] and +0.4239 [+0.3792, +0.4706]
     (`raw/2026-10-28-speedsub-t1-build/probes/probe_viol_ctrl_s0.json`).
2. **"The tick frame" (section 2) is the window's NOW row j** (the tool's `st.j`), the row at which the tool builds its own causal `limit`.
   C-CAUSAL found 0 mismatches against the tool's `build_item --limit causal` at a_t = 0, 1 and 2.
3. **G-NEW is read per seed and must hold at both seeds.** A row fires when, at each CTRL seed, `identity − P10` is not separated-worse and
   `egoc − P10` is separated-worse. The other reading of the sentence is reported beside and decides nothing.
4. **The smoke's controls** (section 4: `--max-windows 200`, CTRL s0 `identity` and `egoc`) are:
   * K-NONSPEED;
   * K-REC;
   * K-REACH (`egoc`, ≥ 99 % of (window, tick));
   * K-SHA;
   * K-ALIGN (the two smoke rollouts carry the same episode ids and windows).

   Any failure stops the chain before a full rollout.
5. **C-FIT's tolerance is the float32 class written in section 2:** 1e-5 m/s absolute on hold-out predictions. The fit read 9.412e-6
   (PASS); the margin is thin and is disclosed.
