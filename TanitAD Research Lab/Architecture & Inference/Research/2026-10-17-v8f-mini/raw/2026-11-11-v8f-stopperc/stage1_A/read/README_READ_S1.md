# STOPPERC stage-1 READ package, arm A (built 2026-11-12 evening, BEFORE any stage-1 arm exists)

Card `theory/stopperc/PREREG_V8F_STOPPERC.md` (FROZEN) section 5 (rules and outcome table) and section 8 (Q3: two replicates); `AMENDMENT_3_A_READINGS.md` (section 4: NULL reuse on model tensors). Code of record of the arms: `3e264bbf3fa2f6f17bea448de09aa972faaf7725` (child of `715ef8fc`). Tier **T1-kin** (self-action open loop, m = 10, causal speed bucket, dead-reckoned nav, logged scene; deterministic planner), eval-141, per seed. CPU, zero training, **no GPU** (every job sets `CUDA_VISIBLE_DEVICES=-1`, as STOPBAL's own jobs do, so `devbox_gpu.lock` is not taken). Estimator: `taniteval.ci` paired episode-cluster bootstrap, B 2,000, seed 0 ("another draw of episodes" only; "another training run" is the floor).

This package re-points the first dose's read (`theory/stopbal/read/`, `dose2/null_banks.py`; imported by path, **not edited**). New files only, all in this folder.

## Run steps for the Master Mind (once the chain has banked the arms)
Inputs: banks `C:/Users/Admin/t2v8f_pulls/stopperc_{NULL_s0,NULL_rep2,A_s0,A_s1}/` (+ `stopperc_NULL_s1`, `stopperc_NULL_rep` only if the reuse rule failed) and `C:/Users/Admin/stopperc_state/NULL_REUSE.json`. Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`.
0. `python -I stopperc_s1_read.py refs`: re-asserts the reuse state and every stage-1 bank (rc 0 clean, rc 2 with one REFUSED line each). `python -I stopperc_s1_read.py tags` prints the arm list the next two steps will use.
1. `python -I make_tree_s1.py`: `git archive` (autocrlf off) of `3e264bbf` from `C:/Users/Admin/tanitad-push` to `F:/TanitAD-scratch/stopperc_read/tree` (about 350 MB, ~10 s); asserts the parent is `715ef8fc`, the diff over the exported paths is exactly four files, the instrument files equal the STOPBAL tree's pins and `v8f_mini.py` differs. Writes `TREE_PINS_S1.json`.
2. `powershell -File launch_wmi_s1.ps1 job_dumps_s1.cmd` (detached, `< NUL`): T1-kin dumps of the stage-1 arms, 2 threads, ~5 min per arm ESTIMATED from the dose-2 dumps (301.5 s MEASURED for BAL-G2_s0). Done marker `DUMPS_S1.done` (`rc=0`). Each arm is REFUSED, nothing launched, if its bank is not whole (section "What refuses").
3. `powershell -File launch_wmi_s1.ps1 job_stopdec_s1.cmd`: the first dose's `stopdec_arm.py` unchanged, on the new tree (env `STOPBAL_TREE`, `STOPBAL_DUMPS`); `out/stopdec/stopdec_<tag>.npz` + `STOPDEC_<tag>.json`. Marker `STOPDEC_S1.done`. Run it after step 2 has `rc=0`.
4. `python -I stopperc_s1_read.py read --out RESULT_STOPPERC_S1.json --verdict VERDICT_STOPPERC_S1.md` (about 3 to 5 minutes CPU). Prints `WROTE ... outcome` and every control problem. The verdict file is generated from the JSON; the Master Mind adds the context.
Dumps of every arm must come from ONE environment (thread drift, OTBAND trap): the reused `BAL-NULL_s1` / `BAL-NULL_rep` dumps were made by the first dose's driver at the same 2 threads (and the OFF-checkpoint identity below shows the tree move does not change them).

## NULL reuse (AMENDMENT_3 section 4)
`NULL_REUSE.json` (written by the chain's `reuse` step) **EQUAL** => `NULL_s1 := BAL-NULL_s1`, `NULL_rep := BAL-NULL_rep` (banks `stopbal_BAL-NULL_*`, dumps in `F:/TanitAD-scratch/stopbal_read/t1`, STOPDEC in `theory/stopbal/read/out/stopdec`, all banked). **DIFFERS** => the `stopperc_NULL_s1`, `stopperc_NULL_rep` banks. **Absent, UNKNOWN, unreadable, not "computed", or not matching the banked NULL_s0 checkpoint / BAL-NULL_s0 pin** => the read REFUSES (rc 2, `outcome: REFUSED` written), never a default. EQUAL additionally re-asserts the two reused banks by checkpoint sha256 against their manifests and RUN.done, and BAL-NULL_s0 against its pin `5169ea95...aa2fdf`. NULL_s0 and NULL_rep2 are always the stage-1 runs.

## Dump tree: it MUST move to 3e264bbf, and the OFF checkpoints are bit-identical on both (MEASURED, see Verification)
Arm A's checkpoint `cfg` carries `stop_aux` / `w_stop_aux` and its state dict `stop_head.*` / `stop_pres_head.*`. The STOPBAL tree (`715ef8fc`) does not know them (`V8FMiniConfig(**ck["cfg"])` raises, strict load fails), so dumps and STOPDEC of A come from the `3e264bbf` tree. The head is built last, computed only in the training forward (`use_labels`), read by nothing in `forward_infer`: the planner at inference is the NULL planner plus the adapter A's loss shaped.

## The frozen rules, copied verbatim (card section 5 and section 8)
> **Rules (committed before any run).** A delta is the arm minus the same-seed NULL. A cell's floor is |NULL_rep − NULL_s0|. A **gain** is separated on the better side AND above the floor; a **cost** is separated on the worse side AND above the floor. Both seeds must agree, else SEEDS-DISAGREE (a third seed, `--seed 10/11`).
> * **P (PRIMARY):** tactical ADE (knots 11-30) on the 117 stop-point windows is a gain at both seeds. Reported beside: the fraction of the 1.5 m ceiling recovered.
> * **G-AUROC:** STOPCUE AUROC of p(STOP_POINT) is a gain at both seeds. The lon-STOP AUROC is not a cost.
> * **G-FP:** the false-positive rate at 0.5 on non-stop windows is not a cost.
> * **NR:** all-window tactical ADE is not a cost at either seed. Four families per band, every cost named; TACTICAL includes STOPDEC's hit and false-positive rates; STRATEGIC PAUSED.

> **Q3 (section 8):** stage 1 adds `NULL_rep2` (`--seed 3`, +40 min). **A cell's floor is max(|NULL_rep − NULL_s0|, |NULL_rep2 − NULL_s0|).**

| outcome | predicate | next step (committed now) |
|---|---|---|
| **PERC-ADOPT** | P, G-AUROC, G-FP, NR at both seeds | the flag joins the recipe; then the stacked arm, and BAL-L at W 4.6991 re-read on the new base, because a better ROC makes the weighting lever's operating point worth moving |
| **PERC-AUROC-ONLY** | G-AUROC + NR, P fails | STOPORC and STOPDEC on the arm, zero training; then BAL-L stacked on it |
| **PERC-P-OTHER-PATH** | P + NR, G-AUROC fails | third seed, plus the CONSORC decomposition |
| **PERC-FP-COSTS** | G-AUROC, but G-FP fails | report; that arm plus the training-window threshold rule of STOPTAU, under its own card |
| **PERC-NULL** | neither P nor G-AUROC | the lever is eliminated with its row. Next: L-DAG if S0-L passed, otherwise the 416x1024 source (Q5) |
| **PERC-REGRESSES** | NR fails at either seed | not adopted. For A, W is halved and the arm re-read; for R, the row is reported |

## Reading choices the card does not fix (numbered; none moves a bar; the Master Mind confirms or replaces before the run, as for STOPBAL r1-r6)
* **s1 (G-AUROC windows).** "STOPCUE AUROC" = `theory/stopcue/stopcue_read.py`'s definition: p(STOP_POINT) (goal code 6) over the **has-label windows (2,365), positives `stop_hit` (131)**; lon-STOP AUROC = the lon STOP probability against the regime stop `v_end_gt < 0.5` (STOPCUE's `auroc_lon_stop_vs_regime_stop`). The first dose's M-gate AUROC (has-and-present windows, 117 stop-point positives) is read **beside** (`G_AUROC_goal_beside_first_dose_windows`), with its own floor; it does not enter the outcome. The two sets may disagree in size only; the read prints both.
* **s2 (G-AUROC gain and cost).** The AUROC delta (arm − NULL) has an interval from `paired_stat_cluster_bootstrap` (AUROC is not a mean of per-window scalars); gain = separated, positive, |delta| above the floor; the AUROC floor = max of the two replicate AUROC gaps. The lon-STOP AUROC is a cost when separated, negative and above its own floor; **a lon-STOP cost makes G-AUROC fail** ("the lon-STOP AUROC is not a cost" read as part of the gate).
* **s3 (G-FP).** The false-positive rate = the share of the non-stop has-and-present windows (2,234) on which p(STOP_POINT) >= 0.5 (the first dose's M-gate fpr, `stopbal_read` r3). A cost = separated, higher, above the max-of-two fpr floor, at either seed. The hit rate (117 windows) is read beside with its floor; it is not a gate.
* **s4 (headline precedence where several predicates hold, as STOPBAL r4; all that hold are printed with their triggers).** REGRESSES > ADOPT > P-OTHER-PATH > FP-COSTS > AUROC-ONLY > NULL. The only literal overlaps: AUROC-ONLY and FP-COSTS both hold when G-AUROC gains, NR is fine, P fails and G-FP is a cost (FP-COSTS is the more specific and is the headline; both triggers print); any predicate and REGRESSES (REGRESSES first, as the card's "NR fails at either seed"). "Holds" means at BOTH seeds, "fails" at NEITHER seed; a mixture that satisfies no predicate is **SEEDS-DISAGREE** (the card's rule), with the per-seed flags printed.
* **s5 (floors are read at seed 0 and applied to both seeds)**, as STOPBAL r2: both replicates (`seed 2`, `seed 3`) are s0-init runs. NULL_rep and NULL_rep2 are the same-window arms; floors per cell, per window set, per decision, per AUROC.
* **s6 (P ceiling beside).** `fraction_of_1p5m_recovered = -delta_P / 1.5`, and beside it the fraction of the STOPORC F-GT point of that seed (1.5392 / 1.4768 m, card section 8).
* **s7 (cells).** Both window sets (stop-point 117, all 2,399) x (operative knots 1-10, tactical 11-30) x the 11 metrics (ADE, FDE, speed, along, accel, headway and time-gap error, cross-track, heading, yaw rate, curvature) + the two TAC accuracies = 48 cells per seed (STOPBAL's). Every cost and gain is named in `named_costs` / `named_gains`; REGRESSES reads NR only, as the card says.
* **s8 (controls, fail closed).** 1 (lever read-back: NULL arms 1.0 / 1.0 and no stop key or flag; A `levers.stop_aux == "on"`, `w_stop_aux` equals the bank argv's `--w-stop-aux`, `--stop-aux on` in it), 2 (T live terms finite, no dead T head, `groups_grad.T` > 0; first dose's `check_selfcheck`), 3 (STOPDEC controls exactly 0.0 / in range, window counts 2,399 / 2,365 / 117, STOPCUE positives 131, planted AUROC 1.0 / 0.5 / 0.0, decisions equal the npz's, window sets identical across arms and equal the banked DAG-OACC windows), 4 (NULL_s0, NULL_rep, NULL_rep2 present and the reuse state clean), 6 (bank whole: ckpt sha256 = manifest = RUN.done, tag, commit `3e264bbf`; dump done file names the bank's sha256 and 141 dumps). A control that fails or cannot be read makes the outcome INVALID. Control 5 (the chain's G-SUITE / pre-launch suite) is NOT checked here. STRATEGIC: PAUSED (PI item 25), always printed.

## What refuses
Reuse state absent / UNKNOWN / unreadable / inconsistent; NULL_s0, NULL_rep or NULL_rep2 missing (no floor: no gain or cost is read); any stage-1 bank missing a file, with a ckpt sha256 differing from its manifest or RUN.done, built at another commit, or with a selfcheck that fails controls 1 / 2 (arm A built with the head off; W differing from the argv; a NULL arm carrying a stop key); a dump whose done file does not name the bank's checkpoint; any window set or model-free array differing from the banked DAG-OACC dump; a lever-less `A` at a seed whose NULL is absent (that seed reads NOT-YET and decides nothing).

## Verification (2026-11-12, dev box, CPU)
See `VERIFICATION_S1.md` (written with the numbers: tests, mutation, OFF-checkpoint tree identity, stand-in smoke read). Pins: `READ_PINS_S1.sha256`.

## NOT verified here
* Nothing has run on a real stage-1 bank, ON checkpoint or NULL_REUSE.json: the key layout of the stage-1 `selfcheck.json` / `BANK_MANIFEST.json` / `RUN.done` is read from the STOPBAL banks and from `stopperc_check.py`'s source (`levers.stop_aux`, `levers.w_stop_aux`), not from a real stage-1 bank. The manifest `tag` is accepted as `A_s0` or `stopperc_A_s0`.
* `V8FMiniConfig(**ck["cfg"])` + strict load of a real ON checkpoint on the `3e264bbf` tree (the chain's G-LIVE8 builds it, the unit tests of the checker do not use a real one).
* That the STOPDEC instrument's `forward_topk` shim attaches on the `3e264bbf` module for an ON model: attached and run for an OFF model in the smoke only if `VERIFICATION_S1.md` says so.
* The 12 h-class wall time of the stage-1 dumps (about 5 min per arm ESTIMATED).
