# P2, Amendment 1: what P2's result decides about P6 (pre-registration, PRE-DATA)

> Written 2026-10-11, before any P2 data exists. It replaces ONE clause of `PREREG_P2.md` ("Next arm, by outcome",
> the NOT PROVEN / REFUTED bullet) and adds one read. Everything else in `PREREG_P2.md` stands: the lever, the
> trainable set, the training arguments, the launch gate's criteria and the stage-2 rule.
> It is landed in git before the first P2 training step and before any P2 score is opened.

## 0. No P2 data exists

- The old pod produced **no P2 gate result, no P2 log and no P2 checkpoint** (INHERITED from the Master Mind's brief of
  2026-10-11; the loss itself is recorded in `STOP_CONSEQUENCES_2026-10-11.md`).
- Nothing below was set after seeing a P2 number. There is none.

## 1. Why this amendment

`PREREG_P2.md` ends: "**NOT PROVEN or REFUTED:** P6 still runs, the paper's full chain from scratch."
That clause made P6 unconditional. The PI has ruled otherwise (2026-10-11, verbatim):

> *"if necessary we can do a full run on a new 7a40, only if we have confidence that the result will be better and
> achieve our goals"* (sic: "7a40")

So a full P6 run on a new pod needs **confidence**, and confidence needs a rule written before the data. Section 4 is that
rule. It replaces the clause above. **Rule Zero still binds:** a NO-GO below names the next lever and its first
experiment; it is never a stopping point.

## 2. What changes in how P2 is run (mechanics, not levers)

- The pod is new. `refe/p2r_pod.sh` is the single chain (`eval/P2R_POD_RUNBOOK.md`).
- Its training arguments are **byte-identical** to `refe/p2_chain.sh`'s (`refe/test_p2r_chain.py` asserts it).
- The PDM labels are **recomputed to completion** (174,912 served sets; the chain refuses to train on fewer).
  The lost labels cannot be diffed against the new ones: same code and same sets, no reference copy (UNVERIFIED equality).
- G3 of the launch gate runs on a small bank with some labels **withheld** (`refe/p2r_gate_prep.py`). After a complete
  relabel every served set carries a label, so on the full bank G3's mutation arm would be inert. The criteria of
  G1-G4 are unchanged; only the bank they are given differs.
- Outputs are pushed to a private HF repo during and after the run. A result exists when the remote listing shows
  `model_final.pt` at the local size with the sha256 in `summary.json`.

## 3. Reads, in order

| read | tokens | P2 against | decides | status |
|---|---|---|---|---|
| **R1** stage 2 | 923, 43 logs | deployed, 85.61 PDMS | ADOPT / REFUTED / NOT PROVEN, rule in `PREREG_P2.md` | **unchanged** |
| **R2** stage 3 | all 12,146, 136 logs | **SFT-4 arm B u400, 88.25 PDMS** | P6 GO / NO-GO (section 4) | **new** |
| R2b | same 12,146 | deployed, 84.46 PDMS | nothing; continuity with SFT-4's +3.79 [+3.05, +4.54] | reported |

- **R1 runs first** (cheap). **R2 runs unless R1 is REFUTED.** R1 REFUTED means P2 is below the deployed system, while
  u400 is above it (+3.04 [+1.21, +4.87] on the same 923 tokens, `RESULT_SFT4.md`), so P6 is NO-GO without R2.
- **Estimator (all reads):** paired log-cluster bootstrap, 10,000 resamples, seed 20260927, cluster unit `log_name`.
  It is the estimator of `RESULT_SFT4.md`. It answers *another draw of logs* and nothing else.
- **Instrument precondition for R2, checked BEFORE P2's scores are opened.** `eval/stage2_sft.py` pairs against fixed
  deployed-run constants (`BASE_CSV`, `BASE_SEAM`), so R2 needs a baseline option or a sibling script. It must show:
  - pairing u400 with itself reads **+0.00 [0.00, 0.00]** on 12,146 tokens;
  - swapping the two arms flips the sign exactly;
  - the token file md5 equals `3a8b3653…` (`RESULT_SFT4.md`) and u400's per-token table md5 is recorded.
- **Handling of the 36 C1 tokens (4 logs):** as declared in the full-navtest read of `RESULT_SFT4.md` (its Deviation 1; reported, not gated).
- **Cost on the dev box (MEASURED, `RESULT_SFT4.md`, u400):** seam 17,497 s plus harness 1,336 s for 12,146 tokens.
- **Token reuse, stated:** the 923 tokens already had two scorer reads (SFT-3 arm A, SFT-4 arm B). R1 is a third.
  R2's gating weight rests on the 12,146 tokens, not on the 923.

## 4. The P6 GO / NO-GO rule (committed now; both outcomes)

Let **Δ = PDMS(P2) − PDMS(u400)**, ×100, on the 12,146 tokens, with the 95 % interval [lo, hi] above.
A PDMS sub-score is **adverse** if the paired interval of (P2 − u400) lies entirely below 0. The sub-scores are the
six the harness reports: NC, DAC, EP, TTC, C, DDC.

| outcome | P6 | what happens next |
|---|---|---|
| **lo > 0 AND no sub-score adverse** | **GO** | finalise and land `PREREG_P6.md` (a DRAFT whose own precondition is "P2 has a verdict", plus the PI's P6-T/C/N/Y decisions); the **spend is the PI's call**, this only supplies the confidence |
| lo > 0 AND some sub-score adverse | **NO-GO** | name the adverse component; next lever = a scorer-side fix aimed at that component, run as the cheaper SFT-4R-style arm first |
| lo ≤ 0 ≤ hi (**NOT PROVEN**) | **NO-GO** | next lever = **SFT-4R** (scorer-only on COMPLETE labels, `PREREG_SFT4R.md`), which isolates the label-completeness confound P2 cannot; then a training-seed replicate of the better arm |
| hi < 0 (**P2 below u400**) | **NO-GO** | the un-detached context or the joint objective hurt; next lever = SFT-4R as above, plus P2's training log (score loss, LoRA movement, held-out ranking) to say which half failed |
| R1 REFUTED | **NO-GO** | as the row above; R2 not run |
| P2 does not finish, or a launch gate fails | **NO-GO by default** | name the blocker (a gate FAIL is a launch blocker, never a footnote) and re-run P2; "no result" is not a result |

- **GO is a direction decision for the next run. It is not a lever claim** (section 6).
- A GO needs the PI's explicit go-ahead before any pod is created. This amendment does not authorise spend.
- Aim of P6, unchanged: the paper's chain from scratch, aimed above the HUMAN row (94.55 PDMS, `RESULT_NAVTEST_FINAL.md`).

## 5. Four metric families (reported per family, never pooled; same 136-log estimator)

| family | reported | status |
|---|---|---|
| **longitudinal** | speed, along-track and acceleration MAE and bias; ego-progress ratio | MEASURED by `families6`; **distance-keeping UNAVAILABLE** (no lead-agent track in the seam) |
| **lateral** | heading, cross-track, final cross-track, curvature and yaw-rate MAE | MEASURED by `families6` |
| **tactical** | lateral, longitudinal and 5-way decision accuracy and kappa | MEASURED; rows carry no interval, so **reported, not tested** |
| **strategic** | – | **PAUSED** by the PI (2026-09-19, `PI_DECISION_QUEUE.md` item 25) and **UNAVAILABLE in NAVSIM**; reported as such, never as a pass |

- The 21 longitudinal and lateral components are compared P2 against u400 and against deployed, and any that
  separates adversely is reported by name. **They do not gate R2**: 27 simultaneous comparisons would give a large
  false NO-GO rate under no effect. The six sub-scores gate because the PI's wording is about the score.
- ADE is not a family and is not reported as a result.

## 6. What a GO does not mean (confounds, stated before the data)

1. **Labels.** u400 trained on **146,185 of 174,912** PDM labels (83.6 %; `PREREG_SFT4.md` Amendment 4). P2 trains on
   all 174,912. Δ therefore mixes three differences: complete labels, the un-detached visual context, and joint
   training of the LoRA. One arm cannot separate them. SFT-4R is the arm that can.
2. **One seed each (H-ESTIM-SEED-1).** The interval is blind to training variance, and a separated interval from
   single-seed arms is necessary, not sufficient. No replicate arm exists for either. On the tiny rig a replicate of
   identical flags produced "separated" differences on 4 of 42 cells (CLAUDE.md, `A0b_replicate`); this rig's own
   floor is unmeasured.
3. **u400 is update 400 of 2,335**, a non-final checkpoint that stage 1 never selected (the 923-token read's
   deviations 1 and 2, `RESULT_SFT4.md`). P2 reports its final checkpoint, unselected.
4. **Inference is deterministic**, so there is no inference-seed variance on either side.
5. **Multiple sub-scores.** Six sub-scores gate the GO. Under no effect some chance of an adverse-separated one exists,
   so a NO-GO can be a false negative. It costs one cheaper arm, not the programme.

## 7. Tier and evidence

- **Tier T1** (the model conditioned on its own actions): NAVSIM v1.1 PDMS, an ego pseudo-simulation of an open-loop
  plan against logged agents. Per the open-versus-closed-loop ruling this is **self-action open loop**, not closed
  loop, and not a driving-in-simulation claim.
- Evidence classes: every number above taken from `RESULT_SFT4.md` or `RESULT_NAVTEST_FINAL.md` is **MEASURED (ours)**.
  The PI quote is from the Master Mind's brief of 2026-10-11 (**INHERITED**; the original ruling text is not in the
  repo). The relabel's equality with the lost labels is **UNVERIFIED**.
