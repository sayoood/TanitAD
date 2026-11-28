# PREREG_V8F_T1_FAMILIES: the TACTICAL and DISTANCE-KEEPING families of the T1-kin result, read on the banked dumps

**Written 2026-10-26, before any number of these rows has been computed on any T1 dump.** Claim row `H-V8F-T1-FAMILIES-1`. Zero training, zero
rollout: it re-reads dumps already banked.

## Why

`RESULT_V8F_T1KIN_HACC.md` (rows 27–28) reports two binding families as NOT COMPUTED:

* TACTICAL / STRATEGIC: "the T1 dumps carry no per-tick T code or S checkpoints";
* distance keeping: "`v8f_t1_eval` passes no lead track".

Both instruments already exist and are not re-implemented here:

* `taniteval.four_families.tactical_from_trajectory`, the canonical factored labeller (`tanitad.refs.refc_tactical.factor_from_kinematics`), which
  needs trajectories only;
* `taniteval.lead_metrics` on the banked lead block `raw/2026-10-22-dk/b1_eval_lead_block_k30.npz` (sha256 `7a879baa…3d9`), through the join guards
  of `code/v8f_dk_read.py`.

Both apply to an executed T1 track exactly as to a plan.

## Inputs

| config | dumps (`raw/…`) | role |
|---|---|---|
| HACC, causal speed bucket (the T1 headline) | `2026-10-26-t1kin-causalbucket/run/dumps/dumps_HACC_s{0,1}_m10_causal.npz` | **decisive** |
| HACC, logged bucket | `2026-10-26-t1kin-hacc/run/dumps/dumps_HACC_s{0,1}_m10.npz` | beside |
| strict (first frame held, causal bucket) | `2026-10-26-t1kin-strict/run/dumps/dumps_HACC_s{0,1}_m10_strict.npz` | beside |

* In every file, `cl` is the arm's executed track, `fl` is the P10 floor's executed track and `g` is the GT, all in the window-origin ego frame.
* There are 2,399 windows over 141 episodes.

## Rows (arm − P10, per seed; "worse" is the direction named)

| row | definition | worse if |
|---|---|---|
| **TAC-LAT** | the per-window indicator lat_label(executed) == lat_label(GT), from `tactical_from_trajectory`'s labeller (lateral 3-way) | the delta is < 0 |
| **TAC-LON** | the same for the longitudinal 3-way label | the delta is < 0 |
| **DK-VIOL** | LEAD-VIOL (`lead_metrics.lead_viol` / `paired_lead_viol`, at its registered at-risk denominator) | the delta is > 0 |
| **DK-HW** | \|headway(executed) − headway(human)\| on LEAD windows where both are finite | the delta is > 0 |
| **DK-TG** | \|time-gap(executed) − time-gap(human)\|, the same | the delta is > 0 |

* **Beside, never decisive:**
  * kappa and the per-class report of each tactical block;
  * min-TTC (censored at 30 s), with `n_closing`;
  * `lead_lost` per arm;
  * LEAD-PROGRESS;
  * the per-class confusion of the executed label against GT.
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, on the windows where both sides are finite (n reported).
  * The seed floor F per row (|mean_s0 − mean_s1| of the arm) rides beside every delta.
  * Lead rows are scored only on the block's LEAD windows. NO_LEAD, NOT_STRAIGHT and NO_LABEL are counted with their reasons and never scored as
    free flow.
* **Tier:** T1 (T1-kin: a self-action OPEN loop, never "closed loop").
  * ⚠️ The lead's logged future does not react to the ego. A track that closes on the lead is scored against where the lead WAS. That is the
    instrument's contract, and it is stated with every number.

## Controls (each must pass, or the read is REFUSED)

| control | must read |
|---|---|
| C1 identity | arm vs itself: every row's paired delta is exactly +0 [0, 0] |
| C2 the labeller on GT | GT vs GT: TAC-LAT and TAC-LON accuracy exactly 1.0 |
| C3 the join | the dump's `v0` equals the block's speed at every joined row to 1e-3 m/s; the block's `ts_rel_s` equals 0.2·(1..30) to 1e-6; no duplicate rows and no missing grid window (the three guards of `v8f_dk_read`) |
| C4 the kinematic contract | `distance_keeping` of the dump's GT path reproduces the block's `gt_*` columns to `v8f_dk_read.CONTRACT_MEDIAN_TOL_M` = 0.25 m (median \|d headway\|) |
| C5 the dumps | the ADE6 delta read from them equals the banked −0.7680 / −0.7336 to 1e-3 |
| C6 deliberate regression | the decisive HACC tracks rolled by one window over the whole set (window i scored with track i+1), compared with P10: TAC-LAT, TAC-LON and DK-HW must each read separated-worse at s0. If any does not, the instrument is blind to a known-bad arm, and the read is REFUSED |

## Outcomes, committed now, read PER FAMILY (never pooled)

* **TACTICAL:**
  * **NO WORSE**: neither TAC row is separated-worse at either seed.
  * **DEFICIT**: either TAC row is separated-worse at ≥ 1 seed.
* **DISTANCE KEEPING:**
  * **NO WORSE**: none of DK-VIOL, DK-HW or DK-TG is separated-worse at either seed.
  * **DEFICIT**: any of them is separated-worse at ≥ 1 seed.
* **REFUSED**: any control fails. The family is then reported REFUSED, never as a pass.

**What each outcome commits:**

* **Both NO WORSE:**
  * rows 27–28 of `RESULT_V8F_T1KIN_HACC.md` are filled;
  * the T1 headline may then say "tactical and distance keeping no worse than P10 at T1", and no more: no "dominance" on these families, since no
    better-bar was registered.
* **TACTICAL DEFICIT:**
  * the per-class confusion localises it: which class, and whether it is the at-speed curvature deficit of `RESULT_V8F_T1_LATCOND` showing as
    lateral-class flips;
  * then the lever is pre-registered before any training.
* **DISTANCE-KEEPING DEFICIT:** a lead-aware longitudinal lever is pre-registered before any training. The deficit is reported in the headline
  beside the ADE6 margin.

## Not covered here (named work items, not passes)

* **STRATEGIC at T1:** the dumps carry no per-tick S checkpoints.
* **The model's own T-code decision quality at T1:** the dumps carry no per-tick T logits.

Both need `v8f_t1_eval` re-run with a per-tick dump of the T and S heads. They remain NOT COMPUTED in the result, with this reason.

## Amendment 1 (2026-10-26): C6 replaced. ⚠️ Written AFTER the decisive rows were computed and seen, so the amended read is POST-DATA

**What failed.** C6 as written failed: the decisive HACC tracks rolled by one window, compared with P10, read *better* on TAC-LAT
(+0.130) and TAC-LON (+0.112) and n.s. on DK-HW at s0. The read was therefore REFUSED, and that refusal stays in the record. The
reasons are measured in `T1KF_READ.json`:

1. **The roll is a near-copy, not a regression.** Consecutive grid windows are 0.8 s apart on a 6 s horizon, so the manoeuvre class
   barely moves. Rolled − unrolled HACC: TAC-LAT −0.007 (n.s.), TAC-LON −0.019, DK-HW +0.15 (n.s.).
2. **The comparator could never show it.** P10 is a weaker tactical arm than even the rolled HACC: TAC-LAT accuracy is 0.593 for P10,
   0.723 for the rolled HACC and 0.729 for HACC.

So the control was written without computing what it tests: the class of `R74` (N-3 of `PREREG_V8F_T1_LATCOND`), a second time
today.

**The disclosure.** This amendment is written after the reader reported the decisive HACC − P10 rows, as data, beside the refusal.
* **What does not change:** the rows, the bars, the comparator, the estimator and the outcome table. Nothing in C6A depends on the
  values of those rows.
* **But the amended read is not blind,** and it is reported as **POST-DATA (Amendment 1)**, never as a blind pre-registered pass.

**C6A, which replaces C6:**
* **The known-bad arm:** for every window i, take the decisive HACC executed track of a window from a **different episode** whose
  dump `v0` is nearest to window i's. Ties go to the lowest index.
* **Why speed-matched:** it keeps the units guard (`check_pred`'s first-step ratio) honest, and it degrades only the fit to the scene:
  the right speed, the wrong road and the wrong lead.
* **Compared with:** the unshuffled HACC tracks, i.e. the arm itself. The control asks whether the instrument sees a known
  degradation of this arm. It does not ask whether that degradation falls below P10.
* **Must read, at s0:** TAC-LAT, TAC-LON and DK-HW each separated-worse. s1 is reported beside.
* **C6 as written** is reported beside as FAILED, superseded by C6A.
* **If C6A fails,** the read stays REFUSED, and the instrument's blindness becomes the work item.

Nothing else moves.
