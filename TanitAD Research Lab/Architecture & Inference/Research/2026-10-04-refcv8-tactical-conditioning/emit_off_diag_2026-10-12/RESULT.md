# RESULT: why V-R8 drives turns worse than V0. The SELECTOR, trained by the listwise loss, is the defect.

2026-10-12, Master Mind. **POST-HOC and DESCRIPTIVE.** This is not a ladder row or a ladder verdict. It is TINY RUNG
(resnet34, 3,000 steps, one training seed per arm). It reads open-loop plans on the eval139 grid (1,112 windows /
139 episodes, K=8): the ladder's own grid, the same open-loop rows the ladder scores, so it is NOT closed-loop.
Estimator: the paired episode-cluster bootstrap over the 139 episodes (`ladder_score.make_draws`), on both sampler
seeds 0 and 1. That interval answers *"would another draw of episodes say this?"*. It does **not** answer *"would
another training run say this?"* (H-ESTIM-SEED-1). The training-replicate floor is quoted beside each gap below.
Host: dev box RTX 4060, `--batch 4` for every arm (PREREG amendment 2). Checkpoints are md5-verified copies of
Thor's.

Pre-registration: `PREREG_EMIT_OFF.md`. Q1 and Q2 are pre-registered there. **Q3 was NOT pre-registered.** It was run
after Q2's numbers and is reported as a post-hoc decomposition.

## The short version

1. **Q1 (pre-registered): emission is not the cause. Training is (OUTCOME B, both arms).** Turning off the 32 extra
   candidates at inference (`emit_off`) closes only **16 / 18 %** (V-R8) and **10 / 14 %** (V-R8d) of the cross-track
   gap to V0 on sampler seeds 0 / 1. None of these is separated.
2. **The gap is on turns (zero-training stratification).** On the 107 GT-turn windows V-R8's cross-track at 6 s is
   **+4.4 / +4.6 m** worse than V0's, and its ADE is **+1.8 / +1.8 m** worse, all separated. GT-straight windows show
   no separated difference. The turn direction is as good as V0's (`dir_correct_turn` not worse), so the turns are
   mis-SHAPED, not mis-directed. The gap sits where v7 and v9 nav AGREE (98 % of it), so the nav source is ruled out.
3. **Q2 (pre-registered): SELECTION, on both arms and both seeds.** V-R8's candidate bank is *better* than V0's on
   turns: the best base-fan candidate is closer to the human path (cross **−0.49 / −0.56 m**, ADE **−0.16 / −0.14 m**,
   separated better). The random-candidate control is also better for V-R8, because its fan is tighter, so a wide fan
   cannot be what makes its oracle look good: a wider fan would *help* an oracle, and V0's fan is the wider one. The
   bank holds the good turn. **The selector does not pick it.**
4. **Q3 (post-hoc): the selector's training target is part of the problem, and the selector's failure to learn even
   that target is the bigger part.** V-R8 trains its selector with the listwise soft-target loss, which REPLACES the
   selection CE (`refc_v3_train.py:5658-5661`). That target scores a candidate on only two things, its speed profile
   (cumulative arc length) and its terminal heading (`refcv8_conditioning.listwise_selection_loss`). **Lateral
   position is not in it.** On turns, cross-track at 6 s:

   | plan | V-R8 s0 / s1 (m) |
   |---|---|
   | best candidate by ADE (oracle-all) | 2.82 / 2.83 |
   | best candidate by the LISTWISE TARGET (a perfect learner of it) | 5.92 / 6.38 |
   | what V-R8's selector actually picks | 14.98 / 15.14 |

   * The target alone costs **+3.1 / +3.6 m** (separated). It is lateral-blind.
   * The selector then misses its own target by another **−9.1 / −8.8 m** (separated). The target is very flat: its
     probability mass spreads over a median of **~33 of 149 rows** (V-R8; 37 for V-R8d). A score trained toward that
     is nearly flat, and its argmax lands far from the target's best row. V-R8 picks one of the 32 extra rows **66 %**
     of the time; the target's best row is an extra row only **36–39 %** of the time.
   * For comparison, V0's selector is trained by the hard selection CE (one target row, the lowest-ADE one). It is
     not good on turns either (regret **+7.2 m** cross vs its oracle), but it is far better than V-R8's (**+12.1 /
     +12.3 m**).

**Verdict for the lever ranking.** The turn regression that makes V-R8 lose to V0 is not in the decoder, the bank, the
emission or the nav. It is in the selector's training: the listwise term replaces the hard CE with a flat,
lateral-blind target. ⭐ **The next lever is to train the selector with the CE again. The cheapest arm is V-R8 with
`--w-r8-listwise 0.0`, ONE flag, no code change, because the CE then trains the selector automatically.** It needs
one tiny-rig training run on Thor. That is a 12th run and the PI capped the ladder at 11, so it waits on the PI
(blocker named in `PREREG_VR8_NOLW.md`).

## Noise floor (training replicate, Thor batch 8, same grid)

V0r vs V0 (same recipe, training seed 1) is itself "separated worse": cross all **+0.16 to +0.19 m**, turns
**+0.83 m**, ADE turns **+0.25 to +0.33 m**. V-R8's turn cross gap (+4.4 / +4.6 m) is about **5×** that floor, and the
selector regret gap in Q3 is larger still. ⚠ These are separated intervals from ONE training run per arm. They rank
levers; they do not prove one (H-ESTIM-SEED-1).

## Four families (Q1, V-R8 `emit_off` vs V-R8 `base` vs V0 `base`, seed 0 / seed 1)

| family | metric | V0 | V-R8 | V-R8 emit_off | note |
|---|---|---|---|---|---|
| LATERAL | cross_abs_6s (m) | 3.147 / 3.114 | 3.867 / 3.810 | 3.750 / 3.685 | emit_off still separated WORSE than V0 |
| LATERAL | yaw_mae_0_2s | 0.0223 / 0.0225 | 0.0313 / 0.0308 | 0.0307 / 0.0299 | same |
| LATERAL | curv_mae_0_2s | 0.0041 / 0.0041 | 0.0058 / 0.0057 | 0.0058 / 0.0056 | same |
| LONGITUDINAL | speed_mae_0_2s (m/s) | 0.300 / 0.297 | 0.275 / 0.276 | 0.271 / 0.271 | R8 BETTER than V0 |
| LONGITUDINAL | speed_mae_2_6s (m/s) | 1.345 / 1.346 | 1.236 / 1.232 | 1.220 / 1.225 | R8 BETTER than V0 |
| LONGITUDINAL | ceil_violation | 0.0522 / 0.0522 | 0.0603 / 0.0603 | 0.0603 / 0.0603 | R8 worse |
| TACTICAL | lat3_f1_v9 | 0.315 / 0.315 | 0.762 / 0.762 | 0.762 / 0.762 | the R8 tactical head is far better |
| TACTICAL | dir_correct_turn | 0.664 / 0.673 | 0.673 / 0.682 | 0.701 / 0.720 | not worse |
| STRATEGIC | — | — | — | — | no strategic instrument on this rig (no route label at the rung); n/a, not dropped silently |

Distance-keeping (headway / TTC) is not in `ladder_score.METRICS` at this rung; it is named as absent here, per family
rule 5. Full per-metric tables: `raw/V-R8_eo_vs_own.json`, `raw/V-R8_eo_vs_V0.json`, `raw/V-R8_base_vs_V0.json` (and
the V-R8d set).

⭐ Note what the families say together: R8 is **better** on speed and on the tactical decision, and **worse** only on
lateral shape on turns. That is consistent with a defect in WHICH candidate is driven, not in what the model knows.

## Artifacts

| what | where |
|---|---|
| pre-registration (Q1, Q2) | `PREREG_EMIT_OFF.md` (this dir) |
| the proposed next arm + its pre-registration | `PREREG_VR8_NOLW.md`, `raw/V-R8-noLW.argv.json` (this dir) |
| instrument (`ladder_eval.py` + the `emit_off` row + `--save-cands`) | `code/ladder_eval_diag.py` |
| Q2 variant builder / Q3 listwise oracle / stratified pair | `code/oracle_variants.py`, `code/listwise_oracle.py`, `code/strata_pair.py` |
| Q1 readouts | `raw/*_eo_vs_*.json`, `raw/*_base_vs_V0.json`, `raw/nav_strata.txt` |
| Q2 / Q3 readouts (every table above) | `raw/q2q3/*.txt` |
| eval dumps (batch 4) and candidate captures | dev box `E:/lgdiag/eval_b4/`, `E:/lgdiag/eval_cands/` (NOT in the repo: ~25 MB of npz; the readouts above are the quotable layer) |
| checkpoints used | dev box `E:/lgdiag/{V0,V-R8,V-R8d}/model_final.pt` (md5 03548ad7 / 1b12e8eb / ac1cef23 = Thor's) |

Controls that make Q3's oracle quotable: `listwise_oracle.py` recomputes the target cost and checks it against the
REAL loss function (`listwise_selection_loss` fed `score = −cost` must return the target's entropy). The match is
≤ 4.8e-7 on every batch. A mutated cost with a lateral term added must break that identity, and it does
(|ΔH| 2.4–3.8). The capture's `base` plans are bit-identical to the Q1 batch-4 `base` dumps (0 / 1,112 windows
differ).

## ADDENDUM 2026-10-12 ~21:45Z: the V-R8r replicate (training seed 1)

Thor batch-8 dumps, same eval139 grid (1,112 windows / 139 episodes), paired episode-cluster bootstrap, sampler seeds
0 / 1, md5-verified against Thor. Readouts: `raw/vr8r/`. Still POST-HOC and descriptive; L1 is scored at I-0.

* **The turn regression REPLICATES on a second training seed, and is larger there.** V-R8r vs V0r (both training seed
  1): GT-turn `cross_abs_6s` **+6.88 / +6.21 m**, ADE **+2.32 / +2.14 m**, all separated worse; all windows cross
  **+0.99 / +1.03 m**. At training seed 0 (V-R8 vs V0, Thor batch 8) it was +4.41 / +4.09 m.
* ⚠ **CORRECTION to "mis-SHAPED, not mis-directed" above: that holds at training seed 0 only.** At seed 1
  `dir_correct_turn` is **0.523 / 0.533** against V0r's 0.645 / 0.636 (separated worse on sampler seed 0, ns on 1)
  and against V-R8's 0.692 / 0.710 (separated worse on both). The R8 recipe also gets the turn DIRECTION wrong more
  often at seed 1.
* **The R8 recipe is far less stable run-to-run on turns.** V-R8r vs V-R8 (same recipe, seed 1 vs 0): GT-turn cross
  **+3.30 / +2.95 m**, separated. That is about **4×** V0's own replicate spread (V0r vs V0: +0.83 / +0.83 m). It fits a
  near-flat selector score that picks among near-ties.
* **This is P1's floor in PREREG_VR8_NOLW (rule unchanged).** V-R8-noLW must beat V-R8 on GT-turn cross by more than
  **3.30 m (s0) / 2.95 m (s1)**. V0 beats V-R8 by 4.41 / 4.09 m on Thor batch 8, so a noLW that only matched V0 would
  read R ≈ 1.3–1.4.
* **Families, V-R8r vs V0r (same pattern as seed 0):** LONGITUDINAL better (`speed_mae_0_2s` −0.016 / −0.017,
  `speed_mae_2_6s` −0.095 / −0.101, separated); TACTICAL `lat3_f1_v9` +0.205 better; LATERAL worse (cross, curvature
  and yaw-rate separated); PERCEPTION drivable IoU +0.030 better, lane IoU −0.008 worse; SAFETY `off_drivable` ns.
  STRATEGIC and distance-keeping are not instrumented at this rung.
* **Consequence for L1.** B-RECIPE needs turn direction-correct and heading-15 GAINS of V-R8 over V0, separated
  with R > 1. Descriptively both training seeds show losses, so L1 will almost certainly FAIL as registered.
