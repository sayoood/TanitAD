# PREREG_V8F_STRATHEADS (DRAFT, Stream C, 2026-11-08). Bars are committed here before any real-data run; the Master Mind freezes the sha.

STRATEGIC IS PAUSED (PI item 25). Every result below is reported as PAUSED, tier T0, never as a pass. Training these heads is allowed in principle; whether to spend the time before item 25 lifts is the PI's call.

## Question
Can three small heads on the frozen ladder world model and the per-frame latent cache (a) read country, illumination, weather, surface and visibility from vision alone, (b) decode route checkpoints at 6 to 30 s from rolled L3 states better than a constant-velocity rollout, and (c) derive strategic goals and context constraints from the L3 state, nav and the detached context posteriors, with the context edge live?

## Arms and controls
* Arm `SH_s0`: the argv in `box2_argvs.txt`, seed 0, 6,000 steps, batch 128, R35_s0 world model.
* Controls inside the run (self-check, before step 1): lever read-back, gradient matrix with exact zeros, WM sha256 before and after, vision-only and goal-path probes with mutations, edges live with a cut mutation.
* Eval controls: constant-velocity baseline per horizon; context-swap run of head_gstr; `logged` versus `zero` checkpoint mode.
* Replicate: a second head seed (`SH_s1`, argv identical but `--seed 1`) is required before any difference between two arms is called a lever effect. One arm's interval answers the episode-resampling question only (H-ESTIM-SEED-1).

## Data and splits
Train: the ladder corpus clips with zcache `M2b_s0/train` and a label row (the 60 excluded clips have none). Eval: the 141 eval clips (23 countries). Eval frames every 5th step (1 s). Estimator: episode-cluster bootstrap over clips, 2,000 draws, `taniteval/ci.py`. Labels: release frames (context, goals), stream A (checkpoints).

## Committed bars (HYPOTHESIS thresholds, chosen before the run; none is derived from data)
| id | quantity (eval-141; ckpt in zero-action mode) | bar |
|---|---|---|
| C-ILL | illumination macro-F1 (4 classes), CI lower bound | >= 0.70 |
| C-WEA | weather macro-F1 over classes with support, CI lower bound | >= 0.40 |
| C-SUR | surface macro-F1 over classes with support, CI lower bound | >= 0.50 |
| C-VIS | visibility macro-F1, CI lower bound | >= 0.50 |
| C-CTY | country macro-F1 over classes with support, point estimate (chance about 0.04) | >= 0.30 |
| K-6 | head xy error at 6 s minus constant-velocity xy error, paired CI | upper bound < 0 |
| K-ALL | same, at each of 9, 12, ..., 30 s | upper bound < 0 at >= 6 of the 8 further horizons |
| K-MASK | share of eval rows with a valid label at 30 s | reported with n; no bar |
| G-TOK | gstr token macro-F1 (classes with support), CI lower bound | above the majority-class macro-F1 on the same rows |
| G-CTX | gstr plain versus ctx-swap (token macro-F1 and constraint MAE), paired CI | separated in the direction "plain is better", else the context edge is reported as not used |
| G-CON | speed-margin and headway-margin MAE | below the MAE of predicting the train mean |
| W-EQ | WM sha256 before and after | identical (exact) |
| S-GRAD | self-check matrix | exact zeros as specified; any other reading stops the run |
Every number in the report carries its arm, artifact path, tier T0 and the estimator name.

## Outcomes committed in advance
* **All C, K and G bars met:** the heads are eligible inputs for the T-conditioning token arms (streams B and D); the report still says PAUSED.
* **C bars met, K missed:** the rolled L3 states carry no route information beyond constant velocity at that horizon. Attribution: `logged` against `zero`, and the head against one fed the NOW state only. Next levers, by expected effect: (1) feed the checkpoint head the pooled visual summary next to the rolled state; (2) train the rolled states with the multi-step latent term to k = 10 (a world-model change, the MM's call); (3) predict residuals to a nav-conditioned prior.
* **C bars missed (weather, surface, visibility likely):** label noise is the first suspect (those labels are a probe plus a VLM, not human labels): re-score on the PI-audited frames before changing the head. Next levers: more past offsets, a wider head, a SigLIP-feature input as a separate arm. Nothing is unfrozen.
* **G-CTX missed (the swap does not change the metric):** the goal head ignores the context. Read the swap effect on the token probabilities first (reported); the cheap lever is a larger weight on the constraint term, whose labels are a function of the context.
* **G-TOK missed:** only 4 of 8 tokens occur and `STOP_AT` is rare; lever: class-weight power 1.0 and a longer run, then a per-token report.
* **Train/eval gap large on C or K:** K-fold heads (5 folds over the train clips) so streams B and D receive out-of-fold `stok` for train clips.
* A REFUSED line from the self-check is a blocker to fix, not a result.

## Next arms ready before the run ends
`SH_s1` (replicate seed), `SH_cv` (checkpoint head without the rolled states, the attribution control), `SH_kfold` (five heads for out-of-fold `stok`).

## Cheap checks to run first (0 GPU)
1. `act_extract.py --check-cache` on 100 clips: `ACT_CHECK.json`. If the max accel difference exceeds 0.05 m/s^2, the logged mode is excluded from the report and `zero` alone is quoted.
2. Compare the L3 state at a tick frame with the non-tick lift on real `zmean` (bit-checked on synthetic input only).
3. Count valid labels per horizon on the train clips: the share at 30 s decides how much the last horizon weighs.

## Reporting rules
STRATEGIC family reported PAUSED with this ruling, never a pass, never silently omitted. Tier T0 on every row. No interval without its estimator name. No claim about the hierarchy's value from this run: it measures three heads on a frozen model.
