# PREREG GRID 3 (+ conditional GRID 4): SH5 C-REACH, attention-sharpness levers first (Research Lab, 2026-10-19, written BEFORE any grid-3/4 number)

Diagnostic grid, NOT a registered SH5 stage. It trains only the planted control (`plant`) on fold 0 (= `fit --stage PRE --folds 0`) via `code/diag_pre3.py` (imports the sealed code read-only). No real arm, no `analyze`, no navtest. Tier: none (an instrument check on fold 0 of the LOCAL SH5 sets). Estimator: point values on fold 0 (n_sets 322), one training seed per cell per seed index. The question each number answers: "does THIS training run of THIS recipe read the plant?" Run-to-run noise is read from the two seeds and from the replicate controls below (H-ESTIM-SEED-1); no CI is claimed.

## What grids 1-2 measured (verified from `work/diag1/diag_summary.json`, `work/diag2/diag_summary.json`; MEASURED)
- No recipe passes. Best final F-AUC: 0.6008 (k12o12 lr 3e-4, 12 ep, s0); best final ALLQ 0.6732 (base, 12 ep, s1). 36 ep lr 1e-3: ALLQ 0.588-0.626, F 0.485-0.548, R^2 K6/o8 0.437 (s0) / 0.644 (s1), K10/o10 0.412 (s0) / 0.166 (s1).
- Ablations: zero_clear moves ALLQ by <= 0.0043 in every cell (max: K10/o10 lr 1e-3 12 ep s1, 0.5689 -> 0.5731), so the CLASSIFIER never reads the planted clearance. On K6/o8 s1 zero_clear drops R^2 0.644 -> 0.024, but **shuffle_clear (same values, wrong positions) keeps it at 0.630**: the regression head reads a TOKEN-LEVEL average of the plant, never the position-matched entry. On s0 neither ablation moves R^2 (0.437 -> 0.433 / 0.437).
- Pre-launch CPU checks (`raw/levers_cpu_check.json`, MEASURED, 2 fold-0 tokens, real ctx): at INIT the sealed branch's logit range over the plant entries is ~1 (beta_eff 0.1-0.2) against the ~30 (beta >= 16) an ideal reader needs (`raw/probe2_ideal_reader.json`). The q/k-init scale was fixed by a rule written in `code/check_levers_cpu.py` before it ran: the smallest c in {2,3,4,6,8} whose layer-0 plant logit range at init is >= 30 -> **c = 6** (range 35.2).
- The trained learners of grids 1-2 were NOT checkpointed (`diag_pre.py` saves no weights), so their trained logit range cannot be read post hoc. Grid 3 re-trains them as **deterministic replicates** (same seed, init and data order) with the attention diagnostic, so the number is measured on the same learners up to GPU non-determinism.

## Grid 3 (`work/grid3.json`; runs S12_s0, S12_s1; 12 epochs = 3 cycles x 4; lr 3e-4; p_sub 32; ablations zero_clear / shuffle_clear / no_plant on every cell)
| cell | the ONE lever vs a tested cell |
|---|---|
| base_k6o8 | none: replicate of grid-1 `base_k6o8_lr3e-4` (s0: ALLQ 0.6489) / grid-2 `base_k6o8_lr3e-4` (s1: 0.6732) |
| base_k8o8 | none: replicate of grid-1 `k8o8_lr3e-4` (s0: 0.6467) |
| temp_kKo8 | learnable per-head temperature, tau = exp(10 s), tau0 = 1 (bit-identical to base at init) |
| cos16_kKo8 | cosine (QK-norm) attention, learnable tau, tau0 = 16 (= the ideal beta) |
| qk6_kKo8 | q and k weights x 6 at init |
K in {6 (registered query), 8 (query octaves = plant octaves)}. tau is clamped at 100.

**Attention diagnostic** (every eval, first 24 fold-0 tokens, `code/diag_attn.py`): per layer and head, beta_eff = the per-query-centred least-squares slope of the logits on probe 2's kernel (octaves = min(K, 8)); logit range over the plant entries; plant attention mass; own-entry mass; top-1 within 0.5 m; tau. Checked: synthetic 16 x kernel reads 16.0001, random -0.07, constant exactly 0.0; softmax(logits) @ V equals the SDPA output to <= 4e-6.

## PASS (fixed before any number; `diag_pre3.learner_pass`)
On BOTH seeds, final cycle, fold 0: F-AUC >= 0.97 AND ALLQ >= 0.95 AND R^2 >= 0.50, AND the read collapses without the planted clearance: zero_clear AND shuffle_clear each read ALLQ <= 0.80 AND F <= 0.80 (above both the flat ideal reader, 0.73-0.77, and every plant-free learner so far, <= 0.70).

## Replicate controls (known values)
base_k6o8 S12_s0 must reproduce 0.6489 ALLQ, base_k6o8 S12_s1 0.6732, base_k8o8 S12_s0 0.6467. A |difference| > 0.03 is reported as the GPU non-determinism floor of this rig; any lever effect smaller than it is not an effect.

## Outcomes and committed next arms
- **O1: >= 1 grid-3 cell passes.** Grid 4 skips itself (`--skip-if-pass`). Next: HANDOFF steps 3-5 with the CHEAPEST passing cell (12 ep; lever order qkinit < temp < cos; K6 before K8).
- **O2: no pass, and some lever cell reaches beta_eff >= 16 (best head, any layer, BOTH seeds) with top-1 >= 0.5.** Sharpness is learned but is not the binding term alone; the binding term is downstream (value read, key-time pooling, head) or the budget. Grid 4 runs. If grid 4 also fails, the committed next lever is the READOUT: a key-time pooling that can express a min over the 9 points (e.g. a softmin pool) on the best lever cell.
- **O3: no pass, and no lever cell reaches beta_eff >= 16.** The recipe does not learn sharpness even when a lever allows it. Grid 4 runs (lr 3e-3 / 60 ep / p_sub 64 on BEST and on base_k8o8). If grid 4 also fails, report **INSTRUMENT DEFECT** with each lever's best number. The next lever is fp32 attention (no bf16 autocast) on BEST at lr 3e-3, then a supervised attention target on the plant (diagnostic only, never the instrument).
- Base replicate trained beta_eff is reported in all outcomes. **If base already reaches beta_eff >= 16, sharpness is NOT the binding term**, whatever the levers do.

## Grid 4 (`work/grid4.json`; conditional, chained after grid 3 only if grid 3 exits 0 and nothing passes)
BEST = the grid-3 LEVER cell with the highest MIN over seeds of final F-AUC (ties: higher min ALLQ) (`diag_pre3.best_lever`). Runs W12_s0/s1 (12 ep): base_k8o8_lr3e-3, BEST_lr3e-3, base_k8o8_psub64, BEST_psub64. Runs W60_s0/s1 (60 ep = 15 x 4, eval at cycles 3/6/9/12/15): base_k8o8_60ep, BEST_60ep. Same PASS rule and outcomes.
