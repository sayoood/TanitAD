# PREREG_V8F_SLOWWM: does a closed-form good-view (temporal-CCA) basis for z1 make the ladder world model plannable? (Master Mind, 2026-11-11, written before any SLOWWM code or run; PI approved the box-2 slot after STOPBAL, 2026-11-11)

Every number carries its evidence class, and every eval number carries its tier.

## Why
* **Theory (PUBLISHED, arXiv:2610.39302, library key `2610.39302`).** A JEPA with an identity predictor and whitened anti-collapse learns the spectral embedding of its positive-pair graph (Prop. 1). Positives that are two observations of one world state ("good views") keep the state and drop per-observation nuisance (Prop. 3). Alignment lives in the leading, gapped directions (Thm. 2, Fig. 6).
* **v8F (MEASURED, source).** The adapter is affine on frozen DINOv3 token features (`v8f_mini.py:1119-1125`). The token mean of the parameter-free layer-normalised features, `nm` (1024-d), is adapter-independent (`pooled_adapt_mean` docstring). So Prop. 1's optimum with temporal positives has a closed form: the top canonical directions between `nm_t` and `nm_{t+Δ}`. It is seed-free and whitened, and its directions are ordered by predictability μ.
* **What our trained adapters do instead (MEASURED, `raw/2026-11-11-prh-jepa/`):**
  * PRH-SEED v2 reads O2: the two trained seeds agree slightly beyond random (+0.0169), but sit nearer DINOv3's high-variance (PCA) directions than the slow subspace (−0.046 at both seeds).
  * PRH-TAC reads NO-TAC-LEAN, with weak power.
  * The slow subspace has to be built; training does not find it.
* **The planner reads on this world-model stream** (PLANTEST, PLANTEST2, PLANTEST3, PT3N, PT3M) are NO-GAIN, and EGOWM is NO-EFFECT (WMPOOL card, "Why"). H-PRH-PLAN asks whether part of that is a latent whose directions are not identified.
* **Admissibility.** The basis is a function of frozen vision features and same-clip time adjacency only: no ego state, no action, no label, no situation-classifier output. It is vision-only at inference by construction.

## Question
On the ladder rig (frozen readers and planner, world model only, T0), does replacing z1's input with a whitened closed-form temporal-CCA basis (SLOW, lag Δ = 1.0 s) make the ladder planner harness read PLANNABLE where R35H does not? And does SLOW beat a whitened PCA basis of the same features (PCA-B), the control that separates "good views" from "any fixed whitened basis"?

## Bases (built once, before any run; fit on the ladder's TRAIN windows' frames only, eval-141 never used in a fit)
* `x_t = nm_t − mean`, over frames with `ok`.
* `C = E[x xᵀ]` (train frames).
* `S = ½(E[x_t x_{t+5}ᵀ] + E[x_{t+5} x_tᵀ])`, over same-clip pairs with both frames `ok` (lag 5 steps = 1.0 s).
* `ridge = 1e-3 · tr(C)/1024`.
* **SLOW:** the top-256 generalised eigenvectors of `S v = μ (C + ridge·I) v`, normalised so that `Vᵀ(C + ridge·I)V = I`. z1 input `u_t = Vᵀ x_t` (256-d, whitened, ordered by μ).
* **PCA-B:** the top-256 eigenvectors of `C`, each scaled to unit variance. z1 input `u_t = Λ^{-1/2} Uᵀ x_t`.
* Each basis is stored as a matrix file with its sha256. The per-frame caches use the existing zcache format ([T,256] f32 per clip, `ok`, `t_s`), carry a manifest `kind ∈ {slow, pca}` plus the basis sha, and cover train and eval.
* The eigenvalue spectra (μ for SLOW, the variance for PCA-B) are written beside the basis (MEASURED, no bar).

## Arms (world model only; everything except z1's input as R35H: strides 3 5, k 10 10 6, K = 3, ctx 3 3 3, 19,164 steps, the same 40,358 windows)
Listed in priority order: a stopped chain still yields a paired seed-0 read.

| order | arm | z1 input | WM init seed |
|---|---|---|---|
| — | R35H (control of record, already trained) | M2b adapter token mean (s0 / s1 adapter) | 0, 1 |
| 1 | **SLOW_s0** | SLOW basis, then `proj` | 0 |
| 2 | **PCA-B_s0** | PCA-B basis, then `proj` | 0 |
| 3 | **SLOW_s1** | SLOW | 1 |
| 4 | **PCA-B_s1** | PCA-B | 1 |
| 5 | **SLOW_rep** | SLOW, for the floor (a same-seed replicate is bit-identical on this rig, `ladder_read/R35B_CKPT_IDENTITY.json`) | 2 |

* The bases are seed-free, so seeds 0 / 1 / 2 differ only in the world model's initialisation and data order.
* ESTIMATED cost: about 3.3 h per run (EGOWM's 8,636–9,161 s, MEASURED), so about 16.5 h for 5 runs, plus about 1 h for the basis and caches. Within the 72 h rule.
* The lever (`--wm-basis {adapter, slow, pca}`, default `adapter`) is named in each argv, read back from the BUILT model, and passes the per-run token route.

## Read (T0; WMPOOL's planner read verbatim, with one change: the deciding read is in trajectory space)
z1 lives in a different coordinate space per arm, so a latent-relative score cannot compare SLOW, PCA-B and R35H. IA-0 skill against Copy-Last is therefore **beside, not deciding**; a slow basis also makes Copy-Last strong by construction.

**Deciding — planner.** The `ladder_plantest` harness (PLANTEST and PLANTEST2 forms, latent cost) on FLAT1 at level 1, contexts 1 and 3 as labelled rows:
* rows: G-TRUE (vision-only goal: the arm's own basis applied to the future frame) minus HOLD, and minus G-RAND;
* families: LONGITUDINAL and LATERAL;
* estimator: paired episode-cluster bootstrap over the harness's episodes, B 2,000. It answers "another draw of episodes"; the training-run question is the floor below.
* TACTICAL: n = 0, no decision output. STRATEGIC: PAUSED (PI item 25).
* PLANNABLE for an arm and seed = G-TRUE separated-better than HOLD and than G-RAND in at least one family, by the harness's own label rule.

**Deciding — floor.** Let `m` be the G-TRUE-minus-HOLD margin per family. `floor = |m(SLOW_rep) − m(SLOW_s0)|` per family and context. A difference between two arms counts only if it exceeds the floor of its family at both seeds 0 and 1.

**Beside, no bar:**
* IA-0 skill per level at the band ends (L1 2.0 s, L2 6.0 s, L3 12 s);
* WMACT's action-sensitivity audit;
* decodability: ridge probes from the rolled-out latent at 2.0 s to the measured future speed and yaw rate (fit on train rollouts, scored on eval-141; the same protocol per arm, so comparable across bases);
* the planner with its latent cost restricted to SLOW's top-64 μ directions (H-PRH-PLAN, zero training, SLOW arms only).

## Committed outcomes (before any run)
| outcome | condition | committed next step |
|---|---|---|
| **SLOW-PLANS** | SLOW is PLANNABLE at both seeds in at least one family; its margin exceeds R35H's and PCA-B's by more than the floor at both seeds in that family | SLOW becomes the world-model latent of record. Write per-level SLOW bases at the level clocks (lags 0.6 s and 3.0 s for L2 and L3), read HWM-PLAN on SLOW, and put L4 (multi-camera good views) to the PI. |
| **BASIS-ONLY** | SLOW and PCA-B both beat R35H beyond the floor at both seeds; SLOW minus PCA-B within the floor | The gain comes from a fixed, seed-free whitened basis, not from good views. PCA-B (simpler) becomes the record. The good-view question moves to L4, the only remaining good-view lever (PI decision). |
| **NO-EFFECT** | neither SLOW nor PCA-B beats R35H beyond the floor at both seeds | The z1 basis is not the planner's limit. The predicted-goal path (HWM-PLAN G4 / G5) and the T-side speed levers get the next box-2 time; L4 is deprioritised. |
| **SLOW-COSTS** | SLOW's margin falls below R35H's by more than the floor at both seeds | Reported as a cost, with the decodability rows. The next step is NO-EFFECT's. |

## Controls (the read is unusable if one fails; each has a deliberate-regression arm that must go red)
1. **Off identity:** with `--wm-basis adapter` (default), the built model and one trainer step on a fixed batch equal R35H bit for bit. Mutation: a default of `slow` goes red.
2. **Reachability:** with `slow` / `pca`, the cache the trainer reads is the named basis (manifest kind and basis sha read back from the BUILT run), and a gradient reaches `wm.proj` from the actual trainer path (argparse → namespace → model → z1 → loss, `file:line` at each hop). Mutation: pointing the trainer at the adapter cache under `--wm-basis slow` is refused.
3. **Basis identities (MEASURED before any run):** on the train frames, `cov(u)` = I within 1e-3 for both bases. On 8 clips, the cached `u` equals the basis applied to recomputed `nm` within 1e-4. SLOW's top μ exceeds PCA-B's top direction's lag-1.0 s autocorrelation; a planted shuffled-time basis must fail that check.
4. **Echo guard:** z1 targets and goal latents are a function of vision only. The basis files are built from `nm` and clip-time adjacency alone; a grep and a hash audit show that no ego, action or label array enters the builder.
5. **Launch:** every lever named in argv and read back from the BUILT model (G-DVB form); per-run token route; pilot of 200 steps on SLOW_s0 before the runs.
6. IMAG1B's controls 2, 3 and 4 on the read, as in the ladder.

## Code changes the lever needs (none exists at freeze)
* the basis builder and cache writer (theory script, beside `theory/ladder/zcache.py`);
* `stack/tanitad/models/v8f_mini.py`: config lever `wm_basis` (default `adapter`) and a G-DVB reader;
* `stack/tanitad/data/v8f_wm_windows.py`: serve the basis cache, with manifest checks;
* the trainer's flag, and its refusal on a cache-kind mismatch;
* the planner harness and IA-0 readers: z1 inputs from the arm's basis;
* the `launch_gate.py` G-DVB entry (by its owner; this card does not edit it).

The code is based on the ladder tree (`1faa1a06`, R35H's code), on its own branch, so R35H stays code-comparable.

## Log
* 2026-11-11: frozen; the sha256 is recorded in `PREREG_V8F_SLOWWM.sha256`. No SLOWWM code, basis or run existed at this time.
