# PREREG_V8F_SPEEDSUB: does the banked ego-only read-out, put into the five speed cells O reads, cut O's ADE6?

**Status: PRE-REGISTERED, NOT RUN (2026-10-28).** Promoted by the Master Mind from the draft (`raw/2026-10-28-speedsub-draft/`, sha256 63946e14…) before any SPEEDSUB number; the Master Mind rewrote one sentence of §2 before promotion (the `limit` admissibility: `limit` is in v8F's own `INFERENCE_INPUT_KEYS`). No SPEEDSUB number exists: the instrument has run on synthetic data only (39 new tests plus SPEEDREAD's 27; 28 of 28 mutants killed), and the `ego` substitution has not been run on any real window. Claim row `H-V8F-SPEEDSUB-1` (to be opened on promotion). Commissioned by `RESULT_V8F_SPEEDREAD.md` §5 item 1 (*"Its ADE6 substitution was NOT run"*) and §6. Base commit read: `01f3d34e`. Base instrument: `v8f_mini_eval_speedread.py` sha256 `fd5ce0bea37c3e0e608144e041f54a337fd4378b38fb312df65787bc864be4fe` (SPEEDREAD, Amendment 1). Staged instrument for this draft: sha256 `a95f4fb14da3fd8f6f89e55dfcb23d5d26a81b09dd24285aba394f36546634ca`.

## 0. Why this experiment, and why it is the cheapest next lever

| step | finding | class |
|---|---|---|
| KINDX / SPEEDREAD `gt` | Replacing T's five speed cells by the GT cuts O's own-code ADE6 by **−0.7164 [−0.8182, −0.6234]** (s0) and **−0.6676 [−0.7613, −0.5753]** (s1). | MEASURED (`raw/2026-10-28-speedread/read/READ.json` `rows.<s>.gt`, sha256 `46543c39…`; T0 ceiling) |
| SPEEDREAD `read` | The ridge read-out (PRIMARY `lin`, vision plus ego) moved ADE6 by −0.0097 [−0.0871, +0.0620] / +0.0022 [−0.0534, +0.0564]. Outcome **NOT IN INPUT**. | MEASURED (same file, `rows.<s>.read`) |
| SPEEDREAD `alt` | The vision-plus-ego MLP cost +0.2122 / +0.2040 m, separated-worse at both seeds. | MEASURED (same file) |
| the arm not run | The ego-only MLP `mlp_ego` has the lowest fold-0 hold-out RMSE of the four fitted read-outs in all five cells, and the lowest eval RMSE in all five cells. It was written to the identity dump as `pred_mlp_ego` and **never substituted**. | MEASURED (`fit/fit.json` `mlp_ego.holdout_rmse_sd`, re-read 2026-10-28; `READ.json` `cells.<s>.mlp_ego.rmse`) |

Fold-0 hold-out RMSE in train-SD units, mean over the five cells (derived from `fit.json`): `mlp_ego` **0.24751**, `lin` 0.27236, `lin_ego` 0.28071, `mlp` 0.29196. Eval RMSE in m/s on eval-141 (`READ.json`), cells `lon_v_target_ms, v_a_ms, v_end_ms, v_lo_ms, v_hi_ms`:

| | s0 `T_own` | s1 `T_own` | `mlp_ego` (both seeds) |
|---|---|---|---|
| RMSE (m/s) | 2.894, 1.919, 2.795, 2.257, 2.093 | 2.747, 1.302, 2.692, 2.143, 1.802 | 2.540, 0.752, 2.557, 1.861, 1.714 |

`mlp_ego` is below `T_own` in every cell at both seeds. The ridge was also lower than `T_own` in `v_a_ms` (RMSE cut 59 % at s0 and 40 % at s1, derived from `READ.json`) and ADE6 did not move. **A lower cell RMSE has already failed once to move ADE6**, so this experiment is a real test, not a formality.

**Why a substitution and not another fit.** SPEEDREAD's lesson is that O responds to the five cells (`gt` moves it by 0.7 m, a constant moves it by +5.3 m) and that RMSE alone does not predict the response. The substitution reads the response directly, needs no training, and reuses the banked read-out unchanged (`readout.pt` sha256 `a205609a…`, MEASURED equal to `fit.json` `readout_sha256` on 2026-10-28). The decision path costs about 1.3 to 2.7 CPU-h (§8).

⚠️ **Disclosure of how this arm was chosen.** The Master Mind commissioned `mlp_ego` after reading SPEEDREAD's beside table, which contains eval-141 RMSE. Two things limit the damage, and neither removes it:
* The selection rule restated on TRAIN only, **fixed here before any SPEEDSUB number**: of the four fitted read-outs, the one with the lowest mean fold-0 hold-out RMSE (train-SD units). `fit.json` alone gives `mlp_ego`. The eval table is not an input to the rule. Control S13 recomputes it from `fit.json` at read time.
* This is the **third** read-out examined on these 141 episodes (`lin` and `mlp` were run; `lin_ego` was not). A separated result at both seeds is therefore a screening result and PASS is not a lever claim (§10).
* ⚠️ The MLP read-outs early-stopped on fold 0, so their fold-0 figure is mildly optimistic against the ridge, whose λ was chosen on folds 1–4. The margin (0.2475 against 0.2724) is large next to that effect, but it is not zero.

## 1. Hypothesis

**H-V8F-SPEEDSUB-1.** Replacing only the five speed cells of the code O reads by the banked `mlp_ego` prediction, a function of the ego block of T's own input at the tick frame (§2), cuts X_s's own-code ADE6 by at least **F/2 = 0.06035 m**, separated, at both seeds.

* X_s = `M2b-mixo_s<s>`, as in SPEEDREAD.
  * s0 sha256 `c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d`, `C:/Users/Admin/box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt`.
  * s1 sha256 `68a0f3356c37b25ce15b5f860f8dc919e23efc8b6bf200a2997f9ad6b521e915`, `C:/Users/Admin/box2_ship/logs/v8f_kindx_s1/runs/M2b_s1_mixo/ckpt_last.pt`.
  * Both are the files SPEEDREAD's run script pinned (INHERITED, `run/run_speedread.sh`); the run script re-checks them.
* F = |M2b_s0 − M2b_s1| = 0.1207 m (INHERITED from `RESULT_V8F_GRID.md`; `F_M` in the instrument and `READ.json` `F_m` both read 0.1207, MEASURED). **F/2 = 0.06035 exactly.** The brief rounds it to 0.0604; the instrument compares against `F_M / 2.0`, so a point estimate of −0.0604 and one of −0.06036 both pass, and −0.0603 does not.
* **Why F/2 and not SPEEDREAD's F.** SPEEDREAD's INFO IN INPUT bar was −F because a pass committed a box-2 training arm. This step commits only a zero-training T1-kin substitution (§6), which carries its own bar and replicate, so the Master Mind set a screening bar of half the rig's run-to-run spread. Reading it plainly: F/2 is 8.4 % (s0) and 9.0 % (s1) of the `gt` ceiling (derived: 0.06035/0.7164, 0.06035/0.6676).
* ⚠️ **The separation condition will bind before the point bar.** SPEEDREAD's `read` CI half-widths were 0.075 (s0) and 0.055 (s1). If the `ego` contrast has a similar width, "upper end below 0" needs a point estimate of about −0.055 to −0.075, which is the same size as the −0.06035 bar. Both conditions are required and neither is relaxed.
* ESTIMATED prior, not evidence: PASS is not the expected outcome. The ridge cut `v_a_ms` RMSE by 40–59 % and moved nothing, and `mlp_ego` improves the other four cells by only about 8–17 % over `T_own` (derived from the table above). The experiment is worth running because NULL and HARMS both name a next lever (§7).

## 2. What `ego` reads (MEASURED from source 2026-10-28; the brief's wording is narrower than the truth)

The brief says the inputs are *"measured ego-motion history only"*. The code says otherwise, and this section is the accurate description. The read-out `mlp_ego` is `ReadOut(ego_only=True)`: design = the 22-dim ego block of `window_vectors`, standardised by TRAIN statistics (`xm`, `xs`, 22 wide in `readout.pt`). It reads **only the tick frame `last_t = F − 1 − a_t`** of each key (not a history), and `frame_ok` for the four history frames.

| channel | dims | what it is (source) | class |
|---|---|---|---|
| `p10f` | 8 | P10 kinematic features at the tick, derived from the measured triple (`v0`, `a0`, `kappa0 = clamp(omega0 / max(v0, 2), ±0.3)` with `omega0` the measured yaw rate); the first two are `v` and `a` (`v8f_mini_data.py` header, PI ruling D-PI-YAWRATE-1; `yaw_kin_from_track`) | measured ego state |
| `nav` | 5 | route / manoeuvre token, validity, progress `d_next / 100` at the tick (`v8f_mini_data.py` `EGO_PERTURB_REBUILT` comment) | route input; admissible at training and inference (PI) |
| `limit`, `limit_ok` | 4 + 1 | speed-limit bucket one-hot and its validity, from the sidecar `refcv6_speed_max_v8_*.jsonl` | **a sidecar label, not ego state** (`v8f_mini_data.py` `EGO_PERTURB_KEPT` comment); an INPUT, never a training sign |
| `frame_ok` | 4 | validity mask of the four history frames | mask |

* **Not read, enforced three ways:** `feats` (pixels), any `lab_t` label, `pose`, the cache's `steer` / `kin3`, S's outputs, any situation-classifier output, any goal path. The substitution writes speed cells, not a goal. (1) `EGO_READ_KEYS` excludes them and a test pins `EGO_READ_KEYS ⊂ READ_KEYS` and `"feats" ∉ EGO_READ_KEYS`. (2) `ReadOut.predict_ego_batch` builds a dict of those keys and nothing else before it computes, so it cannot reach a pixel; a test passes it a batch with NaN labels and 6e4 pixels and a batch without either, and requires identical output. (3) Control S7 requires O's cells to equal the identity run's `pred_mlp_ego` (computed through the scene path) to 1e-5, so the two routes agree on the real windows.
* **Admissibility.** `READ_KEYS ⊂ INFERENCE_INPUT_KEYS` was checked by SPEEDREAD's `extract` (`_stack_imports` raises otherwise) and the new `EGO_READ_KEYS` is a strict subset. These are the non-visual inputs T already reads at inference, so `ego` gives O nothing T does not already have. The 2026-08-03 ruling (*labels may use ego; inference is vision-only*) was made for the scenario classifier; it is not applied here to the planner's T, which the PI has already given measured ego motion (D-PI-YAWRATE-1) and nav. `limit` and `limit_ok` are in v8F's own `INFERENCE_INPUT_KEYS` (`stack/tanitad/models/v8f_mini.py:137`), so T already reads them at inference and `ego` adds no channel. Whether a speed-limit sidecar is an input a deployed car may have is a standing v8F design question that predates this arm and is not settled here.
* ⚠️ **Nav is optimistic on PhysicalAI** (CLAUDE.md): the only route supplier is the ego's own future path, and `nav[:, 3]` is a distance-to-next-manoeuvre. T reads it too, so `ego` is not more privileged than T. But a PASS may be partly carried by `nav` or `limit`, and no ablation is fitted here (no re-fit is allowed). The attribution is a named follow-up (§6).

## 3. The arms and the estimator

Per seed, the instrument's `subevals` runs five modes through `v8f_mini_eval.py run` with SPEEDREAD's carrier (`--tac oracle --tac-groups speed --tac-alpha 1`, `--families`); every non-`own` mode gets the same carrier.

| mode | the five speed cells O reads | role |
|---|---|---|
| `own` | X_s's own | baseline |
| `identity` | X_s's own, through the hook | control S1: equals `own` to the bit; also dumps `pred_mlp_ego` and `pred_const` on every window |
| `gt` | the GT where the window has a v9 row, else own | T0 ceiling and control S6 |
| `ego` | **`mlp_ego(ego block at the tick)`**, standardised by the checkpoint's `cons_mu / cons_sd` | **the arm** |
| `egoshuf` | the same `ego` cells, **shuffled across the 2,399 eval windows** by the fixed permutation `numpy.random.default_rng(20261029).permutation(2399)` | **deliberate-regression arm** |

* No read-out is fitted, tuned or selected in this experiment. The fit was made on TRAIN fold structure before any eval batch (`eval_train_disjoint` true, `train_parity_ok` true in `fit.json`); `readout.pt` is not rewritten.
* Every other cell O reads, S, T's decisions and O are X_s's own (control S2: max |Δ| = 0.0).
* **Tier.** `ego`, `egoshuf`, `identity`, `own`: *T1-class open loop (X_s's own held S / T outputs; only the five speed cells O reads are replaced)* (the instrument's `TIER_ARM`). `gt`: *T0 (reads the eval GT): ceiling / control only, never a result*. The unmodified harness stamps each dump `raw.tier` = `T0-open-loop-own-held-outputs (trunk-probe grid)` (quoted from `RESULT_V8F_SPEEDREAD.md`); the tier of the read is the instrument's stamp, the tier of SPEEDCAL's and SPEEDREAD's ADE6 rows. ⛔ Under `EVAL_DOCTRINE.md` neither is the action-closed-loop T1 tier: **no ADE6 here is a driving-performance number** and none may be quoted as one.
* **Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, paired against `own`, per seed. It answers *"would another draw of EPISODES say this?"* only.
  * It does not answer *"would another TRAINING RUN say this?"* (`H-ESTIM-SEED-1`). The two seeds are O's two training runs and carry that question at n = 2: the decision needs both.
  * It does not answer *"would another READ-OUT FIT say this?"*: there is one fit. A separated result is therefore about two O runs and one read-out, as in SPEEDREAD.
  * The planner is deterministic, so there is no inference-seed term.
* The pair `ego − own` shares X_s and the windows, so the contrast has no training-variance term inside it; the replicate that guards a lever claim belongs to §6.

## 4. Controls: each must read its known value before any row is read; any failure ⇒ REFUSED

| id | must read | catches |
|---|---|---|
| S1 | `identity` − `own` ADE6 max \|Δ\| = 0.0 and plans equal | the hook path changes something by itself |
| S2 | non-speed cells O reads = X_s's own, max \|Δ\| = 0.0, in `identity`, `gt`, `ego`, `egoshuf` | a wider substitution than declared |
| S3 | the hook's tick slice = the model's own `_scene` pad mask (first batch); in `ego` modes also the ego slice's `ok` | the read-out reading other frames than T does |
| S4 | episode ids equal across all five modes | window misalignment |
| S5 | `own` ADE6 = SPEEDREAD's banked `own`, per window max \|Δ\| ≤ 1e-5 m | environment drift (the bar allows a different thread count; SPEEDREAD's own C5 used 0.0 between runs at one count) |
| S6 | `gt` ADE6 = SPEEDREAD's banked `gt`, per window ≤ 1e-4 m | the oracle path |
| S6b | `gt − own` (delta, lo, hi) reproduces `READ.json` `rows.<s>.gt` within 5e-4 (printed to 4 dp) | the paired estimator or windows differ |
| S7 | what O reads in `ego` = this run's identity `pred_mlp_ego`, standardised, ≤ 1e-5 | the ego-only path disagreeing with the scene path on real windows |
| S7b | this run's identity `pred_mlp_ego` = SPEEDREAD's banked one, ≤ 1e-5 m/s | a different read-out, checkpoint or windows than the ones SPEEDREAD scored |
| S8 | `readout_sha256` in every hooked dump = `fit.json` = the pin `a205609a…` | a re-fit or a swapped file |
| S9 | `egoshuf` hook: permutation sha equals the registered one, alignment max \|Δ\| ≤ 1e-6, `n_seen` = 2,399, fixed points ≤ max(3, 1 %) | a shuffle that is not the registered one, or not over the scored windows |
| S10 | `ego` moves ≥ 99 % of windows' cells away from `own`'s; `egoshuf` differs from `ego` on ≥ 95 % | a substitution that never reaches O |
| S11 | train parity (4,512 / 306,566 / `f6a1dee0…`) | re-selection |
| S12 | eval-141 ∩ fit clips = ∅ | fitting on what is scored |
| S13 | the §0 selection rule applied to `fit.json` picks `mlp_ego` | the arm being something other than what the rule names |
| LL-1..3 | SPEEDREAD's family controls (identity reads exactly 0.0 on every row; the ADE6 row equals `SUBREAD.json`; per-arm rows reproduce the dump's own family values to 1e-4) | the family tools reading other windows |

* **The deliberate-regression arm is a control with a committed reading.** `egoshuf` must be **separated-worse at both seeds** (lower end of the CI above 0). If it is not, the instrument cannot see a harm, `decide_sub` returns REFUSED, and no `ego` row is read.
* Reachability (CLAUDE.md, *built, tested and unreachable from its caller*): the flag path is `subevals` → `cmd_evals(flow="sub")` → `Hook(…, MODE_READOUT[mode])` → `new_speed_std` → `predict_ego_batch` → `substitute` → `code_override` O reads. The stub-harness test drives exactly that chain and requires different ADE6 for `ego`, `egoshuf`, `gt` and `identity`; S7 and S10 re-prove it on the real windows, and the run script stops after seed 0 (about 5 min in) if S7, S7b or S10-equivalent fail (`preflight_s0`).

## 5. The decision (both outcomes committed now; `decide_sub()` is this table, REFUSED first, then HARMS, then PASS)

`ego_s` = ADE6(`ego`) − ADE6(`own`) at seed s. "Separated-better": CI upper end < 0. "Separated-worse": CI lower end > 0. Equality at 0.0 is not separated.

| outcome | condition | committed next |
|---|---|---|
| **PASS** | at **both** seeds: separated-better **and** `ego_s ≤ −0.06035` m | §6: the T1-kin zero-training substitution on the DAG-OACC / DAG-ORC base, under its own PREREG |
| **HARMS** | separated-worse at **either** seed (even if the other seed is a PASS-grade win) | §7: PREREG_V8F_SPEEDCELL |
| **NULL** | anything else | §7: PREREG_V8F_SPEEDCELL |
| REFUSED | a control failed, a seed is missing, or `egoshuf` is not separated-worse at both seeds | fix the control; no row is read |

* PASS and HARMS cannot both hold. NULL is reported with a descriptive label that decides nothing: **NULL-SMALL** (separated-better at both seeds, a point estimate above −0.06035 at one or both), **NULL-ONE** (separated-better at one seed only), **NULL-FLAT** (at neither).
* Beside (decide nothing): CAPTURE = `ego_s` / (`gt_s` − `own`); per-cell eval RMSE of `T_own` and `mlp_ego` from the identity dump; ADE2.
* ⛔ A committed criterion is reported as written. A NULL is a NULL even if a family row looks good (§9).

## 6. PASS: the T1-kin arm (the Master Mind's design; two parts are infeasible as written, flagged)

Zero-training substitution of the banked `mlp_ego` on the **DAG-OACC / DAG-ORC** base, re-read at **T1-kin** with FLATDAG's rollout (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; about 260 s per rollout, INHERITED from SPEEDREAD §6 citing FLATDAG). It needs its own PREREG with its own bar, both seeds and a replicate. What this draft can and cannot say:

* **Infeasible as written, 1:** the base checkpoints' paths, shas and the rollout command are **not read in this draft** (UNVERIFIED). SPEEDSUB's PASS says nothing about them: DAG-trained O has seen its own rollouts, so the response to the five cells may differ from M2b-mixo's.
* **Infeasible as written, 2:** the banked read-out was fitted with the eval's age cycle on M2b's `pool4x10` inputs and T's `--kin-mode yaw`. A DAG base must share that kin mode and cell standardisation (`cons_mu / cons_sd`); that is checkable and must be checked first (pair-compatibility, as SPEEDAVG's `pair_violations`).
* **Required first:** the attribution the 22-dim block hides. The read-out can be re-fitted on `train/train_arrays.npz` (ego, targets and masks are stored; no extraction is needed, ESTIMATED minutes of CPU) as `p10f + frame_ok` only, and as `p10f + limit + frame_ok` without `nav`. A PASS carried by `nav` is a route-supply result, not a speed-estimation result.
* **Deployable form** (as SPEEDREAD §6 describes): the read-out as a `forward_infer` speed sub-head, buffers absent by default so every earlier checkpoint stays bit-identical.
* ⛔ ADE6 at T1-class open loop is not a capability claim; the T1-kin read is the first tier at which one can be made.

## 7. HARMS or NULL: the next lever (Rule Zero), named and costed now

**PREREG_V8F_SPEEDCELL: zero-training per-cell attribution.** The measured fact to explain is that `gt` moves ADE6 by 0.7 m while every read-out that lowers RMSE moves it by about 0 or worse. The lever with the largest measured effect is "which of the five cells carries the 0.7 m", and nothing yet ranks them.

* **Modes** (same instrument, same hook, `idx` restricted to one cell): `gt_c` and `ego_c` for each of the five cells, both seeds. 20 evals of the cheap kind (27 to 53 s each at 8 threads in SPEEDREAD's logs). ESTIMATED 1.0 to 1.5 CPU-h at 4 threads.
* **What it decides:** (a) the dominant cell(s) of the ceiling; (b) whether `ego` helps or hurts in that cell; (c) if no single cell carries at least half of the ceiling, the response is to the cells jointly, and the next fit targets their joint structure (the GT cells are mutually consistent; independent regressions are not).
* **Cheapest first, no run needed:** stratify the existing `own` / `gt` / `ego` per-window ADE6 by the harness's turn / stop / lead-brake strata (reporting-only windows), to see whether the ceiling sits in event windows that RMSE does not weigh. Zero GPU, minutes.
* **If it is HARMS** the same experiment answers why lower-RMSE cells hurt, plus a distributional check (cell spread and joint consistency of `mlp_ego` against GT).
* **Not built here:** the cell-subset hook is about 25 lines; it is named, not staged, because this brief scoped one instrument change. The perception route (`PREREG_V8F_LORA` on the A40) stays blocked on three PI decisions, unchanged from `RESULT_V8F_SPEEDREAD.md` §6 (A40 allocation, frames, about 200 GPU-h).

## 8. Run (dev-box CPU, at most 4 threads, `CUDA_VISIBLE_DEVICES=-1`; every stage longer than 4 min runs DETACHED and is handed off by file)

`run/run_speedsub.sh` (staged): checks the instrument, `readout.pt`, both checkpoints and SPEEDREAD's banked inputs by sha256, then `subevals` for s0, the preflight, `subevals` for s1, `subread`, then the families. It sets `SPEEDREAD_THREADS=4`, `OMP_NUM_THREADS=4`. It does not write under `F:/TanitAD-scratch/speedread/`.

```bash
export CUDA_VISIBLE_DEVICES=-1 SPEEDREAD_THREADS=4 OMP_NUM_THREADS=4
TREE=<overlay tree from run/make_tree.sh> INSTR_SHA=a95f4fb14da3fd8f6f89e55dfcb23d5d26a81b09dd24285aba394f36546634ca bash run/run_speedsub.sh
#  subevals --seed S --x <X_s> --expect-sha <sha> --readout fit/readout.pt --fit-json fit/fit.json --expect-readout-sha a205609a…   (5 modes per seed)
#  subread  --fit-json … --eval-dir-s0 … --eval-dir-s1 … --speedread-dir-s0 … --speedread-dir-s1 … --speedread-read-json …/read/READ.json --expect-readout-sha a205609a…
```

* **CPU time, ESTIMATED** from SPEEDREAD's logs (MEASURED there: `evals_s0` 7 min 03 s and `evals_s1` 5 min 57 s for seven modes at 8 threads; per mode `own` 93 s, `identity` 126 s with all read-outs, `gt` 27 s, `read` 53 s, `const` 30 s). SPEEDSUB needs five modes, and its identity run computes two cheap read-outs instead of eight, so about 4 to 5 min per seed at 8 threads, up to twice that at 4. Decision path (evals plus `subread`, under 1 min): **20 to 40 min wall, about 1.3 to 2.7 CPU-h**. Families beside it: `longlat` 12 s per seed (MEASURED in the smoke, `logs/smoke/`), `pairfam` about 3.3 min each at 8 threads in SPEEDREAD (MEASURED, `run_families.out`), four runs, so 25 to 45 min wall at 4 threads, **about 1.7 to 3 CPU-h**. **Total about 3.0 to 5.7 CPU-h.** The 8-thread chain on this box runs until about 10:00, so wall times may stretch.
* **Disk:** under 2 GB on F: (the dumps are small); no extraction, no pull. Peak RAM under 3 GB.
* **Inputs, all MEASURED present 2026-10-28:** `fit/readout.pt` 118,851,106 B; `fit/fit.json`; SPEEDREAD's `eval_s{0,1}/eval_*-speedread-{own,gt}.npz` and `speedread_cells_*-speedread-identity.npz` and `read/READ.json` (**single copy on F:**, not in the repo: if they are lost S5, S6, S6b and S7b cannot run and the run REFUSES, by design).
* **Stop rules:** the preflight after s0 (S7, S7b, S10-equivalent) stops the run before s1 if the ego path disagrees on the real windows.

## 9. The four metric families (beside the decision; ADE is one row of four, never the result; per family, never pooled)

Tier of every row: T1-class open loop (the `gt − own` table is the T0 ceiling and a control). Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes, another draw of EPISODES only. One training run per seed, so a separated cell is a pointer (`H-ESTIM-SEED-1`). About 49 cells are read per contrast; at a nominal 5 % about 2 to 3 would separate with no effect (SPEEDREAD saw 6 and 8 for `read`).

| family | rows and tool | availability |
|---|---|---|
| **LONGITUDINAL** | speed MAE, along-track MAE, accel MAE (`families_longlat_sub.py`, a parametrised copy of SPEEDREAD's `families_longlat.py`, SMOKED on SPEEDREAD's own dumps: reproduces the banked `read − own` rows and passes LL-1..3, `logs/smoke/`) | computable, n = 2,399 |
| | **distance keeping** (DK-VIOL / HW / TG) and the anti-echo block | **UNAVAILABLE in the harness path**: status UNAVAILABLE, n = 0 in all 14 SPEEDREAD dumps (`four_families.py:704`, *"no lead-agent track supplied"*; the anti-echo block lacks the t0 ego speed). The lead block exists (`b1_eval_lead_block.npz` K = 10, `b1_eval_lead_block_k30.npz` K = 30) and `v8f_dk_read.read_dk(preds=…, pairs=…)` takes arbitrary arms, but its CLI is hard-wired to the arms `M1`, `M2b`. **Work item, not built here** (an adapter of about 60 lines). Until it exists the DK half of this family is reported as not computed, with the reason, and never as passing. |
| **LATERAL** | heading, yaw-rate, curvature, cross-track MAE, L1 lateral-accel, L2 curvature and L3 yaw-rate (GT ≥ 5.2 m/s) | computable, n about 2,100 to 2,399 |
| **TACTICAL** | TAC-LAT, TAC-LON, 5-way accuracy, goal-point error and biases, head macro-F1 (`v8f_paired_families.py`, unmodified) | computable, n = 2,399 |
| **STRATEGIC** | S-POS@30/50/80, S heading@30/50/80 (same tool) | **computable and reported (not paused for v8F).** Expect +0.0000 [+0.0000, +0.0000] by construction: the substitution acts on the code O reads after S and T have run, and SPEEDREAD's `read` and `gt` both read exactly 0 on all six rows. That zero is a property of the substitution site, not evidence about strategic quality. |

## 10. Instrument and self-test (synthetic only; this is not a result)

* `stack/scripts/v8f_mini_eval_speedread.py` (staged copy, sha256 `a95f4fb14da3fd8f6f89e55dfcb23d5d26a81b09dd24285aba394f36546634ca`): adds `ego`, `egoshuf`, `subevals`, `subread`, `decide_sub`, `controls_sub`; `MODES`, `evals`, `read`, `decide` and every SPEEDREAD code path are as they were.
* `stack/tests/test_v8f_speedsub.py`: 39 tests in about 6 s on CPU. `test_v8f_speedread.py` (27) is unchanged and still green: **66 passed**.
  * **RED on the unmodified instrument.** With `SPEEDREAD_MODULE` pointing at the pinned original, **31 of the 39 new tests fail** and 8 pass (`logs/pytest_RED_on_original.txt`). Six of the 8 are about SPEEDREAD's own behaviour and are meant to pass. Two more pass and should be named: the hook-level `ego` test, because the original's hook accepts any mode string when handed a read-out, and the no-read-out refusal, which the original also raises. What is RED is the driver (it runs seven SPEEDREAD modes and never produces an `ego` dump), the pixel-free path, the shuffle, the controls and the decision. The in-suite test `test_ego_is_RED_on_the_unmodified_instrument_and_GREEN_on_this_one` pins this.
  * **Existing modes bit-identical.** A same-process differential against the pinned original (hook output, hook stats, cells, controls, decisions, fitted read-outs, saved read-outs, the four SPEEDREAD subparsers) compares equal as JSON, and a stored golden made from the original (`golden/golden_speedread_modes.json`) matches to 1e-6.
  * **Mutation pass:** 28 mutants, each breaking one value-emitting line of the new code, **28 killed**, 0 survived, 0 not applied; the unmutated instrument is green before and after (`logs/mutation_results.json`). The harness only counts a pytest FAILURE as a kill.
  * **End to end on a stub harness:** `subevals` for both seeds, then `subread`, reaches PASS (perfect read-out), HARMS (read-out off by 20 m/s) and REFUSED (wrong read-out sha; a `READ.json` row that does not reproduce).
* **Composition with the real base class (MEASURED, `logs/real_speedavg_base_check.txt`):** the hook subclass, built on the REAL `v8f_mini_eval_speedavg.SpeedAvgHook` from the SPEEDREAD run tree and driven on the fake model, hands O the `mlp_ego` cells to 4.4e-7 (float32) with `nonspeed_max_abs_dev` 0.0 and `tick_mirror_ok` true.
* **The banked read-out loads under the staged instrument (MEASURED 2026-10-28):** `readout.pt` sha256 `a205609a…`, `mlp_ego` is `ego_only`, `xm` is 22 wide, `predict_raw` runs on a 22-dim ego block and equals the original instrument's scene-path result bit for bit; the shuffle source is 2,399 windows with 0 fixed points.
* ⚠️ **Not done:** no real-data smoke. The harness `v8f_mini_eval.py run` has no window cap, so a two-window run would need a harness change this brief does not allow. Real-window reachability is proved by S7 / S7b / S10 and the preflight, in the first minutes of the run.
* The pinned original is stored under `tests/golden/` with its sha256 checked by a test; it is a reference copy, not a second instrument.

## 11. What this does not license, and disclosed design limits

* ⛔ No bar moves after data. The F/2 bar, the permutation seed, the tolerances and the table in §5 are fixed by this text. An amendment after promotion is appended below with its date and reason, and the body stays as promoted.
* ⛔ The read-out is never fitted, tuned or selected on eval-141. The choice of `mlp_ego` rests on TRAIN fold 0 (§0), with the disclosure above.
* ⛔ A PASS is a statement about two O training runs and one read-out fit at T1-class open loop. It is not a lever claim for any trained arm and not a driving result.
* ⚠️ **Limits:** (1) tick frame only, no ego history; (2) nav and the speed-limit sidecar ride in the ego block (§2); (3) the fit used the eval's age cycle (`a_t = i mod 3`, `a_s = a_t + (i div 3) mod 5`), not the trainer's random ages; (4) the `egoshuf` permutation is a global shuffle, so a shuffled window usually borrows cells from another episode; a harm that only shows against near neighbours would not be tested; (5) DK is not computed (§9); (6) `lin_ego` was fitted and is untested here, so the ego question is answered for one of two ego read-outs; (7) the follow-up's DAG base is unread.
* Amendments: none.
