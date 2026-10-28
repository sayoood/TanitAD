# FINDING 1: attribution refits of the banked ego-only read-out (PREREG_V8F_SPEEDSUB section 6, "Required first")

Agent: Sonnet, 2026-10-28. CPU only, 4 threads, no GPU, no eval window read, no substitution run. Nothing staged in the repo.
Scratch dir `F:/TanitAD-scratch/speedsub_t1_prep/` (the refit read-outs live there; this dir holds the findings and copies of the small artifacts).

## 1. Answer

**The read-out's skill survives without `nav`. It does NOT survive without `limit`, and `limit` is the logged speed bucket, which is a future-speed oracle.**

| MLP design (TRAIN fold-0 holdout, mean of 5 cells, SD units, lower is better) | dim | mean RMSE (SD) | mean skill | 3-seed range of the mean |
|---|---|---|---|---|
| `mlp_ego` as banked (`fit.json`) = full 22-dim block | 22 | **0.24751** | 0.9350 | 0.00045 |
| (b) `p10f + limit + limit_ok + frame_ok` (no `nav`) | 17 | **0.26313** | 0.9259 | 0.00133 |
| (c, extra) `p10f + nav + frame_ok` (no `limit`) | 17 | 0.28032 | 0.9156 | 0.00087 |
| (a) `p10f + frame_ok` (neither) | 12 | **0.30849** | 0.8969 | 0.00027 |
| for scale: banked `lin` (scene+ego ridge) / `lin_ego` / `mlp` (scene+ego) | | 0.27236 / 0.28071 / 0.29196 | 0.9208 / 0.9158 / 0.9125 | |
| persistence (cell := measured speed `v`, no fit) | | 0.40898 | 0.8267 | |
| CONST (train mean) | | 1.02519 | 0.0000 | |

Evidence class: MEASURED. Artifact `F:/TanitAD-scratch/speedsub_t1_prep/refit_all.json` (sha256 `0f830665…`), `refit_seeds.json` (`9094a3c7…`), table `REFIT_TABLE.md` (`a69dc635…`), `refit_summary.json` (`2bbfa95a…`). Skill = 1 - MSE/MSE(train-mean CONST), per cell, on the 57,479 fold-0 rows (846 clips); mean over the 5 cells.

* **`nav` removal** costs +0.0156 SD (seed 0), +0.0166 (3-seed means 0.24725 -> 0.26382), against a seed range of 0.0004 / 0.0013 on the two means: real, ~12x the MLP seed floor, and small next to what `limit` carries. The no-nav read-out (0.26313) still beats all three other banked read-outs on fold 0 (best of them: `lin` 0.27236).
* **`limit` removal** costs +0.0327 SD (3-seed, from the full block) and +0.0445 (from `p10f + frame_ok`). Adding `limit` to design (a) closes 73 % of the (a)-to-full gap; adding `nav` first closes 46 %.
* **Ridge twin (`lin_ego`)**: full 0.28071, no-nav 0.28273 (+0.0020), no-limit 0.31899, neither 0.32437. For the linear read-out essentially all of the non-`p10f` skill is `limit` (0.0416 of 0.0437).
* **Per cell (MLP, SD units, full -> no-nav -> neither):** `lon_v_target` 0.329 -> 0.352 -> 0.413; `v_a` 0.1036 -> 0.1054 -> 0.1112; `v_end` 0.329 -> 0.356 -> 0.426; `v_lo` 0.231 -> 0.255 -> 0.295; `v_hi` 0.244 -> 0.248 -> 0.297. `v_a_ms` is almost entirely measured speed (the current speed): all designs sit within 0.01 SD there.

## 2. Beside `T_own` (same cells, m/s)

`T_own` exists only on the eval-141 windows (`READ.json`, sha256 `46543c39…`); the refits' holdout is TRAIN fold 0. Different populations, so the juxtaposition is indicative. Check that it is not wild: `mlp_ego`'s own fold-0 RMSE vs its eval-141 RMSE differ by +1.0 %, +6.3 %, -0.3 %, -1.2 %, +5.4 % per cell (derived from `fit.json` x `train_sd` and `READ.json`).

| RMSE m/s, cells `lon_v_target, v_a, v_end, v_lo, v_hi` | |
|---|---|
| `T_own` eval-141 s0 | 2.894, 1.919, 2.795, 2.257, 2.093 |
| `T_own` eval-141 s1 | 2.747, 1.302, 2.692, 2.143, 1.802 |
| `mlp_ego` fold 0 (= control refit) | 2.566, 0.799, 2.548, 1.838, 1.806 |
| (b) no-nav fold 0 | 2.744, 0.812, 2.754, 2.023, 1.834 |
| (c) no-limit fold 0 | 2.901, 0.833, 2.952, 2.017, 2.117 |
| (a) `p10f + frame_ok` fold 0 | 3.222, 0.857, 3.298, 2.346, 2.194 |

Cells where the fold-0 RMSE is below `T_own` (direct, cross-population): `mlp_ego` 5/5 (s0), 4/5 (s1); (b) 5/5 (s0), 3/5 (s1: `v_end` and `v_hi` are above); (c) 2/5, 2/5; (a) **1/5, 1/5 (only `v_a`)**. The ESTIMATED eval-141 form (mlp_ego's eval RMSE x the fold-0 ratio of each design; ESTIMATED, assumes the transfer ratio is design-independent, not an eval) gives the same counts for (b) 5/5 and 4/5, (c) 4/5 and 2/5, (a) 2/5 and 1/5 (`refit_summary.json` `ESTIMATED_eval141_vs_T_own`).

**Reading.** A read-out of the ego dynamics alone (a) does not beat `T_own` in four of five cells. What beats it is `limit` (and, less, `nav`).

## 3. Why `limit` matters for the follow-up (SOURCE-READ, INHERITED claim re-read at file:line)

* The logged `limit` bucket is the bucket of `SPEED_BAND.v_hi_ms`, "the max of the ego's realised future speed", R^2 0.97 recoverable: `Project Steering/GOALS_AND_CLAIMS.md:11518` (`D-SPEEDBAND-EGO-ECHO-1`, "the PI has decided to feed it anyway, with declaration"), `TanitAD Research Lab/.../raw/2026-10-22-t1/DESIGN_T1_V8F.md:62`, and the T1 tool's own header (`stack/scripts/v8f_t1_eval.py:563` lists "speed bucket from the human's realised future max speed" as an OPEN channel when `--limit logged`). PI_DECISION_QUEUE.md:2525 asks the PI whether v8F's T1 headline should be stated on it.
* T also reads `limit`, so the read-out is not more privileged than T (as SPEEDSUB section 2 says). But the attribution question the section asked ("a route-supply result?") has a different answer than expected: the dominant carrier is the future-speed oracle, not the route.
* **The T1-kin lane uses `--limit causal`** (DAG-OACC and DAG-ORC rollouts; finding 3). There the `limit` input is `speed_max_bin(max measured ego speed over rows [j-20, j])` (`v8f_t1_eval.py:261-272`), a backward-looking bucket, not the sidecar the read-out was fitted on. The fitted weights on `limit` were learned against the oracle. Under `--limit causal` the read-out's strongest input changes meaning, and the expected skill sits between design (a) and the full block. A read-out fitted for the causal lane is therefore a separate fit: `train_arrays.npz` stores the logged `limit` (and `win` = clip, t, a_t, a_s for each window, but no speed history), so the causal bucket would have to be recomputed per window from `native_tracks.pt` (an extract-like step over 306,566 windows; not done here, not checked that the tracks file covers every train clip).

## 4. Control (run first; every number below is MEASURED)

The full `mlp_ego` design, refitted through my code path (`refit_attribution.py`: the SPEEDREAD A1 instrument's own `standardiser`, `fold_of`, `mlp_fit`, `ridge_cv`, `ridge_fit`, sha256 `fd5ce0be…`, imported from `F:/TanitAD-scratch/speedread/tree_f3971165`), reproduces `fit.json`:

| check | deviation | note |
|---|---|---|
| `mlp_ego` best epoch | 8 = 8 | equal |
| `mlp_ego` holdout RMSE (SD), 5 cells | max abs 3.99e-9 | |
| refit MLP state vs `readout.pt` `mlp_ego`, every tensor | max abs 4.77e-7 | float32 summation order at 4 threads |
| refit vs banked holdout predictions | max abs **9.41e-6 m/s** | under 1e-5; thin margin |
| `ym`, `ys`, `xm`, `xs` | exactly 0 | |
| `lin_ego` lambda per cell, CV table, holdout | equal; 5.3e-14; 9.9e-15 | ridge is deterministic |
| `lin_ego` final W / b vs banked | 2.4e-7 / 1.2e-14 | |
| inputs vs `fit.json`: `train_sd`, `const`, `n_present`, `mlp_cfg`, `lambdas` | 0.0, 0.0, equal, equal, equal | n_windows 306,566, n_clips 4,512, fold 0 = 57,479 rows / 846 clips |

**Tolerance, stated honestly:** the instrument has no refit-reproduction tolerance. I used its prediction tolerance (SPEEDSUB S7b, `TOL_OWN_BANKED = 1e-5`) as the bar on predictions, and report every deviation above. The original fit ran at torch's default thread count, this refit at 4, so float32 noise of 1e-7 is expected and was found. Layout control: the 22 columns are `p10f` 0:8, `nav` 8:13, `limit` 13:17, `limit_ok` 17, `frame_ok` 18:22; verified on the arrays (limit one-hot with row sum in {0,1}, `limit_ok` iff a one-hot is set, `frame_ok` binary, `nav` token one-hot, `nav[:,4]` in [0,1]) and against `v8f_mini_data.py:151-157,203-210`. Known-value control: the CONST row reads skill 0.0000 on all five cells.

**Noise floor:** the same MLP at init/order seeds 1 and 2 (`refit_seeds.py`): range of the 5-cell mean is 0.0003 to 0.0013 (table above). This answers "would another MLP seed say this?" for the ranking (a) > (c) > (b) > full: it does. It does not answer "would another fold say this?" (one holdout fold, 846 clips) and it is not an eval.

## 5. Deliverables and limits

* Refit read-outs (F: scratch only, same container as `readout.pt`, plain tensors): `readout_refit_{full,a_p10f_fok,b_nonav,c_nolimit}.pt`, each holding `mlp_ego_<d>` and `lin_ego_<d>`; sha256 in the manifest. `full` is a control copy and is NOT `readout.pt` (weights differ at 4.8e-7).
* ⛔ **Not usable by `subevals` as they are.** The reduced read-outs have `xm/xs` of width 12 or 17; `ego_vectors()` hands them 22 columns. The column map is in `refit_all.json` `designs.<d>.cols`. A substitution needs a `cols` field in `ReadOut` (about 5 lines). Not built: it would be an instrument change and the brief forbids running any substitution.
* One holdout fold, one MLP architecture, tick-frame features only (SPEEDSUB limit 1), ages as the eval's cycle. The attribution is by ablation of a trained read-out family, not by a gradient method; interactions between `nav` and `limit` are visible in the 2x2 (the marginal costs differ by order) and are reported both ways.
