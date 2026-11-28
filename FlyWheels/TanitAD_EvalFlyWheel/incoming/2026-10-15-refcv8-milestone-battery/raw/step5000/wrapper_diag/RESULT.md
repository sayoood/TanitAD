# refcv8 step-5,000 G0 — the wrapper-clause FAIL and the one DETECTION miss, diagnosed

EvalFlyWheel, 2026-10-16, dev box, **CPU only** (OMP_NUM_THREADS=6, no GPU lock taken). Ckpt `E:/refcv8_eval_kit/ckpt/ckpt_5000.pt`
(md5 `a32e3b70…`, strict load clean), launch tree `C:/lgs/Lfb47a58376`, refcv8 battery code copied from
`hold_r8batt/.../code` (md5-equal). Batch = the probe's own 4 rows (perm sha256 `99adbc7e…` == G0's; windows 20658, 368, 10637, 7314).

## Verdict

| question | answer | evidence class |
|---|---|---|
| Q2: merge defect or batch coupling? | **Neither.** The probe's arms run on **different DDIM noise per row**. With all of refcv8's eval draws held fixed per row, micro-batched, permuted and row-alone forwards reproduce the plain forward to ≤ 3.4e-6 on every output and every loss term. | MEASURED (CPU) |
| Root cause | `wrapper_probe_r7.py` neutralises noise only through `torch.randn_like` → zeros. refcv8 draws its noise at three other sites, so nothing is neutralised. **This is an instrument defect, not a model defect.** | MEASURED + source |
| Q3: battery numbers readable? | **Yes, with the PROVISIONAL stamp kept.** Rolls do not micro-batch, and refcv8 has no cross-row coupling. Each row's plan is a valid draw of the stochastic planner, from the same distribution as a batch-1 deployment. No roll fix is needed. | MEASURED (coupling) / source (roll path) |
| Q4: `eval_box3d_det_ap2_automobile_0_20` | **Deterministic cross-platform numerics** is the most likely cause (UNVERIFIED). It is not noise and not coupling. The miss is 1.16× tolerance. | MEASURED (not noise, not coupling); attribution UNVERIFIED |
| wrapper clause after the fix | **PASS on CPU**: max wrapper rel 3.36e-6 against bar 1e-5, W1 and W2 both detected. A GPU re-run of the probe is still owed. | MEASURED (CPU) |

## Q1 — What the clause measures (`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/wrapper_probe_r7.py`)

* **Batch:** the first 4 windows of in-run batch 1. `torch.randn_like` is patched to `zeros_like` for every arm (`:140-141`).
  `torch.manual_seed(0)` runs before each arm (`:81`).
* **Arms** (`:78-99`): `plain` / `repeat` (4 rows), `permuted` (rows reversed), and `micro13` / `micro31` (`MicroBatchForward` sizes [1,3] / [3,1]).
  W1 and W2 are deliberate merge regressions.
* **Statistic:** the scalar LOSS TERMS that `compute_losses_v3` returns (`_scalars`, `:53-60`). They are not output tensors.
  `cascade` = `extra["cascade"]` (`stack/scripts/refc_v3_train.py:5868-5877`). It is the per-stage matched-anchor L1 + CE over `layer_u0_hat` / `layer_logits`.
  `r8_subscore` = `tele["r8_subscore"]`, the masked BCE of `r8_sub_logits` over the emitted fan (`stack/tanitad/train/refcv8_train.py:708-713`).
* **How each statistic is computed** (`:102-111`):
  * `floor = max rel(plain, repeat), rel(plain, permuted)`
  * `wrapper = max rel(plain, micro13), rel(plain, micro31)`
  * `bar = max(3·floor, 1e-5)`
* **The clause** (`:178-188`): judged under P3 (fp32, deterministic). It FAILs if any term has `wrapper > bar` while W1 and W2 are both detected.
* **Both reductions sit downstream of `MicroBatchForward`:** `layer_u0_hat`, `r8_sub_logits` and `anchor_traj` are batch tensors merged by `cat`. The loss then reduces over the merged 4 rows. So the `wmean` / `shared` merge rules (`microbatch.py:104-142`) never touch these two terms.

## Q2 — The discriminating experiment (`r8_wrapper_diag.py`, `r8_wrapper_diag2.py`)

**Where refcv8 draws its noise.** None of these sites is reached by the probe's patch:

| site | location |
|---|---|
| base eps | `torch.randn((b, _nb)+…)` at `stack/tanitad/refs/refc.py:3036`. refcv7 used `torch.randn_like` at `:3030`. |
| extras eps | `r8_gen.randn_eval` at `refc.py:3042` |
| source-anchor choice | `gen.rand_eval` at `refcv8_conditioning.py:446` |

The extras eps and the source-anchor choice come from a fresh generator **seeded by hashing the global RNG state** (`refcv8_conditioning.py:187-208`). A row's noise therefore depends on its position in the call and on what the call drew before it.

| arm (4 rows, P3 fp32, CPU) | per-row outputs, max rel over ALL keys | `cascade` rel | `r8_subscore` rel |
|---|---|---|---|
| **control** `zero_seed1` vs `zero_plain` (noise fully neutralised ⇒ must be identical) | **0.0** | 0.0 (471/471 terms bit-equal) | 0.0 |
| **control** `live_repeat` vs `live_plain` (determinism) | **0.0** | 0.0 (471/471 bit-equal) | 0.0 |
| `zero_micro13` / `zero_micro31` | 3.2e-6 / 4.1e-6 | 3.9e-7 / 5.9e-7 | 0 / 0 |
| `zero_perm` | 2.1e-6 | 0 | 6.4e-8 |
| `zero_row0..3` (each row ALONE, batch 1) | ≤ 2.9e-6 | — | — |
| live noise **replayed per row**: `rp_micro13` / `rp_micro31` / `rp_perm` | 2.9e-6 / 2.9e-6 / 2.5e-6 | 1.8e-7 / 6.1e-8 / 4.2e-7 | 6.2e-8 / 0 / 0 |
| live noise replayed, each row alone `rp_row0..3` | ≤ 3.3e-6 | — | — |
| **noise redraw only** (plain, seeds 1..7 vs seed 0, no wrapper) | up to 1.3 | median 7.3 %, max **22.7 %** | median 1.75 %, max **3.5 %** |

Sources: `raw/diag_summary.json`, `raw/diag2_summary.json`. The clause rule re-applied to the zero-noise arms gives 0 terms over bar out of 367, with max wrapper 3.4e-6.

**How to read the table:**
* The control (the trunk / perception keys and the 471/471 bit-equal pairs) and the rows-alone arms rule out cross-row coupling. Those arms show no cross-row effect in normalisation, allocation, listwise selection or softmax.
* The replayed-noise arms rule out a merge defect in exactly the terms that failed.
* The probe's CUDA numbers sit inside the plain redraw spread:
  * `cascade`: wrapper 12.35 % and floor 2.88 % (`g0.json` `wrapper_control.conditions.P3_fp32_det`)
  * `r8_subscore`: wrapper 1.64 % and floor 0.46 %

  The FAIL is one redraw landing more than 3× another.

**Two further points:**
* The unfixed bar has **no power** on refcv8's planner terms. On GPU P3 the bars are 8.6 % for `cascade`, 11 % for `traj` and 320 % for `r8_alloc_l1`. For comparison, refcv7 step 5,000 under the same probe had floor 0.0 and wrapper 1.5e-6 on `traj` (its `raw/step5000/g0.json`).
* **W1's "detection" of `cascade` / `r8_subscore` is inherited.** W1 == micro13 on every term except `goal_score_absmean` and `map_hires` (GPU P3: 469/471 equal; CPU: all but `goal_score_absmean`). W1's real power comes from `goal_score_absmean` alone.
* UNVERIFIED: on CPU the generator-less base eps is split-invariant (unreplayed `live_micro13` `cascade` 1.8e-7). On CUDA it evidently was not (12 %). This is consistent with per-call Philox offsets, but it was not run on GPU.

## The fix (`wrapper_probe_r8fix.py`, validated by `validate_fix.py` → `raw/validate_fix.json`)

1. **Neutralise all four draw sites for the probe:**
   * `torch.randn_like` → zeros
   * generator-less `torch.randn` → zeros (seeded-generator calls such as `refc_v3.py:3122` are left untouched)
   * `R8Generator.randn_eval` → zeros
   * `R8Generator.rand_eval` → 0.5

   Drop-in: call `wrapper_probe_r8fix.probe(...)` in place of `W.probe(...)` at `g0_refcv7.py:1224`, or set `wrapper_probe_r7.probe = wrapper_probe_r8fix.probe` in `g0_refcv8.py` before `G.main()`; the original probe is captured at import, so this cannot recurse. This is a G0 instrument change, so the Master Mind must register it as an amendment before the re-run. It implements the probe's own stated condition, "DDIM eps zeros (every arm)".
2. **The real probe on CPU, P3, same 4 rows:**

   | probe | clause | max wrapper | bars |
   |---|---|---|---|
   | unfixed | PASS by luck | — | 0.04–0.72 |
   | fixed | **PASS** | **3.36e-6** | back to 1e-5 on `cascade` / `r8_subscore` / `traj` / `r8_*` / `loss`; W1 and W2 detected |

3. **Also exclude `r8_spd_crc` from the clause's terms**, as `goal_gate_grad` already is. It is a CRC of the batch's speed channel (`refc_v3_train.py:6576`). Its 6.37e-2 "floor" under permutation is a hash change, identical on GPU and CPU.
4. **Blocker:** the GPU P3 re-run of the fixed probe (~1 min of GPU) waits for the dev-box GPU lock. That re-run is what flips the registered `wrapper_clause`.

## Q3 — Consequence for the battery

* **The rolls never use `MicroBatchForward`.** Only the `g0_*` modules import it.
* **Roll batch size is 1–3 rows:** the nav conditions of ONE window (`conds_fed = nav_true [+ nav_shuffled] [+ nav_flipped]`, `taniteval/tools/refcv3_arm.py:2106-2117`, `b = len(conds_fed)` at `:2260`). There is also a 1-row nav-null call. `os_filteroff` / `vmaxzero` / `obey30` run under forked or replayed RNG (`r7_roll.py:171-180`, `:227-234`).
* **The battery does score noise-dependent outputs.** `traj` and the selection come from the DDIM fan. With no coupling, each row's plan is a valid draw. Rows of one call get different noise, as they did in refcv7 (`randn_like` per row).
* **Consequence:** numbers at batch 1–3 reproduce the model as deployed in distribution, not draw-for-draw. The 2 inference seeds carry that floor.
* **No battery fix is needed.**

## Q4 — `eval_box3d_det_ap2_automobile_0_20`: in-run 0.33835, replay 0.31453 on all 24 seeds

| check | result |
|---|---|
| support and tolerance | n = 97, tol max(0.02, 2/97) = 0.0206, dev 0.0238 (`g0.json verdict.terms`) |
| inference noise | **ruled out.** sd 0 across 24 seeds. On CPU, all 412 perception loss terms are bit-identical across 10 arms with zero and live noise at seeds 0–7. |
| coupling | **ruled out.** Perception per-row ≤ 2.6e-6 in every CPU arm. GPU P3 micro vs plain on `box3d` is 3.5e-7. |
| numerics sensitivity | the term is numerics-sensitive. Dev-box bf16→fp32 alone moves it +0.0046 (A6 `fp32_s0` 0.31912). bf16 makes `box3d` loss partition-sensitive (P1 wrapper 0.55 % vs P3 3.5e-7). |
| platform difference | Thor ran `--trunk-compile` + bf16 at batch 16. The replay is eager bf16, micro-batched 2,2,3,3,3,3. |

**How refcv7 compares:**
* Same SPEC DETECTION class and A2 low-support rule. A6 explicitly excludes DETECTION (refcv7 `SPEC.md:172, 253`).
* The same key at refcv7 step 5,000 read dev 0.0014, OK.
* **refcv8's perception drift is broader:**
  * DETECTION median 7.6e-4 vs refcv7 1.75e-4
  * MATCHED median 2.4e-3 vs 6.4e-4
  * the in-run is above the replay on 25 of 36 gating box3d `det_ap` keys and on every automobile sibling

  So a single threshold flip is not the whole story.

**Classification:** deterministic Thor-vs-dev-box numerics on a greedy-match metric. This is UNVERIFIED: the measured dev-box floor explains about 19 % of the gap.

**Discriminator (needs Thor, PI/Master Mind):** a seed-0, perception-only replay of the 128 G0 windows on Thor, once with `--trunk-compile` on and once with it off.
* If compile-ON reproduces 0.338 → numerics.
* If it reads about 0.315 → a data/loader difference between the in-run eval set and the replay. That would be a defect.

## Deliverable manifest (all under `C:/Users/Admin/qland/work/r8_wrapper_diag/`, unstaged — Master Mind lands it)

* `RESULT.md`, `HANDOFF.md`
* `r8_wrapper_diag.py`, `r8_wrapper_diag2.py`, `analyse_diag.py`, `summarise_diag2.py`, `validate_fix.py`
* `wrapper_probe_r8fix.py` (the fix)
* `raw/diag_arms.json`, `raw/diag_summary.json`, `raw/diag2_arms.json`, `raw/diag2_summary.json`, `raw/validate_fix.json`, `raw/*.log`, `raw/arms/*.pt`
* `code_r8/` (copied battery code)
* `cache/batch4.pt` (135 MB, the 4-row batch; not for banking)
