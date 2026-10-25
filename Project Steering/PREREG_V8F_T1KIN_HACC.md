# PREREG_V8F_T1KIN_HACC: does HACC's margin over the P10 floor survive when v8F-mini lives on its own motion? (2026-10-26)

**Written before any T1-kin number on real cameras exists for any v8F-mini checkpoint.** Master Mind, 2026-10-26 ~00:45 Berlin.
Claim row: `H-V8F-T1KIN-HACC-1`. Instrument: `stack/scripts/v8f_t1_eval.py` (`D-V8F-T1KIN-1`, design `…/2026-10-17-v8f-mini/raw/2026-10-22-t1/DESIGN_T1_V8F.md`, 23 tests, 6 source mutations RED). Zero training.

## 0. Why now, and why no box 2

* `H-V8F-HACC-1` (MEASURED, T1-class open loop): HACC dominates P10 at both seeds. `PREREG_V8F_HACC.md` names the T1-kin read as the next tier, and said it "needs a PI go for the pooled eval cache".
* That go existed because the cache was believed to be on box 2 only (`DESIGN_T1_V8F.md` §9: *"the pooled DINOv3 eval cache is not on this machine"*).
* MEASURED 2026-10-26: the cache is on the dev box. It lives at `scratchpad/sfresh/cache/eval`, pipeline `ra1a-pooled-v1`, 141 / 141 clips, 0 md5 mismatches (`cache_md5_verify.json`), copied for SFRESH.
* ⇒ The read runs on the **dev-box CPU**, inside the PI's standing allocation (`PI-V8F-2026-10-25d`: *"keep everything on box 2 and the dev-box CPU"*). Box 2's queue is unchanged. `D-V8F-HACC-T1KIN` is answered by this venue and needs no box-2 slot.

## 1. Tier, and what this read can and cannot say

* **Tier T1, in `EVAL_DOCTRINE.md`'s vocabulary: T1-kin, self-action OPEN loop with a closed KINEMATIC state.**
  * The ego state the model reads (`p10f`, `base`, `pose_*`, and the route countdown in `nav`) is rebuilt from its own executed motion at every replan.
  * The camera frames stay logged; T2 (re-perception) is not provisioned.
  * ⛔ **Never "closed loop", never "the model drives".**
* **Two channels stay open, and both flatter the MODEL over the floor.** The floor reads neither.
  * The **camera stream**: logged frames are the human's view.
  * The **speed bucket** `limit`: a PI-declared input (`D-SPEEDBAND-EGO-ECHO-1`).
  * Both are bracketed in this same read (§3: `--scene frozen`, `--limit off`). They are reported, not deferred.
* **Estimator:** `taniteval/tools/t1_eval.py::analyze`, the paired episode-cluster bootstrap. B = 2,000; 141 episodes / 2,399 windows; eval-141, `kin_mode` yaw.
  * It answers *"another draw of EPISODES"*.
  * The two training seeds are the training replicate (`H-ESTIM-SEED-1`).
  * Inference is deterministic, since `forward_infer` does not sample.
* **F = 0.1207 m** (M2b's seed floor, `MODEL_REGISTRY.md` §17); 2F = 0.2415 m.

## 2. Arms (the same windows, the same executor)

| arm | what | checkpoint |
|---|---|---|
| **HACC** (`cl`) | v8F-mini M2b with the baked HACC post-map in its own loop | `E:\Projects\TanitAD-artifacts\v8f-mini-hacc\M2b_s{0,1}_hacc.pt` (sha256 `71de3c61…` / `32e4809d…`, `MODEL_REGISTRY.md` §17.1) |
| **P10** (`fl`) | the P10-yaw-4512 floor driven by the same executor (the checkpoint's own frozen prior) | inside the same file |
| **B** (`cl`, beside) | plain M2b (no post-map) in its own loop | the scored `M2b_s{0,1}.pt` (sha256 `601cb435…` / `6104fb15…`) |

**Settings** (the instrument's defaults, as designed): `--nav deadreckon --scene logged --limit logged --schedule periodic`.

**Replan period** `m` (0.2 s steps per plan):
* **10 is the deciding period.** It is the informative point (`DESIGN_T1_V8F.md` §9.2); there the floor alone already runs away, from 2.487 to 3.443 m.
* **30** is a control.
* **5** is a stress point, beside.

## 3. Runs (CPU, `CUDA_VISIBLE_DEVICES=-1`)

Per seed s ∈ {0, 1}:
1. HACC, m = 30 (control C-1 / C-2);
2. **HACC, m = 10 (PRIMARY / DOMINANCE)**;
3. HACC, m = 5 (beside);
4. HACC, m = 10, `--scene frozen` (bracket: the camera channel closed, doctrine-literal "perception fixed at t0");
5. HACC, m = 10, `--limit off` (bracket: the speed bucket removed);
6. B, m = 10 (beside: attribution of the post-map in the loop).

## 4. Controls, read BEFORE any number of §5 (a failed control stops the read: `REFUSED`, no outcome row)

* **C-1. m = 30 is open loop.**
  * `cl` = `ol` exactly. This is the instrument's C2, re-checked on real data.
  * HACC's `ol` per-window ADE6 equals the banked HACC per-window ADE6 (`raw/2026-10-25-hacc/run/score_M2b_s*_perwindow.npz`) to ≤ 1e-3 m, which covers the CPU-vs-banked float noise. If the banked file has no per-window ADE6, the comparison is made on plans recomputed against the eval GT, same bound.
* **C-2. The floor reproduces its banked open-loop number.** At m = 30, `fl` ADE6 has mean 2.48699 m to within 1e-5, matching `DESIGN_T1_V8F.md` §8.
* **C-3. The post-map is in the loop.** At m = 10, the executed tracks of HACC and B differ: max abs > 1e-2 m at both seeds. A zero here means the baked map never reached the loop.
* **C-4. The instrument's suite** (`stack/tests/test_v8f_t1.py`, which includes the C1 bit-identity, the C4 leak guard and its mutant) passes on the tip used for the runs.

## 5. Read, and the outcome table (applied verbatim; no bar moves after data)

**PRIMARY (per seed):** HACC − P10 at m = 10 on the 6 s ADE of the EXECUTED track against the 30 GT waypoints, paired. The bar is the programme's model-vs-floor bar `H-MINI-K-1`: **point ≤ −2F = −0.2415 m AND the 95 % interval excludes 0**.

**DOMINANCE (per seed):** PRIMARY, AND **no row that `analyze()` emits for HACC − P10 at m = 10 is separated-worse**.
* This covers the four families' rows (longitudinal, lateral, tactical, strategic as available).
* A row that `analyze()` cannot compute is listed with its reason and its n. It neither passes nor fails.

| outcome | condition | next (committed) |
|---|---|---|
| **T1 DOMINATES** | DOMINANCE at both seeds | HACC's margin survives self-action at T1. The brackets (§3 runs 4–5) are reported beside as how much the two open channels flatter it, per seed. The evidence video is re-rendered on the T1 executed tracks. The A40 case (`D-V8F-SPEED-PERCEPTION`) is restated with AR-1 at T1. |
| **T1 BEATS THE FLOOR** | PRIMARY at both seeds, but some row separated-worse at either seed | Report the worse rows by family. That family is the next lever's target, under its own PREREG. When it is acceleration or curvature, the first lever is HACC's λ rule re-fitted on T1 executed tracks of TRAINING windows. |
| **MARGIN COLLAPSES** | PRIMARY fails at either seed | The open-loop margin does not survive the model's own motion. Run 6 attributes it: if B − P10 also fails, the cause is O itself; if B passes and HACC fails, it is the post-map. Next: the attribution's arm, under its own PREREG. |
| **MIXED** | anything else | Reported per seed; seeds before windows. |

**Beside (decides nothing):**
* m = 5 and m = 30;
* B − P10 and HACC − B at m = 10;
* both brackets as HACC − P10 and as bracket − default;
* envelope-clipped windows per arm and m;
* executed accel / jerk extremes;
* T1 − open-loop per arm.

## 6. What this read does NOT license

* ⛔ Any "closed loop" or "drives" wording (§1).
* ⛔ A SOTA claim: the comparator is the programme's own kinematic floor.
* ⛔ Re-choosing m, the bar, or the settings after a number is seen. The PRIMARY is m = 10 with the instrument's defaults, as designed on 2026-10-22, before this checkpoint existed.
