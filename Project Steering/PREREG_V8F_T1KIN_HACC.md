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

## Amendment 1 (2026-10-26 ~00:30 Berlin): the DOMINANCE rows made computable, and a timestamp corrected. Landed before any real-data number was read; no bar moves

**What was found, from the instrument's output schema on a 24-window plumbing smoke (its numbers were not read):** `analyze()` emits only TWO paired HACC − P10 rows, both under `paired_decision_grade.paired_model_minus_floor_t1`: `ade_m` and `speed_mae_mps`. It reports the four families per arm, but not paired. So §5's "no row that `analyze()` emits for HACC − P10 is separated-worse" would have read two rows, while the same sentence promises the four families. The amendment resolves the conflict toward the stricter reading.

1. **The DOMINANCE rows are the four families' per-window components, computed by the canonical code on the executed tracks.**
   - Code: `taniteval/taniteval/four_families.py` (`longitudinal`, `lateral`, and `_seq_geometry` at dt = 0.2 s), with the same masks that code applies.
   - Inputs: the dumped executed tracks `cl` (HACC) and `fl` (P10) against the GT `g`.
   - Paired estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, on the 141 episodes.
   - Rows: ADE2, ADE6, and every scalar `longitudinal` and `lateral` return per window (speed, along-track, acceleration, heading, curvature, yaw rate, cross-track).
   - A component that is undefined on a window (an empty mask) drops that window from BOTH arms of that row; the row's n is reported.
2. **TACTICAL and STRATEGIC are NOT COMPUTED in this read.** The T1 dumps record no per-tick T code or S checkpoints. Each is reported with that reason and its n, as the binding four-family rule requires, and logged as a work item: a T1 dump of the per-tick code and checkpoints.
3. **Distance keeping** is computed if `longitudinal` accepts the eval join's lead track on this window grid. Otherwise it is NOT COMPUTED, with the reason.
4. **Reader controls, before any outcome row is read:**
   - R-1: the reader's per-arm ADE6 equals the arm's full-set ADE in `analyze()`'s output to 1e-6 m.
   - R-2: the reader's paired ADE6 point equals `paired_decision_grade.paired_model_minus_floor_t1.ade_m`'s point to 1e-6 m.
   - R-3: a known-value check on synthetic tracks. An identical track reads exactly 0 on every row, and a constant 1 m lateral offset reads cross-track 1 m and along-track 0.
5. **PRIMARY is unchanged:** HACC − P10 ADE6 at m = 10, point ≤ −2F and the interval excluding 0, at both seeds.

**Timestamp correction.** The header's "~00:45 Berlin" is wrong. The PREREG was written at about 00:20 Berlin and committed as `323624f2`, and the first run started at 00:24:55. The binding order is the commit, which precedes every run. The same wrong label is on `D-V8F-HACC-T1KIN`'s MOOT note.

## Amendment 2 (2026-10-26 ~00:55 Berlin): C-1's second clause cannot pass by construction; it is replaced by its age-free half. Landed before any outcome row was read; PRIMARY unchanged

**What was found** (on the 24-window plumbing smoke and the 17 cached-clip windows; no HACC − P10 row was computed):
- §4 C-1 assumed the T1 tool's tick-0 plan `ol` is the open-loop eval's plan. That holds only when the two use the same sample-and-hold ages.
- The banked open-loop eval runs `--ages cycle`: a_t = i mod 3, a_s = a_t + (i // 3) mod 5, per grid window i. The T1 tool's `--schedule periodic` starts every window at a_t = a_s = 0, and its CLI has no cycle schedule.
- MEASURED (`raw/2026-10-26-t1kin-hacc/ages_probe.log`), the baked `M2b_s0_hacc.pt` on all 17 windows of the cached eval clip:
  - with the eval's cycle ages, it reproduces the banked HACC plan to **6.4e-6 m**;
  - with ages 0, it equals the banked plan where the cycle ages are also 0 (5.4e-6 m), and differs everywhere else by a **median 0.71 m, max 2.81 m**.
- **Window alignment is correct.** On the smoke, the floor's first 10 knots equal HBLEND's banked P10 plans at the mapped grid windows to 3.8e-6 m.

**The amendment:**
1. **C-1 now reads:**
   - (i) at m = 30, `cl` = `ol` exactly (unchanged);
   - (ii) at m = 30, the floor `fl` equals HBLEND's banked P10 plan `P` (`raw/2026-10-25-hblend/score_M2b_s*_perwindow.npz`) at the mapped grid windows, to ≤ 1e-4 m, on all 2,399 windows. The floor ignores ages, so this is the age-free half of the original clause: it checks the window alignment and the state rebuild at tick 0.
2. **The original second clause is reported beside, not gated:** |`ol` − banked HACC plan| per window (max and median), with the measured mechanism above.
3. ⚠ **A consequence stated now: T1-kin − open loop (beside) mixes two effects,** self-action and the ages schedule (periodic from 0 against cycle). It is never quoted as the self-action effect alone. The PRIMARY (HACC − P10, both inside the T1 tool) is not affected: the floor reads no age.
4. C-2, C-3, C-4, R-1..R-3, PRIMARY, DOMINANCE and the outcome table are unchanged.
