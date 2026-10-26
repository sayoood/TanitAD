# L4 · refcv9 fed speed-ceiling relabel: design memo

2026-10-26 · Data Engineering FlyWheel for the Master Mind · **0 GPU, CPU only, no training, no label release** · evidence MEASURED unless stamped · cause table: `L4_CAUSE_DECOMPOSITION.md` · raw: `raw/*.json` · code: `code/*.py`

## 0. Answer

* ⛔ **The brief's premise is wrong, and the correction changes the question.** The fed ceiling is **not** a VLM-CoT band edge. It is the **ego's own realised max speed over [anchor+2 s, anchor+6 s]**, taken once per clip at raw 8.0 s and broadcast to every window of the clip. Source: `s2_geom_emit_v7.py:51,207-211,309-311`, snapped by `build_refcv6_speed_max_window.py`, `refcv6_max_speed.py:91,166-172`. The v9 manifest stamps `SPEED_BAND: "geometry"`. ⇒ **Under the brief's own rule ("an input may not be derived from future ego"), refcv8's current ceiling is itself inadmissible.** It runs on the PI's 2026-10-11 exception. Its leak is **0.224** recovered on train (my instrument). D4 banked **0.238** for the same sidecar (`refc_v3_train.py:977`, INHERITED), against the 0.05 bar.
* **Cause decomposition of the 595/4,754 (12.52 %):**
  * **(ii) temporal broadcast: 579 (97.3 %).**
  * (i) snap, i.e. the >120 km/h clamp: 16.
  * (iii) human above a correct posted limit: not measurable at scale (see §1).
  * (iv) missing band: 0.
  * Train: 75,082/607,899 = 12.35 %, of which (ii) 70,471 and (i) 4,611.
  * The snap is UP and protective. The no-snap raw edge would leave only **41.0 %** compliant.
* ⭐ **RECOMMENDATION: C7, the v9 past-only `speed_n2_kmh` fed through the existing 4-way containing window.**
  * Compliance **96.87 %** eval (4,605/4,754 windows, 139/139 episodes) and **95.52 %** train.
  * Leak **−0.009 / 0.017** recovered (eval / train).
  * Past-only verified: a past-only rule reproduces the release column on 27,465/27,664 rows (99.28 %), and it is invariant to scrambling every future sample.
  * **Zero new data**: the column exists in both v9 releases.
  * **Zero new model code**: `--r8-speed-input n2` exists (`refc_v3_train.py:4372`, `refcv8_train.py:79,87,266-289`).
  * It needs **one PI ruling** (P1).

## 1. Why (ii) dominates, and why (iii) cannot carry the answer

* **One 4-s sample of the human's speed is fed as the ceiling for a ~20-s clip.** Every non-clamp exceedance has its max-speed instant outside the label band (an identity).
  * Exceeding windows by position: before the band 151, overlapping 265, after 179.
  * Steps above the fed class: one step 541, two steps 11.
  * Excess p50 **3.5 km/h**, p90 13.8.
  * 30 km/h class: **374/1,612 = 23.2 %**, which reproduces the amendment.
* **The leak mechanism is position-dependent** (`raw/l4_checks.json`). On windows *before* the anchor the clip label reveals the future: recovered **0.33–0.45** on train, 0.19–0.33 on eval. After the anchor it is ≈ 0. That is why pooled eval (0.030) and pooled train (0.224) differ. Train is the decision-grade number (n = 607,899 windows, 4,572 clips).
* **(iii) A posted limit would make compliance WORSE, not better.** PhysicalAI-AV has no map and no GNSS, so nuPlan limits **cannot be geo-joined**. No geographic overlap is establishable. The only independent reference is the CoT sign reading: 57/4,729 clips, ungrounded 0/57, ego-coupled (SL-SRC-1, INHERITED). On the 36 train clips that carry a unit-stated reading (4,784 windows):
  * the human is at or below the sign on only **52.1 %** (61.1 % with the sign snapped up);
  * the current label reaches 79.8 % on the same windows;
  * 799 of 968 train exceedances there are a human above the sign.
  * ⇒ A legal-limit ceiling and "~100 % compliant against the logged human" are **in tension by construction**.

## 2. Candidates (strict `gt_max_speed_6s ≤ ceiling`)

Sources: `raw/l4_candidates.json`, `raw/l4_extra.json`, `raw/l4_c7a.json`, `raw/l4_tightness.json`.
* **Leak** = D4 "recovered": 5-fold clip-grouped OOF OLS, target max v over [NOW+2, NOW+6] s, base [1, v0], on train (n = 607,899).
* **Leak controls:** constant **0.000**; clip-mispaired label **−0.000**; N2 8-step **0.030** vs D4's 0.0346 (INHERITED).
* **Slack** = ceiling − human max on compliant eval windows. It is the counterweight, because an all-130 control is **99.73 %** compliant at a slack p50 of 91 km/h.

| id | candidate | eval / train compliance | leak (train) | future ego? | slack p50 eval | class shift (eval windows) | refcv8 bars |
|---|---|---|---|---|---|---|---|
| C0 | current: clip anchor [2,6] s, snap-up 4-way | 87.48 % / 87.65 % | **0.224** | **yes** | 13.6 | 30:1612 50:1673 100:1231 120:238 | as registered |
| C1 | no snap, raw `v_hi` | 40.98 / 40.77 | 0.465 | yes | n/a | continuous | refused by loader (`refcv8_train.py:135-145`) |
| C2 | 8-step road-law snap-up | 81.89 / 83.47 | 0.363 | yes | n/a | 8 classes | checkpoint break |
| C3 | drop the 30 step {50,100,120} | 95.12 / 94.63 | 0.131 | yes | 21.2 | 30→50: 1,612 | re-roll, no ckpt break |
| C4 | add 140 (no clamp) | 88.20 / 89.42 | 0.230 | yes | n/a | 120→140: 34 | checkpoint break |
| C5 | per-window own [2,6] s snap-up | 94.11 / 93.38 | **0.397** | yes (echo) | n/a | small | re-roll |
| C6 | clip-span max snap-up | **99.10 / 98.08** | 0.141 | yes | 19.6 | 30:685 50:2157 100:1606 120:306 | re-roll, no ckpt break |
| **C7** | **past-only N2 → 4-way (>120 keeps value)** | **96.87 / 95.52** | **0.017** | **no** | 22.7 | 30:0 50:3020 100:1491 120:209 130:34 | re-roll; OOD for refcv8 |
| C7b | past-only N1 (no urban floor) | 92.74 / 88.77 | 0.032 | no | n/a | keeps 30: 1,100 | re-roll |
| C7a | C7 + one step up if a0>0.25 m/s² (fit on train) | 98.93 / 97.92 | 0.049 (base +a0) | no | **33.8** | lifts 27 % of windows | re-roll |
| C8 | max(C0, C7) | 98.21 / 97.92 | 0.063 | yes | n/a | n/a | re-roll |
| C9 | vision head (design only, §4) | HYPOTHESIS | n/a | no | n/a | n/a | needs a model-free mask |
| T | keep C0, score with +10 km/h | 98.02 / 97.19 | 0.224 | yes | 22.1 | none | 0-GPU re-read; **goalpost for refcv8** |

Notes on the table:
* **Leak audit, categorical first:** "could this value have been computed from the ego's future speed?" is **yes** for C0–C6 and C8, so they are inadmissible under the brief's rule.
* C7, C7b and C7a read only rows ≤ NOW (C7a adds IMU a0 at t0, which is admissible). Their D4 numbers measure legitimate past information and are not an echo.
* ⚠️ The D4 metric measures predictive information, not provenance. With a0 added to the base, C7 reads 0.050. Provenance decides admissibility; "recovered" only sizes it.
* **"What it does to refcv8's bars":** the compliant set would grow from 87.48 % (4,159 windows, 138 episodes) to X:
  * C7 **96.87 %** (139 episodes);
  * C6 99.10 %;
  * T 98.02 %.
  * Only **T** re-reads the banked dumps at 0 GPU. Every relabel changes a model INPUT and needs a re-roll.
  * C7 is out-of-distribution for refcv8, which never saw a ladder without the 30 class. **refcv8's verdict stays on A-R8-CLAMP-2; C7 is a refcv9 label.**
* **Tolerance is a goalpost move for refcv8**, because the strict rule was registered 2026-10-18 and data has since been seen. For refcv9 it is admissible only if fixed **now**. It also leaves C0's 0.224 leak untouched.

## 3. Recommendation: C7, and why not the others

1. **Admissibility.**
   * N2 = snap-up-8 of max(v over [NOW−20 s, NOW]) with an urban floor of 50 km/h. It reads no sample after NOW: reproduced at 99.28 % by an independent past-only rule, and future-scramble invariant 27,664/27,664.
   * Its inference input is ego speed **history** (IMU/odometry), not vision. That is P1.
   * It never carries the situation classifier's output.
2. **Compliance per unit slack.** Among admissible options C7 dominates C7b (+4.1 pp at a comparable leak).
   * C7a buys only +2.1 pp for **+11 km/h** median slack. That is loosening, not information, so it is not recommended.
   * The residual 149 eval windows (3.1 %) are humans accelerating into a faster road type within 6 s. The past cannot know that; only vision can (§4).
3. **Build cost.**
   * **Labels: 0.** `speed_n2_kmh` is in `release/v9_labels_eval139.npz` and `release_train4572/v9_labels_train.npz`.
   * **Trainer: 0.** `--r8-speed-input n2 --r8-v9-labels[-eval]`.
   * **Eval:** the arm reuses the trainer by path (`refcv3_arm.py:229-237`). End-to-end n2 feeding in the battery is **UNVERIFIED** and needs one CPU smoke.
   * **Mask:** `fed_ceiling` (`clamp_posthoc_r8.py:193`) already applies the `over` → value rule for 130.
4. **Mutation test** (`code/test_l4_builder.py`, `raw/test_l4_builder.log`, **SUITE PASS**). Literal road-law expectations: {20→30, 30→30, 50→50, 70→100, 80→100, 100→100, 120→120, 130→130, unknown→+inf}. Production is GREEN and **all five mutants are RED**:
   * M1 snap-DOWN;
   * M2 clamp 130→120;
   * M3 the window reaches the future;
   * M4 the 4-dp m/s ladder read back in km/h (the 57.5 % trap);
   * M5 unknown collapsed to 30.
   * ⚠️ M3 read GREEN on the first run. My random speeds saturated the ladder, so no future sample could move the ceiling. The check now uses slow-past / fast-future data and is pinned with that comment.
5. **Scoring for refcv9, fixed now, before any refcv9 data.**
   * Primary: strict `gt ≤ C7` on eval139, same estimator as A-R8-CLAMP-2, with the mask's n and sha printed.
   * Beside only: all-window, and a `+10 km/h` read.
   * Control: an all-130 mask must reproduce the all-window cells.

## 4. Next lever after C7 (Rule Zero): a vision road-type head, trained on PhysicalAI itself

* **Label:** the window's own future road-type step (C5b; privileged ego labels are allowed for LABELS).
* **Input:** vision only.
* **Upper bound:** C6/C5b, 99.1 %.
* ⛔ **Design constraint:** a model-predicted ceiling **cannot define the compliant set**, because A-R8-CLAMP-2's C3 requires a model-independent mask. The mask would stay on C7 while the head feeds the planner.
* **nuPlan posted-limit head:** LV+PIT have 100 % lane coverage (FS19-2, INHERITED); maps are at `D:/Projects/TanitAD/data/nuplan-maps/`, 970,997,691 B. It is the **legal-limit** variant, measured above at ≈52–61 % against the human, plus a US→25-country domain shift.
* Cost: ESTIMATED GPU probe hours. GPU is outside this brief's authorisation.

## 5. PI decisions (explicit)

* **P1 (blocks C7).** Is past-only ego speed history (max over 20 s) an admissible INPUT for refcv9? Three sources disagree:
  * `refc_v3_train.py:978` cites "MM ruling Q5" admitting n2;
  * FS19-2 says "Still refused: any ceiling derived from ego dynamics";
  * the PI's t0 rule names v0/a0/yaw-rate only.
* **P2.** The urban floor removes the PI's 30 km/h value (the 2026-09-16 four values). Choose between C7 (96.9 %, no 30) and C7b (92.7 %, keeps 30).
* **P3.** Does refcv9's ceiling mean a **legal limit** (vision/nuPlan, ~52–61 % compliant against the human, so the scoring frame must change) or a **demonstrated road-type set speed** (C7)?
* **P4.** If P1 is refused, every admissible ceiling is vision-only (§4), and that needs GPU authorisation. The alternative is to keep C0 under the 2026-10-11 exception and accept its 0.224 leak.
