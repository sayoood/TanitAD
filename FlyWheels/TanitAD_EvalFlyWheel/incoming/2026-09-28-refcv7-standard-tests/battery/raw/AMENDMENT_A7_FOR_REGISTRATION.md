## AMENDMENT A7 (items A7.2–A7.4) — clean text for registration (G0-50,400 diagnosis, 2026-10-04 ~17:10 Europe/Berlin)

**Status:** text for the Master Mind to register. It governs nothing until it is appended to `SPEC.md` and `raw/SPEC_SHA256_AMENDMENT_A7.txt` is written (the SPEC's sha256 after appending, plus the time). It binds only a G0 that **starts after** that file exists, e.g. the refcv8 battery's G0 if it adopts this §2.

**What A7.1 was, so the text is complete.** A2's low-support rule is in force under every amendment from A2 on. This brings the judge into line with A6 item 7 as registered and changes no rule. **The Master Mind applied it in code on 2026-10-04:** `A2_LOWSUPPORT_AMENDS = ("A2", "A5", "A6", "A7")` in `code/g0_refcv7.py`. It is pinned by `code/test_g0_a6_lowsupport.py`.

Under the Master Mind's ruling the step-50,400 record reads as follows:
* G0-A6 as registered (the text): **PASS** — computed by the corrected judge after the coded verdict was read; the criterion is unchanged (`raw/step50400/g0_A6_text.json`).
* G0-A6 as coded: **FAIL** — a judge defect (the A2 low-support tuple omitted A6).
* Step 30,000 likewise reads PASS under the text (`raw/step30000/g0_A6_text.json`).

**Calibration disclosure.** The A7.2 guard constants (factor 2, slack 5) were chosen while the refcv7 step-30,000 and step-50,400 counts below were visible. They bind only checkpoints whose data does not yet exist.

**Measured basis.** All figures are MEASURED, zero GPU; the files are in `raw/g0diag_step50400/`. The numerics-only arm `fp32_s0` uses the same weights, flags, windows and code as the replay, with the trunk in fp32 and NCHW and cuDNN TF32 off.

| step | low-support cells (n < 30) | in-run moved > 0.02 | `fp32_s0` moved > 0.02 | M1 moved > 0.02 |
|---|---|---|---|---|
| 50,400 | 238 | 13 | 20 | 27 |
| 30,000 | 238 | 8 | 8 | 31 |

* In both steps, no cell with n ≥ 30 and no pooled cell moved beyond 0.02 under any of the three.
* Every detection cell is seed-invariant (sd 0 over 24 seeds).
* Per-(seed, batch) DDIM effects correlate across checkpoints at r = 0.60–0.87 (n = 192).

### A7.2 — the DISCRETE-SMALL-N class

1. **Members.** Exactly A2's: a per-class or per-band DETECTION key whose support n is below 30. That covers `*_det_ap{thr}_{cls}_{band}` with cls ≠ all, `*_det_map{thr}_{band}`, and `*_rec@gate_{cls}`. The support n is read from the in-run row's own `*_det_npos_*` / `*_npos_*` keys. Each member is REPORTED with its n, in-run value, replay mean and `fp32_s0` value, and is never gated on its own (unchanged from A2).
2. **Population guard (this one gates).** Count two numbers over the members:
   * N_in = members with |in-run − replay mean| > 0.02;
   * N_num = members with |`fp32_s0` − seed 0| > 0.02 (`fp32_s0` is the A6 numerics arm, which already runs).

   The class **FAILS iff N_in > 2·N_num + 5**. If the `fp32_s0` row is missing, the class is NOT EVALUABLE and therefore FAILS (fail closed).
3. **Power probe M5 (reported; it never VOIDs G0).**
   * **What it does:** inference seed 0, with the class-logit columns of `bus` and `heavy_truck` swapped in BOTH slot heads at eval. This is a vocabulary-order loader defect that leaves every class-agnostic cell and every n ≥ 30 cell untouched.
   * **Detected iff** N_M5 > 2·N_num + 5, OR any gating DETECTION term moves outside its tolerance.
   * **Undetected** ⇒ the blind spot is NAMED: "a rare-class index swap is invisible to G0".
   * Both outcomes are committed here, before any M5 number exists.
4. **Power, stated.**
   * On refcv7 data this class would have had **no power against M1** at 50,400 (27 ≤ 45). It would have detected M1 at 30,000 (31 > 21).
   * M1 detection remains REQUIRED through the other classes; G0 is VOID otherwise, unchanged. Under A6 as registered, M1 is detected on 5 terms at 50,400 and 6 at 30,000.

### A7.3 — the detection packs are banked (reported, never gating)

* **What is banked:** the per-window detection packs for inference seed 0 and for `fp32_s0`, both slot heads, all 128 G0 windows. A pack is the `detection_metrics.window_packs` output: per slot the presence logit, xy and class argmax; and the GT xy, class, pos and ignore flags.
* **Why:** every DETECTION deviation can then be decomposed into COUNTED decisions — presence-order swaps, class-argmax flips, match-radius crossings — each with its margin, instead of being inferred from the AP arithmetic.
* **Storage:** ESTIMATED at under 10 MB. It goes in the G0 artifact or a sibling `.npz`; raw clip ids are never banked.

### A7.4 — the inference-seed draw is reported as one draw (diagnostic, never gating)

* Every 24-seed G0 reports two things:
  * the seed-group F test, seeds 0–7 vs 8–23 (the `PREREG_SEED_GROUP_CHECK_50400.md` form);
  * when an earlier checkpoint's G0 exists, the per-(seed, batch) correlation of `eval_traj` deviations against it.
* **Why:** a fixed seed list reuses ONE draw at every checkpoint. MEASURED: the 50,400 prereg check reads (B), F(15, 7) = 6.61, one-sided p = 0.0088.
* The STOCHASTIC rule (A5, K = 24) is unchanged.
* The battery's inference-seed floor (seeds 0 and 1) is quoted as one draw.

### Unchanged, and what A7 does not license

* **Unchanged:** every other class, tolerance and median; A2's thresholds; A5; A6 items 1–6 and 8; M1 required, else VOID; the M2 and M4 power probes; the wrapper clause; the requires_grad flags.
* **Reporting order:** as registered → A2 → A5 → A6 → A7. A7 is the gate for a G0 that started after it was registered.
* **What A7 does not license:**
  * re-opening a banked verdict;
  * passing a gating DETECTION cell (n ≥ 30, or pooled) that is outside its tolerance;
  * passing a low-support population that exceeds the guard;
  * passing when the numerics arm is missing.

### Implementation owed before any refcv8 G0 (code, zero GPU, Master Mind to assign)

* **`g0_refcv7.py`:**
  * the A7.2 guard inside `judge` when `amend == "A7"`;
  * an M5 mutation arm in the mutations loop (swap two `cls_logits` columns in both slot heads, then restore);
  * pack banking for seed 0 and `fp32_s0` (A7.3);
  * the seed-group report (A7.4, reusing `code/seed_group_check.py`'s test).
* **A literal test with a deliberate-regression arm for each item:**
  * A7.2: a population beyond the guard must FAIL;
  * M5 with columns NOT swapped must read undetected;
  * A7.3: a missing pack must be reported, never silent;
  * A7.4: the F test reproduces the step-5,000 numbers the prereg quotes.
* **Registration gate:** an `a7_registration` check, by file mtime against the G0's `started`, mirroring `a6_registration`.
