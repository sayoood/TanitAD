# RESULT_V8F_T1KIN_STRICT: with BOTH open channels closed (cameras frozen at t0, causal speed bucket), HACC still beats the floor. Outcome: STRICT MARGIN

`PREREG_V8F_T1KIN_STRICT.md` (`963906e8`, commit 01:45:51 +0100, before the first run at 02:30:52). ⚠ Its header's "~01:55" is a mis-estimated clock (erratum in `RESULT_V8F_T1KIN_CAUSALBUCKET.md`).

Read 2026-10-26 by `raw/2026-10-26-t1kin-strict/t1ks_read.py`, applied verbatim. Claim row `H-V8F-T1KIN-STRICT-1`. Zero training.

* **Tier T1 (T1-kin)**, the strictest open-loop configuration the programme can run without re-perception (T2 is not provisioned). At every 2 s replan the model reads only:
  * the FIRST frame, held for all 6 s;
  * its own executed motion;
  * a speed bucket built from the speed it has already driven (`--limit causal`);
  * the dead-reckoned nav.

  ⛔ Not closed loop, and not "drives".
* **Settings:** m = 10, `--nav deadreckon --scene frozen --limit causal --schedule periodic`; HACC checkpoints `M2b_s{0,1}_hacc.pt`; eval-141, 2,399 windows, 141 episodes. Dev-box CPU.
* Each run asserted the tool's sha256 (`d0ac8ce6…`, = `963906e8`'s `v8f_t1_eval.py`) before and after.
* **Artifacts:** `raw/2026-10-26-t1kin-strict/` (reader, read, 2 run JSONs, chain and logs, float32 dumps + manifest); float64 dumps `E:/Projects/TanitAD-artifacts/v8f-mini-hacc/t1kin_strict_dumps_f64_2026-10-26.tar` (282 files, 12,431,360 B, sha256 `b30ddba0…94458de`, `tar -d` verified).
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, with Amendment 1's rows read through `t1k_read.py`'s functions. F = 0.1207 m, 2F = 0.2415 m.

## The read (HACC strict − P10, both in their own loop; lower is better)

| row (family) | s0 | s1 |
|---|---|---|
| **ADE6, PRIMARY (m)** | **−0.7278 [−0.8680, −0.5942]** ✅ | **−0.6413 [−0.7696, −0.5190]** ✅ |
| ADE2 (m) | −0.0122 better | −0.0107 better |
| speed / along-track MAE, LONG | −0.359 m/s, −0.465 m, better | −0.374 m/s, −0.490 m, better |
| accel MAE (m/s²), LONG | +0.023, n.s. | −0.201, better |
| heading MAE (deg), LAT | −0.34, n.s. | −0.39, n.s. |
| yaw-rate MAE (deg/s), LAT | +3.06, worse | +1.52, worse |
| curvature MAE (1/m), LAT | +0.032, worse | +0.014, worse |
| cross-track MAE (m), LAT | −0.512, better | −0.339, better |
| TACTICAL / STRATEGIC, distance keeping | NOT COMPUTED (as in `RESULT_V8F_T1KIN_HACC`) | same |

Mean ADE6: strict 2.716 / 2.802 m, P10 3.443 m.

**Outcome (verbatim): STRICT MARGIN.** PRIMARY holds at both seeds. ⇒ v8F's T1 headline can be stated in the strict configuration.

**Controls (PASS):**
* The floor is untouched: max \|d\| = 0.0 against T1KIN's floor.
* Both channels are closed: the tick-0 plans equal the causal run's to 0.0 m, while the executed tracks differ by up to 22.3 / 17.0 m.

## Beside (decides nothing)

* **Closing the camera channel on top of the causal bucket costs +0.040 / +0.092 m of ADE6** (strict − causal, separated). The live frames carry some information after t0, but most of the margin does not need them.
* **The lateral rows read as before:** curvature and yaw rate are separated-worse at both seeds. `RESULT_V8F_HACC_LAT` shows these two rows are millimetre-noise-dominated below 5 m/s, and `PREREG_V8F_T1_LATCOND` re-reads them well-conditioned.
* **The envelope clips** 169 / 163 strict windows, against 178 floor windows.

## What it is, and what it is not

* **It is the strictest T1 result over the floor that this rig can produce.** From the first frame alone plus its own motion and a past-only speed bucket, HACC beats the kinematic floor, driven the same way, by 0.73 / 0.64 m of 6 s ADE at both seeds.
* ⚠ **Not dominance:** the canonical lateral rows; see above.
* ⚠ **Not closed loop:** no re-perception.
* ⚠ **The baseline is the programme's own floor,** not the state of the art.
