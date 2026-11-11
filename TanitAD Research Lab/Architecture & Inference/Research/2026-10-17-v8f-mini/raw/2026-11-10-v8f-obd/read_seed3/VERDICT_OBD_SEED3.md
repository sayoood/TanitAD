# VERDICT: OBD seed-3 BAND read = BAND-UNSETTLED; recipe of record OBD-CTL (Master Mind, 2026-11-11 ~20:45)

Amendment `AMEND_OBD_SEED3.md` (sha256 `a8a5848f`, verified by the reader), card `PREREG_V8F_OBD.md`, code of record `cdbff971`. Four seed-3 runs banked on box 2 2026-11-11 17:34-20:19 (OBD-CTL / OBD-BAND at seeds 10 and 11, inits M2b s0 / s1). Dumps 20:19-20:38 (four arms, rc 0; checkpoint sha256 equal to the bank manifests). Read `obd_read_seed3.py` -> `RESULT_OBD_SEED3.json` / `.md`, 52.6 s, **0 problems**. Tier **T1-kin** (self-action open loop, m = 10). Estimator: paired episode-cluster bootstrap, B 2,000, seed 0 ("another draw of episodes"); "another training run" is floor G (OBD-BAND_rep vs OBD-BAND_s0) and the four-seed rule. STRATEGIC: **PAUSED** (PI item 25). All numbers MEASURED.

## Outcome, as committed: **BAND-UNSETTLED**
The rule needs one label at 3 of 4 seeds, with at least one at each init. The per-seed labels are s0 BAND-ADDS, s1 BAND-COSTS (both from the existing `RESULT_OBD.json`), s10 BAND-ADDS and s11 BAND-NEUTRAL (this read). No label reaches 3 of 4.

**Committed consequence: OBD-CTL's argv becomes the recipe of record** (the conservative branch: four training runs could not show the band loss consistently). The band's operative-only gain stays recorded beside, with the code-drift caveat of `VERDICT_OBD.md` section 3.

## What the four seeds show (BAND minus CTL, PRIMARY family)
| cell | floor G | s0 | s1 | s10 | s11 |
|---|---|---|---|---|---|
| operative ADE (m) | 0.0013 | -0.0019 gain | -0.0014 gain | -0.0020 gain | -0.0003 |
| operative FDE (m) | 0.0002 | -0.0058 gain | -0.0051 gain | -0.0094 gain | -0.0026 |
| tactical ADE (m) | 0.0688 | -0.0314 | +0.0487 (sep., sub-floor) | +0.0195 | +0.0374 |
| tactical FDE (m) | 0.2464 | -0.0871 | +0.1989 (sep., sub-floor) | +0.1114 (sep., sub-floor) | +0.1392 (sep., sub-floor) |
* The operative gain repeats at three of four seeds, and it is millimetres: 1.4-2.0 mm ADE.
* The tactical band leans the other way at three of four seeds, below floor G each time.
* LONGITUDINAL, LATERAL and TACTICAL family rows per seed are in `RESULT_OBD_SEED3.md`. The operative cross-track gain repeats at s0, s1 and s10 (-1.3 to -2.1 mm).

## What follows (Rule Zero)
The band loss is closed as a recipe lever: its measured effect is a millimetre-scale operative gain against a tactical lean, and the recipe of record is OBD-CTL. The levers with measured headroom are elsewhere: stop-point coverage (`raw/2026-11-10-v8f-stopbal/oracle/VERDICT_STOPTAU.md`: a correct stop decision is worth about 1.5 m tactical ADE on stop-point windows), and the SLOWWM basis read now running on box 2.
