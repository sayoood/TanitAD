# MM ruling on EP-ADMISSION-R1 (Master Mind, 2026-10-19; before any candidate bank exists)

**1. A1-r1 is confirmed as specified.** FAIL iff D_M lower CI > 0 OR D_M point > +0.05 on any seed.
- False refusal of a calibrated candidate: 2.7 % [1.9, 3.9]; the cross-check generator gives 2.4 %.
- Power: 100 % at +0.135; 95 % at +0.05 (the conservative generator).
- The drafted A1 refuses 75.7 % [72.9, 78.3] of the same calibrated replicates. Revision 1's premise is now MEASURED, not estimated.
- ⚠️ A1-r1 is a per-changed-pick calibration test. A candidate that changes few picks is held to a proportionally smaller absolute D_M. The sealed registration says so.

**2. A2 takes a margin: FAIL-A2 iff, for any of NC / DAC / TTC, on any seed, the observed / predicted lower CI > 1.05** (was > 1.0).
- With no margin, six one-sided tests give 7.8 % [6.3, 9.6] false refusal, and 10.0 % with A1. Two independent retrains would compound that toward about 19 % (ESTIMATED).
- At 1.05 the post-hoc table gives 1.9 % false refusal. Power at ×1.25 is DAC 78 %, TTC 47 %, NC 26 %; at ×1.5 it is 99.9 / 96 / 70 %. DAC is the row that carries the measured loss (`D-REFE-EP-RELIAB-1`, share 0.63).
- ⚠️ The 1.05 was CHOSEN on the same replicates that measured it, so it is CONFIRMED on fresh ones before sealing (**EP-ADMISSION-R2-CONFIRM**). Bars, committed here before that run:
  - A2(1.05) false refusal: Wilson upper ≤ 5.0 %;
  - DAC power at ×1.25 ≥ 70 % (point);
  - A1-r1 false refusal: Wilson upper ≤ 5.0 %.
  - The run uses a NEW replicate seed, N ≥ 1,000 per condition, and the same generators and estimator.
  - All three met ⇒ seal at 1.05. Any one missed ⇒ seal at 1.10 (post-hoc table: 0.6 % false refusal, DAC power 54 % at ×1.25, 99.6 % at ×1.5) and report the miss.
- **NC is weak at any margin** (about 195 predicted unsafe pairs over 117 logs). The registration states NC's power and does not claim NC calibration is tested at ×1.25.

**3. The rest is sealed as Revision 1 says.** B's slot source is `row`; 2 seeds; `--rules r1` is named in the sealed registration; P6 is in-sample on S0 (biased toward OVER).

**4. Integration.**
- The staged `ep_admission.py` / `test_ep_admission.py` are LANDED. Draft rules are reproducible bit-for-bit under `--rules draft`.
- The p6-read-prep `MANIFEST.md` and `READY_P6_READ.md` get a dated UPDATE section. `code/CODE_SHA256.json` stays as the record of the prep build; nothing verifies against it.
