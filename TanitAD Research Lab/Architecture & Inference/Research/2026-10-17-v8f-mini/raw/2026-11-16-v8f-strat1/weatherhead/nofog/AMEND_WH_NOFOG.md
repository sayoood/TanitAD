# AMEND_WH_NOFOG: weather without fog (written 2026-11-17 ~21:50Z, before any no-fog number is computed)

**PI ruling, verbatim (2026-11-17):** "Drop fog for now".

**Why.** The PI's blind audit Q3 (`q3result/RESULT_Q3.md`, MEASURED) found the fog label RELABEL CANDIDATE: 2 of 10 answered fog frames agree, 0.20 [0.057, 0.510]. The PI reads the "fog" frames as snow 4, clear 3, rain 1, fog 2.

The label comes from Qwen2.5-VL-3B plus a SigLIP 2 probe trained on its answers (`q3rain/FOG_PROVENANCE.md`). A head scored against that label on fog is scored against noise.

**What changes, for now.** "For now" means until a fog labeller reaches the Q3 bar (agreement >= 0.80 on a blind PI draw). Bringing fog back is the PI's call.

1. **Scoring.** Weather windows whose label is fog are unscored, for every arm, the shipped head and the controls. The weather prediction is the argmax over clear / rain / snow: the fog logit is masked before the argmax. The number of unscored fog windows and episodes is printed.
2. **Weather-head fits from here on.** Fog-labelled windows are excluded from the weather loss (ignore index). That covers the planned re-fit on the adopted driving checkpoint. The shipped output reads clear / rain / snow.
3. **Not changed:**
   - **Arms whose comparison needs parity with banked arms keep the banked label table**, so the comparison stays clean. These are the roundabout N1R / S1R against S1, CKPT120, C0REP and DSCALE, all of which carry STRAT1-style context heads. The weather family is reported there with fog windows unscored, as in point 1.
   - The other four context variables are unchanged.
   - Every bar, estimator, window and outcome rule of the card is unchanged.

**The re-score now (zero training).** `wh_read_nofog.py` imports `wh_read.py` and changes only point 1. It runs on the same run2 T0 dumps and kid files, with the run2 instrument record. K0 (the reader reproduces the banked STRAT1 rows) is checked on the unmasked world, as in the main read. The card's outcome rules apply unchanged.

**Committed reading.**
- If weather still reads WH-DETECTS at both seeds without fog, weather ships as a clear / rain / snow output.
- If it does not detect, the fog-free read is reported as NOT-DETECTED. The card's a-W branch is the next lever.

The probe's beside row (no bar) stays unmasked and is labelled so.
