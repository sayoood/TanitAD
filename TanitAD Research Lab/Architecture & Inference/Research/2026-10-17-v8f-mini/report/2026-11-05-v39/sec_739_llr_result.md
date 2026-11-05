**7.39.6 LLR: the lead head beside the level head, both as one conditioning token of O, reads P-FAILS-FAMILY-EXHAUSTED (`R/raw/2026-11-05-levelhead-llr-result/llr_read/RESULTS.md`, landed `{{LAND_LLR}}`).**

* **Setup [M].** LLR_s{0,1} was trained on box 2 on 2026-11-05 and pulled with no bad file; the pre-check passed: the checkpoint sha256 equals SHA256SUMS and the box-2 marker, and the argv is CLR's plus exactly the three lead tokens. The ten T1-kin rollouts ran on a new read tree carrying the lead-head model, whose known answer (KA-NEW2) reproduced C0 on `logged`, C0 on `reindexed` and CLR on `reindexed` exactly. Bases: C0 (P, G, D) and the banked CLR arrays (attribution LLR − CLR). Deciding scene `reindexed`; paired episode-cluster bootstrap B = 2,000, seed 0, eval-141; one training run per seed.
* **Reading [M].** Outcome **P-FAILS-FAMILY-EXHAUSTED** at both seeds. P: M1 RILLR − RIC0 is separated-better (−0.0125 [−0.0177, −0.0080] / −0.0116 [−0.0163, −0.0074]), but the share closed is 0.235 / 0.195 (bar 0.5); not LEAK-BORNE (logged 0.240 / 0.192). G fails: DK-VIOL is separated-worse at both seeds, DK-TG at s0, DK-HW and along-track MAE at s1 (ADE6 +0.0309 / +0.0540 m, inside the guard). F: 0.0125 against the C0REP floor 0.0362 at s0 (inside) and 0.0116 against 0.0038 at s1 (outside). The rules:

@@T LLR after=## Rules (CL card section 4@@

  The attribution row gives the lead head's share. Against CLR's own checkpoint LLR adds nothing on M1 (RILLR − RICLR −0.0001 [−0.0032, +0.0031] / −0.0039 [−0.0082, +0.0001]; on `logged` −0.0006 / −0.0038, separated-better at s1 only), so LLR's closure is the recovery recipe's, as CLR's was. Families, RILLR − RIC0: LONGITUDINAL 0 better / 2 worse (s0) and 0 / 3 (s1) of 6; LATERAL 4 / 0 (s0) and 6 / 0 (s1); TACTICAL 1 / 0 at both (TAC-LAT); STRATEGIC 3 / 0 at both (S-POS@80 −0.5825 / −1.0269). All 19 rows:

@@T LLR after=## 19 rows: RILLR - RIC0 (reindexed@@

* **The composite RILLRC = LLR + S (sg72) + kbox vs P10 [M].** NOT DOMINANT: D1 and D3 hold, D2 fails on the M1 crawl share alone (+0.0474 [+0.0297, +0.0685] / +0.0504 [+0.0295, +0.0752]); the other 18 rows are better or not separated. All 19 rows of RILLRC − P10 (Read A):

@@T LLR after=## 19 rows: RILLRC - P10@@

* **O's sensitivity in the token wiring [M]** (T0, not a driving result; `R/raw/2026-11-05-levelhead-llr-result/llr_read/SENSITIVITY.md`). The A1 measurement, adapted to the token, reads O-INSENSITIVE at both seeds for the level and, by analogy, for the lead presence. This holds although the token's norm is 1.5338 / 1.5699 times the ego token, and zeroing its output moves the plan −0.0250 / −0.1080 m/s, so O reads the token and does not use the level or the presence in it:

@@T LLRS after=Statistic:@@

  The A2 card's committed branch applies: the vision-head family on frozen single-frame features is exhausted at these forms, and its named levers are T2 re-perception and the A40. The one form still open on the allocated resources is pre-registered in §7.39.7 (Next).
