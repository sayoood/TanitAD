* **Status of this section (v37) [M].** Four reads of fine-tunes trained on box 2, each with a card frozen before its runs and a RESULTS file landed on the branch: RECOVER-FT with its composite and its frame-freeze control (`H-RECOVER-FT-1`, `H-RECOVER-COMPOSITE-1`; `R/raw/2026-11-03-recover-read-result/RESULTS.md`, landed `{{LAND_RF}}`), the two attribution arms that card committed for a HARMS label, RTL and RP01 (`H-RECOVER-NEXT-1`; `R/raw/2026-11-04-recover-next-rtl/RESULTS.md`, landed `{{LAND_RT}}`, and `R/raw/2026-11-04-recover-next-rp01/RESULTS.md`, landed `{{LAND_RP}}`), and the arm their outcome selected, RAPR (`R/raw/2026-11-04-recover-next-rapr/RESULTS.md`, landed `{{LAND_RA}}`). Tier: T1-kin for every read (the model's own plan applied open loop on logged cameras; it says nothing about re-perception; no T2 number exists). Estimator: the paired episode-cluster bootstrap, B = 2,000, seed 0, over 141 episodes / 2,399 windows, which answers *another draw of EPISODES* only. The training-run question is answered by a same-init replicate C0rep (training seed 10 + s) read as a floor (`H-ESTIM-SEED-1`); the planner is deterministic (KA1 bit-identity rerun of each arm's s0, max abs dev 0.0), so inference variance is not a question here. Unlike §7.35 these are TRAINED arms: each is 3,000 steps from the DAG-OACC checkpoint of its seed with DAG-OACC's own argv minus the DAgger swap (parity control KP1: every run's train and window-list sha256 equal DAG-OACC's at the same seed). The frame-freeze control is the FRAMEFREEZE tool package (`R/raw/2026-11-02-framefreeze/`), which has no result of its own; it is read inside each recovery read.
* **Why this arm [M] / [H].** §7.35 located the M1 excess in the ego-history input: the logged history closes 0.939 of it (ORACLE), a constant-twist history at the executed speed does not, and the same history at the logged speed closes 0.599 to 0.633. **H** (a HYPOTHESIS, RECOVER-FT card section 1): teaching O to recover from deviated ego histories makes it take the speed level from the frames, or discount a deviated history. **Recipe R_s:** with probability p = 0.3 a training sample's kinematic history (v, a, ω on every real history frame, and the NOW anchor) is offset by ONE record drawn uniformly from an EMPIRICAL deviation table, the DAgger iteration-2 own-state bank on TRAIN windows only (613,132 records; the bank is refused if it holds an eval-141 clip), each record being own minus logged at the same clock row: (dv, da, dω, dy, dψ). The target is the logged future re-expressed in the perturbed frame and, for the first K = 10 knots, blended onto it from the straight continuation of the deviated heading:

$$g'_i = \mathrm{Rot}(-d\psi)\,\big(g_i - (0, d_y)\big), \qquad \mathrm{target}_i = (1 - w_i)\,(L_i, 0) + w_i\, g'_i, \qquad w_i = 3x_i^2 - 2x_i^3,\ x_i = \tfrac{i+1}{K},$$

  with $L_i$ the logged cumulative arc length to knot $i$ and $\mathrm{target}_i = g'_i$ for $i \ge K$. **C0_s** is the same recipe with p = 0 (the off state returns the same batch, bit-identical, so C0 is *more training on logged data*); **C0rep_s** is C0_s at training seed 10 + s. Read rules (card section 6, frozen before any run): PRIMARY = M1 R − C0 separated-better at both seeds and mean share closed ≥ 0.5; GUARDS = ADE6 R − C0 point ≤ +0.0604 m at each seed and no separated-worse non-crawl row; FLOOR = |R − C0| > |C0rep − C0| on M1 at each seed; HARMS whenever the guards fail.

**7.36.1 RECOVER-FT (R): HARMS (`H-RECOVER-FT-1`).**

| item | s0 | s1 |
|---|---|---|
| M1 R - C0, logged frames | -0.0117 [-0.0185, -0.0050] sep-B | -0.0078 [-0.0143, -0.0019] sep-B |
| share closed (primary.share_closed) | 0.2198 | 0.1318 |
| ADE6 R - C0 | +0.1892 [+0.1060, +0.2739] sep-W | +0.0977 [-0.0337, +0.2218] ns |
| non-crawl separated-worse rows in R - C0 | ADE6, DK-HW, DK-VIOL, TAC-LAT, accel MAE, along-track MAE, cross-track MAE | DK-VIOL, along-track MAE |
| FLOOR abs(R - C0) on M1 (point) | 0.0117 | 0.0078 |
| FLOOR abs(C0rep - C0) on M1 (point) | 0.0362 | 0.0038 |
| FREEZE share closed | 0.1674 | 0.1568 |

* **Reading [M].** M1 moves the right way at both seeds (separated-better) but by 0.2198 / 0.1318 of the C0 − P10 excess (mean 0.1758; bar 0.5), so PRIMARY fails; the guards fail at both seeds (ADE6 above +0.0604 m at both, seven non-crawl rows separated-worse at s0 and two at s1), so the label is HARMS. At s0 the replicate moves further from C0 than R does (C0rep − C0 on M1 +0.0362 and on ADE6 +0.3061 m, both separated): at that seed one more training draw of the CONTROL is a larger effect than the lever. Frame-freeze reads PARTIAL (share closed 0.1674 / 0.1568 under `--scene hold_now`, which replaces every history row's image features by the current frame's, so the frames carry no motion), so the closure R does show is not explained away by the logged frames' motion, and is not robust to it either.

**7.36.2 The composite RC = R + S (sg72) + kbox vs P10: NOT DOMINANT, D2 fails on M1 alone (`H-RECOVER-COMPOSITE-1`).**

| item | s0 | s1 |
|---|---|---|
| RC vs P10: ADE6 | -0.9566 [-1.1678, -0.7514] sep | -0.9344 [-1.1692, -0.6864] sep |
| RC vs P10: M1 crawl share | +0.0482 [+0.0282, +0.0715] sep-W | +0.0571 [+0.0340, +0.0853] sep-W |
| C0C vs P10: M1 crawl share | +0.0587 [+0.0401, +0.0812] sep-W | +0.0604 [+0.0383, +0.0868] sep-W |

* **Reading [M].** D1 holds at both seeds with about 0.7 m of slack over its −0.2415 m bar, D3 holds over all four families (STRATEGIC included: S-POS@30/50/80 separated-better), and D2 fails at one row of 19, the M1 crawl share, exactly the FEAS2 signature of §7.34. The same composite on the control, C0C, fails at the same single row. kbox's own effect on the recovery base (RC − R) is separated-worse on M1 (+0.0080 / +0.0062) while it improves curvature and yaw-rate MAE (−0.0932 / −0.1207 and −12.7951 / −13.0791, separated): the feasibility box trades a little crawl for lateral smoothness.

**7.36.3 The attribution arms RTL (`--recover-target logged`, no rejoin blend) and RP01 (`--recover-p 0.1`), and Q1 (`H-RECOVER-NEXT-1`).** The RECOVER-FT card committed both for a HARMS label (section 8); the PI's box-2 go of 2026-11-03 covered these four runs. Each arm changes ONE flag of R; read by the same rules with R := X. Q1, committed in the RECOVER-NEXT card (section 3) before either arm ran: *"blend harmed iff RTL's guards PASS at both seeds; dose harmed iff RP01's guards PASS at both seeds; both / neither otherwise."*

| quantity | bar | R s0 | R s1 | RTL s0 | RTL s1 | RP01 s0 | RP01 s1 |
|---|---|---|---|---|---|---|---|
| M1 crawl share closed vs C0 | >= 0.5 | 0.2198 | 0.1318 | 0.2131 | 0.1933 | 0.1473 | 0.1342 |
| ADE6 X − C0 (m) | <= +0.0604 | +0.1892 | +0.0977 | +0.1492 | +0.0925 | +0.0017 | +0.0008 |
| non-crawl rows separated-worse vs C0 | none | ADE6, DK-HW, DK-VIOL, TAC-LAT, accel MAE, along-track MAE, cross-track MAE | DK-VIOL, along-track MAE | ADE6, along-track MAE | DK-HW, DK-VIOL, along-track MAE | DK-VIOL | DK-VIOL, accel MAE |
| composite M1 crawl share XC − P10 | not separated-worse | +0.0482 [+0.0282, +0.0715] | +0.0571 [+0.0340, +0.0853] | +0.0484 [+0.0282, +0.0715] | +0.0540 [+0.0318, +0.0805] | +0.0513 [+0.0319, +0.0738] | +0.0559 [+0.0340, +0.0824] |
| kbox's own M1 effect XC − X | not separated-worse | +0.0080 [+0.0032, +0.0142] | +0.0062 [+0.0019, +0.0121] | +0.0066 [+0.0027, +0.0112] | +0.0070 [+0.0025, +0.0127] | +0.0068 [+0.0022, +0.0128] | +0.0050 [+0.0009, +0.0101] |
| floor: abs(X − C0) M1 vs abs(C0rep − C0) | exceeds | 0.0117 vs 0.0362 | 0.0078 vs 0.0038 | 0.0113 vs 0.0362 | 0.0115 vs 0.0038 | 0.0078 vs 0.0362 | 0.0080 vs 0.0038 |
| frame-freeze share closed | robust bar | 0.1674 | 0.1568 | 0.1860 | 0.1899 | 0.1546 | 0.1265 |

* **Reading [M].** Both arms read HARMS, so **Q1 = neither**. Removing the rejoin blend (RTL) leaves R's ADE6 harm in place (+0.1492 / +0.0925 m) and keeps the M1 closure where R had it (0.2131 / 0.1933). Lowering the dose to p 0.1 (RP01) removes the ADE6 harm (+0.0017 / +0.0008 m, inside the guard) but not the longitudinal one: DK-VIOL is separated-worse at both seeds (+0.0235 [+0.0051, +0.0475] / +0.0202 [+0.0052, +0.0374]) and accel MAE at s1 (+0.0933 [+0.0285, +0.1648]), and the M1 closure drops to 0.1473 / 0.1342. Every composite fails D2 on M1 alone, at +0.048 to +0.057 against C0C's +0.0587 / +0.0604; D1 and D3 hold for all of them (RP01C ADE6 vs P10 −1.1647 [−1.3754, −0.9645] / −1.0226 [−1.2422, −0.8039] m, the largest D1 margin of the three recovery composites; RTLC −0.9766 / −0.9353 m, RC −0.9566 / −0.9344 m). Twenty controls, 0 failed, in each of the RTL and RP01 reads. Q1 = neither selects row D of addendum 1 (frozen 2026-11-04 01:02, before any RTL or RP01 rollout): **RAPR as frozen**.
