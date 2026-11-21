# P6-HN screen, amendment B (2026-10-18): five implementation findings settled BEFORE any screen data exists

Applies to the REGISTERED `PREREG_P6HN_SCREEN.md` (sha256 `9eb7cb2fc5e0e27927fc69e83926d2bed7878d866be9bd6f16f24511e7625679`).
Source: the implementation of checklist items 1, 2, 5, 6, 7 (`impl1/RESULT.md`), all on synthetic fixtures. No bank, cache,
label, pass or read exists yet; nothing below was chosen after seeing screen data. Every other clause stands.

**B1. §2.4 wording, no change to the computation.** In the masked two-call form, fan rows never attend to hard-negative
rows, so the fan's per-layer states are exactly those of the fan-only call. A hard-negative row therefore equals "a 65th
candidate in its real set" exactly only at decoder depth 1; at depth > 1 it reads the fan's unperturbed per-layer states.
That is the intended semantics (the deployed selector scores the fan without hard negatives) and is what is built and
tested.

**B2. The analytic identity (the gradient section) is corrected.** "A bank row that duplicates fan row j adds exactly row
j's share of `bce_fan_B`" cannot hold with set-level self-attention: the duplicate's query attends to 65 keys (the fan plus
itself) where row j's attends to 64. The identity is asserted with set-level self-attention NEUTRALISED (it then tests the
loss bookkeeping exactly, fp32 CPU). The attention mask itself stays covered by the registered mutations 3 and 4 (unmask
a padded row; drop the mask so fan rows see hard negatives), which must turn RED. `judge_g14` (checklist item 4) uses this
corrected form, never the literal one.

**B3. I6's fan tolerance is DEFINED here, because the cited calibration chose none.** `raw/rl_fan/atol_calibration/`
reads `CHOSEN_ATOL = null` (its pre-registered 10x rule failed at 0.5). I6 passes a cached scene only if, against RL-S1's
banked frozen fan, the SAME two bounds RL-S1's `fancmp` stage pins (`refe/rls1_pod.sh:24`, `:363`): the catastrophe bound
`FANCMP_ATOL` = **0.5** (max abs delta over every x / y (m) and yaw (rad) value of the set; 2 x the measured 0.25 floor) AND
the mean bound `FANCMP_MEAN_ATOL` = **0.025** (every null arm of the calibration reads a worst-set mean abs x/y delta
≤ 0.0248). Disclosed: the
calibration's "another vehicle's rig" arm (A1) is invisible at ANY tolerance, so I6 cannot detect that class; G15's
fingerprint checks carry it.

**B4. §10's pairing is KEPT; the implementation follows it.** The first implementation applied the adverse rule to an
unpaired seed-mean block because the families instrument exports only aggregates. That deviation is NOT adopted: the
families instrument is extended to export per-set component values, so the trajectory-level components are paired on the
§6 joint draws as registered. Until that export exists, S4 may not read families (the screen's PDM sub-scores are already
paired).

**B5. Three readings made explicit.** (a) SY's SPARSE rule is judged on the point estimate; its log-cluster interval is
reported beside it. (b) A hard negative is PDMS(offset) = 0, i.e. NC = 0 or DAC = 0 (§2, line "Hard negative"), AND
PDMS(anchor) > 0, unchanged. (c) Offsets are rounded to the fan dump's stored precision before labelling, so a δ = 0 row
reproduces the anchor's labels exactly.
