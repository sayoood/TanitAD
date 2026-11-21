# MM rulings on the Option B read side (2026-11-18 ~01:10Z, before any fine-tune checkpoint exists)

Card `../AMEND_OPTB.md` (sha256 7cb0245c, frozen). Build `BUILD_READ.md`. MM rulings on the card: `../MM_RULINGS_OPTB.md`.

1. **C10' PASS is accepted as the card's control.** It was measured once on R35, before any fine-tune checkpoint, as the card requires: s_swap on acceleration over the 141 eval clips with IA-3's folds was 0.2280 [0.2020, 0.2520] (R35_s0) and 0.2223 [0.1952, 0.2473] (R35_s1). The estimator was the episode-cluster bootstrap, B 2,000, seed 0, and it answers "another draw of episodes". Evidence MEASURED, `c10p/RESULT_C10P.json`. So IA-3P-K is calibratable as written.
2. **HARM at either chassis is HARM.** The card's HARM row reads "WORSE at either chassis", so it applies whatever the other chassis reads. When SEEDS-DISAGREE also holds, it is reported beside HARM. The card says the less favourable label's next step applies, and that is HARM's (term off, lambda 0.3). The two rules therefore never conflict on the next step.
3. **IA-3P-K contrast and anchors are accepted.**
   - The contrast is the term's 1 sd, 0.4203 WM units. MECH-MOVED asks whether the term trained the response it targets, so the read uses the term's own contrast.
   - Anchors are every valid start (stride 1). This gives more anchors under the same clip-level bootstrap, so the question the interval answers does not change.
   - Beside the primary read, with no bar: the same read at IA-3's contrast of 1.0 m/s², so a size-dependence of the gain would show. It costs seconds.
4. **The step-0 reads are reference only.** The cycle gain through the train-fit RA is 1.255 / 1.246 (the card predicted about 1.2). IA-3P-K reads 0.990 / 0.891. Neither number is a bar.
5. **Owed before the first arm's read** (MM): a chain that produces each arm's `plans_hwm_oracle_s{0,1}.npz` with the existing HWM-PLAN planner stage, at about 2 h of dev-box CPU per checkpoint (INHERITED from the AMEND_HWMPLAN_3 timings). It also owes the manifest inputs `controls.c2_c3_hold` and `band_met_lost`, which are read from each arm's banked selfcheck and log, not typed by hand.
