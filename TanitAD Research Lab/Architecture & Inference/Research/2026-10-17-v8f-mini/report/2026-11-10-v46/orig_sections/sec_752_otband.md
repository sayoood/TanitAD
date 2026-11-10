<!-- source: Paper/TANITAD_PAPER.md v1.8 at 5bb44006, section 7.24, renumbered for the report -->
### 7.52 The band loss on the 6 s plan, and the levers behind the adopted base (2026-11-10)

**Tier T1-kin throughout** (self-action open loop: m = 10, causal bucket, dead-reckoned nav, logged scene; the lead follows its logged future). **Never a closed-scene driving result.** Evidence class MEASURED; records under `raw/2026-11-09-v8f-otband/read/` (`RESULT_FULL.md`, `out/READ_FULL.json`, `out/FLOOR_TABLE.json`, `out/TOPK_FULL.json`, `out/HEADS_FULL.json`, `out/DAG_FLOOR.json`), register row H-V8F-OTBAND-1, card `PREREG_V8F_OTBAND.md` (frozen, prefix sha256 `fd81c66e`). Estimator: paired episode-cluster bootstrap over 141 episodes (2,399 windows), B = 2,000; it answers "would another draw of episodes say this?" only. Training-run variance is read from a replicate: the same flags and init at training seed 2, whose difference from the arm gives each row's floor (read at seed 0, applied to both seeds). The planner is deterministic (repeated inference differs by 0.0 m), so inference variance does not arise. STRATEGIC is PAUSED (PI item 25).

**The question.** The cascade of §7.50 states it: the operative layer's loss puts no gradient on the tactical layer (`--detach-upper`), and the band loss weights the operative knots (0.2-2.0 s) and the tactical knots (2.2-6.0 s) separately (`--w-op 1/3 --w-tac-traj 2/3`). Five runs on box 2 from M2b, with S, the adapter and the conditioning embeddings frozen, 9,582 steps on logged rows only: OB-CTL (the 30-knot term) and OB-BAND at seeds 0 and 1, and the replicate OB-BAND_rep.

**Finding 1: the band loss improves the short-horizon plan, and the gain reproduces.** OB-BAND minus OB-CTL, operative band, paired:

| row | seed 0 | seed 1 | replicate floor |
|---|---|---|---|
| ADE (m) | -0.0168 [-0.0275, -0.0098] | -0.0149 [-0.0202, -0.0105] | 0.0074 |
| FDE at 2 s (m) | -0.0393 [-0.0678, -0.0214] | -0.0267 [-0.0420, -0.0140] | 0.0008 |
| along-track MAE (m) | -0.0173 [-0.0283, -0.0102] | -0.0157 [-0.0210, -0.0112] | 0.0088 |
| speed MAE (m/s) | -0.0184 [-0.0362, -0.0056] | -0.0258 [-0.0335, -0.0184] | 0.0390 |
| heading MAE (deg) | +0.0311 [+0.0059, +0.0620] | +0.0313 [+0.0003, +0.0681] | 0.0176 |
| TAC-LAT accuracy, 0-6 s plan | +0.0229 [+0.0017, +0.0455] | +0.0371 [+0.0141, +0.0630] | not computed |

ADE, FDE and along-track are separated at both seeds and exceed the replicate floor at both; speed is separated at both seeds and within its floor; the price is 0.03 deg of operative heading. OB-CTL's operative ADE is 0.2571 / 0.2573 m, so the gain is about 6 %. On the tactical band no row moves at both seeds (ADE -0.0235 [-0.1147, +0.0672] / -0.0600 [-0.1539, +0.0306]), and the reason is visible in the replicate: two runs of the same recipe differ on tactical ADE by -0.2869 [-0.4073, -0.1665] m, five to twelve times the band loss's tactical effect (0.287 against 0.060 and 0.024). T's own decisions do not separate between the arms either, while the replicate pair itself differs by +0.0866 [+0.0413, +0.1255] in lateral macro-F1. Under the card as written the outcome is BAND-COST at both seeds: its tactical bars ask for upper bounds at or below zero, and none reached zero; no tactical row was separated on the worse side.

**Finding 2: which part of the adopted base does which work.** DAG-OACC, the adopted v8F base (register rows H-V8F-DAGGER-1 and H-V8F-DAG-OACC-1: M2b fine-tuned on a mix of logged rows and rows from the model's own rollouts, with an acceleration-matching term added), differs from M2b in three ways at once. The banked ladder separates them, all four arms from M2b and re-dumped here in one environment: DAG0 (the fine-tune schedule alone: 3,000 steps, lr 1e-4, p_gt 0.2, the whole model trainable, logged rows), DAG (plus DAgger rows at p = 0.1) and DAG-OACC (plus the acceleration-matching term, weight 49.6969). Each step is paired, seed 0 / seed 1; bold means separated at both seeds and above the band recipe's replicate floor at both:

| row | schedule (DAG0 - M2b) | DAgger rows (DAG - DAG0) | acceleration term (DAG-OACC - DAG) |
|---|---|---|---|
| tactical ADE (m) | -0.187 / +0.032 | **-0.819 / -0.615** | -0.022 / -0.165 |
| tactical heading MAE (deg) | -0.49 / +0.31 | **-4.63 / -2.57** | +0.97 / -0.74 |
| tactical cross-track MAE (m) | -0.02 / -0.11 | **-0.44 / -0.46** | +0.03 / -0.07 |
| TAC-LAT accuracy, 0-6 s plan | +0.011 / -0.045 | +0.100 / +0.192 (separated at both) | -0.002 / +0.012 |
| operative ADE (m) | -0.022 / +0.006 | +0.007 / +0.016 | **-0.062 / -0.079** |
| operative acceleration MAE (m/s^2) | -0.37 / +0.19 | +0.01 / +0.09 | **-0.97 / -1.14** |
| tactical acceleration MAE (m/s^2) | -0.17 / -0.05 | -0.05 / +0.64 | **-2.01 / -2.56** |

The DAgger rows carry the tactical band: position, heading, cross-track and the lateral manoeuvre the plan executes. The acceleration term carries the operative band and the acceleration rows of both bands. The band loss's operative ADE gain of Finding 1 is a fifth to a quarter of the acceleration term's. One interaction is visible in distance keeping: of 743 windows with a lead, the tactical band breaks the human's minimum headway or time gap in 20 / 10 for DAG0, 43 / 28 for DAG and 23 / 22 for DAG-OACC, so the DAgger rows raise those violations and the acceleration term lowers them again.

DAG-OACC's own run-to-run spread bounds this attribution. The DAG-ORC card's replicate (DAG-OACC's argv at training seed 10 / 11, trained with a later trainer file whose added levers read back off, so seed plus trainer version) differs from DAG-OACC by tactical ADE +0.6021 [+0.4134, +0.7978] m and tactical yaw-rate MAE +24.45 deg/s at init s0, and by +0.0215 [-0.0756, +0.1188] m at init s1. The DAgger step exceeds that pair at both inits (0.819 against 0.602; 0.615 against 0.022). The pair also says that at init s0 a second training run of the adopted recipe can plan the 2-6 s band much worse laterally.

**Trajectories, plural (R-OUT, §7.50).** Decoding O once per top-K tactical manoeuvre code (K = 3, the one-shot plan of each window's first tick): the best of the three plans beats the top-1 plan on tactical ADE by 0.35 to 0.39 m in all five runs (OB-CTL s0 -0.3645 [-0.3983, -0.3326]), and the logged manoeuvre pair is among T's top three in 0.895 to 0.901 of 2,026 labelled windows against 0.572 to 0.585 for its first choice. The best-of-3 is an oracle pick and no deployable plan; it is the measured headroom of a trained selection over T's codes, and it fired the card's committed top-K trigger.

**Controls.** The commit tree reproduces the baseline's own tool tree bit for bit at the same CPU thread count (all 141 episode dumps and the tick file); against the banked DAG-OACC dump, made at an unrecorded thread count, the rolled track differs by at most 3.9 mm, so every compared pair is dumped in one environment. Window identity and model-free arrays are asserted per load, checkpoint sha256 is checked against each bank manifest, and the T-head identity control reads 0 [0, 0].

**What this section does not establish.** (i) Nothing here is closed-scene driving. (ii) Each floor is one replicate pair, one draw of training variance; the DAG-ORC pair also carries a trainer change. (iii) The ladder attribution is post-hoc, beside the card, from one run per seed and step. (iv) Best-of-3 is an oracle bound. (v) The band loss has not been read on the DAgger base: that is the next card (DAG-OACC's recipe with `--detach-upper`, with and without the band loss, against a same-commit seed-only replicate). **Stated as method:** on this rig the tactical band's run-to-run spread (0.29 m on the band recipe, up to 0.60 m on DAG-OACC's) is larger than most lever effects, so a separated paired interval from one run per arm is necessary and not sufficient (H-ESTIM-SEED-1), and a lever is read as an effect only above a same-recipe replicate.
