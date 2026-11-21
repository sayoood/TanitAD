# VERDICT: PREREAD_V8F_TOPK_PAIR = DIVERSITY, PREREAD_V8F_CONSORC = CONS-CARRY (speed) (Master Mind, 2026-11-10)

Tier: **T1-kin first tick** (self-action open loop, m 10, causal bucket; the first tick equals the logged history), eval-141. An oracle decomposition and a selection read, **never driving performance**. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0 ("another draw of episodes" only; the planner is deterministic; the run-to-run spread is the REP pair beside). Memory guards under AMENDMENT 1 (`AMENDMENT_1_MEMORY.md`, written 15:38:52, before the s1 / OB-CTL / CONSORC arms; guards only). STRATEGIC: **PAUSED** (PI item 25). All numbers MEASURED, artifacts `topk/out/TOPK_PAIR.json` (s0), `topk/out/TOPK_PAIR_s1ctl.json` (s1, OB-CTL), `consorc/out/CONSORC.json`.

## 1. TOPK_PAIR: DIVERSITY at both seeds (the rule applied verbatim)
Rule: DIVERSITY if PAIR-SEL - TOP1 is not separated, or S < 0.25. Tactical band (knots 11-30), labelled windows n 2,026 / 136-139 episodes.

| arm | logged pair in T's top 3 | PAIR-SEL - TOP1 (m) | GT-PAIR - TOP1 (m) | BEST3 - TOP1 (m) | S |
|---|---|---|---|---|---|
| DAG-OACC s0 | 0.891 | -0.0710 [-0.0875, -0.0555] | -0.1318 [-0.1647, -0.1010] | -0.2973 [-0.3270, -0.2678] | **0.239** |
| DAG-OACC s1 | 0.887 | -0.0635 [-0.0806, -0.0472] | -0.1355 [-0.1719, -0.1011] | -0.2788 [-0.3127, -0.2471] | **0.228** |
| OB-CTL s0 (method control) | 0.896 | -0.1009 [-0.1238, -0.0787] | -0.1593 [-0.1979, -0.1235] | -0.3712 [-0.4091, -0.3334] | 0.272 |

Controls: TOP1 and the GT future equal each arm's dump (max |diff| 0.0); PAIR-SEL equals TOP1 outside K; GT-PAIR equals the matching K-plan (0.0); OB-CTL's TOP1 and BEST3 tactical means equal `TOPK_FULL.json` (|diff| 0.0). **Outcome: DIVERSITY.** No manoeuvre selector is built: picking the logged manoeuvre perfectly among T's top 3 recovers about a quarter of the best-of-3 gap, and a perfect manoeuvre classifier at most -0.13 m. The best-of-3 gap is plan diversity. Committed trigger: the continuous constraints (CONSORC, below).

## 2. CONSORC: CONS-CARRY at both seeds; the largest kind is SPEED
Rule: CONS-CARRY if cost_CONS is separated on the worse side and share_CONS >= 0.7; DEC-CARRY if cost_DEC separated-worse and share_DEC >= 0.3. Windows with GT labels n 2,365 / 139 episodes. share = ratio of MEASURED means (ESTIMATED, no interval).

| arm | TOP1 tac ADE | GT-ALL | H = GT-ALL - TOP1 | cost_DEC (share) | cost_CONS (share) | cost_speed (share) | GT-SPEED-ONLY - TOP1 |
|---|---|---|---|---|---|---|---|
| **DAG-OACC s0** | 2.958 | 1.334 | -1.625 [-1.812, -1.448] | +0.181 [+0.151, +0.211] (0.111) | **+1.417 [+1.258, +1.583] (0.872)** | +2.717 [+2.399, +3.043] (1.673) | -0.386 [-0.462, -0.320] |
| **DAG-OACC s1** | 3.046 | 1.273 | -1.773 [-1.972, -1.574] | +0.167 [+0.132, +0.203] (0.094) | **+1.581 [+1.409, +1.755] (0.892)** | +2.750 [+2.463, +3.040] (1.551) | -0.392 [-0.468, -0.320] |
| M2b s0 (reference) | 3.194 | 1.300 | -1.894 | +0.190 (0.100) | +1.661 (0.877) | +2.836 (1.497) | -0.448 |
| M2b s1 | 3.023 | 1.349 | -1.674 | +0.167 (0.100) | +1.553 (0.927) | +2.488 (1.486) | -0.341 |
| REP s0 (beside) | 3.291 | 1.316 | -1.975 | +0.185 (0.094) | +1.678 (0.850) | +3.426 (1.734) | -0.542 |
| REP s1 (beside) | 2.935 | 1.265 | -1.670 | +0.232 (0.139) | +1.507 (0.902) | +2.550 (1.527) | -0.341 |

Other kinds' cost at DAG-OACC s0 / s1 (all on the worse side): geom +0.701 / +0.844, tte +0.523 / +0.681, angle +0.148 / +0.147, timegap +0.018 / +0.009 (s1 not separated), leadgap -0.007 / +0.004 (not separated). Leave-one-out costs are not additive (speed's alone exceeds |H|: the cells interact, as H-V8F-KIND-1 found on M2b).
Controls (all 6 arms): TOP1 = dump `ol` (0.0), GT future = dump (0.0), IDENT = TOP1 (0.0), OWN-ALL = IDENT (0.0), DAG-OACC s0 TOP1 tactical = the pair read's (0.0); checkpoint sha256: DAG-OACC s0 67265ec1 (= the SUMS pin), M2b s0 601cb435, s1 6104fb15 (= the OBD gate table); `manifest_ok` null (no manifest file beside the arms). **Outcome: CONS-CARRY, kind SPEED** (cost_speed separated-worse and largest at both seeds).

### The four families (DAG-OACC, tactical band; H = what a perfect tactical code buys, cost_CONS = what T's own constraints cost)
| family | row | H s0 / s1 | cost_CONS s0 / s1 | GT-SPEED-ONLY - TOP1 s0 / s1 |
|---|---|---|---|---|
| LONGITUDINAL | along-track MAE (m) | -1.584 / -1.733 | +1.391 / +1.553 | -0.385 / -0.418 |
| | speed MAE (m/s) | -0.691 / -0.748 | +0.567 / +0.640 | -0.184 / -0.201 |
| | accel MAE (m/s2) | -0.058 / -0.063 | +0.044 / +0.045 | -0.022 / -0.028 |
| LATERAL | cross-track MAE (m) | -0.304 / -0.342 | +0.259 / +0.299 | -0.061 / -0.038 |
| | heading MAE (deg) | -0.697 / -0.632 | +1.138 / +0.789 | +0.077 / +0.117 (n.s.) |
| | yaw-rate MAE (deg/s) | +0.247 / -0.046 (n.s.) | +0.221 / +0.532 (n.s.) | +0.116 / +0.004 (n.s.) |
| | curvature MAE (1/m) | +0.009 / +0.001 | +0.004 / +0.012 (n.s.) | +0.003 / -0.001 (n.s.) |
| TACTICAL | logged manoeuvre in T's top 3; PAIR-SEL, GT-PAIR | 0.891 / 0.887; table 1 | | |
| STRATEGIC | | PAUSED (PI item 25) | | |
Distance keeping (headway / time gap) has no row in this read; the timegap and leadgap cells (n 1,061 windows with a lead) carry share <= 0.011.

The headroom is **longitudinal**: along-track is -1.58 / -1.73 of the -1.62 / -1.77 m ADE headroom; cross-track -0.30 / -0.34.
**T's speed accuracy did not move with DAgger:** present-slot speed MSE DAG-OACC - M2b +0.0069 [-0.0055, +0.0197] (s0), +0.0004 [-0.0074, +0.0079] (s1), not separated. DAgger's tactical gain is on O's side, not in T's speed cells.

## 3. What the trigger says, and one premise of it that is wrong
The card's CONS-CARRY / speed trigger reads: "the speed-information levers in flight (EGOWM, LoRA) are the right aim on this base too, and each of their reads adds this decomposition on its own checkpoints."
* **EGOWM cannot move T's code.** Its card (`theory/egowm/PREREG_V8F_EGOWM.md`, Arms) trains the world model only, with "adapter, readers and planner frozen at M2b s{k}; driving cannot change, so T0". The decomposition on an EGOWM checkpoint equals M2b's by construction. EGOWM stays the lever for the planner's use of the world model, not for T's speed cells. The card named it wrongly; the trigger's own logic ("each read adds this decomposition") would have produced a structural zero.
* **LoRA (H-V8F-LORA-1, arm L, pod)** retrains all of v8F-mini on a LoRA-adapted DINOv3-L encoder, so it can move T's speed cells. Its read adds this decomposition on L_s0 / L_s1 against M2b s0 / s1 (the same windows, controls and rows).
* Known before this read: T's speed error is systematic (seeds correlate 0.71-0.85, H-V8F-SPEEDAVG-1) and information-limited (H-V8F-SPEEDCAL-1). What no read has asked: **where** the missing information is. Next lever, CPU, zero training: `PREREAD_V8F_SPEEDSRC` (being built): T's speed-cell error on DAG-OACC stratified by situations fixed before the read (longitudinal regime, lead agent, traffic-light label, curvature ahead, speed level), with a committed rule that names the information source a T-side card must feed T, or says the error is diffuse (then the lever is perception capacity: LoRA, resolution, data scale).
