# OTBAND read, seed 0 only (partial; decides nothing)

2026-11-10, Master Mind. Card `PREREG_V8F_OTBAND.md` (frozen, prefix sha256 `fd81c66e`); the card requires both seeds to agree and the replicate floor, so this is a per-seed reading that **decides nothing**. Seed 1 and OB-BAND_rep are training on box 2.

**Tier T1-kin** (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene), never a closed-scene driving result. **Evidence class MEASURED** (dumps of the banked checkpoints, zero training). **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap` (same windows) and `episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows; it answers "would another draw of EPISODES say this?" only. Numbers: `out/READ_S0_PARTIAL.json`, `out/HEADS_S0_PARTIAL.json`, `topk_read_s0.json`.

## Controls
* **P0** (commit tree vs the banked DAG-OACC dump): FAIL under its rule. Model-free arrays identical; first-tick plan `ol` within 1.7e-5 m; rolled track `cl` within 3.9 mm. **P0b** (the baseline's own tool tree, same settings): PASS-EXACT against the commit tree. So the OTBAND code reproduces the baseline bit for bit; the gap is the banked run's environment. Every compared pair below was dumped here, in one environment (README amendment 1).
* Top-3: `traj30_top1` equals the dumped `ol` exactly (0.0 m, 2,399 windows, both arms); the three plans differ pairwise in 100 % of windows (median largest gap 1.85 m CTL, 1.83 m BAND). T-head identity control 0 [0, 0].

## The card's bars at seed 0: OB-BAND minus OB-CTL
| family | row | operative 0.2-2 s | tactical 2.2-6 s |
|---|---|---|---|
| PRIMARY | ADE (m) | **-0.0168 [-0.0275, -0.0098]** | -0.0235 [-0.1147, +0.0672] |
| PRIMARY | FDE (m) | **-0.0393 [-0.0678, -0.0214]** | -0.0030 [-0.3080, +0.3190] |
| LONGITUDINAL | speed MAE (m/s) | **-0.0184 [-0.0362, -0.0056]** | -0.0139 [-0.1126, +0.0842] |
| LONGITUDINAL | along-track MAE (m) | **-0.0173 [-0.0283, -0.0102]** | +0.0081 [-0.0813, +0.0985] |
| LONGITUDINAL | DK headway abs error (m) | **-0.0427 [-0.0783, -0.0053]** | -0.1449 [-0.2941, +0.0125] |
| LONGITUDINAL | DK time-gap abs error (s) | **-0.0245 [-0.0602, -0.0030]** | -0.0420 [-0.1016, +0.0048] |
| LATERAL | cross-track MAE (m) | -0.0006 [-0.0023, +0.0009] | -0.0353 [-0.0877, +0.0152] |
| LATERAL | heading MAE (deg) | +0.0311 [+0.0059, +0.0620] | +0.3938 [-0.1904, +0.9773] |
| LATERAL | yaw-rate MAE (deg/s) | +0.1339 [+0.0172, +0.2912] | -1.9620 [-4.5450, +0.2681] |
| LATERAL | curvature MAE (1/m) | +0.0035 [+0.0001, +0.0078] | -0.0088 [-0.0305, +0.0112] |

TACTICAL: plan-derived TAC-LAT accuracy +0.0229 [+0.0017, +0.0455], TAC-LON -0.0050 [-0.0250, +0.0142]; T's own decisions (all ticks) T-LAT-F1 +0.0058 [-0.0281, +0.0356], T-LON-F1 +0.0151 [-0.0067, +0.0427], goal macro-F1 +0.0110 [-0.0131, +0.0321]. STRATEGIC: **PAUSED** (PI item 25).

**Seed-0 label, by the card as written: BAND-COST.** B1 holds (every operative bound inside its bar). B2 fails (no separated tactical gain). B3 fails: it asks for upper bounds at or below 0 on tactical speed and along-track MAE, and they are +0.084 and +0.099. B5 fails narrowly: TAC-LON's lower bound is -0.025 against -0.02. B4 fails (tactical yaw-rate -1.96, upper bound +0.27). In words: the band loss improves the operative band, longitudinal and distance keeping included, and leaves the tactical band unchanged within its interval.

## Beside, post-hoc, decides nothing: where the gap to DAG-OACC comes from
Both OTBAND arms trail the adopted base DAG-OACC widely (OB-CTL minus DAG-OACC, tactical ADE +0.734 [+0.505, +1.029] m, FDE +2.21 m, speed MAE +0.91 m/s). The card attributes that row to nothing, because the arms differ from DAG-OACC in init, detach, freeze and step count. A dump of the retrain's own init checkpoint (M2b s0, sha256 `601cb435`, the `levers.init_ckpt` of OB-CTL's summary) splits it:

| tactical band | init M2b | OB-CTL | DAG-OACC |
|---|---|---|---|
| ADE (m) | 4.282 | 3.988 | 3.254 |
| change against init (m) | | -0.294 [-0.497, -0.113] | -1.028 [-1.428, -0.721] |
| speed MAE change (m/s) | | -0.084 [-0.242, +0.067] | -0.998 [-1.235, -0.795] |
| acceleration MAE change (m/s^2) | | -0.237 [-0.497, +0.017] | -2.221 [-2.592, -1.892] |

The retrain improves on its own init. The arms trail DAG-OACC because they start from M2b and train on logged data only (`dagger_n` 0 in the run summary), and the DAgger fine-tune is worth about 3.5 times the retrain on tactical ADE. T's decisions say the same: against DAG-OACC, T-LAT-F1 is level at tick 0 (-0.0056 [-0.0288, +0.0160]) and drops after the model has driven on its own actions (ticks 1-2, -0.1940 [-0.2414, -0.1344]), which is the on-policy robustness DAgger trains.

## Beside: trajectories, plural (R-OUT), one-shot plan of each window's first tick
Best-of-3 minus top-1 ADE (oracle pick, not a deployable plan): OB-CTL operative -0.0153 [-0.0171, -0.0136], tactical -0.3645 [-0.3983, -0.3326]; OB-BAND -0.0133 and -0.3537. The GT manoeuvre pair is among T's top 3 in 0.896 [0.868, 0.922] (CTL) and 0.901 (BAND) of 2,026 labelled windows, against 0.572 and 0.580 for top-1. The card's top-K trigger (a mixture head) needs this gain above the replicate floor on both seeds.

## Next (Rule Zero)
1. Seed 1 and the replicate finish on box 2; this read reruns as written and gives the card's outcome.
2. **The largest measured lever is DAgger, not the band weighting.** The next arm is the band recipe initialised from DAG-OACC (and, or, with its DAgger rows), so that the band loss is read on the base it has to beat. It needs its own pre-registration and a box-2 slot after the queued measured-speed runs.
