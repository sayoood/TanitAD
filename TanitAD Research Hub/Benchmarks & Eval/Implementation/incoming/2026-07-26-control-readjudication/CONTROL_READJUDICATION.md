# CONTROL RE-ADJUDICATION — the half of the harvest that can only REMOVE results

**Date:** 2026-07-26 · **Host:** pod3 (A40, idle — verified 0 MiB / 0 %) + dev box (CPU)
**Feeds:** `Project Steering/BOOST_PROGRAM.md` §7.1 · `PROGRAM_HARVEST.md` H1.3 · `MODEL_REGISTRY.md` §1.2a
**Estimator on every interval in this document:** paired **episode-cluster bootstrap**
(`taniteval/ci.py`, B = 2000), resampling unit = **episode cluster**, never the window.
`overlapping_holdout_se` / `_jack` is **never quoted here**; where a source used it, it is named as a defect.
🔒 No clip UUIDs or raw PhysicalAI content appear in this document or its artifacts.

---

## 0. PRE-REGISTRATION — written before a single number was re-adjudicated

Per BOOST §7.1: *"the list of claims to re-adjudicate is frozen before any re-scoring runs… No claim is
added to the list after seeing a result."* §1 below is that frozen list. Both outcomes are committed here,
in advance:

> **OUTCOME A — controls hold at higher power.** The firewall/leakage/shuffle nulls stay null with a
> materially smaller half-width. The results they gated **stand, on firmer ground**, and each null is
> upgraded from *unpowered* to *powered*.
>
> **OUTCOME B — one or more controls FAIL at higher power.** A control whose null was the licence for a
> downstream claim turns out to be non-null. **Then the downstream claim must be withdrawn, by name.**
> This is the outcome that costs us results and it will not be softened.

**Additionally pre-registered, because it is the finding a pure power-projection cannot produce:**

> **OUTCOME C — the control could never have failed.** If a control's own **minimum detectable effect**
> (MDE) is larger than any leak it was built to catch, then its null carried **no evidence at all**, at
> any n. This is C13's *"a guard that cannot fail is not a guard"* and BOOST §7.3's unfalsifiable-benefit
> rule, applied to the negative-control side. A row in Outcome C is **neither confirmed nor refuted — it
> is void**, and re-running it at higher n is the only thing that can ever give it content.

**The asymmetry that makes this list different from the rest of H1.** For a *treatment* comparison an
unpowered null is an effect we may have discarded — upside. For a **firewall / leakage / shuffle /
circularity / blind-baseline / chance-baseline** check the null is the **desired** verdict, so
*"not separated at n≈40"* is **not a refuted leak — it is a leak we did not have the power to see.**

---

## 1. ⭐ THE FROZEN LIST — 75 control-type nulls (the harvest counted 14)

**How it was built** (`MEASURED`, reproducible): every one of the **1,845 JSON files** in the repo was
parsed (0 parse failures); every node carrying a `separated` flag **or** an interval was extracted; nodes
whose JSON path matched a control vocabulary (`shuf`, `permut`, `leak`, `firewall`, `blind`, `circular`,
`placebo`, `sanity`, `no_signal`, `negative_control`, `control`, `majority`, `random`, `dead_`, `clock`,
`scramble`, `null`, `degener`, `trivial`, `chance`, `mismatch`, `swap`, `decoy`) were retained; the four
**meta-index** files (`harvest_index.json`, `h2/h3/h4_*.json`) were excluded because they index other
results rather than produce them. That yields **231 control nodes, of which 75 are NULLS**. Every one of
the 75 was hand-classified into a family; **0 were keyword false-positives.**

**Containment check:** all **14** rows in `harvest_index.json → h1.firewall_inversion` are present in the
75 (marked ✅). **61 rows are new** (marked 🆕) — the harvest's index only reached nodes carrying an
explicit `separated: false`, so it missed every control expressed as a bare interval (`blind_qwk_ci`), the
whole `lead_gate_posthoc.json` subgroup layer, the `bar-a-selector` counterfactual controls, the
`h2-classifier` chance baselines, and — inside a file it did read — **`firewall.H`, the highest-proximity
firewall row in the program.**

| family | rows | n (episode clusters) | MDE range | what a NULL is supposed to license |
|---|---:|---|---|---|
| `S3-LEAK` | 7 | 73–558 | 0.0275–0.1985 QWK | conditioning alone does **not** move a blind head ⇒ S3 measures perception |
| `S3-BLINDvsCHANCE` | 4 | 73–81 | 0.1140–0.1776 QWK | the blind arm is **at chance** ⇒ the task needs the camera |
| `S3-CLOCK` | 11 | 73–587 | 0.0159–0.2024 QWK | R3: the target is **not** "how much clip is left" |
| `S1-BLINDvsMAJORITY` | 3 | **6–20** | **0.2457–0.5555 acc** | a goal-blind head does **not** beat the trivial majority ⇒ S1 is admissible |
| `SHUFFLE` | 23 | 32–126 | 0.0015–0.0606 | shuffling the feature destroys its signal ⇒ the contrast is causal |
| `DEAD-CONTROL` | 3 | 12–40 | 0.1225–0.3694 m | a perturbation with no physical content has **no** effect ⇒ the envelope is real |
| `CE-CONTROL` | 14 | 40 | 0.0047–0.1169 m | the counterfactual-equal arm is **inert** ⇒ the measured loss is the selector's |
| `CHANCE-BASELINE` | 10 | 322 | 0.0067–0.1679 | the head is **above** random at matched rate |
| **TOTAL** | **75** | | | |

### 1.1 The frozen list, row by row

`prox` = `|effect| / half-width` (screening statistic only — it ranks closeness to separation and does
**not** predict a flip, because the point estimate can move). **`MDE` = the half-width itself**: with a
95 % interval an effect must *exceed* it to separate, so MDE is the smallest leak this control could ever
have detected **at its own n**. ✅ = in the harvest's 14 · 🆕 = added by this sweep.

| # | family | artifact :: node | effect [CI95] | n eps | prox | MDE | in H? |
|---|---|---|---|---:|---:|---:|:--:|
| 1 | `S3-LEAK` | `2026-07-26-4brain-s3/s3_blind_baseline_primary.json`<br>`lon.paired_leak_B2_minus_B1` | +0.1657 [-0.0451, +0.3518] | 73 | 0.835 | **0.1985** | ✅ |
| 2 | `S3-LEAK` | `2026-07-26-s3-decision-grade/s3_parity_vs_nonparity.json`<br>`firewall_and_skill_bars.lon.paired_leak_B2_minus_B1.non_parity` | +0.1657 [-0.0451, +0.3518] | 73 | 0.835 | **0.1985** | ✅ |
| 3 | `S3-LEAK` | `2026-07-26-4brain-s3/s3_blind_baseline_primary.json`<br>`lon.paired_leak_B3_minus_B1` | +0.0524 [-0.0866, +0.1911] | 73 | 0.377 | **0.1389** | ✅ |
| 4 | `S3-LEAK` | `2026-07-26-s3-decision-grade/s3_parity_vs_nonparity.json`<br>`firewall_and_skill_bars.lon.paired_leak_B3_minus_B1.non_parity` | +0.0524 [-0.0866, +0.1911] | 73 | 0.377 | **0.1389** | ✅ |
| 5 | `S3-LEAK` | `2026-07-26-4brain-s3/s3_blind_baseline_sens_h8.json`<br>`lon.paired_leak_B2_minus_B1` | +0.0392 [-0.0740, +0.1466] | 85 | 0.355 | **0.1103** | ✅ |
| 6 | `S3-LEAK` | `2026-07-26-s3-decision-grade/s3_blind_baseline_parity_sens_h8.json`<br>`lon.paired_leak_B2_minus_B1` | +0.0058 [-0.0225, +0.0326] | 558 | 0.211 | **0.0275** | 🆕 |
| 7 | `S3-LEAK` | `2026-07-26-4brain-s3/s3_blind_baseline_sens_h8.json`<br>`lon.paired_leak_B3_minus_B1` | +0.0253 [-0.1175, +0.1561] | 85 | 0.185 | **0.1368** | ✅ |
| 8 | `S3-BLINDvsCHANCE` | `2026-07-26-4brain-s3/s3_blind_baseline_primary.json`<br>`lat.arms.B1_sensor_only.blind_qwk_ci` | +0.1128 [-0.0010, +0.2271] | 81 | 0.989 | **0.1140** | 🆕 |
| 9 | `S3-BLINDvsCHANCE` | `2026-07-26-4brain-s3/s3_blind_baseline_primary.json`<br>`lon.arms.B4_plus_clock.blind_qwk_ci` | +0.1410 [-0.0412, +0.3140] | 73 | 0.794 | **0.1776** | 🆕 |
| 10 | `S3-BLINDvsCHANCE` | `2026-07-26-4brain-s3/s3_blind_baseline_primary.json`<br>`lon.arms.B3_FULL_CONDITIONING.blind_qwk_ci` | +0.1200 [-0.0529, +0.2913] | 73 | 0.697 | **0.1721** | 🆕 |
| 11 | `S3-BLINDvsCHANCE` | `2026-07-26-4brain-s3/s3_blind_baseline_primary.json`<br>`lon.arms.B1_sensor_only.blind_qwk_ci` | +0.0676 [-0.0959, +0.2204] | 73 | 0.427 | **0.1582** | 🆕 |
| 12 | `S3-CLOCK` | `2026-07-26-s3-decision-grade/s3_blind_baseline_parity_sens_h8.json`<br>`lat.paired_clock_B4_minus_B3` | -0.0075 [-0.0233, +0.0085] | 587 | 0.472 | **0.0159** | 🆕 |
| 13 | `S3-CLOCK` | `2026-07-26-4brain-s3/s3_blind_baseline_sens_h8.json`<br>`lon.paired_clock_B4_minus_B3` | +0.0371 [-0.0568, +0.1233] | 85 | 0.412 | **0.0901** | 🆕 |
| 14 | `S3-CLOCK` | `2026-07-26-pod2-eval-host/artifacts/s3_blind_baseline_pod2_parity_primary.json`<br>`lat.paired_clock_B4_minus_B3` | +0.0063 [-0.0121, +0.0245] | 558 | 0.344 | **0.0183** | 🆕 |
| 15 | `S3-CLOCK` | `2026-07-26-s3-decision-grade/s3_parity_vs_nonparity.json`<br>`firewall_and_skill_bars.lat.paired_clock_B4_minus_B3.parity` | +0.0027 [-0.0162, +0.0216] | 558 | 0.143 | **0.0189** | 🆕 |
| 16 | `S3-CLOCK` | `2026-07-26-s3-decision-grade/s3_blind_baseline_parity_primary.json`<br>`lat.paired_clock_B4_minus_B3` | +0.0027 [-0.0162, +0.0216] | 558 | 0.143 | **0.0189** | 🆕 |
| 17 | `S3-CLOCK` | `2026-07-26-4brain-s3/s3_blind_baseline_sens_h8.json`<br>`lat.paired_clock_B4_minus_B3` | -0.0100 [-0.0873, +0.0623] | 95 | 0.134 | **0.0748** | 🆕 |
| 18 | `S3-CLOCK` | `2026-07-26-4brain-s3/s3_blind_baseline_primary.json`<br>`lon.paired_clock_B4_minus_B3` | +0.0209 [-0.1593, +0.2455] | 73 | 0.103 | **0.2024** | 🆕 |
| 19 | `S3-CLOCK` | `2026-07-26-s3-decision-grade/s3_parity_vs_nonparity.json`<br>`firewall_and_skill_bars.lon.paired_clock_B4_minus_B3.non_parity` | +0.0209 [-0.1593, +0.2455] | 73 | 0.103 | **0.2024** | ✅ |
| 20 | `S3-CLOCK` | `2026-07-26-s3-decision-grade/s3_blind_baseline_parity_sens_h8.json`<br>`lon.paired_clock_B4_minus_B3` | -0.0011 [-0.0232, +0.0196] | 558 | 0.051 | **0.0214** | 🆕 |
| 21 | `S3-CLOCK` | `2026-07-26-4brain-s3/s3_blind_baseline_primary.json`<br>`lat.paired_clock_B4_minus_B3` | -0.0014 [-0.1270, +0.1101] | 81 | 0.012 | **0.1186** | 🆕 |
| 22 | `S3-CLOCK` | `2026-07-26-s3-decision-grade/s3_parity_vs_nonparity.json`<br>`firewall_and_skill_bars.lat.paired_clock_B4_minus_B3.non_parity` | -0.0014 [-0.1270, +0.1101] | 81 | 0.012 | **0.1186** | ✅ |
| 23 | `S1-BLINDvsMAJORITY` | `2026-07-26-4brain-gates/S1_RESULTS.json`<br>`firewall.H.blind_vs_majority_paired` | -0.5000 [-1.0000, +0.1111] | 6 | 0.900 | **0.5555** | 🆕 |
| 24 | `S1-BLINDvsMAJORITY` | `2026-07-26-4brain-gates/S1_RESULTS.json`<br>`firewall.NOGOAL.blind_vs_majority_paired` | -0.0667 [-0.3429, +0.2122] | 20 | 0.240 | **0.2775** | ✅ |
| 25 | `S1-BLINDvsMAJORITY` | `2026-07-26-4brain-gates/S1_RESULTS.json`<br>`firewall.E.blind_vs_majority_paired` | +0.0385 [-0.2222, +0.2692] | 20 | 0.157 | **0.2457** | ✅ |
| 26 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_result.json`<br>`ridge|canonical.paired_mae_A_minus_B_shuf` | -0.0014 [-0.0029, +0.0002] | 126 | 0.903 | **0.0015** | ✅ |
| 27 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_result.json`<br>`ridge|canonical.rel_reduction_B_shuf` | -0.0032 [-0.0068, +0.0003] | 126 | 0.890 | **0.0036** | 🆕 |
| 28 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_result.json`<br>`gbm|canonical.rel_reduction_B_shuf` | -0.0116 [-0.0255, +0.0023] | 126 | 0.835 | **0.0139** | 🆕 |
| 29 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_result.json`<br>`gbm|canonical.paired_mae_A_minus_B_shuf` | -0.0052 [-0.0114, +0.0011] | 126 | 0.832 | **0.0063** | ✅ |
| 30 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`subgroups_2s[3].rel_reduction_B_shuf_control` | -0.0287 [-0.0665, +0.0068] | 58 | 0.784 | **0.0367** | 🆕 |
| 31 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_dspeed[2].rel_reduction_B_shuf_control` | -0.0181 [-0.0434, +0.0041] | 126 | 0.762 | **0.0238** | 🆕 |
| 32 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_displacement[1].rel_reduction_B_shuf_control` | -0.0116 [-0.0285, +0.0045] | 126 | 0.704 | **0.0165** | 🆕 |
| 33 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_dspeed[1].rel_reduction_B_shuf_control` | -0.0131 [-0.0357, +0.0077] | 126 | 0.605 | **0.0217** | 🆕 |
| 34 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_displacement[4].rel_reduction_B_shuf_control` | -0.0111 [-0.0352, +0.0087] | 126 | 0.506 | **0.0219** | 🆕 |
| 35 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_dspeed[3].rel_reduction_B_shuf_control` | -0.0091 [-0.0322, +0.0113] | 126 | 0.418 | **0.0218** | 🆕 |
| 36 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`subgroups_2s[4].rel_reduction_B_shuf_control` | -0.0110 [-0.0441, +0.0188] | 93 | 0.351 | **0.0315** | 🆕 |
| 37 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_dspeed[0].rel_reduction_B_shuf_control` | -0.0067 [-0.0265, +0.0122] | 126 | 0.345 | **0.0194** | 🆕 |
| 38 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_displacement[3].rel_reduction_B_shuf_control` | -0.0069 [-0.0304, +0.0136] | 126 | 0.312 | **0.0220** | 🆕 |
| 39 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_displacement[2].rel_reduction_B_shuf_control` | -0.0061 [-0.0280, +0.0131] | 126 | 0.299 | **0.0206** | 🆕 |
| 40 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`subgroups_2s[0].rel_reduction_B_shuf_control` | -0.0090 [-0.0406, +0.0236] | 78 | 0.280 | **0.0321** | 🆕 |
| 41 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_leadpresent[0].rel_reduction_B_shuf_control` | -0.0090 [-0.0406, +0.0236] | 78 | 0.280 | **0.0321** | 🆕 |
| 42 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_dspeed[4].rel_reduction_B_shuf_control` | -0.0046 [-0.0274, +0.0154] | 126 | 0.214 | **0.0214** | 🆕 |
| 43 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_leadpresent[4].rel_reduction_B_shuf_control` | -0.0056 [-0.0365, +0.0214] | 77 | 0.194 | **0.0290** | 🆕 |
| 44 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_leadpresent[3].rel_reduction_B_shuf_control` | +0.0051 [-0.0263, +0.0353] | 78 | 0.164 | **0.0308** | 🆕 |
| 45 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_leadpresent[2].rel_reduction_B_shuf_control` | -0.0039 [-0.0382, +0.0306] | 78 | 0.112 | **0.0344** | 🆕 |
| 46 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`subgroups_2s[1].rel_reduction_B_shuf_control` | +0.0044 [-0.0400, +0.0481] | 48 | 0.101 | **0.0441** | 🆕 |
| 47 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`horizon_leadpresent[1].rel_reduction_B_shuf_control` | -0.0007 [-0.0349, +0.0333] | 78 | 0.020 | **0.0341** | 🆕 |
| 48 | `SHUFFLE` | `2026-07-21-lead-state-gate/lead_gate_posthoc.json`<br>`subgroups_2s[2].rel_reduction_B_shuf_control` | -0.0004 [-0.0725, +0.0487] | 32 | 0.006 | **0.0606** | 🆕 |
| 49 | `DEAD-CONTROL` | `2026-07-26-p1-envelope-revalidation/artifacts/yawext_12ep.json`<br>`conditions.dead_noise[0].paired_along_2s` | +0.1035 [-0.0117, +0.2334] | 12 | 0.845 | **0.1225** | 🆕 |
| 50 | `DEAD-CONTROL` | `2026-07-26-p1-envelope-revalidation/artifacts/yawext_40ep.json`<br>`conditions.dead_shuffle[0].paired_along_2s` | +0.1237 [-0.0708, +0.3305] | 40 | 0.616 | **0.2006** | ✅ |
| 51 | `DEAD-CONTROL` | `2026-07-26-p1-envelope-revalidation/artifacts/yawext_12ep.json`<br>`conditions.dead_shuffle[0].paired_along_2s` | +0.1298 [-0.2663, +0.4725] | 12 | 0.351 | **0.3694** | ✅ |
| 52 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_produced.json`<br>`paired_intervals.ce_control_minus_as_trained.cross_abs_dense_LATERAL` | +0.0081 [-0.0004, +0.0211] | 40 | 0.753 | **0.0108** | 🆕 |
| 53 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_oracle.json`<br>`paired_intervals.ce_control_minus_as_trained.along_abs_dense_LONGITUDINAL` | +0.0235 [-0.0061, +0.0624] | 40 | 0.686 | **0.0342** | 🆕 |
| 54 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_produced.json`<br>`paired_intervals.regret_minus_ce_control_ISOLATES_THE_LOSS.along_abs_dense_LONGITUDINAL` | -0.0529 [-0.1571, +0.0100] | 40 | 0.633 | **0.0835** | 🆕 |
| 55 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_oracle.json`<br>`paired_intervals.ce_control_minus_as_trained.ade_0_2s` | +0.0238 [-0.0101, +0.0685] | 40 | 0.606 | **0.0393** | 🆕 |
| 56 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_produced.json`<br>`paired_intervals.ce_control_minus_as_trained.along_abs_dense_LONGITUDINAL` | +0.0567 [-0.0197, +0.1711] | 40 | 0.594 | **0.0954** | 🆕 |
| 57 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_oracle.json`<br>`paired_intervals.regret_minus_ce_control_ISOLATES_THE_LOSS.along_abs_dense_LONGITUDINAL` | -0.0204 [-0.0584, +0.0105] | 40 | 0.592 | **0.0345** | 🆕 |
| 58 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_oracle.json`<br>`paired_intervals.ce_control_minus_as_trained.miss_at_2m` | +0.0114 [-0.0057, +0.0340] | 40 | 0.574 | **0.0198** | 🆕 |
| 59 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_produced.json`<br>`paired_intervals.regret_minus_ce_control_ISOLATES_THE_LOSS.cross_abs_dense_LATERAL` | +0.0141 [-0.0047, +0.0445] | 40 | 0.573 | **0.0246** | 🆕 |
| 60 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_produced.json`<br>`paired_intervals.ce_control_minus_as_trained.ade_0_2s` | +0.0668 [-0.0271, +0.2066] | 40 | 0.572 | **0.1169** | 🆕 |
| 61 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_oracle.json`<br>`paired_intervals.ce_control_minus_as_trained.cross_abs_dense_LATERAL` | +0.0027 [-0.0025, +0.0073] | 40 | 0.551 | **0.0049** | 🆕 |
| 62 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_oracle.json`<br>`paired_intervals.regret_minus_ce_control_ISOLATES_THE_LOSS.ade_0_2s` | -0.0197 [-0.0651, +0.0158] | 40 | 0.487 | **0.0404** | 🆕 |
| 63 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_oracle.json`<br>`paired_intervals.regret_minus_ce_control_ISOLATES_THE_LOSS.cross_abs_dense_LATERAL` | -0.0020 [-0.0065, +0.0029] | 40 | 0.426 | **0.0047** | 🆕 |
| 64 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_produced.json`<br>`paired_intervals.regret_minus_ce_control_ISOLATES_THE_LOSS.ade_0_2s` | -0.0414 [-0.1702, +0.0421] | 40 | 0.390 | **0.1061** | 🆕 |
| 65 | `CE-CONTROL` | `2026-07-26-bar-a-selector/raw/bar_a_produced.json`<br>`paired_intervals.ce_control_minus_as_trained.miss_at_2m` | +0.0091 [-0.0250, +0.0443] | 40 | 0.263 | **0.0347** | 🆕 |
| 66 | `CHANCE-BASELINE` | `2026-07-26-h2-classifier/artifacts/h2c_results.json`<br>`operating_point.paired_recall_deltas.head_ego - random_at_rate` | +0.1340 [-0.0110, +0.3247] | 322 | 0.798 | **0.1679** | 🆕 |
| 67 | `CHANCE-BASELINE` | `2026-07-26-h2-classifier/artifacts/h2c_results.json`<br>`operating_point.paired_recall_deltas.head_img_ego - random_at_rate` | +0.1013 [-0.0158, +0.2727] | 322 | 0.702 | **0.1442** | 🆕 |
| 68 | `CHANCE-BASELINE` | `2026-07-26-h2-classifier/artifacts/h2c_results.json`<br>`verdict.delta_vs_random` | +0.1013 [-0.0158, +0.2727] | 322 | 0.702 | **0.1442** | 🆕 |
| 69 | `CHANCE-BASELINE` | `2026-07-26-h2-classifier/artifacts/h2c_results.json`<br>`paired_AP_vs_chance.head_ego` | +0.0075 [-0.0011, +0.0317] | 322 | 0.456 | **0.0164** | 🆕 |
| 70 | `CHANCE-BASELINE` | `2026-07-26-h2-classifier/artifacts/h2c_results.json`<br>`operating_point.paired_recall_deltas.head_img - random_at_rate` | +0.0294 [-0.0417, +0.1329] | 322 | 0.337 | **0.0873** | 🆕 |
| 71 | `CHANCE-BASELINE` | `2026-07-26-h2-classifier/artifacts/c12_fix.json`<br>`arms.head_img_ego.paired_AP_vs_chance` | +0.0060 [-0.0004, +0.0395] | 322 | 0.301 | **0.0199** | 🆕 |
| 72 | `CHANCE-BASELINE` | `2026-07-26-h2-classifier/artifacts/h2c_results.json`<br>`paired_AP_vs_chance.heur_decel` | +0.0024 [-0.0031, +0.0153] | 322 | 0.260 | **0.0092** | 🆕 |
| 73 | `CHANCE-BASELINE` | `2026-07-26-h2-classifier/artifacts/h2c_results.json`<br>`paired_AP_vs_chance.head_img_ego` | +0.0027 [-0.0028, +0.0264] | 322 | 0.184 | **0.0146** | 🆕 |
| 74 | `CHANCE-BASELINE` | `2026-07-26-h2-classifier/artifacts/c12_fix.json`<br>`arms.head_img.paired_AP_vs_chance` | +0.0031 [-0.0029, +0.0428] | 322 | 0.136 | **0.0229** | 🆕 |
| 75 | `CHANCE-BASELINE` | `2026-07-26-h2-classifier/artifacts/h2c_results.json`<br>`paired_AP_vs_chance.head_img` | +0.0002 [-0.0042, +0.0092] | 322 | 0.035 | **0.0067** | 🆕 |

⚠️ **Rows 2, 4, 19, 22 and 15 are the SAME measurements as rows 1, 3, 18, 21 and 16**, re-published inside
`s3_parity_vs_nonparity.json`. They are kept as separate frozen rows because they are separate quotable
locations — a reader grepping either file gets the number — and de-duplicating them would hide that a
corrected number has to be fixed in **two** places.

### 1.2 Second list — rows discovered AFTER the freeze

Per BOOST §7.1 nothing may be added to the primary list after a result is seen. **This section is empty
at the time of writing** and any later discovery goes here, clearly marked, and is ranked as *owed*.

---
