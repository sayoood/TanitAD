# CONTROL RE-ADJUDICATION — the half of the harvest that can only REMOVE results

**Date:** 2026-07-26 · **Host:** pod3 (A40, idle — verified 0 MiB / 0 %) + dev box (CPU)
**Feeds:** `Project Steering/BOOST_PROGRAM.md` §7.1 · `PROGRAM_HARVEST.md` H1.3 · `MODEL_REGISTRY.md` §1.2a
**Estimator on every interval in this document:** paired **episode-cluster bootstrap**
(`taniteval/ci.py`, B = 2000), resampling unit = **episode cluster**, never the window.
`overlapping_holdout_se` / `_jack` is **never quoted here**; where a source used it, it is named as a defect.
🔒 No clip UUIDs or raw PhysicalAI content appear in this document or its artifacts.

> ## ⭐ TL;DR — six sentences
>
> 1. The frozen list is **75 control-type nulls**, not the 14 the harvest counted — including
>    **`firewall.H`, the highest-proximity firewall row in the program**, which the harvest missed inside
>    a file it read.
> 2. ⛔ **OUTCOME B: the controls do not hold.** Of the 12 S3 firewall verdicts re-adjudicable at maximum
>    power, **4 changed and every one changed from "control passed" to "control FAILED"**, with **3 of 12
>    sign flips** — reproduced across **three runs on two pods** (§2.3, §3.1.1).
> 3. The longitudinal leak control had **MDE 0.1985 QWK** against a leak that is really **+0.0711** — it
>    was **2.8× too blunt to see the thing it existed to catch**, so it never passed, it was never run.
> 4. 🔴 **`firewall.H` is PROVEN incapable of firing**: its MDE (0.5555) is **222 % of the maximum effect
>    that can physically exist** (0.2500) — a blind head at 100 % accuracy would still have read
>    "not separated". Its `ADMITTED` verdict is zero evidence.
> 5. **Six things must be withdrawn or amended** (§4): the S3 skill bars on both axes (off by **+128 %**
>    longitudinally), the "blind arm is at chance" reading, `S1_T1_SLICE.md:87`'s admissibility sentence,
>    `GATE_RESULTS.md`'s "blocked only on corpus size", and — reaching the whole harvest — **H1's
>    `would_flip` projection, which scores 4/8 = 50 % on the only 8 cases where truth is now measured.**
> 6. ⚠️ **30 of the 75 rows I could not re-power at all** (pod2 holds the 600-episode val and pod2 is
>    training); they are ranked **owed**, not resolved — do not read this as "the controls were checked".


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

## 2. METHOD, and the power I actually had — stated plainly

Two instruments, because only one of them needed a pod.

**Instrument 1 — the higher-n sibling.** For the S3 firewall family a **higher-power run already exists
in the repo** and nobody had paired it against the null it answers. The `2026-07-26-s3-decision-grade`
stream re-ran the *identical* probe (`changed_between_runs: "ONLY --train-cache / --test-cache. Zero code
change"`) on the **parity** corpus, reaching **n = 520–587 episode clusters / 24,987–53,658 windows**
against the original **n = 73–95 / 1,997–5,290**. The harvest's index reached the `.non_parity` key of
`s3_parity_vs_nonparity.json` and **not** the `.parity` key **in the same file** — because the parity row
had `separated: true` and so never entered a `separated: false` sweep. **The answer to the harvest's
"single most dangerous row" was one JSON key away from the question.**

**Instrument 2 — the power ceiling (MDE).** Costs nothing and needs no pod. `MDE` = the row's own 95 %
half-width; with a 95 % interval an effect must *exceed* it to separate. For a **bounded** metric
(accuracy, QWK) the maximum attainable effect is also known, so `MDE vs max-possible-effect` is a
**proof**, not a projection: if MDE exceeds the largest effect that could physically exist, the test
**cannot fire at any observed value** and its null is void. This is C13's *"a guard that cannot fail is
not a guard"* and BOOST §7.3's unfalsifiable-benefit rule, pointed at the negative-control side.

### 2.1 Power actually achieved, per family — no rounding up

| family | rows | power available to me | achieved | class |
|---|---:|---|---|---|
| `S3-LEAK` | 7 | parity run in-repo, ×2 pods | **n = 73 → 520/558**, half-width **÷2.3–4.9** | ✅ **RESOLVED** |
| `S3-BLINDvsCHANCE` | 4 | same | **n = 73/81 → 520/558** | ✅ **RESOLVED** |
| `S3-CLOCK` | 11 | same | **n = 73/95 → 520/587**, half-width **÷4.2–7.7** | ✅ **RESOLVED** |
| `S1-BLINDvsMAJORITY` | 3 | **none** — n is bounded by 30 AlpaSim decision points, not by episodes | **n unchanged**; adjudicated by **power ceiling** | ⚠️ **VOID + OWED** |
| `SHUFFLE` | 23 | **none** — needs a lead-state re-ingest | **n unchanged**; adjudicated by **capacity decomposition** | ⚠️ **CORRECTED + OWED** |
| `DEAD-CONTROL` | 3 | ⛔ needs the 600-ep val, which is **on pod2 and pod2 is training** | **none** | 🔴 **OWED** |
| `CE-CONTROL` | 14 | ⛔ same | **none** | 🔴 **OWED** |
| `CHANCE-BASELINE` | 10 | ⛔ pod2 is running the H2 sweep | **none** (already n = 322) | 🔴 **OWED** |

⛔ **Said plainly: 30 of the 75 rows (40 %) I could not re-power at all.** They are ranked **owed**, not
resolved. Anyone who reads this document as "the controls were checked" is reading it wrong — **45 were
checked, 30 were not.**

### 2.2 ⚠️ The confound in Instrument 1, stated before the result

The parity run changes **both `n` and the corpus**. `MEASURED` from `s3_parity_vs_nonparity.json`: the
lateral majority-class rate moves **0.3255 → 0.8553**, the event rate **0.7178 → 0.1447**, and the
majority class itself changes from `t_5_10` to `t_none`. **This is not a pure power increase** — it is
the same 40-vs-600 composition confound the registry flags in §1.2a.

**Why the parity numbers are nonetheless the decision-grade ones, and it is not a matter of taste:** the
non-parity caches fail an *independent* consistency check. The shipped label artifact for the same
600-episode val records a 25 s curvature-lookahead rate of **0.2076**; S3's parity 12 s event rate is
**0.1447**, correctly *below* it, while the non-parity rate is **0.7178 — 3.5× the 25 s figure**, which
is impossible for the same corpus. **The dev-box caches are a city-heavy sample, not a scaled-down
parity corpus.** So the non-parity rows are **superseded**, not merely under-powered — and that is a
worse verdict for them, not a kinder one.

### 2.3 Reproduction before quoting — done, and it is not a formality

The load-bearing parity firewall exists as **two independently produced artifacts** on **two different
pods** from **two separate cache instances** (`md5` differs: `bb5bd2f…` vs `f4b6a8b…`; `mine_seconds`
1326.6 vs 2469.5). I paired them node-by-node:

| node | run A (`s3-decision-grade`, pod3) | run C (`pod2-eval-host`, pod2) | agree? | Δpoint |
|---|---|---|:--:|---:|
| `lat.paired_leak_B2_minus_B1` | +0.3727 [+0.3163, +0.4292] SEP | +0.3740 [+0.3193, +0.4304] SEP | ✅ | 0.0013 |
| `lat.paired_leak_B3_minus_B1` | +0.3968 [+0.3426, +0.4492] SEP | +0.3902 [+0.3358, +0.4449] SEP | ✅ | 0.0066 |
| `lat.paired_clock_B4_minus_B3` | +0.0027 [−0.0162, +0.0216] not sep | +0.0063 [−0.0121, +0.0245] not sep | ✅ | 0.0036 |
| ⭐ `lon.paired_leak_B2_minus_B1` | **+0.0711 [+0.0308, +0.1127] SEP** | **+0.0796 [+0.0392, +0.1217] SEP** | ✅ | 0.0085 |
| `lon.paired_leak_B3_minus_B1` | +0.2442 [+0.1966, +0.2951] SEP | +0.2539 [+0.2038, +0.3088] SEP | ✅ | 0.0097 |
| `lon.paired_clock_B4_minus_B3` | **−0.0359 [−0.0634, −0.0108] SEP** | **−0.0475 [−0.0761, −0.0198] SEP** | ✅ | 0.0116 |

**6/6 verdicts agree; every point estimate within 0.0116 QWK.** Class `MEASURED`, tier **CONFIRMED**
(two independent artifacts). A **third** independent run was launched by me on pod3 for this task and is
reported in §3.1.1.

---

## 3. THE RESULTS

### 3.0 ⭐ Verdict in one line

> **OUTCOME B. The controls do not hold.** Of the 12 S3 firewall verdicts that could be re-adjudicated
> at maximum power, **4 changed — every one of them from "control passed" to "control FAILED"**, and
> **3 of 12 flipped sign**. Separately, **`firewall.H` is proven incapable of ever firing.** Nothing
> moved in the direction that would have been good news.

### 3.1 `S3-LEAK` + `S3-CLOCK` — 4 of 12 verdicts changed, all against us

`MEASURED` · tier **CONFIRMED** (two independent runs) · raw: `raw/readjudication.json`.
Estimator: paired episode-cluster bootstrap, B = 2000, unit = val episode.

| axis · node · horizon | original (n = 73–95) | **re-adjudicated (n = 520–587)** | power | verdict |
|---|---|---|---:|---|
| ⭐ `lon.paired_leak_B2_minus_B1` H=12 s | +0.1657 [−0.0451, +0.3518] **not sep** | **+0.0711 [+0.0308, +0.1127] SEP** | ÷4.85 | 🔴 **CONTROL FAILS** |
| ⭐ `lon.paired_leak_B3_minus_B1` H=12 s | +0.0524 [−0.0866, +0.1911] **not sep** | **+0.2442 [+0.1966, +0.2951] SEP** | ÷2.82 | 🔴 **CONTROL FAILS** |
| ⭐ `lon.paired_leak_B3_minus_B1` H=8 s | +0.0253 [−0.1175, +0.1561] **not sep** | **+0.1895 [+0.1514, +0.2242] SEP** | ÷3.76 | 🔴 **CONTROL FAILS** |
| ⭐ `lon.paired_clock_B4_minus_B3` H=12 s | +0.0209 [−0.1593, +0.2455] **not sep** | **−0.0359 [−0.0634, −0.0108] SEP** | ÷7.70 | 🔴 **CONTROL FAILS + SIGN FLIP** |
| `lon.paired_leak_B2_minus_B1` H=8 s | +0.0392 [−0.0740, +0.1466] not sep | +0.0058 [−0.0225, +0.0326] not sep | ÷4.00 | ✅ holds (now powered) |
| `lon.paired_clock_B4_minus_B3` H=8 s | +0.0371 [−0.0568, +0.1233] not sep | −0.0011 [−0.0232, +0.0196] not sep | ÷4.21 | ✅ holds (now powered) |
| `lat.paired_clock_B4_minus_B3` H=12 s | −0.0014 [−0.1270, +0.1101] not sep | +0.0027 [−0.0162, +0.0216] not sep | ÷6.27 | ✅ holds (now powered) |
| `lat.paired_clock_B4_minus_B3` H=8 s | −0.0100 [−0.0873, +0.0623] not sep | −0.0075 [−0.0233, +0.0085] not sep | ÷4.70 | ✅ holds (now powered) |
| `lat.paired_leak_B2/B3_minus_B1` (4 rows) | already SEP | still SEP, larger | ÷2.3–2.7 | (never a null) |

⛔ **THE SINGLE MOST IMPORTANT NUMBER IN THIS DOCUMENT.** The original longitudinal leak control had
**MDE = 0.1985 QWK**. The leak that is actually there is **+0.0711 QWK**. **The control was built with
2.8× too little resolution to see the thing it existed to catch.** It did not "pass" — it was
**structurally blind**. Six of the twelve rows have this property (`control_was_VOID_at_original_n`).

### 3.1.1 The third independent run — mine, on pod3

I launched a fresh `run_s3_characterisation.py` on **pod3** (A40, verified idle 0 MiB / 0 %; explicit
PID **1237515**, never `pkill -f`) against the same parity views, writing to `/workspace/s3repro`
(`/workspace` verified by a real `dd` test to accept a **full 3.0 GiB** — 3,145,728,000 bytes, not
truncated; `df` never consulted). It mined all 2,376 train + 600 val episodes (`mine_seconds` 775.2) and
completed. Artifact pulled and **md5-verified end to end** (`da57d963…` on pod3 ≡ `da57d963…` in repo):
`raw/s3_repro_pod3_primary.json`.

⭐ **RESULT — a THREE-way reproduction, and the strongest form of it:**

| node | A (`s3-decision-grade`) | C (`pod2-eval-host`) | **R (this task, pod3)** | 3-way agree | point spread |
|---|---|---|---|:--:|---:|
| `lat.paired_leak_B2_minus_B1` | +0.3727 SEP | +0.3740 SEP | **+0.3727 SEP** | ✅ | 0.0013 |
| `lat.paired_leak_B3_minus_B1` | +0.3968 SEP | +0.3902 SEP | **+0.3968 SEP** | ✅ | 0.0066 |
| `lat.paired_clock_B4_minus_B3` | +0.0027 not sep | +0.0063 not sep | **+0.0027 not sep** | ✅ | 0.0036 |
| ⭐ `lon.paired_leak_B2_minus_B1` | +0.0711 SEP | +0.0796 SEP | **+0.0711 SEP** | ✅ | 0.0085 |
| `lon.paired_leak_B3_minus_B1` | +0.2442 SEP | +0.2539 SEP | **+0.2442 SEP** | ✅ | 0.0097 |
| ⭐ `lon.paired_clock_B4_minus_B3` | −0.0359 SEP | −0.0475 SEP | **−0.0359 SEP** | ✅ | 0.0116 |

**My run reproduces run A to four decimal places on all six nodes** — the pipeline is deterministically
seeded, so the committed artifact is confirmed to be the output of that code on that data. Run C, from a
**different pod and a different cache instance**, agrees on **6/6 verdicts** with a maximum point spread
of **0.0116 QWK**. Operative blind floors agree: lat **0.6534 / 0.6493 / 0.6534**, lon **0.5323 / 0.5420
/ 0.5323**.

⇒ **The longitudinal leak is not a fluke of one run, one pod or one cache.** Class `MEASURED`, tier
**DECISION-GRADE** (three runs, two pods, two cache instances, md5-verified transfer).

### 3.2 `S3-BLINDvsCHANCE` — 4 of 8 arms move from "at chance" to separated

The firewall's own arms were read as **at chance** on the non-parity corpus. `MEASURED` at parity:

| axis · arm | original (n = 73/81) | **re-adjudicated (n = 520/558)** | verdict |
|---|---|---|---|
| ⭐ `lat.B1_sensor_only` (**= the S3-W bar**) | +0.1128 [−0.0010, +0.2271] **not sep** | **+0.2566 [+0.2075, +0.3090] SEP** | 🔴 **FAILS** |
| ⭐ `lon.B1_sensor_only` (**= the S3-W bar**) | +0.0676 [−0.0959, +0.2204] **not sep** | **+0.2881 [+0.2280, +0.3475] SEP** | 🔴 **FAILS** |
| ⭐ `lon.B3_FULL_CONDITIONING` | +0.1200 [−0.0529, +0.2913] **not sep** | **+0.5323 [+0.4780, +0.5832] SEP** | 🔴 **FAILS** |
| ⭐ `lon.B4_plus_clock` | +0.1410 [−0.0412, +0.3140] **not sep** | **+0.4964 [+0.4411, +0.5472] SEP** | 🔴 **FAILS** |
| `lat.B2/B3/B4`, `lon.B2` | already SEP | still SEP | (never a null) |

⇒ **Every blind arm, on both axes, is above chance at full power.** The reading that the sensor-only
arm sat at chance — which is what makes a *withheld-conditioning* S3-W variant look like it measures
perception against a ~0 floor — is **withdrawn**. The S3-W floor is **+0.2566 / +0.2881**, not 0.

### 3.3 🔴 `S1-BLINDvsMAJORITY` — the firewall that could not fail. This is a PROOF, not a power estimate.

The S1 blind-conditioning firewall ADMITTED all three variants. `MEASURED` from `S1_RESULTS.json`, and
the arithmetic is closed-form because accuracy is bounded above by the stated `acc_ceiling = 1.0`, so
`max possible (blind − majority) = 1.0 − acc_major`:

| variant | n dec. pts | n clusters | δ blind−major [CI95] | **MDE** | **max possible effect** | MDE as % of max | **can this test EVER fire?** |
|---|---:|---:|---|---:|---:|---:|---|
| ⛔ **H** "hard" | **8** | **6** | −0.5000 [−1.0000, +0.1111] | **0.5555** | **0.2500** | **222.2 %** | ⛔ **NO — PROVEN IMPOSSIBLE** |
| ⚠️ **NOGOAL** control | 30 | 20 | −0.0667 [−0.3429, +0.2122] | 0.2775 | 0.3000 | 92.5 % | ⚠️ only if blind captured **>92 %** of all headroom |
| ⚠️ **E** "easy" | 26 | 20 | +0.0385 [−0.2222, +0.2692] | 0.2457 | 0.3077 | 79.9 % | ⚠️ only if blind captured **>80 %** of all headroom |

⛔ **Variant H's minimum detectable effect is 2.22× the largest effect that can physically exist.**
Even a blind head at **100 % accuracy** (δ = +0.25) would have returned `separated: false`. Its
`ADMITTED` verdict is not weak evidence — it is **zero evidence**, and no amount of care in reading it
helps. This is the highest-proximity firewall row in the program (prox 0.900) and **it is absent from
the harvest's 14.**

⚠️ **And the firewall asked the wrong question.** It tests `blind` vs **majority**. The leak-relevant
contrast is `blind` vs **chance**, and `MEASURED` from the same file:

| variant | `acc_blind` | `acc_chance` | **blind − chance** |
|---|---:|---:|---:|
| **E** | 0.7692 | 0.4872 | ⛔ **+0.2820** |
| **NOGOAL** (*no goal at all*) | 0.6333 | 0.4833 | ⛔ **+0.1500** |
| H | 0.5000 | 0.5000 | +0.0000 |

⇒ **In the NOGOAL variant — where the model is given no goal whatsoever — option geometry and ego state
alone buy 15 accuracy points over chance.** That is exactly the quantity a circularity firewall exists
to surface, and **it was never tested with an interval.** The E variant buys **28 points**.

✅ **Credit where it is due:** `S1_T1_SLICE.md` §1.3 qualification 2 already says *"At n=26/20 clusters
this test cannot establish that the goal channel helps or that it leaks. ADMITTED here means 'not
refused', not 'certified clean.'"* That is honest and correct. **What this re-adjudication adds is that
for variant H it is not a matter of degree — the test is mathematically incapable of refusing** — and
that the un-intervalled `blind − chance` gap is large on two of three variants.

### 3.4 `SHUFFLE` (lead-state gate) — the control HOLDS in direction, but it changes the number by 2×

23 rows. The two load-bearing ones, `MEASURED`, plus a decomposition the gate did not do:

| cell | treatment `A−B` | shuffle control `A−B_shuf` | \|ctl\|/\|treat\| | capacity cost of 5 noise features |
|---|---|---|---:|---:|
| `ridge\|canonical` | +0.0018 [−0.0021, +0.0054] | −0.0014 [−0.0029, +0.0002] | 0.78 | +0.0014 |
| ⚠️ `gbm\|canonical` | +0.0051 [−0.0040, +0.0143] | −0.0052 [−0.0114, +0.0011] | **1.02** | **+0.0052** |

✅ **The control's DIRECTION is correct** — shuffled features *hurt* (−), real features *help* (+). It is
not leaking. But in the `gbm` cell **shuffling moves MAE by 1.02× as much as the real feature does**,
which means the pre-registered `A → B` contrast is **not capacity-matched**: arm B has 5 more features
than arm A, and part of what `A−B` measures is that capacity cost, working *against* the treatment.

Decomposing (⚠️ **additivity ASSUMED** — class `ESTIMATED`, point estimate only, no CI, because the
per-window data is not in the repo):

`signal = (A−B) + capacity_cost` ⇒ `gbm`: **+0.0051 + 0.0052 = +0.0103**, i.e. a relative reduction of
**2.32 %** against the published **1.159 %** — **2.0× the quoted effect.**

⛔ **The refusal STANDS.** The pre-registered rule is `FAIL ≤ 5 % or CI spans 0`; **2.32 % is still a
FAIL**, and the CI still spans zero. The 12.4 GB + 2–3 eng-day `obstacle.offline` ingest stays refused —
**but the number in the record understates the lead-state signal by half**, and anyone re-litigating the
gate on the published 1.159 % is arguing against the wrong figure. This agrees with the harvest's own
read (*"the refusal stands; more power does not rescue it"*) and adds the reason.

### 3.5 The 30 rows I could NOT re-adjudicate — ranked owed

| pri | family | rows | what unblocks it | cost |
|---|---|---:|---|---|
| **1** | `CE-CONTROL` (`bar-a-selector`) | 14 | the 600-ep val, i.e. **pod2 free** or the val copied to an eval-capable pod | eval-only |
| **2** | `CHANCE-BASELINE` (`h2-classifier`) | 10 | a **larger clip draw** (n = 322 already; more episodes will not help — this needs more *labelled clips*) | CPU |
| **3** | `DEAD-CONTROL` (`p1-envelope`) | 3 | the 600-ep val + v1 ckpt | eval-only |
| **4** | `S1-BLINDvsMAJORITY` | 3 | ⛔ **~103 AlpaSim scenes for the 40-cluster bar, ~513 for the 200-cluster bar** (measured yield 0.39 clusters/scene). **Not a compute problem — a download decision.** | data |
| **5** | `SHUFFLE` posthoc subgroups | 19 | a lead-state re-ingest — **and the gate already refused that**, so these are permanently owed unless the gate is reopened | n/a |

⚠️ Rows in family 2 are the case where **the H1 ×3.4 episode-shrinkage argument does not transfer at
all** — `paired_AP_vs_chance` is limited by labelled positives, not by episode count. Applying the H1
projection to them would be the same category error the harvest flagged for the McNemar row.

---

## 4. ⭐ DOWNSTREAM DEPENDENCY CHAINS — what has to be withdrawn

A leak found in a control that gated a published claim invalidates the claim. Traced explicitly:

### CHAIN 1 — `lon` leak controls fail ⇒ the longitudinal S3 skill bar is wrong by **+128 %**

```
lon.paired_leak_B2/B3_minus_B1  "not separated"  (n=73, MDE 0.1985)
   └─ licensed:  "conditioning alone does not move a blind longitudinal head"
       └─ licensed:  operative_blind_floor(lon) = 0.2334
           └─ CONSUMED BY:  S3_IMPLEMENTATION.md  pre_registered_skill_bars
                            S3 as specified  = 0.2334   ->  MEASURED at parity 0.5323
                            S3-W (withheld)  = 0.0676   ->  MEASURED at parity 0.2881
               └─ CONSUMED BY:  every future S3 longitudinal model score, since
                                "skill = QWK(model) - bar"
```
⇒ **W-1 (WITHDRAW).** The longitudinal S3 skill bars **0.2334** and **0.0676** are withdrawn. A model
scoring QWK 0.45 longitudinally would have been read as **+0.217 skill** and is in truth **−0.082** — a
**win reported where there is a loss**. The lateral bar 0.3898 → 0.6534 has the same defect (**+0.2636**).
🟢 **No arm has yet been scored against these bars**, so nothing published is contaminated *today* — this
is a **prevented** error, not a committed one. `MEASURED`: a repo-wide grep for `0.2334|0.3898|0.0676`
returns only S3's own files, the harvest index, and this document.

### CHAIN 2 — blind arms are above chance ⇒ "S3-W measures perception against ~0" is withdrawn

```
lat/lon B1_sensor_only blind_qwk_ci spans 0   (n=73/81)
   └─ licensed:  "the sensor-only blind arm is at chance"
       └─ licensed:  S3-W (conditioning withheld) has a near-zero floor
           └─ MEASURED at parity: +0.2566 (lat) / +0.2881 (lon), BOTH separated
```
⇒ **W-2 (WITHDRAW).** The reading that the withheld-conditioning variant scores against a ~0 floor.
S3-W remains the *right* variant to run — the decision-grade stream's *"S3-W is not optional; it is the
variant that measures anything"* is **strengthened**, not weakened — but its floor is **~0.26–0.29**, and
`skill = QWK(model) − 0` would overstate every S3-W result by that much.

### CHAIN 3 — 🔴 `firewall.H` cannot fire ⇒ the S1 admissibility sentence must be withdrawn

```
firewall.{E,H,NOGOAL}.blind_vs_majority_paired  "not separated"  (n_cl = 20/6/20)
   └─ licensed:  S1_T1_SLICE.md:87
                 "The S1 target is NOT recoverable from the conditioning channels.
                  It is admissible."
       └─ licensed:  GATE_RESULTS.md §0
                     "S1 · S2 · S4 · HP-4 (the strategic half) are UNBLOCKED ...
                      They are now blocked only on corpus size."
           └─ FEEDS:  STRATEGIC_TACTICAL_PROBLEM_SPEC -- 7 of 9 decision problems
```
⇒ **W-3 (WITHDRAW).** `S1_T1_SLICE.md:87` as written. The correct sentence is: *"the S1 target was not
shown to be recoverable from the conditioning channels **by a test that, for variant H, could not have
shown it at any value**."* The variant-H row carries **zero** evidence (MDE 222 % of max possible);
E and NOGOAL carry almost none (80 % / 92.5 % of max).
⇒ **W-4 (AMEND).** `GATE_RESULTS.md §0`'s *"blocked only on corpus size"* → S1 is blocked on corpus size
**and** on an un-cleared circularity firewall. Materially these are the same fix (more scenes: ~103 for
the 40-cluster bar) — **but they are different claims**, and only one of them is currently written down.
⇒ **ESCALATION:** the un-intervalled **`blind − chance` = +0.15 in the NOGOAL variant** (option geometry
+ ego state alone, *no goal given*) is a circularity signal that has never been tested. It should be the
first thing the enlarged corpus measures.

### CHAIN 4 — the shuffle control ⇒ the lead-state refusal stands, the number does not

```
paired_mae_A_minus_B_shuf  "not separated"  (n=126)
   └─ licensed:  "the A vs B contrast is causal"
       └─ MEASURED: |control| = 1.02x |treatment| in the gbm cell
           => A->B is NOT capacity-matched; corrected signal is 2.0x the published one
               └─ verdict rule FAIL <= 5%:  1.159% -> 2.32%  ==>  STILL FAIL
```
⇒ **No withdrawal.** ⚠️ **Correction to the record:** the lead-state relative reduction is
**~2.32 % (ESTIMATED, point only)**, not 1.159 %. The 12.4 GB ingest refusal is **unchanged**.

### CHAIN 5 — ⭐ the one that reaches the whole harvest: H1's projection scores **4/8**

The harvest's H1 ranks 789 nulls by `prox@600 = prox × 3.4` and flags `would_flip_mean`, calibrating on
**one** end-to-end case (`along_track_vs_cv`, point estimate moved 0.7 %). This re-adjudication produces
**twelve** end-to-end cases — the same probe, low n and high n, both measured. `MEASURED`, raw in
`raw/h1_calibration.json` and `raw/h1_projection_scored.json`:

| what H1 assumes | what 12 measured cases show |
|---|---|
| the point estimate holds (n=1: moved 0.7 %) | **\|movement\| median 75.3 %, max 649 %, mean 167 %** |
| sign is stable | ⛔ **3 of 12 flipped sign** |
| half-width shrinks ×2.8–3.9 (mean 3.4) | ✅ measured **×2.26–7.70, median 3.88** — the *one* assumption that held |
| `would_flip_mean` predicts the verdict | ⛔ **4 correct / 8 scored = 50 %** — 2 TP, 2 FP, 2 FN, 2 TN |

⇒ **W-5 (AMEND `PROGRAM_HARVEST.md` H1).** The ×3.4 half-width projection is **sound** and is confirmed
by 12 independent measurements. The `would_flip` column built on top of it is **not** — on the only
cases where truth is now known it is **indistinguishable from a coin flip**, because the point estimate
moves far more than the half-width shrinks. H1's own caveat (*"it does not predict a flip, because the
point estimate can move"*) is **correct and must be promoted from a footnote to the headline**; the
"**516 of 789 would separate**" figure must be quoted as *"516 rank highly enough to be worth
re-running"*, never as a count of recoverable results.
⇒ **W-6 (RETRACT, harvest H1.1 row 10 / H1.3).** *"the S3 firewall may be hiding a real leak of 0.166."*
**The leak is real — and it is +0.0711, 2.33× smaller than projected.** The direction of the warning was
right; the magnitude came from the n=73 point estimate, which is exactly the quantity H1 says not to trust.

---

## 5. RETRACTION_LOG entries — DRAFTED, NOT FILED

⚠️ Per the brief I have **not** edited `Project Steering/RETRACTION_LOG.md` (siblings are editing it).
These are handed over ready to paste, each with its **root-cause class** per operating-standard rule 4.

> ### R-A · class: **UNDER-POWERED NEGATIVE CONTROL READ AS A PASS**
> **Retracted:** "the S3 blind-baseline firewall found no longitudinal leak" (implied by
> `s3_blind_baseline_primary.json` `separated:false` on `lon.paired_leak_B2/B3_minus_B1`, n=73).
> **Correction:** at parity (n=520, two independent runs) both leaks are **CI-separated**: B2−B1
> **+0.0711 [+0.0308, +0.1127]**, B3−B1 **+0.2442 [+0.1966, +0.2951]**.
> **Root cause:** the control's MDE (0.1985 QWK) was **2.8× larger than the leak it existed to catch**.
> **Generalisation:** *for any firewall/negative control, publish the MDE beside the verdict. A control
> whose MDE exceeds the effect it guards against has not passed — it has not been run.*

> ### R-B · class: **A GUARD THAT CANNOT FAIL** (C13 / BOOST §7.3, negative-control side)
> **Retracted:** `S1_T1_SLICE.md:87` — "The S1 target is NOT recoverable from the conditioning channels.
> It is admissible."
> **Correction:** `firewall.H`'s MDE is **0.5555** against a **maximum possible effect of 0.2500**. The
> test could not have refused the task at **any** observed blind accuracy, including 100 %. E and NOGOAL
> sit at 80 % and 92.5 % of their maxima. Untested: `blind − chance` = **+0.28 (E)**, **+0.15 (NOGOAL)**.
> **Root cause:** the guard was evaluated against a **threshold rule** (`blind ≥ 0.98 × ceiling`) whose
> trigger point lies outside the metric's realised range, and a **paired interval** whose resolution
> exceeded the metric's own bound. Neither could fire.
> **Generalisation:** *before running a guard, state the value that would trip it and check that value is
> attainable. If it is not, the guard is a comment.*

> ### R-C · class: **PROJECTION MISTAKEN FOR PREDICTION** (n=1 calibration)
> **Retracted:** `PROGRAM_HARVEST.md` H1's operative use of `would_flip_mean` / "**516 of 789 (65.4 %)
> would separate at n=600**" as a count of recoverable results, and H1.3's "**a real leak of 0.166**".
> **Correction:** on **12** end-to-end cases, half-width shrinkage is confirmed (×2.26–7.70, median 3.88)
> but the point estimate moves **median 75 %, max 649 %**, with **3/12 sign flips**, and `would_flip`
> scores **4/8 = 50 %**. The named leak is real at **+0.0711**, not 0.166.
> **Root cause:** a projection calibrated on **one** observation (0.7 % movement) was applied as a
> predictor to 789. The single calibration case was a *treatment* contrast on a large, stable effect; the
> control contrasts are small effects on a re-fit model, where the point estimate is the noisy part.
> **Generalisation:** *never promote a screening rank to a prediction on n=1 of calibration; report the
> rank as a work queue and the outcome only when measured.*

> ### R-D · class: **UNCONTROLLED CAPACITY IN A "CAUSAL" CONTRAST**
> **Corrected (not retracted):** lead-state gate relative reduction **1.159 % → ~2.32 % (ESTIMATED)**.
> The `A → B` contrast adds 5 features, and the shuffle control shows those 5 features cost
> **+0.0052 MAE** in capacity alone — **1.02× the size of the treatment effect itself**.
> **Verdict UNCHANGED: FAIL** (bar is ≤ 5 %). The `obstacle.offline` ingest stays refused.
> **Root cause:** the pre-registered primary compared arms of **different width**; the capacity-matched
> arm (`B_shuf` vs `B`) existed in the same run and was never differenced.
> **Generalisation:** *when a treatment adds features, the pre-registered primary must be the
> capacity-matched contrast, not the width-mismatched one — the shuffle arm is already paid for.*

> ### R-E · class: **INSTRUMENT DUPLICATION DEFEATING A PACKAGED, TESTED FIREWALL**
> **Finding (no prior claim to retract):** `taniteval/taniteval/blind_baseline.py` — packaged, documented,
> with `taniteval/tests/test_blind_baseline.py` — is imported by **neither** stream that ran a firewall
> this week. `2026-07-26-4brain-s3/s3_blind_baseline.py` and
> `2026-07-26-4brain-gates/blind_conditioning_baseline.py` are **independent re-implementations**
> (`MEASURED`: neither file contains any `taniteval` import). **Both produced the under-powered nulls in
> R-A and R-B.** The packaged module emits neither an MDE nor a `separated_positive`; had it been the
> single implementation, the fix would have been made once.
> **Generalisation:** *a firewall is infrastructure. Two copies means two power ceilings nobody compares.*

---

## 6. What this changes about how a control is written — one paragraph, binding if adopted

Every one of R-A…R-E has the same shape: **a negative control was reported as a verdict without its
resolution.** The fix is one line in the emitter, not a process:

> **A control node MUST emit `mde` (its own half-width) and, for a bounded metric,
> `max_possible_effect` and `can_fire: bool`. A control with `can_fire: false` is emitted as
> `verdict: VOID`, never as `ADMITTED`/`not separated`.**

This is a ~10-line change to `taniteval/taniteval/blind_baseline.py` — **which is the module both
firewalls should have been importing** (R-E). Doing it there fixes both call sites at once and is the
cheapest structural item in this report.

---

## 7. DELIVERABLE MANIFEST

| # | artifact | what it is | where it lives | only ONE copy? |
|---|---|---|---|:--:|
| 1 | `CONTROL_READJUDICATION.md` | this document — frozen list, results, chains, retraction drafts | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-control-readjudication/` | ❌ |
| 2 | `raw/frozen_list.json` | the **75-row frozen list**, with family, n, prox and MDE per row | same dir | ❌ |
| 3 | `raw/readjudication.json` | per-row original → re-adjudicated verdicts, all four families, with power gain, sign-flip and VOID flags | same dir | ❌ |
| 4 | `raw/s3_two_run_repro.json` · `raw/s3_three_run_repro.json` | the independent-run agreement checks — **3 runs, 2 pods, 6/6 verdicts** | same dir | ❌ |
| 5 | `raw/h1_calibration.json` | 12 measured point-estimate movements + half-width shrinkages | same dir | ❌ |
| 6 | `raw/h1_projection_scored.json` | H1 `would_flip` confusion matrix, 4/8 | same dir | ❌ |
| 7 | `raw/control_sweep_raw.json` | the full 231-node control sweep (75 nulls + 156 separated) — the audit trail for the freeze | same dir | ❌ |
| 8 | `raw/s3_repro_pod3_primary.json` | ⭐ **my third independent parity firewall run** — md5 `da57d963…`, verified identical on pod3 and in repo | `repo:` same dir **and** `pod3:/workspace/s3repro/` | ❌ |
| 9 | `scripts/sweep_controls.py`, `scripts/freeze.py`, `scripts/adjudicate.py` | the sweep, the freeze and the adjudication, re-runnable | same dir | ❌ |

**Nothing produced by this task lives in only one place.** Everything is staged in the repo working tree.
**Nothing was committed and nothing was pushed** (per the operating standard).

**Pods touched:** `pod3` **only** — one job, launched by explicit PID **1237515**, never `pkill -f`.
`pod1` (v2corpus training, step 13,450/30,000), `pod2` (blind-imagination sweep, 97 % GPU) and the
**eval pod** (v5 hierarchical) were **never contacted**. pod3's `/workspace` was verified with a real
`dd` write (**3.0 GiB accepted in full**, 515 MB/s); `df` was not consulted. `/root` was written only for
a 500 MiB probe, which was deleted.
🔒 No clip UUIDs, episode ids or raw PhysicalAI content appear in any artifact.

---

## 8. ⚠️ ESCALATIONS — these need a decision, not an engineer

1. 🔴 **W-1/W-2 are a live trap, not a past error.** The S3 skill bars **0.2334 / 0.3898 / 0.0676 /
   0.1128** are still written in `S3_IMPLEMENTATION.md` as `pre_registered_skill_bars`. **No arm has been
   scored against them yet** — so this is the rare case where the correction lands *before* the damage.
   The moment an S3 arm is scored, a **−0.082 loss becomes a +0.217 win**. **Owner needed: whoever runs
   the first S3 model.** Fix = replace with the parity bars **0.6534 (lat) / 0.5323 (lon)**, S3-W
   **0.2566 / 0.2881**.
2. 🔴 **`firewall.H` is void and S1's admissibility sentence rests on it.** This is a **PI decision**, not
   an engineering one: either accept S1 as *not-yet-firewalled* and say so in the spec, or buy the ~103
   scenes that make the firewall capable of firing. **Do not leave the sentence as written.**
3. ⚠️ **H1's `would_flip` column is 50 % accurate and is currently the input to a re-scoring programme.**
   BOOST §7.1's ranked list is still the right work queue — but if any compute is booked on the strength
   of "516 of 789 would separate", that arithmetic now has 12 counter-observations. **Owner: BOOST §7.1.**
4. ⚠️ **`planner_p2.py` (HEAD) still calls `_jack_scalar`/`_jack_paired`** at lines 373/381/399/442/570/
   574/577 — `overlapping_holdout_se`, which biases the interval **1.107–3.100×** *and* the point estimate
   (**−6.67 %…+11.69 %**, up to **×−4.15 with sign flips**). ✅ **No row in this frozen list was
   adjudicated with it** (`MEASURED`: all 75 carry `paired_episode_cluster_bootstrap` or
   `episode_cluster_bootstrap`) — so this report is clean. **But P2's own closed-loop numbers are not**,
   and P2 is the CEM planner over the frozen v1 world model. **Owner: TanitEval.**
5. ⚠️ **Two independent firewall re-implementations in one week** (R-E), neither importing the packaged,
   tested `taniteval/taniteval/blind_baseline.py`. Both produced nulls this report had to overturn.
   **Whichever stream lands next must delete its copy** — this was already S3's own escalation and it is
   still true, which is the 10-day-README failure mode repeating.

---

## 9. §7.4 — what this unblocks, and what it does not

Per BOOST §7.4 rule 1, every stream must name the stream its result unblocks, or say plainly that it
unblocks nothing.

**UNBLOCKS:**
- **The 4-brain dominance program (S3).** S3 can now be scored, because it finally has **correct bars**
  (0.6534 / 0.5323, S3-W 0.2566 / 0.2881) and a **known-armed R2 on both axes**. Before this, scoring S3
  would have produced a wrong-signed skill number on the first arm.
- **BOOST §7.1's re-scoring programme.** It now has a **measured calibration** (12 cases) instead of one,
  and knows that its ranking is a work queue rather than a forecast. That changes what may be *promised*
  from the re-scoring, not whether to do it.
- **The strategic-brain corpus decision.** Escalation 2 converts "S1 is blocked on corpus size" into a
  concrete, costed bar: **~103 scenes to make the firewall capable of firing at all**, ~513 for a two-arm
  comparison. That is a number a PI can decide on.

**DOES NOT UNBLOCK:**
- The **v4 / restart** decision — no row in this list touches it.
- The **closed-loop** direction — its controls are in the `DEAD-CONTROL` / `CE-CONTROL` families, and
  those are the 30 rows I could not re-power. They remain **owed** and are the natural next agent, the
  moment pod2 frees or the 600-ep val is relayed to an eval-capable pod.

---

## 10. What was NOT done, stated plainly

1. **30 of 75 rows were not re-powered at all** (§2.1, §3.5). They are owed, not resolved.
2. The **capacity decomposition in §3.4 is `ESTIMATED`** — a point estimate assuming additivity, with
   **no CI**, because the lead-state per-window data is not in the repo. It is enough to show the
   published number is ~2× too small and that the FAIL verdict survives; it is **not** decision-grade.
3. I did **not** re-run the S3 non-parity arm. The comparison in §3.1 is
   *non-parity n=73* vs *parity n=520* — **both n and corpus differ** (§2.2). The parity arm is
   decision-grade because the non-parity caches fail an independent consistency check, **not** because
   the difference is purely power.
4. The **prose layer was not exhaustively swept.** I grepped the 222 `.md` files for the S3 bars and the
   S1 firewall claims specifically. A control null that exists only as a sentence, in a workstream I did
   not name, would not be in the frozen list. I am claiming *"no control-type null found where I looked"*,
   not *"none exist"*.
5. `RETRACTION_LOG.md` was **not edited** — the five entries in §5 are drafted for the orchestrator.
