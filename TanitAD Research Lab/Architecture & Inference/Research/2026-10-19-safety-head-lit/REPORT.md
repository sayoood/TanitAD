# SAFE-HEAD-LIT: how generator-scorer planners make the safety rows resolve fast, off-expert proposals, and what REFe should try in what order (2026-10-19)

**Provenance.** Research Lab literature pass (LAB-RUN 2026-10-19) by a Sonnet agent, briefed by the Master Mind, built on tip `d568e539`. No GPU, no experiment, no `git add`, commit or push. The agent's harness refused to let it write this file, so the Master Mind transcribed it from the hand-back. It extends `2026-10-18-ep-rl-lit/REPORT.md` and does not repeat it: yesterday's report asked how strong methods get high EP; this one asks why the NC, DAC and TTC rows mis-score the fast proposals that an EP-led pick change selects. Lever ids here are `SH1`..`SH8`; yesterday's `L1`..`L10` keep their meaning and are cited as "yesterday's L<n>".

**Verification.**
- `raw/verify_published_rows.py`: 102 of 102 tabled rows PASS. The printed numbers of a row must occur, in order, on one visual line of one page of the banked PDF. "p" is the physical PDF page, 1-based, which can differ from the printed page number.
- `raw/verify_quoted_claims.py`: 52 of 52 prose claims PASS against the PDF text. This report quotes nothing longer than a few words and paraphrases the rest.
- `raw/absence_scan.py` (positive control built in): see section 2.3.
- Everything under `raw/` regenerates with `python make_tables.py`, `verify_*.py` and `absence_scan.py`; the library arguments are in each script's docstring.

**Evidence classes** (tags in the text): **PUB** = PUBLISHED, from a banked primary, PDF-verified. **MEAS** = MEASURED by us, register row or artifact named. **EST** = ESTIMATED, the arithmetic is in `raw/make_tables.py`. **HYP** = HYPOTHESIS. **INH** = INHERITED from another agent or document and not re-verified. Intervals in the register rows quoted below answer one question only, "would another draw of LOGS say this" (paired log-cluster bootstrap, one training seed); none of them speaks for another training run or another inference run (`H-ESTIM-SEED-1`).

## 0. Headline

**Needs the Master Mind (integration, before anything else):**
1. **Adopt, amend or reject the proposed rows in section 7** (3 GAC rows, 6 BACKLOG rows, 9 KB lines). They exist only in this file until landed.
2. **`HEADROW-PROBE` is running, and its two outcomes cannot separate a third cause.** CAPACITY-LIMITED and FEATURE-LIMITED leave open a COVERAGE-LIMITED reading (label support on fast, off-expert proposals), which looks FEATURE-LIMITED to a probe fitted on the same sparse labels. Ask the probe agent to save per-proposal OOF predictions together with each proposal's speed ratio, and register `FASTNEG-CENSUS` (0 GPU) **before the probe outcome is read**. Registering it after is a post-hoc stratification.
3. **Library landing.** `library.json` and `LIBRARY.md` are generated files and CRLF, like the blobs at the tip. If the tip moved after `d568e539`, do not overwrite: re-run `raw/bank_commands.sh` (NOT re-run, fallback only) in an export of the new tip. 4 PDFs are new; 17 entries gained a tag and a `cited_by`.

**Findings.**
1. **No banked paper prints a per-head accuracy, AUC or calibration number for NC, DAC or TTC** (31 text layers scanned, with a positive control; section 2.3). The request for "published per-head accuracy" has no answer in the literature we hold; this is an absence claim scoped to those 31 papers plus 4 web searches made earlier in this run (not re-run).
2. **The closest published evidence for off-proposal failure is TOAD Tab.7 p15 (PUB).** On iPad's proposals, a scorer trained on a fixed grid gives DAC 87.3 (GTRS) and 85.1 (SparseDriveV2) against iPad's own 94.9, and a search against the GTRS scorer reaches DAC 77.8. A scorer trained on freely decoded trajectories (DrivoR) gives DAC 97.8 and holds 97.1 under search. What changed between the scorers is the trajectories they were trained on, among other things; it is a comparison of three systems, not a one-factor ablation.
3. **Three kinds of change cut safety failure rates in published single-factor rows** (section 2.1): proposal-anchored scene features (DAC failures -19 % to -49 % relative), per-step or fine-grained safety factors (-30 % to -38 %), and designed hard-negative banks (-3 % to -21 % on strong bases). Backbone scale (Hydra-MDP++, -29 % to -65 %) is the largest, and the least transferable to a frozen REFe trunk.
4. **No paper shows that a better safety head unlocks more EP at a fixed pick rule** (section 3). Where the scorer is the only thing that changes, EP is flat or lower (DA-WAM -1.39, TOAD DrivoR -6.8, FlowR2A knob -2.1). EP and safety rise together only when the encoder or generator improves (iPad +12.7, Hydra-MDP++ backbone +5.3) or when an over-safe reward is rebalanced (ZTRS +20.5 with safety held). For REFe the value of better safety rows is therefore **permissive**: they remove the veto on an EP-led change. That reading is HYP and is what `H-REFE-RL1-NCRES` already says.
5. **REFe has two safety-failure populations, and they need different levers** (section 1): the standing near-twin grazes at the base pick (60 % of DAC-only zeros, within-token DAC AUC 0.46 on them, MEAS) and the fast proposals an EP-led change admits (three arms, indirect, MEAS). The registered HN screen is lateral-only and addresses the first. Nothing registered addresses the second.
6. **The ranking** (section 4.3 and 4.4, EST). If the probe reads CAPACITY-LIMITED: SH2 (row MLP), SH6 (two-scorer ensemble), SH1 (fast-negative bank). If FEATURE-LIMITED: run D1 (0 GPU) first; SH1 is first if it reads COVERAGE, and SH3/SH4/SH5 (scorer-branch, cached `visual_ctx`) are next if the information is in `visual_ctx`; if it is not, only the trunk (SH7/SH8, A40, PI-gated) can help. The scorer's gradient is detached from the trunk (`model.py:784`), so no scorer-only lever can repair a trunk-limited reading.

## 1. The problem as the register states it, and two facts that change the plan

All rows: T1 NAVSIM v1.1 navtest, 12,146 tokens / 136 logs, `x100`, one training seed, paired log-cluster bootstrap 10,000 resamples seed 20260927. STRATEGIC family PAUSED (PI ruling 2026-09-19, ITEM 25). **MEAS**, register rows named.

| row | what it says |
|---|---|
| `D-REFE-RL1-HEADWISE` | The zero-training composite V1 (RL R0B's EP and comfort heads, the deployed scorer's NC, DAC, TTC, DDC, lane): PDMS +0.095 [-0.074, +0.257], **NC -0.124 [-0.237, -0.016] adverse**, DAC -0.115, TTC -0.132, EP +0.530. Every safety head is the base scorer's, yet NC drops. |
| `D-REFE-HEADFIT-F10` | EP-R1's conditional EP row: OOF unsafe-pick rate +0.0102 [+0.0065, +0.0146], dPDMS -1.145 (selection tier). |
| `D-REFE-EPR1UB-1` | An anchor that pushed the EP row's scores on unsafe proposals down (mean sigmoid 0.43 / 0.26 / 0.14) still raised the unsafe-pick rate in all 18 cells (+0.0135 to +0.0182). The leak is in the NC/DAC rows. |
| `D-SRTTC-DEV-F10` | A head-only refit on S0's frozen features does not move the pick (dev gain +0.100 / +0.078 against a +0.30 bar); learning curve FLAT. |
| `D-REFE-P2-FULLTABLE` | Oracle minus pick = 9.41 = non-zero ranking 5.88 + DAC-zero picks 2.50 + NC-zero picks 1.01 + both 0.02. |

**Two facts from the register that this plan depends on and that are easy to lose:**
- **(a) The standing zero pool is a ~0.3 m-resolution problem, not a gross-failure problem** (`D-REFE-ZEROTAIL-3`, MEAS). 60.1 % [51.4, 68.0] of u400's DAC-only selection zeros are ON_PATH near-twin grazes: the pick matches the best-ranked non-zero proposal to a median 0.29 m lateral, with length ratio 1.01 and the same manoeuvre. The DAC head's within-token AUC on those tokens is 0.46, against 0.92 on random tokens. 42.0 % [33.9, 49.4] exist only in NAVSIM's simulated tracking.
- **(b) The label banks do not obviously cover the band an EP-led change selects.**
  - The registered HN screen's offsets are lateral only, 0.15 to 0.60 m (`PREREG_P6HN_SCREEN` section 2.2).
  - The TargetBank's candidate set (`refe/build_scorer_targets.py:144`) has lateral ramps of +-2 and +-4 m and three longitudinal variants (x0.5, x1.5, stopped), plus jerky, reverse and over-curb. Its own header says a proposal outside that basis is extrapolation.
  - Oracle selection moves the longitudinal family away from the human (progress ratio 1.177, speed MAE 0.983 against 0.806; `D-REFE-SELRANK-1`, MEAS). If that ratio is relative to the human's progress, it is the band between the teacher and x1.5 (HYP).
  - The OnPolicyBank relabels the student's 64 proposals, so that band is populated exactly as often as the generator produces it. The label density there is not measured anywhere I could find (**UNVERIFIED**).

**Two populations follow.** **P-A:** the standing near-twin grazes at the base pick (3.51 PDMS points: DAC-zero 2.50 + NC-zero 1.01). **P-B:** fast, off-expert proposals that become picks only when something prefers progress. P-B is currently measured only through its effect (three arms above) and never directly. Section 4 treats them separately.

## 2. Q1: what lets the NC, DAC and TTC outputs stay accurate far from the expert

### 2.1 Published single-factor rows (PUB; `raw/published_deltas.csv`, `raw/published_subscores.csv`)

"rel" is the relative change in the sub-score failure rate, (100 - sub-score), negative = fewer failures. Deltas are in points. Bases differ and several rows change more than the scorer; the last column says which.

| class | source (arXiv, table, PDF page) | change | dNC | dDAC | dTTC | dEP | rel NC / DAC / TTC | caveat |
|---|---|---|---|---|---|---|---|---|
| A. Scorer reads the scene along the proposal | iPad 2505.15111 Tab.4 p8 | BEVFormer + proposal refinement -> ProFormer (proposal-aware spatial cross-attention) | +1.2 | +3.3 | +3.5 | +12.7 | -38.7 / -48.5 / -38.0 | whole-planner encoder, weak base (PDMS 79.4 -> 89.8) |
| A | iPad Tab.4 p8 | + proposal-centric mapping task | +0.2 | +1.4 | +0.1 | +1.7 | -10.5 / -40.0 / -1.8 | aux task |
| A | iPad Tab.4 p8 | + proposal-centric prediction task | +0.3 | +0.4 | +0.5 | +2.1 | -17.6 / -19.0 / -8.9 | aux task |
| A | SafeDrive 2602.18887 Tab.4 p7 | DiffusionDrive head -> ProposalNet (trajectory-guided deformable attention) | +0.1 | +0.8 | -0.7 | +0.9 | -8.3 / -21.6 / +16.7 | TTC worse; one seed |
| B. Per-agent, per-step safety factors | SafeDrive Tab.6 p7 | neither -> both factors | +0.3 | +0.5 | +1.2 | not printed | -37.5 / -33.3 / -30.0 | factors need agent instances |
| B | SafeDrive Tab.6 p7 | per-agent collision factor only | +0.3 | +0.2 | +1.3 | not printed | -37.5 / -13.3 / -32.5 | |
| B | SafeDrive Tab.6 p7 | per-step DAC with 9-point map lookup only | 0.0 | +0.5 | +0.7 | not printed | 0.0 / -33.3 / -17.5 | |
| B | SafeDrive Tab.5 p7 | scene-level -> + fine-grained safety (sparse world) | +0.3 | +0.8 | +0.5 | not printed | -37.5 / -44.4 / -15.2 | |
| C. Sub-score supervision and weighting | Hydra-MDP 2406.06978 Tab.1 p4 | overall-PDM-score-only -> sub-score heads | +0.4 | +2.8 | +0.4 | +2.8 | -16.0 / -25.2 / -5.3 | |
| C | Hydra-MDP Tab.1 p4 | + weighted confidence at inference | +0.2 | +4.4 | +1.0 | +0.2 | -9.5 / -53.0 / -14.1 | an aggregation change |
| C | Hydra-MDP++ 2503.12820 Tab.3 p7 | + weighted confidence | -0.1 | +4.0 | +1.0 | +0.6 | +4.0 / -50.0 / -12.2 | |
| C | Hydra-MDP++ Tab.3 p7 | + perception auxiliary tasks | 0.0 | -0.4 | +0.2 | -0.3 | 0.0 / +10.0 / -2.9 | **worse**; the paper says so (p7) |
| D. Designed negatives (label bank) | Versatile Samples 2609.01799 Tab.5 p8, DiffusionDrive | both designs vs plain scorer | +0.17 | +0.18 | n/p | n/p | -10.6 / -7.3 / n/p | strong base (NC 98.4, DAC 97.5) |
| D | Versatile Tab.5 p8, DiffusionDrive | longitudinal-toward-lead only / lateral-toward-boundary only | +0.15 / +0.08 | +0.10 / +0.20 | n/p | n/p | -9.3 / -4.0 and -5.0 / -8.1 | each alone gives its metric's full gain (p8) |
| D | Versatile Tab.5 p8, MeanFuser | both designs | +0.05 | +0.32 | n/p | n/p | -2.8 / -15.6 / n/p | lateral-only DAC +0.31 |
| D | DA-WAM 2608.19085 Tab.3 p10 | action-conditioned future, without -> with hard negatives | +0.23 | +0.30 | +0.48 | -0.50 | -20.5 / -21.1 / -13.1 | EP down |
| D | DriveSuprim 2506.06659 Tab.5 p7 (EPDMS columns) | single-stage -> coarse-to-fine | +0.3 | -1.1 | -0.1 | +3.9 | -10.7 / +44.0 / +2.9 | **DAC worse**: pruning by easy negatives |
| E. Per-candidate predicted future | WoTE 2504.01941 Tab.3 p7 | trajectory evaluation -> + BEV future states | +0.9 | +1.8 | +1.5 | +1.7 | -31.0 / -25.4 / -18.5 | |
| E | DA-WAM Tab.3 p10 | no future -> action-conditioned future | +0.43 | +0.31 | +0.85 | -0.89 | -27.7 / -17.9 / -18.8 | |
| E | DA-WAM Tab.3 p10 | no future -> shared global future (control) | +0.57 | +0.19 | +1.06 | -2.68 | -36.8 / -11.0 / -23.5 | the future must be per-candidate for EP |
| E | MM-Future 2609.20377 Tab.3 p6 | history only -> paired future | +0.1 | +0.2 | +0.5 | +0.1 | -8.3 / -11.8 / -11.1 | |
| F. Ensembles | GTRS 2506.06664 Tab.2 p4 (stage 1) | Aug -> 2-model / 6-model ensemble | -0.5 / +0.2 | +0.9 / +1.3 | -0.3 / -0.3 | -0.1 / -0.7 | DAC -45.0 / -65.0 | members differ in recipe, not seed |
| G. Backbone | Hydra-MDP++ Tab.1 p7 | ResNet34 -> V2-99 | +1.0 | +2.6 | +2.0 | +5.3 | -41.7 / -65.0 / -29.0 | changes everything, not scorer-isolated |
| G | CLOVER 2605.15120 Tab.20 p30 | scorer backbone DINOv2-S -> DINOv3-B -> Wan2.2-5B | | | | | test top-1 PDMS 92.4 -> 92.7 -> 92.8 | train 95.0 -> 95.1 -> 97.7: the train-test gap grows, test barely moves |

`n/p` = not printed in the table I verified. Two tables I did not turn into a row, on purpose: SparseDriveV2 Tab.6 p15 (attention-scorer EPDMS 87.7 for multi-head attention against 89.9 for deformable aggregation) differs also in the number of proposals (512 against 1024), so it isolates nothing; FlowR2A Tab.4a p8 (TTC 86.9 -> 88.8 -> 94.9 as the **generator** is conditioned on a score, sub-rewards, then per-timestep rewards) is a generator ablation and not a scorer one.

**What the class pattern says (PUB reading, HYP transfer):**
- Classes A and B are the two that **change what the safety output is computed from**. Both do it by making the proposal's own geometry address the scene: a deformable or grid-sampled read along the path (iPad p4/p8, SafeDrive p4, FlowR2A p20) or a per-step factor with a 9-point box lookup into a drivable map (SafeDrive Eq.3-4 p5). REFe's scorer instead flattens (x, y, yaw) into one vector before any attention (`model.py:560-562`).
- Class D is the only one that **changes the labels**, and it is the cheapest to port, because REFe's scorer branch retrains on cached `visual_ctx` without the trunk.
- Class C shows that sub-score structure and aggregation matter (sub-score supervision DAC -25 %; weighting DAC -50 % to -53 %), and that auxiliary perception tasks can hurt. REFe already trains six or seven sub-score outputs and aggregates in probability space (`planner.py:537`), so C is mostly spent.
- Class E lowers all three failure rates but costs EP when the future is shared (DA-WAM -2.68) and needs a new trunk-side predictor.

### 2.2 The only published off-proposal comparison: TOAD Tab.7 p15 (PUB)

iPad's proposals, NAVSIM v2 sub-score columns as printed (NC, DAC, TTC, EP shown). Row labels as in the paper.

| scorer applied to iPad's proposals | NC | DAC | TTC | EP |
|---|---|---|---|---|
| none (iPad's own pick) | 96.2 | 94.9 | 95.8 | 82.0 |
| re-score with DrivoR scorer | 98.7 | 97.8 | 98.7 | 75.2 |
| re-score with GTRS scorer | 95.3 | 87.3 | 95.6 | 69.9 |
| re-score with SparseDriveV2 scorer | 96.7 | 85.1 | 96.4 | 79.4 |
| search with GTRS scorer | 93.3 | 77.8 | 94.0 | 59.9 |
| search with SparseDriveV2 scorer | 96.9 | 86.9 | 96.9 | 74.7 |
| TOAD (search with DrivoR scorer) | 97.7 | 97.1 | 98.4 | 75.6 |

- The text states the mechanism: a fixed-vocabulary scorer gets no supervision away from its set (p7), the DrivoR scorer is forced to score freely decoded trajectories (p7), and the learned scorer is more reliable near the proposals it was trained on (p4). Ours to add: the GTRS and SparseDriveV2 DAC outputs are **worse than no scorer** on another generator's proposals, before any search.
- The safe scorer pays in EP (-6.8). Section 3 returns to this.
- **Scope (HYP):** these are three published systems that differ in backbone, data and heads. TOAD attributes the difference to the trajectories trained on; the table does not isolate it.

### 2.3 Per-head accuracy and calibration: what exists, and the scoped absence (PUB, MEAS scan)

- **CLOVER p8 and p27 (aggregate score, not a head):** for proposals with predicted score 0.95 or higher, the true mean score is 0.9753, P(R* >= 0.90) = 96.69 %, P(R* = 1) = 69.74 %, against a pooled full-score rate of 35.42 %; fixed groups (true >= 0.95 against <= 0.50) are separated with 94.72 % pairwise ranking accuracy. About 30 % of "high-confidence" proposals are not at full score at truth, and the numbers cannot say whether that is NC, DAC, EP or TTC.
- **World4Scorer Tab.2b p7-8:** pairwise ordering accuracy 64.2 % with correct bank-outcome labels, 55.6 % with the bank's outcome labels shuffled, 51.5 % with all labels shuffled. A 512-scene, single-seed rig (511 held-out scenes), so a control on label correspondence, not a benchmark.
- **Scoped absence.** `raw/absence_scan.py` extracted the text layers of **31 banked papers** and looked for a metric word (AUC, AUROC, ECE, calibration, reliability, concordance, pairwise ranking accuracy, precision, recall, F1) within 160 characters of a safety-head word (NC, DAC, TTC and spelled-out forms). **One hit, a false positive** ("reliability of our policy", 2509.17940 p8). Positive control: the minimum text layer is 21,784 characters, and 18 of the 31 papers contain a metric word elsewhere, so the scan can see these words. **Limits:** figures and appendix images carry no text layer; Hydra-NeXt turned up in a web search, is not banked, and no number from it is used here (anything from it would be PUBLISHED-SECONDARY and inadmissible for the register).
- => The in-fan head concordances we already hold (EP 0.887, TTC 0.722 inside the scorer's top-8, `D-REFE-SELRANK-1`, MEAS) and the DAC within-token AUC of 0.46 on near-twin grazes (`D-REFE-ZEROTAIL-3`) are **more per-head information than the literature prints**. The off-proposal per-head measurement is ours to make: it is yesterday's L1 Step A (`L1-STEPA-RUN`, launch-ready, A4/A5 exploit rates) and the stratified census in section 4.5.

### 2.4 Why safety rows fail off-expert, and what the published remedies share

1. **Label support (PUB).** Fixed-vocabulary scorers get no supervision off their set (TOAD p7). The label-bank papers rebuild the bank to cover the region where the deployed policy will pick:
   - CLOVER's 261-candidate pool (Tab.10 p16: 200 constant-speed lateral transitions, 12 off-road, 18 acceleration profiles, 9 stop-go, 10 approach-brake, 12 overshoot; only 37 of 261, 14 %, are longitudinal). At most 180 candidates per scene go to the full evaluator, feasible and near-feasible first with a small infeasible subset kept (p17).
   - World4Scorer's scene-matched bank following CLOVER (p5): 54.5 valid entries per scene on average, and 10.7 % of scenes lack enough trajectories in the required safety categories to fill the quota (p15).
   - Versatile Samples' lateral-toward-boundary and longitudinal-toward-lead banks (p1, p8).
   - DA-WAM's safety-critical hard negatives (p10).
2. **Refit to the deployed distribution (PUB).** FlowR2A's stage 2 freezes everything but the selector and refits it on online decoder proposals labelled by the simulator, plus 32 random vocabulary trajectories drawn under inverse-density weighting, to keep the selector calibrated across the full quality range (pp5, 22), because the stage-1 selector sees only vocabulary trajectories (p5). The paper names the selector as the more actionable bottleneck (p15). CLOVER's loop likewise refits the scorer to evaluator sub-scores (p10). REFe's `OnPolicyBank` is the same idea; what the literature adds is the **second half, the broad random complement**, which REFe's OnPolicyBank does not have (the TargetBank is the nearest, and is a fixed basis).
3. **It is not the features (PUB, one scorer).** Swapping CLOVER's scorer backbone across a large range moves test top-1 PDMS by 0.4 and train by 2.7 (Tab.20 p30). This is one rig and one metric, so it is evidence for a generalisation gap, not a proof against feature limits.
4. **Easy negatives dominate (PUB).** DriveSuprim says easy-to-reject options dominate the training process and gradient (p1) and answers with coarse-to-fine selection and self-distillation from teacher-generated soft labels (p5); its Tab.5 p7 shows the cost of the pruning, DAC -1.1.

## 3. Q2: does a better safety head unlock more EP? (PUB; the answer is "not shown")

Rows where **only the scorer, the label bank or the pick rule moved** (top) and rows where more than the scorer moved (bottom). Deltas in points.

| source | change | dNC | dDAC | dTTC | dEP | reading |
|---|---|---|---|---|---|---|
| DA-WAM Tab.3 p10 | hard negatives vs no future | +0.66 | +0.61 | +1.33 | **-1.39** | safety up, EP down |
| TOAD Tab.7 p15 | re-score iPad with DrivoR scorer | +2.5 | +2.9 | +2.9 | **-6.8** | safety up, EP down |
| FlowR2A 2606.24231 Tab.7 p15 | selector TTC weight 3 vs 1 (an inference-time knob, p15) | +0.3 | +0.1 | +1.1 | **-2.1** | the trade as a free knob |
| Hydra-MDP Tab.1 p4 | + weighted confidence | +0.2 | +4.4 | +1.0 | +0.2 | EP flat |
| Versatile Tab.5 p8 | designed negatives | +0.17 | +0.18 | n/p | n/p (EPDMS +0.34) | small, EP not isolated |
| DriveSuprim Tab.5 p7 | coarse-to-fine | +0.3 | **-1.1** | -0.1 | **+3.9** | the reverse trade: EP bought with DAC |
| ZTRS 2510.24108 Tab.8 p17 | reward safety-only -> + progress | -0.2 | -0.1 | +1.5 | **+20.5** | EP restored (63.3 -> 83.8) with safety held |
| iPad Tab.4 p8 | ProFormer | +1.2 | +3.3 | +3.5 | **+12.7** | encoder change; both rise |
| Hydra-MDP++ Tab.1 p7 | backbone | +1.0 | +2.6 | +2.0 | **+5.3** | backbone; both rise |
| WoTE Tab.3 p7 | + BEV future states | +0.9 | +1.8 | +1.5 | +1.7 | both rise, small |

**Reading (PUB for the rows, HYP for the use):**
- Sharpening the safety output alone is **flat or negative on EP** in every isolating row. The published remedy that raises EP together with safety is a **better encoder or generator**, or an **evaluator-in-the-loop reward** that carries the multiplicative gating (ZTRS).
- No row shows the REFe question directly: "hold the pick rule, improve NC/DAC resolution, and measure how much more progress a progress-preferring rule can then take". TOAD comes closest (search needs an off-proposal-accurate scorer, p6-7), and it ends at the same trade (EP -6.8).
- **The consequence for REFe is permissive, and it is HYP:** better NC/DAC rows are worth what an EP-led change would otherwise lose. First-order size of that loss on V1: NC -0.124, DAC -0.115 and TTC -0.132 (sub-score points) cost about 0.21 + 0.055 = 0.27 PDMS points (EST: 0.239 points of NC x DAC multiplier on ~88, plus 5/12 x 0.132); the EP gain is worth at most 5/12 x 0.530 = 0.22 before multipliers (EST). That first-order arithmetic gives about -0.05 where V1 measured +0.095, so it is an order of magnitude and not a decomposition. Reading it that way, **resolution alone, on V1's picks, is worth roughly +0.2 to +0.3** (EST). The larger prize is a stronger EP-led change that the safety rows can then permit, bounded above by the 5.88-point non-zero ranking term (`D-REFE-P2-FULLTABLE`, MEAS) and not estimable from the literature.

## 4. Q3: mapping onto REFe

### 4.1 Code facts (MEAS, read at tip `d568e539`; paths under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/`)

- `model.py:560-562` `score_q_mlp`: each proposal's (x, y, yaw) over the horizon is flattened into one vector and mapped to 256. The proposal is not a sequence of per-step tokens.
- `model.py:552-553` and `:426` `score_dec`: 4 `CrossBlock`s, each with self-attention across the 64-proposal set, cross-attention to `visual_ctx` (7,680 patch tokens x 256 with PETR-style 3D position embeddings) and an MLP. 4,213,760 parameters (`raw/2026-10-16-safety-head/RESULT.md`).
- `model.py:563` `score_head = Linear(256, 6 or 7)`: 1,799 parameters at 7 outputs. Aggregation is in probability space at `planner.py:537` (sigmoid, then NC x DAC x (5 EP + 5 TTC + 2 C) / 12 for `navsim_v1`).
- `model.py:759` `visual_ctx = scene_proj(visual)` and `:784` `sctx` is detached: **the scorer's gradient does not reach the trunk.** `ctx_cache.py` excludes `score_q_mlp`, `score_dec` and `score_head` from the trunk fingerprint and caches `visual_ctx` at 3.75 MiB per scene (bf16), so any scorer-only change retrains without the trunk. This is why SH1 to SH6 are dev-box-sized and why none of them can repair a trunk-limited reading.
- Because four cross-attention blocks with MLPs sit before the final `Linear`, the **final linear read-out is an unlikely bottleneck (HYP; I would put P(CAPACITY-LIMITED) near 0.2)**. The probe's real value is more likely to be the FEATURE-LIMITED branch, which then needs the finer diagnosis below.

### 4.2 Outcomes of `HEADROW-PROBE`, refined

The probe fits a linear read-out and a 2-layer read-out on S0's cached head-input features, rows NC and DAC first, and reads OOF concordance and pick-level OOF dPDMS and unsafe rate.

| outcome | meaning | what the probe can and cannot say |
|---|---|---|
| CAPACITY-LIMITED | the 2-layer row separates what the linear row does not; the information is in the 256-d token | decisive |
| FEATURE-LIMITED, **scorer-branch** | the 256-d token lacks it, `visual_ctx` has it | **the probe cannot tell this from trunk-limited** (it sees only the token) |
| FEATURE-LIMITED, **trunk** | `visual_ctx` lacks it | same |
| **COVERAGE-LIMITED** (added here) | the information is available, but the rows were never trained on enough fast, off-expert, lead-ward examples to use it | **indistinguishable from FEATURE-LIMITED to a probe fitted on the same sparse labels**; and a *global* flat learning curve (`D-SRTTC-DEV-F10`) is insensitive to it when that stratum is a small share of rows |

### 4.3 The eight levers (ESTIMATED reach; `raw/lever_ranking.csv`, regenerate with `raw/make_tables.py`)

Reach is in PDMS x100 points on the standing zero pool of **3.51** (NC-zero 1.01 + DAC-zero 2.50 picks, MEAS) as a published relative failure-rate reduction (section 2.1, discounted for strong bases) times that pool. SH6 and SH5 are applied to the DAC pool of 2.50 alone. **Three limits on reading it:** (i) a relative reduction in a sub-score failure rate is not a reduction in the zero-pick term; I assume proportionality. (ii) The pool is dominated by near-twin grazes (fact 1a), which the cited rows did not stratify. (iii) For SH1 the reach on the standing pool is not where its value lies (see the note below the table). Cost-equivalent = CPU-h x 0.1 + dev-box GPU-h x 1 + A40-h x 2, **a convention and not a measurement**.

| id | lever | reach (low-high) | cost: CPU-h / dev-box GPU-h / A40-h | reach per cost-equiv | evidence | applies if CAP / FEAT scorer-branch / FEAT trunk (rank) |
|---|---|---|---|---|---|---|
| **SH2** | row MLP: a 2-layer read-out per NC/DAC/TTC row replacing `Linear(256, 7)` (`model.py:563`) | 0 - 0.35 | 0.5 / 0.3 / 0 | 0.50 | HYP: no paper ablates linear against MLP; the probe is this test | yes (1) / no / no |
| **SH6** | two-scorer ensemble, P2-s0 + P2-s1, each scoring a common fan with its own `visual_ctx` (`planner.py:537`, `model.py:699`) | 0.12 - 0.38 | 0 / 3.0 / 0 | 0.083 | PUB (GTRS Tab.2 p4, members differ by recipe); seed-only transfer HYP | partial (2*) / partial (1*) / no |
| **SH1** | fast and lead-ward PDM-labelled negative bank, then a scorer-branch refit on cached `visual_ctx`, as arm C of the registered HN screen | 0.11 - 0.74 | 26.4 / 5.0 / 0 | 0.055 | PUB (Versatile p8, DA-WAM p10, CLOVER Tab.10 p16, FlowR2A pp5/22, W4S p5) + MEAS context; transfer EST | yes (3) / yes (2) / partial (1*) |
| **SH3** | per-step scorer tokens (position, yaw, speed, acceleration per step) in place of the flattened vector, per-step NC/DAC/TTC outputs, continuous TTC label | 0.28 - 0.70 | 20 / 10 / 0 | 0.041 | PUB (SafeDrive Tab.6 p7, ZTRS p16, FlowR2A p20); no printed selector ablation; transfer HYP | no / yes (3) / no |
| **SH4** | proposal-anchored sampling of `visual_ctx` along the projected path inside `score_dec` (grid-sample or deformable) | 0.35 - 0.88 | 0 / 16 / 0 | 0.038 | PUB on weaker bases (iPad Tab.4 p8, SafeDrive Tab.4 p7); transfer HYP | no / yes (4) / no |
| **SH5** | map-geometry factor: `H-REFE-AUXDRV-1` H2 plus a SafeDrive-style factor of the drivable field at 9 box key points, multiplied into p_DAC | 0.32 - 0.64 | 2 / 0 / 14 | 0.017 | PUB (SafeDrive Eq.4 p5, Tab.6 p7) + MEAS reach (`D-REFE-ZEROTAIL-3`: P7.1 Y-share 25.7 %) | no / yes (5) / no |
| **SH7** | per-candidate predicted future conditioning the score (WoTE, DA-WAM, MM-Future form), trunk-moving | 0.28 - 0.53 | 0 / 0 / 60 | 0.003 | PUB; EP -0.2 to -0.9 side effect except WoTE | no / partial (6*) / yes (2) |
| **SH8** | the P6 route, DAC gradient into the trunk | not estimated | 0 / 0 / 169 | n/a | MEAS coverage (58 % of the u400 DAC-only zeros, Y-PDMS 1.48 [1.04, 1.95]); effect UNMEASURED | no / no / yes (held by P6, PI-gated) |

`*` = conditional on a stated test: SH6 on a seed-ensemble transfer (P2-s1 had TTC -0.49 adverse against u400 and a seed floor of 0.33, `H-REFE-P2-REP-1`); SH1-trunk on D1 reading COVERAGE; SH7-scorer-branch only if the future predictor adds information the frozen `visual_ctx` lacks (HYP: it cannot).

**Cost bases:** SH1 CPU = 8,922 S0 tokens x 24 candidates (a design choice: 4 speed scales x 3 lateral phases x 2 profile shapes) x 0.444 s per candidate at stride 2 (MEAS, `build_scorer_targets.py` header) / 3600 = 26.4 CPU-h (EST); SH1 GPU = one more arm in the HN screen, whose arms A and B at 3 seeds cost 8 to 12 dev-box GPU-h (`H-REFE-HNSCREEN-1`) (EST). SH5 = `H-REFE-AUXDRV-1` Amendment A3, 12 to 16 A40-h (EST). SH6 = 1,705 tokens x 2 models x 1.2 to 1.8 s per token for a dump (`l1_stepA/RESULT.md`) plus analysis (EST). SH3, SH4 and SH7 are my estimates from the HN-screen calibration and carry no more weight than that.

**Note on SH1.** Its reach on the standing pool is the weakest part of its case. Its value is that it is the only lever aimed at **P-B**: the label-support gap in the band an EP-led change selects. Neither the register nor the probe measures that population directly, and SH1's first step (D1, 0 GPU) is what measures it.

**Is the ordering robust?** (`raw/lever_ranking.csv`, columns `rank_all_at_low_reach` and `rank_all_at_high_reach`.) SH7 is 6th and 7th at the two ends of every range, and SH5 is 5th and 6th. SH2 moves from 7th (zero reach) to 1st. SH6 is 1st and 2nd; SH1 is 4th and 3rd; SH3 and SH4 are 2nd to 5th. **The only robust statements: the two A40-priced levers are expensive for their reach; SH2 and SH6 are cheap options; the middle of the table cannot be ordered from the literature.**

### 4.4 What to run, by probe outcome

| probe reads | first | then | do not |
|---|---|---|---|
| **CAPACITY-LIMITED** (rows NC/DAC) | SH2 (a head swap on cached features, 0.3 dev-box GPU-h; needs a G-DVB entry) | SH6; D1 and SH1 if the larger head raises the unsafe-pick rate on fast proposals | start with SH3/SH4: they re-pay for a capacity the head already lacked |
| **FEATURE-LIMITED** | **D1 (0 GPU)**, then split three ways below | | assume trunk-limited before D1 and D2: `D-SRTTC-DEV-F10`'s flat curve is global |
| ... D1 reads **COVERAGE** | SH1 (CPU bank, then arm C of the HN screen) | SH6; the SH1 bank then also trains SH3/SH4 | ship SH1 without the replicate arm (below) |
| ... D1 reads no coverage gap, **D2 jumps with map scalars** | SH5 (H2, then the factor) | SH4 | |
| ... D2 jumps with **agent or lead scalars** | SH3 (per-step tokens + continuous TTC) | SH4; if SH3 fails with D2 positive, the information is in the geometry but not in `visual_ctx`: trunk-limited | |
| ... D2 shows **no jump** | stop spending scorer-side GPU: the label is not a function of anything the scene shows at this resolution (compare `D-REFE-ZEROTAIL-3`'s 42 % tracking-made zeros) | SH7/SH8 are the only routes and need the PI | |

### 4.5 The two diagnostics, and the SH1 success criterion

**D1: `FASTNEG-CENSUS` (0 GPU, CPU minutes, selection tier, navtest untouched).** Define each proposal's speed ratio r as its mean speed over the horizon divided by the scene's teacher (or base-pick) mean speed; strata are quartiles of r pooled over the table (the cut points fixed before any label is read). Report per stratum:
- **(a) label density**, as the share of rows in S0's bank, in the TargetBank and in the OnPolicyBank, and the NC/DAC failure rate in that stratum (labels only, no model);
- **(b) within-token AUC for the NC and DAC rows** of the nine SFT-4 sibling scorers on the dev table (3,137 sets / 24 logs, `D-REFE-SELRANK-1`; the same measure that read 0.46 on the near-twin grazes);
- **(c) a per-stratum learning curve**: train on the S0 bank with the fast stratum subsampled to 25, 50 and 100 %, and read the fast stratum's within-token AUC.

A rising (c) means COVERAGE-LIMITED; a flat (c) with low (b) means FEATURE-limited; a high (a) with normal (b) says there is no problem there. Exact input file names for the dev table are **UNVERIFIED in this pass**. Parts (a) and (b) need no probe output; (c) reuses the probe's fit code.

**D2: `PRIVGEOM-PROBE` (CPU, EST 13.5 CPU-h = 1,705 tokens x 64 x 0.444 s).** Add privileged geometric scalars from the PDM calculators (clearance of the ego box to the drivable boundary at 9 key points, time gap and closing speed to the nearest lead, per-step TTC margin) to the 256-d token and refit the linear and 2-layer read-outs. Labels may use privileged signals; this is a diagnostic and never a deployed input. A jump in NC/DAC AUC says the information exists upstream, and **which scalars carry it names the lever** (map: SH5; agents or lead: SH3 or SH7). No jump with oracle geometry says the label is not learnable from the scene at this resolution.

**SH1 success criterion (proposed; the Master Mind writes the prereg).** Arm C (fast bank) against paired arm A (BCE only) and arm B (lateral HN), three training seeds **plus a replicate of A**, because a separated interval from a one-seed arm is necessary and not sufficient (`H-ESTIM-SEED-1`). Endpoints, selection tier: (E1) fast-stratum within-token NC and DAC AUC, paired difference C minus A with its lower bound above 0 on the joint draw across seeds, and no stratum losing more than 0.02; (E2) the unsafe-pick rate of the V1-type composite (the EP-led head-wise composite) falling by at least half against A, which is the actual P-B test; (E3) no regression on arm B's near-twin DAC endpoint. A pass on E1 without E2 means the rows learned the stratum and the pick did not use it: report it as such.

### 4.6 Things the literature says not to do

- Do not add auxiliary perception losses to the scorer expecting safety gains: Hydra-MDP++ p7 measures the opposite (DAC -0.4, PDMS -0.5).
- Do not condition on a **shared** global future: DA-WAM p10 (EP 91.36 -> 88.68).
- Do not distil the overall PDM score only: Hydra-MDP p4 (DAC 88.9 against 91.7 with sub-score heads).
- Do not tune a coarse-to-fine pruning stage expecting free safety: DriveSuprim Tab.5 p7 (DAC -1.1).
- Do not rank the ensemble lever on GTRS's numbers alone: the members differ in recipe.

## 5. Rule Zero: the next lever, and what blocks the rest

**Next lever: `FASTNEG-CENSUS` (D1), 0 GPU, minutes.** It is the cheapest experiment that separates the third cause, it needs no new compute or data, and it turns the probe's FEATURE-LIMITED branch into an actionable one. If it reads COVERAGE, the SH1 bank is next (26.4 CPU-h plus 5 dev-box GPU-h as arm C of an already-implemented screen). **Not run in this pass** (no experiments were allowed); the design is in section 4.5 and the BACKLOG rows in section 7.

**Blockers, named:**
- `HEADROW-PROBE` (running): decides which column of section 4.4 applies, and its per-proposal OOF output is needed for D1(c).
- Dev-box GPU lock: held by other work; SH1, SH6, SH2 and the screen queue behind `L1-STEPA-RUN` (the head bank `NAVTEST-HEADBANK` was cancelled 2026-10-19).
- Amendment adoption for the arm-C extension of the HN screen (Master Mind) and a G-DVB entry for any new trainer flag (launch-gate rule, 2026-09-26).
- A40 allocation (PI) for SH5 (14 A40-h), SH7 (60) and SH8 (169, held by P6). Nothing here needs them unless D2 jumps or D1 and D2 both fail.
- Input file names for the dev table and the S0 bank are UNVERIFIED in this pass.

## 6. Limits

- The reaches are EST and share one weakness (proportionality of failure rate to the zero-pick term); the order of the middle of the table is not informative (section 4.3).
- Every PUB delta compares systems with different bases, and several change more than the scorer (the caveat column says which). Relative failure-rate changes at saturated bases (NC > 98) rest on very small absolute differences, and none of these papers prints an interval or a seed replicate for these rows.
- TOAD Tab.7 is a three-system comparison, not an ablation; Versatile's gains are +0.05 to +0.32, within what one seed can move.
- All MEAS rows quoted here are single-seed, selection or T1 tier as stated, and the intervals answer the episode question only.
- Hydra-NeXt is not banked. SafeAlign-VLA and TPSP (yesterday) remain abstract-only.
- The "P(CAPACITY-LIMITED) near 0.2" prior is a HYP from the architecture, not a measurement.
- Strategic family: PAUSED (ITEM 25); nothing here touches it.

## 7. Proposed register and backlog entries (NOT applied; the Master Mind adopts, amends or rejects)

### 7.1 `GOALS_AND_CLAIMS.md`

- `| **D-LIT-SAFEHEAD-1** (2026-10-19) |` **No banked paper prints a per-head accuracy, AUC or calibration number for the NC, DAC or TTC output of a NAVSIM scorer; the closest published evidence of off-proposal failure is TOAD Tab.7 p15 (fixed-vocabulary scorers give DAC 87.3 and 85.1 on iPad's proposals against iPad's own 94.9; search against the GTRS scorer reaches 77.8; the DrivoR scorer holds 97.1).** In isolating rows a better safety output leaves EP flat or lower (DA-WAM -1.39, TOAD -6.8, FlowR2A -2.1); EP rises with safety only when the encoder or generator improves or an over-safe reward is rebalanced (iPad +12.7, ZTRS +20.5). Scope: 31 banked text layers with a positive control, plus 4 web searches. 102/102 tabled rows and 52/52 claims PASS against the PDFs. `TanitAD Research Lab/Architecture & Inference/Research/2026-10-19-safety-head-lit/` | **PUBLISHED** (banked primaries); the REFe reading is HYPOTHESIS |
- `| **H-REFE-FASTNEG-1** (2026-10-19) |` The NC, DAC and TTC rows are **label-support-limited on fast, off-expert proposals** (COVERAGE-LIMITED, a cause the `HEADROW-PROBE` outcomes cannot separate from FEATURE-LIMITED), so a PDM-labelled bank of fast and lead-ward negatives lifts the fast-stratum within-token NC/DAC AUC and cuts the unsafe-pick rate of an EP-led composite by at least half against a paired BCE-only arm and its replicate, without regressing the lateral HN endpoint. Gate: `FASTNEG-CENSUS` (D1) reads COVERAGE first. Reach on the standing pool EST 0.11 to 0.74; the permissive value on V1 EST +0.2 to +0.3. Cost EST 26.4 CPU-h + 5 dev-box GPU-h. `.../2026-10-19-safety-head-lit/REPORT.md` sections 4.3 and 4.5 | **HYPOTHESIS; nothing run.** Falsified if (c) of D1 is flat, or if arm C does not beat the A replicate. |
- Append to **`H-REFE-RL1-NCRES`**: "UPDATE 2026-10-19 (Lab): the literature has no per-head accuracy table to test this against (`D-LIT-SAFEHEAD-1`); the closest is TOAD's scorer swap. `HEADROW-PROBE` cannot separate a label-support (COVERAGE) cause from FEATURE-LIMITED; `FASTNEG-CENSUS` does, at 0 GPU. The registered HN screen is lateral-only and does not address the fast band."

### 7.2 `BACKLOG.md`

- **FASTNEG-CENSUS** | **NEXT, CPU, minutes, selection tier, navtest untouched.** D1 of `2026-10-19-safety-head-lit/REPORT.md` section 4.5: per speed-ratio stratum, label density in S0/TargetBank/OnPolicyBank, NC/DAC failure rate, within-token AUC of the nine SFT-4 sibling scorers on the dev table, and a per-stratum learning curve. Register the strata cut points before reading a label. Decides COVERAGE-LIMITED vs FEATURE-LIMITED. Needs: the dev-table and S0-bank file names (UNVERIFIED).
- **HEADROW-PROBE addendum** | Save per-proposal OOF predictions and each proposal's speed ratio with the probe's output, so D1(c) can stratify them. **Must be registered before the probe outcome is read.**
- **FASTNEG-BANK** | **BLOCKED on FASTNEG-CENSUS reading COVERAGE.** Fast and lead-ward candidates (24 per S0 token, EST 26.4 CPU-h) added as a `fast` family in `build_scorer_targets.py` `candidate_set`, then arm C of the HN screen (+5 dev-box GPU-h EST, 3 seeds + an A replicate). Needs an amendment to `PREREG_P6HN_SCREEN` and a G-DVB entry for any new flag.
- **PRIVGEOM-PROBE** | CPU, EST 13.5 CPU-h. D2 of the report: PDM-calculator geometry scalars added to the head-input token; the scalars that carry the jump name the lever. Run only if D1 reads no coverage gap.
- **SCORER-ENS-S0S1** | EST 3 dev-box GPU-h, selection tier. SH6: P2-s0 and P2-s1 each scoring a common fan on the dev table. Needs `visual_ctx` for both on the 1,705 tokens (neither exists, `l1_stepA/RESULT.md`); reuses the `make_stepA_bundle.py` forward with `L1-STEPA-RUN`.
- **SCORER-BRANCH-PERSTEP** | **CONDITIONAL** on D2 naming agent or lead scalars (SH3) or on SH5 failing (SH4). EST 10 and 16 dev-box GPU-h. Not before D1 and D2.

### 7.3 Proposed lines for `Architecture & Inference/Research/KNOWLEDGE_BASE.md` (format as in the file)

- [2026-10-19] [lib 2606.07170 TOAD Tab. 7 p15, local full text] On iPad's proposals a fixed-vocabulary scorer's DAC is worse than no scorer (GTRS 87.3, SparseDriveV2 85.1 against 94.9) and search against GTRS reaches 77.8, while the DrivoR scorer gives 97.8 and holds 97.1 under search; EP -6.8 for the safe scorer — the only published off-proposal comparison of safety outputs, three systems not an ablation — `Research/2026-10-19-safety-head-lit/REPORT.md`
- [2026-10-19] [lib 31 text layers scanned, raw/absence_scan.py] No banked paper prints a per-head AUC, ECE or accuracy for NC, DAC or TTC; calibration-type numbers exist only for aggregate scores (CLOVER p8: s >= 0.95 -> true mean 0.9753, P(R* = 1) 69.74 %) and bank-label ordering (World4Scorer Tab.2b: 64.2 -> 55.6 %) — the off-proposal per-head measurement is ours to make — `Research/2026-10-19-safety-head-lit/REPORT.md`
- [2026-10-19] [lib 2602.18887 SafeDrive Tab. 6 p7, Eq. 3-4 p5, local full text] A per-agent per-step collision factor and a per-step DAC with a 9-point drivable-map lookup together move NC 99.2 -> 99.5, DAC 98.5 -> 99.0, TTC 96.0 -> 97.2 (relative failure -30 % to -38 %); the factors need agent instances — levers SH3/SH5 — `Research/2026-10-19-safety-head-lit/REPORT.md`
- [2026-10-19] [lib 2505.15111 iPad Tab. 4 p8, 2602.18887 SafeDrive Tab. 4 p7, local full text] Proposal-anchored scene reads cut DAC failures by 22 to 49 % relative (SafeDrive +0.8 DAC, iPad +3.3), with TTC worse in SafeDrive's row; REFe flattens the proposal before attention (`model.py:560`) — lever SH4 — `Research/2026-10-19-safety-head-lit/REPORT.md`
- [2026-10-19] [lib 2609.01799 Versatile Samples Tab. 5 p8, local full text] Each designed-negative family alone gives its own metric's full gain (lateral-toward-boundary -> DAC, longitudinal-toward-lead -> NC); the gains are +0.05 to +0.32 on strong bases — the fast, lead-ward family is the one REFe's registered HN screen lacks (lever SH1) — `Research/2026-10-19-safety-head-lit/REPORT.md`
- [2026-10-19] [lib 2606.24231 FlowR2A pp5, 15, 22, local full text] The selector's stage 2 freezes everything else and refits on online decoder proposals plus 32 random vocabulary trajectories under inverse-density weighting, to stay calibrated across the full quality range; the authors name the selector the more actionable bottleneck — REFe's OnPolicyBank lacks the broad random complement (lever SH1) — `Research/2026-10-19-safety-head-lit/REPORT.md`
- [2026-10-19] [lib 2605.15120 CLOVER Tab. 20 p30, Tab. 10 p16, local full text] Swapping the scorer backbone from DINOv2-S to Wan2.2-5B moves test top-1 PDMS 92.4 -> 92.8 and train 95.0 -> 97.7: the gap is generalisation, not features (one rig); the 261-candidate bank is 77 % lateral transitions — `Research/2026-10-19-safety-head-lit/REPORT.md`
- [2026-10-19] [lib 2608.19085 DA-WAM Tab. 3 p10, 2606.07170 TOAD Tab. 7 p15, 2606.24231 FlowR2A Tab. 7 p15, local full text] In rows where only the scorer or its labels change, safety rises and EP is flat or lower (DA-WAM -1.39, TOAD -6.8, FlowR2A -2.1); EP rises with safety only with an encoder or generator change (iPad +12.7) or a rebalanced reward (ZTRS +20.5) — better safety rows are permissive, not a source of EP — `Research/2026-10-19-safety-head-lit/REPORT.md`
- [2026-10-19] [lib 2503.12820 Hydra-MDP++ Tab. 3 p7, 2406.06978 Hydra-MDP Tab. 1 p4, 2506.06659 DriveSuprim Tab. 5 p7, local full text] Auxiliary perception tasks cost DAC -0.4 and PDMS -0.5; distilling the overall PDM score only costs DAC 91.7 -> 88.9; coarse-to-fine pruning costs DAC -1.1 — three published things not to do to a safety head — `Research/2026-10-19-safety-head-lit/REPORT.md`

## 8. Banking, admissibility and manifest

**Newly banked (4):** 2609.36438 World4Scorer, 2609.20377 MM-Future, 2609.22317 EditWM (screened, no sub-score ablation), 2602.18887 SafeDrive. **Extended (17 entries, tag `safety-head-lit` and `cited_by`):** 2406.06978, 2406.15349, 2503.12820, 2504.01941, 2505.15111, 2506.06659, 2506.06664, 2510.24108, 2601.05083, 2601.22032, 2603.29163, 2605.15120, 2606.07170, 2606.24231, 2607.00399, 2608.19085, 2609.01799. `python tools/kb_add.py --verify` in the export: **654 entries, 0 orphans**, plus 4 export-only CRLF artefacts (alpasim-egodriver-proto, alpasim-readme, alpasim-tutorial, hugsim-readme) whose blobs match their recorded hashes; those four are not staged. The diff against the tip is `raw/library_delta.json` (no extended entry changed any field other than tags, `cited_by` and note). The staged `library.json` and `LIBRARY.md` are CRLF, as the tip blobs are.

**Not admissible for the registry:** anything from Hydra-NeXt (not banked); any number read only from a paper's abstract. EditWM and Drive-JEPA (2601.22032) were screened for sub-score tables and none was used.

**Deliverable manifest** (all under `C:/Users/Admin/par18/lab19/out/`, mirroring the repo path; listed in `paths.json`, sha256 in `MANIFEST.sha256`; none is committed or staged in git):
- `TanitAD Research Lab/Architecture & Inference/Research/2026-10-19-safety-head-lit/REPORT.md` (this file; carried by the hand-back only)
- `.../raw/`: `verify_published_rows.py` + `.json`, `rows_def.py`, `pdfrows.py`, `verify_quoted_claims.py` + `.json`, `absence_scan.py` + `.json` + `_ids.txt`, `make_tables.py`, `published_subscores.csv`, `published_deltas.csv`, `lever_ranking.csv`, `gap_arithmetic.json`, `library_delta.json`, `bank_commands.sh` (fallback, not re-run).
- `TanitAD Research Lab/Library/library.json`, `LIBRARY.md` (generated, CRLF), and 4 PDFs under `Library/papers/`.
- **These files exist only in this one place until landed.**
