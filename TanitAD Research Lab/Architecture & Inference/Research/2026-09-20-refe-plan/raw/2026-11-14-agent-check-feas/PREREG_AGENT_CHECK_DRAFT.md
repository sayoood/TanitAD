# PREREG_AGENT_CHECK (DRAFT, not frozen): the agent check, D-VDAC-6, on P6

**Status: DRAFT 2026-11-14 by the REFe Research Lab teammate. Not frozen, not run, nothing launched.** The MM freezes it (values marked HYPOTHESIS are fixed at freeze and never tuned afterwards). Style follows `PK/raw/2026-11-09-scorer1p6-prereg/vdac/PREREG_VDAC_P6_A5.md`.
Companion: `FEASIBILITY_AGENTS.md` (same folder; every number below that is not re-derived here is MEASURED there, with its artifact). PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
Evidence classes: MEASURED / PUBLISHED / INHERITED / ESTIMATED / HYPOTHESIS / UNVERIFIED.

## 0. Why this card, and the claim it tests

REFe target PDMS >= 94.55 on navtest; P6 reads 91.196 (gap 3.355). The drivable-area perception line is closed on its levers (A5: HB X-BOUND, HC X-NULL, S1-Q flat ~0.83; `PK/raw/2026-11-09-scorer1p6-prereg/vdac/a5_read/hc_read/READ_HC.md`), and a DAC-only veto projects below the bar even with a perfect raster (`FEASIBILITY.md` section 5). The agent terms are the larger share of P6's gap: TTC term **1.527** (TTC 0 on 4.92 % of tokens) and NC 0 on **1.12 %** (136 tokens) (`PK/raw/2026-11-09-p6-navtest-read/gap_to_human/p6_gap_to_human.json`), regret split `F:/tanitad_scratch/selq2read/p6_e3_regret_split.json`: TTC-0 1.58 [1.29, 1.89], zero-pick selection 2.81 [2.34, 3.30] (NC 136 + DAC 248 tokens).

**Claim under test (H-AGENT-CHECK-1, HYPOTHESIS):** a vision-only occupancy head on P6's frozen trunk, read through a deterministic footprint rule, moves P6's pick by more than the P1 bar (d > 0.4971 PDMS on navtest) by catching the agent failures P6's own NC/TTC heads miss.

**What is already measured and shapes every bar (navtrain fit pack, 8,489 sets, `FEASIBILITY_AGENTS.md` sections 3 and 4):** TRUE agent veto +1.215 PDMS (TTC +1.181, NC +0.219); P6's own NC/TTC heads are strong (pooled AUC 0.973 / 0.952) and miss **confidently** (98.6 % of failing picks have own p_ok > 0.5); on P6's top-5 its own agent-ok AUC is **0.787**; a clean proposal exists for 98.2 % of failing picks.

## 1. Decisions the MM / PI must take before freeze

| id | decision | default in this draft |
|---|---|---|
| Q-A | S0-B runs on the **3,976 fit-pack sets whose DB is local** (48 of 101 logs; held-out 1,361 of 1,705), not on all 8,489 | yes; the other 4,513 follow when the 53 DBs land |
| Q-B | pod time for S1: **ESTIMATED 4.3 to 5.6 L40-h per trained arm, ~10 to 13 L40-h for the S1 set (section 7)** | needs PI approval; none requested before S0-B reads |
| Q-C | which readout is admissible: pose-only footprint (default) vs NAVSIM's own scorer run on a predicted agent LIST (CPU, ~0.5 s per token, 64 proposals) | pose-only first; the list-readout is the committed successor of S0B-READOUT |
| Q-D | whether a single-frame input is acceptable if S0-B says the future is what matters (S0B-STATIC) | PI decision; the card names it, never decides it |

## 2. Data and code pins (to be completed at freeze; MEASURED where a hash is given)

| input | pin |
|---|---|
| P6 checkpoint | sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1` (A5 section 4) |
| fit pack | `F:/tanitad_scratch/p6fit/fit_pack.npz`, sha256 `e58653c57641a659b5c200dc23a6c59791eb2f053c3a9806b2069b680a7d0a2a` (= the pod's, `pod_sha.txt`); 8,489 sets, 101 logs, `ckpt_step` 10,075 |
| held-out logs | `splits/sft_heldout_logs.txt` sha256 `b814998deeb40a9d7839cce2807f8f282ede76c7f2c5ccce97adc064b27f4902` (24 logs) |
| nuPlan DBs (label source) | `E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/` (top level + `trainval/`); the **freeze must list the sha256 of each DB used** (not computed here: 349 GB tree, not touched) |
| NAVSIM devkit for the identity control | v1.1 `F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896` + `F:/Archive/devbox-C/navsim/nuplan-devkit` |
| scripts to freeze | `scripts/ceil_fitpack.py`, `auc_conv.py`, `blind.py` (S0-A, done); S0-B scripts **not yet written** (named in section 3) |
| estimator | paired log-cluster bootstrap, 10,000 draws, seed 20260927, over logs (same as A5 section 7.2): it answers "would another draw of LOGS say this" and nothing about training or inference variance |

## 3. Stage 0: zero GPU, navtrain only, unkeyed

### S0-A (DONE 2026-11-14, MEASURED): oracle ceiling and what a check must beat
Outputs: `ceiling_fitpack.json`, `auc_and_conversion.json`, `p6_agent_blindspot.json`. Not re-run by the card; it sets the denominators below. **Denominator (all 8,489 sets): O_NC+TTC = +1.215; held-out 1,705: +1.666.** For S0-B the denominator is recomputed on the SAME covered sets.

### S0-B (to build; dev box CPU, one process): how much of the oracle survives a deterministic raster readout, and does it need the future?
Builds L-OCC rasters (`FEASIBILITY_AGENTS.md` 2.1 and 2.2, cheap path, no confidence filter) for the covered fit-pack sets, applies the footprint readout (4.1) to the banked proposals (`traj_xy`, `yaw`), and compares with the banked TRUE NC/TTC.

| variant | raster fed to the readout | purpose |
|---|---|---|
| **R-true** | true agents at the 8 future steps | the readout's own ceiling (the DAC analogue of V-GT: 14 % of the oracle) |
| **R-t0** | true agents frozen at their t0 pose for every step | a single-frame perception ceiling: what a head that sees WHERE agents are but not how they move could at best do |
| **R-cv** | true t0 pose + true t0 velocity, constant velocity | privileged-velocity bound: separates "needs motion" from "needs the true future" |
| R-dyn | R-true restricted to dynamic classes | static contact (NC 0.5) vs dynamic |
| **C-SHUF** (control) | the raster of ANOTHER set in the same log (derangement) | label-shuffled control: gain must be <= 0 and capture < 0.10 |
| **C-REG** (deliberate regression) | all-zero raster (must give gain exactly 0.0, identity) and all-one raster (every proposal vetoed, fallback to P6's pick: gain exactly 0.0) | proves the readout and the fallback are wired; a nonzero reading is INVALID |
| C-ID | cheap raster vs NAVSIM `observation` raster on >= 64 pod tokens (pod CPU, minutes) | label identity (G20 analogue); one-cell tolerance reported, bar: >= 0.99 cell agreement |
| G-FID | for R-true: precision and recall of "this proposal fails NC or TTC" against banked truth, over all proposals and over P6's top-5 | the readout's fidelity, reported per class |

Capture fraction `c(X) = gain(X) / O_NC+TTC` on the same covered sets, paired log-cluster bootstrap.

| id | condition (committed now) | next |
|---|---|---|
| **S0B-GO** | c(R-true) >= 0.50 (HYPOTHESIS: leaves margin against the 31 % deployable-to-true conversion measured for the DAC raster, 0.114/0.367, FEASIBILITY.md C2) AND c(R-t0) >= 0.25, controls valid | S1 (section 4), PI pod time (Q-B) |
| **S0B-STATIC** | c(R-true) >= 0.50 but c(R-t0) < 0.25 | the future motion is what carries the gain and P6's single-frame input cannot supply it (UNVERIFIED single time step, FEASIBILITY_AGENTS 5.7). Successor: **a motion-conditioned arm**: the head additionally reads ego-compensated history features, or the S1 head predicts per-cell flow supervised with L-OCC velocity; this is an architecture decision (Q-D). Cost not estimated until the decision |
| **S0B-READOUT** | c(R-true) < 0.50 | the footprint rule is the constraint, not perception. Successor, run in the SAME turn: **R-sim**, the true agent list scored by NAVSIM's own `score_plans` with the head's boxes in place of the logged ones (this is the identity-clean readout; CPU 0.5 to 0.7 s per token MEASURED) and a learned fusion `q = f(R-true features, P6 logits)` fit out-of-fold on fit-extra. If R-sim also reads < 0.50 of the oracle, the oracle itself (sub-scores) is not reachable from any agent-geometry check: **the agent check is closed and the lever moves to a direct top-5 re-ranker (section 6, X-NULL row 3)** |
| **INVALID** | C-SHUF capture >= 0.10, C-REG nonzero, C-ID < 0.99, or the covered-set denominator differs from the S0-A one by label provenance | fix, re-run; no row counts |
| **NOT RUN** | DB missing / script blocked | the row stands "NOT RUN: <reason>"; never read as a refutation |

Cost: **0.3 to 0.5 CPU-h for 3,976 sets (ESTIMATED, 0.2 to 0.4 s per token raster; MEASURED parts 0.070 s query + 0.034 s raster)**; readout minutes; no GPU; not keyed.

## 4. Stage 1 (pod GPU, needs PI approval): train the head, read it on navtrain held-out

Only if S0B-GO (or S0B-STATIC resolved by the PI). **Arms, each single-lever, one seed first:**

| order | arm | what | updates |
|---|---|---|---|
| 1 | **AH-s0** | occupancy head on P6's frozen `visual_ctx` (recipe of HC-s0: 1 m cells 80 x 64, 2 cross-attention blocks, ctx 256), output **8 time steps x {dynamic, static}**, BCE + Dice on L-OCC; trunk frozen, P6 bit-identical; vision-only (G19: ego, command, scorer logits disjoint) | 10,000, batch 8 |
| 2 | **AH-SHUF** | AH-s0's flags + a derangement of the label keys | 2,000 |
| 3 | **AH-REG** (deliberate regression) | the head fed `visual_ctx` replaced by zeros | 2,000 |
| 4 | AH-s1 | AH-s0 with seed 1 (replicate against the rig's seed floor; a same-seed re-run is bit-identical, so the seed MUST change) | 10,000, only on X-GO |

Labels: L-OCC from the 103,139-key navtrain store of the aux drivable set (keys, split, H48 exclusion as A5); trained on navtrain only, **never on navtest** and **never on the 8,489 read sets' logs**: the read sets are `is_heldout` plus `fit_extra`, whose logs are outside the head's training (UNVERIFIED for P6's own training, FEASIBILITY_AGENTS 3.1; the held-out column is the clean read).

**Reads (T0-like perception and a navtrain selection read; never driving performance, never a navtest claim):**
* perception: dynamic-occupancy IoU at 0.5 s and 2.0 s on the held-out frames, per step, per city;
* **entry read (the deciding one):** the head's rasters on the 8,489 fit-pack sets (forward on the pod, images are there) -> the section 3 readout -> paired gain vs P6's pick, per entry (hard veto at tau chosen on fit-extra OUT-OF-FOLD, soft multiplier), scored against banked TRUE sub-scores. **No parameter is tuned on the held-out 1,705.**

| bar | value (HYPOTHESIS, fixed at freeze) |
|---|---|
| **AH-P1** | entry gain on held-out >= **+0.30** PDMS with lower 95 % > 0. Basis: P1 on navtest is d > 0.4971; navtest/navtrain failure-share ratios are 1.45 to 1.93 (FEASIBILITY_AGENTS 3.3), so the navtrain-equivalent is 0.26 to 0.34 |
| **AH-Q** | AUC for agent-ok **within P6's top-5**: head-derived >= **0.90** (P6's own: 0.787, MEASURED) |
| **AH-F** | trained head minus its OWN initial head on AH-Q: lower 95 % > 0 |
| C-SHUF | AH-SHUF entry gain <= 0 and AH-Q < 0.55 |
| C-REG | AH-REG entry gain <= 0 (a zero-information head must not help) |
| capture | reported: gain / (R-true gain from S0-B on the same sets): how much of the readout's ceiling the perception recovers |

## 5. Stage 2 (MM, navtest, KEYED): not designed here
Only after an AH arm reads AH-GO on S1 and a replicate holds. It substitutes the entry into the P6 pick on navtest under the P1 bar 0.4971, the family and gate bars of `PREREG_VDAC_P6.md`, and costs **K + the number of selector reads** the MM names. No navtest truth is read by S0 or S1. Write it as a keyed step only when the MM has the S1 numbers.

## 6. Outcomes, committed before any data (Rule Zero: every row names its successor)

| id | condition | next (committed) |
|---|---|---|
| **AH-GO** | AH-P1, AH-Q, AH-F hold; C-SHUF and C-REG valid | AH-s1 (seed floor), then Stage 2 on the substituted pick; PI pod time ~5 L40-h + forward 0.4 |
| **AH-PERC** | AH-Q holds, AH-P1 fails (perception is there, the pose-only entry cannot convert it) | learned fusion `q = f(predicted raster features along the proposal, P6's six logits)` trained out-of-fold on fit-extra (the PRIVGEOM-SAFE structure, FEASIBILITY.md section 2, with its label re-derivation caveat R54 checked first); same pod, CPU-light |
| **AH-NOQ** | AH-P1 and AH-Q fail, AH-F > 0 (the head learns but not enough) | the registered one-step lever: **20,000 updates, one seed**, then X-NULL if still failing |
| **X-NULL** | both bars fail and AH-F does not separate | agent perception from P6's single-frame features is not moved by this lever. Successors, in this order: (1) motion-conditioned input (Q-D), (2) agent-LIST head + R-sim readout, (3) **direct top-5 re-ranker**: fine-tune the scorer's NC/TTC heads on the confident-miss cases (mine the 2.58 % failing picks and their top-5 neighbours from the label bank; the hard-negative machinery `PK/refe/hn_bank.py`, `hn_fingerprints.py` exists) with the AH-Q metric as the read. This is the cheapest experiment that could still make the agent terms move and must be run in the same turn |
| **INVALID** | a control fails, a gate fact contradicts the arm, or the reader's regression identity fails | fix and re-run the arm; no row counts |
| **NOT RUN** | blocked at preflight, gate or timeout | stands as written, never read as X-NULL; the blocker's smallest fix is the next step |

⛔ A separated one-seed interval is necessary, not sufficient for a lever claim: AH-GO triggers the replicate, never a claim by itself (`H-ESTIM-SEED-1`). The paired bootstrap answers "another draw of logs"; it says nothing about training seeds or the head's inference.

## 7. Cost and approvals (ESTIMATED unless marked)

| stage | what | CPU / L40-h | needs |
|---|---|---|---|
| S0-A | oracle ceilings | done, ~minutes CPU | none |
| S0-B | rasters + readout on 3,976 sets | 0.3 to 0.5 CPU-h; +0.4 to 0.6 CPU-h for the other 4,513 once their DBs land | MM schedules the one process; no pod |
| S0-C | identity control | minutes on a pod CPU | pod access (existing) |
| S1 | AH-s0 4.3 to 5.6 (HC-s0 4.3; the 8 x 2 output channels add compute) + AH-SHUF 0.9 + AH-REG 0.9 + one-time export / chain 2.2 | **~8.3 to 9.6 L40-h** | **PI pod time**; the draft requests nothing now |
| S1 forward on the 8,489 read sets | rasters + entry read | ~0.4 | same |
| conditional: AH-s1 | replicate | 4.3 to 5.6 | only on AH-GO |
| conditional: AH-NOQ 20,000 updates | | 8.6 to 11.2 | only on AH-NOQ |
| navtest read | Stage 2 | MM's keyed look | MM |

Recomputing L-PROP for the 60,000 train sets (NOT needed): ~8.7 to 12 CPU-h (0.52 to 0.73 s x 60,000, MEASURED per call).

## 8. The four metric families and the tier (binding)

Tier: **S0 and S1 are T0-like (true-label or navtrain-open-loop perception and a navtrain selection read); never a driving-performance claim.** Every family below is per-family, never pooled, with the section 2 estimator.

| family | S0 / S1 | S2 (navtest, if earned) |
|---|---|---|
| LONGITUDINAL | the TTC sub-score is the longitudinal agent term: per-sub-score gain (NC, TTC) of every entry, and gap-keeping is not computable at S0/S1 from the banked sub-scores (reason: no lead-agent distance in the fit pack; n = 0). The entry's effect on picked speed (proposal speed from poses) is reported | target-speed accuracy and distance-keeping, paired |
| LATERAL | not computable at S0/S1 (n = 0, the check changes which proposal is picked, not the lateral geometry); cross-track and curvature of the PICKED proposal vs P6's pick reported as descriptive | heading, curvature, yaw-rate, cross-track |
| TACTICAL | selection quality: best-of-64, zero-pick share, top-5 agent-ok AUC, per set | command adherence (`command_adherence_p6final.json` method) |
| STRATEGIC | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25): neither a pass nor omitted nor extended to the other three | PAUSED |
| perception | occupancy IoU per step / city (S1) | from the same forward pass; never deciding |

## 9. Information-disjointness and admissibility
* **Labels use privileged data (agent boxes, futures, ego) offline: admissible. Inference is vision-only:** the head reads images and the calibration table only, as P6's `visual` does; no ego, command, scorer logits, situation class or route (G19 pins it by autograd w.r.t. ego and command: None or exactly 0.0, with a non-zero control).
* The readout uses the proposal's own poses (finite-difference speed for the TTC projection). It uses no tracked-state steering input; the LQR-tracked variant is NOT used (steering-angle vs yaw-rate ruling, PI).
* The goal path and the check stay information-disjoint: the check's output replaces nothing but an NC/TTC-style factor in the pick; the command path is untouched. A G-DVB entry must pin it.
* Launch rule (PI 2026-09-26): no S1 arm starts without a launch-gate PASS token; each new flag gets its G-DVB entry; each reachability is traced flag -> loss with a non-zero gradient on the new parameters, a control that reads exactly 0.0, and a mutation (delete the line that emits the value and require RED). The post-training gate (every head parameter holds Adam state, non-zero first moment) applies.

## 10. Freeze record (to complete)
`frozen_utc:` ____ by the MM. This draft's sha256 is to be recorded OUTSIDE its bytes (`PREREG_AGENT_CHECK.sha256`). Scripts S0-B named in section 3 do not exist yet; the freeze requires them, their tests (C-SHUF, C-REG, C-ID as mutation arms), and the covered-set list sha256 before the first read.
