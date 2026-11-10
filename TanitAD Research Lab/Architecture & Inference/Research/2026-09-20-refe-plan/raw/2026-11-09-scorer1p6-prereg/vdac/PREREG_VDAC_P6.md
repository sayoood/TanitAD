# PREREG_VDAC_P6 (FROZEN): a vision-built drivable-area check in P6's selection (Decision A)

**Status: FROZEN by the Master Mind, by sha256, 2026-11-11, before any VDAC code, run, raster or number exists on P6.** Written
2026-11-10/11 by a design agent. MM decisions at freeze: D-VDAC-1 done (register corrected, R-VDAC-1); D-VDAC-2 the S0-NEAR rule below
stands; D-VDAC-5 H48 is used, and its log list and sha256 go into an amendment written BEFORE S1 trains; S0's population is the local
FIT pack (section 2). Open with the PI: D-VDAC-3 (VDAC-PhiT is not read unless ruled admissible; the default footprint is pose-only)
and D-VDAC-4 (S1 pod time; nothing trains without the PI's go and a launch-gate PASS).
It changes no earlier card. PI 2026-11-10: *"Go ahead with Decision A, write the design card"*. Design: `DESIGN_VDAC_P6.md`;
facts: `FEASIBILITY.md` (F0-F6). Base: P6 = `model_final.pt` sha256 `968a469d...` (`F:/TanitAD_p6read_work/ckpt/run/`).

## 0. What is known before the freeze

- MEASURED, E3 (F1): P6 91.196 vs human 94.551; gate term 3.040; DAC 0 on 2.050 % of tokens (249), NC 0 on 1.120 % (136).
- MEASURED on u400, not P6 (F2): a trained head on a frozen trunk reads IoU [0, 40) 0.723 (C3) / 0.764 (natural navtest) / 0.821
  (held-out navtrain). A pose-only hard veto at margin 0.25 m gains +0.114 [-0.019, +0.226] with that head, +0.367 [+0.015, +0.638] with
  the TRUE raster, +0.794 with the true raster on tracked states, against an oracle DAC veto of +2.546 [+2.196, +2.880].
- ESTIMATED (F5): oracle DAC veto on P6 ~1.52 points; projected true-raster hard veto ~0.22. The bar below is 0.4971.
- No drivable head exists on P6's trunk (F0 C1). The untrained-head floors 0.13-0.27 are an instrument property, not a probe result.

## 1. Arms

**Heads (S1, trained on the pod; `DESIGN_VDAC_P6.md` section 4):** HA-s0 (seed 0), HA-s1 (seed 1), HA-SHUF (seed 0, label keys deranged,
2,000 updates). Recipe = S1c (`PK/eval/PREREG_S1C.md`, 10,000 updates, batch 8, visual tap, depth 2, feed off) with `--ckpt` = P6 and the
held-out set **H48** excluded from the head's labels (list and sha256 written into this card at freeze).

**Selection arms (S2, CPU, no training):** each changes only P6's DAC probability and picks with P6's own logits and proposals.

| arm | DAC factor | role |
|---|---|---|
| **VDAC-Phi** (HA-s0) | p_DAC x Phi(d_k / sigma_b) | **primary** |
| VDAC-Phi (HA-s1) | same, the seed-1 head | replicate (training variance of the head) |
| VDAC-PhiL | p_DAC x sigmoid(a + b d_k + c s_k + e t_k), fitted on H48 with the head out of sample, P6's logits unused | secondary: reported, names the next lever only |
| VDAC-V | AUXVETO's veto (d_k < -0.25 m drops k; all dropped => P6's pick) | comparison: soft vs hard |
| VDAC-SHUF | VDAC-Phi with HA-SHUF | deliberate regression |
| VDAC-PhiT | VDAC-Phi on NAVSIM's LQR-tracked states | only if the PI rules it admissible (D-VDAC-3); otherwise not read |
| O-DAC | oracle: drop every proposal with NAVSIM DAC = 0 | reported ceiling, never deployable |

d_k, s_k, t_k, the footprint, the 0.1 s interpolation, the mask-to-distance transform and sigma_b are defined in `DESIGN_VDAC_P6.md`
section 5.1 and are copied verbatim into the frozen card. **No weight, margin, threshold or member is chosen on navtest.** sigma_b and
VDAC-PhiL's four coefficients are fixed at the end of S1, on navtrain, before the S2 reader runs.

## 2. Stages and their bars

### S0: the ceiling, zero GPU, navtrain only (no navtest read)

Population (MM at freeze): the local navtrain FIT pack `F:/tanitad_scratch/p6fit/fit_pack8.npz` (8,489 sets with P6's logits,
per-proposal NAVSIM sub-scores and proposal poses), restricted to the sets whose token is in `label_store`; set and log counts are
reported before the read. The pod's ~60,000-set bank is NOT read: P7 trains on that pod, and no load is added to a training pod.
The FIT sets are in-sample for P6, so P6 fails on them less often than on navtest; R is a ratio of gains and is read as a ceiling
estimate, and that limitation is printed with it. Field: the TRUE raster from `label_store` (sha256 `4a17db67...`). Arms: O-DAC, V0, V25,
Phi-GT at sigma 0.5 / 1.0 / 2.0 m (pose-only; tracked if admissible). Quantity: R = gain(rule) / gain(O-DAC), PDMS x100 over the sets,
paired log-cluster bootstrap 10,000, seed 20260927. R_max = the largest point R over V0 and the three Phi-GT rules.

| outcome | condition | next (committed) |
|---|---|---|
| **S0-GO** | R_max >= 0.33 | S1 (pod time is the PI's, D-VDAC-4) |
| **S0-NEAR** | 0.28 <= R_max < 0.33 | the MM may authorize the E3 oracle rows (O-DAC, V0, Phi-GT at 1.0 m on the banked P6 table and the navtest store, K + 3) as the tie-break: Phi-GT's E3 gain >= 0.4971 => S1, else S0-STOP |
| **S0-STOP** | R_max < 0.28 | no DAC-only training. Next: the agent-check card (label build from NAVSIM's agent annotations) with the DAC output bundled into the same head arm, plus VDAC-PhiT's ceiling if admissible |
| INVALID | any S0 control fails | fix the instrument, re-read |

S0 controls: an all-drivable field gives Delta = 0.0 exactly and 0 proposals changed; O-DAC's gain >= every rule's; a y-flipped true
field reads below the true field; the reader reproduces P6's banked pick on every set.

### S1: perception on held-out navtrain (H48), BEFORE any navtest read

Tier T0-like (head perception on banked frames, never driving performance). Paired log-cluster bootstrap over H48's 48 logs, 10,000,
seed 20260927; IoU reads reweighted to navtest's city mix (LV 4,064 / BOS 3,731 / PIT 2,437 / SG 1,914 of 12,146).

| id | metric (HA-s0 unless stated) | bar |
|---|---|---|
| **S1-P1** | drivable IoU, x in [0, 40) m, mask `logit > 0` vs label, all H48 frames, city-reweighted | point >= **0.80**; per-city values and the CI reported |
| **S1-Q** | pooled AUC of -d_k for NAVSIM DAC = 0 over P6's proposals on H48 sets that carry P6 labels | point >= **0.90** (the S1 prereg's P3 mark; true-field 2 m ceiling on u400's C3 was 0.932) |
| **S1-F** | trained minus the same seed's INITIAL head, paired, on S1-P1 and S1-Q | both 95 % lower bounds > 0 |
| S1-R | HA-s1 meets S1-P1 and S1-Q | reported; if it fails, every S2 statement carries "single head seed" |
| C-SHUF | HA-SHUF's S1-P1 | must be < 0.50, else **INVALID** (the instrument passes a head without map information) |
| C-FLIP | HA-s0's prediction y-flipped, S1-Q | must be < 0.80, else **INVALID** |

Reported, not deciding: near-boundary SDF MAE and RMS (|SDF_gt| < 3 m), sigma_b, the true-field AUC on the same proposals, P6's own DAC
logit's AUC there (P6 is in-sample on every navtrain log: descriptive only), IoU by band 0-20 / 20-40 / 40-72 m and per city.

| outcome | condition | next (committed) |
|---|---|---|
| **S1-GO** | S1-P1, S1-Q, S1-F hold, controls valid | S2 |
| **S1-COMP** | S1-Q holds, S1-P1 fails, LV >= 0.80 and a non-LV city < 0.70 | HA-CB (city-balanced draws), its own pair of seeds |
| **S1-BOUND** | S1-P1 holds, S1-Q fails | HC (1 m cells, boundary-weighted Huber) |
| **S1-FIT** | both fail, in-sample IoU >= 0.85 | HB (a LoRA-adapted copy of P6's trunk for the head only; P6's trunk untouched) |
| **S1-WEAK** | both fail, in-sample IoU < 0.85 | HB and HC as one registered pair under their own card |
| INVALID | C-SHUF or C-FLIP fails, or a launch-gate fact contradicts the arm | fix, re-run the affected arm |

### S2: the selection read on E3 (only after S1-GO)

E3 = full navtest, 12,146 tokens, 136 logs, T1-class open loop, NAVSIM v1 PDMS x100 by table lookup
(`F:/tanitad_scratch/selq2read/table_p6final_c.npz`, sha256 `3bda2080...`). Paired log-cluster bootstrap, **10,000 draws, seed
20260927**. Every arm vs P6.

| id | metric | bar |
|---|---|---|
| **P1** | VDAC-Phi (HA-s0) - P6 | CI-lo > 0 **and** d > t x s_p = **0.4971** (s_p 0.1791, df 4, t 2.776; the single-run floor of `PREREG_DOSE_P6.md` D1) |
| **P1r** | VDAC-Phi (HA-s1) - P6 | CI-lo > 0 |
| **M** | DAC-zero pick share (pick DAC sub-score exactly 0), VDAC-Phi (HA-s0) - P6 | CI-hi < 0 |
| G1 | TTC, NC, DAC sub-scores of the pick, VDAC-Phi - P6, in points | each >= -0.5 |
| G2 | regret on E2, VDAC-Phi - P6 | CI-hi < +0.5 |
| G3 | overspeed share on E2, VDAC-Phi - P6 | CI-hi < +0.10 |
| G4 | slow-pick share on E2 CRUISE, VDAC-Phi - P6 | CI-hi < +0.10 |

Reported, not deciding: VDAC-Phi - VDAC-V (soft vs hard); VDAC-PhiL - P6; O-DAC - P6 and each arm's capture ratio; the NC-zero pick share;
EP x100 of the pick (P6 holds an EP surplus of 1.195 over the human, F1, and a DAC penalty may spend some of it); tokens where the check
changes the pick, split by P6's pick DAC = 0 / 1 (recoveries vs false moves).

The estimator answers *another draw of logs*. Training variance of the HEAD is read by HA-s1 (P1r); P6's own training variance is not
resampled (P6 is a fixed scorer); inference is deterministic (ASSERTED by a second pass, bit-identical). Stated with every result.

## 3. Outcomes of S2 (committed; label order VD1, VD2, VD3, VD4, VD5)

- **VD1 ADOPT** (P1, P1r, M, G1-G4 hold): VDAC-Phi enters P6's selection (deploying is the PI's call). Next: goal status vs 94.55, then the
  agent check (NC 1.12 % + TTC 1.527 points) under its own card, read on top of VDAC-Phi.
- **VD2 SINGLE HEAD SEED** (P1, M, G1-G4 hold; P1r fails): *separated, single head seed, not yet a lever claim*. Next: a third head seed
  (~3.2 L40-h, PI check) read by these bars unchanged.
- **VD3 MECHANISM, TOO SMALL** (M holds, P1 fails, no guard fails): the check removes DAC-zero picks but not enough PDMS. Next, ranked by
  S0's capture ratios: VDAC-PhiT if admissible, else the larger of VDAC-PhiL's reported gain (its own card) and the agent check.
- **VD4 HARM** (VDAC-Phi - P6 CI-hi < 0, or a G1 sub-score below -0.5): false penalties dominate. Next: if S0's true-field Phi rule was
  positive, the raster is the limit: HC, then HB; otherwise the DAC rule family is closed and the agent check carries the line.
- **VD5 NO MECHANISM** (M fails, P1 fails, no harm): the raster does not reach P6's failing picks. Next: HB if S1-Q's head AUC sits more
  than 0.05 below the true-field AUC on the same proposals, else VDAC-PhiT or the agent check.
- **Unnamed** if none matches; reported as such.

⛔ Every outcome is reported as written. A FAILED arm leaves its next arm above, and no bar moves after a number is seen.

## 4. Gates and controls of the S2 reader (it fails closed)

- **G0-PICK:** P6's banked logits reproduce its banked pick on 12,146 / 12,146 tokens.
- **G0-ID:** the head's navtest forward pass produces scorer logits bit-identical to `proposals_p6final.npz` `logits` on 12,146 / 12,146
  (the head reads the same `visual_ctx` the scorer read; S1c's 1,058 / 1,058 pattern).
- **C-ZERO:** q = 1 everywhere reproduces P6's pick exactly; Delta = 0.0.
- **C-LIVE (reachability):** a planted field that puts one proposal's footprint 5 m off-road on a fixed set of 100 tokens must move each
  of those picks off that proposal, and must leave the other 12,046 picks unchanged. A multiplier that never changes a pick is a switch
  that is off.
- **C-FLIP:** VDAC-Phi with the y-flipped field must read below VDAC-Phi.
- **C-SHUF:** VDAC-SHUF must not pass P1 (if it does, the read is INVALID: the gain is not from map information).
- **C-ODAC:** O-DAC's recovered DAC-zero picks >= every arm's.
- Validity identities: the reader's sigma_b equals S1's banked value to 1e-9; the head's navtest IoU computed by the reader equals the
  harness value (`aux_s1_readout.py`) to 1e-6.

## 5. Four metric families (per arm vs P6, per family, never pooled)

| family | reported |
|---|---|
| **LONGITUDINAL** | EP and TTC sub-scores of the pick; overspeed and slow-pick shares (E2); speed / along-track errors of the pick (the method of `p6_full_families.json`); headway / time-gap where a lead agent exists, with n |
| **LATERAL** | DAC and DDC sub-scores of the pick; DAC-zero pick share (M); heading, curvature, yaw-rate and cross-track errors of the pick; d_k distribution of the pick by band (0-2 s operative, 2-4 s tactical) |
| **TACTICAL** | selection quality: random / P6 / best-of-64 / arm; command adherence (the method of `command_adherence_p6final.json`) |
| **STRATEGIC** | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25): not a pass, not omitted, not extended to the other three |
| perception (T0-like) | head IoU, near-boundary MAE, per city and band on navtest, from the same forward pass; never deciding |

A family that cannot be computed is reported with its reason and n.

## 6. K ledger

One key per arm read on E3: `VDAC-<arm>:<sha256 of the per-token selection scores [12146, 64] float32>:E3` for VDAC-Phi (s0),
VDAC-Phi (s1), VDAC-PhiL, VDAC-V, VDAC-SHUF and O-DAC (**K + 6**; + 1 for VDAC-PhiT if admitted; + 3 if the S0-NEAR oracle rows are
read). S0 and S1 are navtrain reads and are not keyed. E2 guard reads are not keyed (as `PREREG_S1ZSM_P6.md`). Best-of-K statements use
Bonferroni intervals; P1, P1r and M are VDAC-Phi's own tests. The ledger is written by the reader at the real read only.

## 7. Launch-gate checks a training needs (HA-s0, HA-s1, HA-SHUF; no launch without a PASS token)

Profile **`refe_vdac_p6`**, a NEW dict built from `refe_aux_s1c` (shared constants untouched), bound to the commit, the argv sha256 and the
data-manifest sha256s (label store `4a17db67...`, H48 list, P6 checkpoint `968a469d...`).

| check | entry | deliberate regression that must turn it RED |
|---|---|---|
| G-HYG | config dataclasses and the argparse namespace refuse undeclared `aux_drv*` attributes | set `aux_drv_feeed`; remove the guard |
| G-DVB `--ckpt` | the built model's checkpoint sha256 = `968a469d...`; built `nav_mode` = command (a built fact, WIRING_NOTE section 4) | point the argv at u400 |
| G-DVB `--aux-drv` | `aux_drv` declared = head built (`n_params` 2,440,450) | build without the head |
| G-DVB `--w-aux-drv` | named in the argv; equals the weight multiplied into the built loss | drop it from the argv (default 0.0) |
| G-DVB `--aux-drv-lr` | named; equals the head optimiser's peak lr | omit it (refused by `aux_s1_refusals`) |
| G-DVB `--aux-drv-tap` | `visual`; the head's K/V `data_ptr` equals `visual_ctx`'s | tap `scene` (refused on this host) |
| G-DVB `--aux-drv-feed` | `off`; scorer input width unchanged and no autograd path head -> score (all `None`) | wire the feed with the flag off |
| G-DVB `--aux-drv-labels` | store sha256 and frame count (103,139) equal the manifest's | point at the navtest store |
| G-DVB `--aux-drv-exclude-logs` (NEW) | the H48 list's sha256; built fact: labelled frames from H48 that reached the loss = **0**; frames excluded > 0 | remove the filter |
| G-DVB `--aux-drv-shuffle-seed` (HA-SHUF only) | the key map is a derangement: 0 fixed points, identical label multiset | a map with one fixed point |
| G-DVB disjointness | autograd of the head output w.r.t. the ego and command tensors is `None` / exactly 0.0 | add `ego_tok` to the head's context |
| G-LIVE | 2 real-data updates: `aux_bce`, `aux_huber` finite; head parameters non-zero gradient; trunk exactly 0.0; initial `aux_bce` ~ ln 2 +/- 0.15 (S1c read 0.644); the scorer arms bit-identical to a head-less run (feed off) | detach the head's context; `w_aux_drv` = 0 must give exactly 0.0 on the head |
| G-LIVE alignment | on >= 64 real frames, P6's rank-0 targets lie inside the label at the banked rate | flip the label's y axis; shift the lookup by one frame |
| G-CLOCK | label looked up by the same `(log_name, token)` as the images | shift the lookup by one sample |
| G-EVAL | the navtest forward tool builds the identical head (strict load of `aux_drv.pt`) and reproduces P6's banked logits (G0-ID) | load a different seed's head |
| G-CKPT | head parameters and Adam state round-trip; 2 resumed updates bit-identical | drop the head's optimiser state |
| G-SUITE | clean-tree tests, 0 regressions; new tests for `--aux-drv-exclude-logs`, the profile, and the VDAC reader's controls; all ~69 launch_gate-reading tests | - |
| post-training | every head parameter has Adam state (`probe_zerograd3.py` pattern: registered = with-state) | - |

## 8. Registration

On freeze: `GOALS_AND_CLAIMS.md` row `H-VDAC-P6-1` is updated to this card's sha256, with the corrected design input (F0 C1/C2: the
0.18-0.27 IoU are untrained floors on u400; trained frozen-trunk heads read 0.72-0.82; the October veto read is the measured prior).
