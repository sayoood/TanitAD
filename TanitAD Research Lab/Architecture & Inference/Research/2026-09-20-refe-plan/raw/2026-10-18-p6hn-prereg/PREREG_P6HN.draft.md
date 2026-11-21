# P6-HN: expert-proximate hard negatives in P6's scorer loss, as one lever on P6 (pre-registration DRAFT)

> ⛔ **DRAFT. NOT REGISTERED. NOT IN FORCE.** Written 2026-10-18 by the pre-registration design agent, on tip `3a28bc19`.
> No P6-HN number exists yet: no bank, no yield pilot, no arm, no P6 checkpoint. The hash in `PREREG_P6HN.draft.md.sha256`
> records this text only. Registering it is the Master Mind's job: hash, land, and then amend only by a dated amendment written before the read.
>
> **It comes into force only when all five of these hold:**
> 1. P6 has a VALID read under `eval/PREREG_P6.md` (any outcome except INVALID). See §11 for the case where P6 is REFUTED.
> 2. ZT4 has been read on P6's own fan (`D-REFE-ZEROTAIL-3`), so §7's M2 has its P6 baseline.
> 3. The PI has said go on the compute (§12). The PI questions are in `NOTES.md`.
> 4. Gate Y (§4) has passed.
> 5. The launch gate has issued a PASS token for the profile `refe_p6hn` (§3.3).
>
> Evidence classes: MEASURED (ours, with an artifact path) · PUBLISHED (library key) · INHERITED (another doc, not re-verified here) ·
> ESTIMATED · HYPOTHESIS. Every path is relative to `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/` (PK) unless it is a full repo path.

## 0. Question, and the one lever

**Question.** Does adding expert-proximate, safety-critical hard negatives to P6's scorer loss raise full-navtest PDMS over P6, without paying for it in NC or TTC?

**The one lever.** One trainer flag, `--hn-bank <path>` (§3). Everything else is P6's registered argv: same data manifest, schedule, optimiser,
seed 0, labeller, nav mode, yaw form and selection rule.

**Why it is a lever on P6 and not already inside P6.**
- P6 already trains its scorer on online PDM labels of each sample's own 64 proposals, at lag 0 (`eval/PREREG_P6.md` "Scorer";
  `SPEC_P6_PAPER_FAITHFUL.md` §3.1-3.2; INHERITED).
- So the hard negatives that a fan happens to contain are already in P6's loss. The census measured them on u400's fan:
  33.7 % [23.3, 46.7] of held-out sets at 0.5 m ADE (MEASURED, `raw/2026-10-18-hn-census/RESULT.md` T1).
- A term mined from P6's own fan would only re-weight data P6 already sees.
- **What the lever adds is new geometry:** lateral near-twins of the expert that the fan does not produce. On u400's fan the census
  found the following sets with nothing proximate (MEASURED, census T0/T1/T11):
  - 66.3 % at 0.5 m ADE;
  - 46.5 % at 0.3 m cross-track, POST-HOC;
  - 35.5 % with no hard negative anywhere.
- In the sets that already have one, the lever adds sharper near-twins.

**Published form** (PUBLISHED; both PDFs re-read against the banked copies this session, sha256 matches `library.json`):
- **`2608.19085` DA-WAM, §3.4 and Tab. 3.**
  - Method: for each scene, candidate trajectories are retrieved from an offline trajectory bank. The rule is dual: geometric distance to the
    expert below ε_geo, AND safety degradation above ε_safety (their Eq. 11).
  - The negatives are appended to the candidate set. They receive factor and score targets (and ranking targets), and they are excluded
    from expert matching.
  - Tab. 3, matched pair, action-conditioned future, without vs with hard negatives:

    | | PDMS | NC | DAC | EP | TTC |
    |---|---|---|---|---|---|
    | without | 93.46 | 98.88 | 98.58 | 90.47 | 96.33 |
    | with | 93.68 | 99.11 | 98.88 | 89.97 | 96.81 |

    Net: **PDMS +0.22, EP −0.50**, single run, no interval.
- **`2605.19771` BeyondDrive, Tab. 4.** A generator plus a repulsive loss on regression and diffusion planners (not a scorer):
  - LTF\*: PDMS 88.7 → 89.7, DAC 97.0 → 97.9;
  - DiffusionDrive: PDMS 88.1 → 89.2, DAC 96.2 → 97.4;
  - single runs.

**The mechanism it targets** (`D-REFE-ZEROTAIL-3`, INHERITED from the register):
- 60.1 % [51.4, 68.0] of u400's DAC-only selection zeros are ON_PATH near-twin grazes. The median lateral distance to the best non-zero
  proposal is 0.29 m.
- 74 % of them leave the road by less than 0.5 m (median 0.23 m).
- On these grazes the DAC head is near-blind:
  - pairwise DAC accuracy on near-twin pairs is **0.392 [0.268, 0.571]** on u400's DAC-only selection-zero tokens;
  - against **0.963** on a random stratum;
  - P2 reads 0.443 [0.331, 0.587].
  - Source: MEASURED, `raw/2026-10-17-zero-tail-mechanism/ZT4_BASELINE.json`. That instrument is POST-HOC within ZT3.

## 1. The proximity axis: fixed here, with the census peek disclosed

**Disclosure.** The step-0 census (`PREREG_HN_CENSUS.md`, sha256 `5943731c…`) read supply on two axes, and both have been seen:

| axis | status in the census | sets with ≥ 1 proximate hard negative | gross reach (ESTIMATED) |
|---|---|---|---|
| ADE to the teacher plan | pre-registered | **33.7 % [23.3, 46.7]** at 0.5 m | 0.29-0.32 PDMS |
| mean absolute cross-track to the teacher path | **POST-HOC** (T11) | **53.5 % [44.6, 61.6]** at 0.3 m; 60.2 % [50.5, 69.2] at 0.5 m | 0.46-0.57 PDMS |

Source: `raw/2026-10-18-hn-census/RESULT.md` T1, T11, T8; `raw/posthoc2.json`. The census also saw where the sibling scorers' zero picks
sit: about 0.4 m cross-track and about 1.5 m behind the teacher (T12, POST-HOC).

**Decision (fixed now).** Proximity = **mean absolute cross-track distance to the anchor path over the 20 poses. Speed and along-track
are excluded.** Threshold **≤ 0.60 m**, met by construction (§2).

**The grounds.** None of the three depends on a census number:
1. **The mechanism is lateral.**
   - DAC is a footprint-versus-boundary event.
   - The mechanism this lever targets was measured on the lateral axis: ZT3's grazes, median 0.29 m lateral, length ratio 1.01, same manoeuvre.
   - ZT3 was hashed and landed on 2026-10-17, before the census existed.
2. **The four-family rule keeps LATERAL and LONGITUDINAL apart.**
   - ADE adds along-track lag (a speed quantity, which belongs to the EP and TTC family) to a lateral question.
   - A proximity built on ADE would label a slow but laterally correct plan as "far", and a fast plan that grazes the kerb as "near".
3. **The threshold and grid predate the census.**
   - The offset grid ±{0.15, 0.30, 0.45, 0.60} m is LAB-RUN-019 §4.4's step 1, written before the census was hashed (the census prereg cites
     it: "the Lab's step-1 offsets go up to 0.6 m").
   - It brackets ZT3's measured graze depth: median 0.23 m, and 74 % below 0.5 m.

**Why the peek cannot decide this arm's outcome.**
- (a) In this arm the proximity axis is a **training-construction parameter, not a measurement**. The decisive reads in §5-§6 (full-navtest
  PDMS, NC and TTC, the four families) contain no proximity quantity.
- (b) The census numbers describe **u400's fan on the navtrain held-out dev table** (24 logs). The bank in §2 does not draw from any fan,
  and the census never opened navtest.
- (c) No P6-HN number exists anywhere. The decisive read is a newly trained arm, evaluated on navtest.
- (d) Census numbers enter this document only as ESTIMATED reach (§10). They are reported and never gated.

The honest residue: the chosen axis is also the one that looked more favourable post hoc. That is stated here and repeated in the result.

## 2. The hard negatives: construction, definition and admissibility

**Anchor.**
- Each training tuple's own target trajectory `traj`: 20 poses, 4 s, the expert P6 already imitates.
- Rank-0 rows: the teacher plan.
- 3-C lane-rank twins: the re-rolled teacher's first 4 s (`raw/2026-10-17-twin-bank-3c/PREREG_3C.md`).
- Tuples whose anchor path length is < 2.0 m get no offsets: their tangent is undefined. This uses the census's own STATIONARY rule
  (`PREREG_HN_CENSUS.md` §3.6), and these tuples are counted.

**Offsets (trajectory-derived only).**
- 8 candidates per tuple: pose k is shifted by δ along the anchor's unit left-normal at pose k.
  - δ ∈ {−0.60, −0.45, −0.30, −0.15, +0.15, +0.30, +0.45, +0.60} m.
  - The normal comes from the stored yaw if present, else from a central-difference tangent.
  - Heading is kept; the pose index (time) is kept.
- The anchor itself (δ = 0) is labelled too, so there are **9 labels per tuple**.
- **No map geometry is read to build or to choose a negative.**

**Labels.**
- P6's own labeller code path on P6's own D9 parts: `executed_plans8(props, repair=True)` → `score_plans(parts, plans8)` (SPEC §3.1).
- That gives six sub-scores (NC, DAC, EP, TTC, C, DDC) plus PDMS, with A7 repair ON.
- The parts carry the drivable-area map as a **training-time label**. This is exactly how every P6 proposal is already labelled
  (admissible, PI 2026-10-11, SPEC §3.1), so it is no new map use.

**Hard negative (the dual constraint of DA-WAM Eq. 11, in our terms).** An offset candidate is a hard negative iff:
- PDMS(candidate) = 0 (NC = 0 or DAC = 0), AND
- PDMS(anchor) > 0.

NC = 0.5 (static object) is not a hard negative. Sub-types (DAC-only, NC-only, both) are recorded. A tuple whose anchor scores PDMS = 0
contributes none, and it is counted. This also closes the census's open gap: the teacher's own NAVSIM DAC was UNVERIFIED.

**How they enter (score loss only).**
- Each hard negative is appended to that sample's scorer candidates, with its six sub-score targets.
- It enters **no WTA assignment** and **no generator target**, and it is **never** a candidate at selection or inference.
- Weight: `score_hn = score_w × Σ_hn BCE_6(hn) / 64`. Each hard negative weighs exactly what one of the 64 fan proposals weighs.
- The fan term is unchanged bit-for-bit.
- No new free weight is introduced. DA-WAM's extra up-weighting lives in its ranking loss; REFe has no ranking loss, so that part has no
  counterpart here (a declared departure).
- Padding to 8 per sample uses a mask, and masked rows contribute exactly 0.

**Excluded, named.**
1. **Map-derived negatives** are excluded pending the open PI ruling **C9** (BACKLOG PAPER-720; LAB-RUN-019 §3, flags 1-2). These are
   negatives whose geometry comes from lane boundaries, drivable polygons or lane-graph alternatives. If the PI admits them as labels, they
   are a separate arm, never folded into this one.
2. **Negatives retrieved from the archived u400 fan** (190,696 labelled training sets, INHERITED count, census §7). Three reasons:
   their label_version is UNVERIFIED; their collision calculator disagrees with PDM NC on 5,046 of 200,768 held-out proposals (MEASURED,
   census §7); and they come from another checkpoint. This is a candidate next lever (§11, row 6).
3. **Parity.** P6's tuples are not re-selected. The bank only adds scorer candidates to P6's exact data manifest.

## 3. The arm, its control, and its launch-gate entry

### 3.1 Arms

| arm | definition | status |
|---|---|---|
| **C0 = P6** | P6 as registered (`eval/PREREG_P6.md`), seed 0, its `model_final.pt` | reused, not retrained |
| **HN = P6-HN** | P6's minted launch argv, with exactly two changes: `--out <p6hn_s0 dir>` and `--hn-bank <bank path>` | the arm |
| **R = P6-HN-s1** | P6-HN's argv with exactly two changes: `--out <p6hn_s1 dir>` and `--seed 1` | conditional replicate (§9) |

**Base argv.** HN's base is the exact argv that P6's PASS token is minted over. P6's token does not exist yet (P6 is waiting on the 3-C bank
and on the migrated pod, BACKLOG P6-3C). So the argv is **copied into this file by amendment at registration**, before any P6-HN step.

### 3.2 Reachability, proved before the gate (CLAUDE.md "built, tested, and unreachable")

**The hop trace.** The implementer names file:line at each hop, and the reviewer checks them. None exists yet; it is a work item.
- argparse `--hn-bank` (`refe/train.py`)
- → `a.hn_bank`
- → the bank loader, joined on the tuple key, which refuses a bank whose P6 data-manifest sha differs from the token's
- → the batch fields `hn_traj`, `hn_labels`, `hn_mask`
- → the scorer forward over `[fan ; hn]` (`refe/model.py`)
- → the `score_hn` term in `loss`
- → `.backward()`

**Gradient checks on a real-data smoke** (non-zero where it must flow, exactly zero where it must not):

| parameter group | gradient from `score_hn` alone |
|---|---|
| scorer heads | **non-zero** |
| scorer candidate encoder | **non-zero** |
| trunk LoRA (P6's context is un-detached, D3) | **non-zero** |
| trajectory-decoder / generator group | **exactly 0.0**. This is the control that proves the probe works. |
| any group, with `--hn-bank` absent | **exactly 0.0**, and the total loss is bit-identical to P6's smoke at the same commit |

**Mutations, each of which must turn a named test RED:**
1. delete the line that appends the hard negatives;
2. route the hard negatives into WTA;
3. unmask one padded row;
4. a bank built from a different data manifest.

### 3.3 G-DVB and the launch gate (`Project Steering/SPEC_REFCV7.md` §2; `eval/REFE_LAUNCH_GATE.md`; CLAUDE.md "NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS")

**New `gate_checks.DVB_KINDS` row: `--hn-bank`, kind `built` + `data`.**
- **Observed side.** At load, the trainer's `data` event emits:
  - `hn_bank_sha256`;
  - `hn_rows`;
  - `hn_tuples_with_hn`;
  - `hn_max_per_sample = 8`.
- The built loss's term list contains `score_hn`.
- The bank path is fingerprinted into the token like every input path (an unfingerprinted input path FAILS, per REFE_LAUNCH_GATE).
- **Declared side.** The argv value.
- **They must be equal.**

**New profile `refe_p6hn`.**
- The argv must equal P6's minted argv plus exactly the two §3.1 changes, each once, in any order.
- A moved, added, dropped or doubled lever FAILS. The P2-s1 profile has the same form.
- `refe_p6` is amended to **refuse `--hn-bank`**, so P6 cannot carry the lever by accident.

**The other gate checks:**

| check | what must hold |
|---|---|
| G-LIVE | `score_hn` is finite and > 0 on at least one window with `n_hn > 0`; the §3.2 gradient checks hold |
| G-EVAL | the eval loader builds a model whose state-dict keys and shapes equal P6's (the lever adds no parameter), and the eval reads no bank |
| G-CKPT | unchanged |
| G-CLOCK | unchanged (10,075 steps) |
| G-SUITE | 0 regressions |
| G-LIVE-LAB, G-LIVE-MEM, G-EGR | re-measured at launch. G-LIVE-MEM matters because the scorer sees up to 72 candidates instead of 64 |

**Deliberate-regression arms (each must make the gate FAIL):**
- `lever_ignored`: parsed, never appended;
- `hn_to_generator`;
- `hn_in_selection`;
- `moved_lever`;
- `stale_bank`: the bank's bytes changed after minting;
- `empty_bank_in_launch`: a bank with 0 hard-negative rows. An arm that names the lever must exercise it.

## 4. Gates before the arm (CPU; any failure stops it; navtest untouched)

**Gate Y: bank yield, on the navtrain held-out dev table** (3,137 sets / 24 logs, the census table).
- Build §2's 9 labels per key.
- Report, all with the census estimator (log-cluster bootstrap, 24 logs, 10,000 resamples, seed 20260927):
  - the share of sets with ≥ 1 hard negative;
  - the hard-negative rate per |δ| and per side;
  - the sub-type split;
  - the share of anchors that score PDMS = 0.
- **Committed outcomes:**
  - **share ≥ 0.10 ⇒ proceed.**
  - **< 0.10 ⇒ SPARSE.** The arm is not launched. The next lever is §11 row 6's variant under a new prereg: pair-contrast on the fan's
    own twins, which exist in 34.7 % [21.6, 50.6] of held-out sets (MEASURED, census T6).
- This gate is a supply check. It reads no effect.

**Identity controls.** Any failure ⇒ **HN-INVALID**: no number is read.
- **I1.** On ≥ 64 production tuples, the bank's labels equal the online labeller's labels for the same 9 trajectories, bit-for-bit.
- **I2.** Unit tests on a straight and a curved toy path: every offset's mean absolute cross-track to the anchor is |δ| (median error
  ≤ 0.01 m), and the pose times are unchanged. The builder's call graph reaches no map API; this is paired with a control that must find one
  in the labeller.
- **I3.** With the flag off, and again with an empty bank, the G-LIVE smoke's loss is bit-identical to P6's.
- **I4.** P6-HN's navtest picks are bit-identical whether or not the bank file is on disk.
- **I5.** A second full eval of P6-HN's checkpoint is bit-identical in per-token PDMS (as in PREREG_P6).

## 5. Primary endpoint, comparator, estimator, tier

**Primary endpoint.** Δ = PDMS(P6-HN) − PDMS(P6) on the **full navtest: 12,146 tokens / 136 logs, NAVSIM v1.1**, ×100.

**Harness.** The same as for P6's read, unchanged:
- dev box, never the pod;
- `model_final.pt`;
- A7 repair ON;
- command mode;
- `navsim_v1` selection over the 64 generated proposals. Hard negatives are never candidates.

**Estimator.** Paired log-cluster bootstrap over the 136 logs, **10,000 resamples, seed 20260927**, percentile 2.5/97.5. It is reported for
Δ and for P6-HN's level. Every family component and every mechanism read uses the same draws.

**Tier: T1 NAVSIM v1.1.** This is the register's stamp for navtest PDMS. Per PREREG_P6 it is a pseudo-simulation of an open-loop plan
against logged agents, **not closed loop**.

**Reported, not deciding.** P6-HN against the goal of 94.55 (PUBLISHED, DriveZero Tab. A13), using PREREG_P6's three readings verbatim
(GOAL MET / GOAL NOT EXCLUDED / BELOW GOAL).

## 6. Rule (committed now)

**Safety co-gate** (`D-REFE-FREEPOINT-1`).
- Why it exists: R0B gained PDMS +0.55 [+0.24, +0.84] but paid NC −0.25 [−0.45, −0.06] and TTC −0.40 [−0.71, −0.09] (INHERITED, register).
- **The co-gate FAILS iff the upper bound of ΔNC < 0 OR the upper bound of ΔTTC < 0** (paired sub-scores, §5 estimator).
- DDC, and the count of NC-zero picks, are reported beside it.

| outcome | rule |
|---|---|
| **HN-INVALID** | a §4 control or the launch gate failed, or P6's own read is INVALID ⇒ nothing is read |
| **HN-UNSAFE** | the co-gate fails ⇒ **not adoptable, whatever the PDMS** |
| **HN-ADOPT** | LB(Δ) > 0 AND the co-gate passes AND no LONGITUDINAL or LATERAL family component separates adversely (PREREG_P6's own rule) |
| **HN-LEVER** | HN-ADOPT AND LB(Δ) > F (F per §9). Only this outcome is a claim that the lever moved PDMS |
| **HN-REFUTED** | UB(Δ) < 0 |
| **HN-NOT-PROVEN** | everything else |

An HN-ADOPT that is not HN-LEVER is reported as **"separated, single seed, not yet a lever claim"**.

## 7. Mechanism reads (committed; they choose the next lever, never adoption)

**M1, zeros.**
- Δ zero-pick rate (picks with PDMS = 0), in percentage points, paired. Split by DAC-only / NC-only / both.
- **ZEROS-FALL iff the upper bound of Δ(rate) < 0.**

**M2, the head.**
- The ZT3 package (`zt3_analyze.py` + `zt4_baseline.py`), unchanged, run on P6-HN's own 64-proposal fan exactly as ZT4 runs it on P6's.
- Read: the pairwise DAC accuracy on near-twin pairs over each arm's own DAC-only selection-zero tokens.
- **HEAD-MOVES iff the lower bound of (P6-HN − P6) > 0**, using joint log resampling (the same 10,000 draws for both arms).
- Otherwise **HEAD-BLIND**. ZT4's 0.70 bar and u400's 0.392 are reported beside it.
- Instrument work item: the two-arm joint-draw difference is new. Its identity control is an arm minus itself, which must read exactly 0
  with CI [0, 0].

## 8. Four families (binding; per family, never pooled; the instrument that produced `sft4_B_u400_families.json`; same tokens; paired CI on each component)

| family | components |
|---|---|
| **LONGITUDINAL** | speed MAE and bias; along-track MAE and bias; **EP** and **TTC** sub-scores; headway or time-gap to the lead agent where a lead exists. Each carries its n; where there is no lead, the reason is given. EP is watched specifically: DA-WAM paid EP −0.50. |
| **LATERAL** | heading MAE; cross-track MAE; curvature MAE; yaw-rate MAE; **DAC** sub-score; last-pose heading error (D5) |
| **TACTICAL** | lateral, longitudinal and 5-way decision accuracy and κ; command adherence of the selected proposal; selection quality as random / actual / best-of-64 on the same tokens (the random control is mandatory; the 64 are the generated fan only) |
| **STRATEGIC** | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25). This is not a pass, and it is not omitted. |

## 9. What the interval answers, and the replicate

**Logs.** The bootstrap answers one question only: *would another draw of LOGS say this?*

**Inference.** REFe inference is deterministic, asserted by I5. Inference variance is therefore zero by test, not by assumption.

**Training (`H-ESTIM-SEED-1`): a separated CI from one training seed is necessary, not sufficient.**
- **The floor is F = 0.33 PDMS** (MEASURED). P2-s1 minus P2-s0, two seeds that differ in nothing else, read Δ_seed = +0.33 [+0.04, +0.64]
  (`eval/RESULT_P2REP.md`, raw `raw/p2_readout/p2s1_full/r2_result.json`).
  - That pair itself read "separated", which is the rule's own warning.
- If a P6 seed-1 replicate exists by the read, then **F := max(0.33, |Δ_seed(P6)|)**.
- ⚠️ **Scope.** F comes from a warm-start joint fine-tune (P2, about 21 A40-h). The seed variance of a from-scratch, 25-epoch run is
  UNMEASURED and may be larger. F is a one-draw estimate. All of this is inherited from P6, which uses the same F.
- **The replicate arm R (P6-HN-s1) is requested iff the outcome is HN-ADOPT and (LB(Δ) ≤ F, or the point Δ < 2F).**
  - It is then read under an amendment written before its read: the lever claim needs Δ_s1's LB > 0 as well, and a mean Δ > F.
  - Below that line the replicate's absence is justified by the measured F; above it, the claim is reported as single-seed with F named.

## 10. Power: disclosed before the arm (ESTIMATED)

**What the CI can resolve.** Paired-CI half-widths measured on this harness (MEASURED, `eval/RESULT_P2REP.md` and `eval/RESULT_P2.md`):

| comparison | Δ [95 % CI] | half-width |
|---|---|---|
| a seed pair | +0.33 [+0.04, +0.64] | 0.30 |
| P2 vs u400 | +0.57 [+0.02, +1.18] | 0.58 |
| P2-s1 vs u400 | +0.90 [+0.36, +1.59] | 0.62 |

So the half-width h is between 0.30 and 0.62.

| outcome | point Δ needed |
|---|---|
| HN-ADOPT | > h, i.e. > ~0.3 to 0.6 |
| HN-LEVER | > F + h, i.e. > ~0.63 to 0.95 |

**What the lever is expected to deliver** (all ESTIMATED, gross, not net; `D-REFE-ZEROTAIL-4` warns that a gross reach can net about 0):

| source | expected gross Δ |
|---|---|
| census-constrained | 0.29-0.57 |
| LAB-RUN-019's increment over P6 | 0.55-0.94 |
| DA-WAM's measured PDMS effect of this form (PUBLISHED, single run) | **+0.22** |

**Reading.** The likeliest a priori outcome is **HN-NOT-PROVEN**, and HN-LEVER needs the top of the Lab's range.
- M1 is the more sensitive read: zeros are the lever's direct target.
- This is a disclosure, not a reason to move any bar.
- `NOTES.md` Q1 asks whether to screen cheaply first.

## 11. Next lever, by outcome (Rule Zero; committed now)

| # | outcome | next lever, executed in the same run as the verdict |
|---|---|---|
| 0 | P6 itself REFUTED (precondition 1) | P6-HN is not launched on a refuted base. The lever waits for P6's bisection and is re-based onto the deployed system by amendment. |
| 1 | HN-INVALID | fix the failed control and re-run it; nothing is read |
| 2 | HN-UNSAFE | not adopted. First, a 0-GPU attribution of the changed picks: the tokens whose pick differs, split by which sub-score moved. Then, under a new prereg screened on the dev table before any navtest: DAC-only negatives (NC-0 offsets excluded) if the loss traces to NC negatives; otherwise the selection-side safety constraint line (SO-PICKCON) on P6-HN's scorer. |
| 3 | HN-LEVER | deploy P6-HN. The next arm is per-timestep proposal on-road targets (iPad form, LAB-RUN-019 §4.3 row B), as a single-lever arm on P6-HN (needs the PI's GPU go), plus the 0-GPU selection-rule read on P6-HN. Never fold the two into one arm. |
| 4 | HN-ADOPT, not HN-LEVER | the replicate R (§9). The lever claim is decided on it. |
| 5 | NOT-PROVEN or REFUTED, **ZEROS-FALL** | the term acts on its target but pays elsewhere (DA-WAM: EP −0.50). Next: 0-GPU EP-versus-safety calibration of the selection rule on P6-HN's scorer, on the dev table first. |
| 6 | NOT-PROVEN or REFUTED, no ZEROS-FALL, **HEAD-MOVES** | the head learned the offsets, but they do not match the fan's real failures. Next: N1 (the training-scale census, which verifies label_version), then the retrieval form (DA-WAM's own: negatives retrieved from fan banks) or pair-contrast on fan twins, under a new prereg. |
| 7 | NOT-PROVEN or REFUTED, no ZEROS-FALL, **HEAD-BLIND** | the blindness is representational (LAB-RUN-019 §4.4). Next: the trunk, i.e. P7.1, a training-only map-supervised drivable-area aux head on P6 (admissible label, PI 2026-10-11). |

## 12. Compute (ESTIMATED; arithmetic shown)

**Rate.** P6's measured labeller rate is **10.887 sets/s at 7 workers** (64 proposals per set).
- It was measured beside a training run, so it is a lower bound (MEASURED, `raw/2026-10-15-p6-prereg-final/pod_lab_early.txt`, the mirror
  of the pod's `lab_early.json`).
- The tuple count is P6's **189,888** (INHERITED from PREREG_P6; this is the pre-3-C count, an upper bound; the 3-C count is PENDING).

| item | arithmetic | result |
|---|---|---|
| bank, cost ∝ candidates | 189,888 × 9 / 64 = 26,703 set-eq ÷ 10.887 | **2,453 s ≈ 0.68 h** |
| bank, cost ∝ sets | 189,888 ÷ 10.887 | **17,442 s ≈ 4.84 h** |
| bank on the migrated container (4 workers, `p6_pod.sh:201` cores−4; linear) | × 7/4 | **1.19-8.48 h** |
| gate Y pilot (3,137 sets × 9) | 28,233 / 64 = 441 set-eq ÷ 10.887 to 3,137 ÷ 10.887 | **41-288 s** |
| GPU, P6-HN | P6's wall × (1 + scorer share × ≤ 8/64). P6 = **6.8 d** if the labeller keeps up (`eval/P6_ON_11VCPU.md` §6, INHERITED; P6's own measured wall supersedes it). The scorer's share of a step is UNMEASURED, so it is bounded by 1. | **6.8-7.65 A40-days** |
| eval | 2 × 5.3 h navtest reads (read + I5), INHERITED, plus the M2 package (≤ 1 dev-box GPU-h, INHERITED from ZT3) | **~11.6 dev-box h** |
| replicate R (conditional) | the same as P6-HN | **+6.8-7.65 A40-days** |

**Rules on where the bank build runs.**
- It runs in a non-training window, or under P6's overlap rule (nice 19, at or under the overlap cap).
- **Never beside a training run beyond that cap.**

**Labeller load.** The online labeller's load is unchanged, because the bank is offline. Needed: 4.49 sets/s. At 4 workers the linear
ESTIMATE is 10.887 × 4/7 = 6.22 sets/s. That keeps up with 4.49, but it is **below G-LIVE-LAB's 6.735 bar**. P6 meets the same problem
first (BACKLOG P6-3C, "MEASURE FIRST"), and P6-HN inherits P6's resolution.

## 13. Not changed, out of scope

**Not changed:** P6's data, tuples, schedule, optimiser, seed, labeller, nav mode, yaw form, selection rule, eval harness and comparators.

**Out of scope:**
- inference inputs (the lever is training-only);
- map-derived negatives (C9);
- per-timestep targets (a separate arm);
- any change to the census or its readings.

**Sources:**
- `eval/PREREG_P6.md`; `SPEC_P6_PAPER_FAITHFUL.md`; `eval/REFE_LAUNCH_GATE.md`; `eval/P6_ON_11VCPU.md`;
- `raw/2026-10-18-hn-census/` (RESULT.md, PREREG_HN_CENSUS.md);
- `raw/2026-10-17-zero-tail-mechanism/ZT4_BASELINE.json`;
- `eval/RESULT_P2REP.md`; `eval/RESULT_P2.md`;
- `TanitAD Research Lab/2026-10-18-LAB-RUN-019.md` (finding 2, §3, §4.3-4.4);
- `Project Steering/GOALS_AND_CLAIMS.md` (`D-REFE-HN-CENSUS-1`, `D-REFE-FREEPOINT-1`, `D-REFE-ZEROTAIL-3/-4`);
- `Project Steering/SPEC_REFCV7.md` §2;
- Library: `2608.19085` (sha256 `9c942d4a7c5a…`), `2605.19771` (sha256 `d6b1442bb667…`).
