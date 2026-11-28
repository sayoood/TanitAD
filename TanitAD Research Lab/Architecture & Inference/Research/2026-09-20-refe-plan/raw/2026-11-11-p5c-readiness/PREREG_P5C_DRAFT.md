# PREREG P5-C (DRAFT): the REFe student fine-tuned on tracker-aware Tc20s labels, against a matched raw-label fine-tune

**Status: DRAFT, 2026-11-12.** Written before any P5-C label table, training step or read exists. The Master Mind resolves §16, then freezes the card by sha256 (`PREREG_P5C.md` + `PREREG_P5C.sha256`) **before any P5-C training step and before any P5-C read**. After the freeze, nothing below changes without a dated amendment written before the read it affects.

**Evidence classes.** Every number carries one tag:
- **[M `path`]** MEASURED, ours, with its artifact;
- **[P]** PUBLISHED;
- **[I]** INHERITED, another agent or doc, not re-verified here;
- **[E]** ESTIMATED;
- **[H]** HYPOTHESIS;
- **[U]** UNVERIFIED.

**Path keys.**
- `PK` = `E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`, read at the mirror tip `fa67c63f` (`origin/agent/arch-inf-20260803`). E: lags it.
- `SG` = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107`.
- `LS` = `E:/Projects/TanitAD-artifacts/dagland/LOOP_STATE_P6_2026-11-06.md`.

## 0 · What is already known (inputs, not bars)

**The label, at teacher level (navtrain held-out, 1,018 rank-0 targets, 11 logs, NOT navtest)** [M `PK/raw/2026-11-11-teacher-fidelity/tf1/tf1_result_full.json`]:
- Tc20s tracked PDMS 94.36 [93.07, 95.79] against raw T tracked 86.73 [79.87, 92.83]: **+7.64 [+2.29, +13.60]**. DAC +7.07 [+1.83, +13.26].
- **The label is worse as a path, by construction:** Tc20s transparent − T transparent = **−15.87 [−26.53, −6.26]**, DAC −13.36. A pre-distorted command is not meant to be driven untracked.
- P5-S bar PASSED: sim8 −7.07 pp [−13.26, −1.83]; learnability 1.69× (acceleration) and 1.76× (heading rate) against a 2× bar [M `PK/raw/2026-10-11-p5s-smoothed/p5s_compare.json`, register `D-REFE-P5S-SMOOTH-1`].

**The tracker channel, at student level (prior evidence, mixed):**
- P4 F2: test-time compensation of an older student's own plans, full navtest: **+1.07 [+0.68, +1.45]** PDMS. DAC violations 643 → 521: 188 repaired, 66 broken, so **29 %** repaired. 7 family components were adverse because they were read on the pre-distorted reference [M, exploratory, `PK/eval/RESULT_P4_TEACHER_NAVTEST.md` F2].
- ZT5, on dev (navtrain held-out, 2,466 sets, 13 logs): the inadmissible pre-compensation ceiling F_a reads **−0.19 [−0.76, +0.63]**. The tracker "saves about as many picked plans as it breaks" [M, `D-REFE-ZEROTAIL-4`].
- `D-REFE-ZEROTAIL-4` closed the **inference-side** form only. P5-C is training-side and is not covered by that closure.

**Noise floors available today:**
- **s_p = 0.1791**, df 4, t(0.975, 4) = 2.776. This is the pooled seed SD of the 400-update **scorer-only** fine-tunes of P6; the trunk was frozen and the proposals were bit-identical to P6 [M `PK/raw/2026-11-09-scorer1p6-prereg/rep/read/real/rep_read.md`]. t·s_p = 0.4972; the PESS/DIV cards print it as 0.4971 [M `SG/p6base/PREREG_DOSE_P6.md`].
- **F = 0.33**: |P2-s1 − P2-s0| = +0.33 [+0.04, +0.64]. It comes from **P2, a warm-start joint fine-tune** (LoRA + heads, lr 5e-5, ≈1,209 steps), and it is ONE draw [M `PK/eval/RESULT_P2REP.md`; recipe `PK/eval/PREREG_P2.md` Design]. This is the closest recipe to P5-C whose seed noise has ever been measured.
- **F3 = 0.3796**: the largest pairwise difference among C0, C0r and C0r2. It is scorer-only, from the same 7 runs as s_p [M `rep_read.md`].

**Bank and base:**
- The P7 bank `train_grow_cmd_3c_p7` holds **150,603 tuples**: 92,896 rank-0 and 57,707 rank-1. `targets_sha256` is `92af8d66…` [M `PK/SPEC_P7.md`, P7-PINS note].
- All rows carry `sim_hz 10` and step 0. The bank spans 1,075 logs [I, `SG/p5c/labels/probe_rows.json`, the labels agent].
- P7 is training on the pod's L40: step 3,624 at 22:16Z on 2026-11-11, **51.2 s/step**, ETA about 15 Nov 03:45Z without pauses [I, `LS` l.1048, 1073].
- P6: navtest 91.196 [M `SG/p6base/PREREG_REP_P6.md` §0]; `model_final.pt` sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1` [M, same].
- K ledger: **K = 40**, last entry `PESS-10`, 2026-11-11T09:05:50Z [M `SG/p6base/selectors/ledger_K.json`].

## 1 · The question, and the PI ruling that binds this card

**PI ruling, 2026-11-11** (recorded `LS` ~23:15Z, chat). It settles P5 card §8 decision (a).
- Training the REFe student on the P5-S tracker-aware Tc20s labels is **ADMISSIBLE**, on two conditions:
  1. **every P5-C number is reported both LQR-tracked AND tracker-transparent**, so that a gain that only exists through NAVSIM's tracker is visible;
  2. **the claim is worded "the student's plans survive the benchmark's tracker", never "the student drives better".**
- Arm form approved: **C1** = a 2,000-step fine-tune of P7's final checkpoint on Tc20s labels; **C0** = a matched 2,000-step fine-tune on the raw `traj` labels. ESTIMATED ~27 L40-h each.

**Question.** Starting from the same checkpoint, does a 2,000-step fine-tune on Tc20s labels make the student's picked plans survive NAVSIM v1.1's LQR tracker better than the same fine-tune on raw labels does?

**Tier.** T1-class: NAVSIM v1.1 non-reactive pseudo-simulation of the arm's own open-loop plan against logged agents.
- It is **not** the EVAL_DOCTRINE action-closed-loop tier, and T2 is not provisioned.
- No number in this card may be quoted as driving quality.

## 2 · Arms, base, and order

| arm | what | seed | status |
|---|---|---|---|
| **C1** | base + 2,000 steps, WTA target = the Tc20s table (§4) | 0 | approved |
| **C0** | base + 2,000 steps, WTA target = raw `traj` | 0 | approved |
| C0r | C0 with `--seed 1` | 1 | **NOT approved.** Conditional extra arm (§6.3); PI decision |

- C1 and C0 differ in **exactly one argv token**: `--traj-label-table <table>`. `--out` also differs.
- They share the seed, so the data order is the same (common random numbers). A test asserts that the argv diff is exactly these two tokens, and it goes RED on any other difference (§10).
- ⛔ **The card's P5-D control (drop-if-sim8) is NOT run.** The PI's approved control is C0. P5-D stays a candidate later lever; it would need the full-pipeline sim8 pass, up to 11.1 process-h [E, `P5C_READINESS.md` §2].

### 2.1 Base, and what happens if P7 fails its bars (committed)

**Default base B-P7:** P7's `run/model_final.pt`. Its sha256 is bound into the launch token's data manifest (G-DVB `init_from`).

**Fallback base B-P6:** P6's `model_final.pt`, sha256 `968a469d…`, with P6's argv (no yaw table, no aux head). It still uses **the P7 bank**: a reserve-free subset of P6's own training bank, so one label table serves both bases and the reserve stays out of every P5-C input. The navtest reads then use P6's harness, with no yaw table.

**The rule. It reads P7's committed `SPEC_P7.md` §5 verdict, and nothing else:**

| P7 state | base |
|---|---|
| ADOPT, or NOT PROVEN, vs P6 | **B-P7** |
| REFUTED vs P6 (upper bound of Δ < 0) | **B-P6** |
| INVALID: P7 did not reach its final step (a K3 / K6 kill or a lost run), or P7's post-training zero-gradient gate is RED on a **planner** parameter, or P7's read cannot produce a valid verdict (determinism or harness-pin failure) | **B-P6** |

**Why fine-tune from P7 even when it is NOT PROVEN, and not wait:**
- The P5-C contrast (C1 − C0) is internal to the base, and the label lever is orthogonal to P7's four changes (head, reserve, yaw, noise).
- NOT PROVEN means P7 is indistinguishable from P6 at the programme's bar. That is no evidence that P6 is the better base.
- P7 carries the PI-admitted yaw input and the reserve-free bank.
- No other base is scheduled, so waiting buys nothing.
- REFUTED or INVALID switches the base because a label lever is only worth adopting on the best available trunk.
- ⚠️ This rule reads a navtest verdict. That verdict is P7's own, one look, counted under P7's card. It selects between two bases before any P5-C number exists, and both P5-C arms share the choice. Disclosed here; it touches no P5-C contrast.

**Timing (no idle pod, bounded loss):**
- P5-C launches on B-P7 as soon as P7's final step is banked and the gates pass. P7's navtest read runs on the dev box in parallel.
- If that read returns REFUTED or INVALID while arm 1 is training, arm 1 is stopped at its next checkpoint. It is banked as `DISCARDED`, logged, never read. Both arms then restart on B-P6.
- If P7's verdict is not banked when arm 1 finishes, arm 2 does not start until it is. The loss is bounded at one arm.

**Order:** C1, then C0, on the same pod and GPU.
- C1 runs first so that the new label path fails, if it fails, before C0's budget is spent.
- A pod change between or within arms is disclosed as a hardware deviation (as in `PREREG_P6` Amendment (d)), and the arm resumes with `--resume` on the same argv.
- **Neither arm is read before both `model_final.pt` files are hashed and sealed in one manifest** (§11).

## 3 · Fine-tune recipe (identical in C1 and C0; each lever named)

The edits are relative to **P7's token-bound launch argv**. Read P7's `config.json` and its launch token at freeze; anything not listed here is P7's value.

| lever | C1 = C0 | P7 value | why |
|---|---|---|---|
| warm start | `--init-from <base model_final.pt>` (FULL strict load, `train.py` `warm_start` :1086-1098) | from scratch | the PI's arm form |
| optimiser state | **fresh AdamW** (what `--init-from` already does: "the optimiser and schedule start fresh", :2199) | — | see below |
| lr / schedule | **5e-5**, cosine to 0 over 2,000 steps, no warm-up | 2e-4, cosine over 9,075 | see below |
| steps | `--steps 2000`, `--epochs 0` = 512,000 sample-passes = **5.51 scene-epochs** of 92,896 [arithmetic] | `--epochs 25` = 9,075 | the PI's arm form |
| batch | `--batch 4 --accum 64` (effective 256) | same | WTA coverage needs the effective batch (`train.py` :2126-2132) |
| wd, clip, precision | 0.01, 1.0, bf16 + TF32 | same | unchanged |
| data | the P7 bank `train_grow_cmd_3c_p7`, 150,603 tuples, `targets_sha256 92af8d66…`, `--epoch-unit scenes` | same | no row re-selected; the reserve stays out |
| scorer | `--labeller online --label-chunks 2 --scorer-sees-trunk --score-w 0.1 --scorer-mode` as P7, D9 metric cache as P7 | same | the scorer's labels are the student's own tracked proposals in both arms |
| nav, yaw | `--nav-mode command --yaw-wrap off`; `--ego-yaw-table` = P7's pin (B-P7 only) | same | P7 is yaw-trained |
| aux head | **OFF**: no `--aux-drv`, `--w-aux-drv` absent (0.0) | on, w 0.1 | see below; open question Q2 |
| seed | 0 | 0 | same data order as P7's seed, and C1 = C0 |
| `--label-workers` | the pod's derivation (throughput only; not in `IDENTITY_ARGS`, `train.py` :1071) | 13 | `PREREG_P6` Amendment (e) |
| **WTA target** | **C1: `--traj-label-table <table>`; C0: absent** | raw `traj` | **the one lever** |

**Why lr 5e-5:**
- (i) It is P2's registered warm-start lr: a quarter of the paper's peak, "since it warm-starts a converged model" (`PREREG_P2.md` Design). P2's seed replicate is our only joint-fine-tune noise floor F. Keeping its lr keeps F the closest available proxy for this rig; changing lr would make F a worse one.
- (ii) It is the lr P7 itself ran at around step 6,050 of 9,075 (cosine 0.25 × peak at t/T = 2/3) [arithmetic]. So it lies inside P7's own trajectory, not above its peak.
- (iii) Its integrated lr is 5.5 % of P7's (0.05 against 0.9075) [arithmetic]: a perturbation of a converged model, not a re-training.
- Restarting at 2e-4 would re-open P7's full step size on a converged LoRA and push C0 − P7 away from 0.
- Whether 2,000 steps at 5e-5 is enough dose to absorb the label is **[U]**. The learnability readout L (§8.3) measures it, and the failure branches (§12) depend on it.

**Why fresh optimiser state, not P7's moments:**
- (i) **Symmetry.** P7's Adam moments are statistics of raw-label gradients. They match C0's gradients and not C1's, so loading them would tilt the contrast toward C0.
- (ii) It keeps the post-training Adam-state gate (§10.4) informative. With fresh state, a parameter without state never received a gradient. With P7's state loaded, every parameter would already hold state and the gate would be vacuous.
- (iii) It needs no new flag.
- Whether `model_final.pt` holds any optimiser state at all is **[U]**; it is a planner-format file.

**Why the aux head is OFF:**
- The head is training-only, and its weights are **not** in `model_final.pt` (`SPEC_P7.md` §1, "Not in model_final.pt"). `warm_start` loads the planner only; the head state is restored only by `--resume` (`train.py` :1580-1585, :1692).
- So `--aux-drv` under `--init-from` would build a **fresh random head**, and its ≈ ln 2 BCE gradient would flow into a converged LoRA from step 0. That is a perturbation that is neither the label lever nor P7's recipe.
- Warm-starting the head would need a new default-off flag (the defect class this repo has paid for five times) and its own G-DVB entry.
- The cost of this choice: C0 − P7 also contains "2,000 steps without the aux loss". That is why **C0, not P7, is the comparator**.

## 4 · The label table, and the rank-1 policy

### 4.1 Construction

**Builder:** P5-S's code, unchanged (`PK/eval/p5_target_labels.py` blob `294644df`: `iterate20`, cubic, ITERS 3), through the labels agent's thin driver (`SG/p5c/labels/p5c_labels.py`).

**Where:**
- **On the pod, after P7's last step** (MM decision, [I] `LS` ~23:00Z). The pod holds every nuPlan DB.
- The dev box holds DBs for only **482 of 1,075 logs** [I, `probe_rows.json`].
- `NUPLAN_MAPS_ROOT` is set explicitly. The default `D:/...` path fails every row [M `P5C_READINESS.md` §2].

**Key:** (log, token, rank, sha256 of the row's own `traj` as float64 LE bytes). Rank-1 twins share the rank-0 token, so a token key would collide. The sha also proves each label was computed from the row it replaces.

**Labels:** [20, 3] each. A row whose Tc20s raises gets the identity label (label := `traj`, `source: raw_fallback`), and the fallbacks are counted. Labels are computed for **all** 150,603 rows. The rank policy (§4.3) is applied afterwards as a mask in the table meta, so the build does not wait on the rule.

**Table meta:** schema, interp, iters, code blob shas, bank `targets_sha256`, `rank_policy`, `n_rows`, `n_raw_fallback`, `n_differs` per rank, build host, and the table's own sha256 (`P5C_LABEL_TABLE_SHA256`, registered in `launch_gate.py`).

### 4.2 Table gates (CPU, before the launch gate; any FAIL means no launch)

| id | check | bar |
|---|---|---|
| TB-0 | cross-environment bit-identity: the pod's labels on the 482 dev-box logs equal the dev-box build | max abs diff **0.0** on every common row |
| TB-1 | reproduction: the driver on P5-S's 1,018 rows reproduces `p5s_cubic.npz` `labels20_t20_tc20_tw` | bit-identical, **1,018 / 1,018** (60/60 and 100/100 already [M `P5C_READINESS.md` §2, `SG/p5c/labels/CONTROL_P5S_BITIDENT.json`]) |
| TB-2 | knot exactness C2′ on every row | ≤ 1e-9 |
| TB-3 | identity control: `ident` = `traj` bit-for-bit on every row | 100 % |
| TB-4 | coverage: table keys = the bank's keys | **150,603 / 150,603**; `n_raw_fallback` ≤ 1.0 % of rows, else REFUSE (threshold committed here, before data) |
| TB-5 | sanity: each label finite, \|Δyaw vs `traj`\| < π (the `--yaw-wrap off` branch check), `sim_hz` 10 on every row | 100 % |
| TB-6 | learnability, **per rank**: median per-row max \|acc\| and max \|heading rate\| of the labels (origin prepended at 0.2 s, P5 §3 `learnability()`) against the same rows' `traj` | ≤ 2× on both. **Rank-0 FAIL ⇒ STOP** (P5-S bar (c) fails at bank scale; next lever P5-R, card §9). Rank-1 FAIL ⇒ rank-0 only (§4.3) |
| TB-7 | `n_differs` per rank is recorded | it feeds G-LIVE (a) |

### 4.3 Rank-1 policy (mechanical; its inputs exist before any P5-C outcome)

**Inputs:**
- (I1) the repair measurement by the labels agent: `SG/p5c/labels/RANK1_REPAIR.md` with `repair_rank1.json`, and the same tool's rank-0 sample `repair_rank0.json`;
- (I2) TB-6 on the rank-1 rows;
- (I3) TB-2 and TB-3 on the rank-1 rows.

**The quantity:** r_k = sim8-fail rate(T) − sim8-fail rate(Tc20s) on rank k's sample, in percentage points. The estimator is P5 §3's paired log-cluster bootstrap: ratio estimator, 10,000 draws, seed 20260927, 95 % percentile. If the report states another estimator, the MM recomputes r_k from the per-row records with this one. No other estimator is admitted.

**RK0, a stop rule:** if r_0's upper CI bound < 0, the label is harmful on the training bank's own rows. **No launch.** The next lever is P5-R.

**ALL-RANKS iff every one of these holds:**
- RK1: the rank-1 sample has n ≥ 200 rows over ≥ 5 logs;
- RK2: r_1's CI-lo > 0;
- RK3: r_0's point > 0 (the same sign as rank-1, and as P5-S's held-out +7.07 pp);
- RK4: TB-6 passes on the rank-1 rows;
- RK5: TB-2 and TB-3 pass on every rank-1 row;
- RK6: if the report gives P5-S §6's guards (DDC, comfort, lk10, executed-vs-intended), none is adverse in §6's directions. If they are absent, RK6 is not required: the repair tool measures sim8, which is the P5-S primary.

**Otherwise RANK-0 ONLY:**
- In C1, rank-1 rows carry the identity label (`source: raw_rank1`). C0 is unchanged. The bank and the sampling stay P7's.
- The claim is scoped **"Tc20s on rank-0 rows"**.
- Rank-1 tuples are 38.3 % of the bank. Under scene-epoch rotation they are ≈ 31 % of draws [E; the alternation rule is [U]], so the lever then reaches ≈ 69 % of draws.
- This is a narrower single lever, not an unattributable mixture.

The output goes into the table meta as `rank_policy` ∈ {`all`, `rank0`}, and G-DVB checks it.

⚠️ **Disclosure.** While reading `LS` for P7's status, the drafter saw the MM's one-line summary of that measurement. It reports rank-1 +4.67 pp [2.33, 7.21] and rank-0 +3.00 pp [1.06, 5.28], 300 rows each, "0 made worse" [I]. The drafter did **not** open `RANK1_REPAIR.md` or `repair_rank*.json`.
- The rule above is P5-S §9's bar transposed to rank-1 rows, plus the brief's example ("same sign and CI-lo > 0"). No threshold in it was chosen from that line.
- Read at face value, RK2 and RK3 would pass, RK1 needs the log count, and RK4 and RK5 are not yet computed. The estimator of that summary is [U].
- The MM may wish to have the rule checked by someone who has not seen the line.

## 5 · Endpoints and estimator

**Population:** full navtest, 12,146 tokens over 136 logs, NAVSIM v1.1, read on the **dev box, never on the pod**.
- The harness is the pinned post-NT-FIX1 one (`harness_pins.py --verify`; `PREREG_P6` Amendments (b) and (f)): A7 last-heading repair ON, `navsim_v1` selection on the arm's own scorer logits, `model_final.pt`.
- B-P7 arms read through the navtest yaw table `f5ebe24f…`, as P7 does (`SPEC_P7` Amendment YAW).

**Primary, LQR-tracked:** PDMS × 100 of each arm's pick under NAVSIM's LQR + kinematic bicycle. The contrast is **d = C1 − C0**, paired per token.

**PI condition 1, tracker-transparent:** the **same pick** of each arm (the 20-pose plan after A7, exactly what the tracked path converts) scored as executed, with no tracker.
- The builder is E2's `transparent_states` (`PK/raw/2026-11-11-teacher-review/e2e1/e2_score.py`): a cubic spline through the origin and 20 poses, giving 41 states at 0.1 s with analytic velocity, acceleration and yaw. It goes into the same PDMScorer and the same metric cache, against PDM-Closed.
- The contrast is **d_T = C1_T − C0_T**.
- The **tracker dependence** is **TD = d − d_T**.

**Estimator:**
- Paired log-cluster bootstrap over the 136 logs, 10,000 draws, seed 20260927, 95 % percentile.
- **Every arm and both scorers share the same draws:** C1, C0, P7 (B-P7) or P6 (B-P6), and C0r if trained.
- Rates use the ratio estimator.
- It answers *another draw of LOGS* only.

**The three variances, named:**
- **Logs:** answered by the interval.
- **Training run:** NOT answered by the interval. It is handled by the floors and the replicate (§6), per `H-ESTIM-SEED-1`.
- **Inference run:** REFe inference is deterministic. This is **asserted**: a second pass of each arm over a seeded 10 % subset (1,215 tokens, seed 20261112) must be bit-identical in per-token PDMS under both scorers. A mismatch is a defect, investigated before any read.

## 6 · Bars, verdicts, control, replicate

### 6.1 The floor B (committed)

**B = max( t·s_p·√2 , 2F ) = max( 2.776 × 0.1791 × 1.4142 , 2 × 0.33 ) = max( 0.7031 , 0.66 ) = 0.7031** PDMS × 100 [arithmetic on M inputs].

- **Why √2.** The programme's 0.4971 bar (`PREREG_DOSE_P6` D1, `PREREG_PESS_P6` P1) is for ONE trained arm against a FIXED checkpoint (P6). C1 − C0 has two trained arms, and each carries its own training noise. `PREREG_REP_P6` R2's own form, t·s_p·√(1/n₁ + 1/n₂), gives √2 at n = 1 and 1. The shared seed may make the null difference smaller, but that is unmeasured, and `H-ESTIM-SEED-1` forbids assuming it. (Q1.)
- **Why both terms.** s_p is the only pooled seed SD in the programme, but it comes from scorer-only fine-tunes, so it likely **understates** the noise of a joint fine-tune [H]. F comes from the joint-fine-tune recipe P5-C copies, but it is one draw. Neither covers the other's weakness, so the bar is their maximum.
- 2·F3 = 0.759 is reported beside B and decides nothing: it is a scorer-only rig, from the same 7 runs as s_p.

### 6.2 Primary verdict (tracked)

| clause | condition |
|---|---|
| P1-a | CI-lo(d) > 0 |
| P1-b | d (point) > B = 0.7031, strictly |
| G1 (guard) | no tracked sub-score of C1 − C0 (NC, DAC, TTC, EP, C) is separated adverse (CI-hi < 0). This is P2-REP RR1's safety clause, the clause that failed on TTC there [M `RESULT_P2REP.md`] |
| G2 (guard) | no LONGITUDINAL or LATERAL component of the **tracked-executed** families (§8.2) is separated adverse, as in P6/P7's ADOPT rule |
| CTRL-1 | the control is valid (§6.4) |
| P1-c (sufficiency) | the replicate clause (§6.3) |

| verdict | when | wording |
|---|---|---|
| **LEVER** | all gates; P1-a, P1-b, G1, G2, CTRL-1 and P1-c | "C1's plans survive the benchmark's tracker better than C0's", plus §7's qualifier |
| **SEPARATED-SINGLE** | as LEVER, but C0r was not run (not approved, or still training) | "separated, single seed, not yet a lever claim" |
| **SWAMPED-BY-REPLICATE** | P1-a and P1-b pass, P1-c fails | NOT PROVEN: inside this rig's own run-to-run noise |
| **SWAMPED** | P1-a passes, P1-b fails | NOT PROVEN |
| **NULL** | the CI of d contains 0 | NOT PROVEN |
| **REFUTED** | CI-hi(d) < 0 | C1's plans survive the tracker **worse** |
| **GUARD-FAIL** | P1-a and P1-b pass, G1 or G2 fails | NOT ADOPTED; the component is named |
| **INVALID** | any §10 gate or any read control fails, or CTRL-1 fails | no verdict; C1 − C0 is printed as descriptive only |

### 6.3 The replicate (H-ESTIM-SEED-1's sufficient form): conditional, so it is cheap

- **C0r** = C0 with `--seed 1` (argv diff = `--seed`, `--out`).
- **Trigger, committed:** C0r trains **iff** the primary reads SEPARATED-SINGLE, i.e. P1-a, P1-b, G1, G2 and CTRL-1 all pass. That is the only state in which a lever claim is on the table. In every other state no replicate can turn the verdict into LEVER.
- **P1-c:**
  - with d_r = C1 − C0r (cross-seed) and F_p5c = |C0r − C0| (this rig's own seed draw): **CI-lo(d_r) > 0 AND min(d, d_r) > max(B, 2·F_p5c)**;
  - F_p5c is reported next to F and F3;
  - no floor is dropped after it is seen.
- **Cost:** +28–29 L40-h [E, §14], spent only in that one state. **It needs the PI's go.** Without it, the strongest available verdict is SEPARATED-SINGLE.
- ⚠️ Same-seed reruns are bit-identical on this hardware class (MEMORY, box-2 training [I]), so the replicate **must** change the seed.
- **Alternatives, listed and not proposed:**
  - (a) C0r unconditional: +28–29 L40-h, always;
  - (b) a full second seed pair, C1r + C0r: +57 L40-h; then the bar becomes t·s_p·√(2/2) = 0.497, and the verdict reads the mean over 2 pairs plus sign agreement.

### 6.4 The control C0, and what C0 − P7 must read

C0 − P7 measures **fine-tune drift**: 2,000 more raw-label steps with fresh Adam and no aux loss. P6 replaces P7 under B-P6. It is reported under both scorers, with its CI.

- **CTRL-0, mechanical:** G-LIVE (f) (§10.2). At step 0, C0's planner output on real rows is bit-identical to the base's `planner.py` load.
- **CTRL-1, valid control:** C0 must not be separated-worse than the base beyond the floor. **The control is INVALID iff CI-hi(C0 − P7) < 0 AND (C0 − P7) < −B.**
  - A drift inside the floor, or a positive drift, is valid and printed.
  - If CI-lo(C0 − P7) > 0 with (C0 − P7) > B, it is reported as **"the fine-tune recipe moves the base by itself"**. That is a finding, and it changes nothing in C1 − C0.
- **If CTRL-1 fails:** the read is INVALID-CONTROL. The next lever is the recipe itself (§12 O7).

### 6.5 Deploy clause (separate from the label claim)

C1 replaces P7 as the programme's trunk only if all of these hold: verdict LEVER; CI-lo(C1 − P7) > 0 tracked; and no tracked sub-score of C1 − P7 is separated adverse. Otherwise the outcome reads "label lever shown, trunk unchanged".

Goal **94.55** (DriveZero Table A13 [P]): C1's tracked level is reported against it, GOAL MET / NOT EXCLUDED / BELOW GOAL as in `PREREG_P6`. The ≤ ~0.25 harness gap is reported, never subtracted [E, REVIEW_7 §1.2].

## 7 · PI condition 1, operationalised: the tracker-transparent qualifier and the wording

**Every** number in §6 and §8 is printed twice, tracked and transparent, from the same picks, tokens and draws.
- Where a number involves no tracker (training losses, distance of a plan to a label), it is labelled **"no tracker in this number"**, so that its absence from the pair is visible rather than silent.

**Every verdict carries exactly one transparent qualifier, read on the same draws:**

| qualifier | condition | meaning, printed verbatim |
|---|---|---|
| **TM, tracker-mediated** | CI-lo(TD) > 0 and CI-lo(d_T) ≤ 0 | "the gain exists only through NAVSIM's tracker" |
| **TM+C, tracker-mediated with path cost** | CI-lo(TD) > 0 and CI-hi(d_T) < 0 | as TM, plus: "C1's plans are worse as paths by d_T [CI]" |
| **TI, also present untracked** | CI-lo(d_T) > 0 | "the plans also score higher as paths on the PDMS formula" |
| **TU, undetermined** | none of the above | printed with both CIs |

**Wording, binding (PI condition 2).**
- The claim sentence for any positive verdict is: *"C1's plans survive the benchmark's tracker (NAVSIM v1.1 LQR) better than C0's: tracked +d [CI], transparent +d_T [CI], tracker dependence TD [CI], qualifier X."*
- ⛔ "drives better", "safer", "better driving" and their synonyms are banned in every result, register row, report and paper text derived from this card, **including under TI**.
- A TI qualifier says the paths score higher on a formula. It is not a closed-loop result.

**What to expect, stated before data:**
- The label itself reads TM+C at teacher level (transparent −15.87 [M, §0]).
- So **TM+C on the student is the expected signature that the student absorbed the pre-distortion. It is not a defect**, and it does not fail the verdict.
- What TM+C does mean is that the gain is specific to NAVSIM's LQR tracker. Any full-retrain proposal under §12 must carry that sentence to the PI as a deployment caveat: a real vehicle's controller is not NAVSIM's LQR.
- Transparent families are reported and never used as guards (§8.2).

## 8 · Secondary and diagnostic (reported; only G1 and G2 decide)

### 8.1 Sub-scores and pick rates

**Per scorer (tracked and transparent), each as paired CIs for C1 − C0, C0 − base and C1 − base:**
- the sub-scores NC, DAC, DDC, TTC, EP, C;
- the **DAC-zero pick rate**, **PDMS-zero pick rate** and NC-zero pick rate (the pick's own component equals 0; ratio estimator).

**Mechanism counts (ZT4/ZT5's quantity), per arm:**
- **tracker-made** DAC zeros: tracked 0, transparent 1;
- **tracker-saved** DAC zeros: tracked 1, transparent 0;
- C1 − C0 for each, with CIs.
- [H] C1 has fewer tracker-made zeros. For C1, tracker-saved rises by design.

**Transparent comfort C** is printed with E2's caveat: transparent comfort depends on how the states are derived [M `PK/raw/2026-11-11-teacher-review/e2e1/RESULT_E2E1.md`].

### 8.2 Four metric families (binding; per family, never pooled)

**Instrument and frame:**
- The instrument is `families6.py`, the one behind `sft4_B_u400_families.json` and P6/P7's read, on the identical tokens, against the logged human future, with paired CIs.
- Families are computed **twice**: (a) on the **LQR-executed** states, and (b) on the **transparent** 41 states of the plan.
- Which of the two `families6` reads today is **[U]**: the P2 runbook says "executed poses bit-equal to the deployed seam", and that may be the post-A7 8-pose plan rather than the LQR states. The missing reading is work item **W-FAM** (§10.5), and it is done before the read.
- **G2 uses (a) only.** The reason, committed before data: on a pre-distorted plan, (b) is inflated by construction. P4 F2 measured yaw-rate MAE going 3.23 → ~8.4 °/s on exactly such a reference [M, P4 F2]. Guarding on (b) would penalise the mechanism under test.

| family | components | status |
|---|---|---|
| **LONGITUDINAL** | speed MAE and bias; along-track MAE and bias; EP; TTC; headway / time-gap to the lead where a lead exists (with its n, and the reason where none exists) | computable (families6 + NAVSIM). Headway's presence in families6 is [U]; if absent, it is a work item before the read |
| **LATERAL** | heading MAE; cross-track MAE; curvature MAE; yaw-rate MAE; DAC; DDC; **last-pose heading error** reported separately (the A7 seam, §10.5 W-A7) | computable |
| **TACTICAL** | command adherence (turn-active accuracy headline, κ, 3×3 confusion, random-proposal floor; `command_adherence.py`); selection quality, random / actual / best-of-64 (`--dump-proposals`) | computable, as in P6's read. **Tactical goal-setting is NOT computable:** REFe has no tactical goal head, and NAVSIM carries no manoeuvre label beyond the driving command (n = 12,146, reason stated) |
| **STRATEGIC** | — | **PAUSED** by the PI's ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` item 25). Not a pass, not omitted; also unavailable in NAVSIM |

### 8.3 Learnability readout L (student side; NOT navtest; it decides the failure branch, never the verdict)

**Population, mechanical:**
- First choice: the ZT5 dev set (navtrain held-out, 2,466 sets, 13 logs) **iff** all 13 of its logs are absent from the P7 bank (asserted by a script before use).
- Else: a seeded 1,000-row subset of the R-A reserve's rank-0 rows (seed 20261112, stratified by log), never trained on by P7, C0 or C1. This counts as a reserve read and is logged as one.
- Else: **UNAVAILABLE**, with the reason. It is a work item, never a silent omission.
- Disjointness and image availability are both **[U]** today.

**Metrics (Tc20s labels for the population are built by the §4 driver; CPU, minutes [E]):**
- **L1, absorption** (no tracker in this number): the WTA quantity, min over the 64 proposals of L1 to the label. ΔA = L1(C1, Tc20s) − L1(C0, Tc20s), together with L1(C1, T) − L1(C0, T).
  - **ABSORBED iff CI-hi(ΔA) < 0.** Else NOT ABSORBED.
- **L2, executed against intent** (tracked): max-xy, along, cross and heading deviation of each arm's LQR-executed pick from the teacher's intended path T; C1 − C0. [H] C1 < C0.
- **L3:** tracked and transparent PDMS on this population (TF1's scorer). No navtest, K unchanged.
- **L4, learnability of the student's own plans:** median per-plan max |acc| and max |heading rate|, C1 against C0 (P5-S (c)-style, 2× flag).

**Order:** L is computed and banked **before** the navtest read of either arm, so its result cannot be shaped by navtest.

**T0 training diagnostics** (no tracker in these numbers), reported and never gating: WTA traj L1 to each arm's own labels, score loss, scorer-target coverage, `no_parts`, at matched steps.

## 9 · Expected effect and power (stated before data)

- The teacher-level gain (+7.64) does **not** transfer one-for-one. Student-level references: P4 F2 test-time compensation **+1.07 [+0.68, +1.45]** [M, older student]; ZT5 F_a **−0.19 [−0.76, +0.63]** [M, dev].
- Training-side, the fan and the scorer can both adapt, so the effect could exceed F2's. Equally, the tracker "saves about as many as it breaks" (ZT4), so it could be about 0.
- **ESTIMATED tracked effect: 0 to about +1.1 PDMS. The bar B = 0.70 sits inside that range** [E/H].
- ⇒ **This approved design may read SWAMPED even if the label works** [H, roughly even odds]. It is stated here so that a SWAMPED read is not a surprise, and so that the MM and PI can weigh the alternative before the spend (Q5).
- Cheaper ways to gain power do not exist: the noise is training noise, so more tokens do not help. The ways that do exist cost compute: more seeds per arm, or the full-retrain dose.

## 10 · Gates

### 10.1 Launch gate (no training step without a PASS token, CLAUDE.md 2026-09-26)

- **Profile `refe_p5c`**, cloned from `refe_p7` (`launch_gate.py` :379). It registers two argvs, C1 and C0 (and C0r if approved). The PASS token per arm is bound to the commit, the argv sha256 and the data-manifest sha256s: base checkpoint, label table (C1), bank, yaw table, metric cache.
- A test asserts **argv(C1) − argv(C0) = {`--traj-label-table`, `--out`}** and argv(C0r) − argv(C0) = {`--seed`, `--out`}. Any other differing token is RED.
- **Pins:** `P5C_LABEL_TABLE_SHA256`; rows = `P7_BANK_TUPLES` (150,603); `P7_TARGETS_SHA256` `92af8d66…`; `P5C_BASE_SHA256` (P7 final, or P6's `968a469d…` under B-P6); `expect_schedule` {epochs 0, steps 2,000, epoch_len 92,896}.
- ⚠️ A pin edit moves 6 profile fingerprints pinned in 3 tests. Run every launch_gate-reading test (~69) before landing [I, MEMORY].

**G-DVB entries (declared = built; each with a mutation that must FAIL):**

| dest | C1 / C0 | check | RED mutation |
|---|---|---|---|
| `traj_label_table` (data) | table / absent | file sha256 = pin; keys = 150,603 bank keys; the bank's recorded `traj_label_sha256` = the file's; table `rank_policy` = §4.3's output; with the flag off, the bank's `item[3]` = raw `traj` | kwarg dropped at the train-bank ctor (`label_table_not_passed`); lookup reverted at `__getitem__` (`label_lookup_removed`); one key removed (REFUSE); `rank_policy` flipped |
| `init_from` (data) | base / base | sha256 = `P5C_BASE_SHA256`; FULL format; tensor count = the model's | a different checkpoint |
| `steps`, `epochs` | 2000, 0 | cosine `T_max` = 2,000; `expect_schedule` | `--epochs 25` left in |
| `lr` | 5e-5 | `opt.param_groups[0]["lr"]` at step 0 = 5e-5 | 2e-4 |
| `aux_drv`, `w_aux_drv` | off, 0.0 | no head module; no `l_aux` term | `--aux-drv` given |
| `resume` | forbidden at start | `--out` holds no `ckpt_last.pt` at the first start (fresh optimiser) | a pre-seeded `ckpt_last.pt` |
| `ego_yaw_table` | P7 pin (B-P7) / forbidden (B-P6) | as `refe_p7` | as `refe_p7` |

- Every other `refe_p7` entry is unchanged.
- `r["traj"]` is also read at train.py :316 and :446 (the scorer and on-policy banks). They are not used under this profile, because those flags are forbidden. The G-DVB entry states this.
- Held-out and eval banks never receive the table; a mutant that passes it to them must FAIL.

### 10.2 G-LIVE (the lever is a data substitution, so the probes are bit-level, not only gradients)

| id | probe | bar |
|---|---|---|
| (a) | through the **real** DataLoader, `item[3]` = the table label, bit-for-bit, on N ≥ 256 rows of each labelled rank; the differing fraction ≥ TB-7's per rank. With the flag off, `item[3]` = raw `traj` bit-for-bit. Under `rank0` policy, rank-1 rows = raw bit-for-bit | exact |
| (b) | fixed weights, one real batch: `wta_loss(table) − wta_loss(raw)` | non-zero with the flag; **exactly 0.0** without it |
| (c) | identity table (label := `traj`) | reproduces flag-off **bit-for-bit** (loss, gradients, one optimiser step) |
| (d) | deliberate-regression arms `label_table_not_passed` and `label_lookup_removed` | each must make the gate **FAIL** |
| (e) | `check-bank`: table keys against bank rows | 150,603 / 150,603 |
| (f) | **warm-start identity** (P2's G1): before step 1, trajectories and scores on ≥ 64 real rows equal `planner.py`'s strict load of the base | bit-identical; RED arm: a different checkpoint |
| (g) | gradient reach: ‖∇traj_head(table) − ∇traj_head(raw)‖ at fixed weights | non-zero with the table; exactly 0.0 with the identity table |
| (h) | lever isolation: at fixed weights, the score loss under the table = under raw, **bit-for-bit** (the online labeller scores proposals, which do not depend on the target); the traj loss differs | exact |

Inherited from `refe_p7` unchanged: G-LIVE-LAB (identity at the launch worker count, stall < 5 %), G-LIVE-MEM, the `no_parts == 0` guard on the first labeller events, the D7 twin check, and scorer-target coverage 1.0. A breach in any training window **stops** the arm. That is the P6 Amendment (c) defect class.

### 10.3 G-HYG, G-CLOCK, G-EVAL, G-CKPT, G-SUITE

As `refe_p7`, plus:
- **G-CLOCK:** the label is a function of the row's own `traj` and the ego state at its own token; the key carries `traj_sha256`. A one-row key shift must REFUSE.
- **G-CKPT:** `model_final.pt` loads through `planner.py` strictly. The meta carries `init_from_sha256`, plus `traj_label_sha256` and `rank_policy` (C1) or their absence (C0).
- **G-SUITE:** clean tree, 0 regressions, with the new tests in the profile's suite.

### 10.4 Post-training gate (before either arm is read)

- **G-ADAM.** In each arm's final `ckpt_last.pt`, every trainable parameter has AdamW state with step = 2,000 and non-zero `exp_avg_sq` (the `probe_zerograd3.py` pattern: registered against with-state, exact count). The frozen trunk has no state.
  - Control: a copy with one parameter's state deleted must FAIL.
  - This gate is meaningful only because the optimiser state is fresh (§3).
- **Weight checks:**
  - traj-head weights differ from the base in both arms (‖ΔW‖ > 0);
  - C1's traj-head weights differ from C0's (the arms really differ);
  - the frozen trunk is bit-identical to the base in both.
- **Banking:** checkpoints go to HF, private repo name to be set by the MM, verified by the **remote listing** (size and LFS sha256), never by the push status.

### 10.5 Work items that must close before the read (0 GPU on the pod)

- **W-TRANSP:** the navtest-scale transparent scorer. E2's builder exists only on navtrain. Controls, all before any transparent number:
  - (i) the new caller's tracked arm reproduces the pinned harness per-token, bit-for-bit, on the 24-token u400 control (csv sha256 `a71dd17c…`) and on all 12,146 tokens of the base's banked read;
  - (ii) the builder reproduces E2/TF1's `teacher_transparent` bit-for-bit on the 1,018 navtrain rows;
  - (iii) RED arm: feeding the LQR-simulated states into the transparent slot must reproduce the tracked score, and a 0.5 m lateral shift must change DAC on at least 1 token;
  - (iv) a rerun is bit-identical.
- **W-FAM:** families on both frames (§8.2).
- **W-A7:** the labels were built with **no** A7 (P5 / E2 / TF1 `to_navsim(t20)`), while training's online labeller and the read apply A7 (last heading := the previous pose's). 0-GPU check on the 1,018 held-out rows: Tc20s tracked PDMS with A7 against without.
  - [H] the difference is about 0. Index 19's heading is the only change.
  - If the difference is separated, the MM decides before launch, by an amendment, whether the labels are rebuilt with A7.
- **W-L:** the population of the learnability readout (§8.3).

### 10.6 Closure-land rule

The `train.py` / `launch_gate.py` / test wiring is staged in `SG/p5c/hold/` (the wiring agent). It **lands only after P7's final training step**: a land that touches `train.py` while P7 runs voids P7's bound records [I, MEMORY]. Then: land → G-SUITE on the landed commit → launch gate → PASS token → C1.

## 11 · K ledger (`SG/p6base/selectors/ledger_K.json`; K = 40 today)

- **One entry per arm read.** An entry covers both scorers: the transparent score is a second metric on the same picks and the same tokens, and it creates no new candidate for best-of-K selection (Q8).
- `n` is the next free value at read time: 41 and 42 if nothing else is read first.
- P7's own navtest read is counted under P7's card, not here.
- L (§8.3) is not navtest and is not counted.
- **Read order:**
  1. both arms' `model_final.pt` sha256s are written to `SG/p5c/read/SEALED.json` before either navtest pass starts;
  2. both arms are scored in one session;
  3. no per-arm number is printed before both are complete.

```json
{"key": "P5C:C1:<sha256 of C1 model_final.pt>:E3", "n": "<next>", "time": "<UTC at read>", "nominee": "C1", "family": "P5C", "set": "E3", "smoke": false, "scorers": ["lqr_tracked", "tracker_transparent"], "card_sha256": "<PREREG_P5C.md sha256>", "base_sha256": "<P5C_BASE_SHA256>"}
{"key": "P5C:C0:<sha256 of C0 model_final.pt>:E3", "n": "<next>", "time": "<UTC at read>", "nominee": "C0", "family": "P5C", "set": "E3", "smoke": false, "scorers": ["lqr_tracked", "tracker_transparent"], "card_sha256": "<PREREG_P5C.md sha256>", "base_sha256": "<P5C_BASE_SHA256>"}
{"key": "P5C:C0r:<sha256 of C0r model_final.pt>:E3", "n": "<next>", "...": "only if §6.3 triggers and the PI approves"}
```

- §6's tests are this card's own pre-registered tests.
- Any best-of-K statement ("the best arm in the programme") uses a Bonferroni-adjusted interval over all K reads, and prints K (`PROGRAMME_P6_SCORER.md` rule 4).
- The inference-determinism subset pass (§5) is an identity check on the same model and is not counted.

## 12 · Outcomes and the next arm (Rule Zero: no branch ends at a verdict)

| outcome | next lever (each needs its own card; GPU needs the PI's go) |
|---|---|
| **O1 LEVER** | (1) **P5-F proposal to the PI:** the full P7 recipe with `--traj-label-table` from step 0. P7 is its free single-lever control (same argv + 1 flag, seed, bank, hardware class). ~128–130 L40-h [E, `P5C_READINESS.md` §5]. It carries §7's qualifier and, under TM / TM+C, the deployment caveat. (2) The deploy clause (§6.5) decides whether C1 replaces P7 now. (3) In parallel, at 0 GPU first: the TF stream's policy-margin lever L2. 6 of the 10 residual DAC rows at teacher level are the policy's own [M TF1]. |
| **O2 SEPARATED-SINGLE** | C0r (§6.3). **Blocker: the PI's approval of +28–29 L40-h.** If it is declined, the result stands as "separated, single seed, not yet a lever claim", and P5-F is proposed with that wording. |
| **O3 SWAMPED / SWAMPED-BY-REPLICATE / NULL** | Branch on L1. **NOT ABSORBED** (the student did not move toward Tc20s): the dose was too small; P5-F is the lever (the full dose, and the deployable form), proposed to the PI. **ABSORBED** (the student moved, but the plans do not survive better): the tracker channel's net reach on this student is small, as ZT4 found at inference. Next: the boundary-margin levers, ranked by measured gross reach (`D-REFE-ZEROTAIL-3`: trunk DAC gradient 58 % / 1.48; P7.1 map supervision 26 % / 0.65), starting with margin-aware targets (P5-M / TF L2), whose label test runs at 0 GPU first. |
| **O4 REFUTED** | Diagnose with §8.1's tracker-made / tracker-saved counts and L4: did the pre-distortion break plans the tracker used to save? Next: (1) a **Tsel-s** label arm: Tc20s only on the rows where T fails sim8, about 8 % of rows. P5-S measured Tsel-s at 94.40 % dual compliance and 1.49× acceleration on its 82 rows [M `p5s_compare.json`, DERIVED]. It **reuses C0 as its control** (same recipe and seed), so it costs +28–29 L40-h. (2) **P5-R**, the regularised inversion (card §9), with its own 0-GPU label test first. |
| **O5 GUARD-FAIL** | The adverse component names the lever. TTC adverse is the progress-for-TTC signature seen in P2-s1 and in every RL arm [M `RESULT_P2REP.md`], so the next arm is the PI-approved separate-safety-head line. DAC or NC adverse sends it to the margin levers of O3. |
| **O6 INVALID (gate or read control)** | Fix the gate or instrument and **re-read**. A read defect never retrains an arm. A training defect retrains under a dated amendment. |
| **O7 INVALID-CONTROL (CTRL-1)** | The fine-tune recipe damages the base. 0 GPU first: the step-0 loss jump in `metrics.jsonl` (fresh-Adam shock) and the per-window drift. Then the recipe lever: lr 2.5e-5, or a 100-step warm-up, under an amendment, with the PI's pod time. |

**Done means:** the verdict is read **and** the next lever's card is written, or it is blocked on a named PI decision. Name which.

## 13 · What this card does NOT license

- ⛔ No "the student drives better" in any form (PI condition 2), not even under TI.
- ⛔ No closed-loop or real-vehicle claim. The tracker-transparent score is a formula on a path, not a drive.
- ⛔ No claim that Tc20s helps under any controller other than NAVSIM v1.1's LQR.
- ⛔ No P7 or P6 lever claim. No attribution of C0 − P7 to any single component (it bundles fresh Adam, the lr restart and the absent aux loss).
- ⛔ No navtest number is used for training or for choosing anything inside P5-C. The base rule's read of P7's verdict is disclosed in §2.1.

## 14 · Cost (ESTIMATED; the runs' own measurements replace it)

| item | L40-h | class |
|---|---|---|
| C1 | **28–29** = 2,000 × 51.2 s (P7's L40 rate, with the head; P5-C runs without it) + 2 % egress | E, rate [I `LS` l.1048] |
| C0 | **28–29** | E |
| **approved pair** | **≈ 57–58**, against the ~54 the PI approved (which used 57.0 s × 0.863) | E (Q4) |
| C0r, conditional, needs the PI | 28–29, only under O2 | E |
| B-P6 restart, if triggered mid-arm | ≤ the time to P7's verdict, ≈ 6–10 h | E |
| label table | ≈ 1 h of pod CPU before C1 (MM decision) | I |
| reads (dev box) | 2 × ≈ 5.3 h GPU [I, P6-era u400 read] + the transparent pass: ≈ 0.9 process-h per arm at 0.27 s/token [E from P5-S's 0.265 s/row] | I / E |

## 15 · Provenance pins, filled at freeze (blank = not yet known)

- PREREG_P5C.md sha256: `____`
- P7 `model_final.pt` sha256 (`P5C_BASE_SHA256`): `____` (or P6 `968a469d…` under B-P6)
- P7 launch argv sha256 and `config.json` sha256: `____`
- label table sha256 and `rank_policy`: `____` / `____`
- `n_raw_fallback`, and `n_differs` per rank: `____`
- bank `targets_sha256` `92af8d66…` and 150,603 rows: to be confirmed by the trainer's own recompute
- yaw tables: navtrain (P7's pin) `____`; navtest `f5ebe24f…`
- harness pins (`harness_pins.py --verify`): `____`
- commit of the landed wiring, and the per-arm PASS token ids: `____`

## 16 · Open questions for the Master Mind (resolve before freezing)

1. **Bar B.** I commit t·s_p·√2 = **0.7031** (two trained arms; the REP card's R2 form at n = 1, 1), not the single-arm 0.4971 the brief cites. Overrule before freeze if the MM reads the common seed as justifying √1.
2. **Aux head OFF during the fine-tune** (§3). The alternative is to warm-start the head from P7's `ckpt_last.pt` `aux_drv` key. That needs a new flag, a G-DVB entry and a mutation arm. Never a fresh head.
3. **Conditional replicate C0r**, +28–29 L40-h, PI approval. Without it, the ceiling is SEPARATED-SINGLE.
4. **Cost.** P7's measured L40 rate (51.2 s/step [I]) puts the pair at about 57–58 L40-h, against the ~54 approved. Tell the PI.
5. **Power.** The ESTIMATED effect (0 to +1.1) brackets B = 0.70, so the approved design may read SWAMPED with a working label [H]. Is the fine-tune still the right first spend, against P5-F (~130 L40-h, with P7 as its free control)? The PI's arm form stands unless the PI changes it.
6. **The A7 seam** (W-A7): labels built without A7, reads with it. 0-GPU check before launch.
7. **W-TRANSP and W-FAM** do not exist at navtest scale yet. They are pre-read work items, not launch blockers.
8. **K accounting:** one entry per arm covering both scorers (my choice), or one entry per scorer.
9. **Rank-1 disclosure** (§4.3): I saw the MM's one-line summary before writing the rule.
10. **Base switch mid-arm** (§2.1): launch on P7 before P7's read completes, and stop/discard on REFUTED or INVALID. The alternative is to wait about 6–10 h with the pod idle.
11. **The L population** (§8.3): the ZT5 dev logs' disjointness from the P7 bank, and image availability, are [U].
12. **P5-D is dropped** (§2). Confirm.
13. **Ledger home.** P5-C's reads go into the P6-scorer programme ledger, as the brief says. Confirm that P7's read is entered there too, or not.
