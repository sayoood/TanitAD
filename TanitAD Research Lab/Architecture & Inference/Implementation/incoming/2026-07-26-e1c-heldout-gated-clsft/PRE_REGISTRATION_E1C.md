# E1c — held-out-gated closed-loop SFT: the E1b successor with a CORRECTED forgetting guard

**Pre-registration — written and staged BEFORE the fine-tune was launched.**
`2026-07-26 ~02:15 Europe/Berlin` (`2026-07-26 ~00:15 UTC`; pod logs are UTC) ·
`tanitad-pod3` (A40, idle, 0 MiB at write time) · renderer-free imagination /
kinematic closed loop (**NOT** AlpaSim). PI: Sayed.

Evidence-class legend (CLAUDE.md operating standard): `MEASURED` (ours + artifact
path) · `PUBLISHED` (cited) · `INHERITED` (another agent/doc, NOT re-verified) ·
`ESTIMATED` · `HYPOTHESIS`.

---

## 0. Why E1c — the one defect E1b's own data localises

`MEASURED` (`…/2026-07-25-e1b-failure-gated-clsft/E1B_RESULTS.md`, `e1b_eval_result.json`):

E1b's failure-gated CL-SFT moved closed-loop behaviour more than anything this
program has tried — junction corridor-departure@K185 **0.8414 → 0.4144**
(paired Δ **−0.4270 [−0.6838, −0.1648]** SEP), overall **0.5877 → 0.1603**
(Δ **−0.4274 [−0.5161, −0.3378]** SEP, 43 clusters), peak |XTE| **38.94 m → 3.04 m**,
and OOD moved *favourably* (**1.2664 → 1.1339**), so the win is not a
distribution-shift artefact. It was nevertheless pre-registered **BOUND**, because
3 of 4 guardrails failed: open-loop ADE@2s **0.4747 → 0.6693**
(+0.1947 [+0.1415, +0.2522] separated worse), anchor_acc 0.6815 → 0.6163,
anchor_traj_l1 0.1775 → 0.2399.

**The defect, stated as a root-cause class.** The replay branch's loss **fell** on
parity-train (1.826 → 1.613, rp_anchor_acc 0.550 → 0.584) while **held-out**
open-loop ADE rose 41 %. The forgetting guard was monitored on **the very corpus
it replays**, so it could only ever report success.
*Root-cause class (for `Project Steering/RETRACTION_LOG.md`): **training-set
instrument used as a generalisation guard** — the sibling of "trainer val is not
eval output".* A training-set loss is not a generalization guard.

**And E1b measured only two endpoints.** Base (step 0) and a heavily-traded model
(step 4000). The intermediate trajectory is unmeasured and nearly free to measure.
**The frontier — closed-loop gain vs open-loop cost as a function of training
step — is the scientific object of E1c.** The single best point is just its most
useful reading.

**E1c hypothesis** `HYPOTHESIS`: with the guard moved to **held-out** data and the
trajectory instrumented, there exists an *earlier* checkpoint on the same run at
which the closed-loop corridor gain is already CI-separated while the open-loop
cost has not yet become CI-separated-worse. If no such point exists **anywhere**
on the frontier, the closed-loop/open-loop trade is **real and not an artifact of
the guard** — a genuine, publishable result.

---

## 1. What changes vs E1b, and what deliberately does NOT

**Attributability is the design constraint.** The only changed variables are the
**instrumentation and the guard**.

| | E1b | E1c |
|---|---|---|
| base ckpt | `refc-diffusion-base-v21-30k/ckpt.pt` step 29999 | **identical** |
| mined buffer | `mined_buffer.pt` (3,537 states / 362 parity-train eps) | **identical, reused — NOT regenerated** (md5 `a32cfe9bfea4b1b5c196d3bb7f71fa5f`, 2.3 h to rebuild) |
| seed | 0 | **0** |
| lr / warmup / schedule | 2e-5 / 100 / cosine over 4000 | **identical** |
| cl-batch / replay-batch | 16 / 16 | **identical** |
| `lam_cl` / **`lam_replay`** | 1.0 / **1.0** | 1.0 / **1.0 — HELD FIXED ON PURPOSE** |
| encoder | frozen (13,732,945 trainable / 90,458,632 frozen) | **identical** |
| replay corpus | parity-train `e438721ae894` (all 2376) | **identical** |
| **forgetting guard** | replay loss **on the replayed training corpus** | **held-out open-loop, paired vs base, at every checkpoint** |
| **checkpointing** | one rolling `ckpt.pt` (endpoint only) | **17 step-tagged checkpoints → 18 frontier points with base** |

**`lam_replay` is NOT a lever in E1c.** If the frontier shows no acceptable point,
λ becomes the **next** experiment, not this one. Squeezing a better number out of
this run would destroy the attribution.

---

## 2. Substrate (all `MEASURED`, re-asserted at startup)

| item | value |
|---|---|
| REF-C base ckpt | `tanitad-pod3:/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt`, step **29999**, 128 anchors `[128,4,2]`, 104,191,577 params |
| mined CL buffer | `/workspace/e1b/mined_buffer.pt` — **reused**, md5 re-verified at launch |
| replay corpus | `/workspace/pai_epcache/physicalai-train-e438721ae894` — the **sacred parity corpus**, 2376 episodes |
| held-out EVAL set | `/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6` — **the same 44 episodes as E1a and E1b** |
| **leak guard** | mined-episode ids ∩ held-out ids **must be 0**, re-asserted **byte-level at startup** exactly as E1b did (`--assert-disjoint-heldout` REFUSES to train on overlap) |
| harness | E1a `e1a_horizon.rollout` reused VERBATIM (+ E1b's declared additive 2 s capture) |
| estimator | **paired episode-cluster bootstrap**, `taniteval/ci.py`, **B=2000**, resampling the 44 held-out **episodes**. `overlapping_holdout_se` is used **NOWHERE** |

The leaky split `physicalai-val-f1b378f295ae` (78.5 % into parity train) is **not**
used anywhere.

---

## 3. The frontier — what is checkpointed and what is measured where

**Checkpoints (17 + base = 18 frontier points), fixed in advance:**

```
step 0 (= base) , 100 , 250 , 500 , 750 , 1000 , 1250 , 1500 , 1750 ,
2000 , 2250 , 2500 , 2750 , 3000 , 3250 , 3500 , 3750 , 4000
```

Step 100 is added because E1b's `cl_anchor_acc` moves 0.000 → 0.131 inside the
first 250 steps; the early region is where a cheap win would live.

Each checkpoint stores **only the trainable (non-encoder) parameters**
(13,732,945 params ≈ 54.9 MB) rather than the full 417 MB state dict.
`MEASURED` on this model: the **only** non-encoder buffer is the constant
`decoder.anchors`, and the encoder is frozen **including** its BatchNorm running
stats (`encoder.eval()`), so base + trainable-delta reconstructs the full model
**exactly**. This is asserted, not assumed: at step 4000 a **full** state dict is
also written and `base ⊕ delta == full` is checked key-by-key (`max|Δ| == 0`).
Disk: 18 × 55 MB ≈ 1 GB (12 GB `dd` headroom MEASURED on `/workspace`).

**What is measured at every checkpoint (cheap, ~60 s each):** held-out **open
loop** on all 44 episodes / 967 windows — **ADE@2s**, **anchor_acc**,
**anchor_ce**, **anchor_traj_l1**, plus the M1 lateral/longitudinal split.

**What is measured at every checkpoint (expensive, ~12 min each):** the
**closed-loop rollout at K=185** — corridor-departure **overall AND junction**,
window-departure, peak/mean |XTE|, peak |Δψ|, OOD envelope ratio,
out-of-envelope fraction, closed ADE@2s.

**Cost choice, stated as the brief requires.** E1b `MEASURED` per arm: open loop
**49–77 s**, K=185 **732–736 s**, K=20 **556–703 s**. 18 × (open + K185) ≈
**3.8 h** on an idle A40 — affordable, so **closed loop is evaluated at EVERY
checkpoint, not a subset.** The frontier is the deliverable and a subset would
leave it with holes. **K=20 is dropped from the per-checkpoint sweep** (it would
add ~3.3 h and E1a/E1b already established that the 2 s instrument is the one
that hid the effect in both directions); K=20 is run only for **base** and for
the **selected / endpoint** checkpoints, and is **reported, non-deciding**.

---

## 4. PRE-REGISTERED OUTCOMES — both committed here, in advance

**Primary metric.** Closed-loop **corridor-departure rate @ K=185** (18.5 s) on
the held-out 44 episodes, **paired episode-cluster bootstrap** (B=2000), reported
for **overall**, **junction** and **longitudinal** strata.

### 4.1 A checkpoint `c` is a SUCCESS POINT iff ALL SIX hold

| id | condition | direction |
|---|---|---|
| **P1** | corridor-departure@K185 **overall**: paired Δ(c − base) **CI-separated LOWER** (`separated ∧ hi < 0`) | must fire |
| **P2** | corridor-departure@K185 **junction**: paired Δ(c − base) **CI-separated LOWER** | must fire |
| **Ga** | open-loop **ADE@2s**: paired Δ(c − base) CI **includes 0 or is separated LOWER** — i.e. **NOT** (`separated ∧ lo > 0`) | must not fail |
| **Gb1** | open-loop **anchor_acc**: **NOT** (`separated ∧ hi < 0`) | must not fail |
| **Gb2** | open-loop **anchor_traj_l1**: **NOT** (`separated ∧ lo > 0`) | must not fail |
| **Gc** | closed-loop **OOD peak ratio** @K185 overall for `c` **≤ 1.30** (E1a's measured band) | must not fail |

`Gc` is carried over from E1b unchanged: a departure improvement bought by moving
the loop out of the measured perturbation envelope would be confounded.

### 4.2 The two committed outcomes

- **SUCCESS.** At least one checkpoint on the frontier satisfies all six.
  **That checkpoint is the deliverable.**
  **Selection rule, fixed in advance** (so it is not chosen post-hoc): among all
  SUCCESS points take the one with the **most negative paired Δ on
  corridor-departure@K185 overall**; ties (Δ within 0.005) broken by the **smaller
  open-loop ADE@2s Δ**; still tied → the **earlier** step.
- **BOUND.** **No** checkpoint anywhere on the frontier satisfies all six ⇒
  **the closed-loop / open-loop trade is REAL and not an artifact of the guard.**
  That is a genuine, publishable result and will be reported as such — as the
  measured shape of a trade-off, not as a failed attempt.

**The full frontier is reported either way** — closed-loop gain vs open-loop cost
at every one of the 18 points, with intervals and estimator named on each.

### 4.3 The in-training gate (the corrected guard itself)

At every checkpoint step the trainer computes, **on the held-out 44 episodes and
never on the replay corpus**, the paired Δ(current − base) for ADE@2s /
anchor_acc / anchor_traj_l1 using the same estimator, against **base per-window
arrays cached once at step 0 on identical windows**.

- **STOPPING POINT** := the **first** checkpoint at which **Ga** fails
  (open-loop ADE@2s paired Δ `separated ∧ lo > 0`).
- **Declared in advance:** the run **does not halt there** — it continues to step
  4000 and the stopping point is **recorded**, because the frontier beyond the
  gate is exactly what tells us whether the trade is real. Halting would destroy
  the object of the experiment. The stopping point is reported as *"where a
  correctly-guarded early-stopping run would have stopped"*.
- The probe **saves and restores every RNG state** (python / numpy / torch CPU /
  all CUDA devices) around itself, so the instrumentation **cannot** perturb the
  training trajectory. This is what keeps E1c's step-4000 endpoint comparable to
  E1b's.

### 4.4 Multiplicity — declared in advance, not discovered later

Scanning 18 checkpoints and reporting the best is a **multiple-comparison
procedure**. Two things are pre-committed:

1. The SUCCESS rule above uses the **same CI-separation predicate as E1a/E1b**, so
   the headline stays apples-to-apples with the program's other closed-loop
   verdicts.
2. **Additionally reported per checkpoint (non-deciding):** a
   `multiplicity_robust` flag requiring the closed-loop separation to survive a
   **Bonferroni-adjusted** level, `p_delta_gt0 < 0.05 / M` with `M = 18`
   (⇒ `p < 0.002778`), for **both** P1 and P2. If the selected SUCCESS point does
   not carry this flag, the report says so in the verdict line.

Symmetrically, a guardrail that "includes 0" may be a **power** artifact rather
than a genuine null. Also reported per checkpoint, non-deciding: the open-loop
ADE@2s **Δ point estimate and CI width**, plus a `noninferior_0p05` descriptor
(paired Δ upper bound `< +0.05 m`, ≈ 10 % of base ADE). These are descriptors.
**They do not move the verdict.**

### 4.5 Reproduction control (built in, free)

E1c step 4000 re-runs E1b's exact configuration. Its measured endpoint is
therefore a **reproduction test** of E1b. It is reported, and any drift is
reported honestly rather than smoothed — the run is seeded but not
bitwise-determinism-guaranteed (cuDNN, 4 dataloader workers). **The base arm is
also re-rolled**, which reproduces E1a/E1b's base row a third time.

---

## 5. Binding process requirements earned by E1b

1. **THE EVALUATOR IS VALIDATED BEFORE THE DECIDING RUN.** E1b's shipped
   `e1b_eval.py` would have reported **SUCCESS** on a BOUND run: its verdict
   string never consulted the guardrails, several were unimplemented, and the
   open-loop guardrail was unpaired. Before E1c's real run, `e1c_selftest.py`
   drives **the shipped verdict code path** with synthetic per-window arrays
   constructed to fail **each registered guardrail one at a time**, and asserts
   the failing verdict is rendered. It also replays E1b's real measured numbers
   and asserts **BOUND**. The self-test result is reported in `E1C_RESULTS.md`.
   *A guardrail absent from the code is a comment, not a guardrail.*
2. **A BOUND IS NOT RESCUED.** No re-tuning, no re-thresholding, no episode
   re-selection, no post-hoc stratum shopping. If it is BOUND, it is reported as
   BOUND. The thresholds, strata, checkpoint steps, selection rule and estimator
   in this document are frozen at write time.

---

## 6. Falsifiability & honest bounds (stated in advance)

- The closed loop is **map/agent-free** — this measures drift / corridor-keeping,
  **not** collision or off-road safety. A corridor-keeping win is not a
  certified-safety claim.
- The **junction stratum is ~6 windows in 6 episodes** at K=185. An
  episode-cluster bootstrap over 6 clusters is low-powered by construction; the
  junction number is never quoted without that. P1 (overall, 43 clusters) is
  required alongside P2 precisely so the verdict does not rest on 6 clusters.
- **K=185 is the structural ceiling** on this 190–199-frame corpus (E1a §1.4).
- The P1 OOD envelope was MEASURED only to |dlat| ≤ 3.0 m / |dyaw| ≤ 12°;
  `np.interp` clamps beyond, so a reported OOD ratio is a **lower bound** at long
  horizons.
- The recovery target is a **kinematic demonstration** (logged corridor in the
  offset ego frame), not a renderer rollout.
- **Not answered by E1c, by construction:** whether a different `lam_replay`,
  replay drawn from a held-out-like distribution, LoRA-style constraint, or an
  AlpaSim/CARLA loop with agents and a map changes the picture. λ is the next
  lever, not this one.
- **Not a parity change.** Mining and replay stay on parity-train
  `e438721ae894`; evaluation stays on the disjoint held-out 44. Nothing
  re-selects episodes.

---

## 7. Deliverables (into `…/incoming/2026-07-26-e1c-heldout-gated-clsft/`)

`PRE_REGISTRATION_E1C.md` (this file, staged first) · `E1C_RESULTS.md`
(verdict + **full frontier table** + guardrails + strata + lat/lon split +
evaluator self-test) · raw JSONs (`e1c_frontier_result.json`,
`e1c_selftest_result.json`, `e1c_heldout_gate.jsonl`) · training log/metrics ·
scripts (`e1c_common.py`, `e1c_clsft.py`, `e1c_eval.py`, `e1c_selftest.py`,
launchers, `hf_push_e1b_ckpt.py`).

Per the Agent Operating Standard: files are **written into the working tree and
NOT `git add`ed, committed or pushed** — the orchestrator stages.
