# v7 — the architecture, the training recipe, and what the scaled run needs

**Written** 2026-08-26 · **Author** Master Mind · **Supersedes as the QUOTABLE ANSWER**
the patchwork in `PROVEN_TRAINING_SETUP.md` (which is retained as the audit trail of
how each number was arrived at, and of what was retracted).

**Why this document exists (PI, 2026-08-26):** *"The goal was to find the architecture
and training recipe for v7 and prepare the scaled training. We are coming from several
trials v1 until v1.7 and v1.8 where the model did not learn the driving environment and
extract from it the trajectory."*

⛔ **Every T0 number here is a world-model diagnostic and is NEVER driving performance.**
T1 stamps are marked where they exist. ⛔ **Per C166, every cross-arm number names its
arm in the same breath** — a column swap in these tables reverses the verdict.

---

## 1. The v1.x failure, stated as the one measurement that explains it

`MODEL_REGISTRY.md` §1.12, **TIER T1**, measured 2026-08-06:

| metric | v1.6 open | v1.6 closed | v1.7 open | v1.7 closed | CV floor |
|---|---|---|---|---|---|
| ADE (m) | 0.3398 | 0.4714 | 0.2849 | 0.4616 | 0.5352 |
| **S-curve reproduction** | **0.9785** | **0.0538** | **0.9785** | **0.0430** | 0 |

⭐ **And the control that names the cause: the HOLD-ACTION arm reproduces 0.0 % of
S-reversals.** ⇒ The counter-steer in the open-loop eval came from **the true-action
conditioning, not from vision.** v1.x did not extract the trajectory from the scene; it
**echoed the trajectory it was handed**, and scored 97.9 % for doing so.

⇒ **That is the PI's sentence, in numbers: the model did not learn the driving
environment and extract from it the trajectory.**

---

## 2. ⭐⭐⭐ The campaign's central finding is the SAME FACT from the other side

Three results, each MEASURED, compose into one explanation:

| # | finding | evidence |
|---|---|---|
| **A** | **Our `action` channel IS the ego's realised motion in other units.** `v·tan(steer)/L` reproduces the measured yaw-rate at **r = 0.9988** | E-DEC-57, closed form |
| **B** | **In observational driving data the action carries NO information about the future scene.** Against a positive control at **t 8.5–14.3**, the action's marginal is **+0.0755 (t 0.64) / −0.0179 (t 0.14) / +0.1083 (t 1.43)** — null | E-DEC-48b, `n_agents` / `occ_center` / `n_free_cols` |
| **C** | Ten objective terms (O1, O2, O3, O7–O11, O13, PSG) failed, **as a class, not one at a time** | E-DEC-52 and the ledger in `PROVEN_TRAINING_SETUP.md` §6 |

⭐⭐⭐ **A + the v1.x echo are the same event.** Feeding the model `atan(L·κ)` — a
restatement of the realised trajectory — and then asking it to predict that trajectory
is **asking it to copy its own input.** 97.9 % open-loop is what copying looks like;
4.3 % closed-loop is what remains when the copy is removed.

⭐⭐ **B explains why ten objectives died.** Every one of them asked the **action** to
move the **scene latent**, and B measures that this information is not in the data. The
arrow in observational driving runs **SCENE → ACTION**, not ACTION → SCENE: other
traffic evolves largely independently of what we do.

⇒ ⭐⭐⭐ **THE ARCHITECTURAL CONSEQUENCE, AND IT IS THE PI'S THESIS.** *"If the lead
decelerates, the ego must react"* is **not a statement the world model should encode —
it is one the PLANNER should.** A monolithic action-conditioned world model failed for a
reason that is structural, and the structure that fixes it is **the hierarchy**: the WM
predicts scene evolution (largely ego-independent), and reasoning/planning above it owns
the reaction. **The campaign did not merely fail to find an action-conditioning trick —
it produced the argument for why the hierarchy is necessary.**

⚠️ **What this does NOT establish, and must not be quoted as establishing** (PI,
2026-08-26 — the closure was withdrawn): that action-conditioning is closed. Finding **A**
means **a genuine command channel has never been tested**. The PI's directive stands:
**use the AV dataset's ego data — `[yaw_rate, a_long, v]` as measured state** — which is
implemented as `--cond-param omega_accel_v` and has **never been run as an arm**.

---

## 3. What the campaign PROVED — screened, with the arm named

| axis | status | the number, with its arm |
|---|---|---|
| **Collapse** | ✅ **SOLVED** | participation **3.80/3.62 → 25.58/26.96** at parity scale (`rdw8p30k`); 5 arms clear the C149 constant-predictor floor. Two-term+k1 turns rise-then-collapse into rise-then-**plateau** (H-RANK-11: all-six 8.96@16k → 5.55@20k, −38 %; two-term peak 6.748 @6–12k, final 6.623 @24–30k) |
| **Representation** | ✅ **SOLVED, held-out** | **`splitp30k`** `n_agents` **+0.3881** > frozen DINOv3 **+0.2754** > constant **0.0000**; corridor `n_free_cols` **+0.2869**, `occ_left` +0.1325 (E-DEC-29, n=2,400, d_eff=128) |
| **Prediction** | ⛔ **LOCATED, OPEN** | Δz is **64 % drift** (predictable from `z_t` alone, t 65.6); 25 of 2048 directions carry 90 %; the drift-removed residual carries the action in **0 of 8 arms** |
| **Physics / action-conditioning** | 🔶 **OPEN** (closure withdrawn by the PI) | see §2 |

### ⚠️ Three caveats that travel with "representation is solved"

1. **It is target-specific.** `splitp30k` beats DINOv3 on `n_agents` but **LOSES** on
   free space (+0.2869 vs **+0.3701**) and side occupancy (+0.1325 vs **+0.2735**).
2. **No fine localisation.** All eight per-azimuth-column occupancies sit at or below
   zero. E-DEC-29's own wording: *the representation knows roughly how crowded the scene
   is, how much room is ahead, and which side things are on — and does not know WHERE
   any individual object is.*
3. ⛔ **We have the RESULT without the EXPLANATION.** The published lever —
   *"initialisation, 8-arm clean separation"* — is **retracted (C164)**: `postrain30k`
   and `splitp30k` share the same `distill_init.pt` and sit **0.47 apart** on drift, so
   the distilled/scratch grouping was confounded.

---

## 4. ⛔ THE ONE BLOCKER FOR v7 — the dissociation

**No single arm carries the environment AND predicts it.** This, not collapse, is what
stands between here and a scaled run.

| arm | encoder | environment content | does the predictor add? |
|---|---|---|---|
| **`splitp30k`** | frozen distilled | **+0.3864** from `z_t` — the richest measured | ⛔ **NOTHING**: −0.0008 / −0.0320 / −0.0158 at k=1/3/6, **all negative** (t −3.69 / −5.62 / −6.26) |
| **`rdw8p30k`** | trained | +0.0777 | ✅ **substantially** (predictor cos **0.6224**, z 30.55 — 3.3× the best previously measured anywhere in the programme) |

⇒ **Mandate (2) — environment in BOTH encoder and predictor — is satisfied by no arm.**
E-DEC-20's dissociation, in its sharpest form.

⭐ **The leading hypothesis (E-DEC-61), and it is testable:** **O5 manufactures drift when
the encoder is trainable.** Six of seven trained-encoder arms converge to drift
**0.616–0.679** across 7.5k–30k steps and *both* initialisations; the only two arms
outside that band are the **frozen** ones (0.199, 0.337). The load-bearing evidence is a
**within-recipe trajectory**: `postrain10k` **0.359** → `postrain30k` **0.669** — same
recipe, same init, only training length differing. Mechanism: the cheapest way to satisfy
`ẑ_{t+k} ≈ z_{t+k}` is an **easier target**, not a better prediction.

⚠️ **Run-to-run variance is now known** — `postrain30k` 0.669 vs `postrain30k_seed1`
0.679, **~1.5 %** — so a 0.47 gap is emphatically not seed noise. This is the programme's
first variance estimate and every single-seed comparison should be read against it.

---

## 5. The v7 recipe

### 5.1 Settled — carry these into the scaled run

```
--stage S-W                          # scene prediction from the scene
--o5-form l1  --w-o5 1.0             # the two-term core
--w-o6 0.1                           # SIGReg / isotropy
--o5-k 8
--sigreg-subspaces 32 --sigreg-slices 512
--spectrum-accum <ceiling >= d_op>   # rank gate; cheap, retained
--cond-param omega_accel_v           # ADOPTED 2026-08-27 (D2/E-DEC-65): the PI's
                                     # measured-state channel; no-worse drift,
                                     # likely-better prediction (single-seed hedge)
--w-o14 1.0 --o14-mode fut --o14-k 4 # R2 ADOPTED 2026-08-28 (E-DEC-67, ABSORBED):
                                     # the future-observation aux removes the
                                     # E-DEC-63 pixel-marginal (+0.0096 t5.11 ->
                                     # -0.0047 t-3.09); gates held (drift ~band,
                                     # nrmse +2.1%). ⚠️ anti-DISPLACEMENT only —
                                     # drift itself UNCHANGED (0.671 ~ 0.669); the
                                     # drift attractor stays the open front (P0).
                                     # corpus: physicalai-train-e438721ae894 (PARITY)
--o5-target ema                      # ADOPTED 2026-08-29 by the PI (E-DEC-69): the
                                     # largest prediction gain ever measured on the
                                     # trainable line (cos +24.5%, nrmse -9.9%,
                                     # absorption held) at a +3.6% drift cost,
                                     # knowingly accepted.
                                     # ⭐ GATE CLOSED 2026-08-30: the 30k tau-RAMP
                                     # arm read NEUTRAL -- drift 0.6936 vs 0.6952,
                                     # nrmse 0.7408 vs 0.7466, cos 0.7513 vs 0.7524,
                                     # absorption INSIDE_NULL both ways. Every delta
                                     # is ~20x SMALLER than the only seed spread this
                                     # recipe has produced (0.036 at 2k).
                                     # => tau stays FIXED at 0.996; --ema-decay-ramp
                                     # is DROPPED. "Fixed tau is known-suboptimal" is
                                     # true in general and does NOT bind at 30k, where
                                     # tau=0.996 already sits inside the published
                                     # band. THIS WAS THE LAST OPEN FLAG IN THE RECIPE.
```

⛔ **Do NOT add O1, O2, O3, O7, O8, O9, O10, O11, O13 or PSG.** Ten measured failures;
E-DEC-48b explains them as a class. O13's matched pair degraded prediction by **+192.4 %**.

⛔ **`--init-from` is NOT a lever (C164).** Harmless to keep; it does not buy the
separation it was credited with.

### 5.2 ⛔ The two decisions the scaled run is waiting on

| # | decision | the experiment | state |
|---|---|---|---|
| **D1** | **encoder trainable or frozen** | `postrain30k_freeze` — the crossed cell | ⛔ **ANSWERED 2026-08-27: DEGENERATE — drift 0.3905 ✅ but nrmse 0.9301 vs 0.8115 (+14.6 %) ⛔. The v7 encoder stays TRAINABLE**; anti-drift moves to the representational lever (E-DEC-63) |
| **D2** | **the conditioning channel** | `--cond-param omega_accel_v` — the AV ego data as measured state, per the PI | ⭐ **ANSWERED 2026-08-27 (MIXED, prereg OUTCOME): marginal null (t −1.29), drift unchanged, held-out nrmse IMPROVED 14.8 % (~4× the 3.5 % seed band, single seed) — ADOPTED for v7r** |

⭐ **D1 is the v7 decision experiment, not a side quest.** It asks precisely whether
content and prediction can coexist: freezing is the only condition under which drift has
ever been low, and E-DEC-20c records a frozen encoder driving the predictor **~5×
miscalibrated** — so **DEGENERATE is a live branch, pre-registered as such.**

⚠️ **If D1 comes back degenerate**, the dissociation is not solved by freezing and the
next lever is a **partial/staged unfreeze** or a **content-preserving auxiliary that is
not self-generated** — the one property every failed objective shared (E-DEC-7).

---

## 6. Preparing the scaled training — what must be true before the big run

| # | prerequisite | state |
|---|---|---|
| 1 | **D1 answered** (encoder policy) | ▶ running |
| 2 | **D2 answered** (conditioning channel) | ⏸ queued |
| 3 | **A T1 number on a v7 arm** — the programme has **never** evaluated a v7 arm at T1 | 🔶 chain armed on Thor, parity corpus, both arms |
| 4 | **Four metric families reporting** | 🔶 3 of 4: longitudinal anti-echo ✅, lateral ✅, tactical ✅; **distance-keeping** needs `obstacle.offline` staged (engineering now done — the straight-driving gate landed `b6e98043a`); **strategic** needs map-derived option sets |
| 5 | **Tactical + strategic training labels** | ⛔ **Data FlyWheel work (PI's Stage B)** — not started |
| 6 | **REF-C / REF-D comparison harness** | REF-C trained (`refc-diffusion-xl-30k`); **REF-D does not exist**; hierarchy-traversing eval does not exist — **PI's Stage D** |

⭐ **The staged plan (PI, 2026-08-26):** show **operative** capability first; **in
parallel** the Data FlyWheel generates the tactical/strategic data; then train the upper
levels; then show hierarchy dominance over REF-C/REF-D. **Items 1–4 are Stage A. Item 5
is Stage B and can start now — it is not gated on D1.**

---

## 7. ⚠️ The instrument discipline this campaign produced — and why it is load-bearing

Nine claims were retracted in ~24 hours (C156–C166). **Every retraction came from a
control, and most controls were run after the claim was published.** The durable output
may be the discipline rather than any single result:

- **A MEASURED null** — `taniteval/taniteval/null_calibration.py`, 104 draws: the bar for
  this panel family is **|t| ≈ 2.9, not 2.0.** Several older cells at t 2–3 are not
  separable from noise under it.
- **p floored at 1/N** — the smallest resolvable p from N draws.
- **The crossed-cell rule** — a grouped comparison is a lever **only if the groups are
  matched on everything but the label** (C164).
- **Every panel carries** a constant control reading exactly +0.0000, a time-shuffled
  control, and a **positive control that must read a known value** (C159).
- **C166** — a cross-arm claim names its **ARM** beside its number.

⛔ **And the standing limit on everything above: it is all T0.** No v7 arm has been
evaluated at T1, so **no claim about driving is available from this campaign at all.**
Item 3 in §6 is therefore not a formality — it is the first evidence that would bear on
the PI's actual question.


---

## 8. ⭐⭐⭐ HOW EFFECTIVENESS IS PROVEN — the ladder, and the control that would have caught v1.x

**The PI's question:** *"how can we prove the effectiveness of the taken measures?"*

⛔ **Start from why the last attempt's proof failed.** v1.7 cleared every gate it was
measured against and still did not drive: **ADE −16 % vs v1.6, CI-separated, every
non-regression gate green** — and at T1 its S-curve reproduction was **0.0430**. The
proof was not wrong, it was **measured in the wrong tier**. ⇒ **A v7 effectiveness
claim that is not T1 with the echo control is not a claim about driving.**

⭐⭐ **The programme already OWNS the instrument that exposes this: the HOLD-ACTION
control**, which reproduces **0.0 %** of S-reversals when the apparent skill is an echo
of the supplied action. It is the single sharpest test we have, and it is free.

### The ladder — an arm earns the next level only by clearing the previous

| level | what it proves | criterion | ⛔ deliberate-regression arm (must FAIL) |
|---|---|---|---|
| **L0** | the **rig** is valid | constant control reads the no-information value **EXACTLY**; the positive control reads its **KNOWN** value; a raw-pixel floor is present; `n` and `d` printed | an input with the target provably absent |
| **L1** | no collapse | **participation (σ²), val-side, ≥ 8.56** (frozen DINOv3, MEASURED on our frames). ⛔ Never `effective_rank` (C132: they disagree up to 141×) | — ⛔ **rank is NECESSARY, NOT SUFFICIENT (C131):** v1-era had the highest rank ever measured here and no environment interpretation |
| **L2** | the latent **carries** the environment | beats the raw-pixel floor **AND** the constant control **AND** frozen DINOv3 — paired — on `n_agents`, `n_free_cols`, `occ_{l,c,r}` | **`frzrand`** (frozen RANDOM ViT). It spans essentially the pixel subspace and **must not clear L2**; it already reads `lead_gap_m` **equal to the pixel floor to four decimals** |
| **L3** | ⭐ the predictor **ADDS** — *the dissociation gate, and the actual v7 bar* | `zhat` beats `z_t` on the same targets, paired, **\|t\| ≥ 2.9** against the measured null | **`splitp30k`** — a known predictor-dead arm (t −3.69 / −5.62 / −6.26). If it clears L3, the gate is broken |
| **L4** | ⭐⭐⭐ it **DRIVES**, and **not by echo** | **T1.** S-curve reproduction must **exceed the HOLD-ACTION control**; ADE below the **CV floor 0.5352**; all four metric families, paired episode-cluster bootstrap | ⭐ **v1.7 ITSELF** — banked at 0.9785 open / **0.0430 closed**, hold-action **0.0 %**. A harness that does not reproduce that collapse cannot be trusted to detect it in v7 |
| **L5** | the **hierarchy** earns its place | separated from **REF-C** and from the **frozen-DINOv3 WM** (the PI's fallback), paired episode-cluster bootstrap | a **flattened single-level** variant of the same arm, same corpus, same steps |

⭐ **Why v1.7 as the L4 regression arm is the strongest part of this design.** It costs
nothing (the numbers are banked), it is the *exact* failure we are trying not to repeat,
and it makes the harness falsifiable **before** it is used to bless v7. Per the
validation standard: *if the gate does not FAIL the regression arm, a PASS on the fixed
arm means nothing.*

### The four rules that make each level admissible

1. ⛔ **ONE VARIABLE, and diff the LAUNCH COMMANDS, not the intent.** C164 is the cost of
   skipping this: a grouped comparison is a lever **only if the groups are matched on
   everything but the label**, and `--init-from` failed exactly that test.
2. ⛔ **The bar is the MEASURED null — \|t\| ≈ 2.9, not 2.0** (104 draws,
   `taniteval/taniteval/null_calibration.py`), with **p floored at 1/N**.
3. ⛔ **Every hyper-parameter fitted on FIT only.** λ selected on the scored split picks
   maximal regularisation and reads **exactly the floor**, which then beats every noisy
   positive estimate — measured, four times in one afternoon.
4. ⛔ **Name the ARM beside the number** (C166). A column swap in a six-column table
   silently reverses the verdict and nothing in the number itself flags it.

### What the ladder does NOT prove

⚠️ **L1–L3 are all T0 and none of them is evidence about driving.** v1.7 is the proof of
that: it is possible to clear open-loop gates convincingly and reproduce 4 % of the
manoeuvres that matter. **L4 is the first level at which the word "effective" is
admissible**, and no v7 arm has ever reached it.
