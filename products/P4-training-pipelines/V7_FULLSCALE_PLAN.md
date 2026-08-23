# v7 FULL-SCALE TRAINING PLAN

`Owner: TanitAD_TrainingFlyWheel. Created 2026-08-23. PRIORITY 3 of the PI
mandate (2026-08-22): "prepare the FULL-SCALE training plan — scaled encoder
under the same objective, staged gates per /TanitAD_ValidateAIDesign, hardware
plan, checkpoint/snapshot policy (learned from the v6F era)."`

⛔ **BOTH BRANCHES ARE DRAFTED HERE, BEFORE THE DECIDING READOUT EXISTS.** The
champ30k decodability measurement (P4-3) SELECTS a branch; it does not start the
thinking. This is the programme's own both-outcomes discipline applied to a plan
rather than to an experiment.

---

## 0. THE DECIDING READOUT

| | |
|---|---|
| **experiment** | E-TRUNK-3 battery on the champ30k trunk (P4-3) |
| **runner** | `champ30k_decodability.py` (both outcomes in its docstring, written before the number existed) |
| **primary metric** | `lead_gap_m` R², episode-disjoint 5-fold, episode-cluster bootstrap |
| **controls** | C-EGO (identity map must read ~1.0), C-PIXEL raw-pixel floor, DINOv3 reference |
| **tier** | T0-DIAGNOSTIC — a representation property, **never** a driving claim |

**What we already know, and it is the reason this gate exists.** champ30k raised
VAL participation 4.052 → **6.499** (H-RANK-12). E-TRUNK-3 established on v6F that
participation rises (3.29 → 6.94) *while* `lead_gap_m` R² stays pinned below zero
(−0.009 → −0.018). Its own conclusion:

> an anti-collapse regulariser guarantees the dimensions are **used**, never that
> they are **informative**; a variance/isotropy constraint can be satisfied by
> noise.

⇒ **The participation gain is not, by itself, evidence of anything worth
scaling.** Scaling an uninformative trunk buys a bigger uninformative trunk.

---

## 0b. ⭐ THE READOUT LANDED — 2026-08-23. **BRANCH A, CONDITIONALLY.**

`raw/champ30k_decodability.json` · controls reproduced exactly (C-EGO 0.334,
C-PIXEL 0.002, identity map 1.0000) · falsifier 1 OK (5,617 keys match).

| rung | `lead_gap_m` R² | |
|---|---|---|
| v6F @2000 | −0.0093 [−0.0948, +0.0537] | below zero |
| v6F @16000 | −0.0147 [−0.0421, +0.0019] | below zero |
| v6F @18000 | −0.0190 [−0.0502, +0.0010] | below zero |
| v6F @20000 | −0.0176 [−0.0454, +0.0001] | below zero |
| **champ30k @30000** | **+0.1273 [0.0866, 0.1683]** | ⭐ **CI wholly above zero** |
| C-PIXEL floor | +0.0020 [−0.0174, +0.0156] | — |
| DINOv3 reference | +0.3510 [0.2863, 0.4222] | ceiling |

**Under identical frames, folds and probe, champ30k separates where every v6F
rung did not.** WORLD B does not hold for the two-term+k1 objective.

### ⛔ Three constraints that keep this from being a licence to scale

1. **IN-SAMPLE.** All 130 probe episodes are members of the 2,400-episode
   training cache — **100 % overlap** (H-DEC-3). The 5-fold holds episodes out of
   the *ridge fit*, not out of *trunk training*. The comparison survives (every
   rung shares the flaw); the absolute number does not. ⇒ **the clean held-out
   test is a prerequisite to Scale-2, not optional.**
2. **NARROW, AND LONGITUDINAL.** Separated: `lead_gap_m`, `ego_speed` (+0.1823).
   Not separated: `nearest_any_m`, `n_agents_log`, `left_occupied`, `vru_ahead`,
   `occluded_frac`. And `ego_yawrate` reads **−0.4925** — actively anti-predictive.
   ⇒ champ30k learned depth/longitudinal structure and **no lateral structure**.
3. **T0, NOT DRIVING.** E-TRUNK-3 §5 already retracted the step from decodability
   to an architectural prescription: REF-A had *better* decodability and drove
   **2.62 m worse**. This licenses scaling the *objective*; it forecasts nothing
   about ADE.

### What this changes in the ladder below
- **Scale-1 proceeds** (§A2) — the objective demonstrably induces perception.
- ⭐ **Add a LATERAL gate.** The measured deficit is lateral/occupancy, so
  Scale-1's success criterion is `lead_gap_m` retained **AND** movement on
  `left/right_occupied` or `ego_yawrate`. Scaling what already works while the
  lateral channel stays at chance would buy a better rangefinder, not a driver.
- **Gate Scale-2 on the held-out re-measurement** (constraint 1).

---

## 1. BRANCH A — champ30k SEPARATES (lead_gap_m above 0 and above the pixel floor)

*Meaning:* the first v6/v7 checkpoint to acquire environment information. Two-term
+ k1 is a genuine representational advance, and the objective is worth scaling.

### A1. What scales, and what does not
champ30k is **19.34 M of a 300 M budget** — 15.5× headroom. The parameters are
*not* where one would expect:

| group | params | share |
|---|---|---|
| encoder | **0.97 M** | 5 % |
| predictor_op | 7.57 M | 39 % |
| layer_tac | 4.88 M | 25 % |
| layer_str | 3.56 M | 18 % |
| aux | 1.65 M | 9 % |
| planner | 0.69 M | 4 % |
| readout | 0.017 M | 0.1 % |
| interp | 0 | 0 % (no stage may train it — `LADDER_UNTRAINED_GROUPS`) |

⭐ **The encoder is 5 % of a model whose deficit is perceptual.** The scaling
axis is therefore the ENCODER first (`--enc-dim 128 → 384+`, `--enc-depth 3 → 8+`,
the trainer's own defaults), not the predictor.

### A2. Staged ladder (per `/TanitAD_ValidateAIDesign`)
1. **Scale-1 — ⭐ LAUNCHED 2026-08-23, Thor `~/v7tiny/scale1`.**
   `--enc-dim 256 --enc-depth 6 --enc-heads 8`, everything else identical.
   ⚠️ **MEASURED total 23.87 M, not the ≈60 M I estimated** — only the encoder
   scales, 0.97 → **5.49 M** (5.7×); the predictor/layers are unchanged. Trainable
   14.74 M / frozen 9.13 M. Parity VERIFIED at launch (2,400 clips, clip sha256
   `e61a04553df5…`, skip-hash `f09e44db`). Pre-registered with both outcomes in
   `PREREG_E_P4_SCALE1.md`.
   Gate: participation ≥ champ30k's **and** `lead_gap_m` retained **and** ⭐ the
   LATERAL criterion (below). ⛔ If longitudinal rises and lateral stays at
   chance, **STOP scaling this objective** — that is Branch B arriving late, and
   it means the objective, not capacity, is the constraint.
2. **Scale-2 (≈150 M)**: only if Scale-1 moves BOTH.
3. **Scale-3 (≤300 M)**: budget ceiling; `assert_param_budget` enforces it
   (`v6.py:5668-5674`).
4. Only then S-T → S-S → S-J, each gated by `stage_gate.json`.

### A3. Compute
Measured: champ30k = **0.4925 s/step** (14,774.7 s / 30,000 steps) at
`--batch 4 --window 6` on Thor. Scale-1 is ≈3× the params; **do not assume 3×
the time** — Thor is not GPU-bound at this size and the loop has **zero
DataLoader workers** (SPEC §4.2), so throughput may be data-bound and barely
move. ⇒ **P4-6 (`data_wait_s` vs `compute_s`) is a PREREQUISITE**, not a
parallel nicety: without it every scaling estimate is a guess.

---

## 2. BRANCH B — champ30k DOES NOT SEPARATE (WORLD B holds for two-term+k1)

*Meaning:* the participation lever is **exhausted**. Every registered rank lever
has now been tried:

| lever | claim | verdict |
|---|---|---|
| SIGReg weight | H-RANK-3 | REFUTED (1000× sweep flat) |
| term count | H-RANK-4 | SUPPORTED for rank only |
| estimator conditioning | H-RANK-5 | REFUTED both halves |
| capacity ratio | H-RANK-6 | REFUTED |
| k-step rollout k=1 | H-RANK-7 | SUPPORTED for rank only |
| **more training (6k→30k)** | H-RANK-10/12 | **rank plateaus; decodability = this readout** |
| n_stack=1 | H-RANK-8 | ⛔ blocked (cache rebuild, DataFlyWheel) |

⇒ **Do not scale this objective.** Scaling is the most expensive way to learn
what a 19 M probe already told us.

### B1. What happens instead
1. ⭐ **E-ENC-3WAY becomes the whole question** (`PREREG_E_ENC_3WAY.md`).
   Arm **C** (`dino-frozen`) runs first — no encoder training, cheapest, and
   simultaneously the clean re-test of REF-A whose 2.1322 carried two
   residual-init defects.
   ⛔ **Gate release is a Master-Mind/PI ruling** — see §4.
2. ⚠️ **And C is not a free win.** E-TRUNK-3 §5 already retracted "therefore
   freeze the encoder": REF-A used a frozen DINOv2 trunk with far better
   decodability and drove **2.62 m WORSE** (paired [2.0945, 3.2570]).
   **Decodability does not translate into driving.** Arm C must clear a DRIVING
   gate, not a probe.
3. The open question E-TRUNK-3 explicitly left: WORLD B is equally consistent
   with *"this objective never induces perception"* and *"2,376 episodes is too
   little to learn perception from scratch"*. These separate **only** by training
   a different objective on the same corpus — which is what the 3-way does.

---

## 3. COMMON TO BOTH BRANCHES

### 3.1 Checkpoint / snapshot policy (learned from the v6F era)
| rule | why |
|---|---|
| full `ckpt.pt` at `--save-every` is the **resume anchor** | `load_resume` does `load_state_dict(..., strict=True)` (`:4722`) and **refuses** an fp16 snapshot (`:4716-4721`) |
| fp16 snapshots every 2k as `--init-from` artifacts + travel copies | the full ckpt is the artifact that does not travel; `ops/ckpt_fp16_snapshot.py` |
| **done-marker in the same turn** the run finishes | a run without `summary.json{done:true}` is RESURRECTED FOREVER by its supervisor (measured: 2 days) |
| off-box backup of the final ckpt | HF relay ≈118 MB/s from a pod; ⛔ HF quota is a hard ceiling — check before pushing |
| ⛔ never write logs to a quota-bound volume | logs to `/workspace` were SWALLOWED on death; write to `/tmp` and copy |

### 3.2 Hardware plan
| role | machine | note |
|---|---|---|
| training | **Thor** | the only sustained-training machine we hold. ⛔ never add load while it trains |
| probes / eval / cache builds | **dev box RTX 4060** | verified this session: built the champ30k decodability cache at 7.8 fr/s while Thor stayed idle |
| label lab | Colab | ⛔ **no training entry point exists** (`grep -rn train_v6_staged colab/` → nothing) |
| burst | HF Pro | ⛔ METERED — asks first, every time |

⛔ **Code reaches Thor by md5-verified file-ship, never git** (no credentials).
Ship every file the run imports, not just the entry point.

### 3.3 Preflight gates that must be green before any full-scale launch
`--spectrum-accum` sized so **O6 CAN rule** (P4-1 — today the check only prints) ·
residual-init banner reads 0.001 · `assert_param_budget` under 300 M ·
X3 isolation strict pass · md5 code-freshness on Thor · `--gate-probes` file
exists (`:5879`) · **run provenance recorded** (P4-4).

---

### 3.4 ⭐ TWO CONFIG DECISIONS ARE NOW SETTLED BY MEASUREMENT (2026-08-23)

Both are independent of which OBJECTIVE variant wins, so **v7 can be finalised on
them now** while the objective is still being decided.

| decision | value | evidence |
|---|---|---|
| **readout azimuth** | `--readout-grid 4 --readout-grid-w 8 --readout-dim 64` (= 2048, **the incumbent d_op — no extra latent dimension**) | E-DEC-2, LOEO paired: speed **+0.0374 → +0.2830** (t 5.17, **12/12** episodes), `d_ego` **+0.1532 → +0.3601** (t 6.12, 12/12) |
| **rollout depth** | `--o5-k 4` minimum, `8` measured better; ⛔ **NEVER 1** | H-PROOF-7/7b, rolled read-out: cos at j=6 **0.0686 (k=1) → 0.1987 (k=4) → 0.2468 (k=8)**, monotone, advantage GROWING with horizon (2.80× at j=1 → **3.60×** at j=6) |

⛔ **`--o5-k 1` IS A CORRECTNESS BUG, NOT A TUNING CHOICE (C139).** `rollout_transitions`
builds the rollout from `t[1]` — the h=1 head ONLY — rolled `k_roll` times. At
`k_roll = 1`, and with O1/O2/O3 off, **the h=2 / h=4 heads receive no gradient at
all**: their weight norms read **0.02612 / 0.02614 in every arm at 2k steps and at
30k**, identical to five significant figures — the untouched `1e-3` init. Any
evaluation that QUERIES those heads measures an untrained layer.
⚠️ v6F is CLEAN (`o5_k = 60`, all six terms on ⇒ head norms 24.27 / 26.11 / 26.12);
the condition was **introduced by the two-term simplification**, which removed the
only consumers of the h≥2 heads.

⚠️ **Two negative results that bound the readout decision:**
* `rdw20` (20 azimuth bins) is **WORSE than `rdw8` when TRAINED**, although a
  frozen-encoder re-pooling curve put the optimum at 20. The curve was a pointer,
  not a design — do not ship a geometry chosen on a frozen-encoder sweep.
* Widening **reverses on ENVIRONMENT targets**: it helps `n_agents` but HURTS
  `lead_gap_m` and bearing (a scalar depending on one region ahead is diluted by
  more bins). `rdw8` is chosen on EGO evidence; the environment cost is real and
  must be re-measured at full scale.

⚠️ **Scale caveat (H-SCALE-2):** tiny-arm screening is valid for ARCHITECTURE
decisions and invalid for CAPABILITY levels — the readout ranking held across a
4× encoder / 15× steps / 18× data change, while absolute rank moved 3.80 → 8.54
and predictor cos 0.054 → 0.609. **Never quote a tiny-arm number as v7's expected
capability.**

### 3.5 Read-out protocol for any v7 arm (non-negotiable)

⛔ A read-out at h≥2 must **ROLL** the h=1 head, never query `predictor(...)[2]`.
Predictor statistic = **mean-centred cos vs a ≥100-draw permutation null**, report
`z` (C137 retired divergence-over-movement: its denominator was an arm property
spanning 468×). Decodability = **leave-one-episode-out PAIRED** (the episode
bootstrap returns ±1.6 on a 0.2 effect at 12 clusters). Every panel carries a
CONSTANT control reading exactly 0.0000 and a RAW-PIXEL floor, and reports **EGO
AND ENVIRONMENT** — ego alone hid a total absence of scene content for this entire
campaign.

---

## 4. ⛔ THE ONE DECISION I AM NOT TAKING

`PREREG_E_ENC_3WAY.md` §0 says the 3-way *"does not start until an arm
demonstrably clears collapse."*

- champ30k **arrests** collapse — plateau, no fall (H-RANK-11 SUPPORTED).
- champ30k **does not clear** the floor — 6.499 < 8.56 (H-RANK-13 SUPPORTED).
- ⚠️ and the floor itself is contested — 8.56 vs the E-TRUNK-3 reference 40.77
  (H-RANK-16), so "clears the floor" is not currently a well-defined test.

Whether *"collapse arrested + more-training proven not to be the lever"*
satisfies a gate written as *"collapse cleared"* is a **design ruling for the
Master Mind and the PI**. I have prepared both branches so the ruling costs one
message, not one week. **Recommendation:** resolve H-RANK-16 (P4-2) first — a
gate whose floor is ambiguous cannot license or refuse anything.
