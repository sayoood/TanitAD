# H-REFAV1-SURFACE-1 — the paired delta and the cost-surface weight sweep

**status: IN PROGRESS — done: PHASE 1 LANDED (§1) — the decision-grade four-family paired
episode-cluster bootstrap over the three banked Thor dumps, n = 282 windows / 141 clusters,
n_boot 2000, known-value control PASS, floors bit-identical across dumps, LONGITUDINAL
distance-keeping present (n = 90 LEAD windows), STRATEGIC unavailable with reason. THE
REFUTATION SURVIVES, and it hardens: `ccos` compensated is separated WORSE than `cos` on ADE and
on every metric of three families. / next: the GT distance-keeping reference (§1d), then PHASE 2
`SPEC.md` (pre-registered) and the weight + goal-source arms.**

Arch+Inference FlyWheel · 2026-09-05 · checkpoint `refav1-b1-v72-ep3-speed/ckpt.pt` step **21,109** ·
**T1** (self-action open loop) for every planner arm; **T0** for `ol` (world-model diagnostic only).
⛔ Nothing in this document is driving performance (PI ruling 2026-09-02: a planner feeding its own
predictor is still OPEN LOOP).

## The question the PI asked

> *"our goal is to let refav1 drive and prove the performance of WM based architectures and also our
> hierarchy architecture."*

Two open items from `D-REFAV1-CCOS-EVAL`:

1. **Phase 1 — the paired delta.** The `ccos` refutation rested on PER-ARM intervals plus a
   `paired_decision_grade` block carrying only **two** metrics (`ade_m`, `speed_mae_mps`). The
   decision-grade form is the PAIRED episode-cluster bootstrap **on every family separately**.
2. **Phase 2 — the weight sweep** (`H-REFAV1-SURFACE-1`). With the goal term audible (`ccos`
   compensated), is there a weight setting at which refav1's planner beats the trivial controls on
   the four families — or is the iCEM line dead?

⚠️ **Scope statement that travels with every number below:** the world model is SOUND and separately
banked (TURN_L +0.502 [+0.383, +0.579], TURN_R −0.387 [−0.498, −0.267], separated). Everything here
is a statement about the **PLANNER over that model**, never about the world model itself.

---

## 1. PHASE 1 — the decision-grade paired delta (zero GPU)

**Instrument:** `tools/paired_delta_refav1.py` (this stream). It takes N dumps at once, so every
pair is drawn from ONE set of per-window components; components come from
`refav1_arm._components` (four_families' own geometry + the canonical trajectory labeller), never
re-derived. Estimator `taniteval.ci.paired_episode_cluster_bootstrap`, **n_boot 2000**, over the
**141 episode clusters**. ⛔ No composite: every family is reported separately.

**Inputs — the three arms as banked, pulled from Thor by content (not re-run):**

| name | dump | cost metric | weights |
|---|---|---|---|
| `cos` | `/home/nvidia/refav1_ccos/dump_cos_ext` | `cos` (shipped) | shipped (0.02, 0.05, 0.1) |
| `ccos_naive` | `/home/nvidia/refav1_ccos/dump_ccos_naive` | `ccos` | shipped (0.02, 0.05, 0.1) |
| `ccos_comp` | `/home/nvidia/refav1_ccos/dump_ccos_comp` | `ccos` | **(12.859430, 32.148575, 64.297150)** |

⚠️ **The dev box also holds an independent `ccos_comp` replicate, and it is NOT a substitute.**
`C:\Users\Admin\ccos_eval\devbox\rec_ccos_comp.json` differs from the Thor record by up to **0.044**
on 570 shared scalars (worst: `mean_min_ttc_s` 23.4644 vs 23.4251, `along_final_bias_m` −0.6940 vs
−0.6606). Cross-box agreement is ulp-level for GT and the *floors* (`D-REFAV1-CCOS-EVAL` §5) but
**the searched `cl` arm is not ulp-reproducible across boxes**, so the paired delta uses the Thor
dumps that produced the banked headline numbers. *(That is itself a MEASURED fact worth carrying: a
planner arm is a cross-box REPLICATE, not a cross-box identity.)*

### 1a. Controls — all read their known values

| control | expected | read |
|---|---|---|
| **known-value** (`cos.cl − cos.cl`) | exactly 0.0000 [0, 0] on every metric | **PASS**, 0 of 14 metrics failing |
| **same windows** | `ws`, `v0`, `g`, `clip_index` bit-exact across all three dumps | asserted, held |
| **shared floors identical across dumps** | the floors do not depend on the cost metric | `ha` **0.0**, `ha0` **0.0**, `ol` **0.0** (bit-identical); `ha0_ext` **7.63e-06** — float32 ulp, because on the `cos` dump it was added post hoc by `refav1_add_floor.py` while the `ccos` arms computed it inline |
| **lead block** | present, with its n | `b1_eval_lead_block.npz`: **PRESENT**, LEAD **90**, NO_LEAD 63, NOT_STRAIGHT 108, NO_LABEL 21 |

### 1b. ⛔ THE HEADLINE — the refutation survives, and it hardens

**`ccos` compensated − `cos`** (T1 − T1, n = 282 / 141). **Bold = the interval excludes zero.**

| family | metric | Δ (ccos_comp − cos) | verdict |
|---|---|---|---|
| **ADE** | `ade_m` | **+0.1642 [+0.1075, +0.2282]** | separated WORSE |
| | `fde_m` | **+0.4467 [+0.2984, +0.6131]** | separated WORSE |
| **LONGITUDINAL** | speed MAE (m/s) | **+0.1763 [+0.1155, +0.2425]** | separated WORSE |
| | along MAE (m) | **+0.1264 [+0.0800, +0.1763]** | separated WORSE |
| | accel MAE (m/s²) | **+0.1848 [+0.1223, +0.2531]** | separated WORSE |
| | dist-keep headway min (m) | **+0.2786 [+0.0372, +0.6463]** (n = 87) | separated — the plan keeps MORE headway |
| | dist-keep time-gap min (s) | **+0.0483 [+0.0026, +0.1119]** (n = 81) | separated — larger gap |
| | dist-keep min TTC (s) | +0.5616 [−0.0365, +1.3938] (n = 87) | straddles |
| **LATERAL** | cross-track MAE (m) | **+0.0645 [+0.0211, +0.1193]** | separated WORSE |
| | heading MAE (°) | **+0.7940 [+0.2059, +1.4960]** (n = 272) | separated WORSE |
| | yaw-rate MAE (rad/s) | **+0.0159 [+0.0056, +0.0289]** | separated WORSE |
| **TACTICAL** | traj lat correct | −0.0213 [−0.0567, +0.0071] | straddles |
| | traj lon correct | **−0.1028 [−0.1596, −0.0496]** | separated WORSE |
| **STRATEGIC** | — | **UNAVAILABLE, n = 0** | see §1e |

⇒ **Nine of the thirteen available metrics are separated worse; two straddle; the only two that
move in `ccos`'s favour are distance-keeping surrogates** — and those are not error metrics (§1d).
`D-REFAV1-CCOS-ARMS` said *"the direction is not in doubt"*. The paired test agrees and is stronger:
**the refutation stands on the decision-grade estimator, on every family separately, not only on
ADE.**

### 1c. Against the trivial controls — and the fingerprint, now proven at window level

| pair | ADE | LON speed | LAT cross | TAC lon |
|---|---|---|---|---|
| `cos − ha` | +0.0083 [−0.0574, +0.0703] | **+0.2706 [+0.2140, +0.3312]** | **−0.1569 [−0.2131, −0.1072]** | **−0.1277 [−0.1879, −0.0674]** |
| `cos − ha0` | **+0.0158 [+0.0007, +0.0315]** | **+0.0308 [+0.0049, +0.0585]** | **0.0000 [0, 0]** | −0.0035 [−0.0248, +0.0213] |
| `cos − ha0_ext` | +0.0267 [−0.0285, +0.0813] | **+0.2706 [+0.2140, +0.3312]** | **−0.1409 [−0.1844, −0.1015]** | **−0.1277 [−0.1879, −0.0674]** |
| `ccos_comp − ha` | **+0.1725 [+0.0902, +0.2585]** | **+0.4469 [+0.3707, +0.5236]** | **−0.0924 [−0.1649, −0.0170]** | **−0.2305 [−0.2980, −0.1667]** |
| `ccos_comp − ha0` | **+0.1800 [+0.1188, +0.2456]** | **+0.2071 [+0.1401, +0.2781]** | **+0.0645 [+0.0211, +0.1193]** | **−0.1064 [−0.1667, −0.0496]** |
| `ccos_comp − ha0_ext` | **+0.1909 [+0.1119, +0.2724]** | **+0.4469 [+0.3707, +0.5236]** | **−0.0764 [−0.1398, −0.0078]** | **−0.2305 [−0.2980, −0.1667]** |
| `ccos_naive − ccos_comp` | **+0.2152 [+0.1022, +0.3530]** | **−0.0361 [−0.0667, −0.0045]** | **+0.2375 [+0.1245, +0.3751]** | +0.0177 [+0.0000, +0.0390] |

⭐ **`cos − ha0` on the LATERAL family reads exactly 0.0000 with a ZERO-WIDTH interval on all three
metrics — and on `TAC_traj_lat_correct` too.** `D-REFAV1-CCOS-EVAL` called the bit-identical lateral
row "the fingerprint" from four-decimal means. The paired estimator upgrades that from a coincidence
of means to a **window-level identity**: on all 282 windows the shipped planner's lateral output IS
the constant-velocity rollout. There is nothing left to attribute to lateral planning under `cos`.

⚠️ **The compensated arm does not simply lose everywhere.** `ccos_comp − ha0_ext` is separated
**worse** on ADE, on all three core LON metrics and on TAC-lon, but separated **better** on LAT
cross-track (−0.0764) and yaw-rate (−0.0250), with heading straddling. `ha0_ext` holds the recorded
κ₀, which drifts; the compensated planner beats that particular floor laterally while losing to it
longitudinally. **No arm here beats any floor on all four families.**

⭐ **Weight compensation is a real effect, paired-confirmed.** `ccos_naive − ccos_comp` is separated
on ADE (+0.2152), FDE, all three LAT metrics and TAC-lat: compensating the implicit re-weight
recovers a measurable part of the naive flip's damage. It does not recover enough to reach any floor.

### 1d. ⚠️ How to read the distance-keeping rows (and why they are not a win)

`distance_keeping` is computed **on each arm's own predicted path against the lead track** — it is a
safety surrogate, **not an error against ground truth**. A paired delta on it says *"arm B's plan
runs X m further from the lead than arm A's"*, which is only good if the plan is otherwise right.
The straight-line `cos` plan reads **−0.7878 m headway** and **−1.8913 s min-TTC** against `ha`
(both separated): a plan that ignores the lead runs closer to it. `ccos_comp` reads *more* headway
than `cos` largely because it is slower and wanders. **The GT's own headway/TTC is the reference
that makes these rows interpretable, and it is the immediate next work item.**

### 1e. STRATEGIC — UNAVAILABLE, with its reason and n

**n = 0.** No route/goal channel exists in this eval surface. The grid is evaluated `--no-navshuf`
on the v7.2 EVAL labels, and the planner's own strategic input is the **tactical head's imagined
goal token** — model output, not a route label — so scoring the plan against it would score the
model against itself. PhysicalAI-AV ships no map, lane graph, junction annotation or route signal
(CLAUDE.md's read-set table), so no external reference exists either. **This is a WORK ITEM** (the
strategic reference must come from AlpaSim's `map.xodr` or an external corpus), **not a pass and not
an omission.**

### 1f. Verdict on `D-REFAV1-CCOS-ARMS`

**The refutation SURVIVES the paired test and is strengthened.** No amendment to the row's status is
warranted; it is annotated with the paired numbers instead. `ccos` must not become the default; the
shipped `cos` stays.

⚠️ **But `cos` is not thereby vindicated.** Against `ha` and `ha0_ext` its ADE straddles zero, its
longitudinal family is separated WORSE, its tactical-longitudinal is separated WORSE, and its
lateral family is *bit-identical to constant velocity*. The shipped planner ties the trivial
controls by **being** one of them.

**Artifacts:** `raw/paired_delta_phase1.json`, `raw/paired_delta_phase1.md`,
`tools/paired_delta_refav1.py`.

---

*(§2 — phase 2 pre-registration and arms — lands next)*
