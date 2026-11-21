# PRE-REGISTRATION — E-EXP-2: is the per-window λ\* FINDABLE from REF-C's own latents?

**Written and content-pinned BEFORE any statistic below was computed.** Stream: arch-inf.
Date: 2026-08-04 (Europe/Berlin). GPU cost: **one inference-only forward pass on Thor** (no
training, no optimiser, no checkpoint written). All fitting is dev-box CPU.

Predecessor: `…/Research/2026-08-04-planner-hierarchy-sota/PREREG_E_EXP1.md`
(blob `000ed1cdc45da59c7b4ca406f921ba18c024ce4e`, re-verified 2026-08-04).

---

## 1. What E-EXP-1 left open, stated exactly

E-EXP-1 measured a **CEILING**, with an oracle λ, on the already-selected trajectory
(881 windows / 40 val episodes, paired episode-cluster bootstrap):

| | base (128 anchors) | XL (256 anchors) |
|---|---|---|
| `ade_sel` → per-window-oracle-λ ceiling | 0.4728 → **0.3138** | 0.4714 → **0.3064** |
| paired Δ (the ceiling) | **+0.1590** [+0.1196, +0.2006] sep | **+0.1651** [+0.1244, +0.2049] sep |
| matched-DoF **lateral** control (δ rotation) | +0.0161 | +0.0141 |
| share of that arm's own oracle gap | 56.5 % | 53.7 % |

**REACHABILITY is answered. FINDABILITY is open**, and the two cheapest feature sets have
already FAILED (E-EXP-1b, exploratory, LOEO):

- a **single global λ** fitted leave-one-episode-out recovers **NOTHING** (λ lands at 0.999–1.001);
- a **`v0`-decile lookup** for λ is **separably WORSE than doing nothing** on both arms
  (base −0.0039 [−0.0097, −0.0006] sep; XL −0.0021 [−0.0045, −0.0006] sep);
- a `v0`×along-distance decile lookup is worse and not separated.

⇒ **This experiment asks the third and most expensive question: does REF-C's OWN LATENT carry
λ\*, above the `v0` echo?** It is a PROBE. **⛔ No training is launched by this experiment under
any outcome.**

---

## 2. ⭐ THE ECHO FLOOR — why a `v0`-only control is mandatory, not decorative

**MEASURED (ours, E-EXP-1 surface):** `corr(2 s along-track endpoint, v0) = 0.9973`.
A radial scale λ acts almost entirely on the along-track endpoint, so **`v0` alone very nearly
determines the quantity λ is correcting**. Any predictor that reads a latent which encodes speed
would therefore score above zero *without carrying one bit about the residual*.

This is the same defect family as:
- flagship v1's route head being an **exact bijection of the nav it is fed** (369/369 and 81/81)
  and scoring **1.0000** — an echo of its own input read as skill;
- the sitclf leak (labels derived from ego dynamics, ego supplied at inference);
- the REF-A I-JEPA val leak (~80 % of val inside train).

⛔ **A latent arm that does not separably beat the `v0` arm is a speed echo and is reported as
one.** The `v0` arm is built to be STRONG, not a straw man: it gets the same model class, the same
nested selection, and a polynomial expansion (§5, `F1`), because a weak control manufactures a
positive result.

---

## 3. Data, fixed in advance — nothing may be added after a result is seen

**Banked, already on disk (unchanged, read-only):**
- `…/incoming/2026-08-03-s1-climbout/raw/fan_emitted_refc-base-30k.pt` — 128 anchors
- `…/incoming/2026-08-03-s1-climbout/raw/fan_emitted_refc-xl-30k.pt` — 256 anchors
- Both: 881 canonical windows / 40 val episodes, 4 waypoints at `wp_steps=(5,10,15,20)`
  (**0.5 s grid**), step 29999, `nav_mode=follow_constant`.
- Keys read: `fan`, `gt`, `sel`, `eid`, `v0`, `logits`, `emitted_logits`, `prefinal_logits`.

**NEW — the one thing this experiment produces on a GPU** (`stack/scripts/refc_dump_latents.py`,
Thor, `tanitad-edge`, inference only):

| tensor | shape | what it is |
|---|---|---|
| `pooled` | [881, F] | ResNet encoder pooled feature of the **last frame**. F = 8·base_width = **704** (base) / **992** (XL). VISION ONLY. |
| `pooled_seq` | [881, 8, F] | the same, for all 8 frames of the window. VISION ONLY, temporal. |
| `ctx` | [881, d_ctx] | `StrategicCtx` GRU state over `pooled_seq` — REF-C's own strategic summary. VISION ONLY. |
| `measurement` | [881, d_m] | `m = measurement([v0/10, nav one-hot, keep, known])` — the EGO+NAV embedding. **This is the echo path by construction** and is labelled as such everywhere. |

⛔ **This list is CLOSED.** No further tensor may be dumped or added to a feature set after any
statistic in §7 has been read. If a later question needs another tensor, it is a new
pre-registration.

⛔ **No episode re-selection of any kind.** The parity invariant is untouched: the dumper walks the
identical `list_val_episodes(val, 40)` / `WINDOW=8, STRIDE=8` grid as the banked emitter and is
gated on reproducing its fan bit-for-bit (§4).

---

## 4. INSTRUMENT-FAIL branch — no verdict is issued if any of these fires

1. `fan` from the latent dump is **not bit-identical** to the banked `fan` (⇒ the latent rows do
   not correspond to the banked λ\*, and no row-wise statistic is meaningful).
2. `eid`, `gt`, `sel`, `v0` are not bit-identical to the bank.
3. Window/episode counts differ from **881 / 40**.
4. The raster gate in `refc_s1_dump_emitted.assert_raster` fails
   (R-2026-08-02-a: base **accepts a wrong token count and returns a plausible wrong number
   silently**).
5. `load_state_dict` reports any missing or unexpected key.
6. Either **shuffled-latent control** (§5, `C_shuf*`) recovers **separably above 0**.
   ⇒ the fitting pipeline leaks; the whole panel is void, not just that arm.

---

## 5. Arms — one pipeline, ten feature sets, identical protocol

**Target.** Per window *i*, with `f_i` = the SHIPPED SELECTED trajectory (`fan[i, sel[i]]`, 4×2)
and `g_i` = ground truth:

- **grid target** `λ*_i = argmin_{λ ∈ {0.92,0.96,1.00,1.04,1.08}} ADE(λ f_i, g_i)` — HAD's published
  5-point radial grid, verbatim, the same one E-EXP-1 used.
- **regression target** `λ^ls_i = ⟨f_i, g_i⟩ / ⟨f_i, f_i⟩` — the closed-form least-squares optimum.
  Used as the *regression* target because it is continuous and well-conditioned; **it is never the
  score**.

**Score (the only thing a verdict may read).** `RECOVERY = mean(ADE(f)) − mean(ADE(λ̂ f))`, i.e.
**realised ADE after applying the predicted λ**, on all 881 windows. Positive = better.
Predictions are clipped to **[0.92, 1.08]** so no arm may leave the published operator's reach.
Both **continuous** λ̂ and **grid-snapped** λ̂ are reported; the **continuous** one is PRIMARY
(it is the deployable form and it cannot be helped by a lucky snap).

**Protocol — leave-one-episode-out (LOEO) over the 40 val episodes.** For each held-out episode:
standardisation statistics, the ridge coefficients and the hyper-parameter are all fitted on the
other 39 **only**. α is chosen by an inner **5-fold GroupKFold over episodes** from a fixed grid
`α ∈ {1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5}`, minimising realised ADE on the inner held-out folds.
⛔ Nothing about the held-out episode enters the fit, including feature means.

**Model class.** Ridge regression on standardised features → λ̂ (PRIMARY, decides).
Secondary, reported and deciding nothing on its own: **k-NN** on the standardised latent,
`k ∈ {5,10,20,40}` by the same inner CV — a genuinely different function class, to catch a
non-linear latent→λ relation a linear probe would miss.

| id | features | role |
|---|---|---|
| `F0_const` | intercept only | the "no features" floor. Reduces to the LOEO global λ. |
| `F1_v0` | `[v0, v0², v0³, 1/(1+v0)]` | ⭐ **THE ECHO FLOOR.** Deliberately strong. |
| `F2_geom` | `F1` ⊕ the selected trajectory's own geometry (per-waypoint along & cross, chord speeds, net heading, arc length, curvature) | free, non-latent |
| `F3_score` | `F2` ⊕ fan logit statistics (`logits`/`emitted_logits`/`prefinal_logits`: entropy, top1−top2 margin, std, selected logit, selected rank) | free, non-latent |
| **`F4_pooled`** | `pooled` [F] | ⭐ **THE TREATMENT.** Vision latent, last frame. |
| `F5_pooled_seq` | `[pooled_last, mean_t pooled, pooled_last − pooled_first]` [3F] | temporal vision latent |
| `F6_ctx` | `ctx` | REF-C's own strategic summary |
| `F7_all_latent` | `pooled ⊕ ctx ⊕ measurement` | the full latent set (incl. the ego path) |
| `F8_latent_v0` | `F7 ⊕ F1` | the deployable superset |
| `C_shuf` | `pooled`, rows permuted across **all 881 windows** (seed 0) | ⭐ shuffled-latent control |
| `C_shuf_ep` | `pooled`, rows permuted **within each episode** (seed 0) | ⭐ the STRONGER control — preserves episode-level statistics and destroys only the window-level correspondence, so it cannot be passed by episode-level structure alone |

⚠️ **Why two shuffles.** A global permutation destroys episode structure as well as window
correspondence, so passing it is easy and proves little. `C_shuf_ep` is the one that matters. Both
must fail; either one separating above 0 is INSTRUMENT-FAIL (§4.6).

⚠️ **These controls are NOT of the vacuous `C-shuffled` kind.** In the killed re-ranking studies a
permute-then-argmax was a uniform random pick and could only ever score the mean. Here the shuffled
arm still runs the FULL fit and can still overfit noise, so a non-zero recovery is a real,
detectable leak signal.

---

## 6. PRIMARY read + DIRECTION predicates + the DEAD threshold

**Primary quantities (both paired episode-cluster bootstrap, `taniteval.ci`, unit = episode,
`n_boot=2000`, `seed=0`, `reduce="mean"`; ⛔ `overlapping_holdout_se` is never called):**

- **R_latent** = `RECOVERY(F4_pooled)` — and, reported beside it, the best latent arm among
  {F4, F5, F6, F7, F8}. ⚠️ The **best-of-5** is a multiplicity surface, so `F4_pooled` alone is the
  pre-declared PRIMARY and the best-of is reported as secondary with that caveat attached.
- **R_latent − R_v0** = paired delta of realised ADE, `F1_v0` vs `F4_pooled`.

**⭐ THE THRESHOLD, COMMITTED NOW.** θ = **0.0258 m**.

Provenance: K1 measured that the **entire re-ranking class** — every selection lever this
programme has killed — is bounded at **≤ 8.4 %** of the 0.3075 m oracle gap = 0.0258 m. A lever
that recovers less than the *upper bound on a class we already killed* does not earn a parameter.
As a share of the E-EXP-1 ceiling, θ is **16.2 %** (base) / **15.6 %** (XL).

**Three-sided verdict table (each row carries a DIRECTION predicate, not merely separation):**

| condition | verdict | what we do |
|---|---|---|
| `R_latent` **> 0** and CI excludes 0, **AND** `R_latent − R_v0` **> 0** and CI excludes 0, **AND** neither shuffle separates above 0, **AND** `R_latent` point **> θ = 0.0258** | **FINDABLE — WORTH A HEAD** | Specify the **cheapest discriminating training arm** and its parameter cost; ⛔ do NOT launch it. Hand the GPU-day to the PI with both outcomes pre-committed. |
| `R_latent` **> 0** and CI excludes 0, **AND** `R_latent − R_v0` **> 0** and CI excludes 0, **BUT** `R_latent` point **≤ θ** | **FINDABLE — TOO SMALL TO FUND** | State the size plainly. The latent carries λ information but less than the bound on an already-killed class. Do not fund a head. |
| `R_latent` CI **includes 0**, **OR** `R_latent − R_v0` CI **includes 0**, **OR** `R_latent − R_v0` point **< 0** | ⛔ **DEAD — λ\* IS NOT FINDABLE FROM THESE LATENTS** | Report it as a refuted lever, with the size of the ceiling it fails to reach. Demote the along-path residual family for us. |
| either shuffle separates **above** 0 | **INSTRUMENT-FAIL** | No verdict on any arm. Fix the pipeline, re-run, re-pin. |

⛔ **BANNED REFRAMINGS OF THE DEAD BRANCH, committed in advance.** If the DEAD row fires, the
report may **not** say *"needs training to emerge"*, *"the latent was never trained for this"*,
*"a bigger head would find it"*, or *"this motivates an auxiliary loss"*. Those are the moves that
turn a refutation into a funding request. A refuted lever is a result this programme funds; three
have been killed this week on banked fans alone. The permitted statement is: **the information is
not linearly or locally decodable from REF-C's representation at 30k, at this n, above the `v0`
echo — and here is the size of what was missed.**

⚠️ **Two-sided reporting.** Every arm is reported in a three-column table — **better / worse / not
separated** — never as a list of the ones that helped. `F1_v0` being *separably worse than doing
nothing* in E-EXP-1b is exactly the kind of row that must remain visible.

---

## 7. P2 — the FOUR METRIC FAMILIES, per family, never pooled (binding, Sayed 2026-08-02)

Reported for **three trajectory sets on identical windows**: the shipped `sel`, the
**predicted-λ** `λ̂·sel` (best latent arm), and the **oracle-λ** `λ*·sel` (the ceiling).
Estimator: paired episode-cluster bootstrap on the same 881 windows / 40 episodes.
⛔ dt is **0.5 s**, derived from `wp_steps=(5,10,15,20)` via `four_families.infer_dt` — never the
0.1 s default, which inflated every published speed by ×5 and every accel by ×25 (R-2026-08-03).

| family | what is reported | status, declared in advance |
|---|---|---|
| **LONGITUDINAL** | `speed_mae/bias/rmse`, `along_mae/bias/final_bias`, `accel_mae`, `ego_progress`, **and distance-keeping** (headway, time-gap, min-TTC) via `taniteval.lead_metrics` from the banked `val40_lead_block.npz` (**LEAD 270 / NO_LEAD 551 / NO_LABEL 60**), speed-stratified | **AVAILABLE.** This is the family λ acts on. |
| **LATERAL** | heading, yaw-rate, curvature, cross-track | **AVAILABLE.** ⭐ **PRE-REGISTERED ANALYTIC PREDICTION, to be verified numerically:** a radial scale about t0 scales every displacement uniformly, so **heading and yaw-rate are EXACTLY invariant**, **curvature scales as 1/λ**, and **cross-track scales as λ**. If the measured numbers contradict this, it is INSTRUMENT-FAIL, not a finding. |
| **TACTICAL** | trajectory-derived **factored** (lat, lon) manoeuvre via `refc_tactical.factor_from_kinematics` — the canonical labeller — applied to pred and GT on identical footing; agreement, per-class recall, `never_predicted` | **AVAILABLE WITH A STATED CAVEAT:** this is a **trajectory-derived proxy**, using the path tangent where the training label uses pose yaw, and a 0.5 s grid. It is comparable ACROSS the three trajectory sets (identical definition) and is **not** quotable as the trained head's accuracy. ⭐ λ is bearing-preserving, so it can only move the **lon** half — the exact half of the 5-way softmax that is our largest known defect. |
| **STRATEGIC** | goal/route setting | **PARTIALLY N/A, with the reason and the n.** (a) The map-derived option-set path is **impossible on PhysicalAI** — settled at five independent probes: no map, lane graph, junction annotation or route signal; the card says verbatim *"we do not include open maps data"*. n = 0 junctions scorable. (b) What IS computable and reported: λ is **provably bearing-preserving**, so the strategic *direction* is bit-invariant under the operator (verified numerically, and a non-zero result is INSTRUMENT-FAIL); the goal *distance* at 2 s is not, and its error is reported for all three sets. |

⛔ An ADE horizon sweep is **one row of four** and is never presented as "the result".

---

## 8. P3 — sizing, if and only if the lever survives

The comparison that decides is **not** "better than nothing". It is:

1. **Better than the levers already refuted** — θ (§6) is exactly that bar.
2. **Worth its parameters.** Accepted levers in this programme cost **+897 / +385 / +128 / 0**
   parameters; one cost **+272,001** before its own control caught it. The λ head's cost is
   reported as an exact parameter count for the specified arm.
3. ⛔ **No training is launched.** The deliverable is a specification of the cheapest discriminating
   arm — architecture, parameter count, the pre-registered read, and **both outcomes committed in
   advance** — handed to the PI. Provision and spend are the PI's.

---

## 9. What this experiment explicitly CANNOT say

1. **Nothing about closed-loop.** Banked open-loop fans; MEASURED (ours) 0.45 m open-loop →
   1.69 m closed-loop. Open-loop does not predict closed-loop.
2. **Nothing about the flagship**, which has no fan at all (`anchor_decoder is None`, four unimodal
   `Linear(d,2)` heads). Two REF-C arms only.
3. **Nothing about a λ head trained END-TO-END.** This measures whether the information is present
   and decodable in the *frozen* 30k representation. That is the necessary condition; it is not
   the sufficient one, and a PASS here still buys only a pre-registered training arm.
4. **n = 40 episodes** is the bootstrap's unit. Every interval here is a 40-cluster interval and is
   quoted as such.
5. **Nothing about `overlapping_holdout_se`-era numbers.** Every comparison here is recomputed on
   `full_set` means from raw per-window data.

---

## 10. Provenance and pinning

`git hash-object PREREG_E_EXP2.md` is recorded in `raw/prereg_pin.json` **before** the first
statistic and **re-verified at the end of the run**. A changed blob invalidates the whole report.
