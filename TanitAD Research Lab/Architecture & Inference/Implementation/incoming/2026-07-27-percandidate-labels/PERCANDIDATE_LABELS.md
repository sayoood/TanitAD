# The per-candidate label ladder — T0 · T1 · T2 · composite · T3

**Written:** 2026-07-27 (Europe/Berlin; pods and logs are UTC) · **Host:** dev box
(`C:/Users/Admin/venvs/tanitad`, RTX 4060). ⛔ **No pod was contacted, not even read-only.**
**Mandate:** the 4-step ladder pre-registered in
`…/Research/2026-07-27-planner-scorer-inputs/SCORER_INPUTS_RESEARCH.md §7`.
**Question:** Bar A refuted the *objective*; the research says the *target* was never a target.
Do rule-computed per-candidate verdicts exist on our corpus, and do they discriminate?

Evidence stamps on every number: class `MEASURED` / `PUBLISHED` / `INHERITED` / `ESTIMATED` /
`HYPOTHESIS`, **and** tier `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE`.

---

## HEADLINE

> ### T2 **SURVIVES** its pre-registered kill by **4.5×**, and the ladder produced a sharper result than the kill test itself.

Five findings, each `MEASURED` with its artifact.

1. **⭐ The rule label exists, it is common, and it discriminates *within* windows.**
   Over **860,160 per-candidate verdicts** (96 clips × 3,360 windows × 256 candidates), the
   at-fault-collision label fires on **22.65 %** of candidates — CI **[18.98 %, 26.41 %]**, whose
   **lower bound is 3.8× the 5 % kill threshold** — and it **VARIES ACROSS THE FAN in 81.73 %
   [76.13, 86.76] of windows**. TTC: **28.67 %**, varies in **86.28 %**. The kill needs the rate
   leg **and** the ρ leg; the rate leg fails by 4.5×, at **both** threshold operating points.
   *The corpus is not too empty to supervise a rule scorer: mean **53.6** tracked agents per
   window, median 48, max 187, and only **0.03 %** of windows have none.*

2. **⭐⭐ ADE and the rule verdict are not merely uncorrelated — they are in tension.**
   Selecting the **ADE-optimal** candidate from the fan collides at **3.36 % [1.70, 5.36]**.
   Selecting the **rule-optimal** candidate collides at **0.71 % [0.12, 1.70]**. Paired
   episode-cluster bootstrap on the identical windows: **−0.0265 [−0.0432, −0.0128], SEPARATED.**
   **The oracle of the metric we have been optimising is 4.7× more dangerous than the oracle of
   the metric we have not.** Spearman ρ(label, `fan_err`) is **0.149** — the label is nearly
   orthogonal to distance-to-the-realised-future, which is exactly the information a
   distance-trained scorer cannot contain.

3. **⭐ T0 is settled: the 181 km/h candidates are manufactured by the OFFSET HEAD, not the
   vocabulary — and the proof is structural, not statistical.**
   `furthest_point_sample` returns `pool[chosen]`; **256 of 256 anchors are bitwise identical to a
   real human window** in the pool. Demonstration pool max 2 s displacement **37.91 m (68.2 km/h)**;
   the **emitted** fan of a matched architecture reaches **95.27 m (171.5 km/h)** with **p99 at
   88.65 m (159.6 km/h)** — above the **maximum realised displacement anywhere in the val set
   (73.54 m)**. **5.92 %** of all emitted candidates exceed that maximum and **100 % of windows
   contain at least one.** ⇒ the fix is **clamp the refinement**, not rebuild the vocabulary.

4. **⭐ T1: the kinematic clip is FREE, INERT on ADE, and worth −59.9 % of the rule scorer's
   Goodharting. It is a precondition for R1, now measured rather than argued.**
   The head's *own* reachability clamp applied to the candidates removes **72.08 %** of the emitted
   fan; the ADE-oracle survives in **100 %** of windows; the as-trained pick **does not move in a
   single window** (paired Δ **exactly 0.0000**). On the **rule** surface the same clip moves the
   pick from **9.7073 m → 3.8950 m** ADE and its implied speed from **57.2 → 40.6 km/h** (mean ego
   speed 31.7), *while raising* PDMS-lite 0.8067 → 0.8480.
   ⚠️ **T1's own pre-registered kill therefore FIRES**: neither `ade_0_2s` nor `miss_at_2m`
   improves, so the clip is **not a lever on its own** and R2/R3 drop to **CONDITIONAL** — which is
   what the research already recommended them as.

5. **⭐ The imagination cannot rank candidates, and it is a code fix.** `imagine_probes` returns
   `[B, n_probes·len(imag_read), S]` — **32 tokens, invariant to `n_anchors`** (verified at runtime
   at 64 and 256). There is **no candidate axis anywhere in the imagination path**, and
   `V15Decoder._decode` emits all 256 candidates from the same KV. **A feature identical for every
   candidate cannot rank them, by construction.** E-V5-1's negative is over-determined.

⛔ **Nothing was launched, restarted or trained. T3 was NOT run — see §8 for why, and for its
pre-registration.** This is a measurement; the decision is Sayed's.

---

## 0. PRE-REGISTRATION — and an honest statement of what was pre-registered by whom

**⚠️ I did not author the kill.** T2's kill rule is taken **verbatim** from
`SCORER_INPUTS_RESEARCH.md §7`, written and staged **before** this agent existed:

> *"if the NC/TTC labels have < 5 % positive rate AND their Spearman ρ with `fan_err` is < 0.15,
> R1 is REFUTED on this corpus."*

It is honoured **unmodified**, evaluated at **two independent threshold operating points**, and
reported with an episode-cluster bootstrap interval. T1's kill (*"if `ade_0_2s` does not improve
**and** `miss_at_2m` does not improve, R2/R3 drop to CONDITIONAL"*) is likewise verbatim from §7
and **it fires** — reported as such in §4 rather than re-scoped.

What I did commit in advance, before computing it, is the §6 prediction: *the clip should be inert
for the ADE picker and large for the rule picker.* It is written into
`code/t2b_clip_x_rule.py`'s docstring, which was authored before that script was first run.

**Estimator, named in advance and never varied:** episode-cluster bootstrap (`taniteval/ci.py`,
B = 2000), unit = the clip; **paired** form for every contrast, because both arms live on the same
windows. **`overlapping_holdout_se` is never used** — it is not a jackknife, it biases the point
estimate bidirectionally (−6.67 % to +11.69 % over 27 arms) and it has flipped the sign of a paired
delta.

### 0.1 ⚠️ HOST AND CORPUS DISCLOSURE — read this before quoting any number

The brief assumed *"Bar A's committed cache"* holds the fan. **It does not.** Bar A's staged
`raw/*.pt` are per-window ADE dumps; its two 4.07 GiB feature caches are **deliberately not staged**
and live only on `tanitad-eval:/workspace/_bara/` (Bar A §10 says so explicitly). The dev box holds:

| asset | what it is | key |
|---|---|---|
| `physicalai-train-14231cd29c74` (400 eps) | episode cache | ⚠️ **NOT** the parity key `e438721ae894` |
| `physicalai-val-bb543bdf7836` (100 eps) | episode cache | ⚠️ **NOT** the clean val `physicalai-val-0c5f7dac3b11` |
| `pai_probe_cache/labels/{obstacle.offline,egomotion}.chunk_0000` | **96 clips carrying both** | raw PhysicalAI-AV |
| `taniteval/results/fan_refc-xl-30k.pt` | REF-C-XL's **emitted** 256-candidate fan + **real logits**, 881 canonical val windows | committed |
| `…/v5-imagination-selection/raw/v5_v4_windows_reduced.pt` | flagship-v4's real per-candidate `fan_err4`, same 881 windows | committed |

**Consequences, stated rather than glossed:**

- **Nothing here re-selects episodes for an arm comparison.** T2 measures a property of *labels*;
  T0/T1 run on committed fans over the canonical 881 windows. **The parity invariant is untouched.**
- **v4's own emitted fan geometry and `flagship_v4_anchors_dense.pt` are pod-side.** T0's offset-side
  leg and T1's clip therefore run on **REF-C-XL** — the *same* `V15Decoder`, the *same* 256 anchors,
  the *same* unbounded `offset_head` (`refc.py:539-540`) that v4 inherits — **not on v4's tensor.**
- **v4 supports only the COVERAGE side of the fan-size sweep.** Its `base_rank` in the v5 dump is
  *rank-0-real, fan-order thereafter* by that script's own documentation, so it is not a usable
  score ranking.
- 🔒 **No clip UUID or raw content reaches any artifact.** Clips appear as `clip_<sha256[:8]>`.

---

## 1. THE CLOCK TRAP — proved, and the brief's premise is half wrong

🔴 The brief warns: *"`egomotion` spans ~140 s while `obstacle.offline` spans ~20 s **with different
origins**."* **The origins are IDENTICAL. What differs is the SPAN, and that is the trap.**
`MEASURED · CONFIRMED` · artifact `raw/t2a_clock_join.json`.

**The proof, and it can come out either way.** `obstacle.offline` boxes are `reference_frame="rig"`
on 100 % of rows, i.e. ego-relative. A world-static object therefore has a rig-frame track that is
exactly the ego's motion, negated — so `world(track, t) = ego_pose(t + δ) ⊕ rig(track, t)` is
constant **iff δ is the true offset**. We sweep δ over ±2 s and take, **fresh at every δ**, the p10
of per-track world-position dispersion (no track is pre-selected at any δ, so the test cannot be
rigged toward zero).

| δ | median p10 dispersion over 30 clips |
|---|---|
| **0.0 s** | **0.0639 m** |
| +1.0 s | 0.3166 m — **4.96× worse** |

**Best δ = 0.0 s; 93.3 % of clips minimise at exactly 0.0 s individually.** The
`failing_input` self-test (a deliberately wrong +1.0 s) is reported as worse ✅ — the instrument can
render a failing verdict.

**What the trap actually is.** `obstacle.offline` covers **exactly [0, 20] s** (measured span
19.907–20.000 s over 30 clips). `egomotion` carries **~2,000 rows at 100 Hz inside that same
[0, 20] s** *plus a sparse trailing tail whose last timestamp runs from 20.2 s to 140.2 s* — over 30
clips, **2.0 % to 59.2 % of `egomotion` rows lie OUTSIDE the labelled clip, median 21.1 %.** Any
index-based or full-range join silently drags those in. *(That the epcache episodes are 199 frames
at 10 Hz = 19.9 s is an independent confirmation that the clip proper is the [0, 20] s window.)*

---

## 2. T0 — the span audit. **The offset head, not the vocabulary.**

`MEASURED · CONFIRMED` · artifact `raw/t0_span_audit.json`, anchors `raw/anchors_dev256.pt`.

### 2.1 Leg 1 — structural, and it is a proof rather than a measurement

`build_refc_anchors.episode_traj_pool` pools `refb_labels.waypoint_targets` over **every window** of
the corpus; `refc.furthest_point_sample` ends `return pool[chosen]` — **it SELECTS pool members and
never synthesises a centroid** (its docstring says it chooses FPS over k-means deliberately).
⇒ **every anchor IS, verbatim, one real human 2 s trajectory.**

**Verified numerically rather than asserted: `n_anchors_bitwise_identical_to_a_pool_row = 256 / 256`,
max L∞ distance `0.0`.**
*(A first pass used `torch.cdist` in fp32 and returned 0.044 m — enough to have printed "not exact
membership". Over a 40-dim vector with ~38 m entries that is pure rounding. Recomputed in float64
with bitwise comparison. **Precision is a scientific parameter here**, exactly as Bar A's R-2
found for fp16 caching.)*

### 2.2 Leg 2 vs Leg 3 — the numbers

| | source | max 2 s along-track | implied mean speed | p99 |
|---|---|---:|---:|---:|
| **demonstration pool** (71,577 windows) | dev-box train cache | **37.91 m** | **68.24 km/h** | 28.29 m |
| **FPS-256 anchors** | ⊆ the pool | **37.91 m** | **68.24 km/h** | 33.87 m |
| **REALISED future**, canonical val | `fan_refc-xl-30k.pt` `gt` | **73.54 m** | **132.39 km/h** | 73.03 m |
| **EMITTED fan** (`x_in + offset`), canonical val | REF-C-XL, 225,536 candidates | **95.27 m** | **171.50 km/h** | **88.65 m (159.62 km/h)** |

> **Verdict: the superhuman span is manufactured by the unbounded refinement.**
> - **5.92 %** of all emitted candidates exceed **the maximum displacement any real val window ever
>   achieved**, and **100 % of windows contain at least one such candidate.**
> - The emitted fan's **p99 (159.6 km/h) is above the realised maximum (132.4 km/h)** — this is not
>   a thin tail.
> - Mean **per-window** along-track span **104.61 m** (max 112.25 m). The v5 stream independently
>   measured **108.7 m** on **v4's** fan; two different arms, two different harnesses, same shape.
> ⇒ **The fix is R3 (clamp the refinement), not rebuilding the vocabulary.**

⚠️ **Honest limit.** The dev-box pool (68.2 km/h max) is a *different, smaller* corpus than parity —
so its 37.91 m is **not** the parity anchor bound. The load-bearing argument is Leg 1, which is
corpus-independent: an anchor is a real window, so `max(anchors) ≤ max(real 2 s displacement)`. On
the canonical val split that real maximum is **73.54 m**, and the emitted fan is **29.6 % beyond
it**.

---

## 3. T1 — the kinematic clip. **Free, and inert on ADE.**

`MEASURED · CONFIRMED` · artifact `raw/t1_clip_fansize.json`.

**The band comes from physics and our own head, never from held-out error.**
`FlagshipV15Head.select` already computes `reach = sel_accel_max · horizons[-1] · 0.1`
= **2.5 m/s² × 2.0 s = 5.0 m/s** for the *goal* (`flagship_v15.py:139,455`). T1 applies the
**identical** clamp to the *candidates*: keep `v_term ∈ [max(0, v0 − 5.0), v0 + 5.0]`. Nothing is
tuned on `ade_0_2s`.

**Harness fidelity, both directions.** The dump's own recorded pick equals `argmax(logits)` on
**881/881 windows (1.000000)**, and the recomputed as-trained `ade_0_2s` **0.4714** reproduces the
committed headline **0.4714** (`taniteval/results/driving_refc-xl-30k.json`) by an independent path.
A uniform-random pick is reported as far worse (§5 arms).

| | value |
|---|---:|
| candidates removed | **72.08 %** |
| windows left with an empty survivor set | **0.00 %** |
| **ADE-oracle survives the clip** | **100.00 %** of windows |
| oracle `ade_0_2s` | 0.1640 → **0.1640** (unchanged) |
| as-trained `ade_0_2s` | 0.4714 → **0.4714** |
| paired Δ (episode-cluster bootstrap, B = 2000, 40 episodes) | **+0.0000 [0.0000, 0.0000]**, not separated |
| `miss_at_2m` | 0.0159 → **0.0159** |

> ### ⚠️ T1's pre-registered kill FIRES.
> Neither `ade_0_2s` nor `miss_at_2m` improves — **the pick never changed in a single window** —
> so *"the fan's tail is not what is hurting the pick"* and **R2/R3 are CONDITIONAL, not levers.**
> Said plainly, not re-scoped.
>
> **But "inert" is not "useless": the clip deletes 72.08 % of the fan at ZERO cost to the ceiling
> and ZERO change to the pick.** That is the definition of a free precondition — and §6 shows it is
> worth −59.9 % of the rule scorer's Goodharting, which is precisely why the research listed R3 as
> a precondition for R1 and R4 rather than as a win.

### 3.1 The fan-size sweep — and it goes AGAINST shrinking

Random subsets of size K, 8 seeds each, on both surfaces. Surface A has real logits so it shows the
**realisable pick**; v4 supports only **coverage**.

| K | A: oracle | A: **realisable pick** | A: miss@2m | B (v4): oracle |
|---:|---:|---:|---:|---:|
| 4 | 4.5462 | 5.2866 | 0.7250 | 4.9648 |
| 8 | 2.5648 | 3.3678 | 0.5511 | 2.6048 |
| 16 | 1.4438 | 2.2012 | 0.3912 | 1.3568 |
| **20** *(PLUTO / DiffusionDrive)* | 1.1939 | **1.8974** | 0.3234 | 1.1134 |
| 32 | 0.8232 | 1.4086 | 0.2240 | 0.7669 |
| 64 | 0.4753 | 0.9365 | 0.1037 | 0.4800 |
| 128 | 0.2722 | 0.6503 | 0.0390 | 0.3339 |
| **256** *(ours)* | **0.1640** | **0.4714** | **0.0159** | **0.2505** |

> **Monotone degradation, no plateau, on both surfaces.** Shrinking our fan by random subsetting is
> a pure loss: at K = 20 the realisable pick is **4.0× worse** than at K = 256.
>
> ⚠️ **This does NOT refute R2, and saying so would be a C6 confound.** Random subsetting varies
> *coverage* and *size* together. PDM's 15 proposals are a **designed, current-state-conditioned**
> set; DriveSuprim coarse-filters 8,192 → 256 **by score**. The correct comparison is a *designed*
> filter — which is the kinematic clip, and it removes 72 % for free. **What this table does refute
> is the free-lunch reading of LLM-Assist: on our surface a bigger fan does not make the realisable
> picker worse.** `MEASURED (ours) · CONFIRMED` vs `PUBLISHED (LLM-Assist, PDF-VERBATIM): 92.51 at
> 15 → 77.78 at 8,505`. Both are recorded; they are not in contradiction because their filters
> differ, and ours was never Goodharted on ADE (§6 shows it *is* Goodharted on the rule surface).

---

## 4. T2 ⭐ — the label discrimination, against the kill

`MEASURED · CONFIRMED` · artifacts `raw/t2_labels.json` (calibrated),
`raw/t2_labels_conservative.json`, per-window dump `raw/t2_labels_windows.pt`,
intervals `raw/t2c_intervals.json`.

**Scale:** 96 clips · **3,360 windows** · 256 candidates = **860,160 per-candidate verdicts**, CPU
only, ~4 min per operating point.
**Fan:** 256 FPS anchors over real waypoint targets — the `build_refc_anchors` real-data recipe,
i.e. our own vocabulary, **without** an offset head (v4's offset head is pod-side).
**Agent futures:** **log replay** — the `obstacle.offline` tracks *are* the future, exactly as
NAVSIM's NC/TTC teachers use them. **No prediction model was built**, per PLUTO's measured +0.75.

### 4.1 What is buildable, and what is not

| rule | built? | note |
|---|---|---|
| **NC** at-fault collision | ✅ | OBB (separating-axis) sweep vs log-replay tracks |
| **TTC** | ✅ | constant-velocity projection of ego *and* agents, real footprints |
| **C** comfort | ✅ | candidate geometry only |
| **EP** progress | ⚠️ partial | along-heading proxy; **no route exists in the corpus** |
| **DAC / DDC / LK / TL** | ⛔ **NO** | no map in PhysicalAI-AV (settled at five probes). **DAC is one of PDMS's two multiplicative terms — this is a real ceiling, not a nicety.** |

### 4.2 The discrimination table

| label | positive rate | 95 % CI | **varies within-window** | ρ vs `fan_err` (pooled / median-per-window) |
|---|---:|---|---:|---:|
| **NC (at-fault)** | **0.2265** | [0.1898, 0.2641] | **0.8173** [0.7613, 0.8676] | 0.1494 / 0.1467 |
| collision, any | 0.2389 | — | 0.8104 | 0.1624 / 0.1447 |
| **TTC** | **0.2867** | [0.2446, 0.3275] | **0.8628** [0.8107, 0.9072] | 0.1525 / 0.1477 |
| comfort **violation** | 0.9934 | [0.9930, 0.9939] | 0.7702 | 0.1036 / 0.1055 |

**Robustness — the same table at a second, conservative operating point** (ego pad 0.35 m, TTC
horizon 1.0 s instead of 0.0 m / 0.5 s): NC **0.2545**, TTC **0.3917**, within-window variation
0.825 / 0.900. **The verdict does not move.**

> ### VERDICT: the pre-registered kill **DOES NOT FIRE**, at either operating point.
>
> | leg | threshold | measured (calibrated) | measured (conservative) |
> |---|---|---:|---:|
> | positive rate | < 5 % | **22.65 %** (4.5×) | **25.45 %** (5.1×) |
> | \|ρ\| with `fan_err` | < 0.15 | 0.1494 | 0.1387 |
> | **kill fires?** | needs **both** | **NO** | **NO** |
>
> The rate leg fails by 4.5× at the point estimate and by **3.8× at the CI lower bound**. The ρ leg
> is marginal and **irrelevant**, because the rule is a conjunction.
>
> **And read ρ ≈ 0.15 the right way.** A label that were rare *and* uninformative would be useless —
> that is what the kill was written to catch. What we measured is a label that is **common,
> varies inside 4 of every 5 windows, and is nearly ORTHOGONAL to ADE**. Orthogonality is the
> asset: it is precisely the information a distance-trained scorer structurally cannot hold.

### 4.3 The instrument, checked in the direction that can only embarrass it

The **same labeler** applied to the **realised human future**:

| | calibrated | conservative | reference |
|---|---:|---:|---|
| GT at-fault collision | **2.05 %** | 2.83 % | `PUBLISHED (PARA-Drive, PDF-VERBATIM)`: GT "collides" at **0.384 %** under axis-aligned boxes |
| GT any collision | 2.08 % | 3.30 % | |
| GT TTC infraction | 2.02 % | 4.76 % | |
| GT comfort OK | 85.92 % | 85.92 % | |

⚠️ **Stated as a limitation, not tuned away: our labeler flags the human at ~5.3× PARA-Drive's
reference rate.** There is a real false-positive floor (box padding, track interpolation, and the
simplified "agent ahead of the rear axle" at-fault proxy). **The contrast survives it easily** —
the fan is flagged at **22.65 %, i.e. 11.0× the GT floor** — but **no absolute collision rate from
this labeler is quotable as a safety number**, and calibrating it against a held-out reference is
listed as owed work in §9.
⚠️ **The comfort label is SATURATED (99.34 %) and is therefore a degenerate binary here.** Cause:
the anchor vocabulary is **global, not speed-conditioned**, so most candidates imply an infeasible
acceleration from the current `v0`. That is itself the R3 argument (§6) — but it means comfort must
be computed on a *clipped* or scene-conditioned fan or it discriminates nothing.

---

## 5. THE COMPOSITE — a prerequisite, built, and it already changes the verdict

⚠️ **`ade_0_2s` CANNOT ADJUDICATE A SCORER.** `PUBLISHED (same-day closed-loop research)`:
L2/ADE vs closed-loop Driving Score at **ρ = −0.36, p = 0.43**, with Ego Progress the strongest
single predictor (ρ = 0.83). So the composite is built **first**.

**PDMS-lite (no-map)** `= NC × (5·EP + 5·TTC + 2·C) / 12` — PDM's own term list and weights
(`PDF-VERBATIM`, CITATIONS #11), with **DAC dropped because no map exists**. ⚠️ It is therefore
**not comparable to any published PDMS number** and is never to be quoted as one.

| selection arm (same 3,360 windows, same fan) | PDMS-lite | at-fault collision | `ade_0_2s` |
|---|---:|---:|---:|
| **rule-optimal** (PDMS-lite argmax) | **0.8067** [0.792, 0.820] | **0.0071** [0.0012, 0.0170] | 9.7073 |
| **ADE-optimal** (`oracle_in_fan`) | 0.5880 [0.562, 0.612] | **0.0336** [0.0170, 0.0536] | **0.5906** |
| random *(the deliberately failing input)* | 0.4359 [0.410, 0.462] | 0.2167 [0.1771, 0.2580] | 7.5251 |

**Paired episode-cluster bootstrap, B = 2000, unit = clip:**

| contrast | Δ collision rate | 95 % CI | separated |
|---|---:|---|---|
| rule − random | **−0.2095** | [−0.2491, −0.1705] | ✅ |
| **rule − ADE-oracle** | **−0.0265** | [−0.0432, −0.0128] | ✅ |
| ADE-oracle − random | −0.1830 | [−0.2226, −0.1432] | ✅ |
| rule − ADE-oracle, `ade_0_2s` | **+9.1167** | [+8.0964, +10.1814] | ✅ |

> **The two surfaces disagree with statistical separation in BOTH directions on the same windows.**
> The ADE-optimal pick is **4.7× more likely to collide** than the rule-optimal pick; the
> rule-optimal pick is 9.12 m worse in ADE. **This is the single sharpest result in the ladder**:
> Bar A's target and a rule verdict are not two estimates of one quantity — they are different
> objectives, and our entire measurement stack has been reading only one of them.
>
> ⚠️ **The rule arm's 9.71 m ADE is a DIAGNOSTIC, not a proposal.** With no map, no route and no
> speed limit, EP is unbounded and rewards the fastest candidate. **PDM explicitly DROPS its
> speed-limit and no-progress terms because "the generator enforces them"** — we have no such
> generator. This reproduces `PUBLISHED (PLUTO T.VI)` exactly: rule-only maximises progress
> (98.43) and destroys comfort (80.32); the **mix at α = 0.3** beats both. ⛔ And per the standing
> constraint, **rules are TRAINING LABELS here, never inference costs** — PARA-Drive measured that
> removing UniAD's inference-time rule optimiser improved *both* collision (0.40 → 0.16) and L2
> (0.83 → 0.74), and our own REF-C v1.0 recovered 0.0 % from a hand-written cost re-rank.

### 5.1 Does a rule veto destroy the fan's own ceiling?

| veto applied to the fan | windows with a survivor | oracle `ade_0_2s` | ADE-oracle itself vetoed |
|---|---:|---:|---:|
| none | 100 % | **0.5906** | — |
| **NC only** | **99.32 %** | 0.6614 (+12.0 %) | **3.36 %** |
| NC + TTC + comfort | **68.42 %** | 1.6808 | 93.81 % |

**An NC veto is nearly free** (the ceiling moves 12 %, a survivor exists in 99.3 % of windows).
**Adding the saturated comfort veto destroys it** — 31.6 % of windows are left with no candidate at
all. ⇒ **NC/TTC are usable as hard training targets today; comfort must not be, until the fan is
speed-conditioned.**

---

## 6. T1 × T2 — the clip is a PRECONDITION FOR R1, and the prediction was committed first

`MEASURED · CONFIRMED` · artifact `raw/t2b_clip_x_rule.json`. The prediction in
`code/t2b_clip_x_rule.py`'s docstring — *"inert for the ADE picker, large for the rule picker"* —
was written before the script was first run.

The head's own 5.0 m/s clamp applied to the anchor fan (removes **47.61 %**; ADE-oracle survives in
**98.66 %** of windows):

| | unclipped | **clipped** | Δ |
|---|---:|---:|---:|
| **ADE-oracle pick**, `ade_0_2s` | 0.5906 | **0.5906** | **0.0000** |
| **rule pick**, `ade_0_2s` | 9.7073 | **3.8950** | **−59.9 %** |
| rule pick, implied mean speed | 57.20 km/h | **40.64 km/h** | −29.0 % *(mean ego speed **31.72 km/h**)* |
| rule pick, PDMS-lite | 0.8067 | **0.8480** | **+5.1 %** — the score *improves* |
| rule pick, at-fault collision | 0.0071 | 0.0077 | flat |

> **The clip is exactly inert on the surface T1 measured and exactly large on the surface T1 could
> not see.** The 72 % kinematically-unreachable tail was never what the ADE argmax was eating — but
> it is **precisely** what an unbounded rule scorer eats. R3 is not a lever; it is the thing that
> has to exist before R1 is safe. `MEASURED (ours)` + `PUBLISHED (LLM-Assist: the same Goodhart
> signature — progress rising 91.75 → 95.60 while TTC collapses 93.11 → 62.89)`.

---

## 7. THE IMAGINATION IS NOT CANDIDATE-CONDITIONED — verified in source and at runtime

`MEASURED (ours — source read + runtime shape proof) · CONFIRMED` · artifact
`raw/t4_imagination_conditioning.json`.

| evidence | |
|---|---|
| `flagship_v15.py:505` | `imagine_probes(predictor, states, actions, **probes**, read, v0n)`, `probes` is `[M, K, 2]` — a vocabulary of **probe ACTION SEQUENCES**, not candidates |
| `flagship_v15.py:525` | `pr = probes.unsqueeze(0).expand(b, m, k, 2).reshape(b*m, k, 2)` — the same M probes expanded across the **batch**; there is **no candidate axis** |
| `flagship_v15.py:269` | `n_imag_tokens = cfg.n_probes * len(cfg.imag_read)` — **8 × 4 = 32**, independent of `n_anchors` |
| runtime, no checkpoint needed | `n_imag_tokens` = **32 at `n_anchors=64` and 32 at `n_anchors=256`**; `imagine_probes` output shape `[B, 32, S]`, matches expectation |
| `refc.py` `V15Decoder._decode(kv, cond, x_in, t_idx)` | all 256 candidates are emitted from the **same** KV built from those 32 tokens |

> **32 imagination tokens serve all 256 candidates and are identical for every one. A feature that
> is constant across candidates cannot rank them, by construction.**
> ⇒ E-V5-1's negative is **over-determined**, and this is a **code fix, not a research question**.
> The v5 stream already implements the candidate → action-sequence map
> (`V5_IMAGINATION_SELECTION §0.3`); the missing piece is wiring it into the head's **conditioning**
> path rather than only into a post-hoc scorer — which is exactly WoTE's mechanism
> (`PUBLISHED, HTML-SUMM`: no evaluator 81.0 → current-state 83.2 → **world-model futures 85.6**).

---

## 8. T3 — **NOT RUN.** Why, and its pre-registration

⛔ **T3 as briefed is not executable on this host, and I will not substitute something else and call
it T3.** It requires *"Bar A's cache, folds and estimator"*; Bar A's `q_final` / `q0` feature caches
(4.07 GiB each) exist **only** on `tanitad-eval:/workspace/_bara/`, and the brief forbids touching a
pod. Without the scorer's input features there is no head to train — this is a **capability gap, not
a result**.

**Pre-registration, so that whoever runs it cannot reverse-engineer the rule from the outcome:**

| | |
|---|---|
| **arms** | `AS_TRAINED` · `CE_CONTROL` (Bar A's exact refit, original target) · `BCE_RULE` (`conf_head: Linear(512,1) → Linear(512,4)` + 4 sigmoids, BCE on NC/TTC/C/EP) |
| **protocol** | Bar A's 5-fold episode-disjoint cross-fit, its LR grid {3e-5, 1e-4, 3e-4}, its cache-fidelity + failing-input self-tests |
| **inference combine** | Hydra-MDP form `w₁ log S_im + w₂ log S_NC + w₃ log S_TTC + w₄ log(·)`, **weights by grid search on the fit folds only** |
| **primary read** | **PDMS-lite (no-map), not `ade_0_2s`** — §5 shows the two are separated in opposite directions |
| **estimator** | paired episode-cluster bootstrap, B = 2000, unit = episode. Never `overlapping_holdout_se` |
| **CONFIRM** | `BCE_RULE − CE_CONTROL` separated-better on PDMS-lite **and** its at-fault collision rate separated-below `CE_CONTROL`'s |
| **REFUTE** | not separated on PDMS-lite. **Say so plainly; do not re-scope** |
| **preconditions** | the kinematic clip of §3 applied first (§6); comfort **excluded** from the hard veto until the fan is speed-conditioned (§5.1) |
| **one free flag** | `use_q` / `normalize_base` — Slow-Brain measured that **hiding the planner's own scores from the selector improved selection**; ⚠️ sidewalk robots, `HYPOTHESIS` tier, worth a flag not a redesign |

---

## 9. WHAT THIS DOES AND DOES NOT LICENSE

**Settled.**
- **The T2 line is not refuted by our corpus** — the pre-registered kill fails by 4.5× at two
  operating points with an interval. R1 remains live.
- **The 181 km/h span is the offset head's**, proven structurally and measured on a matched arm.
- **The kinematic clip is free** (72 % removed, ceiling and pick unmoved) and is R1's precondition.
- **The imagination cannot rank candidates** — structural, verified twice.
- **`ade_0_2s` and a rule verdict are separated in opposite directions on the same windows.**

**NOT settled, and I refuse to conclude it.**
- ⚠️ **This does not show a rule-trained scorer will WIN.** It shows the label carries signal ADE
  does not. Bar A's lesson is exactly that the step from "signal exists" to "a realisable ranker
  extracts it" must be measured. **That measurement is T3, and I did not run it.**
- ⚠️ **No number here is on v4's own fan except v4's `fan_err4` coverage curve.** T0's offset leg,
  T1's clip and every T2 number use REF-C-XL's fan or the anchor vocabulary.
- ⚠️ **No number here is on the parity corpus for T2.** It is a label property; it is not
  cross-arm comparable and must never enter `MODEL_REGISTRY.md` as a model fact.
- ⚠️ **The labeler has a ~2 % false-positive floor** (5.3× PARA-Drive's reference). Absolute
  collision rates are not quotable; the *contrasts* are.
- ⚠️ **DAC/DDC/LK/TL remain unbuildable.** PDMS-lite is missing one of PDMS's two multiplicative
  terms. This bounds how far a composite can adjudicate for us and is a standing argument for the
  OpenDRIVE / Overture lane-graph work.

---

## 10. DELIVERABLE MANIFEST

Repo root `G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD`, folder
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-percandidate-labels/`.
**All STAGED (`git add`). Nothing committed. Nothing pushed. No branch switched. No pod touched.**

| artifact | where | also elsewhere? |
|---|---|---|
| `PERCANDIDATE_LABELS.md` — this document | `repo:…/` | **repo only** |
| `code/pcl_common.py` — loaders, ego kinematics, OBB/SAT overlap, clip aliasing | `repo:…/code/` | repo only |
| `code/t2a_clock_join.py` — the clock-join proof + failing-input self-test | `repo:…/code/` | repo only |
| `code/t0_span_audit.py` — FPS membership proof + span audit | `repo:…/code/` | repo only |
| `code/t2_labels.py` — the per-candidate labeler, kill test, composite, GT control | `repo:…/code/` | repo only |
| `code/t1_clip_and_fansize.py` — kinematic clip + fan-size sweep, both surfaces | `repo:…/code/` | repo only |
| `code/t2b_clip_x_rule.py` — the clip × rule-scorer interaction (prediction committed in-file) | `repo:…/code/` | repo only |
| `code/t2c_intervals.py` — episode-cluster bootstrap on every headline | `repo:…/code/` | repo only |
| `code/t4_imagination_conditioning.py` — source + runtime proof | `repo:…/code/` | repo only |
| `code/compact_dump.py` — repo-size compaction **with a verification pass** (`fan_err` bitwise identical, `pdms_lite` recomputes to 3e-8) | `repo:…/code/` | repo only |
| `raw/t2a_clock_join.json` — δ sweep, per-clip spans, self-test | `repo:…/raw/` | repo only |
| `raw/t0_span_audit.json` — three legs + verdict | `repo:…/raw/` | repo only |
| `raw/anchors_dev256.pt` — the 256 FPS anchors used throughout (42 KB) | `repo:…/raw/` | repo only |
| `raw/t2_labels.json` · `raw/t2_labels_conservative.json` — both operating points | `repo:…/raw/` | repo only |
| **`raw/t2_labels_windows_compact.pt` · `raw/t2_labels_conservative_compact.pt`** (19.8 MB each) — **per-window × per-candidate labels; EVERY rate, ρ, arm and interval in this report was RE-DERIVED from these files after compaction and reproduced exactly** | `repo:…/raw/` | repo only |
| `raw/t2c_intervals.json` — bootstrap intervals + paired contrasts | `repo:…/raw/` | repo only |
| `raw/t1_clip_fansize.json` — clip + sweep, both surfaces | `repo:…/raw/` | repo only |
| `raw/t2b_clip_x_rule.json` — the interaction result | `repo:…/raw/` | repo only |
| `raw/t4_imagination_conditioning.json` — the conditioning proof | `repo:…/raw/` | repo only |

**Exists in only ONE place — flagged per the operating standard:** nothing. Every artifact is in the
repo. The inputs that are *not* staged are the raw PhysicalAI-AV probe cache
(`C:/Users/Admin/AppData/Local/Temp/claude/pai_probe_cache`, gated-confidential, re-downloadable) and
the dev-box episode caches — deliberately, on both licence and size grounds.

⚠️ **The index contains sibling agents' work.** Per CLAUDE.md's git-hygiene rule this is recorded so
whoever commits knows the index is shared: **I staged only the files listed above and did not
commit, amend, push or switch branch.**

---

## 11. ESCALATIONS — things that must not sit in a file

1. **⭐ R1 IS LIVE AND ITS PRECONDITIONS ARE NOW PRICED.** T2's kill failed by 4.5×; the rule label
   discriminates inside 82 % of windows and is nearly orthogonal to ADE; and the ADE-oracle is
   separated-*worse* on collisions than the rule pick. **T3 is the next spend and it is ~13 GPU-min
   — but it must run where Bar A's feature cache lives (`tanitad-eval:/workspace/_bara/`).**
   Pre-registration is in §8; it needs no further design work.
2. **⛔ ADOPT THE COMPOSITE BEFORE THE NEXT SCORER DECISION.** §5 measures our two candidate targets
   as **separated in opposite directions on identical windows**. Any scorer adjudicated on
   `ade_0_2s` alone from here is adjudicated on the surface that prefers the more dangerous pick.
   PDMS-lite (no-map) is implemented and CPU-only; **DAC is missing and that is a real ceiling.**
3. **⭐ A ONE-LINE, ZERO-RISK CODE CHANGE IS AVAILABLE TODAY: clamp the candidates to the head's own
   reachable band.** It deletes **72.08 %** of the fan, moves the ceiling and the pick by **exactly
   zero**, and makes any per-candidate imagination roll **3.6× cheaper**. `FlagshipV15Head.select`
   already computes `reach`; it is applied to the goal and not to the candidates.
4. **⭐ THE IMAGINATION CONDITIONING BUG (§7) SHOULD BE FILED AS A BUG, NOT A RESEARCH ITEM.** 32
   tokens serve 256 candidates. Until it is fixed, **no imagination-selection result on this head is
   interpretable**, including any future one.
5. **⚠️ A BRIEF PREMISE IS CORRECTED: `egomotion` and `obstacle.offline` share the SAME clock AND the
   SAME origin** (δ = 0.0 s, 93.3 % of clips individually). The hazard is the **span** — **2.0 % to
   59.2 % (median 21.1 %)** of `egomotion` rows lie *outside* the labelled 20 s clip. Any future
   ingest must window to `[0, 20] s` explicitly.
6. **⚠️ OUR LABELER FLAGS THE HUMAN AT 2.05 %, 5.3× PARA-Drive's 0.384 % reference.** Before any
   absolute safety number is quoted from it, it needs a calibration pass (box padding, the at-fault
   proxy, track interpolation). The *contrasts* in §5 are robust to it; the *levels* are not.
7. **⚠️ `obstacle.offline` is a large, unread asset and this run used ONE CHUNK OF IT.** 96 clips
   gave 860 k verdicts and mean 53.6 agents/window. Wiring it into the episode build (our ingest
   still reads 4 of 36 features) is the enabling step for R1 at parity scale — and it is a Data
   Engineering task, not an Architecture one.

---

## 12. FOR `Project Steering/RETRACTION_LOG.md` — root-cause classes

**I did not edit the log; appending to an append-only steering file is the orchestrator's call.**

### R-1 — **C-new: a premise inherited from a brief, wrong in its stated mechanism**

> **The claim:** *"`egomotion` spans ~140 s while `obstacle.offline` spans ~20 s **with different
> origins**"* — carried in the brief with a ×5.7 warning attached.
> **Measured:** the origins are **identical** (δ = 0.0 s; the p10 static-track dispersion is
> 0.0639 m at δ = 0 and 0.3166 m at δ = +1 s). The real hazard is that **2.0–59.2 % (median 21.1 %)
> of `egomotion` rows lie outside the labelled clip**.
> ⚠️ **And this correction is itself a second-order instance of the same class:** my own first draft
> quoted "18.5–34.9 %" from the **four clips I had sampled while writing the loader**, not from the
> 30-clip run that produced the verdict. Caught by re-deriving every quoted figure from its JSON
> before staging. A number that entered a document from a smoke run is a stale number even when the
> real run agrees with its conclusion.
> **Root cause:** a correct *warning* acquired an incorrect *mechanism* in transmission. The
> warning saved the work; the mechanism would have made me apply a nonexistent offset.
> **Rule this earns: a hazard flag is not a hazard model. Prove the mechanism before compensating
> for it — and design the proof so it can return "no offset".**

### R-2 — **C3 (mechanism instead of measurement), avoided — the fp32 membership check**

> **Nearly claimed:** *"the anchors are NOT exact pool members"* (max distance 0.0442 m), which
> would have weakened T0's structural leg into a statistical one.
> **Caught by:** re-deriving in float64 with a bitwise comparison — **256 / 256 exact, L∞ = 0.0**.
> **Root cause:** `torch.cdist` in fp32 over a 40-dim vector with ~38 m entries carries ~4e-2 of
> rounding. This is **Bar A's R-2 in a new place**: in a system whose findings turn on near-ties and
> exact identities, **storage and arithmetic precision are scientific parameters.**

### R-3 — **C-new: the metric we adjudicate on prefers the more dangerous pick**

> **The pattern:** every selector result in this program — Bar A, REF-C v1.2's 47 arms, E-V5-1 — was
> adjudicated on `ade_0_2s`. **Measured here:** on identical windows, the ADE-optimal pick collides
> at **3.36 %** and the rule-optimal pick at **0.71 %**, paired Δ **−0.0265 [−0.0432, −0.0128],
> SEPARATED**. The two objectives are not two estimates of one thing.
> **Rule this earns: a scorer may not be adjudicated on a metric that is blind to the failure the
> scorer exists to prevent.** The composite is a prerequisite, not a follow-up — and it was called
> that in the research a day before this was measured.
