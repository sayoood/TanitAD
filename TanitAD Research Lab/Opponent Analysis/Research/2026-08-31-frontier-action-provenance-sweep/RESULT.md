<title>RESULT — one frontier model has a real command channel, and it also diagnosed our P5 and fixed it with P4's mechanism</title>

# RESULT — E-LAB-OPP-0831

`TanitAD Research Lab · Opponent Analysis (cross-filed to Benchmarks & Evals) · 2026-08-31`
`0 GPU · 0 spend · nothing downloaded · Thor untouched (k60p30k live).`
`⛔ Three evidence tiers per SPEC §3, never merged in a sentence. raw/QUOTES.md carries the tier`
`and retrieval method of every passage. A ⚠️RELAYED item is a verification target, not a claim.`

---

## 0. THE ANSWER

**Three findings, each landing on a live gate item.**

1. ⭐⭐⭐ **The one frontier driving world model I verified as having a genuine command channel
   is also the one that diagnosed P5 and fixed it with P4's mechanism.** `2606.12987` takes its
   actions *"from CAN-bus data as 2D vectors aₜ=(steerₜ, accelₜ)"* — **our exact
   parameterisation, from the bus** — is *"genuinely action-controllable"* (ρ = 0.81 vs −0.18
   for a regression baseline), traces *"limited single-pass motion to a shared-present anchor"*,
   and fixes it with a **1.7 M-parameter Δt=4 "jump" model** that recovers **1.02× GT motion
   magnitude** where single-pass models capture *"less than half"*. **P2, P4 and P5 converge on
   one paper, and the thing that fixed it was a temporally abstract step — not a bigger model.**
2. ⭐⭐ **NAVSIM's scoring basis has moved twice, and one of the moves names our pinned split.**
   The official README records **2025-04-28 (v2.2)**, *"Fixed bug in `openscene_meta_datas` for
   `navhard` and `warmup`… please re-download and use the new data"*, and **2025-09-29**, a fix
   to *"metric filtering where `multiplicative_metrics_prod` and `weighted_metrics` were not
   correctly excluded by the human filter."* On 2026-08-29 this programme **pinned
   navhard-two-stage EPDMS** as a GO target. ⇒ **an EPDMS number is only comparable to another
   EPDMS number on the same side of both dates**, and the second fix touches exactly the
   human-forgiveness machinery that `2608.04896` audits.
3. ⛔ **Our own 2026-08-30 novelty claim is AT RISK and I am not resolving it today.** We wrote
   that *"a HOLD-ACTION FLOOR the model must beat"* is *"not found in any driving world model."*
   The sweep surfaced a driving model reported to run a counterfactual-scaling protocol with an
   unaltered-ground-truth arm. **That report is RELAYED. Per SPEC §3 a novelty claim may not be
   upheld or retracted on relayed evidence** ⇒ it is flagged for verification, with the source
   named, and the paper must not restate the claim until someone has read it.

---

## 1. F1 ⭐⭐ — action provenance: one verified genuine command, and an honest limit on the field-wide half `[VERIFIED + RELAYED, kept apart]`

**H-LAB-OPP-1 — SUPPORTED at the verified tier, UNRESOLVED at the field-wide tier.**

| model | action channel | provenance | tier |
|---|---|---|---|
| ⭐ `2606.12987` DiT World-Action Model (2026-06-11, nuScenes) | `aₜ = (steerₜ, accelₜ)` | *"extracted from **CAN-bus data**"* — **independent of the realised path** | ⭐ **VERIFIED, full text** |
| `2503.20523` GAIA-2 (2025-03-26) | *"ego-vehicle dynamics"* among *"structured inputs"* | ⛔ **the abstract does not state the parameterisation at all** | ⭐ VERIFIED that it is *absent*, not that it is derived |
| ~18 further frontier driving WMs | reported as ego-log-derived (trajectory / speed / curvature / waypoints) | — | ⚠️ **RELAYED — R8, unresolved** |
| **ours** (`physicalai-train-e438721ae894`) | `(steer_road_rad, accel_mps2)`, `steer = atan(wheelbase × curvature)` | **derived from the realised path**, documented in `physicalai.py` as a *"road-wheel angle proxy"* | **MEASURED** (E-DEC-57, gate §P2(b)) |

⭐ **The verified part is already decision-relevant:** a frontier model exists that uses **our
exact 2-D `(steer, accel)` form** but reads it off the bus. That is the cleanest possible
one-variable comparison for P2(b) — same form, different provenance — and it is on **nuScenes**,
a corpus this programme already handles.

⛔ **The part I will not claim.** The attractive headline — *"our action channel is the field
norm, so P2(b) is not our defect"* — rests on the ~18-model half, which is **RELAYED**. And V5
shows why that matters: **GAIA-2's abstract does not state its action parameterisation at all**,
so "everyone uses derived actions" cannot be established by skimming abstracts. ⚠️ **R8 is the
highest-priority verification target in this package**, and until it is done the field-norm
sentence must not enter the paper or a claim row.

## 2. F2 ⭐⭐⭐ — the convergence: P2, P4 and P5 meet in one paper `[VERIFIED]`

`2606.12987`, all quotes verified (V1):

| our gate item | their finding |
|---|---|
| **P5 / L3** — *"the predictor is not transporting the scene, it is restating it"* | *"We trace limited single-pass motion to a **shared-present anchor**"* — the same pathology, named in a **driving** world model, with the deficit measured: single-pass models capture *"less than half"* of ground-truth motion magnitude |
| **P4** — strategic horizon needs *"temporal abstraction… not a longer flat rollout"* | the fix is a **Δt=4 jump transition** `z_t → z_{t+4}` *"conditioned on the four intervening actions"*, applied as *"a 4-step open-loop chain, re-anchoring on its own output at each step"* to reach 8 s |
| **P2** | the same model is *"genuinely action-controllable"*, ρ = 0.81 |
| ⭐ cost | the jump model is **1.7 M parameters** (n_blocks = 2, dim 192) |

⭐⭐ **The reading that matters: the fix for "restating the present" was NOT a better action
channel and NOT more capacity — it was a temporally abstract step, at 1.7 M parameters.**
`[INFERRED transfer]` Our own defect has the matching shape: **88.7 % of our oracle gap is
longitudinal**, i.e. along-track *magnitude*. ⚠️ That is a shape match and an argument for
attention, **not** a measurement about our arms.

⚠️ **Two limits.** (i) It is a **latent diffusion** model with an x₀ objective and residual
anchoring — **not** a teacher-forced latent-regression JEPA, so it does not sit inside the
mechanism the Architecture package identifies, and it changes several things at once.
(ii) 150 held-out nuScenes scenes at 256 × 256, 2 Hz — a compact-scale study by its own
description.

⭐ **A third thing worth carrying to Benchmarks & Evals** — verified, and it re-states our own
doctrine from outside: *"distortion metrics (cosine similarity, SSIM) favor the blurry mean,
masking that the diffusion model is far closer to the real frame distribution"*, with
**KID 0.078 vs 0.375 (4.8×)**. A metric that rewards the mean is the video-domain twin of an
ADE that rewards regression-to-mean.

## 3. F3 ⛔ — our anti-echo novelty claim is AT RISK, and I am deliberately not resolving it `[claim status: FLAGGED]`

**H-LAB-OPP-2 — NOT RESOLVED, by design.**

Recorded on 2026-08-30 (`2026-08-30-anti-echo-control-precedent`): *"⭐ Novel at five probes:
(a) a **HOLD-ACTION FLOOR the model must beat** — not found in any driving world model."*

What the sweep changes:

| item | status | effect on the claim |
|---|---|---|
| `2606.12987` steering sweep with a **regression baseline** at ρ = −0.18 | ⭐ VERIFIED | ⚠️ a **comparison baseline**, not a floor the model must beat. Our claim survives this one — but it is now closer than "not found in any". |
| **Orbis 2** counterfactual scaling ×0.5/×1.5 against an **unaltered-GT arm** (R1) | ⚠️ **RELAYED** | ⛔ **could refute the claim, or could turn out to be a null arm rather than a floor.** Only reading `2607.15898` can tell. |

⛔ **SPEC §3 forbids me from settling this on relayed evidence, and I am holding to that.**
The output is: **the claim is FLAGGED, the source is named, and the paper must not restate it
until `2607.15898` has been read.** ⭐ Leaving a novelty claim standing because checking it was
inconvenient is precisely the failure this discipline exists to prevent — and so is retracting
it on a second-hand sentence.

## 4. F4 ⭐⭐ — NAVSIM comparability: two dated scoring breaks, one of them on our pinned split `[VERIFIED, official doc]`

**H-LAB-OPP-4 — SUPPORTED.** From the official NAVSIM README, fetched today (V4):

| date | change | why it binds us |
|---|---|---|
| **2025-04-28** (v2.2) | *"Fixed bug in `openscene_meta_datas` for `navhard` and `warmup`—If you used `navhard_two_stage/openscene_meta_datas` … please re-download and use the new data."* | ⭐⭐ **names `navhard_two_stage` explicitly** — the split pinned as our GO target on 2026-08-29 |
| **2025-09-29** | *"Fixed a bug in metric filtering where `multiplicative_metrics_prod` and `weighted_metrics` were not correctly excluded by the human filter"* | ⭐ the **human filter** is the reference-conditioned-forgiveness machinery; a number computed either side of this date is not the same quantity |
| latest release named | **v2.2, 2025-04-28** | ⇒ **no NAVSIM v3**; the protocol has been stable in *version* while changing in *scoring basis* — which is the trap |

⇒ ⭐ **A COMPARABILITY RULE, and it is cheap to enforce:** any EPDMS we quote or compare
against must carry **the NAVSIM commit/date it was computed at**, not merely the split name.
Two papers both saying "navhard EPDMS" on opposite sides of 2025-09-29 are not comparable, and
nothing in the version string reveals it.

**And the independent audit, `2608.04896` (2026-08-05), verified at abstract:** under
*"reference-conditioned forgiveness, under which an agent receives credit when the logged human
reference fails a compliance channel"*, *"the route-blind Ignore-All probe and a route-aware
actor-blind probe **outrank human replay and PDM-Closed** over the complete 12,146-token
navtest split."*

⚠️ **Scope it exactly — the caveats are load-bearing and are in their own sentence:** *NAVSIM
v2.2*, *original scene*, ***single-stage*** *scoring*, *"the affected documented-stack
condition"*, *"the audited numerical backend"*. ⇒ **This does NOT automatically indict our
pinned navhard-TWO-stage target** — different split, different staging. ⛔ **And the specific
numbers circulating for this result (a probe at 79.6 against human replay at 74.0) are NOT in
the abstract; I did not open the full text. UNVERIFIED — do not quote them.**

⭐ **What it does establish for us regardless of staging:** a benchmark whose forgiveness rule
can let a **blind** probe outrank human replay is structurally the same failure our own doctrine
names on the model side — *a score that a policy ignoring the input can achieve is not measuring
the capability it is named for.* That is the **external precedent for our floor-arm doctrine**,
and it is a better citation than anything we had.

## 5. F5 / F6 — the two highest-value RELAYED items, recorded as targets not findings `[⚠️ RELAYED]`

- **R2 — Orbis 2 (`2607.15898`) is reported to run a two-rate hierarchy** (an abstract latent at
  ~2 Hz over a ~10 Hz low level, ≈5× stride). ⚠️ If it holds, it is **P4's first driving-domain
  precedent** and a published stride to compare against MM-E16's 1 : ~3 : ~15. **Not a finding
  today.**
- **R3 — ReSim (`2506.09981`) is reported to attribute hallucination under unseen actions to
  the expert-only state-action space of real data**, and to fix it by importing non-expert
  behaviour. ⚠️ If it holds, it is a **third P2 lever — action-space COVERAGE** — neither the
  channel (b) nor the target (d), and it connects directly to L2D's native **13.8 % STUDENT
  (learner-driver)** split found by today's Data Engineering package. **Not a finding today.**

**H-LAB-OPP-3 — UNRESOLVED.** A driving-domain temporal-abstraction precedent is *reported*
(R2) and one is *verified but small* (F2's Δt=4 jump model, 1.7 M params, 8 s). The strong
form — a driving world model with an explicit multi-rate hierarchy — awaits R2's verification.

## 6. ⛔ ESCALATION

1. ⭐⭐ **BENCHMARK COMPARABILITY RULE — for the EvalFlyWheel to adopt.** Every EPDMS/PDMS number
   we quote, ours or theirs, carries **the NAVSIM version AND the changelog date it was computed
   at**. Two dated scoring breaks exist and one names `navhard_two_stage`. ⚠️ Our 2026-08-29
   pinned target should be re-stated with its date basis; the pin itself is not challenged.
2. ⛔ **NOVELTY CLAIM ON HOLD.** `2607.15898` must be read before the paper restates *"a
   hold-action floor… not found in any driving world model."* Assigning this to nobody is how it
   ships unchecked.
3. ⭐ **VERIFICATION QUEUE, ranked** — R8 (the field-norm half) · R1 (the novelty threat) ·
   R2 (P4 precedent) · R3 (the third P2 lever) · R7 (split-dependent EPDMS deltas). Each is
   reading, no compute.
4. ⭐ **`2606.12987` deserves a full read by Architecture & Inference**, not a citation. It is
   the closest published analogue to our three open gate items simultaneously.

## 7. Hypothesis verdicts

| ID | verdict |
|---|---|
| **H-LAB-OPP-1** | ⭐ SUPPORTED at the verified tier (1 genuine-command model found); ⚠️ **UNRESOLVED** field-wide — R8 |
| **H-LAB-OPP-2** | ⛔ **NOT RESOLVED, by design** — flagged for verification, not settled on relayed evidence |
| **H-LAB-OPP-3** | ⚠️ UNRESOLVED — one small verified precedent (Δt=4 jump), the strong form awaits R2 |
| **H-LAB-OPP-4** | ⭐ **SUPPORTED** — two dated scoring breaks in the official changelog, one naming `navhard_two_stage` |

## 8. Deliverable manifest

| artifact | where | only-one-place? |
|---|---|---|
| `SPEC.md` · `PLAN.md` · `RESULT.md` (this) · `COMMS.md` | `repo:TanitAD Research Lab/Opponent Analysis/Research/2026-08-31-frontier-action-provenance-sweep/` | no — staged |
| `raw/QUOTES.md` (tiered passages + the RELAYED verification queue) | same dir | no — staged |
| 9 banked PDFs (`2606.12987` `2607.22535` `2608.04896` `2607.15898` `2506.09981` `2607.22430` `2608.06706` `2511.23369` `2607.05133`) | `repo:TanitAD Research Lab/Library/papers/` + `library.json` | no — staged |
| KB lines | `repo:.../Opponent Analysis/Research/KNOWLEDGE_BASE.md` **and** `.../Benchmarks & Evals/Research/KNOWLEDGE_BASE.md` | no — staged |

**No code changed. No commit, no push, no branch switch. `Mission Plan.md` untouched.**
