# The driving gate's floor cannot turn — CTRV re-adjudication on the canonical val40

**Benchmarks & Eval · 2026-08-02 · MEASURED**
Artifacts: `Implementation/incoming/2026-08-02-ctrv-floor/` (module + 11 tests + prereg + driver +
`raw/ctrv_readjudication.json` 1.3 MB + `raw/ctrv_run.log`).
Resource: **eval pod `tanitad-eval` (A40), CPU-only, 372.9 s**, $0 marginal (standing pod), GPU untouched.
Estimator: **paired episode-cluster bootstrap**, B = 2000, seed 0, unit = val episode; orientation
`floor − model`, positive = model wins. `separated` = CI excludes zero.
Surface: `physicalai-val-0c5f7dac3b11`, **881 windows / 40 episodes**, window 8, stride 8, K = 20 @ 10 Hz.

---

## 0. The one-line finding

**`taniteval/driving.py:304` — `FLOORS = ("cv", "holdv0")` — contains only straight-line predictors,
and CTRV, which is already computed on every window and thrown away, is a *better* trivial floor than
both: ADE 0.5265 m vs CV 0.8377 / hold-v0 0.7876, paired +0.3113 m [0.167, 0.484] separated, and it
wins 423 of 881 windows outright.** Re-scoring the 25 banked arms against it moves **16 of 25 headline
verdicts**, including the deployed flagship-v1's, which drops from *beats the floor* to a **tie**.

---

## 1. What was wrong, and why it was invisible

`stack/scripts/driving_diagnostic.baseline_waypoints()` returns **three** trivial predictors —
`constant_velocity`, `go_straight`, `constant_yaw_rate`. `taniteval/rollout.collect` persists only
`constant_velocity` (`rollout.py`, the `CV.append(...)` line); `driving.py` then adds `hold_v0`,
derived from the persisted `speed` channel. So the canonical tier-0 driving block scores every arm
against **two straight lines**, and CTRV — same information budget (`poses[last]`, `poses[last-1]`, no
future, no privileged state), already computed, zero extra compute — never reaches it.

This was already measured on other corpora and never carried onto the decision surface:
`incoming/2026-07-15-baseline-floor` found CTRV winning **55–58 %** of 26 132 anchors and CV
overstating the floor **4.6× on curves**. `LEADERBOARD.md` §0 even lists a *"CTRV oracle 0.523"* row —
but sourced `INHERITED` from `MODEL_REGISTRY` §0.3, on a different corpus, with no interval and no
per-stratum form. Nothing paired it against a model on the 881 windows the gate decides on.

The community reference is stricter than ours in exactly this direction: the **nuScenes prediction
`PhysicsOracle`** selects the best of **four** dynamics models — constant velocity+yaw, **constant
velocity+yaw rate (CTRV)**, constant acceleration+yaw, **constant acceleration+yaw rate** — precisely
so that a learned model cannot bank credit for turning ([nuscenes-devkit prediction
README](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/README.md),
[tutorial](https://www.nuscenes.org/tutorials/prediction_tutorial.html)). Our floor family is a
best-of-two subset of that, with **both yaw-rate members removed**.

## 2. The instrument precondition, measured before any verdict (C63)

The floors are **backfilled** onto banked `windows_<arm>.pt` dumps, so everything rests on window *i*
here being window *i* there. `ctrv_floor.verify_alignment` measures it rather than assuming it: the
rebuilt `cv` and `gt` must equal the tensors `rollout.collect` persisted.

**MEASURED: `max_abs_diff_cv = 0.0` and `max_abs_diff_gt = 0.0` — bit-exact — on 25 of 27 dumps.**
The two refusals are the 88-window `refc-v12-smoke-*` partial dumps (correctly refused; no verdict is
reported for them).

⚠️ **A harness inconsistency surfaced on the way.** `windows_flagship-v4.1-10k.pt`,
`windows_flagship-v4.2-step4000.pt` and `windows_flagship-v16-ab-ft.pt` label the **same 40 episodes**
with `eid = 808464434, …` — the episode's string uid reinterpreted as an int — where every other dump
uses `0..39`, while their `gt`/`cv`/`speed` tensors are bit-identical to the rest. My first alignment
check required literal `eid` equality and **refused three provably-aligned arms**. Fixed: the check now
compares the *partition* (which is what a cluster bootstrap actually depends on) and reports
`eid_labels_equal` separately. ⛔ **Any code that joins arms across dumps on `eid` will mis-join these
three.** Flagged to the orchestrator (§7).

## 3. The floor itself (model-independent, so it holds for every arm)

| floor | `ade_0_2s` m | `fde_2s` m | `miss_2m` | `lat_abs_2s` m | heading MAE° | wins/881 |
|---|---:|---:|---:|---:|---:|---:|
| constant velocity (CV) — *the incumbent* | 0.8377 | 1.7406 | 0.3042 | 1.0089 | 6.623 | 156 (18 %) |
| hold-v0 — *the incumbent* | 0.7876 | 1.6521 | 0.2917 | 0.9137 | 6.344 | 302 (34 %) |
| **CTRV (speed-gated 2 m/s)** — **missing** | **0.5265** | **1.1272** | **0.1896** | **0.3741** | **3.679** | **423 (48 %)** |
| CTRV (ungated) | 0.5230 | 1.1194 | 0.1896 | 0.3667 | 3.333 | — |
| *best-of-3 **oracle*** (privileged, per-window min) | *0.4820* | — | — | — | — | — |

Paired, model-free: **CTRV − CV = +0.3113 m [0.1674, 0.4844] separated**; **CTRV − hold-v0 = +0.2611 m
[0.1419, 0.4061] separated**. The gate costs a hair on this corpus (gated − ungated = −0.0034 m
[−0.0085, −0.0003]) — i.e. the standstill yaw-noise artifact that motivated the gate is *small on
val40*, so the finding is not a gate artifact either way. The pre-registered primary stays the **gated**
form; the ungated number is 0.7 % better and would only strengthen everything below.

⚠️ *best-of-3 is an **oracle*** — it picks the winner per window using the ground truth, exactly like
nuScenes' PhysicsOracle. It is a reference, never a competitor, and it is labelled as such here because
the previous `0.5005` best-of-3 row was quoted without that label.

## 4. Verdict re-adjudication — 25 arms, headline `ade_0_2s`

| vs CV → vs CTRV | n | arms |
|---|---:|---|
| beats floor → **beats floor** | 6 | `flagship-v16-ab-ft`, `refc-base-30k`, `refc-v12`, `refc-v12-identity`, `refc-v12-k16reg`, `refc-xl-30k` |
| beats floor → **TIE** | 3 | ⭐ **`flagship-30k` (v1, deployed)**, `flagship-speed`, `refc-xl-live` |
| beats floor → **LOSES to floor** | 3 | `refb-v2-30k`, `refb-v2-20k`, `refc-xl` |
| tie → **LOSES to floor** | 4 | `refb`, `refb-10k`, `flagship-v4.1-10k`, `flagship-v4.2-step4000` |
| loses → loses | 9 | the REF-A / no-speed / v2-6k / v3enc family |

**16 of 25 verdicts move.** Under CV, 12 arms "beat the trivial floor"; under CTRV, **6 do**, and the
best surviving margin in the entire fleet is **+0.0890 m**. Two of the six (`refc-v12-identity`,
`refc-xl-30k`) are separated only at `lo = +0.0001` — **marginal, and must be reported as such.**

⭐ **flagship-v1 @30k:** ADE 0.4271 (matches `MODEL_REGISTRY` §6 exactly). vs CV **+0.4106 separated,
favours model**; **vs CTRV +0.0993 [−0.0258, +0.2204] — NOT separated.**
⇒ **`PROJECT_STATE`'s "the FIRST arm below EVERY trivial bar" is a point-estimate statement.** With the
program's own decision-grade estimator, paired on the same windows, flagship-v1 is **statistically tied
with a constant-turn-rate extrapolation** on the headline metric. It is *not* refuted — 0.4271 < 0.5265
— but it is not separated, and the program's own rule ("a claim that decides a GPU-day must be
MEASURED", plus the CI-separation predicate) means the claim must be restated, not repeated.

## 5. Pre-registered verdict: **H-REAL** — and the magnitude is still ~5× smaller

`PREREGISTRATION.md` committed both outcomes and named three flip criteria on `flagship-30k`:
`sustained_turn` ADE, `curv_sharp` heading MAE, overall `lat_abs_2s_m`. **None of the three flipped.**

| stratum / metric | n | model | CV | CTRV-g | vs CV | vs CTRV | shrink |
|---|---:|---:|---:|---:|---|---|---:|
| `sustained_turn` ADE | 142 | 0.5061 | 2.3124 | 0.8460 | +1.8063 model | **+0.3398 [0.153, 0.550] model** | **5.3×** |
| `sustained_turn` lateral | 142 | 0.3462 | 3.9126 | 0.9156 | +3.5664 model | **+0.5694 [0.334, 0.804] model** | **6.3×** |
| `curv_sharp` heading° | 144 | 3.81 | 28.74 | 11.50 | +24.93 model | **+7.69 [4.75, 11.09] model** | **3.2×** |
| overall `lat_abs_2s_m` | 881 | 0.2369 | 1.0089 | 0.3741 | +0.7720 model | **+0.1372 [0.026, 0.252] model** | **5.6×** |

⇒ **H-ARTIFACT is refuted where I registered it. The model really does turn better than the best
trivial predictor.** But **~81–84 % of the published margin was floor artifact**: `verdict.
where_the_win_lives = "lateral only"` survives as a *direction*, and must never again be quoted with a
CV-derived magnitude.

## 6. What the CTRV floor reveals that CV structurally could not (exploratory, NOT pre-registered)

These flips were not registered in advance and are reported as hypothesis-generating, not as results.

| scope | metric | n | vs CV | vs CTRV |
|---|---|---:|---|---|
| overall | `fde_2s` | 881 | +0.8330 model | **+0.2196 tie** |
| overall | `heading_med_2s_deg` | 881 | tie | **floor** (−0.6456) |
| `speed_high` | `ade_0_2s` | 294 | tie (+0.0954) | **floor −0.2154 [−0.386, −0.030]** |
| `speed_top10pct` | `ade_0_2s` | 89 | floor −0.4156 | **floor −0.6173 [−0.782, −0.459]** |
| `speed_top10pct` | `lat_abs_2s_m` | 89 | tie | **floor −0.3837 [−0.543, −0.156]** |
| `speed_top10pct` | heading MAE° | 89 | tie | **floor −1.190 [−1.679, −0.598]** |
| `speed_top10pct` | `pathgeom_crosstrack_m` | 89 | tie | **floor −0.1476 [−0.210, −0.060]** |

⭐ **At the top speed decile CTRV's ADE is 0.0986 m and the model's is 0.7159 m — 7.3× worse — and the
model now loses LATERALLY too, not only longitudinally.** The known high-speed weakness
(`memory/flagship-longitudinal-lever`, `MODEL_REGISTRY` §4.1: 89 % of 2 s sq-error along-track) has
always been framed as a *longitudinal* problem, because a straight-line floor cannot expose a lateral
one on a road that is locally an arc. **The mechanism is coherent:** at high speed the geometry is
close to constant-curvature over 2 s, so CTRV is nearly exact and the model's residual is fully
exposed. This does not overturn the longitudinal reading — it says the high-speed regime is where the
model adds the least of *any* kind, and it is where the trivial predictor is strongest.

⚠️ **`stop_approach` heading MAE reads 102.79° for the model vs ~8.4° for both floors.** At a stop the
2 s displacement is near zero, so a heading derived from it is dominated by noise. This is an
**instrument caveat, not a capability claim** — it is exactly why `driving.py` marks `median` as the
honest reducer for heading (R5). Do not quote the 102.79°.

## 7. Actionable, in priority order

1. ⭐ **Add `ctrv` to the floor family** — intake `2026-08-02-ctrv-floor` (§INTAKE): persist
   `baseline_waypoints(...)["constant_yaw_rate"]` in `rollout.collect` alongside `cv` (zero extra
   compute — it is already computed and discarded), then `FLOORS = ("cv", "holdv0", "ctrv")` with a
   `win.get("ctrv")` fallback so the 25 legacy dumps keep working. Every stratum, paired test and
   verdict in `driving.py` then recomputes against three floors automatically.
2. ⭐ **Restate the "below every trivial bar" claim** in `PROJECT_STATE.md` / `MODEL_REGISTRY` §0.3 —
   ⛔ *escalated, not edited: those are Project Steering files.* The honest form is: *flagship-v1's
   point estimate (0.4271) is below the CTRV floor (0.5265), but the paired interval is NOT separated
   (+0.0993 [−0.026, +0.220]); it IS separated against CV and hold-v0.*
3. **Upgrade `LEADERBOARD.md` §0's floor rows from INHERITED to MEASURED** — done this run (§0 now
   carries CTRV recomputed on the 881 windows with its interval and win count, and the best-of-3 row is
   labelled an oracle).
4. **Fix the `eid` encoding split** (§2) — three dumps use packed string uids. One-line normalisation
   at write time in `rollout.collect`; until then, no cross-arm join may key on `eid`.
5. **Re-run the v4/v5 30k gates with the third floor before the next gate verdict.** The v4 30k gate
   (`incoming/2026-07-28-v4-30k-gate`) published `sustained_turn +1.2416 favours model` and
   `where_the_win_lives = "lateral only"` against the two-floor family; both need the CTRV column
   before they are quoted again.
6. **Propose the midpoint-corrected CTRV as a *stronger* floor** (P1, not this run). The shipped arc is
   forward-Euler and carries a measured **0.3044 m half-step bias on a perfect 1 rad arc**
   (`tests/test_ctrv_floor.py`) — so every margin above is a *lower bound* on the artifact.

## 8. Evidence classes

| claim | class |
|---|---|
| every number in §3–§6 | **MEASURED** — `raw/ctrv_readjudication.json`, eval pod, 372.9 s |
| CV floor 0.8377 / model 0.4271 reproduce the banked gate + registry exactly | **MEASURED** (bit-exact alignment, §2) |
| nuScenes PhysicsOracle is best-of-4 incl. two yaw-rate models | **PUBLISHED** (nuscenes-devkit README / tutorial) |
| CTRV wins 55–58 % on 26 132 anchors (other corpora) | **MEASURED**, `incoming/2026-07-15-baseline-floor` |
| "`0.523` CTRV oracle" in the previous LEADERBOARD §0 | **INHERITED** — different corpus, no interval; superseded by §3 |
| high-speed mechanism in §6 (locally-constant curvature) | **HYPOTHESIS** — consistent with the data, not tested |

**Sources:** [nuScenes prediction README](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/README.md) ·
[nuScenes prediction tutorial](https://www.nuscenes.org/tutorials/prediction_tutorial.html) ·
[Is Ego Status All You Need? (arXiv 2312.03031)](https://arxiv.org/abs/2312.03031) ·
[Open-loop ⊥ closed-loop (arXiv 2605.00066)](https://arxiv.org/abs/2605.00066)
