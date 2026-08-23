# Closed-loop artifact re-run — every circulating interval AND mean, corrected

**Date:** 2026-07-26 · **Compute:** eval pod (`tanitad-eval`, A40, idle) ·
`OMP_NUM_THREADS=8` · **pod1 / pod2 / pod3 untouched** ·
**Nothing `git add`ed, nothing committed, nothing pushed.**

**The standing item this closes:** `closedloop.py` was migrated to
`ci.episode_cluster_bootstrap` on 2026-07-25, but W1-A flagged that *"no
closed-loop artifact has been re-run, so EVERY circulating closed-loop interval
AND mean is still legacy."* This re-runs them.

---

## 0. Harness check — PASSED before any new number was quoted

| check | result |
|--|--|
| **v1 reproduces `0.4271`** | ✅ `0.4271`, CI `[0.3675, 0.4871]`, ci95 `0.0598`, 881 win / **40 val episodes**, B=2000 — **exactly** the pinned triple in `Project Steering/CI_RECOMPUTE_2026-07-20.json` |
| deprecated statistic reproduces its published value | ✅ `0.4522 ± 0.0312` via `ci.overlapping_holdout_se` over `gates.split_by_episode` — so **both sides** of every comparison below are reproducible |
| point-estimate shift on that arm | **+5.868 %** (legacy inflated) · CI width ratio **1.916×** |
| `lateral.py` horizon fix present | ✅ `horizon_provenance` stamped, `horizon_s = 2.0 s` (**not** 0.4 s) |

**⚠️ The eval pod was NOT fully synced.** Of 38 `taniteval` modules, **37 were
byte-identical to the repo and `lateral.py` was NOT** — the pod carried the
**pre-fix** version with no `horizon_provenance` and the 5× sparse-horizon
mislabelling still live. It was replaced (md5 `a3b3d491…`) and `__pycache__`
cleared **before** any decomposition ran. The 141/141 sync predates the
lateral.py fix; anything that ran on this pod between the sync and now used
stale lateral code.

Artifact: `harness_check.json` · script: `harness_check.py`

---

## 1. Inventory — which closed-loop artifacts exist, and can they be recomputed?

**The structural finding: `closedloop.run_and_save` never persisted the
per-window paths.** `collect()` builds `closed_bike / closed_grnd / open_grnd /
open_bike / cv / gt` as `[N,4,2]` tensors and `analyze()` reduces them to
scalars — only the scalars were written. The open-loop panel has
`windows_<arm>.pt` dumps and is re-scorable offline; **the closed-loop panel had
no such surface.** That is why this was a real gap and not bookkeeping: the
deprecated statistic could not be undone by arithmetic on the artifact, the loop
had to be re-driven on a GPU.

**Fixed going forward:** every re-run here also writes `clwin_<arm>.pt` /
`p2win_<arm>.pt`. The next estimator change on this axis is a CPU recompute in
seconds, not a GPU re-run.

| # | artifact | statistic that produced it | raw data? | action |
|--|--|--|--|--|
| A1 | `…/2026-07-19-alpasim-closedloop-v1/results/closedloop_flagship-30k.json` | `overlapping_holdout_se`, self-labelled *"8-split episode-disjoint jackknife (bench.py)"* | ❌ none | **RE-RUN** ✅ |
| A2 | `…/closedloop_flagship-speed.json` | same | ❌ | **RE-RUN** ✅ |
| A3 | `…/closedloop_flagship-nospeed.json` | same | ❌ | **RE-RUN** ✅ |
| A4 | `…/2026-07-22-imagination-closedloop-proof/closedloop_flagship-30k_imagination-proof.json` | **MIXED** — A/B block already `paired_episode_cluster_bootstrap`; all other scalars `overlapping_holdout_se` (the file says so) | ❌ | **RE-RUN** ✅ |
| A5 | `/root/taniteval/results/planner_p2_flagship-30k.json` (**pod-only, not in repo**) | `overlapping_holdout_se`, stamped *"8-split episode jackknife"* — `planner_p2.py` was **never migrated** (`_jack_scalar :373`, `_jack_paired :381`) | ❌ | **RE-DRIVEN** ✅ (see §5) |
| B | `e1a_horizon_*`, `e2a_localize_*`, `e1b_*`, `e1c_*`, `hp3_*`, `regate/*`, `GOAL_MODE_GAP` | `episode_cluster_bootstrap` primary, legacy present only as a **labelled** back-compat block | n/a | **no action** |
| C | `wm_mpc_result*`, `lowood_closedloop`, `lowood_flagship_ci`, `lowood_lanekeep_40ep`, `corridor_*`, `tolerance_rescore`, `powered_departure`, `dagger_result*` | `episode_cluster_bootstrap` / paired, stamped | n/a | **no action** |
| D | `frozen-wm-learned-planner/artifacts/*.json` | **intervals with NO estimator named anywhere in the JSON** | ✅ `perwin_*.pt` | **VERIFIED** — see §6b |
| E | AlpaSim `*_results-summary.json`, `gate1_*`, `gate0_*`, `scenario_stratified_*` | **no interval and no estimator at all** — scene-scored point values | partial | **SUSPECT** — §6 |

---

## 2. Published → corrected (decision-grade)

**Every re-run self-validates.** The migrated `analyze()` still emits the
deprecated numbers under `legacy_overlapping_holdout_se`, computed from the same
`gates.split_by_episode` builder as 2026-07-19. For all three arms the re-run's
legacy block reproduced the published values **18/18 metrics each** — so the
shifts below are **the estimator**, not a new forward pass.

### v1 — `flagship-30k` (THE headline arm; 881 win / 40 ep)

| metric | published (`overlapping_holdout_se`) | corrected (`episode_cluster_bootstrap`) | mean shift | CI ratio |
|--|--|--|--|--|
| **closed_bike ade@2s** | **1.6852 ± 0.0977** | **1.7318 [1.5707, 1.9070]** ± 0.1682 | **−2.69 %** | **1.72×** |
| closed_bike fde@2s | 3.5296 ± 0.2548 | 3.6190 [3.2453, 4.0215] | −2.47 % | 1.52× |
| closed_bike ade@0.5s | 0.2267 ± 0.0086 | 0.2321 ± 0.0171 | −2.33 % | 1.99× |
| closed_bike ade@1s | 0.5838 ± 0.0211 | 0.5995 ± 0.0507 | −2.62 % | **2.40×** |
| closed_bike ade@1.5s | 1.0703 ± 0.0473 | 1.1027 ± 0.1024 | −2.94 % | 2.17× |
| closed_grnd ade@2s | 2.6736 ± 0.1253 | 2.7255 ± 0.2627 | −1.90 % | 2.10× |
| **open_grnd ade@2s** | **0.4522 ± 0.0312** | **0.4271 [0.3675, 0.4871]** | **+5.88 %** | 1.92× |
| **open_bike (bicycle fidelity floor)** | **0.5129 ± 0.0923** | **0.4518 [0.3097, 0.6174]** | **+13.52 %** | 1.67× |
| cv ade@2s | 0.8248 ± 0.1035 | 0.8377 ± 0.2241 | −1.54 % | 2.17× |
| compounding **grounded** Δ@2s | +4.2722 ± 0.2935 | +4.4059 ± 0.5241 | −3.04 % | 1.79× |
| compounding **bicycle** Δ@2s | +2.4774 ± 0.2469 | +2.7008 ± 0.3310 | −8.27 % | 1.34× |
| **divergence >5 m @2s** | **0.2216 ± 0.0431** | **0.2350 [0.1680, 0.3027]** | **−5.70 %** | 1.56× |

**The headline moves the wrong way for the story that was told.** The circulating
claim is *"open-loop 0.452 → closed-loop 1.685"*. Corrected it is
**0.4271 → 1.7318**: the open-loop side was **6 % better** than published and the
closed-loop side **3 % worse**, and the two errors compound in the same
direction. The open-loop→closed-loop collapse the program reasons from is
therefore **larger**, not smaller: **published ratio 3.73× → corrected 4.05×**.
Every doc that says *"open-loop does not predict closed-loop"* is **more** right
than it claimed, not less.

### `flagship-speed` (19k relay) and `flagship-nospeed` (ablation control)

| metric | arm | published | corrected | shift | CI ratio |
|--|--|--|--|--|--|
| closed_bike ade@2s | speed | 1.6564 ± 0.1195 | 1.6915 [1.5220, 1.8723] | −2.08 % | 1.47× |
| closed_bike ade@2s | nospeed | 1.5750 ± 0.0830 | 1.5731 [1.4165, 1.7462] | +0.12 % | 1.99× |
| open_grnd ade@2s | speed | 0.6277 ± 0.0551 | 0.6152 | +2.03 % | 1.39× |
| open_grnd ade@2s | nospeed | 2.9176 ± 0.3558 | 3.0175 | −3.31 % | 1.40× |
| divergence >5 m | speed | 0.2167 | 0.2225 | −2.61 % | 1.48× |
| divergence >5 m | nospeed | 0.2351 | 0.2213 | +6.24 % | 1.35× |

**Every interval on every arm widened — range 1.06×–2.45×**, an independent
replication of the program's 1.28–2.06× finding on a third axis.

### 🔴 VERDICT FLIP — one, and it is a CI-separation that evaporates

| arm | metric | published | corrected | flip |
|--|--|--|--|--|
| `flagship-nospeed` | compounding **grounded** Δ@1s | **+0.2351 ± 0.1410, `separated: True`** | **+0.1667 [−0.1392, +0.4469], `separated: False`** | **TRUE → FALSE** |

The no-speed control's *"the closed loop is CI-resolvably worse than open loop
at 1 s"* does not survive the correct estimator. In the same block Δ@0.5s also
**flips sign** (+0.0386 → **−0.0413** [−0.1906, +0.0915], a −193 % move), though it is not
CI-separated either way — so no verdict rests on it. Δ@1.5s and Δ@2s survive.

**No other separation flipped.** All grounded/bicycle compounding deltas on
`flagship-30k` and `flagship-speed` stay separated, and the headline ordering of
the three arms is unchanged.

### What did NOT move — the honest bound on the blast radius

`lateral_deviation_growth_m`, the whole `comfort` block, and every
`speed_stratified` cell are **bit-identical** between published and corrected.
They were always computed as full-set means over all 881 windows and never
touched `_agg`/`_jack`. The corrected/legacy difference is confined to the
blocks that carried an interval.

### Free upgrade: the imagination A/B now exists on the full 40 episodes

The 2026-07-19 artifacts predate arms A/B, so the re-run adds a block the
originals never had — on the full val set rather than the 12-episode proof:

> **Δ(B − A) ade@2s = −0.1711 [−0.2615, −0.0838]**, `separated: True`,
> `paired_episode_cluster_bootstrap`, 881 win / 40 ep, B=2000 →
> **IMAGINATION_HELPS**

The 12-episode proof's verdict **replicates on 40 episodes**.

Artifacts: `closedloop_flagship-{30k,speed,nospeed}.CORRECTED.json`,
`published_vs_corrected.json` · script: `rerun_closedloop.py`

---

## 3. The imagination proof (2026-07-22) — mixed artifact, cleanly split

Surface: 265 windows / 12 episodes (`ep_00000..ep_00011`), re-run on the A40
(original: local RTX 4060).

**The block that was already decision-grade reproduced to 4 decimals** — across
a different GPU. This is the strongest faithfulness evidence in the whole task:

| quantity | published | re-run | reproduced |
|--|--|--|--|
| A `open_plan_bike` ade@2s | 1.9325 [1.6319, 2.2749] | 1.9325 [1.6320, 2.2749] | ✅ |
| B `closed_bike` ade@2s | 1.7196 [1.4437, 2.0401] | 1.7194 [1.4433, 2.0399] | ✅ |
| **paired Δ(B−A)** | **−0.2130 [−0.3413, −0.0527]** | **−0.2131 [−0.3412, −0.0527]** | ✅ |
| verdict | IMAGINATION_HELPS, separated | IMAGINATION_HELPS, separated | ✅ **unchanged** |

The **legacy** scalars in the same file did move — and this is where the largest
point-estimate error in the whole task lives:

| metric | published | corrected | shift | CI ratio |
|--|--|--|--|--|
| **open_grnd ade@2s** | **0.3177 ± 0.0368** | **0.4045 [0.3128, 0.5149]** | **−21.46 %** | **2.75×** |
| cv ade@2s | 0.9552 ± 0.3813 | 0.8463 [0.4318, 1.3232] | +12.87 % | 1.17× |
| closed_grnd ade@2s | 2.6277 | 2.5963 | +1.21 % | 1.54× |
| closed_bike ade@2s | 1.7315 ± 0.2396 | 1.7194 [1.4433, 2.0399] | +0.70 % | 1.25× |
| closed_bike fde@2s | 3.5459 | 3.5746 | −0.80 % | 1.30× |
| open_bike ade@2s | 0.4451 | 0.4661 | −4.50 % | 1.39× |
| divergence >5 m | 0.2314 | 0.2226 | +3.95 % | — |

**−21.5 % is far outside the −6.67 %/+11.69 % band measured across 27 open-loop
arms**, and the mechanism is visible: with only 12 episodes a 20 % holdout is
2–3 episodes, so the mean-of-split-means is dominated by which episodes were
drawn. **The deprecated statistic's point-estimate bias grows as the episode
count falls** — which means every small-n closed-loop probe is the *worst* place
it was used, not the most forgiving.

Artifacts: `closedloop_imagination-proof.CORRECTED.json`,
`imagination_proof_published_vs_corrected.json` · script:
`rerun_imagination_proof.py`

---

## 4. Lateral / longitudinal decomposition

Run on the persisted `clwin_*.pt` surfaces with the **verified-fixed**
`lateral.py`. Every paired call passes `knot_dt=0.5` **explicitly** — the
closed-loop surface is `sparse_4wp` (4 knots at dense steps [5,10,15,20], so a
knot is **0.5 s**), which is exactly the surface `paired_cross_track`
mislabelled 5× until 2026-07-26. The script **aborts rather than publish** if
`horizon_provenance` is missing or the horizon is not 2.0 s.

**All blocks emitted `horizon_provenance: "explicit"`, `horizon_s: 2.0`.**

### Energy share of squared error — closed loop is LONGITUDINAL, but only just

| path (v1) | longitudinal | lateral | longitudinal share by knot (0.5→2.0 s) |
|--|--|--|--|
| **closed_bike** | **76.1 %** | 23.9 % | 0.893 → 0.855 → 0.819 → **0.734** |
| closed_grnd | 89.8 % | 10.2 % | 0.961 → 0.936 → 0.916 → 0.886 |
| open_grnd | 87.3 % | 12.7 % | 0.874 → 0.899 → 0.924 → 0.854 |
| open_bike | 81.6 % | 18.4 % | 0.964 → 0.935 → 0.869 → 0.790 |
| cv | 30.1 % | 69.9 % | 0.314 → 0.307 → 0.302 → 0.300 |

**The closed loop is where lateral error is manufactured.** Open-loop's
longitudinal share is roughly *flat* across the horizon (0.874 → 0.854); the
closed loop's **falls monotonically 0.893 → 0.734**. So the extra error that
distribution shift adds is disproportionately **cross-track**, and it compounds —
consistent with the program's standing "lateral compounds, longitudinal does
not" finding, now measured on the closed-loop axis for the first time.

Same pattern on the other two arms: `flagship-speed` 70.2 % longitudinal
(0.883 → **0.663**), `flagship-nospeed` 78.9 % (0.895 → 0.777).

### Paired |cross-track| @ 2.0 s (`paired_episode_cluster_bootstrap`, B=2000)

Oriented `closed − open`; **positive = the closed loop is laterally worse**.

| arm | vs `open_grnd` (mean) | vs `open_grnd` (p90) | vs `cv` (mean) |
|--|--|--|--|
| flagship-30k | **+1.0719 [+0.766, +1.411]** sep | **+3.0721 [+2.201, +4.208]** sep | +0.2501 [+0.029, +0.505] sep |
| flagship-speed | +1.0634 [+0.757, +1.425] sep | +3.0466 [+1.874, +4.182] sep | +0.4056 [+0.160, +0.679] sep |
| flagship-nospeed | +0.2700 [+0.054, +0.508] sep | +0.9113 [+0.207, +1.682] sep | −0.0758 [−0.296, +0.125] **tie** |

**The tail is the story: +3.07 m of extra cross-track at p90** versus +1.07 m at
the mean — a 2.9× mean-to-tail ratio. And on v1 the closed loop's lateral error
is **CI-separably worse than constant velocity** (+0.25 m). Both are decisions
the mean alone would have hidden.

Artifact: `latlon_decomposition.json` · script: `latlon_decompose.py`

---

## 5. G4 — the highest-consequence number in the program

`planner_p2_flagship-30k.json` is the artifact behind

> **G4 closed-loop | ADE@2s 1.038 ± 0.202 vs v1 head 1.685 ± 0.098 | PASS —
> 38 % less drift, CI-separated**; divergence 8.7 % ± 4.6 vs 22.2 %

quoted in `MODEL_REGISTRY.md:1456-1458` and `:1683`, in the adjudicating note
`2026-07-19-p2-planner-over-v1.md`, in `V3_HIERARCHICAL_PLANNING_DESIGN.md:216`
(where **1.69 m IS the gate threshold**), `V35_DESIGN.md:91/:276`,
`TANITAD_PAPER.md:1385`, and the 4-brain dominance program's "direction safe"
row.

**Three defects, all MEASURED:**

1. **Both sides are the deprecated statistic.** The artifact stamps
   `"ci": "8-split episode jackknife"`, and `planner_p2.py` **was never
   migrated** — it still aggregates via `_jack_scalar`/`_jack_paired`, which its
   own docstring already flags as *"random holdouts; NOT a jackknife"*. The
   baseline `1.6852` is literally the legacy `closed_bike ade_0_2s` corrected
   above to **1.7318**.
2. **The comparison is not paired and cannot be.** Planner = 221 windows / 20
   episodes / stride 16; head baseline = 881 windows / 40 episodes / stride 8.
   **Different window sets** — so "CI-separated" rests on two independent
   intervals, both too narrow, over non-comparable surfaces.
3. **The planner side is not reproducible at all.** `planner_p2.py:249` draws
   `torch.randn(B,N,K,2)` with **no seed**, so the published `1.0377` is one
   unrepeatable CEM draw.

**Re-drive:** seeded (`seed=0`), same protocol (20 episodes, stride 16,
`CEM_CL` N=48/iters=2), reporting `published_legacy` / `rerun_legacy` /
`rerun_corrected` separately so the **estimator effect** (identical windows,
identical CEM draw) is never conflated with **CEM sampling drift**.

### 🟢 The CEM turned out to be reproducible — which overturns my own prediction

I pre-registered that the unseeded CEM would make the published planner value
unrecoverable, and listed it as SUSPECT S1. **The measurement says otherwise.**
The seeded re-drive reproduced the 2026-07-19 legacy block essentially exactly:

| metric | published legacy | re-run legacy (seeded) | CEM sampling drift |
|--|--|--|--|
| closed_bike ade@2s | 1.0377 ± 0.2022 | 1.0375 ± 0.2023 | **0.02 %** |
| closed_bike fde@2s | 2.1942 ± 0.4550 | 2.1940 ± 0.4552 | **0.01 %** |
| open_grnd ade@2s | 0.4244 ± 0.0573 | 0.4244 ± 0.0573 | **0.00 %** |
| cv ade@2s | 0.7704 ± 0.1704 | 0.7704 ± 0.1704 | **0.00 %** |
| divergence >5 m | 0.0871 ± 0.0460 | 0.0871 ± 0.0460 | **0.00 %** |

The 2026-07-19 run evidently ran from the same default RNG state. **S1 is
withdrawn** — the number is reproducible, therefore correctable, and the shifts
below are purely the estimator on identical windows and an identical CEM draw.
*(Root-cause class: "unseeded ⇒ unreproducible" is an inference, not a
measurement. It was worth 17 GPU-minutes to check instead of asserting.)*

### G4, corrected

| metric | legacy (`overlapping_holdout_se`) | corrected (`episode_cluster_bootstrap`) | estimator effect | CI ratio |
|--|--|--|--|--|
| **planner closed_bike ade@2s** | **1.0375 ± 0.2023** | **0.9799 [0.7456, 1.2312]** | **+5.88 %** | 1.20× |
| planner closed_bike fde@2s | 2.1940 ± 0.4552 | 2.0583 [1.5463, 2.6134] | +6.59 % | 1.17× |
| planner open_grnd ade@2s | 0.4244 ± 0.0573 | 0.4063 [0.3293, 0.4907] | +4.46 % | 1.41× |
| planner cv ade@2s | 0.7704 ± 0.1704 | 0.7214 [0.4680, 1.0360] | +6.79 % | 1.67× |
| **planner divergence >5 m** | **0.0871 ± 0.0460** | **0.0724 [0.0225, 0.1409]** | **+20.30 %** | 1.29× |

**The deprecated statistic inflated every planner number**, and by far the most
on the divergence RATE (+20.3 %) — a rare-event proportion is exactly where a
mean-of-split-means is most biased, because most 20 % holdouts contain few or no
events.

### 🟢 G4 verdict: STILL PASSES — and by a wider margin

| | published | corrected |
|--|--|--|
| planner ADE@2s | 1.038 ± 0.202 | **0.9799 [0.7456, 1.2312]** |
| v1 head baseline | 1.685 ± 0.098 | **1.7318 [1.5707, 1.9070]** |
| **drift reduction** | **38 %** | **43.4 %** |
| intervals overlap? | no | **no** (1.2312 < 1.5707) |
| divergence >5 m | 8.7 % vs 22.2 % (2.5× fewer) | **7.24 % vs 23.50 % (3.2× fewer)** |

**The gate's conclusion is unchanged and its evidence is stronger** — both
numbers moved in the direction that favours the planner. The planner-over-frozen-WM
thesis, and the 30k training run it deferred, stand.

⚠️ **But the "CI-separated" claim is still not what it says it is.** The planner
(221 win / 20 ep / stride 16) and the head baseline (881 win / 40 ep / stride 8)
are scored on **different window sets**, so this can never be a *paired* test —
it is two independent intervals over non-comparable surfaces. The intervals are
now honest and still do not overlap, which is the strongest statement this
design supports; **"CI-separated" in the paired sense remains unproven and would
need both arms re-scored on one common window set.** That is a cheap follow-up
(the head baseline surface is now persisted as `clwin_flagship-30k.pt`).

Artifact: `planner_p2_G4.CORRECTED.json`, `p2win_flagship-30k.pt` (the
per-window surface, now persisted) · script: `rerun_planner_p2_g4.py`

---

## 6. SUSPECT list — numbers I will NOT estimate a correction for

**SUSPECT count: 2 unresolved** (S2, S4). Two more were raised and then
**resolved by measurement rather than assumed** — S1 (§5, the CEM reproduced)
and S3 (§6b, the estimator was right, just unstamped). No correction is
estimated for anything I could not compute.

| # | what | why SUSPECT | what would settle it |
|--|--|--|--|
| ~~S1~~ | ~~the published `1.0377 ± 0.2022` G4 planner value~~ | ~~unseeded CEM — the draw is gone~~ | **WITHDRAWN — see §5.** I predicted this was unrecoverable; the seeded re-drive reproduced it to **0.02 %**, so it is correctable and is corrected (`0.9799 [0.7456, 1.2312]`). Measured, not asserted. |
| S2 | **AlpaSim scene-scored family** — `Flagship_v1_results-summary.json`, `M2_`, `REFC_{base,small,xl}_`, `vs_flag*`, `vs_refc*`, `gate1_summary.json`, `gate0_freefloor_results.json`, `scenario_stratified_*` | **No interval and no estimator at all** — bare point scores. A *different* defect from the legacy estimator, and not fixable by re-aggregation | These carry `per_scene` blocks. A **scene-cluster bootstrap** over the per-scene scores would give a first honest interval — but the scene is the cluster unit, not the window, and the NuRec renderer needed to regenerate anything missing is **unrunnable on this pod** (seccomp/user-namespace, 2026-07-19 INTAKE). Needs a renderer-capable host. |
| S3 | `2026-07-23-frozen-wm-learned-planner/artifacts/*.json` — `bigplanner_*`, `valuemodel_results`, `results*`, `mpc_results` | intervals + `separated` verdicts with **no estimator named anywhere in the JSON** | **RESOLVED — see §6b below.** Downgraded from SUSPECT to *unstamped*. |
| S4 | `taniteval/results/v1-validation.json`, `stack/experiments/p0-arm-compare-smoke/arm_compare.json` | legacy `_jack`/`_agg`; the smoke artifact is 92 windows / 4 episodes with `git_hash: "smoke-sample"` | Both are canaries/footnotes, not decision surfaces. `v1_g1_dryrun_gate.json` already has a `_FIXED` sibling. Re-run only if either is ever cited for a decision — currently neither is. |

### 6b. S3 resolved by measurement, not by reading the source

The frozen-WM scripts *look* like they vendored a correct bootstrap
(`run.py:104-134`, `run40.py:60-81`: resample episodes, percentile CI, paired
form on shared episodes) — but reading code is INHERITED evidence. Their
`perwin_*.pt` dumps are committed, so I re-derived the published intervals
through the canonical `taniteval.ci`:

| published block | published | recomputed via `ci.paired_episode_cluster_bootstrap` | match |
|--|--|--|--|
| `W_minus_oracle` | δ 0.5231 [0.3743, 0.6826] | δ 0.5231 [0.3746, 0.6812] | ✅ |
| `W_minus_cv` | δ 0.1125 [−0.0555, 0.2953] | δ 0.1125 [−0.0562, 0.2973] | ✅ |
| `W_minus_holdv0` | δ 0.1681 [0.0141, 0.3404] | δ 0.1681 [0.0132, 0.3420] | ✅ |

Corroborating: that family's `refs` quote `oracle ade2s = 0.4271 [0.3691,
0.4907]` and `cv = 0.8377` — the **corrected** full-set values, not the legacy
split-means 0.4522 / 0.8248. **MEASURED: the family is on the right estimator
and simply never stamped it.** Fix = add the provenance stamp; **no number
changes**.

Artifact: `frozenwm_estimator_verification.json` · script:
`verify_frozenwm_estimator.py`

---

## 7. Consequence ranking — did the number decide something?

| rank | number | consequence | corrected | status |
|--|--|--|--|--|
| **1** | **G4 `1.038 ± 0.202` vs `1.685 ± 0.098`, PASS** | **GATE.** The verdict that validated the whole planner-over-frozen-WM thesis "at zero training cost" (`TANITAD_PAPER.md:1388`, `V3_…DESIGN.md:220`) — i.e. it **deferred a 30k training run** — and that `4BRAIN_DOMINANCE_PROGRAM.md:64` cites as "direction safe" | **0.9799 [0.746, 1.231] vs 1.7318 [1.571, 1.907]** — 43.4 % less drift, divergence **7.24 % vs 23.50 %** | ✅ **PASS HOLDS, evidence stronger** — but "CI-separated" is unpaired (§5) |
| **2** | `1.69 m` **as the G4 gate THRESHOLD** (`V3_HIERARCHICAL_PLANNING_DESIGN.md:216`, `TANITAD_PAPER.md:1385`) | **GATE DEFINITION.** A threshold, not a result — every future arm is judged against it | **1.7318** (threshold is 2.8 % lax) | needs edit |
| **3** | `MODEL_REGISTRY.md:177-178` §1.2 — *"closed_bike ADE@2s 1.685 ± 0.098, FDE 3.530, divergence 22.2 %. Open-loop 0.452 → 1.685"* | **HEADLINE + upstream source.** ~20 downstream docs cite "REGISTRY §1.2" | **1.7318 [1.571, 1.907]**, FDE **3.619**, div **23.5 %**, open-loop **0.4271** | needs edit |
| **4** | `0.45 → 1.69 m` on **HF model cards** `README_base.md:249`, `README_xl.md:214` | **HEADLINE, EXTERNALLY PUBLISHED.** Highest reputational exposure | **0.4271 → 1.7318** | needs edit (external) |
| **5** | `LEADERBOARD.md:55` + `TANITEVAL_V2_METRIC_SUITE.md:42` R4 | **HEADLINE.** The standing rule that stamps `claim_strength: open-loop/weak` on **every** suite row | direction unchanged, magnitudes move | needs edit |
| **6** | v1 open-loop **0.4522 → 0.4271** inside the INT8/FP8 margin argument | **GPU-DAY.** `DEPLOYMENT_PLAN.md:75` gates the quantization rollout on *"our margin over the kinematic floor is only ~0.05 m (v1 ADE@2s 0.4522 vs floor 0.5005)"* | the **floor does not move** (0.5005 is full-set by construction, `bench.py:511`, registry `:1557` — *"never went through the estimator"*). Only v1 moves, so the margin goes **0.0483 → 0.0734 m, +52 %** | **re-examine — the gate is LESS tight than stated, not more** |
| **7** | `flagship-nospeed` compounding Δ@1s `separated: True` | **KILL-adjacent** — evidence in the no-speed ablation's write-up | **`separated: False`** | **the one flip** |
| **8** | divergence `22.2 %` | HEADLINE, quoted in ~10 docs incl. the Orin/Thor lever analysis | **23.5 %** | needs edit |
| **9** | imagination proof `IMAGINATION_HELPS` | HEADLINE — feeds the v4 imagination thesis | **unchanged, and now replicated at n=40** | ✅ safe |
| 10 | `1.685` as a *comparison anchor* in the low-OOD line (`LOWOOD_HARDENING_REPORT.md:134`, `LOOP_STATE.md:97`: *"1.45 < 1.685"*) | **GPU-DAY** — anchored the Gate-1 YES that funded the low-OOD instrument | **1.7318** — the 1.45 comparison gets **stronger**, verdict unchanged | ✅ direction safe |

**The good news first:** the two verdicts with the most riding on them — **G4
PASS** and **IMAGINATION_HELPS** — both survive the correct estimator, and G4's
margin *widens*. **No decision in the program is reversed by this correction.**
What changes: across **61 recomputed point estimates** the published value was
wrong by **−21.46 % to +20.30 %, bidirectionally — 18 inflated, 43 deflated**
(two near-zero-denominator deltas excluded as meaningless in relative terms).
Every interval was **1.06–2.75× too narrow**. One CI-separation (§2)
evaporates. This replicates, on the closed-loop axis, the −6.67 %/+11.69 %
bidirectional finding measured across 27 open-loop arms — **with a wider range,
because the closed-loop surfaces have fewer episodes.**

**The two that still need care:** #1 (G4's "CI-separated" is two independent
intervals on different window sets, not a paired test) and #6 (a deployment gate
stated as tighter than it actually is).

⚠️ **Two floors, do not confuse them** — I nearly did. `open_bike` **0.5129 →
0.4518** is the *closed-loop harness's* bicycle fidelity floor (TRUE actions
integrated through the bicycle model) and it DID move. The **best-of-3 kinematic
floor 0.5005** used by the deployment gate and by *"first arm below every
trivial bar"* is a **different quantity**, computed full-set by construction
(`bench.py:511`) and explicitly recorded in the registry as *"never went through
the estimator"*. **0.5005, CTRV 0.523 and the no-vision ceiling 0.5735 are all
unaffected by this task.** Only the arm's own number moved against them.

---

## 8. Deliverable manifest

All under `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-closedloop-artifact-rerun/` (repo working tree, **NOT staged**):

| file | what |
|--|--|
| `CLOSEDLOOP_RERUN.md` | this report |
| `harness_check.py` / `harness_check.json` | the 0.4271 + lateral-fix gate |
| `rerun_closedloop.py` | re-run driver (persists `clwin_*.pt`) |
| `closedloop_flagship-{30k,speed,nospeed}.CORRECTED.json` | corrected artifacts |
| `published_vs_corrected.json` | full published→corrected ledger + loop-reproduction check |
| `rerun_imagination_proof.py` | 12-ep proof re-run |
| `closedloop_imagination-proof.CORRECTED.json`, `imagination_proof_published_vs_corrected.json` | corrected proof |
| `latlon_decompose.py` / `latlon_decomposition.json` | lat/lon + paired cross-track |
| `rerun_planner_p2_g4.py` / `planner_p2_G4.CORRECTED.json` | G4 gate re-drive |
| `verify_frozenwm_estimator.py` / `frozenwm_estimator_verification.json` | S3 resolution |
| `_pod_pulled/planner_p2_flagship-30k.json`, `plan_flagship-30k.json` | the pod-only artifacts, now in the repo |
| `artifact_inventory.json` | the machine-readable inventory + SUSPECT classes |
| **`raw_windows/clwin_*.pt` + `p2win_flagship-30k.pt`** | **the per-window closed-loop surfaces — the data whose absence forced this GPU re-run.** 1.5 MB for all five. Now in the repo, so the next estimator change on this axis is a CPU recompute in seconds instead of ~35 GPU-minutes. |

**On the eval pod** at `/root/cl_rerun_20260726/`: the same set plus the run
logs (`rerun.log`, `p2.log`, `imag.log`). **Nothing was written outside that
directory except `taniteval/taniteval/lateral.py`, which was brought up to the
repo version (§0).**

### Escalations (not "please merge" in a doc)

1. **`planner_p2.py` is un-migrated** and still ships `_jack_scalar`/`_jack_paired`.
   Any future G-gate run through it produces a deprecated interval. It needs the
   same migration `closedloop.py` got — I did **not** edit it (out of scope, and
   other agents are active in this tree).
2. **`closedloop.run_and_save` still does not persist `win`.** Until it does,
   every future closed-loop artifact is equally un-recomputable.
3. **The eval pod's `lateral.py` was stale** despite the "141/141 md5-verified"
   sync — the sync predates the fix. Anything run on that pod in the window
   between used stale lateral code.
4. **`MODEL_REGISTRY.md` §1.2 and the G4 rows** carry legacy numbers; the HF
   model cards are externally published. Registry edits are the PI's / owning
   agent's call, not mine.
