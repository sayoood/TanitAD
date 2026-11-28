# `_jack` / `_agg` blast radius — artifact + code sweep

**Date:** 2026-07-25 · **Scope:** every committed result JSON + the full git history of the
estimator modules · **Compute used:** dev-box CPU only. No pod, no GPU, no network.
**`taniteval/` was treated as READ-ONLY** (a sibling agent owns it); `ci.py` was imported from its
file path with `sys.dont_write_bytecode = True`.

**Deliverables in this directory**

| File | What |
|---|---|
| `sweep_jack_artifacts.py` | fingerprints every JSON, classifies it, reads the bias where both blocks exist |
| `recompute_jack_fullset.py` | recomputes 27 arms from the raw `windows_*.pt` window dumps |
| `recompute_hierarchy_seams.py` | recomputes all 42 hierarchy-seam deltas from the artifacts' own full-set fields |
| `jack_artifact_inventory.jsonl` / `.csv` | machine-readable inventory (410 artifacts) |
| `jack_recompute.json` | per-arm + paired recomputation output |
| `jack_hierarchy_recompute.json` | per-seam recomputation output |

---

## 0. Executive verdict

**The bias is real, it is bidirectional, and it is bigger than the program's standing figure —
but the blast radius is narrower than feared, because most of it is already recomputable from
committed data.**

Three findings decide the picture.

**(1) The deprecated statistic moves the POINT ESTIMATE, and the direction is not fixed.**
MEASURED across 24 artifacts that carry both blocks and 27 arms recomputed from raw windows:
the `heldout` split-mean differs from the full-set mean by **−6.67 % to +11.69 %** on the headline
`ade_0_2s`. It is **not** a conservative error — **11 of 27 arms inflated, 16 deflated, none flat**.
Any claim of the form "it errs in the safe direction" is false.

**(2) The interval narrowing extends beyond the program's standing "1.28–2.06×".**
`Project Steering/CI_RECOMPUTE_2026-07-20.json` established that band on **10 arms**
(median 1.51×). I reproduced all 10 **exactly** and extended the set to **27 arms**
(`jack_recompute.json`): the ratio runs **1.107× to 3.100×, median 1.499×**. The median is
stable; the **range is 50 % wider at the top** (`overfit_refa-dynin-15k`, 3.100×) and reaches
lower at the bottom (`flagship-v4.2-step4000`, 1.107×). **`CLAUDE.md` and
`Project Steering/MODEL_REGISTRY.md` both quote `1.28–2.06×` as the program figure and should be
widened to the measured range.**

**(3) The blast radius is bounded and mostly already repaired.** Of **410** committed result
JSONs, **378 never touched the estimator**, **27 are fully recomputable from committed data**
(24 carry both blocks; 3 hierarchy artifacts carry the full-set components beside the `_jack`
deltas), and **only 5 are genuinely SUSPECT** — 4 closed-loop artifacts and 1 smoke file.

**The one number that is both load-bearing and unrecoverable is not in the artifact tree at all:
P2's CEM planner `0.893 ± 0.114`.** It carries D-033 — the entire v3 pivot — and
`taniteval/taniteval/planner_p2.py` is the **only** module that survived both 2026-07-25
migrations. It still emits `_jack_scalar` / `_jack_paired` and **no `full_set` block whatsoever**
(`planner_p2.py:373-381`, `:399`, `:570-574`). Its result JSON exists nowhere in the repo
(3 independent probes, §7).

**One verdict flip found: `refb-10k` "beats CV".** Split-mean says **no** (0.8255 vs CV 0.8248);
full-set says **yes** (0.8372 vs CV 0.8377). Neither is CI-separated, so the honest verdict is
**TIE** — but the registry §6 ✗ and the LEADERBOARD's +0.0005 have been contradicting each other
for this exact reason.

**Two floor-verdict flips found in the hierarchy panel**, both confirming and one extending the
HPP-1 finding (§5).

**No gate verdict flips.** Every RESTART / FAIL / KILL I could trace survives the correction —
v3enc, v4.1, v4.2 all clear or fail their bars by margins far larger than the bias (§6).

---

## 1. Method, and why it can be trusted

The sweep is validated against five corrections the program derived independently, by other
routes, before this audit. All reproduce **exactly**:

**`Project Steering/CI_RECOMPUTE_2026-07-20.json` — 10 arms, 10/10 exact match** on
`published_mean`, `full_set_mean`, `boot_ci95`, `boot_lo`, `boot_hi`, and on all three of its
paired deltas (`+0.0443`, `+0.1642`, `+2.62`). That file is the program's own reference recompute;
my pipeline is a bit-level reproduction of it, extended from 10 arms to 27.

Plus:

| Published correction | Source | My recomputation | Match |
|---|---|---|---|
| REF-C-XL − v1 full-set gap `0.0443` (was `0.006` split-mean) | `MODEL_REGISTRY.md` C6 retraction | `+0.0443`, paired CI `[−0.0544, +0.1465]` | EXACT |
| v1.6 − v1 `+0.0104`, CI `[−0.0888, +0.1147]`, not separated | `v16_vs_v1_paired_bootstrap.json` | `+0.0104`, CI `[−0.0888, +0.1147]` | EXACT |
| REF-C base − XL `+0.0013 [−0.0316, +0.0281]` | `MODEL_REGISTRY.md:1258` | `+0.0013 [−0.0316, +0.0281]` | EXACT |
| v3enc − v1 `+1.5383 [+1.2697, +1.8159]` | `paired_v3enc10k_vs_flagship30k.json` | `+1.5383 [+1.2697, +1.8159]` | EXACT |
| ctx→tactical `+0.0439 → +0.0148` (×2.97) | `HPP1_UNBLOCK_REPORT.md` | `+0.0148`, ratio `2.966` | EXACT |

And the per-arm reproduction is exact on both blocks for every arm with a committed bench JSON
(6/6 arms × 2 blocks × model and CV, to 4 decimals — printed by `recompute_jack_fullset.py`).

**Window-set alignment proof (MEASURED).** All 25 full-size `windows_*.pt` dumps have
**byte-identical `gt` and `cv` tensors** and **identical episode boundaries** — the same 881
windows in the same order. Three dumps (`v16-ab-ft`, `v4.1-10k`, `v4.2-step4000`) use the real
episode ids instead of `0..39`.

> ⚠️ **A caveat in `MODEL_REGISTRY.md:518-527` names the wrong mechanism.** It says
> `split_by_episode` *"hashes the id values"*, so the two eid families' `heldout` numbers
> *"come from different random partitions … and must never be compared directly."*
> **It does not hash.** `stack/tanitad/instruments/checks.py:49-58` (`i3_episode_split`) does
> `torch.randperm(len(episode_ids))` over the **positions** of `sorted(set(int(e)))` — so an
> **order-preserving** relabelling yields **identical** partitions. MEASURED: the real ids in all
> three dumps are order-preserving w.r.t. file index, and `split_by_episode` returns
> **byte-identical val indices for all 8 seeds**. My recomputation reproduces the published
> `0.4886` exactly *using the real-id labels*, which could not happen if the partitions differed.
>
> The registry's *conclusion* — don't compare those split-means — is right; its *reason* is wrong,
> and the wrong reason is load-bearing, because it implies "same-family split-means ARE
> comparable". They are not: the defect is the **bias**, which is present within a family too.
> (The caveat would become true if a relabelling were ever order-*changing*; that is a real hazard,
> just not the one that happened here.)

---

## 2. Code archaeology — the timeline

### 2.1 The lineage

| When | Commit | What |
|---|---|---|
| 2026-07-08 | `bf533bb`, `8dfbc25` | first `def _agg` in the repo (bakeoff harness intake) |
| 2026-07-15 | `e8fca8e` | `_agg` in `run_baseline_floor.py` (the trivial-floor emitter) |
| 2026-07-19 | — | `closedloop.py` v1 emits `_agg`/`_jack` compounding + divergence (`incoming/2026-07-19-alpasim-closedloop-v1/closedloop.py:301,312,371,400-412`) |
| **2026-07-20 08:20** | **`a91bef8`** | **the entire TanitEval harness enters git for the first time** |
| 2026-07-20 | `e2862bf`, `ec0dba5` | `MODEL_REGISTRY.md` created; REF-C-XL "0.458 ± 0.057" published |
| 2026-07-20 | `df32781`+`ci.py` | `ci.py` created — `overlapping_holdout_se` named honestly, bootstrap added |
| 2026-07-25 00:32 | `df32781` | productionization; `bench.run()` starts emitting `cluster_bootstrap` as PRIMARY |
| 2026-07-25 | `52d089a` | `run_gate.py` gains the **REFUSE** path (`:663-720`) |
| 2026-07-25 16:01 | `00f7a2b` | `closedloop.py` migrated; legacy quarantined under `legacy_overlapping_holdout_se` |
| 2026-07-25 16:40 | `a28922c` | `hierarchy.py` migrated; `_jack` quarantined |
| **still open** | — | **`planner_p2.py` NOT migrated** — `_jack_scalar`/`_jack_paired` are its only estimators |

### 2.2 The critical gap in the record

**Every headline in `MODEL_REGISTRY.md` predates `a91bef8` (2026-07-20 08:20).** Before that
commit the harness existed only on `tanitad-eval:/root/taniteval`. A commit-level "which code
version produced which number" timeline **cannot be built from git for anything before
2026-07-20**, and I do not claim one. What *can* be established — and is, below — is what the
artifacts themselves carry, and that is stronger evidence than a code timeline anyway.

### 2.3 The verdict-bearing change nobody logged

At `a91bef8`, `bench.run()` decided the "beats CV" flag on the **split-means**:

```python
beats = _agg(model_split)["ade_0_2s"]["mean"] < _agg(cv_split)["ade_0_2s"]["mean"]
```

At HEAD (`bench.py:215`) it decides on the **full-set bootstrap mean**. So `beats_cv_ade_0_2s`
in every pre-2026-07-25 bench artifact is a **split-mean verdict**. That is where the `refb-10k`
flip comes from (§4).

### 2.4 Live call sites of the deprecated estimator at HEAD

| Module | Line | Status |
|---|---|---|
| `taniteval/taniteval/ci.py` | `:61` | the named, deprecated function — correct to keep |
| `taniteval/taniteval/bench.py` | `:112`, `:222` | `heldout` emitted beside `full_set` + `cluster_bootstrap`; PRIMARY is the bootstrap |
| `taniteval/taniteval/closedloop.py` | `:484`, `:502`, `:753` | **quarantined** under `legacy_*`; `summary` reads the bootstrap (`:598`) |
| `taniteval/taniteval/hierarchy.py` | `:323` | **quarantined** under `LEGACY_BLOCK` (`:121`) |
| **`taniteval/taniteval/planner_p2.py`** | **`:373`, `:381`, `:399`, `:442`, `:570`** | **NOT migrated — sole estimator, no `full_set` emitted** |
| `stack/scripts/run_gate.py` | `:663-720` | **REFUSES** to adjudicate on it (fail-loud) |
| `stack/scripts/eval_flagship_v4.py` | `:491` | carries `REGISTRY_V1_HELDOUT` as a reference constant |

---

## 3. Classified artifact inventory

410 JSONs scanned under `taniteval/results/`, `TanitAD Research Hub/`, `Project Steering/`,
`Benchmarks & Eval/`, `stack/`, `DataEng/`.

| Verdict | n | Meaning |
|---|---|---|
| **SAFE** | **378** | no split-mean block and no `_agg`/`_jack`-shaped metric dict anywhere in the file |
| **CORRECTED** | **24** | carries BOTH `heldout` and `full_set` → the bias is readable directly |
| **CORRECTED_POINT_ONLY** | **3** | hierarchy panels: point estimate correctable from the artifact's own full-set components; interval is not |
| **SUSPECT** | **5** | through the estimator, NOT recomputable from committed data |

Notable SAFE classes, spot-checked rather than assumed:

* All 27 `driving_*.json` (the v2 tier-0 suite) — they carry
  `estimator.deprecated_and_refused: "overlapping_holdout_se"` and an
  `episode_cluster_bootstrap` headline. Genuinely SAFE.
* `scaleab_refc-*.json`, `paired_v3enc10k_vs_flagship30k.json`, `v16_vs_v1_paired_bootstrap.json`
  — bootstrap-native.
* The **departure-power cross-fit** (`incoming/2026-07-24-refccl/powered_departure.json`) and the
  **n=40 low-OOD closed-loop panel** (`lowood_lanekeep_40ep.json`) — **SAFE**, paired
  episode-cluster bootstrap. (They have a *horizon* exposure — every number is a 2-second number —
  but that is a different instrument problem, not this one.)
* The **AlpaSim n=12 panel** (`flagship_vs_refc_suite_results.json`) — **SAFE on this axis**; it
  uses a *scene*-cluster bootstrap, a different estimator entirely.
* The **imagination-in-the-loop A/B** (`closedloop_flagship-30k_imagination-proof.json`,
  `imagination_comparison`) — **SAFE**: `paired_delta_B_minus_A_ade@2s` carries
  `estimator: paired_episode_cluster_bootstrap`. The `IMAGINATION_HELPS` verdict is **not** an
  estimator artifact. *(I expected otherwise and checked before writing it down.)*

### The 5 SUSPECT artifacts

| File | What is unrecoverable | Consequence |
|---|---|---|
| `incoming/2026-07-19-alpasim-closedloop-v1/results/closedloop_flagship-30k.json` | `closed_bike_ade@2s 1.6852`, `closed_grnd 2.6736`, compounding deltas, `divergence_rate 0.2216` | **the closed-loop gap claim** |
| `…/closedloop_flagship-speed.json`, `…/closedloop_flagship-nospeed.json` | same block | the speed-channel closed-loop A/B |
| `incoming/2026-07-22-imagination-closedloop-proof/closedloop_flagship-30k_imagination-proof.json` | the `summary` scalars (`closed_bike 1.7315`, `closed_grnd 2.6277`, `divergence 0.2314`) — **not** the A/B verdict | headline scalars only |
| `stack/experiments/p0-arm-compare-smoke/arm_compare.json` | 216 split-mean dicts | **none** — 92 windows / 4 episodes, `git_hash: "smoke-sample"`. Footnote. |

No closed-loop per-window dump exists (`windows_*.pt` covers open-loop only), and closed-loop
rollout is imagination-in-the-loop — it needs a GPU. **These cannot be corrected without a re-run.**

---

## 4. Recomputation — single arms (27 arms, raw window dumps)

`published (heldout split-mean) → corrected (full-set mean)` for `ade_0_2s`, with the
episode-cluster bootstrap interval. Full table in `jack_recompute.json`.

| Arm | published | corrected | bias | bootstrap CI95 | widening |
|---|---:|---:|---:|---|---:|
| **flagship-30k (v1)** | **0.4522** | **0.4271** | **+5.88 %** | [0.3675, 0.4871] | 1.917× |
| **flagship-v16-ab-ft (v1.6)** | **0.4886** | **0.4375** | **+11.69 %** | [0.3423, 0.5501] | 1.299× |
| flagship-speed (19k) | 0.6277 | 0.6152 | +2.03 % | [0.5422, 0.6951] | 1.387× |
| flagship-nospeed (control) | 2.9176 | 3.0175 | −3.31 % | [2.5450, 3.5444] | 1.404× |
| flagship-v2-6k | 6.1790 | 5.9396 | +4.03 % | [4.3273, 7.6249] | 1.284× |
| **flagship-v3enc-10k** | **2.1072** | **1.9654** | **+7.21 %** | [1.6556, 2.2859] | 1.560× |
| **flagship-v4.1-10k** | **0.8707** | **0.8522** | **+2.17 %** | [0.7468, 0.9800] | 1.420× |
| **flagship-v4.2-step4000** | **1.0490** | **0.9869** | **+6.29 %** | [0.8795, 1.1088] | 1.107× |
| **refc-xl-30k** | **0.4577** | **0.4714** | **−2.91 %** | [0.3896, 0.5556] | 1.451× |
| refc-base-30k | 0.4523 | 0.4728 | −4.33 % | [0.3835, 0.5699] | 1.875× |
| refc-xl (earlier ckpt) | 0.5645 | 0.6048 | −6.67 % | [0.5170, 0.7009] | 2.058× |
| refc-xl-live | 0.4703 | 0.4788 | −1.77 % | [0.3977, 0.5638] | 1.446× |
| refc-v12 | 0.4671 | 0.4625 | +0.99 % | [0.3781, 0.5486] | 1.390× |
| refc-v12-k16reg | 0.4546 | 0.4576 | −0.66 % | [0.3742, 0.5438] | 1.506× |
| refc-v12-identity | 0.4577 | 0.4714 | −2.91 % | [0.3896, 0.5556] | 1.451× |
| **refb-10k** | **0.8255** | **0.8372** | **−1.40 %** | [0.6753, 1.0218] | 1.746× |
| refb (5k) | 0.8682 | 0.8629 | +0.61 % | [0.6928, 1.0385] | 2.115× |
| refb-v2-20k | 0.6462 | 0.6435 | +0.41 % | [0.5410, 0.7516] | 1.922× |
| refb-v2-30k | 0.5921 | 0.5913 | +0.14 % | [0.4766, 0.7131] | 1.727× |
| refa-dinov2 | 2.1322 | 2.1675 | −1.63 % | [1.9081, 2.4212] | 1.409× |
| refa-dynin-30k | 2.9196 | 3.0471 | −4.18 % | [2.4984, 3.6878] | 1.511× |
| overfit_refa-dynin-{5,15,20,30}k | 3.755 / 3.694 / 3.016 / 2.920 | 3.831 / 3.782 / 3.114 / 3.047 | −2.0 to −4.2 % | — | 1.37–3.10× |
| refc-v12-smoke-{t0,reg} | 0.5521 / 1.4860 | 0.5573 / 1.5466 | −0.9 / −3.9 % | — | 1.29 / 1.50× |
| **CV floor (all arms)** | **0.8248** | **0.8377** | **−1.54 %** | — | — |

Two more from artifacts with both blocks but no window dump:

| Arm | published | corrected | bias | file |
|---|---:|---:|---:|---|
| flagship-v4.2b-step4000 (dry-run) | 0.9170 | 0.8604 | +6.57 % | `incoming/2026-07-25-v4-gate-dryrun/raw/flagship-v4.2b-step4000-dryrun.json` |
| refc-small-30k | 0.5007 | 0.5261 | −4.82 % | `incoming/2026-07-22-refc-small-30k/refc-small-30k.json` |
| flagship-v1.5 a/ab/abc × ckpt/best (8 arms) | 0.5437–0.6792 | 0.5349–0.6784 | +0.11 to +3.73 % | `incoming/2026-07-20-vtarget-validation/*.json` |
| flagship 5k gate (reset-speed4b) | 2.3431 | 2.2322 | +4.97 % | `stack/experiments/reset-speed4b/flagship_5k_gate.json` |
| REF-A 4b 30k gate | 2.1355 | 2.1688 | −1.54 % | `stack/experiments/reset-speed4b/refa4b_gate_30k.json` |

### `beats_cv` verdicts — one flip

| Arm | old flag (split-mean) | new flag (full-set) | paired-separated? | verdict |
|---|---|---|---|---|
| **refb-10k** | **False** | **True** | **False** | **FLIP → honest answer is TIE** |
| refc-v12-smoke-t0 | True | True | False | win was never separated — should read TIE |
| all other 25 | unchanged | unchanged | consistent | no flip |

---

## 5. Recomputation — paired arm comparisons

Same 881 windows, same 40 episodes, paired episode-cluster bootstrap B=2000.
`legacy Δ` = difference of the two published split-means (how the gap was actually read).

| Comparison | legacy Δ | **true Δ** | ratio | paired CI95 | separated? | flip? |
|---|---:|---:|---:|---|---|---|
| **refc-xl-30k − flagship-30k** | +0.0055 | **+0.0443** | **×0.124** | [−0.0544, +0.1465] | no | no (C6, already retracted) |
| **refc-base-30k − flagship-30k** | +0.0001 | **+0.0457** | **×0.002** | [−0.0555, +0.1506] | no | **the legacy gap was ~0 — it read as a dead heat** |
| **refc-xl-30k − refc-base-30k** | +0.0054 | **−0.0013** | **×−4.154** | [−0.0316, +0.0281] | no | **SIGN FLIP** |
| **flagship-v16-ab-ft − flagship-30k** | +0.0364 | **+0.0104** | **×3.5** | [−0.0888, +0.1147] | no | no |
| flagship-v3enc-10k − flagship-30k | +1.6550 | +1.5383 | ×1.076 | [+1.2697, +1.8159] | yes | no |
| flagship-v4.1-10k − flagship-30k | +0.4185 | +0.4251 | ×0.984 | [+0.3294, +0.5364] | yes | no |
| flagship-v4.2-step4000 − flagship-30k | +0.5968 | +0.5598 | ×1.066 | [+0.4611, +0.6741] | yes | no |
| refb-v2-30k − flagship-30k | +0.1399 | +0.1642 | ×0.852 | [+0.0430, +0.2851] | yes | no |
| refa-dynin-30k − flagship-30k | +2.4674 | +2.6200 | ×0.942 | [+2.0945, +3.2570] | yes | no |
| flagship-nospeed − flagship-speed | +2.2899 | +2.4023 | ×0.953 | [+1.9137, +2.9304] | yes | no |

**The sign flip is new.** `refc-xl-30k − refc-base-30k` reads `+0.0054` (XL worse) under the
split-mean and `−0.0013` (XL better) on the full set. Neither is separated, so the substantive
conclusion — "104 M ties 252 M" — **stands**, and the registry already publishes the corrected
`+0.0013`. But the demonstration that `_jack` can flip a *sign* on real program data, not just in
synthesis, belongs in the retraction log.

**The `refc-base-30k − flagship-30k` row is the most under-reported.** The legacy gap was
`+0.0001` — a *perfect* dead heat, the most persuasive possible "REF-C base equals the flagship"
number. The true gap is **457× larger** (`+0.0457`) and still not separated. Same conclusion,
utterly different evidence.

### 5.1 Hierarchy seams — 42 deltas recomputed

The hierarchy panel publishes each seam's full-set component means (`hierarchy._mean`,
`hierarchy.py:393-397`) beside a `_jack` delta. So **every `_jack` delta in every hierarchy
artifact is correctable in its point estimate with no re-run.** Full table in
`jack_hierarchy_recompute.json`; the consequential rows:

| Artifact | delta | published | **true** | ratio | flag |
|---|---|---:|---:|---:|---|
| `hierarchy_flagship-30k.json` | `maneuver_acc/delta_real_vs_mean` | 0.0439 | **0.0148** | ×2.966 | **FLOOR FLIP** (0.02) |
| `hierarchy_flagship-v4.2b-dryrun.json` | `maneuver_acc/delta_real_vs_mean` | 0.0298 | **0.0091** | ×3.275 | **FLOOR FLIP** (0.02) |
| `hierarchy_flagship-30k_v1.json` | `goal_latent_cos/delta_real_vs_mean` | 0.0030 | **0.0009** | **×3.333** | largest ratio in the panel |
| `hierarchy_flagship-30k_v1.json` | `maneuver_acc/delta_real_vs_zero` | −0.1163 | **−0.0415** | ×2.802 | |
| `hierarchy_flagship-30k_v1.json` | `maneuver_acc/delta_real_vs_mean` | −0.0399 | **−0.0227** | ×1.758 | |
| `hierarchy_flagship-30k.json` | `goal_latent_cos/delta_real_vs_mean` | 0.0084 | **0.0050** | ×1.680 | both fail floor 0.01 |
| `hierarchy_flagship-30k_v1.json` | `wp_ade_2s/delta_real_vs_zero` | 2.6491 | **4.1370** | ×0.640 | **56 % understated** |
| `hierarchy_flagship-30k.json` | `h18_grounded_vs_ungrounded` | 2.6979 | **2.9568** | ×0.912 | true effect is LARGER |
| `hierarchy_flagship-30k.json` | `maneuver_acc/delta_real_vs_zero` | −0.0588 | **−0.0805** | ×0.730 | |
| `hierarchy_flagship-30k.json` | `wp_ade_2s/delta_real_vs_mean` | 0.0336 | **0.0437** | ×0.769 | |

This **independently confirms HPP-1** (×2.97 / ×3.28 / ×1.76 / H18 = +2.9568, all exact) and
**extends it**: HPP-1 covered the ctx→tactical seam; the ×3.333 on `goal_latent_cos` in the
19k panel and the ×0.640 understatement on `wp_ade_2s/delta_real_vs_zero` are new.
**No sign flips** in the real hierarchy artifacts. The intervals are **not** recomputable —
the panel does not persist per-window arrays.

> The verdict string in `hierarchy_flagship-30k.json:254` — *"hierarchy 2/3 seams load-bearing …
> ctx→tactical LOAD-BEARING (man Δ0.0439 …)"* — is built from the `_jack` values and is wrong
> in its own artifact. On the corrected point estimates the seam fails all three floors.

---

## 6. Consequence-ranked findings

**Rank 1 — `P2 CEM planner 0.893 ± 0.114` decided the v3 pivot and cannot be recomputed.**
`planner_p2.py` is the only module that survived both migrations; it emits **no `full_set`
block at all**, so even a fresh run of the *current* code would not produce the corrected number
without a code change. Its result JSON is absent from the repo (verified 3 ways, §7). The
number carries **D-033** ("supervised heads demote to proposal priors"), IMP-2 "Confirmed 90 %",
and PC2 of the Hierarchy Proof Program. `Benchmarks & Eval/LEADERBOARD.md:588` already says it
*"survives only under the deprecated estimator"* — nobody acted on that.
**SUSPECT. What would settle it:** migrate `planner_p2.py` to `ci.py` (add a `full_set` block and
`paired_episode_cluster_bootstrap`) and re-run one arm on the eval pod. ~1 GPU-hour.

**Rank 2 — `0.452 m` is the split-mean, and it is the program's most-quoted number.**
MEASURED: full-set **0.4271**, bias **+5.88 %**. It is the deployed-v1 headline in
`CLAUDE.md` itself, and the string `0.452` appears on **11 lines of `PROGRAM_OVERVIEW.md`,
12 of `TANITAD_PAPER.md` and 26 of `MODEL_REGISTRY.md`** (`grep -c`; not all are bare quotes).
It is also the **WM-canary reference for the entire v4 gate line**. The registry's own table is honest —
it prints both columns — but every derived document dropped the name.
`v1_g1_dryrun_gate.json` shows the gate literally adjudicating on it:
`"primary": {"value": 0.4522, "provenance": "heldout (DEPRECATED overlapping_holdout_se) ±0.0312"}`.
**CORRECTED.** *(Already fixed in the reader: `v1_g1_dryrun_gate_FIXED.json` reads 0.4271 via
`cluster_bootstrap`. The doc layer is not fixed.)*

**Rank 3 — the closed-loop panel is entirely un-recomputed, and its open-loop leg is the biased
number.** `closed_bike_ade@2s = 1.6852` and `open_grnd_ade@2s = 0.4522` are both split-means.
The famous "open-loop 0.45 → closed-loop 1.69" gap therefore has a **corrected denominator**
(0.4271) and an **uncorrectable numerator**. The divergence rate `0.2216` and every compounding
delta are `_jack` outputs with a **one-sided** `separated` predicate (`|mean| − ci95 > 0`) —
the most permissive form of the defect. **SUSPECT.** **What would settle it:** re-run
`python3 -m taniteval.closedloop --arm flagship-30k` (+`-speed`, `-nospeed`) on the eval pod once
it is free; the HEAD code already emits the bootstrap as primary and quarantines the legacy.
~30 GPU-min for all three (the original run was 46.6 s wall).

**Rank 4 — the hierarchy floor-verdict flips.** `ctx→tactical maneuver-acc` clears the 0.02
practical floor at `0.0439` and fails it at `0.0148`; same on v4.2b (`0.0298 → 0.0091`).
This is the seam the v3.5 and v4 designs were premised on. **CORRECTED (point only)** —
already caught by HPP-1 today; this sweep confirms it independently and adds the ×3.333 and
×0.640 rows. **What would still settle the intervals:** re-run `taniteval.hierarchy` on the
three arms with the migrated code.

**Rank 5 — `refb-10k` "beats CV" flips, and the CV floor circulates as two different scalars.**
`0.8248` (split-mean) is quoted bare in `MODEL_REGISTRY.md:156`/`:1480` and as the
`PROGRAM_OVERVIEW.md:124` column header "Beats CV 0.825?"; `0.8377` (full-set) is the
LEADERBOARD's bar. `refb-10k` sits *between* them, which is why the two documents give opposite
verdicts for the same arm. **CORRECTED — and the honest answer is neither ✗ nor ✓ but TIE**
(paired bootstrap not separated).

**Not findings (checked, cleared):** v3enc RESTART, v4.1 FAIL, v4.2 kill, the AlpaSim n=12
panel, the n=40 low-OOD panel, the departure-power cross-fit, the D1 frozen-WM `0.599`, and the
imagination A/B verdict. The first three survive the correction by margins 10–100× the bias;
the rest never went through the estimator. *(The frozen-WM `0.599` and the closed-loop panels
carry a separate, larger exposure — n=12 and a 2-second horizon — which is out of this sweep's
scope and already owned elsewhere.)*

---

## 7. What I could NOT determine, and what would settle it

1. **Which code version produced each pre-2026-07-20 number.** The harness entered git at
   `a91bef8` on 2026-07-20 08:20; everything before that ran from `tanitad-eval:/root/taniteval`
   with no version record. *Settled by:* nothing available in-repo. The artifacts' own blocks are
   the better evidence and are what this report uses.

2. **The closed-loop point estimates** (4 artifacts). No per-window closed-loop dump exists and
   the rollout needs a GPU. *Settled by:* one `taniteval.closedloop` re-run per arm on the eval
   pod (~10 GPU-min each).

3. **P2's `0.893 ± 0.114`.** Absence verified three ways — `git ls-files | grep planner_p2`
   (only the `.py`), a filesystem `find` for `*planner_p2*` (only `.py` + worktree copies +
   `.pyc`), and a content grep for its distinctive emitter keys `closed_bike_fde2s` / `g1_delta`
   across every JSON in the repo (**zero hits**). *Settled by:* migrating `planner_p2.py` and
   re-running one arm.

4. **Hierarchy-seam INTERVALS.** Point estimates corrected; the paired episode-cluster bootstrap
   needs per-window arrays the panel does not persist. *Settled by:* re-run of
   `taniteval.hierarchy` (the migrated code emits them).

5. **Rate-style `_jack` blocks with no full-set twin** — e.g.
   `consistency/maneuver_vs_trajectory/agreement 0.8569 ± 0.0373`. The panel publishes `kappa`
   full-set but not the agreement rate, so the rate's split-mean has no counterpart. Small
   consequence (a coherence read, not a gate). *Settled by:* the same hierarchy re-run.

6. *(RESOLVED during the sweep — kept as a record of the correction.)* I first wrote that
   `CI_RECOMPUTE_2026-07-20.json` could not be located. A second probe found it at
   `Project Steering/CI_RECOMPUTE_2026-07-20.json`. It contains exactly the 10 arms behind the
   `1.28–2.06×` band (median 1.51×), all 10 of which I reproduce bit-for-bit. So the band was
   **never wrong, only under-sampled** — 27 arms give 1.107–3.100×, median 1.499×. Nothing
   outstanding; the documents just need the wider range.

7. **Whether any *prose* number I did not trace came through the estimator.** This sweep is
   artifact- and code-complete but not prose-complete: a claim quoted in a doc with no named
   source JSON cannot be classified. The catalogue assembled alongside this sweep lists 48 such
   claim sites. *Settled by:* the doc-layer estimator-naming enforcement already proposed as
   R4/F7 in the chief-scientist review.

---

## 8. Recommended actions (for the orchestrator — I have staged nothing)

1. **Widen the narrowing band** in `CLAUDE.md` and `MODEL_REGISTRY.md`: `1.28–2.06× (10 arms)` →
   **`1.11–3.10×, median 1.50× (MEASURED, 27 arms, jack_recompute.json)`**. The old band was
   under-sampled, not wrong — all 10 of its arms reproduce exactly.
2. **Fix the mechanism in the v1.6/v1 partition caveat** (`MODEL_REGISTRY.md:518-527`).
   `i3_episode_split` permutes **positions**, it does not hash id values; the two families'
   partitions are byte-identical here (MEASURED, 8/8 seeds). Keep the "don't compare those
   split-means" conclusion, but attribute it to the **bias** — otherwise the text implies
   same-family split-means are safe to compare, and they are not.
3. **Log two new retraction rows** under the existing class: the `refc-xl − refc-base` **sign
   flip** on real data, and `refb-10k`'s **beats-CV flip** (correct answer: TIE).
4. **Escalate `planner_p2.py`** — the only unmigrated module, holding the highest-consequence
   unrecoverable number in the program. This is an integration escalation, not a doc note.
5. **Queue the three re-runs** (closedloop ×3 arms, hierarchy ×3 arms, planner_p2 ×1 arm) for the
   next free eval-pod window. Total ≲ 2 GPU-hours; it closes every remaining SUSPECT.
