# pod2 as the n ≥ 200 eval host — corpus verified, harness stood up, the ladder's power re-measured

**Date:** 2026-07-26 (Europe/Berlin; pods log UTC) · **Stream:** 4-Brain Dominance Program
**Host:** `tanitad-pod2` (A40, 0 MiB / 0 % at start — its 30 k flagship run had finished)
**Constraints honoured:** pod1 (TRAINING) never touched · the 600-episode cache **read only, never
moved or modified** · `OMP_NUM_THREADS=8` on every job · nothing `git add`-ed, committed or pushed.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED`
(another agent/doc, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. Headline

| # | Result | Class |
|:--:|---|---|
| **1** | ✅ **The 600-episode CLEAN build is present, loadable and intact on pod2** — 600 `ep_*.pt`, **0 read errors**, 600 unique `sha256(poses)`, `T ∈ [188, 205]`, 66 GB. | `MEASURED` |
| **2** | ✅ **Byte-level DISJOINT from the parity train: 0 / 600.** Re-derived on **pod2's own copy** of `physicalai-train-e438721ae894` (2,376 eps), not inherited from pod3's. | `MEASURED` |
| **3** | ⭐ **The ORDER-PRESERVING PREFIX property holds: 40 / 40 positions match.** The eval pod's published 40 are `val600[0:40]` element-for-element (positions `[0,1,…,39]`), so 40 → 600 **adds episodes and re-selects none. Parity holds.** | `MEASURED` |
| **4** | ⭐ **Functional proof of the same fact:** pointing the harness at the **600** build with `--episodes 40` reproduces **881 windows** and **`ade_0_2s = 0.4271 [0.3675, 0.4871]`** — bit-identical to the registry's published v1 row, CI bounds included. | `MEASURED` |
| **5** | ✅ **pod2 is stood up and CLEAN.** `taniteval` **217/217 files md5-identical**, `stack` **344/344 md5-identical**, **0 mismatches, 0 missing**. `corridor.py` present **and exercised**; `lateral.py` emits `horizon_provenance` with `horizon_s = 2.0` on the sparse surface (the stale `0.4 s` signature is **absent**). | `MEASURED` |
| **6** | ⭐ **The sys.path audit found SIX other `stack`/`taniteval` trees on this box. NONE is on `sys.path`**, and all 46 loaded `taniteval.*`/`tanitad.*` modules resolve inside the two verified roots. The 30 `sys.path` entries the submodules' `sys.path.insert` calls produce collapse to exactly **two distinct trees**. | `MEASURED` |
| **7** | ⛔ **The ladder's real ceiling is NOT the episode count — it is the JUNCTION stratum.** Overall clusters stay at **600** out to K = 190. The junction stratum peaks at **232** (K = 20) and **falls below 200 between K = 70 and K = 75**. HP-2 is the binding problem, not HP-1. | `MEASURED` |
| **8** | ⭐ **The closed-loop co-primary at K = 185 DOES reach n ≥ 200 at 600 episodes: 599 windows over 596 episode-clusters** (vs 41 / 40 published) — a **14.9× increase in clusters**. It does not need a shorter horizon *for pooled power*. It needs one for **junction** power. | `MEASURED` |
| **9** | ⛔ **The inherited "stride cannot buy windows" is FALSE as literally stated, and right in substance.** At K = 185, stride 8 → 1 multiplies windows **×5.9** (599 → 3,548) and changes clusters by **0**. Stride buys correlated rows 0.1 s apart inside the same episode; the resampling unit is the episode, so it buys **no power**. | `MEASURED` |
| **10** | ⭐ **Recommended horizon for the next gate: register `K = 60` (6.0 s) as primary, with `K = 70` (7.0 s) as the documented hard maximum** and K = 185 retained REPORT-ONLY-pooled. K = 70 is the **largest** K at which every E1a stratum — junction included (**204**) — clears the 200 two-arm bar on the maximum-possible parity val; K = 60 keeps a margin (**207**). It also satisfies the gate report's *own* §10.1 instruction to "register at a horizon where the envelope holds". | `MEASURED` |
| **11** | ⭐ **The capacity was used: v1 now has a reference at n = 600** — `ade_0_2s = 0.4108 [0.3956, 0.4273]`, 13,198 windows / 600 clusters, 901 s. **CI half-widths shrank ×2.8–3.9 (mean ≈ 3.4) against the √15 = 3.87 that √n predicts** — the §4.4 projection, now MEASURED. ⛔ **Not a correction to 0.4271**: a different deployment, and the 600 is an *easier* corpus (CV floor 0.8377 → 0.6917). | `MEASURED` |
| **12** | ⭐⭐ **A verdict FLIPPED on power alone.** `along_track_vs_cv` goes from δ 0.2543 **[−0.0278, +0.5304] "tie"** at 40 episodes to δ **0.2525 [+0.1926, +0.3104] "model wins, separated"** at 600. **The effect moved 0.7 %; only the interval moved.** A real effect had been sitting under the 40-episode noise floor — caught on the program's own reference arm. | `MEASURED` |
| **13** | ⚠️ **The repo drifted by 7 files in the 35 minutes this ran** (§2.6) — the staleness mechanism, live. **No number here depends on any of them**, and I did **not** re-sync a sibling's mid-edit tree. ⭐ **Two of the 7 are `taniteval/clhorizon.py` and `taniteval/ood.py` — precisely the K-sweep and envelope instruments §4.5's recommendation needs. Escalated as an integration (§9.8), not left in a doc.** | `MEASURED` |
| **14** | ⭐⭐ **S3's power re-run on pod2's own caches reproduces the pod3 figures EXACTLY — 12/12 strata, window counts and cluster counts.** Lateral **558 decision-point / 139 event clusters** (yield 0.93), longitudinal **520 / 312** (0.8667); train 2,206 / 2,056. Two hosts, two disks, two data surfaces (a 1.86 MB poses view vs the real 66 GB cache), digit-for-digit. **The brief's `558 lat / 520 lon` is confirmed.** | `MEASURED` |
| **15** | ⛔ **…but the S3 SKILL BARS are not reproducible to the precision they are quoted at.** Every firewall *verdict* survives (R1 not refused, R2 armed on both axes, R3-lat clear, R3-lon separated-and-negative), yet the bars move across hosts on identical code and identical data: **lat 0.6534 → 0.6493, lon 0.5323 → 0.5420**. `skill = QWK − bar`, so a marginal arm would be adjudicated by which pod fitted the bar. **Pin it before it decides anything (§9.9).** | `MEASURED` |

---

## 1. The corpus — verified here, not inherited

### 1.1 Method, and why the key is `sha256(poses)`

Every row below is keyed on **`sha256` of the raw `poses[T,4]` float32 bytes**, read via
`torch.load(..., mmap=True)` so only the ~3 KB poses pages fault in and the 117 MB `frames_u8`
never leaves disk. Cost: **13.3 s** for the 600, **71.5 s** for the 2,376-episode parity train.
The cache directory was **never written to**.

⛔ **`episode_id` was checked and is unusable as a key — reproduced independently here.** pod2's copy
of the parity train holds **2,376 episodes and 2,342 unique `episode_id`s** (34 collisions); the 600
build holds **600 and 596**. On `episode_id` the val "overlaps" the train by **20 / 600 = 3.33 %**;
on bytes it is **0**. The 20 are collisions, not leakage.

⚠️ **`physicalai-val-f1b378f295ae` was never touched.** It does not exist on pod2 — pod2's `_epcache`
root holds only `physicalai-train-e438721ae894` and `physicalai-val-0c5f7dac3b11`, so the
`sorted(glob("*val*"))[-1]` resolver has nothing leaky to prefer here.

Scripts: `scripts/corpus_verify.py`, `scripts/prefix_and_disjointness.py`.
Artifacts: `artifacts/verify_val600_pod2.json`, `artifacts/verify_train_pod2.json`,
`artifacts/verify_evalpod40.json`, `artifacts/prefix_disjointness_result.json`.

### 1.2 What is actually on the disk

`MEASURED`, pod2 `/workspace/data/physicalai_phase0/_epcache/`:

| cache | `ep_*.pt` | read errors | unique `sha256(poses)` | unique `episode_id` | `T` range | size |
|---|---:|---:|---:|---:|---|---:|
| `physicalai-val-0c5f7dac3b11` | **600** | **0** | **600** | 596 | `[188, 205]` | **66 GB** |
| `physicalai-train-e438721ae894` | **2376** | **0** | **2376** | 2342 | `[188, 205]` | — |

⭐ **A structural corroboration nobody asked for and that I checked anyway:** the train directory
carries **24 `skip_*` markers** (`skip_01798 … skip_01941`) beside its 2,376 episodes — matching
`stack/tanitad/data/parity_manifest.json`'s `skip_indices` (24 entries, `1798 … 1941`) exactly. That
is the **`f09e44db` skip-hash structure**, confirmed on pod2's own disk rather than quoted.
*(The `ls | wc -l` count of 2401 is 2376 + `DONE` + the 24 skip markers — it is not 2,400 episodes.)*

### 1.3 A. Disjointness — `MEASURED`, byte level

> ### ✅ **0 / 600 (0.0 %).** No episode of the 600-build shares a `sha256(poses)` with any of the
> 2,376 parity-train episodes.

| pair | overlap by `episode_id` | overlap by **bytes** |
|---|---:|---:|
| clean val @600 vs parity train (pod2's own copy) | 20 / 600 = **3.33 %** | ✅ **0 / 600 = 0.00 %** |

This is an **independent re-derivation**, not a re-read: the S3 agent hashed pod3's copy of the
parity train and relayed a poses-only view; this run hashed **pod2's** copy of both sides on one
host. Both give 0. The collision count (2,376 → 2,342) reproduces to the episode.

### 1.4 B. The PREFIX property — ⭐ the fact that preserves parity

Set equality is not enough. What makes "40 → 600 re-selects nothing" true is that the published set
is the **first 40 in order**, so `list_val_episodes(VAL, 40)` on the 600 build returns *the same
episodes in the same order* as it does on the eval pod's 40-episode deployment.

> ### ✅ **CONFIRMED: 40 / 40 positions match, element-for-element.**
> `published40[i].sha256(poses) == val600[i].sha256(poses)` for every `i ∈ [0, 39]`, and the
> published episodes occupy positions **`[0, 1, 2, …, 39]`** of the 600 — contiguous, in order,
> starting at 0. Set equality also holds (it is implied, not relied on).
> ⇒ **Moving from the published 40 to the 600 ADDS episodes and RE-SELECTS NONE. `Parity is sacred`
> is not violated.**

⚠️ **The statistics still must be re-run, not rescaled.** 600 episodes is a different `n` and a
different window set; a number on 600 is not comparable to a number on 40 by arithmetic.

### 1.5 Two further corroborations, both independent of the hashing

1. ⭐ **The window arithmetic closes on a third source.** `range(0, T − W − K, stride)` with `W = 8`,
   `stride = 1`, `K = 20` over the 600 measured `T` values gives **102,532 windows** — exactly the
   `n_windows: 102532` recorded in pod2's `/workspace/v15/labels_val_v4_provenance.json` for the same
   600-episode build. Same corpus, two unrelated derivations.
2. ⭐ **The functional proof (§2.5).** Pointing the *shipped harness* at the 600 build with
   `--episodes 40` produced **881 windows / 40 episodes** and `ade_0_2s = 0.4271 [0.3675, 0.4871]`,
   bit-identical to the registry. If the first 40 of the 600 were not the published 40, this number
   could not land on the published value's CI bounds.

---

## 2. pod2 stood up as an eval host

### 2.1 What was placed where

`taniteval` submodules hard-code three absolute paths via `sys.path.insert` — `/root/TanitAD/stack`,
`/root/TanitAD/stack/scripts`, `/root/taniteval` — so the deployment reproduces the eval pod's layout
exactly rather than inventing a new one.

| path | what | how |
|---|---|---|
| `/root/taniteval` | the `taniteval` package root (217 files) | tar from the dev-box working tree, md5-verified |
| `/root/TanitAD/stack` | the `stack` tree (344 files) | same bundle |
| `/root/valdata/physicalai-val-0c5f7dac3b11` | **symlink** → `/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11` | ⚠️ a link in `/root`; **the cache itself is untouched** (601 entries before and after) |
| `/root/models/flagship-30k/ckpt.pt` | **symlink** → `/workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt` | v1 FINAL, step 29999. No 3.3 GB copy made |
| `/root/TanitAD/s3/` | the four S3 files, md5-identical to the repo copies | for §3 |

⚠️ **Deliberately excluded from the bundle:** 17 files, **all** training-run artifacts
(`p0-sA01…/model.pt`, `p0-sB00…/model.pt`, `p0-s000…/model.pt`, six `fan_*.png`, their
`config.json`/`metrics.json`/`REPORT.md`) totalling ~2.0 GB. **Zero `.py` files were excluded** —
verified, not assumed (`artifacts/md5_compare.json`).

⚠️ **A layout trap, recorded because it cost a run:** a script placed at `/root/` cannot
`import taniteval`. `sys.path[0]` becomes `/root`, which contains the *directory* `taniteval` with no
`__init__.py`, so the name binds to an empty namespace package and every submodule import fails with
`ImportError: cannot import name 'bench' from 'taniteval' (unknown location)`. **Run from
`/root/taniteval`.** This is a property of the deployment layout, not of pod2.

### 2.2 md5 verification — the sync is byte-identical

`MEASURED`, `scripts/md5_manifest.py` run on both ends, same algorithm, same skip rules:

| tree | repo files | pod2 files | common | **mismatch** | missing | extra |
|---|---:|---:|---:|---:|---:|---:|
| `taniteval` → `/root/taniteval` | 217 | **217** | 217 | **0** | **0** | 0 |
| `stack` → `/root/TanitAD/stack` | 361 | **344** | 344 | **0** | **0** | 0 |

The 17-file gap in `stack` is exactly the deliberate artifact exclusion above. The bundle's own md5
(`3e0f379b7ae5b8b8da8082cd6af7ab23`) matched on both ends before extraction.

### 2.3 ⭐ THE `sys.path` AUDIT — every tree, because probing the obvious one is what failed

The lesson being mechanised (`INHERITED` — this agent did **not** re-verify the eval pod's staleness,
and does not need to): the eval pod was **62 % stale with `corridor.py` entirely missing**, and a
**second** stale tree (`/root/TanitAD/stack`) was hard-coded by every submodule — so a probe of the
obvious tree declared the pod clean when it wasn't. The audit below is `MEASURED` regardless of
whether that history is; it is designed so that the same defect could not hide here.

**(a) `sys.path` after the full decision-grade import set** (`taniteval` + `bench`, `ci`,
`closedloop`, `corridor`, `data`, `hierarchy`, `lateral`, `pathspeed`, `registry`, `rollout`,
`runner`, `tanitad`, `tanitad.data.parity`): **30 entries**, produced by the submodules' repeated
`sys.path.insert` calls. They collapse to **exactly three distinct paths, two distinct trees**:

| distinct entry | exists | hosts | verified root |
|---|:--:|---|---|
| `/root/taniteval` | ✅ | `taniteval` | ✅ `/root/taniteval` |
| `/root/TanitAD/stack` | ✅ | `tanitad` | ✅ `/root/TanitAD/stack` |
| `/root/TanitAD/stack/scripts` | ✅ | — (flat modules) | ✅ inside `/root/TanitAD/stack` |

plus the five stdlib / `dist-packages` entries, **none** of which hosts a `tanitad` or `taniteval`
package. `PYTHONPATH` and `TANITEVAL_STACK_OVERRIDE` were both **unset** for the audit run.

**(b) Both verified roots re-hashed *at import time*, not at sync time:**

| root | seen / manifest | mismatch | missing | clean |
|---|---:|---:|---:|:--:|
| `/root/taniteval` | 218 / 217 | **0** | **0** | ✅ (the +1 is the preflight script itself) |
| `/root/TanitAD/stack` | 344 / 344 | **0** | **0** | ✅ |

**(c) Every loaded `taniteval.*` / `tanitad.*` module resolved:** **46 modules, 46 inside a verified
root, 0 outside.**

**(d) ⭐ Absence probed at more than one location — every `tanitad`/`taniteval` tree on the whole box**
(`find / -maxdepth 6`), and whether it is on the path:

| tree | on `sys.path`? | verified? | note |
|---|:--:|:--:|---|
| `/root/taniteval` | ✅ yes | ✅ yes | this deployment |
| `/root/TanitAD/stack` | ✅ yes | ✅ yes | this deployment |
| `/workspace/TanitAD/stack` | ❌ no | — | ⚠️ **a stale git checkout at `0f93b98`** — the exact shadowing hazard |
| `/workspace/v15/evalsrc` | ❌ no | — | v1.5 era eval source |
| `/workspace/tmp/v2_stack/{stack,taniteval}` | ❌ no | — | v2 era |
| `/workspace/phase0-build/stack` | ❌ no | — | build-time copy |
| `/workspace/speed_input/stack` | ❌ no | — | speed-input experiment |

> ✅ **Verdict: PASS.** Six other trees exist; **none** of them is importable from the eval
> environment, and nothing loaded from one. ⚠️ **This is a statement about the environment as
> configured** (`cwd = /root/taniteval`, `PYTHONPATH` unset). A future job that exports
> `PYTHONPATH=/workspace/TanitAD/stack` re-opens it. Re-run `scripts/preflight.py` from any new
> invocation shape before quoting a number.

Artifact: `artifacts/preflight_pod2.json` (full 30-entry list, all 46 module paths, all 8 trees).

### 2.4 The standing preflights

| # | check | result | class |
|:--:|---|---|---|
| **P1** | `corridor.py` **present AND EXERCISED** | ✅ **PASS.** Imported, and *run*: `cross_track_from_paths` → `corridor_block` (episode-cluster bootstrap) → `stratified` on a synthetic 12-window / 4-episode set. Constants read back: `CORRIDOR_HALFWIDTH_M 1.75`, `CORRIDOR_GRID_M (1.0, 1.75, 2.5)`, `JUNCTION_DEG 10.0`. `horizon_seconds(185) = 18.5`; `horizon_ceiling(205) = 196`, `horizon_ceiling(198) = 189`. | `MEASURED` |
| **P2** | `lateral.py` emits `horizon_provenance`, `horizon_s = 2.0` on the sparse surface | ✅ **PASS.** `paired_cross_track(..., step=4)` on a 4-knot surface → **`horizon_s = 2.0`**, `horizon_provenance = "inferred_from_knot_count"`, `n_knots = 4`; with `knot_dt=0.5` → `horizon_s = 2.0`, provenance `"explicit"`. `from_sparse_windows` stamps `surface="sparse_4wp"`, `dt_s = 0.5`. ⭐ **The stale `0.4 s` signature (`step * DT`, the 5× bug) is ABSENT.** Estimator on the delta: `paired_episode_cluster_bootstrap`. | `MEASURED` |
| **P3** | the val chokepoint | ✅ **PASS.** `data.list_val_episodes(VAL, 40)` → 40 files, `ep_00000.pt … ep_00039.pt`; guard reports `episodes_present 600`, `registered_deployments [40, 600]`, `parity: true`, `checked: true`. | `MEASURED` |
| **P4** | ⭐ **v1 reproduces 0.4271** | ✅ **PASS — exactly.** See §2.5. | `MEASURED` |

### 2.5 ⭐ P4 — v1 on pod2, against the published row

`python3 -m taniteval.runner run --model flagship-30k --episodes 40`, 101.0 s wall, `ckpt_step 29999`.

| quantity | pod2, this run | registry (`PUBLISHED`) | verdict |
|---|---|---|:--:|
| `ade_0_2s` full-set | **0.4271089434623718** | **0.4271** | ✅ |
| episode-cluster bootstrap, B = 2000 | **0.4271 [0.3675, 0.4871]**, se 0.0305 | **0.4271 [0.3675, 0.4871]** | ✅ **CI bounds identical** |
| `n_windows` / `n_episodes` | **881 / 40** | 881 / 40 | ✅ |
| `fde@2s` · `miss_rate@2m` · `tms` | 0.9075 · 0.0454 · 0.0978 | 0.9075 · 0.0454 | ✅ |
| vs CV | Δ **+0.411 [+0.205, +0.624] SEPARATED** | — | — |

Estimator named: **`episode_cluster_bootstrap`, B = 2000, unit = val episode**.
`overlapping_holdout_se` is present in the JSON as `legacy_overlapping_holdout_se` and is **not**
quoted anywhere here. Artifact: `artifacts/RESULT_v1_40ep_preflight.json`.

⚠️ **One honest caveat carried forward, not hidden:** the runner stamps
`pc2: the scored pass did NOT traverse the hierarchy … actions_source=expert_future`, i.e. this is
`wm_fidelity_ade_2s`, not a hierarchy or driving result. That is true of the published 0.4271 too —
it is the same protocol, which is exactly why it is the right preflight.

⭐ **A new artifact this produced:** the guard computed `episode_uid_sha256 =
75a4d429be8cef8ea47a319e2033d792ee9eecbff033fad27dbb624b5634df20` over the 600 uids. The manifest
carries **no expected digest** for the val split (`uid_source: count-only-unrecorded`), so the check
degraded to COUNT-ONLY. **This digest is now MEASURED on a verified cache and could be registered**
— escalated in §9.4, deliberately not acted on here.

---

### 2.6 ⚠️ The repo drifted **while this ran** — measured, and it is the same hazard

The sync and its md5 proof are a **point-in-time** statement. Re-running the manifest against the
repo working tree ~35 minutes later, at the end of this work:

| tree | changed | new | removed |
|---|---:|---:|---:|
| `taniteval` | **1** (`taniteval/closedloop.py`) | **4** (`taniteval/clhorizon.py`, `taniteval/ood.py`, `tests/test_clhorizon.py`, `tests/test_ood_guard.py`) | 0 |
| `stack` | **1** (`scripts/run_gate.py`) | **1** (`tests/test_run_gate_card_render.py`) | 0 |

> ⚠️ **pod2 is 7 files behind the repo as of the end of this run.** This is not a defect in the sync
> — it is the eval-pod staleness mechanism happening live, in 35 minutes, because sibling agents are
> writing. **A tree verified once is verified once.**

✅ **No number in this report depends on any of the 7.** The v1 evals go through
`runner → rollout → bench → driving → ci`; `corridor` and `lateral` were exercised directly;
`closedloop` was **imported but never called**. All were byte-identical at run time
(`artifacts/preflight_pod2.json` re-hashed both roots *at import*, not at sync).

⛔ **I deliberately did NOT re-sync.** `closedloop.py` and `run_gate.py` are under active edit by a
sibling agent; pushing a mid-edit tree onto the eval host is how a half-written module gets a number
attributed to it. **The re-sync is two commands and belongs to whoever runs the next eval:**

```bash
# 1. on the dev box, from the repo root (MSYS: use /c/... not C:/... for the output path)
tar --exclude=__pycache__ --exclude='*.pyc' \
    --exclude='stack/experiments/p0-s*' --exclude='stack/experiments/p0-fan-viz' \
    -czf /tmp/eh.tgz taniteval stack
python scripts/md5_manifest.py --root taniteval --out /tmp/md5_local_taniteval.json
python scripts/md5_manifest.py --root stack     --out /tmp/md5_local_stack.json
scp /tmp/eh.tgz tanitad-pod2:/root/

# 2. on pod2 — NOTE: mkdir before tar, --no-same-owner, and NOT behind `set -e`
ssh tanitad-pod2 'rm -rf /root/_s && mkdir -p /root/_s
  tar --no-same-owner --no-same-permissions -xzf /root/eh.tgz -C /root/_s 2>/dev/null
  rm -rf /root/taniteval /root/TanitAD/stack && mkdir -p /root/TanitAD
  mv /root/_s/taniteval /root/taniteval && mv /root/_s/stack /root/TanitAD/stack
  rmdir /root/_s'

# 3. ALWAYS, before quoting any number — run from /root/taniteval, not /root
scp scripts/preflight.py tanitad-pod2:/root/taniteval/_preflight.py
scp /tmp/md5_local_*.json tanitad-pod2:/root/          # the manifests _preflight.py checks against
ssh tanitad-pod2 'cd /root/taniteval && OMP_NUM_THREADS=8 \
  python3 /root/taniteval/_preflight.py /root/preflight_pod2.json'
```

⚠️ `_preflight.py` compares against `/root/md5_pod2_{taniteval,stack}.json`. Regenerate those on the
pod **from the freshly extracted tree** *and* diff them against the dev-box manifests — hashing only
the pod side proves the pod is self-consistent, not that it matches the repo.

⭐ **And the drift is good news for §4.5.** Two of the new files are precisely the instruments the
horizon recommendation needs:

* **`taniteval/clhorizon.py`** — "the HORIZON-CAPABLE closed loop", a **K-parameterised** port of the
  `incoming/` one-off driver, pinned bit-identical to it by `test_clhorizon.py`. **This is the thing
  that can run a K sweep {20, 40, 60, 70} at all.** Before it, the registered co-primary was
  reachable only through a driver in `incoming/`.
* **`taniteval/ood.py`** — "the OOD/EXTRAPOLATION guard, with its SATURATION declared", which records
  that `np.interp` **clamps** at `|dlat| = 3.0 m` so **every long-horizon OOD ratio the program has
  quoted is a lower bound**. **This is the thing that can check whether K = 60 is inside the
  envelope** — the `HYPOTHESIS` §4.5 explicitly refuses to assume.

**Escalated in §9.8: these two streams are halves of one next step and neither knows about the
other.**

---

## 3. The power the ladder actually gets at 600 episodes

### 3.1 ⭐ Per-stratum episode-cluster yield — the E1a strata, MEASURED on all 600

This is HP-2's bar (`≥ 200 / stratum`) and the CDR co-primary's headline stratification, computed
with the **shipped** code — `driving_diagnostic.net_heading_change_deg` (the exact call
`rollout.collect:161` makes) and `taniteval.corridor.strata` — over the real window starts of all 600
episodes. Poses-only, no GPU. `junction = |net heading change over the first 2 s| ≥ 10°`, held FIXED
across K so strata stay comparable. **Unit = episode cluster.**

| K | s | windows | **overall** | **junction** | longitudinal | other |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 2.0 | 13,198 | **600** ✅ | **232** ✅ | 380 ✅ | 351 ✅ |
| 30 | 3.0 | 12,554 | 600 ✅ | **227** ✅ | 377 ✅ | 348 ✅ |
| 40 | 4.0 | 11,399 | 600 ✅ | **218** ✅ | 369 ✅ | 347 ✅ |
| 50 | 5.0 | 10,799 | 600 ✅ | **213** ✅ | 369 ✅ | 344 ✅ |
| 55 | 5.5 | 10,202 | 600 ✅ | **207** ✅ | 364 ✅ | 339 ✅ |
| **60** | **6.0** | 10,198 | 600 ✅ | **207** ✅ | 364 ✅ | 339 ✅ |
| 65 | 6.5 | 9,599 | 600 ✅ | **204** ✅ | 358 ✅ | 338 ✅ |
| ⭐ **70** | **7.0** | 9,554 | 600 ✅ | ⭐ **204** ✅ | 358 ✅ | 338 ✅ |
| ⛔ **75** | **7.5** | 8,999 | 600 ✅ | ⛔ **196** | 356 ✅ | 336 ✅ |
| 80 | 8.0 | 8,399 | 600 ✅ | ⛔ 193 | 350 ✅ | 334 ✅ |
| 90 | 9.0 | 7,799 | 600 ✅ | ⛔ 188 | 346 ✅ | 333 ✅ |
| 100 | 10.0 | 7,198 | 600 ✅ | ⛔ 172 | 343 ✅ | 331 ✅ |
| 120 | 12.0 | 5,399 | 600 ✅ | ⛔ 137 | 327 ✅ | 322 ✅ |
| 150 | 15.0 | 3,554 | 600 ✅ | ⛔ 103 | 315 ✅ | 304 ✅ |
| 185 | 18.5 | 599 | **596** ✅ | ⛔ **58** | 287 ✅ | 251 ✅ |
| 190 | 19.0 | 558 | 558 ✅ | ⛔ 57 | 269 ✅ | 232 ✅ |

> ⛔ **THE FINDING. The ladder's binding constraint is the junction stratum, not the episode count.**
> Overall clusters never drop below 558 anywhere in the admissible range. **Junction crosses the 200
> two-arm bar between K = 70 (204) and K = 75 (196)** and collapses to **58 at K = 185**.
>
> ⚠️ **And the margin is thin even at its best.** The junction stratum's maximum is **232 / 600 =
> 0.387 yield**, against the feasibility threshold of `200/600 = 0.333`. HP-2 clears the bar by
> **16 % at K = 20 and by 2 % at K = 70.** It is the tightest problem in the ladder and it has no
> headroom for a further restriction (a stricter `junction_deg`, an option-set filter, or an
> OOD sub-stratum will push it under).

⚠️ **A kinematic signature, never a topology** (`corridor.py:165`). There is no lane graph in this
corpus; `junction` here is `|Δheading| ≥ 10° / 2 s`. HP-2's "multi-option" stratum and HP-4's
"junction topology classes" need the **VectorMap connectivity instrument**, which is a *different*
blocker from n and is not resolved by this work.

Artifacts: `artifacts/stratum_yield_600.json`, `artifacts/stratum_yield_600_fine.json`.
Script: `scripts/stratum_yield.py`.

### 3.2 ⭐ Per-HP: what is now runnable at n ≥ 200, and what is still short

Power bars read from `4BRAIN_DOMINANCE_PROGRAM.md` §3.2 (`PUBLISHED`, this program). The unit is the
**episode cluster** in every row — that is what the episode-cluster bootstrap resamples, and it is
why no amount of window-level data or stride reduction fixes a short problem (§4.3).

| HP | bar | the arithmetic on the 600 | verdict |
|:--:|---|---|---|
| **HP-1** advantage grows with horizon | n ≥ 200, 2-arm, paired Δ CDR & ADE at K ∈ {20, 60, 120, 185}, closed-loop | overall clusters **600 · 600 · 600 · 596** (windows 13,198 · 10,198 · 5,399 · 599). Ratio to bar: **3.0× · 3.0× · 3.0× · 2.98×** | ✅ **RUNNABLE at all four K.** The interaction test across K is fully powered on the pooled surface |
| **HP-2** advantage concentrates at decision points | ≥ 200 **per stratum** | junction: **232** (K=20) → **204** (K=70) → ⛔ **196** (K=75) → ⛔ **58** (K=185). longitudinal / other ≥ 251 everywhere | ⚠️ **RUNNABLE ONLY FOR K ≤ 70.** ⛔ **NOT runnable at K ≥ 75** — the stratum, not the corpus, is the wall. Max headroom **16 %** (232 vs 200) |
| **HP-3** route-conditionality | ≥ **40** decision clusters | S3 decision-point clusters at 600: **558 lat / 520 lon** (§3.3). The published 40 gave **37 / 34** — *below the 40 bar* | ✅ **RUNNABLE with ~14× margin.** The single largest unlock: it went from **failing the single-arm bar** to 13.9× over it |
| **HP-4** compositional generalisation to unseen junction **topologies** | ≥ 40 clusters **per topology class** | ⛔ **not an n problem.** PhysicalAI carries no lane graph, so there are no topology classes to hold out. Corpus-side upper bound if the instrument existed: **232 junction clusters ⇒ at most 5 classes at ≥ 40 each** | ⛔ **BLOCKED — on the VectorMap connectivity instrument, not on episodes.** Unchanged by this work |
| **HP-5** structure substitutes for data | 4 seeds × 3 data fractions, **20.2 GPU-days** | ⛔ **not a val-power question.** It is a *training* budget; the val side needs only a decision-grade ADE per cell, which the 600 provides | ✅ **UNBLOCKED on the eval side**, gated on GPU-days exactly as before |
| **HP-6** recovery / re-planning after perturbation | ≥ 200 | measured on all windows after a lateral offset, i.e. the **overall** stratum: **600** clusters at K ≤ 150, **596** at K = 185 | ✅ **RUNNABLE.** ⚠️ If the PI scopes it to junction approaches it inherits HP-2's K ≤ 70 wall |
| ⭐ **HP-7** branch-mean collapse | ≥ 200, at &#124;options&#124; ≥ 2, closed-loop | ⛔ **corpus-side upper bound = 232** (K=20 junction clusters), and `\|options\| ≥ 2` is **strictly stronger** than `\|Δheading\| ≥ 10°/2 s`, so the true yield is **≤ 232 and probably below 200** | ⛔ **MARGINAL AT BEST, and the option-set instrument does not exist yet.** Do not book compute on HP-7 before its yield is measured directly |
| ⭐ **HP-8** decision persistence | ≥ 200, `branch_flip_rate` over the last **10 s** of approach | needs a branch **and** a 10 s window: junction clusters at **K = 100 (10.0 s) = 172** ⛔; at K = 20 (if the 10 s is lead-in rather than horizon) **232**, again an upper bound before the `\|options\| ≥ 2` restriction | ⛔ **SHORT on the closed-loop reading (172 < 200)**, marginal on the lead-in reading. **The surface must be pinned before this is registered** |

> ### ⭐ The one-line answer
> **HP-1, HP-3, HP-5 and HP-6 are now runnable at n ≥ 200 (HP-3 by a factor of 14).**
> **HP-2 is runnable only at K ≤ 70.**
> **HP-4, HP-7 and HP-8 are still short — and none of them for a reason a bigger val fixes.**
> HP-4 and HP-7 need the **option-set / VectorMap instrument**; HP-8 needs that *and* clears the bar
> only if it is measured on a lead-in rather than a 10 s closed-loop horizon.

⚠️ **What this does NOT say.** "Runnable at n ≥ 200" is a statement about the **resampling unit**, not
about effect size. A comparison can be fully powered by this table and still be uninformative if the
effect is small or the estimand is degenerate — S3's parity re-run is the standing example (the
lateral problem is **85.5 % `t_none`**, so 558 decision-point clusters rest on **139 event-carrying**
episodes). **Report both counts or neither.**

### 3.3 ⭐ S3's yields — confirmed on this host, per stratum, and they reproduce EXACTLY

The brief asked for S3's **558 lat / 520 lon** to be confirmed on pod2. It was re-run end-to-end
here, on **pod2's own copies of both caches** — which is a materially different setup from the
original: the S3 agent mined a **1.86 MB poses-only VIEW relayed to pod3**, this run mined the
**real 66 GB cache in place**.

**Provenance of this run** (`artifacts/s3_power_pod2_parity_primary.json`):
`train = /workspace/…/_epcache/physicalai-train-e438721ae894` (2,376 files, `is_parity_cache: true`)
· `val = …/physicalai-val-0c5f7dac3b11` (600 files, `is_parity_cache: true`) ·
`H_S3 = 12 s`, `MIN_TTM_S = 1.0`, band edges `2 / 5 / 10 s` — the pre-registered primary, no change ·
estimator `episode_cluster_bootstrap (taniteval/ci.py) B = 2000; overlapping_holdout_se NEVER used`.
The four S3 files were **md5-verified identical** to the repo copies before the run
(`s3_labels.py c1b08d7b…`, `s3_blind_baseline.py ed0af06a…`, `run_s3_characterisation.py 620a8273…`,
`test_s3_labels.py 7122c718…` — the same digests the S3 report records), and `pytest -q` →
**20 passed**, matching pod3 and the dev box.

**The headline yields — `MEASURED` on pod2:**

| axis | clusters with a **decision point** | clusters with an **EVENT** | yield | ≥ 40 | ≥ 200 |
|---|---:|---:|---:|:--:|:--:|
| **lateral** | ⭐ **558 / 600** | ⛔ **139** | 0.9300 | ✅ | ✅ |
| **longitudinal** | ⭐ **520 / 600** | **312** | 0.8667 | ✅ | ✅ |

Train-side: **2,206 lat / 2,056 lon** clusters of 2,376.

**Per stratum, val (the answer the brief asked for):**

| axis | stratum | windows | **episode clusters** | ≥ 200? | majority rate |
|---|---|---:|---:|:--:|---:|
| **lat** | city `< 8 m/s` | 8,533 | **235** | ✅ | 0.6589 |
| **lat** | mid `8–15 m/s` | 10,387 | **224** | ✅ | 0.8303 |
| **lat** | highway `≥ 15 m/s` | 15,417 | **229** | ✅ | ⛔ 0.9809 |
| **lon** | city `< 8 m/s` | 5,209 | ⛔ **178** | ⛔ | 0.3467 |
| **lon** | mid `8–15 m/s` | 7,200 | ⛔ **179** | ⛔ | 0.5517 |
| **lon** | highway `≥ 15 m/s` | 12,578 | **214** | ✅ | 0.7615 |

Train-side per stratum: lat **878 / 956 / 916**, lon **689 / 738 / 860** — all ≥ 200.

> ### ⭐ **12 / 12 strata reproduce EXACTLY — window counts *and* cluster counts — across two hosts,
> two disks and two different data surfaces.**
> `558 / 139 / 0.9300` and `520 / 312 / 0.8667` land digit-for-digit on the pod3 figures. So does
> every stratum row above, and every train row. **The brief's `558 lat / 520 lon` is confirmed**, and
> the poses-only-view methodology the S3 agent used is now validated against the real cache rather
> than argued for.

⭐ **A fifth independent corroboration of the corpus fell out of it.** The miner's own log reports
`[physicalai-val-0c5f7dac3b11] DONE 600 eps … rows=102532` — **102,532** mined rows, matching
`labels_val_v4_provenance.json`'s `n_windows` *and* §4.1's stride-1 arithmetic *and* §2.3's coverage
figure. Four unrelated derivations, one number.

⚠️ **What the yields do NOT license, restated because it is the trap.** 558 counts episodes with an
admissible *decision point*; **139** carry an actual manoeuvre onset, and the lateral option set is
**85.5 % `t_none`**. Highway is the largest stratum by windows and the **thinnest by signal**
(majority rate **0.9809**). An arm that separates only on `t_none` recall has demonstrated nothing
about timing. **Report the decision-point count and the event count together, or neither.**

⛔ **Two S3 conclusions this run leaves exactly as the parity re-run left them** — unchanged, and
worth not re-litigating: the longitudinal **city (178)** and **mid (179)** strata **fail** the 200
two-arm bar while **highway (214) passes**, and the skill bars are the parity ones
(**lat ≈ 0.65 / lon ≈ 0.53**; S3-W **≈ 0.26 / ≈ 0.29** — see §3.3.1 for why these are now written to
two decimals). ⚠️ **The pre-parity bars (0.3898 / 0.2334) must not be quoted against any arm** — they
would score a −0.10 loss as a +0.16 win.

#### 3.3.1 ⭐ The firewall reproduces every VERDICT — and the BARS move by ±0.01

The blind-conditioning firewall was re-run too. Every **count** is identical to pod3's
(`n_train` 136,484 lat / 99,036 lon · `n_test` 34,337 / 24,987 · `n_test_episodes` **558 / 520**), so
the labelled data is the same data. Every **refusal verdict** is identical:

| rule | pod3 | **pod2** | verdict |
|---|---|---|:--:|
| **R1** circular (blind ≥ 0.98) | not refused (0.6534 / 0.5323) | not refused (**0.6493 / 0.5420**) | ✅ same |
| **R2** echo, lat `B2 − B1` | +0.3727 [+0.3163, +0.4292] sep | **+0.3740 [+0.3193, +0.4304]** sep | ✅ same |
| **R2** echo, lat `B3 − B1` | +0.3968 [+0.3426, +0.4492] sep | **+0.3902 [+0.3358, +0.4449]** sep | ✅ same |
| **R2** echo, lon `B2 − B1` | +0.0711 [+0.0308, +0.1127] sep | **+0.0796 [+0.0392, +0.1217]** sep | ✅ same |
| **R2** echo, lon `B3 − B1` | +0.2442 [+0.1966, +0.2951] sep | **+0.2539 [+0.2038, +0.3088]** sep | ✅ same |
| **R3** clock, lat `B4 − B3` | +0.0027 [−0.0162, +0.0216] **not** sep | **+0.0063 [−0.0121, +0.0245] not** sep | ✅ same |
| ⚠️ **R3** clock, lon `B4 − B3` | −0.0359 [−0.0634, −0.0108] **sep, NEGATIVE** | **−0.0475 [−0.0761, −0.0198] sep, NEGATIVE** | ✅ same — **still needs PI adjudication** |

> ⛔ **BUT the pre-registered skill bars are NOT reproducible to the precision they are quoted at.**
>
> | bar | pod3 | **pod2** | Δ |
> |---|---:|---:|---:|
> | S3 lateral | 0.6534 | **0.6493** | **−0.0041** |
> | S3 longitudinal | 0.5323 | **0.5420** | **+0.0097** |
> | S3-W lateral | 0.2566 | **0.2591** | **+0.0025** |
> | S3-W longitudinal | 0.2881 | **0.2881** | **0.0000** |
> | lon `B3 − B1` | +0.2442 | **+0.2539** | **+0.0097** |
> | lon `B4 − B3` | −0.0359 | **−0.0475** | **−0.0116** |
>
> Same code (md5-identical), same corpus (every count and every stratum identical), same estimator,
> same seeds — **different host.** The mined *labels* are deterministic; the **fitted blind heads are
> not**, to about **±0.01 QWK**. `HYPOTHESIS` for the mechanism: floating-point non-determinism in the
> BLAS/threading path of the fit (pod3 ran at a different thread count on a different CPU). **I did
> not instrument it and am not asserting the cause.**

⚠️ **Why this matters more than 0.01 sounds.** `skill = QWK(model) − bar`, and the bar is quoted to
four decimals as if it were a constant. An arm scoring **0.6510** clears the pod2 bar and fails the
pod3 bar. **Every conclusion in this report and in the S3 report survives** — the deltas are ~10× the
noise and every verdict is unchanged — but a *marginal* arm adjudicated against a 4-decimal bar would
be adjudicated by which pod fitted it.

> ⭐ **Recommendation (escalated, §9.9): pin the bar before it decides anything.** Fix
> `OMP_NUM_THREADS`, record it in the artifact, re-fit **n ≥ 5 times** and publish the bar as
> **`mean ± spread`** — or freeze one fitted bar as *the* registered constant with its host and thread
> count stamped. Quoting `0.6534` bare invites exactly the kind of retraction
> `RETRACTION_LOG.md` collects.


Artifacts: `artifacts/s3_power_pod2_parity_primary.json`,
`artifacts/s3_coverage_pod2_parity_primary.json`,
`artifacts/s3_option_set_pod2_parity_primary.json`,
`artifacts/s3_blind_baseline_pod2_parity_primary.json`.
Script: `scripts/s3_extract.py` (the cross-host diff).


---

## 4. What the horizon costs — the K vs window-yield curve

### 4.1 The rule, and its validation before use

The window rule is one line, `taniteval/rollout.py:130`:

```
starts = list(range(0, T - window - K, stride))          # window = 8
```

so an episode of length `T` yields `ceil((T − W − K)/stride)` windows at horizon K and **zero** once
`T − W − K ≤ 0`. `corridor.horizon_ceiling` states the same cap: `K_max(T) = T − W − 1`.

⭐ **Validated against three independently MEASURED points before it was used for anything:**

| point | source | computed here | match |
|---|---|---:|:--:|
| 40 published eps, `W=8`, `stride=8`, `K=20` → **881** windows | `MODEL_REGISTRY.md`; 30 of 30 decision-grade result JSONs | **881** | ✅ |
| 40 published eps, `W=8`, `stride=8`, `K=185` → **41** windows | `GATE_30K_RESULTS.md` §6.2 (`MEASURED`) | **41** | ✅ |
| 600 eps, `stride=1`, `K=20` → **102,532** windows | pod2 `/workspace/v15/labels_val_v4_provenance.json` | **102,532** | ✅ |

Three for three, on real per-episode `T` values rather than assumed ones. Everything below is
arithmetic on a rule that reproduces every published window count this program has.

### 4.2 The curve — `stride = 8`

`MEASURED` episode lengths: the **600** have `T ∈ [188, 205]`, mean **198.89**, and **15 of them are
shorter than the published 40's minimum of 198**. Structural K ceiling: **196** for *some* episode,
**179** for *every* episode.

| K | horizon | **600: windows** | **600: episode clusters** | win/ep | 40: windows | 40: clusters |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 2.0 s | 13,198 | **600** | 22.00 | 881 | 40 |
| 40 | 4.0 s | 11,399 | **600** | 19.00 | 761 | 40 |
| 60 | 6.0 s | 10,198 | **600** | 17.00 | 681 | 40 |
| 80 | 8.0 s | 8,399 | **600** | 14.00 | 561 | 40 |
| 90 | 9.0 s | 7,799 | **600** | 13.00 | 521 | 40 |
| 100 | 10.0 s | 7,198 | **600** | 12.00 | 481 | 40 |
| 120 | 12.0 s | 5,399 | **600** | 9.00 | 361 | 40 |
| 140 | 14.0 s | 4,198 | **600** | 7.00 | 281 | 40 |
| 150 | 15.0 s | 3,554 | **600** | 5.92 | 238 | 40 |
| 160 | 16.0 s | 2,399 | **600** | 4.00 | 161 | 40 |
| 170 | 17.0 s | 1,799 | **600** | 3.00 | 121 | 40 |
| **185** | **18.5 s** | **599** | ⭐ **596** | **1.00** | **41** | 40 |
| 190 | 19.0 s | 558 | **558** | 0.93 | 38 | 38 |
| 192 | 19.2 s | 3 | **3** | 0.01 | 0 | 0 |
| 196 | 19.6 s | 3 | **3** | 0.01 | 0 | 0 |

⛔ **The cliff is between K = 190 and K = 192**, where only the three 205-frame episodes survive.
`K ≥ 197` is structurally impossible on this corpus at any deployment.

### 4.3 ⛔ Stride — the inherited claim, tested rather than repeated

> **Inherited:** *"`starts = range(0, T−W−K, stride)` with T = 198–205 leaves ≤ 12 frames of slack —
> stride cannot buy windows."*

**`MEASURED`, and it is FALSE as literally stated:**

| K | stride 8 | stride 4 | stride 2 | stride 1 | clusters gained |
|---:|---|---|---|---|---:|
| 120 | 5,399 w / 600 c | 10,797 / 600 | 21,550 / 600 | 42,532 / 600 | **0** |
| 150 | 3,554 / 600 | 6,553 / 600 | 12,550 / 600 | 24,532 / 600 | **0** |
| **185** | **599 / 596** | 1,184 / 596 | 1,789 / 596 | ⭐ **3,548 / 596** | **0** |
| 190 | 558 / 558 | 561 / 558 | 567 / 558 | 579 / 558 | **0** |

At K = 185, stride 8 → 1 multiplies **windows ×5.9** (on the published 40: 41 → **244**, ×6.0). The
"≤ 12 frames of slack" argument only bites at **K ≈ 190**, where 558 → 579 is a 4 % change.

> ⭐ **But the conclusion the claim was made to support is CORRECT, and it gets sharper: stride
> changes the episode-cluster count by EXACTLY ZERO, at every K, at every stride.** Sub-stride
> windows sit **0.1 s apart inside the same episode**; they are near-perfectly correlated, the
> episode-cluster bootstrap resamples *episodes*, and 3,548 rows in 596 clusters carry the same
> information as 599 rows in 596 clusters. **Stride buys rows, never power.** State it that way — the
> "cannot buy windows" form is falsifiable, and it is false.

### 4.4 ⭐ Does the closed-loop co-primary reach n ≥ 200 at 600 episodes?

> ### ✅ **YES — on the pooled surface. 599 windows over 596 episode-clusters at K = 185**, versus the
> gate's **41 / 40**. That is a **14.9× increase in the resampling unit**, clearing the 200 bar by
> **2.98×**. **The co-primary does not need a shorter horizon for pooled power.**

**What it buys, projected.** The gate's `MEASURED` co-primary is
`corridor_departure_rate @1.75 m, K = 185 = 0.6388 [0.5565, 0.7128]` (episode-cluster bootstrap,
B = 2000, 41 win / 40 ep) — half-width **0.0781**.

| | n clusters | half-width | class |
|---|---:|---:|---|
| gate, published 40 | 40 | **0.0781** | `MEASURED` |
| **the 600 build** | **596** | ⭐ **≈ 0.0202** (×0.259) | `ESTIMATED` |

⚠️ **`ESTIMATED`, and it is a projection, not a measurement.** It assumes `1/√n_clusters` scaling at
an unchanged departure rate and ~1 window per cluster. A binomial cross-check at `p = 0.6388` agrees
on the *factor* (1.96·SE: 0.1489 → 0.0386, also ×0.26) while differing in level, because the
bootstrap interval is not binomial. **The real interval must come from running the block on the 600.**

⭐ **The √n assumption is no longer bare, though — it was checked on a real 40 → 600 eval this run**
(§5.1): eight open-loop metrics' episode-cluster-bootstrap half-widths shrank by **×2.8 – 3.9,
mean ≈ 3.4**, against the **×3.87** that `√15` predicts. So the *scaling* is `MEASURED` on this
corpus and this estimator; only its **transfer to a 1-window-per-cluster closed-loop rate at
K = 185** stays `ESTIMATED`. ⚠️ Note the open-loop metrics had **22 windows per cluster** to average
within — the K = 185 corridor rate has **1.00** — so if anything the projection is optimistic.

> ⛔ **The horizon still has to come down — for a DIFFERENT reason than the brief anticipated.**
> Pooled power at K = 185 is fine. **HP-2's junction stratum is not: 58 clusters, 3.4× under the
> bar** (§3.1). And the gate report's own warning survives intact — at K = 185 there is **1.00 window
> per episode**, so the bootstrap has **no within-cluster averaging at all**. The 600 fixes the
> *number* of clusters; it cannot manufacture a second window inside a 199-frame clip.

### 4.5 ⭐ The horizon the next gate should register

⚠️ **Attribution, checked at source rather than repeated.** The brief carried *"the 30 k gate report
recommends an intermediate K ≈ 60–120"*. **`GATE_30K_RESULTS.md` does not say that.** What it says
(§10.1, quoted) is: *"either register at a horizon where the envelope holds or re-validate P1 out to
18.5 s"*, and §6.4 records **`54.6 %` of rolled-out steps and `90.2 %` of windows leaving the
`|dlat| ≤ 3.0 m` P1 envelope at K = 185**. `gate_emitters.py:374` separately notes that a **`K ≥ 100`**
co-primary needs a closed-loop rollout. **The `60–120` band is `INHERITED` and unsourced; the
envelope instruction is `PUBLISHED` and is the one that binds.** The recommendation below satisfies
the *envelope* instruction and the power arithmetic simultaneously — which is a stronger argument
than the band would have been.

Crossed with the measured stratum yields:

| candidate | overall | **junction** | longitudinal | other | verdict |
|---|---:|---:|---:|---:|---|
| K = 20 (2.0 s) | 600 | **232** ✅ (+16 %) | 380 | 351 | the current instrument; demoted by the gate because it hides the failure ~168× |
| ⭐ **K = 60 (6.0 s)** | 600 | ⭐ **207** ✅ (+3.5 %) | 364 | 339 | **the safe register.** Inside the report's own band, every stratum powered |
| ⭐ **K = 70 (7.0 s)** | 600 | ⭐ **204** ✅ (+2.0 %) | 358 | 338 | **the maximum admissible horizon.** Every stratum powered, no margin left |
| K = 75 (7.5 s) | 600 | ⛔ **196** (−2 %) | 356 | 336 | ⛔ **the first K at which HP-2 fails** |
| K = 120 (12.0 s) | 600 | ⛔ **137** (−32 %) | 327 | 322 | pooled-only; HP-2 unmeasurable |
| K = 185 (18.5 s) | 596 | ⛔ **58** (−71 %) | 287 | 251 | pooled-only, 1.00 window/episode |

> ### ⭐ RECOMMENDATION
> **Register the next gate at `K = 60` (6.0 s) as the primary closed-loop horizon, with `K = 70`
> (7.0 s) documented as the hard maximum and `K = 185` retained as a REPORT-ONLY pooled co-primary.**
>
> * **`K = 60` clears the 200 two-arm bar on every E1a stratum — junction included (207) — on the
>   maximum possible parity val.** It is 3× longer than the demoted 2 s instrument and carries
>   **10,198 windows over 600 clusters** (17 windows/episode, so the bootstrap has real
>   within-cluster averaging, which K = 185 does not).
> * ⚠️ **It also serves the gate's binding instruction, and this part is a `HYPOTHESIS` I am
>   labelling as one.** §6.4 measured **54.6 % of steps / 90.2 % of windows outside the `|dlat| ≤
>   3.0 m` P1 envelope at K = 185**. A 6 s rollout accumulates far less lateral drift than an 18.5 s
>   one, so K = 60 is *likely* to sit inside the envelope — **but that has NOT been measured, and it
>   must be, on v4 and not on v1** (the envelope constants come from `lowood_flagship_ci.json`, a v1
>   artifact). **Do not register the bar until the envelope is checked at the chosen K.**
> * **`K = 70` is the hard ceiling for any *stratified* verdict.** Above it, HP-2 is not measurable at
>   parity **by any deployment, ever** — 600 is the corpus maximum (`randperm(3000, seed 0)`, first
>   600 to val), so there is no larger val to move to.
> * **`K = 185` stays worth reporting** — 596 clusters, projected half-width ≈ 0.02 — but as a
>   **pooled** number only, and **never** with a junction breakdown.
> * ⚠️ **Set the bar from the interval, not the point estimate** (`GATE_30K_RESULTS` §6.2), and quote
>   **K, `n_windows` AND `n_episode_clusters`** with every corridor number. A corridor number without
>   its K and its n is not admissible.
> * ⚠️ **Pre-register both outcomes before the run.** The discriminating question the gate report
>   actually poses — *where does the loop leave the envelope, and where does CDR saturate* — is
>   answerable on a K sweep {20, 40, 60, 70} at **full stratified power**, which is strictly more
>   informative than one number at K = 185 that HP-2 cannot use.

Artifacts: `artifacts/horizon_yield.json` (K = 20…190, both deployments, 4 strides),
`artifacts/horizon_analysis.json` (stride test, CI projection, the n ≥ 200 crossing).
Scripts: `scripts/horizon_yield.py`, `scripts/horizon_analysis.py`.

### 4.6 ⭐ The stratum reconstruction, cross-validated against the gate's own artifact

Before any of §3.1's 600-episode stratum numbers were used, the same code was run on the **first 40
episodes of the 600** — which §1.4 proves *are* the published set — and compared to the gate's own
paired artifact `coprimary/paired_v4_vs_refc_K185.json`, which records `n_by_stratum` from the real
closed-loop rollout.

| K = 185, published 40 | gate artifact (`MEASURED`, closed-loop rollout) | this reconstruction (poses-only) | match |
|---|---|---|:--:|
| overall | **41 win / 40 ep** | **41 win / 40 ep** | ✅ |
| **junction** | **6 win / 6 ep** | **6 ep** | ✅ |
| longitudinal | **18 win / 18 ep** | **18 ep** | ✅ |
| other | 17 win / **16 ep** | **16 ep** | ✅ |

And at K = 20 the same 40 episodes give **881 windows** — the published number, a fourth time.

> ⭐ **The poses-only stratum reconstruction reproduces the gate's stratification episode-for-episode.**
> That is what licenses §3.1's 600-episode numbers: they come from the same call
> (`net_heading_change_deg`, `corridor.strata`, `junction_deg = 10.0`) on the same window starts, and
> the method has been checked against a rollout that actually ran.

**The 40 → 600 unlock, per stratum** (`MEASURED`, episode clusters):

| stratum | K = 20: 40 eps → 600 | K = 60: 40 → 600 | K = 185: 40 → 600 |
|---|---|---|---|
| overall | 40 → **600** (15.0×) | 40 → **600** | 40 → **596** (14.9×) |
| **junction** | 22 → **232** (10.5×) ✅ | 19 → **207** (10.9×) ✅ | 6 → **58** (9.7×) ⛔ |
| longitudinal | 24 → **380** (15.8×) ✅ | 23 → **364** ✅ | 18 → **287** (15.9×) ✅ |
| other | 24 → **351** (14.6×) ✅ | 22 → **339** ✅ | 16 → **251** (15.7×) ✅ |

⚠️ **A precision note for the gate report, offered as a fix rather than a retraction.** §6.3 correctly
discloses `JUNCTION | 6 / 6`. The **headline table at line 23** quotes the junction number
(`0.8432 [0.7874, 0.8919]`) beside the n-column value **`41 win / 40 ep`**, which is the *overall*
n. A reader who stops at the headline inherits a junction interval attributed to 40 clusters when it
rests on **6**. One column, and it is the same class of defect the val-parity work was chartered to
close. **Suggested fix: `41/40 overall · 6/6 junction`.**

Artifact: `artifacts/stratum_yield_40.json`.

---

## 5. What was run on the new capacity

### 5.1 ⭐ v1 (`flagship-30k`) on all 600 episodes — the ladder's reference arm at n = 600

`python3 -m taniteval.runner run --model flagship-30k --episodes 600` · **901.4 s** wall ·
`ckpt_step 29999` · **13,198 windows / 600 episode clusters** · estimator
**`episode_cluster_bootstrap`, B = 2000, unit = val episode**.
Chosen over the REF-C-base eval the brief suggested because **REF-C-base is not on pod2** (it is a
1.25 GB move from the eval pod, §7) and because *every* two-arm HP-x comparison is paired against v1
— without a v1 reference at n = 600 the ladder has nothing to pair to.

| metric | **40 episodes** (881 win) | **600 episodes** (13,198 win) | CI half-width ratio |
|---|---|---|---:|
| `ade_0_2s` | 0.4271 [0.3675, 0.4871] hw **0.0299** | **0.4108 [0.3956, 0.4273]** hw **0.0080** | **×3.76** |
| `ade@0.5s` | 0.0720 [0.0629, 0.0824] | 0.0714 [0.0684, 0.0750] | ×2.94 |
| `ade@1s` | 0.1467 [0.1252, 0.1704] | 0.1391 [0.1330, 0.1457] | ×3.59 |
| `ade@1.5s` | 0.2670 [0.2261, 0.3073] | 0.2576 [0.2470, 0.2690] | ×3.69 |
| `fde@2s` | 0.9075 [0.7851, 1.0306] | 0.8704 [0.8397, 0.9038] | ×3.83 |
| `miss_rate@2m` | 0.0454 [0.0239, 0.0681] | 0.0415 [0.0339, 0.0497] | ×2.80 |
| `tms_openloop` | 0.0978 [0.0701, 0.1304] | 0.1077 [0.1000, 0.1154] | ×3.91 |
| `rmse` | 0.6563 [0.5662, 0.7445] | 0.6427 [0.6145, 0.6727] | ×3.07 |

> ⭐ **The intervals shrink by ×2.8–3.9, mean ≈ ×3.4, against the `√15 = 3.87` that √n predicts.**
> **The √n scaling assumption behind §4.4 is therefore `MEASURED` on this corpus and this
> estimator** — which is what makes the K = 185 corridor projection (half-width 0.078 → ≈ 0.020)
> worth acting on. ⚠️ **The corridor projection itself stays `ESTIMATED`**: these eight metrics had
> **22 windows per cluster** to average within, the K = 185 corridor rate has **1.00**, and it is a
> rate rather than a mean. The scaling transfers *plausibly*, not provably.

⛔ **This is NOT a correction to the published 0.4271, and must never be quoted as one.** It is a
**different deployment** of the same corpus — a strict order-preserving superset. Two things make
them non-interchangeable:

* **n differs** (40 vs 600 clusters, 881 vs 13,198 windows).
* ⚠️ **The 600 is an EASIER corpus by the trivial floor.** `CV ade_0_2s` moves **0.8377 → 0.6917**,
  so the extra 560 episodes are, on average, more predictable than the published 40. The model's
  margin over CV therefore *falls* (Δ **+0.4106 → +0.2809**) even though its absolute ADE improves.
  **Any 40-vs-600 delta is confounded by corpus composition and must not be read as a model result.**

`MODEL_REGISTRY.md`'s v1 row stays **0.4271 [0.3675, 0.4871] @ 40 eps / 881 win**. The new number is
`v1 @ 600 eps = 0.4108 [0.3956, 0.4273]` and should be registered **as its own row**, with its n.

### 5.2 ⭐⭐ Three driving-panel verdicts FLIP at n = 600 — and one of them flips on power alone

`taniteval.driving` tier-0, paired against the trivial floors,
`paired_episode_cluster_bootstrap`, B = 2000:

| paired test | 40 episodes | **600 episodes** | what moved |
|---|---|---|---|
| ⭐ **`along_track_vs_cv`** | δ **0.2543** [**−0.0278**, 0.5304] ⛔ **not separated → "tie"** | δ **0.2525** [**+0.1926**, +0.3104] ✅ **separated → "model"** | ⭐ **the effect is IDENTICAL (−0.7 %); only the interval moved.** A pure power flip |
| `speed_mae_vs_cv` | δ −0.0032 [−0.1285, +0.1182] ⛔ tie | δ **+0.0504** [+0.0209, +0.0770] ✅ model | ⚠️ effect **changed sign** — power **and** composition |
| `speed_mae_vs_holdv0` | δ +0.0108 [−0.0991, +0.1179] ⛔ tie | δ **+0.0527** [+0.0264, +0.0769] ✅ model | ⚠️ effect grew ~5× — power **and** composition |
| `cross_track_vs_cv` | δ 0.7720 ✅ model | δ 0.4178 ✅ model | unchanged verdict, smaller margin (easier corpus) |
| `cruise_speed_vs_holdv0` | δ −0.2122 ✅ **floor wins** | δ −0.1761 ✅ **floor wins** | unchanged — v1 still loses cruise speed to hold-v0 |

The one-line summary the runner prints moves from
`win lives: lateral only · tracks speed > CV: False` to
`win lives: both axes · tracks speed > CV: True`.

> ⭐ **`along_track_vs_cv` is the clean demonstration of what n ≥ 200 buys.** The point estimate is
> **0.2543 → 0.2525** — a 0.7 % change, i.e. nothing — while the interval narrows **×3.9** and the
> verdict goes from *"tie"* to *"model wins, CI-separated"*. **A real effect was sitting under the
> 40-episode noise floor the whole time.** This is exactly the failure mode the ladder's n ≥ 200 bar
> exists to prevent, caught on the program's own reference arm.
>
> ⚠️ **The other two flips are NOT clean** — their effect sizes moved too (one changed sign), so
> power and corpus composition are confounded there. **The honest decomposition** — re-running the
> 600-episode paired test restricted to the common 40 episodes — was **not done here** and is the
> correct follow-up. I am not attributing those two to power.
>
> ⚠️ **And a standing caution:** "v1 tracks speed better than CV" is now `separated` on 600 episodes
> at δ **+0.0504 m/s**. That is a *statistically* separated **5 cm/s** difference. Separation is not
> materiality — the longitudinal-blindness finding is not overturned by this, and `cruise_speed`
> still favours the trivial floor on **both** deployments.

Artifacts: `artifacts/RESULT_v1_600ep.json`, `artifacts/v1_40_vs_600.json`.


---

## 6. Operational notes — earned on this run

| # | note | class |
|:--:|---|---|
| **1** | ⛔ **A script placed at `/root/` cannot `import taniteval`.** `sys.path[0]` becomes `/root`, which holds the *directory* `taniteval` with no `__init__.py`, so the name binds to an empty namespace package: `ImportError: cannot import name 'bench' from 'taniteval' (unknown location)`. **Always run from `/root/taniteval`.** | `MEASURED` |
| **2** | ⚠️ **`tar -xzf` of a Windows-authored bundle fails under `set -e`** on `Cannot change ownership to uid 197609` — the *files extract fine*, the chown does not. Use `--no-same-owner --no-same-permissions`, and do not put `tar` behind `set -e`. | `MEASURED` |
| **3** | ⚠️ **MSYS `tar` reads `C:/…` as a remote host** (`Cannot connect to C: resolve failed`). Use `/c/Users/…` for local paths on the dev box. | `MEASURED` |
| **4** | ✅ **`OMP_NUM_THREADS=8` held throughout**, per the 9×-slowdown finding. With one CPU miner + one GPU eval concurrently, load average stayed **13–20 on 96 cores**, CPU **~6 % user**, RAM **40 / 503 GB**. **No stall** of the kind two uncapped miners produced on pod3 — the miner sat in state `D` at 13 % CPU, i.e. **IO-blocked, not CPU-starved**. | `MEASURED` |
| **5** | ⚠️ **The real cost driver on pod2 is MooseFS read latency, not threads.** Poses-only mining runs at **~0.76 s/episode alone** and **~1.6 s/episode while a GPU eval is reading the same cache** — against pod3's **0.223 s/ep** on a 1.86 MB local poses view. **Budget ~3.5× the pod3 figure for a poses-only job against the real 66 GB cache, and ~7× if an eval is running.** Serialise big jobs when wall-clock matters; they do coexist correctly, just slower. | `MEASURED` |
| **6** | ⚠️ **`taniteval.runner` writes into `/root/taniteval/results/`**, overwriting the repo-synced result JSONs of the same name. Back up any result you need to keep *before* the next run of the same arm (done here: `RESULT_v1_40ep_preflight.json`). It also means the md5 tree check must be run **before** the first eval, or it will report expected mismatches under `results/`. | `MEASURED` |

## 7. What pod2 now is, and what it costs to keep

| | |
|---|---|
| **Host** | `tanitad-pod2` (A40 46 GB, 96 cores, 503 GB RAM, `/` overlay 500 G with 500 G free) |
| **Corpus** | the **only** copy of the 600-episode CLEAN val, 66 GB, verified this run, **read-only** |
| **Harness** | `taniteval` 217 files + `stack` 344 files, byte-identical to the repo, at the two hard-coded paths |
| **Arms present locally** | v1 (`flagship4b-speedjerk-30k`), v2, v3enc, v4 / v4.1 / v4.2 / v4.2b, v1.5 a/ab/abc, v1.6-ab-ft, REF-B 30k, REF-C-diffusion-small-v21-30k, phase0-30k (the no-speed control) |
| ⚠️ **Arms NOT present** | **`refc-base-30k`** — the D-030 middle rung — and `refc-xl-30k`. They live on the eval pod. A two-arm HP-x against REF-C-base needs a **1.25 GB** move (HF relay, ~2 × 10 s at the measured 118 MB/s, plus md5) |
| **Marginal cost of an n = 600 open-loop eval** | ~15× the 40-episode run's 101 s of GPU, plus the 66 GB cache read |
| **What must NOT happen** | the cache must not be moved, deduplicated, or "tidied". It is the only copy, and every n ≥ 200 comparison in the ladder depends on it |

## 8. Deliverable manifest

All paths relative to
`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-pod2-eval-host/`
in the **repo working tree on the dev box**. ⛔ **Nothing was `git add`-ed, committed or pushed.**

| file | what |
|---|---|
| `POD2_EVAL_HOST.md` | this report |
| `artifacts/verify_val600_pod2.json` | per-episode `sha256(poses)`, `T`, `episode_id` for all **600** |
| `artifacts/verify_train_pod2.json` | the same for pod2's **2,376**-episode parity train |
| `artifacts/verify_evalpod40.json` | the same for the eval pod's **published 40** |
| `artifacts/prefix_disjointness_result.json` | the disjointness verdict + the 40-position prefix check |
| `artifacts/md5_compare.json`, `md5_{local,pod2}_{taniteval,stack}.json` | the byte-level sync proof, at sync time |
| `artifacts/md5_local_{taniteval,stack}_T2.json` | the **same trees ~35 min later** — the 7-file drift of §2.6 |
| `artifacts/preflight_pod2.json` | the full `sys.path` audit, tree re-hash, 46 module paths, 8 trees, P1/P2/P3 |
| `artifacts/RESULT_v1_40ep_preflight.json` | v1 @ 40 → **0.4271 [0.3675, 0.4871]** |
| `artifacts/RESULT_v1_600ep.json` | v1 @ 600 (see §5) |
| `artifacts/horizon_yield.json` | K = 20…190 × stride {8,4,2,1} × {40, 600} |
| `artifacts/horizon_analysis.json` | stride test, CI projection, the n ≥ 200 crossing |
| `artifacts/stratum_yield_600.json`, `stratum_yield_600_fine.json`, `stratum_yield_40.json` | E1a strata per K |
| `artifacts/s3_*_pod2_parity_primary.json` | the S3 power / coverage / firewall re-run on pod2's caches |
| `scripts/*.py` | every script used, runnable as-is |

**On pod2** (`tanitad-pod2`), left in place and re-usable:

| path | what |
|---|---|
| `/root/taniteval`, `/root/TanitAD/stack` | the verified harness |
| `/root/valdata/physicalai-val-0c5f7dac3b11` | symlink → the 600 build (the cache itself untouched) |
| `/root/models/flagship-30k/ckpt.pt` | symlink → v1 FINAL |
| `/root/TanitAD/s3/` | the four S3 files, md5-identical to the repo |
| `/root/{verify_*,preflight_pod2,stratum_yield_*,RESULT_v1_*}.json`, `/root/s3out/` | the same artifacts, pod-side |
| `/root/taniteval/_preflight.py` | ⭐ **re-run this before quoting any number from a new invocation shape** |

## 9. Escalations — integration, not a note in a README

1. ⭐ **The next gate's registered horizon.** §4.5 recommends **K = 60 primary / K = 70 maximum /
   K = 185 report-only-pooled**. This needs a PI decision *before* the gate is written, because
   registering K ≥ 75 makes HP-2 unmeasurable at parity **permanently** — 600 is the corpus maximum
   and there is no larger val to move to. **Owner: whoever writes the next `GATE_*.md`.**
2. ⭐ **HP-7 and HP-8 should not be booked yet.** Their corpus-side upper bound is **232** clusters
   (K = 20 junction) *before* the `|options| ≥ 2` restriction, and HP-8's closed-loop reading is
   already at **172**. Both also need the option-set instrument that does not exist. **Measure the
   `|options| ≥ 2` yield directly before any GPU is committed.** **Owner: 4-Brain Dominance Program.**
3. ⚠️ **A one-column fix to `GATE_30K_RESULTS.md` line 23** — the junction co-primary is quoted
   beside `41 win / 40 ep`, the *overall* n, while it rests on **6 / 6** (correctly disclosed in
   §6.3). Suggested: `41/40 overall · 6/6 junction`. **Owner: the gate report's author.**
4. ⭐ **A val uid digest is now MEASURED and could be registered.** The guard computed
   `episode_uid_sha256 = 75a4d429be8cef8ea47a319e2033d792ee9eecbff033fad27dbb624b5634df20` over the
   600 uids of a cache this report verifies at the byte level, with **0 read errors** and **0 overlap
   with the parity train**. The manifest's val entry is `uid_source: count-only-unrecorded`, so every
   val check in the program degrades to COUNT-ONLY. **The command is
   `scripts/make_parity_manifest.py --record --split val --cache-dir <this cache>` on pod2.**
   ⚠️ **I did NOT run it** — writing the shared manifest is a program-wide change and it belongs to
   the parity guard's owner, not to a passing agent. **Owner: `tanitad/data/parity.py`'s author.**
5. ⚠️ **`refc-base-30k` is not on pod2.** Every REF-C two-arm comparison at n ≥ 200 needs a 1.25 GB
   checkpoint move (HF relay, or the eval pod → dev box → pod2 path). Cheap, but it is a prerequisite
   nobody has scheduled. **Owner: whoever runs the first ladder comparison.**
6. ⭐ **`MODEL_REGISTRY.md` should gain a v1 @ 600 row — as a NEW row, never as an edit to 0.4271.**
   `flagship-30k @ 600 eps = ade_0_2s 0.4108 [0.3956, 0.4273]`, 13,198 win / 600 clusters,
   `episode_cluster_bootstrap` B = 2000, artifact `artifacts/RESULT_v1_600ep.json`. ⚠️ It must carry
   **its n and the CV floor of its own deployment (0.6917, vs 0.8377 on the 40)** — the two
   deployments differ in difficulty, and a reader who sees `0.4271 → 0.4108` without that will read
   an improvement that is not there. **This is exactly the class of error `RETRACTION_LOG.md`
   exists for; I am flagging it rather than writing the row.**
   **Owner: `MODEL_REGISTRY.md`'s maintainer.**
7. ⭐ **The `along_track_vs_cv` flip should be checked against every "tie" this program has
   published on 40 episodes.** One tie on the reference arm turned out to be a real, CI-separated
   effect once n went to 600, with the point estimate moving 0.7 %. **Any decision that rested on a
   40-episode "not separated" is now suspect on power grounds** — not wrong, *unpowered*, which is a
   different and cheaper thing to fix now that the host exists.
   **Owner: 4-Brain Dominance Program / whoever owns the tie-based scoping decisions.**
8. ⭐⭐ **THE INTEGRATION: this recommendation and `taniteval/clhorizon.py` + `taniteval/ood.py` are
   halves of one step, and neither stream knows about the other.** A sibling agent landed both
   modules in the repo working tree **during this run** (§2.6). `clhorizon.corridor_rollout` is
   K-parameterised, so it is **the only thing in the package that can run the K sweep {20, 40, 60,
   70}** §4.5 asks for; `ood.py` is **the only thing that can answer whether K = 60 sits inside the
   `|dlat| ≤ 3.0 m` envelope**, which §4.5 flags as an unmeasured `HYPOTHESIS` and refuses to assume.
   Meanwhile this work supplies the **stratum yields that say which K is admissible at all** — which
   neither module measures. **One run on pod2 closes all three: `clhorizon` sweep × `ood` envelope ×
   600 episodes.** ⚠️ This is written here **and must be raised directly**, per the standing rule
   that cost an orthogonality instrument 10 days inside a README.
   **Owner: PI, to pair the two streams before either books GPU.**
9. ⛔ **The S3 skill bars are quoted to 4 decimals and reproduce to ~±0.01 across hosts** (§3.3.1).
   Same md5-identical code, same corpus (every count and stratum identical), different pod:
   **lat 0.6534 → 0.6493, lon 0.5323 → 0.5420, lon `B3−B1` +0.2442 → +0.2539**. Every *verdict*
   survives — the deltas are ~10× the noise — but `skill = QWK(model) − bar` means an arm at **0.6510**
   clears one pod's bar and fails the other's. **Fix `OMP_NUM_THREADS`, stamp it in the artifact,
   re-fit n ≥ 5 and publish `mean ± spread`; or freeze one fit as THE registered constant with its
   host recorded.** ⚠️ Do this **before** the bar adjudicates an arm, not after.
   **Owner: S3's author / `PRE_REGISTRATION_S3.md` §5.3.**

## 10. What was deliberately NOT done

* ⛔ **pod1 was never contacted.** It is training.
* ⛔ **The 600-episode cache was never written to, moved, deduplicated or re-selected.** Only `ls`,
  `du` and `torch.load(..., mmap=True)` touched it; the symlink lives in `/root`.
* ⛔ **`physicalai-val-f1b378f295ae` was never opened.** It is not on pod2 at all.
* ⛔ **No `git add`, no commit, no push.**
* ⛔ **No interval from `overlapping_holdout_se` is quoted anywhere in this report.** Where a result
  JSON carries one it is named `legacy_overlapping_holdout_se` and ignored.
* ⛔ **The eval pod was touched read-only** (one `torch.load(mmap=True)` pass over 40 episodes'
  poses, 0.0 s, to obtain the published set's hashes). A sibling agent's code work there was not
  disturbed; nothing was written except one probe script under `/root/_corpus_verify_probe.py`.
