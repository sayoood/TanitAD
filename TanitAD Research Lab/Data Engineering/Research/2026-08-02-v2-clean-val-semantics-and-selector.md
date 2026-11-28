<title>A3 unblocked: pool column semantics, and why "6.77× headroom ⇒ feasible" was one axis short (2026-08-02)</title>

# The clean v2-line validation split — semantics established, feasibility re-measured, selector built

**Date:** 2026-08-02 (local, Europe/Berlin; the session ran past midnight into 08-03).
**Author:** Data Engineering agent (weekly run). **Branch:** `agent/data-engineering-20260802`
(worktree `C:/Users/Admin/wt-de-0802`, D-026).
**Package:** `Data Engineering/Implementation/incoming/2026-08-02-v2-clean-val-selector/`
(`pool_columns.py`, `clean_val_select.py`, `make_results.py`, `tests/` **27 ✓**, 3 JSON artifacts).
**Backlog item executed:** `Project Steering/BACKLOG.md` **A3** — 0-GPU, unblocked, and the only
data-engineering item on the live pull-list.

Every number below is **MEASURED (ours, this session)** unless tagged otherwise, and every one is
regenerable with `python make_results.py` (~6 s, dev-box CPU, **$0**, no GPU, no pod, no network).
Artifact: `selector_comparison.json`.

---

## 0. Headline

1. **The blocker is gone.** All **34** semantic checks on `v2_pool_scored.parquet` PASS on the real
   **18,988 rows**. `lk` is a frame COUNT out of `nlab`; `stopped/city/hw` are per-window
   FRACTIONS that partition the window. Both are now a contract in code with corruption tests.
2. ⛔ **A3's "6.77× headroom ⇒ FEASIBLE" is a one-axis number.** On the four-family axis set it is
   **1.07×**; add one more tactical axis and it is **0.77× — infeasible**. Headroom is a property
   of the axis set, never of the corpus.
3. ⛔ **Cell-matching — the design A3 proposed — does not produce a balanced val.** It leaves
   max |d| **0.3997** (10 of 13 axes over the 0.10 bar). Covariate balancing reaches **0.0094**.
4. **And matching cannot be fixed by finer cells**, because the remainder is skewed *inside* every
   matched cell: median |d| **0.359**, p90 0.915, max 2.351.
5. ⛔ **The bigger finding: a "clean v2 val" is not clean for v1.** A v2-only-clean 600-clip draw
   contains **62 clips from v1's TRAIN split** (and 24 from v1's val). The shipped manifests
   therefore exclude the entire parity selection — and once they do, **600 clips is not available**
   (headroom 0.95). **n=400** is the largest fully-clean, fully-balanced split.
6. **Delivered:** frozen candidate manifests at n=400 (sha256 `abe041db72a045b3…`) and n=300, each
   with a disjointness proof, a per-family balance table, and the per-cell pool-depth cost.

---

## 1. The blocker, and why it was a contract problem and not an arithmetic slip

`fe400f0` (2026-07-29) recorded the stop:

> ⛔ `stopped`, `city` and `hw` are not clean 0/1 columns … and **`lk` is not a rate at all**
> (train mean 80.55, remainder 120.24). My first pass computed `needed_in_val = 48330` …
> **Each column needs its semantics established before it becomes a stratification axis.**

The semantics were never missing — they are in the emitting code, `score_v2_pool.py:64-88`, which
reuses `stack/scripts/refb_labels.py` verbatim. What was missing was anything that made a misread
*fail* rather than propagate. So the deliverable is a registry with identities, not a note.

| class | columns | what it means | may it be a stratum axis? |
|---|---|---|---|
| **BINARY** | `junction`, `has_turn`, `has_brake`, `has_stop` | presence in the 20 s window | yes |
| **FRACTION** | `stopped` (v<1.0), `city` (1≤v≤12), `hw` (v>12), `stop_frac` (v<0.5) | share of frames — **not flags**, and `stopped+city+hw ≡ 1` | yes (as a mean) |
| ⛔ **COUNT** | `lk,tl,tr,ac,bs` and `lk2…bs2` | frames per maneuver class, **out of `nlab`** | **no — divide by `nlab` first** |
| **SCALAR** | `mean_v`, `dist_m`, `net_head`, `cum_head`, `T`, `nlab`, `win_s` | physical quantities | yes |

**The identities that prove it (all PASS, n=18,988):** `lk+tl+tr+ac+bs ≡ nlab` (and the v2 twins) ·
`nlab ≡ T − 20` · `stopped+city+hw ≡ 1` · `stop_frac ≤ stopped` · `cum_head ≥ net_head` ·
`has_turn ≡ (tl+tr>0)` · `has_brake ≡ (bs>0)` · `has_stop ≡ (stop_frac>0)` · 25 domain checks.

**The corrected column, and a nuance worth keeping.** `nlab ≡ 179` for **every** clip (T ≡ 199), so
`lk` and `lk/nlab` are related by a constant — the misread was a pure **scale** error and did not
reorder anything. Corrected: `lk_rate` train **0.4500** vs remainder **0.6718** — which reproduces
`V2_CORPUS_DESIGN.md`'s target "lane_keep 45.0 %" exactly, the cross-check that the reading is right.

**A second correction, small but load-bearing for provenance.** The pool has **18,988 rows and
18,987 unique clip_ids** — clip `32ad1a3a-c407-4cd3-afd8-243315e5f37a` is registered under **two
chunks (1573 and 3117)** with byte-identical scores, and it **is** in the v2 selection. Both
figures are in circulation (`V2_CLEAN_VAL_PLAN.md` says 18,987, `score_v2_pool.py`'s docstring says
18,988) and neither said which quantity it was. `load()` now de-duplicates and the count is
recorded in the artifact.

## 2. Feasibility: the 6.77× was one axis wide

A3 measured headroom on `junction` alone. Re-measured over axis sets, all at n=600, on the same
9,987-clip remainder:

| cell axes | cells | binding cell | **headroom** | max n at ≥3× depth |
|---|---|---|---|---|
| `junction` ← **the A3 number** | 2 | `1` | **6.77×** | 1,353 |
| `junction + has_turn` | 4 | `0\|1` | 4.05× | 813 |
| `junction + speed` | 6 | `1\|s2` | 1.98× | 396 |
| ⭐ `junction + has_turn + speed` | 12 | `0\|1\|s2` | **1.07×** | **213** |
| `+ has_brake` | 23 | `0\|1\|s2\|b1` | **0.77× ⛔ infeasible** | 153 |

The binding cell throughout is **no-junction / turning / fast** — the remainder holds **81** such
clips and a matched 600-split needs **76 of them (93.8 %)**. This is not a corpus fact and must
never be quoted bare: *headroom belongs to its axis set*, exactly as an exponent belongs to its fit
window. `feasibility()` therefore takes `cell_axes` and returns it inside the result.

## 3. The design A3 proposed does not produce a balanced val

Four selection arms, n=600, drawn from the same remainder, scored against the v2 TRAIN corpus on
the four-family axis set (13 axes; Cohen's d standardised on the train sd; bar |d| < 0.10):

| arm | max \|d\| | median \|d\| | axes over bar | max KS | cell L1 | max cell census |
|---|---|---|---|---|---|---|
| ⭐ **greedy balanced** | **0.0094** | 0.0037 | **0 / 13** | 0.1221 | 0.4778 | **0.500** |
| ⭐ **hybrid (cell quota + balance)** ← shipped | 0.0532 | 0.0234 | **0 / 13** | 0.1866 | **0.0047** | 0.938 |
| **cell-quota random** ← the A3 design | **0.3997** | 0.1591 | 10 / 13 | 0.2576 | 0.0047 | 0.938 |
| uniform draw | 1.2148 | 0.5613 | 10 / 13 | 0.4913 | 1.0587 | 0.250 |

**Cell quotas match cells and nothing else.** The quota arm nails the joint-cell distribution
(L1 0.005) and still lands 40× further from balance than the balancer, worst on `ac_rate`
(+0.400) and `tl_rate` (−0.396) — the tactical family. The hybrid keeps the cell match *and*
clears the bar on every axis, and is what the manifests ship. Selection costs **0.2 s**.

**Seed stability** (hybrid, n=600, 5 seeds): max |d| 0.0094–0.0125, pairwise clip overlap
**0.795–0.827**. The solution is mostly determined by the data, not by the draw.

## 4. Why no selection from this remainder is exchangeable with train

Inside each matched cell, train and remainder still differ — 10 cells (n≥30 both sides) × 9
continuous features, standardised mean difference:

| | median \|d\| | p90 | max |
|---|---|---|---|
| within-cell skew | **0.359** | 0.915 | 2.351 |

Per feature: `mean_v` **0.666** · `dist_m` 0.666 · `bs_rate` 0.499 · `ac_rate` 0.449 ·
`lk_rate` 0.338 · `tl_rate` 0.317 · `cum_head` 0.304 · `net_head` 0.161 · `stop_frac` 0.151 ·
`tr_rate` 0.039.

**Mechanism:** the remainder is the *residue* of a quota selector. Within any cell the selector
took the clips that filled its quota, so what is left is systematically what it declined. Matching
on cell membership cannot undo a difference that lives inside the cell — and adding cells to catch
it makes the counts infeasible (§2). This is also why the balanced arm's **max KS stays at
0.12–0.19** against a 0.057 critical value: the split is **mean-balanced, not distributionally
exchangeable**, and no draw from this remainder can be.

⇒ **This is a real limit on what option B can deliver**, and it should travel with any v2-line
number scored on the split — alongside the caveat `LOOP_STATE.md` already carries (a v2-line result
on a v2-line val is an internal baseline and cannot sit beside v1's 0.4271).

## 5. ⛔ The finding that changed the deliverable: a clean v2 val is not clean for v1

A first 600-clip draw was checked against the parity selection
(`…/physicalai/r0/phase0_selection.parquet`, 3,000 clips, split by the canonical rule
`sorted ids → torch.randperm(seed 0) → first 20 % val` → 2,400 train / 600 val):

| overlap | clips |
|---|---|
| v2-only-clean val (600) ∩ **v1 TRAIN** | ⛔ **62** |
| v2-only-clean val (600) ∩ v1 val | 24 |
| remainder (9,987) ∩ parity selection | 1,689 → **parity-free remainder 8,298** |

Scoring **v1** on that split would have scored it partly on its own training data — C64 in mirror
image, and it would have been invisible because the split's disjointness proof was written against
the *v2* corpus. `load()` now takes `exclude_paths` and the shipped manifests exclude the whole
parity selection.

**Also measured, and it answers a version of backlog item A5 without a pod:** at **clip**
granularity, **256 of v1's 600 parity-val clips (42.7 %) sit inside v2corpus's 9,000-clip training
selection** (and 1,311 of the 9,000 come from the parity 3,000). ⚠️ This is **not** the same
statistic as C64's *"21 of 40 eval episodes"* — different unit, different denominator. Compare
within a granularity, never across. It is an independent confirmation of the contamination, from
two parquet files, with no pod2 access.

## 6. What is shipped, and the size the evidence supports

Fully-clean = disjoint from v2corpus's training selection **and** from the entire v1 parity corpus
(remainder 8,298):

| n | max \|d\| | axes over bar | cell L1 | max cell census | file |
|---|---|---|---|---|---|
| 600 | — | — | — | **1.000 ⛔** (headroom 0.95, cell exhausted) | not shipped |
| ⭐ **400** | **0.0409** | 0 / 13 | 0.0084 | 0.694 | `v2_clean_val_manifest.json` |
| 300 | 0.0158 | 0 / 13 | 0.0144 | 0.528 | `v2_clean_val_manifest_n300.json` |

**Recommendation: n=400.** It is the largest split that is clean against both arms, balanced on
every axis of all four families, and still leaves ~31 % of the binding cell for a later split.
n=300 is the choice if the ≤60 % pool-depth bar is treated as hard.

⚠️ **The 600 that `V2_CLEAN_VAL_PLAN.md` proposed (to match the parity val's 600 so interval widths
are comparable) is not available.** A 400-clip val will produce wider intervals than the 600-clip
parity val; that is a consequence of the contamination in §5, not a choice.

⛔ **Not authorised, and not done here:** freezing a v2-line val is a corpus decision and belongs to
the PI. These are candidates with hashes.

## 7. What this does and does not establish

| claim | status |
|---|---|
| pool column semantics are established and machine-checked | ✅ **MEASURED** — 34/34 on 18,988 rows, corruption-tested |
| "a matched clean v2 val is feasible at 600 with 6.77× headroom" | ❌ **CORRECTED** — 1.07× on the four-family axis set; 0.77× with one more tactical axis |
| "stratify on the selector's axes" produces a train-like val | ❌ **FALSIFIED** — max \|d\| 0.3997, 10/13 axes over bar |
| a balanced 400-clip split, clean against **both** arms, exists | ✅ **MEASURED** — max \|d\| 0.0409, sha256 banked |
| the split is an exchangeable sample of the v2 training distribution | ❌ **NO** — within-cell median \|d\| 0.359; max KS 0.12–0.19 vs 0.057 critical |
| disjointness is provable at drive level | ❌ **NOT PROVABLE** — no session id, clip-local clock (§8) |
| a v2-line number on this val is comparable to v1's 0.4271 | ❌ unchanged — different corpora *and* different val surfaces |
| the balancing recipe transfers to other corpora | 🟡 **HYPOTHESIS** — it is corpus-agnostic code, untested elsewhere |

## 8. Absence, probed at four locations (per the standing rule)

**PhysicalAI-AV ships no session/drive identifier, and no absolute clock.**

| probe | result |
|---|---|
| `clip_index.parquet` | 3 columns: `clip_is_valid, chunk, split` |
| `metadata/data_collection.parquet` | 5 columns: `country, month, hour_of_day, platform_class, radar_config` |
| `metadata/feature_presence.parquet` | 36 columns — all per-feature presence flags |
| `labels/egomotion/*.parquet` `timestamp` (731 clips, 8 chunks) | starts at **−200,000 to −188,331 µs** and spans ~137 s ⇒ **clip-local microseconds**, not epoch |

⇒ neither an id join nor the **L2D-style time-overlap dedup** (which retired that trap for
`yaak-ai/L2D` on 2026-07-22 by finding 150 frames at 0.000000 m GPS disagreement) can be run here.
Combined with the standing fact that `egomotion` carries no lat/lon, PhysicalAI clips have **no
cross-clip identity linkage on any axis we hold**. **Clip granularity is the provable ceiling** and
`assert_disjoint()` says so in the artifact rather than implying more.
**Falsifier:** if a future probe finds an absolute clock or a session key in the remaining feature
families, this bound lifts and the split should be re-derived at drive level.

## 9. Resource declaration (G-I)

| | |
|---|---|
| resource | **dev-box CPU only** (RTX 4060 idle; no pod, no Colab, no network) |
| wall-clock | ~2.6 h total session; the measurement itself **6.1 s** (`make_results.py`) |
| cost | **$0** |
| why not the eval pod | the question is a property of two parquet files — no model, no GPU, no checkpoint is in the path. Spending an A40 on it would have been waste, and the eval pod's value this week is the arms it is scoring. |
| reproduce | `python make_results.py` in the package dir; `pytest tests -q` → 27 ✓ |

## 10. Escalations

1. ⭐ **PI decision (C64 option B):** the split size is **400, not 600**, and the reason is
   contamination (§5), not preference. Freeze `v2_clean_val_manifest.json`
   (sha256 `abe041db72a045b3…`) or the n=300 variant — or reject option B on the exchangeability
   limit in §4. Both outcomes are recorded in advance here.
2. **Orchestrator:** intake `pool_columns.py` into `stack/tanitad/data/` and
   `clean_val_select.py` into `stack/scripts/` (additive; 27 standalone tests). Then any consumer
   of `v2_pool_scored.parquet` — the next corpus selector included — passes `validate_pool()` first.
3. **`Project Steering/BACKLOG.md` A3** should carry the corrected headroom and the v1
   contamination; the row as written would send the next reader back to the 6.77× figure. Updated
   on this branch.
4. **A5 is partly answered** (§5, clip granularity, from parquets) — the pod2 replay is still the
   exact check, but it is no longer the only source of the number.

## 11. Fleet context this run

- **Monday's outputs:** no `tools-devenv` note exists for this week (the last is 2026-07-21); the
  fleet's weekly-agent cadence has been superseded by the autonomous main loop since ~07-24. I
  consumed the main loop's record instead — `LOOP_STATE.md`, C64/C65, and the DE-touching commits
  `fe400f0 / 2b7fe3f / 82692f2 / af78e86`.
- **`Data Engineering/Research/STATE.md` was 15 days stale** (LAST_RUN 2026-07-18) while the work
  it should describe — TanitDataSet, the L2D adapter, the lead-state gate, the v2 corpus — landed
  through the main loop. Updated this run, with the gap named rather than smoothed over.
