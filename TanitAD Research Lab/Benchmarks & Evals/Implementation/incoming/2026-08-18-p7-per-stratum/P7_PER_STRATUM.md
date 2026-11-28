# Per-stratum P7 — the T3 gate row is now COMPUTABLE, and it is a free parameter until the strata are pre-registered

**2026-08-18** · branch `agent/arch-inf-20260803` · base HEAD `8cf49ab` ·
dev-box CPU only · **Thor untouched, zero GPU** · evidence class on every claim

---

## 0. Headline

**Built, tested, measured, staged.** `taniteval/taniteval/p7_strata.py` + a CLI +
21 tests. F-9's escalation *"F-9's OWN GATE ROW IS NOT COMPUTABLE TODAY"* is
**closed** — the row can be adjudicated on the canonical val40 today, with an
`obstacle.offline`-derived stratifier that survives the goal/situation
disjointness ruling.

⛔ **AND THE FIRST REAL READ FOUND THE DEFECT THE GATE WAS WRITTEN TO CATCH.**
MEASURED on both banked fan arms: **pooled P7 PASSES while an interaction-rich
stratum FAILS.** For `refc-xl-30k` / `selector_entropy`, pooled ρ **0.4656**
[0.3663, 0.5324] — a clean pass — while the 20–40 m following band reads ρ
**0.0973** [−0.2664, 0.4400]. **62.5 % of the val40 windows (551/881) are
free-flow**, and they carry the pooled number.

⛔ **BUT THE VERDICT MOVES WITH THE BAND EDGES, AND THAT IS THE FINDING THAT
MATTERS MORE.** MEASURED, five stratifications of the *same* 881 windows:

| interaction cut | `selector_entropy` | `endpoint_dispersion_m` |
|---|---|---|
| **LEAD vs NO_LEAD** (edge-free, n=270/21 ep) | **PASS** | **PASS** |
| 3 bands at 20/40 m | **FAIL** (mid band) | **FAIL** (mid band) |
| 3 bands at 15/35 m | **FAIL** (mid band) | PASS |
| 3 bands at 14/45 m | **FAIL** (far band) | PASS |
| median split at 24.58 m | **FAIL** (far half) | PASS |

⇒ **"interaction-rich strata" is not yet a definition, and until it is
PRE-REGISTERED the T3 row can be made to pass or fail by choosing where the
bands sit.** That is the same failure class as the learning-curve exponent whose
value depended on the fit window. **A T3 arm must not be graded until the strata
are fixed in writing.** §6 proposes the pre-registration.

⭐ **A control I built was WRONG and its own replacement pins the mistake.** The
obvious permutation control — shuffle the spread once, put a cluster-bootstrap
interval around it — produced ρ **+0.1998, CI [0.0133, 0.4009]** on real windows:
"significant" from pure noise. The interval was correct; the **question** was
wrong. A null is a distribution over shuffles, not one shuffle plus an interval.
Replaced by :func:`permutation_null` and pinned by
`test_a_single_shuffle_plus_an_interval_is_not_a_null`.

---

## 1. What was missing

**MEASURED (two probes, re-verified here):** `stack/scripts/w7_roll_rerank.py`
holds P7 (`P7_GATE_RHO = 0.3`, `cluster_bootstrap_spearman`) and has **no
stratification support**; a repo-wide sweep across `stack/` and `taniteval/`
finds none. The gate row, quoted from its two independent locations by the F-9
cell:

> *"T3 | interaction curriculum … | **P7 calibration ρ ≥0.3 held on
> interaction-rich strata, not just pooled**"* —
> `…/2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:66`

**What P7 actually asks** (so the strata cannot quietly change the question): does
a proposal fan's **spread** rank the **realised error of the candidate the arm
selected**? A Spearman ρ between two per-window scalars. Stratifying means
computing it *inside* window subsets — never pooling them.

---

## 2. The strata — what they are computed from, and why that is admissible

**Source: `obstacle.offline`** — the dataset's own 3D agent cuboids — through
`taniteval.lead_source`'s causal in-corridor lead rule, banked per eval window as
`val40_lead_block.npz`.

| | |
|---|---|
| ✅ **not ego-derived** | The stratifying quantities are lead **presence** and lead **gap** — facts about *other traffic*. `stack/tanitad/data/situations.py` derives the situation labels from **ego dynamics**, so an ego-state cut (speed, accel, yaw-rate, curvature) would cut on the label's own source. Sayed 2026-08-03. |
| ✅ **not model-derived** | T3's *training* score descends from the **P8 decoder**. A stratum cut on the arm's own occupancy prediction would let the thing being graded choose the strata that grade it. `O4`'s own docstring already admits the obstacle join as *"frozen-probe/eval-strata material, never a training-time selector"* — this is exactly that sanctioned use. |
| ⚠️ **ego speed is present and deliberately unused** | The lead block carries `speeds` (ego speed at t0). It is used **only as the trivial-proxy control** (§4), never to stratify. |

This is **enforced, not asserted**: `assert_stratifier_admissible` requires a
written declaration (`name`, `kind`, `derived_from`, `why_admissible`) and
**raises** on an ego- or model-derived kind unless an explicit `override_reason`
is passed, which is then stamped into the report JSON. Pinned by three tests; the
ego refusal message names `situations.py`.

⚠️ **THREE STATES, NEVER TWO.** `NO_LABEL` is its own row and **never** enters an
interaction verdict. `obstacle.offline` spans ~20 s while `egomotion` runs
20–140 s, so most of a long clip is unlabelled; folding that into `NO_LEAD` would
manufacture free-flow and flatter every arm.

**Census — MEASURED, with the inclusion rule beside every count (C110):**

| stratum | inclusion rule | n windows | n episodes |
|---|---|---|---|
| `NO_LABEL` | no `obstacle.offline` for the clip, or t0 outside the labelled span | 60 | 5 |
| `NO_LEAD` | labels present AND no causal in-corridor vehicle ahead (≤80 m, ±2 m lat) | 551 | 33 |
| `LEAD_le20m` | `state == LEAD` AND `gap_m < 20` | 109 | 10 |
| `LEAD_20_40m` | `state == LEAD` AND `20 ≤ gap_m < 40` | 84 | 14 |
| `LEAD_ge40m` | `state == LEAD` AND `gap_m ≥ 40` | 77 | 13 |
| **total** | the canonical val40 window grid | **881** | **40** |

`gap` convention inherited unchanged from `lead_source`: `along − size_x/2`, rig
origin to the lead's **rear face**, rig frame x-forward / y-left. Of the 40 val
episodes, **21 contain at least one LEAD window**; **1 is entirely `NO_LABEL`**.

### 2.1 The join, and the guard that runs before any arithmetic

The lead block and the fan dumps are joined **positionally** (both are
`rollout.collect`'s window order). MEASURED — the alignment is exact and
independently cross-checked:

* `gt`, `cv`, `speed` are **byte-identical** across `windows_refc-base-30k.pt`,
  `fan_refc-base-30k.pt` and `fan_refc-xl-30k.pt` (`max|Δ| = 0.0`);
* the `eid` sequence is identical across all four artifacts (881/881);
* the lead block's independently-reconstructed `speeds` matches the dumps'
  `speed` to **max |Δ| = 1.81e-3 m/s** — a registration cross-check, not an
  assumption.

`assert_aligned` re-checks n and the full `eid` sequence and **exits non-zero**
before any arithmetic. A mismatched pair would otherwise produce a complete,
plausible, wrong report.

---

## 3. The measurement — MEASURED, T0

**Tier: T0.** Banked fan dumps come from `rollout.collect`, which is fed the
**expert's true future actions**. Every number below is a WM / instrument
diagnostic and **none of it is a driving claim** (EVAL_DOCTRINE).

⚠️ **These are not T3 arms.** No T3 arm exists yet. `refc-xl-30k` (256 candidates)
and `refc-base-30k` (128) are the arms whose **fan** dumps are banked with `eid`
on the canonical val40, so they are what the instrument can be demonstrated on.

**Estimator: episode-cluster bootstrap** (2000 draws, seed 0), resampled with
`taniteval.ci._draws` — the programme's single resampler. ⛔ `overlapping_holdout_se`
is not used anywhere in this module.

### `refc-xl-30k` (raw: `p7_per_stratum.json`)

| stratum | n / ep | `selector_entropy` ρ [cluster-bootstrap 95 %] | `endpoint_dispersion_m` ρ [cluster-bootstrap 95 %] |
|---|---|---|---|
| `NO_LABEL` | 60 / 5 | **REFUSED** (min-n) | **REFUSED** (min-n) |
| `NO_LEAD` (free-flow) | 551 / 33 | +0.5251 [0.4106, 0.5954] pass | +0.5854 [0.4573, 0.6519] pass |
| `LEAD_le20m` | 109 / 10 | +0.5734 [0.1473, 0.7719] **PASS** | +0.5743 [0.2074, 0.7616] **PASS** |
| `LEAD_20_40m` | 84 / 14 | +0.0973 [−0.2664, 0.4400] **fail** | +0.2103 [−0.2068, 0.5643] **fail** |
| `LEAD_ge40m` | 77 / 13 | +0.3451 [0.0161, 0.6099] **PASS** | +0.4758 [0.1139, 0.7061] **PASS** |
| *pooled — **NOT** the gate read* | 881 / 40 | *+0.4656 [0.3663, 0.5324]* | *+0.5377 [0.4285, 0.6011]* |

### `refc-base-30k`

| stratum | n / ep | `selector_entropy` ρ | `endpoint_dispersion_m` ρ |
|---|---|---|---|
| `NO_LEAD` | 551 / 33 | +0.3979 [0.2684, 0.4883] pass | +0.5391 [0.3948, 0.6154] pass |
| `LEAD_le20m` | 109 / 10 | +0.4462 [−0.0540, 0.7116] **fail** | +0.4994 [0.0260, 0.7410] **PASS** |
| `LEAD_20_40m` | 84 / 14 | +0.1734 [−0.0217, 0.4492] **fail** | +0.2971 [0.0106, 0.5888] **fail** |
| `LEAD_ge40m` | 77 / 13 | +0.2628 [−0.0714, 0.5842] **fail** | +0.3158 [−0.0201, 0.6324] **fail** |
| *pooled — **NOT** the gate read* | 881 / 40 | *+0.3621 [0.2469, 0.4460]* | *+0.5134 [0.3977, 0.5858]* |

**Every bracket above is an `episode_cluster_bootstrap_percentile_95` interval**
(`bracket_kind` is carried in the JSON on every one, C109). Gate = ρ ≥ 0.3 **and**
the interval excluding 0.

⭐ **The structural point:** the pooled column passes in all four arm × measure
combinations, and in six of the eight interaction-rich rows above the gate is not
held. **A pooled P7 on this corpus is close to a free-flow P7** — 551 of 881
windows.

`pooled` is emitted stamped `is_gate_read: false`; there is deliberately no
pooled-only mode, and the verdict is computed **only** from strata flagged
`interaction_rich: true`.

---

## 4. Controls — PER ARM, not per study (C107)

Computed for each arm on its own windows. Full numbers in the JSON;
`selector_entropy` shown.

| stratum | **positive** (err + N(0, 0.5 sd), seeded) | **trivial proxy** (ego speed at t0) | **permutation null** (500 shuffles) |
|---|---|---|---|
| `NO_LEAD` | ρ +0.766 [0.640, 0.840] **detected** | ρ −0.404 — gate NOT reached | median −0.004, 2.5–97.5 % **[−0.089, 0.071]**, p 0.000 |
| `LEAD_le20m` | ρ +0.863 [0.763, 0.913] **detected** | ρ +0.414 — CI includes 0, gate NOT reached | median −0.004, **[−0.183, 0.165]**, p 0.000 |
| `LEAD_20_40m` | ρ +0.744 [0.565, 0.850] **detected** | ρ −0.351 — gate NOT reached | median −0.004, **[−0.236, 0.218]**, p **0.376** |
| `LEAD_ge40m` | ρ +0.603 [0.447, 0.767] **detected** | ρ −0.347 — gate NOT reached | median −0.011, **[−0.234, 0.222]**, p 0.004 |

*(the permutation-null brackets are `permutation_null_dispersion_not_a_ci` — a
null's own spread, **not** a confidence interval. C109.)*

Reading them:

1. ⭐ **The positive control is detected in EVERY reported stratum, including the
   one that fails.** So the `LEAD_20_40m` failure is **not** "the instrument
   cannot see anything at n = 84 / 14 episodes". A **strong** signal is visible
   there.
2. ⚠️ **But that does not make the fail a clean fail.** The permutation null at
   `LEAD_20_40m` spans **[−0.236, 0.218]** and the observed ρ 0.0973 sits at
   **p 0.376** — indistinguishable from noise. The honest statement is: *at
   n ≈ 80 the instrument resolves a STRONG ρ but cannot separate ρ ≈ 0.3 from
   ρ < 0.3, because the interval half-width there is ≈ 0.35 against a threshold
   of 0.3.* MEASURED interval widths: **0.706 at n = 84 / 14 ep**, **0.329 at
   n = 270 / 21 ep**, **0.185 at n = 551 / 33 ep**.
3. ✅ **The trivial proxy never reaches the gate in any stratum, for either arm**
   — the fan's spread is not a repackaged ego scalar. *(Observation, not a claim:
   ego speed is **negatively** rank-correlated with the selected candidate's
   error in three of four strata.)*
4. ✅ **Constant control** returns `nan` with no gate pass in every stratum — the
   rank convention is not minting order out of memory layout.
5. ✅ **The null is centred on 0** everywhere (medians −0.011 … +0.013).

---

## 5. Band-edge sensitivity — the load-bearing caveat

Same 881 windows, same lead block, five interaction cuts (`refc-xl-30k`; the
`raw/p7_sens_*.json` + `raw/p7_lead_vs_nolead.json` artifacts):

| interaction cut | strata (n / ep) | `selector_entropy` | `endpoint_dispersion_m` |
|---|---|---|---|
| **LEAD vs NO_LEAD** (no edge choice) | 270 / 21 | ρ +0.4529 [0.2537, 0.5829] **PASS** | ρ +0.5241 [0.3317, 0.6344] **PASS** |
| 20 / 40 m | 109/10, 84/14, 77/13 | **FAIL** (mid) | **FAIL** (mid) |
| 15 / 35 m | 76/9, 108/16, 86/13 | **FAIL** (mid) | PASS |
| 14 / 45 m | 68/9, 133/17, 69/13 | **FAIL** (far) | PASS |
| median split 24.58 m | 134/14, 136/17 | **FAIL** (far half) | PASS |

Two things follow, and they are different:

* **`selector_entropy` fails the interaction gate under every 3-band or 2-band
  proximity split, and passes only when LEAD is left whole.** The *which* band
  fails moves; the fact that one does, does not. That is a genuine
  resolution-dependent finding about this arm.
* **`endpoint_dispersion_m`'s verdict is edge-sensitive** — FAIL at 20/40 only,
  PASS at three other cuts. Its 20/40 FAIL should not be quoted as a property of
  the arm.

⛔ **Therefore the gate is a free parameter until the strata are pre-registered.**
Choosing the edges after seeing the ρs is exactly the window-shopping the
programme already outlawed for learning-curve exponents.

---

## 6. What a T3 arm needs — the runbook, and the pre-registration this forces

**Pre-registration proposed (both outcomes committed in advance):**

| | |
|---|---|
| **PRIMARY interaction-rich stratum** | `LEAD` vs `NO_LEAD` — **no edge choice exists**, n = 270 / 21 ep on the canonical val40, and the measured interval half-width (≈ 0.16) is the only one on this corpus that comes near resolving a 0.3 threshold. |
| **SECONDARY, reported always, gating never** | the 3-band 20/40 m split, as a *resolution check*, with its band n / ep printed. A band-level fail is a flag for investigation, **not** a T3 kill, until §5's edge-sensitivity is resolved by more data. |
| **`NO_LABEL`** | reported, never gating. |
| **Controls** | all four, **per arm**. A stratum whose positive control is not detected is reported `power: INSUFFICIENT`, and a negative there means "not enough data", never "not calibrated". |

**To grade an actual T3 arm:**

1. Evaluate the arm on the **canonical val40** so the banked lead block applies —
   `taniteval/results/` window order, 881 windows, 40 episodes. Its fan dump must
   carry `fan`, `logits`, `sel`, `gt`, `eid`.
2. ```
   python taniteval/tools/p7_per_stratum.py \
       --fan <arm fan dump.pt> --arm <name> \
       --lead ".../2026-08-04-distance-keeping-arms/raw/val40_lead_block.npz" \
       --out p7_per_stratum_<arm>.json
   ```
3. If the arm evaluates on **different** windows, a new lead block must be built
   first (`taniteval/tools/build_lead_block.py`, or the `.npz` builder in
   `…/2026-08-04-distance-keeping-arms/code/`). `assert_aligned` will refuse the
   mismatch rather than silently mis-join it.
4. ⚠️ **W7-shaped arms do not need the CLI.** `p7_per_stratum(spread, err, eid,
   strata, stratifier=…)` takes any two per-window scalars, so a W7 arm passes
   its `cost` and realised `err` straight in. The CLI is one adapter, not the
   contract.

---

## 7. What is NOT computable, with the arithmetic

1. **`NO_LABEL` gets no ρ.** n = 60 windows over **5 episodes**, below the
   `MIN_N_EPISODES = 8` floor. Refused with its counts, not reported as a number.
   *(It is not interaction-rich either way, so nothing gates on it.)*
2. **Only the canonical val40 has a banked lead block.** MEASURED: the local
   `physicalai-val-bb543bdf7836` episode cache (**100** episodes) is a **different
   split** from the eval's val40 — a content match of the per-window ego-speed
   vectors finds **no exact match for any of the 40** (best-case residual
   **2.47 m/s**, worst **32.2 m/s**), and its `episode_id` is only the clip
   uuid's first 4 ASCII bytes, which is ambiguous for **99 of 100** episodes
   against the 4,613-clip selection union. ⇒ per-stratum P7 on any *other* window
   set needs the lead block built for that set, not a re-derivation from this
   cache.
3. **The pre-built `lead_gate_windows.parquet`** (104,994 rows,
   `n_ahead_50m` / `n_vru_near` / `ttc_s`) covers **614 clips**, of which
   **2 of the 100** local val-cache clips appear (4 across all splits). It is a
   different clip sample and **cannot** substitute for the val40 lead block.
4. **What would make the fine bands decisive.** The 3-band split's interval
   half-width is ≈ 0.35 at n ≈ 80 / 13 ep versus a 0.3 threshold. Widths scale
   roughly as 1/√n_episodes (**ESTIMATED**, from the three measured points 0.185 /
   0.329 / 0.706 at 33 / 21 / 14 episodes — *not extrapolated beyond 2×*). Halving
   the width needs ≈ 4× the episodes carrying a lead. Two routes, both real:
   **(a)** extend `obstacle.offline` coverage per episode — labels span ~20 s
   while episodes run longer, so 60 windows are `NO_LABEL` and much of each clip
   is unusable; **(b)** a larger eval split. Neither is an instrument change.

⚠️ **A richer stratifier is available and NOT used here.** `n_ahead_50m`
(agent count ahead) and `n_vru_near` exist in the lead-gate schema and would give
a density stratifier rather than a lead-proximity one — closer to T3's
"multi-agent kinematic entropy". They are **not** in the banked val40 block, which
carries only the single causal lead. Adding them is a lead-block rebuild, not an
instrument change, and is the single highest-value follow-up (§9).

---

## 8. Retraction-class note — the control I killed

**Class: an interval answering a different question than the one asked** (same
family as `overlapping_holdout_se`, `df` on a pod, cgroup `usage_in_bytes`).

I first implemented the permutation control as *"shuffle the spread once within
each stratum, then run the same cluster bootstrap"*. On synthetic data it passed.
On real windows it reported ρ **+0.1998** with interval **[0.0133, 0.4009]** for
`refc-xl-30k` / `LEAD_20_40m` — an interval **excluding 0 for pure noise**. The
bootstrap was not broken: it correctly bracketed *that one shuffle's* ρ, and one
shuffle at n = 84 lands anywhere in **[−0.236, 0.218]**. A null is a distribution
over shuffles. **Caught by running the instrument on real data before publishing
it**, which is why the demonstration run is part of the deliverable and not an
afterthought.

---

## 9. Escalations — routed, not left in a README

1. ⛔ **PI / gate owner: the T3 strata must be PRE-REGISTERED before any T3 arm is
   graded** (§5, §6). Until then the row can be made to pass or fail by choosing
   band edges. Proposal in §6; it needs a decision, not an implementation.
2. ⛔ **`MODEL_REGISTRY.md` / gate emitters owner:** pooled P7 is quotable today as
   though it were the calibration of the arm. On this corpus it is **62.5 %
   free-flow**. Any registry row quoting a pooled P7 ρ should carry that, or carry
   the per-stratum read.
3. ⚠️ **Lead-block owner (Architecture & Inference):** rebuild `val40_lead_block`
   with `n_ahead_50m` / `n_vru_near` so the stratifier can be **agent density**,
   not just the single causal lead. That is the cut T3's "multi-agent kinematic
   entropy" actually names, and it is one builder change (§7.4).

---

## 10. Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `p7_strata.py` — the instrument (strata, estimator, admissibility guard, controls) | `repo:taniteval/taniteval/p7_strata.py` | **NEW, staged** |
| `p7_per_stratum.py` — CLI + alignment guard + `.npz`/`.pt` lead loader | `repo:taniteval/tools/p7_per_stratum.py` | **NEW, staged** |
| `test_p7_strata.py` — 21 tests | `repo:taniteval/tests/test_p7_strata.py` | **NEW, staged** |
| `p7_per_stratum.json` — the 20/40 m read, 2 arms × 2 measures + all controls | `repo:…/incoming/2026-08-18-p7-per-stratum/raw/` | **NEW, staged** |
| `p7_lead_vs_nolead.json` — the edge-free primary read | same | **NEW, staged** |
| `p7_sens_15_35.json`, `p7_sens_14_45.json`, `p7_sens_24.58_1000.json` | same | **NEW, staged** |
| this report | `repo:…/incoming/2026-08-18-p7-per-stratum/P7_PER_STRATUM.md` | **NEW, staged** |

**Nothing is stranded** — no pod, no worktree was used; the run is dev-box CPU
only. **Suites, run separately, `PYTHONUTF8=1`, no CPU contention:**
`taniteval/tests` **1157 passed** (132.9 s, 1 pre-existing warning);
`taniteval/tests/test_p7_strata.py` **21 passed**.

⚠️ **This deliverable modified NO existing file — all seven repo artifacts are
NEW.** `stack/scripts/train_v6_staged.py`, `stack/tanitad/models/v6.py` and
`stack/tests/test_v6_domain_mix.py` show as changed in the working tree; those are
a **sibling agent's** concurrent work, are not staged by me, and are not touched
here. My staged set is exactly the seven rows above.
