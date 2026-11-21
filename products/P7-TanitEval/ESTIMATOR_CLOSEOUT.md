# ESTIMATOR CLOSEOUT — `overlapping_holdout_se` removed from every live emission path

**Product** P7-TanitEval · **Date** 2026-08-28/29 · **Branch** `claude/zen-bose-eb35e9`
**Authority** PI ruling 2026-08-28 ("fix"): `overlapping_holdout_se` is FORBIDDEN as a
decision-grade estimator.
**Evidence class** MEASURED (this clone, dev box, CPU only, no GPU). Every number below is
recomputed from the 27 banked `taniteval/results/windows_*.pt` per-window artifacts through the
same `bench.run` / `taniteval.ci` code path the harness uses (B = 2000, seed 0).

---

## 0. TL;DR

Three live emission paths were named in the brief. **All three are closed.** Closing them
surfaced **four more** that were not: three are closed here, one is escalated because fixing it
moves every D1 verdict in the programme.

MEASURED over the 25 distinct arms (27 dumps minus the two C126 duplicates):

| quantity | measured |
|---|---|
| `ade_0_2s` point-estimate shift, `(primary − legacy)/legacy` | **−10.46 % to +7.14 %**, bidirectional |
| the same shift as `(legacy − primary)/primary` (the `CLAUDE.md` framing) | **−6.66 % to +11.68 %** |
| arms where the legacy value was INFLATED / DEFLATED / flat | **11 / 14 / 0** |
| CI narrowing factor (primary ci95 ÷ legacy ci95) | **1.127× to 3.100×, median 1.499×** |
| leaderboard positions that MOVE when ordered on the primary | **9 of 25** |
| paired head-to-head claims that change SIGN | **2 of 7** |
| paired head-to-head claims that LOSE their separation | **1 of 7** |
| `miss_rate@2m` shift on the two flagship arms | **−24.6 % and −25.8 %** |

> ⚠️ **Two published ranges that look contradictory are the SAME measurement.**
> `CLAUDE.md` says *"−6.67 % to +11.69 %"*; `test_runner_gate_print.py` says *"−10.5 % … +7.1 %"*.
> They differ only in the denominator — `(legacy−primary)/primary` versus `(primary−legacy)/legacy`.
> Both are reproduced above to two decimals so nobody retracts one against the other.
> **State the denominator whenever this range is quoted.**

---

## 1. WHAT WAS LIVE, AND WHAT IS NOW CLOSED

### SITE 1 — `bench.run` emitted the legacy block TWICE, the second time under a bare key

`taniteval/taniteval/bench.py:303` wrote `"heldout": legacy_block` — **the same dict object** as
the correctly quarantined `LEGACY_BLOCK` at `:302`. The bare key carries no verdict word, so
`gate_guard`'s deciding-name rule cannot see it; and `bench.run` was the **only** block emitter in
the harness that never called `assert_no_deprecated_estimator` (`closedloop`, `hierarchy`,
`planner_p2`, `driving`, `corridor`, `lateral` and `strategic_probes` all did).

Five consumers read it as the arm's headline number:

| consumer | what it did with a mean-of-split-means |
|---|---|
| `report.py:72,74` | **ORDERED the dashboard leaderboard on it**, and printed it as `ADE ± ci95` |
| `runner.py:409-410` | fell back to it, labelled `heldout(DEPRECATED)`, and wrote it into `golden.json` |
| `refc_rerank.py:384,385,529` | published it as `ade2s_heldout_baseline` / `_oracle` |
| `efficiency.py:1712` | paired it with latency as `cost_per_accuracy.ade_0_2s_heldout` |
| `generalization.py:921` | printed it as the OOD generalisation read |

**Closed by a TOMBSTONE, not a deletion.** The `heldout` key survives with the same metric keys
and its `estimator` / `deprecated` labels, and **no `mean`, `ci95` or `std`**. Both halves are
load-bearing and pull opposite ways:

* `run_gate._deprecated_present` searches the literal `("heldout", "model")` path and accepts a
  node on `"mean" in n` **OR** `n["estimator"] == DEPRECATED_ESTIMATOR`. Dropping the numbers
  leaves the tripwire firing through the second branch — the branch written for exactly this
  shape. **Deleting the key would have silently disarmed the gate's own refusal**, trading one
  silent failure for a worse one. Verified live and pinned.
* Nothing is quotable out of it. The values remain **bit-identical** one key over, under
  `legacy_overlapping_holdout_se`, so every published figure stays traceable.

`bench.run` now calls `bench.assert_no_deprecated_estimator(out)` **before** attaching either
quarantine key — the same shape as its six sibling modules.

### SITE 2 — `recompute_ci.py`, an unguarded standalone caller

`taniteval/recompute_ci.py:91` called `ci.overlapping_holdout_se` from **outside**
`ENFORCED_ROOTS`, which covered `taniteval/taniteval` and `taniteval/tools` but **not the
`taniteval/` top level**, where ~35 standalone drivers live. It was unguarded by accident, not by
decision.

The call itself is legitimate: this file is the instrument that MEASURES the defect, and refusing
to compute the BEFORE column would destroy the evidence that the AFTER column is needed. What it
lacked was a name that said so, and a guard that had ever looked at it.

* `ENFORCED_ROOTS` now includes `(REPO/"taniteval", "*.py")`. **MEASURED before adding: zero new
  violations, name or shape** — so the extension costs nothing and closes the gap. Non-recursive
  on purpose: `taniteval/tests` contains deliberate reproductions of the arithmetic that exist to
  PROVE the guard fires, and policing the guard's own negative controls is the self-match trap
  again.
* `naive_published` → **`reproduce_overlapping_holdout_published`**. The name IS the exemption:
  `gate_guard.is_declared_estimator_name` matches `overlapping_holdout`, which is the documented
  mechanism that makes a quarantined reproduction legal and a silent one not. `naive_published`
  announced nothing — the same defect as site 3, in a different file.
* It now reads the BEFORE value from **either artifact vintage** (`heldout` on pre-closeout
  files, `legacy_overlapping_holdout_se` on post-closeout ones), so the ledger tool keeps
  reproducing history across the change instead of reporting "no published value" for a file that
  plainly has one.

### SITE 3 — `driving_diagnostic.mean_ci`, the UNNAMED CLONE

`stack/scripts/driving_diagnostic.py:167-177` re-typed `1.96 * std / sqrt(n)` over per-split
means, under a name no grep for `jack` or `overlapping_holdout` would ever return, emitting **no
`estimator` field at all**. Two independent audits missed it because both searched for names.

> ⚠️ **The brief named two callers; there are six.**
> `stack/experiments/reset-speed4b/eval_refa4b_grounded.py:199-200` and
> `.../eval_grounded_rollout_4b_speed.py:184-185` (both named), plus
> `stack/scripts/compare_arms.py:375-376`, `stack/scripts/eval_grounded_rollout_4b.py:191-192`,
> `stack/scripts/eval_metric_rollout.py:173-174`, and `driving_diagnostic.py`'s own
> `probe_ceilings` at `:265,:287-290`. Absence found at one location is not absence — and neither
> is a call-site census.

**Closed by declaration + self-labelling, with the arithmetic untouched:**

1. `mean_ci` → **`overlapping_holdout_mean_ci`**, with `mean_ci` kept as a back-compat alias so
   all six callers keep working unchanged. **Both spellings were already in
   `gate_guard.BANNED_CALLS`** — the guard authors wrote this exact rename down as the intended
   end state in a comment at `gate_guard.py:56-58`, and it had never been done.
2. The output now carries `estimator: "overlapping_holdout_se"` and `deprecated: True`, so
   `driving.assert_no_deprecated_estimator` **REFUSES** any block containing it. Before, the
   guard walked straight past — an unlabelled interval is invisible to a guard that keys on
   labels, which is how it survived at all.

> ⚠️ **The arithmetic is preserved VERBATIM, ddof=1 included.** It is deliberately NOT routed
> through `ci.overlapping_holdout_se`, which uses `np.nanstd` (**ddof=0**). Delegating would have
> rescaled every published D-number by `sqrt((n−1)/n)` = **0.935 at n=8**. A reproduction that
> changes the number is not a reproduction. Pinned by a bit-identity test against the removed body.

---

## 2. FOUR MORE SITES THE BRIEF DID NOT NAME

### CLOSED-4 — the arithmetic-shape detector had NO CALLER

`gate_guard.scan_source_shapes` / `scan_file_shapes` / `scan_paths_shapes` — 120 lines whose own
module docstring says admissibility is settled by
`tests/test_no_jack_in_gates.py::SHAPE_ALLOWLIST`. **That allowlist did not exist, and neither
did any caller**: a repo-wide search for `scan_paths_shapes` returned only the definitions in
`gate_guard.py` itself.

The detector was correct and tested by nobody — an instrument structurally unable to report the
answer it is cited for, the same class as `df` hiding the per-pod quota and as the `_files_under`
skip bug that scanned 373 of 373 files as zero.

**Now wired**, with an EXHAUSTIVE and EXACT census: every shape in enforced scope is either
allowlisted with a written reason or listed as a known-open defect, and the test fails if the
detected set differs from their union **in either direction**. A new site fails on the day it is
written; a known-open site that is "fixed" by renaming it into self-exemption **also** fails,
because it vanishes from a census that requires it to be there.

### CLOSED-5 — the shape detector was still NAME-KEYED in its NUMERATOR

Found by writing the deliberate-regression arm the brief demanded. `spread = np.std(vals)`
followed by `1.96 * spread / sqrt(n)` **escaped the detector**, because `spread` is not in
`_DISPERSION_NAME_RE` and `_se_names` only tracked names bound to a *whole* `dispersion/sqrt(n)`
expression, never to a bare dispersion. That is the "one rename behind" failure the shape detector
exists to end, one level down.

`gate_guard._dispersion_names` now follows the data in the numerator too (fixpoint, so
`a = np.std(v); b = a; 1.96*b/sqrt(n)` is caught). **MEASURED: zero new census entries** — it
closes a hole without moving the count. A false-positive control (`1.96 * mean / sqrt(n)`) is
pinned so the widening cannot creep.

### CLOSED-6 — the DECLARED-name exemption had a hole exactly where it was widest

Also found by a deliberate-regression arm. Renaming `naive_published` to
`reproduce_overlapping_holdout_published` correctly exempted it from the SHAPE detector — and
**simultaneously made a verdict computed from its output invisible to the NAME guard**, because
the new name was in neither `BANNED_CALLS` nor `BANNED_CALL_RE`.

`is_banned_callable` and `banned_import_aliases` now treat a declared name (`_DECLARED_RE`:
anything containing `overlapping_holdout` or `jackknife`) as banned. The asymmetry with the shape
detector is the point: **declaring the estimator buys a function the right to CONTAIN the
arithmetic, never the right to DECIDE with it.** MEASURED: zero new name violations in scope.

### CLOSED-7 — two live GATE scripts adjudicated G1/G2/G3 on the deprecated block

`stack/scripts/eval_flagship_v15.py:238-250` and `eval_flagship_v16.py:256-265` read
`res["heldout"]["model"]` and decided:

```
"G1_beat_refc_xl_final_0.458": bool(m["ade@2s"]["mean"] < 0.458),
"G2_beat_v1_0.4522":           bool(m["ade@2s"]["mean"] < 0.4522),
"G3_miss_le_0.10":             bool(m["miss_rate@2m"]["mean"] <= 0.10),
```

That is **a gate decided by the banned estimator**, and `gate_guard` missed it twice over:
(a) the taint enters through a **dict key**, not a banned CALL, so taint propagation has no
source to start from; and (b) the verdict keys are spelled `G1_beat_…`, which `_DECIDING_RE`
(`(^|_)(pass|passed|verdict|gated?_ok|admissible)($|_)|^G\d+_pass`) does not match.

Both scripts now read `cluster_bootstrap`. `stack/scripts/refc_v12_eval.py:246` (a print of the
same block) is migrated with them.

> ⚠️ **I did NOT broaden `_DECIDING_RE` to `^G\d+_`.** MEASURED: it would flag three
> reporting-only keys in `planner_p2.py`'s legacy block (`G1_head_minus_planner_ade2s`,
> `G1_delta` ×2) **and still not catch these two scripts**, because the missing piece is a taint
> source, not a verdict name. Recorded as OPEN-2 rather than papered over with a change that
> costs false positives and buys nothing.

### OPEN-1 ⛔ — `tanitad/eval/gates.run_d1` is a FOURTH unnamed clone, and it decides the D1 gate

`stack/tanitad/eval/gates.py:249-251`:

```python
ade      = sum(ades) / n                                        # mean of split means
ade_std  = (sum((a - ade) ** 2 for a in ades) / max(1, n - 1)) ** 0.5
ade_ci95 = 1.96 * ade_std / n ** 0.5                            # overlapping_holdout_se
...
passed   = admissible and ade < thr                             # <- DECIDED ON IT
```

`ades` are the per-split means of `split_by_episode(episode_ids, val_frac, s)` for
`s in seed..seed+n_splits-1` — **8 OVERLAPPING random 20 % holdouts of one episode pool**, the
exact protocol the PI ruled out. The verdict string publishes `ADE@1s=<ade>±<ade_ci95>` with no
estimator name. `driving_diagnostic.mean_ci`'s own docstring said *"route-resampled protocol,
matching gates.run_d1"* — the two are siblings, and only one of them was in the brief.

**Not closed here, deliberately.** `run_d1` is the D1 gate for the whole programme; changing its
point estimate moves every D1 verdict ever published, and that is a PI decision, not an agent's.
It is enumerated in `SHAPE_KNOWN_OPEN` with this reasoning checked in beside the code, so it
cannot silently disappear.

**⇒ ESCALATION 1 (PI): authorise a D1 re-adjudication.** The work is a day of CPU: replace the
mean-of-split-means with the full-set point estimate + `ci.episode_cluster_bootstrap` over the
episode ids `run_d1` already receives, keep the legacy value under a `_LEGACY` key, and re-issue
every banked `GateReport`.

### OPEN-2 — the name guard cannot see taint that enters through a dict key

`gate_guard.scan_source` starts taint at a **banned CALL**. A gate that does
`m = res["heldout"]["model"]` and then `bool(m["ade@2s"]["mean"] < thr)` has no banned call
anywhere, so the guard is silent. CLOSED-7 was found by reading, not by the guard.

**⇒ ESCALATION 2 (P7 backlog):** add a *taint-by-key* source — any subscript of a literal key in
`{"heldout", LEGACY_BLOCK, "legacy_overlapping_holdout_se"}` taints the binding. This needs its
own false-positive census (`recompute_ci.py` and `test_runner_gate_print.py` legitimately read
those keys) and is a separate work package.

---

## 3. THE NUMBER-MOVEMENT LEDGER

**Metric:** `ade_0_2s` (= `ade@2s`, the headline). **OLD** = what a consumer read out of the bare
`heldout` key (`overlapping_holdout_se`: the mean over 8 overlapping random 20 % holdouts, ± its
`1.96·std/√8`). **NEW** = what it reads now (`cluster_bootstrap`: the **full-set** point estimate
with a percentile episode-cluster interval). Both columns recomputed from the same
`windows_<key>.pt`. Rows ordered by the NEW value — this **is** the corrected leaderboard.

| # | arm (`windows_<key>.pt`) | n win | OLD `heldout` mean ± ci95 | NEW `cluster_bootstrap` mean [lo, hi] | delta | delta % | CI widen |
|---|---|---|---|---|---|---|---|
| 1 | `flagship-30k` | 881 | 0.4522 ± 0.0312 | **0.4271** [0.3675, 0.4871] | -0.0251 | -5.55 % ↓ | ×1.92 |
| 2 | `flagship-v16-ab-ft` | 881 | 0.4886 ± 0.0800 | **0.4375** [0.3384, 0.5530] | -0.0511 | -10.46 % ↓ | ×1.34 |
| 3 | `refc-v12-k16reg` | 881 | 0.4546 ± 0.0563 | **0.4576** [0.3742, 0.5438] | +0.0030 | +0.66 % ↑ | ×1.51 |
| 4 | `refc-v12` | 881 | 0.4671 ± 0.0613 | **0.4625** [0.3781, 0.5486] | -0.0046 | -0.98 % ↓ | ×1.39 |
| 5 | `refc-xl-30k` | 881 | 0.4577 ± 0.0572 | **0.4714** [0.3896, 0.5556] | +0.0137 | +2.99 % ↑ | ×1.45 |
| 6 | `refc-base-30k` | 881 | 0.4523 ± 0.0497 | **0.4728** [0.3835, 0.5699] | +0.0205 | +4.53 % ↑ | ×1.88 |
| 7 | `refc-xl-live` | 881 | 0.4703 ± 0.0574 | **0.4788** [0.3977, 0.5638] | +0.0085 | +1.81 % ↑ | ×1.45 |
| 8 | `refc-v12-smoke-t0` | 88 | 0.5521 ± 0.1244 | **0.5573** [0.3966, 0.7180] | +0.0052 | +0.94 % ↑ | ×1.29 |
| 9 | `refb-v2-30k` | 881 | 0.5921 ± 0.0685 | **0.5913** [0.4766, 0.7131] | -0.0008 | -0.14 % ↓ | ×1.73 |
| 10 | `refc-xl` | 881 | 0.5645 ± 0.0447 | **0.6048** [0.5170, 0.7009] | +0.0403 | +7.14 % ↑ | ×2.06 |
| 11 | `flagship-speed` | 881 | 0.6277 ± 0.0551 | **0.6152** [0.5422, 0.6951] | -0.0125 | -1.99 % ↓ | ×1.39 |
| 12 | `refb-v2-20k` | 881 | 0.6462 ± 0.0548 | **0.6435** [0.5410, 0.7516] | -0.0027 | -0.42 % ↓ | ×1.92 |
| 13 | `refb-10k` | 881 | 0.8255 ± 0.0992 | **0.8372** [0.6753, 1.0218] | +0.0117 | +1.42 % ↑ | ×1.75 |
| 14 | `flagship-v4.1-10k` | 881 | 0.8707 ± 0.0821 | **0.8522** [0.7423, 0.9785] | -0.0185 | -2.12 % ↓ | ×1.44 |
| 15 | `refb` | 881 | 0.8682 ± 0.0817 | **0.8629** [0.6928, 1.0385] | -0.0053 | -0.61 % ↓ | ×2.12 |
| 16 | `flagship-v4.2-step4000` | 881 | 1.0490 ± 0.1035 | **0.9869** [0.8787, 1.1119] | -0.0621 | -5.92 % ↓ | ×1.13 |
| 17 | `refc-v12-smoke-reg` | 88 | 1.4860 ± 0.1289 | **1.5466** [1.3535, 1.7398] | +0.0606 | +4.08 % ↑ | ×1.50 |
| 18 | `flagship-v3enc-10k` | 881 | 2.1072 ± 0.2020 | **1.9654** [1.6556, 2.2859] | -0.1418 | -6.73 % ↓ | ×1.56 |
| 19 | `refa-dinov2` | 881 | 2.1322 ± 0.1821 | **2.1675** [1.9081, 2.4212] | +0.0353 | +1.66 % ↑ | ×1.41 |
| 20 | `flagship-nospeed` | 881 | 2.9176 ± 0.3558 | **3.0175** [2.5450, 3.5444] | +0.0999 | +3.42 % ↑ | ×1.40 |
| 21 | `refa-dynin-30k` | 881 | 2.9196 ± 0.3937 | **3.0471** [2.4984, 3.6878] | +0.1275 | +4.37 % ↑ | ×1.51 |
| 22 | `overfit_refa-dynin-20k` | 881 | 3.0159 ± 0.2913 | **3.1138** [2.6402, 3.6705] | +0.0979 | +3.25 % ↑ | ×1.77 |
| 23 | `overfit_refa-dynin-15k` | 881 | 3.6937 ± 0.1887 | **3.7818** [3.2177, 4.3875] | +0.0881 | +2.39 % ↑ | ×3.10 |
| 24 | `overfit_refa-dynin-5k` | 881 | 3.7550 ± 0.4630 | **3.8307** [3.2216, 4.4916] | +0.0757 | +2.02 % ↑ | ×1.37 |
| 25 | `flagship-v2-6k` | 881 | 6.1790 ± 1.2845 | **5.9396** [4.3273, 7.6249] | -0.2394 | -3.87 % ↓ | ×1.28 |

**Excluded duplicate dumps (C126 — not counted in any census above):**

| excluded duplicate dump (C126) | n win | OLD | NEW | delta % |
|---|---|---|---|---|
| `overfit_refa-dynin-30k` | 881 | 2.9196 ± 0.3937 | 3.0471 [2.4984, 3.6878] | +4.37 % |
| `refc-v12-identity` | 881 | 0.4577 ± 0.0572 | 0.4714 [0.3896, 0.5556] | +2.99 % |

### 3.1 Leaderboard positions that MOVE — 9 of 25

| arm | old rank (legacy) | new rank (primary) |
|---|---|---|
| `flagship-v16-ab-ft` | 7 | **2** |
| `refc-base-30k` | 2 | **6** |
| `refc-v12` | 5 | **4** |
| `refc-xl-30k` | 4 | **5** |
| `refc-xl-live` | 6 | **7** |
| `refb-v2-30k` | 10 | **9** |
| `refc-xl` | 9 | **10** |
| `flagship-v4.1-10k` | 15 | **14** |
| `refb` | 14 | **15** |

`flagship-v16-ab-ft` moves **five places**, from 7th to 2nd, on the largest single-arm bias in
the corpus (−10.46 %). Nothing about the model changed.

### 3.2 The other three metric families (the ADE-only table is one row of four)

Per the binding four-families rule, the movement is NOT confined to ADE — and on `miss_rate@2m`
it is **four to five times larger**:

| arm | metric | OLD mean ± ci95 | NEW mean [lo, hi] | delta % | CI widen |
|---|---|---|---|---|---|
| `flagship-30k` | `fde@2s` | 0.9437 ± 0.0630 | 0.9075 [0.7851, 1.0306] | -3.84 % | ×1.95 |
| `flagship-30k` | `miss_rate@2m` | 0.0602 ± 0.0121 | 0.0454 [0.0239, 0.0681] | -24.58 % | ×1.83 |
| `flagship-30k` | `tms_openloop` | 0.1070 ± 0.0229 | 0.0978 [0.0701, 0.1304] | -8.60 % | ×1.31 |
| `refc-xl-30k` | `fde@2s` | 0.9724 ± 0.1243 | 1.0061 [0.8301, 1.1875] | +3.47 % | ×1.44 |
| `refc-xl-30k` | `miss_rate@2m` | 0.1459 ± 0.0390 | 0.1419 [0.0943, 0.1918] | -2.74 % | ×1.25 |
| `refc-xl-30k` | `tms_openloop` | 0.2032 ± 0.0103 | 0.2135 [0.1928, 0.2349] | +5.07 % | ×2.05 |
| `refc-base-30k` | `fde@2s` | 0.9543 ± 0.1066 | 1.0031 [0.8148, 1.2087] | +5.11 % | ×1.85 |
| `refc-base-30k` | `miss_rate@2m` | 0.1353 ± 0.0336 | 0.1419 [0.0874, 0.2000] | +4.88 % | ×1.68 |
| `refc-base-30k` | `tms_openloop` | 0.1894 ± 0.0160 | 0.1957 [0.1755, 0.2160] | +3.33 % | ×1.26 |
| `flagship-v16-ab-ft` | `fde@2s` | 1.0352 ± 0.1731 | 0.9297 [0.7113, 1.1856] | -10.19 % | ×1.37 |
| `flagship-v16-ab-ft` | `miss_rate@2m` | 0.1424 ± 0.0453 | 0.1056 [0.0488, 0.1714] | -25.84 % | ×1.35 |
| `flagship-v16-ab-ft` | `tms_openloop` | 0.1038 ± 0.0089 | 0.0947 [0.0834, 0.1057] | -8.77 % | ×1.26 |

`miss_rate@2m` is the metric `G3_miss_le_0.10` is decided on. On `flagship-v16-ab-ft` the legacy
value 0.1424 and the primary 0.1056 are both above 0.10, so the ❌ verdict holds — but the
primary interval [0.0488, 0.1714] **straddles the threshold**, so the honest reading is
"undecided", not "fails".

### 3.3 Head-to-head claims: 2 SIGN FLIPS and 1 LOST SEPARATION

OLD = difference of per-split means ± `overlapping_holdout_se` of the per-split differences.
NEW = `ci.paired_episode_cluster_bootstrap` on the same windows (always paired — the arms are not
independent, so a quadrature combination of two single-arm intervals is invalid).

| claim | OLD paired delta (split-means ± ohse) | NEW paired episode-cluster bootstrap | change |
|---|---|---|---|
| registry leaderboard rows 1 vs 2<br>`refc-xl-30k` - `flagship-30k` | +0.0055 ± 0.0652 (tie) | **+0.0443** [-0.0544, +0.1465] (tie) | direction + verdict hold |
| REF-C scale A/B (base vs XL)<br>`refc-xl-30k` - `refc-base-30k` | +0.0054 ± 0.0097 (tie) | **-0.0013** [-0.0316, +0.0281] (tie) | **SIGN FLIP** |
| H1: flagship v1 vs REF-B v2<br>`refb-v2-30k` - `flagship-30k` | +0.1399 ± 0.0768 (SEPARATED) | **+0.1642** [+0.0430, +0.2851] (SEPARATED) | direction + verdict hold |
| D-A5 / H4: flagship vs REF-A frozen<br>`refa-dynin-30k` - `flagship-30k` | +2.4674 ± 0.3894 (SEPARATED) | **+2.6200** [+2.0945, +3.2570] (SEPARATED) | direction + verdict hold |
| v1 vs v16 ab-ft<br>`flagship-v16-ab-ft` - `flagship-30k` | +0.0364 ± 0.0695 (tie) | **+0.0104** [-0.0888, +0.1147] (tie) | direction + verdict hold |
| v1.2 rescorer vs k16reg<br>`refc-v12-k16reg` - `refc-v12` | -0.0125 ± 0.0083 (SEPARATED) | **-0.0049** [-0.0199, +0.0098] (tie) | **SEPARATION LOST -> tie** |
| frozen REF-C selection vs v1.2 rescorer<br>`refc-v12` - `refc-xl-30k` | +0.0094 ± 0.0097 (tie) | **-0.0089** [-0.0245, +0.0059] (tie) | **SIGN FLIP** |

Reading the three that change:

1. **REF-C scale A/B — SIGN FLIP.** Legacy: base (104.2 M) beats XL (251.9 M) by 0.0054 m.
   Primary: XL beats base by 0.0013 m. Neither is separated, so the honest verdict under either
   estimator is **tie** — but the *direction* reverses, and a scale study whose headline is
   "smaller is as good as bigger" must not be quoted from the legacy direction.
   `MODEL_REGISTRY.md:2773` already records this swap; this is its paired confirmation.
2. **Frozen REF-C selection vs the v1.2 learned rescorer — SIGN FLIP.** Legacy: the rescorer is
   0.0094 m WORSE. Primary: it is 0.0089 m BETTER. Still a tie either way.
3. **`refc-v12` vs `refc-v12-k16reg` — SEPARATION LOST.** Legacy: −0.0125 ± 0.0083,
   **SEPARATED** — a published win. Primary: −0.0049 [−0.0199, +0.0098], **not separated**.
   ⇒ **A claimed win becomes a tie.** This is the single most consequential row in the ledger:
   the other two flips were ties in both framings, this one was a positive result that does not
   survive the correct estimator.

### 3.4 Banked artifacts that carry a number produced by a closed path

**`taniteval/results/eff_*.json` × 21** — `cost_per_accuracy.ade_0_2s_heldout` + `ade_ci95`, the
values `efficiency.panel_rows` renders in the dashboard's efficiency table. Every one is
recomputable and recomputed here:

| eff_*.json | key | OLD ade_0_2s_heldout ± ade_ci95 | NEW cluster_bootstrap [lo, hi] | delta % |
|---|---|---|---|---|
| `eff_flagship-30k.json` | `flagship-30k` | 0.4522 ± 0.0312 | **0.4271** [0.3675, 0.4871] | -5.55 % |
| `eff_flagship-30k.replicate-fp32-2153.json` | `flagship-30k` | 0.4522 ± 0.0312 | **0.4271** [0.3675, 0.4871] | -5.55 % |
| `eff_flagship-nospeed.CONTAMINATED-20260720-215601.json` | `flagship-nospeed` | 2.9176 ± 0.3558 | **3.0175** [2.5450, 3.5444] | +3.42 % |
| `eff_flagship-nospeed.json` | `flagship-nospeed` | 2.9176 ± 0.3558 | **3.0175** [2.5450, 3.5444] | +3.42 % |
| `eff_flagship-speed.json` | `flagship-speed` | 0.6277 ± 0.0551 | **0.6152** [0.5422, 0.6951] | -1.99 % |
| `eff_refa-dinov2.CONTAMINATED-20260720-215641.json` | `refa-dinov2` | 2.1322 ± 0.1821 | **2.1675** [1.9081, 2.4212] | +1.66 % |
| `eff_refa-dinov2.json` | `refa-dinov2` | 2.1322 ± 0.1821 | **2.1675** [1.9081, 2.4212] | +1.66 % |
| `eff_refa-dynin-30k.CONTAMINATED-20260720-215903.json` | `refa-dynin-30k` | 2.9196 ± 0.3937 | **3.0471** [2.4984, 3.6878] | +4.37 % |
| `eff_refa-dynin-30k.json` | `refa-dynin-30k` | 2.9196 ± 0.3937 | **3.0471** [2.4984, 3.6878] | +4.37 % |
| `eff_refb-10k.CONTAMINATED-20260720-220049.json` | `refb-10k` | 0.8255 ± 0.0992 | **0.8372** [0.6753, 1.0218] | +1.42 % |
| `eff_refb-10k.json` | `refb-10k` | 0.8255 ± 0.0992 | **0.8372** [0.6753, 1.0218] | +1.42 % |
| `eff_refb.CONTAMINATED-20260720-215957.json` | `refb` | 0.8682 ± 0.0817 | **0.8629** [0.6928, 1.0385] | -0.61 % |
| `eff_refb.json` | `refb` | 0.8682 ± 0.0817 | **0.8629** [0.6928, 1.0385] | -0.61 % |
| `eff_refc-base-30k.json` | `refc-base-30k` | 0.4523 ± 0.0497 | **0.4728** [0.3835, 0.5699] | +4.53 % |
| `eff_refc-xl-30k.CONTAMINATED-20260720-220309.json` | `refc-xl-30k` | 0.4577 ± 0.0572 | **0.4714** [0.3896, 0.5556] | +2.99 % |
| `eff_refc-xl-30k.CONTAMINATED-smoke-2136.json` | `refc-xl-30k` | 0.4577 ± 0.0572 | **0.4714** [0.3896, 0.5556] | +2.99 % |
| `eff_refc-xl-30k.json` | `refc-xl-30k` | 0.4577 ± 0.0572 | **0.4714** [0.3896, 0.5556] | +2.99 % |
| `eff_refc-xl-live.CONTAMINATED-20260720-220226.json` | `refc-xl-live` | 0.4703 ± 0.0574 | **0.4788** [0.3977, 0.5638] | +1.81 % |
| `eff_refc-xl-live.json` | `refc-xl-live` | 0.4703 ± 0.0574 | **0.4788** [0.3977, 0.5638] | +1.81 % |
| `eff_refc-xl.CONTAMINATED-20260720-220144.json` | `refc-xl` | 0.5645 ± 0.0447 | **0.6048** [0.5170, 0.7009] | +7.14 % |
| `eff_refc-xl.json` | `refc-xl` | 0.5645 ± 0.0447 | **0.6048** [0.5170, 0.7009] | +7.14 % |

*(The `.CONTAMINATED-*` files are quarantined latency runs and are listed for completeness; their
ADE column moves identically because it comes from the same banked dump.)*

**`taniteval/results/eval_v16_flagship-v16-ab-ft.json`** — `heldout.ade_0_2s = 0.4886`,
**no `cluster_bootstrap` block at all**. This is the artifact behind the v1.6 G1/G2/G3 verdicts
(CLOSED-7). Recomputed: **0.4375 [0.3384, 0.5530]**, −10.46 %.

| gate | threshold | legacy read | corrected read | verdict change |
|---|---|---|---|---|
| `G1_beat_refc_xl_final_0.458` | REF-C-XL, legacy **0.458** → primary **0.4714** | 0.4886 ≥ 0.458 → ❌ | 0.4375 vs 0.4714; paired Δ **−0.0340 [−0.1060, +0.0511]**, not separated | ❌ **→ TIE (undecided)** |
| `G2_beat_v1_0.4522` | flagship v1, legacy **0.4522** → primary **0.4271** | 0.4886 ≥ 0.4522 → ❌ | 0.4375 vs 0.4271; paired Δ **+0.0104 [−0.0888, +0.1147]**, not separated | ❌ → **TIE (undecided)** |
| `G3_miss_le_0.10` | fixed 0.10 | 0.1424 > 0.10 → ❌ | 0.1056 [0.0488, 0.1714] | ❌ holds, interval straddles |

> ⚠️ **BOTH SIDES OF A THRESHOLD MUST BE CORRECTED TOGETHER.** `G2`'s threshold 0.4522 is itself
> v1's legacy split-mean. Correcting only the numerator (0.4886 → 0.4375) against an uncorrected
> 0.4522 would manufacture a **PASS** out of nothing. Against v1's true 0.4271 it is a tie. This
> is the same defect `gate_guard`'s docstring records for `planner_p2`'s G4 — *"compared against
> a threshold that was itself a mean-of-split-means"* — and it is the single easiest way to turn
> this closeout into a false positive.

**`taniteval/results/refc-base-30k.json`, `refc-xl-30k.json`** — `heldout` only, no
`cluster_bootstrap` (pre-2026-07-20 format). Recomputed above: 0.4523 → **0.4728** and
0.4577 → **0.4714**.

**`taniteval/results/flagship-v3enc-10k.json`, `flagship-v4.1-10k.json`,
`flagship-v4.2-step4000.json`** — carry BOTH blocks. On re-run only the bare `heldout` key becomes
a tombstone; no number is lost and the primary was already present.

**`taniteval/results/golden.json` — DOES NOT EXIST.** The `runner.regression` change is therefore
**prospective only**: there is no banked golden file whose entries could have been read off the
deprecated estimator. Stated explicitly because "no movement" is a finding, not an omission.

### 3.5 UNRECOMPUTABLE — what would be needed

| artifact / number | why it cannot be recomputed here | what would be needed |
|---|---|---|
| `refc_rerank` published headline (`ade2s_heldout_baseline` / `_oracle`, the λ and top-K curves) | the `<key>.json` it writes is **not in this repo** — the banked copies live under `stack/experiments/pod-rescue-20260802/pod3/root/taniteval/results/`, and the λ/top-K curves need the **decode FAN** (`fan_refc-*.pt` exist for `base-30k`/`xl-30k` only, not for the `refc-v12*` arms) | re-run `taniteval/taniteval/refc_rerank.py` on a GPU with the `refc-xl-30k` checkpoint to regenerate the fan, then re-derive the curve. **~1 GPU-hour.** The single-arm `ade2s` values ARE recomputed in §3 from the banked window dumps; only the per-λ curve is missing. |
| `generalization` `gen_*.json` | **no `gen_*.json` is banked in `taniteval/results/`** | re-run `python -m taniteval.runner generalize --model <key> --corpus <c>`; needs the val corpora for each OOD corpus |
| dashboards (`dashboard.html`) | not a tracked artifact; regenerated from `results/` on demand | nothing — it will pick up the corrected ordering on the next `report.build()` |
| the 6 `driving_diagnostic.mean_ci` call sites' banked outputs (`d1_*`, `compare_arms`, `eval_metric_rollout`, `eval_grounded_rollout_4b*`) | **their numbers DO NOT MOVE** — the arithmetic is byte-identical; only the labels are new | nothing. This is a labelling fix, not a value fix, and saying so is the honest answer rather than inventing a correction. |
| every pre-2026-07-20 registry row with no surviving `windows_*.pt` | the per-window artifact is gone; only the split-mean survives | the checkpoint + a re-eval. Where a row has no dump, it must be quoted as `legacy overlapping_holdout_se` or not at all. |

---

## 4. `MODEL_REGISTRY.md` — corrections needed (⛔ NOT EDITED BY ME)

The registry is the quotable source and was not touched. **Most of the headline correction is
already in it** — §6's leaderboard (`:2805-2807`) already publishes the bootstrap values, `:2209`
already retracts the "REF-C-XL finishes 0.006 m behind v1" claim (the true gap is 0.0443 m, 8×
larger), and `:2773` already records the REF-C base/XL swap. What remains:

| registry line | issue | correction |
|---|---|---|
| `:2204` REF-C comparison table | quotes `0.458 ± 0.057` / `0.470 ± 0.057` / `0.5645 ± 0.0447` — all legacy, unlabelled in that row | add the primary column: `refc-xl-30k` **0.4714** [0.3896, 0.5556]; `refc-xl-live` **0.4788** [0.3977, 0.5638]; `refc-xl` **0.6048** [0.5170, 0.7009] (**+7.14 %**, the largest positive shift in the corpus) |
| `:636` v1.6 table + `:640-641` verdicts | the row is a **cross-arm comparison of split-means**, which the same document rules invalid at `:668` — and G1/G2/G3 are adjudicated on it | replace with the corrected reads in §3.4 above. **G1 and G2 move from a hard ❌ to an undecided TIE.** Correct BOTH sides of each threshold. |
| `:1936` "Result (ADE@2s heldout)" run table | a whole results table in the deprecated estimator | re-issue from §3's table, or stamp the column header `legacy overlapping_holdout_se — NOT decision-grade` |
| §3.3 head-to-head rows | the `refc-v12` vs `refc-v12-k16reg` **separation does not survive** | re-state as a tie: −0.0049 [−0.0199, +0.0098], paired episode-cluster bootstrap |
| any row quoting the shift range | `-6.67 % … +11.69 %` at `:695` etc. is `(legacy−primary)/primary` | keep it, but **name the denominator**; the sibling range `−10.46 % … +7.14 %` is the same measurement the other way round |

---

## 5. TESTS

Every change ships with its regression test, and **every closure has a deliberate-regression arm**
that re-introduces the exact defect and requires the guard to catch it.

| file | what it pins |
|---|---|
| `taniteval/tests/test_estimator_closeout.py` **(new, 27 tests)** | all three briefed sites + the four found ones; the tombstone exposes no number; `run_gate`'s tripwire still fires on it; `runner.regression` REFUSES (and refuses a *tombstoned* artifact identically, so it is named rather than silently dropped); the leaderboard is ordered on the primary, proven with an **order-inversion fixture**; `mean_ci`'s numbers are bit-identical to the removed body |
| `taniteval/tests/test_no_jack_in_gates.py` **(extended)** | `ENFORCED_ROOTS` gains the `taniteval/` top level; the **shape census** is exhaustive and exact in both directions; a scope-collapse canary (`> 200` files must actually be scanned); three deliberate-regression fixtures for the shape guard; a false-positive control |
| `taniteval/tests/test_runner_gate_print.py` **(updated)** | the old `heldout is LEGACY_BLOCK` identity pin is **superseded** by a two-sided pin: the KEY must survive (tripwire) and the NUMBERS must not (quotability) |
| `stack/tests/test_driving_diagnostic.py` **(extended)** | `mean_ci is overlapping_holdout_mean_ci`; the output self-labels; the ddof=1 arithmetic is unchanged |

**Measured suite counts (this clone, `PYTHONPATH=<worktree>/taniteval;<worktree>/stack`):**

| suite | before | after | delta |
|---|---|---|---|
| `tools/tests` | 365 passed | **387 passed, 0 failed** | +22, **none of them mine** — a sibling agent staged `tools/tests/test_media_manifest.py` (+19) and `test_release_gate.py` (+3) into this worktree mid-session |
| `taniteval/tests` | 1229 passed / 6 failed / 22 skipped | **1264 passed / 6 failed / 22 skipped** | **+35 passed, 0 new failures** |
| `stack/tests` (gate/eval/ci subset, 848 selected) | — | 833 passed / 15 failed / 10 errors | **no failure references any module I touched** (verified by grep) |

> ⚠️ **The brief's baselines are stale, and the difference is not mine.**
> It quotes `tools/tests` **362** and `taniteval/tests` **1235 passed / 0 failed**. Measured at
> HEAD `49eb9b8` *before any edit*: `tools/tests` **365**, `taniteval/tests` **1229 / 6 failed**.
> All 6 pre-existing failures are `taniteval/tests/test_c2_published_policy.py`, and all 6 have
> one cause: **the file hardcodes `TanitAD Research Hub` at line 32**, while the PI's 2026-08-27
> rename made the tree `TanitAD Research Lab`. The artifacts exist — I verified
> `v5_v1_windows_reduced.pt` and 9 siblings are present under the Lab path and git-tracked. It is
> a one-line fix in another product's test and I did not sweep it into this commit.
> **⇒ ESCALATION 3: `taniteval/tests/test_c2_published_policy.py:32` `_HUB` → the Lab path.**
> The same rename breaks `stack/tests/test_build_parity_guard.py` and
> `stack/tests/test_eval_contamination.py` (10 errors). The other three `stack` failures
> (`test_loss_determinism`, `test_v6_s2_loss`, `test_x4_layer_spectrum`) are a pre-existing
> `train_v6_staged.py:2845 stage_a_losses() got an unexpected keyword argument
> 'stopgrad_factual'` signature mismatch — unrelated to estimators and untouched by me.

---

## 6. FILES CHANGED

| file | change |
|---|---|
| `taniteval/taniteval/bench.py` | `HELDOUT_ALIAS` + `_tombstone_node` / `_alias_tombstone`; `assert_no_deprecated_estimator` added and called before the quarantine keys are attached |
| `taniteval/taniteval/runner.py` | `regression()` REFUSES instead of falling back; `_has_deprecated_block` accepts both artifact vintages; `--allow-missing-primary` downgrades fatal→reported but never substitutes |
| `taniteval/taniteval/report.py` | `_primary()` gatekeeper; leaderboard ordered and rendered on `cluster_bootstrap`; no-primary arms shown as `NO PRIMARY INTERVAL`; the panel note rewritten |
| `taniteval/taniteval/refc_rerank.py` | `ade2s_heldout*` → `ade2s_primary*` + `ci_lo`/`ci_hi`/`estimator`; protocol string corrected |
| `taniteval/taniteval/efficiency.py` | `cost_per_accuracy` reads the primary and RAISES on a deprecated-only artifact; historical rows render marked `DEPRECATED est.` |
| `taniteval/taniteval/generalization.py` | the `[gen]` print reads `cluster_bootstrap` and names its estimator |
| `taniteval/taniteval/gate_guard.py` | `_dispersion_names` (CLOSED-5); declared names are banned callables (CLOSED-6) |
| `taniteval/recompute_ci.py` | `naive_published` → `reproduce_overlapping_holdout_published`; reads either artifact vintage; scope note |
| `stack/scripts/driving_diagnostic.py` | `mean_ci` → `overlapping_holdout_mean_ci` (+ alias); output self-labels; `DEPRECATED_ESTIMATOR` constant |
| `stack/scripts/eval_flagship_v15.py`, `eval_flagship_v16.py` | G1/G2/G3 adjudicate on `cluster_bootstrap` (CLOSED-7) |
| `stack/scripts/refc_v12_eval.py` | the summary print reads `cluster_bootstrap` |
| `taniteval/tests/test_estimator_closeout.py` | **new** |
| `taniteval/tests/test_no_jack_in_gates.py`, `test_runner_gate_print.py`, `stack/tests/test_driving_diagnostic.py` | updated / extended |
| `taniteval/results/estimator_closeout_ledger.json` | **new** — the raw per-arm ledger behind §3 |
| `products/P7-TanitEval/ESTIMATOR_CLOSEOUT.md` | this document |

---

## 7. ESCALATIONS

1. ⛔ **PI — authorise a D1 re-adjudication (OPEN-1).** `tanitad/eval/gates.run_d1` decides
   `passed` on a mean-of-split-means under `overlapping_holdout_se`. Fixing it moves every D1
   verdict in the programme. ~1 CPU-day; enumerated in `SHAPE_KNOWN_OPEN` so it cannot vanish.
2. **P7 backlog — taint-by-key in `gate_guard` (OPEN-2).** The name guard cannot see a verdict
   whose input arrives through `res["heldout"]["model"]`. Needs its own false-positive census.
3. **P7 — `test_c2_published_policy.py:32` `TanitAD Research Hub` → `TanitAD Research Lab`.**
   One line; unblocks 6 red tests and restores a clean `taniteval/tests` baseline. Same rename
   breaks `stack/tests/test_build_parity_guard.py` and `test_eval_contamination.py`.
4. **Registry owner — apply §4.** ⛔ I did not edit `MODEL_REGISTRY.md`. The `refc-v12` vs
   `refc-v12-k16reg` **separation loss** is the row that changes a claim, not just a number.
5. **P7 — re-run `refc_rerank` on a GPU** to close the one UNRECOMPUTABLE row (§3.5).
