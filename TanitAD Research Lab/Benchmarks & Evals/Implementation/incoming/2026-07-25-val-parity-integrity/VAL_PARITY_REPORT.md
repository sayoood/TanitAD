# Val-side parity integrity — the mirror of Wave-1 B, on the side that touches published ADEs

**Date:** 2026-07-25 · **Scope:** `taniteval/` + the eval-side of `stack/scripts/`
**Status:** implemented, tested, left in the working tree — this agent did **NOT** `git add` / commit / push.
**Compute:** dev box only. No pod SSH, no GPU (pod1/pod2/pod3 training, eval pod mid-transfer).
**Suites:** `taniteval` **286 → 334 passed** · `stack` **867 → 924 passed, 3 skipped**. +105 tests, 0 regressions.

---

## 0. The residual Wave-1 B escalated, and the two things it turned out to be

WAVE1_B_REPORT.md §7.2:

> *"13 evaluators glob a val cache with no integrity check. A truncated val cache silently changes every
> published ADE the same way a truncated train cache changes every arm. This needs an owner."*

That is true, and it is the smaller half. Auditing the 13 surfaced a **second, worse** defect in the same
line of code.

### ⭐ FINDING 1 — the inherited val-dir resolver *selected the leaky split*

**MEASURED** (dev box, `python -c "sorted(['physicalai-val-0c5f7dac3b11','physicalai-val-f1b378f295ae'])[-1]"`):

```
sorted:     ['physicalai-val-0c5f7dac3b11', 'physicalai-val-f1b378f295ae']
[-1] picks: physicalai-val-f1b378f295ae      <-- the 78.5 %-LEAKED split
```

`sorted(Path(cache_dir).glob("*val*"))[-1]` — the "newest dir wins" convention **10 of the 13 evaluators**
inherited verbatim from the trainers — is a **lexicographic** max, and `'0' < 'f'`. Wherever both splits are
materialised under one epcache root, that line does not merely *fail to check* which split it got: it
**actively prefers the leaked one**. Pinned permanently as
`test_the_legacy_resolver_would_have_SELECTED_the_leaky_split` in both suites.

### FINDING 2 — the count hole is not hypothetical on the val side; it is the normal state

The val corpus exists in **at least three different sizes**, all under the identical directory name
`physicalai-val-0c5f7dac3b11`:

| where | episodes | evidence | class |
|---|---|---|---|
| `<epcache root>` on the training pods | **600** | `labels_val_v4_provenance.json` `n_episodes`; `parity_skipset.sh` asserts `len(val)==600`; `parity_manifest.json` | MEASURED |
| `tanitad-eval:/root/valdata/` | **40** → the 881 stride-8 windows every published number is quoted over | `MODEL_REGISTRY.md:61`; 30/30 decision-grade `taniteval/results/*.json` carry `n_episodes=40, n_windows=881` | MEASURED |
| `tanitad-pod (pod1):/root/valdata/` | **12** → 265 windows | `…/2026-07-23-lower-ood-closedloop-source/P1_DECISION_GRADE_FINDINGS.md:19`, `LOWER_OOD_CLOSEDLOOP_DESIGN.md:96` | INHERITED (another agent's pod scan) |

So the naive fix — assert `== 600` — would have **refused every eval this program has ever run**, and the
naive-cautious fix — assert `<= 600` — catches nothing. `list_val_episodes(VAL, 40)` against the 12-episode
deployment returns **12 files and scores them**, and the resulting ADE is published beside 881-window numbers
with nothing to distinguish it. This is the near-miss class WAVE1_B_REPORT §5 recorded: *a guard that fires
in the wrong failure mode costs a finished run.*

---

## 1. Per-evaluator audit (MEASURED — state at `f57ce2a` + the sibling's staged Wave-1 B)

`BEFORE` = what the entry point asserted about its val cache before this work. `AFTER` = what it asserts now.
Every `AFTER` check runs **before a single episode file is opened** (the fixtures are empty `ep_*.pt`
markers, so "the guard raised" and "the guard ran before `torch.load`" are the same assertion).

### 1.1 `stack/scripts/` — the 13 Wave-1 B listed, plus 2 it did not

| # | Evaluator | Val cache via | BEFORE (MEASURED) | AFTER |
|---|---|---|---|---|
| 1 | `evaluate_checkpoint.py` (`:176`) | `sorted(glob("*val*"))[-1]` → `glob("ep_*.pt")[:n]` | **NOTHING**, and the resolver picks the LEAKY split | `resolve_val_dir` + `assert_val_cache(requested=--episodes)` |
| 2 | `compare_arms.py` (`:165`, `:179`) | same, ×2 (frame val **and** REF-A DINO feature val) | **NOTHING** ×2 | guard on both; feature dir guarded directly when it holds `ep_*.pt` |
| 3 | `driving_diagnostic.py` (`:424`) | same | **NOTHING** | guard |
| 4 | `d1_probe_capacity.py` (`:97`) | `sorted(glob("*val*"))[-1]`, `[:12]` | **NOTHING** | guard, `requested=12` |
| 5 | `d3_decompose.py` (`:117`) | same, `[:16]` | **NOTHING** | guard, `requested=16` |
| 6 | `run_spectral.py` (`:47`) | same | `assert val_dirs` (dir EXISTS — not which, not how many) | guard |
| 7 | `geom_sanity.py` (`:59`) | same | `assert val` (existence only) | guard |
| 8 | `resolution_probe.py` (`:97`) | same | **NOTHING** | guard |
| 9 | `eval_grounded_rollout_4b.py` (`:139`) | same | **NOTHING** | guard |
| 10 | `eval_metric_rollout.py` (`:124`) | same | **NOTHING** | guard |
| 11 | `eval_behavior.py` (`:823`) | same — **not in Wave-1 B's list**, found by re-probing | **NOTHING** | guard |
| 12 | `eval_flagship_v4.py` (`:113`, `:133`) | explicit `--val-cache`, `glob("ep_*.pt")` ×2 | `raise if not files` (empty only) | `assert_val_cache` in **both** dataset builders |
| 13 | `eval_flagship_v15.py` (`:124`) | explicit `--val-cache`, `os.listdir[:n]` | **NOTHING** | `assert_val_cache(requested=n)` |
| 14 | `eval_flagship_v16.py` (`:113`) | explicit `--val-cache`, `os.listdir[:episodes]` | **NOTHING** | `assert_val_cache(requested=episodes)` |
| 15 | `refc_v12_eval.py` (`:221`) | `taniteval.data.list_val_episodes` | leaky refusal only | inherited — full guard via the chokepoint |

**BEFORE summary: 0 of 15 checked the episode count. 10 of 15 used a resolver that prefers the leaked split.**
Two had an existence check (`assert val_dirs`), which is not an integrity check.

### 1.2 `taniteval/` — decision-grade modules (all already routed through one chokepoint)

`data.list_val_episodes` was already the single entry point for every decision-grade module, and it already
hard-refused the leaky split (`allow_leaky=False`, added `df32781`). What it did **not** do was count.

| Module | BEFORE | AFTER |
|---|---|---|
| `runner` (`:60/:170/:225`), `closedloop` (`:903`), `hierarchy` (`:1116`), `pathspeed` (`:445`), `efficiency` (`:1635`), `refc_rerank` (`:241`), `planning` (`:253`), `planner_p2` (`:591`), `bench` (`:587`), `generalization` (`:899/:927`), `strategic_probes` (`:484`) | leaky-hash substring refusal; **no count, no cache identity, no requested-vs-delivered** | full guard, inherited from the chokepoint; every module pinned individually by `test_every_decision_grade_module_routes_through_the_chokepoint` |

### 1.3 `taniteval/` — modules that BYPASSED the chokepoint (found by re-probing, not assumed)

| Module | BEFORE | AFTER |
|---|---|---|
| `cam_overlay.py:121` | hardcoded clean-val path, bare `glob` | routed through `list_val_episodes` |
| `flagship_overlay.py:184` | `sorted(Path(VAL).glob(...))` | routed |
| `corpus_overlay.py:391` · `direct_overlay.py:208` · `plan_fan.py:710` · `plan_fan_clips.py:259,299` | `sorted(Path(corp["root"]).glob(...))` from the CORPORA registry | routed (NON-PARITY corpora warn and run, as before) |
| `label_overlay.py:297` | **`--val` DEFAULTS to the LEAKY split**, `glob.glob`, silent | `list_val_episodes(allow_leaky=True)` + `--allow-leaky` — the audit is still allowed, it now **announces itself** |

### 1.4 The second name and the second path (brief item 3 — "absence at one location is not absence")

Probing beyond the 13 for a *second argument name* (`--pai-val-cache`) and a *second default* found a whole
family with the **leaky split hardcoded as the default value**:

| Script | BEFORE | AFTER |
|---|---|---|
| `run_branchb_transfer.py:229` · `run_idm_parity_validation.py:121` · `run_v1_encoder_char.py:167` | `--val-cache` **default = `physicalai-val-f1b378f295ae`**, consumed by `R.select_episodes`, nothing refused it | `--allow-leaky-val` opt-in; without it `assert_val_cache` **refuses** |
| `run_idm_proof.py:271` · `run_idm_ft.py:271` | `--pai-val-cache` required; Wave-1 B guarded the **train** cache at `:218`, the val cache stayed unchecked, and both docstrings point it at the leaky split | same opt-in + refusal |
| `run_idm_pipeline_derisk.py:154` | `--val-cache` default = leaky, **and never read** — a dead argument advertising the leaked corpus | annotated as dead + a do-not-wire-without-the-opt-in note |
| `route_label_audit.py` · `vlm_route_labels.py` · `vlm_kin_crossval.py` | `--val` default = leaky; legitimate (they audit route LABELS, produce no ADE) but **silent** | `parity.note_leaky_audit(...)` — discloses, stamps `decision_grade: False`, does not refuse |

**This is the answer to brief item 3: the refusal did NOT cover everything.** It covered `taniteval`'s
decision-grade chokepoint and (after Wave-1 B) the trainers. It did not cover 10 `stack` evaluators whose
resolver *preferred* the leaked split, nor 8 scripts that named it as a default argument.

---

## 2. What I wired

### 2.1 One guard, extended — `stack/tanitad/data/parity.py` (no second implementation)

Added a val-side surface to the sibling's module rather than a parallel one:

| New | What |
|---|---|
| `resolve_val_dir(root, *, label)` | the fixed replacement for `sorted(glob("*val*"))[-1]`: a dir referencing the registered CLEAN key wins outright; a leaky-only root is refused; when a leaky dir is skipped it **prints what the legacy rule would have picked**. Raises `AssertionError` (not `SystemExit`) when nothing matches — WAVE1_B §5's lesson: "you gave me no val dir" must not kill a finished run at its metrics write. |
| `assert_val_cache(cache_dir, *, label, requested, decision_grade)` | THE evaluator-facing guard: leaky refusal → absent/empty is a loud non-fatal `checked=False` → `check_uids(mode="subset")` (count bound + uid digest **when the manifest has one**) → **deployment check** → requested-vs-delivered. |
| `guard_val_split(root, ...)` | resolve + assert, for the `*val*`-pattern callers. |
| `val_deployments()` | the registered admissible counts, read from the manifest. |
| `note_leaky_audit(path, *, label, why)` | the ONE sanctioned way to touch a leaky split: discloses, returns `decision_grade: False`, never licenses a number. |

Two upstream hardenings, both additive:

* `check_uids(..., subset_note=...)` — the default subset warning ("must not be cross-compared with
  full-corpus arms") is right for a *train* subset and **wrong** for a val *deployment*, where every arm
  shares the same 40-episode prefix.
* `check_uids(mode="subset")` now verifies the **digest** when the observed set is the FULL set and the
  manifest carries a digest but no uid list. Without this, a manifest written by `build_entry` (digest-only)
  would have silently degraded every subset-mode caller to count-only while looking at the entire corpus.

### 2.2 The manifest — registered val deployments, with citations, no invented hash

**The val entry stays COUNT-ONLY.** No committed artifact enumerates the val uids; the Wave-1 B agent
correctly refused to derive one, and I refused too. Pinned by
`test_val_manifest_is_count_only_and_no_hash_was_invented`.

What I *added* is the thing that makes a count check meaningful for a corpus deployed as subsets — the
admissible counts as **data with evidence**, not a magic number in code:

```json
"known_deployments": [
  {"n_episodes": 600, "role": "full build (the epcache split dir on the training pods)",
   "evidence": "labels_val_v4_provenance.json n_episodes=600; parity_skipset.sh asserts len(val)==600",
   "evidence_class": "MEASURED"},
  {"n_episodes": 40,  "role": "canonical TanitEval deployment -> 881 stride-8 windows (THE published open-loop statistic)",
   "evidence": "MODEL_REGISTRY.md:61; 30 of 30 decision-grade taniteval/results/*.json carry n_episodes=40, n_windows=881",
   "evidence_class": "MEASURED"}
],
"deployments_seen_but_NOT_admissible": [
  {"n_episodes": 12, "where": "tanitad-pod (pod1):/root/valdata/…",
   "why_not": "a PARTIAL deployment; not the 40-episode/881-window statistic every published number is quoted over",
   "evidence_class": "INHERITED … Used only to REFUSE, never to license a number."}
]
```

Manifest diff: **+33 / −3 lines**; the 2 376-entry train uid list and its digest `9877bef64da3…` are byte-identical.

### 2.3 ⚠ THE ONE POD COMMAND THAT UPGRADES THIS TO A CONTENT CHECK

Until this runs, the val side is **COUNT + CACHE-IDENTITY**, and it says so in its own log line on every run
(`test_count_only_mode_says_it_is_count_only` pins that the line names the command). On a pod where
`scripts/pod_ops/compute_skipset.py` has just printed **`VERDICT MATCH`**:

```bash
PYTHONPATH=/workspace/TanitAD/stack python3 \
  stack/scripts/make_parity_manifest.py --record --split val \
  --cache-dir /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11
# then bring the changed stack/tanitad/data/parity_manifest.json back to the repo and stage it
```

`--record` writes **both** the digest and the uid list, so subset-mode prefix checking activates for the
40-episode deployment too, with **no code change** — pinned end-to-end by
`test_uid_digest_IS_enforced_once_the_manifest_carries_one` / `test_uid_digest_is_enforced_the_moment_one_is_recorded`.

### 2.4 `taniteval/taniteval/data.py` — the chokepoint

`list_val_episodes(val_dir, n, allow_leaky=False, allow_partial=False, label=None)` now calls the shared
guard before listing. Three deliberate design choices:

1. **Late import, hard refusal.** `tanitad.data.parity` is imported inside the guard, not at module load —
   a bare `ImportError` at import time would take down `--help` and every non-val path on a pod whose
   `stack/` checkout is behind. If it *is* missing, the eval **refuses** with "sync `stack/` to this
   machine": a run that cannot verify its val cache must not quietly produce a number.
2. **Absence is not a violation.** A missing or empty val dir returns `checked=False` and a loud line; the
   caller's own `assert files` owns that case. This is what keeps the pre-existing productionization tests
   (which run off-pod, where `/root/valdata` does not exist) honest rather than merely green.
3. **`last_val_parity()`** exposes the integrity record so an emitter can stamp it — see §5.

---

## 3. Test evidence — RED → GREEN

Two new files, **105 new tests**, both suites green.

### RED (guard present, evaluators NOT yet wired)

`taniteval/tests/test_val_parity.py`, run 1 — **9 failed, 28 passed**; run 2 after refining the bypass
regex — **7 failed, 30 passed**, every failure a real bypass site:

```
FAILED test_viz_modules_also_route_through_the_chokepoint[cam_overlay]
FAILED test_viz_modules_also_route_through_the_chokepoint[flagship_overlay]
FAILED test_viz_modules_also_route_through_the_chokepoint[corpus_overlay]
FAILED test_viz_modules_also_route_through_the_chokepoint[direct_overlay]
FAILED test_viz_modules_also_route_through_the_chokepoint[plan_fan]
FAILED test_viz_modules_also_route_through_the_chokepoint[plan_fan_clips]
FAILED test_viz_modules_also_route_through_the_chokepoint[label_overlay]
```

`stack/tests/test_val_parity_evaluators.py` — **3 failed, 54 passed**, all three the leaky-default probes
whose guard insertion had silently no-op'd on a callsite mismatch (`run_branchb_transfer`,
`run_idm_parity_validation`, `run_v1_encoder_char`). *That failure is itself the argument for asserting the
wiring per file rather than trusting a bulk edit.*

### The permanent PREMISE PINS (the analogue of the sibling's substring pin)

| Test | Asserts the OLD behaviour, forever |
|---|---|
| `test_the_legacy_resolver_would_have_SELECTED_the_leaky_split` ⭐ | `sorted(glob("*val*"))[-1].name == LEAKY`, and that `resolve_val_dir` picks CLEAN on the same root |
| `test_the_old_glob_would_have_PASSED_a_truncated_val_cache` | `sorted(glob("ep_*.pt"))[:40]` returns 12 files from a 12-episode cache, no error, correctly-named dir |
| `test_no_evaluator_still_uses_the_leaky_selecting_resolver` | regex over all 14 evaluators, comments stripped, so *documenting* the bug does not fail the test |

### GREEN

```
taniteval/tests/test_val_parity.py        37 passed
taniteval/tests/test_runner_gate_print.py 11 passed
stack/tests/test_val_parity_evaluators.py 57 passed
```

| Suite | baseline | after |
|---|---|---|
| `taniteval` | **286 passed** (21.6 s) | **334 passed** (23.2 s) |
| `stack` | **867 passed, 3 skipped** (64.9 s) | **924 passed, 3 skipped** (69.8 s) |

Plus a **compile + import sweep over all 45 touched files** (24 `stack/scripts`, 20 `taniteval` modules,
`parity.py`): `0 failures`. A syntax or import error in an evaluator would otherwise surface only at launch,
on a pod, after a model load — it caught 3 broken import insertions during this work.

### Coverage map (selected)

| Test | Guards against |
|---|---|
| `test_truncated_val_cache_is_refused` ⭐ | 12 episodes in a correctly-named clean-val dir |
| `test_unregistered_episode_counts_are_refused[1,11,12,39,41,599,601]` | every off-by-N truncation and over-fill |
| `test_requesting_more_episodes_than_present_is_refused` | "asked 40, cache holds 12" — the silent rescoring |
| `test_geom_sanity_loader_picks_the_clean_split_over_the_leaky_one` | end-to-end: with BOTH splits present the evaluator now reads the clean 40 where it read the leaked 79 |
| `test_leaky_default_probes_now_require_an_explicit_opt_in` (×5) | §1.4's second-name family |
| `test_leaky_label_audits_disclose_instead_of_running_silently` (×3) | the sanctioned-but-silent class |
| `test_absent_val_dir_is_not_a_parity_violation` · `test_missing_val_dir_stays_an_AssertionError_not_a_refusal` | WAVE1_B §5's near-miss, re-applied |
| `test_non_parity_corpora_warn_but_never_block` | comma / cosmos / OOD keep working |
| `test_val_deployments_are_registered_data_with_citations` | the counts stay evidenced, never folklore |

---

## 4. Blast radius — assessed honestly, with what I could and could not determine

### 4.1 Could a published ADE have been computed on a TRUNCATED val cache?

**RULED OUT for every committed `taniteval` result — MEASURED.** Enumerating all 36 committed result JSONs
carrying window/episode counts:

```
n_episodes=40  n_windows=881   x30      <- the canonical deployment
n_episodes=None n_windows=881  x4       <- pre-dates the n_episodes field; same window count
n_episodes=4   n_windows=88    x2       <- driving_refc-v12-smoke-{reg,t0}, labelled smoke
```

`n_windows` is a near-unique fingerprint of the episode set at fixed window/stride: the 40-episode
deployment gives **881**, the 12-episode pod1 deployment gives **265**
(`LOWER_OOD_CLOSEDLOOP_DESIGN.md:96`, independent source). **Every** decision-grade committed result carries
881. The four `n_episodes=None` files (`refc-xl-30k`, `refc-base-30k`, `eval_v16_flagship-v16-ab-ft`,
`v1-validation`) also carry `n_windows=881` — the field was added later, the corpus was the same.

### 4.2 Could a published ADE have been computed on the LEAKY split?

**RULED OUT for the `taniteval` ADE path — MEASURED, over the full git history, by two independent probes.**

* `git log --all -S "f1b378f295ae" -- taniteval/` returns exactly **two** commits, and the only file
  involved is `label_overlay.py` (a label-audit **video renderer**, no ADE, introduced `ec0dba5`).
* `runner.VAL` has been `/root/valdata/physicalai-val-0c5f7dac3b11` since the harness's **first** commit
  (`git show a91bef8:taniteval/taniteval/runner.py` → `VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"`),
  i.e. *before* the `allow_leaky` refusal existed. The refusal hardened a property that already held.

**NOT ruled out — and already a known, corrected instance — outside the ADE path.** `MODEL_REGISTRY.md`
§Branch-B (corrected 2026-07-25) records that the Branch-B transfer `*_val` **R²** numbers were computed on
`f1b378f295ae` via `run_branchb_transfer.py`'s leaky default (§1.4). Those are R², not ADE, and the registry
already carries the correction and the reasoning that the *ordering* is conservative. My change makes the
same script refuse that split without `--allow-leaky-val`.

### 4.3 The one thing I could NOT determine, and exactly what would settle it

**I cannot determine from committed artifacts whether the two val split dirs ever co-existed under a single
epcache root** — which is the precondition for FINDING 1 to have actually fired rather than merely being
latent.

What the committed evidence shows is a *two-root* pattern, which is suggestive but **not proof**:

* every committed command naming `physicalai-val-0c5f7dac3b11` uses `/workspace/data/physicalai_phase0/_epcache`
  (pod1/pod2) or `/root/valdata` (eval pod) — e.g. `results/trainlogs/flagship-v4.1-10k_config.json`;
* every committed command naming `physicalai-val-f1b378f295ae` uses `/workspace/pai_epcache` (pod3) —
  `run_idm_*`, `vlm_*`, `route_label_audit`, `label_overlay`;
* `MODEL_REGISTRY.md:37` states the roots as pod1/pod2 `…/_epcache` · pod3 `/workspace/pai_epcache`.

**This is not a clean bill of health and I am not writing one.** The `[-1]`-resolver evaluators take
`--cache-dirs <root>` as a free argument, so what matters is what was on disk at each invocation, not what
the docs say. There is also one *published* artifact from that family: `LEADERBOARD.md §8`, the camera-frame
D1/D2/D3 gate ladder from **2026-07-12** ("exact training val (comma+pai)"), produced by
`evaluate_checkpoint.py` / `driving_diagnostic.py`. It is explicitly marked **SUPERSEDED, different unit**,
and no current decision rests on it — but it is the one place a `[-1]`-selected val could have reached a
published table.

**What would settle it — one read-only command per pod, no GPU, ~1 s:**

```bash
ls -d /workspace/data/physicalai_phase0/_epcache/*val*   # pod1, pod2
ls -d /workspace/pai_epcache/*val*                       # pod3
ls -d /root/valdata/*val*                                # eval pod, pod1
# and the counts that upgrade §2.3 at the same time:
for d in <each dir>; do echo "$d $(ls "$d"/ep_*.pt | wc -l)"; done
```

If any single root lists **both** split dirs, every number ever produced by the §1.1 family against that root
must be re-derived, starting with `LEADERBOARD.md §8`. I could not run this (pods are training; the brief
forbids pod access), and I will not guess the answer.

### 4.4 Why the question was this hard — and the fix so it never is again

**No result JSON recorded which val cache produced it.** §4.1 had to be answered by *inferring* the corpus
from `n_windows == 881`. That is a reconstruction, not a record. `runner.run_one` now stamps
`res["val_parity"] = data.last_val_parity()` into every result JSON — corpus key, cache dir, episodes
present, episodes listed, the deployment matched, the content-check strength, and `decision_grade`. The next
audit **reads** it.

---

## 5. Second task — the deprecated gate-facing print (`runner.py:153`)

### The before

```python
hm = res["heldout"]["model"]                                  # overlapping_holdout_se
print(f"[run] {key} step={…} n={res['n_windows']} "
      f"ade@2s={hm['ade_0_2s']['mean']:.3f}±{hm['ade_0_2s']['ci95']:.3f} …")
```

`closedloop.py`, `driving.py` and `hierarchy.py` had all been migrated. This line — **the number a human
reads when gating an arm** — had not.

### The measurement (MEASURED, dev box, no GPU, over all 27 committed `results/windows_*.pt`)

Re-ran `bench.run` on every committed fixture and compared the two blocks. On **v1 `flagship-30k`**:

| metric | legacy `heldout` (`overlapping_holdout_se`) | primary `cluster_bootstrap` (`episode_cluster_bootstrap`) | shift | CI-width ratio |
|---|---|---|---|---|
| `ade_0_2s` | **0.4522 ± 0.0312** | **0.4271 [0.3675, 0.4871]** ±0.0598 | **−0.0251 (−5.6 %)** | **1.92×** |
| `fde@2s` | 0.9437 ± 0.0630 | 0.9075 [0.7851, 1.0306] ±0.1227 | −0.0362 (−3.8 %) | 1.95× |
| `miss_rate@2m` | 0.0602 ± 0.0121 | 0.0454 [0.0239, 0.0681] ±0.0221 | −0.0148 (**−24.6 %**) | 1.83× |
| `tms_openloop` | 0.1070 ± 0.0229 | 0.0978 [0.0701, 0.1304] ±0.0301 | −0.0092 (−8.6 %) | 1.31× |

`0.4522` is *exactly* the registry's published v1 headline; `0.4271` is its `full_set`. The gate-facing line
has been printing the deprecated-estimator mean the whole time.

Across all 27 arms:

* **CI width ratio primary/legacy: 1.11 – 3.10×, median 1.50×** — re-confirming the 1.28–2.06× program
  finding on this side, and exceeding it on the REF-A overfit arms (3.10×).
* **Point estimate shift: −10.5 % … +7.1 %, median +1.4 %.** `_agg` averages 8 overlapping 20 % holdouts;
  the primary point estimate is the full-set metric. They are different statistics, not different intervals
  on one statistic.
* ⭐ **The cross-arm RANKING changes in 10 of 27 positions** — e.g. legacy ranks `refc-base-30k` 2nd where
  the primary ranks `flagship-v16-ab-ft` 2nd; `refb` and `flagship-v4.1-10k` swap at ranks 14/15.

This is consistent with the finding the brief flagged (`_jack` biasing point estimates up to ×4.29 with sign
flips): the same defect, milder in magnitude here, but **it moves the leaderboard order**.

### The after

```python
cb = res["cluster_bootstrap"]["model"]
vs = res["cluster_bootstrap"]["model_vs_cv_paired"]
a  = cb["ade_0_2s"]
print(f"[run] {key} step={…} n={res['n_windows']} eps={res['n_episodes']} "
      f"ade@2s={a['mean']:.3f} [{a['lo']:.3f},{a['hi']:.3f}] fde={…} miss@2m={…} tms={…} | "
      f"vs CV Δ{vs['delta']:+.3f} [{vs['lo']:.3f},{vs['hi']:.3f}] "
      f"{'SEPARATED' if vs['separated'] else 'tie'} | "
      f"estimator={a['estimator']} B={a['n_boot']} ({res['wall_s']}s)")
```

Three changes beyond the estimator swap: the interval is a **[lo, hi]**, not a `±` (a percentile CI is not
symmetric); the **paired** model-vs-CV separation is on the same line (an unseparated win is a tie, not a
win); the **estimator and B are printed**, so the number can never be requoted without its construction.

### The quarantine

`bench.run` now emits `legacy_overlapping_holdout_se`, matching `closedloop.LEGACY_BLOCK` /
`hierarchy.LEGACY_BLOCK` so one grep finds every quarantined number in the harness. It carries `_estimator`,
`_why_kept`, `estimator_note`, and — MEASURED per artifact, not cited from a doc —
`ci_width_ratio_new_over_legacy` **and** `point_estimate_shift_primary_minus_legacy`.

> **`heldout` is retained as an alias to the same dict, deliberately.**
> `stack/scripts/run_gate.py:666` `_deprecated_present` searches exactly `("heldout", "model")` to decide
> fail-loud-vs-fallback. Renaming the key would have **silently disarmed the gate's own refusal** of the
> deprecated estimator — the precise opposite of the intent. Five other consumers (`report`, `efficiency`,
> `refc_rerank`, `generalization`, `_extract2`) also read it. Pinned by
> `test_heldout_alias_survives_because_the_gate_keys_on_that_name`, and the legacy numbers are asserted
> **bit-identical** to the pre-migration block so every published figure stays reproducible.

`runner.regression()` was migrated too: it read `heldout` means with an 8 % tolerance, so a real regression
could hide inside a ±10.5 % estimator gap. It now reads `cluster_bootstrap`, falls back to `heldout` only for
results written before that block existed, and **prints which arms fell back** so a mixed golden file cannot
pass as a like-for-like comparison.

---

## 6. Deliverable manifest

All paths relative to the repo root. **Nothing was `git add`ed, committed or pushed.** Nothing lives on a pod
or in a worktree. 35 modified + 3 new files, **+824 / −69 lines**.

**New**

| Path | What |
|---|---|
| `taniteval/tests/test_val_parity.py` | 37 tests — the chokepoint, the guard API, the premise pins, per-module routing |
| `taniteval/tests/test_runner_gate_print.py` | 11 tests — the gate-print migration + the quarantine |
| `stack/tests/test_val_parity_evaluators.py` | 57 tests — per-evaluator wiring, the leaky-default family, functional end-to-end refusals |
| `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-25-val-parity-integrity/VAL_PARITY_REPORT.md` | this report |

**Modified — `stack/`**

`tanitad/data/parity.py` (val-side surface) · `tanitad/data/parity_manifest.json` (registered deployments) ·
`scripts/`: `evaluate_checkpoint.py` · `compare_arms.py` · `driving_diagnostic.py` · `d1_probe_capacity.py` ·
`d3_decompose.py` · `run_spectral.py` · `geom_sanity.py` · `resolution_probe.py` ·
`eval_grounded_rollout_4b.py` · `eval_metric_rollout.py` · `eval_behavior.py` · `eval_flagship_v4.py` ·
`eval_flagship_v15.py` · `eval_flagship_v16.py` · `run_idm_proof.py` · `run_idm_ft.py` ·
`run_branchb_transfer.py` · `run_idm_parity_validation.py` · `run_v1_encoder_char.py` ·
`run_idm_pipeline_derisk.py` · `route_label_audit.py` · `vlm_route_labels.py` · `vlm_kin_crossval.py`

**Modified — `taniteval/`**

`taniteval/data.py` (the chokepoint) · `taniteval/bench.py` (quarantine) · `taniteval/runner.py` (gate print,
val provenance, regression) · `cam_overlay.py` · `flagship_overlay.py` · `corpus_overlay.py` ·
`direct_overlay.py` · `plan_fan.py` · `plan_fan_clips.py` · `label_overlay.py`

---

## 7. Left open — escalated here, not filed in a README

1. **⚠️ THE POD `ls` THAT CLOSES THE BLAST RADIUS (§4.3).** One read-only command per pod. If any single
   epcache root lists **both** val split dirs, `LEADERBOARD.md §8`'s D1/D2/D3 ladder and anything else from
   the §1.1 family against that root must be re-derived. **This is the only open question that could still
   invalidate a published table, and I could not answer it without pod access.**
2. **⚠️ The val uid digest (§2.3).** One `--record` run on a `VERDICT MATCH` pod upgrades both splits from
   count to content. Until then a *substituted* 40-episode val of the right size passes. The code path is
   already written and tested; only the data is missing.
3. **Registry/leaderboard follow-up on §5.** Every headline ADE quoted from the `heldout` block is a
   different statistic from the primary — and the *ranking* changes in 10 of 27 positions. Re-emitting
   §1/§6 of `MODEL_REGISTRY.md` and `LEADERBOARD.md §2` from `cluster_bootstrap` is a CPU-only job
   (`python -m taniteval.runner driving-all`) and is **out of this workstream's surface** — it needs the
   Benchmarks & Eval owner. Flagged, not done, because silently re-ranking the leaderboard from a
   side-quest is worse than leaving it flagged.
4. **`stack/scripts/eval_flagship_v15.py` / `v16.py` vendor `taniteval` from a pod path.** They import
   `bench` from `--vendor`, so they will get the migrated quarantine only once that copy is refreshed. The
   guard I added to them is in-repo and unaffected; the *estimator* they report is not.
5. **Builders should self-verify (Wave-1 B §7.3, still open and now also true for val).** Adding
   `make_parity_manifest.py --verify --cache-dir <dir>` as the last step of any val build would catch a
   quota death at the moment it happens rather than at the next eval.
6. **`run_idm_pipeline_derisk.py`'s dead `--val-cache`.** Annotated, not removed — removing a CLI argument
   other scripts' launch lines may pass is a compatibility change, not a cleanup.

---

## 8. Evidence classes

| Claim | Class |
|---|---|
| pre-fix per-evaluator state (§1 BEFORE columns) | **MEASURED** — file:line + the RED test runs |
| `sorted(glob("*val*"))[-1]` selects the leaky split | **MEASURED** — dev-box sort, pinned in both suites |
| val deployed at 600 / 40 / 12 episodes | **MEASURED** (600, 40) · **INHERITED** (12 — another agent's pod scan, used only to refuse) |
| 40 eps → 881 windows; 12 eps → 265 windows | **MEASURED** — `MODEL_REGISTRY.md:61`; `LOWER_OOD_CLOSEDLOOP_DESIGN.md:96` |
| every committed decision-grade result is 881 windows | **MEASURED** — enumeration of all 36 result JSONs |
| the leaky split never reached a `taniteval` ADE | **MEASURED** — `git log --all -S` over the full history + the first-commit `VAL` constant |
| Branch-B `*_val` R² used the leaky split | **INHERITED** — `MODEL_REGISTRY.md` §Branch-B, corrected 2026-07-25; not re-verified here |
| whether both splits ever shared one root | **UNDETERMINED** — §4.3 names the exact command that settles it |
| legacy-vs-primary shift, CI ratios, ranking changes | **MEASURED** — `bench.run` over the 27 committed `windows_*.pt`, this dev box |
| test results (286→334, 867→924, RED 7/9/3 → GREEN) | **MEASURED** — `C:/Users/Admin/venvs/tanitad` (py 3.13.5, torch 2.11.0+cu128, pytest 9.1.1) |
