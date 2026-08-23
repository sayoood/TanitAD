# Wave-1 B — parity integrity in `stack/`

**Date:** 2026-07-25 · **Scope:** `stack/` only (sibling agents held `taniteval/` and `Project Steering/`)
**Status:** implemented, tested, staged-by-orchestrator (this agent did NOT `git add`/commit/push)
**Compute:** dev box only. No pod SSH, no GPU.

---

## 0. The hole, stated exactly

`CLAUDE.md §Invariants` declares the parity corpus SACRED. The enforcement was a **path substring
match** and nothing else.

**MEASURED** (`stack/scripts/train_flagship_v4.py:566-584` @ `f57ce2a`) — the entire pre-fix check:

```python
tc = str(Path(train_cache).resolve()).replace("\\", "/")
if PARITY_KEY not in tc:
    raise SystemExit("PARITY VIOLATION: ...")
# The skip-hash is a property of that build key, not of an on-disk sidecar in
# the split dir; the loop consumes the FULL split (every ep_*.pt, no --episodes
# subsetting knob exists), so episode re-selection is structurally impossible.
```

The comment is the bug. "The loop consumes the FULL split" is true and is precisely why a
**truncated** split is invisible: the loop consumes every `ep_*.pt` that is *there*. The
`/workspace` MooseFS quota — which `df` does **not** show, and which has already killed a build in
this program — stops a build mid-way and leaves a correctly-**named** directory with fewer episodes.

**MEASURED, red-before-green** (`tests/test_parity_manifest.py::test_flagship_v4_assert_parity_refuses_truncation`,
run against `f57ce2a` code): a synthetic cache holding **1 200 of 2 376** episodes in a directory named
`physicalai-train-e438721ae894` → `Failed: DID NOT RAISE`. It passed. Silently.

The three other loader families did not even reach a check: they went straight into `torch.load`
(`RuntimeError: mmap can only be used with files saved with torch.save(...)` — i.e. the guard never
existed, the *file contents* were the first thing that objected).

### Nuance the brief asked me to verify, not assume

> *"the 'refuses to launch unless the build reproduces the key' guarantee reportedly holds for the
> REF-B shell pipeline but NOT for the Python flagship trainers."*

**CONFIRMED, with a correction to how strong the shell guarantee is.**

- `stack/scripts/pod_ops/refb_pipeline.sh:15,98-100` does gate on `EXPECT_KEY=physicalai-train-e438721ae894`
  and calls out to `/workspace/parity_skipset.sh`; `stack/scripts/pod_ops/compute_skipset.py:87` is a
  **real** count+hash gate — `counts_ok = (n_built == 2376 and skips == 24)` and
  `hash_ok = sha256(sorted skip clip ids) == f09e44db0004…85457`, exit 3 → `PARITY_HOLD`.
- **But** it is a *build-time, pod-side, shell-only* gate. It runs when the REF-B pipeline builds the
  cache. It does **not** run when a trainer is launched against a cache that already exists — which is
  every launch after the first, every restart, every cross-pod move, and every other arm.
- It also depends on `/workspace/parity_skipset.sh` being deployed (`SCRIPT = "/workspace/parity_skipset.sh"`,
  `compute_skipset.py:29`) with a self-contained backstop that reads `r0/r0_selection.parquet` — both
  pod-side. Nothing in this repo can run it.

So the guarantee was: **one arm, one moment, on one pod.** Every Python trainer, on every launch, had
the substring match or nothing.

---

## 1. Per-trainer audit (MEASURED — state at `f57ce2a`, before this work)

`BEFORE` = what the trainer actually asserted about the parity corpus at startup.
`AFTER` = what it asserts now. All `AFTER` checks run **before any model reaches the GPU**.

| # | Trainer / entry point | Reads parity corpus via | BEFORE (MEASURED) | AFTER |
|---|---|---|---|---|
| 1 | `scripts/train_flagship_v4.py` | `_assert_parity` (`:566`) then `glob("ep_*.pt")` (`:832`) | **path substring only**; comment claimed re-selection "impossible" | `parity.assert_parity_corpus(require=True)` on train **+ val**; count + uid-sha256; provenance into `config.json` |
| 2 | `scripts/train_flagship4b.py` | `_cache_split` (`:150`) | **NOTHING** — `glob` → `load_episode` | guard in `_cache_split`, strict (subset when `--episodes N`) |
| 3 | `scripts/refb_train.py` | `load_cached_episodes` (`:189`) | **NOTHING** | guard in the loader, strict/subset |
| 4 | `scripts/refc_train.py` | imports #3's loader (`:63`) | **NOTHING** (inherited) | inherited from #3 |
| 5 | `scripts/refa_train.py` | `load_feature_episodes` (`:113`) | **NOTHING** | guard, **subset mode** (DINO dirs are a `--train-n` prefix by construction) |
| 6 | `scripts/refa_train4b.py` | imports #5's loader (`:59`) | **NOTHING** (inherited) | inherited from #5 |
| 7 | `scripts/train_flagship_v15.py` | `V15Dataset` `eids` from label caches | only "the 3 caches agree with **each other**" (`:93`) | `assert_eids_parity` vs the manifest; train/val keys passed explicitly |
| 8 | `scripts/train_flagship_v16.py` | `V16FramesDataset` (`eids` + epcache dir) | `os.path.exists` on **the first 5 eids** (`:123`) — a 1 200-ep cache passes it | `assert_parity_corpus` on the cache dir (key auto-detected from the path) **+** `assert_eids_parity` on the label eids |
| 9 | `scripts/refc_v12_train.py` | distilled `sh_*.pt` shard cache + `manifest.json` | **NOTHING** | `check_shard_cache_parity`: recorded `src` must be a parity key and `episodes` must equal 2 376 |
| 10 | `scripts/refc_v12_cache.py` (the shard builder) | imports #3's loader (`:63`) | **NOTHING** (inherited) | inherited from #3 |
| 11 | `scripts/finetune_traj.py` | `load_cached_episodes` (`:169`) | **NOTHING** | `load_parity_cache` per resolved split dir |
| 12 | `scripts/train_dynamics_encoder.py` | `--pai-cache` (SIDE model) | **prose only** — docstring `:22-24` claims it "never reads the WM parity key"; nothing enforced it | `assert_side_model_firewall` — the firewall is now an assertion (`ParityViolation` if pointed at parity) |
| 13 | `tanitad/train/train_worldmodel.py` `--data cached` | `mk()` (`:132`) — the bake-off arm path, a **third** copy of the same glob | **NOTHING** | guard inside `mk()` |
| 14 | `scripts/v15_prep.py` | `os.listdir(--cache)` (`:94`) — the **mint point** for every v1.5/v1.6/v4 label cache | **NOTHING** | guard on `--cache`; a truncated epcache can no longer mint a self-consistent short label cache |
| 15 | `scripts/run_idm_proof.py` `select_episodes` (`:201`) — shared by `run_idm_ft`, `run_idm_parity_validation`, `run_idm_pipeline_derisk`, `run_idm_downstream_ablation`, `run_v1_encoder_char`, `run_branchb_transfer`, `run_camcond_ablation` | `Path(pai_cache)/f"ep_{idx:05d}.pt"` | **NOTHING**, and worse: `if not p.exists(): continue` **absorbs** truncation by design | guard on the cache dir before selection (one call covers all 7 probes) |

**Summary of the BEFORE column: 1 of 15 entry points had any parity check at all, and that one check
could not detect a truncated corpus.** Nothing recomputed a skip-hash. Nothing asserted 2 376.

Not in scope (eval-side, or the sibling agents' territory), listed so the gap is visible rather than
forgotten: `eval_flagship_v4.py`, `eval_flagship_v15/16.py`, `evaluate_checkpoint.py`,
`eval_behavior.py`, `compare_arms.py`, `driving_diagnostic.py`, `d1_probe_capacity.py`,
`d3_decompose.py`, `run_spectral.py`, `geom_sanity.py`, `resolution_probe.py`,
`eval_grounded_rollout_4b.py`, `eval_metric_rollout.py` — all glob a val cache with no integrity
check. See §7.

---

## 2. What I implemented

### 2.1 One shared module — `stack/tanitad/data/parity.py` (new)

Not copy-pasted into 15 call sites (the codebase already carries a 4× copy-pasted window class; this
would have been worse). Public surface:

| Function | Used by |
|---|---|
| `assert_parity_corpus(cache_dir, *, label, require, mode)` | every dir-based loader |
| `assert_eids_parity(eids, *, label, corpus_key, mode)` | v1.5 / v1.6 (no dir to glob) |
| `assert_not_parity(*paths, label)` | the dynamics-encoder SIDE firewall |
| `resolve_split_dir` / `scan_cache_dir` / `scan_skip_markers` / `corpus_key_of` | shared plumbing |
| `check_uids` / `uid_digest` / `load_manifest` / `manifest_entry` / `build_entry` | the check + the manifest |
| `ParityViolation(SystemExit)` | the refusal type |

**Checks performed** when a path references a registered parity key:

1. `len(episodes) == manifest.episode_count`
2. `sha256("\n".join(sorted(uids))) == manifest.episode_uid_sha256`
3. the known-**leaky** split `physicalai-val-f1b378f295ae` is refused outright, naming the clean
   replacement (`physicalai-val-0c5f7dac3b11`) — MODEL_REGISTRY §Branch-B, corrected 2026-07-25:
   62 of its 79 populated episodes (78.5 %) are IN the parity train set

**Two modes.** `strict` = set equality. `subset` = the loaded set must be the **sorted prefix** of the
manifest set — what a declared `[:n]` knob produces (`dino_precompute --train-n`, `--episodes N`). Subset
mode still refuses foreign / renumbered / substituted caches, and prints the shortfall LOUD
(`⚠ SUBSET — 400 of 2376 … NOT strict parity and must not be cross-compared with full-corpus arms`).

**Deliberate escape hatch, deliberately visible.** A path referencing no registered key is not an
error — toy, comma2k19 and the v2 9 000-clip corpus (`4b7eeeac222d`) must keep working. It prints one
`[parity] ⚠ NON-PARITY` line and returns `parity=False`. Only `require=True` (which
`train_flagship_v4` has always used) refuses. **There is no environment variable that turns the
content check off** — the opt-out is per-call and shows up in the trainer's own argv.

**Failure is loud, early and actionable.** Real output (MEASURED, this dev box):

```
==============================================================================
PARITY VIOLATION [--train-cache] — corpus physicalai-train-e438721ae894
==============================================================================
  cache      : …/bad/physicalai-train-e438721ae894
  episodes   : 1200 loaded, 2376 expected   <-- TRUNCATED by 1176
  missing    : ep_01200.pt, ep_01201.pt, … (+1170 more)
  extra      : (none)

  The canonical corpus is SACRED (CLAUDE.md §Invariants). A truncated or
  re-selected episode set breaks cross-arm comparability INVISIBLY — every
  number produced against it is void. Refusing to train.

  Most common cause: the /workspace MooseFS quota filled mid-build. `df` does
  NOT show that quota — verify with a real dd write test, not df.
  Rebuild:  scripts/rebuild_pai_rolling.py --expect-key e438721ae894 --skip-idx …
  Re-verify: scripts/pod_ops/compute_skipset.py   (hash + count verdict)
  If (and only if) the cache is verified good and the manifest is stale, re-record
  it with scripts/make_parity_manifest.py --record and commit the diff.
==============================================================================
```

and the success path:

```
[parity] --train-cache: physicalai-train-e438721ae894 VERIFIED — 2376 episodes,
uid sha256 9877bef64da3… matches the committed manifest (skip-hash f09e44db).
```

### 2.2 The committed manifest — `stack/tanitad/data/parity_manifest.json` (new, 60 KB)

```
train  physicalai-train-e438721ae894 : 2376 episodes
       episode_uid_sha256 = 9877bef64da35f384b380b23ab0e760f3ef5396c6f3e849d5de81c7243ac7386
       skip_indices       = 24 × [1798, 1835, 1841 … 1898, 1941]
val    physicalai-val-0c5f7dac3b11   : 600 episodes, episode_uid_sha256 = null (COUNT-ONLY)
```

**Episode identity = the `ep_%05d.pt` basename.** `tanitad/data/epcache.py:104-110` writes
`ep_{i:05d}.pt` where `i` is the position in the **ordered** source list, and `skip_{i:05d}` for a clip
that failed to build. The index is therefore the stable identity *within a build key*, and the build
key `e438721ae894` is itself a hash of the ordered clip ids + build params (`epcache.cache_key`).
Every derived cache preserves it: `dino_precompute.py:95` (`o = dst / f.name`) and the v15/v16 `eids`
lists are the same strings, so one uid space covers the epcache, the DINO feature dirs and the label
caches.

### 2.3 Which path I took for the manifest — and what I did NOT do

I could not read the pod-side corpus (no pod access). **I did not invent or hardcode any hash.**
Here is exactly where the train entry comes from.

**Source (MEASURED):** `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-25-v2-corpus-qa/parity_profile.csv`
— a committed, READ-ONLY pod-side scan of the canonical cache dir
(`parity_profile.json`: `cache_dir = /workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894`,
`episodes: 2376`, `ok: 2376`, `bad: []`). 2 376 rows, one `ep_*.pt` per row, all unique.

**Three independent cross-checks, all enforced in code** (`make_parity_manifest.py:from_profile_csv`
raises `REFUSING to write` if any fails — verified by
`test_generator_cross_checks_reject_a_tampered_profile`):

| Cross-check | Result |
|---|---|
| the 24 indices absent from the 2 376 must be the skipset, endpoints matching `rebuild_pai_rolling.py:32`'s independently-written `--skip-idx 1798,1835,…,1941` | **1798..1941 MATCH**, exactly 24 |
| `sum(T_out)` over the 2 376 rows vs `total_frames` in `2026-07-24-parity-corpus-profile/corpus_profile.json` — scanned on a **different pod, different path** (`tanitad-pod3:/workspace/pai_epcache/…`) | **472 627 == 472 627 MATCH** |
| the count vs `n_episodes` in `2026-07-22-v4-labels/labels_train_v4_provenance.json` | **2 376 MATCH** |

**The val entry is COUNT-ONLY on purpose.** 600 is MEASURED (`labels_val_v4_provenance.json`
`n_episodes`; `scripts/parity_skipset.sh` asserts `len(val) == 600`). But **no committed artifact
enumerates the val uid set.** I could have *derived* one — the val split has 0 skips, so the uids are
almost certainly `ep_00000.pt … ep_00599.pt` — and I deliberately did not: that is a HYPOTHESIS, and
a false refusal on a 30 k launch is not a cheap error. The count check alone already refuses a
truncated val cache, which is the failure this workstream closes; the uid digest additionally refuses
a *substituted* set of the right size, and is a one-command upgrade (§3).

**What the manifest does NOT cover** (stated in the file's own `limitations` field): it pins **which
episode slots are present**. It does not hash episode *content*, so a same-named file with different
tensor bytes is not caught here — that is the build key's and `compute_skipset.py`'s job.

### 2.4 `--parity-manifest` mechanism + self-recording

`stack/scripts/make_parity_manifest.py` (new):

- `--from-profile-csv` — regenerate the train entry on the dev box (what produced the committed file)
- `--record --cache-dir <dir> --split {train,val}` — the **self-recording** path, pod-side
- `--verify --cache-dir <dir>` — check a live cache, non-zero exit on violation, no writes
- **`--record` refuses to overwrite an entry that already has a uid digest unless `--force`** — a
  truncated cache must never be able to quietly re-record itself into a passing manifest
  (`test_generator_refuses_to_overwrite_a_recorded_digest`)

Every guard call also takes `manifest_path=…`, so an alternative manifest can be threaded through
without touching the module.

### 2.5 The `filtering.py` skipset (brief item 4)

`stack/tanitad/lake/filtering.py:66` `CORRUPT_SKIPSET["physicalai_av"]` is still an empty set of
**clip ids** — and that is now *documented precisely* rather than left as an unexplained blank, plus a
half-fix:

- **HAVE (new):** `filtering.PARITY_SKIP_INDICES` — the 24 skip **positions**, read from the committed
  manifest (one source of truth, not re-typed). The skipset is now reproducible from this repo at
  **index level**.
- **MISSING:** the ordered 2 400-entry train **clip-id** list that turns a position into a UUID. It
  derives from `<root>/r0/r0_selection.parquet` (3 000 rows) + the seed-0 `torch.randperm` 80/20 split,
  and lives pod-side with the **gated** PhysicalAI-AV dataset.
- **Why it stays missing:** PhysicalAI-AV is tier `firewalled` / `gated-confidential`
  (`filtering.tier_of`, §1 of that file) — recipe-only, never content. Committing the UUIDs into
  `stack/` would contradict the licence axis the lake is built on. This is a **licence** boundary, not
  an oversight, and it is now written down at the site.
- Consequence, stated honestly: `f09e44db…` (a sha256 over clip **ids**) is **not** reproducible from
  this repo and will not be. The trainers' integrity check keys on the **index** level instead, which
  is what makes the gap tolerable. Regenerating the clip-id hash is one pod command:
  `bash scripts/parity_skipset.sh` or `python scripts/pod_ops/compute_skipset.py`.

---

## 3. ⚠️ ONE ACTION REQUIRED ON A POD (before the next v1.5/v1.6/v4 val run)

The val entry is count-only. To upgrade it to a content check, on a pod where
`scripts/pod_ops/compute_skipset.py` has just printed **`VERDICT MATCH`**:

```bash
PYTHONPATH=/workspace/TanitAD/stack python3 \
  stack/scripts/make_parity_manifest.py --record --split val \
  --cache-dir /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11
# then bring the changed stack/tanitad/data/parity_manifest.json back to the repo and stage it
```

Optional but recommended, and free: re-record the **train** entry the same way (`--split train
--force`) from a `VERDICT MATCH` cache. If the recorded digest differs from
`9877bef64da3…`, the derivation in §2.3 was wrong and the divergence is itself the finding — the
`--force` guard exists so that this is a deliberate, reviewed act.

Also free, and worth wiring into `refb_pipeline.sh` / any rebuild: post-build verification is now one
command that needs no GPU and no pod-side script deployment —
`python3 stack/scripts/make_parity_manifest.py --verify --cache-dir <split dir>`.

---

## 4. Test evidence — RED → GREEN

New file: `stack/tests/test_parity_manifest.py`, **30 tests**.

### RED (guard module present, trainers NOT yet wired)

```
7 failed, 18 passed in 2.99s
FAILED  test_flagship_v4_assert_parity_refuses_truncation
FAILED  test_flagship4b_cache_split_refuses_truncation
FAILED  test_refb_loader_refuses_truncation
FAILED  test_refa_feature_loader_refuses_foreign_episodes
FAILED  test_finetune_traj_loader_refuses_truncation
FAILED  test_v15_v16_label_caches_check_their_eid_lists
FAILED  test_dynamics_encoder_keeps_its_parity_firewall
```

with the two diagnostic failure shapes that *are* the audit finding:

- `train_flagship_v4._assert_parity(<1200-of-2376 cache>)` → **`Failed: DID NOT RAISE ParityViolation`**
  — the old substring check passed a truncated corpus.
- `train_flagship4b._cache_split` / `refb_train.load_cached_episodes` /
  `refa_train.load_feature_episodes` → `RuntimeError: mmap can only be used with files saved with
  torch.save(…)` — no check existed at all; execution reached `torch.load` on the truncated cache.

The fixture deliberately uses **empty** `ep_*.pt` marker files, so "the guard raised" and "the guard
ran *before* any tensor was unpickled" are the same assertion.

### GREEN (after wiring)

```
tests/test_parity_manifest.py .............................. [100%]
30 passed in 3.3s
```

`test_the_old_substring_check_would_have_PASSED_the_truncated_cache` stays in the suite permanently
as the premise pin: it asserts `PARITY_KEY in str(truncated_dir)` — i.e. the entire old enforcement,
satisfied by the bad cache.

### Full suite

| | result |
|---|---|
| baseline @ `f57ce2a` (before any edit) | **837 passed, 3 skipped**, 84.3 s |
| after Wave-1 B | **867 passed, 3 skipped**, 69.6 s |

**+30 tests, 0 regressions, green.** All 22 edited/related scripts also import cleanly (`0 FAILURES`
on an explicit `importlib` sweep) — a syntax or import error in a trainer would otherwise only surface
at launch.

### Test coverage map

| Test | Guards against |
|---|---|
| `test_truncated_cache_is_refused` ⭐ | the headline: 1 200 of 2 376 in a correctly-named dir |
| `test_substituted_episode_set_of_the_right_size_is_refused` | right count, wrong episodes (count alone is not enough) |
| `test_extra_episodes_are_refused` / `test_empty_split_dir_is_refused` | over-full and empty caches |
| `test_manifest_is_self_consistent` / `…reproduces_the_24_clip_skipset` | 2 376 + 24 tile the 2 400 sources; endpoints 1798/1941 |
| `test_val_entry_is_count_only_and_says_so` | the honesty invariant — no invented val hash |
| `test_refusal_names_the_rebuild_and_reverify_commands` | a 3 a.m. refusal must say what to do (`MooseFS`, both commands) |
| `test_subset_mode_accepts_the_canonical_prefix_but_not_foreign_ids` | REF-A prefix dirs stay legal; foreign ids don't |
| `test_non_parity_corpus_warns_but_does_not_block` / `…is_refused_when_the_caller_requires_parity` | toy/comma/v2 keep working; v4 still hard-requires |
| `test_known_leaky_val_split_is_always_refused` | `physicalai-val-f1b378f295ae` |
| `test_parity_firewall_blocks_a_side_model_from_the_parity_corpus` | the dyn-encoder firewall |
| 7 per-trainer wiring tests + `test_train_worldmodel_cached_path…`, `test_idm_probe_family…`, `test_refc_v12_shard_cache…` | each family individually |
| `test_missing_split_dir_stays_an_AssertionError_not_a_refusal` | see §5 |
| `test_lake_filtering_skipset_is_index_reproducible_from_the_repo` | §2.5 |
| `test_generator_refuses_to_overwrite_a_recorded_digest` / `…cross_checks_reject_a_tampered_profile` | the manifest cannot be laundered |

---

## 5. One near-miss worth recording (root-cause class: *"a guard that fires in the wrong failure mode
costs a finished run"*)

My first `resolve_split_dir` raised `ParityViolation` (a `SystemExit`) when **no** split dir matched.
`refa_train.py:493` and `refb_train.py:557` deliberately `except AssertionError` around their
**optional** val-metrics block. That change would have converted "no val dir" — a supported
configuration — into a hard exit **at the metrics write, after a finished 30 k run**. The suite was
green either way; only reading the catch sites caught it. `resolve_split_dir` now raises
`AssertionError` with the original message, and `test_missing_split_dir_stays_an_AssertionError_not_a_refusal`
pins it. *"You gave me no val dir" is not a parity violation.*

---

## 6. Deliverable manifest

All paths relative to the repo root. **Nothing was `git add`ed, committed or pushed** — the
orchestrator stages. Nothing lives on a pod or in a worktree.

**New**

| Path | What |
|---|---|
| `stack/tanitad/data/parity.py` | the shared guard (the only place the logic exists) |
| `stack/tanitad/data/parity_manifest.json` | the committed episode manifest (60 KB) |
| `stack/scripts/make_parity_manifest.py` | generate / `--record` / `--verify` |
| `stack/tests/test_parity_manifest.py` | 30 tests |
| `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-25-wave1-parity-integrity/WAVE1_B_REPORT.md` | this report |

**Modified** (all inside `stack/`)

`scripts/train_flagship_v4.py` · `scripts/train_flagship4b.py` · `scripts/train_flagship_v15.py` ·
`scripts/train_flagship_v16.py` · `scripts/train_dynamics_encoder.py` · `scripts/refa_train.py` ·
`scripts/refb_train.py` · `scripts/refc_v12_train.py` · `scripts/finetune_traj.py` ·
`scripts/v15_prep.py` · `scripts/run_idm_proof.py` · `tanitad/train/train_worldmodel.py` ·
`tanitad/lake/filtering.py`

(`refc_train.py`, `refa_train4b.py`, `refc_v12_cache.py` and the six other `run_idm_*`/`run_*` probes
are covered through the loaders they import — no edit needed, verified by import sweep.)

---

## 7. Left open — escalated, not filed in a README

1. **⚠️ Pod action (§3).** The val uid digest requires one `--record` run on a pod. Until then val is
   count-only. **This is the only thing between here and a full content check on both splits.**
2. **Eval-side has the identical hole.** 13 evaluators glob a val cache with no integrity check
   (§1, "not in scope"). A truncated *val* cache silently changes every published ADE the same way a
   truncated train cache changes every arm. `taniteval/` already hard-refuses the leaky split
   (`data.list_val_episodes(..., allow_leaky=False)`) but does not check counts. **This needs an owner
   — it is the mirror image of what Wave-1 B just closed, and it touches the published numbers
   directly.** I did not touch `taniteval/` (sibling agent held it this wave).
3. **Builders should self-verify.** `rebuild_pai_rolling.py` checks `cache_key == --expect-key` but
   never re-counts after the build. Adding
   `make_parity_manifest.py --verify --cache-dir <dir>` as its last step (and as the last step of
   `refb_pipeline.sh`) would catch a quota death at the moment it happens rather than at the next
   launch. Cheap; not done here to keep the diff to the audited surface.
4. **Content-level integrity is still unproven from the repo.** The manifest pins *which* episodes,
   not *what is in them*. The build key covers the build; nothing re-verifies tensor bytes after a
   cross-pod HF relay (`md5` is verified for the transfer, not for the cache). A per-episode
   `sha256` sidecar written at build time would close it — a real cost (2 376 hashes over ~260 GB),
   worth pricing before committing to.
5. **`compute_skipset.py` is pod-side and unversioned in its dependency.** It shells out to
   `/workspace/parity_skipset.sh` and falls back to a backstop that reads
   `r0/r0_selection.parquet`. Both are pod-local. If a pod is lost, the clip-id hash becomes
   unreproducible — the same reconstruction risk the registry already flags for TanitEval.

---

## 8. Evidence classes used

| Claim | Class |
|---|---|
| pre-fix per-trainer state (§1 BEFORE column) | **MEASURED** — file:line at `f57ce2a` + the RED test run |
| 2 376 episodes / the 2 376 uids / the 24 skip indices | **MEASURED** — committed pod scan `parity_profile.csv`, triple cross-checked (§2.3) |
| val = 600 episodes | **MEASURED** — `labels_val_v4_provenance.json`, `parity_skipset.sh` |
| val uids = `ep_00000..ep_00599` | **HYPOTHESIS** — deliberately NOT committed (§2.3) |
| `f09e44db…` = sha256 over sorted skip clip ids | **MEASURED** — `parity_skipset.sh`, `compute_skipset.py:EXPECT_HASH` |
| `physicalai-val-f1b378f295ae` is leaked (78.5 %) | **INHERITED** — MODEL_REGISTRY §Branch-B, corrected 2026-07-25; not re-verified here (no pod). Used only to *refuse*, never to license a number |
| test results (837→867, RED 7/18 → GREEN 30) | **MEASURED** — this dev box, `C:/Users/Admin/venvs/tanitad` (py 3.13.5, torch 2.11.0, pytest 9.1.1) |
