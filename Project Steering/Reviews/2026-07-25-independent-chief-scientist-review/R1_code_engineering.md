# R1 — Code & Engineering Audit (Independent)

**Auditor role:** independent senior ML-systems engineer (adversarial, read-only).
**Date:** 2026-07-25 · **Scope:** `stack/`, `taniteval/`, CI/packaging, `CLAUDE.md` traps.
**Method:** primary sources only — actual code read at file:line; `pytest --collect-only`
run on both suites with the project venv (`C:/Users/Admin/venvs/tanitad`). No pods touched,
no training/eval run, nothing staged.

Evidence classes: **MEASURED** (I read it / ran it) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.
Every severity-bearing claim below is MEASURED unless tagged otherwise.

---

## 1. Executive verdict

The codebase is **bimodal**. The *library core* — `taniteval/ci.py`, `stack/tanitad/data/_contract.py`,
`epcache.py`, `ckpt_io.py`, the leakage guards, the fail-loud contracts — is genuinely strong,
correctly reasoned, and unusually well-tested (the CI estimator has a test file that pins the exact
statistical defect it replaced). The *trainer/orchestration layer* is the opposite: **13 trainer
scripts totalling ~7,861 LOC** with scripts importing scripts, a core window/label class
**copy-pasted into four files** and imported by the flagship *from a reference-arm script*, the
multi-day training **loops and checkpoint-resume paths untested**, **no training determinism** beyond
`manual_seed`, and **parity "enforced" by a directory-name substring** rather than a content hash.
Two numerical-rigor gaps touch decisions: the closed-loop headline CIs still use the *deprecated*
estimator the program retired for being 1.28–2.06× too narrow, while the corrected one sits 40 lines
away in the same file. Nothing here is fraud or a smoking-gun correctness bug in the shipped metric —
the shipped `ci.py`, windowing alignment, and leakage guards are correct — but the reproducibility and
trainer-hygiene story is well below the rigor the rest of the program holds itself to.

**Overall engineering-maturity grade: B− (library core A−/B+; trainer + reproducibility layer C).**

---

## 2. Findings

### F1 — Parity is enforced by path-substring, not content; keys are unreproducible from the repo · **HIGH** · confidence HIGH
`stack/scripts/train_flagship_v4.py:566-584` `_assert_parity()` is the flagship trainer's entire
parity check, and it is:
```
tc = str(Path(train_cache).resolve())...
if PARITY_KEY not in tc:            # line 574 — SUBSTRING match on the path
    raise SystemExit("PARITY VIOLATION ...")
```
It verifies only that the string `physicalai-train-e438721ae894` appears in the `--train-cache`
path. It never counts episodes (no `== 2376` assertion exists in any trainer —
MEASURED: `grep 2376 stack/**` returns only comments/docstrings) and never recomputes the skip-hash.
The comment at :579-581 explicitly rationalises this ("episode re-selection is structurally
impossible"), but a **truncated** corpus is not a re-selected one: a build that stopped early at,
say, 2,000 episodes (the exact MooseFS-quota failure mode CLAUDE.md warns about, and which
`epcache.build_episodes_cached` tolerates per-source at `epcache.py:117-128`) lands in a
correctly-named dir and **passes**. Cross-arm comparability — the stated foundation of every H1/H4
verdict — then silently rests on mismatched corpora.

Compounding it: the parity keys **cannot be reproduced or verified from the repo at all.**
`stack/tanitad/lake/filtering.py:66` ships `CORRUPT_SKIPSET = {"physicalai_av": set()}` — **empty** —
with the real 24 clip-ids "pod-side with the gated dataset (not committed)". The one *real* content
gate, `stack/scripts/pod_ops/compute_skipset.py` (sha256 over sorted skip-ids, cross-checks
`built==2376 && skips==24`, `compute_skipset.py:85-91`), depends on `/workspace/parity_skipset.sh`
and `r0/r0_selection.parquet` — **neither committed**. So `e438721ae894`/`f09e44db` are opaque
constants off-pod; the REF-B shell pipeline (`refb_pipeline.sh:97-135`) gates hard on them, but the
Python flagship trainer does not, and no in-repo test can exercise the gate.

*Why it matters:* the gated data legitimately can't live in the repo, but an **episode-id manifest +
content hash** could, and that would turn the substring check into a real one. Today, the strongest
reproducibility claim in the program ("parity is sacred") is, in the committed code, a `str in str`.

### F2 — Closed-loop headline / compounding / divergence CIs use the DEPRECATED estimator · **HIGH** · confidence HIGH
`taniteval/taniteval/closedloop.py` reports its headline closed-loop ADE/FDE, the compounding-error
delta (closed−open — the program's stated "distribution-shift cost the open-loop ADE hides"), the
divergence rate, and the speed-stratified drift **all through the deprecated overlapping-holdout
estimator**:
- `_agg()` :382-390 and `_jack()` :393-405 are literally `1.96*std/sqrt(n_splits)` over 8 overlapping
  random 20% holdouts — the construction `ci.py` documents as **1.28–2.06× too narrow** and the
  program formally retired.
- Used for the headline block at :462 (`suite_ci`→`_agg`), the compounding delta at :483-486, and the
  divergence/stability `separated` claims at :495 via `_jack`'s `separated = abs(mean)-ci>0` (:405).
- The **correct** `ci.episode_cluster_bootstrap` is imported and used **only** for the imagination
  A-vs-B arm (:528-575).

This is not a hidden bug — the protocol dict self-labels it DEPRECATED (:606-608) — but the *number a
reader quotes* ("closed_bike ade@2s = X ± ci95", printed at :689; "closed−open Δ@2s separated") is on
the retired estimator. Contrast `taniteval/driving.py`, which **was** migrated (uses
`episode_cluster_bootstrap`, :453/:551, MEASURED). Closed-loop is, by the program's own memory, the
*more* decision-relevant axis ("open-loop ADE does not predict closed-loop"), yet it kept the weaker
estimator. Fix is nearly free — the right function is in the same process.

### F3 — Trainer sprawl, a 4× copy-pasted window class, and flagship←reference coupling · **HIGH** (debt) · confidence HIGH
MEASURED (`wc -l`): 13 trainers, 7,861 LOC — `train_flagship4b.py` 631, `train_flagship_v4.py` **1396**,
`train_flagship_v15.py` 412, `train_flagship_v16.py` 648, `refa_train.py` 538, `refa_train4b.py` 459,
`refb_train.py` 601, `refc_train.py` 594, `refc_v12_train.py` 383, plus `experiments/*` copies.
- **The core window/label producer is duplicated four times.** `class FailLoudWindowDataset(
  EpisodeWindowDataset)` is defined independently in `scripts/refb_train.py:121`,
  `experiments/refb-v2/refb_train_v2.py:121`, `.../refb_train_v3.py:124`, `.../refb_train_v4.py:124`.
- **The flagship imports its data contract from a reference arm.**
  `scripts/train_flagship4b.py:53` `from refb_train import FailLoudWindowDataset`. The canonical arm's
  window/label emission (pose_last, future_poses, nav/maneuver fields) is thus *defined in REF-B's
  trainer script*; any edit there (as the v2/v3/v4 forks already do) can silently move the flagship's
  data contract. The flagship also stitches `cosine_lr` from `train_worldmodel`, `start_cache_guard`
  from `finetune_traj`, and `refb_labels` — a web of `sys.path` script-to-script imports, not a library.
- **`train_flagship_v4.py` contains a full `_training_loop` (:591-) and `load_checkpoint_v4` (:544)
  whose own module docstring says the loop "is NOT launched here ... reuses `train_flagship4b.py` /
  `train_flagship_v16.py` machinery" (:4-8).** So a 1,396-line trainer carries an ambiguous/duplicate
  loop that is either dead or a fourth divergent copy of the loop logic. Either way it is a
  maintenance and correctness hazard.

The *good* news, and why this is debt not a correctness bug: the **base** windowing
(`tanitad/data/_contract.py:104-139`) is single-sourced and its alignment is **correct** —
`pose_last = poses[t+w-1]`, `future_poses = poses[t+w : t+w+H]`, episode-level split, no leakage
(MEASURED). The risk is divergence *among the four subclasses*, not the base.

### F4 — No training determinism beyond `manual_seed` · **MEDIUM** · confidence HIGH
Every trainer sets `torch.manual_seed(args.seed)` (`train_flagship4b.py:230`, `refb_train.py:384`,
`refc_train.py:360`, `train_flagship_v16.py:455`, `train_flagship_v4.py:774`) but **none** set
`torch.use_deterministic_algorithms`, `cudnn.deterministic/benchmark`, or a DataLoader
`worker_init_fn`/`generator` (MEASURED: grep across all five trainers returns only `manual_seed`
and the mixing-dataset seed). Consequences: (a) DataLoader worker RNG (any augmentation/sampling
inside workers) is unseeded → batch order/content not reproducible across runs; (b) cuDNN
nondeterminism → a re-run of the "same" config does not reproduce the loss curve bit-for-bit. For a
program that makes GPU-day continue/restart calls off small ADE deltas and treats parity as sacred,
non-reproducible *training* is a reproducibility hole as real as the data one. `MixedWindowDataset`
takes a seed (`mixing.py`), which is necessary but not sufficient.

### F5 — The most expensive-to-fail paths are the least tested · **MEDIUM-HIGH** · confidence HIGH
Collection is clean: **839 tests (stack) + 153 (taniteval)**, no import/collection errors (MEASURED).
Coverage is strong on models, losses, labels, lake, metrics, and `ci.py`. But it is thin exactly where
a silent failure costs days:
- **No test exercises any multi-day training loop or checkpoint-resume.** `load_checkpoint_v4`
  (`train_flagship_v4.py:544`), the `_training_loop`, and the canary controller's step-through have no
  test (MEASURED: grep for `load_checkpoint|_training_loop|resume` over `stack/tests` +
  `taniteval/tests` hits only the `.pyc` of `test_ckpt_io`, which tests only `atomic_archive`, not
  resume). Only `train_worldmodel.train` has a smoke (`test_smoke_train.py`).
- **Three trainers have zero test files importing them**: `train_flagship_v15`, `train_flagship_v16`,
  `refc_v12_train` (MEASURED per-trainer grep). `train_flagship_v16` is the checkpoint/canary machinery
  the v4 docstring says the real runs *reuse*.
- **The parity guard `_assert_parity` is untested** — no test constructs a truncated/misnamed cache
  and asserts refusal (which would also have surfaced F1).
`test_train_flagship_v4.py` is good where it exists (smoke finiteness across curriculum phases,
preflight asserts, from-scratch freeze gate) — but it validates *assembly*, never the *loop*.

### F6 — Feature-flag combinatorics concentrated in one loss function · **MEDIUM** · confidence HIGH
`flagship_loss` (`stack/tanitad/train/flagship_losses.py:167-427`) branches on ~12 independent
`getattr(cfg,"v2_*",…)`/lever switches in a single body: `v2_ego_to_planners` (:201),
`v2_ego_dropout` (:214), `speed_input` (:227), `v2_nav_dropout` (:240), `v2_anchor_tactical` (:283),
`goal_traj_head` (:297), `v2_traj_jerk` (:306), `v2_route_from_vision` (:338), `v2_invdyn_gradscale`
(:353), `v2_encoder_ego_decorr` (:380), `rollout_k>1` (:258), `v2_fa_dropout` (:261). That is ~2^10
reachable path combinations (MEASURED: 26 distinct `v2_*` flags exist repo-wide); the tests exercise a
handful of named configs. Each lever is individually documented and defaults to a no-op, which is the
mitigating factor — but the interaction surface is effectively untested, and `getattr(...,default)`
guards mean a **typo in a flag name fails silently to the default** rather than erroring.

### F7 — Percentile bootstrap: `mean ± ci95` misreports skewed reducers; coverage only validated for the mean · **LOW-MEDIUM** (numerical rigor) · confidence MEDIUM
`ci.py` is the strongest module in scope and its logic is correct: episode is the resampling unit
(`_draws` :98-104), paired shares the draw (:201-234), fail-loud on empty/mismatched input, provenance
in every dict. Two nuances:
1. It returns raw **percentile** bounds (:187) *and* a symmetric `ci95=(hi-lo)/2` (:192) alongside the
   full-set `mean`. The runner prints `mean ± ci95` (`runner.py:155`). For the **skewed** reducers the
   suite deliberately added — `rms` (:114) and `p90/p10` (:123) — the percentile interval is asymmetric,
   so `mean ± ci95 ≠ [lo,hi]` and the printed half-width misdescribes the actual interval. For the
   `mean` reducer at n_ep=40 this is negligible (near-symmetric by CLT); it bites the tail reducers.
2. `test_ci.py::test_coverage_cluster_vs_naive` validates ~91–98% coverage **only for the mean reducer
   on a symmetric Gaussian random-effect corpus** (:150-176). Coverage of the `rms`/quantile reducers,
   and of a right-skewed per-window error distribution, is unvalidated. No BCa/bias-correction; percentile
   is known to under-cover skewed statistics. `DEFAULT_N_BOOT=2000` is adequate for 95% bounds, not for
   tight tail quantiles. *Recommendation, not a bug:* report `[lo,hi]` (never `mean±ci95`) for non-mean
   reducers, and extend the coverage test to `rms`.

### F8 — Minor / low-severity (grouped)
- **Consistent off-by-one under-utilisation** drops the single last valid window per episode:
  `_contract.py:120-121` `range(frames-window-max_horizon)` excludes the last valid `t`; same pattern
  `rollout.py:76` and `closedloop.py:308`. Harmless (fewer windows), but it means the last GT pose is
  never a target. **LOW.**
- **Closed-loop action-window duplication:** `closedloop.py:193-203` overwrites the window's last action
  with the executed `a_exec` *and then* appends `a_exec`, so after the slide the executed action occupies
  the last two slots and the genuine `a_t` is discarded. Harness-only (does not touch the leaderboard
  rollout), but the duplication looks unintended over K=20 ticks. **LOW — verify against predictor action
  alignment.**
- **Per-batch class weighting** in `_class_weighted_ce` (`flagship_losses.py:111-118`) recomputes CE
  weights from each batch's class counts (clamp 10), injecting batch-composition noise into the
  maneuver/route gradient scale; per-corpus weights would be steadier. **LOW.**
- **Hard-coded pod paths** at import time: `rollout.py:16-17`, `runner.py:31-32`, `closedloop.py:82-84`
  `sys.path.insert(0,"/root/...")`. Harmless off-pod (missing paths are no-ops) but couples the eval
  package to one machine layout; `conftest.py` already solves this centrally and these predate it. **LOW.**

### Credits (measured strengths — do not regress these)
`ci.py` + `test_ci.py` (the defect is pinned with a known-answer synthetic corpus and a coverage
criterion); `_contract.py` windowing alignment; `epcache.py` collision-safe full-path keying with
read-only legacy fallback; `ckpt_io.atomic_archive` (temp+rename, self-healing); the runner leakage
guard that drops train-set episodes from val and *refuses* a decision on <8 clean episodes
(`runner.py:62-76`); `driving.py`'s correct migration to the cluster bootstrap; and the REF-B pipeline's
real content-hash parity gate. These are A-grade.

---

## 3. Engineering-maturity grade — **B−**

**Justification.** The program has genuinely industrial instincts where it has invested: a
single-sourced, correct data contract; a statistically correct, well-tested decision-grade CI
estimator that was *rebuilt from a documented defect* with a coverage test; leakage guards that
fail loud and refuse under-powered decisions; atomic checkpointing. That is A-/B+ work and rare.
But engineering maturity is gated by the weakest link on the path to a shipped number, and three of
those links are C-grade: (1) **reproducibility is asserted, not enforced** — parity by substring
(F1) and no training determinism (F4); (2) **the trainer layer is unconsolidated** — 13 scripts,
a 4×-duplicated core dataset, flagship←reference coupling, and an ambiguous duplicate loop (F3);
(3) **the costliest failure paths are untested** — loops, resume, and the parity guard itself (F5).
The two CI-hygiene gaps (F2, F7) are cheap to close but currently let a retired estimator back the
most decision-relevant axis. Net: **B−**, held down from B by F1/F2/F5 being decision-touching, held
up from C by an unusually strong tested library core.

---

## 4. Concrete proposals (prioritized)

| # | Proposal | Addresses | Payoff | Effort |
|---|----------|-----------|--------|--------|
| **P1** | **Commit an episode-id manifest + content hash** for the parity corpus (ids only, no gated frames) and make `_assert_parity` recompute `sha256(sorted(loaded episode_uids))` and assert `count==2376` + hash match — the same check `compute_skipset.py` does, but in-process for **every** trainer, not just the REF-B shell. Add a test that a truncated/misnamed cache is refused. | F1, F5 | Turns the #1 reproducibility claim from convention into enforcement; catches the known quota-truncation failure before a multi-day run. | **M** |
| **P2** | **Delete `_agg`/`_jack` from `closedloop.py`; route its headline, compounding, divergence and stability blocks through `ci.episode_cluster_bootstrap` / `paired_episode_cluster_bootstrap`** (already imported at :528). Mirror `driving.py`. | F2 | Removes the retired-estimator from the most decision-relevant axis; ~1 hr, no new deps. | **S** |
| **P3** | **Make training reproducible:** add a `seed_everything(seed)` helper (`manual_seed` + `cuda.manual_seed_all` + `use_deterministic_algorithms(True, warn_only=True)` + cudnn flags) and a DataLoader `generator=` + `worker_init_fn` seeded from `seed`. Call it in all five trainers. Log the resolved flags to `config.json`. | F4 | Same-config re-runs become comparable; restart decisions stop riding on RNG noise. | **S-M** |
| **P4** | **Consolidate the window dataset into one library class.** Move `FailLoudWindowDataset` (nav/maneuver/pose_prev fields) into `tanitad/data/` next to `EpisodeWindowDataset`; have the flagship + all REF-B forks import it; delete the 3 experiment copies. Stop `train_flagship4b` importing from `refb_train`. | F3 | Eliminates the flagship←reference coupling and 4-way divergence risk on the exact class that defines cross-arm comparability. | **M** |
| **P5** | **Add a CPU smoke that runs ~10 steps of each real trainer's loop, saves a checkpoint, resumes, and asserts step/opt/controller state continuity.** Parametrize over the trainers; reuse the `smoke_config` fixtures that already exist. | F5 | Covers the untested resume/loop path where a silent corruption costs days; would have caught a stale `load_checkpoint_v4` contract. | **M** |
| **P6** | **Resolve `train_flagship_v4.py`'s loop ambiguity:** either delete `_training_loop`/`load_checkpoint_v4` and import them from the canonical trainer, or update the docstring and make it the single loop. One loop, one owner. | F3 | Removes ~400 LOC of possibly-dead duplicate; kills a divergence source. | **S-M** |
| **P7** | **CI-report hygiene in `ci.py`:** for non-`mean` reducers, print/emit `[lo,hi]` and drop the symmetric `ci95` (or flag it `asym=True`); extend `test_ci.py` coverage to the `rms` reducer and a right-skewed corpus. | F7 | Stops `mean±ci95` misdescribing skewed intervals; validates the tail reducers actually in use. | **S** |
| **P8** | **Add a GitHub Actions (or equivalent) job that runs `pytest --collect-only` + the CPU-only subset** on push. Today there is no `.github/` workflow (MEASURED — none in `git ls-files`); `pyproject.toml` marks slow/sim tests but nothing runs the fast suite automatically. | F5, general | Makes "`pytest -q` must stay green" (CLAUDE.md) an enforced gate, not an honour system. | **S** |

*(The `git commit -- <pathspec>` SEGFAULT and the ~1 MB/s cross-pod relay documented in CLAUDE.md are
real engineering defects but are environment/tooling issues outside the code under audit; P1–P8 are the
in-code levers. The segfault in particular deserves its own root-cause spike — a partial-index commit
crashing is not normal and the current workaround, "whole-index `-F` commit only," weakens the
concurrent-agent safety the same doc demands.)*

---
*End R1. No files staged; orchestrator to stage per the batch protocol.*
