# TanitAD Repo Triage — 2026-07-20

**Scope:** read-only audit of branch/worktree/stash sprawl. No branch was checked out, nothing merged,
nothing deleted, no working tree touched. Every claim below is derived from `git log/show/diff/cat-file/ls-tree`.

**Baseline:** `HEAD = agent/tools-devenv-20260717 @ b57bb47`. 56 local branches, 43 worktrees, 1 stash.

> **Mid-audit update.** HEAD advanced to `547c8ec` while this ran, and one of the new commits is
> `db05620 Merge branch 'main' into agent/tools-devenv-20260717`. **`main` is now 0 ahead of HEAD** — §2's
> recommendation was independently executed during the audit. Every other finding was re-verified against
> `547c8ec` and still holds (`tools/` still empty, the `shutil.copy2` bug still live in all 3 trainers, all
> branch ahead-counts unchanged). Counts below are stated post-merge.

**Headline**

| | count |
|---|---|
| Branches with commits not in HEAD | **25** (26 at baseline; `main` merged mid-audit) |
| → **MERGE** (valuable, not superseded) | **8** |
| → **ARCHIVE** (tag, then delete) | **8** |
| → **DROP** (superseded / already applied) | **9** (was 10; `main` now merged) |
| Branches fully contained in HEAD (0 ahead) | 31 (30 deletable; 1 is HEAD itself) |
| Worktrees safe to prune | **41 of 43** |
| Worktrees **DO-NOT-PRUNE** | **2** (main tree + `agent-a5ab55fd6191cb521`) |
| Stash | **DROP** (contents already live) |

**Three things that would have been genuinely lost**

1. **A live, unfixed ops bug.** `agent/prod-opt-20260718` carries `ckpt_io.py::atomic_archive`, the fix for
   non-atomic gate-milestone archiving. HEAD still has the bare `shutil.copy2` in **all three trainers** —
   `stack/scripts/train_flagship4b.py:417`, `stack/scripts/refb_train.py:375`,
   `stack/experiments/reset-speed4b/refa_train_plus.py:540`. A kill mid-copy leaves a truncated
   `ckpt_step{N}.pt` that the `not arch.exists()` guard then refuses to re-archive — the gate protocol
   later `torch.load`s a corrupt milestone. Given pod2's documented OOM-kill / Errno122-quota history this
   is not hypothetical.
2. **An agent explicitly recommended a merge that never happened.** HEAD's own
   `TanitAD Research Hub/Architecture & Inference/Implementation/orthogonality_verification/README.md`
   says the 2026-07-10 orthogonality instrument on `worktree-agent-arch-inf-20260710` is theoretically
   superior to the replacement that was drafted, that it reproduces exactly (`iso_ratio_active = 0.254`),
   that the replacement was **withdrawn** in its favour, and it recommends the orchestrator merge it.
   It was never merged. HEAD's `stack/tanitad/eval/spectral.py` has `effective_rank`/`energy_knee` only —
   no `isotropy_ratio`, `participation_ratio`, `condition_number`, `rms_offdiag_correlation`.
3. **The pod ops bundle exists on exactly one disk.** `agent/phase0-supervised-hardening` (7 commits,
   `supervise_run.sh`, `pod_boot_hook.sh`, `pod2_ops_daemon.sh`, `parity_skipset.sh`, …) is **local-only —
   never pushed to origin**, and none of its 9 files exist in HEAD. This is the tooling the pod-recovery
   runbook depends on.

**Bonus finding (worse than the branch sprawl):** `origin/main` is at `0f93b98` — **59 commits behind HEAD**,
and it does not even contain `9413fb7`. GitHub has been stale for a week. (The local `main` merge that landed
mid-audit does not change this — `origin/main` is still unpushed.)

**Gap:** no `gh` CLI on this box, so **GitHub PRs were not inspectable**. If any of the 34 `origin/*` branches
has an open PR with review comments, that context is not reflected here. Everything below is from local refs
plus `git branch -r`.

---

## 1. Unmerged-work table

`n` = `git rev-list --count HEAD..<branch>`. "in HEAD?" is blob-level (`git cat-file -e HEAD:<path>` +
`git rev-parse` comparison), not commit-level.

### 1a. MERGE — 8 branches

| # | Branch | n | What the work IS | Why MERGE |
|---|---|---|---|---|
| 1 | `agent/tools-devenv-20260720` | 2 | `tools/` — `ci_gate.py` v2 (suite manifest), `gpu_tripwire.py` (CUDA device-parity), `session_guard.py` (D-026 stranded-work session-end gate), `ci.ps1`, `session_guard.ps1`, `README.md` + 585 lines of tests. +2641/−444. | **`tools/` does not exist in HEAD at all** (`git ls-tree HEAD -- tools/` is empty). Newest of the whole ci-gate lineage and strictly supersedes branches #19–#22. Highest single-branch value. |
| 2 | `agent/prod-opt-20260718` | 2 | Combined deploy-tick harness (**11.16 ms / 89.6 Hz measured**, 1.59×), predictor CUDA-graph attack (2.57×), **atomic milestone-archive fix (`ckpt_io.py`)**, numerics-safety sweep + tests, VRAM fp16/fp32 json. +2010/−52. | Contains the **live bug fix** described in the headline. All 12 Implementation files absent from HEAD; HEAD only has the derived `Research/2026-07-18-nan-class-sweep.md`. |
| 3 | `agent/phase0-supervised-hardening` | 7 | Pod ops bundle: `supervise_run.sh` (boot-persistent auto-resume + heartbeat + flock, incl. the fd-close restart fix), `pod_boot_hook.sh`, `install_pre_start_hook.sh`, `pod2_ops_daemon.sh` (memory-relief + disk monitor), `parity_skipset.sh` (reap-independent skip-set → clip_ids + sha256), `reap_built_mp4s.sh`, `flagship_phase0.run.env`, `pai_build.run.env`, LF `.gitattributes`. +400/−0. | **All 9 files ABSENT-IN-HEAD** and the branch is **local-only, never pushed**. This is the documented pod2 recovery path. Merge *and* push — single point of failure today. |
| 4 | `agent/data-engineering-20260718` | 4 | ZOD loader (`zod.py` 407 L + 245 L tests, geometry falsifier PASS), curve-rebalance measured on real bytes (243 L + 145 L tests, refutes "74 % straight" as a comma/highway property), ZOD-access escalation to §4 blocked-items. +1771/−14. | HEAD name-drops ZOD in `OWN_DATASET_PLAN.md` / sensor survey / BACKLOG but ships **no loader**. Both Implementation packages and both research notes are absent from HEAD. (Loader is gated on ZOD access — merge the code, keep the blocker flagged.) |
| 5 | `agent/arch-inf-20260718` | 1 | E1+E2 rerun on the **OPERATIVE** flagship-speed @19k: `blind_rollout_flagship.py`, 2-seed result JSONs, `run_orthogonality_flagship.py`, `test_flagship_parity.py`. σ-dissipation reproduces; readout isotropy converging → drops the pre-reset caveat. +1032/−52. | HEAD's `belief_rollout_diagnostic/` holds only the base250cam `blind_rollout.py` (2026-07-17 seeds). The flagship-operative measurement is the one that removes the caveat, and it is not in HEAD. |
| 6 | `worktree-agent-arch-inf-20260710` | 1 | Orthogonality/isotropy admissibility instrument: `spectral_orthogonality.py` (active-subspace `isotropy_ratio`, `participation_ratio`, `condition_number`, `rms_offdiag_correlation`), `run_orthogonality.py`, 109 L tests, `2026-07-10-orth_step6500.json`, theory note. +815/−32. | **HEAD's own README recommends merging this exact branch** (see headline #2). `stack/tanitad/eval/spectral.py` lacks every one of these functions. Gates the D-021 sizing claim on LeJEPA's optimal-planning precondition. |
| 7 | `worktree-agent+bench-eval-20260711` | 2 | D1 ADE statistical-power audit: `d1_ade_power_audit.py` (208 L) + 87 L tests + `run_d1_bootstrap.py` + intake. Verdict: the step-21k "regression" sits **inside the estimator's noise band**. +774/−18. | No bootstrap / CI / power machinery anywhere in HEAD (`git grep -i bootstrap` hits only prose). This is the mechanical antidote to the repeated "read a metric, raise a false alarm" failure mode. |
| 8 | `agent/phase0-highway-dataset` | 3 | (a) `95a777f` scenario-stratified PhysicalAI selection **including highway**, +364/−125 in `physicalai_r0.py`; (b) `7214011` batch the f-theta crop → **~12× less build memory**; (c) `e0b6d2b` `PAI_DECODE_THREADS` cap (default 4). | **MERGE PARTIAL — cherry-pick `7214011` + `e0b6d2b` only.** Both are cheap, self-contained decode hardening absent from HEAD. **`95a777f` → ARCHIVE:** the Phase-0 corpus is frozen at the canonical 2376-episode set for cross-arm parity; re-selecting breaks it. ⚠ **Conflict warning:** HEAD's `stack/tanitad/data/physicalai.py` has moved on (+36 net lines of later per-clip-cy / two-rig work) — cherry-pick, do not merge the branch. HEAD's `physicalai_r0.py` is untouched since `886beb3` and has no highway/stratification at all. |

### 1b. ARCHIVE — 8 branches (tag, then delete)

| # | Branch | n | What it IS | Why ARCHIVE not MERGE |
|---|---|---|---|---|
| 9 | `worktree-agent-a1d4c9c7201f6aacb` | 1 | Axis-6 GO verdict in `PRE_FLIGHT_VALIDATION.md`: from-scratch grounding drops the **true oracle ceiling to 0.68 m** (ridge_a1, fit R²=0.983); SIGReg-relaxation A/B **flat**. +37/−12. | The gate it unblocked is long past, so merging the doc is stale. **But 0.68 m appears nowhere in HEAD** (`Project Steering`, `Benchmarks & Eval`, `HYPOTHESIS_LEDGER` all clean) and HEAD's checkbox is still `- [ ]` unticked. **Hand-lift the 0.68 m ceiling into the LEADERBOARD** — it is the honest in-distribution denominator next to CV 0.83 m. |
| 10 | `agent/prod-opt-20260711` | 2 | INT8 weight-quant curve, windowing fail-loud, imagination-NLL logvar clamp. | **Largely already in HEAD by other means:** `int8_quant/{int8_quant_curve.py, int8_quant_step6500.json, run.log}` are **byte-identical**, `2026-07-10-contract-windowing-failloud/{windowing.py, tests}` **byte-identical**, and the logvar clamp shipped as `incoming/2026-07-17-imagination-logvar-clamp/bounded_logvar.py`. Unique remainder = `imagination_nll_overflow/` sweep data + 2 research notes. Cherry-pick the notes if wanted. |
| 11 | `worktree-tools-devenv-20260715` | 2 | ci.ps1 + `ci_check.py` + `test_ci.py` under `stack/scripts`, plus `RESIM_ROADMAP.md` and the note that flagged **3 parallel unmerged ci-gate branches**. | Gate itself superseded by `tools/ci_gate.py` v2 (#1). **Rescue `TanitAD Research Hub/Tools&DevEnv/RESIM_ROADMAP.md`** — ABSENT-IN-HEAD — and the 2026-07-15 note, then archive. Its own final commit is a written record that this pile-up was flagged 5 days ago and not acted on. |
| 12 | `worktree-agent-bench-eval-20260710` | 1 | D3 decomposition audit + **Compounding Ratio** adoption ("rel falls with k" is a normalization artifact). `i4_compounding.py`, `i4_horizon_audit.py`. | Superseded in substance: HEAD ships `stack/scripts/d3_decompose.py` (superlinear/compounding analysis) and the LEADERBOARD D3 row now reports a **ratio 1.30** — the corrected normalization. Keep the audit note for provenance. |
| 13 | `worktree-arch-inf-20260711` | 1 | D1 probe-capacity ladder; "anisotropy-taxes-linear-probe" **REFUTED**; probe is D≫N underdetermined. | Superseded: HEAD ships `stack/scripts/d1_probe_capacity.py` (ridge @ 3 alphas + MLP) — the promoted version of `probe_capacity_ladder.py`. Unique remainder = the research note + the REFUTED ledger entry. |
| 14 | `worktree-opponent-20260711` | 1 | SC-14 red-light barrier scenario + Zoox/Tesla 2nd-source evidence (viol-rate 0 vs 1, OKRI −82 %). | Superseded: HEAD's `SCENARIO_DATABASE.md` already carries SC-14 **with its numbers** (`rule_barrier 0.0 / soft_prior 1.0`) and states it "Shares the Stop-Arm Gate oracle structure"; HEAD ships `2026-07-24-stop-arm-gate-scenario/stop_arm_gate.py`, the same barrier-vs-soft-prior oracle. The module is a near-duplicate. |
| 15 | `agent/data-engineering-20260711` | 2 | Focal-canonicalization validated on the trained encoder (**wrong intrinsics ≈ 10–15× latent drift**) + L2D semantic-label survey (Apache-2.0, 4 219 nav cmds). | L2D survey superseded by HEAD's `2026-07-19-tanitdataset-v1-semantic-survey.md` + `TANITDATASET_V1_STRATEGY.md`. The **focal-invariance measurement** is not obviously in HEAD (D-016 R1 rectify is the *fix*, not the encoder-drift validation) — cherry-pick `Research/2026-07-11-focal-invariance-validation-and-sc13-sourcing.md` if you want the number on record. |
| 16 | `agent/data-engineering-20260710` | 1 | WorldModel-Synthetic pose probe = **NO-POSE** + video-only loader (`tanitad_worldmodel_synth.py`, 267 L). | Superseded: HEAD's `2026-07-15-worldmodel-pose-gate-and-pandaset-geometry.md` records the pose gate **CLOSED (pose-less)**. The loader is moot given the verdict; keep the probe for the audit trail. |

### 1c. DROP — 10 branches

| # | Branch | n | Verdict / what supersedes it |
|---|---|---|---|
| 17 | ~~`main`~~ | ~~1~~ → 0 | **RESOLVED mid-audit** by `db05620`. `9413fb7 docs(data-eng): OWN_DATASET_PLAN` was blob-identical to HEAD's copy, so the merge added no content — as predicted in §2. `main` is now fully contained in HEAD; the remaining action is `git push origin main`. |
| 18 | `claude/modest-merkle-bd99ff` | 1 | Tip == `9413fb7` == `main`. Pure duplicate ref. |
| 19 | `claude/mystifying-bhaskara-8371db` | 1 | Tip == `9413fb7` == `main`. Pure duplicate ref. |
| 20 | `worktree-prod-opt-20260710` | 1 | Tip `1a5754d` is a **strict ancestor** of `agent/prod-opt-20260711` (#10). Nothing unique. |
| 21 | `agent/tools-devenv-20260718` | 1 | Tip `c4d8451` is a **strict ancestor** of `agent/tools-devenv-20260720` (#1). |
| 22 | `worktree-agent-tools-devenv-20260711` | 2 | `2026-07-11-ci-gate` intake — superseded by `tools/ci_gate.py` v2 (#1). |
| 23 | `worktree-agent+tools-devenv-20260710` | 1 | `2026-07-10-ci-script` intake — **explicitly retired** by branch #11's own recommendation; superseded by #1. |
| 24 | `agent/opponent-20260715` | 1 | SC-13 stationary-lead — HEAD ships `2026-07-31-stationary-lead-scenario/stationary_lead.py` + tests + the SC-13 database entry. Same work redone under the narrative-clock date. |
| 25 | `worktree-agent-opponent-20260710` | 1 | SC-13 stationary-lead, earliest of the three attempts. Same supersession as #24. |
| 26 | `fix-b1-traj-head` | 1 | **Already applied by other means:** `stack/tanitad/models/traj_head.py`, `stack/scripts/finetune_traj.py`, `stack/tests/test_metric_dynamics.py` are **byte-identical** to HEAD; `metric_dynamics.py` and `eval_metric_rollout.py` differ only because HEAD evolved past them. Merging would regress. |

### 1d. Fully contained in HEAD — 30 branches, 0 ahead

All safe to delete (`git branch -d` will succeed; no `-D` needed). Keep `agent/tools-devenv-20260717` (HEAD).
13 of them are pinned by a worktree — remove the worktree first (§4).

```
agent/arch-inf-20260715              agent/opponent-20260717              worktree-agent-a1e1c3a8d0d3d0a52
agent/arch-inf-20260717              agent/opponent-20260720              worktree-agent-a226003185d9b3a6d
agent/benchmarks-eval-20260715       agent/phase0-eval-harness            worktree-agent-a5ab55fd6191cb521
agent/benchmarks-eval-20260717       agent/data-lake-architecture         worktree-agent-a6ce858293916afe5
agent/data-engineering-20260715      agent/data-lake-phase-a-impl         worktree-agent-a9c489b04fdced0df
agent/data-engineering-20260717      bench-eval/backlog3-synthetic-...    worktree-agent-aa92806a9da24de9b
integration/phase0                   tanitscena                           worktree-agent-ab2829da05397168d
validate/axis6-precheck              worktree-agent-a081f0523a43ee547     worktree-agent-ac42943e3ab12d876
worktree-agent-ad9bc5fd1f5168a59     worktree-agent-aeaf2a241cc950641     worktree-agent-aebfd8ecfe01a2ba5
worktree-agent-benchmarks-eval-...   worktree-prod-opt-20260717           (+ agent/tools-devenv-20260717 = HEAD, KEEP)
```

---

## 2. `main` reconciliation — *executed mid-audit*

> **Status:** `db05620 Merge branch 'main' into agent/tools-devenv-20260717` landed while this audit ran.
> `git rev-list --count HEAD..main` is now **0**. The analysis below stands as the record of *why* that merge
> was content-free, and the `origin/main` half of the problem is still open.

**Confirmed contents of `9413fb7`:** exactly one file,
`TanitAD Research Hub/Data Engineering/OWN_DATASET_PLAN.md` (+358 lines) — the owned/redistributable
dataset survey and licensing verdict (permissive core: comma2k19 MIT, Cosmos-Drive-Dreams CC-BY-4.0,
WorldModel-Synthetic OpenMDW-1.1, PandaSet CC-BY-4.0, Udacity MIT, CARLA; ZOD CC-BY-SA-4.0 as a separate
copyleft shard; nuScenes/Waymo/Argoverse/KITTI/BDD100K/… blocked).

**Is merging it safe/valuable? Safe, but worthless — the content is already in HEAD.**

```
main:9413fb7 blob = 51fc59a55957b6523561a7f9e835d8e7e7bb36f9
HEAD          blob = 51fc59a55957b6523561a7f9e835d8e7e7bb36f9   <- identical
```

HEAD acquired the identical file via `aee8084 hub(data-eng): D-016 R1 pinhole rectify…`. So
`git merge main` produces an empty-content merge commit. **Do not do it.**

**The real reconciliation problem is the opposite direction:**

```
git rev-list --left-right --count main...HEAD        ->  1   57
git rev-list --left-right --count origin/main...HEAD ->  0   59
origin/main = 0f93b98   (does not even contain 9413fb7)
```

`git diff HEAD..main` = **215 files, −40 692 lines**: the entire data lake (`stack/tanitad/lake/*`), REF-C
(`refs/refc.py`, `refc_train.py`), the AlpaSim closed-loop harness, `compare_arms.py`, `watch_gates.py`,
`validate_data.py`, the whole 4-brain reset work, every hub research note since 07-13, `PROGRAM_OVERVIEW.md`,
the W32/W33 progress reports, the paper. **`main` and GitHub are a week stale.** Fast-forward `main` to the
integration tip and push — that is the single highest-leverage action in this document.

---

## 3. The stash — **DROP**

```
stash@{0}: WIP on worktree-agent-a6ce858293916afe5: 4279641 flagship-4b: merge full 4-brain …
git stash show --stat stash@{0}
 .claude/settings.local.json | 8 +++++++-
 1 file changed, 7 insertions(+), 1 deletion(-)
```

The full patch is **only** `.claude/settings.local.json` — 7 added permission-allowlist entries
(`ssh -o ConnectTimeout=8 tanitad-pod …`, `Read(//c/Users/Admin/tanitad-data/eval/**)` ×2,
`nvidia-smi --query-gpu…`, `nvidia-smi --query-compute-apps…`, `timeout 20 ssh tanitad-pod …`).

**Every one of those strings is already present in the live `.claude/settings.local.json`** (verified by
`grep -cF` per entry — all returned ≥ 1). Despite the alarming subject line, **there is no 4-brain code in
the stash** — the subject is just the base commit it was taken on. Nothing to lose.

**Verdict: DROP.** (`git stash drop stash@{0}`)

---

## 4. Worktrees — 43 total

Checked with `git --no-optional-locks -C <path> status --porcelain` (no index writes, no working-tree touch).
`git worktree prune --dry-run` reports **0 prunable** — every registered path exists on disk, so all removals
must be explicit. Note `.claude/settings.local.json` is shared and shows dirty in many worktrees; that is
noise, not work.

### 4a. DO-NOT-PRUNE — 2

| Worktree | Branch | Why |
|---|---|---|
| `G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD` | `agent/tools-devenv-20260717` | **The main tree — your live session.** 1 uncommitted file: `TanitAD Research Hub/Data Engineering/TANITDATASET_V1_STRATEGY.md`. |
| `.claude/worktrees/agent-a5ab55fd6191cb521` | `worktree-agent-a5ab55fd6191cb521` | **~486 lines of uncommitted TanitResim work.** See below. |

**What is in `agent-a5ab55fd6191cb521`** (`git diff --stat`, uncommitted, never committed anywhere):

```
 stack/scripts/replay_app.py          |   7 +
 stack/tanitad/resim/export.py        |  99 ++++++++-
 stack/tanitad/resim/static/app.js    | 277 ++++++++++++++++++++++++++-
 stack/tanitad/resim/static/style.css |  39 +++
 stack/tests/test_resim.py            |  62 ++++++
```

It adds an **`ego_poses` → per-window kinematic maneuver strip** to the TanitResim session bundle: export-time
maneuver classes computed from ground-truth ego poses via `scripts/refb_labels.maneuver_labels` (the *same*
class REF-B targets, but arm-independent so every bundle gets a strip even a main-only run), per-episode
`maneuver_counts` histogram for the home-card ribbon, SPA timeline strip + badge, graceful null when poses
are missing or shorter than the label horizon, plus 3 new tests. HEAD's `resim/export.py` has **no
`ego_poses` parameter and no per-step `maneuver` field**. This is a direct implementation of the standing
"show the model's decoded tactical maneuver on the viz" preference. **Commit it before touching this worktree.**

⚠ Its base commit is `4279641` (2026-07-12) while HEAD has moved on — expect to rebase/replay the hunks
rather than merge the branch.

### 4b. Live agent worktrees — 12, all clean, prune only if the loop is stopped

`C:/Users/Admin/wt-*` is the current off-Drive generation (memory: torch/venv reasons). All 12 report zero
uncommitted changes. 7 of them pin unmerged branches; removing the worktree does **not** delete the branch.

| Path | Branch | ahead |
|---|---|---|
| `C:/Users/Admin/wt-h` | `agent/phase0-supervised-hardening` | 7 |
| `C:/Users/Admin/wt-de-0718` | `agent/data-engineering-20260718` | 4 |
| `C:/Users/Admin/wt-tools-0720` | `agent/tools-devenv-20260720` | 2 |
| `C:/Users/Admin/wt-prod` | `agent/prod-opt-20260718` | 2 |
| `C:/Users/Admin/wt-ai-0718` | `agent/arch-inf-20260718` | 1 |
| `C:/Users/Admin/wt-opp` | `agent/opponent-20260715` | 1 |
| `C:/Users/Admin/wt-tools-0718` | `agent/tools-devenv-20260718` | 1 |
| `C:/Users/Admin/wt-ai` | `agent/arch-inf-20260717` | 0 — stale |
| `C:/Users/Admin/wt-be` | `agent/benchmarks-eval-20260715` | 0 — stale |
| `C:/Users/Admin/wt-de` | `agent/data-engineering-20260715` | 0 — stale |
| `C:/Users/Admin/wt-de-0717` | `agent/data-engineering-20260717` | 0 — stale |
| `C:/Users/Admin/wt-opponent-20260720` | `agent/opponent-20260720` | 0 — stale |

### 4c. PRUNE LIST — 29 worktrees under `.claude/worktrees/` (names only)

All clean or `.claude/settings.local.json`-only. **`agent-a5ab55fd6191cb521` is deliberately excluded.**

```
agent+bench-eval-20260711
agent+tools-devenv-20260710
agent-a081f0523a43ee547
agent-a1d4c9c7201f6aacb
agent-a1e1c3a8d0d3d0a52
agent-a226003185d9b3a6d
agent-a6ce858293916afe5
agent-a9c489b04fdced0df
agent-aa92806a9da24de9b
agent-ab2829da05397168d
agent-ac42943e3ab12d876
agent-ad9bc5fd1f5168a59
agent-aeaf2a241cc950641
agent-aebfd8ecfe01a2ba5
agent-arch-inf-20260710
agent-bench-eval-20260710
agent-benchmarks-eval-20260710
agent-data-engineering-20260710
agent-data-engineering-20260711
agent-opponent-20260710
agent-tools-devenv-20260711
arch-inf-20260711
modest-merkle-bd99ff
opponent-20260711
prod-opt-20260710
prod-opt-20260711
prod-opt-20260717
tanitscena
tools-devenv-20260715
```

Plus the 5 stale off-Drive ones once the loop is idle: `wt-ai`, `wt-be`, `wt-de`, `wt-de-0717`,
`wt-opponent-20260720`. **Total prunable: 34; leaves 9 (main + 8 live).**

---

## 5. High-value shipped work not in mainline — the short answer

Ranked by what it costs you to keep losing it.

| Rank | Artifact | Branch | Status in HEAD |
|---|---|---|---|
| 1 | **`ckpt_io.py::atomic_archive`** — fixes silent gate-milestone corruption | `agent/prod-opt-20260718` | **BUG STILL LIVE** in all 3 trainers |
| 2 | **Pod ops bundle** (`supervise_run.sh` + boot hook + ops daemon + parity skip-set) | `agent/phase0-supervised-hardening` | ABSENT — and **local-only, unpushed** |
| 3 | **`tools/`** — ci_gate v2, gpu_tripwire (CUDA device-parity), session_guard (D-026) | `agent/tools-devenv-20260720` | Directory does not exist |
| 4 | **Orthogonality/isotropy admissibility instrument** | `worktree-agent-arch-inf-20260710` | ABSENT; **HEAD's own README asks for it** |
| 5 | **TanitResim maneuver-strip** (~486 L, uncommitted) | worktree `agent-a5ab55fd6191cb521` | ABSENT; not committed anywhere |
| 6 | **D1 ADE statistical-power / bootstrap audit** | `worktree-agent+bench-eval-20260711` | No bootstrap machinery in HEAD |
| 7 | **ZOD loader + curve-rebalance measurement** | `agent/data-engineering-20260718` | Docs reference ZOD; no loader |
| 8 | **Flagship-operative blind-rollout + orthogonality** (drops the pre-reset caveat) | `agent/arch-inf-20260718` | Only the base250cam run is in HEAD |
| 9 | **PhysicalAI decode hardening** (12× build-memory cut, PyAV thread cap) | `agent/phase0-highway-dataset` | ABSENT |
| 10 | **0.68 m from-scratch oracle ceiling** | `worktree-agent-a1d4c9c7201f6aacb` | Number appears nowhere in HEAD |

**Eval harnesses:** the good news is that the big ones *did* land — `2026-07-19-alpasim-closedloop-v1`,
`2026-07-16-eval-metric-suite`, `2026-07-17-openloop-l2-egostatus-shortcut`, `2026-07-15-baseline-floor`,
`stack/scripts/{d1_probe_capacity,d3_decompose,watch_gates,compare_arms}.py` are all in HEAD. The gap in eval
is **statistical rigour** (#6), not coverage.

**Training fixes:** #1 is the only real one stranded. The speed-channel / 4-brain / K-step work all reached
mainline.

**Data tooling:** #7 and #9. The highway *re-selection* is deliberately not recommended (parity).

---

## 6. Prioritized action list — commands to run (NOT executed)

Run from `G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD`. Paths are quoted for the spaces.
Nothing below was run by the triage.

### P0 — stop the bleeding (do these first)

```bash
# 0a. Push the local-only ops branch BEFORE anything else. It exists on one disk.
git push -u origin agent/phase0-supervised-hardening

# 0b. Commit the stranded TanitResim maneuver-strip work (486 lines, uncommitted, nowhere else).
git -C ".claude/worktrees/agent-a5ab55fd6191cb521" add \
    stack/scripts/replay_app.py stack/tanitad/resim/ stack/tests/test_resim.py
git -C ".claude/worktrees/agent-a5ab55fd6191cb521" commit -m \
  "resim: kinematic maneuver strip from ego poses (export + SPA + tests)"
#    ^ base is 4279641; expect to cherry-pick/rebase onto HEAD, not merge.

# 0c. Un-stale GitHub. origin/main is 59 behind and blocks every PR-shaped workflow.
#     The local main<->HEAD merge already happened mid-audit (db05620), so this is now just:
git branch -f main agent/tools-devenv-20260717     # main has no unique content (see §2)
git push origin main
```

### P1 — merge the 8 (suggested order; each is independent)

```bash
git merge --no-ff agent/prod-opt-20260718            -m "merge(prod-opt): atomic milestone archive + combined deploy-tick + CUDA-graph predictor"
git merge --no-ff agent/phase0-supervised-hardening  -m "merge(ops): pod supervised auto-resume, boot hook, ops daemon, parity skip-set"
git merge --no-ff agent/tools-devenv-20260720        -m "merge(tools): ci_gate v2 + gpu_tripwire + session_guard"
git merge --no-ff worktree-agent-arch-inf-20260710   -m "merge(arch-inf): orthogonality/isotropy admissibility instrument (per 07-17 verification)"
git merge --no-ff agent/arch-inf-20260718            -m "merge(arch-inf): operative flagship blind-rollout + orthogonality"
git merge --no-ff worktree-agent+bench-eval-20260711 -m "merge(bench-eval): D1 ADE statistical-power audit"
git merge --no-ff agent/data-engineering-20260718    -m "merge(data-eng): ZOD loader + measured curve rebalance"

# #8 is a CHERRY-PICK, not a merge — physicalai.py has diverged in HEAD.
git cherry-pick 7214011 e0b6d2b     # batched f-theta crop (~12x mem) + PAI_DECODE_THREADS cap
# Do NOT take 95a777f (highway re-selection) — breaks the frozen 2376-ep parity corpus.

# After the tools merge, apply the fix it carries to the three live trainers:
#   stack/scripts/train_flagship4b.py:417
#   stack/scripts/refb_train.py:375
#   stack/experiments/reset-speed4b/refa_train_plus.py:540
#   shutil.copy2(ckpt, arch)  ->  ckpt_io.atomic_archive(ckpt, arch)
```

### P2 — rescue the orphan artifacts, then archive-tag

```bash
# Two files worth lifting out before their branches are tagged away:
git checkout worktree-tools-devenv-20260715 -- "TanitAD Research Hub/Tools&DevEnv/RESIM_ROADMAP.md"
git checkout agent/data-engineering-20260711 -- \
  "TanitAD Research Hub/Data Engineering/Research/2026-07-11-focal-invariance-validation-and-sc13-sourcing.md"
# And by hand: put the 0.68 m from-scratch oracle ceiling into Benchmarks & Eval/LEADERBOARD.md
#   (source: worktree-agent-a1d4c9c7201f6aacb : Project Steering/PRE_FLIGHT_VALIDATION.md)

# Tag the 8 ARCHIVE branches (annotated tags keep the history reachable after branch deletion):
git tag -a archive/2026-07-12-axis6-go-verdict          worktree-agent-a1d4c9c7201f6aacb -m "Axis-6 GO: from-scratch oracle ceiling 0.68 m; SIGReg-relax A/B flat"
git tag -a archive/2026-07-11-prod-opt-int8-logvar      agent/prod-opt-20260711           -m "INT8 curve + windowing fail-loud + imagination-NLL logvar clamp (mostly landed)"
git tag -a archive/2026-07-15-ci-gate-v1                worktree-tools-devenv-20260715    -m "ci.ps1/ci_check.py gate v1 + RESIM_ROADMAP (superseded by tools/ci_gate.py v2)"
git tag -a archive/2026-07-10-d3-compounding-audit      worktree-agent-bench-eval-20260710 -m "D3 decomposition audit + Compounding Ratio"
git tag -a archive/2026-07-11-d1-probe-ladder           worktree-arch-inf-20260711        -m "D1 probe-capacity ladder; anisotropy-taxes-linear-probe REFUTED"
git tag -a archive/2026-07-11-sc14-red-light-barrier    worktree-opponent-20260711        -m "SC-14 red-light barrier scenario (superseded by stop-arm-gate oracle)"
git tag -a archive/2026-07-11-focal-invariance-l2d      agent/data-engineering-20260711   -m "Focal-canonicalization validation + L2D semantic-label survey"
git tag -a archive/2026-07-10-worldmodel-pose-probe     agent/data-engineering-20260710   -m "WorldModel-Synthetic pose probe = NO-POSE + video-only loader"
git push origin --tags
```

### P3 — worktree prune (34)

```bash
# 29 under .claude/worktrees/ — agent-a5ab55fd6191cb521 deliberately EXCLUDED (see 4a/0b)
for w in agent+bench-eval-20260711 agent+tools-devenv-20260710 agent-a081f0523a43ee547 \
         agent-a1d4c9c7201f6aacb agent-a1e1c3a8d0d3d0a52 agent-a226003185d9b3a6d \
         agent-a6ce858293916afe5 agent-a9c489b04fdced0df agent-aa92806a9da24de9b \
         agent-ab2829da05397168d agent-ac42943e3ab12d876 agent-ad9bc5fd1f5168a59 \
         agent-aeaf2a241cc950641 agent-aebfd8ecfe01a2ba5 agent-arch-inf-20260710 \
         agent-bench-eval-20260710 agent-benchmarks-eval-20260710 agent-data-engineering-20260710 \
         agent-data-engineering-20260711 agent-opponent-20260710 agent-tools-devenv-20260711 \
         arch-inf-20260711 modest-merkle-bd99ff opponent-20260711 prod-opt-20260710 \
         prod-opt-20260711 prod-opt-20260717 tanitscena tools-devenv-20260715; do
  git worktree remove --force ".claude/worktrees/$w"
done

# 5 stale off-Drive worktrees — ONLY when the agent loop is stopped
git worktree remove C:/Users/Admin/wt-ai
git worktree remove C:/Users/Admin/wt-be
git worktree remove C:/Users/Admin/wt-de
git worktree remove C:/Users/Admin/wt-de-0717
git worktree remove C:/Users/Admin/wt-opponent-20260720

git worktree prune -v
```

### P4 — branch deletion (after P1–P3 land)

```bash
git stash drop stash@{0}                              # §3: settings.local.json entries already live

# 9 DROP branches (main is no longer among them — it merged mid-audit)
git branch -D claude/modest-merkle-bd99ff claude/mystifying-bhaskara-8371db \
              worktree-prod-opt-20260710 agent/tools-devenv-20260718 \
              worktree-agent-tools-devenv-20260711 worktree-agent+tools-devenv-20260710 \
              agent/opponent-20260715 worktree-agent-opponent-20260710 fix-b1-traj-head

# 8 ARCHIVE branches — only after the tags in P2 exist and are pushed
git branch -D worktree-agent-a1d4c9c7201f6aacb agent/prod-opt-20260711 \
              worktree-tools-devenv-20260715 worktree-agent-bench-eval-20260710 \
              worktree-arch-inf-20260711 worktree-opponent-20260711 \
              agent/data-engineering-20260711 agent/data-engineering-20260710

# 30 fully-merged branches (safe -d; §1d — now includes main, which you may prefer to keep). Verify first:
git branch --merged HEAD | grep -v -E "^\*|agent/tools-devenv-20260717"
# then:  ... | xargs -n1 git branch -d

# Finally clean the remote of the branches you deleted locally, e.g.:
# git push origin --delete worktree-agent-opponent-20260710 ...
```

**End state:** 56 branches → ~8 (HEAD, `main`, and the small set you actively iterate on) + 8 archive tags.
43 worktrees → 9.
