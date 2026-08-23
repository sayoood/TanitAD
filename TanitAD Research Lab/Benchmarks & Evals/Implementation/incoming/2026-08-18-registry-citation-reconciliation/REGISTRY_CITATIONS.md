# Registry citation reconciliation — 2026-08-18

**Task:** make HEAD's suite green by reconciling the registry-citation lint failures, without
touching a single registry number. **Scope:** citation paths, anchors, and citation formatting in
`Project Steering/MODEL_REGISTRY.md` ONLY. **Author:** registry-citation subagent, clone
`C:/Users/Admin/wt-tanitad-local`, branch `agent/arch-inf-20260803`.

**Baseline correction for the caller:** the brief named the failing tests as
`stack/tests/test_registry_lint.py` / `stack/tests/test_registry_paths_allow.py`. They live at
**`tools/tests/`**, not `stack/tests/` (MEASURED — `stack/tests/` has no such files; the pytest
invocation from the brief exits 4 "file not found"). Baseline on this clone (then-HEAD
`64575912`): **3 failed / 37 passed** across the two files, failures identical to the brief's
description.

---

## 1. The three failing tests → fourteen defective citation sites

| test | defect it caught |
|---|---|
| `test_seeded_sidecar_pointers_resolve_against_the_live_repo` | 1 inline drift-pointer with an unresolvable `src` (site 14) |
| `test_real_registry_has_no_dead_or_malformed_citations` | 6 MISSING + 7 NOT_A_PATH backticked citations (sites 1–13) |
| `test_real_registry_exit_code_is_not_a_hard_failure` | same 13 (exit-code view of the same defects) |

Mechanics (MEASURED from `tools/registry_paths.py` source): a backticked token ending in a known
extension is a citation; `{…}`/`*`/`?`/`<` ⇒ NOT_A_PATH; a repo-relative path that resolves
nowhere (after suffix-, loose-tail- and basename-fallbacks) ⇒ MISSING; a leading-`/` token ⇒
NOT_CHECKED "absolute pod path" (never MISSING, `stranded` reported); a no-slash token ⇒ NAME_ONLY
(never a defect). The allowlist `tools/registry_paths_allow.json` could excuse pattern-tokens, but
`tools/` is out of my scope — **every fix below therefore lives in the registry text itself, and
the allowlist was not touched** (its occurrence-counted tokens were also left byte-identical, so
no ALLOW_STALE/ALLOW_COUNT_MISMATCH was triggered; verified by the green run).

## 2. Per-citation verdicts

Evidence classes: **MEASURED** unless stamped otherwise. "HF tree" = unauthenticated
`huggingface.co/api/models/Sayood/tanitad-flagship-v5f-w120/tree/main` listing via
`truststore.inject_into_ssl()`, 2026-08-18, from this box.

| # | citation (as was) | verdict | evidence | fix applied |
|---|---|---|---|---|
| 1 | `_pod_backup/pod2-2026-08-03/ckpts/flagship4b-phase0-30k_ckpt.pt` | **not a repo artifact BY DESIGN** — dev-box copy, deliberately git-ignored | `.gitignore:48` = `_pod_backup/**/*.pt`; the dir IS tracked (`git ls-files` shows `BACKUP_LOG.txt` beside it); the row itself already states "git-ignored … NOT in the repo" | formatting only: split into dir-token `_pod_backup/pod2-2026-08-03/ckpts/` + name-token `flagship4b-phase0-30k_ckpt.pt` so the linter stops classifying a documented non-repo path as a dead repo path. Every path character preserved |
| 2 | `stage-a-predictor/ckpt_stage_a.pt` | **unbanked pod ckpt; durable copy on HF** | emitters `stack/scripts/train_stage_a.py:102,:909`, `stack/scripts/t1_v58f_chain.sh:31` (pod path `/workspace/experiments/stage-a-predictor/`); HF tree: `release/v58f/ckpt/ckpt_stage_a.pt` (3,248,642,418 B) mapped by `stack/scripts/release_v58f.py:24` | cite the true pod path (linter class: absolute-pod-path, NOT_CHECKED) + the verified HF location |
| 3 | `w7_gate_k{8,32,64}.json` | **(a) banked in-repo** — brace form hid it | all three tracked at `…/incoming/2026-08-07-hierarchical-wm-redesign/` (`git ls-files`) | three explicit citations (first with the resolvable `…/incoming/…` form → EXISTS) |
| 4 | `w7-full-roll/w7_eval_windows.pt` | **(b) UNBANKED, gone** | pod4 per the W7-FULL header in the same section; emitter `stack/scripts/w7_roll_rerank.py:715`; NOT in the HF release (tree listed); pods gone per `stack/scripts/sel_winners_curse_law.py:13`; never in git history (`git log --all --diff-filter=A`) | full pod path + `[⛔ UNBANKED …]` marker |
| 5 | `p8_gate_attempt{1,2}.json` | **(a) banked in-repo** | both tracked in the 2026-08-07 hub dir | two explicit name citations |
| 6 | `i4a/flagship-v5f-w120-30k-i4a-{none,zero,shuffle}.json` | **(b) pod5 gone; durable copies on HF** | HF tree: `release/v58f/gates/i4a_{none,zero,shuffle}.json` (11.8 KB each), local-stem mapping per `release_v58f.py:40–45`; never in git history | pod dir + local stems in prose + verified HF locations. ⚠️ noted: the ops-bundle `i4a_chain.sh:24` writes `i4a_$MODE.json` — a stale naming variant; the release manifest's stems are the ones proven by the HF push |
| 7 | `w7-repaired-w4r-k32/w7_gate.json` | **(b) pod5 gone; durable copy on HF** | HF tree: `release/v58f/gates/w7_w4r_k32_gate.json` (8,406 B), mapping per `release_v58f.py:38` | full pod path + verified HF location |
| 8 | `ph0_mini/v2/ph0_v2.json` | **(b) UNBANKED, gone** | pod4 per the PH0-v2 header; out-dir `stack/scripts/ph0_v2_chain.sh:20`; FOUR absence probes: tree find, `git log --all --diff-filter=A`, HF release tree (only the separate `ph0_pilot_novlm/` set is there), hub `Evaluation/Videos/ph0-vlm-overlay-2026-08-12/` (mp4s only) | full pod path + `[⛔ UNBANKED …]` marker |
| 9 | `lf0-bev-lead/lf0_gate.json` | **(b) UNBANKED, gone** | pod4 per the LF0 header; out-dir `stack/scripts/lf0_chain.sh:17`; not in HF release; never in git history | full pod path + `[⛔ UNBANKED …]` marker |
| 10 | `w7-prog-{01,05}/w7_gate.json` | **(b) UNBANKED, gone** | pod4 per the W7-PROG header; run-dir names from the citation + `PREREG_W7_PROG.md` arm names; no in-tree emitter names an absolute path, so **none is asserted** (the marker keeps dir names only) | name-citation + run dirs in prose + `[⛔ UNBANKED …]` marker |
| 11 | `four_families/ff_{stageA,v5f30k}_{cl,ol,ha}.json` | **(b) pod5 gone; durable copies on HF** | HF tree: all six under `release/v58f/gates/four_families/` with the SAME basenames, + `ff_comparison.json` (55,705 B); pod path was already in the row's own text | stems kept in prose (un-backticked), verified HF location added |
| 12 | `pc6_ridge_*.json` | **(a) banked in-repo** — glob form hid it | 5 files tracked at `…/incoming/2026-08-17-probe-positive-control/raw/` (`git ls-files`): `pc6_ridge_{nullmatched,orc010,orcdir,s09000,s11250}.json` | family stem kept (un-backticked `pc6_ridge_*`) + the real directory added |
| 13 | `taniteval/results/*.json` | **(a) banked in-repo** — glob form | the directory holds **exactly 73** `*.json` (MEASURED `ls | wc -l`) = the claim's own count | reworded to "all **73** JSONs in `taniteval/results/`" — same claim, no glob token |
| 14 | inline pointer `<!-- src: …/2026-08-16-jack-in-gates/raw/g1_g4_both_estimators.json#G1.arms.tactical_head -->` (registry table row `plan_flagship-30k`) | **(c) citation-mechanics defect, doubly** | `tools/registry_lint.py:432` joins `repo / src` LITERALLY (no ellipsis resolution) → unreadable; AND the field names a dict — `G1.arms.tactical_head` is an object whose `corrected_mean` = 3.3839, the exact number the row quotes (MEASURED from the JSON) | full repo-relative src (`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-16-jack-in-gates/raw/g1_g4_both_estimators.json`) + field extended to `.corrected_mean`. The pointer now BINDS and verifies the row's 3.3839 against its raw JSON |

**Not touched on purpose:** `gate_step{1k,5k,10k}.json` — the allowlist's single ratcheted
`unresolved` entry (its occurrence count is load-bearing); every `heldout`/`full_set` value; all
registry numbers, statuses, parity keys, claim texts.

## 3. Proof: zero changed numbers

Mechanical check over the whole file, old (`git show HEAD:…`) vs new, multisets compared:

```
decimals (\d+\.\d+):              only-old: {}   only-new: {'5.8': 4}
comma-grouped (\d{1,3}(,\d{3})+): IDENTICAL multisets (345 tokens)
scientific (\d+e-?\d+ forms):     IDENTICAL multisets (82 tokens)
```

Nothing was removed or altered (`only-old` is empty). The four new `5.8` tokens are the version
literal **"v5.8f"** inside the four added "NOT in the v5.8f HF release" markers — a release name,
not a measurement. All other new digits sit inside location metadata prescribed by the task's own
UNBANKED-marker format: `/workspace/...` paths, emitter line numbers (`:17`, `:20`, `:715`), the
verification date 2026-08-18, and `release/v58f/...`. The full `git diff --word-diff` is appended
in §6 — every `[-removed-]` span is a citation token, every `{+added+}` span is a path or
location marker.

## 4. Verification

```
$ python -m pytest tools/tests/test_registry_lint.py tools/tests/test_registry_paths_allow.py tools/tests/test_registry_paths.py -q
64 passed in 0.53s
```

`tools/registry_paths.py --only-bad` after the fix: MISSING 0, NOT_A_PATH 0, allow issues 0,
unresolved ratchet 1/1 (unchanged), EXISTS 186→188, NOT_CHECKED 29→34 (the five new
absolute-pod-path citations), NAME_ONLY 164→173. All 15 inline pointers bind.

**Full `stack/tests` suite:** see §5 tail. ⚠️ It CANNOT run in this clone the obvious way while
the G: mount is down — see the environment finding below, which also explains the collection
errors a naive run produces.

### ⚠️ Environment finding (pre-existing, NOT caused by these edits)

`pytest stack/tests` in the clone dies at collection with `OSError: [Errno 22] Invalid argument`
on the first 8 files. Root cause (MEASURED): the venv carries an **editable install of `tanitad`
pointing at the G: mount** —

```
$ python -c "import importlib.util as u; print(u.find_spec('tanitad').origin)"
G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack\tanitad\__init__.py
```

(`site-packages/__editable__.tanitad-0.0.1.pth`). With G: flapping, `find_spec` intermittently
succeeds while `get_data` (the bulk source read) raises Errno 22 — a plausible-looking test
failure that has NOTHING to do with the tree under test. A `.md` edit cannot cause an import-time
OSError; the same 8 errors reproduce with the registry edit stashed-equivalent (single-file run,
twice, minutes apart). **Fix used, same family as the pod rule "`PYTHONPATH=/workspace/TanitAD/stack`
is REQUIRED":** run the suite as
`PYTHONUTF8=1 PYTHONPATH=/c/Users/Admin/wt-tanitad-local/stack python -m pytest stack/tests -q`
— `sys.path` outranks the appended editable meta-finder, so every import stays inside the clone.

## 5. Suite tails

Registry tests (the deliverable): `64 passed in 0.53s` (see §4).

Full `stack/tests` with the PYTHONPATH override (exit code captured before any pipe; run alone,
nothing else on the CPU):

```
2 failed, 4276 passed, 8 skipped, 2 xfailed, 10 warnings in 471.96s (0:07:51)
PYTEST_EXIT=1
FAILED stack/tests/test_loss_determinism.py::test_NEGATIVE_CONTROL_the_enumerator_can_fire_AND_NAMES_THE_RIGHT_FILE
FAILED stack/tests/test_v4_labels.py::test_mintability_report_names_the_gaps
```

**Both failures are PRE-EXISTING and cannot observe my edit — verdict: NO NEW FAILURES.**
Evidence (MEASURED):

1. Neither test file nor the code under test mentions the registry at all
   (`grep -c "MODEL_REGISTRY|Project Steering"` = 0 in `test_loss_determinism.py`,
   `test_v4_labels.py`, `scripts/v4_labels.py`, `models/sigreg.py`), and
   `git status` shows the registry is the tree's ONLY modification — the tests exercise
   committed code exclusively.
2. `test_v4_labels::test_mintability_report_names_the_gaps` asserts the literal `lead_state`
   appears in `mintability_report()["not_mintable_needs_data"]` — a HARDCODED dict. Commit
   `4276613a` ("The stale lead refusal is RETIRED at all four live sites", 2026-08-18)
   reworded that value to "the lead **state** EXISTS" (space, not underscore) and no
   `lead_state` literal survives in the committed blob. `git merge-base --is-ancestor`
   confirms `4276613a` precedes my baseline `64575912` — **broken before my session began**,
   by the lead-wiring stream's own doc-rot fix outrunning its test.
3. `test_loss_determinism::test_NEGATIVE_CONTROL_…` fails because the RNG-site classifier
   `_site()` (test file line ~171) counts a frame as in-repo iff `"TanitAD" in f.filename` —
   and this clone lives at `wt-tanitad-local`, so NO frame ever matches and every draw reports
   `<outside the repo>`. The test fails at ANY commit in ANY clone whose path lacks the string
   "TanitAD"; on the canonical `G:\…\TanitAD` checkout it passes. Same trap family as the
   CLAUDE.md `df`-on-pods entry: a probe whose scope assumption (the repo's dirname) travels
   silently with it.

Registry-relevant suites, all green post-fix: `tools/tests` 64/64 (§4).

## 6. The word-diff (verbatim `git diff --word-diff -- "Project Steering/MODEL_REGISTRY.md"`)

```
diff --git a/Project Steering/MODEL_REGISTRY.md b/Project Steering/MODEL_REGISTRY.md
index 23b91440..44144160 100644
--- a/Project Steering/MODEL_REGISTRY.md	
+++ b/Project Steering/MODEL_REGISTRY.md	
@@ -150,7 +150,7 @@ Grounding heads live **outside** the model (separate ckpt keys) so a vanilla `Wo
| Field | Value |
|---|---|
| **Status** | **SUPERSEDED** — killed by the 2026-07-14 speed reset at step **22,950**; retained as the causal ablation control |
| **Location** | 🔴 **`tanitad-pod2` IS TERMINATED.** The run dir `/workspace/experiments/flagship4b-phase0-30k/` went with it, and it is **absent from pod4's rescue dump** (MEASURED 2026-08-03, read-only `ls /workspace/rescue/experiments/`). **Reachable copies — of `ckpt.pt` ONLY:** (a) dev box [-`_pod_backup/pod2-2026-08-03/ckpts/flagship4b-phase0-30k_ckpt.pt`,-]{+`_pod_backup/pod2-2026-08-03/ckpts/` file `flagship4b-phase0-30k_ckpt.pt`,+} **3 302 176 350 B**, md5 `74be81035699c362e2fd0e5197880506` (per `_pod_backup/pod2-2026-08-03/ckpts/BACKUP_LOG.txt`) — ⚠️ **git-ignored by `.gitignore:48`, so it is NOT in the repo and one disk failure ends it**; (b) HF `Sayood/tanitad-flagship-4b-phase0`. ⛔ **`config.json`, `train_log.jsonl` and the per-gate JSONs have no reachable copy.** |
| **⛔ UNRESOLVED citation** | This row used to cite `gate_step{1k,5k,10k}.json`, which is wrong twice over: a **shell brace expansion names no file**, and the stem is wrong as well. MEASURED 2026-08-03 from the emitters — `stack/scripts/watch_gates.py:213` and `stack/scripts/evaluate_checkpoint.py:201` both write `f"gates_step{step}.json"` — so the real name is **gates_step<full-integer>.json** (plural `gates`, no `k` abbreviation), and **no `gate_step*` writer exists anywhere in the tree**. ⛔ **I did not rewrite it to a guess**: the host is terminated and the dir is not in the rescue dump, so neither the filenames nor the gate steps can be verified, and a confidently wrong citation is worse than an obviously broken one. Tracked as the single `unresolved` entry in `tools/registry_paths_allow.json`; resolving it means finding the run dir, then lowering that file's `max_unresolved`. |
| **Distinguishing flags** | `speed_input=false`, `action_dim=2`, `jerk_weight=0.0`, `aux_accel=false`, `rollout_k=4` |
| **Params (from run config)** | encoder 87,121,280 · operative 96,607,490 · tactical_pred 26,534,912 · tactical_policy 22,736,141 · strategic_policy 8,385,027 · h15 22,055,683 · grounding_heads 13,432,338 → **total_model 263,440,533 / trainable 276,872,871** ✅ |
@@ -1403,7 +1403,10 @@ after on the full held-out W3 pack: **lateral gain 0.27 → 0.971/0.966** (gate
**longitudinal sign 0.745/0.787 → 1.0/1.0** (gate ≥0.95); lateral sign stays 1.0;
longitudinal gain 0.972 (reported); **P6 subspace stays exactly 3-dim**; no-harm passed.
The single root defect behind the action echo, the three scoring failures and W7's ceiling
is closed at head-scale cost. Repaired ckpt: [-`stage-a-predictor/ckpt_stage_a.pt`;-]{+`/workspace/experiments/stage-a-predictor/ckpt_stage_a.pt`+}
{+(pod copy gone — hosts terminated; durable copy verified 2026-08-18 on HF+}
{+`Sayood/tanitad-flagship-v5f-w120` at `release/v58f/ckpt/` `ckpt_stage_a.pt`, per the+}
{+`stack/scripts/release_v58f.py` manifest + HF tree listing);+}
artifact `stage_a_gate.json`. W7-on-repaired (K=32) ran ~07:35Z: gate FAIL by INSTRUMENT
COMPOSITION (the frozen-trunk-trained W4 head/selector don't compose with the repaired
trunk — §1.14), while roll-cost calibration nearly doubled (ρ 0.716) — the repair's
@@ -1441,7 +1444,8 @@ barely differ, so the cost drowns. **Why: W3 measured the WM's action-response g
DEFECT: the trunk under-weights actions in its rollout.** ⇒ **Stage-A post-training
(V18 E3.4: L_ctrl gain repair, targets measured by W3 — lateral gain into [0.5, 2],
longitudinal sign ≥95 %, preserve the 3-dim action subspace) is THE critical path** for
selection AND closed-loop capability; W7 re-runs after it. Artifacts:
[-`w7_gate_k{8,32,64}.json`.-]{+`…/incoming/2026-08-07-hierarchical-wm-redesign/w7_gate_k8.json` + `w7_gate_k32.json` + `w7_gate_k64.json`.+}

**W7-ON-REPAIRED (stage-A trunk, K=32) — MEASURED 2026-08-11 ~07:35Z [T0, 881 grid]: gate
FAIL (selected 2.3468 vs thr 0.4505; frac closed −2.27) — but the failure is INSTRUMENT
@@ -1491,7 +1495,9 @@ and **you cannot repair a trunk and keep its planner**; (2) that sentence IS the
training argument for v6 — consumers must be (re)trained ON the trunk they consume
(S-W → S-T → S-S), and argmin-over-a-large-fan must be replaced by a
noise-robust rule (top-m aggregation / sharpened cost), pre-registered before it is used.
Artifact: `w7_full_gate.json`; per-window arrays [-`w7-full-roll/w7_eval_windows.pt`.-]{+`/workspace/experiments/w7-full-roll/w7_eval_windows.pt`+}
{+[⛔ UNBANKED — lived on pod4 (terminated) per this block's header, emitter `stack/scripts/w7_roll_rerank.py:715`;+}
{+never committed, and NOT in the v5.8f HF release (tree listed 2026-08-18)].+}

**P8 BEV-OCCUPANCY READOUT (attempt 2) — MEASURED 2026-08-12 ~00:05Z [T0-diagnostic,
881 grid, pod4]: GATE PASS — the PREDICTED latent retains the environment.** Attempt 1's
@@ -1542,7 +1548,7 @@ quote the enc arm, or quote the pred arm *with* this. Also: the join flagged at
sensor's 120° while the encoder saw the 117° sub-frame, which puts unseen agents in the
*visible* bucket and can only SHRINK this gap ⇒ the banked number is **conservative**.
Artifacts: `…/incoming/2026-08-16-p4-fov-predicate/P4_FOV_PREDICATE.md` +
`raw/p4_predicate_identity.json`; both [-`p8_gate_attempt{1,2}.json`-]{+`p8_gate_attempt1.json` + `p8_gate_attempt2.json`+} annotated in place.

**I4a IMAGINATION ABLATION — MEASURED 2026-08-11 ~19:40Z [T0, 881 grid]: the imagination
channel is LOAD-BEARING, not decorative.** Three arms, same checkpoint, same grid, only the
@@ -1554,8 +1560,12 @@ result: shuffling preserves the marginal statistics and destroys only the
window↔consequence correspondence, so the planner is reading imagination as CONTENT, not
as a bias term. ⚠️ Caveat stamped: the head was TRAINED with imagination present, so this
measures the dependence of THIS architecture, not the value of retraining without it;
I4b (occluded-split stratification) is the next refinement. Artifacts: [-`i4a/flagship-v5f-w120-30k-i4a-{none,zero,shuffle}.json` (pod5).-]{+three JSONs in+}
{+pod5 `/workspace/experiments/i4a/` (local stems flagship-v5f-w120-30k-i4a-none/zero/shuffle;+}
{+pod terminated, never committed) — durable copies verified 2026-08-18 on HF+}
{+`Sayood/tanitad-flagship-v5f-w120` at `release/v58f/gates/` as `i4a_none.json` /+}
{+`i4a_zero.json` / `i4a_shuffle.json`, per the `stack/scripts/release_v58f.py` manifest ++}
{+HF tree listing.+}

**W4r + W7-w4r — MEASURED 2026-08-11 ~19:10Z [T0, 881 grid]: the repair arc closes on ONE
remaining stale part.** W4r (unicycle head refit ON the stage-A trunk, 4000 steps, trunk
@@ -1570,7 +1580,9 @@ component, and it sits in W7's PRUNER, not its cost.** ⇒ **W7-FULL queued (top
shortlist, selector-free): roll-cost + kinematic cost over the whole healthy fan — the
first selection read of the fully-repaired pipeline with NO stale part anywhere** (pod4,
behind p8c; W4r head relayed via HF /battery/). Artifacts: `w4r_gate.json`,
[-`w7-repaired-w4r-k32/w7_gate.json` (pod5).-]{+`/workspace/experiments/w7-repaired-w4r-k32/w7_gate.json` (pod5, terminated — durable copy+}
{+verified 2026-08-18 on HF `Sayood/tanitad-flagship-v5f-w120` at `release/v58f/gates/` as+}
{+`w7_w4r_k32_gate.json`, per the `stack/scripts/release_v58f.py` manifest + HF tree listing).+}

**P1 LEAD-GAP RESOLUTION — MEASURED 2026-08-11 ~17:20Z (two runs, pod4): the instrument
was fixed AND the failure survived it — MODEL VERDICT "missing state variable".**
@@ -1779,7 +1791,10 @@ and rejecting the record for them discarded a good `goal_kind`.
and it says yes. Still open: the processor reports `fps=24` for a 2 fps sample (a temporal
mismatch that B4's hindsight premise depends on), `alpamayo_rows = 0` on every clip (engine D
contributed nothing), and SAM3's real API is installed but not yet wired.
Artifacts: [-`ph0_mini/v2/ph0_v2.json`-]{+`/workspace/ph0_mini/v2/ph0_v2.json`+} (every prompt + raw model output banked
per [-call),-]{+call) [⛔ UNBANKED — lived on pod4 (terminated) per this block's header, out-dir per+}
{+`stack/scripts/ph0_v2_chain.sh:20`; never committed, and NOT in the v5.8f HF release+}
{+(tree listed 2026-08-18)],+}
`ph0_mini/v2/viz/` (overlay MP4 + stills); instruments `stack/scripts/ph0_v2.py`,
`ph0_v2_chain.sh`, `ph0_v2_overlay.py`, 44 CPU tests.

@@ -1850,7 +1865,9 @@ R²(enc) ≤ 0, every transform failed, 2-layer MLP ceiling **−0.334**) — re
**the lead gap is not readable from this latent in any form yet probed**. (3) It converges with
the T1 result above: the closed loop is ~99 % longitudinal and over-accelerates (progress ratio
1.7279, speed bias +9.3892 m/s), and the model cannot see the vehicle in front of it. Artifacts:
[-`lf0-bev-lead/lf0_gate.json` (pod4);-]{+`/workspace/experiments/lf0-bev-lead/lf0_gate.json` (pod4, terminated) [⛔ UNBANKED — out-dir per+}
{+`stack/scripts/lf0_chain.sh:17`; never committed, and NOT in the v5.8f HF release (tree listed+}
{+2026-08-18)];+} instrument `stack/scripts/lf0_bev_lead.py` + `lf0_chain.sh`,
21 CPU tests.

**W7-PROG — MEASURED 2026-08-12 ~05:40Z [EXPLORATORY, 881 grid, pod4]: PRE-REGISTERED
@@ -1882,14 +1899,18 @@ anti-degeneracy weight, and W7-style self-consistency selection is retired as a
A cost that is *negatively* calibrated across windows is worse than an uninformative one, which is
the mechanism behind the ADE regression and independent evidence that this cost family is not
merely under-tuned. ⚠️ EXPLORATORY stamp holds: this re-uses the W7 scoring windows, so no arm
here is quotable as a v5.8f number. Artifacts: [-`w7-prog-{01,05}/w7_gate.json`-]{+`w7_gate.json`+} + `rules.json` [-(pod4).-]{+in each arm run dir,+}
{+w7-prog-01/ and w7-prog-05/, on pod4 [⛔ UNBANKED — pod terminated, never committed, and NOT in+}
{+the v5.8f HF release (tree listed 2026-08-18)].+}

Instrument change that made this possible: `t1_eval.py` now calls
`all_families(win, tactical_from_traj=True, tier=t)`. At T1 nothing steers the rollout but the
arm's own actions, so the driven path IS its manoeuvre decision; at T0 the same block is
stamped as substantially an ACTION ECHO so a teacher-forced tactical number can never be read
as skill. Artifacts: [-`four_families/ff_{stageA,v5f30k}_{cl,ol,ha}.json`-]{+the six per-tier JSONs ff_{stageA,v5f30k}_{cl,ol,ha}.json+} + `ff_comparison.json`
[-(pod5:`/workspace/experiments/t1-v58f/four_families/`).-]{+(pod5:`/workspace/experiments/t1-v58f/four_families/`, terminated — all seven durable, verified+}
{+2026-08-18 on HF `Sayood/tanitad-flagship-v5f-w120` at `release/v58f/gates/four_families/`, same+}
{+basenames, per the `stack/scripts/release_v58f.py` manifest + HF tree listing).+}

## 2. REF-A — the frozen-encoder arm (H4)

@@ -2790,7 +2811,7 @@ deleted so every previously published number stays traceable. Sources: the per-r
| 13 | Flagship **no-speed** (ablation control) | `flagship-nospeed` | ~22 000 | 263.4 M | 3.0175 [2.5450, 3.5444] <!-- src: taniteval/results/driving_flagship-nospeed.json#headline.ade_0_2s.mean --> | 5.0282 | 0.7423 | ✗ | *2.9176 ± 0.3558* |
| 14 | REF-A dyn-in 4B | `refa-dynin-30k` | 29 999 | — | 3.0471 [2.4984, 3.6878] <!-- src: taniteval/results/driving_refa-dynin-30k.json#headline.ade_0_2s.mean --> | 4.7642 | 0.7412 | ✗ | *2.9196 ± 0.3937* |
| 15 | Flagship **v2** (killed) | `flagship-v2-6k` | 6 000 | 272.9 M | 5.9396 [4.3273, 7.6249] <!-- src: taniteval/results/driving_flagship-v2-6k.json#headline.ade_0_2s.mean --> | 12.4011 | 0.8524 | ✗ | *6.179 ± 1.2845* |
| — | Flagship v1 tactical **head** (not rollout) | `plan_flagship-30k` | 29 999 | — | **3.3839** [2.8336, 3.9722] <!-- src: [-…/2026-08-16-jack-in-gates/raw/g1_g4_both_estimators.json#G1.arms.tactical_head-]{+TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-16-jack-in-gates/raw/g1_g4_both_estimators.json#G1.arms.tactical_head.corrected_mean+} --> ⚠️ **the "🟥 no windows dump — legacy only" that stood here is REFUTED (2026-08-16):** `clwin_flagship-30k.pt`'s `plan_direct` **is** this arm — it reproduces the legacy 3.1501 ± 0.3472 **bit-exactly at 4 dp**, and its `full_set` mean is 3.3839. Same stale-absence class (**C69**/**C70**) as the P2 row above | — | — | ✗ | *3.38 (3.150 ± 0.347 in the P2 pass)* |

*Arms recomputed but not ranked here (same 881 windows; full table in `jack_recompute.json`):*
**v1.6** `flagship-v16-ab-ft` 0.4375 [0.3423, 0.5501] (legacy 0.4886 — the largest single-arm bias in the
@@ -3615,7 +3636,8 @@ performance.** T1 capability claims live in §1.12, §1.13c and §5.
**Substrate:** frozen `v6F-SW-30k` snapshots (`/home/nvidia/ckpt_snaps`, fp16 weights-only).
**Estimator:** paired episode-cluster bootstrap throughout. **Instrument:** `pc6_linear_readout`
ridge — ⛔ **pass `intercept_col=-1`**; the default is deliberately the incumbent (biased) behaviour
so banked [-`pc6_ridge_*.json`-]{+`pc6_ridge_*` JSONs under `…/incoming/2026-08-17-probe-positive-control/raw/`+}
reproduce bit-exactly (C92).

### 12.1 ⛔ The 40:1 pooling bottleneck is REFUTED — the ENCODER is the constraint (C104, 2026-08-18)

@@ -3701,7 +3723,7 @@ fires** — `parity.py` §9 checks a cache against *its own* corpus digest, and
is a different corpus by construction. ⇒ **Whoever runs that build MUST call
`parity.filter_train_clips()` first.**

✅ **BLAST RADIUS ON PUBLISHED NUMBERS: ZERO** — all **73** [-`taniteval/results/*.json`-]{+JSONs in `taniteval/results/`+} opened;
`registry.py:288` lists three eval corpora, **none Alpamayo**; every registry hit sits in §11
*PRODUCED DATASETS*. The aug120 numbers that exist are **label-quality only**, and the ADE numbers
near the word "Alpamayo" are the **Alpamayo-2-Super model** on the 290-clip OOD-val corpus — a
```

## 7. Escalations

1. **Genuinely unrecoverable artifacts** (pods terminated, absent from the HF release, never in
   git history) — the registry now says so at each site instead of citing dead paths:
   `w7_eval_windows.pt` (W7-FULL per-window arrays), `lf0_gate.json`, the two W7-PROG
   `w7_gate.json` + `rules.json`, and `ph0_v2.json`. If any pod4/pod5 disk image or rescue dump
   still exists anywhere, these five are the pull list; otherwise the affected numbers rest on
   the registry text alone (the W7-FULL GATE json itself IS banked in-repo — only the per-window
   arrays are gone; `sel_winners_curse_law.py` already documents that consequence).
2. **Cheap banking opportunity (recommended follow-up, ~200 KB total):** the HF release
   `Sayood/tanitad-flagship-v5f-w120` `release/v58f/gates/` holds small JSONs the registry can
   currently only cite as remote: `i4a_{none,zero,shuffle}.json`, `w7_w4r_k32_gate.json`, and
   `four_families/ff_*.json` + the FULL `ff_comparison.json`. Pulling them into a hub incoming
   dir (md5 against the release `MANIFEST.json`) would upgrade five §1.14/§1.15 citations from
   "durable on HF" to in-repo EXISTS. Out of my scope tonight (it creates new artifacts beyond
   the named deliverables).
3. **Basename collision worth knowing:** in-repo `…/2026-08-07-hierarchical-wm-redesign/ff_rescore_val40_demo/ff_comparison.json`
   is a 3,916 B DEMO; the §1.15 artifact of the same name is the 55,705 B file on HF. The
   registry's `ff_comparison.json` name-citation sits directly beside the HF location now, so a
   reader lands on the right one — but anyone resolving by basename search will hit the demo
   first. Banking (2) dissolves this.
4. **The suite-vs-G: coupling** (§4) deserves a line in the repo docs / CLAUDE.md traps list:
   while G: is down, any pytest run in a clone silently imports NOTHING locally without the
   PYTHONPATH override, and the failure it produces (Errno 22 at collection) does not name G: at
   all. Same trap family as `df`-on-pods: a probe reporting the wrong scope, dressed as an answer.
5. **Brief's test paths were stale** (§ baseline correction): the registry lint tests live under
   `tools/tests/`, not `stack/tests/`.
6. Mid-task, HEAD advanced `64575912` → `c54017c5` (sibling's dump-lead-wiring commit; touches
   only `taniteval/` + its own hub dir — VERIFIED disjoint from the registry). Staging was
   re-verified after that move, per the CLAUDE.md end-of-turn rule.
7. **Two pre-existing `stack/tests` failures, neither mine to fix** (§5): (a)
   `test_v4_labels::test_mintability_report_names_the_gaps` — the lead-wiring stream's commit
   `4276613a` retired the `lead_state` wording inside `stack/scripts/v4_labels.py` but not the
   test's grep token; one-line fix belongs to that stream. (b)
   `test_loss_determinism` negative control — `_site()`'s `"TanitAD" in filename` repo test
   breaks in any differently-named clone/worktree; deriving the root from the test file's own
   `_STACK` path would fix it for every checkout. Flagged, not fixed: both sit outside my named
   scope (`stack/tests` is not mine; the brief allows test edits only for the two registry lint
   tests, and only on proven layout migration).
