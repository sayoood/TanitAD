# `maneuver_acc` was scored against v1 labels for every `--v2` arm — fixed, pinned, and **not re-scorable from anything we banked**

**MEASURED 2026-08-17 (ours)** · **0 GPU, 0 pods contacted** · raw record:
`raw/maneuver_label_mismatch.json` · dev box `C:/Users/Admin/venvs/tanitad`.

Escalated by `…/incoming/2026-08-17-diryaw-reread/DIRYAW_REREAD.md` §1.5 / §5 item 3.
**⛔ `MODEL_REGISTRY.md` NOT TOUCHED. `TANITAD_PAPER.md` NOT TOUCHED. Nothing committed, nothing
pushed.** The rows the PI must correct are listed in §6.

---

## ⛔ ESCALATION TO THE PI — three things, first

1. ⭐ **The blast radius is ONE banked number set, not sixteen.** `flagship-v2corpus-30k` is the
   **only** `--v2` arm with a banked hierarchy panel. Its `seam_ctx_to_tactical.maneuver_acc`
   (**real 0.0239**, n = 418 / 19 eps) is scored against v1 net-yaw labels on a curvature-trained
   head. The **deployed v1** (`flagship4b-speedjerk-30k`) and **`flagship-v1arch-v2bal-30k`** are
   **v1-labelled and completely clean** — no registry headline moves.
2. ⛔ **I cannot give you the corrected number, and no weaker estimator can either.** `maneuver_acc`
   needs `man_pred` — the tactical head's argmax — and **`man_pred` is on disk NOWHERE** (four
   independent probes, §4). The lf19 episode poses are not in the repo either. **A GPU re-run of
   `hierarchy.run` with `labels_v2=True` is the only path.** I have not substituted anything.
   ⭐ **But the instrument gap that caused this is now closed** (§4.1): the panel banks the
   per-window arrays it used to drop **and derives BOTH label families at scoring time**, so this
   class of question will never again cost a GPU pass. That also closes `DIRYAW_REREAD.md` §5 item
   1 — the reason **no Δκ in this programme has ever carried an interval**.
3. ⚠️ **This fix does NOT clear the paper's κ-collapse.** `TANITAD_PAPER.md:1867-1868`'s
   0.253 → 0.0072 is built from `man_pred` vs `traj_dir` and **never touches `man_tgt`**. It stays
   exactly where the DIR_YAW re-read left it: confounded, and stacked on a genuinely degenerate
   head. Do not let the two be merged in a single retraction.

---

## 1. The defect, from source

`taniteval/taniteval/hierarchy.py:592` at HEAD:

```python
man_tgt = rl.classify_maneuver(pl[:, 2], fut[:, GOAL_H - 1, 2],
                               pl[:, 3], fut[:, GOAL_H - 1, 3]).long()
```

— the **v1** labeler, with no branch on the arm's `cfg.v2_labels`. It reaches
`rec["man_corr_real"]` (`:639`), `rec["man_corr_{mean,zero}ctx"]` (`:685-686`) →
`seam_ctx_to_tactical.maneuver_acc` (`:820`) → `load_bearing` / `content_matters` (`:829-833`) →
`verdict` (`:1267`) and the legacy block (`:1079`, `:1107`, `:1137`).

⛔ **It is not a threshold disagreement.** v1's turn test is `|dyaw| > YAW_TURN_RAD = 0.15` **rad**;
v2's is `|dyaw/arc| ≥ CURV_TURN_MAN_PER_M = 1/60` **per metre** (`refb_labels.py:107-108` vs
`:344`). Different physical quantities — no gate value makes them commensurable, which is why the
fix is a branch and not a constant.

⚠️ **And the shift is not even signed per window.** v2 moves a gentle motorway curve *out* of the
turn classes, but at low speed a short arc gives high curvature under a small net yaw, so v2 can
also declare a turn v1 misses. **Do not assume the correction only lowers turn counts.**

---

## 2. Blast radius — which arms, from `MODEL_REGISTRY.md` only

### 2.1 The three v2-labelled arms

| § | arm | registry evidence (verbatim) | banked hierarchy panel? |
|---|---|---|---|
| **1.3** | `flagship4b-v2-30k` | Exact command = §1.2 plus `--v2` *"(which implies `--speed-input`, `--labels-v2`…)"*; lever table `v2_labels` **`true`** | ⛔ **none** |
| **1.4** | `flagship4b-v3enc-30k` | Exact command read live from `ps`: `--speed-input --v2 --staged-levers`; *"every **decode-side** lever (anchored tactical, gated intent, goal decode, **labels-v2**, jerk…)"* unchanged from v2 | ⛔ **none** |
| **1.7** | `flagship-v2corpus-30k` | Exact command carries `--v2`; Levers row `v2_labels true` *"read from **both** config.jsons and confirmed equal to §1.3"* | ✅ **yes — the one affected panel** |

⚠️ **§-number correction to `DIRYAW_REREAD.md`.** It cites v3enc as *"§1.6.x"* (`:97`) and
*"§1.6/§1.7"* (`:269`). Per the registry's own headings, v3enc is **§1.4** (`MODEL_REGISTRY.md:443`)
and §1.6 (`:944`) is the numberless "variants that are not versions" table. The *fact* is unchanged;
the citation is wrong.

### 2.2 Confirmed UNAFFECTED (so nobody re-opens them)

| § | arm | why |
|---|---|---|
| **1.2** | **`flagship4b-speedjerk-30k` — THE DEPLOYED v1** | exact-command row carries no `--v2`, no `--labels-v2` |
| **1.9** | `flagship-v1arch-v2bal-30k` | *"Every `v2_*` lever in its own `config.json` is `false`"* — `v2_labels` named in that list |
| **3.5** | `refb-refbpatch-v2-30k` | `--arch-v2` is an **architecture** flag, not `--labels-v2` |
| **4.3** | REF-C line | *"`labels_v2` was never set in `refc_train.py`"* |
| **1.5.x** | v4 line | `train_flagship_v4.py --labels v3` is the **route**-label axis, a different flag |

### 2.3 The one affected banked panel — BEFORE, in full

`…/Evaluation/Implementation/incoming/2026-08-02-four-family-panel/hier_v2corpus-lf19.json`
(md5 `a1b12be962bfb28a19f1512c42b85eed`, **byte-identical duplicate** at
`stack/experiments/pod-rescue-20260802/eval/root/taniteval/results/hier_v2corpus-lf19.json`).
Arm `flagship-v2corpus-30k`, arch `flagship-worldmodel-v2`, `ckpt_step` **29999**,
**n = 418 windows / 19 episodes**. Estimator on every delta:
`paired_episode_cluster_bootstrap`, B = 2000.

| `seam_ctx_to_tactical` member | real | mean_ctx | zero_ctx | Δ(real−mean) | CI95 | separated | **label-dependent?** |
|---|---|---|---|---|---|---|---|
| **`maneuver_acc`** | **0.0239** | **0.0239** | 0.0526 | **+0.0000** | [+0.0000, +0.0000] | no | ⛔ **YES — mis-scored** |
| `wp_ade_2s` (m) | 19.603 | 19.4487 | — | −0.1542 | [−0.9281, +0.5089] | no | ✅ no |
| `goal_latent_cos` | 0.4609 | 0.4616 | — | −0.0007 | [−0.0029, +0.0015] | no | ✅ no |

`load_bearing = false`. `maneuver_acc.delta_real_vs_zero` = **−0.0287** [−0.0622, −0.0024]
(two-sided separated, `separated_positive` false).

The same-grid **v1** control `hier_v1-lf19.json` (`flagship4b-speedjerk-30k` @ 30000, n = 418/19) is
**unaffected**: `maneuver_acc` real **0.6746**, mean_ctx 0.6818, Δ −0.0072 [−0.0766, +0.0622].

⚠️ **What the "0/3 seams beneficial" verdict rests on.** Two of the three members are
label-independent and neither is close to its floor, so **the only way that verdict could move is
through `maneuver_acc`** — and its post-fix value is not derivable offline (§4). ⛔ I therefore do
**not** claim the verdict survives; I claim it is **at risk on exactly one of three members**.

### 2.4 Where the affected number is quoted downstream

⚠️ These are **documents, not the registry** — listed so the PI can act, per §6.

* `Project Steering/V5_FLAGSHIP_DEEP_REVIEW.md:73` — *"hierarchy seams beneficial 0/3 | 0/3"*
  (the second column is v2corpus).
* `TANITAD_PAPER.md:1868-1883` — the *"0/3 seams beneficial on both"* sentence.

**No `maneuver_acc` number for a v2 arm appears in `MODEL_REGISTRY.md` at all** — §1.9's
`seams_beneficial_of_3 = 0` and TACTICAL κ 0.6033 belong to the **v1** `flagship-v1arch-v2bal-30k`,
which is clean. ⇒ **No registry row's *value* changes because of this defect.** (§6 still asks for
two provenance notes.)

---

## 3. The fix — one definition, resolved the way the trainer already resolves it

⛔ **No new resolution rule was invented.** The trainer's rule is `cfg.v2_labels` on `StackConfig`
(`stack/tanitad/config.py:272`), set by `--v2` and overridden by `--labels-v2` / `--no-labels-v2`
(`train_flagship4b.py:361, 398-399`). The eval loader **already rebuilds each arm's `cfg` from its
own `config.json`** — it simply never handed it to the panel. The fix hands it over.

| # | file | change |
|---|---|---|
| 1 | `stack/scripts/refb_labels.py` | **NEW** `window_maneuver_labels_for(pose_last, future_poses, horizon, *, v2)` — the single flag→labeler mapping, next to the two functions it selects between. `v2` is **keyword-only and REQUIRED**: a caller that has not resolved the arm's family fails at the call site instead of inheriting a plausible default. |
| 2 | `stack/scripts/train_flagship4b.py` | `FlagshipWindowDataset.__getitem__`'s `if self.labels_v2: … else: …` replaced by that one call. **This is what makes it ONE definition** — the duplicate branch is gone. |
| 3 | `taniteval/taniteval/loaders.py` | **NEW** `resolve_labels_v2(entry, cfg)`; `load()` now returns `cfg` and `labels_v2`. A registry entry MAY declare `labels_v2`, but a declaration that **contradicts** the run config **raises** — a silent disagreement here is the exact failure this ends. |
| 4 | `taniteval/taniteval/hierarchy.py` | `run(..., labels_v2=False)`; `:592` → `rl.window_maneuver_labels_for(pl, fut, horizon=GOAL_H, v2=bool(labels_v2))`; **NEW** `MANEUVER_LABEL_KEY` stamp at panel top level **and** inside `seam_ctx_to_tactical`, naming the exact labeler; **NEW** `per_window` block and `maneuver_acc_under_other_label_family` (§4.1). |
| 5 | `taniteval/taniteval/rollout.py` | `_man_gt(..., labels_v2=False)` and `collect(..., labels_v2=False)` — same defect, feeds `win["maneuver_gt"]` → four-families TACTICAL. |
| 6 | `taniteval/taniteval/planning.py` | `run(..., labels_v2=False)`; `:145` same fix. |
| 7 | `taniteval/taniteval/runner.py` | `run_hierarchy` passes `labels_v2=bool(L["labels_v2"])`. |
| 8 | `taniteval/tools/eval_four_families.py` | passes it to **both** `hierarchy.run` and `rollout.collect`, and prints `maneuver_labels=v1\|v2` in the progress line. |

⭐ **Every future panel is self-describing.** A JSON with no `maneuver_label_version` predates this
fix and used **v1 regardless of how the arm was trained** — the note says exactly that, so a bare
number can never be mis-read again the way it was for a year.

### 3.1 The refactor is proven INERT — no v1 arm can move

**MEASURED**, random-draw equivalence of the new call against the exact pre-fix expressions:

| path | mismatches |
|---|---|
| trainer v1 branch (`classify_maneuver(p_last[2], p1[2], …)`) | **0 / 2000** |
| trainer v2 branch (`window_maneuver_labels_v2`) | **0 / 2000** |
| `hierarchy.py` v1 path, batched | **0 / 500** |
| `rollout._man_gt` v1 path, incl. episode-tail clamping | **0 / 500** |

⇒ no published v1 number and no training target can shift.

### 3.2 ⚠️ A pin was updated DELIBERATELY — flagging it because the PI owns that decision

`stack/tests/test_v5_trainer_v2_val.py::test_train_flagship4b_is_untouched_by_this_change` hashes
`train_flagship4b.py`. Change #2 moved it. Per **that test's own instruction** I confirmed the edit
is intended, confirmed no arm is mid-run on this trainer (the live 30k is `train_v6_staged.py`),
recorded the reason in its docstring, and updated the hash
`702a683b…` → **`371d8786…`**. The inertness proof above and the unchanged, green
`tests/test_labels_v2_wiring.py` — which asserts the emitted `maneuver_label` against direct
`refb_labels` calls on **both** settings — are why the edit is safe. **It is not a silent
re-baseline.**

---

## 4. ⛔ THE RE-SCORE: not possible from banked artifacts. Stated plainly, with the probes.

`maneuver_acc = mean(man_pred == man_tgt)`.

* `man_tgt` — **recomputable** at zero GPU *if* the poses are available.
* `man_pred` — the tactical head's 5-way argmax. **Requires the checkpoint and a forward pass.**

**Four independent probes, all negative:**

| probe | result |
|---|---|
| recursive key walk over **all 1,929 JSON files** in the 8 content roots for list-valued `man_pred` / `maneuver_pred` / `man_tgt` / `man_corr_real` / `maneuver_gt` / `man_corr_meanctx` | **0 hits** |
| **all 60** `rollout.save_windows` dumps (`taniteval/results/windows_*.pt` + every `incoming/**/*.pt`) | keys are `pred/gt/cv/eid/speed/head_deg/wp_steps` (+`pred_dense/gt_dense` on 2). **No manoeuvre or route key on any file.** `decision_fn` — the only code path that would write `maneuver_pred`/`maneuver_gt` — was never used to produce a banked dump. |
| repo-local search for `ep_*.pt` episode caches (the **poses**) | **none in the repo** — the val40 cache lives on Thor. So even `man_tgt` cannot be re-derived here. |
| the repo's own prior findings — `recompute_hierarchy_seams.py:16-19` (*"the paired episode-cluster bootstrap needs the per-window arrays, which the hierarchy panel does not persist"*) and `DIRYAW_REREAD.md` §3.4 | agree independently |

⇒ **There is no BEFORE→AFTER table in this report for `maneuver_acc`, because producing one would
require inventing a number.** The BEFORE column is §2.3 in full; the AFTER column needs one GPU
pass. ⛔ **I did not substitute `overlapping_holdout_se` or any other estimator to fill it.**

### 4.1 ⭐ Instrument gap — DIAGNOSED, AND CLOSED IN THIS CHANGE

`hierarchy.py` built every array this needed **in memory** and persisted only their means. That is
the same omission `DIRYAW_REREAD.md` §3.4 found from the other side — `gt_net_yaw` / `traj_net_yaw`
built and dropped, so **no Δκ in this programme has ever carried an interval**. One writer, one
fix, so I did both rather than filing a second work item.

**Closed:**

1. **`per_window` block** — `eid`, `man_tgt`, `man_tgt_alt`, `man_pred`, `man_pred_meanctx`,
   `man_pred_zeroctx`, `gt_dir`, `traj_dir`, `gt_net_yaw`, `traj_net_yaw`, `valid`, aligned
   row-for-row. This is exactly what a **paired episode-cluster bootstrap on Δκ** needs, and it is
   the first time it will exist. It carries `complete: true/false` and names any missing key —
   ⛔ a short block must never *look* finished.
2. ⭐ **Both label families are now derived at scoring time.** The panel emits
   `seam_ctx_to_tactical.maneuver_acc_under_other_label_family` — the **same predictions** scored
   against the other family's targets — plus `label_disagreement_rate`, the share of windows on
   which the families disagree at all. **Cost: one extra labeler call on tensors already in hand.**
   `test_the_other_familys_number_is_banked_so_a_rescore_costs_no_gpu` asserts this diagnostic is
   **bit-equal to an actual re-run under the other family**, so it is a measurement, not a promise.
   ⇒ **This exact defect can never again cost a GPU pass to correct.**
3. Panels assembled from a pre-2026-08-17 record emit
   `{"status": "UNAVAILABLE", "reason": …}` — the contract `gate_sensitivity` already uses. A
   quietly-absent number reads as a number nobody needed.

⚠️ **Cost:** ~+60 KB of JSON per panel (11 arrays × ~880 windows), against one GPU re-run per
question. ⚠️ **What is still NOT banked:** the raw `pose_last` / `future_poses` (~1.5 MB/panel).
Banking `man_tgt_alt` makes them unnecessary for the v1↔v2 question specifically; a *third* label
family would still need a re-run. Stated so nobody assumes more coverage than exists.

⛔ **This does NOT retroactively fix the banked panel.** `hier_v2corpus-lf19.json` was written
before any of it. §4's verdict stands: **that number still needs one GPU pass.**

### 4.2 The one thing that IS measurable now, and its limit

`MODEL_REGISTRY.md` §1.7 publishes the label-family gap on that arm's own **train** corpus
(`physicalai-v2bal-4b7eeeac222d`, 9,000 clips): **28.04 % turns under v1, 18.83 % under v2 — the
same clips either way.** ⚠️ That is the **train** corpus, not the 418-window lf19 val grid, and per
§1 the per-window shift is not signed. It establishes that the families disagree **materially**; it
does **not** bound the correction. Evidence class: **PUBLISHED (registry, cited)** — do not transfer it.

---

## 5. Every other label call in the eval path, and its conditionality

⚠️ Absence at one location is not absence — this is the whole eval path, not just `hierarchy.py`.

| site | call | conditional at HEAD? | feeds | action |
|---|---|---|---|---|
| `hierarchy.py:592` | `classify_maneuver` | ⛔ no | `maneuver_acc`, seam verdict | ✅ **FIXED** |
| `planning.py:145` | `classify_maneuver` | ⛔ no | `plan_*.json` manoeuvre acc / turn recall (paper `:1236`) | ✅ **FIXED** (⚠️ this panel has **no caller in the repo**) |
| `rollout.py:118` `_man_gt` | `classify_maneuver` | ⛔ no | `win["maneuver_gt"]` → four-families TACTICAL | ✅ **FIXED** (no banked dump ever used it) |
| `hierarchy.py:599-600` | `nav_command` / `route_target` | ⛔ no | `seam_nav_to_strategic` | ⛔ **NOT fixed — see below** |
| `planning.py:151,153` | same | ⛔ no | `route_acc_follow` | ⛔ not fixed |
| `strategic_probes.py:284`, `:118` | `nav_command` / `route_target` | ⛔ no | HP-3 route-swap counterfactual | ⛔ not fixed |
| `refc_eval.py:90` | `nav_command` | ⛔ no, but **scoped** to `nav_mode == "oracle"` | REF-C oracle upper bound, explicitly *"never a leaderboard number"* | report only |
| `four_families.py:800-801` | `factor_from_kinematics(...)` with **`kappa` omitted** ⇒ `refc_tactical`'s **v1** branch | ⛔ no | TACTICAL family of every four-families panel | ⚠️ report only — see below |
| `label_overlay.py:220` | `window_maneuver_labels_v2` | ⛔ no — **hardcoded to v2** | video overlay only | harmless mirror-image hardcode |
| `blind_baseline.py:731-739` | `route_target` / `_v2` / `_v21` | ✅ **yes, all three** | circularity registry | ⭐ **the pattern to copy** |
| `stack/tanitad/replay/arms.py:546` | `window_maneuver_labels` (v1) | ⛔ no | ⚠️ a **second, unrelated** `maneuver_acc` in `replay/stats.py:120` | ⚠️ **name collision** — a repo-wide grep hits both. Correct today (REF-B replay arms are v1); would mis-score a v2 arm. |

**Why the route/nav sites were NOT fixed, deliberately.** There are **three** route label versions
(v1; v2 via `cfg.v2_labels`; **v2.1** via a *separate* flag `cfg.v21_route_labels`). A single
boolean cannot express three, and `MODEL_REGISTRY.md` §1.3/§1.4 record arms trained on the
**pre-v2.1 (broken)** route labels whose route readings the registry already calls
*"uninterpretable"*. ⛔ Guessing a mapping here would be a **new defect**, not a fix. It needs its
own pre-registered pass. Scope stated, not narrowed.

**Why `four_families.py:800-801` was not changed.** Both raters there share the same gate (the
arm's driven path vs the human's driven path), so it is **self-consistent** — a label-*definition*
question, not "a head scored against labels it never saw". Different severity, different fix.

---

## 6. What the PI must correct — I changed none of these

⛔ **I did not edit `MODEL_REGISTRY.md` or `TANITAD_PAPER.md`.**

| # | where | what |
|---|---|---|
| 1 | `MODEL_REGISTRY.md` §1.7 (`flagship-v2corpus-30k`) | add a provenance line: its banked `seam_ctx_to_tactical.maneuver_acc` (**0.0239**, n = 418/19) was scored under **v1** labels while the arm trained on **v2**; **withdrawn pending a `labels_v2=True` re-run**. **No published value in this section changes.** |
| 2 | `MODEL_REGISTRY.md` §1.4 (`flagship4b-v3enc-30k`) | note that **any future** hierarchy eval of `ckpt_step10000.pt` must run with `labels_v2=True` (now automatic via `--run-config`). Nothing banked, nothing to retract. |
| 3 | `TANITAD_PAPER.md:1868-1883` | the *"0/3 seams beneficial on both"* clause: the v2corpus side's `maneuver_acc` member is mis-scored. The other two members are label-independent, so the sentence is **at risk on one of three**, not refuted. |
| 4 | `Project Steering/V5_FLAGSHIP_DEEP_REVIEW.md:73` | same caveat on the *"0/3 \| 0/3"* row. |
| 5 | `TANITAD_PAPER.md:1867-1868` (κ 0.253 → 0.0072) | ⚠️ **unchanged by this fix.** It is a different defect (`man_pred` vs `traj_dir` rater mismatch, `DIRYAW_REREAD.md` §1.3). Keep the two retractions separate. |
| 6 | `RETRACTION_LOG.md` | root-cause class: **"a scoring path hardcoded one side of a versioned definition while the producing path branched on it"** — the same class as the nav-echo and the C6 marginal-decoder confounds: *an instrument that cannot see a distinction the thing it measures was built around.* |
| 7 | `DIRYAW_REREAD.md:97, :269` | §-citation fix: v3enc is **§1.4**, not §1.6.x. |
| 8 | `Project Steering/BACKLOG.md` | ✅ **`DIRYAW_REREAD.md` §5 work items 1 and 3 can both be CLOSED** — item 3 (branch `man_tgt`) is this fix; item 1 (persist the per-window gate inputs) is §4.1. Item 2 (sweep `kappa_turn_subset`) and item 4 (GPU re-score of v2corpus) remain OPEN. |
| 9 | `Project Steering/BACKLOG.md` | **NEW item, 0 GPU:** the nav/route label family (§5) — needs its own pre-registered pass because there are three versions, not two. |

---

## 7. Suite

`cd stack && PYTHONUTF8=1 OMP_NUM_THREADS=6 pytest -q` — quoted exactly, nothing varied.

⚠️ **One environment note, reported because it is a trap:** bare `pytest` on this dev box resolves
to `C:/Users/Admin/AppData/Local/hermes/hermes-agent/venv/Scripts/pytest`, which collects **190
errors in 17 s** and looks like a catastrophic regression. The project interpreter is
`C:/Users/Admin/venvs/tanitad/Scripts/pytest`. Same trap family as reading `df` on a pod: a probe
answering about the wrong scope. Only the resolution of the word `pytest` was changed; no flag, no
env var was added or removed.

* **Pre-fix baseline reproduced:** 3750 passed / 0 failed / 7 skipped / 2 xfailed — as stated in the
  brief.
* **With the source fix, before the pin update:** 3749 passed / **1 failed** — the single failure
  was the deliberate `train_flagship4b.py` hash pin (§3.2), which is the guard doing its job.
* **Final:** **3765 passed / 0 failed / 7 skipped / 2 xfailed** = baseline 3750 + the 15 new pin
  tests.

⚠️ **The `taniteval` package has its OWN suite and it is NOT in that 3750** — `stack/pyproject.toml`
sets `testpaths = ["tests"]`, so `cd stack && pytest` never collects `taniteval/tests`. Since most
of this change lives in `taniteval/`, that suite was run too:
`cd taniteval && pytest -q tests` (with `stack`, `stack/scripts`, `taniteval` on `PYTHONPATH`) —
**1092 passed / 0 failed**. ⚠️ It caught a real back-compat break during this work:
`test_hierarchy_ci.py` calls `_assemble` on a synthetic record, so the §4.1 arrays were absent and
the new block raised `KeyError`. Fixed by emitting `status: UNAVAILABLE` with a reason instead —
**the guard did its job; I did not relax it.** The stack suite alone would not have caught this.
⚠️ Do **not** use `taniteval/tests/run_all.py` to judge this: it is the pod's no-pytest fallback and
cannot supply fixtures, so it reports **160 spurious `TypeError: missing positional argument`
failures** on fixture-based tests **whether or not anything is broken**. That is a property of the
runner, not a regression — and it is exactly the "a probe answering the wrong question looks like an
answer" trap. The new pin file was deliberately placed in `stack/tests/` so it is inside the
mandated suite.

⚠️ **One further guard fired and was respected, not weakened.**
`taniteval/tests/test_ego_guard.py::test_the_trainer_is_untouched_so_pod1_stays_resumable` asserts
the literal string of the eval package name never appears in `train_flagship4b.py`, so the trainer
cannot drift into the eval import graph. My first comment named the eval module and tripped it. **I
reworded the comment; I did not touch the guard.** (Trainer hash therefore settled at
`371d8786…`, not the intermediate `4595f68e…`.)

---

## 8. The pin, and its NON-VACUITY proof

`stack/tests/test_maneuver_label_family.py` — **15 tests, 0 GPU, 0 corpus, 0 checkpoint.**

Everything behavioural runs on one fixture, `_gentle_curve` — a constant-radius 200 m left curve at
20 m/s — on which the two families **provably disagree**: net yaw **0.2 rad > 0.15** (v1 says
`turn_left`) while curvature **0.005 < 1/60 per m** (v2 says `lane_keep`). It is exactly the case
`classify_maneuver_v2`'s own docstring names. `test_fixture_actually_separates_the_two_label_families`
asserts that separation first, so nothing below it can pass vacuously.

The pin includes an **executable end-to-end `hierarchy.run`** over stub brains (3 episodes, 12
windows), not only a signature check:
`test_hierarchy_maneuver_acc_actually_moves_with_the_family` asserts the two accuracies **sum to
1.0** — arithmetic from the geometry, not a golden file. The stub's manoeuvre head is restricted to
`{lane_keep, turn_left}` precisely so that identity holds.

⭐ **NON-VACUITY, MEASURED — not asserted.** The eight fixed files were restored to their HEAD
(pre-fix) content in place and the pin was run against that code path:

**12 of 13 tests then present FAILED** — `hierarchy.run`, `planning.run` and `rollout.collect` had
no `labels_v2` parameter at all, `loaders.resolve_labels_v2` did not exist, the dispatcher did not
exist, and the source guard found the unconditional `rl.classify_maneuver(` call in all three eval
modules. The files were then restored from a scratch copy (**no git operation** — no `checkout`, no
`stash`, no branch switch) and the pin went green again. The only test that passed pre-fix is the
fixture-separation proof, which is a fact about `refb_labels`, not about the fix. The two §4.1
banking tests were added afterwards and also fail pre-fix (`PER_WINDOW_KEYS` did not exist).

---

## Deliverable manifest

| artifact | repo path | state |
|---|---|---|
| `MANEUVER_LABEL_MISMATCH.md` (this file) | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-17-maneuver-label-mismatch/` | staged |
| `raw/maneuver_label_mismatch.json` (every number above, machine-readable, incl. both md5s) | same dir | staged |
| the dispatcher | `stack/scripts/refb_labels.py` | staged |
| trainer routed through it | `stack/scripts/train_flagship4b.py` | staged |
| `resolve_labels_v2` + handle | `taniteval/taniteval/loaders.py` | staged |
| the fix + version stamp + `per_window` banking + cross-family diagnostic | `taniteval/taniteval/hierarchy.py` | staged |
| same fix | `taniteval/taniteval/rollout.py`, `taniteval/taniteval/planning.py` | staged |
| callers threaded | `taniteval/taniteval/runner.py`, `taniteval/tools/eval_four_families.py` | staged |
| the pin (15 tests) | `stack/tests/test_maneuver_label_family.py` | staged |
| trainer hash pin, deliberately updated + reason recorded | `stack/tests/test_v5_trainer_v2_val.py` | staged |

**Nothing committed, nothing pushed, no branch switched. No GPU used, no pod contacted.
`MODEL_REGISTRY.md`, `TANITAD_PAPER.md` and `…/2026-08-17-w120val-sign-adjudication/` untouched.**
