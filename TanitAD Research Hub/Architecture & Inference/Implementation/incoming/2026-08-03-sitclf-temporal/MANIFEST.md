# sitclf TEMPORAL — DELIVERABLE MANIFEST

**Date** 2026-08-03 · **Stream** situation classifier / temporal content · **0 pod GPU-h** — the
whole study ran on the dev box (RTX 4060). No pod was touched; `tanitad-new` and `tanitad-pod4` were
left training. **Scope owned:** `stack/tanitad/eval/sitclf.py`, `stack/tanitad/data/situations.py`,
`stack/scripts/emit_situation_labels.py`, `sc_train*.py`. ⛔ `stack/tanitad/refs/` NOT touched.

---

## 1. Where every artifact lives

| artifact | path | what it is |
|---|---|---|
| **pre-registration** | `…/incoming/2026-08-03-sitclf-temporal/PRE_REGISTRATION.md` | **REPO, STAGED.** Written and staged **before** any held-out number from this run was read. Required by the parent pre-reg §7 ("any follow-up is a new pre-registration"). |
| **report** | `…/2026-08-03-sitclf-temporal/TEMPORAL_HYPOTHESIS.md` | **REPO, STAGED.** |
| the experiment | `…/2026-08-03-sitclf-temporal/run_temporal.py` | **REPO, STAGED.** 21-arm ladder (window × motion basis × architecture), all controls, the four families |
| results | `…/2026-08-03-sitclf-temporal/results_temporal.json` | **REPO, STAGED.** every arm, interval, control, verdict |
| per-row scores | `…/2026-08-03-sitclf-temporal/results_temporal.scores.npz` | **REPO, STAGED.** out-of-fit scores for **all 42 arms** (21 real + 21 permuted-feature nulls) — a 0-GPU re-analysis surface |
| **subspace diagnostic** | `…/2026-08-03-sitclf-temporal/subspace_diag.py` → `results_subspace.json` | **REPO, STAGED.** the label-free, classifier-free measurement that settles H-T2. ⭐ **This is the quotable subspace artifact** — see §3. |
| independent verifier | `…/2026-08-03-sitclf-temporal/verify_scores.py` | **REPO, STAGED.** re-derives every headline from the banked scores alone, so the JSON cannot agree with itself |
| table renderer | `…/2026-08-03-sitclf-temporal/render_tables.py` → `tables.md` | **REPO, STAGED.** every table is generated from the JSONs, never retyped |
| run logs | `…/2026-08-03-sitclf-temporal/run_log.txt`, `run_log_subspace.txt`, `run_log_verify.txt` | **REPO, STAGED.** |
| **temporal primitives** | `stack/tanitad/eval/sitclf.py` | **REPO, STAGED.** `temporal_difference`, `diff_reparam`, `undiff_reparam` |
| **tests** | `stack/tests/test_sitclf.py` (+8, 31 total in file) | **REPO, STAGED.** including the instrument contract in §2 below |

### Deliberately NOT in the repo

| artifact | where | why, and how to regenerate |
|---|---|---|
| the feature substrate (410 MB) | `C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz` + `.meta.json` | 99,477 × 2048 fp16 v1 latents, built by BACKLOG B4. Too large for git and fully reproducible with `…/2026-08-03-sitclf-matched-capacity/build_substrate.py`. Its `.meta.json` is copied into the B4 run directory and its provenance block is embedded in `results_temporal.json`. |

**Nothing in this stream lives only on a pod or only in a worktree.**

---

## 2. ⭐ The instrument contract — why a null here is interpretable

A null result ("a motion basis does not beat the appearance basis") is worthless unless the
instrument could have detected a motion gain had one existed. `stack/tests/test_sitclf.py::
test_a_MOTION_ONLY_signal_is_INVISIBLE_to_appearance_pca_and_VISIBLE_to_motion_pca` builds the case
where the answer is known by construction — a large-variance appearance direction carrying a slow
walk, a tiny-variance direction carrying a fast flip, and a label depending only on the sign of the
change along the latter. It asserts that a rank-1 **appearance** PCA lands near chance (< 1.35×
base) and a rank-1 **motion** PCA recovers the label (> 3× base). **If that test ever fails, the
motion arms in `run_temporal.py` never had a chance and no null from them may be quoted.**

Companion contracts already in the file: `test_causal_window_is_causal_and_never_crosses_a_clip`
and `test_head_learns_a_signal_that_needs_a_LONG_window` (win=16 reaches AP 1.00 on a target
decidable only 12 frames back, win=4 sits at chance) — the window plumbing is validated too.

---

## 3. ⚠️ One superseded block inside a banked artifact — read this before quoting

`results_temporal.json` → `controls.H_T2_SUBSPACE_DIAGNOSTIC` was computed with the temporal
difference centred on the **appearance** mean rather than its own, which inflates the appearance
basis's apparent share of the Δ variance (reported 0.9520 at rank 16; correct value **0.8808**). The
defect was found after the hour-long fit had passed that stage.

**The block is left untouched rather than edited after the fact.** `results_subspace.json` carries a
`_supersedes` pointer and **is the quotable artifact** for every subspace number.
`run_temporal.py` has been corrected so a re-run is right.

Second defect, same class, also disclosed rather than hidden: in this run the `NEG_FEAT__CPOS_*`
columns were **byte-identical to their real arms**, because `build_features` did not apply the clip
permutation to the ego block. It never touched any **vision** arm's null and never entered the C-POS
predicate (which reads `paired_vs_reference`), but the `paired_vs_own_null` column of the two
`CPOS_*` rows is degenerate by construction and **must not be read**. `run_temporal.py` is fixed.

---

## 4. Escalations

*(filled at the end of the run)*
