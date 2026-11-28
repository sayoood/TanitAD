# sitclf PER-SITUATION HORIZON — DELIVERABLE MANIFEST

**Date** 2026-08-03 · **Stream** situation classifier / per-situation `(window, lead_s)` ·
**0 pod GPU-h** — the whole study ran on the dev box. No pod was touched; `tanitad-new` (v5f) and
`tanitad-pod4` (v1arch) were left training.
**Scope owned:** this run directory, the B4 substrate's ego sidecar, and **one additive function**
in `stack/tanitad/eval/sitclf_deploy.py` (+ its tests). Nothing existing in `stack/` was changed.

---

## 1. Where every artifact lives

| artifact | path | what it is |
|---|---|---|
| **pre-registration** | `…/2026-08-03-sitclf-horizon/PRE_REGISTRATION.md` | **REPO, STAGED.** Written and staged **before** any number in this study was read (staging verified with `git ls-files --cached`, not with an exit code). Both outcomes, the registered personal prediction and the explicit "no per-situation gain" branch are in it |
| **report** | `…/2026-08-03-sitclf-horizon/PER_SITUATION_HORIZON.md` | **REPO, STAGED.** |
| **tables** | `…/2026-08-03-sitclf-horizon/TABLES.md` ← `render_tables.py` | **REPO, STAGED.** Generated from the JSONs; nothing retyped |
| the experiment | `…/2026-08-03-sitclf-horizon/run_per_situation_horizon.py` | **REPO, STAGED.** the 5×5 grid × 2 substrates, the selection procedure, all five controls, both questions |
| results | `…/2026-08-03-sitclf-horizon/results_horizon_ps.json` | **REPO, STAGED.** every cell, every arm, every interval, the pre-registered verdict |
| per-row scores | `…/2026-08-03-sitclf-horizon/results_horizon_ps.scores.npz` | **REPO, STAGED.** out-of-fold score columns for all 25 grid cells + 7 arms, plus the onset universe — a 0-GPU re-analysis surface |
| **C-POW pre-commit** | `…/2026-08-03-sitclf-horizon/c_pow_precommit.json` | **REPO, STAGED.** positive clusters written to disk **before any score was read** |
| **C-FID-PARENT** | `…/2026-08-03-sitclf-horizon/cfid_parent.py` → `cfid_parent.json` | **REPO, STAGED.** reproduces the sibling stream's banked lead-3.0 row |
| **four families** | `…/2026-08-03-sitclf-horizon/four_families_ps.py` → `results_four_families.json` | **REPO, STAGED.** LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC per situation, never pooled, on **both** the causal and the legacy ego block |
| **P4 — ego audit + rebuild** | `…/2026-08-03-sitclf-horizon/rebuild_causal_ego.py` → `ego_leak_audit.json` | **REPO, STAGED.** the audit over ALL 500 clips and the causal rebuild |
| ⭐ **independent verifier** | `…/2026-08-03-sitclf-horizon/verify_event_yardstick.py` → `verify_event_yardstick.json` | **REPO, STAGED.** re-derives every Q-B headline from the banked scores using the **promoted `stack` function**, so the JSON cannot agree with itself. **12/12 cells, 0 mismatches** |
| ⚠️ **winner's-curse diagnostic** | `…/2026-08-03-sitclf-horizon/oracle_winners_curse.py` → `oracle_winners_curse.json` | **REPO, STAGED. POST-HOC, not pre-registered** — measures how much of `C-ORACLE-PS`'s margin is selection bias. Changes no verdict |
| **Q-B verdicts** | `…/2026-08-03-sitclf-horizon/qb_verdict.py` → `qb_verdict.json` | **REPO, STAGED.** the Sec 5 predicates applied to Q-B, which the run script wrote for Q-A only |
| **P3 cost analysis** | `…/2026-08-03-sitclf-horizon/trainer_gap_cost.json` | **REPO, STAGED.** exact parameter counts + what would have to be built + the stale-claim correction |
| run logs | `…/2026-08-03-sitclf-horizon/run_log_{ego,cfid,ps,curse,families}.txt` | **REPO, STAGED.** |

### ⭐ Promoted into `stack/` — the one code change this stream made

| artifact | path | why it is not left in an analysis script |
|---|---|---|
| **the event-level yardstick** | `stack/tanitad/eval/sitclf_deploy.py::event_anticipation_report` | **REPO.** `precision_recall_at_budget` and `anticipation_lead_s` both take `y`, which is a function of `lead_s`, so **neither can compare two heads at different horizons**. This one takes the ONSETS and no label. Leaving it in one analysis script is the stranded-instrument failure the operating standard exists to prevent |
| its tests | `stack/tests/test_sitclf_deploy.py` (+8, **43** in file) | **REPO.** including the two load-bearing ones: the look-back **never crosses a clip boundary**, and an unwarned onset contributes **no lead** rather than a 0 s that would reward silence |

### Deliberately NOT in the repo

| artifact | where | why, and how to regenerate |
|---|---|---|
| the B4 feature substrate (410 MB) | `C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz` + `.meta.json` | 99,477 × 2048 fp16 frozen-v1 latents. Too large for git, fully reproducible with `…/2026-08-03-sitclf-matched-capacity/build_substrate.py`. Its `.meta.json` now carries this stream's **quarantine stamp** (§3) |
| **the CAUSAL ego sidecar** (1.2 MB) | `C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.ego_causal.npz` | ⚠️ **EXISTS IN ONE PLACE.** It is a pure function of the episode caches and `situations.kinematics(causal_pre=True)`, regenerated in **~2 s** by `rebuild_causal_ego.py`, and it carries a `_provenance` blob. Kept out of git because it is a derived binary beside a 410 MB substrate that is itself out of git — the substrate and its sidecar must not become separable |

**Nothing in this stream lives only on a pod or only in a worktree.**

---

## 2. Reproduction order

```
python rebuild_causal_ego.py            # P4: audit + causal sidecar + quarantine stamp  (~5 s)
python cfid_parent.py                   # C-FID-PARENT: must PASS before anything else   (~1 min)
python run_per_situation_horizon.py     # the study                                      (~20 min)
python verify_event_yardstick.py        # the independent verifier — must PASS           (~5 s)
python oracle_winners_curse.py          # the post-hoc control-on-a-control              (~1 min)
python qb_verdict.py                    # the Sec 5 predicates applied to Q-B            (~1 s)
python four_families_ps.py              # the four binding families, 0-GPU re-analysis   (~9 min)
python render_tables.py                 # TABLES.md
```

`OMP_NUM_THREADS=6` is set for every step (torch spawns ~113 threads per process and concurrent
arms then make no progress — MEASURED 2026-07-27, 7 arms at GPU `sm` 0–6 % for 50 minutes).

---

## 3. ⭐ P4 — what changed on disk outside this directory

`rebuild_causal_ego.py` writes **two** things outside the run directory, both additive:

1. `C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.ego_causal.npz` — **new file**, the causal
   `E` block. Nothing is overwritten.
2. `C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.meta.json` gains an `ego_block_defect`
   key — the **quarantine stamp**. The substrate's own arrays are **untouched**.

⛔ **The 410 MB substrate was deliberately NOT rewritten.** Rewriting it in place would silently
break the bit-reproducibility of every banked run that cites it — including the sibling temporal
study and B4 — and this stream's own `C-FID-PARENT` depends on that reproducibility. A sidecar plus
a stamp on the documented provenance surface is the honest form.

---

## 4. Escalations

See the report's headline section. The load-bearing ones:

1. ⭐ **A stale absence-claim is in circulation and I am correcting it, not repeating it:**
   *"the situation classifier still has no promoted trainer"* is **FALSE at HEAD** —
   `stack/scripts/sitclf_train.py` exists (commit `49e2229`) and already exposes `--win` for
   exactly this study. It was true when the sibling stream wrote it and the trainer landed the same
   day. Same root-cause class as the operating standard's own rule: *absence found at ONE location
   is not absence*.
2. **The B4 substrate's ego block is quarantined, not deleted.** Anyone computing regime strata
   from it must read the sidecar. The report names every arm that was scored on it.
3. The sibling stream's citation escalation (`refc.py:1112-1117`) is **already logged** as
   `R-2026-08-03-cite` in `RETRACTION_LOG.md` — no action needed, checked rather than assumed.
