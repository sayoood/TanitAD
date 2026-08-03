# DELIVERABLE MANIFEST — P8 (scenario classifier) + P9 (IDM), 2026-08-03

Every artifact and **where it lives**. All paths absolute-from-repo-root
`G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD`.
**Everything is STAGED in the working tree (`git add`) and NOTHING is committed or pushed** —
staging verified with `git ls-files --cached`, never with an exit code.
**Nothing is stranded on a pod or a worktree: 0 pod GPU-h was used, all compute ran on the dev box.**

---

## Library code — `stack/` (in the pytest suite)

| path | contents |
|---|---|
| `stack/tanitad/eval/ap_ci.py` | `average_precision`, `ap_lift`, `ap_episode_cluster_bootstrap`, `paired_ap_episode_cluster_bootstrap`, `stat_episode_cluster_bootstrap`, `paired_stat_episode_cluster_bootstrap`. **The first admissible interval for an AP or any set-level statistic in this program.** Resampling delegates to `taniteval.ci._draws`, asserted identical draw-for-draw. |
| `stack/tanitad/eval/sitclf.py` | `clip_runs`, `causal_window`, `cluster_folds`, `CausalSitHead` (= `sc_train.SitHead` with WIN a parameter), `train_sit_head`, `predict_sit_head`, `late_fuse_scores`. |
| `stack/tanitad/eval/idm_families.py` | The IDM's four-family instrument: `geometry(wp, dt)`, `manoeuvre_classes` (**factored** lateral / longitudinal + the legacy `mixed`), `confusion`, `balanced_accuracy(require_all=)`, `longitudinal`, `lateral`, `tactical`, `strategic`, `all_families`, `ade`. |
| `stack/tests/test_ap_ci.py` | 14 tests incl. estimator identity with `taniteval`, cluster-vs-frame SE (1.94×), and the estimator's own negative control. |
| `stack/tests/test_sitclf.py` | 14 tests incl. causality/no-clip-crossing, out-of-fit proof by refit, and a long-window contract (win16 AP 1.00 vs win4 at chance). |
| `stack/tests/test_idm_families.py` | 16 tests incl. **bit-parity with `taniteval.four_families._seq_geometry`**, the 5× cadence trap, factored-vs-mixed manoeuvre loss, and one regression test per bug found in §"bugs" below. |

**`pytest -q` from `stack/`: 1680 passed, 12 skipped, 2 xfailed** (44 of those tests are new; **0 of the new tests skip**).

---

## P8 — scenario classifier · `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-sitclf-optimisation/`

| file | what |
|---|---|
| `SITCLF_OPTIMISATION.md` | The result. |
| `run_sitclf_opt.py` | Runner: baseline, WIN sweep with the architecture held fixed, late fusion, 3 negative controls, paired AP-lift bootstrap. |
| `results_sitclf_opt.json` | **B=2000** (program default) — primary. |
| `results_sitclf_opt_b400.json` | **B=400** independent replication; same verdict on every row. |
| `results_sitclf_opt_b400.scores.npz` | **11.4 MB — the out-of-fit per-frame scores for every arm I fitted**, plus labels, validity, clusters and folds. Re-analysable at 0 GPU by anyone; no retrain needed to re-score or re-interval. |
| `run_log.txt`, `run_log_b400.txt` | Full run logs. |

**Source substrate (pre-existing, unmodified):** `…/2026-07-26-situation-classifier/artifacts/heldout_frames.npz`.

---

## P9 — IDM · `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-03-idm-four-families/`

| file | what |
|---|---|
| `IDM_FOUR_FAMILIES.md` | The result, incl. the leak finding. |
| `run_idm_four_families.py` | Runner: leak-audited split by content, encode via `run_idm_proof.load_encoder` (**reused, not reimplemented**), the shipped 3-seed ensemble, 4 families, 3 negative controls, a leak-sensitivity arm, paired CIs. |
| `results_idm_four_families.json` | All four families + ADE + intervals for 4 arms + the leaked arm. |
| `run_log.txt` | Full run log. |

**Inputs (pre-existing, unmodified):** head `…/2026-07-27-fleet-sync-idm-steer/idm_head_v4_steer_ens3.pt`; encoder `C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt` (step 29999, off-repo, unchanged); frames `C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f` (off-repo, unchanged).

---

## ⚠️ NOTE FOR WHOEVER COMMITS — the index contains OTHER agents' work

At hand-off `git diff --cached --name-only` shows **50 staged paths**, of which **12 are mine**
(the 6 `stack/` files above + the 6 P8/P9 hub files). The rest belong to concurrent siblings:
`stack/experiments/nurec-gsplat/`, `stack/scripts/lan_probe.py`, `stack/tanitad/data/lan.py`,
`stack/tanitad/eval/route_cf.py`, `stack/tanitad/models/readout.py`, `stack/tanitad/refs/refc.py`,
`stack/scripts/refc_train.py`, `stack/tests/test_lan.py`, `stack/tests/test_readout_onnx_pool.py`,
the `2026-08-03-thor-*` hub folders, `2026-08-03-sota-scan/`, and `_pod_backup/pod2-2026-08-03/`.

Per CLAUDE.md's git-hygiene rule: a pathspec-free `git commit -F <msgfile>` commits the **whole
index**, so it is admissible here **only** after listing it and confirming every entry is intended
programme work — and the commit message must say the siblings' deliverables are included.
`git commit -- <pathspec>` **segfaults on this repo** and is not the escape hatch.

---

## Bugs found in my own work, fixed, and pinned by a regression test

1. **Leaked comprehension variable** in `run_idm_four_families.py` made all three "paired vs control" contrasts compare against the *same* arm. The tell was three identical deltas. Fixed to index `arms[n][1]`.
2. **Balanced accuracy degenerates inside an episode bootstrap.** Turns are <1 % of a highway corpus, so many resamples contain no turn; with one class present, "mean recall over present classes" scores a **blind constant predictor at 1.0**, and the first run duly reported the blind control at **[0.3333, 1.0000]**. `balanced_accuracy(..., require_all=True)` now returns `nan` for such draws; they are dropped and **counted** (1,727/2,000 survived on the lateral axis). Pinned by `test_balanced_accuracy_require_all_blocks_the_bootstrap_degeneracy`.

---

## ⚠️ ESCALATIONS — these need a decision, they are not doc notes

1. ⛔ **The IDM has no verified held-out comma substrate, and the local val cache is its training pool.** Two probes (§5 of `IDM_FOUR_FAMILIES.md`): a 12.4× speed-MAE gap in the impossible direction, and `idm5_ensemble.json.leak_check`'s `cmx_00008 / cmx_00020` indices matching the local cache exactly. The program's content-clean audit covers **42 of 121** comma training episodes. **`E-IDM-CLEAN20` cannot be run as planned**, and every comma number the IDM publishes needs re-reading in this light.
2. ⛔ **`MODEL_REGISTRY.md` has no `idm_head` row and no `sitclf` entry** — zero matches for `idm_head`, `sitclf`, `situation`, `head_img`. Under the source-of-truth rule every number in both streams is admissible as **raw eval JSON only**. Needs a PI ruling: registry gap, or probes are out of registry scope.
3. ⛔ **`MODEL_REGISTRY.md` §8.1 #6 (line 1852) still publishes the withdrawn 77.5 %-leaked "held-out speed R² 0.930"**, and the leaky-cache audit's own fix-table does not list it as corrected.
4. ⚠️ **`head_img_ego` is the deployed sitclf arm and it is separated-worse than its own ego ablation.** The fix is a config change, not a retrain — but it is a change to a shipped component and belongs to the PI.
5. ⚠️ **A sitclf → `four_families.py` adapter still does not exist.** `ap_ci.stat_episode_cluster_bootstrap` was the missing piece; the adapter itself is an open work item.
