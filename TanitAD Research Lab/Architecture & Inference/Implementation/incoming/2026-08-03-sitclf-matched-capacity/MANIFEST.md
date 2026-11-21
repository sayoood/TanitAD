# B4 — matched-capacity vision head: DELIVERABLE MANIFEST

**Date** 2026-08-03 · **Stream** STREAM C / BACKLOG **B4** · **Substrate** dev box (RTX 4060),
**0 pod GPU-h** — no pod was touched; `tanitad-new` and `tanitad-pod4` were left training.

---

## 1. Where every artifact lives

| artifact | path | what it is |
|---|---|---|
| substrate builder | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-sitclf-matched-capacity/build_substrate.py` | **REPO, STAGED.** v1-frozen 2048-d camera latents + the canonical situation labels over 500 local clips |
| the sweep | `…/2026-08-03-sitclf-matched-capacity/run_matched_capacity.py` | **REPO, STAGED.** the capacity ladder, all controls, the four families |
| results | `…/2026-08-03-sitclf-matched-capacity/results_matched_capacity.json` | **REPO, STAGED.** every arm, interval, control and family |
| per-row scores | `…/2026-08-03-sitclf-matched-capacity/results_matched_capacity.scores.npz` | **REPO, STAGED.** out-of-fit scores for **all 28 arms** (14 rungs + 14 permuted-feature nulls) over 99,477 rows — 0-GPU re-analysis surface |
| index-leak probe | `…/run_index_probe.py` · `results_index_probe.json` · `results_index_probe.scores.npz` | **REPO, STAGED.** the control that the clip-permuted null is a genuine chance floor (index-only arm + a time-rolled null) |
| table renderer | `…/render_tables.py` → `tables.md` | **REPO, STAGED.** every table in the report is generated from the JSON, never retyped |
| run logs | `…/run_log.txt` · `…/run_log_index_probe.txt` | **REPO, STAGED.** |
| report | `…/2026-08-03-sitclf-matched-capacity/MATCHED_CAPACITY.md` | **REPO, STAGED.** |
| capacity primitives | `stack/tanitad/eval/sitclf.py` | **REPO, STAGED.** `head_param_count`, `width_for_param_budget`, `ridge_scores`, `ridge_param_count` |
| precision-at-budget | `stack/tanitad/eval/sitclf_deploy.py` | **REPO, STAGED.** `precision_recall_at_budget`, wired into `four_family_report`'s TACTICAL block |
| degenerate-draw hardening | `stack/tanitad/eval/ap_ci.py` | **REPO, STAGED.** `_bounds` — an all-nan bootstrap now reports, it no longer raises from inside numpy |
| tests | `stack/tests/test_sitclf.py` (+11) · `stack/tests/test_sitclf_deploy.py` (+5) · `stack/tests/test_ap_ci.py` (+3) | **REPO, STAGED.** |
| retraction | `Project Steering/RETRACTION_LOG.md` → **R-2026-08-03-h** | **REPO, STAGED.** the "2,049 vs 2.17 M" cross-stream numeric transplant |
| corrected doc | `…/2026-08-03-sitclf-fusion-wired/SITCLF_VISION_ONLY.md` | **REPO, STAGED.** §4 and §6.4 corrected in place with a pointer to the retraction |

### Deliberately NOT in the repo

| artifact | where | why, and how to regenerate |
|---|---|---|
| the feature substrate (410 MB) | `C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz` + `.meta.json` | 99,477 x 2048 fp16 v1 latents. Too large for git and fully reproducible: `python build_substrate.py --out <path>` — ~15 min on the 4060 from the local episode caches and `C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt`. **The `.meta.json` (provenance, trunk step, per-situation counts) IS copied into the run directory.** |

**Nothing in this stream lives only on a pod.** The 2048-d features the *banked* sitclf arms consumed
did — on `pod3:/workspace/sitclf/feats` — and pod3 is gone; that is precisely why the substrate had to
be rebuilt (see §2).

---

## 2. Why the substrate is a REBUILD, and what that costs

BACKLOG B4 needs the image FEATURES. The banked bundle
`…/2026-07-26-situation-classifier/artifacts/heldout_frames.npz` stores **scores only**. Probed three
ways at HEAD, all negative:

1. `find` for `clip_0*.npy` across the repo and `C:/Users/Admin` — nothing;
2. `heldout_frames.npz` and `scores.npz` bank `E` (ego) and the score columns but never `F`;
   `_pod_backup/` holds pod2 *model* checkpoints only;
3. `tanitad-thor` — no `sitclf` directory and no PhysicalAI **train** cache (only
   `physicalai-val-0c5f7dac3b11`).

**IDENTICAL to the banked run:** the encoder (v1 `flagship4b-speedjerk-30k` @ step 29999, encoder
87,022,848 + readout 98,432 params, STRICT-loaded, frozen, `uint8 -> /255`), the label detectors
(`tanitad.data.situations`, constants **byte-identical** to the banked `sc_situations.py` by diff),
`lead_s = 3.0`, `cross=None`, and the ego block `[v, alon_pre, omega_pre] / EGO_SCALE`.

**DIFFERENT:** the episodes. The parity cache `physicalai-train-e438721ae894` is not on this box; the
local caches are `physicalai-train-14231cd29c74` (400) + `physicalai-val-bb543bdf7836` (100).
⇒ **absolute APs here are NOT comparable to the banked table.** Every claim is a WITHIN-substrate
paired contrast on identical rows, which is what a capacity comparison needs. No training corpus is
re-selected or re-hashed; this is an offline probe of a frozen encoder's latents, not a model arm.

---

## 3. Escalations

* **B4 is CLOSED by this run** — `BACKLOG.md` row B4 has been struck through and pointed here
  (DONE, staged), including the correction of its retracted "~2k to ~2.17M" range. The true span of
  the two arms the finding is about is **129 -> 417,028**; the ladder built here covers
  **129 -> 2,207,572**, so both readings are answered.
* **The P8 recommendation "the real lever is the head, not the camera" needs the qualifier this run
  supplies**: it holds on `intersection`, REVERSES on `lane_change`, and cannot be decided on
  `roundabout` — which is the situation it was originally read off.
* **The next experiment this points at is B5** (frozen VIDEO-pretrained encoder), not another head
  sweep: `tf_pca64_d128` beats `tf_pca16_d128` (separated), i.e. the head is INPUT-starved.
* **The situation classifier still has no promoted trainer.** `sc_train.py` / `sc_train_v2.py` remain
  in hub `incoming/` only. This run promotes the *primitives* (`ridge_scores`, `head_param_count`,
  `width_for_param_budget`) into `stack/`, not the trainer. Promoting the trainer is a separate work
  item and is a prerequisite for anything that wants to re-fit a situation head in production.
