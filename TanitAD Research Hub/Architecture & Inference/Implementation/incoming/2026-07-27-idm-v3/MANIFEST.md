# IDM v3 — deliverable manifest

**Agent:** `idm-v3` · **Date:** 2026-07-27 · **Pod:** `tanitad-eval` (A40).
pod1 / pod2 / pod3 untouched throughout.
**Git state: I committed nothing and pushed nothing.** Everything below was
`git add`ed only. ⚠️ **However, a sibling agent's commit `fdc5b4f` ("five
refutations, one 3x storage win, and a struck VALIDATED row") swept my
then-staged work into itself** — the known whole-index behaviour documented in
`CLAUDE.md`. So the docs, the code and the `stack/` loader fix are now **in
HEAD** rather than in the index, and the later artifacts (results JSON, the
checkpoint, this manifest, the HF receipt) are **staged on top**. Nothing was
lost; the state is recorded here so an auditor is not surprised by it.
The HuggingFace push is separate and was authorised by the brief.

⚠️ **`/root` on the eval pod silently truncates** — a `dd` requesting 3.00 GB
wrote 2.79 GB and exited 0. All v3 output went to `/workspace/idm3`.

## Documents

| file | what | lives |
|---|---|---|
| `IDM_V3.md` | **the detailed approach the PI asked for** — literature, geometry substrate, label before/after, geometry arms vs controls, per-channel per-corpus results, what still fails, escalations | repo (staged) |
| `PRE_REGISTRATION_IDMV3.md` | bars frozen before the arms ran; §5b declares the arms added mid-run; §5c the sourcing-discipline note | repo (staged) |
| `CITATIONS.md` | ~50 papers, per-entry **fetch-depth ledger**, the scale-vs-rotation table | repo (staged) |
| `MODEL_CARD_IDM_V3.md` | the card pushed to HF as `README.md` | repo (staged) + **HF** |
| `MANIFEST.md` | this file | repo (staged) |

## Code

| file | what | lives |
|---|---|---|
| `idm3_geom.py` | the geometry substrate + the physics (eq. 1/2) that generates every prediction | repo + `tanitad-eval:/workspace/idm3/` |
| `idm3_labels.py` | Phase 2 — A0 reproduction, speed-binned yaw audit, the repair, before/after | repo + pod |
| `idm3_geomtest.py` | the sharp 0-GPU geometry test (partial correlation + the direct correction + shuffled control) | repo + pod |
| `idm3_arms.py` | the arm ladder: side-info conditioning, physics parametrisation, HL-Gauss ordinal head, `--corpus pai` | repo + pod |
| `idm3_analyze.py` | all paired episode-cluster bootstraps + the repair audit | repo + pod |
| `idm3_paitest.py` | the decisive within-PhysicalAI test, scored with paired CIs | repo + pod |
| `idm3_ship.py` | assembles + scores the two-expert composite, emits the weights-only checkpoint | repo + pod |
| `idm3_a0.py` | persists the deployed head's predictions on the identical windows | repo + pod |
| `push_idm_v3.py` | the HF push, with a **confidentiality audit** that refuses non-whitelisted checkpoint keys | repo (staged) |
| `pai_geom_table.json` | per-clip camera geometry, **keyed by episode index — no clip UUIDs** | repo + pod |

**Nothing is stranded on the pod.** Every script above exists in the repo; the
pod copies are the execution copies. The only pod-only artifact is
`/root/idm2/lat/` (104 encoded episodes, 108 MB), which is **regenerable in 102 s**
by the v2 `idm2_encode.py` and was not created by this work.

## Raw JSON — one file per number quoted

All under `results/`, all staged:

| file | contents |
|---|---|
| `labels_v3.json` | A0 reproduction, speed-binned yaw audit, the repair at 5 thresholds, long_accel provenance, the first (underpowered) geometry correlation |
| `geomtest_v3.json` | the sharp geometry test: partial correlations, the direct correction, the shuffled control, the oracle ceiling |
| `arms_v3.json` | wave 1 — 14 arms × 3 seeds, per channel, per corpus |
| `arms_v3c.json` | wave 4 — the HL-Gauss family + discretisation ceilings |
| `arms_v3pai.json` | the within-PhysicalAI ladder |
| `compare_v3.json` | every paired episode-cluster bootstrap + the **repair audit** |
| `paitest_v3.json` | the within-PhysicalAI paired contrasts |
| `ship_v3.json`, `ship_tra.json`, `ship_rot.json` | the shipped composite and its two experts |

⚠️ **`arms_v3b.json` does not exist:** wave 2a died silently after `HrotG` seed 0
with no traceback. `Hrot`/`Htra` (3 seeds each) completed and are quoted from the
log; `HrotG`/`HtraG` were **not re-run** and nothing is quoted from them. The
high-value remainder (the HL-Gauss family) was re-run as wave 4 (`arms_v3c.json`).

## Model weights

| artifact | where | detail |
|---|---|---|
| `idm_head_v3.pt` | repo (staged) · `tanitad-eval:/workspace/idm3/out/` · **HuggingFace** | 4,301,848 params, 94 tensors, md5 `0125640133215c738526e40a13dc362f`, 17,238,745 B |
| `idm_head_v3_R0.pt` / `_V2R.pt` | pod only | the two single-expert checkpoints the composite is assembled from; **regenerable** by `idm3_arms.py --save-ckpt` |

### HuggingFace push receipt

| | |
|---|---|
| repo | **`Sayood/tanitad-idm-head-v3`** |
| revision | `8efe8c3a9274e4ca193e4c8c06e6c5333bcbb78e` |
| files | `README.md` (the model card), `idm_head_v3.pt`, `.gitattributes` |
| md5 | `0125640133215c738526e40a13dc362f` |
| visibility | **private** |
| pushed as | `Sayood` |

Full receipt: `hf_push_receipt.json`.

🔒 **Confidentiality:** the push script whitelists checkpoint keys and scans the
metadata for `clip_id` / `uuid` / `episode_id` / `frames` / `poses` / corpus
paths, refusing the upload on any hit. **Weights and scalar config only** — no
frames, no poses, no clip identifiers, no per-clip geometry table.
🔑 The token was read in place via `tanitad.keys.load_keys()` into the
environment; never printed, echoed, copied, or passed in argv.

🔴 **ACTION REQUIRED (I cannot do this via the API):** the repo is **private**
but the **manual-approval gate must be enabled in the HF repo settings UI** to
match the precedent of the four Phase-0 checkpoint repos.

## Changes OUTSIDE this directory — these need review

| file | change | status |
|---|---|---|
| `stack/tanitad/data/comma2k19.py` | `HEADING_MODE_LEGACY/HOLD`, `hold_heading_through_standstill()`, `heading_mode=` on `actions_and_poses` and `build_episode`. **Default is LEGACY**, so every existing cache and published number stays byte-identical | **in HEAD (fdc5b4f)** |
| `stack/tests/test_comma2k19.py` | 6 new tests: the defect reproduced, the repair, wrap-safety, the no-observable-frames no-op, the byte-identical default, the rejected mode, the measured threshold | **in HEAD (fdc5b4f)** |

`cd stack && pytest -q` — **green: 1200 passed, 7 skipped** (and 12/12 on `test_comma2k19.py`).

## Escalations (headline, per operating-standard rule 3)

1. **Flip the comma2k19 heading default and rebuild the corpus** — the fix is
   implemented and tested but opt-in. Needs an owner and a rebuild decision.
2. **Re-issue every published comma2k19 `yaw_rate` number.** The deployed head's
   goes **0.105 → 0.811**. `MODEL_REGISTRY.md` + `idm_head_v1_card.json`.
3. **Four files hard-code a wrong constant camera height** (`rr_log.py:93-94`,
   `taniteval/cam_overlay.py:29`, `taniteval/clhorizon.py:87`,
   `scripts/viz_trajectory_fan.py:43`). Measured truth: **per-clip, 1.245–1.607 m**.
4. **Enable HF gating on `Sayood/tanitad-idm-head-v3`** (UI-only).
5. **Retrain the v3 recipe on A0's 160-clip corpus** to recover `steer`
   (0.408 vs 0.742 is a data-budget regression, not a recipe finding).
