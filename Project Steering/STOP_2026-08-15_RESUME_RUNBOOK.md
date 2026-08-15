# Pod stop 2026-08-15 — final state, what is banked where, and how to resume

**PI, 2026-08-15:** *"So now I will stop the pods, so I want you to stop training and
pipeline. Commit and push all valuable and necessary contents. Document any missing things.
Push any valuable data as archive to hf. Document how to resume the training and pipeline."*

Everything below is MEASURED at stop time unless marked otherwise.

---

## 1. Final state at stop

| stream | state |
|---|---|
| **v6F S-W training** (pod5) | **stopped cleanly at step 6 300**; last checkpoint **step 6 250**, verified loadable (573 tensors), snapshot `ckpt_final_stop.pt` md5 `01a0c5e8` == live `ckpt.pt`. Trainer killed by explicit PID after the snapshot; no supervisor existed. |
| training history | two gradient-spike episodes (≈3 450–3 850, peak gnorm 354 076; and ≈5 150), both self-recovered; loss band ~2.3–3.3 at stop; 17.37 s/step throughout. |
| **aug120 pipeline** (pod4) | **completed on its own**: all **201/201** runnable Alpamayo clips processed at 120° (bridge → VLM → SAM3), every batch pushed incrementally (`AUG120_DONE`). |
| **PH1 fusion** | 600-clip w120-val set fused and on HF (`fused_w120val/`): 175 corroborations, 41 conflicts, 56 with the Alpamayo layer. |
| **P-battery** | 2.5 k and 5 k runs + speed-echo controls on HF. Echo confirmed at both (structural). Encoded curve moving (−2.30 → −0.74 at speed k=10). P3/P6 verdict still pending. |
| **G1** | reviewed (PI-delegated) and **closed**: 0/31 verifiable; SAM3 sign-class reliability flagged. Evidence sheet archived. |

## 2. What is banked, and where

**GitHub (`claude/tanitad-resumption-handoff-92zx39`, PR #2)** — all code and docs, pushed
continuously; working tree clean at stop.

**HF `Sayood/tanitad-v6` (model, public+gated-manual):**
`v6F-SW-30k/{ckpt.pt, ckpt_final_stop.pt, config.json, metrics.json, train_log.jsonl,
train.out, weights_fp16.pt}` · `pbattery_5k/` · `pbattery_2k5/` · `ops/` (the shipped
module set).

**HF `Sayood/tanitad-ph0-aug120` (dataset, public+gated-manual):**
`batch_*/{v2,sam3}` (the 201 Alpamayo-at-120° labels) · `fused_w120val/` (600 fused
records + summary) · `bridged_w120train_2400/` (the 5.4 GB bridged corpus — the previously
stranded artifact) · `epcache_oodval_290/` (32 GB OOD-val episode cache) ·
`w120val_600/{ego,clips.json}` · `g1_evidence/` (crops + the graded sheet).

⚠️ **Verify the two large archives landed before deleting the pods** — far-side listing,
not the push log (§5 has the command). Everything else is small and was verified.

## 3. How to resume TRAINING on a fresh pod

1. **Clone the branch** → all code including `train_v6_staged.py` with the gc-split flags.
2. **Venv, in this exact order** (the twice-measured torch trap):
   pipeline extras `--no-deps` first, then **torch pinned LAST**:
   `uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0" "torchvision==0.23.0"`.
   Verify with a **real conv2d on CUDA**, never `import torch`.
3. **Caches:** `hf download Sayood/tanitad-physicalai-w120-256x640cyl` — the
   `physicalai-train-…` (85 GB) and `physicalai-val-…` (21 GB) directories, straight into
   `/workspace/data/`. Parity is preserved by the skip-hash + episode-uid check, not bytes.
4. **Checkpoint:** `hf download Sayood/tanitad-v6 v6F-SW-30k/ckpt.pt v6F-SW-30k/config.json`.
5. **Relaunch with the EXACT flags from the banked `config.json["args"]`** plus
   `--resume auto`. The load is strict: config drift ⇒ a *refused* restart (by design —
   recoverable, never silent). `--out` must contain the downloaded `ckpt.pt`.
   Optional speed lever, safe with the same checkpoint: `--enc-grad-checkpoint off`
   (rollout checkpointing stays on — it is the OOM guard).
6. **Re-arm the safety loops** (both banked in `stack/ops/`):
   `hf_push_loop.py` (push + far-side verify every cycle — the old loop's silent-failure
   class is written up in `POD_HANDOVER_2026-08-13.md §4b`) and
   `pbattery_watcher.py` with `TARGET_STEP=10000` (runs battery + echo control at the
   milestone, pushes, restarts whatever it paused).
7. **Budget:** 30 k from 6 250 ≈ **4.8 days** at 17.4 s/step on one A40. The gc-split /
   batch-size levers (unmeasured) are the candidates to shorten this.

## 4. How to resume the PIPELINE on a fresh pod

1. Deps (`--no-deps`): `sam3`, `open_clip_torch`, `opencv-python-headless`,
   `imageio`/`imageio-ffmpeg`, `lm-format-enforcer`, `qwen_vl_utils`. HF token present
   (`facebook/sam3` is gated to the account).
2. **The 201-clip Alpamayo-at-120° pass is DONE** — do not re-run; outputs are on HF.
3. **Next pipeline work, in order:**
   a. **Fuse the aug120 batches** (`ph1_fuse.py` over each `batch_*/` — NOT yet done;
      only the 600-clip val set is fused).
   b. **The 4 472 clips without w120 caches:** derive the chunk→clip index from the
      nvidia source's per-feature parquet chunks
      (`camera/camera_front_wide_120fov/*.zip` + parquet siblings), then
      `v2_compressed.py build --only-clips <list>` in bounded batches feeding the
      aug120 flow. This is the single biggest remaining data job (~8–10× everything
      processed so far).
   c. **G1 native-resolution re-run** — same 31 rows, crops from the source chunks;
      the pipeline-fidelity review is banked in `G1_RESULT.md`.
   d. **OOD-290 redo:** `/workspace/ph0_ood` outputs were INVALID (the t−200 ms channel
      bug, fixed in `epcache_to_pilot.py` before any label was banked). Re-bridge from
      the `epcache_oodval_290` archive with the fixed script, then VLM → SAM3 → fuse.

## 5. Verification commands (run before pod deletion)

```python
from huggingface_hub import HfApi
api = HfApi()
i = api.dataset_info("Sayood/tanitad-ph0-aug120", files_metadata=True)
pref = {}
for f in i.siblings:
    k = f.rfilename.split("/")[0]
    pref[k] = pref.get(k, 0) + 1
print(pref)   # expect bridged_w120train_2400 ≈ 4800+, epcache_oodval_290 = 291
m = api.model_info("Sayood/tanitad-v6", files_metadata=True)
print({s.rfilename: s.size for s in m.siblings if "v6F-SW-30k/" in s.rfilename})
```

## 6. Missing / open at stop — the honest list

1. **P3/P6 verdict** — never produced on v6 (the 5 k run predated the pod-side
   `V6Grounding` patch; the patch is applied and committed; the 10 k watcher will get it).
2. **BEV (P8) probe** — not ported to v6; O3 trains but the probe-side BEV eval is v5-only.
3. ~~**aug120 batch fusion** (§4.3a) — labels exist, fusion not run.~~ **DONE 2026-08-15**:
   all **201/201** clips fused and on HF (`fused_aug120/`, far-side verified) — 88 corroborations,
   10 conflicts, 201 with the Alpamayo layer. **What remains is the SAM3 leg, not the fusion** —
   see the corrected item 11. Record: `…/incoming/2026-08-15-aug120-fusion/AUG120_FUSION_RESULT.md`.
4. **The 4 472-clip build** (§4.3b) — scoped, not started.
5. **E-ENC in ViT-5 form** — the 384×8-vs-768×12 question was answered only for the plain
   ViT; the running encoder's width has no measurement in its current architecture.
6. **Batch-assembly / DataLoader measurement** — the trainer builds batches synchronously
   on the main thread; 42.8 % GPU util says there is headroom; unmeasured.
7. **SAM3 sign-class threshold study** — two-thirds of best sign crops contained no sign
   (G1_RESULT.md); sign channel flagged, not yet re-thresholded.
8. **Tasks 12/13/16** — v5.8f registry closure; D6 amendment; E-H1/W5 6 s baseline on the
   frozen v5f trunk (needs a free GPU).
9. **`ph0_ood/` on pod4 is invalid data** — superseded; do not archive or reuse (§4.3d).
10. The **sitclf label-provenance probe** (the `head_img` leak question) remains open from
    the earlier campaign.
11. ⚠️ **CORRECTED 2026-08-15 — this item understated the gap by 14×.** As written it said only
    **`batch_00184` (8 clips) has v2 labels but NO SAM3 output** (`B184_SAM3_ABSENT` at archive
    time). That is true, and it is **not the whole defect**: MEASURED at fusion, **115 of the 201
    aug120 clips (57.2 %) have no SAM3 record at all**, spread across *every* batch. Root cause —
    `aug120_pipeline.py` never passed `--n` to `ph0_sam3.py`, whose default is **4**
    (`ph0_sam3.py:411`), so each batch got SAM3 on its first 4 clips only while printing
    `SAM3_RC=0`. **Fixed in `aug120_pipeline.py`**; the labels are not retro-filled. All 201 clips
    are fused with the 115 marked as *named* partials (`perception.absent`), never silent zeros.
    Re-derivable as stated (shards are in the w120 corpus on HF), now **~30 min of GPU for 115
    clips**, then a re-fuse (fuser resumes; minutes, no GPU).
    Detail: `…/incoming/2026-08-15-aug120-fusion/AUG120_FUSION_RESULT.md §3`.
12. **`ckpt_final_stop.pt` == `ckpt.pt`** (md5-verified at stop). It exists as a separate
    name so that a future resumed run overwriting `ckpt.pt` on HF can never destroy the
    step-6250 stop point. Resume from either; they are the same bytes today.
