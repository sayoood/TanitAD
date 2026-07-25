# IDM reconstruction validation — SUBSTRATE (P0 bank)

Date: 2026-07-25. Agent: idm-youtube-validation (reconstruction + proof + video leg).
Goal: reconstruct an ego trajectory PURELY FROM VIDEO with the persisted non-causal IDM
head, prove it against ground truth on a **held-out** clip, render the standard overlay video.

## The clip chosen (held-out, WITH ground truth)

- **`ep_00009` — episode_id 808596784** from the PhysicalAI **val** cache
  `physicalai-val-0c5f7dac3b11` (eval pod `/root/valdata/`).
- T=199 frames @10 Hz (~20 s). Speed 11.1–17.6 m/s (sustained, real longitudinal content),
  yaw span ~3.0 rad with a substantial LEFT turn (35 `left` maneuver frames, |yaw_rate|max
  0.46 rad/s). Chosen so the reconstructed path visibly CURVES — a non-trivial test of both the
  trustworthy (speed / longitudinal) and the moderate (yaw) channels, not a straight constant-v
  clip. Survey of all 40 val eps: `scratchpad/idmval_survey.py` (MEASURED).
- Episode contract: `frames_u8 [199,9,256,256] u8` (encoder-ready 3-stack), `poses [199,4]=x,y,yaw,v`,
  `actions [199,2]=steer,accel`, `maneuvers [199] int64` (GT), `episode_id`.

## Held-out proof (this clip was in NO training set)

Doubly held-out — unseen by BOTH the head and its frozen encoder:
1. **Head** `idm_head_v1` trained ONLY on tags `tr_a_[0:60]` + `tr_b_[0:60]` + `cm_[0:40]`
   (source: `idm_head_v1_train.py::TRAIN_TAGS`, in this repo dir). Those are PhysicalAI
   **train**-cache episodes + comma2k19 — **zero** val-cache episodes.
2. **Encoder** flagship-v1 (`flagship4b-speedjerk-30k`) trained on the parity **train** corpus
   `physicalai-train-e438721ae894` (2376 train eps) — never the val cache.
3. PhysicalAI train vs val are disjoint splits (distinct datasets, content hashes
   `e438721ae894` vs `0c5f7dac3b11`); this is the same split every program val number rests on.
   The head's own card already treats these 40 val eps as its held-out `val_parityval` set.
   Basis: dataset split invariant (not independently re-checked against the 2376 train-id list,
   which lives on pod3) — evidence class for the disjointness: INHERITED (parity invariant).

## Substrate located / verified (all on eval pod `tanitad-eval`, GPU free: 0 MiB used)

| thing | path | verify |
|---|---|---|
| frozen encoder = flagship-v1 | `tanitad-eval:/root/models/flagship-30k/ckpt.pt` | md5 `b5f07d9e3dd2ca643949bc86832e6585` == card; step 29999; state_dim 2048 (MEASURED) |
| persisted IDM head | `tanitad-eval:/root/idmval/idm_head_v1.pt` | md5 `fa4462f0b898b036be729c790278b823` == card `weights_md5`; relayed read-only from `tanitad-pod3:/workspace/tmp/yt_val/results/idm_head_v1.pt` via dev box |
| val cache (GT) | `tanitad-eval:/root/valdata/physicalai-val-0c5f7dac3b11/` | 40 eps, format dumped |
| model code | `tanitad-eval:/root/v4eval/stack` (+ scp'd `scripts/idm_head.py`,`run_idm_proof.py`) | `load_encoder` + `encode_frames` smoke OK |

NOTE — `tanitad-flagship-4b-phase0/ckpt.pt` on the eval pod is a DIFFERENT md5
(`74be8103…`) = the no-speed ablation control per the registry; it is NOT the encoder and was
not used.

## Pipeline (reuses shipped modules; nothing re-implemented)

video frames_u8 → `run_idm_proof.encode_frames` (frozen v1 encoder+readout) → z[T,2048]
→ `idm_head.build_windows` (non-causal 9-frame windows) → `IDMHead` → per-center
{speed,yaw_rate,steer,long_accel} + 2 s ego trajectory. Metrics via `idm_head.evaluate`
(speed/yaw R², ADE@2s, per-horizon DE) against CAN-derived GT from `poses`/`actions`.
(The YouTube-specific `yt_idm_reconstruct.py` GeoCalib/anonymiser/HUD path is bypassed — this
clip is a cached PhysicalAI episode with known geometry and CAN GT, so those stages don't apply;
the reusable core — encode → build_windows → head → evaluate — is identical.)
