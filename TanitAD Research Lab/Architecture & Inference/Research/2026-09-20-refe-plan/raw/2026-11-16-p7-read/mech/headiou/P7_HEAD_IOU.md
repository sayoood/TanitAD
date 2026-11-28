# P7 auxiliary drivable-area head: IoU over [0, 40) m and head-vs-scorer DAC AUC on the 1,058 C3 navtest tokens (2026-11-16/17)

Tier: T0-like (head perception on teacher-forced navtest frames, planner fp32 ctx path). Never driving performance. Evidence class for every number: MEASURED (ours), artifact paths below, all under `HOLD = E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/headiou/`.
Estimator for every interval: paired/unpaired **log-cluster bootstrap**, 10,000 resamples, 95 %, seed 20260927 (`proxy_eval.boot` scheme, as `aux_s1_readout.py`). It answers "another draw of LOGS" only; not another training run (H-ESTIM-SEED-1), not another inference run.
n = 1,058 tokens, 123 logs (all 1,058 labelled; the C3 set is failure-enriched: 416 u400-DAC-zero + 642 others, a DIAGNOSTIC, never a rate).

## Headline
| read | value | CI | n | artifact |
|---|---|---|---|---|
| **IoU [0, 40) m, pooled over cells (registered K2 read)** | **0.8225** | [0.7896, 0.8525] | 1,058 frames | `readout_perception.json` arms.H1-s0.p1.all."0-40" |
| IoU [0, 40) m, per-token mean (cells 4..23 of 40) | **0.8107** | [0.7837, 0.8375] | 1,058 tokens | `token_iou.json` |
| per-token median / share of tokens with IoU >= 0.80 | 0.841 / 64.2 % | | 1,058 | `token_iou.json` |
| IoU [0, 40), tokens with u400 DAC = 0 | 0.7657 | [0.7089, 0.8186] | 416 | `readout_perception.json` p1.dac0 |
| IoU [0, 40), tokens with DAC = 1 | 0.8577 | [0.8402, 0.8738] | 642 | p1.dac_ok |
| IoU bands 0-20 / 20-40 m | 0.8489 [0.8164, 0.8765] / 0.7953 [0.7591, 0.8298] | | 1,058 | p1.all |
| SDF MAE near boundary (+-3 m) | 1.289 m | [1.164, 1.412] | 436,390 cells | p2 |
| **P3 AUC, head min-SDF along proposal footprints, DAC=1 vs 0** | **0.8105** | [0.7698, 0.8469] | 67,712 proposals (45,269 pass / 22,443 fail), 1,058 tokens | p3.pooled.head |
| P3 AUC, scorer (see caveat) | 0.7874 | [0.7311, 0.8310] | same | p3.pooled.scorer |
| head minus scorer | +0.0232 | [-0.0139, +0.0600] | same | p3.pooled.head_minus_scorer |
| head AUC, never-trained P7-architecture head (seed 0) | 0.6603 | [0.6246, 0.6940] | same | floor.init |
| trained minus init | +0.150 | lower bound +0.104 | same | floor.trained_minus_init |

## Branch triggered (SPEC_P7 section 5, NOT-PROVEN branch; spec text itself not present on E:, quoted as the BRIEF words it)
**IoU >= 0.80 branch: the head is good (K2 reads K2_CLEARED, decided on the point estimate over all labelled frames, as the prereg words it); the lever is the scorer feed (feed is off in P7, `feed_built` False), not the 1 m grid / `--w-field`.**
Honest margin: the point estimate clears 0.80 by 0.022 pooled (0.011 per-token mean), but both 95 % intervals include 0.80 (lower bounds 0.790 / 0.784); 36 % of tokens are below 0.80 and the head is clearly worse exactly where DAC fails (0.766 on the 416 DAC-zero tokens vs 0.858 on the rest). So "head is good enough to be read" is supported on the point estimate; "head IoU is clearly above bar" is not.
P3 context: head AUC 0.81 is above the u400 scorer (+0.023, CI spans 0), below the prereg's 0.90 evaluate threshold; decision label from the readout: H2_PROVISIONAL (floor lower bound +0.104 > 0).
**P3 caveat (do not over-read):** the "scorer" in P3 is the banked u400 scorer on u400's 64 proposals (`table.npz` logits[..., 1], 7-component SFT-4 model); the head is P7's. It is P7-head vs u400-scorer on u400's proposals, not P7-head vs P7-scorer. The P7 scorer's own DAC AUC on its own proposals needs a P7 DAC-labelled proposal table, which was not built here.

## Controls (all passed, before the real read was trusted)
1. **Head weights byte-equal**: all 43 head tensors of `ckpt["aux_drv"]` (2,440,450 params) equal the adapter's loaded head by `torch.equal` AND sha256 of bytes; key sets equal; no non-head tensors in the entry. `proof/proof_head_weights.json` (ckpt sha256 `2613fc17...`, matches the BRIEF).
2. **Trunk byte-equal**: all 643 `model` tensors of `model_final.pt` equal the planner's loaded model (n_unequal 0). `main/proof_run.json`.
3. **Tap**: the ctx the head reads (score_dec input, as the instrument taps it) equals the trainer's tap (scene_proj 2nd output) bit-for-bit on all 1,058 tokens (max abs 0.0). `main/proof_run.json`.
4. **Zeroed-head control** (arm H1-s1): IoU [0, 40) = 0.0000 [0, 0] (empty mask; a zero logit is not > 0), P3 AUC 0.5000. `readout_perception.json` H1-s1. Never-trained head (H1-s2): IoU 0.196, AUC 0.660: the trained head is far above both.
5. **Run-twice determinism** on 2 tokens: head_pred (logit, sdf_m, all 3 arms) and the cached ctx + scorer logits bit-identical. `proof/proof_determinism.json`.
6. **Planner-path identity**: the adapter's scorer logits equal P7's own seam dump (`stage2_p7_full.proposals.npz`, 12,146 x 64 x 6) **bit-for-bit on all 1,058 tokens** (max abs delta 0.0). `main/identity.json`. This required the ego-yaw shim: P7 is a yaw-trained checkpoint and the instrument's planner path alone feeds ego[4] = 0 (without it logits differed by up to 9.6; the visual ctx is image-only so the IoU would not have moved, but the identity then could not vouch for the path). The adapter applies `refe/ego_yaw.py patch_planner_class` with the navtest table, as `stage2_sft.py --ego-yaw-table` did.

## What was run
`p7_adapter.py` (NEW; monkeypatches `aux_navtest_forward.py` globals, edits nothing on disk): head loader from `aux_drv` (arms: trained / ZERO:: / FRESH::), trunk+tap proofs, identity vs P7's dump, rule `navsim_v1` (P7 has a 6-output scorer; the instrument's `navsim_v1_lane` needs 7), E: paths (the dev box's D: is now E:; archive on F:). Instrument code = mirror tip `6a9f9816` (`agent/arch-inf-20260803`), exported to `F:/tanitad_scratch/p7headiou/tree/`. Chain `chain.py` ran detached under the shared lock (instrument's own `acquire_gpu_lock`, released in `finally`; lock file confirmed absent after): 1,058-token forward 1,645 s wall, 1.495 s/token, 123 logs, peak 2.44 GB allocated. Readout `aux_s1_readout.py perception` with the pinned navtest store (sha 985a3bc7...) and the u400 csv/C3 table.
Not done / UNVERIFIED: the BRIEF's `taniteval/ci.py` was not used; the registered readout's bootstrap (`proxy_eval.boot` scheme) was used for every interval. The head reads the planner's fp32 ctx; the trainer saw bf16 ctx (documented ~1.05-1.46 % rel L2 difference on u400 in the instrument's smoke), so the number is the deployment-path read.

## Cheapest next arm (one paragraph)
No new trunk training. Because the head clears 0.80 on the point estimate and beats the u400 scorer on DAC-AUC while P7's picks did not move the DAC-zero share (+0.09 pp [-0.19, +0.39]), the lever is making the scorer consume the head. Two steps, cheapest first: (1) a 0-GPU offline head-veto read on the banked artifacts: use `main/head_pred_H1-s0.npz` min-SDF-along-footprint to re-rank/veto P7's own 64 proposals on the 1,058 C3 tokens (the `eval/PREREG_AUXVETO.md` family; needs P7's proposals scored against the harness DAC, i.e. a P7 table, which `stage2_p7_full.proposals.npz` has proposals for but not DAC labels) and read DAC/PDMS delta before any training; commit both outcomes first (veto lifts DAC-pass share of picks => build H2; no lift => the head's errors sit where the scorer fails and a 1 m grid / `--w-field` arm is next). (2) If (1) lifts, H2 = scorer-only fine-tune with feed ON (zero-init projection, function preserving), frozen trunk and the P7 head, seeds with an N control and a replicate (H-ESTIM-SEED-1), 4 metric families reported.

## Manifest
All under `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/headiou/` (single copy, NOT staged in git: the folder is outside the repo; the MM lands it):
`P7_HEAD_IOU.md`, `HANDOFF.md`; code `p7_adapter.py`, `proof_head.py`, `compare_det.py`, `token_iou.py`, `chain.py`, `run_chain.cmd`; results `readout_perception.json`, `token_iou.json`, `main/head_pred_H1-s{0,1,2}.npz` (s0 sha256 `515a7840...`), `main/identity.json`, `main/proof_run.json`, `main/run.json`, `proof/proof_head_weights.json`, `proof/proof_determinism.json`; logs `chain.log`, `logs/`.
Scratch (deleted, logged in HANDOFF.md): `F:/tanitad_scratch/p7headiou/ctx_full`, `detA_ctx`, `detB_ctx`. Kept: `F:/tanitad_scratch/p7headiou/tree/` (instrument export, ~25 MB, needed by the adapter), `pk/`, small helper scripts.
