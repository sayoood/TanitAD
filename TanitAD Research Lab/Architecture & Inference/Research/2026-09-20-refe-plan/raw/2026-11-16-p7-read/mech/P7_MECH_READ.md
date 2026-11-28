# P7 mechanism read (SPEC_P7 section 5, NOT-PROVEN branch) - 2026-11-16

Tier: T1 NAVSIM v1.1 PDMS (self-action open loop), the harness CSVs; the head items are T0-like and were NOT run.
Caveat: `SPEC_P7.md` is not present in the E: tree (find over repo and `PK` found none); its section 5 is quoted only as the BRIEF quotes it. UNVERIFIED against the spec text.

## 1. DAC-zero share, P7 vs P6 (DONE)
MEASURED. Inputs: per-token harness CSVs `data/refe_navtest/score/refe_p6final/refe_p6final.csv` (md5 2b8f7701..., equals the `BASELINES`/r2_result base pin) and `.../refe_stage2_p7_full/refe_stage2_p7_full.csv`. n = 12,146 tokens, 136 logs, same token set. Estimator: paired log-cluster bootstrap, 10,000 resamples, seed 20260927, 95 % (the `proxy_eval.boot` form stage3 uses; answers "another draw of logs" only, not training or inference variance). Script and output: `dac_zero.py` (scratch), `dac_zero_p7_vs_p6.json` (hold).

| | P6 | P7 |
|---|---|---|
| DAC == 0 tokens | 249 (2.050 %) | 260 (2.141 %) |
| DAC mean (x100) | 97.950 | 97.859 |

- Zero-share diff P7 - P6: +0.0009 [-0.0019, +0.0039] (+0.09 pp). DAC sub-score diff: -0.0009 [-0.0039, +0.0019]. Both intervals straddle 0.
- Churn: P7 fixed 119 P6 DAC-zero tokens and broke 130 others; 130 are zero in both. DAC is binary in both arms (0 non-binary values).
- Reading: the aux head did not reduce DAC failures. 249 -> 260 is net noise on a rig whose single-seed floor is unmeasured (H-ESTIM-SEED-1); 249 of 12,146 is the whole DAC headroom (about 2 PDMS points).

## 2. Head IoU over [0, 40) m on navtest (NOT RUN: over the CPU budget; plan below)
Instrument FOUND: `PK/eval/aux_navtest_forward.py` (+ `test_aux_navtest_forward.py`), readout `PK/eval/aux_s1_readout.py` (P1 IoU in longitudinal bands, P3 AUC, K2 fires on IoU), store `aux_drivable/label_store_navtest` (12,146 keys, sha 985a3bc7...), results `PK/raw/2026-10-13-aux-navtest-forward/RESULT.md`. It drives the full planner path (nuPlan scenario build, route, 4-camera DINOv3-ViT-L trunk) one token at a time.
Cost (from RESULT.md, GPU RTX 4060, MEASURED there): 3.0 s/token warm, 10.8 s/token cold + 54 s per log; 1,058 C3 tokens = 38 min to 5 h on the GPU. On CPU the ViT-L x 4 cameras trunk is slower by an ESTIMATED 5-20x (not measured by me): 1,058 tokens is several to tens of CPU-h, all 12,146 tokens is days. Both exceed the 3 CPU-h bar, and the instrument takes the shared GPU lock. Stopped per the brief.
P7 checkpoint facts (MEASURED, `F:/TanitAD_p7read/run/model_final.pt`): `aux_drv` is a separate top-level entry (not in `model`), head depth 2, 2,440,450 params, tap `visual`, feed off (`feed_built` False), w_aux 0.1, grid cell **2.0 m**, x0 -8, nx 40, ny 32 (so [0, 40) m = cells 4..23). The head is a read-only auxiliary; P7's scorer does not read it (feed off), so a head IoU cannot by itself move DAC.
Adapter needed: the tool loads the u400 trunk and an external `aux_drv.pt`; for P7 it must load the trunk from `model_final.pt` and the head from `ckpt["aux_drv"]` (identity check against the P7 dump logits in `stage2_p7_full.proposals.npz` is available: 12,146 x 64 x 6).

## 3. Head-vs-scorer DAC AUC (NOT RUN)
Same forward as item 2 (needs head SDF at each proposal footprint); P3 in `aux_s1_readout.py` reads it on the 1,058 C3 tokens x 64 proposals. Blocked by the same cost.

## 4. Branch triggered
PDMS NOT PROVEN (-0.12 [-0.51, +0.28]) triggers the section-5 NOT-PROVEN branch. Its discriminator (head IoU < 0.80 even trunk-trained => 1 m grid or `--w-field`, D-AUX-4) is UNRESOLVED: head IoU is unmeasured. What IS measured: the DAC-zero share did not move (+0.09 pp, CI spans 0), which is consistent with either (a) head IoU < 0.80 (head too coarse; 2 m cells already coarser than the 1 m grid the branch names) or (b) a good head the scorer never reads (feed off). Those two are separable only by the head IoU.

## 5. Cheapest next arm (spec)
No training. One GPU-lock window on the dev box (the pod stays on C0): adapt `aux_navtest_forward.py` to the P7 checkpoint and run the chunked resumable command from its `HANDOFF.md` over the 1,058 C3 tokens (38-53 min GPU, warm), then `aux_s1_readout.py` for P1 IoU in [0, 40) m (log-cluster bootstrap) and P3 AUC head vs scorer. Commit before the run: IoU >= 0.80 => the head is fine and the lever is the scorer-side feed (`feed on`, D-AUX); IoU < 0.80 => train the 1 m grid / `--w-field` arm. If the PI will not release the GPU, the same read needs a CPU run chunked over several agents (hours), which I do not recommend.
