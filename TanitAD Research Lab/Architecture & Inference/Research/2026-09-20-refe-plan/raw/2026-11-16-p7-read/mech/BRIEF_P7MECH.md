# BRIEF: SPEC_P7 §5 NOT-PROVEN branch: the mechanism read (DAC-zero share, head IoU, head-vs-scorer DAC AUC) that picks P7's next lever. Agent: Sonnet. MM, 2026-11-16

Read `Project Steering/AGENT_OPERATING_STANDARD.md` first.
- **Binding:** stage, never commit or push; end with a manifest.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/`. **Scratch:** `F:/tanitad_scratch/p7mech/`.
- **Dev box:** CPU only, `CUDA_VISIBLE_DEVICES=-1`, TMP in scratch, serial runs. Free commit is tight (~3-5 GB). Before any process expected above 1 GB, check `(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB` and wait (in a script) while it reads below 4 GB.

## The rule being executed
`SPEC_P7.md` §5 (repo `PK/SPEC_P7.md`, `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`).
- **Mechanism endpoints (reported, not gating):** the DAC sub-score and the DAC-zero share vs P6; head IoU over [0, 40) m on navtest and on the reserve (bar 0.80, K2); head-vs-scorer DAC AUC.
- **Next arm, NOT PROVEN branch:** "read the DAC-zero share and head IoU; head IoU < 0.80 even trunk-trained ⇒ the 1 m grid or the field-loss arm (`--w-field`, D-AUX-4)".

## Facts (MEASURED, 2026-11-16)
- **P7 read, complete and valid:** chain.DONE, determinism rep bit-identical. Work dir `E:/Projects/TanitAD-artifacts/p7_full_read/`; stage-2 outputs in `F:/TanitAD_p7read_tip/<PK>/raw/2026-10-05-sft-stage2/p7_full/` (`result.json`, `p7_full_seam.npz`).
- **Per-token harness CSV:** `E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p7_full/refe_stage2_p7_full.csv`.
- **Proposals dump:** `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7_full.proposals.npz`, which holds all 64 proposals and the scorer logits.
- **P7:** 91.0724 PDMS (DAC 97.8594). vs P6: −0.12 [−0.51, +0.28], NOT PROVEN.
- **P6's per-token CSV and seam:** find them through `PK/eval/stage3_full.py` `BASELINES['p6']`, which pins them by md5.
- **P7 checkpoint:** `F:/TanitAD_p7read/run/model_final.pt` (sha256 `2613fc17…`). It carries the aux drivable-area head (`aux_drv`).
- **The reserve:** `reserve_sessions.json` (R-A `53f3bcd4…`); see SPEC_P7 §0–§3 and `PK/raw/` for the P7 card.

## Deliverable, in priority order (bank each result as you go)
1. **DAC-zero share, P7 vs P6, same 12,146 tokens.** Paired log-cluster bootstrap (`taniteval/ci.py`, or `proxy_eval.boot`, as stage3 uses). Also give the DAC sub-score difference.
   - Report the DAC-zero tokens P6 had and P7 fixed, and the reverse.
2. **Head IoU [0, 40) m on navtest** (P7's aux head vs the drivable-area ground truth the head was trained on).
   - FIND the existing instrument first: SPEC_P7 §1–§3, `PK/refe/` (aux head, `aux_drv`), `PK/raw/2026-10-22-p7-wire-head/`, P7 VDAC reads.
   - If an instrument exists, run it and report n and the estimator. If none exists, write the smallest correct one, test it on a known case, then run it.
   - It needs the model forward on CPU over navtest frames: estimate the cost first. If it is above ~3 CPU-h or needs the GPU, STOP and report the plan; the dev-box GPU is shared and C0 trains on the pod.
3. **Head-vs-scorer DAC AUC**, if the dump plus head output allow it cheaply.
4. **`P7_MECH_READ.md`:** each number with its evidence class, estimator and n; which SPEC_P7 branch it triggers (head IoU < 0.80 ⇒ 1 m grid or the `--w-field` arm); the cheapest next-arm spec, in a paragraph; and `HANDOFF.md` (≤ 2k tokens).

## ⛔ Do not
- Commit, land or push.
- Start any training.
- Use the pod (C0 is training there) or box 2.
- Re-run the navtest seam.
- Treat a trainer-log number as a result.
- Delete outside your scratch (literal paths; log each deletion).
- Exceed about 150k context.

**Report:** at most 300 words, plus a manifest.
