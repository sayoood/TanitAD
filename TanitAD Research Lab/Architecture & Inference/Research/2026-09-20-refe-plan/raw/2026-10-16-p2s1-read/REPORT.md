# P2-s1 navtest read: built to start by itself (RR1 / RR2 → P6's training-variance floor F)

**Master Mind, 2026-10-16 ~00:30 Berlin.**
- Built by a Sonnet agent; the MM re-ran both suites on this tree:
  - `test_p2rep_read.py` **ZZTEST_PASS 83**;
  - `test_p2s1_autoread.py` **ZZTEST_PASS 73**.
- The detached auto-reader was **launched** on the dev box at ~00:25 Berlin (cmd.exe pid 33716). It polls HF and stays idle until P2-s1's FINAL marker; P2-s1 is due ~10-16 12-13Z.

## What it does
- Rule: `eval/PREREG_P2_REPLICATE.md` §2–§3. Instrument: R2's, unchanged (`eval/P2_READOUT_RUNBOOK.md` §3).
- Sequence:
  1. It waits for the chain's FINAL marker: `run/summary.json done:true` + `model_final_sha256`, AND `run/model_final.pt` listed with an equal server LFS sha256, AND the `ZZP2REP_FINAL_OK` sha in `chain.log` when present.
     - A `done:true` alone is not the marker: a light egress tick pushes it before the checkpoint.
  2. It downloads the checkpoint and **refuses on any sha256 mismatch**.
  3. It runs runbook §3 verbatim with `--name p2s1_full`.
  4. It runs `p2rep_read.py`.
- `p2rep_read.py` imports stage 3's own `load_table`/`read_pair`/`decide`/`families_vs`, so the bootstrap is the program's `proxy_eval.boot` (10,000 resamples, seed 20260927). Nothing is reimplemented.
- It maps to exactly one §3 row: REPLICATED / REPLICATED, FLOOR-BOUND / NOT REPLICATED / INVALID.
- Markers:
  - `out/P2S1READ_DONE` = a recorded outcome;
  - `out/P2S1READ_FAILED` = INVALID or any failure, with the phase and the reason.

## Identity and controls (MEASURED, T1 NAVSIM v1.1, banked tables)
- **Identity:** P2-s0 vs u400 from the banked CSVs reproduces P2's banked R2 **bit-for-bit**: +0.5714677935 [+0.0179166053, +1.1755185399], and all six sub-scores.
- **Zero control:** P2-s0 vs P2-s0 gives Δ_seed exactly 0.0 with CI [0, 0]. The read then **refuses as INVALID**, which guards against P2-s0's table being read as P2-s1's (that would return REPLICATED with F = 0).
- **Mutations:** 10 source mutations of the reader go RED, and 8 of the auto-reader.
- **Dry run:** scripted HF polls, a hash disagreement, a corrupt-download refusal, command assembly and preflight on the real inputs. The evidence is in `dryrun_evidence.log`.

## Open (named, not blocking)
- The 1.29 GB `hf_hub_download` under `HF_HUB_DISABLE_XET=1` is UNVERIFIED at full size.
- `stage2_sft.py` takes the dev-box GPU lock itself, so the reader proves idle and then hands over without holding it.
- E: had 16 GB free.

## Families
As R2 reports them: LONGITUDINAL, LATERAL and TACTICAL vs u400 and vs P2-s0. STRATEGIC is **PAUSED** (PI 2026-09-19), never a pass.
