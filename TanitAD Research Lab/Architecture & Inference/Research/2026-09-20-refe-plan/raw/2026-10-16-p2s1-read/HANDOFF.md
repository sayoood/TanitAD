# HANDOFF: the P2-s1 navtest read, built to start by itself (2026-10-16)

State: built and tested, **nothing launched** (the auto-reader has never run for real; the 5.3 h read has not run). 0 pod, 0 eval/ or refe/ edits.
Rule: `eval/PREREG_P2_REPLICATE.md` s2-s3. Instrument: R2's, unchanged (`eval/P2_READOUT_RUNBOOK.md` s3).

## Launch (one command, detached, safe to start NOW: it only polls HF until P2-s1 ends, ~2026-10-16 12-13Z)
```
powershell -NoProfile -File "<package>\raw\2026-10-16-p2s1-read\launch_p2s1_autoread.ps1"
```
`out/` is created beside the script. Same command = resume (a FAILED marker is cleared; a DONE marker is left alone; `-Extra "--force"` re-runs).
Before it: `--dry-run` foreground (no GPU): `C:\Users\Admin\venvs\tanitad\Scripts\python.exe p2s1_autoread.py --dry-run` -> `ZZP2S1AUTOREAD_DRYRUN_PASS` (evidence: `dryrun_evidence.log`).
Needs: `D:`/`E:` letters resolving (`subst D: E:\` if D: vanished), Keys.txt readable (token read in place, never printed/written/argv, stripped from stage 2/3 children), E: free >= 6 GiB (it had 16).
Land `out/`-free: do not stage `out/` until the read is over.

## What it waits for (the chain's FINAL marker, from `p2rep_pod.sh` + `p2r_sync.py final`)
`run/summary.json` `done:true` + `model_final_sha256`, AND `run/model_final.pt` listed with server LFS sha256 equal, AND (when pushed) `logs/chain.log` `ZZP2REP_FINAL_OK model_final sha256 <hex>` equal.
`done:true` alone is NOT the marker: a light tick pushes summary.json before model_final.pt. Disagreeing hashes for 6 polls = FAILED. Then: download to `E:/Projects/TanitAD/data/refe_runs_eval/p2s1/model_final.pt`, REFUSE on any sha256/size mismatch, preflight (stage2_sft.py md5 `831a8d76...`, score_e.py + families6.py scratch deps, token file, `p2rep_read.py --check-inputs`), GPU gate (`wait_idle.py`, then lock + chain_active free), stage 2 attempt 1 (**ends `INCOMPLETE (seam failed)` by design**; continues only if the seam .npz exists), attempt 2, `stage3_full.py --name p2s1_full ... --expect-ckpt-sha256`, `p2rep_read.py`.
Lock note: `stage2_sft.py` takes `devbox_gpu.lock` itself (O_EXCL, own body); the reader does NOT hold it (that would deadlock stage 2) - it proves idle+free, then hands over in the same second. The lock is currently held by another job: the gate waits (<= 24 h).

## Watch with zero model turns
Files in `out/`: `autoread.log` (tail it), `autoread_state.json` (phase, poll_n, last_why, step_elapsed_s), `stage2_att1.out`, `stage2_att2.out`, `stage3.out`, `read.out`.
One background Monitor: `until [ -f "$OUT/P2S1READ_DONE" ] || [ -f "$OUT/P2S1READ_FAILED" ]; do sleep 120; done`. Expect DONE about 6 h after the final marker (seam 17.5 ks + harness 1.3 ks + families).

## Markers
- `P2S1READ_DONE`: a RECORDED outcome (`REPLICATED` | `REPLICATED, FLOOR-BOUND` | `NOT REPLICATED`) with RR1, Delta_seed, F, D_s0, D_s1, paths, instrument md5s. Full read: `out/read/p2rep_result.json`, `out/read/P2REP_READ.md`, line `ZZP2REP_READ ...`.
- `P2S1READ_FAILED`: names `phase` and `reason`. **An INVALID read is FAILED** (no outcome is recorded; the reason lists the failed gates). Typical: sha mismatch (re-run), seam died (stage2_att1.out), harness not PASS, a stage-3 gate failed (fix the instrument, relaunch: stage 2 reuses the seam).

## What the Master Mind does with the outcome (PREREG_P2_REPLICATE s3; numbers from `p2rep_result.json`)
1. Copy the E: raw dirs into the package (runbook s4): `raw/2026-10-05-sft-stage2/p2s1_full/`, `raw/p2_readout/p2s1_full/` (`r2_result.json`), the arm CSV `score/refe_stage2_p2s1_full/`, seam `.npz` + `.report.json` with md5s, and `out/read/`. Convert nothing: the banked CSVs are CRLF (md5 `bed5687f...`, `b17c4fbac...`); `git archive` gives LF, and `p2rep_read` accepts both forms.
2. Write `RESULT_P2REP.md` with the rule's wording + amendment s6 confounds; update `GOALS_AND_CLAIMS.md` (D-REFE-P2 rows) in the same turn.
3. **F = `RR2.F`** (|mean Delta_seed|, interval in `RR2.delta_seed_ci95`) goes to `PREREG_P6.md` "Training variance" as ONE draw, never a variance.
4. REPLICATED: lever at the mean of the two Deltas. FLOOR-BOUND: quote "direction replicated, size within run-to-run noise". NOT REPLICATED: next lever SFT-4R (Rule Zero: run it).
STRATEGIC is reported PAUSED (PI 2026-09-19), never a pass.

## Tests (dev box, tanitad venv; no GPU)
`test_p2rep_read.py` -> `ZZTEST_PASS 83` (~1.5 min; identity +0.57 [+0.02, +1.18] bit-equal to P2's banked R2; zero control exact; every outcome row; 10 source mutations RED).
`test_p2s1_autoread.py` -> `ZZTEST_PASS 73` (~1 min; real path with stub commands; 8 source mutations RED).
