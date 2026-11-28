# HANDOFF: P5-C read chain (REFe agent, 2026-11-17 ~08:20Z). Context ended early by design (never wait on a long job).

## Ready (hold = E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain/)
- `RUNBOOK_P5C_READ.md` (stage table, inputs/outputs, GPU vs CPU, ESTIMATED times from the P7 read, verdict + C0r rule, failure map).
- `p5c_read_chain.sh` (resumable, markers, serial, commit guard, exit-trap GPU-lock release, refuses on sha mismatch), `run_p5c_read.cmd`, `launch_p5c_read.ps1` (WMI, CreateFlags 0x08000200, writes launch_env.cmd because WMI children do not inherit env, sets PATH to Git usr\bin).
- `tools/`: `commit_guard.py` (integer GB, no locale), `hf_pull.py` (HF pull + sha vs pod file, token read in place), `verdict_line.py` (verdict + C0r trigger), `make_standins.py` (dry run only).
- Code pinned: mirror sha `751f3ae703097507b37548057393ae9a0d76fd78`; tip export for the dry run `F:/tanitad_scratch/readchain/tip` (re-creatable; real run needs `F:/TanitAD_p5cread_tip`, command in the runbook section 1; the export takes ~1 h on F:).

## The one command (real run, after C1 + C0 are on HF)
`$env:P5C_TIP='F:/TanitAD_p5cread_tip'; & '<hold>\launch_p5c_read.ps1'` -> `E:/Projects/TanitAD-artifacts/p5c_read/out/VERDICT.txt`. ESTIMATED ~11 h wall, ~8.8 h dev-box GPU (2 x 4.4 h seams, P7: 15,990 s), 2 x 0.6 h harness score, ~1.2 h CPU read stages.

## State of the dry run (NOT FINISHED; the only open deliverable)
- RUNNING detached: driver `dry_run_p5c.ps1` -> `p5c_dry_driver.sh`: (1) seal regression arm (C0's sha file zeroed; the chain must exit 4 at `arms_verified`), (2) identity run (P7 seam/dump/csv as C1 and C0, P6 seam as base, 64 tokens, read in `--rehearsal`).
- It is BLOCKED ON THE FREE-COMMIT GUARD (rule >= 6 GB): free commit was 2 GB because another session's python (PID 34828, ~14.9 GB private) holds memory; the chain waits up to 720 min (`P5C_COMMIT_WAIT_MIN`). Nothing of mine uses it. Check: `python tools/commit_guard.py`.
- Progress: `dryrun_driver/driver.log` (two "exit" lines when done), `dryrun_regression_seal/CHAIN.log`, `dryrun_identity/CHAIN.log`, markers in `*/markers/`.
- KNOWN DRY-RUN DESIGN POINT: `p5c_seal.verify` requires C1 != C0 and both != base, so the dry run seals two stand-ins (P7 file + 1 byte, + 0 byte, made by `tools/make_standins.py` on F:) while the seams are P7's. The identity control reads the seams, the seal reads the stand-ins; say so in the report.
- Untested until it runs (expect first-run bugs): every stage after `seal`. Suspects: `p5c_families.py --python` (navsim venv used for both families_run and fam_ci; if fam_ci fails there use the tanitad venv), `strategic_family.py` on full seams (time unmeasured), `verdict_line.py` key names (`d_headline.d_tracked`, `d_transparent`).
- Expected dry-run outcomes to verify and bank: seal regression exit 4; identity: `d_tracked` point 0 with CI [0, 0] for C1 - C0 (arms identical), strategic self-vs-self exactly 0, `read.json` verdict `REHEARSAL:...`; `identity_red` DETECTED; `det_compare` PASS. If C1-C0 is not exactly 0, that is a defect to report, not to fix away.

## Next steps for the next agent
1. When `driver.log` has both lines, read the outputs, fix chain bugs (edit hold files only; landed tools stay untouched, patches go to the hold), re-run (resume by markers; re-run the driver after clearing the failing stage marker only).
2. Write `DRYRUN_REPORT.md` (identity result, regression result, per-stage MEASURED wall times; replace ESTIMATED strategic/convert numbers in the runbook).
3. Clean scratch (literal paths, log size): `F:/tanitad_scratch/readchain/dry_models/` (2.6 GB), `F:/tanitad_scratch/readchain/tip` only if no real run is near (re-creatable; a real run needs its own tip anyway), `F:/tanitad_scratch/readchain/*.py|PREREG.md`, `<hold>/dryrun_*/tmp`. Deletions so far: none (a partial first export was overwritten in place).

## Open questions
HF layout under `p5c/c0|c1` (UNVERIFIED); base P7 vs P6 (P6 has no dump: mixed transparent forms); selection quality (64 harness seams per arm, 4-6.4 h CPU each) is not in the chain; whether stage-2 logs (which print per-arm harness lines) may be read before `chain.DONE` (treated as no).

Deletions logged: E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p5c_readchain\driver.log, dryrun_identity.stdout.log, dryrun_identity\ (stale output of a killed earlier driver; ~0 bytes of results, only empty dirs/logs).
