# BRIEF: HNSCREEN-FP-COST, implement the MM ruling (b′) (Master Mind, 2026-10-19 ~22:55)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/hnfp3/`.
- A CPU read (PRIVGEOM-CLEAN) is running and free RAM swings from 4 to 10 GB. Run one heavy Python process at a time, each under 2 GB, and gate on free RAM ≥ 4 GB before starting a test file (reuse `C:/Users/Admin/par18/hnfp2/suite_runner.py`'s RAM gate).
- CPU only. Never commit or push.

**The ruling to implement:** `C:/Users/Admin/par18/hnfp3/MM_RULING_FPCOST.md`. Read it fully. Its numbered rule and its RED arms R1-R7 are the spec.

**Base:** the mirror tip only (`git -C C:/Users/Admin/tanitad-push fetch -q origin agent/arch-inf-20260803`, then pin `SHA=$(git rev-parse FETCH_HEAD)` ONCE and use the SHA everywhere after; FETCH_HEAD moves under you).

The previous package is `C:/Users/Admin/par18/hnfp2/`:
- it built the content fingerprint;
- `out/` is its landed package, with RESULT.md and COST.md;
- `suite_runner.py` and `cfg_*.json` run base-vs-mine over every test file that names an edited file.
The code is in `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/`: `launch_gate.py`, `hn_fingerprints.py`, `ctx_cache.py`, the HN screen chain and reader, and `test_hn_content_fp.py`.

## Deliverables, in PRIORITY ORDER
1. Find every caller that re-verifies the ctx cache inside one HN screen chain (`file:line` per hop). Write the call map first, in `hnfp3/CALLMAP.md`.
2. Implement (b′) with the smallest change, behind an explicit mode, with these defaults:
   - **content** everywhere except the HN screen chain's intermediate calls;
   - the chain writes the START anchor and the END verify;
   - the reader refuses without `END_VERIFY_OK`.
3. Tests:
   - R1-R7 as named tests;
   - plus mutation arms: delete the single line that enforces each refusal and require the test to go RED (record each mutant's result);
   - R7 is an identity arm (bit-for-bit equal to today's behaviour in content mode).
4. Measure the reads per screen, on a small synthetic cache, by counting full reads (instrument the reader): it must equal 4 per screen, with today's count of 17 as the control.
5. The base-vs-mine suite over every test file that names an edited file, full form (reuse hnfp2's runner). Required: 0 new failures.
   - It is slow (about 90 min). Launch it detached (WMI `Invoke-CimMethod Win32_Process Create`, `cmd /c "... < NUL >> log 2>&1"`), write `hnfp3/HANDOFF.md` (2k tokens or fewer: state, how to check, the exact staging step), and END. Do not wait on it.
6. Stage the package:
   - `hnfp3/out/` with `paths.json` items `[repo path, abs src, "raw"]`;
   - the repo dir `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-hnscreen-fp-cost/`, with `RESULT.md`, `CALLMAP.md`, the ruling, test logs, the mutant table, and the edited `refe/` files at their repo paths;
   - a `code_vs_tip.diff`.

## Hand-back (300 words or fewer)
The call map summary, the reads-per-screen measurement (4 vs 17), the R1-R7 and mutant results, the suite status (or its handoff), and the manifest.
