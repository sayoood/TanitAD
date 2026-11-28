# BRIEF: W-A7-RJ5, re-read W-A7 on the RJ5-based labels (PREREG_P5C_A3 section 5; definition, estimator and bar from A2 section 6, unchanged). Agent: Sonnet. MM, 2026-11-17

Read `Project Steering/AGENT_OPERATING_STANDARD.md` first.
- **Binding:** stage, never commit or push; end with a manifest.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_wa7rj5/`. **Scratch:** `F:/tanitad_scratch/wa7rj5/`.
- `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Read the frozen documents from the mirror `C:/Users/Admin/tanitad-push` (`git fetch -q origin agent/arch-inf-20260803`; FETCH_HEAD; `core.autocrlf=true`, so use `git show` / `cat-file`):
  - `PK/raw/2026-11-11-p5c-readiness/PREREG_P5C_A2.md` section 6 (W-A7: definition, estimator, bar, controls, INVALID branch);
  - `PREREG_P5C_A3.md` section 5 (W-A7-RJ5: the 1,018 rank-0 keys over 11 logs, each with an RJ5 traj whose sha equals its bank row; rolling cost 0);
  - A3.1 / A3.2 / A3.3 for any amendment touching W-A7.
- **The earlier W-A7 work:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/w_a7/` and the A2 hold. Find the W-A7 instrument there and in `PK/raw/2026-11-11-p5c-readiness/`.
- **RJ5-based labels:** the selected C1 table `P5C_label_table_p5r-rj5_rank0.json.gz`, sha256 `3b198314…`.
  - Local copy: `F:/tanitad_scratch/rj5nt/label_a33/`.
  - HF: `Sayood/tanitad-refe-scorer1p6/evidence/2026-11-16-p5c-label-table-a33`.
  - Its provenance sidecar `C1_provenance.jsonl.gz` sits beside it.

## Deliverable
1. Run W-A7 exactly as A2 section 6 defines it, on the RJ5-based labels for the 1,018 W-A7 rows.
   - First check the membership claim of A3 section 5: 1,018 / 1,018 rows have an RJ5 traj, and each traj sha equals its bank row's.
   - Run every control A2 section 6 requires. If a control fails, the read is INVALID: say so, and do not "fix" it.
2. Report the result against the A2 bar exactly as written, with its estimator and n, in `WA7_RJ5_READ.md`.
   - Note whether the one A3.3 fallback row (token `d06ff3cbd9025da7`) is among the 1,018.
   - Write `HANDOFF.md` (≤ 2k tokens).
3. Dev box, CPU only (`CUDA_VISIBLE_DEVICES=-1`), serial, TMP in scratch, with a free-commit guard (wait while < 4 GB). If the instrument needs a GPU or more than ~4 CPU-h, STOP and report a plan.

## ⛔ Do not
- Change the definition, estimator or bar.
- Commit, land or push. Train.
- Use the pod (C0 trains there) or box 2.
- Print or copy secrets.
- Delete outside your scratch.
- Exceed ~150k context.

**Report:** at most 300 words, plus a manifest.
