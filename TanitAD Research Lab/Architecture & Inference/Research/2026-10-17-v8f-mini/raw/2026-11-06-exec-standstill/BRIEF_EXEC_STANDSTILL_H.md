# Brief: run the frozen executor-standstill read at T1-kin on eval-141 (AGENT H; CPU only)

**Card:** `PREREG_EXEC_STANDSTILL.md` (same folder), FROZEN; its sha256 is in `FROZEN_SHA.txt`. ⛔ Never edit it.

**Rules.**
* The preamble `E:/Projects/TanitAD/Project Steering/AGENT_OPERATING_STANDARD.md`.
* CPU only (`CUDA_VISIBLE_DEVICES=-1`), no ssh, no downloads, no new packages. Never commit, stage or delete: move aside. LF endings,
  and the banned words.
* Python is `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`. Write Python with the Write tool, never a bash heredoc.
* You own `motion_c/exec_standstill/` ONLY. Another agent edits `motion_c/t2v8f/`: read it, never write it.
* Run long jobs detached: WMI `cmd /c ... < NUL` (memory `devbox-drive-letter-and-detached-jobs.md`). Run one heavy job at a time:
  the dev box's C: has filled before.
* Never stay alive waiting more than about 4 minutes. Write `HANDOFF_H.md` (≤ 2k tokens: state, job, paths, the exact next step) and
  end. A fresh agent finishes from it.
* ≤ 150k context.

**Read first:**
* The card.
* `motion_c/t2v8f/AUDIT_NUREC_SCORE.md`: the mechanism, and the T1 reader's standstill rows.
* How the T1-kin tool is run: the T1 tool is `v8f_t1_eval_ri_lf6.py` (sha256 `949c4b7a...`). Find its source, cache and checkpoint
  paths in `motion_c/lf6_result_pkg/` (its HANDOFF and build scripts) and in `motion_c/t2v8f/HANDOFF_D.md` LEFT 1.
* The banked base rows: `R/raw/2026-11-05-motion-c-ri/ri_read/RESULT_RI_C0.md` in the repo mirror
  `git -C C:/Users/Admin/tanitad-push show <tip>:<path>`.

## Items, in order
1. **The variant.** Make a COPY of the T1 tool under `exec_standstill/code/` with a `--exec-eps` flag (default 1e-3 = base). Below eps
   the executed heading is held and the yaw rate is 0. Prove the gates of card section 4 on a small window subset, each with its RED arm:
   * G-ID: eps 1e-3 reproduces the banked rollouts bit for bit;
   * G-MOV;
   * G-PHYS;
   * G-P10.
2. **The read.** C0_s0, C0_s1 and P10 under base and under hold (eps 0.1), with DAGOACC beside, on all eval-141 windows.
   * Compute E = M1(C0) − M1(P10) per seed and executor, and ΔE with the paired episode-cluster bootstrap (B 2000, seed 0).
   * Report all 19 rows beside, with lateral and heading rows per start class.
   * Report the card's yaw-rate diagnostic.
3. **`RESULT_EXEC_STANDSTILL.md`.** The outcome per card section 5, with every number's tier, artifact path and estimator. If the
   outcome is 1 or 2, add the blast-radius list: which banked T1-kin launch rows used the base executor.

## Hand-back (≤ 300 words)
Gates (with RED arms), the outcome with ΔE and CIs at both seeds, the diagnostic, wall time, open items, and a deliverable manifest.
