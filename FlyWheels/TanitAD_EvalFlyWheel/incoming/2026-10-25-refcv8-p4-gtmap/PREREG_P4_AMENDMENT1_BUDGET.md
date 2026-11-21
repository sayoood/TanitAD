# PREREG_P4 — Amendment 1 (OPERATIONAL ONLY): the wall-clock guard, not the design

Master Mind, 2026-10-25 08:2x CET, written BEFORE any planner-side output of P4 has been read.

* Run 2's seed-1 roll stopped at its own budget probe (`ZZP4HALTBUDGETZZ` 07:28:01 CET): 150 windows in 626 s projected
  24,180 s for both seeds, above `--budget-max-s 16200` (4.5 h, PREREG_P4 sec. 9).
* The probe ran while the NT A1 launch gate's dev-box pinned suite shared the CPU (07:17–07:55) and a desktop screensaver held the
  GPU at 41–45 % with no compute app, so the projection is an upper bound.
* **Change:** `--budget-max-s 16200` → `30000` (8.3 h). **Nothing else changes:** design, arms, windows (all 4,754 S2 / S6),
  inversion hyper-parameters, controls, verdict rules and their order, the seal of `PREREG_P4.md`
  (sha256 `d9fc8245d14ff492bb9e0748204d5469c6f6c18be360ba0806b64cb0d0a9c6dc`).
* Run 2's partial outputs (`run/` → `run_aborted_budget_run2/`, delta cache → `delta_cache_aborted_run2`) are moved aside
  **UNREAD**; run 3 recomputes from window 0.
* Reason this is admissible: the guard is a cost control, stated as such in sec. 9; raising it cannot move any verdict.
