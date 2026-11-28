# DEVIATIONS / readings added after the seal (PREREG_L2.md sha256 4267cd68…, Amendment 1 c463ba55…)

Nothing below changes an arm, a bar, a split or the calibrator family.

* **D1 (reading of an ambiguous phrase, fixed before any CAL number):** PREREG §6 G says "no METRICS row separated worse … at both seeds". `l2_score.py` reads it STRICTLY, as PLAN_REFCV9 B9-0
  words it ("no family separated worse … at both inference seeds" = the guard must hold at EACH seed): a row separated worse at ANY rolled seed fails G. The offending rows are listed per seed.
* **D2 (Amendment 1):** roll order and the CAL-s1 early stop are operational, sealed (c463ba55…) before any roll.
* **D3 (code, no definition changed):** the TRAIN loader reads the 6 s future from `decisions/gt_future_ext` (steps 50 and 60) instead of an S6 panel, because L1's train dump has no S6 panel; it was
  verified bit-equal to the panel-based class on the EMA-30k train dump (`smoke/train_loader_check.py`: same 3,749 windows, same classes). `l2_roll.py` reads W from `model.cfg.core.window` via the
  top-level model (the L1 route) before the seam module's own cfg.
* **D4 (exploratory, NOT in the prereg, post-seal):** `result/Z4_nav_split_EXPLORATORY.json` — the remaining false-decel split by the per-frame nav token. Reported as exploratory only.
