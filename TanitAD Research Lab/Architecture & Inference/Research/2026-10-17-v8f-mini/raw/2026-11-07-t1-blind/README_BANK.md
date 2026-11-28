# T1-kin BLIND read (2026-11-07): VISION-USED, so RENDER-GAP-BINDS at T2

* **Card:** `cards/PREREG_T1_BLIND.md`. The first 4,574 bytes are frozen, sha256 `9619414a...`. The log follows: the scoping entry, written before any BLIND1 row existed, and the result entry.
* **Result:** `RESULT_T1BLIND.md` / `.json` and `raw/READ_T1BLIND.json`. Tier T1-kin, eval-141, paired episode-cluster bootstrap B 2,000.
  * All gates PASS at both seeds: G-ID bit for bit against the banked RIC0; G-MEAN on 115,152 received frames; G-EGO on 2,399 tick-0 windows.
  * ADE6 B = BLIND1 − RIC0 is +0.8313 [+0.5576, +1.1530] / +0.8158 [+0.5309, +1.1454] m.
  * M1 B is −0.0246 / −0.0415, separated-better: without its tokens, the planner crawls less.
* **What it commits:**
  * the token-use gates in the LoRA card (`cards/DRAFT_PREREG_T2_DOM_LORA.md`, open item 5);
  * the render-robustness card `cards/PREREG_T2_RALIGN.md` (frozen `a6c84433`, written before any GAP48 token exists).
* **Theory:** `cards/THEORY_CRAWL_AMEND_4.md` (`016c2950`). It states T6 (render-domain information loss) and its rival T6′ before this read. VISION-USED keeps T6 and drops T6′.
* **Beside:**
  * `gapscale/`: the yardstick for the stage-1 render gap, descriptive only;
  * `raw/DK141_TABLE_T1BLIND.md`: distance keeping on the tick-0 plan.
* **Code:** `t1blind_run.py` (a wrapper over the pinned T1 tool; the done markers pin its sha), `t1blind_gates.py`, `t1blind_read.py`, `t1blind_result.py`, `t1blind_dk.py`, the chain, `tests/` and `logs/`.
* **Rollouts:** on the dev box at `F:/TanitAD-scratch/motion_c/t1_blind/out/`.
