# GAP48 and RALIGN (2026-11-07): the render gap on 48 scenes, and an inference-time map that fails its mechanism gate

* **`gap48/`:** the box-2 job that measured the NuRec render gap on the 48 read scenes. It renders at the logged pose with actors attached (`objects`) every 10th tick: 960 pairs. `RESULT_GAP48.md` is descriptive and decides nothing.
  * Render to real: cosine 0.8401 [0.8307, 0.8503].
  * Render to the BLIND mean: 0.7098. Real to the BLIND mean: 0.7427. So a render lies FARTHER from the mean than its real frame does.
  * The chain's first launch refused on a false clip mismatch. Windows Python's CRLF left a CR in the expected byte count. The fix is in `gap48_chain.sh`; the earlier copy is kept beside it.
  * The 157 MB of token pairs are not in the repo. They are on box 2 at `D:/t2v8f/pull_gap48/` and on the dev box at `F:/TanitAD-store/gap48_pull/`. `gap48_pull_summary/GAP48_RESULT.json` is the job's own summary.
* **`ralign/`:** the card `PREREG_T2_RALIGN.md` (frozen `a6c84433`; `cards/`, with its log), implemented and tested on synthetic data before any token existed (41 tests).
  * **Result: ALIGN-FAILS-MECH** (`ralign/out/M_ALIGN.json`). The held-out median cosine rises from 0.8401 to 0.8937 (+0.0536), and the level misses 0.90 by 0.0063.
  * The same-fold regression arm is flagged, and reaches only 0.9015 itself.
  * By the card, no T2 stage runs, and the next lever is encoder adaptation. The maps are in `ralign/out/maps/`. The identity package for G-IDMAP was built and is run beside, deciding nothing.
* **`cards/THEORY_CRAWL_AMEND_4_LOG.md`:** the T6 log. P6 is VISION-USED, so T6 survives. Section 5 (a) does not hold, because renders are not mean-like. Section 5 (b) is ALIGN-FAILS-MECH.
* **`lora_L_launch/`:** the LoRA arm L's launch evidence, copied from the PI's GPU pod.
  * Gate token and its verify; the batch-64 timing; L_s0's argv, selfcheck and supervisor log.
  * `train_log.jsonl` to step 200: 2,258 s at step 50 and 8,994 s at step 200, so 44.97 s/step MEASURED.
