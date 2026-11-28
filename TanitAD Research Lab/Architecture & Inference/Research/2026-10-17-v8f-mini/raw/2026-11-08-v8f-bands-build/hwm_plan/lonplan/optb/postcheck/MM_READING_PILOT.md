# MM reading: Option B pilot (box 2, commit 86da79cd, 2026-11-18 ~04:00Z)

Evidence class MEASURED. The pilot files were pulled from box 2 and sha256-checked by the chain: `C:/Users/Admin/optb_state/pilot/` (PILOT.json, PILOT.verified.json, summary.json, selfcheck.json, train_log.jsonl, gate_token.json).

* **Stop rules (MM, BUILD_OPTB step 4): both PASS.**
  * **Speed:** mean 0.548 s/step (median 0.543), against a limit of 1.0.
  * **Memory:** peak GPU memory 3,508 MiB, sampled by nvidia-smi 540 times. That reading includes box 2's idle 663 MiB, so it is an upper bound on `max_memory_allocated`. The limit is 6.5 GB.
* **Projection (the chain's permit):**
  * training alone projects 6.1 h for 10 x 4,000 steps, inside the 72 h limit;
  * the pilot's wall time was 2,727 s for 200 steps, of which about 110 s was training. So about 44 min per arm is overhead (gate, start-up, bank);
  * that gives about 80 min per arm, about 13.5 h in all (ESTIMATED from these MEASURED parts), inside the card's 14-16 h.
* **Trainable-parameter pin:** `n_trainable` = **1,978,562** at all three reads (summary and selfcheck). That is 2,011,970 minus `wm.proj` (33,408).
  * Recorded here as the pin. Every arm's banked `selfcheck.json` must read this number, and the MM checks it at the read.
  * `optb_check.py` is not edited under a running chain: it is a member of the shipped manifest.
* **TARGET-READ (beside, no bar):** the term trains down fast in the pilot.
  * `rsp/L1` read 0.0297 / 0.0148 / 0.0122 / 0.0064 at steps 50 / 100 / 150 / 200.
  * `rsp_g_aa` read 0.941 / 1.001 / 0.962 / 1.015, with `s_a` held at 0.2008 (the frozen constant).
  * The term meeting its own readout is the case the card's OPTB-NULL (i) row names ("the WM met its own readout without a response a second readout sees"). The cross-fitted IA-3P-K read, on a readout the term never saw, is what tells the two apart. Nothing is concluded from the pilot.
* The chain started FTRSP_s0 at once.
