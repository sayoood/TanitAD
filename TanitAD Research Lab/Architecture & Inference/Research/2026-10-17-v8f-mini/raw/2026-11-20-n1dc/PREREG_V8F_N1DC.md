# PREREG_V8F_N1DC: a direct path from T's code into O's plan head

**FROZEN 2026-11-20, before any N1DC arm, build or number exists.**
* PI go, first-hand in the MM session: *"Yes, N1DC next"* (answer to "Park the weight line and take N1DC next (about 1.5 pod-h)?"). Then: *"continue and accelerte autonmously the progress"*.
* Basis: decision 6.1(c) of `raw/2026-11-19-n1dw/read/verdict/MM_READING_N1DW_VERDICT.md`. The N1DW weight line is parked. Its verdict of record stays as landed (94636bf7), and the hb0.2 overlap read is reported beside it when it lands.
* This card is the MM's draft of 2026-11-19, updated at freezing in four places, each marked **[F]**: the executor of record, the conditional arm, the coordinator-review controls carried from N1DW, and the STRATEGIC S1-slot view.

## 0. Why this lever, and what it aims at (all MEASURED, tier T0 unless marked)
The chain of zero-training reads on N1D (D1-HO, 5,736 windows / 497 held-out episodes, tree probe, paired episode-cluster bootstrap, B 2,000, seed 0) places the roundabout decision's loss at ONE step:

| stage | N1D_s0 | N1D_s1 | read |
|---|---|---|---|
| route text alone | 0.6044 | 0.6044 | PROBE-N1D-FEAT |
| route text + ego state at NOW | 0.6855 | 0.6855 | PREREG_N1D_CEIL |
| T's 60-column code that O reads | 0.6702 | 0.6783 | PREREG_N1D_TCODE (CODE-CARRIES) |
| O's plan-head input | 0.7028 | 0.6937 | PROBE-N1D-FEAT (PROBE-PRESENT-UNUSED) |
| route + ego + plan feature | **0.7306** | **0.7287** | PREREG_N1D_CEIL (CEIL-AT-PLAN): the stated **target** |
| the plan O decodes, tick 0 (T0) | **0.6266** | **0.6349** | PREREG_N1D_T0T1 (DECODE-LOSES) |
| the rolled track (T1-kin/old) | 0.6196 | 0.6499 | NAVRBD |

* The decision is carried to O's plan head and lost in decoding.
* Today O reads T's code only as ONE of its conditioning tokens. The cross-attention reader `read_o` mixes it with the ego, nav, limit, checkpoint and age tokens and with the scene, before `head_o` sees the pooled result (`v8f_mini.py` `run_o`).
* This lever gives `head_o` a direct, linear read of the same code.
* **Target, from CEIL:** D1-HO TAC-LAT 0.73, from the decoded 0.63. The room is about +0.10 (ESTIMATED), 2.6 bars.

## 1. The lever (one flag, default off)
* `--o-code-skip on|off`. The default `off` builds nothing and is the parent, bit for bit.
* With `on`: `o_code = nn.Linear(CODE_DIM, n_q * d, bias=False)`. It is zero-initialised and built last under a forked RNG, as `e_nav_rb` is (PREREG_V8F_NAVRB2).
* In `run_o`: `h = read_o(...).flatten(1)`, then **`h = h + o_code(code)`**, then `head_o(h)`.
  * `code` is exactly the tensor `run_o` already receives: after teacher forcing (`code_mix`), `detach_upper` and `o_interface`. So the skip reads what `e_code` reads, no more.
  * Step 0 therefore equals the parent bit for bit (x + 0.0 = x).
* **Admissibility:** the code is T's output. It reads only `INFERENCE_INPUT_KEYS`, with no situation classifier and no label at inference (`assert_no_situation_classifier_input` stays green). Nav is an input at training and at inference.

## 2. Arms
Base: N1D's recipe and initialisation. Code: the N1DW commit 134f5da1 (branch `agent/v8f-n1dw-20261119`) plus this flag, at `--rb-window-w 1.0` (its default, which builds nothing).

| arm | argv | GPU (ESTIMATED from the N1DW chain: about 12.5 min per arm) |
|---|---|---|
| N1DC_s0 | N1D_s0's banked argv + `--o-code-skip on` | ~13 min |
| N1DC_s1 | N1D_s1's banked argv + `--o-code-skip on` | ~13 min |

* With gates and the D1-HO and eval-141 dumps on CPU, the total is about **1.5 to 2 LoRA pod-h** (ESTIMATED).
* **[F] Conditional arm:** the draft's N1DC at the carried weight needed N1DW to read RBW-GAIN. N1DW reached no row, so that arm does not run under this card.
* Pod order: after the ADD45 GPU chain and the N1DW hb0.2 overlap read, under `flock /root/gpu_queue.lock`, one GPU job at a time.

## 3. Read
* **[F] Executor of record: T1-kin/hb0.2** (`stack/scripts/v8f_t1_eval_hb.py --kappa-max 0.2`, bcf267ce; PI ruling D16-D: cards frozen from 2026-11-20 read under the bounded executor). T1-kin/old is reported beside every cell.
  * N1D's comparator dumps under hb0.2 on D1-HO come from the overlap read (`raw/2026-11-19-n1dw/overlap_hb/`). On eval-141, N1D is re-dumped under hb0.2 (2 dumps plus their zero twins, CPU).
* **Deciding cell:** D1-HO kinematic TAC-LAT, delta **N1DC - N1D**, paired, per seed.
* **Bar:** max(0.0383, F_HO 0.0424, |N1D_s1 - N1D_s0|, |N1DC_s1 - N1DC_s0|), with both seed draws read under the executor of record.
* **Beside, no bar:**
  * N1DC - N1R;
  * LONGITUDINAL and LATERAL on D1-HO, as N1DW R1. Distance-keeping there is NOT-COMPUTABLE (n = 0: no lead block on the held-out clips);
  * the **T0 decoded-plan TAC-LAT** of N1DC against N1D's 0.6266 / 0.6349 (does decoding approach 0.73?);
  * Z-USE: the skip zeroed at inference (`o_code` output x 0), real minus zero.
* **Guard:** the four families on eval-141, no cost at either seed (as N1DW 5.2). Distance-keeping is computed there.
  * **STRATEGIC is reported in full** (PI 2026-11-16).
  * **[F]** The S1-slot view `n1dw_view_s1slot.py` (landed 94636bf7) runs in the chain, so S_ROUTE / S1 cells come out with the read.
* **Estimator:** paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES" only. Training variance is the two seeds; the executor is deterministic.

## 4. Outcomes (committed now)
| outcome | condition | next |
|---|---|---|
| **N1DC-GAIN** | D1-HO gain above the bar at BOTH seeds, and the guard holds at both | carried; s10/s11 (PI go) |
| **N1DC-ONE-SEED** | gain at one seed only, guard holds at both | s10/s11 before any claim (PI go) |
| **N1DC-GAIN-COST** | gain at one or both seeds, with a guard cost at one seed | reported as such. Attribution against N1D on the cost cell, as N1DW section 3.3. Back to the PI with the cost family named |
| **N1DC-COST** | guard cost at both seeds, or D1-HO a cost at both | not adopted; the cost family gets a card |
| **N1DC-NO-GAIN** | otherwise | the decoder is not access-limited. Next: an auxiliary route-decision head on O's plan feature (kinematic label, CE), the T0T1 card's lever, under its own card |

* **[F]** The N1DC-GAIN-COST row is added at freezing. It closes the outcome gap N1DW met (a gain at one seed with a one-seed guard cost had no row).
* An arm that misses its bar FAILED and is reported so, and the next arm is built (RULE ZERO).
* Bars, arms, executor and outcomes do not change after any number is seen.

## 5. Controls (build, before any launch)
* **Z0:** with the flag off, the tree is bit-identical to 134f5da1 for the state dict and every loss term.
* **Z1:** with the flag on, the step-0 forward, every loss term and every output are bit-identical to the parent (zero-init). Checked on CPU.
* **REACH:** `o_code.weight` gets a NON-ZERO gradient at step 0, and a control parameter must read exactly 0.0. The zero-grad gate runs after training.
* **Mutation:** delete the one `h = h + o_code(code)` line, and REACH must go RED.
* **G-DVB entry** for the new flag (the gate refuses a new lever without one). Full launch gate, with per-arm tokens. The R2 comparator compares STEP-0 terms only, on the same seed.
* **[F] R2:** every bit-identity test runs on CPU, fp32, one thread.
* **[F] K-CLIP:** the share of logged steps with `grad_norm` above `clip_grad`, per arm, read with the N1DW `kclip.py`, reported beside.
* **Clean judged tree:** the stray `eval_grid_windows.json` is removed from the pod tree, and G-SUITE re-runs on the clean tree.
* **Read controls** (as the N1DW read): K1 self-pair exactly 0, K-CONST 0.338215, and Z-EVAL. K-RE is read against the banked N1D under the same executor.
