# PREREG_V8F_N1DC (DRAFT for the PI's go; frozen only on that go, before any N1DC arm exists): a direct path from T's code into O's plan head

**Status:** DRAFT by the Master Mind (MM), 2026-11-19. Nothing is built and no arm exists.
* It is frozen, and its sha256 recorded, only when the PI says go.
* It changes no bar or outcome of PREREG_V8F_N1DW, whose read is in progress.

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
| the rolled track (T1-kin) | 0.6196 | 0.6499 | NAVRBD |

* The decision is carried to O's plan head and lost in decoding.
* Today O reads T's code only as ONE of its conditioning tokens. It is mixed by the cross-attention reader `read_o` with the ego, nav, limit, checkpoint and age tokens and with the scene, before `head_o` sees the pooled result (`v8f_mini.py` `run_o`).
* This lever gives `head_o` a direct, linear read of the same code.
* **Target, from CEIL:** D1-HO TAC-LAT 0.73, from the decoded 0.63. The room is about +0.10, 2.6 bars.

## 1. The lever (one flag, default off)
* `--o-code-skip on|off` (default `off` = nothing built, bit for bit the tip).
* With `on`: `o_code = nn.Linear(CODE_DIM, n_q * d, bias=False)`, zero-initialised and built last under a forked RNG, as `e_nav_rb` is (PREREG_V8F_NAVRB2).
* In `run_o`: `h = read_o(...).flatten(1)`, then **`h = h + o_code(code)`**, then `head_o(h)`.
  * `code` is exactly the tensor `run_o` already receives: after teacher forcing (`code_mix`), `detach_upper` and `o_interface`. So the skip reads what `e_code` reads, no more.
  * Step 0 therefore equals the parent bit for bit (x + 0.0 = x).
* **Admissibility:** the code is T's output. It reads only `INFERENCE_INPUT_KEYS`; no situation classifier, no label at inference (`assert_no_situation_classifier_input` stays green). Nav is an input at training and inference.

## 2. Arms (base = N1D's recipe and initialisation, code = the N1DW commit 134f5da1 + this flag)
| arm | argv | GPU (ESTIMATED from the N1DW chain: about 12.5 min per arm) |
|---|---|---|
| N1DC_s0 | N1D_s0's banked argv + `--o-code-skip on` | ~13 min |
| N1DC_s1 | N1D_s1's banked argv + `--o-code-skip on` | ~13 min |

* Plus gates and the D1-HO and eval-141 dumps on CPU: about **1.5 LoRA pod-h** in all.
* **Conditional, only if N1DW reads RBW-GAIN** (not asked now): N1DC at the carried weight, s0/s1.

## 3. Read
* **Deciding cell:** D1-HO kinematic TAC-LAT, delta **N1DC - N1D**, paired, per seed.
* **Bar:** max(0.0383, F_HO 0.0424, the within-arm seed draws |N1D_s1 - N1D_s0| and |N1DC_s1 - N1DC_s0|).
* **Beside, no bar:**
  * N1DC - N1R;
  * LONGITUDINAL and LATERAL on D1-HO (as N1DW R1);
  * the **T0 decoded-plan TAC-LAT** of N1DC against N1D's 0.6266 / 0.6349 (does decoding approach 0.73?);
  * Z-USE: the skip zeroed at inference (`o_code` output x 0), real minus zero.
* **Guard:** the four families on eval-141, no cost (as N1DW 5.2). STRATEGIC is reported in full.
* **Estimator:** paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES" only; training variance is the two seeds.

## 4. Outcomes (committed on freezing)
| outcome | condition | next |
|---|---|---|
| **N1DC-GAIN** | D1-HO gain above the bar at BOTH seeds, guard holds at both | carried; s10/s11 (PI go); combine with N1DW if that gained |
| **N1DC-ONE-SEED** | gain at one seed only | s10/s11 before any claim |
| **N1DC-COST** | guard cost at both seeds, or D1-HO a cost at both | not adopted; the cost family is carded |
| **N1DC-NO-GAIN** | otherwise | the decoder is not access-limited. Next: an auxiliary route-decision head on O's plan feature (kinematic label, CE), the T0T1 card's lever, own card |

## 5. Controls (build, before any launch)
* **Z0:** with the flag off, the tree is bit-identical to 134f5da1, for the state dict and every loss term.
* **Z1:** with the flag on, the step-0 forward, every loss term and every output are bit-identical to the parent (zero-init). Checked on CPU.
* **REACH:** `o_code.weight` gets a NON-ZERO gradient at step 0. A control parameter must read exactly 0.0. The zero-grad gate runs after training.
* **Mutation:** delete the one `h = h + o_code(code)` line, and REACH must go RED.
* **G-DVB entry** for the new flag (a new lever without one is refused by the gate). Full launch gate, per-arm tokens. The R2 comparator compares STEP-0 terms only, on the same seed.
* **Clean judged tree:** the stray `eval_grid_windows.json` is removed from the pod tree, and G-SUITE re-runs on the clean tree.
