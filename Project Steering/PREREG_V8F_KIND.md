# PREREG_V8F_KIND: which KIND of T's constraint cells is worth ADE6 to O (written 2026-10-25, before any instrument or data)

**Evidence class of the motivation:** MEASURED (`TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-t-attrib/T_ATTRIB.md`, register `D-V8F-T-ATTRIB-1`). **Tier of every read below: T0** (GT-conditioned): a diagnostic of how O consumes T's code, never a driving-performance claim. **Estimator:** paired episode-cluster bootstrap, B = 2,000, over the 141 eval episodes (2,399 windows). It answers another draw of EPISODES, not another training run (`H-ESTIM-SEED-1`). This instrument has no training; its seed replicate is the second banked checkpoint.

## Why this experiment, and why now

* Feeding O the GT constraint code closes the T-to-GT ADE6 gap: 1.2045 m at M2b_s0, 1.0508 m at M2b_s1 (`s9_lever_table.out`).
* T's error is fit-limited and shared by every trained arm (R70). Every measured lever moved it by 0–6 %, and the gap is CONVEX in T's accuracy.
* Two estimators of which cells O reads disagree by about 10× on the SPEED cells: a size-agnostic dose model (0.033 m) and a cross-arm association (0.78–0.90 m per SD²).
* Both are ESTIMATED. A per-kind oracle MEASURES it. That decides whether any T-side lever aimed at one kind of cell can clear 2F, or whether the next lever must be on O.
* Cost: eval only, on the two banked checkpoints, no training. ESTIMATED under 1 h of box 2. It needs the PI's box-2 go.

## The partition (six KINDS of the 26 constraint cells; each cell in exactly one kind)

| kind | cells (`CONS_NAMES`, `stack/tanitad/models/v8f_mini.py`) |
|---|---|
| `speed` | `lon_v_target_ms`, `v_a_ms`, `v_end_ms`, `v_lo_ms`, `v_hi_ms` |
| `angle` | `lat_theta_deg`, `turn_dyaw_deg`, `turn_exit_psi_deg` |
| `tte` (time to event) | `turn_t_start_s`, `turn_t_end_s`, `lc_t_cross_s`, `lon_t_reach_s`, `lon_t_start_s` |
| `timegap` | `lead_tg_s`, `lead_tg_min_s` |
| `geom` | `turn_d_start_m`, `turn_len_m`, `turn_r_arc_m`, `turn_exit_x_m`, `turn_exit_y_m`, `lc_d_cross_m`, `lon_d_reach_m`, `stop_x_m`, `stop_y_m` |
| `leadgap` | `lead_gap_m`, `lead_gap_min_m` |

The union of the six kinds must equal the existing `constraints` composite index for index (5 + 3 + 5 + 2 + 9 + 2 = 26). A test pins it. The existing groups and composites stay byte-identical.

## The evals (per seed, M2b_s0 and M2b_s1, banked checkpoints, sha256-checked)

1. **Identity controls:**
   * T's own code (`--tac` off) must reproduce the banked eval.
   * The full oracle (`all`) must reproduce the banked `-tacorc` eval.
   * Both must match per window, with ADE6 Δ exactly +0 [0, 0].
2. **Single kind** `k` (GT in kind k's cells only, T's own code everywhere else): six evals.
3. **Leave one kind out** `all-k` (GT everywhere except kind k): six evals.
4. **The union** `constraints` (GT in all 26 cells, T's decisions): one eval.

**Reads per kind (paired, same windows):**
* gain_k = ADE6(own) − ADE6(k);
* necessity_k = ADE6(all−k) − ADE6(all);
* additivity A = Σ_k gain_k / gain_union.

## The committed reading

| outcome | condition | next |
|---|---|---|
| **ONE KIND CARRIES** | some kind k has gain_k ≥ 2F = 0.2415 m, separated, at BOTH seeds | a T-side arm aimed at kind k, under its own PREREG. SPEED points to T's speed variance (two-seed code average / EMA); ANGLE, GEOM or TTE to the encoder (raises LoRA's prior); TIMEGAP or LEADGAP to lead perception |
| **DIFFUSE** | no kind clears 2F at both seeds, and A ≥ 0.8 at both | no single T-side lever; next is O-side: train O on CELL-WISE mixed codes, under its own PREREG |
| **INTERACTIVE** | no kind clears 2F at both seeds, and A < 0.8 at both | O needs most cells right at once; same O-side next, stronger prior; T-side single-kind arms deprioritised |
| **MIXED** | the seeds disagree on the row above | no branch from this PREREG; each seed reported; the SPEED read below still stands |

**Named in advance, decides nothing alone:** gain_speed ≥ 0.20 m at both seeds ⇒ the association was right and the size-agnostic dose model wrong for speed. ≤ 0.05 m at both ⇒ the reverse.

⚠️ **The confound, stated before the data.** O was trained on WHOLE codes: GT with probability p_gt (1.0 → 0.2), else T's own, one switch per sample over the whole code (`stack/tanitad/models/v8f_mini.py:620-622` at tip `a61d3170`). A single-kind oracle hands O a MIXED code it never trained on, so gain_k is a LOWER bound on kind k's worth.
* ONE KIND CARRIES is robust to that: a lower bound that clears 2F clears it.
* DIFFUSE and INTERACTIVE cannot tell "the kinds are individually small" from "O cannot use a mixed code". Both point to the same O-side arm, which removes the mismatch, and this instrument re-reads its result.

**Beside every oracle arm, deciding nothing:** the four families (longitudinal incl. distance keeping K = 10, lateral, tactical, strategic: NOT PAUSED for v8F-mini) and Λ.

## Refusals (rc 2)

* An identity control misses exactly +0.
* The kind partition is not a partition of the 26 cells, or its union is not `constraints`.
* An eval's window set or episode list differs from the banked one, or a checkpoint's sha256 differs from the banked run.
* An eval or a family is missing. A missing family is refused, never dropped.

## Amendment 1 (2026-10-25, written before any KIND data, from `raw/2026-10-25-navturn/NAVTURN_AUDIT.md`)

* **Beside, deciding nothing:** every single-kind gain_k and the union gain, split by WINDOW STRATUM on eval-141:
  * **JUNCTION:** a dominant turn segment in the label band with `turn_is_turn` = 1;
  * **CURVE:** `curve` = 1, a ≥ 20° segment that the nav's speed gate never announces (`stack/scripts/build_v9_labels.py:828-831`);
  * **NO-TURN:** every other window.
* The oracle replaces T's code per window, so this split IS the curve-only / junction-only oracle the audit proposes, at no extra eval.
* Each stratum carries its n windows and n episodes and the paired episode-cluster bootstrap within the stratum. A stratum with fewer than 10 episodes is reported with its n and no interval.
* **Why:** 74 % of the 210 "route-silent" turn windows are fast bends (R71).
  * If the angle / tte / geom gains sit in CURVE windows, the lever is perception of bends (the encoder: it raises LoRA's prior), not a route input.
  * If they sit in JUNCTION windows, a predicted-goal input stays on the list.
* The committed outcome table above is unchanged.
