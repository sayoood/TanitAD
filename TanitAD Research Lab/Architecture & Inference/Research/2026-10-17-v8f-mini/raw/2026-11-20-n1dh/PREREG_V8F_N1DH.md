# PREREG_V8F_N1DH (FROZEN 2026-11-20, before any build, run or number): an auxiliary route-decision head on O's plan feature, whose posterior conditions the plan head

Status: FROZEN by the MM at the sha256 beside this file (`PREREG_V8F_N1DH.sha256`), before any code, arm or number of this card. The body is the design agent's draft unchanged; the FREEZE RECORD and the closing MM review bind where they are more specific.

## FREEZE RECORD (MM, 2026-11-20 20:02Z UTC)
* **PI go, first-hand:** "Yes, after lever C": two arms, `cond` at s0 and s1, about 1.5-2 LoRA pod-h (ESTIMATED upper; measured N1DC walls in the MM review). It runs on the LoRA pod only after the ROLLHIST chain ends, under `/root/gpu_queue.lock`. A projection above 2.4 pod-h (+20 %), or any third arm, goes back to the PI.
* **Basis:** the N1DC outcome of record is N1DC-COST (verdict d1dece12; under the per-cell wording it would be NO-GAIN). Under either reading N1DC is not adopted, and the decoder-side lever both cards name is this one.
* **Design, MM decisions** (section 7 items 2-5, as recommended): a detached posterior; the CE only on rows where O read T's own code; CE weight 0.5, untuned; no permuted-label arm.
* **Guard rule, stated per SEED** (unlike the N1DC card's wording): the guard holds at a seed iff no cell is a cost at that seed and there is no ROUTE-COST. Per-cell costs at both seeds are printed beside it.
* **Named guard risks:** the N1DC cost cells (s0 TAC-LON; s1 operative 0-2 s ADE, along-track, headway). The HYPOTHESIS from the N1DC cost attribution's post-hoc reading (INVALID as written, a320bf6d) is printed beside: an extra input trained into O's decoder can move operative longitudinal precision through the weights.
* **Admissibility additions** (MM review item 1): control K-LIM for the logged speed-bucket path, beside the own-code mask for the GT-code path.
* **Build:** any code fact the build finds that changes a lever, argv, cell, bar or outcome is an amendment written before any arm trains. A pure wiring fix is logged.

Draft of 2026-11-20, paper only (no build, pod, GPU or pytest). Template: `raw/2026-11-20-n1dc/PREREG_V8F_N1DC.md` (card sha256 2eae7964). Code read at `bcf3ba3f` (N1DC's tip) via the mirror; `file:line` below are at that commit. Raw root: `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/`.
Basis: the committed next lever of N1DC-NO-GAIN (PREREG_N1DC section 4) and of the T0T1 reading (`2026-11-19-n1dw/t0t1/MM_READING_N1D_T0T1.md`, "Next"). Evidence class of every number: MEASURED (ours, artifact named) unless marked.

## 0. Why this lever (MEASURED, tier T0 unless marked)
* D1-HO, 5,736 windows / 497 held-out episodes, 3-class kinematic TAC-LAT, majority share 0.338215. N1D decodes the roundabout decision at T0 at 0.6266 (s0) / 0.6349 (s1) (`t0t1/MM_READING_N1D_T0T1.md`); O's plan-head input carries 0.7028 / 0.6937 under the TREE probe (`probe/MM_READING_N1DWPROBE.md`, PROBE-N1D-FEAT).
* **Correction to the brief's wording:** the 0.70 is a tree probe. The LINEAR (logistic) probe of the same feature reads **0.6844 / 0.6850** (`2026-11-19-n1dw/probe/result/READ.txt` lines 11, 15). A `Linear(1024 -> 3)` head is bounded near the logistic figure, not 0.70. Honest aim: 0.685 to 0.73 (CEIL-AT-PLAN 0.7306 / 0.7287), against the decoded 0.63.
* N1DC (linear skip from T's code into O's plan head) gained nothing: +0.0045 / +0.0068 against bar 0.042364, Z-USE about 0 (`2026-11-20-n1dc/read/MM_READING_N1DC_VERDICT.md` section 2). Access to T's code was not the limit; the decision is lost when the plan head turns `h` into a 6 s track.
* The idea: make the decision an explicit supervised intermediate of O (a posterior over lane_keep / turn_left / turn_right, CE on the kinematic label) and let the track decoder read it.

## 1. The lever, from source
**O's plan feature.**
* `V8FMini.run_o`, `stack/tanitad/models/v8f_mini.py:1077`: `h = self.read_o(cond, sc, pad).flatten(1)`; `head_o` reads it at `:1081`.
* Shape `[B, n_q * d]` = 4 x 256 = **1024** (`n_q = 4` at `:214`, `d = 256` at `:207`; trainer defaults `--d 256`, `--n-q 4`, `v8f_mini_train.py:356-357`; `head_o = _mlp(fl, 2*d, 2*nb, zero_last=True)`, `:897,902`). It is exactly the tensor the PROBE-N1D-FEAT extractor hooked (`probe/code/n1dwprobe_extract.py:10,86`). `h` already contains T's code via the `e_code(code)` token (`:1071`).

**The label (K = 3).**
* From the GT 6 s track: `taniteval/taniteval/four_families.py:1124` `maneuver_kinematics(wp, dt)` gives `(dyaw, dv, v0, v1)`; `stack/tanitad/refs/refc_tactical.py:202` `factor_from_kinematics(...)` gives `(lat, lon)`. With `kappa=None` it is the v1 gate `|dyaw| > YAW_TURN_RAD = 0.15` (`refc_tactical.py:148`). Classes `LAT_LANE_KEEP, LAT_TURN_LEFT, LAT_TURN_RIGHT = range(3)` (`:137`). D1-HO's reader uses the same two calls verbatim (`2026-11-19-n1dw/read/code/n1dw_ho_read.py:76-81`, `CL` at `:42`), at `DT = 0.2`.
* This is NOT T's 5-way `lat_logits` (`N_LAT = 5`, `v8f_mini.py:44`; trained at `:1262` from the v9 `lab_t["lat_al"]`). The 3-way kinematic label is the deciding cell's own label.
* Training source: `batch["gt"]` (30 knots at 0.2 s) and `batch["gt_ok"]` (`:1272`). **The trainer has no kinematic-label code today** (grep `maneuver_kinematics|factor_from_kinematics` in `v8f_mini_train.py`, `v8f_mini.py`, `v8f_mini_data.py` at bcf3ba3f: 0 hits; control grep on `four_families.py`: hits). The build adds `route_stage(batch) -> (route_y [B] long, route_ok [B] bool)` beside `level_stage` (`v8f_mini_train.py:2817`), run BEFORE `ego.apply` / `dag.apply` as `level_stage` and `rb_mask` are. `taniteval` is already on the trainer's `sys.path` (`v8f_mini_train.py:41`). `route_ok = gt_ok.all(1)` (and `aux_ok` where present). Training-set class shares are a build record (ESTIMATED lane_keep-heavy; D1-HO is 1/3 each). CE is unweighted.

**Flags** (new, default OFF, each with a G-DVB entry as `--o-code-skip` had, `v8f_mini_train.py:133`):
* `--aux-route off|ce|cond`. `off` builds nothing = every earlier arm bit for bit (N1DC pattern: cfg field `v8f_mini.py:262`, validation `:301`, `built_levers` `:1437`, `declared_new_keys` `v8f_mini_train.py:819`).
  * `ce`: `route_head = nn.Linear(1024, 3)` trained by CE, conditions nothing.
  * `cond`: the same plus the conditioning below.
* `--w-aux-route W` (default 0.0; `ce`/`cond` with W = 0 refused as declared-and-inert, as `bad_o_code` does, `v8f_mini_train.py:1934`). Recommended **W = 0.5**, the coefficient T's own lat CE carries in the total (`v8f_mini.py:1346`, `0.5 * T["lat"]`) (ESTIMATED choice, untuned).
* Loss: `total = total + cfg.w_aux_route * T["route"] if cfg.aux_route != "off" else total` in the `:1346` block, one line (a mutation deletes it).

**Conditioning (`cond`).**
* `p = softmax(route_head(h).float(), -1)`, `[B, 3]`. `route_emb = nn.Linear(3, 1024, bias=False)`, **zero-initialised**, built LAST under a forked RNG after `o_code` (`:939-943` pattern).
* In `run_o`: `h_c = h + route_emb(p_in)`; `head_o(h_c)`. `route_head` reads `h` BEFORE the add, so there is no loop.
* Step 0: `route_emb = 0`, so `h_c = h` and the forward, every loss term but `T["route"]`, and every output are the parent's (Z0 / Z1). An additive term on `h` can be bit-identical; an extra token cannot (it changes `cond.mean(1)` and the softmax keys, the comment at `:1062-1064`).

**Detach: recommended `p_in = p.detach()`.**
* Attribution: without detach the decoder loss reshapes the posterior and `route_head` becomes an extra trainable layer inside `head_o`'s path, no longer "the route decision"; Z-USE is then not a clean ablation.
* The posterior stays a supervised decision, shaped by CE alone.
* The other option, no detach, is named and NOT recommended: it is a different lever (free capacity for the decoder).
* CE gradient into `h` is kept: it is the representation effect of the supervised head, which Z-USE separates from the conditioning effect.

## 2. One lever or two
* The lever is "decision supervised AND handed to the decoder". `ce` alone is a representation-only variant.
* **Recommend 2 arms: `cond` only, N1DH_s0 and N1DH_s1.** The `ce` attribution is available at zero training from Z-USE: real minus posterior-zeroed is the conditioning path's contribution, and the rest of N1DH - N1D is the representation effect (paired, per seed). One variable moves per arm.
* 4 arms (`ce` and `cond`, both seeds) cost about twice as much (section 6). Run `ce` only if Z-USE reads about 0 AND the gain is separated, under a PI go, with the zero-training read already in hand.

## 3. Admissibility (CLAUDE.md BINDING rules)
1. **Head inputs at inference contain nothing the label is derived from.**
   * The label is a function of the GT FUTURE track. `forward_infer` calls `_forward(g, 0.0, False)` (`v8f_mini.py:1169-1176`): `p_gt = 0`, no labels; `_KeyGuard` records keys read, which must lie in `INFERENCE_INPUT_KEYS` (`:198-199`, pinned by `assert_no_situation_classifier_input`, `:1532-1555`). `h` is a function of those keys only.
   * **Training-only exposure (a concern, not an inference violation).** `p_gt` anneals 1.0 to 0.2 (`v8f_mini_train.py:359-360`); `code` is replaced by the GT code with probability `p_gt` (`v8f_mini.py:1147-1151`), and the GT code (`_code_gt`, `:982`) carries label-derived `lat_al` / `lon_al`. So on teacher-forced rows `h` holds a function of what the CE predicts, and the head can learn "read the GT code", a shortcut absent at inference (there the code is T's own prediction). This is the nav-echo class, and an exposure bias `e_code` already carries in the parent.
   * Mitigation recommended: restrict `T["route"]` (and so the head's training signal) to rows where O read T's OWN code. It needs `_forward` to emit the substitution mask it already draws (`use_t`, `:1150`) as one new bool `out` key, no new RNG draw. A control reads the head's accuracy on a probe batch split by that mask; a large gap names the shortcut. PI question 3.
2. **The conditioning is the planner's own decision, not a goal input.** `p` is computed from `h`, hence from inference inputs only; no goal, route point or classifier signal is added (nav enters only via `_nav_tok`, as today, an input at training and inference as ruled). There is no situation classifier in V8FMini (`:20-21`, `:1532`).
3. **O's plan feature does not carry a situation-classifier output.** `h` is read from ego, nav, limit, ckpt, `e_code(code)`, age and scene tokens (`:1065-1077`). **A shared trunk does feed both:** `h` already carries T's tactical decision through `e_code`, so the posterior is partly T's own `lat_logits` re-read through O. T is a brain of this model, not the situation classifier, and its code is already O's input in the parent (N1D); the head turns an existing internal into a supervised readout. Not a back door; state it in the card.
4. **Vision-only at inference:** labels use the future (offline, allowed); the head's inputs at inference are the model's own inference inputs.

## 4. Read, bar, guard, estimator, outcomes
* **Executor of record: T1-kin/hb0.2** (`v8f_t1_eval_hb.py --kappa-max 0.2`, blob sha256 74fd8445, PI ruling D16-D); T1-kin/old beside. N1D comparators on D1-HO from `2026-11-19-n1dw/overlap_hb/`; on eval-141 from the N1DC re-dumps.
* **Deciding cell:** D1-HO kinematic TAC-LAT, **N1DH - N1D**, paired, per seed. **Bar:** max(0.0383, F_HO 0.0424, |N1D_s1 - N1D_s0|, |N1DH_s1 - N1DH_s0|), both seed draws under the executor of record (N1DC's was 0.042364).
* **Beside, no bar:** N1DH - N1R; N1DH - N1DC; LONGITUDINAL and LATERAL on D1-HO (distance-keeping NOT-COMPUTABLE, n = 0).
  * T0 decoded-plan TAC-LAT of N1DH against N1D's 0.6266 / 0.6349.
  * **Z-POST:** the route head's own argmax accuracy on D1-HO at T0, against the logistic 0.685 and CEIL 0.73 (did the CE extract the decision?).
  * **Z-USE: the posterior zeroed at inference** (`route_emb.weight` = 0, the exact "output x 0" as N1DC's `_zuse` copies), real minus zero, per seed, paired.
* **Estimator:** paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES" only; training variance is the two seeds; the executor is deterministic.
* **Guard:** four families on eval-141 (2,399 windows / 141 episodes), executor of record, N1DH - N1D, floor F2; STRATEGIC in full; S1-slot view in the chain; distance-keeping computed.
* **Guard cost rule, per SEED (the N1DC wording left it open):**
  * a cell is a COST at a seed iff separated beyond F2 in the worse direction at that seed;
  * the guard HOLDS at a seed iff none of the 24 cells is a cost at that seed;
  * "guard cost at both seeds" = each seed has at least one cost cell, the same cell or not (the N1DC label-of-record reading, `MM_READING_N1DC_VERDICT.md` section 1); "at one seed" = exactly one seed has any cost cell;
  * whether the cells coincide is reported beside with the cell names and never changes the label.
* **Named risks, reported first:** N1DC's cost cells, s0 TAC-LON (-0.0154, floor 0.0150) and s1 operative 0-2 s ADE (+0.0015), along-track (+0.0012), headway abs error (+0.0176) (floors 0.0005 / 0.0005 / 0.0039). No tolerance; same rule.

| outcome | condition | next |
|---|---|---|
| **N1DH-GAIN** | D1-HO gain above bar at BOTH seeds, guard holds at both | carried; s10/s11 (PI go) |
| **N1DH-ONE-SEED** | gain at one seed, guard holds at both | s10/s11 before any claim |
| **N1DH-GAIN-COST** | gain at one or both seeds, guard cost at one seed | reported; attribute on the cost cell (zero-twin method of N1DC costattr); back to the PI with the family named |
| **N1DH-COST** | guard cost at both seeds, or D1-HO a cost at both | not adopted; the cost family gets a card |
| **N1DH-POST-WEAK** | no gain AND route-head T0 accuracy below 0.685 at a seed | the CE did not extract the decision; next: a nonlinear head (`_mlp`) under its own card |
| **N1DH-NO-GAIN** | otherwise (head >= 0.685, decoder still about 0.63) | the loss is in the decoder's geometry, not in access to the decision. Next: a mode-routed plan head (K = 3 residual branches of `head_o`, selected by the posterior; architecture change, PI go) |

* A gain with Z-USE about 0 is a representation effect and is reported so; `ce` is then the clean follow-up.
* A missed bar FAILED and is reported so; the next arm is built (RULE ZERO). Bars, arms, executor and outcomes do not change after any number is seen.

## 5. Controls (before any launch)
* **Z0:** flag off: state dict and every loss term bit-identical to `bcf3ba3f`. **Z1:** `cond` at step 0: forward, every loss term but `T["route"]`, every output bit-identical to the parent. CPU, fp32, one thread.
* **REACH:** `route_head.weight` gets a NON-ZERO CE gradient at step 0; `route_emb.weight` gets a non-zero gradient with `head_o`'s last layer given a small seeded weight (zero at init makes every upstream gradient exactly 0, as the N1DC probe does, `v8f_mini_train.py:1966`); a control parameter reads exactly 0.0; with the detach, the decoder terms put EXACTLY 0.0 on `route_head.weight` (CE term removed in that probe).
* **Mutations that must go RED:** (a) delete the `h_c = h + route_emb(p_in)` line: REACH on `route_emb` RED; (b) delete the CE loss line: REACH on `route_head` RED; (c) deliberate regression: permute `route_y` within the batch in the stage: the head's accuracy on a held probe batch must fall to the class-share chance and K-LABEL must go RED.
* **K-LABEL:** `route_stage` equals D1-HO's `labels()` on the same GT tracks, bit for bit.
* **G-DVB entries** for both flags (the gate refuses a lever without one); a default-disabling flag is named in argv; full launch gate with per-arm tokens; `strat1_check.py` kind `H` as N1DC's kind `C` (`strat1_check.py:229-270,466-477,534-539`).
* **K-CLIP, clean judged tree, read controls** as N1DC: K1 self-pair exactly 0, K-CONST 0.338215, K-RE against banked N1D, Z-EVAL. Class record: training-set shares of `route_y` and the masked share, in `selfcheck.json`.

## 6. Cost (ESTIMATED)
* N1DC's per-arm GPU time is not in the records I may read; its card gave about 12.5 min per arm (ESTIMATED from the N1DW chain, `PREREG_V8F_N1DC.md` section 2), 3,000-step runs (`build/b2/n1dc_arm.sh`). The head adds a `Linear(1024, 3)` and a `Linear(3, 1024)`: negligible compute.
* **2 arms (recommended): about 1.5 to 2 LoRA pod-h** with gates, mint, post checks (N1DC's figure).
* **4 arms:** about 2.5 to 3.5 pod-h.
* Extra dev-box CPU work beyond N1DC: Z-POST and the own-code vs GT-code split (minutes). The build is bigger than N1DC's (a label stage, a head, a mask output).

## 7. Questions for the PI (MM's recommended answer)
1. Go for N1DH, 2 arms (`cond`, s0/s1, about 1.5 to 2 pod-h)? Yes; no `ce` arms now.
2. Detach the posterior into the conditioning? Yes (attribution; supervised meaning).
3. Restrict the CE to rows where O read T's own code (removes the teacher-forcing shortcut), or leave it unmasked and read the gap as a control? Restrict (one bool output, no new RNG draw).
4. CE weight 0.5 (T's own lat CE coefficient), untuned? Yes; no sweep before a first read.
5. An extra deliberate-regression ARM (permuted labels, s0, about 0.2 pod-h)? No: mutation (c) and K-LABEL carry it at zero pod time.

## MM review (2026-11-20 ~20:40Z; folded in at freeze)
1. **A second training-only path to the label, besides the GT code.** `h` also reads `e_lim(limit)` (`v8f_mini.py:1071` at bcf3ba3f).
   * In training, `limit` is the clip's logged speed bucket, built from the ego's realised speed over [t0 + 2 s, t0 + 6 s] (row `H-V8F-CAUSALBUCKET-1`). That is future information, and it correlates with a turn. At T1-kin inference the bucket is causal (`ROLLOUT["limit"] = "causal"`, `v8f_dagger.py:46` at 2bda9e1a).
   * The own-code mask does not remove this path. It is an exposure the parent N1D already carries, not one N1DH adds, so it does not block the lever.
   * The card names it in section 3. Control K-LIM: the head's accuracy on a probe batch with `limit` replaced by the causal bucket, against the logged one. A large gap names the shortcut.
2. **Cost, MEASURED from the pod** (`/root/strat1/n1dc/runs/N1DC_s{0,1}/train/summary.json`):
   * trainer time 633 s and 644 s for 3,000 steps;
   * mint to ARM_DONE 11 min 53 s and 12 min 6 s.
   * Two N1DH arms are about 0.4 pod-h of training. With G-SUITE and the D1-HO dumps, the draft's 1.5-2 pod-h is the upper bound.
3. **Decided by the MM, not the PI** (design choices, not spend), as the draft recommends:
   * 2 arms, `cond` only (Z-USE carries the conditioning-versus-representation split);
   * a detached posterior;
   * the CE restricted to own-code rows;
   * CE weight 0.5, untuned;
   * no permuted-label arm (mutation (c) and K-LABEL carry it).
   ⇒ The PI's question is only the go for about 1.5-2 LoRA pod-h, queued after ROLLHIST.
