# v8F-mini (track M, stream M1-build) -- the code, tested, ready to train. NOTHING WAS TRAINED.

Date 2026-10-17. Spec: `Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` §13 and `PREREG_V8F_MINI.md` (read in full; neither changed).
Evidence classes: MEASURED = ours, artifact named. ESTIMATED / UNVERIFIED are marked. No GPU was used (CPU only, as instructed).

## 0. Headlines (integration items first)

1. **The residual base is NOT the banked P10.** Per the orchestrator correction, every path integration goes through `taniteval/floors_kin.py` with `action_units="steer"`. The refit P10 (same recipe, 600 train clips, stride 8, correct unit) reads **ADE6 2.5744 m [2.327, 2.829]** (n 2,399 windows, 141 episodes, episode-cluster bootstrap B 2,000; `raw/p10/p10_fit.json`) against the banked 2.9257 m. **Pipeline control (MEASURED):** the same script in the LEGACY unit reproduces the banked recipe at 2.9281 m (banked 2.9257, delta 0.0024), so the refit is the banked recipe with one thing changed. Correct-unit kd_x is 2.9396 m (legacy 3.5584). **Quote 2.5744 as "P10 (steer unit)"; do not quote 2.9257 as the bar.** The PREREG text cites the banked numbers (2 s 0.2229 / 6 s 2.9257) -- it should be amended by the Master Mind (H-MINI-K-1 is defined as arm minus P10; the arm is compared with the refit P10, `raw/p10/p10_eval_pred6.npy`).
2. **`stack/tanitad/data/clip_clock.py` is missing from this branch's tree** (it exists in commit `82c2331a`; `git ls-files` shows no copy). `build_v9_labels.py build` and `refc_v3_train.py` import it, so v9 labels cannot be (re)built on this checkout. v8F-mini does not import it (it reads finished releases). Needs a landing decision.
3. **Use the 4,572-clip v9 train release (md5 `83e30bd3…`), not `f63ece41…`.** The latter has 4,369 clips; against the 4,572-clip cache it leaves 203 clips (13,783 of 310,642 windows, 4.4 %) unlabelled. The newer release is a strict superset (4,369 clips bitwise identical, `qland/hold_v9train4572/RESULT.md`) and gives 100 % window coverage (310,642 windows). The code accepts both md5s; the default path/md5 follow the task (`f63ece41…`); HANDOFF.md uses `83e30bd3…`. Decision for the Master Mind.
4. **The frozen SPEED rule is not satisfiable by the human on 6.3 % of windows** (GT-plan control, below): the v8 max-speed sidecar is ONE limit per CLIP, not per window. This is an instrument property, not a bug; it does not change the H-MINI-CONS-1 bar (arm vs P10), but the training penalty asks the model to be more conservative than the human there. `M1-nocons` isolates the effect.
5. **Files to land** (all new, none staged, per the session amendment): see §9. Nothing existing was modified.

## 1. Design against the PI spec, line by line (PLAN §13.2)

| PI spec | What is built | Where |
|---|---|---|
| Inputs: scene tokens, current + short history | DINOv3-L pooled 4x10 (or 8x20) tokens, `hist` 4 frames (0.6 s, = refav1's window); `hmax` 10 frames cached per window so that older tick frames exist | `V8FMiniWindows`, `V8FMini._scene` |
| Inputs: t0 ego dynamics = the 8 P10 features | `p10` rows of the cache = `kin_features((v0, a0, steer0))`, standardised with P10's own (xm, xs) | `V8FMini._ego`, `P10Prior.standardise` |
| Inputs: max target speed one-hot {30,50,100,120} | v8 sidecar `refcv6_speed_max_v8_*.jsonl` -> 4-vector + ok flag (per clip) | `load_speed_sidecar`, `e_lim` |
| Inputs: nav command + distance to manoeuvre | v9 per-frame `nav_token` (3-way one-hot) + `nav_d_next_m/100` (clipped 0..3) + args-valid flag. No side/dyaw/lookahead args (not "distance"), no nav-args dropout | `build_label_join`, `e_nav` |
| S -> checkpoints (x, y, heading) at 30/50/80 m of arc, ego frame | `head_s` -> 3x3, on a straight-ahead nominal (30,0,0)...; trained by L1 on position+heading against v9 RC-A (sigma 8 m), valid-masked | `run_s`, loss `ckpt` |
| T reads the checkpoints -> v9 goal/action logits + constraint regressions | `run_t`: checkpoint token + inputs + scene -> 22 goal logits (masked BCE, w=0 IGNORE), 5 lateral + 7 longitudinal logits (partial-label CE: -log sum_allowed p), 26 constraint cells (Huber on standardised targets, NaN masked) | `run_t`, `loss` |
| O reads T + checkpoints -> 6 s trajectory (60 x 0.1 s) as a residual on P10 | `run_o`: P10 (30 knots @0.2 s) upsampled to 60 points (odd indices ARE the knots) + a ramped cosine-basis residual, **last layer zero-init** | `run_o`, `upsample_prior` |
| P10 from a frozen fitted MLP passed in; zero-init residual reproduces P10 exactly | `P10Prior` (frozen, `requires_grad` False). **Test: `traj30 == prior(...)` bit-exact; a non-zero head moves it** | `test_zero_init_residual_reproduces_p10_exactly` |
| Sample-and-hold: O uses T's last 0.5 s tick, T uses S's last 1 s tick | each layer is evaluated at its OWN tick frame (NOW - age); lower layers read the held output; held checkpoints are re-expressed in the reading layer's frame with measured odometry (`pose_*`) | `_forward`, `transform_ckpt` |
| Scheduled sampling p_GT 1.0 -> 0.2 | per-sample Bernoulli(p_gt): GT RC-A (S->T, valid-masked) and GT tactical code (T->O) vs predicted; linear schedule over the run; never at inference | `p_gt_schedule`, `_forward` |
| `navonly` flag removes the scene input from S and T | S and T readers are BUILT without a scene path (`uses_scene=False`); O keeps the scene. Test: S/T outputs are invariant to the features iff navonly | `test_navonly_builds_no_scene_path_in_S_and_T` |
| `assert_no_situation_classifier_input()` | `forward_infer(batch)` reads exactly `INFERENCE_INPUT_KEYS` (recorded by a key guard); an AST scan of the module's code finds no situation-classifier identifier; a model that peeks at `lab` is caught | `assert_no_situation_classifier_input`, `test_no_situation_classifier_input` |
| Losses: L1 ckpt; CE + masked regression; ADE/FDE; consistency penalties SPEED/STOP/TURN/LEAD, differentiable | all present. Penalties are hinges in the rule's own units; **hinge > 0 <=> the numpy frozen rule says "violated"** on 400 random boundary-heavy cases (all four mutation-checked) | `consistency_penalties`, `test_torch_penalties_agree_with_the_numpy_rules` |
| Hard R1 speed-cap projection at inference (+ Kamm mu 0.7) | `project_speed(traj, cap_ms, mu)`: caps each interval's speed at min(limit-from-INPUT, sqrt(mu g / kappa)), re-samples the same geometry; known-value tests (cap -> exactly the cap, cap above -> identity, R 10 m @15 m/s -> 8.3 m/s, still on the circle) | `project_speed`; `v8f_mini_eval.py run --project` |
| Each layer runs at its own cycle time | S 1 Hz / T 2 Hz / O 10 Hz as ages (below) | -- |
| 4th brain (WM): H-JEPA z1/z2/z3 on O/T/S cycles | `WMAux` wraps S3's `HJEPAStack` (3 levels, strides (2,2)); a = gamma 0, b = beta 0.08 / gamma 100; aux weight 0.1 | `WMAux`, arms M2a/M2b |

Parameters (MEASURED, `bench`): **M1 5.21 M trainable** (adapter 0.28, cond 0.03, S 1.32, T 1.46, O 2.12), **M2 7.21 M** (+2.0 WM). The plan's "40-120 M heads" were not needed on cached 4x10 features; the capacity is a lever if M1 misses K-MINI (width `--d`, depth, 8x20 grid).

## 2. Interpretations (each frozen before any data; confirm or overrule)

Consistency rules (docstring of `taniteval/v8f_consistency.py`, I1-I7): SPEED/STOP/LEAD are evaluated on the plan's overlap with the label band (tau >= 2.0 s); STOP active iff the STOP_POINT goal is set, `lon_d_reach_m` finite and the stop is inside the 6 s plan; TURN active iff a TURN goal is set and the segment overlaps the plan by >= 1.0 s, heading change read from path tangents with a 1 deg dead band (a plan that does not turn violates it); LEAD uses the logged lead position and the builder's own time-gap definition with the label `lead_tg_min_s`; CONSISTENCY(a) = fraction of ALL windows violating ANY active rule (per-rule rates over all and over active windows are both reported); self-consistency (b) = 1 - the same against the model's OWN decoded goals (TURN/STOP at p > 0.5, LEAD iff argmax lon class is FOLLOW); nav-only baseline turn = a 90 deg R = 12 m arc then straight (MEASURED: the empirical |nav dyaw| of announced turns is median 78 deg, p10 32, p90 109 -- 90 is a fair nominal).

Model/training interpretations:
* **J1 Ages are quantised to the 0.2 s cache grid**: T ticked 0..2 steps ago (< 0.5 s), S ticked 0..4 steps before T (< 1.0 s), so S's output can be up to 1.2 s old at O (nested causality). The true tick phases (0.5 s) cannot be represented on a 0.2 s grid.
* J2 Eval ages: default `--ages cycle` (deterministic, covers the cycle); `fresh` (all 0) is the optimistic variant.
* J3 The strategic layer ALSO sees the scene (PLAN's stated interpretation); `M1-navonly` is the literal nav+distance version.
* J4 H-MINI-TAC-1 "goal macro-F1" is ambiguous: reported as lateral (5-way) and longitudinal (7-way) macro-F1 against the majority class and `M1-navonly` (paired), plus the 22-token goal macro-F1 (point estimate; the all-negative majority is 0 by construction, so that bar is trivial). **Open question for the Master Mind: which is the bar?**
* J5 The training SPEED/STOP/TURN/LEAD penalties use exactly the metric rules against the GT constraints at NOW; a second penalty set (weight 0.25) is against T's own detached predictions to train (b).
* J6 The WM levels sit at 0.2 / 0.4 / 0.8 s (strides 2, 2), the grid's nearest multiples to 0.1 / 0.5 / 1.0 s. z1 is the mean-pooled adapter token per frame. WM collapse is not guarded here (see `2026-10-17-hjepa-collapse-diag`).
* J7 `--amp` (bf16) is implemented with the geometry/prior kept in fp32, but **UNVERIFIED on a GPU**.

## 3. What the floors say (MEASURED, `raw/eval/floors_report.json`, 2,399 windows, 141 episodes, B 2,000)

| floor | ADE6 | ADE2 | CONSISTENCY(a) any [SPEED / STOP / TURN / LEAD] |
|---|---|---|---|
| P10 refit (steer unit) | **2.5744** [2.327, 2.829] | 0.2187 (first 10 knots of the 6 s fit) | **0.3235 [0.276, 0.369]** [0.243 / 0.034 / 0.036 / 0.139] |
| kd_x (steer unit) | 2.9396 | 0.2981 | 0.4010 |
| damp50 (steer unit) | 2.9336 | 0.3655 | 0.3130 |
| constant velocity | 4.1280 | 0.5650 | 0.2960 |
| **GT plan (control)** | 0 | 0 | **0.0792** [0.063 / 0.003 / 0.000 / 0.013] |

* The GT control must be ~0 if the rules match the labels. It is not: SPEED 6.3 % (the clip-level limit, median excess 1.7 m/s; 86 of 152 are in the 30 km/h bin) and LEAD 1.3 % (0.2 s tolerance vs chord-vs-log speed at long time gaps, 32 of 1,027 active). STOP 0.3 %, TURN 0. **This is the instrument's floor and is reported with every arm.**
* P10 ignores a TURN goal in 55 % of the 158 windows where one is active (no nav input) -- the room M1 should take. Event strata (P10 ADE6): turn 4.21 m (n 158), stop 3.86 m (n 81), lead braking 3.05 m (n 319).
* **Nav-only kinematic checkpoint baseline** (vs v9 RC-A, valid windows 2,322 / 2,319 / 2,294): position error **2.01 m @30, 4.82 m @50, 11.15 m @80**; heading 7.9 / 11.0 / 17.0 deg.
* The four binding families run through `taniteval.four_families` (`--families`): longitudinal and lateral computed; the lead-track distance-keeping half of LONGITUDINAL is reported by the CONSISTENCY LEAD rule instead (the family's own lead-track input is not built -- a work item); tactical-from-trajectory computed; STRATEGIC is un-paused and is the H-MINI-STRAT-1 checkpoint error.

## 4. Tests (MEASURED, CPU; 70 passed in 47 s together with the 18 existing `floors_kin` tests)

* `taniteval/tests/test_v8f_consistency.py` **25 passed** (each rule fires / holds / both sides of every boundary; nav baseline known values; macro-F1). Mutation checks: relaxing the SPEED / STOP / TURN / LEAD tolerance each turns the suite red.
* `stack/tests/test_v8f_mini.py` **27 passed** (incl. 5 parametrised live-checks and 1 test that skips if the local eval release is absent -- it ran). Mutation checks (each turned a named test red): swapped odd/even upsample, join `k = 2j+1`, flipped scheduled-sampling comparison, scene path forced on in navonly, swapped tick ages, non-zero-init head, and the four torch tolerances (SPEED 0.5, STOP 2 m, TURN 1 deg, LEAD 0.2 s).
* End to end (synthetic world, in the S4 and v9 formats): train 8 steps -> **crash at step 6 -> resume from the step-4 checkpoint -> final weights equal the uninterrupted run (1e-5)** -> `summary.json {"done": true}` only at the end -> eval (`run`, `--project`) -> `compare`. A checkpoint written by a different argv is refused.
* Launch gate (`v8f_mini_train.py`): every argparse flag has a `FLAG_CHECKS` entry (test); declared-vs-built refuses a built/declared mismatch for navonly, consistency, wm, a data fraction that was not applied, and **"a lever declared that does nothing" (penalties on with weight 0; WM on with weight 0)**; `live_check` (3 optimiser steps on a real batch): every declared term finite, every trainable parameter non-zero gradient except the DECLARED-dead set (M2a: the IDM heads), the frozen prior's gradient exactly 0.0, and the zero-init head's gradient non-zero at step 0 -- red on a deliberately broken model (O head unreachable). The 18 existing `floors_kin` tests were re-run and pass (this stream modified no existing file).

## 5. Join facts (full report: `LABEL_JOIN_REPORT.md`)

Key: cache step j = v2ep frame k = 2j = v9 row k (clock exact; `v_now_ms` vs pose speed over 2,365 windows: mean 0.005, p99 0.058, max 0.32 m/s). 5 hand-checked windows: release RC-A is **bit-identical** to `build_v9_labels.rc_fields` re-run from the ego log, and within 0.03-2.3 m of the unsmoothed human path (pinned in the test). Eval: **139 of 141 grid episodes have v9 labels (2,365 of 2,399 windows)**; the 2 episodes are `38fe80c1-caf1-45ea-810b-2681ea488c6d`, `204b15b4-bd0a-4155-b01e-572c55197d0e` (34 windows). Their strategic/tactical/consistency cells are masked (trajectory metrics use all 2,399).

## 6. Open questions / not done

1. PREREG amendment: cite the refit P10 (see 0.1).
2. Which v9 train release (0.3); which "goal macro-F1" (J4).
3. The SPEED rule's per-clip limit (0.4) -- window-local limits would need a data-engineering change; not invented here.
4. **Not built / not verified:** no feature cache existed (S4 had produced none), so no real-feature forward pass has run; step time and GPU memory are ESTIMATED in HANDOFF.md; `--amp` and 8x20 are UNVERIFIED; the 2 unlabelled eval episodes were NOT built (the builder needs `clip_clock.py` or two sidecar rows, and the 3-D agent join for lead fields; egomotion parquets, VLM records and v2ep poses for both are local); D3/LoRA are out of scope.
5. Seeds: the code is seeded and resumable; the seed floor F needs two seeds per arm (HANDOFF commands).

## 7. Reachability audit (the "built, tested, unreachable from its caller" class)

Traced argparse -> config -> built attribute -> loss term for every lever: navonly (`FLAG` -> `make_config` -> `Reader.uses_scene` -> `built_levers`), consistency (weights read back from the model), wm a/b (gamma read back from the built `HJEPAStack`), p_gt (schedule in the loop, used in `_forward`), data fraction (clips read back), steps-mult (S computed from the 100 % window count). Defaults are `consistency` ON, `wm` off; **a flag whose default disables a mechanism is named in HANDOFF.md argv** (`--wm a|b`, `--navonly`, `--no-consistency` are the only ways to leave M1's defaults).

## 8. Artifact classes

MEASURED: all numbers above (paths in the manifest). ESTIMATED: GPU memory/time (HANDOFF). INHERITED: the banked P10 numbers (used only as the legacy-unit control). Registry rows for the arms are NOT written here -- there are no trained results.

## 9. Deliverable manifest

See the end of the hand-back message and `MANIFEST.txt`.
