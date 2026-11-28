# Note N1 to card PREREG_V8F_PROOF_ADD70 (LONGROLL10K): the wrapper of record of the card and three repairs made before any run

Written by the coordinator fork after the first build of the wrapper was handed back and before any run of the card, before the smoke on the two-clip mini grid has returned (time: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). No number of the card exists. Arms, pairs, estimators, names, consequences and limits of the card are unchanged.

## 1. What the first build left open

The first build (`make_longroll10k.py`, which writes `v8f_t1_eval_reader_rule_rzgfhp.py`) was checked on stubs only. Three points would have made its rule arms or its strategic read wrong at P above 0:

1. The lead rule sets the car's own progress at the first tick of a window to 0 by definition (`distrule/v8f_t1_eval_rule_mix.py`, line 59; headway definition D4). After a pre-roll the car is not at the logged position at t0, so the first-tick rule would have used a gap that ignores the carried longitudinal offset. Later ticks and the second look project the car's position and were right.
2. The rule's arc tables are those of the 30-step logged path from the state's t0. A 60-step pre-roll in one piece would have run past them (clipped index).
3. Two arrays of the pinned tick recorder that the strategic instrument reads (`pose_w0`, `s_ckpt_w0`) would have stayed in the frame of the car's own pose at t0, while ground truth and labels are in the logged frame.

## 2. The wrapper of record of the card

`v8f_t1_eval_reader_rule_rzgfhq.py`, written by `make_longroll10k_b.py` from the first build by exact-once edits (the first build's files are kept unchanged). With P = 0 it behaves as the wrapper of card ADD67. Repairs:

* R1. When a pre-roll is active, for every re-based state at its first tick the wrapper computes a_m0 = the arc coordinate of the car's position at t0 projected on the logged path (the rule's own `progress`, own position in the logged frame at t0) and subtracts it from the gap g0 that the tick-level call of `distance_rule` receives; the second look and later ticks are untouched. A state at the logged position gets exactly 0. The rule-log columns `a_m` and `g0` of first-tick rows keep the definition values; the value used is g0 - a_m0, and a_m0 is saved per window (`preroll_offsets_<run>.npz`, key `a_m0_first_tick`). A reader that rebuilds g0 from the rule log at P above 0 has to subtract it.
* R2. P must be a multiple of 30. The pre-roll runs as P / 30 segments of 30 steps; after every segment the state is re-based as before the main phase (t0 advanced by 30, the logged frame pose of the new t0 set). R1 applies at the first tick of every later segment and of the main phase.
* R3. `pose_w0` of a pre-rolled state is the logged pose at t0, so `s_ckpt_w0` (computed from it by the pinned code) is in the logged frame. Every other recorded key is a world pose, a quantity in the frame at the tick, or free of a frame, and is left as the pinned recorder writes it.

The pinned tool and the files in `distrule/` are not edited. Local checks on stubs with planted wrong arms: `check_preroll_b.py`. The path extension of the rule's projection is 1,000 m on each side, so an offset behind the logged pose is carried with its sign.

## 3. Controls added to section 7 of the card

* l1 reads `check_preroll_b.py` (which reruns the items of `check_preroll.py` on the wrapper of record and adds the items of R1 to R3).
* l10 (first-tick offset term): in `R0LrCarry` (logged pre-roll) the largest absolute a_m0 is at or below the tolerance of l3; in every rule run at P above 0 an a_m0 exists for every window, and the rule-log rows of first ticks satisfy g0(log) = g_src (the definition), so the value used differs from the logged one by a_m0 exactly.
* l11 (segments): every P = 60 run reports two pre-roll segments per window and no clipped arc index.
* l12 (recorder frame): in `R0LrCarry` the tick-file arrays equal those of `R0LrFresh` within the tolerance of l3.

The smoke that fixes the tolerance of l3 runs with the wrapper of record (`pod_lr_smoke_b.sh`).

## 4. What stays as the card says

The headway shares are read from the dumped path of the window, which is in the logged frame at t0 and therefore carries the offset of the pre-roll in every step; nothing in the headway reader changes. A window that is already unsafe at its first step because of the pre-roll counts as unsafe: the model's own motion produced it.
