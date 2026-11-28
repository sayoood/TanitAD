# GAPRULE_DEFS -- kit of card PREREG_V8F_PROOF_ADD94 (GAPRULE10K)

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), UE_s0 and UE_s1. The card file is not edited.

## Where scoring reads the lead (evidence for item 1)
The wrapper j (`v8f_t1_eval_reader_rule_rzgfhqetj.py`) changes ONE object: `X.STATE["label"]`, set at L1567 (`build_rule_label(a, mine.rule_join, T.D.load_join)`; parent rzgfhqet L1546).
* Readers of that object: the rule planner `ReaderPlanner.__init__` L938 (`self.label = X.STATE["label"]`) and the mid-look rule `L822`; both belong to the lead RULE.
* `W.install(T, "mix", a, None, ...)` L1570 gets J = None, so `install` builds no second LabelSource (`distrule/v8f_t1_eval_rule.py` L226: `label = LabelSource(J) if MODE_CFG[mode][0] == "label"`, and mode "mix" is not "label").
* `T.cmd_run(a)` L1572 gets the namespace `a` with `a.eval_join` unchanged (no assignment to it anywhere; check_wrapper_j static fact A3). The pinned tool (sha256 61e140ed..., on the pod only, not on the dev box) reads
  `a.eval_join` itself; the rule's lead speed and acceleration come from `attach_logged(out, store, J_)` with the pinned tool's join J_, not from the label object.
* The measures C, U, F are computed offline by the reader from the dumps and the evaluation join (`smoothreq/read_smoothreq10k.py` L548 `J = H.load_join()`, L598 `H.label_windows(J, ...)`), never from a rule join.
Result: scoring does not share the redirected object. The pinned tool's source is not readable here; that part of the statement rests on the call path above and is checked on the pod by control j0
(G0 through the new argument equals banked RLc, largest difference 0) and j2.
A rule join must equal the evaluation join in `clip`, `off`, `n`, `lead_valid` or the wrapper refuses.

## Files (all in gaprule/)
| file | role |
|---|---|
| make_wrapper_j.py -> v8f_t1_eval_reader_rule_rzgfhqetj.py | 4 asserted edits (header, flag `--rule-join`, label source, run record fields `rule_join`, `rule_join_sha256`) plus one helper `build_rule_label`; diff vs rzgfhqet: 25 lines added, 3 removed (1585 -> 1607 lines) |
| make_planted_joins.py | `--eval-join NPZ --out-dir NEWDIR`; `--selftest`; writes G0, GR1, GR2, GW05, GC10, GC20 and PLANTED_JOINS_RECORD.json (numpy only) |
| make_gaprule_queue.py -> pod_gaprule10k_queue.sh | 12 rollouts, mirrors the queue of card ADD85 (rollout function, slot rule, markers, DRY=1, timeout) |
| check_wrapper_j.py | control j2 on the pod: j generation without `--rule-join` equals rzgfhqet; with G0 equals it; missing join refused |
| make_read_gaprule.py + read_gaprule_frag.txt -> read_gaprule10k.py | reader, stages `controls` and `names` |
| test_gaprule.py | control j4 tests (20) |

## Choices (the card is silent)
* Clips are seeded by clip NAME (array `clip` of the join, `load_join` in stack/tanitad/data/v8f_mini_data.py L231 returns it); without it by clip index. Seed text `<clip>|<W|R|C>|94`, first 8 hex digits of sha256.
* R is generated over ALL frames of a clip and kept where valid; lag-1 is measured on consecutive valid frames of one clip. Normalisation (mean 0, sd 1) is over all valid frames of the join, population sd.
* G0 has no limits (it must equal the label gap); the limits (0.5 m, 60 m) apply to the five planted arms on valid frames only. For GC arms the realised error is the ratio planted/label - 1; where the label gap is exactly 0 the ratio is undefined and the noise term is used (counted by nothing: expected zero such frames).
* lead_gap keeps its stored dtype; the error statistics are taken in float64 before the cast.
* Pod root: planted joins, record, markers, logs and the small folder under `/workspace/v8proof/gaprule10k`; the rollout folders under `$W/t1` of the model queue (`/workspace/v8proof/integ10k`) so the pull of the reader is unchanged. Step markers `ro_<arm>` are in `$GP/markers`.
* The queue drops the two start checks for the reader checkpoint and the q50 map (form RLc uses neither). Free memory bar: 28 GB (`I10_MIN_FREE_GB`), as the model queue.
* Reader: the names stage works from the `payload` of an all-pass controls file and binds it by sha256; any control failing in the file, even with an edited `controls_all_pass`, refuses. Share of the contact gap: interval is that of C(X)-C(RR) over the POINT gap (the denominator is not resampled).
  Families and the strategic family come from the engine pair read as in the model reader; a family the rule does not move carries the reason text (`FAM_REASON`). The strategic section is printed only as the engine gives it.
* `--wrapper-check` takes the CHECK_WRAPPER_J_DONE.txt of check_wrapper_j.py (last line `ZZ CHECK-WRAPPER-J PASS`; its `.hashes` file beside it has the wrapper hash), `--tests-out` the pytest output (`N passed`, no failed).

## Order of use
1. dev box: `python make_wrapper_j.py`; `python make_planted_joins.py --selftest`; pytest (command in the head of test_gaprule.py); `python make_gaprule_queue.py`; `python make_read_gaprule.py`.
2. pod: `make_planted_joins.py --eval-join <join_eval141.npz> --out-dir $GP/joins` (one run; refuses an existing dir); `check_wrapper_j.py ... --workdir $GP/check_j --g0-join $GP/joins/G0.npz` (give `--old` the rzgfhqet file).
3. pod: `GR_SEED=0 bash pod_gaprule10k_queue.sh` and `GR_SEED=1 ...` (or ALL); `DRY=1` first.
4. read: `read_gaprule10k.py --stage controls ...` then `--stage names --controls <file> --out <new file>`.

## Arms
| arm | planted join | run name |
|---|---|---|
| G0 | label gap | I10RLcKF5P30Hb20jG0_s<k> |
| GR1 / GR2 | gap + 1 / 2 m of R | ...jGR1, ...jGR2 |
| GW05 | gap + 0.5 m of W | ...jGW05 |
| GC10 / GC20 | gap * (1 + 0.10 / 0.20 C) | ...jGC10, ...jGC20 |
