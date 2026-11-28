# HANDOFF_RECLAB (build agent, no GPU, nothing staged/committed/pushed, no pod writes)

BASE: code of record **97fe0a56** (rollhist branch tip), NOT the arch-inf tip: the tip lacks `--dagger-bank-readout`/`--detach-upper` that RH's argv uses (FACTS 0). Tree: `F:/TanitAD-scratch/reclab/tree97/{stack,taniteval}`.

DONE
* FACTS_RECLAB.md: every label tensor (gt, s_h waypoint-derived -> re-derived; lab*, cons, lead_gap_h, limit unchanged), matcher cut order, labeller inputs, wall (0.28 s / 1,000 ok, 0.51 s / 1,000 off; ~163 s per 613k-record bank; no sidecar). STOP not triggered.
* Mechanism: `--dagger-label {human,recovery_v1}` (default human); `v8f_recovery.py` (V1 labeller verbatim, sha pinned, + `label_bank`, `arc_from_label`); `DaggerBank.attach_recovery`/`_index`/`gather`; `swap_rows` installs gt + s_h; FLAG_CHECKS `dagger_label`; bad_dagger clauses (non-default at p 0 refused); dagger_probe accepts the label (and checks it reached the batch). Diff vs originals: tmp/*.orig.
* Tests (logs: test_log_final.txt): gold vs 97fe0a56 blob (bit for bit), identity np.array_equal, reachability (>0 recovery, exactly 0 human; newly eligible >0 / 0), semantics table, G-REC gate, trainer wiring + loss reach, 21 ported labeller tests. Mutants all RED (MUTANTS_RESULT.json; M9 was an equivalent mutant, dropped).
* Pod chain staged in `pod/` (clone of the RHp3 chain; marker protocol unchanged): rl_argv.py (RH + exactly `--dagger-label recovery_v1` + `--out`), rl_arm.sh (strips the label for the stock pair check), chain, marker watch, launch, patch_rl.py (checker kind RL). DRY_MINT_PLAN.md.

NOT DONE / NOT RUN (honest)
1. Real-shard reachability: pod bank shards are 533 MB each (> 300 MB cap), so NOT pulled. The reachability proof is on the synthetic bank plus planted off-route records. Next: range-read the first ~3k rows of one shard member-by-member over ssh (npz is stored, uncompressed) and run `label_bank` with `logged_cache.npz` grids; or run it on the pod in a dry step.
2. Wall on the real bank with the real clip store (labeller wall measured on bank-table records only).
3. `patch_rl.py` is written but NOT applied or tested; `rl_argv.py`/`rl_arm.sh` bash -n / ast checked only; the checker mutants (second flag, other bank, human under RL, missing label, dose 0.3) not run.
4. Read extension (`RH_AKIND=RL`, `dry_check.py`): NOT done. Pattern: how `RHK` is wired in `rh_common.py` (:36, :121), `dry_check.py` (:18, :147), `rh_read.py` (:304, NEXT rows).
5. Pod commit: RL needs a NEW commit (pins, blob list, G-SUITE re-run, launch-gate pins). DRY_MINT_PLAN section 0.

DELETIONS (F:/TanitAD-scratch/reclab/DELETIONS.log): arch-inf export, taniteval/results. Still to delete at the very end: tmp/pt*, mutants/tree (re-creatable), research/ copy.

TEST RESULT (test_log_final.txt, whole files, 7 files): 276 passed, 2 skipped, 3 FAILED:
* `test_v8f_dagger_train.py::test_the_single_applying_lines_exist_exactly_once`: was my doing (it pins the literal call `make_dagger(a, J, wins, clips, info) `, now `..., info, store)`). I updated that one anchor line in the test (stack/tests/test_v8f_dagger_train.py:1012); NOT re-run after the edit.
* `test_v8f_dagger.py::test_kb5_the_oracles_are_collected_before_every_bank_fixture...`: an ORDER artifact (my file imports the dagger fixtures and ran first in that invocation); the same file alone passed 57/57 after my source edits. NOT re-run in dagger-first order.
* `test_v8f_ego_dagger_compose.py::test_flag_off_reproduces_the_unmodified_tip_bit_for_bit`: assertion is about `t2_bank_*` summary keys, which my change does not touch; probably pre-existing on 97fe0a56. UNVERIFIED (the control run on the original files timed out at 200 s).
Also deleted at the end: tmp/pt*, mutants/tree. CHANGED_FILES.txt now also lists test_v8f_dagger_train.py (edited, sha not recomputed).
