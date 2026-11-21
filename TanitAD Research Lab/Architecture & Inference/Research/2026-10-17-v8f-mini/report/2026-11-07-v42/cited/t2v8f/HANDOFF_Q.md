# HANDOFF Q (dominance read re-pinned with --arm) -- 2026-11-07. DONE: items 1 to 5. IN FLIGHT: `audit/finish_q.sh` (mutation run on the pinned file, full suite, PINS_Q).

MEASURED = ours, CPU. Nothing committed, staged or deleted (own scratch only); no box 2, no pod, no download; `motion_c/t2_prereg/` and `motion_c/theory/` untouched; `stage4/score/` (the read of record) never written.
Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, `CUDA_VISIBLE_DEVICES=-1`, LF endings.

## FOURTH PIN: `code/read_t2_dominance.py` sha256 `7fd65ffdf62ae74f5609e3ac259588826bb84b51e3a3a84968b299332a3f35f9` (line 1 of `code/read_t2_dominance.py.sha256.txt`; the MM enters it in the cards' logs)
Third pin `d6664718...` is kept beside it as `code/read_t2_dominance.pin3.py` (sha checked). The one other place a pin is hard-coded: `FROZEN["code/read_t2_dominance.py"]`, `tests/test_t2v8f_stage4.py` line 58 (updated; its 5 pin/artifact tests green). `scene_exclusions.py` and `read_stage2.py` UNCHANGED.

## What the read takes now
* `--arm NAME` (default `C0C`): cells `<tc>|NAME_s0` / `NAME_s1` are read against P10 in C0C's place (give the name without the seed: `LC`, `EGAIN`). `--arm-stage N` (default 4).
* Stage 4 holds C0C only. An arm in a stage beyond 4 needs `--arm-root` (rollouts dir), `--arm-exclusions`, `--arm-overlay` (`<tar>::pkg/MANIFESTN.json`, G2 a) and `--arm-gate-log` (a green pytest log, G2 b); `--arm-result` defaults to `STAGEN_RESULT.json` beside the root. The last two are beyond the brief's list: without them G2 (a)/(b) would skip the new stage; absent they refuse by name (`G2A_OVERLAY_MANIFEST_MISSING stage N`, `G2B_LOG_MISSING stage N`).
* Stages 2, 3 and 4 are still read (their `--exclusions2/3/4` are required with `--arm-exclusions`: all or none); the union of the excluded scenes over every stage read leaves BOTH the arm and P10 (rule 4).
* Named refusals before anything is read (SystemExit, nothing written): `ARM_NAME_FORM`, `ARM_NAME_SEED_SUFFIX`, `ARM_NAME_RESERVED` (P10, ZERO, BLIND, C0, DAGOACC), `ARM_STAGE_BELOW_4`, `ARM_STAGE4_IS_C0C`, `ARM_STAGE4_EXTRA_INPUTS`, `ARM_ROOT_REQUIRED`. A pull that holds no such cell: gate `ARM` (`ARM_ABSENT_IN_PULL`, lists the T2 cells present), outcome GATE-FAIL; the gate key exists only when it fails.
* Output: `DOMINANCE_T2.json` keys `arm`, `arm_stage`, `floor`, `argv`; the .md has an `Arm read:` line with the argv.
Command for a stage-5 arm (stages 2 to 4 as in the read of record): `bash audit/read_record_args.sh $T/code/read_t2_dominance.py <out-dir> --arm EGAIN --arm-stage 5 --arm-root <pull5>/rollouts --arm-exclusions <EXCLUSIONS5.json> --arm-overlay "<tar>::pkg/MANIFEST5.json" --arm-gate-log <log>` (add `--arm-result` if not beside the root). EXCLUSIONS5.json: `scene_exclusions.py` with `--sources` = the concatenated 62-scene list, as for stage 4.

## Identity (item 2, binding): `logs/q_identity.json` -> `identical: true`
The pinned script, on the real pulls with the read of record's arguments (`audit/read_record_args.sh`; stages 3 and 4 on `C:/Users/Admin/t2v8f_pulls/...`), wrote `stage4/score_q_identity/`. Compared by `code/compare_dominance_reads.py`: `decision`, all gates, both seeds' 19 rows (`by_seed`), `seed_gap_beside`, `r_scn` and the `.md` are equal; outcome NOT-DOMINANT, 48 scenes, 93 starts. Excluded from the comparison: `script_sha256`, the four keys the re-pin adds, and `card_sha256_now`, the sha of the whole append-only card, which moved because the MM appended a log entry (21:56) after the record (21:54); the frozen prefix is intact in both (`card_sha_ok` true). This extra exception is beyond the brief's two; it is evidenced in the file.

## Tests and mutations
* `tests/test_t2v8f_dominance_q.py`: 14 tests, green (synthetic world + a fifth stage with arm `EGAIN`, its own result/root/exclusions; scene excluded only in stage 5 leaves P10; scenes excluded elsewhere leave the arm; wrong names refuse by name; `--arm C0C` explicit equals the default, JSON equal except argv; G2 a/b/e for the arm's stage). The 57 existing dominance, G2, R-SCN and scene-exclusion tests are green against the new file.
* `code/run_mutations_dom_q.py` carries P's 40 mutations (one anchor reworded, overridden) and 40 new ones: **80 of 80 RED** on the byte-identical working copy (`logs/mutations_dom_q.dev.log`; the first run found 5 uncaught, the tests were tightened, the second is clean). P's own runner now stops on an anchor error: use the Q runner.

## ENVIRONMENT (MEASURED today)
Directory creation: C: about 1 ms, E: about 0.2 s, F: about 1 s. A synthetic world takes 42 s to build on E:, so the suite scratch is on C: (guarded: it needs 1500 MB free; C: showed 0.8 to 3 GB today) and the large rehearsals on E:, not F:.

## IN FLIGHT / NEXT (a fresh agent or the MM)
`logs/Q_FINISHED.txt` ends `ZZ Q-CHAIN-DONE ZZ` when `audit/finish_q.sh` is done: `logs/mutations_dom_q.log` (expect `mutations RED: 80 of 80`), `logs/pytest_full_Q.log` (expect 167 + 14 = about 181 passed; P's 167 took 12.5 min on a healthy F:), `PINS_Q.txt`. If C: was low the suite is reported as NOT started: run it with a basetemp that has room. Cleanup for the PI: `E:/Projects/TanitAD-artifacts/scratch_q_t2v8f/` (rehearsal scratch), `F:/TanitAD-scratch/t2v8f_Q/`, `C:/Users/Admin/pytest_tmp_Q/`.
## UNVERIFIED until the stage-5 driver exists
Its layout is assumed to follow stage 4: all 62 scenes, T2 and RR cells named `NAME_s0/_s1`, a result with `stage: "stage5"`, `manifest_sha256` (the stage-1 package) and `manifest5_sha256` (its overlay). A stage that ran otherwise is refused by name, never read loosely.
