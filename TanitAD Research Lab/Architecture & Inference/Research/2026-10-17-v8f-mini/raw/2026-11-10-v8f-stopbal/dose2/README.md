# STOPBAL second dose: box-2 chain and read additions (code and tests only; nothing is launched from here)

Spec: `theory/stopbal/AMENDMENT_3_DOSE2.md` (frozen; sha256 `460fd54846348d9b654dbc8bcbfa083768c1ae82bec10a21d6df180276a2441f`, equals `AMENDMENT_3_DOSE2.sha256`). Code of record `715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf`.
Arms: BAL-G2 (`--goal-pos-weight-stop 20.81`) and BAL-L2 (`--lon-class-weight-stop 22.08`), seeds 0 and 1, DAG-OACC s{k}'s argv verbatim, `--out` moved, 3,000 steps. Order: BAL-G2_s0, BAL-L2_s0, BAL-G2_s1, BAL-L2_s1.
The banked first-dose BAL-NULL s0 / s1 (reference) and BAL-NULL_rep (floor) are reused, never retrained.

## Launch (the Master Mind lands the package first; then, from Git bash)

```
export STOPBAL_COMMIT=715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf
export STOPBAL_SSH="$(command -v ssh)" STOPBAL_SFTP="$(command -v sftp)"
bash E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/dose2/dose2_chain.sh
```

Launch it detached (WMI `cmd /c ... < NUL`), not from an agent shell. Optional: `STOPBAL_AMENDMENT=<file>` (12 h rule amendment, as in the first dose), `STOPBAL_RESUME=1` after a stop has been inspected.
The chain refuses, before any box call, when: `STOPBAL_COMMIT` is not 40 lowercase hex (rc 8); a package file is missing (rc 8); a banked NULL arm is missing or changed (rc 8, `null_banks.py`); the transport variables are unset, relative or of the wrong kind (rc 16);
its state dir collides with `stopbal_state`, `slowwm_state` or the predecessor dir (rc 2).
It then waits for the SLOWWM chain: DONE.txt (first bytes `ZZ SLOWWM-DONE`), or a STOP.txt whose head (the text before ` :: `) does not end in ` rc=12`. A STOP with rc=12 keeps it waiting. The busy marker `BOX2_BUSY.txt` must also be free.
State dir `C:/Users/Admin/stopbal_dose2_state`; sentinels `ZZ STOPBAL2-DONE` / `ZZ STOPBAL2-STOP` / `ZZ STOPBAL2-ARM-REFUSED`, each in its own file, never in chain.log. Banks: `C:/Users/Admin/t2v8f_pulls/stopbal_<tag>/`.

## Wall time (ESTIMATED)

First dose, MEASURED in `C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/bfe78918-15e8-439b-8566-593ca3297bd8/tasks/b49ubuz8x.output`: pilot 6 min (mean 0.6842 s/step), runs of 38, 41, 40 and 41 min including the pulls and the cleanup.
ESTIMATED here: 4 runs x about 40 min = about 2.7 h, plus ship (about 5 min), the pilot and its gates (about 7 min): **about 3 h from the start of the ship stage; plan for 3.5 h**. The wait for SLOWWM comes before this and is not counted.
Projection basis for the 12 h rule: 4 x 3,000 steps = 12,000 steps (`stopbal_check.py project`); it passes up to 3.6 s/step, about 5 times the first dose's measured step time.

## Read order (after the chain wrote `ZZ STOPBAL2-DONE`)

All from `theory/stopbal/dose2/read/`, CPU only, `python -I` (`C:/Users/Admin/venvs/tanitad/Scripts/python.exe`), launched detached.
1. `python -I dose2_read.py refs`: re-asserts the banked NULL arms (checkpoint sha256 against BANK_MANIFEST.json and RUN.done, lever read-back 1.0). Rc 1 and no read if any is missing or changed.
2. `job_dumps_dose2.cmd`: T1-kin dumps of the four new tags only, into the first dose's dump folder `F:/TanitAD-scratch/stopbal_read/t1` (one environment). The NULL dumps are the first dose's and stay. Done file `DUMPS_RUN2.done` (rc=0).
3. `job_stopdec_dose2.cmd`: STOPDEC of the four new tags into `dose2/read/out/stopdec` (`--split-json`). The NULL arms' STOPDEC files stay in `read/out/stopdec`.
4. `python -I dose2_read.py read --out <file.json> [--levers BAL-G2 BAL-L2]`: `stopbal_read.py` with BAL-G2 / BAL-L2 as lever arms and the expected levers 20.81 / 1.0 and 1.0 / 22.08 in control 1; refuses first when a NULL bank is not whole; the report carries the NULL checkpoint sha256.
5. Write the verdict against the card's outcome list as in `read/VERDICT_STOPBAL.md`; the first dose's rows stay as measured. Prediction written before any number (AMENDMENT_3_DOSE2.md): BAL-FP-COSTS for both levers.

## Package map

| file | role |
|---|---|
| `dose2_chain.sh` | the chain: NULL-bank check, wait for SLOWWM, ship, pilot, 12 h permit, four runs |
| `stages/dose2_run.sh`, `stages/dose2_ship.sh` | the first dose's stage scripts with only the declared lines changed (diffed in `tests/test_dose2_unit.py`) |
| `stages/stopbal_common.sh` | wrapper: points the first dose's common file at this package |
| `stopbal_check.py` | table layer over the first dose's checker (tags, weights, pilot, argv file and its pin) |
| `frozen/argvs_dose2.json`, `argvs_dose2.txt`, `argvs_dose2.meta.json`, `make_argvs_dose2.py` | the four argvs, sha256 `f00267cb...0127f4` |
| `null_banks.py` | the banked NULL arms must be whole |
| `jobs/*.cmd` | byte copies of the first dose's box-2 jobs |
| `read/dose2_read.py`, `read/job_dumps_dose2.cmd`, `read/job_stopdec_dose2.cmd` | read additions |
| `tests/`, `read/test_dose2_read.py` | unit tests and the dry run |
| `PACKAGE_MANIFEST.sha256` | sha256 of every file of the package |

## Tests (all run, results in `tests/RESULTS.md`)

* `python -I tests/test_dose2_unit.py <scratch>`: stage and job files against the first dose's, tables, planted selfchecks, red arms.
* `bash tests/test_dose2_dryrun.sh`: the chain over the first dose's stubs and stand-in trainer, no box call; waiting and released SLOWWM gate, collision guard, red arms (wrong weight, wrong commit, missing NULL bank).
* `python -I read/test_dose2_read.py <scratch>`: read additions with planted fake banks.
* Scratch only under `F:/TanitAD-scratch/dose2_build/`.
