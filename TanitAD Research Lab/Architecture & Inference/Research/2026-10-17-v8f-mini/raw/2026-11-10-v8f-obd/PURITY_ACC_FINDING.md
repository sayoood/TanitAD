# PURITY_ACC_FINDING: the launch purity probe does not read the acceleration term (LAUNCH BLOCKER for OBD-CTL / OBD-BAND)

Code of record: commit `202849da013b5d74c0b0086dd9865fcda6836018`, read with `git show 202849da:<path>` (files below). Nothing here is applied anywhere; `stack/` is held.
Card: `PREREG_V8F_OBD_DRAFT.md` Controls item 3 ("every O term of the arm's own loss, the acceleration term included, puts exactly 0.0 gradient on S, T and the token embedding"; "if [the launch probe] does not [read it], the probe is extended with its own test and mutation").

## The gap (evidence class: SOURCE READ at 202849da + MEASURED on CPU, tiny synthetic model, see below)
* `stack/tanitad/models/v8f_mini.py:1190` `T["acc"] = accel_term(out["traj30"], ...)` (always logged) and `:1239` `total = total + cfg.w_acc * T["acc"] if cfg.w_acc > 0.0`: with `--w-acc 49.6969` the acceleration term is in the total, a function of O's plan `traj30` only, so an O term.
* `stack/scripts/v8f_mini_train.py:1601` (in `oband_probe`, def at `:1543`) `keys = ("op", "tac") if band else ("ade", "fde")`: under `--detach-upper` the probe back-propagates only these keys to S / T (`up` list) and O's head. **`acc` is not in `keys`**; `selfcheck.json` `otband_probe.purity.by_term` therefore holds `{ade, fde}` (CTL) or `{op, tac}` (BAND) and no `acc`.
* `acc_probe` (`:1957`, called `:3378`) does read the term but only checks that the total moves by `w_acc x acc` and that `head_o` / `read_o` get gradient (`head_o_grad_diff`, `read_o_grad_diff`); it reads nothing on S or T.
* So the DAG-OACC term that carries the card's second question (the acceleration lead) is the one O term the purity gate never reads.
* MEASURED (tiny model, CPU, scratch export of 202849da, script reproduced in `purity_acc/`): with `detach_upper` and `w_acc 49.6969` the acc term puts exactly 0.0 on S/T and 16.46 on head_o; without detach it puts 0.0554 on S/T; `oband_probe(...)["purity"]["by_term"]` keys: `['ade','fde']` (CTL-like), `['op','tac']` (BAND-like). **The code is pure today; the probe cannot say so, and could not say otherwise.** A leak carried by the acc term alone passes the launch gate (shown below: the unpatched probe stays green on it).
* Second, smaller gap (SOURCE READ only, no test written): `oband_probe` runs on `probe` (`:3333`-`:3335`, the logged windows), not on a DAgger-swapped batch; `dagger_probe` (`:2676`) checks that the swapped batch reaches the loss and that every group has gradient, not purity. The card says "on a batch that contains DAgger rows". Proposed (UNTESTED, left out of the diff): `oband_probe(model, dag.apply(probe, dag.PROBE_STEP, p=1.0), cfg)` recorded as `selfcheck["otband_probe_dagger"]` when `dag.active`. The OBD checker does not require it.

## Proposed diff (`PURITY_ACC_PROPOSED.diff`, 2 lines) + test + deliberate regression
```
         keys = ("op", "tac") if band else ("ade", "fde")
+        if cfg.w_acc > 0.0:
+            keys = (*keys, "acc")
```
Test: `purity_acc/test_v8f_oband_acc.py` (for `stack/tests/`, imports the helpers of `test_v8f_otband.py`): `by_term` is `{ade,fde,acc}` on a CTL-like and `{op,tac,acc}` on a BAND-like model, unchanged `{ade,fde}` at `w_acc 0`; regression 1: detach removed -> "a detach is missing"; regression 2 (the mutation): the acc term alone is wired to S's head (`+ 1e-6 x sum(head_s params)` on `T["acc"]`) -> "O's acc term puts gradient".
Run on the scratch export (CPU, `CUDA_VISIBLE_DEVICES=-1`):
* patched trainer: `5 passed`; the existing `test_v8f_otband.py -k "probe or purity or detach"`: `8 passed`.
* unpatched trainer (diff reverted): `3 failed, 2 passed`; the failed ones are the two key-set tests and the acc-only leak test, i.e. the leak passes the current probe.

## What the OBD package does about it
* `obd_check.py` requires `purity.by_term` keys `{ade,fde,acc}` (CTL) and `{op,tac,acc}` (BAND), each with `upper_grad_abs_sum == 0.0` and `head_o > 0`. At `202849da` the real selfcheck cannot carry `acc`, so **every CTL and BAND gate refuses until a commit with this probe change exists** (OBD-NULL needs no probe and is unaffected). This is deliberate: the blocker is mechanical, not a README line.
* To launch: the held `stack/` session lands the diff + test + mutation, the Master Mind names the new commit in the card before freeze and sets `OBD_COMMIT` to it (the ship step requires the needle `if cfg.w_acc > 0.0:` plus `"acc")` in the shipped trainer). Argvs do not change (the card's code of record is the only thing that moves).
* The checker's `ACC_PROBE_REQUIRED` constant flips this requirement off only for the unit test that proves the refusal; it is not an environment switch.
