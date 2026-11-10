# IMPL_CHOICES_OBD: what the launch package decides that the card does not fix

Card: `PREREG_V8F_OBD_DRAFT.md` (draft; never edited by the package builder). Anything I think the card needs is listed here, as a proposal for the Master Mind. Evidence classes: SOURCE (read at commit `202849da`), MEASURED (ours, path named), ESTIMATED, UNVERIFIED.

## 1. Provenance of the package
Forked from `theory/otband/box2/` (read only): same stages (wait, ship, pilot, permit, runs, bank, cleanup), same fail-closed transport, same token route. Changed: names (`OBD_*`, tags `OBD-*`), the tag table (NULL / CTL / BAND, one conditional arm), the gate assertions (nothing frozen, DAgger and acceleration levers, no token arms), the post check (S, T and O must move; seed, DAgger record, init), the 12 h rule, the wait condition (EGOWM sentinels), NULL parity, the DAgger-bank binding. The OTBAND freeze identity and the strategic-token arms are gone.

## 2. Argvs (`make_argvs_obd.py`, MEASURED)
Each arm is DAG-OACC s{k}'s `argv.json` token list with `--out` replaced by `D:\tanitad\runs\v8f_obd\<tag>` and the card's flags appended; `*_rep` replaces the value of the existing `--seed` in place. The generator asserts that the diff against the DAG-OACC argv is exactly that, that every argv parses with the parser of `202849da` (trainer sha256 `1ddf491d...`, asserted), and that `argv_sha` of the DAG-OACC argv at this trainer equals the recorded one (`0ee46fc1...`, `2d846e7c...`). `--w-traj 0` is appended after the DAG-OACC tokens: DAG-OACC's argv has no `--w-traj`, so nothing is duplicated. The s1 bank sha256 is the full 64 hex of DAG-OACC's `argv.json` (`36c7aea964dafe9423253aa479c2fc6cdfc93848395b593315cba812f5f04d4d`); the 60-hex form in the task text (`...f0`) is a truncation and is refused by a test.

## 3. What the selfcheck does NOT record (SOURCE) and how the gate covers it
* **Training seed**: not in `selfcheck.json` (only in `summary.json`, the checkpoint and `argv_sha`). The gate reads the pinned argv's `--seed` against the tag table and requires the selfcheck's `argv_sha` to be the sha of the pinned gate line; the post check reads `summary.seed` and checkpoint `seed`.
* **`detach_upper`**: not in `levers`; for CTL / BAND it is `otband_probe.detach_upper`; for NULL there is no `otband_probe` at all (the trainer writes it only when a band / token / detach lever is on, `v8f_mini_train.py:3463`). The gate takes the absence as the NULL evidence and refuses a probe that says otherwise; the post check reads `cfg.detach_upper` from the checkpoint for all three kinds.
* **Parameter counts per group** are recorded (`levers.params`); the gate asserts equality with DAG-OACC's, not just "> 0". `dagger_seed`, `dagger_p`, the bank sha256 (built and declared), `w_acc`, `p_gt` are recorded and asserted.

## 4. Purity of the acceleration term: a launch blocker, made mechanical
`PURITY_ACC_FINDING.md`. The checker requires `acc` among the purity keys (`ACC_PROBE_REQUIRED = True`, `purity_keys()`); the ship step requires the probe-change needle. This is stricter than the card's "if it does not, the probe is extended ... and the commit is named here before freeze": until that commit exists, no CTL / BAND arm can pass the gate, and the pilot (a BAND job) stops the chain at once. Proposal for the card: name the new commit in "Code" before freeze.
Second gap, not required by the checker: the purity probe runs on the logged probe batch, not on a batch with DAgger rows swapped in (the card says "on a batch that contains DAgger rows"). The finding lists a proposed (untested) extension. If the PI wants the card's wording kept literally, that extension is a second held-code change.

## 5. Order, pilot, 12 h rule
Run order is the task's: NULL_s0, NULL_rep, CTL_s0, BAND_s0, CTL_s1, BAND_s1, BAND_rep (+ conditional NULL_s1 last). The pilot is 200 steps of OBD-BAND_s0 and also gates the full-length BAND_s0, CTL_s0 and NULL_s0 argvs (steps_total is read before the 12 h decision). The rule projects the seven planned runs at the pilot's BAND step time and does not include the conditional arm (printed beside it). I kept OTBAND's amendment file (`pilot_sha256` + `commit`) as the only way past a refusal: the card says that case goes to the PI, and the file is how the PI's yes is carried. The card's own ESTIMATE is 4.5 h; at the MEASURED 2,276 s per run the projection is 4.4 h (unit test).

## 6. NULL parity: the card's file-sha rule cannot hold by construction (SOURCE), proposal
The card compares `OBD-NULL_s0`'s `ckpt_last.pt` sha256 with DAG-OACC s0's. `v8f_mini_train.py` `save()` (`:3518`-`:3523`) writes `argv_sha` (the sha of the argv, which contains `--out`), `cfg`, `p10_path`, `dagger` (bank path) and the optimiser state into the checkpoint. `--out` differs by the card's own design, so the files cannot be byte-identical even if every weight is. In addition, DAG-OACC ran an earlier trainer file (the card names `30819a6f`; `202849da`'s is `1ddf491d`), so numeric parity of the weights is itself unproven.
Consequence with the card's rule as written: OBD-NULL_s1 is always appended (about 38 min of GPU on the MEASURED 2,276 s per run, 87 MB of checkpoint). That costs little and is what the card says to do on a parity failure, so the default basis is `file`, unchanged. The parity command also records a **tensor-level comparison** (`model` state dicts, shape, dtype, bytes; the DAG-OACC checkpoint is read from `v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt` on the dev box) in `NULL_PARITY.json`, so the Master Mind sees "file differs, weights identical" or "weights differ" without a second pass. `OBD_PARITY_BASIS=tensors` switches the decision to that comparison; I did NOT make it the default because the card does not say so.
Proposal for the card: define NULL parity on the `model` tensors (and say "weights" instead of "checkpoint"), and say whether the optimiser state counts.

## 7. Other choices
* **Wait**: the EGOWM sentinels as given in the task (`DONE.txt` prefix `ZZ EGOWM-DONE`, `STOP.txt` prefix `ZZ EGOWM-STOP`, directory `C:/Users/Admin/egowm_state`); a marker file without the prefix does not count (tested). The busy marker is the shared `C:/Users/Admin/box2_ship/BOX2_BUSY.txt`, claimed noclobber, released only if still ours.
* **DAgger bank**: `bank.json` on box 2 is re-hashed against DAG-OACC's sha256 at every preflight and bound into the token (a field the OTBAND token did not have). The trainer re-hashes it too (`--dagger-bank-sha256`); the double check costs one file read.
* **Gate strictness beyond the card's list**: `acc_probe` and `dagger_probe` pass with their controls exactly 0.0; `info.dagger.coverage.blocked == false`; no `wm_only` / level / lead head / token; no new init keys; per-group G-LIVE gradient > 0 for all seven groups.
* **`_rep` arms** keep `--dagger-seed` at the init's index (the card: "only the listed flags"). Whether the swap's draws depend on `--seed` is UNVERIFIED; if they do not, the replicate varies training order and dropout only, and the floor is a floor of that.
* **Disk**: box-2 floor 5 GB per run and 1 GB for ship, dev-box C: floor 2,000 MB per pull (`OBD_NEED_C_MB`); constants, not measured here.
* **Cleanup**: the pilot's cleanup now also removes the NULL_s0 gate dir (the pilot gates it).
* **Exit code 14** (new): the parity reference is unreadable.
* **Test hooks** that weaken nothing on a real launch: `OBD_TEST_DAGOACC_SHA` (the dry run's stand-in checkpoint is not DAG-OACC's), `OBD_FAKE_*` and `OBD_ROOT` (box-side, only meaningful with the fake root), `OBD_TEST_ARGVS`.

## 8. Things in the card I could not check
* "Seven runs ... about 4.5 h": plausible against 2,276 s per run (7 x 2,276 s = 4.4 h) but the pilot is the real measure.
* The card's launch probe name `oband_probe` is the trainer's function (`v8f_mini_train.py:1543`); its selfcheck key is `otband_probe`. The checker reads the key, the needle names the function.
