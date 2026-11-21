# OTBAND: box-2 launch package (O/T band-split retrain)

Card: `theory/PREREG_V8F_OTBAND.md` (frozen). Choices: `theory/otband/IMPL_CHOICES_OTBAND.md`. Frozen argvs: `theory/otband/argvs_otband.json` (copied to `frozen/argvs_otband.json`, sha256 pinned in `otband_check.py`; not a token changed).
Code under test: commit `202849da013b5d74c0b0086dd9865fcda6836018` (set `OTBAND_COMMIT` to it), branch `agent/v8f-otband-20261109`. Pattern: `theory/ladder/box2/` (read, never edited). **Nothing here has touched box 2**: it was exercised only over a stub transport and a fake box folder (evidence class below).

## What the Master Mind does before launch
1. Put the commit above (or its successor) in `C:/Users/Admin/tanitad-push` and note its full 40-hex sha. The ship step exports that commit's trainer import closure; it refuses a commit whose trainer, model or data module lacks the OTBAND needles (`otband_check.NEEDLES`).
2. Decide the token-dump blocker is still open (OB-BAND-TOK stays out; see the table in `otband_check.py`).
3. Box 2 needs `D:\tanitad\env.cmd` (PY, RUNSAFE), `launch.ps1`, the M2b start states under `runs\v8f_mini_full`, and the v9 data files named in the argvs, as the ladder found them. No zcache is read.
4. Launch from a session background task (the ladder's log notes WMI launches ended silently):
```
export OTBAND_COMMIT=<40 lowercase hex>
export OTBAND_SSH="$(command -v ssh)" OTBAND_SFTP="$(command -v sftp)"      # absolute paths; PATH is never searched; OTBAND_DRY must be unset
[ OTBAND_AMENDMENT=<file> ]                                                  # only if the pilot projects above 72 h and the PI amended
bash E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/otband/box2/otband_chain.sh
```
State: `C:/Users/Admin/otband_state` (`OTBAND_STATE`). Banks: `C:/Users/Admin/t2v8f_pulls/otband_<tag>/` (beside the ladder's `ladder_<tag>`).

## Order and gates
1. **Wait**: the chain polls (`OTBAND_WAIT_POLLS` x `OTBAND_WAIT_POLL_S`, default 1152 x 300 s = 96 h) until the ladder has ended, then refuses with rc 12 if it never does. "Ended" = `C:/Users/Admin/ladder_state/FULL_DONE.txt` starts with `ZZ LADDER-FULL-DONE`, or `FULL_STOP.txt` starts with `ZZ LADDER-FULL-STOP`, and in both cases the shared busy marker `C:/Users/Admin/box2_ship/BOX2_BUSY.txt` is absent. The chain then claims that same marker (noclobber), so the two chains exclude each other. Before the pilot and before every run it also requires no `python.exe` on box 2 and GPU memory used below 1,500 MiB (`OTBAND_GPU_IDLE_MIB`); an unreadable probe is a refusal.
2. **ship**: tar of the commit's closure to `D:\tanitad\otband\ship`, every put re-hashed, overlay onto the OTBAND-owned tree `D:\tanitad\otband\repo` (changed files copied aside first; nothing deleted). The shared `D:\tanitad\repo` is never touched.
3. **pilot** (`PILOT_OB-BAND_s0`): the exact OB-BAND_s0 argv plus `--steps 200`, `--out` moved to `D:\tanitad\otband\pilot_out\OB-BAND_s0`. The same job also gates the full-length OB-BAND_s0 and OB-CTL_s0 argvs (a real selfcheck each), so `steps_total` is read before the 72 h decision. `PILOT.json`: mean s/step over steps 50 to 200, peak GPU MiB, compute- or loader-bound (median utilisation >= 60 %, ESTIMATED heuristic), projection. Re-derived on the dev box from the pulled log (`PILOT.verified.json`).
4. **72 h rule** (`otband_check.py permit`): projected hours for the five planned runs (5 x 9,582 steps at the pilot step time) must be at most 72, else the chain stops (rc 3) and the case goes to the PI; an amendment file containing `pilot_sha256=<sha256 of PILOT.verified.json>` and `commit=<OTBAND_COMMIT>` lets it go on. The projection with the two token arms is printed beside it.
5. **runs**: OB-CTL_s0, OB-BAND_s0, OB-CTL_s1, OB-BAND_s1, OB-BAND_rep. Each: disk read, GPU idle, argv pin, gate, token, training under the token, post check, bank, cleanup.

## What the gate asserts per run (`otband_check.gate_problems`, from the trainer's `selfcheck.json`)
Argv: the shipped line equals the frozen tokens; a mismatch names the first differing token (`pin-fail`). Gate line = the argv with `--out` moved to `D:\tanitad\otband\gate_out\<tag>` and `--selfcheck-only` appended; the selfcheck's `argv_sha` must equal the sha of that line.
Read from the selfcheck (real key names at `8e9499e0`): `declared_vs_built == "PASS"`; `steps_total` (200 pilot, 9,582 full); `S_steps_per_epoch_100pct == 4791`; `info.{train_sha256, excluded_sha256, n_clips, n_excluded, batch, md5_overridden, excluded_in_train}`; `info.init_ckpt.{sha256 (M2b start state), applied, n_mismatch_after_load == 0, source_arm, source_seed, path, declared_new_keys}`; `levers.kin_mode`; `levers.params.{S, adapter, cond} == 0` (the frozen set, read back from the BUILT model's requires_grad counts) and `T`, `O` > 0; `levers.{w_op, w_tac_traj}` (present only in band arms, 0.3333333 and 0.6666667; absent in OB-CTL); `live.{pass, missing_terms, unexpected_dead, terms, control_prior_grad_sum == 0.0, head_o_grad_step0 > 0, groups_grad}` (band terms `ade_op fde_op ade_tac fde_tac op tac` live in band arms, absent in OB-CTL; frozen groups read exactly 0.0); `otband_probe.{pass, otband, detach_upper == true, contribution.w_op / w_tac_traj (weight, term, moved_total_by equal to weight x term), purity.upper_grad_abs_sum == 0.0, purity.head_o > 0, code_edge_max_dknot > 0, token_edge}`; `levers.{recover.recover_p, dagger_p, ego_perturb.ego_perturb_p} == 0`.
Each assertion has a planted-bad test in `tests/test_otband_check.py`.

## Token
Written only by `otband_check.py gate` on box 2, after the assertions pass, to `D:\tanitad\otband\runs\<tag>\gate\gate_token.json`. It binds tag, commit, argv sha256, the six data-file md5s, the sha256 of the shipped tree manifest, the trainer's sha256 and the selfcheck's sha256. The run step recomputes all of it and refuses to train without a matching token (`token-fail`). Each bound field has a test that changes it and expects a refusal, as has a hand-written token.

## Post check (per run, on box 2) and bank
Steps reached (`summary.json` `steps` == 9,582 and `done`, checkpoint `step`), `argv_sha`, checkpoint exists, and every S, `adapter` and `cond` tensor of `ckpt_last.pt["model"]` is bit-identical to the M2b init checkpoint named in the argv (shape, dtype, bytes; tests change one bit in each of the 11 frozen key families). It also requires that some T or O tensor moved, and reads the BUILT config the checkpoint carries (`cfg.detach_upper == true`, `w_traj`, `w_op`, `w_tac_traj`, `stok_dim`), which is the only post-run reader of `w_traj` and `detach_upper` for OB-CTL.
Bank (dev box, each pull sha256-checked against box 2's own `certutil` hash, 3 attempts): `ckpt_last.pt`, `summary.json`, `train_log.jsonl`, `RUN.done`, `gate_token.json`, `selfcheck.json`, plus `argv.txt`, `BANK_MANIFEST.json`, `BANK.done`, `CLEANUP.json`. The token is re-verified against the frozen argv on the dev box.
Cleanup: only after the bank is verified does the dev box put `BANK_VERIFIED_<gid>` on box 2 and start a cleanup job. That job deletes the run's own temporary gate dir (`D:\tanitad\otband\gate_out\<tag>`; for the pilot also its out dir and the two full-length gate dirs) by literal path, refuses any folder whose parent is not the named one, logs (path, bytes) to `CLEANUP.json` (pulled into the bank), and the dev box asserts the dirs are gone. It never touches a checkpoint; the five checkpoints stay on box 2.

## Failure behaviour (item 8): what I chose and why
* Ship, pilot, the 72 h rule, a ladder or GPU wait that times out, a post-check failure, a trainer crash, a pull mismatch, low disk, a transport or package problem: **the chain stops** (`STOP.txt`, a named exit code), and no later arm starts. These either say something about the shared setup or leave a half-finished run that a human should look at before the GPU is used again.
* A **gate / pin / token refusal of one full run** (stage rc 11) writes `ARM_REFUSED_<tag>.txt` and the chain **goes on with the next run**: the arms are independent (own init, own out dir, own token), the refusal happened before any GPU use, and a wrong BUILT lever in one arm says nothing about the others. The chain then ends with `STOP.txt` (rc 4), never `DONE.txt`, listing the refused arms. A refusal inside the pilot (which also gates OB-BAND_s0 and OB-CTL_s0) stops the chain.
* Exit codes of the chain: 3 72 h rule; 4 gate/pin refused; 5 job or transfer failed; 6 disk low (box 2 D: below 5 GB, a constant in `otband_check.py` / `otband_common.sh`; box 2 read 11 GB free recently, nothing measured here; the pilot and each run need 5 GB, ship 1 GB); 7 pulled copy differs; 8 package, commit or transport; 9 ship refused; 10 pilot unreadable; 11 box-2 timeout or unreadable probe; 12 box busy, GPU not idle, or ladder not ended; 13 blocked tag.
* Sentinels, each in its own file (never in `chain.log`): `STOP.txt` (`ZZ OTBAND-STOP`), `DONE.txt` (`ZZ OTBAND-DONE`), `ARM_REFUSED_<tag>.txt`. Resume with `OTBAND_RESUME=1`; finished stages are skipped, a stopped run restarts from `ckpt_last.pt` (the trainer resumes by itself and refuses a changed argv).

## Isolation on box 2
Own root `D:\tanitad\otband\` (tree, checker, ship, runs, gate dirs, pilot out). The ladder's `D:\tanitad\ladder`, `D:\tanitad\repo` and `runs\v8f_ladder*` are never written; the dry run asserts byte-identity of both folders. **One deviation from "own root", forced by the frozen argv:** `--out` is `D:\tanitad\runs\v8f_otband\<tag>` (not a token I may change); that folder is written only by this chain and is not read by the ladder. Reads outside the root are read-only: the M2b start states and join/p10 files under `runs\v8f_mini_full`, `cache_full`, `data\v9` (the ladder reads the same files).

## Fail-closed transport
All traffic goes through `$OTBAND_SSH` and `$OTBAND_SFTP` (absolute paths, never searched on PATH). Unset, relative, absent, or a file carrying the stub marker without `OTBAND_DRY=1`, or `OTBAND_DRY=1` with a file that lacks the marker (a real client in test mode) all refuse with rc 16 before any call. The tests set both variables to the stubs under `tests/` by absolute path, put poison `ssh`/`sftp`/`scp` first on PATH, and assert no breach and zero calls on each refusal.

## Tests (dev box, CPU, no torch)
* `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I tests/test_otband_check.py <scratch>` -> `UNIT ok=<n> bad=<m>`.
* `bash tests/test_otband_dryrun.sh` (about 5 minutes; scratch folder deleted at the end) -> `DRYRUN ok=<n> failed=<m>`. The real job `.cmd` files run on Windows `cmd.exe` against a fake box folder; only the trainer is a stand-in. Result logs: `tests/UNIT_RESULT.log`, `tests/DRYRUN_RESULT.log`.

## UNVERIFIED (nothing below has run on box 2 or with the real trainer)
* Everything on the real box: the ship overlay into a fresh own tree (the export list is the ladder's; whether the trainer's closure plus that list is a complete tree is unproven; a missing file shows up as a named `gate-fail` with the trainer log), `launch.ps1`, `certutil` and `nvidia-smi` output forms, `env.cmd`.
* `steps_total == 9,582` and `S_steps_per_epoch_100pct == 4791`: derived from the ladder README (19,164 = 4 x 4791) for non-wmonly M2b windows. A different value makes the pilot's gate refuse and name `steps_total`.
* The data-identity constants and the M2b start-state sha256 are inherited from the ladder checker (same files, same corpus); not re-read from this trainer's selfcheck.
* The stand-in trainer writes the key names I read from the source at `8e9499e0`; a real selfcheck may differ in a detail the source reading missed.
* `OB-BAND-TOK` arms: blocked; the tag table makes adding them a data change (`blocked`, `order`, the token dir in the argv file and then its pinned sha256).
* The pilot's step time is of a 200-step run that includes warm-up inside steps 1 to 50; the projection is ESTIMATED from steps 50 to 200.

## Findings in the code at `8e9499e0` that bear on the gate (read from source, UNVERIFIED by running)
1. `selfcheck.json` does **not** record the frozen set, `w_traj` or `detach_upper` for OB-CTL: `frozen_groups` is compared inside `declared_vs_built` only. The gate reads the frozen set from `levers.params` (trainable counts per group), `detach_upper` from `otband_probe.detach_upper`, and for OB-CTL relies on `declared_vs_built == PASS` plus the post-run checkpoint `cfg`.
2. OB-CTL now carries `otband_probe` with `otband: detach-only` and the purity of `ade` and `fde` (commit `202849da`); the gate requires both.
3. The purity probe now reads `op` and `tac` on band arms (`purity.by_term`, commit `202849da`); the gate requires both. MEASURED in `test_v8f_otband.py` on CPU; on a real batch UNVERIFIED until the first gate.
4. `live_check` still requires T's gradient and head_o's gradient; with S, adapter and cond frozen, `groups_grad` for those must read 0.0, which this gate asserts. WM parameters stay trainable in these arms (the freeze set is S, adapter, cond) and WM trains on a frozen adapter; that is the card's design, noted so the post check's "11 frozen key families" is not read as "everything but T and O is frozen".
