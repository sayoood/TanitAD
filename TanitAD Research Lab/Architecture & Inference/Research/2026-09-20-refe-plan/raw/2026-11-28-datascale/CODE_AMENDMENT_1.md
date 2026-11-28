# CODE AMENDMENT 1 to the DATA-SCALE build (M6 learning-curve driver), 2026-11-28

**Timing.** Made BEFORE any real-data fit: no 185-log index exists yet, no DATA-SCALE head or learning curve has been trained, no gate token minted for it, nothing read. `PREREG_DATASCALE.md` and its sha256 are unchanged (card sections 5 and 6 included: the learning-curve definition, the sizes, the arms and the decision rules are not touched). `CODE.sha256` is updated only for the two changed files.

**The finding (MM, 2026-11-28).** `refe/safehead_lc.py` `run()` calls `T.cv_select(...)`, i.e. it TRAINS heads (5 folds x every size x 2,000 steps), but never called `T.require_token(...)`. The trainer's `main` does (`safehead_train.py` `run`: refuses an active regression arm, then `require_token(argv)`, unless `--smoke-steps` / `--build-only`). So M6 could have trained without a launch-gate PASS, which CLAUDE.md "NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS" (PI 2026-09-26) forbids. Its docstring said "not token-gated"; that was the defect, written down. Class: "built, tested, and unreachable from its caller" in reverse -- a guard that exists for one caller and is absent from the sibling that trains.

**The change.**
1. `safehead_lc.py` `run()`, after the existing refusals (cv5m / no permute / no smoke / no build-only; `--save-steps` includes 0), BEFORE the output directory is touched and before any fit:
   - refuse if `os.environ.get(T.ARM_ENV)` (a training launch never runs under a regression arm);
   - `tok = T.require_token(rest)`, where `rest` is the trainer argv left after the lc's own flags (`--parent-logs`, `--sizes`, `--parent-cv-select`, `--atol`);
   - `tok` (token path, `token_sha256`, profile, binding) is recorded in `lc.json` under `"token"`.
   - **No bypass flag or environment variable.** The token binds `rest` (argv sha256 incl. `--out`, tree, data manifests), so a real M6 run gates `rest` with profile `refe_ds_ep_s0` or `refe_ds_sh_s0` (the s0 recipe: 2,000 steps, the registered save steps). Docstring corrected.
2. `test_safehead_ds.py`:
   - `test_learning_curve` now runs on a REAL minted PASS token (`mint_env`: `L.run_checks` / `L.finalize` / `GT.load_key`, as `test_e2e` does) for the lc's own trainer argv, at the profile's pinned 2,000 steps and save steps (sizes `5,N` on the synthetic cache; its banked parent is the trainer's `cv5` run with `--smoke-steps 2000`, tokenless). Every assertion keeps its meaning: CTRL-PARENT PASS at <= 1e-9, the 1e-6-off bank FAILS (exit 1), the N-log set equals the trainer's own cv5m, each set's table recomputes. New: `lc.json` records the token (profile, sha256). The token binds `--out`, so the FAIL arm re-uses the emptied same `--out` (`--parent-cv-select` is not in `rest`).
   - NEW `test_lc_token_gate`: with no token, with a token minted for a DIFFERENT argv (`--lr` and `--seed` swapped), and under an active regression arm, `safehead_lc.py` exits non-zero ("REFUSED" / "does not verify") before any fit and writes no `lc.json`.

**Mutation proof** (`test_logs/`): with the `require_token` call replaced by `tok = None` in a COPY of `safehead_lc.py`, `test_lc_token_gate` goes RED (`mutation_gate_test_RED.log`: the no-token run exits 0 and fits, `ZZSHLC_DONE`). With the line deleted outright, `test_learning_curve` itself goes RED on the missing `token` record (`test_safehead_ds_MUTATED.log`). The mutated copy is not staged; the tree was restored byte-identical to the staged file.

**No profile moved.** `safehead_lc.py` is in no profile (the pin tests, `test_safehead_ds.py` `test_profiles_registered` and `test_launch_gate.py --static` are green with no fingerprint change); `safehead_train.py`, `safehead_gate.py`, `launch_gate.py` are untouched.

**Verification** (serial, commit-free memory >= 8,000 MB, CPU only; logs in `test_logs/`): `test_safehead_ds.py` full **ZZSHDS_PASS 21**; `test_safehead.py` rc 0; `test_safehead_next.py` **ZZSHNEXT_PASS 17**; `test_ds_read.py` **ZZDSREAD_PASS 4**; the 8 profile-pin tests rc 0; `test_launch_gate.py --static` **ZZLG_STATIC_PASS 1184**. (`test_launch_gate.py --static` needs git to accept a temp repo on E:; it fails the same way on the UNMODIFIED files without `GIT_CONFIG_*` `safe.directory=*` in the process environment: an environment property, not this change.)

Diffs: `code_amendment_1_lc.diff`, `code_amendment_1_test.diff`. Hashes: `CODE_AMENDMENT_1.sha256`, `CODE.sha256` (updated).
