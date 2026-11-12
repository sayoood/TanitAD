# STOPPERC stage 1 box-2 package (NOT RUN, 2026-11-12)

Card `../../PREREG_V8F_STOPPERC.md` section 5, Q3; `../../AMENDMENT_3_A_READINGS.md`; gate list `../RESULTS.md`. Code of record: `3e264bbf3fa2f6f17bea448de09aa972faaf7725` (branch `agent/v8f-stopaux-20261112`, child of `715ef8fc`).
Template: `../../../stopbal/box2/` (BAL-NULL_s0/_s1/_rep ran from it). **This package does not copy it: it re-points it.** `stopperc_check.py` imports the STOPBAL checker (`stopbal_check.py`, sha256 `70e260a6...5212`, pinned, shipped unchanged beside it) and patches its tags, paths, argv loader and gate; `stopperc_chain.sh` sources STOPBAL's `stages/stopbal_common.sh` in place (ssh/sftp absolute paths, `put_verified`, `pull_verified`) and its `stopbal_ship_tree.py` / `stopbal_refresh.py`. The STOPBAL folder must stay where it is.

## Runs and cost
| tag | what | est. |
|---|---|---|
| NULL_s0 | BAL-NULL_s0's argv, `--out` only | 40 min |
| NULL_rep2 | NULL_s0 with `--seed 3` (Q3; BAL-NULL_rep is `--seed 2`) | 40 min |
| A_s0, A_s1 | DAG-OACC s0 / s1 argv + `--stop-aux on --w-stop-aux W --stop-aux-extr D:\tanitad\stopperc\data\stopaux_extr_join4572.json` | 44 min each |
| NULL_s1, NULL_rep | CONDITIONAL, only if the reuse check says DIFFERS | 40 min each |
Total 168 min = 2.8 h (ESTIMATED, card); 4.1 h with the conditional pair; plus ship, W smoke, gates, pulls. The chain refuses above 12 h.

## How the MM launches it
1. PI's stage-1 go with the stage-0 results (card section 8). Box 2 free: `C:\Users\Admin\box2_ship\BOX2_BUSY.txt` absent (the SLOWWM chain holds it until about 23:30 on 2026-11-12, ESTIMATED). The chain REFUSES (rc 12), it does not wait.
2. Dry checks on the dev box: `python -m pytest tests/test_stopperc_check.py -q -p no:cacheprovider --basetemp <existing parent>/b` (36 passed on 2026-11-12), `bash -n stopperc_chain.sh`.
3. Launch detached, the way the newest chain is (`../../../slowwm/box2/`, `restart_adopt_2026-11-12/ADOPT.md`): WMI `cmd /c ... < NUL` running `"C:\Program Files\Git\usr\bin\bash.exe" -l -c "STOPPERC_SSH=<abs ssh> STOPPERC_SFTP=<abs sftp> bash <this folder>/stopperc_chain.sh"`. Never `Git\bin\bash.exe`.
4. State: `C:/Users/Admin/stopperc_state` (`chain.log`, `STOP.txt` prefix `ZZ STOPPERC-STOP`, `DONE.txt` prefix `ZZ STOPPERC-DONE`, `W.txt`, `NULL_REUSE.json`). Resume after a stop: inspect, then `STOPPERC_RESUME=1`. Banks: `C:/Users/Admin/t2v8f_pulls/stopperc_<tag>/`.

## What each stage refuses on
* **start:** package or STOPBAL files missing; base checker sha differs; camera table (`C:/Users/Admin/stopaux_extr/stopaux_extr_join4572.json`) not sha256 `77b6e8dc...e361`; STOP.txt present; busy marker held; projection above 12 h.
* **ship:** tree tar/manifest not for the commit, any put whose box-2 sha256 differs, table on box 2 differs, `SHIP_RESULT.json` not ok.
* **W smoke:** box 2 busy or GPU not idle; `probe_n_pos` 0 (named STOP, rc 14; **the sampler is not changed**: a STOP-stratified probe batch is a code change landed first); W script not the pinned one (`064baba6...4de63`); W not a 4-decimal number. W is written to the A argvs only through `STOPPERC_W`, and the frozen argv file keeps `W_PLACEHOLDER` (so its sha stays pinned). The final A argv is gated again by the normal `stage` (W is bound in the token).
* **each run (`stopperc_check.py stage`):** the base gate (G-HYG/DVB/LIVE of STOPBAL: DAG-OACC architecture, init sha, DAgger bank, data md5, acc/dagger probes) PLUS: G-DVB1 (`levers.stop_aux`, `w_stop_aux` equal the argv), 2 (`params.STOPAUX` 132,098), 3 (vision-only), 4 (table path, sha256, coverage_train >= 0.99, 4,512 clips), 5 (fresh keys = the 8 stop keys; off arms carry no `stop_*` key, no flag), 6 (A argv = template + exactly the 3 flags, W filled); G-LIVE1-7 from `stop_aux_probe`/`live`; G-LIVE8 (build half) after training: the eval route `V8FMiniConfig(**ck["cfg"])` + strict load builds the ON checkpoint. Token binds commit, argv sha256, data md5s, table sha256, W and the checker sha256.
* **reuse check (G-DVB7, after NULL_s0):** tensor-basis comparison of NULL_s0's model with BAL-NULL_s0's (pinned file sha256 `5169ea95...aa2fdf`, `C:/Users/Admin/t2v8f_pulls/stopbal_BAL-NULL_s0/`). Both the tensor result and the whole-file sha are written. A whole-file sha cannot match by construction (the checkpoint embeds `argv_sha`, which holds `--out`; the STOPBAL parity note says the same), so the card's "sha match" is read on tensors. EQUAL: BAL-NULL_s1 / _rep reused. DIFFERS: NULL_s1 and NULL_rep are appended. UNKNOWN: STOP.

## What is verified, and what is not
MEASURED (dev box, CPU): 36 unit tests pass; `make_argvs_stopperc.py` parses all 6 argvs with the trainer export of `3e264bbf`; `stopbal_ship_tree.py build` succeeds at the commit (69 members, includes `v8f_stopaux.py`); `bash -n` passes.
DRY RUN (added 2026-11-12 ~21:45): `tests/test_stopperc_dryrun.sh` (the STOPBAL fake-box harness, ported) reads `DRYRUN ok=113 failed=0` for the agent and again on the MM's re-run, with no BREACH. It found one real bug: a W of 0.0000 passed the W smoke. The chain now stops there (rc 14), see `tests/DRYRUN_FIXES.md`. Two test hooks are inert on a real launch. The launch waiter (`C:/Users/Admin/stopperc_launch/waiter.sh`) starts the chain only on the MM's `MM_DRYRUN_OK.json`, whose sha256s must equal the current package files.
NOT verified: the real trainer, GPU and transport (the dry run uses a stand-in trainer and stubs, and skips G-LIVE8's eval build); the keys `stop_aux_probe.*` are read from the trainer source at `3e264bbf`, not from a real ON selfcheck (none exists); `eval_build_problems` is untested against a real ON checkpoint; `forward_infer` independence of the head is pinned only by the tree's own test, not re-run on the checkpoint.
