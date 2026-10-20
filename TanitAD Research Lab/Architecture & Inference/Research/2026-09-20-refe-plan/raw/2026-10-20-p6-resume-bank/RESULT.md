# P6 chain: refuse a RESUME whose bank differs from the run's own, before any stage spends (2026-10-20)

**Headline (integration):** three edited files, staged under `p6ident/out/`, for the NEXT ship. The Master Mind lands them. The pod runs `bcc04b5` and is untouched. `SHIP_MANIFEST.json` carries sha256 entries for `refe/p6_pod.sh`, `refe/p6_prep.py`, `refe/test_p6_chain.py`; they go stale on landing and must be regenerated at the ship.

Base: tip `b6422ba5ea94e26750372a7bc66f598a6da395ce` (`agent/arch-inf-20260803`, pinned once). Base blobs: `p6_pod.sh` 968958c4, `p6_prep.py` 11572a35, `test_p6_chain.py` 685c8be9 (the exports hash to them, `git hash-object --no-filters`).

## What changed

| file | change | where |
|---|---|---|
| `refe/p6_pod.sh` | `resume_bank_check()` and two calls | def `p6_pod.sh:259-270`; call 1 `:271` (right after `recover`, before `inputs`); call 2 `:425` (right after `ARGS` / `ARGS_SHA` at `:421-423`, before `lab_stage`, so before lab / mem / egr / launch gate / preflight / train) |
| `refe/p6_prep.py` | new subcommand `resume-bank` (`resume_bank()`, `_norm_bank_path()`) | `p6_prep.py:237-269`, CLI `:301`, `:330` |
| `refe/test_p6_chain.py` | static properties + 12 mutants, helper tests, real-chain sandbox tests, one line in 4c | `check_chain` block, `mutations()`, `rb_*` / `resume_bank_*_tests` before `main()`, `4l` call at the end of layer 4 |

**What the run records.** `train.py` writes `<run>/config.json` ONCE, on a fresh start: `cfg_out = {"argv": sys.argv, "args": {... vars(a) ...}, "identity": ident, ...}` at `train.py:1500-1503`, `json.dump` at `train.py:1510-1511`. `args.targets` is the parsed `--targets` (argparse, `train.py:1933`). It is read instead of `argv` because the parsed value is the same for `--targets=X`, an abbreviated flag and a re-ordered argv.

**Behaviour.**
- `$RUN/config.json` absent: return at once, no call, no log line. A fresh start is byte-for-byte unchanged.
- Present: `"$PY" p6_prep.py resume-bank --config "$RUN/config.json" --bank "$CBANK"`.
  - rc 0 (same bank): `resume bank ok: ZZRESUME_BANK_SAME ...` is logged and the chain proceeds.
  - rc 3 (different): stdout and `chain.log` get `ZZP6_RESUME_BANK_MISMATCH run=<path> chain=<path> run_n_tuples=<n> run_targets_sha256=<12> -- ...`, then `refuse`.
  - rc 4 (config.json unreadable, not JSON, no `args.targets`, `args.targets` not a string, or a directory): `ZZP6_RESUME_BANK_UNREADABLE`, refuse.
  - Any other rc (the check itself crashed): `ZZP6_RESUME_BANK_UNCHECKED`, refuse. Fail closed.
- No stage after the refusal starts. The `on_exit` trap still pushes the final `chain.log` to HF.

**Why the PATH, and why the trainer's check stays.** The path is a proxy that needs one small file and no dataset load, so it can run before any spend. The authority on a bank's identity is the trainer's resume identity: `n_tuples`, `per_rank` and `targets_sha256` (`train.py:1036-1055`, `_identity_diff`). `train.py:993-996` deliberately does NOT compare paths ("a bank moved to another disk is the same bank"). The two fail differently:
- a bank moved with the same bytes is refused HERE (point `P6_CBANK` at the run's recorded path; the trainer then decides on content);
- a bank rebuilt IN PLACE under the same path passes here and is refused by the trainer's preflight, after lab / mem / egr / gate have spent. See limit 1.

## One extension beyond the brief (flagged)

The brief placed the check at 7b (after ARGS). I added a second call right after `recover`, before `inputs` / `commands` / `cache` / `lanes`. Reason: `recover` can restore `config.json` from HF onto a fresh pod, and on a pod with no stage markers `commands` / `cache` (about 24 CPU-h, ESTIMATED in the chain's own comment) / `lanes` would run on the wrong bank before 7b is reached. In the incident the markers survived, so those stages were no-ops and the 7b call alone would have caught it.

Measured in the sandbox (`resume_bank_sandbox_tests`):
- Call 1 alone refuses with zero `build_metric_cache` / `annotate_commands` / `command-table` calls.
- The contrast arm (only the 7b call) runs `annotate_commands` and `build_metric_cache build` first, and refuses only after them.
- The 7b call alone, on a pod whose markers are done, refuses after `ARGS sha` is logged and before lab.
- `P6_STOP_AFTER=recover` is now also a cheap bank check.

The two calls are independent in the tests, and a static property pins each call's position. Delete either call and `check_chain` goes RED. Say if you want call 1 removed; it is one line plus its mutants.

## Tests (CPU only, `CUDA_VISIBLE_DEVICES=-1`, venv `tanitad` Python 3.13.5; MEASURED, logs in `logs/`)

The brief's (a)-(d) are in `test_p6_chain.py` (not a new file, because the launch gate's G-SUITE lists `test_p6_chain.py` explicitly at `launch_gate.py:269-270`; a new file would not be in the suite).

- (a) mismatch refuses with the marker; no lab / mem / egr / launch_gate / preflight / train stub call, no stage file, no `PASS_*.json`; the first call is the one that refuses, so `ARGS` was never defined.
- (b) match proceeds as before: the whole chain, `ZZP6_DONE`, both calls read `ok`, gate and trainer run.
- (c) no `config.json`: zero `resume-bank` calls and no log line. This holds in the lanes run and in the existing full-chain run 4c (one assertion added there).
- (d) unparseable, empty, or no-`args.targets` config.json refuses with `ZZP6_RESUME_BANK_UNREADABLE`. The helper unit tests also cover binary, list, `targets: null`, blank, list-valued, `args` a string, a missing file and a directory (12 cases, each rc 4).

**Counts, same trees, before / after** (`ZZ...PASS n`; none dropped):

| test (names `p6_pod.sh` or scans it) | before | after |
|---|---|---|
| `test_p6_chain.py` | **351** | **432** (+81: 12 static mutants, 29 helper, 39 sandbox, 1 in 4c) |
| `test_p6_cpu_probe.py` | 111 | 111 |
| `test_p2r_chain.py` | 139 | 139 |
| `test_rls1_chain.py` | 522 | 522 |
| `test_rls1_queue.py` | 254 | 254 |
| `test_flock_rw.py` (scans every `refe/*.sh`) | 59 | 59 |

`test_p6_chain.py` takes about 9.3 min before and 10.7 min after on this busy box.

**Mutation checks of the new lines.** Each mutant is a copy of the real files, run through the guard tests; every one goes RED. Logs `logs/mutant1..4.log`.

| mutant | goes RED at |
|---|---|
| 1. delete the refuse line `echo "$msg"; refuse "$msg"` | static `RESUME_BANK_NEVER_REFUSES`; sandbox (a): rc 0, no refusal, lab / mem / egr / gate / train all ran |
| 2. flip the comparison in `p6_prep.resume_bank` (`==` to `!=`) | helper "same bank" assertion (rc 3 on an identical bank); sandbox (a): the mismatch is waved through |
| 3. the same-bank branch refuses (a MATCH refused) | sandbox (b): rc 1, refused on a matching bank, while (a) still passes |
| 4. unreadable config treated as fresh (`4) return 0`) | static `RESUME_BANK_REFUSAL_RETURNS`; sandbox (d): the chain ran on |

The suite also carries these mutants itself, so they are re-checked on every run of `test_p6_chain.py`: 12 static mutants (delete either call, move call 2 behind mem, delete the guard or the refuse line, mismatch only logs, unreadable or failed check returns, fresh-start line changed, wrong file or wrong bank), the flipped-comparison and unreadable-as-fresh mutants of `p6_prep.py`, and 4 behavioural script mutants (refuse deleted, branches swapped, unreadable returns, wrong `--bank`) that must break (a), (b) and (d).

## Limits (read these)

1. **Path, not content.** A bank rebuilt in place under the same path passes this guard. The trainer's `targets_sha256` catches it, after lab / mem / egr / gate have spent. A content check in the chain would compare `identity.targets_sha256` from `config.json` with `_digest_files` over `$CBANK/targets_rank*.jsonl` (`train.py:1036-1055`). Its cost on the 167,300-tuple bank is UNVERIFIED; I did not add it because the brief says the path.
2. **A run that has `ckpt_last.pt` but no `config.json`** is treated as fresh, as briefed. The trainer's preflight identity check still guards the resume.
3. **Stages before `recover`** (dd write test, env, weights hash) still run first, because `config.json` may only arrive from HF in `recover`.
4. **Not run on the pod and not against a real P6 `config.json`.** The structure comes from the writer's source (`train.py:1500-1511`) and the brief's description (INHERITED). The sandbox runs the REAL `p6_pod.sh` and the REAL `p6_prep.py` under bash with stubbed tools, as `test_p6_chain.py` always does. Before the next ship, on the A40, spot-check (read-only): `python -c "import json;print(json.load(open('/workspace/data/refe_runs/p6_paper_faithful/config.json'))['args']['targets'])"` should print the 3-C bank path.
5. **The real fix for the incident is on the pod:** `/workspace/ops/p6_start.env` needs `P6_CBANK=$DATA/refe_navtrain/train_grow_cmd_3c`. This guard makes a missing one loud and cheap instead of silent and 25 A40-minutes.
6. Test environment: the export of `refe/` alone was not enough for `test_p2r_chain` / `test_rls1_*`; for those the package's `code/ eval/ splits/ report/ raw/rl_pool raw/rl_fan raw/2026-10-15-rls1-adam-dose` and `Project Steering/MODEL_REGISTRY.md` were also exported from the same tip. Before and after run on identical trees.

## Proposed register rows (not edited; the Master Mind lands them)

- `TRAPS.md` (class: relaunch re-defaults a knob): *an env file that omits a knob silently re-defaults it. The chain compares its inputs with the run's own record (`config.json`) before spending. Measured 2026-10-20: P6 relaunched on the pre-3C bank (189,888 vs 167,300 tuples), ~25 A40-min spent and a PASS token minted for the wrong argv; only the trainer's preflight refused.* Fix: `p6_pod.sh` `resume_bank_check` (this package).
- `BACKLOG.md`: optional content check in the chain (limit 1), and the same guard for `p2r_pod.sh` / `rls1_pod.sh` / `sft_pod.sh` if they take a bank knob (UNVERIFIED which do).

## Manifest

| artifact | where | exists in |
|---|---|---|
| `refe/p6_pod.sh`, `refe/p6_prep.py`, `refe/test_p6_chain.py` (edited) | `C:/Users/Admin/par18/p6ident/out/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/` | staged copy; tested copy in `p6ident/repo/` is byte-identical |
| this `RESULT.md` and `logs/` (before / after test logs, 4 mutant logs) | `.../2026-09-20-refe-plan/raw/2026-10-20-p6-resume-bank/` under the same `out/` | staged only |
| `out/paths.json` | `C:/Users/Admin/par18/p6ident/out/paths.json` | one place |
| `HANDOFF.md`, `BRIEF.md` | `C:/Users/Admin/par18/p6ident/` | one place |
