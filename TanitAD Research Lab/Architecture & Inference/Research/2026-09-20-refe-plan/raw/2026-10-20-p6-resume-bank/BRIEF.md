# BRIEF: P6 chain, refuse a RESUME whose bank differs from the run's own, BEFORE any stage spends (Master Mind, 2026-10-20 ~07:35 Berlin)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/p6ident/`. CPU only (`CUDA_VISIBLE_DEVICES=-1`). TEMP / TMP / pytest basetemp under your work dir; delete only what you create.
- Never ssh to the pod; never commit or push. The pod runs ship `bcc04b5` and must not change: this fix is for the NEXT ship.

## Why (MEASURED 2026-10-20 on the A40, by the MM)
P6 (`p6_paper_faithful`) trained on the 3-C bank: `P6_CBANK=$DATA/refe_navtrain/train_grow_cmd_3c`, 167,300 tuples, ARGS sha `570619873b1ebaa0`. After a pod reboot the MM relaunched from `/workspace/ops/p6_start.env`, which lacked `P6_CBANK`, so `refe/p6_pod.sh:33` defaulted CBANK to the pre-3C bank (189,888 tuples, ARGS `6a13af85…`).
The chain then re-ran G-LIVE-MEM, G-EGR and the launch gate on the wrong bank (~25 min of A40), RE-MINTED `lg/PASS_bcc04b52233e.json` and its evidence files for the wrong argv, and only the trainer's PREFLIGHT refused: `REFUSING TO RESUME -- the run's identity changed: n_tuples 167300 -> 189888; targets_sha256 '8775e73ee948' -> 'aab23357d77b'`.
The run's `$RUN/config.json` records `args` (incl. the targets path), `argv`, and `identity` {n_tuples, per_rank, targets_sha256, ...}.

## Start
Tip: `git -C C:/Users/Admin/tanitad-push fetch -q origin agent/arch-inf-20260803`, pin `FETCH_HEAD` once. Export `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/` (and whatever its p6 tests import; keep the repo layout). Read `refe/p6_pod.sh` lines 1-120 and 390-530, and how `refe/test_p6_chain.py` stubs a chain run.

## Do
1. In `refe/p6_pod.sh`, immediately after ARGS / ARGS_SHA are defined (7b) and BEFORE the mem stage (`key_done mem`): if `$RUN/config.json` exists (a resume), compare the `--targets` the run was STARTED with (read it from config.json, `args.targets` or the `--targets` value in `argv`; choose the one the trainer actually writes, and cite `file:line` in train.py that writes it) with `$CBANK`. Different => `refuse` with a named marker `ZZP6_RESUME_BANK_MISMATCH run=<path> chain=<path>` and NO stage after it may start (no mem, egr, launch_gate, preflight, train). Equal, or no config.json => unchanged behaviour. Use the chain's Python (`$PY`), not `jq`. A config.json that cannot be parsed => refuse (do not treat as "fresh").
   - Note in a comment why the check compares the PATH and why the trainer's own preflight identity check (targets_sha256) remains the content check.
2. Tests in `refe/test_p6_chain.py` (or a new `refe/test_p6_resume_bank.py` if cleaner): (a) mismatch refuses with the marker and NO mem/egr/launch_gate/preflight/train marker or stub call; (b) match proceeds as before; (c) no config.json proceeds; (d) unparseable config.json refuses. Mutation-check the new lines (delete the refuse line => (a) must go RED; flip the comparison => (b) RED).
3. Run the FULL `test_p6_chain.py` before and after (baseline count from the tip; none may drop), plus any test that names `p6_pod.sh` (grep `refe/test_*.py`). Report counts.
4. Manifest `p6ident/out/paths.json`: `[repo path, abs src, "raw"]`, `"base=<tip blob>"` on every edited existing file. Short `RESULT.md` + logs under `.../2026-09-20-refe-plan/raw/2026-10-20-p6-resume-bank/`. `HANDOFF.md` (<= 600 tokens).

## Hand-back (<= 150 words)
`file:line` of the guard and of what train.py writes, the test counts before/after, mutation result, manifest path.
