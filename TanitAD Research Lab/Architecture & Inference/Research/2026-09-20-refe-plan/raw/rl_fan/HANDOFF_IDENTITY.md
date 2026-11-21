# HANDOFF -- RL-S1 frozen-fan identity evidence (2026-10-12, dev box GPU)

**State.** `refe/rl_fan_dump.py` (+ its suite, ZZTEST_PASS 28), `refe/rl_pool_available.py`, `--dump-scores`, the coverage evidence and the consumer proof are built and staged
(see the dated section at the end of `eval/HANDOFF_RL_S1.md`). The ONE open item is the GPU evidence, because the dev-box GPU is held by another agent's
`ladder_eval_diag.py` (~7.7 GB; my model load OOMs). I did not kill it.

**What is waiting.** A background waiter (my processes, started 2026-10-12 ~15:10 local) blocks until no other GPU job process exists and the GPU is under 700 MiB for
4 consecutive 15-second polls, then runs the instrument:

    D:/Projects/TanitAD-artifacts/_rlfan_work/wait_and_run_identity.sh   ->   eval/rl_fan_identity.py --work D:/Projects/TanitAD-artifacts/_rlfan_work
                                                                              --py C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe --old-keys 500
    log:     D:/Projects/TanitAD-artifacts/_rlfan_work/identity_run.log     (ends with ZZIDENTITY_FINISHED rc=<n> when done)
    output:  D:/Projects/TanitAD-artifacts/_rlfan_work/identity/identity_devbox.json   (+ new_*.log, old_dump.log)

Check it with: `tail -5 .../identity_run.log` (empty = still waiting) and `nvidia-smi --query-gpu=memory.used --format=csv,noheader`.
The waiter may not survive the end of the agent session that started it; if `identity_run.log` is empty AND no `wait_idle.py` / `wait_and_run_identity.sh` process exists, start it again:
`nohup bash D:/Projects/TanitAD-artifacts/_rlfan_work/wait_and_run_identity.sh &` (it only waits, then runs; it never touches a busy GPU).

**What the instrument measures (read `eval/rl_fan_identity.py` docstring).**
* A = new dumper vs the OLD dumper (`onpolicy_dump.py` on the first 500 keys; same GPU, same bank): REQUIRED EXACT.
* C = invariance across batch 1 / 2 / 4, shifted batch composition and `--workers 2`: REQUIRED EXACT.
* B = new and old vs the banked A40 step-10,075 sets (cross-hardware: RTX 4060 vs A40): MEASURED, reported with magnitude. A mismatch is a DEFECT, never a footnote.
Marker on stdout: `ZZFANID_A_AND_C_EXACT` or `ZZFANID_A_OR_C_NOT_EXACT`.

**Exact next step (a fresh Sonnet agent, ~15 minutes of work):**
1. Confirm `ZZIDENTITY_FINISHED` in `identity_run.log`; copy `identity/identity_devbox.json` to `raw/rl_fan/identity_devbox.json`.
2. Re-run the consumer proof on GENUINELY dumped records: `python eval/rl_fan_labeller_proof.py --sets <identity/new_full/...> --work <scratch on D:> --n 6 --run-labellers --out raw/rl_fan/labeller_proof_dumped.json`
   (the navsim venv is used inside the script; set `TMP`/`TEMP` to a D: dir, C: is full).
3. Report dev-box s/set (from `identity_devbox.json`) and peak GPU; put the two numbers into the dated section of `eval/HANDOFF_RL_S1.md` (section 4 table, rows "PENDING -- GPU").
4. If A or C is not exact: stop, name the first differing key and the max abs difference, and diagnose (candidates: `torch.set_num_threads`, TF32 flag order, `calib` argument, batch composition) -- do NOT relax the requirement.
5. Stage with exact paths (`git add <path>`, then `git diff --cached --name-only`); never commit, never push; never ssh to a pod.
