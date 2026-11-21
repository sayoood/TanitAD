# HANDOFF — SAFEHEAD T-horizon arms still running after the agent's budget (2026-10-16)

**State.** Prereg `PREREG_SAFEHEAD.md` (sha256 `2103a1a3…30ca`, verified) + addendum A1 (`ffcfbc1e…773b`). Identity controls ALL PASS (`controls.json`), A1 lever reach PASS (`ttcpen_reach.json`).
Two detached queues (WMI-launched, below-normal priority, ONE python run at a time, 6 threads, ~3.7 GB RAM each):
- `run_queue.sh`: banked P controls -> P K=50 SH_SG, SH_F, SH_A -> T K=250 SH_SG, SH_A, SH_F. Final line `ZZSH_QUEUE_DONE`.
- `run_queue2.sh` (A1): waits for `ZZSH_QUEUE_DONE`, then FULL+TTCPEN0.25 at P K=50 and T K=250. Final line `ZZSH_QUEUE2_DONE`.
Progress: `logs/queue.txt` (one line per finished arm, with rc and seconds); per-arm logs `logs/sh_<regime>_<design>_K<K>.log`. Cost: ~18 s/update per scorer at 6 threads (SH_A trains two scorers: ~2x). T arms ~75 min (SH_SG, SH_F), ~150 min (SH_A).

**⚠ Value of what is still running (decided 20:00Z): the validity row read V-NO, so every SH row is already SH-UNTESTABLE-TTC and A2 read A2-PROGRESS-ROWS (`RESULT.md` s.1). The T arms and A1 can only add f at T; none can change a row.** The MM may stop them: kill by explicit PID — queue-1 bash 32048 / 27864 / 34780 and its python child (SH_SG T: 38184 / 20424; later arms get new PIDs, look them up), queue-2 bash 31272 / 35468.

**How to check.** `tail logs/queue.txt`. A python process is alive if `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` shows `sh_closed.py` / `sh_ttcpen.py`.

**Exact next step when `ZZSH_QUEUE_DONE` lands.**
1. `cd <this package>; PYTHONUTF8=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe sh_summarize.py` -> `summary.json` + printed rows. It applies PREREG_SAFEHEAD.md s.8 mechanically (V first, then per design; a missing T arm reads PROVISIONAL, never GO).
2. Copy the T table and the final rows into `RESULT.md` s.4-5 (replace the PROVISIONAL rows; do not change the P rows).
3. When `ZZSH_QUEUE2_DONE` lands: read `out/sh_s0_{P,T}_FULL_sepon_lr1_lam1_K{50,250}_ttcpen0.25.pkl` against FULL lam 0 with `sh_summarize.read_arm` (view `own`), and apply A1's rows (`PREREG_SAFEHEAD_A1_TTCPEN.md`).
4. Hand the MM the rows. Do NOT launch an A40 arm (PI decision); the trainer change is `SPEC_SAFETY_HEAD_FLAG.md`.

**Stopping the queues** (if the MM needs the CPU): kill by explicit PID (never `pkill -f`): the bash PIDs are in `logs/queue.txt` ("queue start pid", "queue2 armed pid" are Git-bash PIDs; find the Windows PIDs of `bash.exe` with `run_queue` in the command line, then the python child).
