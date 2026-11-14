# Detach probe: `DETACHED_PROCESS` kills the read chain's console children (MM, 2026-11-15)

## Why
- **The rehearsal failure.** The P7 read-route rehearsal (P6 checkpoint, `--only families`) launched with `p7_read_chain.py --detach` at 2026-11-14 07:14Z.
  - Both transparent-scorer shards (`transp_s0`, `transp_s1`) exited at once with rc **3221225786 (0xC000013A, CTRL_C / console close)**.
  - Both logs are 0 bytes; the result is `families.FAILED`.
  - The detached chain itself survived and logged the failure.
- **The flags.** `--detach` used `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | DETACHED_PROCESS`.
- **Known context** (MM memory, 2026-11-13). Since ~18:22Z that day, a new console on this box is handed to Windows Terminal, and a plain WMI `cmd /c` dies within seconds unless it is started with `CREATE_NO_WINDOW`.

## Probe (`probe.py`, this folder)
- **Setup.** One launcher starts two parents and exits at once:
  - A: the `--detach` flags as they were;
  - B: the same flags without `DETACHED_PROCESS`.
- **Each parent** starts a child exactly as `step_families` does (`Popen`, no creation flags). The child sleeps 20 s, and the parent records its return code.
- **Interpreter:** `C:/Users/Admin/venvs/tanitad`, the read chain's.

| mode | flags | result (MEASURED, 2026-11-14 23:16Z = local 2026-11-15 00:16) |
|---|---|---|
| A | `NEW_PROCESS_GROUP \| NO_WINDOW \| DETACHED_PROCESS` | the child started (its log was created), then the child **and** the parent vanished: no result file, no surviving process after 65 s |
| B | `NEW_PROCESS_GROUP \| NO_WINDOW` | the child ran its 20 s and returned **rc 0**; the parent wrote `result_B_nowindow.json`. Both outlived the launching shell |

**Reading.**
- A parent with no console (`DETACHED_PROCESS`) makes every console child open its own console, and that is what dies.
- A hidden console inherited from the parent (B) does not.
- The probe settles mode A vs B on this box. The Windows Terminal mechanism behind it is INHERITED (the memory note), not re-measured here.

## Fix
- **`eval/p7_read_chain.py`.** A new `detach_flags()` returns `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`, without `DETACHED_PROCESS`, and `--detach` uses it.
- **`eval/test_p7_read_chain.py`.** A new check, C10b, asserts both flags are set and `DETACHED_PROCESS` is not. Mutation arm 6 restores `DETACHED_PROCESS`, and C10b must catch it (M1).
- **Tests:** `ZZP7CHAIN PASS 17/17` on a tip `f6cb5c80` export of `eval/` (the baseline before the fix was 16/16).
- **Consequence for the real P7 read** (P7 ends ~2026-11-16 06:15Z, ESTIMATED): launch it from a tree that carries this fix, or start the chain itself via WMI with CreateFlags `0x08000200`.
