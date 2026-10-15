# Master Mind re-verification of the R0c build (2026-10-15)

The agent's counts were re-run, not inherited.
- **Tree:** a fresh copy of the tip `066ca5a`'s code tree, plus the test fixtures. 066ca5a is d257647 plus the `test_p2rep_chain.py` env-isolation fix.
- **Patches:** `PATCH_1..7` applied in order with `git apply` (each checked first with `--check`), and `test_rl_r0c.py` added.
- **Script:** `r0c_verify.sh` (the Master Mind's scratchpad). Per-suite output: `mm_verify_counts.log`.

| suite | torch | result | agent's claim |
|---|---|---|---|
| `test_rl_r0c.py` | 2.11 | ZZTEST_PASS **26** | 26 |
| `test_rl_r0c.py` | 2.7.1 (the pod's build) | ZZTEST_PASS **26** | 26 |
| `test_rl_s1b.py` | 2.11 / 2.7.1 | **18 / 18** | 18 |
| `test_rl_epo.py` | 2.11 | **42** | 42 |
| `test_launch_gate.py --static` | 2.11 | ZZLG_STATIC_PASS **659** | 659 |
| `test_aux_s1_wiring.py` | 2.11 | **20** | 20 |
| `test_aux_g12_bf16.py` (CUDA, RTX 4060) | 2.11 / 2.7.1 with `CUBLAS_WORKSPACE_CONFIG=:4096:8` | **7 / 7** | 7 |
| `test_rls1_chain.py` | 2.11 | **392** | 392 |
| `test_s1aux_chain.py` (not in the agent's list) | 2.11 | **385** | (unpatched tip: 385) |
| `test_p2rep_chain.py` (not in the agent's list) | 2.11 | **353** | (unpatched tip: 353) |
| `test_p6_chain.py` (not in the agent's list) | 2.11 | ZZP6CHAIN_PASS **341** | (unpatched tip: 341) |

**What the last three show:** `launch_gate.py` and `gate_checks.py` sit in the S1, P2-s1 and P6 gates too, and the patches leave all three chains' suites unchanged.

**Status:**
- **Not applied to `refe/` on purpose.** Running R0c is a PI decision (about 3.5 A40-h, beyond RL-S1b's authorised budget). Keeping the patches out also keeps S1's re-ship tree minimal while its H arms train. The patches apply cleanly on 066ca5a.
- **Not yet built:** the pod chain wiring (`rls1_pod.sh`, `test_rls1_chain.py`). It is in build as `chain/` in this package.
- **Not yet verified:** R0c's bf16/GPU path on the A40.
