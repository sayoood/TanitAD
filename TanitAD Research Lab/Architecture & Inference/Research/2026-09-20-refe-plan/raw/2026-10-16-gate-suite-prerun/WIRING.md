# WIRING.md: the 3-line chain change (NOT applied; no `*_pod.sh` is touched by this package)

Chains on the pod are SHIPPED trees (`SHIP_MANIFEST.json` binds their bytes), so wiring is a separate Master Mind step: edit the chain, re-ship, re-land.
Nothing in this package calls `suite-prerun` or passes `--suite-evidence` (checked: `grep -l "suite-prerun\|suite-evidence" refe/*.sh` is empty); `launch_gate.py run` WITHOUT the flag is exactly today's gate.

## The change, shown on `refe/p2rep_pod.sh` (P2-s1)

`cd "$PKG/refe"` is done at line 142, before the WAIT stage (line 157), and `export REFE_BACKBONE_ROOT=...` at line 141, so the calls are relative like the existing `launch_gate.py run` and see the same env.

```sh
# (1) in stage 1, BEFORE `wait_clear` (line 157): start the pre-run beside the WAIT, lowest CPU priority, GPU hidden, the SAME env the gate's `run` gets
( exec nice -n 19 env -u HF_TOKEN OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= "$PY" launch_gate.py suite-prerun --profile refe_p2rep --pkg "$PKG" --out "$R/suite_prerun.json" > "$R/suite_prerun.log" 2>&1 200>&- ) & SPID=$!

# (2) in stage 5, immediately BEFORE the `launch_gate.py run` line (203): let a pre-run that is still going finish (G-SUITE is the LAST of the gate's checks; it is rarely still going)
[ -n "${SPID:-}" ] && { wait "$SPID"; log "ZZP2REP_SUITE_PRERUN rc=$? $(grep -o 'ZZSUITE-PRERUN-[A-Z]*' "$R/suite_prerun.log" | tail -n 1)"; }

# (3) on the `launch_gate.py run` line (203-204): one more argument, nothing else changes
    --gate-sets "$R/gate_sets" --gate-pdm "$R/gate_pdm" --gate-bank "$R/gate_bank" --p2-gate-json "$R/gate/gate.json" --suite-evidence "$R/suite_prerun.json" \
```

A missing, failed, stale or mismatching pre-run is NOT an error for the chain: the gate logs `G-SUITE: pre-run evidence REJECTED (...) -> running the suite LIVE`, runs it exactly as today and records the reasons in the token (`checks.G-SUITE.suite_source`). The chain needs no new refusal path. `ZZSUITE-PRERUN-FAILZZ` (exit 1) means a suite script FAILED in the pre-run: the live run will fail the same way, so read `suite_prerun.log` early; that is a bonus of starting it in the WAIT.

## ⚠ Per-chain: WHERE the pre-run may start (a variable the chain exports AFTER the pre-run started makes the evidence mismatch; MEASURED in the chains' source)
| chain | exports the gate's `run` sees that the suite env binds | earliest safe start of (1) |
|---|---|---|
| `p2rep_pod.sh` | `REFE_BACKBONE_ROOT` (line 141, before the WAIT) | stage 1, as shown |
| `s1aux_pod.sh` | `REFE_BACKBONE_ROOT`, `AUX_LABEL_STORE`, `AUX_NAVTEST` (220); `NUPLAN_DATA_ROOT`, `REFE_NUPLAN_DB_ROOT`, `REFE_NAVTRAIN_YAML` (291) | after line 291 |
| `p2r_pod.sh` | `REFE_BACKBONE_ROOT` (97) AND `NUPLAN_DATA_ROOT`, `REFE_NUPLAN_DB_ROOT`, `REFE_NAVTRAIN_YAML` (175, a later stage) | after line 175 (the data stage), or repeat the three exports before (1) |
| `p6_pod.sh` | `REFE_BACKBONE_ROOT` (193), `NUPLAN_*` / `REFE_NUPLAN_DB_ROOT` / `REFE_NAVTRAIN_YAML` (304, 325) | after line 325 |
| `rls1_pod.sh` | `REFE_BACKBONE_ROOT` (262), `NUPLAN_DATA_ROOT`, `REFE_NUPLAN_DB_ROOT`, `REFE_NAVTRAIN_YAML` (378) | after line 378 |

Starting too early is not unsafe, only wasted: the gate sees `environment differs from the pre-run's`, runs the suite live, and says so in the token.

## What the chain MUST keep equal between (1) and the gate's G-SUITE step (a mismatch costs the 22 min, never strength)
| bound | how the chain keeps it equal |
|---|---|
| the shipped tree | no re-ship between the two (the chain's stage-2 verify re-checks it anyway) |
| the profile name | the same `--profile` |
| the commit | the same `SHIP_MANIFEST.json` in `$PKG` (the pre-run reads the commit from it, like `run`) |
| gate code (`launch_gate.py`, `gate_checks.py`, `gate_token.py`) | same tree |
| the interpreter and its installed distributions | the same `$PY`; ⛔ no `uv pip install` between them (TRAPS.md: it can silently replace torch; here it also voids the evidence, which is the correct outcome) |
| env vars `REFE_* AUX_* NUPLAN_* P2REP_* P6CPU_* S1C_* DZ_* TANITAD_*`, `PYTHONPATH`, `PYTHONHASHSEED`, `PKG` (value HASHES; a name containing TOKEN/KEY/SECRET/PASSWORD is never read; `TANITAD_GATE_KEY_FILE`, `TANITAD_GATE_FP_THREADS`, `TANITAD_GATE_ARM`, `REFE_GATE_NESTED` are not bound) | both calls are children of the same chain shell, so what the chain exports is seen by both; do not set one on only one of the two calls |
| the key | `TANITAD_GATE_KEY_FILE` is exported at line 48, before both; the pre-run creates the key if the pod has none, as `run` would |
| NOT bound, deliberately | `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, `HF_TOKEN`. The pre-run MUST hide the GPU (`CUDA_VISIBLE_DEVICES=`) because stage 1 happens while another arm trains on it, and the live `run` sees it. Every suite script rewrites `dev = "cuda"` to CPU itself (`test_rl_epo.py:368`, `test_aux_s1_wiring.py:162`; the only real-GPU test, `test_aux_g12_bf16.py`, is in no suite), and a script that skipped for want of a GPU makes the evidence FAIL (a skip is not a pass) |
| age | at most 12 h between (1) finishing and the gate's G-SUITE step |

## ⚠ Escalation: the pre-run adds CPU and RAM load to a pod that is TRAINING

CLAUDE.md invariant: *"Never add GPU/RAM load to a pod that is training."* In stage 1 the pod is by definition still busy (RL-S1 / S1 hold their locks and the GPU). The pre-run adds **no GPU load** (the GPU is hidden; no suite script needs it), but it adds **CPU and RAM**. `nice -n 19` protects CPU priority only. The measured peak RSS, thread count and CPU seconds of the real suite on the dev box are in tables/measure_p2rep.txt; compare them with the pod's free RAM BEFORE wiring, and note TRAPS.md's *torch spawns ~113 threads per process* (hence `OMP_NUM_THREADS=4`, the value the live gate run uses). If the pod has no RAM headroom, start (1) at the END of the WAIT (after the upstream chain logs its DONE marker) instead of at its start: the saving is then the gate's G-SUITE minus the time between that marker and the GPU being idle, but no load lands on a training pod. That choice is the Master Mind's / PI's.

## Order of the gate's own checks (why a pre-run started in WAIT is almost always finished in time)
`run_checks` runs G-HYG, G-DVB, G-LIVE, G-CLOCK, G-EVAL, G-CKPT, **then G-SUITE**. The pre-run has until G-CKPT is over (about 18 of the 40 idle minutes on P2-s1, 2026-10-15), not only until `run` starts.
