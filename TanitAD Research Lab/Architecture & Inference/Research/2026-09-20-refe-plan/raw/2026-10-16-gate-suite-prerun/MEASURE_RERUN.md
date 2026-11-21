# Real-profile measurement: rerun by the Master Mind, and a correction to the savings table

**Master Mind, 2026-10-16 ~08:30Z.**

## 1. The agent's measurement never ran
`measure_p2rep_failed_key_in_tree/` is the agent's run, kept as a record. Every step is rc 1 in 0.1–0.3 s with `verdict None`.
- **Cause:** the harness put `TANITAD_GATE_KEY_FILE` inside `--pkg`.
- **The gate refused it:** `gate_token.GateError: refusing a key file inside the code tree`. That is the correct behaviour, because a key inside the tree would enter the tree digest.

## 2. The rerun (MEASURED; `measure_p2rep/`, dev box, profile `refe_p2rep`, tree = ship `22b4068`)
The key was a throwaway written OUTSIDE the tree (`--out` dir); it is not banked. `key_id` in the JSON is only its fingerprint.

| step | s | G-SUITE | path |
|---|---|---|---|
| `suite-prerun` | 237.8 | verdict **PASS**, evidence 2,058 B | — |
| `run --only G-SUITE --suite-evidence` | **2.9** | **PASS** | `accepted` (binding re-measure 0.4 s) |
| `run --only G-SUITE`, no flag (today's gate) | **201.3** | **PASS** | live (`suite_source null`) |

- **The accepted path and the live path agree (PASS / PASS).** The shortcut changes the time, not the verdict.
- Both runs exit rc 1 because `--only G-SUITE` leaves the other seven checks `MISSING` (`run_*/FAIL_22b406875ae8.json`). A partial run never mints a token, which is correct.

## 3. ⚠️ Correction: the ESTIMATED rows of `tables/savings_estimate.txt` are wrong, and the slow part is ONE script
The table scales dev-box seconds by a single ×6.03 ratio, anchored on P2-s1's 22-min G-SUITE. The pod's own G-SUITE runs say otherwise. All pod numbers below are MEASURED from pod pre-run or suite logs read 2026-10-16 (`/workspace/ops/*_gsuite_exact.json`, `p6suite.log`):

| profile | pod G-SUITE | the table's ESTIMATE |
|---|---|---|
| `refe_rls1c` (R0c) | **238.5 s** (rl_epo 74.6, launch_gate 82.1, rl_r0c 60.6, rl_s1b 20.3, rl_reward 0.8) | 39 min |
| `refe_aux_s1c` (S1c) | **479 s** (aux_drivable_head 118.4, aux_s1_wiring 112.8, s1c 104.0, launch_gate 75.8, rl_epo 67.2, rl_reward 0.8) | 137 min |
| `refe_p2rep` (P2-s1) | **1,218 s** pre-run (2026-10-15 05:33Z); **22 min** in the live gate (15:41→16:03Z) | 22 min (the anchor) |
| `refe_p6` (P6, 2026-10-14 suite run) | **~2,059 s**: launch_gate 59, **p2r_chain 1,290**, p6_levers 369, p6_chain 341 | 61 min |

- ⇒ **On the pod, `test_p2r_chain.py` alone takes 1,290 s. On the dev box it takes 5–14 s.**
- ⇒ The profiles that include it (P2, P2-s1, P6, field) are slow. The ones that do not (R0c, S1c) take 4–8 min.
- ⇒ The ×6.03 ratio is an artefact of the one profile it was anchored on.

My earlier HYPOTHESIS here was that the 22-min anchor ran beside P6's CPU stages. **It is REFUTED by the timeline:** P6's chain log shows its first stage at ~16:15Z, after that G-SUITE had ended (16:03Z), and the 1,218-s pre-run reproduces the slowness alone.
⇒ **The larger and cheaper lever is the cause of `test_p2r_chain.py`'s 1,290 s on the pod.** It is ~21 min per P2/P6-family launch, and fixing it needs no evidence-acceptance logic at all. This lever saves 4–8 min for R0c/S1c-type launches, and ~20–34 min where `test_p2r_chain` is in the suite.

## 4. Status
- The lever is landed **UNWIRED** (`WIRING.md`). No chain passes `--suite-evidence`.
- It was re-verified on the landing tree, meaning tip `3ac9f85` with the `test_rl_s1b` anchor fix plus these three files. The suites cover every `refe/test_*.py` that names `launch_gate` or `gate_token` (rule R39): `logs/landing_sweep/`.

## 5. The cause of `test_p2r_chain.py`'s 1,290 s on the pod, and its fix (MEASURED 2026-10-16)
- **Cause.** Section 3a2 (`token_ok`) mints a token with `GT.data_manifest(argv)` on the chain's REAL argv, then calls the gate's real `launch_gate.py verify` about 8 times.
  - On the pod, `/workspace/data/...` exists. So every mint and every verify fingerprints the real 413,184-file image listing and the train_grow bank, and sha256s the 1.29 GB init checkpoint, over NFS.
  - Observed live: the verify child sat in `D` state on `rpc_wait_bit_killable`, and the test process had read 1.3 GB.
  - On the dev box the paths are absent, the fingerprints read "missing" on both sides, and the test takes seconds.
- **Fix (test only; the gate is untouched).** The scenarios rewrite the data root to `/zz_absent_test_root/data/` on every host. An assert pins that the chain's argv still names `/workspace/data/`. The token LOGIC under test is unchanged; fingerprinting the real data is the gate's own job at launch.
- **A/B on the pod**, P6's tree copy (c1dc5d3), sequential runs, `logs/pod_p2r_ab/`:
  - as shipped: **1,226 s**, `ZZTEST_PASS 139`;
  - with the one-line fix: **6 s**, `ZZTEST_PASS 139`.
  - Outputs are identical except egress byte counts that carry a timestamp.
  - Dev box with the fix: 15 s, 139.
- **Reach.** The fix lands with this commit and reaches the next ship of a profile whose suite includes `test_p2r_chain.py` (P2, P2-s1, P6, field). P6 keeps c1dc5d3: a re-ship would rebuild stages it has already built.

## 6. Landing verification (rule R39, extended by R40)
Tree: tip `3ac9f85` plus the three gate files plus the two test fixes. Every `refe/test_*.py` naming `launch_gate`/`gate_token` was run in its FULL form (`logs/landing_sweep/`). All rc 0:

| suite | checks |
|---|---|
| launch_gate static | 645 |
| gate_suite_prerun | 144 |
| rl_s1b | 18 |
| gate_gpu_probes | 12 |
| p2r_chain | 139 |
| p2rep_chain | 353 |
| p6_chain | 341 |
| p6_levers | **145, 11/11 mutations RED** (after the R40 anchor fix) |
| aux_g12_bf16 | 7 |
| aux_s1_wiring | 20 |
| rl_r0c | 26 |
| rls1_chain | 512 |
| rls1_queue | 134 |
| s1aux_chain | 404, 0 skips |
| s1c | 242, 0 skips |

Two notes on the table:
- `rls1_queue` failed once under heavy contention, on a kill-monitor READY timing race. It passed alone.
- `s1c` first reported 220 with 2 skips, because this tree lacked the navtest list. It reported 242 once the list was added.
