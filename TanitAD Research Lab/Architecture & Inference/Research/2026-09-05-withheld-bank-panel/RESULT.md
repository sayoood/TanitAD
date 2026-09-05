# Withheld-bank panel on the v7-tiny rig (H-EGO-LIT-4) — RESULT

status: PANEL RUNNING (2026-09-05 ~10:20 Europe/Berlin) — the six arms are executing SEQUENTIALLY on the dev-box RTX 4060 under `C:\Users\Admin\run_wbank\panel.sh`; A0_fixed in flight (~0.40 s/step ⇒ ~13.5 min/arm, ~80 min for six). Done before this: the `--withheld-bank {fixed,pred,random,none}` switch (`refc.py::roll_bank` / `refc_v3.py` / `refc_v3_train.py`, commit `79586fc`), `stack/tests/test_withheld_bank.py` 12/12, the scorer + table renderer + launcher (`924deaf`), SPEC.md, and the trainer smoke on the epcache. The scoring→table path is validated END-TO-END on the banked smoke report (`mk_tables.py smoke_panel_report.json` exit 0, 16,789 bytes). Next: A1..A5, then `panel_score.py 40 640`, then `mk_tables.py`, then the verdict + `GOALS_AND_CLAIMS.md`.

Owner: Architecture & Inference FlyWheel agent (Claude Opus 5), resumed 2026-09-05 after the predecessor died at a model rate limit while waiting for the GPU.
Pre-registration: `H-EGO-LIT-4` in `Project Steering/GOALS_AND_CLAIMS.md` (both outcomes committed there before this panel ran); reading rules in `SPEC.md`.
Compute: dev-box RTX 4060 only, one arm at a time, `OMP_NUM_THREADS=6`. The live run `refcv4b-b1-v72-40k` on pod `tanitad-refcv3` is untouched; Thor is not used.

## Scope stamp (binding, repeated on every number below)

**RIG scope** — v7-tiny rung, ~19 M params, **48 non-parity training episodes**, 2,000 steps, one seed (0). The corpus `physicalai-train-14231cd29c74` references **no registered parity key**, so nothing here is cross-arm comparable with the parity arms and **nothing here may enter `MODEL_REGISTRY.md`** (H-SCALE-2). This panel validates a **DESIGN** and a **GATE**, never a model claim.

## Arms

| arm | one variable vs A0 | status |
|---|---|---|
| A0_fixed | `--withheld-bank fixed` (today's 10 m/s roll) — the control | RUNNING |
| A1_pred | `--withheld-bank pred` (withheld rows at the model's own DETACHED `g_tac` 2 s speed, after step N) | queued |
| A2_random | `--withheld-bank random` (random-marginal CONTROL — right distribution, zero information) | queued |
| A3_drop25 | `--ego-dropout 0.25` (bank fixed) | queued |
| A4_none | `--withheld-bank none` (speed-blind vocabulary, the field's default) | queued |
| A5_regress | `--ablate-frames` — the DELIBERATE REGRESSION; an echo by construction, never a model | queued |

**Warm-up N (pre-registered reading rule, MEASURED from A0's own log):** the first logged step at which the 5-row running mean of `withheld_speed_mae` falls below 2.5 m/s. A0's log reads **N = 450** (`raw/arms/WARMUP_N.txt`); A1 and A2 SHARE it, so A1-vs-A2 isolates one variable — the speed SOURCE.

## Ops findings from the resume (MEASURED, worth carrying forward)

1. ⛔ **The GPU-wait launcher could never have fired.** `wait_and_launch.sh` counted busy jobs with
   `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'refav1_arm.py|refcv3_arm.py|t1_eval.py' }`
   — a filter whose own PowerShell command line **contains the pattern**, so the query matched **itself**. `procs` read a constant **4** for over an hour (including while GPU utilisation fell to 5 %), of which: the querying `powershell.exe` (self-match), two shell wrappers, and an `ssh.exe` to **Thor** (not a local GPU consumer at all). `n` could never reach 0, so the launch condition was unreachable. This is the documented self-match trap (`CLAUDE.md`) in a `Win32_Process` costume: the fix is to make the emitted token disjoint from the searched token, or — as here — to enumerate processes **without** filtering on the pattern and read them.
2. ⚠️ **`wc -c < file` is a METADATA probe, not a content probe.** During a G: outage it returned the correct 802,715 bytes for a file whose every content read failed — GNU `wc -c` takes the size from `fstat()` without reading. Using it as the mount's health control produced a "the mount is up for the shell but down for Python" reading that was **wrong**. The true content probe (`cat`/`head -c`) failed on target **and** control 15/15, which is what justified the Drive-client restart under the PI's standing authorisation; the mount returned on the first poll after it. *Same family as `df` on a pod and `free` on Thor: a probe reporting the wrong scope, read as an answer.*

## Numbers

None yet. Every number in this file will carry its evidence class, its T-tier, its arm and its artifact path under `raw/<arm>/`. ⛔ ADE is one row of the four families and never the gate (`H-ECHO-4`).
