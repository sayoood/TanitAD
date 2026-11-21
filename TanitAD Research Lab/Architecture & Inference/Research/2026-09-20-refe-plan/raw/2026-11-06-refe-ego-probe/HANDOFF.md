# HANDOFF -- REFe ego-input reliance probe (T0)   written 2026-11-06

**Deliverable:** answer "is REFe's closed-loop crawl a POLICY property or an artefact of OOD sim inputs?" on REAL nuPlan frames. Pre-registered: `PREREG.md` (+ Addendum A1). Nothing is committed/staged: MM lands.

## State
* Instrument built + smoked (8 tokens / 4 logs, run twice): `refe_ego_probe.py`, `make_subset.py`, `analyze_ego_probe.py`. Smoke controls: C1 determinism 8/8 bit-identical, C2 `table.npz` identity 8/8 bit-identical logits + picks, executed-proposal max |d| 0.0 m.
* FULL RUN (all 1,058 C3 tokens / 123 logs, 8 arms, A0r determinism replicate on every 4th token): LAUNCHED detached via WMI 2026-11-06 21:44:01 Berlin (cmd.exe PID 572 -> python PID 30964), holds the shared GPU lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` (owner `refe_ego_probe`). ETA ~64 min from first log (~3.7 s/token, falling) => expect `full/DONE` about 22:50-23:00 Berlin. Command line: `launch_cmd.txt`.
* Everything lives in `E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/` (run dir `full/`). The package (PK) was only read.

## How to check (no model turns)
`cat full/STATUS.json` (stage, done, eta_min) | `tail full/run.log` | marker: `full/DONE` (success; JSON with controls_pass) or `full/FAILED` (error text). A kill -9 / power loss writes no marker: then STATUS.json `updated` is stale and the pid is dead.
If it died mid-run: relaunch with `--resume` (keeps `full/partial.pkl`, per-log checkpoint) after deleting the marker; same command line as in `full/stdout.log` first line / `launch_cmd.txt`.

## When DONE: the analysis (CPU, ~1 min)
```
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I analyze_ego_probe.py --run full --out full/analysis --boot 2000
```
Writes `full/analysis/READOUT.md` (all tables + the mechanically evaluated COMMITTED DECISION R1/R2/R3/R4) and `readout.json`. Read the controls line FIRST (C1/C2/C3/C4/C5); if `controls_pass` is false the verdict is provisional.
Then write `RESULT.md` here: verdict by the committed rule, b_A0 / AX0-A0 / GREY-A0 / DEAD-A0 with CIs, ax gain, selection-vs-generation, strata split, limits (see PREREG "Limits"), tier T0 stamp, estimator named (paired log-cluster bootstrap). Do not move a threshold after seeing the data.

## Smoke numbers (8 tokens: NOT evidence, plumbing only; CIs from n=8/4 logs)
`smoke2/` (8 tokens, 4 logs, 4 CRUISE; identical to `smoke/` bit for bit; `smoke/` kept as evidence of the C3-check float32 bug, since fixed). controls_pass TRUE: C1 4/4 bit-identical (A0r vs A0 traj, 64 proposals, logits, pick, trunk output), C2 8/8 logits bit-identical to `table.npz` + picks equal + executed 8-pose proposals max |d| 0.0 m, C3 8/8 on every arm, grey-memo fresh checks 2/2 bit-identical, real ego slots 4,5 non-zero on 0/8 tokens (the known wiring defect).
Per-arm vs A0 on all 8 (ADE between picks m | dv@4s m/s): AX0 1.08 | -0.60 ; AXP 1.13 | +0.07 ; AXM 1.30 | -0.97 ; DEAD 0.020 | +0.007 ; GREY 1.93 | -0.91 ; SWAP 2.20 | -1.11. CRUISE b_A0 (plan v4 - GT v4) -0.73 [-1.17,-0.29] m/s on n=4 tokens/2 logs (NOT a result). ax gain (v4 under +0.5 minus -0.5) +0.76 [0.62,0.90] m/s per m/s^2 on CRUISE n=4. Smoke only says: DEAD barely moves the plan (2 cm), ax and images move it by metres; the verdict needs the full run.

## Integration (for MM)
* Result goes in `GOALS_AND_CLAIMS.md` only after RESULT.md exists. T0 diagnostic; not a driving claim.
* If R1/R2 hold: the lever is on the policy side (ego-token reliance / slow bias); if R3 holds: look at sim-side ego slots (yaw_rate, steer, ax) and image domain.
* Env note: D: is absent; every path here is E:/ (repo, data) or F:/ (navsim archive, `TANITAD_NAVSIM_ROOT` resolver). `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` is the shared GPU lock (owner `refe_ego_probe`).
