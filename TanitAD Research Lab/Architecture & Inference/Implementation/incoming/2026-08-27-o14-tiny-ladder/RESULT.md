# RESULT — the O14 tiny ladder (5 arms, 2k) and what it could and could not read

**Measured** 2026-08-27 · MEASURED (ours; Thor training, dev-box reads) ·
T0-DIAGNOSTIC · prereg `PREREG_O14_FUTURE_OBS.md` + AMENDMENTS A1/B · raws here.

## The headline finding is about the INSTRUMENT'S SCALE, not the objective

⭐ **The pixel-marginal the absorption test targets does not exist at 2k steps.**
Base (w=0): −0.0033 (t −0.59); w=0.1: −0.0045; w=1.0: −0.0042 — all null, no dose
effect. The E-DEC-63 baseline (+0.0096, t 5.11) is a **30k** phenomenon: the
encoder's displacement of pixel-predictive content BUILDS with training. Recorded
as Amendment B BEFORE further reads; the absorption primary moved to `o14fut30k`.

## Gates (vs base, per Amendment B) — PASSED

| arm | drift | nrmse | cos |
|---|---|---|---|
| base (w=0) | 0.4531 | 0.9876 | 0.183 |
| fut 0.1 | 0.4576 | 0.9886 | 0.184 |
| fut 1.0 | 0.4733 | **0.9746** | 0.246 |
| rec 1.0 | 0.4704 | 0.9769 | 0.233 |
| **DR (shuffled)** | **0.4533** | 0.9895 | 0.179 |

Worst drift +4.5 % relative (~3× seed band); the DR arm exactly at base (correct
inertness); fut/rec 1.0 marginally better nrmse and visibly higher cos (0.24 vs
0.18 — the aux does SOMETHING at 2k, in the right direction, below decision grade).
⚠️ Scale fact stated: at 2k every arm reads at the mean-predictor floor; the gate
is vs-base relative only.

## Defects found and fixed by the ladder itself (each test-pinned)

1. `o14_head.` lacked an S-W **introduction permission** — arm 2 refused its init
   (`STAGE_MAY_INTRODUCE`). 2. The **TaskStop-orphan respawner**: a stopped read
   chain's inner bash survived and double-wrote probe artifacts; killed
   parent-first, poisoned artifacts purged, probes re-run clean. 3. The rig's
   drift-validity band is 30k-calibrated; 2k arms sit ~0.45 with clean controls.

## State

`o14fut30k` (the matched 30k pair, ONE variable vs `postrain30k`) is RUNNING;
its absorption read against +0.0096 decides R2's place in the v7r recipe.
