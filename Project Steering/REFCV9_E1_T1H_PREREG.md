# T1h / T1hs — pre-registration (frozen 2026-11-05 10:20Z, before any T1h data)

Written by the Master Mind before T1h or T1hs has trained, been gated or been scored. The confirmatory zero-training check
(`e1/t1_warp_probe/t1h_confirm/`) is still running at the time of writing; its outcome decides only whether the arms are worth asking the PI
for, never their bar.

## Arms
| arm | argv | what it isolates |
|---|---|---|
| **T1h** | `stack/ops/runs.d/r9e1_T1h.argv.json` = T1 + `--r9-bev-warp-supervise-hist` + `--r9-bev-warp-fuse-init identity` + `--r9-bev-warp-fill-only` | the history lever, made reachable (supervise the history-only cells, copy-init the fuse, fill only cells the current lift cannot see) |
| **T1hs** | T1h + `--r9-bev-warp-shuffle` | the CONTENT control: the same supervision set (geometry-only mask) and the same fuse, with deranged history |

One training run each (seed as T1). Rig, windows, estimator and floor = E1's (REFCV9_DESIGN 4.3): T0 label level, 2,308 DEV windows /
70 episodes, paired episode-cluster bootstrap B=2000, floor = |C1 − C0| (one seed pair).

## Keep rule (unchanged from 4.3; NOT relaxed for T1h)
T1h is KEPT iff, against **C0**:
1. the primary `nf_prec_0_10` (corridor precision 0–10 m incl. lift-invalid cells) is separated in the good direction, AND
2. |Δ| > 2 × floor (2 × 0.0457598 = 0.0915196), AND
3. no other perception primary is separated worse beyond 2 × its floor.

## Attribution (both must hold to call it a HISTORY lever)
4. **T1h vs T1hs** (paired, same windows): `nf_prec_0_10` separated in T1h's favour. If T1h passes 1–3 but NOT 4, the gain is credited to the
   supervision / init change, not to temporal history, and is reported that way (a "supervision lever"), never as "history works".
5. Mechanism read (diagnostic, not a bar): T1h's final |gate| mean (T1: 0.0077). Reported with the verdict.

## Reported alongside (not bars)
`nf_prec_0_5` (the camera-blind band, C0 0.2211), `nf_prec_5_10`, box AP@1/2 m, and the four families row (LONGITUDINAL / LATERAL /
TACTICAL not computable on the rig, n = 0; STRATEGIC PAUSED by PI ruling 2026-09-19).

## Committed outcomes and next levers
* **T1h KEPT and attributable (1–4)** → T1h replaces T1 in K2's composition (the FLIP drops its flags from refcv9-full's forbidden levers).
* **T1h KEPT, not attributable (1–3, not 4)** → keep the supervision / init change as its own lever; history remains untested-positive.
* **T1h FAILS 1–3 with an OPEN gate (|gate| mean > 0.05)** → history is used but does not move the primary at 0.5 / 1.0 s → next lever: longer
  lags (1.5 / 2.0 s) so more of the 0–10 m band is history-covered.
* **T1h FAILS with a SHUT gate (|gate| mean ≤ 0.05)** → the dead start persists → next lever: a gate lr multiplier / non-zero gate init.

Noise caveat (binding): one training run per arm; the 2 × floor clause carries run-to-run noise from ONE seed pair (H-ESTIM-SEED-1).

## Addendum 2026-11-05 14:10Z — clause 4 tightened, before any T1h data

Written before T1h or T1hs has been gated, trained or scored (neither arm has run; the PI has not yet approved their compute). Clause 4 is
TIGHTENED, never relaxed: T1hs read with `--ref T1h` must be **separated worse than T1h beyond 2 × floor** on `nf_prec_0_10` — the convention
the verdict already applies to T1s — instead of "separated in T1h's favour". Why: one run per arm, and a separated CI from one-seed arms is
necessary, not sufficient (`H-ESTIM-SEED-1`: two runs differing in nothing cleared "separated" on 9.5 % of cells). The verdict reports the
first-written reading too: `NOT-WORSE-BEYOND-FLOOR` = separated worse but inside 2 × floor, which clause 4 as first written would have
attributed and this addendum does not. Implemented in `stack/scripts/r9_e1_verdict.py` (`ARMS["T1h"]`, `ARMS["T1hs"]`, `control_outcome`).
