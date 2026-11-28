# PREREG_DOSE_P6 — Amendment A1: `--epochs 2.0` so the arm can reach 1,600 updates (FROZEN 2026-11-10)

Author: Master Mind, 2026-11-10, before any S1ZD run, log line or number exists. The card `PREREG_DOSE_P6.md` (sha256 `037c127235b51c374780f31ad2e93855ac7c141a4051f89498f40f1beff52ebb`) stays frozen and unedited; this file amends its section 1 only.

## What was found

The S1ZD build's reachability check (launch gate, `dose_reach_refusals`; G-DVB's schedule row) found that the card's two-value argv cannot run 1,600 updates. MEASURED from the trainer source at the landed tip:
- `scorer_finetune.py:1503` `--epochs` default 1.0.
- `scorer_finetune.py:2786-2788`: `total = ceil(len(tr_idx) / batch * epochs / accum)`, then `min(total, --max-updates)`. With 60,000 train sets, batch 8, accum 8 and epochs 1.0 that is 938, so `--max-updates 1600` alone ends the run at update 938 on a 938-update cosine schedule.
- `--epochs` is read at that one line and nowhere else in the trainer (grep: line 1503 declares it, line 2786 uses it). The training loop cycles the data loader until `total` (`while not done: for batch in dl`), and the loader's shuffle generator is seeded once from `--seed`. So `--epochs` changes nothing except the cap.

The programme has hit this trap before (`launch_gate.py` RLS1_RECIPE note, TL-1 2026-10-12: "--epochs 2 SILENTLY ends at 500 updates").

## The amendment

S1ZD's argv moves **three** values from S1Z's, not two: `--max-updates 400` becomes `1600`, `--out`, and **`--epochs 2.0` is added**. Then `ceil(60,000 / 8 x 2.0 / 8) = 1,875`, and `--max-updates 1600` binds. The run is the card's 1,600-update dose: LR warmup 50, then cosine over 1,550 updates, about 1.71 passes over the 60,000 training sets.

Nothing else changes: the seed (0), S1Z's measured λ_z, the mutants, the evaluation, the estimator, the bars D1 (0.4972) and D2 (0.6089), the guards, the goal, the outcomes and the cost (about 4.1 L40-h, one P7 pause). The PI's go for S1ZD training (2026-11-10, "I approve training for S1ZD") was for this cost.

The launch-gate profile `refe_dose_s1z_p6` and `dose_pod.sh` carry `--epochs 2.0` as a required, pinned value, and the gate still refuses the card's original two-value argv (it cannot reach its own `--max-updates`).
