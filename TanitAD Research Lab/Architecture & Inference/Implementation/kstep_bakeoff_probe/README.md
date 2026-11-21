# kstep_bakeoff_probe — K-step rollout lever, first measured arm (2026-07-09)

**Not an intake package** — this is a burst/experiment prototype (D-020 §4), separate from the MVP
stream. It changes no `stack/` code; it *drives* the shipped stack (`train_worldmodel`, `eval.bakeoff`,
`eval.gates`, `evaluate_checkpoint`) to measure the `kstep_rollout` bake-off lever.

## What it does
Trains two arms at **matched compute** on real comma2k19, identical except `train.rollout_k`
(asserted one-factor via `tanitad.eval.bakeoff.lever_diff`), and scores each through the D1–D3 gate
runner. Reduced-but-REAL probe config (d256/enc6/pred4/128px/9-ch, no tactical, no H15, 11.74 M params)
so two arms fit a Wednesday wall-clock on the RTX 4060 — **directional, not decision-grade** (P8).

## Run
```bash
python kstep_bakeoff_probe.py \
    --data-root C:/Users/Admin/tanitad-data/comma2k19/extracted \
    --episodes 28 --steps 2000 --seed 0 --k-base 1 --k-variant 2 \
    --out <out_dir>
```
Needs the `tanitad` venv (torch 2.11 cu128). Writes `<out_dir>/kstep_bakeoff_result.json` +
per-arm `k{1,2}/{config,metrics,model}.pt|json`.

## Result (2026-07-09, RTX 4060, $0) — see `results/2026-07-09-kstep_bakeoff_result.json`
- Rollout ≈ free: **+0.5 % wall-clock, 0 params**.
- D2 P1 direction-acc **saturated at 1.0** both arms → falsifier metric ceiling-limited.
- Discriminative `imag_rel`: **K=2 cuts 1-step error vs persistence 2.914→1.049 (−64 %)**, but **no help
  at 4-step** (I4 1.451→1.645) → **K must cover the decode horizon**.
- D1 FAIL + D3 BLOCKED both → **no decision-grade claim (D-004)**.

## Decision-grade follow-up (backlog P0 #2b)
Swap `probe_config` for the operative config + load the pod2 step-8k `ckpt_full.pt`; sweep K∈{1,2,4};
primary metric **`imag_rel` per horizon** (NOT dir-acc). **D-018: escalate before touching the trained config.**

Full analysis: `../../Research/2026-07-09-kstep-rollout-bakeoff-and-lejepa-identifiability.md`.
