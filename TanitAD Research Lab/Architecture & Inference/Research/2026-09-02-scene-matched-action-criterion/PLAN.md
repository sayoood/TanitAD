# PLAN — E-ARCH-SMAS-1

`2026-09-02 · Architecture & Inference · compute budget: 0 GPU, seconds of CPU`

## Priority order (a killed run still yields value at every step)

| # | step | yields on its own | state |
|---|---|---|---|
| 1 | Read the two banked actdiv JSONs by explicit key (the k=8 file is mislabelled inside) | the three arms' `action_spread` / `scene_spread` at h1/h2/h4, from raw, not prose | ✅ done |
| 2 | Log-additive decomposition of the one-variable ratio change | F1 — 73.7 % of the fall is the denominator | ✅ done |
| 3 | Convert the pre-registered ratio bar into its action-side equivalent | F2 — the "10×" bar actually demanded 17.1× | ✅ done |
| 4 | Tabulate what the raw ratio would report for a *successful* arm | F3 — a 2× action rise reads as "+17 %, inert" | ✅ done |
| 5 | Recompute the arms under a pinned denominator (SMAS) | F4 — verdict unchanged, magnitude and bar changed | ✅ done |
| 6 | Run the control panel | admissibility: all four controls read their known values | ✅ done |
| 7 | Write the criterion so L-1's arm can be pre-registered against it | the deliverable | ✅ done |

## Compute

* **0 GPU.** Pure re-analysis of two banked JSONs; no checkpoint is loaded and no
  forward pass is run. The dev-box 4060 was checked free of Python compute before
  and after this pass; Thor and the A40 pod were **not touched**.

## Owners / integration

* **Owner:** Research Lab (daily pass 2026-09-02).
* ⛔ **The Lab does not amend a pre-registration.** F2 says `PREREG_MM_E19`'s bar
  was mis-specified; that is escalated to the Master Mind in `COMMS.md`, not
  edited here.
* The criterion is written to be *adoptable* — a prereg can cite it by id
  (`E-ARCH-SMAS-1`) and name its reference arm — but adopting it is a programme
  call.

## Reproduce

```bash
python code/scene_matched_criterion.py --out raw/smas.json
```

Source paths are constants inside the script and are echoed into
`raw/smas.json._sources`. The run **refuses** if C0 fails on any arm, so a broken
source read stops the analysis instead of producing a plausible number.
