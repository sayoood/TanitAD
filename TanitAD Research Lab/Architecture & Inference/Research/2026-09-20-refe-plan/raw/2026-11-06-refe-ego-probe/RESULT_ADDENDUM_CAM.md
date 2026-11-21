# Addendum to RESULT.md: per-camera ablation on the CRUISE subset (2026-11-07, Master Mind)

**Tier T0** (open loop, one step, logged ego history). Model u400 (`8b4dba48...`). 154 CRUISE tokens over 51 logs. Paired log-cluster bootstrap, B = 2000 ("another draw of logs" only; one checkpoint, deterministic planner).
Controls PASS: C1 39/39 bit-identical; C2 logits bit-identical to the banked table 154/154; C3 every arm reached the input; C4 6/6 grey cache rechecks identical. Full tables: `cam_full/analysis/READOUT.md`.

Two arms added to `refe_cam_probe.py`: **F0ONLY** (cameras 1-3 set to grey 128/255, front kept) and **NOF0** (front set to grey, cameras 1-3 kept).

| arm | mean plan v4 / v0 | dv@4s vs A0 (m/s) | share planning a >15 % slow-down |
|---|---|---|---|
| A0 (all real) | 0.934 [0.796, 1.051] | - | 0.409 |
| F0ONLY (front only) | 0.825 [0.764, 0.867] | -0.730 [-1.382, -0.104] | 0.532 |
| NOF0 (no front) | 0.517 [0.465, 0.560] | -3.249 [-3.889, -2.420] | 0.935 |
| GREY (all grey) | 0.645 [0.603, 0.687] | -2.273 [-3.217, -1.175] | 0.987 |
| GT (logged drive) | 1.001 | - | 0.013 |

**Reading (MEASURED, descriptive; no pre-registered rule covered these two arms).**
1. The front camera carries most of REFe's speed signal: with the front alone the plan keeps 0.825 of v0; without it, 0.517. That is below all-grey, so three real side and rear views with a grey front read as a stronger "slow down" than no image at all.
2. This was the anchor for TanitScenarioFlyWheel's straight-road cruise probe on Cosmos frames (sim F0-only 0.85 against this 0.825). Their full 154-token round trip of these frames through their rig then showed the rig transparent (`D-REFE-EGOREL-1, amendment`), so the image-informativeness hypothesis in RESULT.md item 5 is REFUTED.

**Files:** `refe_cam_probe.py` (the two arms; asserts a 5-D, 4-camera image tensor), `refe_a0_inputs.py` (forward pre-hook recording the exact ego and goal fed to the model), `export_cruise_a0.py`, `make_cruise_tokens.py`, `cam_launch_cmd.txt`, `cam_full/` and `a0in/` logs, status and meta.
Per-token arrays (`cam_full/results.npz`, `a0in/results.npz`, `a0in/a0_model_inputs.jsonl`) hold nuPlan-derived per-token ego states and poses and are NOT in the repo. They are on the dev box at `E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/` and, for the sim, in the private bundle on thor3.
