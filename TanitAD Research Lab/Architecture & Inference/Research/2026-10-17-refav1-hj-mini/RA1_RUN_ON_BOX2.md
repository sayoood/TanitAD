# RA-1 on Box 2 — job file (refav1-HJ mini: R0 / A0 / A1 / A2 / A1x, + exploratory A1h)

Spec: `Project Steering/PREREG_HJ_STAGE0.md` (H-HJ-SENS-1 / H-HJ-K-1 / H-HJ-COST-1). **No bar is changed.** Box 2 = Windows, RTX 4060 8 GB,
venv `D:\tanitad\venv`, repo `D:\tanitad\repo`, cache `D:\tanitad\cache\{train,eval}` (the S4 pooled cache; `train` = 600 clips first).
Code: `stack/tanitad/models/ra1_hjepa.py`, `stack/tanitad/data/ra1_pooled.py`, `stack/scripts/ra1_hjepa_{train,eval}.py`,
`stack/tests/test_ra1_hjepa.py`, and ONE changed shared file: `stack/tanitad/models/hjepa.py` (`action_sensitivity(..., per_window=False)` —
additive, default unchanged, 45 existing tests green). The queue: `code/ra1_box2_queue.cmd`.

**PREREG Amendment 1 (A1-cell SELECTION) is implemented: every trainer call carries `--holdout-n 60` and the trainer REFUSES to run without it.**
The last 60 of the 600 stage-A TRAIN clips (sha12 order) are withheld from training for EVERY arm; each eval scores them (`--train-cache`, already in
the queue's eval line) and `verdict` selects the A1 cell on **seed-0 holdout ADE@6 s — never on eval-141**, which scores K once, for the chosen cell.
Each run's `meta.json` records the holdout clip list and its sha256 (also in `DONE` and every checkpoint).

## 0. Preflight (no GPU, ~2 min) — do not train until both are green
```
cd /d "D:\tanitad\repo"
set "CUDA_VISIBLE_DEVICES=-1"
set "PYTHONPATH=D:\tanitad\repo\stack;D:\tanitad\repo\taniteval"
"D:\tanitad\venv\Scripts\python.exe" -m pytest -q stack\tests\test_ra1_hjepa.py stack\tests\test_hjepa.py
set "CUDA_VISIBLE_DEVICES="
"D:\tanitad\repo\TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-refav1-hj-mini\code\ra1_box2_queue.cmd" gate
```
`gate` runs G-HYG/G-DVB/G-LIVE on the REAL cache for all six arms and prints a VRAM estimate. Every path in the queue is set with
`set "VAR=value"` and expanded only inside double quotes because the package path contains `&`. `RA1_DRYRUN=1` prints the commands instead.

## 1. Run order (each job = train, then eval; skipped if its `eval.json` exists; a killed queue is simply re-run — checkpoints resume bit-exactly)
| stage | command | runs (seeds) |
|---|---|---|
| 1 | `ra1_box2_queue.cmd stage1` | R0 (s0,s1) · A0 β 0.04 and 0.08 (s0,s1) · A1 grid β{0.04,0.08}×γ{50,100} (s0) = **10 runs** |
| — | `ra1_box2_queue.cmd stage1verdict` | prints `A1_best_cell` (lowest seed-0 **holdout** ADE@6 s; rule recorded in the JSON as `A1_best_rule`; exit 4 and `status` = `A1 SELECTION REFUSED` if a cell has no holdout numbers) |
| 2 | `ra1_box2_queue.cmd stage2 <BETA> <GAMMA>` | best A1 cell (s1) · A2 (s0,s1) · A1x (s0) · A1h (s0, EXPLORATORY) = **5 runs** |
| 3 | `ra1_box2_queue.cmd final <BETA> <GAMMA>` | `verdict.json`: SENS / K / COST bars, the A1x gate, the branch the PREREG selects |

Single-arm form (what the queue runs): `python stack\scripts\ra1_hjepa_train.py --arm A1 --beta 0.08 --gamma 100 --seed 0 --grid 4x10 --holdout-n 60 --cache D:\tanitad\cache --p10-prior <p10_prior.pt> --out D:\tanitad\runs\ra1\A1_b0.08g100_s0 --steps 6000 --device cuda`
(`p10_prior.pt` = `...\2026-10-17-v8f-mini\raw\p10\p10_prior.pt`, the steer-unit refit). The secondary 8×20 grid is `--grid 8x20` (A2 estimate 5.5 GB; use `--batch 32` if it OOMs).
Launch detached (the dev-box lesson; UNVERIFIED on Box 2): `powershell -NoProfile -Command "([wmiclass]'Win32_Process').Create('cmd /c \"<queue.cmd> stage1 > D:\tanitad\runs\stage1.log 2>&1\"')"`.

## 2. Wall time and memory
* **MEASURED (this dev-box CPU, A1 real clips, batch 64, 30 steps):** 0.83–1.09 s/step (`s_per_step` is divided by the steps in the log window).
* **ESTIMATE, UNVERIFIED on the 4060:** 0.1–0.3 s/step ⇒ **10–30 min per arm** at 6,000 steps; the plan's own figure is 1–2 h/arm. Queue: 15 runs ⇒ 2.5–7.5 h (or 15–30 h on the plan's figure) + eval ≈ 3–10 min each (CPU; UNVERIFIED at 141 episodes — the 8-clip smoke took 1.5 s).
* **VRAM, ESTIMATE** (params×16 B + hook-counted activations×8 B + fp32 token buffers, ×1.25): R0 1.27 GB · A0/A1/A1x 1.68 GB · A1h 1.98 GB · A2 1.76 GB at batch 64, 4×10 (**5.49 GB at 8×20**). The trainer logs `cuda_peak_gb_MEASURED` — that is the measurement; compare it on the first log line.
* Disk ≈ 70 MB per `latest.pt`/`final.pt`; ≈ 2 GB for the 15 runs.

## 3. What the arms are (so a reader does not have to open the code)
Common: frozen pooled DINOv3-L tokens → trainable adapter (LN → per-token Linear 1024→64 → flatten 40×64 → MLP → z¹ of 128) → 6 s readout (MLP 256) over
`[LN(z¹…zᴸ), 8 standardised P10 features]` as a **residual on the frozen steer-unit P10 trajectory**; loss = `w_traj·ADE_m + world-model terms`; AdamW, 6,000 steps, batch 64 segments of 19 frames (3.6 s), cosine LR with 200 warm-up steps.
* **R0** no world model (no predictor, SIGReg, IDM): readout over `[z¹, P10]`. **A0** 2-level H-JEPA, SIGReg only (γ = 0 at every level). **A1** A0 + IDM at both levels. **A2** A1 + level 3 (s₃ = 3 ⇒ 1.8 s). **A1x** A1 with each window's frames **time-shuffled** before the WM loss (the readout path is untouched) — the deliberate regression. **A1h** A1 + history dropout (below).
* Action = `(a, steer, v)` standardised; **steer is a road-wheel angle, not a curvature** (R46) and every path integration uses `action_units="steer"`.
* **A1h (EXPLORATORY; orchestrator addendum 2026-10-17; never decides H-HJ-SENS-1).** The WM **does** take history: each level's causal predictor attends over up to 3 past latents *and* their actions (`context = 3`; the first position of every chunk has none). A1h wraps every level's predictor so that, with probability `--hist-drop` (0.5) per (window, position), that position is predicted from itself alone (context 1) — the copycat remedy of 2010.14876. The flag reaches the input path (test: p = 1 ⇒ position *i* equals its context-1 prediction and is blind to every earlier latent/action; p = 0 ⇒ bit-identical to A1; eval ⇒ identity; no parameter added; the flag changes the loss of an otherwise identical model). The SENS rollout is context-1 at eval for every arm.

### Decisions the PREREG does not fix — identical across arms, overrule before launch if you disagree
1. The (β, γ) grid is read as **LEVEL 1's** weights; upper levels use the paper's DROID level-2 values (β 0.16, γ 50, act-SIGReg 0.005). A0 sets every γ to 0 and keeps β; A0 is run at **both** β so the SENS comparison is β-matched.
2. `w_traj = 10` (trajectory loss in metres vs the WM terms; at γ = 100 the IDM term is ≈ 100 at init and dominates the adapter's gradient — this is the paper's regime, but it is a real free constant).
3. **R0 reads `[z¹, P10]`**; A-arms read `[z¹, z², (z³), P10]` (the extra input is a pointwise MLP of z¹).
4. **Best-A1-cell selection is made on the withheld TRAIN holdout, NOT on eval-141** (PREREG Amendment 1; the first build selected on the same eval-141 windows that score K, which is the "tune on the data you score" trap). The last 60 of the 600 stage-A clips (sha12 order) are withheld from training for EVERY arm (`--holdout-n 60`; 540 clips train); the cell = the lowest seed-0 mean ADE@6 s over those 60 clips' valid frames (`A1_cells_seed0_holdout_ade6` in `stage1_verdict.json`). eval-141 scores K once, for the chosen cell; every cell's eval-141 ADE@6 s is still listed (`A1_cells_seed0_eval141_ade6_NON_DECIDING`) and decides nothing. Same 60 clips at 600 and at 4,572 (the 600 is the nested sha12 prefix). **DISCLOSED CAVEAT:** the frozen P10 prior (`v8f_mini_fit_p10.py`) was fit on the first 600 clips by sha12, so the holdout clips are IN P10's fit set (P10 is in-sample there, out-of-sample on eval-141). It is the same P10 for every cell, so the offset is common to all four cells, but the READOUT's residual is smaller on the holdout than it would be out-of-sample, which can bias the cell ranking toward cells that correct P10 less. The cure (refit P10 on the 540 train clips) changes the banked prior and is not done here.
5. COST: goal frame excluded; L1 on its own 0.2 s clock (primary), L2 on s = 3; the L1-on-L2's-clock variant is reported beside it. L3 is exploratory.
6. Event strata come from GT ego motion with thresholds fixed in `ra1_pooled.strata_labels` (turn |Δyaw₆ₛ| ≥ 0.35 rad; stop; brake). **Lead braking and distance-keeping are NOT computable from the pooled cache** (no lead track) and are reported as such.

## 4. Reading results
Per run `…\eval\eval.json` (+ `perwindow.npz`): `instrument_ok` (self-swap SENS control = exactly 0.0; P10 − P10 = +0.0000 [0,0]; ADE(gt,gt) = 0; no parameter without Adam state) — **exit code 3 / `[FAIL] eval` means an instrument control failed: read nothing else**.
Tiers: SENS and COST = **T0** (recorded future actions / true +6 s latent); K and the four families = **T1-class open-loop trajectory** (the model supplies its own trajectory; it does not drive). STRATEGIC family = **PAUSED** (PI ITEM 25), never a pass.
Estimator: paired episode-cluster bootstrap, B = 2,000 — it answers *another draw of EPISODES* only; training-seed variance enters `verdict` through the 2-seed floor |A1₍s0₎ − A1₍s1₎|; inference-seed variance is 0 (deterministic readout and rollout).
`verdict.json` carries the PREREG branch verbatim and the A1x gate first: **A1x not losing to A1, separated ⇒ instrument broken, nothing else is read.** A failed bar is reported FAILED and the named next lever runs (RULE ZERO) — SENS FAIL ⇒ γ 200 → IDM at level 2 only → time-repulsion floor.
Exit codes: trainer `2` = launch gate refused (no training happened), eval `3` = instrument control failed. Also: the trainer exits `2` (argparse) when `--holdout-n` is missing, and verdict `4` = no A1 cell could be selected (none supplied, or a cell has no holdout numbers: read `status`).
Holdout numbers: `eval.json["holdout"]` and the `hold_*` arrays of `perwindow.npz` (per-window ADE@6 s on the withheld clips; kept apart from the eval-141 `ade6`). `eval` re-derives the holdout from the train manifest and refuses a checkpoint whose recorded holdout/train split no longer matches (`HOLDOUT DRIFT` / `TRAIN-SPLIT DRIFT`).
