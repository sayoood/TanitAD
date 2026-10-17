# HANDOFF -- train M1, M2 and H-DATA-SCALE-1 on the 8 GB box (the cache must exist first)

State: the code is complete and tested (CPU). **No feature cache existed, so no arm has run on real features.** The first action on box 2, after the S4 cache is built, is the `--selfcheck-only` launch gate below. Nothing waits on a job in this package.

## What box 2 needs (copy; sizes MEASURED)
* repo checkout with the new files (RESULT.md §9): `stack/`, `taniteval/` on `PYTHONPATH`; python with torch (CPU-tested 2.x), numpy.
* the S4 cache: `<CACHE>/{train,eval}/clips/*.pt` + `manifest.jsonl` (pooled 4x10, ~4 MB/clip, ~18 GB train).
* `v9_labels_train.npz` + `.manifest.json` (`83e30bd3…`, 264 MB; the loader needs the sibling manifest), `lead_train4572.npz` (14 MB), `refcv6_speed_max_v8_train.jsonl`.
* eval: `v9_labels_eval139.npz` + manifest (8 MB), `lead_eval139.npz`, `refcv6_speed_max_v8_eval.jsonl`, and from this package `raw/eval141_kin.npz`, `raw/eval_grid_windows.json`, `raw/join_eval141.npz` (prebuilt), `raw/p10/p10_prior.pt` (74 KB, the frozen correct-unit prior), `raw/eval/floors_perwindow.npz`.
* `CACHE` below = e.g. `D:/refav1_hj_cache`; `OUT` = any run dir. Windows paths use `/`.

## 0. Gate (do this first; refuses a bad config, ~1 min)
```
python stack/scripts/v8f_mini_train.py --arm M1 --out OUT/gate --cache-dir CACHE \
  --v9-train V9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 \
  --lead-train V9/lead_train4572.npz --speed-train V9/refcv6_speed_max_v8_train.jsonl \
  --join-train OUT/join_train.npz --p10 PKG/raw/p10/p10_prior.pt --selfcheck-only
```
It builds and saves the join (minutes, once), then runs declared-vs-built and the live check; read `OUT/gate/selfcheck.json` (`live.pass`, `control_prior_grad_sum` 0.0, `info.n_windows_all_clips` ~ 310,642). `S` = ceil(windows / batch) = **4,854 steps at batch 64**.

## 1. M1 and its controls (2 seeds each; main budget 4S = 19,416 steps -- my choice, the PREREG is silent)
Common: `A="--cache-dir CACHE --v9-train … --v9-train-md5 83e30bd3… --lead-train … --speed-train … --join-train OUT/join_train.npz --p10 PKG/raw/p10/p10_prior.pt --batch 64 --steps-mult 4"`
```
for s in 0 1: python stack/scripts/v8f_mini_train.py --arm M1        --seed $s --out OUT/M1_s$s        $A
for s in 0 1: python stack/scripts/v8f_mini_train.py --arm M1-navonly --seed $s --out OUT/M1-navonly_s$s $A
for s in 0 1: python stack/scripts/v8f_mini_train.py --arm M1-nocons  --seed $s --out OUT/M1-nocons_s$s  $A
```
`M1-navshuf` is NOT a separate training run: evaluate the M1 checkpoints with `--nav-shuffle` (below). Resume = rerun the same command (same argv; a different argv is refused). A run is finished iff `OUT/<run>/summary.json` has `"done": true`.

## 2. M2 (the H-JEPA WM auxiliary; hmax 16 is automatic)
```
for s in 0 1: python stack/scripts/v8f_mini_train.py --arm M2a --seed $s --out OUT/M2a_s$s $A
for s in 0 1: python stack/scripts/v8f_mini_train.py --arm M2b --seed $s --out OUT/M2b_s$s $A
```
M2a's selfcheck lists the IDM heads as DECLARED dead (gamma 0) and requires exactly that; M2b requires them alive.

## 3. H-DATA-SCALE-1 (use M1's recipe, or M2's if it wins K-1; 5 settings x 2 seeds; S = one epoch at 100 %)
```
for f in 0.25 0.5 1.0: for s in 0 1: … --arm <WINNER> --data-frac $f --steps-mult 1 --seed $s --out OUT/DS_f${f}_S_s$s $A'
for m in 2 4:          for s in 0 1: … --arm <WINNER> --data-frac 1.0 --steps-mult $m --seed $s --out OUT/DS_f1.0_${m}S_s$s $A'
```
(`$A'` = `$A` without `--steps-mult`.) The 4S/100 % cell IS the main M1/M2 run: do not retrain it. Clips are nested by sha12 (25 % < 50 % < 100 %); the optimiser-step budget is independent of the fraction (the gate reads the clip count back).

## 4. Evaluate (grid = 141 episodes, 2,399 windows)
```
python stack/scripts/v8f_mini_eval.py run --ckpt OUT/M1_s0/ckpt_last.pt --tag M1_s0 --eval-cache CACHE --out-dir OUT/eval --project --families
python stack/scripts/v8f_mini_eval.py run --ckpt OUT/M1_s0/ckpt_last.pt --tag M1-navshuf_s0 --nav-shuffle --eval-cache CACHE --out-dir OUT/eval
python stack/scripts/v8f_mini_eval.py compare --runs M1_s0=OUT/eval/eval_M1_s0.npz M1_s1=… M1-navonly_s0=… M1-navshuf_s0=… … --out OUT/eval/compare.json
```
`floors` (P10 refit, kd_x, GT control, nav-only baseline) are already computed: `PKG/raw/eval/floors_report.json`. `compare` prints K-1 (ADE6 vs the REFIT P10, both seeds, seed floor F, `K1_pass`), CONSISTENCY(a) and (b), STRAT, TAC. Eval default uses `--ages cycle`; add `--ages fresh` for the optimistic variant.

## Resource estimates (ESTIMATED -- nothing was run on a GPU)
* Memory: 5.2 M params (M1), 7.2 M (M2) -> optimiser < 0.1 GB; activations at batch 64 ~ 0.3 GB of fp16 features + ~0.5 GB of adapter outputs/attention; **well under 8 GB** (batch 256 should also fit). RAM: the clip LRU defaults to ~1.5 GB (`--lru`).
* Time: compute per step is small (~65 GFLOP at batch 64); the loop is **input-bound** (per step 64 windows x 0.8 MB gathered on the CPU). Guess 0.05-0.15 s/step: one epoch S ~ 4-12 min, a 4S run ~ 16-50 min, the whole H-DATA-SCALE-1 (18 S-units for 2 seeds, sharing the 4S cell with the main run) ~ 2-4 GPU-hours. CPU reference (MEASURED, a loaded dev box): ~1 s/step at batch 32.
* Disk: cache ~18 GB train + ~0.6 GB eval; join npz ~100 MB.

## Known traps for the operator
* `--v9-train-md5` must match the file (the gate refuses otherwise). With the 4,369-clip release 203 cache clips run unlabelled (their tactical/strategic losses are masked, trajectory trains).
* A zero-init head means the O trunk gets no gradient at step 0 -- by design; the gate's live check looks after 3 steps.
* Do not edit a flag mid-run and `--resume`: the argv hash differs and the checkpoint is refused.
* `--amp` is UNVERIFIED; leave it off for the first run.
* The eval cache path in `v8f_mini_eval.py run` defaults to `C:/Users/Admin/refav1_hj_cache`; pass `--eval-cache`.
