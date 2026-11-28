# PILOT_COMMANDS -- H-L7-1 PILOT (PREREG_L7_SCORER.md sec. 2 "PILOT"), L7-E and L7-F, training seeds 0 and 1

Written 2026-10-22 by the l7prep agent (dev box, CPU only). **NOT RUN.** Scorer training is not authorised until the PI's go; every command below
was checked only up to its first optimizer step (`--epochs 0` dry run, see section 3). Evidence classes: MEASURED = produced on this box today, artifact
named; ESTIMATED = arithmetic on a MEASURED number.

The PILOT is never deciding (PREREG sec. 2). It trains on **train139 s0 + s1** (two inference seeds of the same 4,855 windows, 139 TRAIN clips) and may be
read on eval139 **once**, labelled PILOT. The registered read is the full bank's.

## 0. Environment (every command)

```bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -X utf8"
L7=C:/Users/Admin/refcv8_pf/l7prep                  # work dir: code copy (pkg/code), sliced stack (stack_min), targets, gt6
RUN="$PY $L7/pkg/code/run_isolated.py --stack $L7/stack_min"
B=D:/tanitad-archive/l7cap_thor_1022                # the four banks copied from Thor (ZZL7ARCHDONEZZ 12:33:05)
T=$L7/targets                                       # targets built below
```
`-I` ignores PYTHONPATH, and the venv's editable `tanitad` points at G:; `run_isolated.py` puts `stack_min` (tanitad/{refs/refcv7_heads, refs/refcv7_oracle,
train/config_hygiene} at 701922c, `refs/__init__.py` emptied) first. `pkg/code/` = the changed package code (the code-hold copies are byte-identical except
CRLF line endings).

## 1. Inputs (already built today; rebuild only if a bank changes) -- MEASURED, 83-86 s per bank, peak working set 1.75-2.02 GB

```bash
export L7_STACK_SCRIPTS=$L7/stack_scripts            # refb_labels.py @701922c (build_gt6 imports it)
for b in train139_s0 train139_s1 eval139_s0 eval139_s1; do
  $PY $L7/pkg/capture/code/build_gt6.py --dump $B/$b/dump --out $L7/gt6/$b                     # S6 GT (1..6 s) from decisions/ep*.npz, ~2 s
  $RUN $L7/pkg/code/l7_targets_pai.py --dump $B/$b/dump --s6 $L7/gt6/$b \
       --agents-npz $B/$b/dump/l7cap_labels.npz --out-npz $T/$b.npz --out-json $T/$b.json       # --chunk 8 default; ~85 s
done
```

## 2. The PILOT (four runs, sequential; the launch gate on this box owns the other cores)

```bash
for mode in E F; do for seed in 0 1; do
  $RUN $L7/pkg/code/train_l7.py \
     --fan-npz  $B/train139_s0/dump/fan_extras.npz $B/train139_s1/dump/fan_extras.npz \
     --targets-npz $T/train139_s0.npz $T/train139_s1.npz \
     --mode $mode --seed $seed --out-dir C:/Users/Admin/refcv8_pf/l7pilot/L7${mode}_ts${seed} \
     > C:/Users/Admin/refcv8_pf/l7pilot_L7${mode}_ts${seed}.log 2>&1
done; done
```
Fixed by the script to the PREREG sec. 1 values: d_model 128, 2 layers, 4 heads (E/F are MLP trunks), AdamW lr 3e-4 wd 0.01, batch 32, <= 30 epochs, patience 3
**on the validation loss** (`--early-stop-on loss`, the default; `pick` restores the first version's criterion, both are logged every epoch), T = 0.5 m,
Huber beta 0.1, validation = 10 % of the TRAIN bank's episodes (14 of 139, seed 20261021), both inference seeds of a validation episode on the validation side.
`--seed 0/1` is the TRAINING seed on the SAME bank and the SAME split (PREREG sec. 5).

Each run writes `best.pt` (loadable by `refc_scorer.load_scorer`), `train_log.json` (per-epoch train loss, val loss / pick cost / own-pick cost / oracle cost,
**per-head held-out AUC -- K7: the `comf` AUC must be >= 0.95**), and prints `ZZEP ...` per epoch and `ZZL7TRAIN ...` at the end.

### Expected wall-clock (ESTIMATED from MEASURED per-batch times; `preflight_l7.json`)

| run | params | fwd+bwd+loss per batch-32 (median of 6) | batches/epoch (8,730 train windows) | est. s/epoch (x1.15 for optimizer + python) incl. both evaluate() calls | <= 30 epochs |
|---|---|---|---|---|---|
| L7-E | 154,504 | 0.0365 s | 273 | **12.3 s** | <= 6.2 min |
| L7-F | 155,400 | 0.0353 s (+ 0.0058 s features) | 273 | **14.0 s** | <= 7.0 min |

How many epochs patience 3 will take is NOT known (no scorer has been trained); the bound is 30. **Four runs sequentially: <= 27 min worst case** (+ ~15 s setup each; bank load 1 s,
peak working set 1.4 GB, commit 1.9 GB in the preflight). The optimizer step itself was NOT measured (the preflight takes none); the x1.15 is an allowance. The box is shared
with a launch gate (4 cores), so wall-clock may stretch.

## 3. Verified up to the first step: the exact command lines with `--epochs 0` (DRY RUN) -- MEASURED today

`--epochs 0` runs everything above except the loop: provenance check, bank load, episode split, model + optimizer construction, writes `dry_run_log.json`, takes
**no step and writes no `best.pt`**. All four PILOT command lines (E/F x seed 0/1) printed:

```
ZZL7DRYRUN {"mode": "E", "seed": 0, "n_train": 8730, "n_val": 980, "n_train_episodes": 125, "n_val_episodes": 14, "params": 154504, "keep_mean_candidates": 133.856, "provenance": ["train", "train"]}
ZZL7DRYRUN {"mode": "E", "seed": 1, ... "params": 154504 ...}      ZZL7DRYRUN {"mode": "F", "seed": 0/1, ... "params": 155400 ...}
```
Preflight on the same banks: `preflight_l7.json` (34 gates, 33 blocking all PASS, 1 non-blocking FINDING about the DAC label, see PREFLIGHT_L7.md).

## 4. Not in this file (and why)

* **The PILOT's single read on eval139** (L7-E / L7-F pick vs P-OWN at both fan seeds) is the registered picker/battery lane (`picker/code/build_picker_trees.py` +
  `pk_lane.py`, paired episode-cluster bootstrap in `taniteval/ci.py`), which is outside this package and was not touched. What is verified for it: the new eval
  banks carry every key the fanceil banks carry (K3: `anchor_traj`, `sel_score`, `refined_logits`, `reach_keep`, `sel_idx`, `traj`, `t`, `ws` bit-identical to
  fanceil s1 at the Thor capture), eval targets reproduce the banked FANCEIL own-pick / oracle / random S2 ADE (preflight gate), and `refc_scorer.load_scorer(best.pt)`
  round-trips (test). No eval number of any scorer exists or was read.
* L7-S (scene tokens), R-SAFE reads, and the NAVSIM arms: not part of the PILOT.
