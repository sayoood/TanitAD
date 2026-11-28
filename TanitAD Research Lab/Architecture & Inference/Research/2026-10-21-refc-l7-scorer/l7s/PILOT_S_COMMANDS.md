# PILOT_S_COMMANDS -- H-L7-1 PILOT-S (PREREG_L7_SCORER.AMENDMENT-1_DRAFT.md), L7-S at training seeds 0 and 1

Written 2026-10-22 by the l7s_prep agent (dev box, CPU only). **NOT RUN. Scorer training needs the PI's go AND a sealed amendment.** Everything below was checked only up to
the first optimizer step (a `--epochs 0` dry run and a zero-step preflight). Evidence classes: MEASURED = produced on this box today, artifact named.

## 0. Environment (every command) -- same as `PILOT_COMMANDS.md` sec. 0

```bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -X utf8"
L7=C:/Users/Admin/refcv8_pf/l7prep      # pkg/code = the landed package code, stack_min, targets (built 2026-10-22 12:4x)
W=<dir holding the landed code/ + the new files>   # where scene_bank.py / train_l7.py / read_pilot_s.py sit; the dev-box work copy is C:/Users/Admin/refcv8_pf/l7s_prep/code
RUN="$PY $W/run_isolated.py --stack $L7/stack_min"
B=D:/tanitad-archive/l7cap_thor_1022; T=$L7/targets; CACHE=C:/Users/Admin/refcv8_pf/l7s_prep/scene_cache
```

## 1. Unpack the scene members (data prep, no training; TRAIN banks only -- the CLI refuses a path containing "eval") -- MEASURED 50 s wall, 0.59 GB peak, a cold re-unpack reproduces the cache files byte for byte

```bash
$RUN $W/scene_bank.py --npz $B/train139_s0/dump/l7cap_scene.npz $B/train139_s1/dump/l7cap_scene.npz --cache $CACHE
```
Why: `l7cap_scene.npz` is DEFLATE-compressed (agent `[4855,300,384]` fp16 = 1.12 GB + bev `[4855,480,96]` = 0.45 GB per bank), so the two banks do not fit the 4 GB budget in RAM.
The cache is raw `.npy`, content-addressed (`<member>.<crc32>.<bytes>.npy`), CRC32-verified while writing and again on every open. s0 and s1 are bit-identical, so **one 1.57 GB file set
serves both**. It is derived data: deleting it only costs the 50 s. `train_l7.py --mode S` does the same unpack lazily if the cache is empty.

## 2. The PILOT-S (two runs, sequential)

```bash
for seed in 0 1; do
  $RUN $W/train_l7.py --fan-npz $B/train139_s0/dump/fan_extras.npz $B/train139_s1/dump/fan_extras.npz \
     --targets-npz $T/train139_s0.npz $T/train139_s1.npz --mode S --scene-cache $CACHE --seed $seed \
     --out-dir C:/Users/Admin/refcv8_pf/l7pilot/L7S_ts${seed} > C:/Users/Admin/refcv8_pf/l7pilot_L7S_ts${seed}.log 2>&1
done
```
Every sec. 1 setting is the E / F default. The scene file is read from the same `dump/` as each `fan_extras.npz` (so the train-vs-eval refusal already covers it). Per epoch the log carries
`val` and `val_scene_shuffled` (control C-S1); the run ends with `train_log.json`, `best.pt` (loads with `refc_scorer.load_scorer`) and `ZZL7TRAIN ...`.

## 3. Read it (mechanical -- the amendment's sec. A3 as code)

```bash
$PY $W/read_pilot_s.py C:/Users/Admin/refcv8_pf/l7pilot/L7S_ts0/train_log.json C:/Users/Admin/refcv8_pf/l7pilot/L7S_ts1/train_log.json
```
It reads the validation slice only (refuses a log that is not the E / F slice: 980 windows, own 0.99895, oracle 0.35874, validation-row sha12 `5af23a842e87`) and prints O1 / O2 / O3 / WITHHELD
with `nu`, `M_own`, `M_E`, `g_own`, `g_E`, `h`. **No eval139 read.**

## 4. What was verified (MEASURED today)

| check | result | artifact |
|---|---|---|
| `--epochs 0` dry runs, E / F / S x seed 0 / 1 | n_train 8,730, n_val 980, 125 / 14 episodes, keep 133.856, S params **480,904**, scene dims bev 96 / agent 384, banks in identical order; E / F lines byte-identical to the landed trainer's | `l7s/PREFLIGHT_L7S.log` |
| zero-step preflight on the real banks, 4 repeats | G1-G6 PASS: split == E / F split; join + independent re-reads (numpy memmap, zip stream); 8 finite losses; **all 54 parameter tensors get a finite gradient (16 scene-branch tensors, none zero)**; weights bit-identical after (no optimizer); shuffle map cross-episode | `l7s/PREFLIGHT_L7S.json`, `.log` |
| time / memory | 0.37-0.39 s per batch of 32, 126-133 s per epoch, <= 63-66 min per run at 30 epochs (box shared with a launch gate); peak working set 1.09 GB, commit 1.68 GB | `l7s/PREFLIGHT_L7S.log` |
| package tests | 48 passed, 1 skipped (the skip is the pre-existing "eval139 s1 targets not built on this box" and reads no eval data); 6 production mutants (positional join, no CRC re-verify, shuffle ignored, no duplicate-key check, pad dropped, wrong `--agent-dim`) all RED | `code/tests/` |
