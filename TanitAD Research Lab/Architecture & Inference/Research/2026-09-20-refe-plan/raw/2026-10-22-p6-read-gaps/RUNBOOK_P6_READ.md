# RUNBOOK: P6's pre-registered navtest read, start to finish (dev box, 2026-10-22; supersedes the dry run's Q5)

Rule: `eval/PREREG_P6.md` (IN FORCE). Nothing here changes an endpoint, comparator, estimator, rule or family. Tier of every number: **T1** (NAVSIM v1.1 PDMS, self-action open loop).
Estimator: paired log-cluster bootstrap over the 136 logs, 10,000 resamples, seed 20260927; it answers *another draw of LOGS* only. Evidence class of a number you produce: MEASURED (ours, with its artifact).
⛔ **ONE look, on `model_final.pt`.** No other P6 checkpoint scores or plans on a navtest token. Smokes use NAVTRAIN tokens (step 2).
⛔ Never `--exploratory`, `--goal-fix off`, `--sanitize-goal` for the read. Never ssh to the pod. Never hold a live agent for the 5 h: start detached, write a `HANDOFF.md`, end.

```bash
PKG="E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"   # stage2_sft.py reads and writes under this E: path (D: and E: are the same volume)
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe          # tanitad venv (numpy, torch)
PYD=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe    # harness / seam venv
DATA=E:/Projects/TanitAD/data/refe_navtest
OUT=D:/Projects/TanitAD-artifacts/p6_final
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_XET=1
```

## What the Master Mind lands first (all staged in `p6gaps/code/`, paths relative to the repo root; nothing else changed)
`PK/eval/stage2_sft.py` (G2, G3) · `PK/eval/stage3_full.py` (G1; its `STAGE2_SFT_MD5` pin moves with stage2) · NEW `PK/eval/harness_pins.py`, `PK/eval/selection_quality.py`, `PK/eval/test_p6_read_gaps.py`,
`PK/eval/harness/` (21 files + `HARNESS_PINS.json`) · `PK/raw/2026-10-18-sel-rank/code/srttc_s0_dump.py` (G4; CRLF file) · `PK/raw/2026-10-19-p6-read-prep/code/build_candidate_bank.py` (G4).
`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Land them as blobs (`hash-object -w`, no CRLF filter on `srttc_s0_dump.py`: its pin is a raw sha256 of CRLF bytes).

## 0. Preflight (about 6 min, CPU; each line has an expected reading)
1. Both trees equal: `md5sum "$PKG/eval/stage2_sft.py" "D:/${PKG#E:/}/eval/stage2_sft.py"` equal; `$PKG/eval/harness/HARNESS_PINS.json` present.
2. **Harness guard:** `$PY "$PKG/eval/harness_pins.py" --verify` -> `"ok": true`, `n_files 21`, `caller.variant` = `pre_NT_FIX1` or `NT_FIX1`. (stage2 runs the same guard first and refuses on any mismatch.)
3. **Tests:** `$PY "$PKG/eval/test_p6_read_gaps.py"` -> `ZZP6GAPS PASS 31/31` (about 3 min; it reads banked u400 / P2 tables: known-value controls, no P6 file). `$PY "$PKG/eval/selftest_stage3_full.py"` -> `11/11`.
   `$PY "$PKG/eval/stage3_full.py" --selfcheck` -> `R2_PRECONDITION_MET=True` (about 5 min; also proves the banked families6 reproduces u400's banked families block).
4. Dev box: `ls C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active` both ABSENT; `nvidia-smi --query-gpu=memory.used --format=csv` <= 3,500 MiB; free RAM > 4 GB.
5. Disk: D: needs about 1.5 GB for the read (model 1.29 GB, seam 4 MB, score 26 MB) **plus about 0.2 GB for the proposal dump and table**; the selection-quality columns write 64 seams (0.34 GB, MEASURED for P2) and 64 harness dirs (about 6 files x 1 MiB each = 0.4 GB on this exFAT). Free D: >= 5 GB.
6. Environment the harness itself needs (outside the repo, unchanged): `E:/Archive/devbox-C/navsim/` (navsim tree, `exp/w3_navtest_v1`, `data/openscene`), `C:/Users/Admin/navsim-crun/`, `E:/Projects/TanitAD/taniteval/adapters/navsim.py` (families6's adapter).

## 1. Fetch the checkpoint (2-4 min)
`TOK=$(grep -oE 'hf_[A-Za-z0-9]+' E:/Projects/TanitAD/Keys.txt | head -1); export HF_TOKEN=$TOK` (never print it), then
`$PY -I -c "from huggingface_hub import hf_hub_download as d; [d('Sayood/tanitad-refe-p6-paper-faithful', f, local_dir='D:/Projects/TanitAD-artifacts/p6_final') for f in ('run/model_final.pt','run/summary.json')]"`
(the `run/` prefix of the FINAL file is UNVERIFIED). `summary.json` must say `"done": true`; `SHA=<its model_final_sha256>`; `sha256sum $OUT/run/model_final.pt` must equal `$SHA`. **A result exists only when it matches.** `CK=$OUT/run/model_final.pt`.

## 2. No-navtest smoke of the real file (about 3 min GPU; takes the lock itself, never waits)
```bash
export P6DRY_PKX="$PKG"        # the dry-run driver wants a dir holding eval/ and refe/: the repo package is one
cd "$PKG/raw/2026-10-22-p6-read-dryrun"
$PYD -I code/p6_navtrain_plan.py plan --sel raw/tokens_sel.json --ckpt $CK --device cuda --limit 12 --repeat 2 --out $OUT/smoke.npz
$PY  -I code/p6_compare.py --a $OUT/smoke.pass0.npz --b $OUT/smoke.pass1.npz --mutate
```
Expect `ZZPLAN_OK 12`, `format refe-full-v1`, `nav_mode command`, `IDENTICAL true`. (Needs `D:/Projects/TanitAD-artifacts/p6dry/hf/logs/commands.json`.) Navtrain tokens only.

## 3. The read (about 5.3 h detached; stage 2 takes the GPU lock for the seam only)
```bash
ARGS="--ckpt $CK --name p6_full --rule navsim_v1 --tokens $DATA/navtest_full_tokens.json --dump-proposals"
nohup bash -c "$PY $PKG/eval/stage2_sft.py $ARGS; test -f $DATA/seams/stage2_p6_full.npz && $PY $PKG/eval/stage2_sft.py $ARGS && $PY $PKG/eval/stage3_full.py --name p6_full --ckpt $CK --expect-ckpt-sha256 $SHA --base u400 --also p2 --expect-rule navsim_v1" > $OUT/p6_full.out 2>&1 &
```
* `--dump-proposals` (G2) makes the seam also write `$DATA/seams/stage2_p6_full.proposals.npz` (about 95 MB, all 64 proposals + the scorer's logits from the SAME forward pass): the input of step 5. It does not change the seam's poses (the seam command is the old one plus exactly that flag: tested).
* Attempt 1 ends `ZZSTAGE2 p6_full INCOMPLETE (seam failed)` at the 36-token frame-control gate: EXPECTED (36 declared C1 tokens, max 6.18063877792441 m, `RESULT_SFT4.md` Deviation 1). The `test -f` guard continues only if the seam npz exists. Attempt 2 reuses the seam and scores the whole filter without `--tokens`.
  Its `ZZSTAGE2` line is P6 vs the DEPLOYED system and ends `proposal_dump=ok`; **`proposal_dump=MISSING` means the dump was not written: stop and read `seam.log`, the seam cannot be re-run cheaply (4.8 h).**
* Stage 2 first runs the harness guard (HARNESS PIN GUARD) and refuses with the offending file named if the banked harness is not the pinned one. The harness it runs is `$PKG/eval/harness/` (banked), no longer a temp scratchpad.
* `stage3_full.py` pairs P6 with **C1 = u400** (block `R2`) and **C2 = P2** (block `R2b`, the new `p2` baseline: pins md5 `bed5687f...` csv, `49a1f6ec...` seam, families `d80a2599...`). Its GO / NO-GO words are P2's amendment words: **do not quote them for P6**; apply step 4.
* Wait with ONE background Monitor on `$OUT/p6_full.out` (a second `INCOMPLETE`, a `Traceback`, a `HARNESS PIN GUARD`, or `ZZSTAGE3` ends it). Write `HANDOFF.md` (this file, the command, `$OUT/p6_full.out`, `$SHA`) and end.
* Outputs: `$PKG/raw/2026-10-05-sft-stage2/p6_full/{result.json,score_status.json,p6_full_families.json}` (harness PASS 12,146/12,146 required), `$DATA/score/refe_stage2_p6_full/refe_stage2_p6_full.csv`, `$DATA/seams/stage2_p6_full.npz` (+ `.report.json`: `nav_mode command`, `driving_command_set 12146`, `rule navsim_v1`, `rows 12146`, `misses 0`), and `$PKG/raw/p2_readout/p6_full/r2_result.json`.

## 4. Apply PREREG_P6's rule (by hand, from `r2_result.json`; the numbers, never stage 3's words)
For each comparator X in {C1 = block `R2`, C2 = block `R2b`}: delta = `X.primary.mean`, `[lo, hi]` = `X.primary.ci95`; adverse families = `R2.families_vs_base.adverse` (C1) / `R2b.families_vs_also.adverse` (C2) (longitudinal + lateral components; PREREG's family rule, reported by stage 3, NOT gated by it).
* **ADOPT** iff `lo > 0` AND the adverse list is empty. **REFUTED** iff `hi < 0`. Otherwise **NOT PROVEN**. State it per comparator, separately.
* **Goal 94.55** (PUBLISHED, Table A13): P6's level and its level CI come from the arm's own csv, in seconds, with the registered estimator and NO dependence on step 5 (the 4-6 h columns): `$PY $PKG/eval/selection_quality.py level --arm-csv $DATA/score/refe_stage2_p6_full/refe_stage2_p6_full.csv --out $PKG/raw/p2_readout/p6_full/level_p6_full.json` -> `ZZSELQ_LEVEL <mean> [<lo>, <hi>] n=12146 logs=136` (`<mean>` must equal `R2.primary.arm_mean_x100` from `r2_result.json`; control: the same command on P2's csv prints `88.8184 [87.8611, 89.7701]`, tested). GOAL MET iff point >= 94.55 AND ADOPT vs C1; NOT EXCLUDED iff point < 94.55 <= upper bound; BELOW GOAL iff upper < 94.55. The harness difference to the paper (<= ~0.25, ESTIMATED) is reported, never subtracted.
* **Lever claim:** needs `lo` > the seed floor F = 0.33 PDMS x100 (P2 seed replicate, `eval/RESULT_P2REP.md`); otherwise report "separated, single seed, not yet a lever claim". A P6 seed-1 replicate is requested only if ADOPT and delta < 2 F.
* Also report (not gating): the six sub-scores, the per-epoch 923-token snapshots, the in-run series.
* **STRATEGIC family: PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25): write "PAUSED", never a pass, never omitted.

## 5. TACTICAL rows
* **Lateral / longitudinal / 5-way decision accuracy and kappa:** in `p6_full_families.json` (`families.refcv6.tactical`, no interval) and in `r2_result.json` `tactical_vs_base`.
* **Selection quality, random / actual / best-of-64 (G2)** from the dump: 64 harness columns on CPU, about **4 to 6.4 h** on 4-6 workers (MEASURED for P2: 777,344 token-columns, `raw/2026-10-16-p2-full-table/PLAN.md`); RAM guard waits below 4 GB. Start it AFTER step 3 finishes (CPU contention otherwise), detached:
```bash
D=$DATA/seams/stage2_p6_full.proposals.npz
nohup bash -c "$PY $PKG/eval/selection_quality.py score --dump $D --name p6_full --workers 4 && $PY $PKG/eval/selection_quality.py table --dump $D --name p6_full --arm-csv $DATA/score/refe_stage2_p6_full/refe_stage2_p6_full.csv && $PY $PKG/eval/selection_quality.py read --table $DATA/seams/selq/p6_full/table_p6_full.npz --out $PKG/raw/p2_readout/p6_full/selection_quality_p6_full.json" > $OUT/selq.out 2>&1 &
```
  `score` is resumable (rerun the same command; PASSed columns are skipped). It scores the whole-filter path (no `--tokens`; 12,146 rows per column). Markers: `ZZSELQ_SCORE ... OK 64/64`, `ZZSELQ_TABLE ... OK` (gates: every cell valid, **PDMS at the pick equals P6's own harness csv bit for bit**), `ZZSELQ_READ actual ... random ... best64 ... ok`.
  `INVALID` means an instrument failure (random >= actual), never a result. Reference: P2's own table reads actual 88.8184 / best-of-64 98.2311 / random 73.0031 (`raw/2026-10-16-p2-full-table/anatomy.json`; reproduced to 1e-9 by `test_p6_read_gaps.py`).
* ⛔ **Command adherence of the selected proposal: NOT IMPLEMENTED, BLOCKED ON A MASTER-MIND DECISION** (RESULT.md "G2b": PREREG names the row but does not define it). Report the row as `NOT COMPUTED: PREREG_P6 does not define the metric; question to the MM open`, never a pass, never omitted.

## 6. Determinism assertion (PREREG, about 5.3 h; run after step 3 if the PI wants it before the registry row)
`stage2_sft.py` with `--name p6_full_rep` (same ARGS, new name, WITHOUT `--dump-proposals`), then per-token equality: `$PY -I -c "import csv;f=lambda p:{r['token']:r['score'] for r in csv.DictReader(open(p)) if r['token']!='average'};a=f('$DATA/score/refe_stage2_p6_full/refe_stage2_p6_full.csv');b=f('$DATA/score/refe_stage2_p6_full_rep/refe_stage2_p6_full_rep.csv');print(len(a),a==b)"` must print `12146 True`. A mismatch is a defect to investigate BEFORE any read is quoted.

## 7. After
Record the CSV and seam md5, `$SHA`, the proposal-dump md5 and the pins file sha256 (`result.json` -> `harness_pins.pins_file_sha256`) in the registry row; update `Project Steering/GOALS_AND_CLAIMS.md` in the same turn; bank `$PKG/raw/2026-10-05-sft-stage2/p6_full/` and the arm files under `$PKG/raw/p2_readout/p6_full/arm/` (these exist on the E: disk only until copied). Delete `$OUT/smoke*`; keep `model_final.pt` until the row lands.

## Traps met while building this (each would have cost the read)
* **The scoring step would have died AFTER the 4.8 h seam.** The tip's `eval/score_navtest_refe.py` (NT-FIX1, commit 9e8d11f) loads `run_v1_fix1.py` from W3's code dir and byte-pins `run_v1.py`; the old scratch `w3e/code` had neither `run_v1_fix1.py` nor a pin-matching `run_v1.py`. Fixed by banking `run_v1_fix1.py` and setting `RUN_V1_FIX1_ALLOW_DRIFT=1` (the banked `run_v1.py` is the original with exactly six `D:/` -> `E:/`). Evidence: `raw/harness_ctl/`.
* `navsim_v1_win.py:49` pointed at `.../Temp/.../scratchpad/w3e/e1/navsim_win.py`: the harness reached into the scratchpad even for the E1 wrapper. The banked copy resolves it beside itself.
* `grep -c $'\r'` counts EVERY line in this shell: count CRs with python.
* The dry run's "families md5 a6a381b4" was `p2_full/deployed_families.json`; P2's own families file is `d80a2599` (CRLF) / `2bedac30` (LF).
