# TF-2b HANDOFF (2026-11-12). STATE: BLOCKED ON DATA, prereg frozen, no fresh row exists.

**Blocker.** 11 of the 13 fresh logs (362 of 687 samples) have no nuPlan `.db` on this box. Only
`2021.07.09.20.59.12_veh-38_03292_04331` (236 rows) and `2021.06.14.18.13.35_veh-26_04547_04710` (89 rows) are indexed by
`navtrain_scenarios.index_dbs(splits)`. Searched: `E:\Projects` and `F:\Archive` (-Depth 8), `D:\` and `G:` (absent), box 2 `D:\` (absent). The
local nuPlan is the VAL split (1,381 DBs, `arrange_splits.json`) plus the 2 DBs above; the missing logs belong to the nuPlan TRAIN split, so
they probably still sit on a pod volume or other box. The prereg (section 4) makes a run on fewer logs DESCRIPTIVE ONLY, and shrinking the
list is a new prereg. UNBLOCK = copy the 11 `.db` files (names in `tf2b_samples.txt`, column 1) into
`E:\Projects\TanitAD\data\nuplan\nuplan-v1.1\splits\` (about 27 MB each, ~0.3 GB total); or the PI declines and TF-2b stays unrun.

**Frozen.** `PREREG_TF2B.md` sha256 `b0ed43c2...a14b` (`PREREG_TF2B.sha256`), 2026-11-11T23:22:42Z. Also frozen by sha inside it:
`tf2b_samples.txt`, `tf2b_roll.py`, `tf2b_mknpz.py`, `tf2b_analyze.py`; `../tf2/tf2_score.py` is TF-2's, unchanged.

**Verified pre-row (TF-2 population only).** Roll script in control mode, 3 banked rows: max |dxy| 0.0 m vs the bank, 3.3 s/row (not 1.4).

**Exact next steps once the DBs are in place** (work dir `C:/Users/Admin/tf2b_scratch/w` holds refe/ eval/ code/ p5s/ exported from
mirror tip origin/agent/arch-inf-20260803, with tf2b_roll.py copied into eval/; re-export if missing). PY=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe,
`CUDA_VISIBLE_DEVICES=-1`, check FreeVirtualMemory > 10 GB, at most 2 processes.
1. C0: `$PY eval/run_p4.py --log <w>/c0.log --hz 10 --threads 1 --script tf2b_roll.py -- --control --p5s <w>/p5s/p5s_cubic.npz --out <w>/c0.json`
2. Rolls, 2 shards: `... --script tf2b_roll.py -- --samples <tf2b>/tf2b_samples.txt --shard 0/2 --out <w>/roll_s0.jsonl` (and 1/2). Resumable.
3. `$PY tf2b_mknpz.py --w <w> --rows <w>/roll_s*.jsonl --out <w>/p5s/p5s_cubic.npz.NEW --c0 <w>/c0.json`. Do NOT overwrite the banked p5s npz
   (tf2_score reads `<w>/p5s/p5s_cubic.npz`; use a second --w dir holding the new npz under that name, with refe/ eval/ symlink-free copies).
4. Score: `tf2_score.py --w <w2> --e2score E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/teacher_review/e2e1 --out <w2>/tf2b_s{0,1}.jsonl --shard i/2`,
   PYTHONPATH=`F:\Archive\devbox-C\navsim\navsim-3e8291bfa89ff247231e0227778840cd0a036896;<w2>\refe`. Control rows for C1: run tf2_score on the 30 C0 keys
   against the ORIGINAL npz (`--only`), compare with `../tf2/rows/*.jsonl`.
5. `$PY tf2b_analyze.py --rows <w2>/tf2b_s*.jsonl --ctl <c1 rows> --tf2 ../tf2/rows/*.jsonl --roll-ctl <w>/c0.json --out tf2b_result.json --prereg-sha b0ed43c2...`
6. Write `RESULT_TF2B.md` as pre-registered. WORKS => re-resolve `../PREREG_TF_N1_DRAFT.md` section 4 (candidate = CERT); else it stays Tc20s.
If the job exceeds ~1 h, write a fresh handoff and end.

**Deletion log.** `F:/tanitad_scratch/tf2b` (26 MB, own mirror-tip export, re-creatable; moved to C: because F: showed 20 GB free) deleted;
two smoke files in the C: work dir (tiny). `C:/Users/Admin/tf2b_scratch/w` is kept (re-creatable, about 30 MB) for the continuation.
