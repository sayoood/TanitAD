# snap40k -- the step-40,000 EMA snapshot, built and tested against a fake Thor, NOT started

EvalFlyWheel, 2026-10-18. Single copy of this package: `C:/Users/Admin/qland/hold_r8batt/stage/.../2026-10-15-refcv8-milestone-battery/` (not staged, not committed).
**Thor was touched read-only, three times** (`ssh -n`: `ls`, `grep`, `stat`, `tail`, `date`, `df`); no process started there, nothing written, no checkpoint loaded.

## Why a snapshot at all

The trainer writes `ckpt_<N>.pt` / `ckpt_ema_<N>.pt` only at `MILESTONES = (5000, 15000, 20000, 30000)` (`refc_v3_train.py:232`). `ckpt.pt` (1,560,403,737 B, MEASURED on Thor 20:17) is rewritten
IN PLACE every 500 steps (`torch.save(..., ck)` at :10815, not tmp+rename) and the line `[v3:hier] ckpt step N -> ckpt.pt` is printed AFTER the write (:10832). So the line for 40000 proves the file is complete at
that moment, and the next write (step 40,500) destroys it. **Timing, MEASURED:** the 32,000 file landed at 20:17:27 and 31,500 at 18:54:48, i.e. 1 h 22 min per 500 steps ->
**step 40,000 at ~2026-10-19 18:20 (+-), step 40,500 at ~19:42: a ~1.4 h window.** The 5-minute poll and a ~1-3 min copy fit inside it with a wide margin.

## Commands (exact; none was started)

```
# 1. arm the snapshot (CPU/IO only; no lock needed; read-only on Thor).  Polls every 300 s; gives up after 60 h.
nohup sh C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code/snap40k_r8.sh > E:/refcv8_eval_kit/chain/snap40k.out 2>&1 &
#    markers (opaque): E:/refcv8_eval_kit/chain/snap40k.log
#      ZZS40STARTZZ -> ZZS40WAITZZ (every 12 polls) -> ZZS40SEENZZ -> ZZS40DONEZZ        success; E:/refcv8_eval_kit/ckpt/ckpt_full_40000.pt + .md5 + .provenance.json
#      ZZS40ABORTLATERZZ  a later ckpt line was already in train.log (40,000 is gone)      ZZS40TOOLATEZZ / CHANGEDZZ  it moved during the copy
#      ZZS40MD5MISMATCHZZ / SIZEMISMATCHZZ / INCONCLUSIVEZZ / BADSIZEZZ / NODISKZZ         ZZS40SSHFAILZZ / SSHDEADZZ (24 consecutive failed probes = 2 h) / TIMEOUTZZ
#    exit codes are NOT evidence: read the markers and the files.

# 2. CPU, after ZZS40DONEZZ (needs >= 4 GB free commit; refuses below it; ~2.5 GB peak; no GPU):
set PYTHONPATH=C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval
python -X utf8 code/extract_ema_r8.py extract --src E:/refcv8_eval_kit/ckpt/ckpt_full_40000.pt --config E:/refcv8_eval_kit/ckpt/config.json \
       --out-dir E:/refcv8_eval_kit/ckpt --expect-step 40000 --provenance thor_md5=<the md5 in ckpt_full_40000.pt.md5>
#    -> ckpt_40000.pt, ckpt_ema_40000.pt, ckpt_40000.provenance.extract_ema.json ; prints one JSON line (ok / problems)
```

## What each check asserts

| check | asserts | refuses when |
|---|---|---|
| probe (remote) | prints ONLY an opaque `ZZ<last step>ZZ`; the searched text never reaches the output stream (CLAUDE.md echoed-pattern trap); tested against the REAL Thor read-only: `ZZ32000ZZ` | no token = a counted failure, never "no checkpoint line" |
| snapshot accept | remote `stat` (mtime+size) before and after the scp identical; size == 1,560,403,737 on both ends; remote md5 == local md5 (both 32 hex chars, else INCONCLUSIVE); train.log still shows no ckpt line beyond 40000 | any of those -> the `.part` is deleted, no file under the final name |
| `extract_ema_r8.py extract` | the loader's strict build on the 40k weights; the REAL `WeightEMA(model, run decay)` loads `ckpt['ema']` (key set, shapes, decay, `num_updates == step == 40000` all enforced by the class); the pair is written by the trainer's OWN `_save_ema_model`; the EMA payload equals `slim_rolling_ckpt_r8.derive_ema_model` on EVERY key (two independent derivations); EMA differs from live on >= 1 float tensor; tied-storage partition preserved | any disagreement -> `ok:false` with the problems listed |

⚠ **`ck['step'] == 40000` inside the file is checked only by `extract` (`--expect-step`), not by the snapshot script** (no python on Thor, no 1.5 GB `torch.load` in a poll loop).

⚠ **A trap `extract` works around (pinned by a regression arm):** `refcv7_loader.build_model` ends with `requires_grad_(False)` on every parameter, and `WeightEMA` shadows only TRAINABLE parameters + float buffers, so
`WeightEMA(loader_model, decay).load_state_dict(ckpt['ema'])` raises (key sets differ: on the tiny model "covers 9 tensors, this model 3"). The trainable pattern is restored from the checkpoint's own shadow keys
(per object, so a tied parameter is trainable if ANY of its names is shadowed), and the rebuilt shadow's key set must then equal the checkpoint's.

## Known-value controls that ran (CPU)

* `extract_ema_selftest.json` -- a tiny model TRAINED with a real `WeightEMA` (tied parameter, frozen layer, float + int buffers): the file `extract_core` writes (with the trainer's own `_save_ema_model`) equals
  `WeightEMA.model_state_dict` of the objects that trained, key for key; shadowed keys equal the shadow; frozen / int keys equal the live value; the tied twin equals its twin; 14/14 checks incl. four
  deliberate-regression arms (no requires_grad restore, wrong decay, EMA not at the step, a corrupted shadow -> reads RED against the training EMA). Mutation: deleting the restore kills the selftest.
* `check_real_30k_pair.json` -- on the REAL `ckpt_30000.pt` / `ckpt_ema_30000.pt` (md5 in the file): 1,165 keys in both, identical key set, shapes and dtypes; EMA meta keys == the trainer's writer
  (`model, step, ema_decay, num_updates, kind`), `kind == ema_model_only`, step == num_updates == 30000, decay 0.9998 == `config.json['ema']['decay']`; **821 tensors differ from live, all float** (the same COUNT, 821, as
  the banked 5,000 / 10,500 pairs), every non-float tensor bit-equal; tied-storage partition equal. So the set of keys `extract` writes for 40k is, by construction, this set.
* `pytest_test_snap40k_r8.log` -- 11 passed against a FAKE Thor (stub ssh = `sh -c` locally, so the real probe/stat/md5 pipeline runs; stub scp can corrupt a byte, add a byte, touch the source mid-copy, or let the
  next ckpt line appear mid-copy): waits until the 40000 line appears; later line -> ABORT; each corruption -> refused and nothing under the final name; dead ssh -> loud; already done -> no-op; a static guard that no remote
  command writes/kills/starts anything.
* `pytest_test_extract_ema_r8.log` -- 5 passed (selftest with the local writer and with the trainer's own; mutation; tied-parameter restore; `check_real_pair` regressions).

## UNVERIFIED until a real full `ckpt.pt` is on the dev box

`extract_ema_r8.py extract` end to end on a real 1.56 GB file (none is on the dev box; the selftest + the real-pair key-set assertion are what exist); the `build_model` strict load of the 40k weights; the peak memory
(~2.5 GB estimated); the actual scp speed over `tanitad-thor-wifi` (the LAN is ~63 MB/s MEASURED earlier; wifi may be slower; even 5 MB/s is 5 min).

## Files (all NEW, in `code/`)

`snap40k_r8.sh` `extract_ema_r8.py` `test_snap40k_r8.py` `test_extract_ema_r8.py`; here: `extract_ema_selftest.json`, `check_real_30k_pair.json`, `pytest_test_snap40k_r8.log`, `pytest_test_extract_ema_r8.log`.
