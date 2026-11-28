# RULE: the registered TRAIN139 clip and window rule for the L7 scorer bank

* **Author.** Architecture & Inference agent (l7cap), dev box, 2026-10-22, clock read 00:10 +0200 (Berlin).
* **Status.** PROPOSED for registration under `H-L7-1` (`PREREG_L7_SCORER.md` sec. 2, "PILOT", sealed sha256 `f9c1431c...`). Nothing here edits that prereg.
* **Evidence classes.** MEASURED = produced by `code/build_train139_windows.py` on 2026-10-22; INHERITED = read from the cited artifact.

## The rule (it is not new; it is the RETIME train-fit / refcv6-A6 sample, reproduced and written down)

| part | definition | source |
|---|---|---|
| **clips** | the **139 smallest `sha12(clip_id)`** among train-cache clips that are not in eval139 and that have a train label record AND a train max-speed sidecar row (pool 4,369). `sha12(x) = sha256(x.encode()).hexdigest()[:12]` | refcv6 A6 `select_record.json`, 2026-09-26 (INHERITED); re-derived here (MEASURED) |
| **windows** | for each selected clip, every `V3Dataset` index entry `(e_i, t)` with `t % 5 == 0`. `t` runs over `range(T - W - 20)` (`tanitad/data/_contract.py:121`, `W = 8`, `max_horizon = 20` stacked rows). `ws = t + W - 1` | `roll_seed_r8_train.py` (the RETIME train-fit roll); re-derived here from the cache manifest alone (MEASURED) |
| **seeds** | two inference seeds, 0 and 1, on the SAME windows | `PREREG_L7_SCORER.md` sec. 2 |

Counts: **139 clips, 4,855 windows** (33 to 35 per clip). Window file: `TRAIN139_WINDOWS.json`, sha256 `5ed3283fbbca2fde3e226f0a13bb85b0c41d20c786db3a834ee77510bd20f774`
(`TRAIN139_WINDOWS.json.sha256`). It is keyed by `clip-sha12-<12 hex>` only. It contains no clip id and no 8+ hex prefix of one
(checked by the builder: 0 leaks; the sha12 values themselves are hashes, not id prefixes).

## What was checked (all MEASURED, `TRAIN139_VERIFY.json`, `smoke/p2_build.log`)

| control | result |
|---|---|
| **K-CLIP** the rule, run on the pool list + label ids + sidecar ids + eval139 ids, reproduces the banked `train_clips_sha12.json` | **equal** (139/139) |
| **K-CLIP** the same set equals the file names in the dev-box A6 cache dir | **equal** (139/139) |
| **K-DISJOINT** intersection with eval139 by sha12 | **0 of 139** |
| **K-WIN** windows re-derived from the cache manifest (episode length T per clip) equal the windows the banked train-fit roll actually rolled (`trainfit/step30000ema_train139_pf/dump_s0`) | **equal on all 139 clips, 4,855 = 4,855** |
| K-WIN as first coded used `range(T - W - 20 + 1)` and read 4,861 vs 4,855 on 6 clips | **FAILED as coded, fixed** to `range(T - W - 20)` (the dataset's own bound); the failure is the reason the bound is quoted from `_contract.py:121` |
| the pool list is Thor's `data/refcv6_train_clips.txt` (md5 `bb0e2263...`, 4,369 ids), read over `ssh cat`; the ids stayed in the session scratch and are not banked | |

## Window grid versus eval139's per-frame windows (MEASURED, same builder)

| | eval139 (`s2_windows.json.sha12.json`) | TRAIN139 |
|---|---|---|
| stride | 5 ticks (0.5 s), every consecutive difference is 5 | 5 ticks |
| `ws` convention | `ws = t + W - 1`, `W = 8` | same |
| per-clip phase `t mod 5` of the first window | **varies by clip** (all five phases occur) because the eval list comes from the banked refcv4b grid | **0 on every clip** (A6 / RETIME precedent) |
| windows | 4,754 over 139 clips (34.2 per clip) | 4,855 over 139 clips (34.9 per clip) |

So the **stride and the horizon rule are the same; the phase is not**. Aligning the phase would need a different, unregistered rule and a different roll; it is
not needed for a scorer bank (the scorer sees one window at a time), and it is stated here so nobody assumes more.

## Parity and hygiene

* **Parity.** This is a scorer bank. The canonical training corpus (`physicalai-train-e438721ae894`, 2,376 episodes, skip-hash `f09e44db`) is neither read nor re-selected.
  The sample draws only from clips the model was already trained on (all 4,572 train clips), so it changes no corpus.
* **In-sample caveat (inherited from `DESIGN.md` sec. 3.4).** The generator was trained on these clips, so their fans are closer to GT than eval fans. The pilot is the
  place to measure that: report the pick-to-oracle gap on train139 beside eval139's 0.215 m. It is monitored, never tuned on.
* **Clip-id rule.** No banked file here holds a full clip id or an 8+ hex id prefix. `TRAIN139_WINDOWS.json` and the capture's
  `registered_windows_check` use `clip-sha12-<sha256(id)[:12]>`. The Thor cache builder derives raw ids on Thor from Thor's own pool list and prints counts only.
* **Regenerate.** `python code/build_train139_windows.py --pool-list <refcv6_train_clips.txt> --out-dir <dir>` (needs the dev-box A6 cache manifest and the banked trainfit dump).

## Scale-up to the full bank (NOT registered here; a proposal for the Master Mind)

`PREREG_L7_SCORER.md` sec. 2 asks for train4572 at 1 window in 4. The same rule extends without a new degree of freedom: rank every train clip
(label + sidecar present, not in eval139) by `sha12`, cut into **shards of 139 clips** (rank `139k .. 139k+138`; shard 0 is TRAIN139, so the pilot is a strict
subset), and use `t % 20 == 0` (a 2 s stride; or keep `t % 5 == 0` and thin by `ws`). One shard is one capture job. A shard is what fits one process: the scene bank is
about 0.3 MB per window (see `RESULT.md`), so a 139-clip shard is about 1.5 GB per seed and the full 2 x 39 k plan is about 23 GB, which cannot sit in one process's RAM.
