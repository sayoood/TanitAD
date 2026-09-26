# PLAN — W3: NAVSIM v1.1 `navtest` (runtime, metric cache, reference reproduction, frame bank)

Brief: BUILD_PLAN.md row **W3**. Pre-registration: `SPEC.md` (blob `48562058…`, staged
2026-09-19T13:41:38Z with `raw/` empty — `raw/SPEC_PREREG_HASH.txt`). Every heavy step runs at
**BELOW_NORMAL** priority, ONE worker, RAM floor 3 GB, and the devkit is never edited.

## Step order, and why this order

| # | step | why here | state |
|---|---|---|---|
| 1 | v1.1 runtime: the C: venv + `PYTHONPATH` → the D: v1.1 tree, asserting `navsim.__file__` | the cheapest thing that can fail; no new venv was needed (§Deviation 0) | ✅ |
| 2 | 20-token smoke: cache → C7 loader control → CV / HUMAN / STOP / CV-replicate | prices the full run and proves the harness BEFORE 12,146 scenes are paid for | ✅ |
| 3 | frame-bank smoke on shard 0 + **KB1** against E2's builder | the stitch must be proven bit-exact before 32 shards are processed | ✅ 20/20 |
| 4 | artifacts + four families + `criteria_check` on the smoke arms | the criteria gaps (ego-enforcement block, None-valued metrics) must be closed before the full artifacts | ✅ 0 violations |
| 5 | **full metric cache** (12,146 scenes, D:) | gates every number | ⏳ chain S1 |
| 6 | CV / HUMAN / STOP on the full split | BAR-W3-1/2 + the floors | ⏳ chain S2 |
| 7 | export (136 logs) → analysis → artifacts | per-log tables, C5, the STOP decomposition, the four families | ⏳ chain S3/S4 |
| 8 | frame bank, 32 verified shards | needed only by the model arms, which wait for a GPU gap anyway | ⏳ chain S5 |
| 9 | refcv4b on navtest | PI: our-model inference waits for a GPU gap | QUEUED (§Queue) |

`code/after_a8_chain.py` runs 5→8 unattended (detached, resumable, off-switch
`raw/chain_w3/STOP`), so the work survives this session.

## Deviations from the brief, and their evidence

**0. No new venv on D:.** The brief allowed one if the dependencies were incompatible. They are
not: the existing C: venv (py 3.9.25, torch 2.0.1+cpu, numpy 1.23.4, shapely 2.0.7, hydra 1.2.0)
satisfies v1.1's `requirements.txt`, and `nuplan-devkit` is at exactly v1.1's pin (`ce3c323` =
tag `nuplan-devkit-v1.2`). `PYTHONPATH` wins over the venv's own navsim 2.0.0 (a `.pth` appended
after site-packages) — and the wrapper ASSERTS it rather than trusting it. ⭐ The v1 **data** is
on D: as the PI ruled; only the interpreter is shared.

**1. The metric cache is built ONE LOG PER PROCESS** (`run_v1.py cache --per-log`), then one
final devkit pass writes the metadata CSV. MEASURED 2026-09-20: `cache_data` builds a SceneLoader
over ALL 136 logs before caching anything — 2.1 GB RSS, ~2 min — and a sibling agent's CPU
training spike (33.8 GB committed, system available → 419 MB) tripped the RAM guard and cost the
whole pass. Per log: ~0.4 GB, resumable, and a spike costs one log. ⛔ No devkit file is edited:
only `train_test_split.scene_filter.log_names` is overridden.

**2. The frame bank stores UNIQUE frames, not one file per scene.** navtest windows are 4
consecutive 2 Hz frames at stride 1, so a frame serves up to 4 tokens; D: is exFAT with **1 MiB
clusters**, so a per-scene `.npy` + `.src.npy` layout would cost ~36 GiB allocated and 24,292
files. The bank is one `.npy` per shard + one index + one `src` map per rig. **KB1 proves the
per-scene bytes are identical to E2's builder** (20/20 bit-exact, `raw/KB1_frame_bank_vs_e2.json`).

**3. Per-token rows are streamed to JSONL as the devkit produces them**
(`<label>_rows.jsonl`). The v2 runner scored all 5,912 navhard scenarios on 2026-09-19 and then
died in aggregation, losing the compute. `run_v1.py` can rebuild the CSV from that stream and
labels it `csv_reconstructed_from_rows_jsonl`.

## What each file does

| file | what |
|---|---|
| `code/navsim_v1_win.py` | the wrapper: import assertion (C8), E1's loader patch (C7), observation-only hooks, RAM guard, JSONL row stream |
| `code/run_v1.py` | driver + count guards: `select-smoke`, `cache [--per-log]`, `check-loader`, `score` |
| `code/w3_agents_v1.py` | `StopAgent` (the floor) and `SeamAgentV1` (model arms), v1.1 interfaces |
| `code/export_navtest_inputs.py` | the devkit's own AgentInput / human future / CV plan per token → D: |
| `code/analyze_navtest.py` | SPEC verdicts, C5, C6, the STOP decomposition, paired deltas |
| `code/artifacts_navtest.py` | TanitEval artifact + four families + `criteria_check` per arm |
| `code/build_navtest_frames.py` | the frame bank, per verified shard (extract → stitch → delete jpgs) |
| `code/bank_finalize.py` | KB2/KB3 over the whole bank (every token re-hashed) |
| `code/run_bridge_navtest.py` | refcv4b → seam file, E2's bridge functions + a device for the GPU gap |
| `code/after_a8_chain.py` | the unattended chain (S1…S5) with the training-process log and RAM waits |
| `taniteval/taniteval/bench/plugins/navsim_v1.py` | **W1's CLI plugin** (my exclusive slot) |

## Queue — refcv4b on navtest (blocked on the bank + a GPU gap)

```
python code/run_bridge_navtest.py --arms A1_ego_cmd,A2_vision_pure,A3_ego_nocmd,A4_blind_ego_cmd \
    --inputs D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz \
    --bank   D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/frame_bank \
    --out    raw/bridge_navtest --device auto --gpu-gap
python code/run_v1.py score --label A1_navtest --arm SEAM:raw/bridge_navtest/seam_A1_ego_cmd.npz \
    --cache-name navtest --patch-loader
```
Bar (pre-registered, SPEC §7): **A1 > max(STOP, CV)** paired on the identical tokens; stretch:
A1 > the published Ego Status MLP (65.6 / 66.4).
