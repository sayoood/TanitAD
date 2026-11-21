# PREREG G-BOX-OVERFIT, ADDENDUM NT: the frozen frame set for the NT launch (profile `refcv8-nt`)

**Registered:** 2026-10-22, about 20:50 Berlin, BEFORE any NT box-overfit number exists. No G-BOX-OVERFIT run has been made on NT data.

**Parent protocol:** `stack/ops/refcv8_box_audit/raw/PREREG_G_BOX_OVERFIT.md`, md5 **`594c71196cc5bbd527fee40b2cb0e3f1`**. This addendum changes ONE thing, the frame set. Everything else is the parent's.

**Changing any literal below after a run is a goalpost move.** It needs a dated amendment with the reason, and the run before it stays on the record.

## 1. Why

The NT launch (H-NT-DECIDE-1 A1, `stack/ops/runs.d/nt-a1-s0.argv.json`) trains on the NavSim token-episode cache `/home/nvidia/r8ns/nt/v2ep_a1`. The parent's 16 frames are PhysicalAI train4572 clips. None of them is in that cache.

MEASURED on Thor, bind `701922c66a4d`, `box_closure.json`: the harness exited 1 with `[gbo] 16 frame-set clips are not in the TRAIN cache`.

The parent tests the box path "AS BUILT AND CONFIGURED FOR THE LAUNCH" on real TRAIN frames. For the NT launch, those frames must come from the NT TRAIN corpus.

## 2. What is UNCHANGED

These are exactly as implemented in `stack/scripts/g_box_overfit.py` for refcv8:

- the parent's §3 (model, trainable sets, loss), with A13's launch optimiser, A17's schedule and A19's MAIN-only binding;
- §4 (the scoring rule, the IGNORE / DontCare handling), with A10's reconciliation that DROPPED rows count as IGNORE;
- every §5 PASS bar, literally:
  - AP@2 m BEV ≥ 0.90;
  - precision ≥ 0.90 AND recall ≥ 0.90 at the declared gate;
  - |Σ confident − N_POS| / N_POS ≤ 0.10;
  - Hungarian median centre error ≤ 0.30 m, median |Δl| + |Δw| ≤ 0.30 m, median |Δz| ≤ 0.15 m;
  - greedy-TP class accuracy ≥ 0.90;
  - loss finite at every step, and presence(2,000) ≤ 0.25 × presence(0);
- §6 (the must-fail arms and controls C1 to C4) and §7 (what a result means);
- N = 2,000 steps, batch 4, seed 0;
- the selector: `…/2026-09-26-box-head-audit/code/gbo_select.py` md5 **`f9f93f92b60a979b581b8f78be2ec6db`** and `vis_rules.py` md5 **`99cd007c0fd4e539ca38fa299a948fbf`**, run BYTE-IDENTICAL. Its rule is the parent's §2 rule.

## 3. What CHANGES: the frame set and its count literals

| item | refcv8 (parent) | **refcv8-nt (this addendum)** |
|---|---|---|
| file | `raw/visibility/gbo_frameset.json` | `raw/visibility/gbo_frameset_nt.json` |
| md5 | `b291404c36f83c3e397e3b90367e8e7b` | **`bb60278ce8d763cc5151cbcc1cb32735`** |
| frames | 16 | **16**, all at window t = 0 (NOW = 7) |
| POSITIVE (N_POS) | 113 | **122** |
| IGNORE after A10 (N_IGN) | 77 (28 IGNORE + 49 DROPPED) | **108 (58 IGNORE + 50 DROPPED)** |
| criterion 3 | \|Σ conf − 113\| / 113 ≤ 0.10 | **\|Σ conf − 122\| / 122 ≤ 0.10** |
| C1 | Σ confident = 113 | **Σ confident = 122** |
| C3 | 113 / 77, per frame, md5 | **122 / 108, per frame (POS; IGN + DROP), md5 `bb60278c…`** |

## 4. How the frame set was made (labels only)

The selector's INPUT was built by `stack/ops/refcv8nt_box_audit/code/gbo_nt_input.py` (md5 `2dfeffb77e6e79ed54dcfc793fa94a06`). The selector itself was not touched.

**Pool.** The parent's input pool rule is `bha_dump.py --set train`, applied verbatim to the NT launch's `--v2-cache` manifest:
- clip ids sorted by sha12, every k-th, the first 64;
- per clip, `evenly(4)` over the AGENT-LABELLED windows.

On NT this gives:
- 99,400 clips, k = 1,553, 64 clips from 60 logs;
- one window per clip. The launch's `--window-starts first` keeps exactly ONE window per token-episode, t = 0 (`restrict_window_starts`), so the pool is **64 windows**, all agent-labelled (refcv8: 256).

**Held-out.** The pool passed the launch's own K3 guard:
- 0 of the 64 clips' logs is in `nt_heldout.json` (sha256 `4b5b1aee…`, 174 held-out logs);
- 0 token is a selection-scan sample token;
- eval139 lives in the separate `--eval-cache` and cannot enter the pool.

**Rows.** The rows come from the HARNESS's own data path, `g_box_overfit.TrainerAdapter`. It builds DATA ONLY: no model is built and no checkpoint is opened.
- The trainer's `V3Dataset._agent_item(ep, t + 7)` target block is split by `vis1.vis1_split`.
- `target` is the trainer's `visible_target_filter`.
- `vis_frac` = n_vis / n_full, in float64, from the launch's VIS-1 sidecar `nt_train_plus_eval139_vis1_sidecar.npz` (sha256 `49cd78da…`).
- That sidecar's pixel counts are the audit's own z-buffer of the GT cuboids, `vis_zbuf.py` lines 39-144 (md5 `3721a83f…`), on the NT stitch camera's observed mask.
- An unknown visibility reads NaN, which `vis_rules.label` calls DROPPED and the harness counts as IGNORE: the same row on both sides.

The resulting selector input is `raw/visibility/vis_train_boxes_nt.json`, md5 `84eafdd79f84ff4d022c4fb69b06f6e3`: 2,965 GT rows, 818 in the target filter.

**No model output was read.**
- The adapter's `sys.addaudithook` trace (`raw/visibility/gbo_nt_input_record.json`, `opened_files`) lists:
  - the launch argv and the held-out guard;
  - the NT label files: token table, extrinsics, v7 labels, clip clock, 2-D and 3-D agent joins, road-type sidecar, v9 labels, VIS-1 sidecar;
  - the join-defect mask config;
  - the cache manifest and the 64 pool `.v2ep.pt` payloads.
- The payloads are decoded by `__getitem__`, but no pixel-derived quantity enters a row.
- No checkpoint, prediction or model file appears.
- The selector's own trace (`raw/visibility/gbo_nt_selector_trace.json`, written by `code/run_frozen_selector.py` md5 `2b1c087715db4a97e1701db34540e45c`) shows it opened itself, `vis_rules.py`, its one input file, and its output.

**Reproduced three times:**
- two independent adapter runs on Thor gave byte-identical inputs (`84eafdd7…`) and frame sets (`bb60278c…`);
- the selector re-run on the dev box from the banked input printed the same md5.

## 5. The result

- 44 eligible clips; 16 frames.
- **122 POSITIVES:** automobile 69, person 52, rider 1.
- **58 IGNORE + 50 DROPPED = 108**, all IGNORE after A10.

**Class coverage under the rule's own clause** ("a group with fewer than 3 available is taken whole"):
- **large vehicle has 0 available.** The NT class map sends every vehicle to `automobile`.
- **rider has 1 available,** taken whole: `e4322aedb9a0`.
- **person has ≥ 3.**

| clip sha12 | window t | POS | IGN | DROP | positives by class |
|---|---:|---:|---:|---:|---|
| `0000197af212` | 0 | 3 | 0 | 0 | automobile 1, person 2 |
| `07b5bc10a928` | 0 | 12 | 12 | 13 | automobile 5, person 7 |
| `0bb3a9b2e306` | 0 | 4 | 6 | 6 | automobile 3, person 1 |
| `13d1dc8e6ebf` | 0 | 9 | 4 | 5 | automobile 9 |
| `17ce6715044c` | 0 | 8 | 1 | 0 | automobile 6, person 2 |
| `1fc1c33c305e` | 0 | 11 | 3 | 4 | automobile 5, person 6 |
| `27ca69f81d07` | 0 | 7 | 5 | 0 | automobile 4, person 3 |
| `3004bf3a0a5e` | 0 | 8 | 8 | 2 | automobile 2, person 6 |
| `3c27a8531e42` | 0 | 4 | 3 | 0 | automobile 2, person 2 |
| `480e8dc4bd37` | 0 | 8 | 1 | 2 | automobile 3, person 5 |
| `4c2d96df2e2a` | 0 | 5 | 1 | 1 | automobile 5 |
| `501f5b1e5087` | 0 | 8 | 0 | 2 | automobile 5, person 3 |
| `543af0cdb6c4` | 0 | 7 | 0 | 0 | automobile 7 |
| `5837cf2a9038` | 0 | 3 | 0 | 0 | automobile 3 |
| `5c335a287a60` | 0 | 10 | 2 | 0 | automobile 7, person 3 |
| `e4322aedb9a0` | 0 | 15 | 12 | 15 | automobile 2, person 12, rider 1 |

## 6. Known defects, stated rather than fixed

- **The frame set's `split` field is WRONG for NT.** It reads `TRAIN (bha_dump train set: refcv6-b1-416x1024-train)`. That is a string literal inside the frozen selector, so it cannot be corrected without editing either the selector or its output.
  - The true split is the NT TRAIN cache `/home/nvidia/r8ns/nt/v2ep_a1`, as recorded in `gbo_nt_input_record.json`.
  - The `source` / `source_md5` fields ARE correct.
- **The harness needed two NT data-path repairs, beyond the frame set.** Both are in `g_box_overfit.TrainerAdapter._build_items`, and both are inert for an argv without the flag, so refcv8's path is unchanged. Both were MEASURED on Thor, 2026-10-22:
  1. `--r8-speed-unknown-always`: the vendored loader's `enable_max_speed_v6` refuses the all-UNKNOWN road-type sidecar. The harness now attaches it the way `train()` does, through `enable_r8_speed_label(…, all_unknown=True)`.
  2. `--window-starts first`: the loader built 40 windows per token-episode, and the v9 join refused `k = 10`. The harness now applies `restrict_window_starts` at dataset construction, as `train()` does, through the loader's `dataset_cls` hook.
- **This addendum makes NO claim about NT box performance, generalisation or visibility.** It is the parent's capacity test on NT TRAIN frames.
