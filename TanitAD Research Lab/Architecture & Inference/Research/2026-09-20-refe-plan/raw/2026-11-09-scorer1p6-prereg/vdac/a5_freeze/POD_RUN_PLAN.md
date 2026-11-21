# POD_RUN_PLAN: VDAC A5 pair (HB-s0, HB-SHUF, HC-s0, HC-SHUF) on the L40 pod, inside P7 pauses

Companion to `PREREG_VDAC_P6_A5.md`. Every command below exists in the code at `a01be4a4`; `file:line` is quoted for each stage name and flag. No IP, port, pod id or token appears in this file (the pod address is whatever the MM's usual access uses; the HF token file is resolved in place by the scripts).
`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. `<LANDED>` = the 40-hex commit the MM lands the card at (its `refe/` blobs must equal `a01be4a4`'s).

## 1. Which pod directory (decision, with the reason)

**Ship to a FRESH `/workspace/refe-plan-vdac`** (the old tree of the same name is first moved aside to `/workspace/refe-plan-vdac.s1a_tree`). Reasons, all MEASURED from the code:
* `vdac_pause_p7.sh:62` **hard-codes** `bash /workspace/refe-plan-vdac/refe/vdac_chain.sh`; a differently named directory would run the OLD chain unless that pinned file is edited (PREREG section 9.6). `vdac_chain.sh:36` (`PKG=${VDAC_PKG:-$WS/refe-plan-vdac}`) accepts another path, the pause script does not.
* The directory holds the S1a-era tree, and `gate_token.TREE_ROOTS = (refe, code, splits)` (`gate_token.py:40`): changing `refe/` inside it voids any token minted against the old tree. **Re-minting is intended**: the A5 arms have new names and argv, each arm mints its own token in its own `gate_<arm>` stage (`vdac_chain.sh:287`), and the HA tokens belong to arms that are finished (HA-s0, HA-SHUF trained in S1a; HA-s1 never runs, because HA-s0 missed S1-GO). The tokens live in `/workspace/data/refe_vdac/lg_<arm>/`, **outside** the package directory, as does the HMAC key (`vdac_chain.sh:58`, `TANITAD_GATE_KEY_FILE=$ROOT/launch_gate.key`), so moving the package aside destroys no token and no key.
* A FRESH directory (not a copy laid over the old one) is required: `tree_manifest` compares every `.py .sh .yaml .yml .json .txt .patch` under `refe/ code/ splits/` with the manifest (`gate_token.py:41,746-760`, chain precheck `vdac_chain.sh:164-175`); a stale file left over from the old tree is an `extra` and refuses the chain.
* **Never** `/workspace/refe-plan-p7` (P7's own package; the pause script resumes P7 from it, `vdac_pause_p7.sh:32`) and never `/workspace/refe-plan-stage0` (the Stage 0 package; `vdac_stage0_pause_p7.sh:20`).
* The old moved-aside tree is re-creatable from git, so it may be deleted later by its literal path under the auto-delete rule once nothing needs it; log the size. Do not delete it in this run.

## 2. What to ship (the whole tree, not a hand-picked list)

A 2026-11-12 Stage 0 pause failed because a lazily imported file was not shipped. So: **the whole of `refe/`, `code/`, `splits/` from `<LANDED>`, plus the six `eval/` files the VDAC suite reads outside those roots.** MEASURED by the static scan `refe/p7_ship_list.py derive --profile refe_vdac_p6` on an export of `a01be4a4` (146 files, 157 closure scripts, closure digest `843e1b6fccc8779cafa7f6a68a25607ff9f37de61e61ec849ac640d8d0f86fbc`; the result is in this hold as `vdac_ship_list_a01be4a4.json`; the static scan is an over-approximation, the traced run was not made, so shipping the whole of the three roots is the safe form):

| outside the three roots | read by |
|---|---|
| `eval/harness_pins.py` | `refe/test_ego_yaw.py` |
| `eval/rls1b_arms.json` | `refe/test_launch_gate.py` (the G-SUITE member, `--static --no-banked`), `test_hn_content_fp.py`, `test_hn_ctx_anchor.py` |
| `eval/rls1d_arms.json`, `eval/rls1e_arms.json` | `refe/test_launch_gate.py` |
| `eval/P2REP_POD_RUNBOOK.md`, `eval/S1_POD_RUNBOOK.md` | `refe/test_p2rep_chain.py`, `refe/test_s1aux_chain.py` |

The chain's own `NEED` list (`vdac_chain.sh:158`) names the three NEW files (`refe/aux_hb.py`, `refe/vdac_probes.py`, `refe/test_vdac_hb.py`) but **not `refe/vdac_a5_layer.py`**, which `vdac_layer.py` imports (the peel through the A5 layer, `test_vdac_gate` V7): shipping the whole `refe/` is what covers it.
**SHIP_MANIFEST changes for the 3 new files: none by hand.** `launch_gate.py ship-manifest` WALKS the shipped tree (`gate_token.py:745-760`) and records every file, so `refe/aux_hb.py`, `refe/vdac_a5_layer.py` and `refe/test_vdac_hb.py` enter the manifest by being in `refe/`; it refuses any file that is not its blob at `<LANDED>` (`gate_token.py:768-785`). The tracked `SHIP_MANIFEST.json` at the root of PK is a stale, unrelated copy: do not ship it; write a new one.

## 3. Sequence

### Step A (dev box, CPU): export, manifest, pod-side check list
```
git -C C:/Users/Admin/tanitad-push ls-remote origin <branch>          # equals the local tip, BEFORE building (a stale remote-tracking ref is a cached claim)
S=F:/tanitad_scratch/a5ship; mkdir -p "$S"
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive <LANDED> "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe" "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/code" "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/splits" "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval" | tar -x -C "$S"
P="$S/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
C:/Users/Admin/venvs/tanitad/Scripts/python.exe "$P/refe/launch_gate.py" ship-manifest --pkg "$P" --commit <LANDED> --git-dir C:/Users/Admin/tanitad-push/.git --repo-prefix "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/" --out "$P/SHIP_MANIFEST.json"
```
(`launch_gate.py:2254-2259`; success line `ZZSHIP_OK <n> files tree <16 hex> commit <12 hex>`.) Then prune `$P/eval` to the six files of section 2 (the tree manifest covers only `refe/ code/ splits/`, so pruning `eval/` does not touch it; shipping all of `eval/` is also acceptable, 17 MB). Pack `refe code splits` + the six eval files + `SHIP_MANIFEST.json` as one tar with LF files (a CRLF kit runs Python but a bash script dies: `thor-kit-export-autocrlf`); verify one `.sh` blob hash and `bash -n refe/vdac_chain.sh refe/vdac_pod.sh refe/vdac_pause_p7.sh` on the export.
Copy `vdac_ship_list_a01be4a4.json` with the tar (it is valid only while `refe/` equals `a01be4a4`'s; if `<LANDED>` changed any `refe/` blob, regenerate with `python refe/p7_ship_list.py derive --pkg "$P" --out <list> --profile refe_vdac_p6`).

### Step B (pod, no GPU): place the tree and the frozen cards
```
mv /workspace/refe-plan-vdac /workspace/refe-plan-vdac.s1a_tree        # skip if it does not exist
mkdir -p /workspace/refe-plan-vdac && tar -xf <tar> -C /workspace/refe-plan-vdac
python /workspace/refe-plan-vdac/refe/p7_ship_list.py check --pkg /workspace/refe-plan-vdac --list <vdac_ship_list_a01be4a4.json>      # ZZP7SHIP_OK files=146, in seconds
```
Then put the card and its amendments in the data root and re-hash (the chain checks only the card, A1, A2 by bytes: `vdac_chain.sh` precheck loop, so this step is the MM's own check of A3, A4, A5): `/workspace/data/refe_vdac/PREREG_VDAC_P6_A5.md` with `sha256sum` equal to `<CARD_SHA256>`.
**Do NOT run a `pytest` or the suite on the pod's CPU while P7 is training** (never add load to a training pod); the pod-side tests run inside the pause (step D).

### Step C (dev box, CPU): the tests the build agent could not run (see PREREG 6.2)
Run where `D:\Projects\TanitAD\data` exists: `refe/test_launch_gate.py` in full, then the ~69 launch_gate-reading tests (`pin-edit-moves-profile-fingerprints`: six profile fingerprints pinned in three tests), `refe/test_vdac_gate.py`, `refe/test_vdac_hb.py` (expected `ZZTEST_PASS 133`), `refe/test_aux_s1_wiring.py` (`ZZTEST_PASS 20`), `refe/test_dump_scores.py` (`ZZTEST_PASS 16`), with `S1_READ_V1=<frozen s1_read.py>` for the Stage 0 tests. A failure here blocks the ship.

### Step D (pod, GPU, in the P7 pause): the launch-gate token and the arm
**The token is minted by the chain, inside the pause, per arm, by `launch_gate.py run --profile refe_vdac_p6 --arm-name <ARM>` (`vdac_chain.sh:287`), after the arm's own preflight** (G12 tap, G19, G20, G21, G-LIVE, G-CKPT, the exclusion audit and the NOFILTER and EGO mutants, all on the GPU) and **after the live G-SUITE** (`VDAC_SUITE`, `launch_gate.py:922`: `test_rl_reward`, `test_rl_epo`, `test_aux_drivable_head`, `test_aux_s1_wiring`, `test_launch_gate --static --no-banked`, `test_s1c --no-e2e`, `test_vdac_gate`; `test_vdac_hb.py` is NOT a member). The token lands in `/workspace/data/refe_vdac/lg_<ARM>/PASS_*.json` and is verified again at train start (`token_ok`, `vdac_chain.sh:301`). It is never hand-written. Profile: **`refe_vdac_p6`** (`vdac_chain.sh:27`).
`launch_gate.py suite-prerun` could take the suite off the GPU clock, but the chain does not pass `--suite-evidence` (`vdac_chain.sh:287`), so a pre-run would not shorten the pause: not used.
**Find the two PIDs P7 needs** (no `pgrep -f`, which self-matches; the pause script re-checks both `/proc/<pid>/cmdline`, `vdac_pause_p7.sh:24-25`):
```
ps -eo pid,args | grep -E '[p]7_pod\.sh|[t]rain\.py'
```
Launch, detached (header of `vdac_pause_p7.sh`: "Runs on the pod, detached"; ONE stage per pause; stage names `vdac_pause_p7.sh:18-19,21`):
```
setsid nohup bash /workspace/refe-plan-vdac/refe/vdac_pause_p7.sh <P7_POD_SH_PID> <TRAIN_PY_PID> A5HB >> /workspace/data/refe_vdac/pause.out 2>&1 < /dev/null &
```
The script: waits for P7's next `ckpt_last.pt` (up to 60 min, `vdac_pause_p7.sh:36-38`), stops P7 by explicit PID, confirms the GPU empty, runs the stage's arms one chain each (`:56-66`), and ALWAYS resumes P7 (EXIT trap) unless `/workspace/data/refe_vdac/HOLD_P7` exists. Terminal lines in `/workspace/data/refe_vdac/pause.log`: `PAUSE_STOPPED_EARLY`, `P7_RESUMED`, `P7_HELD`; per arm `chain <ARM> rc <n>`; the arm's own log `/workspace/data/refe_vdac/chain_<arm lower>.log` ends `ZZVDAC_CHAIN_DONE <ARM>` on success and carries `REFUSED: stage=` or `ZZVDAC_CHAIN_FAIL` on failure (`vdac_chain.sh:13`).
Watch with a script or a Monitor, never a model turn (the TOKEN BUDGET rule); a watcher's filter must not contain the pattern it searches for.

| order | stage argument | arm | chain stages (`vdac_chain.sh`) | L40 minutes (ESTIMATED; draft section 5) | pause timeout (`vdac_pause_p7.sh:59`) |
|---|---|---|---|---|---|
| 1 | `A5HB` | `HB-s0` | precheck, inputs, egr, dd, preflight (+ exclusion audit, NOFILTER mutant, G19/G20, EGO mutant), gate (suite + token), train 10,000 updates, run check, post-training gate, export (once per tree, ~6), egress | train 264 to 456 + ~33 chain overhead | 36,000 s |
| 2 | (same stage, next arm) | `HB-SHUF` | same, 2,000 updates | 53 to 91 + ~33 | 14,400 s |
| 3 | `A5HC` | `HC-s0` | same, 10,000 updates (5,120 queries) | ~258 + ~30 | 21,600 s |
| 4 | (same stage, next arm) | `HC-SHUF` | same, 2,000 updates | ~52 + ~30 | 7,200 s |
| | | **total** | | **~12.7 h lower, ~16.5 h upper** (the PI's cap 16.7 includes the finished Stage 0) | |

If an arm's chain does not end in `ZZVDAC_CHAIN_DONE <ARM>`, the stage's remaining arms are NOT run and P7 resumes (`:65`): a failed HB-s0 means HB-SHUF is not run; its row is NOT RUN and HC proceeds in its own pause.
Per-arm facts (all argv are `vdac_pod.sh ARM=<ARM> MODE=argv`, one source for both the pod and the gate): arms and flags `vdac_pod.sh:16-26`, `launch_gate.py:906-913`; chain arm table `vdac_chain.sh:18-26`; `VDAC_STOP_AFTER` stage names `precheck`, `inputs`, `egr`, `dd`, `preflight_<arm lower>`, `gate_<arm lower>`, `train_<arm lower>`, `export` (`vdac_chain.sh:62,176,205,216,225,282,293,356,367`).

### Optional but RECOMMENDED before HB trains: the preflight-and-gate dry-run pause (memory and speed are UNMEASURED)
HB-s0 is the one arm whose memory (batch 8 through a trunk copy with LoRA backward, no activation checkpointing: PREREG 9.1) and step cost (the 4.4 to 7.6 L40-h bracket) are unmeasured. The chain supports a stop after a named stage, inherited through the pause script's environment:
```
VDAC_STOP_AFTER=gate_hb-s0 setsid nohup bash /workspace/refe-plan-vdac/refe/vdac_pause_p7.sh <P7_POD_SH_PID> <TRAIN_PY_PID> A5HB >> /workspace/data/refe_vdac/pause.out 2>&1 < /dev/null &
```
It runs HB-s0's preflight, both mutants, the live suite and the token, writes the stage markers (`/workspace/data/refe_vdac/stage_hb-s0/<name>.ok`, keyed by the tree sha), logs `stopping after gate_hb-s0`, the pause logs "chain HB-s0 did not finish: the remaining arms are NOT run" and P7 resumes. Cost ~0.4 L40-h plus one P7 interruption. It prints `gpu_mem_gb` and `s_per_microbatch` in `/workspace/data/refe_vdac/hb-s0/preflight.log` (BUILD4 section 6 item 8): send them to the PI before the real pause. The real pause then SKIPS the stages whose markers carry the same tree sha and goes to training (token re-verified at train start). **Do not re-ship between the dry run and the real pause** (a new tree sha voids the markers and the token). If the preflight refuses on memory, PREREG 9.2 applies BEFORE any step trains.

### Step E (pod): what to pull back per arm
The chain's egress (`vdac_chain.sh:373-376`) pushes to the private HF repo `Sayood/tanitad-refe-scorer1p6` under `vdac/<arm lower>`: `aux_drv.pt` (format `aux_drv/2` for HB), `head_pred_h48x.npz`, `scores/head_pred_update<N>.npz`, `sft.log`, `sft.jsonl`, `preflight.log`, `mutate_exclude.log`, `launch_gate_<ARM>.log`, `run_check.log`, `postgate.log`, plus the once-per-tree `vdac/export/h48_proposals_subscores.npz(.json)`; it verifies the listing (`ZZEGRESS_OK`) and records the LFS sha256 of `aux_drv.pt` in `<arm dir>/lfs_sha.json` next to the pod file's sha256 (`ZZVDAC_LFS_SHA ... equal True`). **Not pushed, pull by the usual pod-to-dev copy** (small): `/workspace/data/refe_vdac/<arm lower>/scores/head_pred_step0.npz` (the S1-F init file, PREREG 7.5), `<arm>/preflight.jsonl` (G21's raw fields incl. `step0_ctx_max_abs_diff`, `gpu_mem_gb`, `s_per_microbatch`), `<arm>/mutate_ego.log`, `/workspace/data/refe_vdac/lg_<ARM>/PASS_*.json`, `token_verify_<ARM>.json`, `chain_<arm lower>.log`, `pause.log`, `<arm>/lfs_sha.json`.
`model_sft_A.pt` / `model_sft_B.pt` (2.6 GB each) are NOT banked by the egress: they stay on the pod until the MM decides (nothing in this plan deletes the pod's only copy).
Download from HF with `HF_HUB_DISABLE_XET=1` and the tanitad venv (`hf-upload-disable-xet`); verify each npz by sha256 against the LFS sha256 (the S1a procedure, `inputs_manifest.json` note).

### Step F (dev box, CPU): the read command per arm
Precondition: the reader's freeze record exists (PREREG 7.4), and the manifest of PREREG 7.5 is written BEFORE the pulled files are opened. Manifest = the S1a/Stage 0 `manifest_s1f_v2.json` (HA-s0, HA-SHUF, HA init, export, store, h48_logs) PLUS, per new arm, files `<x>_s0_a` (`scores/head_pred_update10000.npz`), `<x>_s0_b` (`head_pred_h48x.npz`), `<x>_shuf_a` (`scores/head_pred_update2000.npz`), `<x>_shuf_b`, `<x>_init_a` (`scores/head_pred_step0.npz`), `<x>_init_b` (HB: HA's `head_init0_b.npz`; HC: absent, PREREG 7.5), an `arms` row each (`role` trained / shuf / init, `cell` 2.0 for HB and 1.0 for HC, `shuf`, `init`), and `"pairs": [{"x": "HB-s0", "ref": "HA-s0"}, {"x": "HC-s0", "ref": "HA-s0"}]` (schema: `eval/s1_read_v2.py` docstring).
```
CUDA_VISIBLE_DEVICES=-1 S1_READ_V1="<PK>/raw/2026-11-09-scorer1p6-prereg/vdac/s1a_read/s1_read.py" C:/Users/Admin/venvs/tanitad/Scripts/python.exe "<PK>/eval/s1_read_v2.py" --manifest <manifest_a5_<arm>.json> --out <read_dir>/<arm>
```
(`s1_read_v2.py:510-514`: `--manifest`, `--out`, `--n-boot` default 10,000.) The frozen reader is selected by `S1_READ_V1`: the tip's `eval/s1_read.py` has another sha256 and the tool REFUSES it (`load_v1`). One command reads one arm with its controls and its paired delta; run it for HB-s0 when `A5HB` finished and for HC-s0 when `A5HC` finished; the pair's branch read follows both.

## 4. What the plan needs that is NOT built (blocks, smallest fix)
Stated in PREREG section 9. For THIS plan: **nothing blocks the launch of any of the four arms except the two items that are UNMEASURED, not missing**: (a) HB memory at batch 8 (9.1/9.2; the dry-run pause measures it before any step trains), (b) `test_launch_gate.py` past line 2313 and the Stage 0 tests, UNVERIFIED until run where the data exists (step C). Blocks of the READ, not of the launch: HC's extra-log init head (7.5, fallback pre-registered), every in-sample number of HB / HC (7.6, not deciding), G-EVAL at S2 (9.5), the replicates HB-s1 / HC-s1 (9.3, only on X-GO).

## 5. Deliverable manifest of the pod run (what exists where, when the plan has run)
| artifact | where | single copy |
|---|---|---|
| per arm: head, held-out and H48 predictions, logs | pod `/workspace/data/refe_vdac/<arm>` AND HF `vdac/<arm>` (sha256-verified) | two |
| `head_pred_step0.npz`, `preflight.jsonl`, tokens, chain logs | pod only until pulled; then the read hold | pull, then two |
| `model_sft_A.pt`, `model_sft_B.pt` | pod only | **single copy**: do not delete |
| H48 export | pod + HF `vdac/export` + the S1a copy (sha256 `14560686...`) | three |
