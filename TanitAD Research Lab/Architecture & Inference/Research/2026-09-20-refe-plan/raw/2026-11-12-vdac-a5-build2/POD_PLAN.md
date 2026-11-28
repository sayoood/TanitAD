# POD_PLAN: VDAC A5 Stage 0, one P7 pause (build 2 of the A5 build order: the forward producer)

`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Written 2026-11-12 against git mirror tip `2207ef03eeae8d048c7d02ef8c8410fe08a7d101`. Nothing here has been run on the pod or on real data: every command below is UNVERIFIED on the pod (the tools are tested on CPU with a tiny model and synthetic stores, `FREEZE_BUILD2.md`). Evidence class of each number is stated inline.

**Needs the PI / MM (provision, spend):** one P7 pause of about 0.25 L40-h (ESTIMATED, section 4) on the pod; the A5 draft's own cost line is 0.12 (budget 0.15, upper 0.25), this plan is higher only because it adds the bank-keys step and the 1,705 rebuilt-initial-head frames (section 4).

## 1. What runs where

| step | tool | where | device | what it reads / writes |
|---|---|---|---|---|
| 0 | ship the package (section 2) | dev box -> pod, BEFORE the pause (copying files adds no GPU / RAM load) | - | `/workspace/refe-plan-stage0/` |
| 1 | `vdac_stage0_pod.sh` inside the pause (below) -> `vdac_stage0_forward.py bank-keys` | pod | CPU, no model | the P6 TRAIN bank's rows -> `stage0/bank/bank_keys.tsv`; refused unless its row count equals the run's data event `train_sets` |
| 2 | `vdac_stage0.py make-list` | pod | CPU | the 2,048-frame list + the 256 C0-ID frames; their sha256 go to `stage0/FROZEN_SHAS.txt` BEFORE the forward exists |
| 3 | `vdac_stage0_forward.py --mode both` | pod | **GPU (L40), P7 stopped** | `stage0/fwd/{preds_insample,preds_repro256,head_init0_rebuilt_a,head_init0_b}.npz`, `stage0_forward.json` |
| 4 | pull back (section 5) | pod -> dev box | - | about 75 MB |
| 5 | `vdac_stage0.py read`, then `vdac_stage0.py s1f` | dev box | CPU | `in_sample_iou.json`, `stage0_read.json`, `s1f_read_v2.json` |

**Order correction against the task wording:** `s1f` consumes `read`'s `in_sample_iou.json` (`vdac_stage0.py:433-435`, it refuses an `S0-INVALID` or absent file), so `read` runs first and `s1f` second.

## 2. Package to ship (a NEW directory; never over `refe-plan-vdac` or `refe-plan-p7`)

`gate_token.TREE_ROOTS = (refe, code, splits)`: changing `refe/` inside a package dir that holds a minted token VOIDS that token (`vdac_chain.sh` header, line 6). Stage 0 needs no token, so it gets its own directory `/workspace/refe-plan-stage0/`, same relative layout as `PK`:

| file (relative to `PK`) | why | sha256 |
|---|---|---|
| `refe/vdac_stage0_forward.py`, `refe/vdac_stage0_pod.sh` | this build | `FREEZE_BUILD2.md` section 1 |
| `refe/vdac_stage0.py`, `eval/s1_read_v2.py`, `raw/2026-11-09-scorer1p6-prereg/vdac/s1a_read/s1_read.py` | the consumer + the frozen reader it asserts (`58aa3a77...`) | `FREEZE_BUILD1.md` section 1 |
| `refe/model.py`, `ckpt_io.py`, `aux_drivable_head.py`, `scorer_finetune.py` (imported for its parser only; NOT run), `train.py`, `rl_reward.py`, `field_loss.py`, `nav_command.py`, `ego_yaw_table.py` | the import closure of the forward, MEASURED by importing the tool on CPU and listing the loaded files of the tree | git tip blobs |
| `refe/vdac_pod.sh`, `refe/scorer1p6_pod.sh` | regenerate the run's argv (`MODE=argv` prints it and exits before any python) | git tip blobs |
| `splits/sft_heldout_logs.txt` | the 24 SFT held-out logs | git tip blob |

No pinned file is modified (none of these is edited; `scorer_finetune.py` is imported for `build_parser()` only). `h48_logs.txt` is already on the pod at `/workspace/data/refe_vdac/h48_logs.txt` (copied and re-hashed by the VDAC chain; the tools assert sha256 `2ae13935...`). Verify the shipped files by sha256 against the tip blobs before the pause (cheap, CPU-light).

## 3. The exact commands

### 3.1 The pause wrapper: reuse `refe/vdac_pause_p7.sh`, with these edits (a sibling file, `vdac_stage0_pause_p7.sh`; NOT built here, the MM lands and reviews it)

`vdac_pause_p7.sh` already: waits for P7's next `ckpt_last.pt`, stops the launch script then the trainer by explicit PID, confirms the GPU holds no compute process (`nvidia-smi --query-compute-apps`), runs the stage, and ALWAYS resumes P7 from an EXIT trap (unless `HOLD_P7` exists). Edit only:

1. stage selector: add `S0) ARMS="" ;;` to the `case "$STAGEARG"`, and update the usage line to `S1a|S1b|S0`. S0 needs no `S1B_GO` file.
2. replace the `for A in $ARMS ... done` loop (the chain invocations) by:
   ```
   say "stage0 starting (timeout 5400 s)"
   timeout 5400 bash /workspace/refe-plan-stage0/refe/vdac_stage0_pod.sh >> $OUT/stage0_pause.out 2>&1 < /dev/null
   say "stage0 rc $? (last: $(grep -E 'ZZSTAGE0_' $OUT/stage0/stage0.log 2>/dev/null | tail -1 | cut -c1-160))"
   ```
   (5,400 s = 90 min against the 0.25 L40-h = 15 min estimate: a generous kill bound, not a plan.)
3. launch it the way S1a is launched (explicit PIDs of `p7_pod.sh` and its `train.py`): `bash vdac_stage0_pause_p7.sh <p7_pod.sh pid> <train.py pid> S0`.

### 3.2 What the stage script runs (`MODE=plan bash vdac_stage0_pod.sh` prints exactly this; `bash -n` clean)

```
# env: VDAC_WS=/workspace  STAGE0_PKG=/workspace/refe-plan-stage0  OUT=/workspace/data/refe_vdac/stage0   (all defaults)
env ARM=HA-s0 MODE=argv VDAC_WS=/workspace bash -c "bash $PKG/refe/vdac_pod.sh > $OUT/ha_s0_argv.txt"
cd $PKG/refe
env CUDA_VISIBLE_DEVICES= python vdac_stage0_forward.py bank-keys --argv-file $OUT/ha_s0_argv.txt --run-dir $RUN --out-dir $OUT/bank
env CUDA_VISIBLE_DEVICES= python vdac_stage0.py make-list --out-dir $OUT/list --bank-keys $OUT/bank/bank_keys.tsv --store $STORE \
    --h48-logs $ROOT/h48_logs.txt --sft-heldout $PKG/splits/sft_heldout_logs.txt --banked-pred $RUN/scores/head_pred_update10000.npz
# LSHA / RSHA = frame_list_sha256 / repro256_sha256 from $OUT/list/stage0_frames.json, written to $OUT/FROZEN_SHAS.txt, logged as ZZSTAGE0_LIST_FROZEN
python vdac_stage0_forward.py --mode both --run-dir $RUN --argv-file $OUT/ha_s0_argv.txt --out-dir $OUT/fwd \
    --aux-sha256 281320071fea --p6-sha256 968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1 \
    --store $STORE --h48-logs $ROOT/h48_logs.txt --sft-heldout $PKG/splits/sft_heldout_logs.txt \
    --insample-list $OUT/list/stage0_frames.tsv --insample-list-sha256 $LSHA --repro-list $OUT/list/stage0_repro256.tsv --repro-list-sha256 $RSHA \
    --banked-pred $RUN/scores/head_pred_update10000.npz --banked-sha256 6b9ac2bc1bc5989e \
    --pod-step0 $RUN/scores/head_pred_step0.npz --pod-step0-sha256 $S0SHA --pod-h48x $RUN/head_pred_h48x.npz --pod-h48x-sha256 64d2f71be45154bd \
    --workers 8 --batch 16
```
with `RUN=/workspace/data/refe_vdac/ha-s0`, `ROOT=/workspace/data/refe_vdac`, `STORE=/workspace/data/aux_drivable/label_store`.

Provenance of the declared hashes (the tools REFUSE if the pod's bytes differ): `968a469d...` P6 `model_final.pt` (repo docs, 40 mentions; verified against the file by the tool); `281320071fea` `aux_drv.pt` LFS prefix (A5 draft section 1); `6b9ac2bc1bc5989e` / `64d2f71be45154bd` = `head_pred_update10000.npz` / `head_pred_h48x.npz` in `raw/2026-11-09-scorer1p6-prereg/vdac/s1a_read/inputs_manifest.json` (`head_s0_a`, `head_s0_b`; the HF copies were sha256-verified equal to the LFS sha at download, FREEZE_BUILD1). **`head_pred_step0.npz` has no earlier recorded sha256** (it is not in the S1a manifest): `$S0SHA` is read from the pod's file at run time and is a RECORD, not a prior freeze (marked RUNTIME in the script and in `FROZEN_SHAS.txt`). If the MM can bank it to HF before the pause, freeze it there instead.

### 3.3 What the tool refuses (so a pause is not spent on a wrong run)

Before any GPU use: argv does not parse with the trainer's own `build_parser`; argv differs in ANY key (except `out`, `dump_scores`) from the namespace the run recorded in `model_sft_A.pt` meta; `aux_drv.pt` is not `aux_drv/1`, carries a feed, records another head config / label store / update count than the argv; its head does not load strictly; it is bit-identical to a freshly built seed-0 head in every tensor; the run's data event disagrees with the rebuilt head's built facts; a sha256 (list, P6, aux, banked, pod files) differs; the frame list fails the consumer's own `verify_list` (quota, hashes, H48 / SFT exclusion, labelled, city); a frame has no / several bank rows; `--out-dir` already holds an output. **GPU guard:** default REFUSES when `nvidia-smi --query-compute-apps` lists ANY process (or cannot be read), checked at start and again right before the model moves to the GPU; `--allow-shared-gpu` is the only override and the plan never passes it.

## 4. ESTIMATED L40 minutes, with the arithmetic

Frames through the trunk (each distinct frame once; both heads are applied to a batch that needs them):

| set | bank | frames | source of the count |
|---|---|---|---|
| in-sample (trained head) | TRAIN | 2,048 | `make-list` rule: 4 cities x 512 (`vdac_stage0.py` `PER_CITY`) |
| h48x (seed-0 head) | TRAIN | 1,502 | S1a `head_s0_b` frames (A5 draft section 1: "1,502 extra-log H48 frames") |
| step 0 (seed-0 head) | HELD-OUT | 1,705 | S1a `head_s0_a` / `head_pred_step0.npz` (A5 draft: "1,705 frames") |
| C0-ID (trained head) | HELD-OUT | 256, a subset of the 1,705 (same 24 logs, same dump order: UNVERIFIED that the key sets are equal; worst case +256) | `make-list` rule |
| **distinct trunk forwards** | | **5,255** (upper 5,511) | 2,048 + 1,502 + 1,705 (+ 256) |

* GPU compute: 5,255 x 0.066 s = **347 s = 5.8 min = 0.096 L40-h** (upper 5,511 x 0.066 = 364 s = 0.101 L40-h). 0.066 s = one third of P6's 0.197 s per sample (INHERITED from `PREREG_SCORER1_AMENDMENT_3.md:24` through the A5 draft section 1; "forward = 1/3 of training" is an ASSUMPTION, no h48x-forward timing is logged: `grep h48x_forward` finds the event name and no timing anywhere in the repo). The head adds under 1 % (ARITHMETIC: about 25 to 30 GFLOP per frame for 2 cross blocks over 7,680 x 256 context against about 4.6 TFLOP for ViT-L on 4 x 1,920 tokens).
* CPU parts, each ESTIMATED from the draft's "load and bank open about 0.05 L40-h": bank-keys (opens the bank) about 3 min; make-list (re-hashes the label store shards, about 1,192 files) about 1 min; the forward's own model load + bank open about 3 min. The bank is opened twice (bank-keys, forward) because it cannot be opened before the pause without adding RAM load to a training pod.
* Image decode: the draft's upper case is "0.25 if frame decode dominates"; this plan uses 8 workers with P7 stopped. The pod's vCPU count is not asserted here (`P6_POD_RUNBOOK.md` asks for at least 24, `P6_ON_11VCPU.md` measured an 11-vCPU quota); 8 workers fit either, but decode could still dominate on 11. UNVERIFIED.

| | arithmetic | L40 min | L40-h |
|---|---|---|---|
| lower | 5.8 forward + 1 list + 3 bank-keys + 3 load | 12.8 | 0.21 |
| central | 5.8 + 1 + 3 + 3 + 2 (writes, hashing, margins) | 14.8 | **0.25** |
| upper | 6.1 + 1 + 3 + 3 + decode-bound forward x 2 | about 24 | 0.40 |

Of which the GPU is computing about 6 minutes; the rest is CPU work inside the pause (the cost of not loading a training pod). P7 additionally loses nothing but the wait: the wrapper stops it at a fresh checkpoint and resumes from `ckpt_last.pt`.

## 5. What to pull back (about 75 MB, sha256-verified)

From `/workspace/data/refe_vdac/stage0/`: `fwd/preds_insample.npz` (about 21 MB: 2,048 x 40 x 32 x 4 B x 2), `fwd/preds_repro256.npz` (about 2.6 MB), `fwd/head_init0_rebuilt_a.npz` (about 17.5 MB), `fwd/head_init0_b.npz` (about 15.4 MB), `fwd/stage0_forward.json`, `list/stage0_frames.tsv`, `list/stage0_repro256.tsv`, `list/stage0_frames.json`, `FROZEN_SHAS.txt`, `head_pred_step0.npz` (about 17.5 MB, the pod's own file: it is `s1f`'s `--init-pod-step0`), `stage0.log`, `ha_s0_argv.txt`. Route: the one used for S1a (`hf_egress.py` to a private HF path under `Sayood/tanitad-refe-scorer1p6`, `HF_HUB_DISABLE_XET=1`, verify by sha256), or the framed-base64 pull if the PTY is used (a plain base64 pull goes silently corrupt: `tanitad-traps`). The route is the MM's choice; it is not scripted here.

## 6. The dev-box read (after the pull)

Inputs already on the dev box or re-downloadable: the label store (`E:/Projects/TanitAD-artifacts/aux_drivable/label_store`, sha prefix asserted), `splits/h48_logs.txt`, `splits/sft_heldout_logs.txt`, the S1a inputs from HF (`head_pred_update10000.npz`, `head_pred_h48x.npz`, the SHUF pair, the export; FREEZE_BUILD1 section 3) with `inputs_manifest.json` re-pointed to them, `s1a_read.json`.

```
python vdac_stage0.py read --out-dir <d>/read --frame-list list/stage0_frames.tsv --frame-list-sha256 <LSHA> \
    --preds-insample fwd/preds_insample.npz --banked-pred <dl>/head_pred_update10000.npz --banked-sha256 <6b9ac2bc...> --preds-repro fwd/preds_repro256.npz \
    --store <label_store> --h48-logs splits/h48_logs.txt --sft-heldout splits/sft_heldout_logs.txt --s1a-json <s1a_read.json>
python vdac_stage0.py s1f --out-dir <d>/s1f --base-manifest <S1a manifest re-pointed> --in-sample-json <d>/read/in_sample_iou.json \
    --init-pod-step0 head_pred_step0.npz --init-rebuilt-a fwd/head_init0_rebuilt_a.npz --init-rebuilt-b fwd/head_init0_b.npz
```
`read` gives S0-FIT / S0-WEAK / S0-INVALID mechanically; `s1f` first checks the rebuilt seed-0 head reproduces the pod's step 0 (sign agreement at least 99.9 %), then runs the frozen reader v2.

## 7. Risks, each named (nothing here is hidden as a footnote)

1. **C0-ID can fail for numeric reasons, not a tool defect.** The trainer's dumps are bf16-autocast at batch 16 in index order; the tool uses the same autocast, batch 16 and index order for the 1,705 held-out and the 1,502 h48x frames (same composition), but not for the 256 (they sit inside the 1,705 batches here, so the composition is the same too). Cells within bf16 noise of logit 0 can still flip: the bar is 99.9 % sign agreement over 256 x 1,280 = 327,680 cells (327 cells allowed). UNVERIFIED how many flip. If the forward ends rc 3, read `fwd/stage0_forward.json` (`reproduction[*].sign_agreement`, `max_abs_logit_diff`) before concluding anything: the consumer's own `read` is the deciding C0-ID (agreement and IoU within 0.002); an `S0-INVALID` is "fix the tool, re-run", not a Stage 0 result.
2. **`real_context` / `real_banks` (the bank and model construction on real data) never ran on real data.** It copies `scorer_finetune.py:1736-1760, 1771-1776` line for line and its row-selection helpers are unit-tested, but the first execution against the real bank is the pod. `vdac_stage0_forward.py bank-keys` is that first execution on CPU: its row count must equal the run's data event `train_sets` (60,000 per A3, INHERITED) or it refuses, before the GPU is touched.
3. `head_pred_step0.npz` has no earlier sha256 (section 3.2).
4. Whether the 256 C0-ID keys are a subset of the 1,705 step-0 keys is UNVERIFIED (cost only).
5. The sibling pause wrapper and the pod stage script are untested on a pod (`bash -n` clean; `MODE=plan` output checked by eye against section 3.2).
6. The A5 draft's alternative branch (S1-F on the 1,705-frame subset when the rebuilt head fails to reproduce step 0) is NOT built in `s1f` (FREEZE_BUILD1 section 4 item 3). The rebuilt head is `build_head(seed 0)` with its own forked RNG stream, tested bit-identical across builds and different from seed 1, so a failure there would be a real finding.
