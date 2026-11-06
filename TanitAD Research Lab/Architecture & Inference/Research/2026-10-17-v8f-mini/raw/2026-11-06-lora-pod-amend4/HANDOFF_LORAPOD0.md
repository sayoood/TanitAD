# HANDOFF_LORAPOD0 (Phase 0 of PLAN_V8F_LORA_POD, 2026-11-06 ~22:05 dev-box time)

Pod address: `C:/Users/Admin/lorapod/pod_direct.txt`; helpers `podenv.sh` (source it, then `pssh`, `pscp`). Nothing here holds the address.

## State

| item | state | evidence |
|---|---|---|
| 1 stack at `939176e7` | DONE | `stack`, `taniteval` in `/root/tad`; `tanitad.__file__` = `/root/tad/stack/tanitad/__init__.py` (MEASURED) |
| 2 DINOv3-L | DONE | `/root/models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors`, 1,212,559,808 B, sha256 = `dcb2e451…8179` (full hash matched); `/root/models/dinov3_sha_check.json` |
| 3a input staging | DONE | `/root/data/stage_report.json`: parity gate kept 4,713 dropped 0; timestamps 4,713; egomotion zip 4,713; intrinsics vs HF max diff 1.14e-13; mp4 ledger complete |
| 3b byte gate (3 clips) | **FAILED 0/3** | `/root/data/gate_result.json`; full build NOT launched |

Deps added to `/root/venv` (torch asserted `2.11.0+cu128` after each): av 18.0.0, pandas 3.0.3, pyarrow 24.0.0 (same ffmpeg libs as the dev box), torchvision 0.26.0+cu128 (`--no-deps`). Not installed: transformers (dev box has 5.15.0), peft, scipy.

## The gate diff (MEASURED)

Pod-built v2ep for `8f387977`, `bf5ec76e`, `dca157a0` differ from the banked hashes in file bytes. Against the dev-box banked `8f387977`: actions, poses, frame geometry, clip_id all equal; PNG-decoded frames differ in **1.377 % of values, by exactly ±1 level, in all 201 frames**; signed mean -0.0007 level (two-sided; Thor's offset was +0.75).

Cause, localised by `C:/Users/Admin/lorapod/rect_probe.py` and `grid_apply.py`: the raw decoded frames (hash `2093f457…`), the observed mask and the intrinsics are identical on both hosts. The float32 grid from `cylindrical_grid` differs (pod Linux torch 2.11.0+cu128, EPYC 7702, vs dev-box Windows torch 2.11.0+cu128, i9-12900F) at every `ATEN_CPU_CAPABILITY`. Feeding the dev-box grid to the pod still gives a different `grid_sample` output. So the production CPU rectify is bit-reproducible only on the Windows torch build (dev box and box 2, which agree), not on this Linux pod. No pod-side setting found that fixes it. Not tested: a token-space effect (needs the GPU encoder).

## Pod paths

`/root/ops/v2ep_pod.py` (stage | gate | run), lists, `banked_hashes.json`; `/root/data/v2ep/{train,eval}` (the 3 gate clips only); `/root/data/v2ep_ledger.jsonl`; `/root/data/stage` (inputs: ~3.9 GB); logs `/root/logs`. `run` refuses without a passing gate; `--waive-gate "<reason>"` overrides and records `/root/data/gate_waiver.json`.

## Projection if the build is released (ESTIMATED)

3 concurrent clips: 22.8–28.9 s build each (+2 s download) on this EPYC; 13 workers give about 2.1 s/clip, so 4,713 clips take **about 2.8 h** and about 174 GB. Resumable, 13 workers, per-clip log, `ZZ V2EP DONE ZZ` / `ZZ V2EP FAIL ZZ` come from the wrapper below (not yet launched).

Launch (only after a decision): `cd /root/ops && nohup bash -c 'PYTHONPATH=/root/tad/stack:/root/tad/stack/scripts /root/venv/bin/python -u v2ep_pod.py run --waive-gate "<reason>" > /root/logs/v2ep_run.log 2>&1; rc=$?; [ $rc -eq 0 ] && echo "ZZ V2EP DONE ZZ" >> /root/logs/v2ep_run.log || echo "ZZ V2EP FAIL ZZ rc=$rc" >> /root/logs/v2ep_run.log' < /dev/null > /dev/null 2>&1 &`
Progress: `tail -3 /root/logs/v2ep_run.log; wc -l /root/data/v2ep_ledger.jsonl`.

## Decision needed (Master Mind / PI)

* **A. Build on this pod and accept ±1 frames.** Identity against box 2's cache then cannot be bit-exact, so Phase 1 takes the card's committed B′ route: re-encode train and eval on this GPU from the pod-built frames, retrain B′ at seeds 0 and 1, take F′ from B′. Base and arm share device and frame set. Needs an Amendment naming the Linux-CPU frame route.
* **B. Keep byte identity.** Build on a Windows host (dev box 8.2 s/clip, about 10.5 h, plus a 165 GB ship; or box 2) and ship to the pod.
* **C. First a cheap GPU discriminator** (seconds, needs a go and transformers): encode the pod-built and banked `8f387977` frames on this GPU and read the pooled-token rel-L2 against tau 1e-3. It prices A.

## Next step

Decide A, B or C. For A, launch the command above. For C, install transformers, then run the builder's `encode_episode` on both frame sets.
