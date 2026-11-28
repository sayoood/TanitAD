# POD_TRAIN_RECIPE: v8F-mini hierarchy vs flat (read-only research, 2026-11-13)

M = mirror `C:/Users/Admin/tanitad-push`, ref `origin/agent/arch-inf-20260803` (tip e262e2cc at read time).
R = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini` (all `raw/...` paths are under R).
VERIFIED = read in the file named. INFERRED = my reading.

## 0. Note on the brief (VERIFIED)
`C:/Users/Admin/t2v8f_pulls/` holds NO flat/hierarchy pair (no argv there contains `--o-interface`; folders are ladder_*, egowm_*, obd_*, otband_*, slowwm_*, stopbal_*, stage*; those are `--wm-only` or other T2 arms).
The matched pairs are in the mirror (argv.json + summary.json) and the dev-box checkpoints on E:.

## 1. Entry point and commit
* Trainer: `stack/scripts/v8f_mini_train.py`; eval: `stack/scripts/v8f_mini_eval.py`; T1 eval: `stack/scripts/v8f_t1_eval.py` (VERIFIED: `set TRAIN=`/`set EVAL=` in `R/code/v8f_mini_box2_queue.cmd` lines 104-105; v8f_t1_eval argparse in M).
* Env on box 2: `PYTHONPATH=%REPO%\stack;%REPO%\taniteval` (queue line 106). torch 2.11.0+cu128, transformers 5.15.0 (memory box2-second-computer.md).
* Commits: FLAT pair (pair 1) gate token `commit` = `ea67b7a0137f2b1c591da355072541064f4af58d` (`raw/2026-10-26-flat/gate_token.json`). DAG pair (pair 2) token `binding.commit` = `b43c7a9a1cf6d6d8f739fcf7b1740b366c241050` (`raw/2026-10-27-dagger-run/M2b_s0_dag/launch_gate_token.json`). Pod B-prime tree = `939176e7` (`C:/Users/Admin/lorapod/chain_lorapod2.sh` line 73, manifest `lorapod/ship2/commit_939176e7_manifest.json`).
* Box 2 repo is a `git archive` of the tip, not a checkout (memory). Key-file sha256 list is in the gate token.
* INFERRED: for a pod use commit `939176e7` or later (it already ran this trainer on a pod, section 7), and sha-check `v8f_mini_train.py` against the token's key_files.

## 2. Exact argv (VERIFIED, `argv.json` files; flags identical except where marked)
Common prefix (box-2 paths; replace roots on the pod):
```
--arm M2b --seed S --grid pool4x10 [--o-interface cut] --out <OUT>
--cache-dir D:\tanitad\cache_full
--v9-train D:\tanitad\data\v9\v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759
--lead-train D:\tanitad\data\v9\lead_train4572.npz
--speed-train D:\tanitad\data\v9\refcv6_speed_max_v8_train.jsonl
--join-train D:\tanitad\runs\v8f_mini_full\join_train.npz
--p10 D:\tanitad\runs\v8f_mini_full\p10\p10_prior.pt
--kin-mode yaw --tracks D:\tanitad\cache_full\native_tracks.pt
--exclude 540:600 --batch 64 --steps-mult 4 --device cuda
```
No `--steps` is given in pair 1: steps = 19,164 come from `--steps-mult 4` x S 4,791 (summary.json `steps`, `S`). The arm table `M2b` sets p_gt 1.0->0.2, hist 4, hmax 16, code_mix whole, w_tac 1, w_cons 0.5 (summary `levers`, flat runs).

### Pair 1: from scratch, 19,164 steps, batch 64, seeds 0 and 1
| arm | folder | argv difference | argv_sha |
|---|---|---|---|
| HIER (B = M2b) | `raw/2026-10-26-flat/runs/M2b_s{0,1}`; box 2 `D:\tanitad\runs\v8f_mini_full\M2b_s{0,1}`; ckpt sha256 s0 `601cb435...c6f8`, s1 `6104fb15...c2e6` (`raw/full/MANIFEST.md`) | none | s0 `1225d98ff87cbe17` |
| FLAT | `raw/2026-10-26-flat/runs/M2b_s{0,1}_flat`; ckpt dev box `E:/Projects/TanitAD-artifacts/v8f-mini-flat/M2b_s{0,1}_flat.pt` (87 MB) | `--o-interface cut` after `--grid pool4x10`, out `...\M2b_s<S>_flat` | s0 `33d9e5e27e44a23b` |
Seed flag `--seed`. Param count identical (7,214,495 trainable, summary.json). Verdict of this pair: NOT ESTABLISHED (`raw/2026-10-26-flat/flat_read.json` key `VERDICT`).
Banked checkpoint folders on E: for HIER: `C:/Users/Admin/box2_ship/logs/v8f_mini/runs/M2b_s*` (MANIFEST.md); not found elsewhere.

### Pair 2: DAgger fine-tune of pair 1, 3,000 steps
Same common prefix, plus (VERIFIED `raw/2026-10-27-dagger-run/M2b_s0_dag/argv.json`, `raw/2026-10-27-flatdag/box2/M2b_s0_flatdag/argv.json`):
```
--init-ckpt <pair-1 ckpt of that arm and seed> --steps 3000 --lr 1e-4 --warmup 100
--p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0 [--o-interface cut]
--dagger-p 0.1 --dagger-bank <bank dir> --dagger-bank-sha256 <sha> --dagger-seed S
```
* HIER = `M2b_s{S}_dag` (init M2b_s{S}; no `--o-interface`; s0 bank sha `197a68bb...aa8f`). FLAT = `M2b_s{S}_flatdag` (init `M2b_s{S}_flat`; `--o-interface cut`; bank `dagger_bank_flat\s{S}`; s0 sha `1fa89521...b540c3`, s1 `f5247555...efebbdf6`). Control `..._dag0` / `..._flatdag0` = same argv with `--dagger-p 0`.
* Banks are produced by queue stages `bankdag` / `bankflat` (queue lines 1170 / 84) from the frozen parent checkpoint, TRAIN only. Banks are small-ish dirs on box 2; E: copies `E:/Projects/TanitAD-artifacts/v8f-flatdag/bank_s{0,1}`. Size not read.
* `--batch 64` kept (prefix). Dev-box FLATDAG ckpts: `E:/Projects/TanitAD-artifacts/v8f-flatdag/M2b_s{S}_flatdag/ckpt_last.pt`.

## 3. Data inputs
| input | box-2 path (argv above) | where a copy exists |
|---|---|---|
| train token cache `cache_full/train` (4,575 files; DINOv3-L pooled fp8, `pool4x10`) | `D:\tanitad\cache_full` | box 2 only: 89.0 GB train (`raw/2026-11-08-hwm-horizon-review/AUDIT_DATA.md:183`); ~98 GB incl. eval per `raw/2026-11-06-lora-pod-amend4/HANDOFF_LORAPOD1.md`. Per-clip 4.356 MB (`raw/2026-10-25-d3-scoping/D3_SCOPING.md:52`). NOT on this dev box (`raw/2026-10-25-lora-route/ROUTE.md:60`). NOT found on HF (searched Sayood/tanitad-* names in R, listed in section 8). |
| eval cache 141 clips | `cache_full\eval` | dev box `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache` (2.947 GB, md5-equal to box 2, `RESULT_V8F_SFRESH.md:36`); pod copy of 146 files `E:/Projects/TanitAD-artifacts/lorapod_eval/eval` |
| `native_tracks.pt` | `cache_full\native_tracks.pt` | `E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt`, `.../lorapod_eval/native_tracks.pt`, `C:/Users/Admin/lorapod/ship2/native_tracks.pt` (sha 1db431f1 on pod = dev box, LOOP_STATE line 525) |
| v9 labels train/eval (md5 83e30bd3.. / 6b5c7f20..) , `lead_train4572.npz`, `refcv6_speed_max_v8_train.jsonl` | `D:\tanitad\data\v9\` | HF private dataset `Sayood/tanitad-refcv8-v9-labels` (`FlyWheels/.../hf_push_v9_thor2.py:12`) holds the v9 label release; whether it holds lead/speed files: not verified. Pod already used `/root/data/v9/...` for B-prime (`lorapod_eval/bprime/M2b_s0/argv.json`). |
| `join_train.npz`, `p10_prior.pt`, `p10_eval_pred6.npy` | `D:\tanitad\runs\v8f_mini_full\` | `C:/Users/Admin/lorapod/ship2/{join_train.npz,p10_prior.pt}` (sha pinned in chain_lorapod2.sh `verify-inputs`); p10 also in the repo at `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/raw/p10_yaw4512/` |
| manifests (clip sets) | | `C:/Users/Admin/lorapod/ship2/ref_box2_{train,eval}_manifest.jsonl` |
Train set facts (VERIFIED summary.json): 4,512 clips, 306,566 windows, train_sha256 `37f4f9ee...cc1`, excluded (A1 holdout 540:600) sha256 `f6a1dee0...50f9`.
Source frames HF dataset (VERIFIED `raw/2026-11-06-lora-pod-amend4/v2ep_pod.py:29`): `Sayood/tanitad-v7-training-corpus`; also `Sayood/tanitad-physicalai-w120-256x640cyl` (2,400 `*.v2ep.pt`, 84.98 GB; `GOALS_AND_CLAIMS.md:7209`). A pod can REBUILD the token cache from these (section 7).

## 4. Launch gate on box 2 (VERIFIED)
* `stack/scripts/launch_gate.py` has NO profile for `v8f_mini_train.py` (probe in `chain_lorapod2.sh` lines 160-176: "binds only refc_v3_train.py"). Gate = trainer in-process `--selfcheck-only` (G-DVB, G-LIVE with frozen-prior control exactly 0.0, probes), whose PASS is turned into a token by the chain (`gate_token.json`; token kind `launch-gate-token`, argv sha256 + commit + data sha256).
* Command template (queue lines 857-870): `<py> stack\scripts\v8f_mini_train.py --arm M2b --o-interface cut --out <gate_dir> <prefix> --selfcheck-only`. Real train line must equal the gated line (argv hash asserted).
* Chains per prereg: `code/box2_ops/chain_v8f_flat.sh` (gate, seed 0, seed 1, 4 evals each, reader `code/v8f_flat_read.py`). Queue stages: `gateflat`, `trainflat S`, `evalflat S`; `gatedag`, `traindag`, `bankdag`, `bankflat`. The wait loop sha-checks the parent checkpoints before training.
* Pre-reg card rule: `PREREG_V8F_FLAT.md` (R, one lever, both seeds always, H-ESTIM-SEED-1). INFERRED: also obey CLAUDE.md "no training without a launch-gate pass".

## 5. Wall clock and GPU (VERIFIED)
* Box 2 = RTX 4060 8 GB (memory box2-second-computer.md). M2b_s0 15,741 s (4.37 h), M2b_s1 16,082 s, M2b_s0_flat 14,368 s, s1_flat 14,337 s (summary.json `seconds`). DAG fine-tunes: 2,273 s (s0_dag), 2,263 s (s1_flatdag).
* Pod precedent: B-prime (= M2b argv with pod paths, `lorapod_eval/bprime/M2b_s0/argv.json`, argv_sha `a2b68127d72210f0`) took 3,565 s (59 min) per seed at 0.185 s/step on the RTX 4000 Ada LoRA pod, reading pod-built cache `/root/data/cache_pod` (summary.json `seconds`; LOOP_STATE line 517/544).
* A40 time: not stated anywhere I read; INFERRED < 1 h per 19,164-step arm.

## 6. T1-kin eval (VERIFIED `raw/2026-10-27-dag-reread/argv/FLAT_s0_m10_causal_heads.argv`, `raw/2026-10-27-flatdag/t1/argv/FLATDAG_s0_m10_causal_heads.argv`)
```
python stack/scripts/v8f_t1_eval.py run --ckpt <ckpt_last.pt> --tag <TAG>
 --tracks native_tracks.pt --eval-cache <eval cache dir>
 --eval-grid R/raw/eval_grid_windows.json --eval-join R/raw/join_eval141.npz
 --out-dir <out> --replan-steps 10 --nav deadreckon --scene logged
 --schedule periodic --limit causal --dump-ticks <out>/ticks --dump-heads --device cpu
```
(`--limit causal` is the strict T1-kin variant, `RESULT_V8F_T1KIN_STRICT.md`; `--device auto` default, cpu used on dev box.) Standard per-arm T0-style eval (queue `evalflat`): `v8f_mini_eval.py run --ckpt ... --tag M2b-flat_s<S> %EC% %ER% --project --families` where EC = `--v9-eval ...v9_labels_eval139.npz --lead-eval lead_eval139.npz --speed-eval refcv6_speed_max_v8_eval.jsonl --kin-mode yaw --p10 <p10> --out-dir <eval dir>`, ER = `--eval-cache <cache> --tracks <tracks>` (queue lines 112-113), plus three tagged variants (`--tac absmask`, `--tac oracle`, `--tac oracle --tac-groups all-latcons --tac-alpha 1`).

## 7. Pod route for the training cache (INFERRED from VERIFIED parts)
`lorapod/cache_pod.py` + `run_cache_pod.sh` re-encoded the 4,572-clip cache on a pod from HF v2ep frames (IDENTITY_POD: pod tokens differ from box 2 by rel-L2 ~2.4e-2 because fp8 + different GPU; NOT bit-identical). So a pod pair trained on a pod cache is a new, internally matched pair, not a reproduction of box 2's checkpoints. Pod cache build ~1.4 h GPU + CPU-bound ~2.8 s/clip.

## 8. Not found / where I looked
* A HF copy of box 2's `cache_full` train: grepped HF repo names in R (Sayood/tanitad-: v8f-otband-read, v8f-plantest3-runs, lorapod-ckpts, rollout-recovery, v7-training-corpus, physicalai-w120-256x640cyl, context-parts-v1, context-labels-v1, band-labels-v10, refcv8-v9-labels): none says it holds cache_full. The pod's `cache_pod` was to be banked to HF (HANDOFF_LORAPOD1.md next step 2); actual repo name not found.
* Pair 2 DAG bank sizes: not found.
* Box-2 chain logs hold SSH aliases/ports: deliberately not read or copied.

## Manifest
This file only. Top unknowns: (1) where the token cache for a pod comes from (no verified HF copy of cache_full; must rebuild per section 7 or ship ~89 GB); (2) A40 wall-clock per arm and whether `lead_train4572.npz`/`refcv6_speed_max_v8_train.jsonl` are on HF; (3) the commit to use (ea67b7a0 vs b43c7a9a vs 939176e7; pod proven at 939176e7 for M2b, flat/dag code added later).
