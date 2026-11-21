# HANDOVER_KIT: running card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD, notes N1 and N2) on the pod of the tree owner session

Tier: T0 (probes on cached tokens; no model is trained); never driving performance. Budget rule of the PI's go: 3 to 5 pod-h estimated, the smoke first, a projection above 6 pod-h stops the kit (exit 4) and goes back to the PI.
The probe runs in the STUDY tree (shipped here), not in the tree of commit cdbff971; nothing of the other session's tree, caches or label files is changed. The kit deletes and moves nothing.

## 1. What to copy to the pod

From `F:/TanitAD-scratch/v8proof/dscale_lead/` on the dev box:

| file | sha256 (first 16) | goes to |
|---|---|---|
| `ship/dscale_tree.tar.gz` (48 MB unpacked: `tree/stack/tanitad`, `tree/stack/scripts`, `tree/taniteval/taniteval`) | 24392737b179e1de | unpack into `<KIT>/`, gives `<KIT>/tree` |
| `ship/p10_prior.pt` (the frozen P10 prior of the study, kin mode yaw) | 14a062a91078266a | `<KIT>/p10_prior.pt` |
| `pod_dscale_lead_run.sh`, `tok_probe_u.py`, `dscale_precheck.py`, `dscale_clockcheck.py`, `dscale_tokens_same.py`, `STUDY_EVAL_MD5.tsv`, `STUDY_TRAIN_MD5.tsv`, `TREE_PINS.txt` | pinned in `TREE_PINS.txt` (the kit refuses a difference) | `<KIT>/` |

`<KIT>` is any new folder on the pod, for example `/root/dscale_lead/kit`.

## 2. Variables

```
export KIT=/root/dscale_lead/kit TREE=/root/dscale_lead/kit/tree OUT=/root/dscale_lead/out1   # OUT must be new
export P10=$KIT/p10_prior.pt
export TRACKS=<native tracks .pt that covers all 14,512 union clips>                          # the one fact the kit does not know
```

Defaults already in the script (change only if a path differs): `PY=/root/venv/bin/python`, `CACHE=/root/data/cache_union_dscale`, `EVAL_CACHE=/root/data/cache_pod`, the four union files under `/root/dscale10k/merged/`, `LEAD_CORPUS=/root/strat1/inputs/data/v9/lead_train4572.npz`, `BSET=/root/dscale10k/mix/bset_clips.txt`, `LOCK=/root/gpu_queue.lock`, `THREADS=4`, memory floor 20 GB (cgroup v1 files).
Cache layout: the study tree reads `<cache>/train/manifest.jsonl`. If the union cache has its manifest at the root, the kit builds `$OUT/cache_view` with two links (`train` to the union cache, `eval` to `$EVAL_CACHE/eval`); nothing is copied.

## 3. Stages, in this order (`DRY=1 bash pod_dscale_lead_run.sh <stage>` prints every command and runs nothing)

1. `bash $KIT/pod_dscale_lead_run.sh precheck`  CPU only, minutes. Pins, input hashes, counts, d1 (+ planted id must fail), d2, d2j, d2n, d2g, the clock check d2c (+ planted shift must fail; it loads 800 cached clips on CPU), the evaluation-set check d2e (+ planted swap must fail). A STOP here means an input is not what the card assumes: send `$OUT/precheck/` back before anything else.
2. `... cpucheck`  CPU only, no GPU lock: two training steps on the union and two with the corpus list. It shows that the study tree reads the union files. Watch the memory on the first call (on the corpus alone the build peaks at 2.1 GB).
3. `... smoke`  200 steps of four arms under the GPU lock; writes `$OUT/SMOKE_ESTIMATE.json`; exit 4 `ESTIMATE-ABOVE-6H` if the projection of the five folders exceeds 6 h.
4. `... run`  five folders, one GPU process each under the lock, resumable through markers.
5. `... extract`  evaluation states under the lock.  6. `... checklog`  CPU; writes `$OUT/DSCALE_LEAD_DONE.txt`.

Each stage refuses to overwrite an existing folder and writes `$OUT/DSCALE_LEAD_STOP.txt` with the reason on a stop.

## 4. What comes back to the coordinator fork (the read runs on its side)

A tar of `$OUT` without the checkpoints: `precheck/`, `SMOKE_ESTIMATE.json`, `dscale_lead.log`, `markers/`, `states/`, `checklog/`, `DSCALE_LEAD_DONE.txt`, and from every run folder `r1_batches.json`, `TRAIN_DONE.json` and per arm `train_log.jsonl`, `train_meta.json`. Put it under `F:/TanitAD-scratch/v8proof/dscale_lead/back/` (or name another dev-box path) with its sha256.

## 5. State of the checks (see `TIMES_NOTE.md`, line of note N2)

Run on the pod of the coordinator fork, CPU only, on stand-in data: the three test files (51 tests) and the mutation list (22 of 22 red) under a stand-in test runner (that pod has no pytest); the precheck, the clock check and the evaluation-set check with their planted failures; 20 training steps from the shipped tree. Not exercised anywhere yet: the union files, the cache layout of this pod, the tracks file, and the reader end to end. The stages precheck and cpucheck exist to find such a fault without GPU time.
