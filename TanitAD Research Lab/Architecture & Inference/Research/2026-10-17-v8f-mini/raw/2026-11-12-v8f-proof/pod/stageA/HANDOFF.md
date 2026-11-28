# HANDOFF stage A (pod cache rebuild), written 2026-11-13 ~06:25 pod clock

## STATE: STOPPED at the byte gate, needs ONE lead decision (not running)
`STAGEA_STOP.txt` on the pod: byte gate FAILED 0/3 (the 3 banked clips: 8f387977, bf5ec76e, dca157a0 differ in file bytes).
This is the SAME result as the previous pod (HANDOFF_LORAPOD0: pod CPU rectify vs dev-box bytes, PNG frames differ by +-1 level in ~1.4 % of values; the previous pod then ran `--waive-gate` under a recorded decision, and IDENTITY_POD measured pod tokens at rel-L2 ~2.4e-2 vs box 2). I did NOT waive: a waiver is a decision I was not given. I did not re-measure the diff pattern here.
Everything before the gate passed: inputs sha-ok, DINOv3-L weights sha-ok (dcb2e451...), stage ok (parity gate kept 4,713/0 dropped, timestamps 4,713, egomotion 4,713, intrinsics vs HF mismatch 0).
**To continue (frames + encode, detached):** on the pod
`cd /workspace/v8proof/stageA && STAGEA_WAIVE_GATE="<reason, e.g. PI route B-prime: pod cache is a new internally matched cache>" nohup setsid bash stageA.sh > /workspace/v8proof/hierrep/stageA.log 2>&1 < /dev/null &`
(idempotent: skips the 3 gate clips already built; the waiver is recorded by v2ep_pod in `data/podwork/gate_waiver.json`). Then arm `poll_stageA.sh`.
Expected duration after the waiver: ~6-7 h (ESTIMATED: previous pod 25.9 s/clip build at 13 workers = 1,750 clips/h; here 5 workers = ~700 clips/h for 4,713 clips; GPU encode trails, 1.4 h GPU, 2.8 s/clip CPU-bound; both share the 7.65-core quota). Disk: ~90-100 GB frames (deleted at the end) + ~25 GB cache.

## Environment (MEASURED)
`/workspace/venv`: torch 2.8.0+cu128 (unchanged before/after; `torch.zeros(1).cuda()` ok), torchvision 0.23.0+cu128. Installed with the full `pip freeze` as constraints:
transformers 5.15.0 (box-2 version), huggingface_hub 1.33.0, safetensors 0.8.0, av 19.0.2, pandas 3.1.1, pyarrow 26.0.1, truststore 0.10.5 (+ tokenizers 0.22.2, hf-xet 1.8.0 unused: HF_HUB_DISABLE_XET=1).
Box 2 had torch 2.11.0; the pod's is 2.8.0 (not changed, by rule). Freezes: `hierrep/freeze_before.txt`, `freeze_after.txt`, `constraints.txt`.

## What the script does (`/workspace/v8proof/stageA/stageA.sh`, local copy in this folder)
inputs sha check -> `code/stageA_fetch.py` (DINOv3-L into HF cache `/workspace/v8proof/hf`, sha pinned) -> `v2ep_pod.py stage` -> `gate` -> `v2ep_pod.py run --workers 5` (CPU frames from HF `Sayood/tanitad-v7-training-corpus`, 4,713 = 4,572 train + 141 eval) in parallel with `cache_pod.py --threads 2` (GPU encode, trailing) -> `code/stageA_final.py` (counts, manifest vs `train4572.txt`, weights sha, STAGEA_REPORT.json) -> deletes the re-creatable scratch (`data/podwork/{v2ep,_hf,stage}`, logged in `hierrep/deleted.log`) -> `STAGEA_DONE.txt`. Any failure writes one line to `STAGEA_STOP.txt`.

## Paths (pod)
code `/workspace/v8proof/stageA/{stageA.sh,code/,ops/}` ; tree `/workspace/v8proof/tree` (commit 3e264bbf)
cache root (`--cache-dir`) `/workspace/v8proof/data/cache_pod/{train,eval}/{clips,manifest.jsonl,meta.json}` ; inputs `/workspace/v8proof/data/inputs/{join_train.npz,p10_prior.pt,native_tracks.pt}` (sha verified, = pins in chain_lorapod2.sh)
markers/logs `/workspace/v8proof/hierrep/` : `STAGEA_DONE.txt|STAGEA_STOP.txt`, `stageA.log`, `v2ep_{stage,gate,run}.log`, `cache_pod.log`, `STAGEA_REPORT.json` (written at the end).
`native_tracks.pt` is in `data/inputs/` only (the previous pod also kept a copy inside cache_pod/; add `--tracks /workspace/v8proof/data/inputs/native_tracks.pt`).

## Deviations from the previous pod's route
* Paths mapped to /workspace/v8proof; HF cache standard (not /root/identity_pod/hf); weights path via env `DINO_SAFETENSORS`; builder `pull_encode_pooled.py` taken from the mirror (`...2026-10-17-refav1-hj-mini/code/`, sha a7dee388, identical bytes to what cache_pod loads).
* Code tree commit 3e264bbf, not 939176e7 (stage/builder files only; not compared against 939176e7).
* Workers 5 (not 13), encode threads 2; no box-2 comparison files; no waiver by me (see STATE); eval cache (141) built too, since the ledger and DONE rule need 4,713.
* Frames and staged sources are deleted at the end (HF is the source); the cache is kept.

## Named blockers
1. Byte-gate waiver decision (above).
2. `data/v9/` is EMPTY. `v9_labels_train.npz` (md5 83e30bd3..., 4,572 clips), `lead_train4572.npz`, `refcv6_speed_max_v8_train.jsonl` exist ONLY on box 2 (make_argvs.py says so; not on the dev box; HF `Sayood/tanitad-refcv8-v9-labels` holds the 4,369-clip release, md5 f63ece41..., and `inputs/lead_train.npz` of unknown clip set; `Sayood/tanitad-band-labels-v10` holds v10 labels). I put NO substitute at the binding paths. The trainer opens these three only when `--join-train` does not exist or `--fit-join` is given (v8f_mini_train.build_data, per make_argvs.py), and the shipped join exists, so training may not need them; verify with the G-DVB/selfcheck before launch. Otherwise they must be shipped from box 2 (not touched by me).
3. torch 2.8.0 on the pod vs 2.11.0 on box 2: affects only numerics, not checked.
