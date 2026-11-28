# FACTS_LEVER3 — re-running RH stage B (RH_s2 recipe) on the dev box (2026-11-28)

Evidence classes: MEASURED = read/computed this session at the stated path; INHERITED = read from a record, not re-verified; ESTIMATED.
Code = commit 97fe0a56 of C:/Users/Admin/tanitad-push (read with `git show 97fe0a56:<path>`). Nothing run on a GPU. Box 2 not touched.

## 1. INPUT INVENTORY

### 1a. How the trainer finds clips (MEASURED, source at 97fe0a56)
* ONE root only. `stack/tanitad/data/v8f_mini_data.py:120-129` `read_manifest(cache_dir, split)` reads `<cache>/<split>/manifest.jsonl` (clip, n_frames). The manifest is only a clip LIST, it carries no file location.
* `v8f_mini_data.py:533` `self.dir = Path(cache_dir) / split / "clips"`; clips opened by `self.dir / f"{clip}.pt"` (`:552`, `:576`). Same pattern in `stack/tanitad/data/ra1_pooled.py:138`. The trainer takes one `--cache-dir` (`v8f_mini_train.py:3100-3101`); the bank generator passes it through (`v8f_dagger_bank.py:258, :435`).
* So: NO multi-root, NO list of roots. All 14,512 files must appear as `<cache>/train/clips/<clip>.pt`.
* Several drives with the code as it is: (a) NTFS directory junctions do NOT work (a junction is a directory; the code needs one FILE per clip). (b) File symlinks would work, but creating one failed here ("Administratorrechte erforderlich", MEASURED in F:/TanitAD-scratch/rollhist/lever3/symtest); needs Windows Developer Mode or an elevated shell = a settings decision for the user. (c) Hardlinks only within one volume. (d) No single drive holds the full-size union (section 4). Cleanest no-privilege route: shrink the clip files (4.B) so ONE drive holds them.
* Drives (MEASURED): F: and C: NTFS, E: exFAT (no links at all), G: = Google Drive mount (do not use).

### 1b. Per input of the RH_s2 argv
| input (pod path) | copy now | size | re-creatable by |
|---|---|---|---|
| `--cache-dir /root/data/cache_union_dscale` (14,512 clips = 4,512 corpus + 10,000 10k) | NONE as a union; pieces below | ~304 GB decimal | build from the two parts |
| corpus part, POD encode (`cache_pod/train`, enc sha 308f36c9) | GONE with the pod (INHERITED LOOP_STATE_2026-11-02.md l.4474: not banked) | 91,109 MB | re-encode ~4 GPU-h on L40 (INHERITED) |
| corpus part, DEV-BOX encode (enc sha 361bc287) | F:/TanitAD-scratch/speedread/cache/train (manifest.jsonl + 4,572 clips = 4,512 corpus + 60 holdout 540:600; `du` 89 GiB; no meta.json beside it). Also on box 2 (INHERITED, not touched) | 89 GiB | exists |
| 10k part (pod encode, enc 308f36c9) | HF dataset Sayood/tanitad-datascale-10k `cache/s000_pilot200 … s041.cache.tar`, 42 tars, 208,760,043,520 B (MEASURED list_repo_tree). Each tar: `train/clips/*.pt`, `train/manifest.jsonl`, `train/meta.json`, `train/pod_provenance.jsonl`. Not downloaded | 208.76 GB | pull from HF |
| `--v9-train merged/v9_labels_union.npz` (md5 9881b170…, 2,042,576,298 B, sha256 c630dab0…) | NO COPY. Its manifest json is in the records tar | 2.04 GB | merge (1c) |
| `--lead-train merged/lead_union.npz` (md5 73693175…, 277,242,630 B, sha256 e96c2d5e…) | NO COPY | 277 MB | merge (1c); needs the corpus lead file (gap below) |
| `--speed-train merged/speed_union.jsonl` (2,794,831 B, sha256 fd808a26…) | IN the records tar: HF Sayood/tanitad-v8f-rollhist-runs:records/mm_records_1124.tar.gz member `dscale10k/merged/speed_union.jsonl` (tar downloaded to F:/TanitAD-scratch/rollhist/lever3/rec/records/, 210,244,733 B; member size matches, sha not re-hashed) | 2.8 MB | direct from tar |
| `--join-train merged/join_union.npz` | NO COPY, and not a real input: argv has `--fit-join`; `v8f_mini_train.py:3101-3113` rebuilds the join (`D.build_label_join`) from manifest + v9 + lead + speed and saves it to that path | n/a | the trainer on first start |
| `--tracks merged/native_tracks_union.pt` (51,494,151 B, sha256 90229524…, 14,653 tracks) | NO COPY | 51 MB | merge (1c) |
| `--p10 /root/strat1/inputs/p10/p10_prior.pt` | NOT IDENTIFIED: no record found carries its sha. Candidates (MEASURED sizes/sha): E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/p10/p10_prior.pt (74,539 B, sha256 14a062a91078266a…, same "strat1/sfresh inputs" family, likeliest); repo blob stack/tanitad/configs/v8f_p10/p10_prior.pt (74,475 B, af5a29dd…); repo raw/p10_yaw/p10_prior.pt (59f0fd92…) | 74 KB | the P10 fit (`stack/tanitad/models/p10_prior.py`, `--kin-mode yaw`); GAP: confirm which file |
| `--init-ckpt …/UE_s2/A/ckpt_last.pt` | HF dataset Sayood/tanitad-v8f-dscale10k-runs:UE_s2/A/ckpt_last.pt, 86,970,901 B (= the stage-A sha 6d434525… named in the chain log, INHERITED) | 87 MB | pull |
| `--dagger-bank /root/rollhist/bank/s2` (manifest sha256 60b4eaca…) | shards NOT local and NOT on HF (rollhist-runs has no bank). Only bank.json / PROVENANCE.json / BANK_CHECK.json / S3_BANK_CHECK.json (0.7–9.9 KB) at F:/TanitAD-scratch/rollhist/seed3/pod_s3_run/bank/s2. Chain log: "s2 bank kept 2.0G" | 2.0 GB | regenerate: `stack/scripts/v8f_dagger_bank.py` from the UE_s2/A ckpt, readout causal_g1x (`--readout causal_g --acc-gain 1 --path-exec --kappa-max 0.2`), 985,048 windows |

### 1c. The merge (MEASURED: records tar `dscale10k/merge.sh`, `dscale10k/merged/merge_record.json`)
`dscale10k_merge.py merge --corpus-manifest <pod cache_pod/train/manifest.jsonl> --tenk-manifest in/manifest_10k.jsonl --v9 <corpus> <10k> --lead <corpus> <10k> --speed <corpus> <10k> --tracks <corpus> <10k> --out-dir merged`. It drops the 60 holdout clips (540:600) from the corpus (4,572 -> 4,512), union = 14,512.
Script copies (sha256 89d4039966cd…, 35,115 B): F:/TanitAD-scratch/datascale/pod/deliver/study/dscale10k_merge.py (also `.../gate/pod/`, `.../pod/x/study/`, F:/TanitAD-scratch/dscale10k/reps2/pod_ref/); repo path "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-15-v8f-datascale/build/pod/study/dscale10k_merge.py". An older 33,298 B copy (sha 308847bd…) is at F:/TanitAD-scratch/datascale/build/. Which one the pod ran is INHERITED-likely the 35,115 B one (merge_record does not hash the script).
| merge input | copy now |
|---|---|
| corpus manifest (pod, sha256 d5c17f3a…) | not the dev-box file (content_md5 differ, section 2). The union manifest.jsonl (9,677,662 B, sha256 4d2d4479…) with pod content_md5 for all 14,512 clips IS in the records tar |
| tenk manifest `in/manifest_10k.jsonl` (6,668,743 B, sha256 f2928333…) | records tar `dscale10k/in/manifest_10k.jsonl` |
| v9 corpus (264,220,433 B, sha256 b2efe864…) | MEASURED match: F:/TanitAD-scratch/strat1/ckpt120build/labels/rec/v9_labels_train.npz |
| v9 10k (561,638,443 B, sha256 74648b91…) | HF Sayood/tanitad-datascale-10k:labels/v9_labels_datascale10k.npz |
| lead 10k (30,402,235 B, 5c13b204…) | HF …:labels/lead_datascale10k.npz |
| speed 10k (1,926,195 B, 2489c03d…) | HF …:labels/speed_sidecar_datascale10k.jsonl (also records tar) |
| tracks 10k (35,261,029 B, 7eb96925…) | HF …:labels/native_tracks_datascale10k.pt |
| tracks corpus (16,591,688 B, sha256 1db431f1…) | MEASURED match: E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt and F:/TanitAD-scratch/strat1/ckpt120build/in/native_tracks.pt |
| speed corpus `refcv6_speed_max_v8_train.jsonl` (880,186 B, md5 bab41adf…) | no standalone copy; its rows are inside speed_union.jsonl (4,512 kept) |
| lead corpus `lead_train4572.npz` (13,739,581 B, md5 d4336a44…, sha256 25494fd9…) | GAP: not found under E:/Projects, F:/TanitAD-scratch, E:/TanitAD-scratch, C:/Users/Admin (find, depth 9). E:/Projects/TanitAD-artifacts/v9labels/inputs/lead_train.npz is a DIFFERENT file (13,633,497 B, md5 67451737…). Route: rebuild with the v9 lead builder from the corpus, verify md5 d4336a44; or take it from box 2 (INHERITED: it was fed to the pod from strat1/inputs/data/v9/) |
Gaps with no copy: lead_train4572.npz; identity of p10_prior.pt; the s2 bank shards (regenerable); the pod corpus encode (regenerable, ~4 GPU-h L40).
The argv pins `--v9-train-md5 9881b170…` (union npz md5), so the re-merge is checkable against merge_record.json; the trainer refuses another md5.

## 2. ENCODE IDENTITY (MEASURED: F:/TanitAD-scratch/rollhist/lever3/ENCODE_IDENTITY.json, script identity.py)
* Pod eval tar `pod_data/mm_evalcache_1124.tar`: 2,963,589,120 B, sha256 cc2170f1baf5e289f7c7b3f0dcc2e117c5345166d8a67651f08b6ece5f535831 VERIFIED equal to the banked value; 148 members; streamed with tarfile, never extracted.
* Dev-box eval cache of the same 141 clips: E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval (meta.json encoder_module_sha256 361bc287c306a8f4…, torch 2.11.0+cu128, transformers 5.15.0). Pod meta.json: 308f36c949979495…, same torch/transformers.
* Result over 141 clips, 0 errors: **bitwise equal 0/141** on both grids.
  * pool4x10 rel-L2 (pod vs dev, fp8 dequantised to float32): mean 0.02411, median 0.02375, min 0.02090, max 0.02889.
  * pool8x20 rel-L2: mean 0.02768, median 0.02714, min 0.02455, max 0.03321.
* Corpus part: dev-box F:/TanitAD-scratch/speedread/cache/train manifest vs the pod's union manifest, 4,512 common clips: content_md5 equal 0/4,512.
* Consequence: a union of the dev-box corpus part (361bc287) and the HF 10k part (308f36c9) is a MIXED-encode cache (~2.4% per-token rel-L2 between the encodes). The pod's training data cannot be bit-reproduced without re-encoding the corpus part. Whether 2.4% matters is the MM's call (no training measurement here).
* Deleted after use: the downloaded tar, logged in F:/TanitAD-scratch/rollhist/lever3/DELETIONS.log (path, 2,963,589,120 B).

## 3. 10k PART'S ENCODE (MEASURED)
* HF Sayood/tanitad-datascale-10k, `train/meta.json` read by HTTP range through the tar headers of two shards (s001_smoke20, s041; no shard downloaded): both record `encoder_module_sha256 308f36c949979495cd454ac7a20a2cc1dffe0369577cb91f44183d7af52101fd` (`/root/tad/stack/scripts/dinov3_fp8_encode_ship.py`), DINOv3 ViT-L/16, torch 2.11.0+cu128, pipeline ra1a-pooled-v1, bf16 encode, fp8 e4m3fn stored. The other 40 shards were not opened.
* Total cache shard size 208,760,043,520 B = 208.76 GB = 194.4 GiB (42 tars).

## 4. DISK LAYOUT
Free now (MEASURED): C 104 GiB, E 113 GiB (exFAT), F 84 GiB (F already holds the 89 GiB dev corpus cache). G is Google Drive.
One clip file = 20.9 MB (T=101): pool4x10 4.14 MB, pool8x20 16.55 MB (80%), mean 0.21 MB, rest ~0.01 MB (MEASURED on an eval clip). The run uses `--grid pool4x10`; grep at 97fe0a56 finds no read of `pool8x20` (only an argparse choice and a docstring "optional") and no read of clip-file md5/bytes in trainer, data class or bank generator.
* A. FULL-SIZE union (~304 GB decimal, ~283 GiB): fits no drive and cannot be spread with the code as it is (1a). Works only with file symlinks (Developer Mode / admin) or a trainer change (multi-root lookup in `ClipStore` and `ra1_pooled`): both decisions for the MM/PI.
* B. POOL4x10-ONLY union (ESTIMATED 14,512 x ~4.36 MB = ~63 GB decimal = ~59 GiB): repack each clip while extracting it from its shard tar, dropping only `pool8x20` (the pool4x10 tokens stay bitwise equal to the source). One volume, no links, no code change:
  * Put `cache_union/train/clips` on C: (104 -> ~45 GiB free), bank 2 GiB and labels ~3 GiB next to it -> ~40 GiB free. Alternative E: (exFAT, 113 -> ~54 GiB). F: would end near 20 GiB: not recommended.
  * Build peak: one 5.2 GB shard tar at a time in F:/TanitAD-scratch/rollhist/lever3/dl, repack, delete that tar by literal path (rule 1), 42 times. Corpus part: repack from F:/TanitAD-scratch/speedread/cache/train (leave the source, it is not this agent's).
  * OPEN: the union manifest rows carry bytes/md5/content_md5 of the FULL files; after repacking those fields describe files that no longer exist. Whether `launch_gate.py` / G-DVB hashes the cache manifest is NOT verified; check before choosing B.
* Bank: 2 GiB next to the cache, fits either.

## 5. COMPUTE
* CPU i9-12900F: 16 cores / 24 logical. RAM 31.84 GiB. Commit limit 67,008,580 KB = 63.9 GiB total; free commit at the probe 8.07 GiB (the box is shared and loaded; memory note: gate jobs need >= 5.5 GB commit). GPU RTX 4060 8 GB, shared; GPU lock qland/work/refcv7/devbox_gpu.loc (INHERITED memory note).
* Bank generator entry: `stack/scripts/v8f_dagger_bank.py gen` (+ a merge step that writes bank.json), `--device auto|cpu`; device resolved at `:274` (auto = cuda if available); `rollout_chunk` defaults to cpu, so it needs no GPU strictly. Args: `--ckpt --m2b-argv --cache-dir --tracks --join-train --eval-grid --out-dir --shard --n-shards --expect-clips 14512 --expect-windows 985048 --readout causal_g --acc-gain 1 --path-exec --kappa-max 0.2` (the 4,512 / 306,566 values in its help are the old corpus numbers). `--limit-clips N` = smoke (bank marked PARTIAL).
* L40 references (MEASURED from local copies): stage B 3,000 steps = 615.77 s (`seed3/pod_s3_run/runs/RH_s2/train/summary.json`-adjacent log: "DONE … seconds 615.77"; train.log). Bank s2: rh_s3_chain.log "step bank_s2" 18:21:44Z to "step RH_s2" 20:19:08Z = 7,044 s (117 min) for 4 parallel shards + merge + check over 985,048 windows (pod CPU quota 15.3 cores; rollouts were CPU-bound, GPU near 0%, INHERITED memory note). Dev-box time ESTIMATED 2-4 h, unmeasured.
* Smoke would need: a 20-60 clip cache (bank `--limit-clips`, trainer `--max-clips`); the launch-gate PASS token (binding rule; G-LIVE does a real-data forward); init ckpt from HF; union v9/lead/speed/tracks or a slice from the merge tool; venv python C:/Users/Admin/venvs/tanitad/Scripts/python.exe with `;` PYTHONPATH; the shared GPU lock. Batch 64 on the 87 MB model with 40-token grids on 8 GB: untested.

## Not run / unverified
Records-tar member sha vs merge_record (size only); p10 identity; the other 40 shard metas; whether the launch gate pins the cache manifest; lead_train4572 provenance; any GPU work.
