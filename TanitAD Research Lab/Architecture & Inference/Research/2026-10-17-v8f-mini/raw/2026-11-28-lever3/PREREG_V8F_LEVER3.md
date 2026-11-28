# PREREG_V8F_LEVER3 (FROZEN): does a longer stage B cut RH's rollout-age cost? Rebuilt on the dev box

**Status:** FROZEN 2026-11-28 after the PI's go ("Plan it on dev box", then "Yes, build and run"), before any build step. Nothing had run except the p10 identity check of section 3, which reads an input, not a result.

**Choices fixed at the freeze:**
* Build root: `C:/Users/Admin/lever3/cache_union` (NTFS), a `train/clips/<clip>.pt` layout.
* Argv templates: `rollhist/argv/RH_s0.tmpl.json` and `RH_s1.tmpl.json` from the records tar (`records/mm_records_1124.tar.gz`), with `render_report.json` naming readout causal_g1x and no problems for either seed.
* Executor: ADD154 had not read at the freeze, so the head closing of row 218 is NOT in the executor. The executor is row 211's reader-form rule.
* The eval cache's sha256 is named in an amendment frozen before any rollout of this card.
**Facts used:** `FACTS_LEVER3.md` and `ENCODE_IDENTITY.json` in this folder (MEASURED 2026-11-28 by an MM agent, CPU only).

## 1. Why
* RH (stage B of the recipe of record, PI 2026-11-24) under readout K is the candidate model (row 206).
* The candidate's own age cost (RH under K, after 6 s of its own driving minus the same 1,271 windows started fresh) has not been read. Two neighbouring reads frame it (MEASURED, box 2):
  * the line model under K loses +14.204 m of tactical track error (label form; the C5 reference of the specification, row 206);
  * RH with the rule under the causal readout loses +16.611 m [+15.464, +17.825], with every family WORSE (row 205, C5).
  * Aged, RH under K is 1.028 m [-1.838, -0.316] better than UE under K (row 206, R3). So the recipe of record removes a small part of an age cost of about 14 to 17 m.
* The untried form of lever 3 is a **longer stage B** at the adopted own-history share p 0.1. A larger share was measured already: RHp3, p 0.3, failed G-T0.
* Stated guess before data: the longer stage B lowers the 6-s tactical track error by about 1 to 3 m, with no family worse fresh.

## 2. What changes because the pod is gone (MEASURED)
* **The pod's corpus encode no longer exists.** On the 141 eval clips, the dev-box encode (encoder_module 361bc287) differs from the pod encode (308f36c9): 0 of 141 clips bitwise equal; pool4x10 rel-L2 mean 0.0241 (median 0.0237, max 0.0289).
  * The 10k part on HF is the pod encode, 308f36c9. A union rebuilt here therefore mixes encodes: the 10k part (10,000 clips) from the pod, the corpus part (4,512 clips) from the dev box.
  * **So no arm here is "RH as trained".** Every comparison of record is made between arms trained on the SAME rebuilt union (section 4). RH itself (pod) is printed beside only.
* **The trainer reads clips from one root** (`v8f_mini_data.py:533,552,576`: `<cache>/train/clips/<clip>.pt`). Directory junctions do not help, and file symlinks are refused here. No dev-box drive holds the 283 GiB union.
  * The union is therefore REPACKED to the pool4x10 tensors only, about 59 GiB (ESTIMATED), on one drive (C: or E:).

## 3. Build (no GPU; each step has its control, and a failed control stops the build)
1. **10k part.** Pull the 42 shard tars of `Sayood/tanitad-datascale-10k` from HF one at a time: tar sha256 against the hub LFS sha, extract, repack each clip to pool4x10, then delete the tar and the extract by literal path and log it.
   * Control b1: on every clip, the repacked pool4x10 tensors are byte-equal to the source file's.
2. **Corpus part.** Repack `F:/TanitAD-scratch/speedread/cache/train` (4,512 train clips of the union, dev-box encode) the same way. Control b1 again.
3. **Labels.**
   * Re-merge v9, lead and tracks with `dscale10k_merge.py` at 97fe0a56. Control b2: the merged v9 file's md5 equals the argv record, 9881b170a889f743c5b0894f9ad13f1d. If it does not, stop and report the difference.
   * The corpus lead file `lead_train4572.npz` (md5 d4336a44) is copied from box 2 when box 2 is idle, with its md5 checked.
   * Control b2 also covers the other merged files: lead_union sha256 e96c2d5e…, native_tracks_union sha256 90229524… (14,653 tracks), and speed_union.jsonl (from the records tar) sha256 fd808a26…, each equal to `merge_record.json`.
   * `p10_prior.pt` (MEASURED 2026-11-28, `p10id/P10_IDENTITY.json`): no pod record carries its sha. The frozen prior saved in UE_s2/A's checkpoint (sha256 6d434525…) equals `E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/p10/p10_prior.pt` (sha256 14a062a9…) on all 7 tensors, bitwise. The two repo copies (`configs/v8f_p10`, `raw/p10_yaw`) differ by up to 28.9. That file is the card's `--p10`.
     * Control b5: the same equality holds for UE_s0/A and UE_s1/A before their arms start.
   * `join_union.npz` is rebuilt by the trainer under `--fit-join`.
   * The union manifest is written anew for the repacked files (their bytes and md5). At 97fe0a56, grep finds no reader of a clip file's bytes or md5 in the trainer, the data class or the bank generator. The gate fingerprints every data path on the launch host, so the new cache binds its own fingerprint.
   * Control b4, the argv: each arm's argv equals RH_sK's recorded pod argv, except paths, the bank and its sha256, `--steps` (D9K only) and nothing else.
4. **Control b3, the reach of the repack.** The trainer's first batch and the bank generator's first chunk, from the repacked union, are byte-equal to the same calls on a 20-clip original-format cache. Deleting the pool4x10 tensor from one clip must make both fail (a mutation).
5. **Own bank per seed.** `v8f_dagger_bank.py` from UE_sK.A (HF `tanitad-v8f-dscale10k-runs`) under `--readout causal_g --acc-gain 1 --path-exec --kappa-max 0.2`, over the rebuilt union's TRAIN windows. BANK_CHECK and coverage as in the ROLLHIST card. The bank is new, because its tokens are.
6. **Launch gate.** `launch_gate.py` PASS for every arm, bound to the commit, the argv sha256 and the new data-manifest sha256s. G-LIVE includes a GPU-memory smoke on the RTX 4060 at batch 64; if it does not fit, the arm stops and comes back to the PI. The batch is not changed under this card.

## 4. Arms (seeds 0 and 1; a third seed only if a cell lands within the floor of a bar)
| arm | stage B | steps | role |
|---|---|---|---|
| `D3K_sK` | RH's argv, rebuilt union, own bank | 3,000 | the same-environment control (the recipe as adopted, rebuilt here) |
| `D9K_sK` | the same, `--steps 9000` (warmup 100; the cosine runs over its own 9,000 steps) | 9,000 | the lever |

The cosine schedule ends at 0.1 x lr at `--steps`, so a state saved inside a run is not an end state. The control is a separate run.

## 5. Reads (dev-box CPU, the fork's frozen readers and pins, unchanged)
* **Executor:** T1-kin/hb0.2, readout K, the candidate's reader-form rule as of row 211 (every step). The head closing of row 218 enters only if card ADD154 reads SCENE-CARRIED before this card freezes.
* **Primary, the age cost (C5 form):** `D9K - D3K` on the 1,271-window grid after a 30-step pre-roll (6 s). Tactical track error, with the four families, U, C and F beside.
* **Fresh:** `D9K - D3K` on eval-141: four families, U, C, F.
* **Estimator:** paired episode-cluster bootstrap pooled over the two seeds. It answers "another draw of episodes"; the two seeds answer "another training run".
* **Floor:** the spread of `D3K` over the seeds.
* **Eval cache:** every read uses the eval cache of the fork's RH rollouts of row 206, named by its sha256 when the card freezes, so the beside read compares like with like.
* **Beside, names nothing:**
  * `D3K - RH (pod)` at both ages, which shows the environment shift of the rebuild;
  * each arm's own age cost (aged minus fresh on the same 1,271 windows, the C5 form), which reads the candidate's age cost for the first time. That is up to 4 more rollouts.

## 6. Names (committed now)
| name | condition | next |
|---|---|---|
| `LEVER3-LONGER-B-HELPS` | aged tactical track error BETTER (interval below 0, both seeds below 0, \|D\| > floor); no family WORSE fresh or aged; U not HIGHER at either age | the candidate's stage B becomes 9,000 steps, back to the PI with the cost of re-reading the candidate |
| `LEVER3-NO-GAIN` | aged tactical track error not separated | the age cost is not a length lever at p 0.1; the next is a deeper own-history bank (states rolled up to 6 s), which needs a generator change and its own card |
| `LEVER3-COST` | aged better, but a family WORSE or U HIGHER | to the PI with the trade-off |
| `LEVER3-BLOCKED` | any build control fails, or the GPU smoke does not fit | report the control and its difference |

## 7. Cost (ESTIMATED; PI go needed)
* **Build, no GPU:**
  * the 10k download is about 209 GB; at an assumed 10 to 20 MB/s that is 3 to 6 h, network-bound;
  * repack, labels and controls take about 2 h of CPU;
  * the launch gate takes about 2 h of CPU, one pytest at a time behind the commit-headroom gate.
* **Banks:** 2 seeds x 117 min on the L40 (`rh_s3_chain.log`). On this box's 16 cores, ESTIMATED 2 to 5 h each.
* **Training:** the L40 ran 3,000 steps in 616 s. On the RTX 4060, ESTIMATED 3 to 4 x slower: D3K about 0.6 h and D9K about 1.8 h per seed, about 5 h of the shared dev-box GPU for both seeds. That is under the dev-box GPU lock, which another session holds at times.
* **Reads:** 8 rollouts (2 arms x 2 seeds x 2 ages), plus up to 4 for the age cost beside; about 1 to 1.5 h of CPU.
* **Total:** about 1.5 to 2 days of dev-box wall time, mostly download and banks; about 5 GPU-h. No pod and no spend.
* **Disk (free now, MEASURED: C: 104 GiB, E: 113 GiB exFAT, F: 84 GiB):** about 59 GiB for the repacked union, 4 GB of banks and 3 GB of labels, on C: (NTFS), which ends near 40 GiB free.
  * One 5.2 GB shard tar at a time is staged on F: and deleted after its repack.
  * The build stops if C: falls below 25 GiB free.
  * E: is the fallback.
