# FEASIBILITY: L2 SAFE-HEAD on the dev box (RTX 4060 8 GB)

Author: REFe scoping agent, 2026-11-26. READ-ONLY investigation; no GPU, no downloads, nothing staged or committed.
Code citations are `PK/refe/<file>:<line>` at mirror branch `agent/arch-inf-20260803` (PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`). The E: working tree does not contain `raw/2026-11-2*`; everything was read with `git show` from `C:/Users/Admin/tanitad-push`.
Evidence classes: MEASURED (probe today) / PUBLISHED-OURS (a banked doc) / INHERITED / ESTIMATED / UNVERIFIED.

## VERDICT: FEASIBLE ONLY WITH X (not a pod)

**X = 4-camera navtrain pixels for the training tokens, staged on the dev box.** They are on no local disk (6 logs only), not on HF, and not in the pod-end bank. They are the single blocker. Labels, the trunk forward and the head training all run on the dev box or are trivial.
- The pixels are public NAVSIM/nuPlan data (`POD_END_AUDIT.md:35`: 77.5 GB, 372,552 files, RE-CREATABLE, NOT on HF). Fetching them is a download, so it needs a PI go (this agent may not download).
- Also needed: a small new head-only trainer (none exists, see d) and a feature-dump script that taps the post-decoder features. Both are CPU-testable.
- A pod is NOT required for any step. The training step is seconds; the forward is ~1 s per token on the 4060 (MEASURED precedent, c).

**Decision for the PI:** approve staging navtrain 4-camera sensor blobs, processed log-by-log and deleted after frame extraction (about 2.8 GB raw per 4-camera log, MEASURED from the 7 local logs: 19.7 GB / 7). A first dev-box run of ~3,000 tokens needs ~35 logs. No pod, no new compute spend.

## a. On-policy navtrain proposal sets and tracked labels: what exists

| item | where | size / count | generator | labels |
|---|---|---|---|---|
| **p6base_labels** (`train_sets/onpolicy_rlfan_p6t_s0of1.jsonl`, `train_pdm/pdm_onpolicy_rlfan_p6t_s0of1.jsonl`, `fit_sets`, `heldout_sets`, `fit_pack/fit_pack.npz`) | HF `Sayood/tanitad-refe-p5c` `pod_end_2026-11-26/bundle_00.tar` (small files, fit, heldout) and `bundle_01.tar` (train_sets, train_pdm, fit_pack). Per-member sizes: `E:/Projects/TanitAD-artifacts/pod_end_audit/bank/manifest.jsonl`. NOT local. | train_sets 2,240,021,832 B; train_pdm 163,737,322 B; fit_sets 253,104,505 B; fit_pdm 18,528,127 B; heldout_sets 63,695,274 B; fit_pack.npz 160,700,266 B. Each bundle is a ~5.3 GB tar, so pulling these members means streaming a whole bundle | **P6's fan** (`MODEL=/workspace/data/refe_p6/model_final.pt`, `raw/2026-11-09-scorer1p6-prereg/labels/p6_labels_chain.sh:19`; sha checked at `:127`). **NOT C1.** | NAVSIM PDM on the executed plans: `onpolicy_relabel_pdm.py` (`p6_labels_chain.sh:88`) -> `navsim_pdm_targets.score_plans` -> `PDMSimulator.simulate_proposals` (`navsim_pdm_targets.py:110`) = the NAVSIM v1.1 LQR-tracked frame; it is the same function the online labeller calls (`online_labeller.py:12`). nc/dac/ep/ttc/c/ddc/pdms per proposal, 64 per set. **Tracked.** |
| token count of p6base | HANDOFF default N_TRAIN=60,000 plus 3,137 heldout plus ~5k fit; the real count is in `labels_manifest.json` / `reports/train.json` inside bundle_00/01 | **UNVERIFIED** (the byte ratio train_pdm/heldout_pdm hints at up to ~110k; do not quote without the manifest) | | |
| **older on-policy sets** (`refe_navtrain/onpolicy`) | HF `Sayood/tanitad-refe-navtrain-banks` `onpolicy_sets_logs_full.tar.zst` (2.07 GB zst, 11.28 GB raw, 27 files) | 27 files | the REFe final/SFT-era model, not C1 | older labelling path; not checked here |
| **C1's own navtrain on-policy sets** | **DO NOT EXIST as a bank.** The SCORER-FT arms labelled online: `--labeller online --label-workers 13 --metric-cache .../metric_cache` (`raw/2026-11-24-scorerft-read/argv_c.json`; local copy `E:/Projects/TanitAD-artifacts/scorerft_read/read_out/argv_c.json`). The online labeller pipes proposals to workers (`online_labeller.py:158`) and nothing persists the sets. | | | |
| C1 / arm C per-proposal tables | local `E:/Projects/TanitAD-artifacts/scorerft_read/propscore/full_SFTC_s{0,1}of2.jsonl` (arm C = C1 + 1,000 steps), P7 F/C fans in `field_read/propscore` | 12,146 navtest tokens x 64, tracked | C-arm / P7 | tracked. ⛔ **NAVTEST = EVAL ONLY**, never training; usable for the final read only. |

So L2's "C1's own 64 proposals per token" must be **generated fresh** (a C1 forward over navtrain tokens) and **labelled fresh** (NAVSIM tracked labeller, CPU). The only banked navtrain sets are another model's fan (P6, and older).

## b. `sctx`, the safety heads, and whether head-only can run from a cache

- `sctx = visual_ctx = scene_proj(visual)`, shape `[B, N_cam*P, dec_width]` = **[B, 7,680, 256]** (4 cameras x 1,920 patches at 512x960, patch 16; `model.py:68-71,87,98`). In the forward it is `visual_ctx.detach()` when `detach_scorer_context` (default **True**, `model.py:152`; used at `model.py:784`).
- Bytes: bf16 = 7,680 x 256 x 2 = **3.93 MB per token** (MEASURED: `E:/Projects/TanitAD/data/refe_trunk_cache/d7c59f4f/navtrain_train/shard_00000/visual_ctx.npy` = 251,658,368 B / 64 tokens).
- **The six outputs are one Linear.** `score_head = nn.Linear(dec_width=256, n_score_components=6)` (`model.py:563`, `:96`). Order NC, DAC, EP, TTC, C, DDC (`scorer_finetune.py:1639`). **1,542 parameters in total** (1,536 weights + 6 biases). The safety rows NC/DAC/TTC/DDC are rows 0,1,3,5 = **1,028 parameters**. The EP row (row 2) stays untouched by writing back only the four rows.
- Upstream of the head: `score_q_mlp` -> 4 x `CrossBlock` (`score_dec`) -> `score_head` (`model.py:699-709`). Scorer total ~4.3M params (ESTIMATED from the block structure at width 256, about 1.05M per block plus 81k q-MLP; consistent with the 34.4 MB `scorer_states_last.pt` in the pod-end bank). The score loss reaches only these three modules when context and trajectories are detached (`scorer_finetune.py:3-6`).
- **Head-only from a cache: YES, at two cache levels.**
  1. **Post-decoder features `s` [64, 256]**: with `score_q_mlp` and `score_dec` frozen, `score_head(s)` is a pure function of `s`. Cache `s` once per token (64 x 256 x 2 B = **32 KiB per token**). Nothing upstream of the head trains. This is the cheap route.
  2. **`visual_ctx` [7,680, 256]** (3.93 MB per token): needed only if `score_dec` or `score_q_mlp` should also train. With the trunk frozen and the context detached, no head input depends on a trainable part upstream. NOTE: C1 and arms C/T/P were trained `--scorer-sees-trunk` (`detach_scorer_context=False`, `train.py:1337-1338`); for L2 the trunk is simply not updated, and the forward is unchanged.
- Capacity caveat (scientific, not feasibility): level 1 is a 1,028-parameter probe on frozen 256-d features. Whether the boundary errors are linearly recoverable from `s` is UNKNOWN. `D-REFE-HEADATTR-1` (`Project Steering/GOALS_AND_CLAIMS.md:16294`) says the heads are confidently wrong at the boundary (pick's DAC output above the passing proposals' median on 70 % of 198 DAC-failed tokens); a re-fit of the same linear read-out may or may not fix that. A small cached set is the cheap decision test.

## c. Cached sctx / trunk outputs for navtrain, and what producing them needs

- **For C1: none anywhere** (local C:/E:/F:, HF `tanitad-refe-p5c`, `tanitad-refe-navtrain-banks`; probed by listing). `ctx_cache.py` binds a cache to the trunk fingerprint and refuses another trunk (`StaleCache`).
- **A local cache exists but for the wrong trunk:** `E:/Projects/TanitAD/data/refe_trunk_cache/d7c59f4f/` (9.0 GB): `navtrain_train` n=1,600, `navtrain_val` n=300, `navtest_sub200` n=200; arrays `visual_ctx, scene_ctx, q0, ego, goal, calib, teacher, props [n,64,20,3], logits [n,64,6]`. Checkpoint `snap_epoch015.pt`, step 4,933, md5 `d7c59f4f...` (an early snapshot, NOT C1); manifest `gpu: NVIDIA GeForce RTX 4060`, built 2026-09-27/28. Unusable as C1 evidence. Its use here is a **timing precedent**: 1,600 tokens written between 23:57 and 00:22 (shard mtimes) = **~0.94 s per token** end to end on the 4060 (MEASURED from file times; includes decode, forward, write).
- **Inputs needed to produce the C1 features:**
  - C1 checkpoint `p5c_c1/model_final.pt`: HF `Sayood/tanitad-refe-p5c` `p5c/c1/` (1.52 GB). The DINOv3 ViT-L/16 weights have an identical-sha copy at `E:/Projects/TanitAD/data/backbones/dinov3-vitl16` (POD_END_AUDIT row).
  - **Navtrain 4-camera pixels: THE BLOCKER.** Pod path `/root/tad/navtrain_pixels`, 77.5 GB, 372,552 jpg, RE-CREATABLE from public sensor blobs, NOT on HF (`POD_END_AUDIT.md:35`). I also listed `tanitad-lorapod-ckpts`, `v8proof-fork-bank`, `tanitadruled-pod-transfer`, `datascale-10k`, `refe-vitl16-navtrain10-grow-final`, `physicalai-w120-256x640cyl`: no navtrain / nuplan / pixel paths. Local nuPlan camera: `nuplan-camera/mini` = 64 logs but **CAM_F0 only** (REFe is 4-camera and refuses a mismatch, `model.py` forward); `nuplan-camera/scenarios4` = 7 logs with all four cameras (19.7 GB), **6 of them in `navtrain.yaml`** (MEASURED, name match). At the average 103,288 / 1,192 = ~87 tokens per log that is ~520 tokens (ESTIMATED).
  - Banks/targets: `train_grow_cmd_3c_p7` (192 MB zst on HF `tanitad-refe-navtrain-banks`), calib table, ego-yaw table (both banked per audit rows 1-2), `splits/navtrain.yaml` (1,192 logs).
  - Labels (tracked): NAVSIM v1.1 devkit + venv at `F:/Archive/devbox-C/navsim/`, maps in `.../navsim/data/maps/`, nuPlan DBs. Local trainval DBs: **1,381; 214 of the 1,192 navtrain logs** (MEASURED, name match). `navsim_pdm_targets.metric_cache_parts` builds from the scenario/DB directly, so the 41 GB metric cache is not needed on this path (UNVERIFIED on the dev box; do a 1-log smoke). A token needs both its DB and its pixels.
- **Cost to make:** GPU ~0.94 s/token (above); CPU labels 4.49 sets/s at 6 workers (INHERITED, SPEC_P6:187; the dev box has 24 logical cores, 31.8 GB RAM). Per 3,000 tokens: forward ~47 min GPU, labels ~11 min at 6 workers. Per 10,000 tokens: forward ~2.6 h, labels ~37 min. Pixels ~0.83 MB per token (77.5 GB / 93k frames, ESTIMATED): 3,000 tokens ~2.5 GB extracted, 10,000 ~8.3 GB, so it fits E:/F: if extracted log-by-log.
- Storage fit (free: C 7.0, E 15.0, F 17.3 GB, MEASURED): the `s` cache is 32 KiB/token (10k tokens = 0.33 GB). A `visual_ctx` cache is 3.93 MB/token: at most ~3,000 tokens on E:. So level 1 is the only practical level.

## d. SCORER-FT trainer: detach / head-only / freeze modes

- **`scorer_finetune.py`** (SFT-4 era trainer): always freezes everything except `SCORER = ("score_q_mlp","score_dec","score_head")` (`:92`, `:1765-1769`; the env `REFE_SFT_MUTATE_UNFREEZE` is a deliberate regression that re-opens `traj_head`, `:1770-1772`). So **trunk-frozen scorer-only is its default, not a flag.** It has **no head-only mode**: no flag trains only `score_head` or only rows 0,1,3,5. It reads `--onpolicy` / `--pdm-labels-train` sets and needs `--images` (live trunk) unless `--ctx-cache DIR` (default `""` = live trunk, `:1689`), which serves cached `visual_ctx` (cache trunk fingerprint must equal the checkpoint's). `--ctx-cache` is refused with `--aux-drv` (`:1361`). Defaults: `--lr 3e-5`, `--batch 8`, `--accum 8`, `--b-mode listnet`, `--lam 0.3`.
- **`train.py`** (the C/T/P arms' trainer): flag `--scorer-sees-trunk` (`train.py:2258`, `dest=detach_scorer_context`, `store_false`). Absent = `detach_scorer_context=True`. Trace: `train.py:1337-1338` sets `cfg.detach_scorer_context=False` when named -> `model.py:152` (default True) -> `model.py:784` `sctx = visual_ctx.detach() if self.cfg.detach_scorer_context else visual_ctx`. With the flag the PDM loss reaches the backbone LoRA; without it the context is detached. Arm C/T/P argv names the flag explicitly (`argv_c.json`).
- **"P-detach" (`--rank-detach-ctx`) is NOT in `train.py` or `scorer_finetune.py` at the tip** (grep: 0 hits in both). It exists only as spec text in `raw/2026-11-20-scorer-ft/PREREG_SCORER_FT.md` and the P card (PL2). It would put the rank term on a second scorer pass over a detached context: an ablation of the rank loss, not a safety-head trainer.
- **Conclusion for L2:** a new small script is needed: load C1, dump `s` via the existing forward, train `score_head` rows 0,1,3,5 with a hinge/focal loss on cached `s`. It reuses `model.score_trajectories` / `score_dec` for the feature dump (`model.py:699`). A new flag in a pinned trainer needs a G-DVB entry (launch-gate rule 3); a separate script avoids that but still needs a launch-gate profile if it counts as a training launch (PI rule 2026-09-26). Flag for the MM.

## e. Estimated memory and wall time, 1,000 head-only steps on the RTX 4060

**Level 1 (rows 0,1,3,5 of `score_head`, from cached `s`) - recommended.**
- Params 1,028 (+ Adam states 2 x 1,028 fp32, 8 KB). Batch of 256 sets: 256 x 64 x 256 x 2 B = 8.4 MB. FLOPs per step: 256 x 64 x 256 x 4 outputs x 2 x 3 ~= 0.1 GFLOP.
- **Memory: well under 0.5 GB GPU (or none; it runs on CPU).** **Wall: ~10 s on GPU, ~1 min on CPU per 1,000 steps.** Data: 10,000 tokens x 32 KiB = 0.33 GB in RAM.
- The real cost is producing `s` and the labels (c): ~0.94 s/token forward on the 4060 (shared GPU lock), labels on CPU in parallel.

**Level 2 (also train `score_dec` + `score_q_mlp`, ~4.3M params, from cached `visual_ctx`) - not recommended.**
- Per set: K/V projection 2 x 7,680 x 256 x 256 = 1.0 GMAC per block x 4 blocks ~ 8 GFLOP, plus attention ~2 GFLOP: ~10 GFLOP forward, ~30 GFLOP fwd+bwd. Batch 16: ~0.5 TFLOP/step. At an assumed 4 TFLOP/s sustained bf16 (ESTIMATED, ~25 % of peak) ~0.12 s/step, so 1,000 steps ~2 min of compute.
- IO: 16 x 3.93 MB = 63 MB per step = 63 GB per 1,000 steps (5-10 min from disk at 100-200 MB/s, ESTIMATED). GPU memory: ctx 63 MB + saved K/V projections ~0.5 GB + attention (SDPA assumed) so under ~2 GB. The limit is **cache storage (3.93 MB/token, at most ~3,000 tokens on E:)**, not GPU memory.

Honest caveat: the compute is trivial; the schedule risk is the pixels download and the shared GPU lock (a forward over N tokens holds the lock about N seconds; run it in per-log units like `ctx_cache.py`, which takes the lock once per unit).

## f. Verdict and exact build steps

**FEASIBLE ONLY WITH X, X = navtrain 4-camera pixels staged locally (PI go for a public-data download).** Not a pod item.

Build steps (all dev-box, none needs a pod):
1. **PI go** on staging navtrain 4-camera sensor blobs for a chosen log list (start: ~35 logs for ~3,000 tokens; ~2.8 GB raw per 4-camera log; extract frames to 512x960 jpg, delete the raw zips after, rule 1 of the delete policy). Use the 6 locally complete logs (`nuplan-camera/scenarios4`) for the first smoke (~520 tokens, ESTIMATED).
2. Pick logs from `splits/navtrain.yaml` whose nuPlan DB is local (214 of 1,192, MEASURED), outside the 24 held-out logs (`splits/sft_heldout_logs.txt`) and outside navtest (eval only).
3. C1 `model_final.pt` from HF `tanitad-refe-p5c` `p5c/c1/` (1.52 GB, a download: PI go); DINOv3 ViT-L is already local. Check the C1 meta (nav_mode command, `traj_label_*`, `config_for_checkpoint`) so the forward matches the C1 build (the P6 dump once got nav_mode wrong, `labels/HANDOFF.md` defect note).
4. **Feature + fan dump** (new script, GPU, shared lock, per-log units): forward C1 on each token with `calib`; write the 64 proposals `[64,20,3]` and `s` (post-`score_dec`, `[64,256]`, bf16). ~0.94 s/token (ESTIMATED from an earlier checkpoint). Write set records like `rl_fan_dump.py` so labelling is unchanged.
5. **Tracked labels** (CPU): `onpolicy_relabel_pdm.py --sets <dir> --out <dir> --workers 6..12` on `F:/Archive/devbox-C/navsim/`. Smoke on 1 log first and compare against a banked P6 log labelled the same way. UNVERIFIED that `index_dbs()` works on the local trainval DB layout.
6. **Head trainer** (new, small, CPU or GPU): rows 0,1,3,5 only; boundary-focused hinge/focal on tokens near the pass/fail boundary; EP row and every other parameter byte-identical to C1 (assert it). Needs its own launch-gate profile.
7. **Eval** on full navtest (12,146 tokens): another C1 forward, ~3.2 h on the 4060 (ESTIMATED), or reuse a navtest context if one exists for C1 (none found). Four metric families, paired log-cluster bootstrap, tier T1 (binding).

Zero-download-free option today: none for L2 itself. A related CPU-only precheck: fit a calibrator on `p6base_labels/fit_pack/fit_pack.npz` (P6 logits + PDM labels, ~8,000 sets, "no images needed", `labels/HANDOFF.md`). It is P6, not C1, so it only says whether re-fitting the heads helps at the boundary at all. It needs bundle_01 (5.36 GB tar, a download).

## UNVERIFIED / INHERITED list
- UNVERIFIED: p6base token count; whether bundle members stream out of HF without pulling the whole ~5.3 GB tar; dev-box `index_dbs()` on the local DB layout; that the dev-box NAVSIM venv runs `onpolicy_relabel_pdm.py` unchanged; C1's nav-mode/calib build; C1-forward throughput on the 4060 (0.94 s/token is from an earlier checkpoint of the same architecture).
- INHERITED: 4.49 sets/s at 6 workers (SPEC_P6:187).
- The "6 of 7" and "214 of 1,192" overlaps compare log-name strings only; a DB or zip existing is not proof it is complete.
- HF was probed by file listing only (no downloads): `tanitad-refe-p5c` (95 files, 50.4 GB), `tanitad-refe-navtrain-banks` (19 files, 45.1 GB), plus the five repos named in c.

## Deliverable manifest
| artifact | where | copies |
|---|---|---|
| `FEASIBILITY_L2.md` (this file) | `E:/Projects/TanitAD-artifacts/safehead/FEASIBILITY_L2.md` | **ONE place only**; not staged (task override); the Master Mind lands it |
| scratch (extracted source files, HF listing helper) | `C:/Users/Admin/AppData/Local/Temp/sh/` | deleted at the end of the run (own temp) |
