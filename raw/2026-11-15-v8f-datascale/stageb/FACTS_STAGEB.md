# FACTS_STAGEB (2026-11-19): every measured fact behind PREREG_DSCALE_STAGEB_DRAFT.md

Evidence classes: M = MEASURED here (command and output named), I = INHERITED (another doc, not re-verified here), E = ESTIMATED (arithmetic on M), U = UNVERIFIED.
Mirror tip read: `cc7014dc` (branch `agent/arch-inf-20260803`). `RD` = `raw/2026-11-15-v8f-datascale` in the mirror. Pod reads were read-only (`ls`, `cat`, `grep`, manifest and npz reads on CPU). The pod GPU and its processes were not touched. One scratch file written on the pod (`/tmp/p_minus_u.json`, 2,400 bytes) was deleted by literal path.

## A. Arms, trees, checkpoints

| id | fact | class | source |
|---|---|---|---|
| A1 | B set = 4,512 clips, 306,566 windows (the 4,572-clip release minus the 60 RA-1 holdout clips, `--exclude 540:600`). U set = B set + 10,000 data-scale clips = 14,512 clips, 985,048 windows. | I | card `CARD_DATASCALE_TRAIN_DRAFT.md` lines 46-47 and 9; `LAUNCH_X_s0.md` (985,048 windows for U_s1) |
| A2 | UE = U set, stage A 61,568 steps (4 x S_U). BL = B set, stage A 61,568 steps. Stage B = 3,000 steps, `--w-acc 49.6969 --detach-upper --dagger-p 0.1`, DAgger bank from the arm's own stage-A checkpoint over its own windows. | M | pod `/root/dscale10k/argv/{UE_s0.A,BL_s0.A,U_s0.B.tmpl,B_s0.B.tmpl,U_s1.B.tmpl,B_s1.B.tmpl}.json` (read by `cat`/python) |
| A3 | BL_s0.A argv = `--arm M2b --seed 0 ... --cache-dir /root/data/cache_pod ... --exclude 540:600 ... --steps 61568`. A BL_s1.A argv is that line with `--seed 1` and `--out .../BL_s1/A` (B_s1 stage A used seed 1, B_s1.B.tmpl `--seed 1 --dagger-seed 1`). | M (BL_s0.A) / E (BL_s1.A by analogy) | same files |
| A4 | Stage-B templates differ between U and B only in the data flags (`--cache-dir --v9-train --v9-train-md5 --lead-train --speed-train --join-train --tracks`, `--exclude` on B, `--fit-join` on U). Seed-1 templates differ from seed-0 in `--seed` and `--dagger-seed` (0 to 1) and `--out`. | M | the four `.B.tmpl.json` files above, compared by eye |
| A5 | Stage-A checkpoints present on the pod and banked on private HF `Sayood/tanitad-v8f-dscale10k-runs`: UE_s0.A sha256 `2fd4b747...`, BL_s0.A `0b374ded...`, UE_s1.A `f13df8ce...`. UM_s0.A / UM_s1.A checkpoints are also on the pod (86,971,413 bytes). All of UE/BL are 86,970,901 bytes. | M | pod `/root/dscale10k/banker_ledger.jsonl` (grep of the three tags), `ls -la runs/*/A/ckpt_last.pt`, `block3.log` lines 11 and 20, `blockreps1.log` line 4 |
| A6 | BL_s1 does not exist: `runs/` has BL_s0 only (`ls /root/dscale10k/runs`: BL_s0, B_rep, B_s0, B_s1, R_s0, UE_s0, UE_s1, UM_s0, UM_s1, U_s0, U_s1, X_s0). No stage-B run exists for UE, BL or UM. | M | pod `ls` |
| A7 | UE_s0 and UE_s1 ran stage A on tree `cdbff971` (UM on `648d4ca3`); BL_s0 ran in block 3 from `block3.sh` (same tree directory). The BL commit field itself was not read. | I / U | `AMENDMENT_UM_A2.md` section 2; BL commit UNVERIFIED |
| A8 | Pod GPU is an RTX 4000 Ada; `nvidia-smi` read 0 % utilisation, name `NVIDIA RTX 4000 Ada Generation` (query-only). Pod root overlay 500 GB, 116 GB free (`df`, which is not authoritative for the volume quota, see the pod `dd` trap). The gate wants 60 GB free on `/root`. | M / I | pod `nvidia-smi --query-gpu`, `df -h /root`; `LAUNCH_X_s0.md` Disk line |

## B. Measured walls on this pod (cost per arm)

Chain-log wall = timestamp of "stage A done" to "stage B done" (bank4 + selfcheck + mint + stage B). Source: pod `/root/dscale10k/block{1,2,3}.log`, `blockreps1.log`.

| id | arm | measured | source line |
|---|---|---|---|
| B1 | stage A, UE_s0 (61,568 steps) | 21:59:30 to 01:53:55 = **3.91 h** | `block3.log:10-11` |
| B2 | stage A, BL_s0 (61,568 steps) | 01:54:00 to 05:28:47 = **3.58 h** | `block3.log:19-20` |
| B3 | stage A, UE_s1 (61,568 steps) | 05:45:42 to 09:03:01 = **3.29 h** | `blockreps1.log:3-4` |
| B4 | stage A, B-set 19,164 steps | B_s0 3,607.5 s class: 23:41:37 to 00:42:05 = 1.01 h | `block1.log:2-3` |
| B5 | union stage B incl. bank, U_s1 | 17:56:47 (A done) to 20:03:11 (B done) = **2.11 h**; bank4 6,603 s (`block2.log:24`), mint 19:47:52, run start 19:49:05, stage B 19:49:05 to 20:03:11 = 846 s (`block2.log:25-27`) | `block2.log:21-27` |
| B6 | union stage B, U_s0 | bank4 6,567 s (`block1.log:20`), mint 11:00:31, stage B 11:01:44 to 11:15:24 = 820 s (`block1.log:21-23`) | `block1.log:20-23` |
| B7 | corpus stage B incl. bank, B_s0 / B_rep / B_s1 | B_s0: 00:42:05 to 01:26:48 = 0.75 h (bank4 2,027 s, `block1.log:6`); B_rep: 12:18:46 to 13:04:22 = 0.76 h (bank4 2,049 s, `block1.log:31`); B_s1: 15:44:53 to 16:52:17 = **1.12 h** (bank4 3,367 s, `block2.log:10`) | `block1.log:3-9,28-34`, `block2.log:7-13` |
| B8 | stage B alone, corpus | B_s0 01:16:45 to 01:26:48 = 603 s; B_rep 627 s; B_s1 617 s | `block1.log:8-9,33-34`, `block2.log:12-13` |
| B9 | eval dumps, eight stage-A arms, one environment, 2 threads, pod CPU | read 13:26Z to 13:40Z = 14 min for eight arms (T1-kin four families, T0 mechanism, STRATEGIC) | `MM_READING_REPS1.md` "Run" |
| B10 | the step time does not depend on clip count (0.188 s per step at 19,164 steps) and bank walls scale with window count | M / I | `LAUNCH_X_s0.md` "Cost" |

Arithmetic (E, from B1-B8):
* BL_s1 stage A: 3.58 h (B2), range 3.29 to 3.91 h (B1-B3).
* Stage B per arm: UE 2.11 h (B5; B6 gives about 2.07 h), BL 1.12 h on the slowest measured bank (B7; 0.75 h on the fastest).
* Deciding set = UE_s0.B 2.11 + UE_s1.B 2.11 + BL_s0.B 1.12 + BL_s1.A 3.58 + BL_s1.B 1.12 = **10.04 pod-h** (upper); with the fast BL bank 0.75 h: 9.30 pod-h.
* UM beside (UM_s0.B, UM_s1.B, union windows): +4.22 pod-h = 14.26 pod-h.

## C. The 60 + 10,000 + eval-141 set arithmetic (requirement 3)

Script run on the pod, CPU, read-only (python reading `manifest.jsonl` files and `bset_clips.txt`). Output verbatim:

```
U 14512 14512 P 4572 4572 E 141 T10 10000 Bset 4512 4512 X 10000 shards 10000 10000
P-U 60 U-P 10000 P&E 0 U&E 0 E-P 141
Bset-U 0 Bset-P 0 U&T10 10000 Bset&T10 0 shards-T10 0 T10-shards 0
PminusU & E 0
```

Reading (M): U = `cache_union_dscale/train/manifest.jsonl` (the clip list UE and UM trained on: both argvs carry `--cache-dir /root/data/cache_union_dscale`, `UE_s0.A.json`, `UM_s0.A.json`). P = `cache_pod/train/manifest.jsonl` (corpus cache, 4,572). E = `cache_pod/eval/manifest.jsonl` (eval-141). T10 = `dscale10k/in/manifest_10k.jsonl`; the 42 shard manifests under `cache_10k/s*/train/manifest.jsonl` hold the same 10,000 ids (shards-T10 0, T10-shards 0).

| id | fact | class | source |
|---|---|---|---|
| C1 | All 10,000 encoded pool clips are in the union training list (U & T10 = 10,000). No encoded pool clip exists that UE or UM did not train on. | M | output above |
| C2 | The B-set list (`dscale10k/mix/bset_clips.txt`, 4,512 ids) is inside U and P; UM's `--tac-src-clips` file. UM trains on the 10k rows too (tactical row weight 0 on them, every other term live). | M / I | output above; `CARD_DSCALE_MIX.md` section 3.2 |
| C3 | Encoded caches present on the pod: `cache_pod` (train 4,572, eval 141), `cache_pod_smoke` (1 + 1 clips), `cache_10k` (10,000, 42 shards), `cache_union_dscale` (14,512), `cache_x_dscale` (10,000, a view). A `find` for other `clips` directories outside `/root/data` returned none. | M | pod `ls`, `find / -xdev ... -name clips` (60 s limit) |
| C4 | Only the pod holds the 10k encoded cache; the dev box does not (195 GB, "Not on the dev box; F: has 31 GB free"). Box 2 was not inspected here. | I / U | `FEASIBILITY_NAVRBD.md` section 1b; box 2 UNVERIFIED |
| C5 | NAVRBD held-out grid `eval_grid_x_feasible.json`: 497 episodes, `n_windows` 5,736, source "NAVRBD step 0: rbdist rule ... stride 4, t = 3 mod 4". Checked against the union list: **497 of 497 clips are in the union list, 497 of 497 are in the 10k, 0 in the B set.** It is therefore held out from N1R / N1D only; for UE and UM it is trained-on data. | M | pod python on `cache_union_dscale/train/manifest.jsonl`, `in/manifest_10k.jsonl`, `mix/bset_clips.txt` with the ids of `step0/in/grid/eval_grid_x_feasible.json` (mirror `raw/2026-11-16-v8f-strat1/navrbd/step0/in/grid/`); output `grid clips 497 in union list 497 in B set 0 in 10k 497` |
| C6 | The 60 clips in P but not in U are the RA-1 holdout (`--exclude 540:600`). They are encoded (in `cache_pod/train`), absent from U by id, absent from eval-141 (PminusU & E = 0), and present in the corpus join: **60 clips, 6,056 join rows (steps), 5,996 labelled steps** in `/root/strat1/inputs/join_train.npz` (md5 prefix `9ebc7a48cbca`). They are corpus-distribution clips. | M | pod numpy read of the join (`clip`, `n`, `has`, `off`); `CARD_DATASCALE_TRAIN_DRAFT.md` line 15 |
| C7 | Eval window count of the 60 on the eval-141 grid rule is not computed (the grid and eval join are not built for them). Rate-based estimate from eval-141: 2,399 windows / 141 episodes = 17.0 per episode, so about 1,020 windows for 60 episodes. | E | arithmetic on `MM_READING_REPS1.md` (141 episodes / 2,399 windows) |
| C8 | Scale pool = 133,929 eligible clips after exclusions (census); 10,000 are encoded; so **123,929 eligible pool clips are not encoded** (E, assumes the 10k lie inside the 133,929). The census removes our train4572, eval-141, the deployed val40 digests, parity_train2400 digests, challenging and OOD-val clips. | M / E | pod `/root/datascale/selection/CENSUS_ELIGIBLE.json` (last step `left` 133,929) |

## D. Encode rate and the cost line for a held-out slice (requirement 3)

| id | fact | class | source |
|---|---|---|---|
| D1 | 10k pipeline: start 2026-11-15 18:58:24, `ZZ DATASCALE-10K DONE ZZ` 2026-11-16 02:47:01 = 7.82 h for 10,000 clips = 1,279 clips/h end to end; `enc_clips_per_h_last_h=1364`. | M | pod `/root/datascale/run10k.log` lines 1 and last |
| D2 | Pilot (200 clips) stage rates: stage 3,856 clips/h; build 1,816 clips/h wall with 13 workers (24.83 s per clip per worker); encode 1,354 clips/h wall (2.61 s per clip, GPU); cache 20.88 MB per clip. | M | pod `/root/datascale/PILOT_REPORT.json` |
| D3 | Geometry labels for new clips exist as a build (GEO10K chains 1 to 5, finished 2026-11-16 11:47 with `ZZ GEO10K DONE ZZ`; v9 release md5 `395221440e3a3c20f17dd381297af4f0`). Per-clip label, join, track and nav-text build time for a new slice was not measured. | M / U | pod `/root/geo10k/chain*.log`; per-clip time UNVERIFIED |
| D4 | T1-kin dump rate on eval-141: 2,399 windows in about 5.5 min on the dev box (0.14 s per window at 2 threads); on the pod 8 arms in 14 min (B9). | I / M | `FEASIBILITY_NAVRBD.md` section 1b; B9 |
| D5 | Encode cost for N = 500 clips (E): 500 / 1,364 = 0.37 h of pod GPU for the encode step, 500 / 1,279 = 0.39 h end-to-end wall for stage + build + encode; label, join and grid build time added on CPU (U). | E | D1, D2 |
| D6 | Eval dump of a 500-clip slice, six arms (BL s0 s1, UE s0 s1, UM s0 s1) at about 17 windows per clip = 8,500 windows per arm: 0.14 s per window gives about 20 min per arm on the dev box; pod rate (B9) is about 1.75 min per 2,399 windows, so about 6 min per arm. CPU only, no GPU. | E | D4, B9 |

## E. Stage-A reads already banked that bear on this card

| id | fact | class | source |
|---|---|---|---|
| E1 | F_cell (TAC-LAT, B pair B_rep - B_s0) 0.0767; F_U = abs(UE_s1 - UE_s0) 0.0342; F2 0.0767. | M | `RD/reps1/read/FLOORS_REPS1.json` |
| E2 | TAC-LAT T1-kin, beside: UE_s0 - BL_s0 -0.0855 [-0.1216, -0.0492] (cost, above F); UE_s1 - BL_s0 -0.0513 [-0.0808, -0.0246] (cost, below F). BL exists at seed 0 only. | M | `RD/reps1/read/MM_READING_REPS1.md` |
| E3 | T0 mean ADE6 (m): BL_s0 2.0629, UE_s0 1.9671, UE_s1 2.0376. UE_s0 - BL_s0 -0.0958 [-0.183, -0.0053] separated (policy-rolled ADE6, T0). | M | `RD/reps1/read/RESULT_REPS1.json` `ade6_stage_A_T0`; `RD/read_b3/result/MM_READING_DSCALE10K_B3.md` section 1 |
| E4 | T0 GT-oracle ADE6 (beside): BL_s0 0.7712, UE_s0 0.8281, UE_s1 0.8327; UE0 - BL +0.0569 [+0.0248, +0.0961] separated; UE1 - BL +0.0615 [+0.0291, +0.0973] separated; UE1 - UE0 +0.0046 [-0.0275, +0.0351] not separated. Direction: UE is worse than BL on this metric. Its exact definition is in `v8f_dscale_read.py` and was not re-read here. | M / U | `RESULT_REPS1.json` `mechanism_stage_A_T0.pairs.{UE0-BL,UE1-BL,UE1-UE0}.beside.oracle_ade6` |
| E5 | T's latcons RMSE (T0): BL_s0 0.6822, UE_s0 0.6864, UE_s1 0.6851, i.e. r = +0.6 % and +0.4 % against BL; stage-A T floor abs_r 0.0308 (n 2,351); the label read is NOT T-DATA-LIMITED on UE0-BL and UE1-BL. | M | same json, `rmse_group.latcons`, `stage_A_T_floor` |
| E6 | The rate of "separated" between two runs of one lever-free pair on the v7-tiny rig was 4 of 42 family cells (9.5 %), about twice nominal (rule `H-ESTIM-SEED-1`, `CLAUDE.md`). Used here as the reason a separated CI is necessary and not sufficient. | I | project `CLAUDE.md` |
| E7 | Stage-A mixed-seed reading issue: BL has only seed 0, so UE_s1 - BL_s0 mixes seed variance into the contrast. | M | `MM_READING_REPS1.md` "Beside" |
| E8 | K-TREE: this device is not bit-reproducible (A vs C same code: 45 differing cells, max rel 2.0e-6). Therefore a same-seed re-run cannot read the floor as exactly zero by design, and a stage-B replicate must use a different stage-A seed. | M | `RD/reps1/ktree/KTREE_RESULT.json` (summarised in `MM_READING_REPS1.md`) |
| E9 | Headline cells (decide the label): L1 op speed MAE, L2 op headway abs error; A1 op yaw-rate MAE, A2 op curvature MAE; T1 TAC-LAT acc, T2 TAC-LON acc; S1 checkpoint position error at 80 m, S2 S_CKPT skill at 80 m. | I | parent card `CARD_DATASCALE_TRAIN_DRAFT.md` lines 156-175 |
| E10 | Fog windows unscored (PI ruling 2026-11-17, `drop-fog` memory); STRATEGIC reported in full (PI 2026-11-16). | I | memory index |

## F. Not established here (UNVERIFIED)
1. The BL_s0.A commit id (A7).
2. Box 2 and the dev-box E:/F: archives hold no encoded clip set outside the pod list: only the dev-box absence of the 10k cache is sourced (C4). A shallow `find` for `meta.json` under `*train*` on F:, E:, D: returned nothing in 90 s; this is not a full search.
3. Per-clip label, join, track and grid build time for a new slice (D3).
4. The exact semantics of `oracle_ade6` (E4).
5. Whether the T1 tool has a T0 mode for the LONGITUDINAL, LATERAL and TACTICAL cells (the banked T0 quantities are ADE6, GT-oracle, latcons, goal F1 and the STRATEGIC cells at first tick).
