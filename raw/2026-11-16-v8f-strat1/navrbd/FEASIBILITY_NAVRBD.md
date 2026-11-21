# FEASIBILITY_NAVRBD (2026-11-18): what the RB-DIST training arm needs, and what the held-out pool holds

Design agent, no training, no GPU, nothing under `E:/Projects/TanitAD/stack`, pod read-only. Evidence classes: MEASURED (M, artifact named), INHERITED (I), ESTIMATED (E), UNVERIFIED (U).
Code base read: `F:/TanitAD-scratch/strat1/navrb2/tree/stack/` (= `4dedac45` + navrb2.patch; the arms N1R / S1R trained from it, commit `e71782a3`).

## 1a. What the NAVRB2 trainer path supports today (M, file:line in `navrb2/tree/stack/`)

| item | fact |
|---|---|
| flags | `--nav-rb off|on`, `--nav-rb-npz`, `--nav-rb-sha256`, `--nav-rb-variant real|zero`, `--nav-rb-lr-mult` (`scripts/v8f_mini_train.py:218-225`; flag classes `:122-124`) |
| channel width | **1**. `e_nav_rb = nn.Linear(1, d, bias=False)`, zero-init, built last under a forked RNG (`tanitad/models/v8f_mini.py:909-913`); added to `e_nav(nav)` in S, T, O (`:918-921`) |
| modes | `NAV_RB_MODES = ("off","on")` (`v8f_mini.py:168`); `NAV_RB_VARIANTS = ("real","zero")` (`tanitad/data/v8f_mini_data.py:372`) |
| table | `build_navtext_nav_rb.py` writes `{split}_nav_rb` [N, 1] in {0, 1}: 1 iff the navtok row at the step has flag 0 and an `RB_*` / `ROTARY*` class (`state()`; rule in its docstring). It reads `cls_id`, `flag`, `age_s`, `vocab` only |
| loader | `nav_rb_arrays` hard-checks shape `(N, 1)` and values in {0, 1} (`v8f_mini_data.py:385-391`); window item reads `J["nav_rb"][rows of idc]` (`:798-799`), collate stacks it (`:849-850`) |
| exit number | **not in the channel.** The table has `{split}_text_i` (index into `texts.json`) and `{split}_dist_m`; `navtok.parse_full` returns `exit_n` and `dist_m` (rbdist card section 2). `build_navtext_nav_rb.py` does not read them |
| countdown | **not in the channel.** The 5-channel text nav gives a countdown (channel 3) only for side classes; a roundabout class has `side == 0`, so its row is `[1,0,0,0,0]` with no distance (`build_navtext_nav.py`, `clip_nav`: `if sd == 0: nav[j] = [1,0,0,0,0]`) |
| dead-reckoned channel | channel 3 is the router distance minus the own arc travelled since the row's time (`clip_nav` for training; the pinned evaluator `v8f_t1_eval.py:215-219` does `row[3] = clip(row[3] - arc[j]/100, 0, 3)` per tick). `nav_rb` is **held from t0** in the study wrapper `v8f_t1_eval_rb.py:77-84` (`self.nav_rb[t0+1:] = self.nav_rb0`) |
| DAgger swap | `swap_rows` rebuilds only channel 3; `nav_rb` is held as logged (`v8f_mini_train.py:3105`, kept-object assert) |
| gates | `bad_navrb` G-DVB (`:1773-1815`), `navrb_probe` G-LIVE (`:1820-1860`, zero-channel control exactly 0.0, step-0 identity, planted ones when the batch carries none), declared-dead rule for N1Z (`:2027-2028`), optimiser group `navrb` (`:3715-3723`), checker kinds `N1R S1R S1dR N1Z` and `rb_gate_problems` (`scripts/strat1_check.py:35-37,147`), report check `rb_report_problems` (`:427`) |

**What an exit + distance variant needs: more than a table build.** A new table alone cannot be read: the model's `Linear(1, d)`, the loader's `(N, 1)` / {0,1} checks and the checker's kind map all assume one binary column. The change is (i) a new table builder, (ii) a width lever on the module, (iii) loader / probe / checker / evaluator-wrapper edits. No change to `stack/tanitad` training math; the new key is `e_nav_rb.weight` with shape `[d, 15]`. Listed file by file in `BUILD_PLAN_NAVRBD.md`.

## 1b. The held-out pool: the 10,000 data-scale clips

**Where each ingredient lives (M, pod read-only `ls` / `df`, and the pulled `join_X.npz`):**

| ingredient | exists | where |
|---|---|---|
| clip list, disjointness | yes | `join_X.npz` clip array, 10,000 clips; overlap with train4572 **0**, with eval-141 **0** (M, set intersection on the pulled join and on `ckpt120build/in/join_train.npz`, `strat1/read/inputs/join_eval141.npz`) |
| TAC-LAT labels | yes | `join_X.npz` key `lat_cls` (1,008,482 steps, `has` 825,841 labelled): geometry labels of the 10k release `v9_labels_datascale10k.npz` (pod `/root/dscale10k/in/labels/`, HF `Sayood/tanitad-datascale-10k` `labels/`). Classes populated: LANE_KEEP, TURN_L, TURN_R only (classes 3, 4 count 0, as on train) |
| route text | yes, raw | `navigation.parquet` (pod `/root/datascale/meta/navigation/`, 67,084,481 bytes; the 10k selection is covered 9,980 / 10,000, `datascale/navprobe/NAVPROBE.md`). Probe: **9,980 of 10,000** clips have rows, 20 have none (M, `out/RB10K_COUNTS.json`); step flags 0 / 1 / 2 / 3 = 996,477 / 6 / 1,999 / 10,000 |
| navtok table (`cls_id`, `flag`, `dist_m`, `text_i`) for the 10k | **not built** | derived here in memory only; a persisted table is a build step (`BUILD_PLAN` file 1) |
| text-nav npz (`navtext_nav.npz` eval-split layout) for the 10k | **not built** | needs the 10k table + `native_tracks_datascale10k.pt` (pod `/root/dscale10k/in/labels/`) |
| cached encoder features | yes, pod only | `/root/data/cache_x_dscale/train/clips/` = 10,000 symlinks into `/root/data/cache_10k/s0xx/train/clips/` (195 GB, fp8 pooled DINOv3; per-clip about 20.9 MB); on HF as 42 shard tars. Not on the dev box; F: has 31 GB free |
| 10k join for the evaluator | yes | pod `/root/dscale10k/merged/join_X.npz` (73,149,031 bytes, sha256 prefix `f383ab78`), pulled to `F:/TanitAD-scratch/strat1/navrbd/in/` |

**RB windows by the rbdist selection rule** (state at the NOW row via the unedited `navtok/build_table.build`, flag 0, class `RB_*` / `ROTARY*`; labelled = `has` and `lat_cls >= 0`; `t >= 3` as RB-JOINT; grid = the eval-141 grid rule, stride 4, `t = 3 mod 4`). MEASURED, script `probe/count_rb10k.py`, output `out/RB10K_COUNTS.json`:

| set | windows | episodes |
|---|---|---|
| RB steps, all | 36,951 | 558 |
| RB, labelled | 30,499 | 548 |
| RB, labelled, t >= 3 | 29,787 | 546 |
| **RB, labelled, eval grid (stride 4, t = 3 mod 4)** | **7,594** | **545** |
| RB TURN windows (TURN_L / TURN_R, labelled, t >= 3) | 12,024 | 344 |

* Exit-number mix on RB labelled t>=3 windows: none 7,259; exit 1 7,113; 2 10,001; 3 4,431; 4 759; 5 81; 6 143. Distance valid on 92.9 %. Label counts LANE_KEEP 17,763, TURN_L 6,810, TURN_R 5,214.
* **Control that the probe is the rbdist rule:** the same script on the train join reproduces rbdist's **24,722 labelled RB windows / 426 episodes** and RB-JOINT's **24,207** (t >= 3) exactly (`out/TRAIN_CONTROL_COUNTS.json`), and the train table flags (456,111 / 36 / 799 for flag 0 / 1 / 2).
* **The RB-JOINT note's "about 930 episodes" (E) is not what the 10k holds: 548.** The 10k clips carry the RB state on 5.5 % of clips against 9.3 % on train4572 (426 of 4,572). Still **34 times** eval-141's 16 episodes and 32 times its 235 windows.
* Exit mix is close to train (train labelled: exit 1 6,768, exit 2 8,805, exit 3 2,721), unlike eval-141 (exit 3 was R 7 of 7). The held-out set is in-distribution for the text channels, which is the point.

**Which tier runs on it, where, at what cost.**
* **T1-kin** (`v8f_t1_eval.py` + the `_rb` wrapper, `--eval-cache` clips dir, `--eval-grid`, `--eval-join`, `--tracks`, `--nav-text-npz`) is the primary tier (EVAL_DOCTRINE). It needs the 10k cache, so it runs **where the cache is: the pod**, CPU (`CUDA_VISIBLE_DEVICES=-1`, 128 cores, 503 GB RAM, `free`), GPU idle at probe time (RTX 4000 Ada, 0 %). Pulling the cache to the dev box would need 545 clips about 11.4 GB (E) only if shard tars could be opened per clip; they cannot (250 clips per tar, about 5 GB), so it is not offered.
* All arms of one comparison must be dumped **in one environment and one thread count** (memory: cross-environment bit-identity bars fail). So the N1R pair (banked checkpoints, pulled from HF `datasets/Sayood/tanitad-v8f-strat1-runs`) and the N1D pair are all dumped on the pod for D1-HO. The eval-141 guard reads dev-box dumps for both arms of its pair, as NAVRB2.
* T0 on the 10k is also possible on the pod and needs the same cache; T1-kin is the tier that carries a capability claim, T0 is a decodability tier.
* **Cost (E):** the dev-box rate is 2,399 windows in about 5.5 min (STRAT1, I) = 0.14 s per window at 2 threads. 7,594 windows = about 18 min per arm at that rate; pod CPU at many threads is faster (U, not measured). Five dumps (N1R_s0, N1R_s1, N1D_s0, N1D_s1, shuffled arm) = about 1.5 h CPU (E). No GPU-hours.
* Disk on the pod: dumps are small (about 6 MB per 141 episodes, I); 134 GB free.

**Pod state read (M):** `nvidia-smi` RTX 4000 Ada, 0 % utilisation, 2 MiB used; NAVRB2's four arms all show `ZZ ARM ... DONE ZZ`; `zerograd_N1R_s0.log` reads `control 'e_nav_rb\.weight' ok (1 trained matches)`, `VERDICT: PASS`.

**Measured arm wall on this pod (M, log mtimes, `/root/strat1/navrb2/runs/`):** N1R_s0 mint 06:56 to run end 07:08 (12 min); S1R_s0 07:08 to 07:24 (16); N1R_s1 07:24 to 07:39 (15); S1R_s1 07:39 to 07:54 (15). So **about 0.25 pod-h per arm** with mint, post and gate included, against the card's 0.16 L40-h (E) for the 3,000 steps alone.

## 1c. Facts that shape the card
1. The first label to read is the **HO floor** `F_HO = |N1R_s1 - N1R_s0|` on D1-HO. It needs only banked arms, so it can be computed before any N1D arm exists, and it fixes the bar in advance.
2. DSCALE10K's U and X arms trained on these 10k clips; they are not pairs for this card. The N1R arms are fine-tunes of M2b (train4572 minus 60 clips) and have never seen the 10k.
3. Left-hand-traffic countries sit in the 10k (25 countries, stratified); the exit side given an exit number depends on driving side, which the route text does not carry. The text channels cannot resolve it; this is a stated risk (card section 9).
