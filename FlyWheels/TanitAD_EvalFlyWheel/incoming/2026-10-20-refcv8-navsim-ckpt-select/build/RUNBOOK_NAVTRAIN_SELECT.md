# RUNBOOK: refcv8 checkpoint selection on the NavSim navtrain sample (D-R8-CKPT-RULE + AMENDMENT 1)

EvalFlyWheel build, 2026-10-21. From "the shards are on Thor" to the decision JSON. Nothing in this file has been executed on navtrain data: **no navtrain data exists on any reachable machine yet** (GROUNDWORK B1). Every tool below was tested on NAVTEST inputs or synthetic fixtures; the one contact with real navtrain data is a header-only inventory of the 4.5 GB `navtrain_current_25.tgz` on the dev box (section 6), which decodes no image, reads no metadata, writes no sample and scores nothing. No PDMS or sub-score has been produced on any navtrain token.

Evidence classes: MEASURED (ours, artifact named) · INHERITED (another record, not re-measured) · ESTIMATED. Every wall time is ESTIMATED unless marked.

**What this selects.** refcv8's deliverable checkpoint, EMA-40k vs EMA-48,852, by the registered rule: arms `R8_RT` at both checkpoints and inference seeds 0 and 1 on one fixed navtrain sample; choose EMA-40k iff at **both** seeds the paired difference (40k minus 48,852, PDMS, log-cluster bootstrap B 2000) is positive with a 95 % interval excluding 0 **and** the smaller seed's margin exceeds 2x the inference floor |d_s0 - d_s1|. Anything else, and every refusal, keeps **EMA-48,852**. Tier: NavSim open-loop PDM scorer (T1-family).

---

## 0. Who must do what first (blockers; a build cannot clear any of them)

| # | needed | owner | status |
|---|---|---|---|
| B1 | **approve the download S1**: metadata tgz + `navtrain_current_1.tgz` + `navtrain_history_1.tgz`, 20,353,459,264 B, public, cc-by-nc-sa-4.0 (or S25, 9,990,506,732 B, at most 1,517 tokens) | PI | open |
| B2/B4 | AMENDMENT 1 in GOALS_AND_CLAIMS (readable = metadata row + 4 history frames x {CAM_L0, CAM_F0, CAM_R0}; seed-1 R8_RT via `derive_rt_seed.py`) | MM | filed 2026-10-21 per the decision script's header; the launch script gate G0 needs its blob sha written to `AMENDMENT_ACK.txt` (S8) |
| B3 | accept `tanitad-train` + the read-only scipy overlay as the inference venv (`tanitad-edge` has no `timm`) | PI/MM | open |
| B6 | one backend: rolls on Thor CUDA, scoring on the Thor CPU backend. Dev-box floors are **mixed-backend** and never enter this decision | rule | binding |

## 1. Stage table

All hosts are Thor unless stated. `$R=/home/nvidia/r8ns`, `$C=$R/code`, `$N=$R/navtrain`, `$OUT=$R/navtrain_sel`, `$X=$C/navtrain` (this build's code).

| stage | what | interpreter | success marker (read the artifact, never an exit code) | ESTIMATED wall |
|---|---|---|---|---|
| S0 | ship + pin the code | dev box -> Thor | `sha256sum -c CODE.sha256` all OK | 1 min |
| S1 | download S1 | `curl`/`hf` on Thor | each file's sha256 = its LFS oid | 11 min (INHERITED 31.2 MB/s) |
| S2 | extract the trainval metadata | `tar` | `>= 1,192` navtrain logs present | 3-6 min |
| S3 | **fix the sample** | `python3` | `ZZSAMPLE status=FIXED count=<n >= 1000>` | 6-10 min (header scan of 13.3 GB of tgz; MEASURED 35 MB/s compressed on the dev box, Thor unmeasured) |
| S4 | export the AgentInput records | navsim-cpu venv | `ZZEXPORT_DONE tokens=<n> missing=0` | 1-3 min (MEASURED 334 tokens in 14 s incl. import on the dev box) |
| S5 | build the 416x1024 frame bank | tanitad-train venv | `ZZBANK_DONE tokens=<n> frames=<k>` | 20-40 min |
| S6 | stage maps + trainval pickles on tmpfs; preflight | navsim-cpu venv | `ZZPREFLIGHT_OK problems=0` | 2 min (maps 1.3 GB at 34 MB/s MEASURED) |
| S7 | metric cache | driver `python3`, devkit in navsim-cpu | `ZZCACHE_DONE rows=<n> expected=<n> problems=0` | 15-40 min (MEASURED on the dev box: 20 tokens / 4 logs in 18.2 s single run, 35.1 s as 4 chunks, each run paying its own map load; W3's first smoke20 took 101.6 s under contention) |
| S8 | the rolls (GPU) | tanitad-train | `ZZROLLSDONE` | 27-31 min + ~1 min controls |
| S9 | score 7 arms (CPU pool) | driver `python3`, devkit in navsim-cpu | seven `ZZSCORE_DONE` | ~6 min at the pool rate (INHERITED 44 scenes/s), up to ~25 min |
| S10 | decide | tanitad-train (numpy) | `ZZSELECT verdict=<...> choice=<...>` | seconds |

**Order is forced**: S8's gate G3 refuses while any `python` of user `nvidia` is alive, so S7 (cache, CPU) must have *finished* before S8, and S9 runs after S8. S4, S5 and S7 may overlap only if `MemAvailable >= 35 GB` and no GPU job is live; the default is serial. Sum S1..S10 = about 1.5-2.5 h (GROUNDWORK section 4).

## 2. Variables used below (Thor shell)

```
R=/home/nvidia/r8ns; C=$R/code; N=$R/navtrain; OUT=$R/navtrain_sel; X=$C/navtrain
NAVSIM_PY=/home/nvidia/venvs/navsim-cpu/bin/python           # py3.9, navsim v1.1 (3e8291b), nuplan; the Thor scorer's venv
TRAIN_PY=/home/nvidia/venvs/tanitad-train/bin/python         # torch 2.13+cu130, timm 1.0.29 (blocker B3)
V11=/home/nvidia/navsim-thor/src/navsim-v11
W3=/home/nvidia/navsim-thor/code/w3                          # byte-identical copy of W3's code (run_v1.py, w3_agents_v1.py)
YAML=$V11/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml     # sha256 c37fea567a0cfdbc29076cca893d4f5dd32db59baec18ae214527206d6b64e6f (identical on Thor, MEASURED)
SAMPLE=$N/NAVTRAIN_SAMPLE_TOKENS.json
INP=$N/inputs/navtrain_sample_inputs.json.gz; BANK=$N/frame_bank; LOGS=$N/navsim_logs/trainval
TMPFS=/dev/shm/navsim                                        # <= 7 GB at any time (backend DESIGN); this selection needs about 4-5 GB
```

## 3. S0 - ship and pin (dev box -> Thor)

The new code is `build/code/*.py` of this package (bare-name hashes in `build/CODE.sha256`, the whole package in `build/MANIFEST.sha256`): `navtrain_common.py`, `fix_navtrain_sample.py`, `export_navtrain_inputs.py`, `build_navtrain_frames.py`, `navtrain_cache_score.py`, `make_score_jobs.py`, `select_ckpt_r8.py`. Also ship `raw_hf_tree_navsim.json` (the shards' LFS oids) and the groundwork's `r8ns_navtrain_select_LAUNCH.sh`, `derive_rt_seed.py`, `rt_assert.py`, `r8ns_bridge_wrap.py`, `r8ns_cuda_controls.py` if they are not already under `$C/thor/` (GROUNDWORK section 7 says the LAUNCH script is NOT on Thor).

```
ssh tanitad-thor "mkdir -p $X $C/thor $N/inputs $OUT/pool_logs"
cd /c/Users/Admin/refcv8_pf/navsim_ckpt && scp build/code/*.py build/CODE.sha256 raw_hf_tree_navsim.json tanitad-thor:$X/
scp navsim_ckpt/code/r8ns_navtrain_select_LAUNCH.sh tanitad-thor:$C/thor/
ssh tanitad-thor "cd $X && sha256sum -c CODE.sha256 | grep -vc ': OK$'"      # must print 0 (and CODE.sha256 must list 7 files: wc -l)
```

Pins that must hold on Thor before anything runs (each is a literal in the code that checks it):

| pin | sha256 / md5 |
|---|---|
| `taniteval/ci.py` of the launch tree (`$C/tree/taniteval/taniteval/ci.py`) | `38db571e368f02dc4c0d9df7b47333ebbb082e43695617ba0dfba881668cd73e` |
| `derive_rt_seed.py` (accepted file) | `bebdc5c1ab41b775a60c80d4d79510bc7af6f4b4d12c374526985408e260e278` |
| `$C/retime/retime_stage.py` (the RETIME stage) | `b4bc7b79aa9a0ac02a170b4e633da0de4cd6026f76923c8d2706a0b2244f08ee` |
| deploy table `refcv8_retime_deploy_gains_pf.json` | `3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7` |
| frozen table `FROZEN_GAINS_RETIME_train_pf.json` | `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80` |
| `ckpt_ema_40000.pt` / `ckpt_ema_48852.pt` (md5) | `57807456bd7169800be159ff95871d22` / `b4364029b68f29a33eeb4aebbbbd002b` |
| navtrain scene filter yaml | `c37fea567a0cfdbc29076cca893d4f5dd32db59baec18ae214527206d6b64e6f` |

```
sha256sum $C/retime/retime_stage.py | cut -c1-64     # == b4bc7b79...  (the launch script only checks the file EXISTS;
                                                     #  boot8.load_retime_stage falls back to OTHER stage paths when the given one is
                                                     #  missing -- MEASURED while building the acceptance test)
```

## 4. S1 - download (blocked on PI approval B1)

Revision `a76f840b65e972bc45e56c2adced897498e9a026` of `OpenDriveLab/OpenScene`; URLs, bytes and LFS oids are in `NAVTRAIN_SAMPLE_TOKENS.json` `acquisition_options_for_PI`:

| file | bytes | sha256 (LFS oid) |
|---|---|---|
| `openscene-v1.1/openscene_metadata_trainval.tgz` | 7,052,210,642 | `193138ac68b29d977d4438a95dfda35a07ff4ad3efb1800b3ee47d6e1dcb3764` |
| `navsim/navtrain_current_1.tgz` | 9,469,285,958 | `b5513fdc8138ed97e7346a2c137a66c09042ac78fde4865c67104c0bb99e0576` |
| `navsim/navtrain_history_1.tgz` | 3,831,962,664 | `49b4fe766224dd8079511ebbc362e9b74ece8cbbfd57980f38a0316447ecc06b` |

Destination `$N/dl/` (251 GB free on Thor). If the HF xet path fails on Thor wifi use `HF_HUB_DISABLE_XET=1` (memory note: LFS fallback worked). Verify each file's sha256 against the table; `fix_navtrain_sample.py --verify-sha256-from` re-checks the two shards (S3). No other file is needed; the selection reads no lidar and no other camera.

## 5. S2 - extract the trainval metadata

```
mkdir -p $N/_meta_x $N/navsim_logs && tar -xzf $N/dl/openscene_metadata_trainval.tgz -C $N/_meta_x
PKL=$(dirname "$(find $N/_meta_x -name '*.pkl' | head -1)"); echo "ZZPKLDIR $PKL"
mv "$PKL" $N/navsim_logs/trainval && ls $N/navsim_logs/trainval | wc -l
```

**UNVERIFIED layout**: the dev box's *test* metadata extracts to `openscene-v1.1/meta_datas/test/<log>.pkl` (MEASURED, `D:/Archive/devbox-C/navsim/data/openscene/openscene-v1.1/meta_datas`), so trainval is expected at `openscene-v1.1/meta_datas/trainval/`; the `find` above does not assume it. The directory must hold one `<log_name>.pkl` per trainval log (the navtrain filter names 1,192 of them; extra logs are harmless). The fixer reports any navtrain log without a pickle as `NO_METADATA`.

## 6. S3 - fix the sample (BEFORE any score exists, never revisited)

```
python3 $X/fix_navtrain_sample.py --filter-yaml $YAML --metadata-dir $N/navsim_logs/trainval \
    --shard $N/dl/navtrain_current_1.tgz --shard $N/dl/navtrain_history_1.tgz \
    --verify-sha256-from $X/raw_hf_tree_navsim.json --out $SAMPLE | tee $N/fix_sample.out
grep -o 'ZZSAMPLE status=[A-Z]* count=[0-9]* logs=[0-9]* sha256=[0-9a-f]*' $N/fix_sample.out
```

Rule applied (registered, with AMENDMENT 1): logs sorted by `log_name`; a token is readable iff its metadata row exists, the devkit keeps its window (4 history + 10 future, route present), and its 4 history frames x {CAM_L0, CAM_F0, CAM_R0} are in the shards; take all readable tokens of the first logs, stop at the first log that takes the total to >= 2,000.

| marker | meaning | next |
|---|---|---|
| `status=FIXED count>=1000` | the sample | go to S4 |
| `status=UNDERPOWERED` | fewer than 1,000 readable tokens | **stop**: the rule's own clause keeps EMA-48,852; report, do not score |
| exit 2 | a FIXED/UNDERPOWERED sample file already exists with different content | the sample is never rewritten; investigate, do not delete |

Real-archive check (MEASURED, `raw/real_shard_inventory_navtrain_current_25.json`, headers only, no image decoded, no metadata read, no sample written): `fix_navtrain_sample.inventory` over the real 4,496,159,137 B `navtrain_current_25.tgz` on the dev box took **128.4 s = 35 MB/s compressed** (a tgz must be gunzipped to walk its headers; one core) and returned **4,551** L0/F0/R0 jpgs in 38 logs, **set-equal** to the same set derived independently (different mechanism, different regex) from the earlier `tar -t` listing `members.txt`. So `member_rel` parses the real layout `navtrain_current_N/<log>/<CAM>/<hash>.jpg`, and S3 plus S5 each cost at least one gunzip pass over 13.3 GB (about 6 min each at that rate; Thor's aarch64 rate is unmeasured).

Yield is unknown until the files are read: shard 1 opened with 16 logs / 1,309 tokens in the pilot (MEASURED, F0-only), the average shard holds 3,228 (ESTIMATED), so FIXED at 1,000-2,000 is possible and still admissible.

## 7. S4 - export the per-token AgentInput records

```
cd $X && PYTHONPATH=$V11 $NAVSIM_PY export_navtrain_inputs.py --sample $SAMPLE --logs-root $N/navsim_logs/trainval \
    --navsim-tree $V11 --w3-code $W3 --filter-yaml $YAML --out $INP --manifest $N/inputs/manifest.json | tail -3
```

Records are W3's schema, produced by the devkit's own `SceneLoader.get_agent_input_from_token` / `Scene.get_future_trajectory` / `ConstantVelocityAgent`. Control (MEASURED, `tests/test_export_navtrain_inputs.py`): the same script on the 4 navtest logs of W3's smoke export reproduces all **334 / 334** banked records exactly. It stubs `Scene._build_map_api` (the export reads no map; Thor has Las Vegas only); the stub changes no record (that control is the proof). `ZZEXPORT_REFUSED` means a sample token the devkit does not produce (or one it produces that the sample does not name); **no file is written** in that case. `--navsim-tree` must be exactly the imported devkit (`navsim_is_from`; a parent directory of the v1.0 devkit used to pass a prefix test).

## 8. S5 - the 416x1024 frame bank (KB3 + hard refusal on any missing frame, B7)

```
cd $X && TANITAD_REPO=$C/tree PYTHONPATH=$C/tree/stack:$C/tree/taniteval OMP_NUM_THREADS=6 \
  $TRAIN_PY build_navtrain_frames.py --sample $SAMPLE --inputs $INP --shard $N/dl/navtrain_current_1.tgz --shard $N/dl/navtrain_history_1.tgz \
    --logs-root $N/navsim_logs/trainval --bank $BANK --threads 6 \
    --frames416 $C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/frames416.py
```

| marker | meaning | next |
|---|---|---|
| `ZZBANK_DONE tokens=<n> frames=<k>` | bank written, re-read through the `BankNavtest416` layout, every token's sha256 re-derived; `DONE.json` last | S6 |
| `ZZBANK_REFUSED_missing_frames` | some token lacks one of its 12 jpgs; `REFUSED.json` names every affected token | the sample was fixed on a different shard set than the one given here; **never** drop tokens |
| `ZZBANK_REFUSED_<other>` | sample not FIXED / sha does not re-derive / inputs do not cover the sample / frame failed to stitch / all-black frame (KB2) / duplicate member with different bytes | read `REFUSED.json` |

Scratch: extracted jpgs only for the tokens needed (about 1-2 GB), deleted at the end (`--keep-scratch` to keep). The bank is `frames_s00.npy` (about 1.3 MB per unique frame) + `index.json`.

## 9. S6 - maps and pickles on tmpfs, preflight

**Maps: ALL FOUR cities are required, whatever the sample's cities are.** MEASURED from source: the nuPlan `GPKGMapsDB` constructor loads the `lane_connectors` layer of every one of `MAP_LOCATIONS` (`nuplan-devkit/nuplan/database/maps_db/gpkg_mapsdb.py:97-101`), so Thor's `navsim_maps/nuplan-maps-v1.0` (Las Vegas only) cannot build a map API for the cache or the scorer. Sizes: sg-one-north 317,575,168 B, us-ma-boston 254,087,168, us-nv-las-vegas-strip 523,767,808, us-pa-pittsburgh-hazelwood 186,761,216, plus `nuplan-maps-v1.0.json`. The v1.1 devkit reads them as `nuplan-maps-v1.0`.

```
# (1) DEV BOX (Git Bash), relative paths: MSYS tar reads "C:/x" as a remote host
cd /d/Archive/devbox-C/navsim/data && tar -cf - maps | ssh tanitad-thor-wifi "mkdir -p /dev/shm/navsim/data && tar -xf - -C /dev/shm/navsim/data"

# (2) THOR shell: the trainval pickles of the SAMPLE's logs only (a few hundred MB), where OPENSCENE_DATA_ROOT points
mkdir -p $TMPFS/data/openscene/navsim_logs/trainval
python3 -c "
import json, shutil
s = json.load(open('$SAMPLE'))
for ln in sorted(set(s['token_log'].values())):
    shutil.copy('$N/navsim_logs/trainval/%s.pkl' % ln, '$TMPFS/data/openscene/navsim_logs/trainval/')
"

# (3) THOR shell
python3 $X/navtrain_cache_score.py preflight --profile thor --sample $SAMPLE
```

Needs, precisely (`navtrain_cache_score.py` docstring; each pinned by `tests/test_navtrain_cache_score.py`):

* `NUPLAN_MAPS_ROOT=$TMPFS/data/maps` with `nuplan-maps-v1.0.json` and `<city>/<version>/map.gpkg` x 4;
* `OPENSCENE_DATA_ROOT=$TMPFS/data/openscene` with `navsim_logs/trainval/<log>.pkl` for every sample log (caching reads it with `sensor_blobs_path=None`; scoring with `sensor_blobs_path=<root>/sensor_blobs/trainval`, a directory the seam agent never opens - it need not exist);
* `NAVSIM_EXP_ROOT=$TMPFS/exp/r8sel_navtrain` (its own scratch, never W3's `w3_navtest_v1`);
* no jpg, lidar or sensor blob is read by the cache or the scorer.

`ZZPREFLIGHT_REFUSED` lists `MAP_GPKG_MISSING:<city>`, `MAPS_JSON_LACKS:<city>` or `LOG_PICKLE_MISSING:<log>`; fix and rerun.

## 10. S7 - the navtrain metric cache (CPU)

```
nice -n 19 ionice -c3 setsid python3 $X/navtrain_cache_score.py cache --profile thor --sample $SAMPLE --out $OUT --label r8sel_cache \
    > $OUT/cache.out 2>&1 < /dev/null &
# read $OUT/cache.out for ZZCACHE_DONE / ZZCACHE_FAILED / ZZCACHE_REFUSED_preflight ; the evidence is $OUT/cache/r8sel_cache.verify.json
```

The driver loads W3's `run_v1.py` by path and calls its `cmd_cache` UNCHANGED (count guard C9), re-pointing only the split (`train_test_split=navtrain`, the sample's log_names and tokens) and the roots. Then it re-verifies the cache **independently**: exactly one metadata CSV, its rows name exactly the sample's tokens (set equality, not counts), every row's file exists, the pickles on disk are the same set.

**Honest status of the Thor-native build.** Thor has only ever *scored* (never *built* a metric cache): `rasterio` there is an import stub that raises on use (backend DESIGN item 4). Whether `metric_caching` touches a raster layer is **untested**. Fallback, validated pattern (W3 built the navtest cache on the dev box and the backend loads Windows-built caches through `thor_compat`):

1. on the dev box with `--profile devbox --chunk-tokens 800 --data <dir holding navsim_logs/trainval/<sample logs>.pkl>` (the dev box holds only `navsim_logs/test` today): the line limit of Windows is 32,767 characters and the sample as a Hydra override is about 19 characters per token, so a 2,000-token line is refused by `check_cmdline` before launch; the driver builds whole-log chunks and re-emits the metadata CSV with the devkit's own `save_cache_metadata` (MEASURED, `tests/test_navtrain_cache_score.py::test_the_windows_chunked_build...`: chunked build of the 20 navtest smoke tokens = W3's single-run cache, **20 / 20 pickles identical in every field except `file_path`**);
2. ship `exp/r8sel_navtrain/metric_cache_navtrain_sample` (about 1.3 MB per token, ESTIMATED 2.6 GB) to `$TMPFS/exp/r8sel_navtrain/` as `tar | ssh`, then rerun the `preflight`/`score` steps. The sample's trainval pickles and the four maps are needed on the dev box too (the dev box holds only `navsim_logs/test` today).

## 11. S8 - the rolls (GPU) - the groundwork launch script, unchanged

```
echo "<blob sha of Project Steering/GOALS_AND_CLAIMS.md at the commit that carries AMENDMENT 1>" > $OUT/AMENDMENT_ACK.txt     # MM: git rev-parse FETCH_HEAD:"Project Steering/GOALS_AND_CLAIMS.md"
ssh tanitad-thor "cd $R && nohup flock -n /home/nvidia/refcv7_post/thor_gpu.lock bash $C/thor/r8ns_navtrain_select_LAUNCH.sh > $OUT/select.out 2>&1 < /dev/null &"
```

Read `$OUT/select.out` for `ZZROLLSDONE` or one `ZZREFUSE_<why>`:

| marker | gate | meaning |
|---|---|---|
| `ZZREFUSE_no_amendment_ack` | G0 | `AMENDMENT_ACK.txt` absent |
| `ZZREFUSE_sample` | G1 | sample not FIXED, < 1,000, or its sha256 does not re-derive |
| `ZZREFUSE_missing_<file>` / `no_logs` / `md5_<step>` | G2 | an input, the bank index, the RETIME table/stage or a checkpoint md5 is wrong |
| `ZZREFUSE_gpu_busy` / `python_busy` | G3 | the box is not idle (S7 still running?) |
| `ZZREFUSE_K0_<step>` / `token_<step>` | | CUDA determinism control failed / no GPU token |
| `ZZREFUSE_post_gate` | G4 | any seam partial, any CV stand-in, KPR not PASS, or not exactly 4 `R8_RT` seams |
| `ZZREFUSE_rt_table` | | `RT_ASSERT.json` verdict is not PASS |
| `ZZROLLSDONE` | | four `R8_RT` seams (2 checkpoints x 2 seeds), the PRIOR_ha0p seam, all with zero stand-ins |

Outputs: `$OUT/ema40000/bridge/` and `$OUT/ema48852/bridge/`, each with `seam_R8_RT.npz`, `seam_R8_RT_s1.npz`, manifests; `seam_PRIOR_ha0p.npz` is read from the 48,852 directory. The first time this runs on navtrain the script is itself untested on navtrain data (it was written 2026-10-20, never executed; the navtest smoke exercised the same bridge, MEASURED GROUNDWORK section 3).

## 12. S9 - score the seven arms on the Thor CPU backend

Seven arms: `R8_RT` and `R8_RT_s1` at each checkpoint (4), `PRIOR_ha0p`, official `STOP`, official `CV`. Labels and paths come from `select_ckpt_r8` so the decision script finds them.

```
python3 $X/make_score_jobs.py --layout-root $OUT --sample $SAMPLE --driver $X/navtrain_cache_score.py --out $OUT/jobs.jsonl      # ZZJOBS_DONE n=7
cd $OUT && nohup setsid python3 /home/nvidia/navsim-thor/code/thor/thor_pool.py --jobs $OUT/jobs.jsonl --state $OUT/pool_state.json > $OUT/pool.out 2>&1 < /dev/null &
```

Pool policy (backend code, INHERITED): at most 4 scorers, launch only if `MemAvailable >= 35 GB`, SIGTERM live jobs below 20 GB or when `/home/nvidia/NAVSIM_YIELD` exists. A killed job is not relaunched by default; its count guards refuse the partial output. Pool progress prints `ZZ<done>-<fail>-<alive>-<queued>-<availGB>ZZ`.

Each job writes `$OUT/scores/<label>/<label>.{csv,counts.json,binding.json}`. The **binding** is written only after the scorer's own counts say PASS and an independent re-read of the CSV finds exactly the sample's token set; it carries the sample sha256, the seam sha256, the CSV sha256 and `split: navtrain`, which `select_ckpt_r8.load_arm` requires. Markers per job (in `$OUT/pool_logs/<label>.log`): `ZZSCORE_DONE` · `ZZSCORE_SKIPPED_banked` (same input already bound) · `ZZSCORE_REFUSED_label_bound_to_a_different_input` (exit 3; a label is never rescored over another arm) · `ZZSCORE_REFUSED_preconditions` (preflight or cache verification failed; names the problem) · `ZZSCORE_FAILED` (counts not PASS, or the CSV's token set differs; `<label>.FAILED.json`). Forcing a rescore needs `R8CS_FORCE_RESCORE=1` and a reason in the report.

MEASURED control (dev box, `tests/test_navtrain_cache_score.py::test_end_to_end_...`): cache build + official STOP scoring + binding on the 20 NAVTEST smoke tokens of W3 gave per-token scores **bit-for-bit equal to W3's banked `STOP_smoke20.csv`** (8 columns x 20 tokens); a re-run was `SKIPPED_banked`, and the same label with `--official CV` was refused. The navtest control binding says `split: navtest, control: true`, which the decision script rejects (`BINDING_SPLIT_navtest`, tested).

## 13. S10 - the decision

```
PYTHONPATH= $TRAIN_PY $X/select_ckpt_r8.py --layout-root $OUT --sample $SAMPLE --out $OUT/DECISION.json --md $OUT/DECISION.md \
    --ci-path $C/tree/taniteval/taniteval/ci.py \
    --expect-ci-sha256 38db571e368f02dc4c0d9df7b47333ebbb082e43695617ba0dfba881668cd73e
```

Marker `ZZSELECT verdict=<CHOOSE_EMA_40000|DEFAULT_EMA_48852|INCOMPLETE_DEFAULT_EMA_48852> choice=<...> reasons=<n>`.

* `INCOMPLETE_...`: **numbers are withheld** and `reasons` lists every refusal (`BINDING_*`, `SEAM_*`, `ARM_*`, token-set or stand-in problems, < 1,000 tokens, controls that failed their known value). The registered default EMA-48,852 holds; nothing is "rescued" by editing a bar.
* The two readings the registered text leaves open are fixed in the script header before any score exists: the margin is `min(d_s0, d_s1)` (the mean is printed beside), and a bootstrap needs >= 2 clusters.
* Controls inside the run: 48,852 - 48,852 reads exactly 0 [0, 0]; a token-shuffled pairing widens the interval; STOP / CV / PRIOR rows and the sub-scores print **beside** the verdict and never decide.

## 14. After the decision

1. Bank into the repo (stage, do not push): `NAVTRAIN_SAMPLE_TOKENS.json`, `fix_sample.out`, the inputs manifest, the bank's `BUILD_REPORT.json`/`DONE.json`, `cache/*.verify.json`, every `scores/*/*.counts.json` + `binding.json` + CSV, `DECISION.json`/`.md`/`.spec.json`, `RT_ASSERT.json`, `select.out`. A result that lives only on Thor is not done.
2. Update `GOALS_AND_CLAIMS.md` `D-R8-CKPT-RULE` in the same turn with the verdict, the evidence class (MEASURED, tier T1-family, backend Thor), and the artifact paths.
3. The single navtest read (A-R8-NS1) uses the **chosen** checkpoint only. The non-chosen checkpoint's plans are never read or scored on navtest (rule item 5).
4. Never reopen the sample, never rescore an arm to move the verdict, never compare against a dev-box floor without saying MIXED-backend.

## 15. What is tested and what is not

Totals (banked in `raw/`): **194 tests pass** (all in `raw/pytest_main.log`, 3 min 4 s), **76 meta-tests pass** (`raw/pytest_mutation.log`, `raw/pytest_mutation_chain.log`) proving **63 deliberate regressions go RED**, each with its named literal tests among the failures (`raw/mutation_record.json`), plus 5 more mutation tests inside the `derive_rt_seed` acceptance file. Every expectation is a literal, never an expression over the code under test.

| piece | tested here | not tested |
|---|---|---|
| `select_ckpt_r8.py` | 55 tests, 17 mutations; synthetic worlds with known deltas, file-level refusals | a real navtrain score |
| `derive_rt_seed.py` | 11 tests (5 mutation tests): bit-identical R8_RT / R8_RT_s1 reproduction against the Thor smoke rows, both checkpoints | navtrain rows |
| `fix_navtrain_sample.py` | 27 tests, 9 mutations; the rule on fixtures; window control on 334 real W3 navtest tokens; header inventory of the REAL 4.5 GB `navtrain_current_25.tgz` (4,551 jpgs = the independent `tar -t` listing) | real metadata + history shards; Thor scan speed |
| `export_navtrain_inputs.py` | 23 tests, 9 mutations; on the real devkit **334 / 334** records equal W3's banked export; exact-tree check (a parent dir of the v1.0 devkit used to pass) | Thor's navsim-cpu venv; trainval pickles |
| `build_navtrain_frames.py` | 24 tests, 6 mutations; real E2 stitch with real navtest rigs and synthetic 1920x1080 jpgs; read back through the real `BankNavtest416` | real navtrain jpgs; Thor timing |
| `navtrain_cache_score.py` | 45 tests, 14 mutations; the devkit's own Hydra configs composed with the driver's overrides; dev-box cache + official STOP score + binding = W3's banked `STOP_smoke20.csv` cell for cell; chunked build = single-run build (20 / 20 pickles, every field but `file_path`) | **Thor-native metric cache build (rasterio stub)**; any navtrain metadata |
| `make_score_jobs.py` | 5 tests, 4 mutations; labels and paths cross-checked against `spec_from_layout` | the Thor pool run |
| the chain | 4 tests, 4 mutations: the fixer's own output is accepted by every consumer; the real record builder's export feeds the real bank builder; tokens the fixer excludes for a missing jpg = tokens the builder refuses (B7) | - |
| the LAUNCH script | its navtest-smoke ancestor ran (GROUNDWORK section 3) | **never executed as written**; gates G0-G4 untested on navtrain |
| `openscene_metadata_trainval.tgz` layout | no | inferred from the test-split tarball only (section 5) |
| pre-existing warnings | two, from E2's `build_frames.py` (non-writable numpy array; NaN cast to uint8 in a synthetic-pixel stitch) | not this build's code |

## 16. Refusals worth knowing by name

* `ZZSAMPLE status=UNDERPOWERED`: fewer than 1,000 readable tokens; stop, EMA-48,852 holds.
* `exit 2` from the fixer: a FIXED/UNDERPOWERED sample already exists with different content.
* `ZZEXPORT_REFUSED`: sample token missing from (or extra in) the devkit's scene list; no file written.
* `ZZBANK_REFUSED_missing_frames`: B7; no stand-in is ever substituted.
* `ZZPREFLIGHT_REFUSED`: a map city, a log pickle or a path is missing.
* `ZZCACHE_FAILED` with `ROW_TOKENS_DIFFER` / `PICKLE_TOKENS_DIFFER`: the cache is not the sample.
* `ZZSCORE_REFUSED_label_bound_to_a_different_input`, `ZZSCORE_FAILED`: see section 12.
* `ZZREFUSE_*`: the launch script's gates (section 11).
* `ZZSELECT verdict=INCOMPLETE_DEFAULT_EMA_48852`: section 13.
