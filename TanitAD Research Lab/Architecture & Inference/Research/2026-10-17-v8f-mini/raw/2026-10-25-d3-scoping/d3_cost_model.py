"""D3 scoping probe 3: the cost table. PURE ARITHMETIC over constants, each tagged with its evidence class and source.
No file is read, nothing is downloaded, no GPU. Every OUTPUT of this script is ESTIMATED (arithmetic on the constants);
every INPUT is tagged. Change a constant and the table follows.
Run:  python -I d3_cost_model.py > d3_cost_model.out
"""
# ---------------------------------------------------------------- constants: (value, class, source)
C = {
    # raw bytes per new clip (MB = 1e6 B)
    "mp4_b1_mb":        (13.057, "MEASURED", "mean of the 4,719 local B1 front-wide mp4s, 61.617 GB (dev box, tanitad-data/physicalai/camera/camera_front_wide_120fov)"),
    "mp4_corpus_mb":    (21.40, "ESTIMATED", "2.05 GB per camera chunk zip (pull_camera.py docstring, INHERITED) / 95.8 clips per chunk outside B1 (MEASURED, clip_index.parquet; 306,152 clips / 3,146 chunks overall) = 21.4; B1's own mean is 13.06, so B1 is NOT size-representative"),
    "ts_mb":            (0.0094, "MEASURED", "4,719 timestamps parquets, mean 9.4 KB"),
    "ego_mb":           (0.415, "MEASURED", "4,800 egomotion_alpamayo parquets, mean 415 KB"),
    "obs_mb":           (0.517, "MEASURED", "obstacle.offline member, corpus mean 517,098 B (2026-09-06-b1-train-join/README.md)"),
    "chunk_fixed_mb":   (0.262, "MEASURED", "per chunk zip: 256 KiB tail + 2 x 1-byte probes = 262,146 B (same README); paid once per zip per chunk (camera, egomotion, obstacle)"),
    "calib_chunk_mb":   (0.04, "MEASURED", "camera_intrinsics.chunk_NNNN.parquet ~37-44 KB (dev box calibration/camera_intrinsics)"),
    # derived artifacts per new clip
    "v2ep_mb":          (36.99, "MEASURED", "22,192,351,528 B / 600 clips (raw/s4c/check_report_train600.json); 256x640 png v2ep"),
    "cache_4x10_mb":    (4.356, "MEASURED", "pool4x10 fp8 + mean fp16 + kinematics, T=101, re-saved from a real cache clip (torch.save) -> 4,355,909 B"),
    "cache_both_mb":    (20.905, "MEASURED", "real cache clip with pool4x10+pool8x20, 20,904,597 B; 141 eval clips mean 20.899 MB (manifest_eval.jsonl)"),
    "v9_mb":            (0.058, "MEASURED", "v9_labels_train.npz 253,667,918 B / 4,369 clips"),
    # seconds per clip
    "dev_build_s":      (8.23, "MEASURED", "v2ep build on the dev box CPU, 6 workers: 600 clips in 82.3 min (raw/s4c/build_train600.log); eval-141 5.62 s/clip"),
    "box2_allin_s":     (16.66, "MEASURED", "box 2, SAFE MODE, build+encode+verify+delete, 3,972 clips: 10-18 12:14 -> 10-19 06:37 = 66,180 s (box2_ship/logs/chain_ra1_full.out)"),
    "box2_encode_s":    (5.01, "MEASURED", "box 2 SAFE MODE encode, 423 clips, 4x10+8x20 (box2_ship/logs/encode_A2s.log); 4.98 on 64 clips; 141-eval manifest enc_s mean 4.20 / median 3.62"),
    "dev_encode_s":     (4.86, "ESTIMATED", "dev-box RTX 4060, 38 clips, MEDIAN under a contended GPU (INCIDENT_gpu_touch.md: 'do not quote as a rate'); banked 4.72"),
    "v9_s":             (0.228, "MEASURED", "v9 build wall_s 997.4 / 4,369 clips (v9_labels_train.manifest.json); eval139: 31.8 s / 139 = 0.229"),
    "s2_emit_s":        (0.0045, "MEASURED", "s2_geom_emit_v7 emit_one 4.5 ms median, 60 clips, 1 process (GOALS_AND_CLAIMS D-TACGT-COVERAGE-IS-A-WRITER-LIMIT)"),
    "obs_pull_s":       (0.25, "MEASURED", "obstacle.offline range pull: 4,440 members, 2.673 GB wire, 17,391 requests, 18.5 min -> 0.25 s/clip, 2.41 MB/s"),
    "obs_join_s":       (0.408, "MEASURED", "b1 train agent join: 30.1 min dev-box CPU / 4,427 clips"),
    "lead_pass_s":      (0.5, "ESTIMATED", "NOT MEASURED anywhere in the export; stated as a guess, order of the join"),
    # rates (MB/s)
    "lan_mb_s":         (14.64, "MEASURED", "dev box -> box 2: 51,480,904,147 B in 3,517 s (ZZSHIP-DONE, box2_ship/logs/ship_trainall.log; transfer_check.json)"),
    "wan_slow_mb_s":    (1.83, "ESTIMATED", "INFERRED from file mtimes of the B1 camera bank: 61.29 GB / 9.31 h (2026-08-29 16:06 -> 08-30 01:24); not a transfer log"),
    "wan_mid_mb_s":     (2.41, "MEASURED", "obstacle.offline range pull, 2.673 GB / 18.5 min (request-bound, small members)"),
    "wan_fast_mb_s":    (5.0, "ESTIMATED", "floor: ~2 GB egomotion tar + timestamps tar + parquet staged in <= 420 s on 2026-10-18 (stage.log: 11:00:52 -> 11:09:54 minus 122 s of local sha256)"),
    # chunks touched by a uniform draw (MEASURED by sampling the pool, probe_pool_gates.out)
    "chunks_4512":      (2408, "MEASURED", "uniform draw of 4,512 from the 301,433-clip pool"),
    "chunks_40000":     (3146, "MEASURED", "uniform draw of 40,000 (every chunk)"),
    # trainer
    "train_steps_s":    (1.217, "MEASURED", "M2b base B on box 2: 1.217 steps/s (RESULT_V8F_GRID.md); 19,164 steps = 4.37 h per seed"),
}
FREE = {  # GB free, MEASURED 2026-10-25 with df and Get-Volume on this box; box 2 is INHERITED
    "dev C: (NTFS, internal)": 64.6, "dev E: (exFAT, artifacts)": 2.1, "dev F: (NTFS, external Transcend)": 787.0,
    "dev G: (Drive mount, abandoned by PI)": 61.4,
    "box 2 D: (last read 2026-10-19 06:37, INHERITED, not re-probed: ssh forbidden)": 85.9,
}
v = {k: x[0] for k, x in C.items()}


def hrs(s):
    return s / 3600.0


print("== constants (value | class | source) ==")
for k, (val, cls, src) in C.items():
    print(f"  {k:16s} {val:>12} | {cls:9s} | {src}")
print("\n== free disk, GB ==")
for k, g in FREE.items():
    print(f"  {k:78s} {g:>7.1f}")

for N, ck in ((4512, "chunks_4512"), (40000, "chunks_40000")):
    print(f"\n==================== +{N:,} new clips (ESTIMATED = arithmetic on the constants above) ====================")
    ch = v[ck]
    for tag, mp4 in (("B1-like clip sizes (13.06 MB mp4)", v["mp4_b1_mb"]), ("corpus-mean clip sizes (21.4 MB mp4)", v["mp4_corpus_mb"])):
        per = mp4 + v["ts_mb"] + v["ego_mb"] + v["obs_mb"]
        fixed = ch * (3 * v["chunk_fixed_mb"] + v["calib_chunk_mb"])                    # camera + ego + obstacle zips, one calibration parquet
        gb = (N * per + fixed) / 1e3
        print(f"-- download, {tag}: {per:.2f} MB/clip + {fixed / 1e3:.2f} GB range overhead over {ch} chunks = {gb:.1f} GB")
        for nm, r in (("slow 1.83", v["wan_slow_mb_s"]), ("mid 2.41", v["wan_mid_mb_s"]), ("fast 5.0", v["wan_fast_mb_s"])):
            print(f"     at {nm} MB/s : {hrs(gb * 1e3 / r):7.1f} h")
        print(f"     (mp4 + timestamps only, for the LAN ship to box 2: {(N * (mp4 + v['ts_mb']) / 1e3):.1f} GB -> {hrs(N * (mp4 + v['ts_mb']) / v['lan_mb_s']):.2f} h at 14.64 MB/s)")
    print("-- disk produced / transient")
    print(f"   cache 4x10 only : +{N * v['cache_4x10_mb'] / 1e3:8.1f} GB      cache 4x10+8x20: +{N * v['cache_both_mb'] / 1e3:8.1f} GB      v9 npz: +{N * v['v9_mb'] / 1e3:.2f} GB")
    print(f"   per 400-clip chunk transient (mp4 + v2ep): {400 * (v['mp4_b1_mb'] + v['v2ep_mb']) / 1e3:.1f}-{400 * (v['mp4_corpus_mb'] + v['v2ep_mb']) / 1e3:.1f} GB   (v2ep is {N * v['v2ep_mb'] / 1e3:,.0f} GB if never deleted)")
    print("-- compute wall, hours")
    b2 = N * v["box2_allin_s"]
    b2_enc = N * v["box2_encode_s"]
    dev_b = N * v["dev_build_s"]
    dev_e = N * v["dev_encode_s"]
    v9 = N * (v["v9_s"] + v["s2_emit_s"])
    lead = N * (v["obs_pull_s"] + v["obs_join_s"] + v["lead_pass_s"])
    print(f"   V1 box 2 does build+encode (the RA-1 FULL recipe)      : box-2 hours {hrs(b2):7.1f}   (dev box: download + stage + ship only)")
    print(f"   V2 dev box does build (6 workers) + encode (4060)      : dev hours  {hrs(dev_b + dev_e):7.1f} serial, {hrs(max(dev_b, dev_e)):7.1f} if CPU and GPU overlap; box-2 hours 0 (ships the 4.36 MB/clip cache: {N * v['cache_4x10_mb'] / v['lan_mb_s'] / 3600:.2f} h)")
    print(f"   V3 dev builds v2ep, box 2 ENCODES only                 : box-2 hours {hrs(b2_enc):7.1f}   dev CPU hours {hrs(dev_b):7.1f}  (ships v2ep {N * v['v2ep_mb'] / 1e3:,.0f} GB at 14.64 MB/s = {hrs(N * v['v2ep_mb'] / v['lan_mb_s']):.1f} h, overlapped)")
    print(f"   labels (v9 + S2 emit, all CPU)                         : {hrs(v9):7.2f} h        lead block (obstacle pull + join + lead-pass, ESTIMATED): {hrs(lead):7.2f} h")
    print(f"   one M2b training run at B's 19,164 steps               : {19164 / v['train_steps_s'] / 3600:.2f} h per seed (fixed steps); {38328 / v['train_steps_s'] / 3600:.2f} h if S doubles with the clips (steps-mult 4 on a 2x manifest)")
