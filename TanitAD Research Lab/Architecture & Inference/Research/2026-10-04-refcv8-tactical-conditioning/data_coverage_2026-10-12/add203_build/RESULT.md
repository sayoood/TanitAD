# refcv8 train set 4,369 -> 4,572 clips: what was built, what still blocks (2026-10-12, 0 GPU)

Layer key: VIEW = train symlink view, CACHE = B1 416x1024 v2 cache, LAB = v8 labels, JOIN = agent joins. Evidence MEASURED unless tagged; artifact paths in `BUILD_OUTPUTS.json` and `raw/`.

## 1. Built (all NEW paths; every pre-existing Thor file byte-identical: `raw/thor_BEFORE_md5.txt` == AFTER)
| artifact | Thor path | md5 | counts |
|---|---|---|---|
| train view | `/home/nvidia/data/refcv8-b1-416x1024-train-4572/` | manifest `77dafb50af0aedd1b940cf12c12518ef` | 4,572 symlinks = 4,369 old (identical targets) + 203; n eval139 = 0 |
| combined extrinsics | `/home/nvidia/data/refcv8_train4572_eval139_extrinsics.json` | `dba6da20b874670aef32f2451b312837` | 4,711 keys (4,508 old untouched + 203) |
| pose-sync sidecar | `/home/nvidia/data/refcv8_pose_sync_sidecar.jsonl` | `894fe1d9e5420dc7f8eadcbe20c791e5` | 4,711 rows, 0 skipped, K4 4,653 checked worst 6.2e-9 s |
| clip-clock sidecar | `/home/nvidia/data/refcv8_clip_clock_sidecar_ts33.jsonl` | `c81c2739b5bef58e3c7e389245bd506a` | 4,686 rows = 4,653 position-inversion + 33 camera-timestamp |
The view is built like the old one (absolute symlinks into `physicalai-b1-w120-416x1024cyl`); the manifest is the old manifest's entries carried bit-for-bit plus `v2_dataset._scan_meta` of the 203 (the function `load_or_build_manifest` calls) -- 1.5 s wall, no frame read, instead of re-scanning 4,572 files.

## 2. Controls (all passed; the build refuses otherwise)
- View: C1 three random old clips re-scanned by `_scan_meta` == old manifest entries (torch.equal); C2 manifest `files` == the trainer's `_list_clips(new view)`; C3 old entries are the old objects; C4 `sum(T_out) - 28*n` reproduces 746,946 with an integer 28. `raw/view_report.json`.
- Extrinsics: 3 known clips re-run == stored; the old 4,508-key file re-serialises byte-identically, so the old entries are untouched.
- Clock/pose-sync: the new builds reproduce every old row exactly (4,483/4,483 clock, 4,508/4,508 pose-sync). `raw/controls_clock_posesync.txt`.
- Camera-timestamp clock route (for stationary clips): validated on ALL 4,653 moving clips, max |d grid_start_s| 4.27e-07 s, max |d dt_s| 6.2e-09 s.

## 3. Guards on the 4,572 set
| guard | value | limit | verdict |
|---|---|---|---|
| (a) clip-clock unverified clips (`assert_label_clock_true`, per CLIP, refc_v3_train.py:4258-4281) | plain rebuild: **55/4,572 = 1.203 %** (33 of the 203 are stationary: "fewer than 20 moving rows" -- 26 of the 58 no-map, 7 of the 145 no-agent) | 1 % | **FAIL as built by the stock builder** |
|  | with the 33 filled from the camera-timestamp route: **22/4,572 = 0.481 %** (the same 22 as refcv7; the 3 eval139 unclocked left unclocked on purpose) | 1 % | PASS |
| (b) map-GT coverage of train windows (`require_map_coverage`, perception_targets.py:203; counted as `no_file`) | **746,946/781,635 = 0.9556** (old 1.0000, config.json map_fine_stats.train) | 0.90 floor | PASS |
| (c) windows n; one epoch = floor(n/16) | **n = 781,635**; ADD adds 34,689 windows (170.88/clip); **48,852 steps/epoch**; 3 windows dropped/epoch | -- | -- |
Masking, not "no objects": a clip without a map file emits `map_label False` / `map_fine_label False` (refc_v3_train.py:3976-3996, 4068-4086); a clip with no agent join emits NO_LABEL, a different state from labelled-clear (`_agent_item`, :3799-3820). Agent-labelled windows fall to ~93.3 % (ESTIMATED from the 58 clips' labelled-frame fraction 0.9562; old measured 0.9636).
At 9.212 s/step: `--steps 30000` = 480,000 samples = **0.614 epoch** (76.8 h); one epoch = 48,852 steps = 125.0 h. (Changing n also breaks the audit's FINDING that refcv8 replays refcv7's first 480,000 permutation positions: same seed, different n = different permutation.)

## 4. The agent join does not change (the brief's assumption was off)
The 58 no-map clips are ALREADY in the 2-D and 3-D joins (58/58). Of the 145 no-agent clips only 13 have obstacle data, and the 2026-09-06 builder dropped them for cause: 6 have 0-row obstacle parquets, 7 fail the per-clip frame-alignment gate (true error 0.25-1.05 frames). Re-running the builder on those 13 reproduces the same 6 + 7 (`raw/join_ctl16.log`). Forcing them in would condition on agents at the wrong instant -- refused. So `--agent-join`, `--join3d`, `--v7-labels`, `--speed-max-sidecar-v6` (4,572/4,572 rows) and `--agent-cls-weight b1` (its guard compares the join's corpus line, unchanged) all stay as they are.

## 5. TWO HARD BLOCKERS the brief did not list (found reading the smoke argv against the trainer; NOT built)
1. **v9 train release** (`--r8-v9-labels`, md5 `f63ece41...`): 4,369 clips, 869,278 rows. `row_for_now` raises `V9LabelError` for any window of the 203 at launch (v9_labels.py:112-115). Needs `build_v9_labels.py` lead-pass + lc-pass + build over the new manifest with the ts33 clock sidecar (the release's `now_s` is checked to 1e-6 s against the trainer clock). The 4,369-clip build took 997 s. Control: the old 4,369 clips' rows must be identical.
2. **VIS-1 sidecar** (`--vis1-sidecar`): 4,508 clips; `require_clips` refuses on any absent clip (vis1.py:277-283), and every clip needs `clip_mask_ftheta`. Needs `precompute_vis1_sidecar.py` rows for the 58 joined clips plus zero-frame entries for the 145 unjoined (or a builder change so unjoined clips are emitted), merged into a new npz; its sha256 replaces the pin at `launch_gate.py:572`.
Quality items (not refusals): `--join-defect-masks` was mined on the 4,369+139 -- re-mine for the 58 newly-active joined clips (`mine_join_label_defects.py`); `nav_compliance_tau_train.json` (tau 0.18063741505146028) and `map_hires_class_weights_train_100x30.json` are derived from the 4,369-clip view (map weights are GT-only, so the 203 add nothing; tau would shift slightly) -- re-derive for provenance or keep the literal and say so.

## 6. Launch deltas (old -> new) -- Master Mind edits launch_gate.py / SPEC from this
| argv / pin | old | new |
|---|---|---|
| `--v2-cache` | `/home/nvidia/data/refcv6-b1-416x1024-train` | `/home/nvidia/data/refcv8-b1-416x1024-train-4572` |
| train `_v2manifest.pt` md5 (gate pin, pose-sync `inputs_md5`) | `3c9f8bc8583938ff60c57d4d97fcd8c0` | `77dafb50af0aedd1b940cf12c12518ef` |
| `--agent-rig-extrinsics` | `refcv6_train_eval139_extrinsics.json` (`3bd70d98...`) | `refcv8_train4572_eval139_extrinsics.json` (`dba6da20...`) |
| `--pose-sync-sidecar` | `refcv6_pose_sync_sidecar.jsonl` (`ce00a130...`) | `refcv8_pose_sync_sidecar.jsonl` (`894fe1d9...`) |
| `--clip-clock-sidecar` | `refcv6_clip_clock_sidecar.jsonl` (`78466f99...`) | `refcv8_clip_clock_sidecar_ts33.jsonl` (`c81c2739...`); `--label-clock-max-unverified` stays default 0.01 |
| `--steps` | 30000 (0.643 ep at n=746,946) | pick against n=781,635: 30000 = 0.614 ep; 1 epoch = 48,852 |
| `--r8-v9-labels` / `--r8-v9-md5` | train.npz `f63ece41...` | **BLOCKED (section 5.1)** |
| `--vis1-sidecar` / launch_gate.py:572 | `vis1_sidecar_refcv6b1_train4369_eval139.npz` | **BLOCKED (section 5.2)** |
| `--agent-join`, `--join3d`, `--v7-labels`, `--speed-max-sidecar-v6`, `--eval-*`, `--map-gt-root` | -- | unchanged |
Scope caveat: the Drive/session-level train/eval disjointness is still UNPROVABLE (audit section 4); only clip-id disjointness is asserted.
