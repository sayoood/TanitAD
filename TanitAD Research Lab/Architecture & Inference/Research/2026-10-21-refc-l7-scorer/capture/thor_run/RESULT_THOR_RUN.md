# H-L7-1 capture -- the Thor run is COMPLETE and its identity control K3 PASSES (no L7 number exists)

Filed 2026-10-22 12:27 Berlin (clock read). Master Mind. Evidence class: MEASURED, artifacts in this folder.

## What ran

`launch/chain_l7cap_yield_thor.sh` (pid 1355745, started 2026-10-22 00:37) waited for the R8-PICKER rolls
(`ZZPKWDONEZZ` 06:02), then ran the four registered capture steps on Thor's GPU, each checking
`NT_LAUNCH_PENDING` first (none was set):

| step | windows | seed | finished | rc | wall |
|---|---|---|---|---|---|
| eval139 s1 (the K3 control) | 4,754 | 1 | 06:27 | 0 | 1,410.8 s |
| train139 s0 | 4,855 | 0 | 06:52 | 0 | 1,450.1 s |
| train139 s1 | 4,855 | 1 | 07:17 | 0 | 1,442.8 s |
| eval139 s0 | 4,754 | 0 | 07:43 | 0 | 1,456.3 s |

Per step (`*/fan_roll_s*.json`): label, agent-label and map-label windows = all windows; 0 scene-missing windows;
0 nav-zero calls; 0 calls without `nav_true`; K3a (fan at the selected index minus the shipped trajectory)
**0.0 m**; K3b (trajectory equals the sidecar `nav_true` plan) **true**; K4 (E9 argmax equals `sel_idx`)
**1.0**. train139 at both seeds reproduces the REGISTERED window rule exactly (`TRAIN139_WINDOWS.json` sha256
`5ed3283f…`: 139/139 clips, 4,855 windows, 0 mismatches, 0 missing). Every step's clip-id scrub: CLEAN,
0 residual over 282 npz + 6 text files.

## K3 -- capture identity (PREREG_L7_SCORER.md control K3): PASS, exact

The new eval139 s1 capture against the banked Thor fanceil fan
(`/home/nvidia/r8ns/fanceil/out/s1/dump/fan_extras.npz`, md5 `6e6699d9…`), on every array the two files share
(`K3_eval139_s1_vs_fanceil_s1.json`, computed on Thor CPU):

| array | shape | max abs diff | elements different |
|---|---|---|---|
| `anchor_traj` (the registered K3 array) | 4754 x 149 x 8 x 2 | **0.0** | **0** |
| `anchor_logits`, `refined_logits`, `sel_score`, `reach_keep` | 4754 x 117 / 149 | 0.0 | 0 |
| `traj` | 4754 x 8 x 2 | 0.0 | 0 |
| `sel_idx`, `clip_index`, `t`, `ws` | 4754 | 0.0 | 0 |

10 of 10 shared arrays are bit-identical, so Thor's sampler is deterministic for this roll and the
R8-PICKER K1 envelope fallback is not needed. The new capture carries 13 additional arrays (the L7 labels and
selection internals) that the old fan never had.

## What this does and does not settle

* It settles that the L7 TRAINING data (train139 at two inference seeds) and its held-out eval capture exist,
  are complete on the registered windows, and reproduce the banked fan exactly where they overlap.
* It produces **no L7 number**. The scorer itself is not trained: **scorer TRAINING awaits the PI's go**
  (registered in `GOALS_AND_CLAIMS.md`, rows `H-L7-1` and `H-L7-1 capture`).
* After R8-ORACLE (NO-HEADROOM in the tactical layer, 2026-10-22) and R8-PICKER (a GT-free re-pick is real but
  small and fails A2 on a lateral cell), this scorer is the remaining SELECTION lever the registered results
  point at; the DECODER lever is the NT-DECIDE A1 navtrain fine-tune.

## Not in git

The four `dump/` directories, 1.5 GB each (fan_extras, l7cap_labels, l7cap_scene, refcv6_extras and the
per-episode npz): on Thor at `/home/nvidia/r8ns/l7cap/out/<step>/dump/`, md5 of the four large arrays per
step in `*/MD5_L7C*.txt`, and copied to the dev box at `D:/tanitad-archive/l7cap_thor_1022/` (2026-10-22).
