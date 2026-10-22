# R8-PICKER registered read: artifacts that exist but are NOT banked in git

Written 2026-10-22 by the EvalFlyWheel when the hold was built. Everything here lives on the dev box under
`C:/Users/Admin/refcv8_pf/picker/` (and, for the dumps, on Thor). Nothing listed here is the only copy of a
number: `PICKER_R8.json` holds every number in `RESULT.md`, and the dumps below are the inputs it was computed from.

Why each group is not banked is stated per section. A pulled dump set over 30 MB is not banked; full PhysicalAI
clip ids are never banked (only sha12 labels).

## 1. Pulled capture dumps (4 files, 118,535,326 bytes in total, over the 30 MB limit)

Thor originals: `/home/nvidia/r8ns/picker/out/R0/` and `/home/nvidia/r8ns/picker/out/R1/`.
Dev-box copies: `C:/Users/Admin/refcv8_pf/picker/runs/R{0,1}/dump/`.
R0 is seed 0 (FULL ARM, `roll_s0.json`, `fan_roll_s0.json`); R1 is seed 1 (`--os-only`).

| dev-box path | Thor path | bytes | md5 | sha256 |
|---|---|---|---|---|
| `runs/R0/dump/fan_extras.npz` | `/home/nvidia/r8ns/picker/out/R0/dump/fan_extras.npz` | 53,542,928 | `eda87e6a5c3088d8ba7a45a7b9cca0b5` | `6165d7fcb15dc6ddbef4dda851dfd8919639d0087b114d8e753932a613612e61` |
| `runs/R0/dump/refcv6_extras.npz` | `/home/nvidia/r8ns/picker/out/R0/dump/refcv6_extras.npz` | 9,803,860 | `44a90a2ef038b1ec52395700cf773249` | `c7e676ce6c11466d08ff8d8d68f79f4abdefe70980ad5c484b968d05c579b38d` |
| `runs/R1/dump/fan_extras.npz` | `/home/nvidia/r8ns/picker/out/R1/dump/fan_extras.npz` | 53,547,441 | `00917687d93687c6d7fbb1279339b624` | `750e6e1f1800502b6e4ed2c1f60b5e727a83ec65d96ca3688c5e40fee7342a23` |
| `runs/R1/dump/refcv6_extras.npz` | `/home/nvidia/r8ns/picker/out/R1/dump/refcv6_extras.npz` | 1,641,097 | `f9f4c1ef2bccda7eaf4622d8e3696255` | `9f6884aaeb864c6546fbb3277a919c47e6cb4e7aeccbaabdbd835c89ba4fe806` |

The two `fan_extras.npz` sha256 values equal the `fan_sha256` that `BUILD_PICKER_TREES.json` recorded for seed 0 and
seed 1 (MEASURED, string-compared at hold-build time). The md5 column was recomputed on the dev box at hold-build
time and equals the Thor-side `MD5_R0.txt` / `MD5_R1.txt` below (also banked as `runs/R{0,1}/MD5_R*.txt`, with the
`md5_check.txt` results, all OK).

`MD5_R0.txt` (verbatim):

```
eda87e6a5c3088d8ba7a45a7b9cca0b5  dump/fan_extras.npz
44a90a2ef038b1ec52395700cf773249  dump/refcv6_extras.npz
bc360b9e1353db3fd7abedfdc089bfe0  fan_roll_s0.json
0781fc3dffb9c27436c04bf7786fa434  roll_s0.json
```

`MD5_R1.txt` (verbatim):

```
00917687d93687c6d7fbb1279339b624  dump/fan_extras.npz
f9f4c1ef2bccda7eaf4622d8e3696255  dump/refcv6_extras.npz
26bb5b75219ed765f6ccd7c36f5a1128  fan_roll_s1.json
31414fbda1f58b92d950ae3042260a21  roll_s1.json
```

## 2. Derived lane arrays (6 files, 19,330,076 bytes): `DEPLOY8_POS_<tree>_s{0,1}.npz`

Per-window 8-slot deploy positions written by the lane's RETIME_DEPLOY8 step (log `lane_d8.log`) and READ by
`code/analyze_picker_r8.py:182`. Not banked because they are derived; regenerating them needs the dumps above plus
the trees of section 3, about 15 minutes of CPU per tree (MEASURED from the lane timestamps in
`logs/pull_analyze.log`: 06:06 to 06:22, 06:22 to 06:37, 06:37 to 06:52). Their sibling JSON
(`FORWARD_DEPLOY8_<tree>.json`) IS banked under `result/lane/<tree>/pfpkg/raw/retime_deploy/`.

| dev-box path (under `lane/`) | bytes | md5 |
|---|---|---|
| `pkship_pf/pfpkg/raw/retime_deploy/DEPLOY8_POS_pkship_pf_s0.npz` | 3,221,678 | `683cb833cd7fc84a75f8dc133d44b1bc` |
| `pkship_pf/pfpkg/raw/retime_deploy/DEPLOY8_POS_pkship_pf_s1.npz` | 3,221,678 | `20a6edce53d06ff164ab1de4e136d55d` |
| `pkbase_pf/pfpkg/raw/retime_deploy/DEPLOY8_POS_pkbase_pf_s0.npz` | 3,221,678 | `e052460766bff974cdbb407899a9793d` |
| `pkbase_pf/pfpkg/raw/retime_deploy/DEPLOY8_POS_pkbase_pf_s1.npz` | 3,221,678 | `9fafe69dd9e98be26f0d79f07df1dfe0` |
| `pkmbr32_pf/pfpkg/raw/retime_deploy/DEPLOY8_POS_pkmbr32_pf_s0.npz` | 3,221,682 | `bef283ddcb3f18258dba4de0a89c673d` |
| `pkmbr32_pf/pfpkg/raw/retime_deploy/DEPLOY8_POS_pkmbr32_pf_s1.npz` | 3,221,682 | `ab39a48ca5c5fbc7b6557fb4b715e2b5` |

## 3. The three registered trees `archive/pk{ship,base,mbr32}_pf/` (1,697 files, about 84.6 MB each)

Each tree is the banked refcv8 per-frame battery tree (`C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf`)
with the picker's selection spliced into the plan rows (`code/build_picker_trees.py`; parameters, K3a identity
residuals and per-tree splice counts are in the banked `BUILD_PICKER_TREES.json`). Not banked because they are 254 MB
in total and `windows_s0.json` / `windows_s1.json` in each carry the FULL PhysicalAI clip ids as keys (139 clips).
The picks themselves ARE banked: `result/archive/PICKS_s0.npz`, `PICKS_s1.npz` (integer and boolean arrays only).
Rebuilding a tree from the dumps was not re-run when this hold was built (UNVERIFIED as a clean-directory rebuild).

## 4. Earlier reads, not part of the registered read

`archive/pkident_pf`, `archive/pv{ship,base,mbrpx,mbrrule}_pf` and the matching `lane/` directories (35 files, about
8.7 MB each) belong to the K9 identity lane and the PROVISIONAL reads of 2026-10-21. Their results are already on the
tip as `K9_LANE_IDENTITY.json` and `PROVISIONAL_S1*.md|json`. `dryrun/` (13 files, 250,566 bytes), `smoke/`
(57 files, 812,045 bytes) and `code/__pycache__/` are dry-run and cache debris; the dry-run and smoke results are on
the tip as `DRYRUN_PICKER_R8.json`, `DRYRUN_RESULT.md`, `K1_SMOKE.json`.

## 5. Empty files (0 bytes, not staged)

`logs/dev_waiter.out`, `logs/lane_pkbase_pf.out`, `logs/lane_pkident.out`, `logs/lane_pkmbr32_pf.out`,
`logs/lane_pkship_pf.out`, `logs/lane_pvbase.out`, `logs/lane_pvmbrpx.out`, `logs/lane_pvmbrrule.out`,
`logs/lane_pvship.out`. The lane runner wrote nothing to stdout or stderr; its real output is the banked
`result/lane/<tree>/lane*.log`, `verdict.log` and `parity_s*.log`.

## 6. One package file deliberately left out

`thor/PREREG_R8_PICKER_ADDENDUM1.md` (dev box) is the UNREDACTED sealed Addendum 1; it carries two clip-id prefixes.
The tip holds the redacted banked copy at the package root (`PREREG_R8_PICKER_ADDENDUM1.md`, blob `86427322e7e2`),
whose own hash differs from the seal by construction; the sealed original stays on the dev box with sha256
`994bd84910dd920deec948bdc2217b6948e8729121cf0eea3becf8329de1ba89`, which matches its seal file.

## 7. Redaction applied to banked logs

`result/runs/R0/pkroll.log` and `result/runs/R1/pkroll.log`: the `[k/139] <8 hex>-<3 hex>` clip-id prefix on 139
progress lines each (278 in total) is replaced with `<sha12-redacted>` in the STAGED copy; every other byte is
unchanged. The dev-box originals are unmodified (md5 `7b438919397e3ba7fca93efd05155fd7` for R0, `63f632245dc5839e21204cf5ac3a9d7e` for R1).
