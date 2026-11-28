# The D:-only scene-tap probe artifacts, banked to HF and verified by content (Master Mind, 2026-10-16)

**Why.** The scene-tap probe units and the C3 navtest scene-context features lived only on the dev box's external drive (D: = E:), the one copy of the evidence behind A2 / A3R and the aux shift-locate study (operating standard 3: an artifact on one disk is not done).

**What was banked** to the private model repo `Sayood/tanitad-refe-s1aux`:

| local (dev box) | repo path | files |
|---|---|---|
| `D:/Projects/TanitAD-artifacts/scene_tap_probe/` | `probes/2026-10-15-scene-tap-probe/` | 41 |
| `D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full_scene_ctx/` | `probes/2026-10-16-aux-navtest-forward/c3_full_scene_ctx/` | 2,120 |

**Verification (MEASURED, `manifest.json`, `BANK_DONE`).**
- Every local file has a remote entry: **2,161 / 2,161**.
- Every entry has the same size.
- **2,157** are LFS, and each one's sha256 equals the local sha256.
- The 4 non-LFS entries are small JSON/JSONL and are checked by size only. They are listed in `manifest.json` with `lfs_checked: false`.
- `bank_probe.py` refuses a non-private repo. It reads the token in place and never prints it. It uploads with `HF_HUB_DISABLE_XET=1`.

**The local copies are KEPT.** The S1c read and the aux-veto feasibility study may re-read them. ⚠️ `du` reports 2.1 GB for `c3_full_scene_ctx`, but that is exFAT allocation: the data is 2,116 × 65.7 kB ≈ 0.14 GB.
