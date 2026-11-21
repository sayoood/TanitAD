# Landing log: the E:-tree stranded set (2026-10-18)

Landed by the Master Mind from `AUDIT.md`'s `paths.json` (1,979 entries: the LAND tier, one recovered index-only blob,
and this audit package), in four commits on `agent/arch-inf-20260803`:

| chunk | commit | files | notes |
|---|---|---|---|
| 1/4 | `f077d33` | 668 | |
| 2/4 | `f415332` | 795 | one package exceeded the 120 MB chunk target (289.3 MB) |
| 3/4 | `88fed18` | 305 | historical pod addresses redacted in 4 files (2 copies of `sft4r_chain.sh`, its diff, a GAC E-only-lines copy) |
| 4/4 | `27f0eec` | 209 | 2 files EXCLUDED, see below |

* **75 sources had paths over Windows' 260-character limit** (`LongPathsEnabled` = 0, not changed); they were read through
  the `\\?\` prefix into short copies, byte-verified, and landed under their full repo paths.
* **Excluded (2), never altered:** a data value in each matches the lander's secret-scan port pattern (a float in a data column; the
  value is deliberately not quoted here, since quoting it trips the same scan). Data is not edited to pass a scan. Both are backed up on C: with their sha256, and off-site on the same HF prefix (content-verified, `offsite/HF_VERIFIED_excluded.json`):
  * `TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/raw/d6_scene_table_step30000.csv` (`137a033f…`)
  * `TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/raw/p4_mask/mask_R7_A1.index.jsonl` (`b5ab6eb9…`)
* **LAND-HOLD-SCRUB (11):** landed in the next commit from the audit's mechanically scrubbed copies (`<redacted-ip>`,
  `<port>` placeholders; 1-5 changed lines each; scan clean).
* **BULK-ONLY + LAND-LARGE (2,252 files, 4.17 GB):** NOT in git. Their only copy was the external E:/D: drive; a second
  copy now lives on internal C: at `C:/Users/Admin/estrand_bulk/` with `MANIFEST.sha256.json` (every file sha256-verified
  on copy, 2026-10-18). **Off-site copy DONE and content-verified (2026-10-18 14:22 Berlin):** the private HF dataset
  `Sayood/tanitad-internal-data`, prefix `stranded-bulk-2026-10-18/`, holds all 2,252 files plus `MANIFEST.sha256.json`
  (4,173,996,554 bytes). Every file was re-read and checked against the hub: the LFS sha256 or the git blob sha1, and the
  manifest's own sha256. 2,253 / 2,253 OK, 0 bad (`offsite/HF_VERIFIED.json`). Uploader: `offsite/hf_bulk_up.py`. The token
  is read in place, never written.
* **Integration items** (29 package-local code variants, 47 candidate edits, 13 LANDING_READY manifests) are triaged in
  `integration/` (separate agent); landing these files does not apply any of them to `stack/` or the registers.
