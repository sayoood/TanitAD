# run1 files deliberately not banked (MM 2026-11-28)

| file | sha256 | why |
|---|---|---|
| `index_sealed.csv` | `f7426318271aebc71e9e256b3dc7f2edfa96dd57d57190342a30679e87434c03` | identical to SAFE-HEAD run1's sealed index (same 35-log cache, `F:/safehead_cache`) |
| `read_eph/VERDICT_EPHEAD.json` (full, 170107418 bytes) | `3dbf11ffa2e61b5dad49f9cf08ab06bded942cca94cd3f7877250026fb2915b0` | 170 MB; the slim copy drops only the three per-window `fam_*.json` dumps |
| `read_eph/fam/families/fam_{C1pick,EPH0,EPH1}.json`, `fam/pass/rows_s*of2.jsonl`, `fam/plans/plans_*.npz`, `read_eph/seams/seam_*.npz` | - | per-window intermediates (~270 MB), re-created by `scripts/run_fam_chain_eph.sh` from the banked heads |
| `gate_*/work/` | - | launch-gate scratch (~143 MB per arm) |
