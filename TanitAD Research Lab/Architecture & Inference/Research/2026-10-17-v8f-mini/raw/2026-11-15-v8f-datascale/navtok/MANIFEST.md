# MANIFEST (navtok package, 2026-11-16). Nothing staged, committed or pushed; the MM lands from this folder.

Package: `F:/TanitAD-scratch/datascale/navtok/` (dev box, one copy each; nothing is held only on the pod).

| artifact | path (package) | notes |
|---|---|---|
| report (tasks 1, 3, 4) | `NAVTOK_REPORT.md` | trainer nav trace at file:line, table, agreement |
| parser | `navtok.py` | frozen 44-class vocab, 0/9,864 unparsed |
| tests | `test_navtok.py` | 33 pass, incl. deliberate-regression arm |
| parse report | `navtok_report.json` | counts per class, unparsed list (empty) |
| per-step table | `navtok_table.npz` (sha256 `2836369e3e14a8c3c49a6b1a42116c36cda522ca42f98a6bcbc8c4e4d45c8992`, 2,300,115 B) + `navtok_table_report.json` | train 461,518 steps, eval 14,237 steps, aligned to the joins |
| table builder | `build_table.py` | T0 = -195,000 us (ESTIMATED) |
| agreement | `agreement.json`, `disagreements.csv`, `analyze.py` | confusion, distances, T0 scan, 60 samples per category and split |
| pre-registration draft | `PREREG_NAVTEXT_DRAFT.md` | NOT RUN, not frozen |
| inputs (re-creatable) | `raw/nav_sub.parquet`, `raw/nav_sub.npz`, `raw/texts.json` (cut from the pod's `navigation.parquet`), `raw/join_train.npz` (pod `/root/runs/bprime/`), `raw/join_eval141.npz` (repo), `raw/train4572.txt`, `raw/clips141.txt` | the pod originals of the three nav files are deleted; the source `navigation.parquet` stays on the pod |
| pod extract scripts | `src/pod_extract_ids.py`, `src/pod_extract_npz.py` | how the nav files were cut |

Only-one-place: everything above exists only in this package. The MM should land it (suggested repo folder: a dated `raw/` folder under the datascale package). `raw/` is about 38 MB, mostly `join_train.npz`.

## Deletions (by literal path, this session)

| path | size | why safe |
|---|---|---|
| pod `/root/datascale/navtok/ex.py` | 527 B | copied to `src/pod_extract_ids.py` |
| pod `/root/datascale/navtok/ex2.py` | 552 B | copied to `src/pod_extract_npz.py` |
| pod `/root/datascale/navtok/nav_sub.npz` | 1,403,679 B | package copy, sha256 equal |
| pod `/root/datascale/navtok/nav_sub.parquet` | 1,716,477 B | package copy, sha256 equal |
| pod `/root/datascale/navtok/texts.json` | 296,284 B | package copy, sha256 equal |
| pod dir `/root/datascale/navtok/` | empty | created by me |
| package `src/v8f_mini_train.py` | 242,468 B | copy of tip `ccec0f28` file, re-creatable by `git show` |
| package `src/v8f_mini_data.py` | 74,764 B | same |
| package `src/v8f_mini.py` | 93,155 B | same |
| package `src/build_v9_labels.py` | 65,773 B | same |
| package `src/PREREG_DAG_OACC_ref.md` | 9,705 B | same |
| `__pycache__`, `.pytest_cache` in the package | small | test caches |

Not touched: the pod's encode processes and `/root/datascale/{state10k,w,keep10k,ledgers10k}`. No GPU load. No address, port or token written anywhere.
