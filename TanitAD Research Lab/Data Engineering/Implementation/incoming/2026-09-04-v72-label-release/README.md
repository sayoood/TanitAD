# v7.2 label release — banked 2026-09-04

⛔ **Why this exists.** Both `refav1-b1-v72-ep3-speed` (21,109 steps, one full
epoch) and `refcv3-b1-v72-30k` (40,284 steps, published as
`Sayood/tanitad-refc-v3`) trained on these labels, and **neither blob was tracked
by git**. Only v7.0 was. They existed on local disk plus pod copies only — one
disk failure from making every v7.2 result unreproducible.

| file | bytes | md5 |
|---|---|---|
| `raw/s2_labels_v7.2_train.jsonl.gz` | 1,999,886 | `0ff902130ce76886b8a925eceed9e3a5` |
| `raw/s2_labels_v7.2_eval.jsonl.gz` | 65,787 | `aa12c948f062181c3297265b51526ec5` |

The train md5 is the one stamped in `MODEL_REGISTRY.md` §2.4 and printed in every
refav1 join log; the eval md5 is the one every refcv3 eval record carries.

⚠️ **Not placed under `data/`** — that path is gitignored at `.gitignore:16`
deliberately. This follows the v7.0 precedent
(`…/2026-08-24-label-extraction-overnight/raw/s2_labels_v7.jsonl.gz`).

⛔ A **pre-schema-fix** pair sits beside the source, quarantined as
`*_PRE-SCHEMA-FIX.DO-NOT-USE.jsonl.gz`. It is deliberately NOT banked. Verify any
copy by md5 against the table above before use — the two pairs differ by only a
few KB and are easy to confuse.

**Schema:** `schema_version = "s2-geom-v7"`, `vocab = "v7"`, 4,572 records.
`load_v7_labels` (`stack/tanitad/data/v7_labels.py:163`) asserts record count,
schema and vocab, and returns the file's md5 in its manifest — so *which blob did
this run read* is answerable from the run's own config.
