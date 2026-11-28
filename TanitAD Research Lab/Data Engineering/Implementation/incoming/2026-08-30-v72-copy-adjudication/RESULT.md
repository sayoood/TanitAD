# v7.2 COPY ADJUDICATION — the artifact is the bytes, the path is not

**Date** 2026-08-30 · **Owner** DataFlyWheel · **Trigger** Master Mind finding
**Status** RESOLVED — trap disarmed, gate deliverable banked

---

## 1. The finding (Master Mind, not mine)

Both v7.2 label names resolved to **two copies with two distinct md5s**, and
`tanitad/train/intrain_eval.py` had pinned its hash guard to the wrong pair:

| copy | bytes | md5 |
|---|---|---|
| `release/_v72_verify/labels/s2_labels_v7.2_train.jsonl.gz` | 1,997,527 | `0a586fe814e08bac81ae18e312d0dc24` |
| `release/v72/s2_labels_v7.2_train.jsonl.gz` **(canonical)** | 1,999,886 | `0ff902130ce76886b8a925eceed9e3a5` |
| `release/_v72_verify/labels/s2_labels_v7.2_eval.jsonl.gz` | 65,727 | `bffc9df52fc4e6afd58ec5e509fe2a41` |
| `release/v72/s2_labels_v7.2_eval.jsonl.gz` **(canonical)** | 65,787 | `aa12c948f062181c3297265b51526ec5` |

⛔ **The guard would have REFUSED the published artifact and ACCEPTED the
superseded one.** The pins were captured through
`glob.glob("C:/Users/Admin/**/<name>")[0]` — first hit of a recursive glob — so
the *subject* of the hash assertion was whichever copy the filesystem yielded.

---

## 2. Adjudicated BY CONTENT (`code/adjudicate_v72_copies.py`)

Not by which directory is named "verify", and not by which explanation sounded
right. Three readings were possible and only one survives:

| reading | prediction | observed |
|---|---|---|
| pre-schema-fix debris | same clips, identical payloads, `schema_version` differs | ✅ **this one** |
| payloads drifted | some records differ in fields the fix never touched | ✗ zero |
| different release | clip sets differ | ✗ identical |

```
TRAIN   n=4572   clip ids identical=True   payload diffs=0
        _v72_verify   schema_version=['s2_labels_v7.2']   release=[None]
        v72           schema_version=['s2-geom-v7']       release=['v7.2']
EVAL    n=147    clip ids identical=True   payload diffs=0   (same fields differ)
```

⇒ **The `_v72_verify` pair is the pre-fix blob. Superseded. Nothing was lost.**

⚠️ **ONE CORRECTION TO THE READING.** The Master Mind reported *"a test reading it
died with `JSONDecodeError` on line 1"*. **These files parse fine as JSON** —
4,572 and 147 records, cleanly. What refused them was the **loader's schema
check** (`v7_labels.EXPECTED_SCHEMA = "s2-geom-v7"`), not JSON validity. The
distinction matters because it points at a different fix: a parse failure says
*the writer is broken*; a schema refusal says *the writer set a contract field to
a release string*, which is what actually happened. Their `release=[None]` — the
key did not exist yet — is independent confirmation.

---

## 3. What I changed — and why a RENAME, not a delete

`code/quarantine_and_build_eval_digests.py` moved both blobs to
`release/_superseded_pre_schema_fix/` **under new names**:

```
s2_labels_v72_train.PRE-SCHEMA-FIX.DO-NOT-USE.jsonl.gz
s2_labels_v72_eval.PRE-SCHEMA-FIX.DO-NOT-USE.jsonl.gz
```

⭐ **The rename IS the fix.** The defect was a match on **filename**. Moving the
files while keeping their names would have left the trap fully armed for the next
recursive glob. Renaming breaks it. md5s were re-checked after the move and are
unchanged, and a `README.md` in that directory states all of the above — because
the next reader has now been demonstrated to be a machine.

The bytes are kept rather than destroyed: 2 MB, the finding stays reproducible,
and nothing references them. **If the Master Mind would rather they were gone,
the directory is one `rm -rf` away.**

---

## 4. The EvalFlyWheel deliverable

`raw/eval_v72_clip_digests.txt` — `sha256(clip_id.encode("utf-8")).hexdigest()`
for the **147** v7.2 eval clips, one per line, sorted, with a header.

* Built from `release/v72/clip_index_eval.json`, re-asserted **on the written
  file** (digests round-trip; the val40 intersection recomputed from the file
  itself reads **6**).
* **Digests only** — no clip id is recoverable, so nothing confidential travels
  and the `confidentiality` field on the val40 digest file is not engaged.
* `sha256(file)` = `be39fb29575115f0…`

⚠️ **IT IS A DRIFT DETECTOR, NOT A DISCOVERY INSTRUMENT.** The 6 is true **by
construction** — `build_v72.py` asserted it before writing the split. A gate
reading 6 today has confirmed nothing new. Its value is the day the eval set
changes and nobody re-checks. The written file says so in its own header so a
future reader cannot quote it as a finding.

---

## 5. The class, which is the part worth keeping

**Identifying an artifact by LOCATION instead of by CONTENT.** The warning was
already in our own tree — `LabelManifest.to_dict()` says *"md5 is the identity —
six copies of this blob exist under three roots and their md5s differ"* — and the
module that pinned the wrong copy sits beside it.

⚠️ **It recurred while writing this very fix.** I reached for
`v72/clip_index_v7.2_eval.json` (the **HF** name) and got `FileNotFoundError`;
on disk the file is `v72/clip_index_eval.json`. The upload renamed it. That is
the same confusion, caught only because the wrong path happened not to exist —
which is luck, not method. **Where a name differs between local disk and the
remote, state both**; the manifest now does.

Related: **C82** (price the artifact the CONSUMER reads) and the DE-C152
artifact-swap lesson — *a true measurement taken on the wrong file reads exactly
like an answer*.
