
<!-- S4-RETIER-MECHANISM-PRICED-PI-ITEM-19-2026-09-22 -->

### ⚠️ 2026-09-22 — the §4 pass's only open question is a PUBLISHED-REPO LAYOUT choice, and the only self-consistent option DELETES files ⇒ PI item 19, with a safe default

MEASURED by me, read-only, no writes to Thor or HF
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-s4-retier-mechanism/`).
Continues `b513a26`, which verified the pass's premise.

**The ruling is made; only the mechanism was open.** `corpus_publisher.gt_path()` routes validated
clips to `semantic_maps/gt/` and flagged clips to `semantic_maps/gt_flagged/`, so re-tiering changes
a clip's **directory** — which is a change to a published artifact's layout, not merely a judgement.

**MEASURED, so the options could be priced rather than argued:**

| fact | value |
|---|---|
| local GT sources still present for all 77 flagged clips | **77 / 77** (`corpus/out/`, 4,546 npz, 14 GB) |
| their total size | **148.6 MB** (mean 1.93, max 3.10) |
| publisher's commit operations | **`CommitOperationAdd` only** — no delete path exists today |

| option | storage | self-consistent? |
|---|---|---|
| A — manifest-only, bytes stay in `gt_flagged/` | 0 MB | ⛔ a consumer globbing `gt/` gets 78 fewer maps than the manifest promises |
| B — add to `gt/`, keep the flagged copy | ~150 MB | ⛔ mirrored: `gt_flagged/` holds 78 clips the record calls validated |
| **C** — add to `gt/`, **delete** from `gt_flagged/` | 0 net | ✅ **the only one** |

⛔ **C is not mine to take.** It permanently removes files from a published dataset repo; the §4
ruling authorises a **re-tier**, which is a judgement, not a deletion. ⚠️ Stated fairly in the
queue: the file C deletes is one the same pipeline produced and is simultaneously re-uploading — an
argument FOR C, and not a reason to skip asking.

⭐ **Safe default recorded as B** if no ruling arrives before production ends (**ETA today 17:44**):
nothing in B is irreversible and C stays available afterwards at any time. Its cost is stated, not
hidden — ~150 MB, and an untidy directory — and the pass will record the duplicate **explicitly**,
naming **both** paths on each re-tiered entry so the record cannot mislead even while the layout is.

⛔ **A is explicitly NOT the default despite being free.** A file under `gt_flagged/` that the
record calls *validated* is exactly the *artifact that cannot be read in isolation* failure the
anchor-units disaster taught, and unlike B's untidiness it would be permanent and silent.

⇒ **PI queue item 19.** Nothing else waits on it; the pass is written against whichever option is
chosen and runs after DONE.
