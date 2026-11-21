# E17 follow-up: `fetch_corpus_clips.py --local-mirror` — the rebuild stages from disk, and the content check is kept

**Date** 2026-09-19 · **Owner** DataFlyWheel · **Asked by** Master Mind · **0 GPU**
⛔ **The corpus rebuild was NOT run** — it waits on the SAM3 corpus (ETA 2026-09-22 ~18:30).
Only a three-clip staging smoke test was run, into a scratch root that was then removed.

## What was built

| file | what |
|---|---|
| `stack/scripts/corpus_mirror_stage.py` | ⭐ NEW. Pure functions, no network: `reconcile_expected`, `verify_sources`, `stage`, `verify_file` |
| `stack/tests/test_corpus_mirror_stage.py` | ⭐ NEW. **9 tests, 9 pass** (`raw/pytest_corpus_mirror_stage.txt`) |
| `…/2026-09-16-256x1024-cache/code/fetch_corpus_clips.py` | `--local-mirror`, `--local-timestamps-tar`, `--local-egomotion-tar`, `--link copy|hardlink`. ⭐ **Additions only: 0 lines removed or changed, 135 added** — the HF path is byte-identical. |

```
python fetch_corpus_clips.py --ids ids.txt --root <staging root> \
    --local-mirror C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov \
    --local-timestamps-tar D:/Projects/TanitAD-artifacts/_s2build-copy-20260919/release/tanitad-v7-training-corpus/timestamps/timestamps.tar \
    --local-egomotion-tar D:/Projects/TanitAD-artifacts/_s2build-copy-20260919/release/tanitad-v7-training-corpus/egomotion/egomotion_alpamayo.tar \
    --receipt receipt.json
```

## The content check is kept — and tightened

* **The expected hashes still come from HF at the pinned revision (`a0cf20df`), from TWO
  independently produced sources that must agree:** the uploader's `camera/camera_sha256.json`
  and the Hub's own LFS metadata. HF disagreeing with itself is a hard failure, not a choice.
* **Every mirror file is hashed BEFORE anything is staged, and one mismatch stages NOTHING.** The
  HF path exits on the first bad file; a half-staged root looks usable and is not.
* **Every copy is hashed again after it is written**, so a bad copy is caught as well as a bad
  source.
* **The two tars, when given locally, are checked against their HF LFS hashes.**
* ⛔ **The run deletes only what IT downloaded** — never the operator's local tars (verified: both
  still present after the smoke test).

## Proof that the tests can fail — mutation, not inspection

The source sha256 comparison was removed from a copy of the module and the suite re-run:

| mutant | result |
|---|---|
| source hash check removed | ⛔ **2 tests RED**, 7 green (`raw/mutation_source_hash_removed.txt`) |

⭐ **The mutation found a real gap, and a test now closes it.** With the source check gone, COPY
mode was still refused — by the copy re-hash — but only **after** staging an earlier file, so the
"stages nothing" guarantee broke, and the one-byte-tamper test went red for exactly that reason.
**HARDLINK mode has no copy to re-hash, so the source check is its ONLY guard.** The first draft had
no hardlink tamper test; the mutant would have staged a tampered file silently.
`test_one_byte_tamper_refuses_in_hardlink_mode_too` was added and reads **DID NOT RAISE** under the
mutant. ⭐ Both tamper tests flip **one byte at the SAME size**, which a size check passes and only
the hash catches.

## The smoke test: three clips, end to end

`raw/smoke_receipt.json`, `raw/smoke_run.log` — MEASURED:

* HF sha table and LFS metadata **agree** on all 3 clips at `a0cf20df`;
* 3 mp4 staged, every source hashed first and every copy re-hashed; **0 mp4 downloaded**;
* both tars taken from D:, each **sha256 = HF LFS**; timestamps 3/3; egomotion zip 3 members;
* wall **690 s** — almost all of it reading 2 GB of tars from D: (see below).

## ⚠️ D: is slow, and one of my own earlier claims needs a condition

`raw/d_drive_io.json`, MEASURED today:

| D: (exFAT, external) | MB/s | conditions |
|---|---|---|
| read + sha256 | **18.7** | 319 MB sample |
| write, fsync | **34.7**, **32.5** | two 268 MB trials, uncontended by this session |
| write, fsync | ⛔ **8.7** | while this session read the 2 GB tar from D: |
| C: read + sha256 (reference) | 826.5 | all 61.6 GB of the mirror |

The 416×1024 rebuild must write **~10.2 MB/s** on average (446 clips/h × 82.0 MB/clip, the 2026-09-16
package's measured rate, INHERITED). ⇒ **D: has ~3× headroom when nothing else is reading it — and
falls BELOW the need when something is.** My C3 result said *"no disk here is the bottleneck; the
build is CPU-bound"*: that holds **only uncontended**. ⛔ So: **do not overlap the rebuild with other
heavy D: I/O**, and put the mp4 mirror on C: (it already is) — only the tars and the cache touch D:.
An appended note in the C3 package records this.

## Reproduce

```
python -m pytest stack/tests/test_corpus_mirror_stage.py      # 9 pass, no network
```
