# The 18 diverged Data Engineering files from the D: sync — HEAD is authoritative for all 18, and nothing is merged

**Date** 2026-09-19 · **Owner** DataFlyWheel · **Asked by** Master Mind (task 1 of 3, PI-approved)
**Inputs** `untracked_differ` (18 paths) in `C:/Users/Admin/qland/d_sync_stage2.json`
(banked as `raw/d_sync_stage2_input.json`) · backup ref `refs/backup/d-pre-sync-20260919`
= `57e2944` · HEAD = `37645fc`.

## Verdict

| class | n | what the exact test proves | authoritative |
|---|---|---|---|
| **BLOB-IDENTICAL** | 5 | backup blob **is** HEAD's blob. The sync compared on-disk bytes (CRLF worktree vs LF blob) and flagged a difference that git does not have | HEAD (= backup) |
| **REDACTION-ONLY** | 10 | `HEAD == redact(backup)` **exactly** (9 files) or as JSON (1 file). `redact()` maps each raw clip id → sha12, each 8-hex clip-id **prefix** → the sha12 of its full id, and the session scratchpad path → `<scratchpad>` | **HEAD** |
| **HEADER+BACKUP** | 2 | HEAD = an **8-line** `SUPERSEDED 2026-09-13` notice prepended to the backup, byte-exact | **HEAD** |
| **STUB-FILLED** | 1 | the backup is HEAD's first 188 lines plus a `SEE_MANIFEST` placeholder; HEAD replaced the stub with the real manifest (+1,007 lines) | **HEAD** |
| UNEXPLAINED | **0** | — | — |

⇒ **Files to merge: 0. Files written to the worktree: 0. Nothing was staged from the backup.**
MEASURED · `raw/resolution.json` (per-file blobs, line counts, verdicts, landing commit).

## Why HEAD, per file — dates, content, run

* **Redaction (10).** Every backup-only line is a raw clip id, an 8-hex id prefix, or a
  scratchpad path that embeds a session id. HEAD carries the same lines with those replaced
  by sha12. Clip ids are gated-confidential (`deployed_val40_clip_digests.json`:
  *"per-clip sha256 ONLY"*), so ⛔ **merging any of these backups would re-leak ids into the
  repo.** The backup is the author's pre-landing working copy; HEAD is the redacted landing.
  Landed by `7b0f27c`, `56015c1` and `8ce3390`, all 2026-09-13 (the Qwen-Drive and SAM3
  usage-review runs).
* **Supersession notice (2).** `build_frames.py` and `render_sample.py` (2026-09-11 package):
  HEAD adds the 2026-09-13 notice that the renderer used camera geometry outside the one
  Qwen-Drive was trained on. The backup predates that finding. Landed by `7b0f27c`.
* **Stub (1).** `2026-09-13-sam3-only-road-map/RESULT.md`: the backup is the report before
  its manifest was written. HEAD is the 2026-09-15 extension (`1638b54`, the SAM3
  whole-corpus production report).
* **Identical (5).** Nothing to decide.

⚠️ **One consequence for re-running, stated so nobody is surprised.** The redaction also
renamed scratchpad file references in code (for example an image path keyed by an 8-hex
prefix now reads with the sha12). Those scratchpads were session-local and are gone anyway,
so neither copy is re-runnable as written. HEAD is a faithful, redacted **record**.

## The classifier was tested by mutation, not trusted

A verdict of "nothing to merge" is only evidence if the classifier can say otherwise.
`code/mutation_check.py` plants genuine backup-only content into real backup texts and calls
the **real** classifier:

| case | file | result |
|---|---|---|
| control + M1 append a code line | `temporal_filter.py` | REDACTION-ONLY → **UNEXPLAINED** ✅ |
| control + M2 change one number | `sam3_paint_ours.json` | REDACTION-ONLY → **UNEXPLAINED** ✅ |
| control + M3 edit a body line | `build_frames.py` | HEADER+BACKUP → **UNEXPLAINED** ✅ |
| control + M4 line before the stub | `sam3-only-road-map/RESULT.md` | STUB-FILLED → **UNEXPLAINED** ✅ |
| control + M5 add a JSON key | `sample_plan.json` | REDACTION-ONLY → **UNEXPLAINED** ✅ |

**5/5 mutations detected, 5/5 controls reproduced.** MEASURED · `raw/mutation_check.json`.

⚠️ **A defect in my own first pass, caught by the same discipline.** The first redaction
test used `\b` word boundaries and reported 3 files UNEXPLAINED. In `seq_<8-hex prefix>` the
underscore is a word character, so `\b` never fires and a prefix survives. Hex-boundary
lookarounds fixed it. The files were never unexplained; the regex was.

## Reproduce

```
python code/resolve_divergence.py raw/d_sync_stage2_input.json raw/resolution.json
python code/mutation_check.py     raw/d_sync_stage2_input.json raw/mutation_check.json
```

Both read the backup through the LOCAL ref only. 🔒 No output contains a raw clip id
(scanned: 0 hits, against a same-breath control that reads 6 in the backup).
