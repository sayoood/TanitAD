# BANKED — nine files, sha256-verified on the destination

**Written** 2026-08-30 · **Owner** DataFlyWheel
**Status** ✅ **CLEARED** — the mount recovered after ~2 h and all nine landed.

⭐ **Kept as the record of the debt, not deleted.** It says what was owed, where it
went, and what the mount was doing while it could not go there — which is the part
an audit needs and the part that vanishes if the file is removed once it is empty.

⚠️ **STILL OUTSTANDING: nothing on disk, one thing on HF.** The HF copy of
`V72_MANIFEST.json` remains stale pending PI approval (see below). And these files
are **banked but NOT staged** — the worktree constraint keeps me out of the main
checkout's index, and the Master Mind sweeps them in with their own queue.

⚠️ *This file was corrupted once and rewritten: a `Get-Content | Set-Content`
round-trip in Windows PowerShell 5.1 decoded the UTF-8 bytes as cp1252 and wrote
the mojibake back. **Never round-trip a UTF-8 file through PowerShell 5.1 without
`-Encoding utf8` on the READ as well as the write.** Edit it with the file tools.*

## Destination

`TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-30-v72-copy-adjudication/`

⚠️ The **Research Lab** tree, not Research Hub — the rename is binding, and the
worktree at `.claude/worktrees/interesting-tharp-463cf3` is a **stale checkout**
whose HEAD predates it. Bank from the MAIN checkout.

| source (C:, under `_s2build/release/`) | destination | sha256 |
|---|---|---|
| `RESULT_v72_adjudication.md` | `RESULT.md` | verify at copy |
| `adjudicate_v72_copies.py` | `code/adjudicate_v72_copies.py` | verify at copy |
| `quarantine_and_build_eval_digests.py` | `code/quarantine_and_build_eval_digests.py` | verify at copy |
| `band_coverage_census.py` | `code/band_coverage_census.py` | verify at copy |
| `patch_v72_manifest_bands.py` | `code/patch_v72_manifest_bands.py` | verify at copy |
| `v72/eval_v72_clip_digests.txt` | `raw/eval_v72_clip_digests.txt` | `be39fb29575115f0…` |
| `v72/V72_MANIFEST.json` | `raw/V72_MANIFEST.json` | `dc04a976e954491b…` |
| `band_census.json` | `raw/band_census.json` | verify at copy |

`bank_to_drive.py` performs the copy with verified retries. It confirms each
directory by a separate `is_dir()` read, because on this mount `mkdir` reports
success and creates nothing.

⚠️ **The HF copy of `V72_MANIFEST.json` is STALE.** The band-coverage note is in
the local manifest only. Re-push to `Sayood/tanitad-v7-training-corpus` (private)
is **pending PI approval** — it modifies a published artifact. The label blobs it
describes are untouched (md5s asserted unchanged on every patch), so nothing a
trainer reads has changed; only the documentation lags.

## DONE — the `V72_MANIFEST.json` band-coverage note (MM-E15)

Generated from `band_census.json`, never typed. Every figure carries
`{train, eval, train+eval}`, asserted on read-back.

1. **`>30 s` is OUT OF HORIZON by design.** `emit_one` loads
   `RAW_T0_S + LOOKAHEAD_S + 5.0` = 35 s so a manoeuvre straddling the 30 s edge
   has poses to clip against. Starts inside that margin (train 256 / eval 12) are
   a by-product of defining the edge, not content the label claims — which is why
   they are correctly NOT in `unassigned_manoeuvres`. ⚠️ The eval side is
   proportionally worse: **10.3% vs 7.4%**, on only 116 manoeuvres.
2. **`operative_s` declares temporal SCOPE, not token coverage.** No manoeuvre is
   assigned to it and no `a_op`/`g_op` exists — and **that is correct, not a gap**:
   the operative layer is supervised *continuously*, by trajectory regression
   (`ade_dense_m`/`fde_last_m` at T1). ⚠️ But a count taken from `bands` reads it
   as zero supervision. It is not; it is differently supervised.
   *(Earlier wording "declared but never populated" was withdrawn — true of the
   tokens, and it would send a reader hunting for labels never meant to exist.)*
3. **`t_start == 30.0` is claimed by nothing** (train 8 / eval 1) — start-time
   census says strategic, strict overlap says no band.
4. ⭐ **Counts must name three things: convention, corpus, and operator.** 54/332/453
   are three conventions with three right answers; train-only and train+eval differ
   again; and `>=` vs `>` at t = 30.0 differs a third time — that last one produced
   264 vs 256 for the same quantity inside one document. The note states its own
   operator as **strict `> 30.0`**.

5. ⭐ **Withdrawn readings are RECORDED, not merely removed.** Three readings that
   are true on their face and mislead anyone acting on them — "operative_s is
   declared but never populated", "the [6,8) accounting disagrees ~8× with the
   band arithmetic", "the pre-fix blobs are unparseable JSON" — each with why it
   was withdrawn. Deleting a wrong reading guarantees the next person who
   rediscovers the same fact reaches for the same wording; a record saying *this
   was tried and here is why it misleads* is what stops the loop. Same principle
   as naming the superseded blobs `DO-NOT-USE` rather than deleting them.

⚠️ **NOTE, do not FIX.** v7.2 is released and the trainer is being wired to it;
changing band assignment would alter labels under a live consumer. Whether the
operative layer should own manoeuvres, and whether [6,8) stays a hole, are
HIERARCHY questions for the PI.

## Also on C:, already applied, not owed to the repo

`_superseded_pre_schema_fix/` — the two pre-schema-fix v7.2 blobs, **renamed**
(`…PRE-SCHEMA-FIX.DO-NOT-USE.jsonl.gz`) so a recursive filename glob cannot select
them, plus a `README.md`. Master Mind ruled: **keep the bytes, do not delete.**

## State of the mount

Listings work; `New-Item -ItemType Directory` reports success with no directory
created; `Set-Content` into an existing directory returns `Falscher Parameter`;
`git` returns `fatal: error reading .git`. 70 verified retries over 7 minutes
landed 0 of 4 files, and it has stayed degraded for ~90 minutes since. The Master
Mind's side is down in the same window with four commits queued — so waiting on
the peer is not a fallback.
