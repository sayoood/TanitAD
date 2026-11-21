# PLAN — E-OPP-CW1-1

`2026-09-02 · Opponent Analysis · compute budget: 0 GPU, ~10 min CPU + one 9.5 MB fetch`

## Priority order (a killed run still yields value at every step)

| # | step | yields on its own | state |
|---|---|---|---|
| 1 | Read CW-1's own text and locate every repo site carrying a DriveFuture navhard number | whether the fence is currently observed | ✅ done |
| 2 | Read the banked primary `2605.09701` | the CW-1 resolution | ⛔ **blocked — the file is truncated** |
| 3 | Read the banked primary `2606.07170` (TOAD) | F3/F4/F6 — the DrivoR correction, the real leaderboard, the search-vs-re-ranking warning; **and an independent DriveFuture level** | ✅ done |
| 4 | Survey all 310 banked entries for readability | F5 — how big is the blocker's class | ✅ done (1 of 310) |
| 5 | Refetch + repair the truncated primary, verify by content | unblocks step 2 and fixes the evidence base | ✅ done |
| 6 | Re-attempt step 2 on the repaired copy | ⭐ F1/F2 — CW-1 resolved, plus the scorer-vs-world-model decomposition | ✅ done |
| 7 | Escalate; propose backlog rows | the deliverable | ✅ done |

⭐ **Step 3 was deliberately placed before the repair.** TOAD independently
reproduces DriveFuture's 55.5, so had the refetch failed, CW-1 would still have had
a partial external answer and the pass would still have shipped F3–F6.

## Compute

* **0 GPU.** PDF text extraction and byte checks on CPU; one 9.45 MB HTTPS fetch
  from arXiv (`truststore.inject_into_ssl()` — `certifi` fails behind this box's
  TLS proxy).
* ⛔ Thor and the A40 pod (REF-C v3 H-arm, ~35 h remaining): **not touched**.

## Owners / integration

* **Owner:** Research Lab (daily pass 2026-09-02).
* **Escalated:** CW-1's closure (PI / Master Mind), the `LEADERBOARD.md`
  corrections (EvalFlyWheel), backlog row 32's wedge, `kb_add.py`'s completeness
  gap, and the search-arm requirement for I-1 / L-3. See `COMMS.md`.

## Mutations made to shared state, and how they are reversible

| change | safeguard |
|---|---|
| overwrote `Library/papers/2605.09701_*.pdf` | written with a sha256 read-back; the replaced content was provably unreadable (no `%%EOF`, 14.5 % of the true size) |
| updated that entry's `bytes` / `sha256` in `library.json` | atomic `os.replace`; the **old hash is recorded in the entry's own note**; full pre-repair index backed up to `raw/library.json.pre-repair.bak` |
| regenerated `LIBRARY.md` | via `kb_add.py --reindex` only — ⛔ **never hand-edited** |

## Reproduce

```bash
python code/library_integrity.py --out raw/library_integrity.json
```

⚠️ `raw/library_integrity.json` is the **PRE-REPAIR** survey — it is the evidence
for F5 and is kept as measured. Re-running it now reads 310/310 OK; the
single-entry post-repair check (`verdict OK, n_pages 24`) is recorded in
`RESULT.md`.
