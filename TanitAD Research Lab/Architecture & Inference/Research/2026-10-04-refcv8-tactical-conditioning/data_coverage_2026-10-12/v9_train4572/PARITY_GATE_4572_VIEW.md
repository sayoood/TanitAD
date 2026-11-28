# The 4,572-clip refcv8 train view through the parity ingest gate (2026-10-12)

**Why this exists.** The launch gate's G-SUITE-PINNED check (B7, dev box, commit `a0b9faf`, full-commit archive) ran
33 pinned test files, **749 tests, 1 failed, 5 skipped**. The one failure was
`test_build_parity_guard.py::test_every_derived_corpus_writer_is_gated_or_classified`:
`stack/scripts/build_refcv8_add203_view.py` (added in `2b510e5`) writes the train view's manifest but never called
`parity.guard_corpus_build`, the programme's check that a supervision corpus holds none of the 40 deployed-val
episodes. The builder had its OWN eval139 / v8-eval disjointness checks, but not this one. The gate would have refused
the refcv8 launch on it.

**Fix.** The builder now calls `parity.require_ingest_gate` + `parity.guard_corpus_build(new_ids, role="train",
mode="refuse")` after it assembles the 4,572 ids and BEFORE it writes anything, and it records the result in its
report and `_VIEW_RECORD.json`. Test on a tip export (`8b289e2`, stack + taniteval + the suite's one outside evidence
file): **unpatched 1 failed / 23 passed (the same test as B7), patched 24 / 24 passed.**

**The view already built on Thor, re-checked through the same call (MEASURED).** Manifest
`train4572_v2manifest.pt` md5 `77dafb50af0a…` (the one the v9 4,572 release was built against), 4,572 unique clip
ids, tip `parity.py`, `role="train"`, `mode="refuse"`:

| quantity | value |
|---|---|
| clips in | 4,572 |
| inside the 40 deployed-val episodes (the disqualifying overlap) | **0 of 40** |
| inside the parity train corpus `physicalai-train-e438721ae894` (control: must read non-zero for a B1-derived set) | 190 |
| oracles loaded | parity train 2,400 clip digests, deployed val 40 |
| kept | 4,572 (refuse mode, nothing removed) |
| decision_grade | true |

So the built view needs no rebuild; only the builder was missing the call.
