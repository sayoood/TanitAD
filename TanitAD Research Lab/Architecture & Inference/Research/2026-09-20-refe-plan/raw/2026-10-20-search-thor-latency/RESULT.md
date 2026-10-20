# RESULT: SEARCH-THOR-LATENCY instrument (2026-10-20). Integration needed: the Master Mind lands `out/` and schedules the Thor slot.

CPU only on the dev box; nothing run on Thor or any pod; nothing committed, pushed or `git add`ed; no register edited. Base: mirror tip `6a0cfaed` (fetched, pinned).

## Headline
`thor_search_latency.py` is written, CPU-tested (117 tests, 50/50 guard mutations caught) and run once on the real P2-s0 checkpoint, the real carve-B ctx and fan, 5 tokens, dev-box CPU. **That run is not a Thor number.**

| arm (dev-box CPU, fp32, 4 threads, n = 5 tokens, 1 repeat, 1 warm-up token; MEASURED, `raw/devbox5/`) | p50 ms | p95 ms | p99 ms | scorer calls / rows per token |
|---|---|---|---|---|
| (a) scorer-only pick | 61.1 | 72.7 | 74.9 | 1 / 64 |
| (b) search C8 (M = 64) | 870.3 | 930.6 | 930.8 | 14 / 896 |
| (c) search O4 (M = 256) | 2,703.3 | 2,790.0 | 2,800.8 | 38 / 3,392 |
Ratio to (a) at p50: 14.3x (C8), 44.3x (O4); O4 over C8: 3.11x (the ruling's "about 2.9x" was ESTIMATED). All controls green: identity 0 of 5 mismatches in every arm, determinism, call counts, n = 0, floor, sees-work. n = 5 and one repeat: the percentiles are only descriptive, and an earlier run on the previous code revision read 58.5 / 797.5 / 2,402.9 ms at p50 (about 10 % run-to-run; that JSON was superseded and not kept).

## What the numbers do not say
(a) is the scorer-only pick on banked post-trunk ctx; the trunk and the proposal generator are not timed, so `ratio_to_a` over-states the ratio to a full forward. Only the `added_ms_vs_a` is safe to quote across baselines. No pass / fail and no deployability claim (DQ4).

## Proposed register text (the Master Mind decides; nothing edited here)
- `H-SEARCH-P2-1` / C2 amendment section 7: the status "script to be written" becomes "BUILT, CPU-tested (117 tests, 50/50 mutations), real-data CPU smoke on 5 tokens; Thor run pending a slot; `inference_cost_status` stays UNMEASURED until a Thor JSON with `controls_ok` and `is_thor_number` true exists". Deviations from section 7 are listed in `RUN_ON_THOR.md` section 6 (the amendment's text should be read with them).
- Evidence class for a future Thor number: MEASURED (ours, Thor, nvpmodel mode and autocast named in the JSON), one search seed, post-trunk ctx.

## Manifest
See `out/paths.json` (every artifact is in the repo dir under `out/`; the source files are also in `C:/Users/Admin/par18/thorlat/`). Anything that exists in only one place: the dev-box run copies of Step B / L1 code it imports are NOT part of this package (they are in the repo at the tip).
