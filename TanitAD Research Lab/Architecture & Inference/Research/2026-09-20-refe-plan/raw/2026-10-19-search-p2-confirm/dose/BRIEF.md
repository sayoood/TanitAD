# BRIEF: SEARCH-DOSE amendment draft + a reachability proof (Master Mind, 2026-10-20 ~03:20)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/doseamend/`.
- CPU only. Write SMALL files only: C: has about 0.1 GB free and D: has 0.
  - Put pytest basetemp under your work dir; no .pt files, no bundles.
  - If C: drops below 40 MB, stop and write HANDOFF.md.
- Never commit or push. No GPU, no pod.
- Never read any B-REST output or label. None exists yet.

## Why
`MM_RULING_SPC.md` DQ7 (`C:/Users/Admin/par18/spc/`) committed the SEARCH-DOSE arm's BAR before C1:
- the pinned O4 config (`stepB_config_O4.json`, sha256 `1ebd7196…`, M = 256);
- everything else identical to C1's arm, on the SAME B-REST bundle;
- CPU only;
- C1's outcome rules verbatim;
- the dose contrast (M256 − M64, paired per token and seed) descriptive only.

"The rest goes in its own amendment, sealed before its launch." Write that amendment now, so it is sealed before C1 reads.

## Deliverables
1. **`doseamend/AMENDMENT_DOSE_DRAFT.md`** (DRAFT; the MM seals). It states:
   - the object and the pins (config sha, ckpt sha, seeds, token list = B-REST's pinned list `3a85cc81…`);
   - the run order after `ZZSPC_DONE`: which C1 outcomes trigger it, and INERT's special case;
   - the estimator;
   - the outcome table, verbatim from C1;
   - the four families (STRATEGIC PAUSED);
   - the CPU cost: ESTIMATE it from Step B's measured M = 64 rate, scaled, and say how;
   - peak disk;
   - every gate with its RED arm;
   - the next arm per outcome.
   - Put the decisions to me as DQ1..DQn with your recommendation.
2. **REACHABILITY: the binding rule "built, tested and unreachable from its caller", CLAUDE.md.** Prove from source that C1's chain (`C:/Users/Admin/par18/spc/code/`) can run the search with the O4 config on the B-REST bundle WITHOUT a code change: argv → config load → the sha pin → M actually used in the search loop. Quote `file:line` at each hop.
   - If a hop is missing, implement the smallest change: a flag whose DEFAULT keeps C1's behaviour bit for bit.
   - Add a test proving that M = 256 is what the search loop receives (a non-trivial assertion), and a control proving that the default still gives M = 64.
   - Run a mutation check on the new lines, plus the full spc suite. Report the counts.
   - Use only a tiny synthetic input: NO real bundle, NO label.
3. **`doseamend/HANDOFF.md`** (1.5k tokens or fewer) and `doseamend/out/paths.json` (`[repo path, abs src, "raw"]`). Repo dir: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-search-p2-confirm/dose/`.

## Hand-back (300 words or fewer)
- The DQs.
- The reachability chain with `file:line`, and any change made.
- Test and mutation counts.
- The CPU cost estimate.
- The manifest path.
