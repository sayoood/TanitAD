# BRIEF: HN screen chain, a CONFIG-LOAD check of token_check_names, before S0 (Master Mind, 2026-10-20 ~06:30)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/hntok3/`.
- CPU only. C: has about 0.35 GB free: TEMP / TMP under your work dir, delete only what you create. If C: drops below 60 MB, stop and write the HANDOFF.
- Never commit or push.

## Start
Tip `2f1969a` (mirror `C:/Users/Admin/tanitad-push`; run `git fetch`, then pin the tip once).
- Export `.../2026-09-20-refe-plan/refe/`, `eval/`, `taniteval/` (.py / .json / .md), `.../raw/2026-10-17-zero-tail-mechanism/` and `.../raw/2026-10-20-hnscreen-token-names/`, keeping the repo layout.
- Read the BACKLOG row `HNSCREEN-TOKEN-NAMES` (its last cell) and `.../mintguard/RESULT.md`.

## Do
1. In `refe/hnscreen_devbox.py`, when the config is loaded or frozen and BEFORE stage S0, refuse with a named marker and a nonzero rc if:
   - any VALUE of the effective `token_check_names` is not in `launch_gate.PROFILES[cfg gate.profile]["checks"]` (import the REAL `launch_gate`; never copy the list);
   - or the profile itself is unknown.
   - No stage may have started; assert on the stage markers.
2. **Tests:**
   - an override naming `G-HYG` passes membership, by design (it is a real check); note why that is acceptable here: the mint guard catches status, and the AST / real-token tests already exist;
   - an override naming `G14` refuses before S0;
   - an unknown profile refuses;
   - the default passes.
   - Mutation-check the new lines.
   - Run on your exported tree: `test_hnscreen_chain.py`, `test_hnscreen_ctx_chain.py`, `raw/2026-10-20-hnscreen-token-names/test_hnscreen_token_names.py`. Report counts before and after; none may drop.
3. Manifest `hntok3/out/paths.json`: `[repo path, abs src, "raw"]`, with `"base=<the tip's blob>"` on every edited existing file. Put a short `RESULT.md` and the logs under `.../raw/2026-10-20-hnscreen-token-names/configguard/`. Write `HANDOFF.md` (600 tokens or fewer).

## Hand-back (150 words or fewer)
The `file:line`, the counts, and the manifest.
