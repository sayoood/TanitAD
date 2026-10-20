# BRIEF: HN screen chain, refuse at the MINT stage on a token-name mismatch (Master Mind, 2026-10-20 ~06:00)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/hntok2/`.
- CPU only. C: has about 0.35 GB free: TEMP / TMP under your work dir, delete only what you create. If C: drops below 60 MB, stop and write the HANDOFF.
- Never commit or push.

## Start
Tip `08ff54e` (mirror `C:/Users/Admin/tanitad-push`; run `git fetch`, then pin the tip once).
- Export `.../2026-09-20-refe-plan/refe/`, `eval/`, `taniteval/` (.py / .json / .md) and `.../raw/2026-10-17-zero-tail-mechanism/`, keeping the repo layout.
- Read `.../raw/2026-10-20-hnscreen-token-names/RESULT.md` and the BACKLOG row `HNSCREEN-TOKEN-NAMES` (its follow-up).

## Do
1. In `refe/hnscreen_devbox.py`, at the stage that verifies or mints each seed's PASS token, BEFORE any GPU stage:
   - if any VALUE of the effective `token_check_names` is absent from the token's `checks`, STOP with a named marker, e.g. `SCREEN-INVALID: token check '<name>' absent`;
   - also stop if that check's status is not PASS.
   - A wrong run-box override must fail here, not at the end.
   - Find where the chain first reads the token, and quote `file:line`.
2. **Tests:**
   - A config override naming a check the real profile does not mint must stop at the mint stage, and no later stage may run. Assert on the stage markers.
   - The default must pass.
   - Add a mutation for the new guard line(s).
   - Run on your exported tree: `test_hnscreen_chain.py`, `test_hnscreen_ctx_chain.py`, `raw/2026-10-20-hnscreen-token-names/test_hnscreen_token_names.py`. Report the ZZTEST counts before and after; none may drop.
3. Manifest `hntok2/out/paths.json`: `[repo path, abs src, "raw"]`, with `"base=<the tip's blob>"` on every edited existing file. Put a short `RESULT.md` and the logs under `.../raw/2026-10-20-hnscreen-token-names/mintguard/`. Write `HANDOFF.md` (800 tokens or fewer).

## Hand-back (150 words or fewer)
The `file:line` of the change, the counts, and the manifest.
