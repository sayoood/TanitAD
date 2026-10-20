# BRIEF: fix HNSCREEN-TOKEN-NAMES, a launch blocker of the P6-HN screen (Master Mind, 2026-10-20 ~05:30)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/hntok/`.
- CPU only. C: has about 0.38 GB free: TEMP / TMP in a new dir under your work dir, delete only what you create. If C: drops below 60 MB, stop and write the HANDOFF.
- Never commit or push. No GPU, no pod, no Thor.

## Start from the tip
`065744d` on `agent/arch-inf-20260803` (mirror `C:/Users/Admin/tanitad-push`; run `git fetch`, then pin the tip once). HNFP3 just landed there.
- Export `.../2026-09-20-refe-plan/refe/` and `eval/` into your work dir, keeping the repo layout.
- The reader tests also need `taniteval/` (repo root; .py / .json / .md suffice) and `.../raw/2026-10-17-zero-tail-mechanism/`.
- Read the BACKLOG row `HNSCREEN-TOKEN-NAMES` (it has every `file:line`).

## The defect (MEASURED from source by the MM; re-verify at the tip before changing anything)
- `refe/hnscreen_devbox.py:74` defaults `gate.token_check_names` to `{"G14": "G14", "G15": "G15"}`.
- `token_checks()` returns False on a missing key, and `assemble_gates` reads G14 / G15 from it.
- The real `refe_hnscreen` profile (`refe/launch_gate.py:370`, which inherits `refe_rls1`'s `checks` at `:316`) mints only `G-HYG, G-DVB, G-PREFLIGHT, G-SUITE` (+ `G-CLOCK` NA). It judges G14 / G15 INSIDE G-PREFLIGHT (`:1133-1145`), required by `gate_checks.py:1026-1028`.
- So a REAL token reads G14 = G15 = False, and the screen ends SCREEN-INVALID after its GPU spend.
- The chain test passes only because its STUB token writes checks named G14 / G15 (`refe/test_hnscreen_chain.py:155`).
- Line numbers may have moved with HNFP3; find them again.

## Do
1. Default `token_check_names` to `{"G14": "G-PREFLIGHT", "G15": "G-PREFLIGHT"}`. Keep the config override.
2. **Invariant test** (the one that would have caught it): every VALUE of the chain's effective `token_check_names` is in `launch_gate.PROFILES[cfg.gate.profile]["checks"]`. Import the REAL `launch_gate`; do not copy the profile.
   - RED arm: the old default must fail it.
   - A second RED arm: a profile whose checks lack `G-PREFLIGHT` must fail it.
3. **Real-token test:** mint a token the way the real gate writes `checks` (use the real profile's check list, `status: PASS` for each), and assert `assemble_gates` reads G14 = G15 = True. With the old default it must read False.
4. Move the chain tests' stub (`test_hnscreen_chain.py`, and `test_hnscreen_ctx_chain.py` if it has the same stub) to the REAL profile's check names. Keep every existing arm green; the mutant counts must not drop.
5. Run, on your exported tree: `test_hnscreen_chain.py`, `test_hnscreen_ctx_chain.py`, `test_hn_ctx_anchor.py`, `eval/test_hn_screen_read.py`, and your new test. Report each ZZTEST count before and after.
   - Do NOT run `test_launch_gate.py` (disk-heavy) unless you changed `launch_gate.py`. You should not need to.
6. Write a manifest `hntok/out/paths.json`:
   - `[repo path, abs src, "raw"]`, and for each EDITED existing repo file a 4th element `"base=<the tip's blob sha>"`.
   - New tests and a short `RESULT.md` go under `.../raw/2026-10-20-hnscreen-token-names/`.
   - Write `HANDOFF.md` (1k tokens or fewer).

## Hand-back (200 words or fewer)
- The changes, with `file:line`.
- The before / after counts.
- The RED arms.
- The manifest.
