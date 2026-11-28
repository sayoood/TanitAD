# Search / probe log — 2026-09-15 EPDMS Stage-1 fingerprint

V-1 first: `library.json` checked for every target before any download.

| # | probe | result |
|---|---|---|
| L1 | library: DriveFuture `2605.09701`, TOAD `2606.07170`, Pseudo-Sim `2506.04218`, DrivoR `2601.05083` | all four banked with PDF |
| L2 | library: CLOVER, RAP | **not banked** → banked today `2605.15120`, `2510.04333` |
| L3 | library: GuideFlow `2511.18729` | entry exists; ⛔ **no PDF file in `papers/`** (glob on id → 0). Second probe: `ls papers | grep 2511.18729` → 0. Not fingerprinted; GuideFlow's post-fix value is taken from TOAD Tab. 6 |
| L4 | ids for CLOVER and DrivoR: repo grep | CLOVER `2605.15120` (`Architecture & Inference/Research/2026-07-27-planner-scorer-inputs/CITATIONS.md`); DrivoR `2601.05083` (`Opponent Analysis/Research/2026-08-28-navsim-v2-camera-lane`) |
| F1 | `s1s2_fingerprint.py`: pypdf text, ±700 chars around each `PDM-Closed` / `PDM-C` | see JSON. DriveFuture 1 mention / 0 tokens; CLOVER **0 mentions** (second probe: regex `PDM` without suffix, in the keyword pass → the only PDM hits are PDMS/EPDMS metric names) |
| F2 | keyword pass DrivoR: `bug`, `54.6`, `56.3`, `after official` | Tab. 3 post-fix caption; Tab. 14 pre/post split; Issue #151 named |
| F3 | keyword pass DriveFuture: `55.5`, `navhard`, `two-stage` | abstract + Fig. 1 + "as of April 2026 ranks 1st"; no PDM-Closed navhard row |
| F4 | keyword pass CLOVER: `48.3`, `navhard` | *"matching the strongest reported result"*; navtest reported for both the updated (90.4) and original (87.2*) code, navhard version unstated |
| ⛔ E1 | NAVSIM GitHub Issue #151 itself | **not fetched this pass**. The mechanism is RELAYED through DrivoR's description → `E-BE-FIX151-1` |
