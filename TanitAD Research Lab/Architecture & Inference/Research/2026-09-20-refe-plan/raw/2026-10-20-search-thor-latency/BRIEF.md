# BRIEF: write the search's Thor latency instrument (SEARCH-THOR-LATENCY) (Master Mind, 2026-10-20 ~04:15)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/thorlat/`.
- CPU only on the dev box. **Do NOT connect to, copy to, or run anything on the Jetson Thor or any pod.** It may be training; adding load there is forbidden. You WRITE and TEST the instrument; the MM schedules the Thor slot.
- C: has about 0.4 GB free: small files only. If C: drops below 60 MB, stop and write the HANDOFF.
- Never commit or push.

## Why
Any navtest (C2) pass of the CEM search on P2's own scorer is capped at "selection-tier gain, deployability UNMEASURED" until the search's per-token latency on the deployment target is measured (`MM_RULING_C2AMEND.md` DQ3 / DQ4, on tip `6a0cfae`).
- The protocol is in the C2 amendment draft §7: `.../raw/2026-10-19-search-p2-confirm/c2/AMENDMENT_C2_DRAFT.md`, read from the mirror `C:/Users/Admin/tanitad-push` (fetch, then pin the tip).
- The search: Step B's P2HEAD CEM search (`stepB.py`, `tts_cem.py`), configs C8 (`stepB_config.json`) and O4 (`stepB_config_O4.json`, M = 256). Find them on the tip, under the l2-stepB / search-p2-confirm packages.

## Deliverables
1. **`thor_search_latency.py`.** Per token at batch 1, it times:
   - (a) P2's pick-only forward (the baseline);
   - (b) the full search, C8;
   - (c) the full search, O4.
   - Output: p50 / p95 / p99 ms per token, the ratio to (a), and the counts of scorer calls.
   - **Controls:**
     - warm-up excluded and stated;
     - an identity check that the timed search returns the SAME output as the untimed reference on every timed token (timing must not change the answer);
     - the device, clocks and power mode recorded (on Thor, `nvpmodel -q` and `jetson_clocks` status, read-only);
     - a no-op timing floor.
   - **Inputs:** 200 carve-B tokens, from the banked carve-B fan, never a label. Say which file, and its sha.
   - No pass / fail, per DQ4.
2. **Tests (CPU, tiny synthetic inputs):**
   - the call count for C8 vs O4 matches the config;
   - the identity control goes RED when the timed path is perturbed;
   - the percentile maths;
   - the refusal when the device fields are missing.
   - Plus a mutation check of the guard lines.
3. **`RUN_ON_THOR.md`:** the exact steps for the MM to run it on Thor:
   - what must be copied (code, checkpoint, tokens) and the size;
   - the expected runtime (ESTIMATE it from the dev-box CPU timing you measure on 5 tokens, and say it is not a Thor number);
   - the preflight that refuses to start if Thor is training. Find how the programme detects that (Deploy + Opt docs, `TRAPS.md` Thor entries: three standard memory probes lie on Thor, so read that trap first).
4. `HANDOFF.md` (1k tokens or fewer) and `out/paths.json` (`[repo path, abs src, "raw"]`). Repo dir: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-20-search-thor-latency/`.

## Hand-back (250 words or fewer)
The dev-box CPU ms per token for (a) / (b) / (c) on 5 tokens (MEASURED, dev-box CPU, not Thor), the test and mutation counts, what RUN_ON_THOR needs, and the manifest.
