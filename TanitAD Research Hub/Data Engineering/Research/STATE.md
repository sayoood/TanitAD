# STATE — Data Engineering

LAST_RUN: 2026-07-07 (W1→W2, Tuesday agent)
QUALITY: full (G-A…G-F, G-D1, G-D2 met; suite 40✓/1 skip)

## This run (2026-07-07)
- Entry: backlog #1 (comma2k19 loader) found **already shipped by Sayed as D-009** mid-session
  (`comma2k19.py`, `base250cam` 6-ch 2-frame contract). Did NOT overwrite; pivoted to validate + close gaps.
- **Real-data validation** (HF token via `tanitad.keys`): loader's `_decode_video` decodes real comma HEVC
  → [200,3,256,256] @ ~105 fps; A8 real ≈ 0.053. Decode path de-risked before A40 spend.
- **Contract reconciliation:** generalized shared `assert_contract(ep, channels=1|6|None)`; added
  `test_comma2k19_contract.py` (ties D-009 loader to the shared contract home + A8 check); exported
  comma2k19 from `tanitad.data.__init__`.
- **Docs:** data card (`2026-07-07-comma2k19-data-card.md`, all G-D1 fields) + research note
  (`2026-07-07-comma2k19-validation-and-h7.md`, H7 deltas + findings).
- **Findings:** Windows `|` path bug (extract on Linux pod only); A8 weak on raw highway RGB → change-
  weighting justified; H7 gains LAOF + Sensorimotor-WM.

## Next (backlog, priority order)
1. **Steering-ratio calibration log** (H7 artifact) — per-segment residual when seed IDM trains on comma2k19.
2. **Per-dataset A8 statistics harness** (backlog #4) — `frame_change_fraction` distribution per corpus →
   data-driven change-weighting schedule for the W2 bake-off.
3. **On the Linux A40 pod:** pull+unzip Chunk_1 (8.7 GB), add `av` to `[real]` extra, smoke `build_episode`
   on 3 segments, wire into Stage-A. ~1–2 engineer-h, 0 new code.
4. PhysicalAI-AV loader hardening + license review note (backlog #2, D-002) — second corpus.
5. Focal-canonicalization prototype (backlog #3) — deprioritized until heterogeneous video (GoPro/BDD/OpenDV).

## HANDOFF
None — all work committed and pushed, suite green. Real segment cached in scratchpad (not committed).
