# Deliverable manifest — 2026-07-22 YouTube/IDM data-pipeline groundwork

**Task:** de-risk the intrinsics gap + the licensing question ahead of the IDM cross-rig proof verdict.
Built only the proof-independent parts (scope discipline — the full ingest/labeler is gated on
`stack/scripts/run_idm_proof.py`, running separately, NOT touched by this task).

**Operating rules honored:** STAGE-NEVER-PUSH (all artifacts `git add`ed, nothing committed/pushed,
branch unchanged) · primary sources only (calib facts from `stack/tanitad/data/calib.py` + its tests;
external claims verified by web search this task, marked `PUBLISHED`) · fail-loud (limitations marked).

| artifact | path (all `repo:` — staged, in the working tree) | one copy only? |
|---|---|---|
| Pipeline design (S0–S5, front-end result, parity firewall) | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-22-youtube-idm-pipeline/PIPELINE_DESIGN.md` | no — in repo |
| ⭐ f-theta front-end prototype (runnable) | `repo:.../2026-07-22-youtube-idm-pipeline/ftheta_frontend_prototype.py` | no — in repo |
| Prototype MEASURED result (JSON) | `repo:.../2026-07-22-youtube-idm-pipeline/ftheta_frontend_result.json` | no — in repo |
| Canonical sample frame (visual proof, 256×256 @ f_eff≈266) | `repo:.../2026-07-22-youtube-idm-pipeline/canonical_sample.png` | no — in repo |
| Licensing / ToS tier verdict | `repo:.../2026-07-22-youtube-idm-pipeline/LICENSING_TIER_ANALYSIS.md` | no — in repo |
| This manifest | `repo:.../2026-07-22-youtube-idm-pipeline/MANIFEST.md` | no — in repo |

**Nothing lives on a pod or worktree.** No pod compute was used (the round-trip is CPU geometry on the
dev box). No new code was added to `stack/` — the front-end **reuses** primitives already in HEAD
(`calib.focal_crop_resize`, `calib.ftheta_crop_resize`, `comma2k19.stack_frames`), so `pytest -q` is
unaffected.

## Headline result (MEASURED, `ftheta_frontend_result.json`)
Canonicalization **round-trips cleanly to f_eff ≈ 266** on both camera branches:
pinhole `focal_crop_resize` → **266.545** (+0.21 %); wide `ftheta_crop_resize(center="principal")` →
**266.02** (+0.007 %); `stack_frames` → `[2, 9, 256, 256]` (the 9-ch encoder contract). Verified on a
real comma2k19 night-highway frame from our lake. The intrinsics-estimate tolerance is quantified:
`f_eff_true = F_REF·(f_true/f_est)` (a ±10 % focal error → ±10 % f_eff).

## Licensing verdict (one line)
Raw frames = **`refuse` to re-host / `nc`-caveats internal-only** (same call as TLD); IDM pseudo-labels
**inherit the source tier** (annotations-only URL layer = `nc` pending review); a YouTube-pretrained WM
= **internal-research OK, provenance-stamped, public/commercial release gated on legal review**. Ship
pointers, never bytes (the OpenDV model).

## Escalations (operating standard rule 3)
1. **Intake review** — these are `incoming/` deliverables; the Data-Engineering owner should intake
   `PIPELINE_DESIGN.md` + `LICENSING_TIER_ANALYSIS.md` into the hub proper and decide whether the
   licensing verdict's `HYPOTHESIS` rows (annotations-only shipping; YouTube-pretrained WM release) go
   to Sayed + legal.
2. **Post-proof integration** — if the IDM proof PASSES, the single new module this line needs is a
   **per-video intrinsics estimator** (GeoCalib wrapper + the VP cross-check already prototyped here);
   the canonicalization + 9-ch contract are already in HEAD and need no merge.
3. **No decision here changes a GPU-day** — all evidence is MEASURED (dev-box geometry) or PUBLISHED.
