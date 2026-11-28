# HANDOFF EP-GEOM (L3), 2026-11-27

State: card, lib, fit, read, tests done. Fit ran: **NULL-HELDOUT** (see FIT_REPORT.json). Therefore **no navtest decision read** (epgeom_read.py refuses). L3 does not advance; the committed next arm is **EP-HEAD**.

MM to land (staging root = `E:/Projects/TanitAD-artifacts/l3epgeom/stage/`, same repo path `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-11-27-epgeom/`): all files in this folder.
After landing, update `GOALS_AND_CLAIMS.md` `H-REFE-EPGEOM-1`: NULL-HELDOUT, T0, navtrain only, navtest untouched.

Re-run checks (CPU, ~1 min):
- `cd <folder>; TMP=F:/TanitAD_l3_tmp C:/Users/Admin/venvs/tanitad/Scripts/python.exe test_epgeom.py`
- `C:/Users/Admin/venvs/tanitad/Scripts/python.exe epgeom_fit.py --out-dir <folder>` (rewrites FROZEN_PARAMS.* with a new timestamp; values reproduce)

Only if the MM wants a decision read anyway (it would violate the card's §3, so it needs an amendment frozen first): `epgeom_read.py --frozen FROZEN_PARAMS.json --frozen-sha256 <FROZEN_PARAMS.sha256> --tables <propscore>/full_C1_s0of2.jsonl <propscore>/full_C1_s1of2.jsonl --out EPGEOM_READ.json --picks-out epgeom_picks.json` after the navtest dump is complete (136 logs).

Next: EP-HEAD under its own card (SAFEHEAD recipe, row 2 on `tr_ep`); cache `F:/safehead_cache/` (only copy, do not delete).
