# MANIFEST: 2026-10-19-epadm-searchbank (staged under C:/Users/Admin/par18/epadm_search/out/; nothing committed, pushed or `git add`ed)
| artifact | repo path (under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-epadm-searchbank/`) | lives in |
|---|---|---|
| RESULT, amendment draft, SIM_DESIGN (+ `.sha256`, `.hashed_at.txt`), HANDOFF, this MANIFEST | `RESULT.md`, `AMENDMENT_SEARCHBANK_DRAFT.md`, `SIM_DESIGN.md`, `SIM_DESIGN.sha256`, `SIM_DESIGN.hashed_at.txt`, `HANDOFF.md`, `MANIFEST.md` | staged `par18/epadm_search/out` + working copy `par18/epadm_search/src` |
| adapter, simulation, margin tool, fixture, tests, post-hoc script | `code/epadm_searchbank.py` (c4d6d76f), `epadm_sim_c1.py` (eca00c86), `apply_a2_margin.py` (b9c2c80d), `fixture_synth.py` (4c5fe787), `test_searchbank.py` (b4477887), `posthoc_diag.py` (5eedd015) | same |
| C1 geometry (token counts per log only) | `raw/geometry_c1.json` (1c599856) | same |
| results | `raw/results/sim_results.json` (099a370a), `replicates.npz` (58fb3a2e), `posthoc_diagnostics.json`, `identity_carveB.json`, `adapter_report_carveB.json`, `ranked_logs_carveB.tsv`, `pc3.json`, `repro.json`, `cost.json`, `run_part0.json`, `run_part1.json` | same |
| logs | `raw/logs/test_searchbank.log`, `run_part0.log`, `run_part1.log`, `build_carveB.log` | same |
| carve-B adapted banks (regenerable, 4 x 24 units, ~3 MB) | NOT staged | **ONE place only: `C:/Users/Admin/par18/epadm_search/work/carveB_bank2`** |
Frozen inputs (read from the mirror tip `fb7ad9a27187dd4db814d3f032b431ac7288d153`, pinned once): `2026-10-19-p6-read-prep/code/ep_admission.py` f0193372; `2026-10-19-ep-admission-r1/code/ep_admission_sim.py`; `2026-10-19-ep-reliab-picks/code/epreliab_lib.py` 72f493fd; `pickrule_lib.py` 555c0a34; `fastneg_lib.py` 0c0b399b. The package imports them by relative repo layout; none is copied into it.
Integration: land the package dir; the amendment draft is a DRAFT; no register was edited (proposed rows are in `RESULT.md` section 5).
